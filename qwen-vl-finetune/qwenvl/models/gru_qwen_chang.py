"""
ZeRO-3-native GRU-Qwen model (chang).

Replaces the old `qwenvl/models/gru_qwen.py` plain-nn.Module wrapper that breaks
under DeepSpeed ZeRO-3 (explicit `.to(device)` + manual warm-start into empty
partitioned params). Here the whole tree (Qwen backbone + projector + frozen GRU)
is a real `PreTrainedModel`, so a single `from_pretrained` builds it under one
`deepspeed.zero.Init()` and fills every weight via HF's zero3-aware loader — no
`.to(device)`, no manual `load_state_dict`.

Key/structure note: the backbone is held as `self.qwen`, so the state-dict keys
are exactly `qwen.*` / `projector.net.*` / `trajectory_gru.*`, matching the
existing alignment `model.safetensors` (no weight rewrite needed).

Injection (matches the alignment scheme): the frozen GRU encodes per-slot action
prefixes, the trainable projector maps the last valid hidden state (256) into the
Qwen embedding space (4096), and those vectors overwrite the `<gru>` placeholder
token embeddings in-place-by-blend (equal length → mrope position_ids stay valid).
"""

from typing import Optional

import torch
import torch.nn as nn

from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# The alignment used <gru> as the trajectory placeholder token (id 151669 in the
# BASE_GRU_DIR tokenizer). The embedding row is overwritten by the projector
# output, so the exact id only needs to match what the data pipeline emits.
DEFAULT_GRU_TOKEN_ID = 151669


class TrajectoryGRUEncoder(nn.Module):
    """Notebook-compatible GRU encoder used for trajectory action features.

    Structure mirrors `allignment training/qwenvl/models/gru_qwen.py` so the
    `trajectory_gru.*` keys line up with the alignment checkpoint.
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 256, embedding_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode_sequence(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        padded_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return padded_out


class ProjectorMLP(nn.Module):
    """GRU hidden (256) -> intermediate (1024) -> Qwen embedding (k * hidden).

    Structure mirrors `allignment training/qwenvl/models/projector.py` so the
    `projector.net.*` keys line up with the alignment checkpoint.
    """

    def __init__(self, gru_hidden_dim: int = 256, qwen_hidden_dim: int = 4096,
                 intermediate_dim: int = 1024, k: int = 1):
        super().__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.qwen_hidden_dim = qwen_hidden_dim
        self.intermediate_dim = intermediate_dim
        self.k = k
        self.output_dim = k * qwen_hidden_dim
        self.net = nn.Sequential(
            nn.Linear(gru_hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Qwen3VLGRUForConditionalGeneration(Qwen3VLPreTrainedModel):
    """Qwen3-VL + frozen trajectory GRU + trainable projector, ZeRO-3-native.

    `self.qwen` is a full `Qwen3VLForConditionalGeneration`. We build it from
    `config` (NOT from_pretrained, NO `.to(device)`); the outer `.from_pretrained`
    loads `qwen.*` / `projector.*` / `trajectory_gru.*` together, zero3-aware.
    """

    config_class = Qwen3VLConfig
    # Inherit the backbone's no-split modules so ZeRO-3 / activation checkpointing
    # wrap the decoder/vision layers correctly.
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(
        self,
        config: Qwen3VLConfig,
        gru_input_dim: int = 7,
        gru_hidden_dim: int = 256,
        gru_embedding_dim: int = 128,
        projector_k: int = 1,
        gru_token_id: int = DEFAULT_GRU_TOKEN_ID,
    ):
        super().__init__(config)

        self.qwen = Qwen3VLForConditionalGeneration(config)

        qwen_hidden = getattr(getattr(config, "text_config", config), "hidden_size", None)
        if qwen_hidden is None:
            qwen_hidden = getattr(config, "hidden_size")

        self.trajectory_gru = TrajectoryGRUEncoder(
            input_dim=gru_input_dim, hidden_dim=gru_hidden_dim, embedding_dim=gru_embedding_dim
        )
        self.projector = ProjectorMLP(
            gru_hidden_dim=gru_hidden_dim,
            qwen_hidden_dim=int(qwen_hidden),
            intermediate_dim=1024,
            k=projector_k,
        )

        self.gru_token_id = int(gru_token_id)
        # The trajectory GRU is frozen by default (the aligned projector is the
        # only GRU-side module trained further in SFT).
        for p in self.trajectory_gru.parameters():
            p.requires_grad = False

        self.post_init()

    # ---- HF plumbing proxied to the inner backbone -------------------------
    def get_input_embeddings(self):
        return self.qwen.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.qwen.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.qwen.get_output_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.qwen.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        self.qwen.enable_input_require_grads()

    @torch.no_grad()
    def _encode_gru_last_state(self, gru_features: torch.Tensor, gru_lengths: torch.Tensor) -> torch.Tensor:
        """Frozen-GRU encode -> per-slot last valid hidden state.

        gru_features: (N, max_t, 7), gru_lengths: (N,) -> returns (N, hidden_dim).
        """
        # Move the *input data* onto the GRU's weight device+dtype (e.g. bf16 when
        # the model is loaded in bf16) so the cuDNN GRU sees matching tensors. This
        # is data movement, not parameter movement — unrelated to the ZeRO-3 rule
        # against `.to(device)` on parameters during construction.
        p = next(self.trajectory_gru.parameters())
        gru_features = gru_features.to(device=p.device, dtype=p.dtype)
        lengths = gru_lengths.clamp(min=1, max=gru_features.size(1))
        hidden = self.trajectory_gru.encode_sequence(gru_features, lengths)  # (N, max_t, H)
        last_idx = (lengths - 1).to(dtype=torch.long, device=hidden.device)
        rows = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[rows, last_idx, :]  # (N, H)

    def _inject_gru(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        gru_features: torch.Tensor,
        gru_lengths: torch.Tensor,
        has_gru: Optional[torch.Tensor],
        gru_token_id: int,
    ) -> torch.Tensor:
        """Replace each nav sample's single <gru> embedding with its projected
        trajectory vector (matches the alignment textonly scheme). Equal length
        keeps mrope position_ids valid; non-nav rows (has_gru=0) are untouched.

        gru_features: (S, T, 7) — one trajectory per sample. The collator stacks a
        singleton slot as (S, 1, T, 7), which we collapse here.

        Works for BOTH collators: padded (input_ids [B, L], one nav row = one <gru>)
        and flattened/packed (input_ids [1, total_len], all samples concatenated so
        the k-th <gru> in the stream belongs to the k-th nav sample). Each nav
        sample's <gru> gets ITS OWN projected trajectory vector.
        """
        if gru_features.dim() == 4:
            gru_features = gru_features[:, 0]                              # (S, T, 7)
            gru_lengths = gru_lengths[:, 0] if gru_lengths.dim() == 2 else gru_lengths

        # frozen GRU -> last valid state -> trainable projector -> one vec / sample
        last = self._encode_gru_last_state(gru_features, gru_lengths)     # (S, 256)
        projected = self.projector(last).to(inputs_embeds.dtype)         # (S, H)

        # <gru> positions in the (possibly packed) token stream.
        mask = input_ids == gru_token_id                                 # (rows, L)
        n_ph = int(mask.sum())
        if n_ph == 0:
            return inputs_embeds                                         # all-QA batch

        # Replacement vectors = the nav samples' projected vectors, in sample order.
        # Non-nav samples emit no <gru>, so placeholders occur once per nav sample in
        # order -> they line up row-major with projected[has_gru]. Selecting by has_gru
        # (instead of masking input rows) is what makes packed [1, L] batches work too.
        if has_gru is not None:
            src = projected[has_gru.bool().to(projected.device)]         # (n_nav, H)
        else:
            src = projected
        if src.shape[0] != n_ph:
            raise ValueError(
                f"<gru> placeholder count ({n_ph}) != nav-sample count ({src.shape[0]}); "
                "each nav sample must carry exactly one <gru>."
            )

        # Clone (avoid in-place on the embedding graph), then write each <gru> row.
        out = inputs_embeds.clone()
        out[mask] = src
        return out

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        gru_features: Optional[torch.Tensor] = None,
        gru_lengths: Optional[torch.Tensor] = None,
        has_gru: Optional[torch.Tensor] = None,
        motion_token_id: Optional[int] = None,
        **kwargs,
    ):
        # Build text embeddings ourselves so we can splice in GRU vectors; the
        # backbone then recovers image positions from inputs_embeds (input_ids=None).
        inputs_embeds = self.qwen.get_input_embeddings()(input_ids)

        if gru_features is not None and gru_lengths is not None:
            tok_id = int(motion_token_id) if motion_token_id is not None else self.gru_token_id
            inputs_embeds = self._inject_gru(
                inputs_embeds, input_ids, gru_features, gru_lengths, has_gru, tok_id
            )

        outputs: CausalLMOutputWithPast = self.qwen(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            use_cache=False,
        )
        return outputs
