import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict

from qwenvl.models.projector import ProjectorMLP
from transformers import AutoConfig


class TrajectoryGRUSFTEncoder(nn.Module):
    """
    GRU encoder for the GRU-SFT modality.
    Input: one-hot encoded action sequences [B, T, 4].
    Output: hidden state sequences [B, T, hidden_dim].
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def encode_sequence(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        padded_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return padded_out  # [B, T, hidden_dim]


class GRUSFTQwenModel(nn.Module):
    """
    Qwen VL model augmented with a GRU action-sequence encoder.

    Architecture:
        gru_features [B, T, 4] (one-hot actions)
        → TrajectoryGRUSFTEncoder → [B, T, 256]
        → ProjectorMLP (256 → 1024 → qwen_hidden) → [B, T, qwen_hidden]
        → prepend to text+image embeddings
        → Qwen forward → LM loss
    """

    def __init__(
        self,
        qwen_model_id: str,
        projector_k: int = 1,
        motion_token_id: Optional[int] = None,
        device: str = "cuda",
        dtype=None,
        tune_qwen_vision: bool = False,
        tune_qwen_lm: bool = True,
        tune_projector: bool = True,
    ):
        super().__init__()

        self.device_str = device
        self._debug_once = False
        self.motion_token_id = motion_token_id

        # Load Qwen model
        self._load_qwen_model(qwen_model_id, device, dtype)

        # GRU encoder (input_dim=4 for one-hot {0,1,2,3})
        self.trajectory_gru_sft = TrajectoryGRUSFTEncoder(input_dim=4, hidden_dim=256).to(device)

        # Projector MLP: 256 → 1024 → qwen_hidden
        qwen_hidden_dim = getattr(self.qwen.config, "hidden_size", None)
        if qwen_hidden_dim is None:
            # nested text_config
            text_cfg = getattr(self.qwen.config, "text_config", None)
            if text_cfg is not None:
                qwen_hidden_dim = getattr(text_cfg, "hidden_size", None)
        if qwen_hidden_dim is None:
            qwen_hidden_dim = self.qwen.get_input_embeddings().weight.shape[1]

        self.projector = ProjectorMLP(
            gru_hidden_dim=256,
            qwen_hidden_dim=qwen_hidden_dim,
            intermediate_dim=1024,
            k=max(1, int(projector_k)),
        ).to(device)

        self._set_trainable_parameters(tune_qwen_vision, tune_qwen_lm, tune_projector)

    # ------------------------------------------------------------------

    def _load_qwen_model(self, model_id: str, device: str, dtype):
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_type = getattr(config, "model_type", "").lower()

        if "qwen3" in model_type or "qwen3" in model_id.lower():
            from transformers import Qwen3VLForConditionalGeneration
            model_cls = Qwen3VLForConditionalGeneration
        elif "qwen2_5" in model_type or "qwen2.5" in model_id.lower():
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            from transformers import Qwen2VLForConditionalGeneration
            model_cls = Qwen2VLForConditionalGeneration

        kwargs = {"trust_remote_code": True, "attn_implementation": "flash_attention_2"}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        self.qwen = model_cls.from_pretrained(model_id, **kwargs).to(device)

    def _set_trainable_parameters(self, tune_qwen_vision: bool, tune_qwen_lm: bool, tune_projector: bool):
        for p in self.qwen.parameters():
            p.requires_grad = False

        inner = self.qwen.model if not hasattr(self.qwen, "visual") else self.qwen
        if tune_qwen_vision and hasattr(inner, "visual"):
            for p in inner.visual.parameters():
                p.requires_grad = True

        if tune_qwen_lm:
            for p in self.qwen.parameters():
                p.requires_grad = True

        for p in self.trajectory_gru_sft.parameters():
            p.requires_grad = True

        if not tune_projector:
            for p in self.projector.parameters():
                p.requires_grad = False
        else:
            for p in self.projector.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------

    def load_warmstart_weights(self, safetensors_path: str):
        """
        Load Qwen backbone weights from a previously saved model.safetensors.

        Handles two key formats:
          - Keys prefixed with "qwen." (saved as GRUSFTQwenModel / GRUQwenModel state dict)
          - Raw Qwen model keys (saved directly from a HF Qwen model)

        GRU encoder and projector keys are expected to be missing (randomly initialized).
        """
        from safetensors.torch import load_file

        path = Path(safetensors_path)
        if not path.exists():
            print(f"[GRU-SFT] Warm-start checkpoint not found: {path}. Skipping.")
            return

        weights = load_file(str(path))

        has_qwen_prefix = any(k.startswith("qwen.") for k in weights)
        if has_qwen_prefix:
            # Strip "qwen." prefix and load into self.qwen
            qwen_weights = {k.removeprefix("qwen."): v for k, v in weights.items() if k.startswith("qwen.")}
            missing, unexpected = self.qwen.load_state_dict(qwen_weights, strict=False)
        else:
            # Assume raw Qwen model keys
            missing, unexpected = self.qwen.load_state_dict(weights, strict=False)

        print(
            f"[GRU-SFT] Warm-start loaded {len(weights)} keys. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}."
        )
        if missing:
            print(f"[GRU-SFT]   Missing (first 5): {missing[:5]}")
        if unexpected:
            print(f"[GRU-SFT]   Unexpected (first 5): {unexpected[:5]}")

    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.qwen, "gradient_checkpointing_enable"):
            self.qwen.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self):
        if hasattr(self.qwen, "gradient_checkpointing_disable"):
            self.qwen.gradient_checkpointing_disable()

    # ------------------------------------------------------------------

    def forward(
        self,
        gru_features: torch.Tensor,
        gru_lengths: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        dev = next(self.parameters()).device

        gru_features = gru_features.to(dev)
        gru_lengths = gru_lengths.to(dev)
        if input_ids is not None:
            input_ids = input_ids.to(dev)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)
        if labels is not None:
            labels = labels.to(dev)

        # Encode GRU sequence → [B, T_gru, hidden]
        gru_hidden = self.trajectory_gru_sft.encode_sequence(gru_features, gru_lengths)
        # Project → [B, T_gru, qwen_hidden]
        projected = self.projector(gru_hidden)

        # Get text (+ vision token) embeddings
        if input_ids is not None:
            input_embeds = self.qwen.get_input_embeddings()(input_ids)
            projected = projected.to(dtype=input_embeds.dtype, device=input_embeds.device)
        else:
            input_embeds = None

        labels_for_model = labels.clone() if labels is not None else None

        if input_embeds is not None:
            combined_embeds = input_embeds
            combined_attention = attention_mask if attention_mask is not None else torch.ones(
                input_embeds.shape[:2], device=dev, dtype=torch.long
            )

            placement_stats = []
            for b in range(combined_embeds.shape[0]):
                seq_len = int(gru_lengths[b].item())
                seq_len = max(1, min(seq_len, projected.shape[1]))

                if self.motion_token_id is not None and self.motion_token_id >= 0 and input_ids is not None:
                    motion_positions = (input_ids[b] == int(self.motion_token_id)).nonzero(as_tuple=False).squeeze(-1)
                else:
                    motion_positions = torch.empty(0, dtype=torch.long, device=combined_embeds.device)

                if motion_positions.numel() == 0:
                    valid_positions = (
                        combined_attention[b].nonzero(as_tuple=False).squeeze(-1)
                        if combined_attention is not None
                        else torch.arange(combined_embeds.shape[1], device=combined_embeds.device)
                    )
                    motion_positions = valid_positions[:seq_len]

                use_n = min(int(motion_positions.numel()), seq_len)
                if use_n > 0:
                    use_pos = motion_positions[:use_n]
                    combined_embeds[b, use_pos, :] = projected[b, :use_n, :]
                    if labels_for_model is not None:
                        labels_for_model[b, use_pos] = -100

                placement_stats.append((int(motion_positions.numel()), use_n, seq_len))
        else:
            combined_embeds = projected
            combined_attention = torch.ones(
                projected.shape[:2], device=dev, dtype=torch.long
            )

        model_kwargs: Dict = {
            "inputs_embeds": combined_embeds,
            "attention_mask": combined_attention,
        }

        if labels_for_model is not None:
            model_kwargs["labels"] = labels_for_model

        if pixel_values is not None:
            model_kwargs["pixel_values"] = pixel_values.to(dev)
        if image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = image_grid_thw.to(dev)
        if pixel_values_videos is not None:
            model_kwargs["pixel_values_videos"] = pixel_values_videos.to(dev)
        if video_grid_thw is not None:
            model_kwargs["video_grid_thw"] = video_grid_thw.to(dev)

        if not self._debug_once:
            print(
                f"[GRU-SFT][debug] gru_features={tuple(gru_features.shape)} "
                f"projected={tuple(projected.shape)} combined={tuple(combined_embeds.shape)}"
            )
            if input_ids is not None:
                print(f"[GRU-SFT][debug] motion_token_id={self.motion_token_id}")
                print(f"[GRU-SFT][debug] placement_stats=(found,use,gru_len) {placement_stats}")

        outputs = self.qwen(**model_kwargs)

        if not self._debug_once and hasattr(outputs, "logits"):
            print(f"[GRU-SFT][debug] logits={tuple(outputs.logits.shape)}")
            self._debug_once = True

        return {
            "loss": outputs.loss if hasattr(outputs, "loss") else None,
            "logits": outputs.logits if hasattr(outputs, "logits") else None,
        }

    # ------------------------------------------------------------------

    def print_trainable_parameters(self):
        def _count(mod):
            total = sum(p.numel() for p in mod.parameters())
            trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            return total, trainable

        qt, qtr = _count(self.qwen)
        gt, gtr = _count(self.trajectory_gru_sft)
        pt, ptr = _count(self.projector)
        total = qt + gt + pt
        trainable = qtr + gtr + ptr

        print("\n[GRU-SFT] Trainable Parameters:")
        print(f"  qwen:               total={qt:,}  trainable={qtr:,}")
        print(f"  trajectory_gru_sft: total={gt:,}  trainable={gtr:,}")
        print(f"  projector:          total={pt:,}  trainable={ptr:,}")
        print(f"  all:                total={total:,}  trainable={trainable:,}\n")
