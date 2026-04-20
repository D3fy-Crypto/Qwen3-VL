import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict

from qwenvl.models.projector import ProjectorMLP
from transformers import AutoConfig, AutoModelForCausalLM


class TrajectoryGRUEncoder(nn.Module):
    """Notebook-compatible GRU encoder used for trajectory action features."""

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


class GRUQwenModel(nn.Module):
    """
    Combines GRU trajectory encoder with Qwen VL model via a learned projector.
    
        Architecture:
         action features (B, T, 7) -> frozen notebook GRU (B, T, 256)
         -> projector MLP (256 -> 1024 -> qwen_hidden)
         -> prepend projected trajectory tokens to text embeddings
         -> Qwen LM logits/loss
    """

    def __init__(self, 
                 qwen_model_id: str,
                 gru_checkpoint_path: Optional[str] = None,
                 projector_k: int = 1,
                 motion_token_id: Optional[int] = None,
                 device: str = "cuda",
                 dtype = None,
                 tune_qwen_vision: bool = False,
                 tune_qwen_lm: bool = False,
                 tune_projector: bool = True):
        """
        Args:
            qwen_model_id: HuggingFace model ID for Qwen (e.g., "Qwen/Qwen2.5-VL-7B")
            gru_checkpoint_path: Path to notebook GRU checkpoint (best_model.pt)
            projector_k: K value for projector output dimension (K * 4096)
            device: Device to load models on
            dtype: Data type for model (torch.float16, torch.bfloat16, etc.)
            tune_qwen_vision: Whether to train Qwen vision encoder (if applicable)
            tune_qwen_lm: Whether to train Qwen language model
            tune_projector: Whether to train projector MLP
        """
        super().__init__()
        
        self.qwen_model_id = qwen_model_id
        self.gru_checkpoint_path = gru_checkpoint_path
        self.projector_k = max(1, int(projector_k))
        self.device = device
        self.tune_qwen_vision = tune_qwen_vision
        self.tune_qwen_lm = tune_qwen_lm
        self.tune_projector = tune_projector
        self.motion_token_id = motion_token_id
        
        # Load Qwen model first to derive hidden size.
        self._load_qwen_model(qwen_model_id, device, dtype)

        # Build notebook-compatible GRU encoder and load frozen checkpoint weights.
        self.trajectory_gru = TrajectoryGRUEncoder(input_dim=7, hidden_dim=256, embedding_dim=128).to(device)
        self.gru_checkpoint_meta = {}
        if gru_checkpoint_path:
            checkpoint_path = Path(gru_checkpoint_path)
            if checkpoint_path.exists():
                ckpt = torch.load(checkpoint_path, map_location=device)
                state_dict = ckpt.get("model_state_dict", ckpt)
                missing, unexpected = self.trajectory_gru.load_state_dict(state_dict, strict=False)
                self.gru_checkpoint_meta = {
                    "keys": list(ckpt.keys()) if isinstance(ckpt, dict) else [],
                    "missing": missing,
                    "unexpected": unexpected,
                }
                print(f"[GRU-Qwen] Loaded GRU checkpoint from {checkpoint_path}")
            else:
                print(
                    f"[GRU-Qwen] GRU checkpoint not found at {checkpoint_path}; using randomly initialized encoder"
                )
        for param in self.trajectory_gru.parameters():
            param.requires_grad = False
        self.trajectory_gru.eval()

        qwen_hidden_dim = getattr(self.qwen.config, "hidden_size", None)
        if qwen_hidden_dim is None:
            qwen_hidden_dim = self.qwen.get_input_embeddings().weight.shape[1]
        
        # Project GRU hidden states into Qwen token embedding space.
        self.projector = ProjectorMLP(
            gru_hidden_dim=self.trajectory_gru.hidden_dim,
            qwen_hidden_dim=qwen_hidden_dim,
            intermediate_dim=1024,
            k=self.projector_k,
        ).to(device)
        self._debug_once = False
        
        # Set trainable parameters
        self._set_trainable_parameters()

    def _load_qwen_model(self, model_id: str, device: str, dtype):
        """Load appropriate Qwen model variant."""
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        # Try different model classes for compatibility
        model_class = None
        if "qwen3" in model_id.lower() or getattr(config, "model_type", "") == "qwen3_vl":
            try:
                from transformers import Qwen3VLForConditionalGeneration
                model_class = Qwen3VLForConditionalGeneration
            except ImportError:
                pass
        
        if model_class is None and "qwen2.5" in model_id.lower():
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                model_class = Qwen2_5_VLForConditionalGeneration
            except ImportError:
                pass
        
        if model_class is None:
            try:
                from transformers import Qwen2VLForConditionalGeneration
                model_class = Qwen2VLForConditionalGeneration
            except ImportError:
                model_class = AutoModelForCausalLM
        
        # Load with dtype casting
        kwargs = {"trust_remote_code": True}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        
        self.qwen = model_class.from_pretrained(model_id, **kwargs).to(device)

    def _set_trainable_parameters(self):
        """Configure which parameters should be trained."""
        # Freeze Qwen entirely by default.
        for param in self.qwen.parameters():
            param.requires_grad = False
        
        # Unfreeze vision encoder if requested
        if self.tune_qwen_vision and hasattr(self.qwen, 'visual'):
            for param in self.qwen.visual.parameters():
                param.requires_grad = True
        
        # Unfreeze language model if requested
        if self.tune_qwen_lm and hasattr(self.qwen, 'language_model'):
            for param in self.qwen.language_model.parameters():
                param.requires_grad = True
        
        # Projector is trainable by default
        if not self.tune_projector:
            for param in self.projector.parameters():
                param.requires_grad = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Proxy HF Trainer gradient-checkpointing calls to the underlying Qwen model."""
        if hasattr(self.qwen, "gradient_checkpointing_enable"):
            self.qwen.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self):
        """Proxy HF Trainer gradient-checkpointing disable calls to the underlying Qwen model."""
        if hasattr(self.qwen, "gradient_checkpointing_disable"):
            self.qwen.gradient_checkpointing_disable()

    def forward(self,
                gru_features: torch.Tensor,
                gru_lengths: torch.Tensor,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.Tensor] = None,
                image_grid_thw: Optional[torch.Tensor] = None,
                pixel_values_videos: Optional[torch.Tensor] = None,
                video_grid_thw: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs):
        """
        Forward pass through GRU-Qwen model.
        
        Args:
            gru_features: Trajectory features, either (batch_size, traj_seq_len, 7)
                or per-slot prefixes (batch_size, num_slots, prefix_len, 7)
            gru_lengths: Valid lengths matching gru_features shape, either (batch_size,)
                or per-slot lengths (batch_size, num_slots)
            input_ids: Text token IDs (batch_size, text_seq_len)
            attention_mask: Attention mask for text tokens (batch_size, text_seq_len)
            pixel_values: Vision input (for multimodal Qwen models)
            image_grid_thw: Image grid info (for some Qwen variants)
            labels: Target token IDs for loss computation (optional)
            
        Returns:
            Dict with loss/logits for Trainer compatibility
        """
        if gru_features is None or gru_lengths is None:
            raise ValueError("gru_features and gru_lengths are required for GRU-Qwen training")

        gru_features = gru_features.to(self.device)
        gru_lengths = gru_lengths.to(self.device)
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        with torch.no_grad():
            if gru_features.dim() == 4:
                bsz, num_slots, max_t, feat_dim = gru_features.shape
                flat_features = gru_features.reshape(bsz * num_slots, max_t, feat_dim)
                flat_lengths = gru_lengths.reshape(bsz * num_slots)
                flat_lengths = flat_lengths.clamp(min=1, max=max_t)

                flat_hidden = self.trajectory_gru.encode_sequence(flat_features, flat_lengths)
                flat_last_idx = (flat_lengths - 1).to(dtype=torch.long)
                flat_last = flat_hidden[
                    torch.arange(flat_hidden.shape[0], device=flat_hidden.device),
                    flat_last_idx,
                    :,
                ]

                projected = self.projector(flat_last).reshape(bsz, num_slots, -1)
                gru_hidden = flat_hidden.reshape(bsz, num_slots, max_t, -1)
            else:
                gru_hidden = self.trajectory_gru.encode_sequence(gru_features, gru_lengths)
                projected = self.projector(gru_hidden)

        if input_ids is not None:
            input_embeds = self.qwen.get_input_embeddings()(input_ids)
            projected = projected.to(dtype=input_embeds.dtype, device=input_embeds.device)
        else:
            input_embeds = None

        labels_for_model = labels.clone() if labels is not None else None

        if input_embeds is not None:
            combined_embeds = input_embeds
            combined_attention = attention_mask if attention_mask is not None else torch.ones(
                input_embeds.shape[:2], device=self.device, dtype=torch.long
            )

            placement_stats = []
            for b in range(combined_embeds.shape[0]):
                if gru_lengths.dim() == 2:
                    seq_len = int((gru_lengths[b] > 0).sum().item())
                else:
                    seq_len = int(gru_lengths[b].item())
                seq_len = max(1, min(seq_len, projected.shape[1]))

                if self.motion_token_id is not None and self.motion_token_id >= 0:
                    motion_positions = (input_ids[b] == int(self.motion_token_id)).nonzero(as_tuple=False).squeeze(-1)
                else:
                    motion_positions = torch.empty(0, dtype=torch.long, device=input_ids.device)

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
                projected.shape[:2], device=self.device, dtype=torch.long
            )
        
        model_kwargs = {
            "inputs_embeds": combined_embeds,
            "attention_mask": combined_attention,
        }
        
        if labels_for_model is not None:
            model_kwargs["labels"] = labels_for_model
        
        if pixel_values is not None:
            model_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = image_grid_thw
        if pixel_values_videos is not None:
            model_kwargs["pixel_values_videos"] = pixel_values_videos
        if video_grid_thw is not None:
            model_kwargs["video_grid_thw"] = video_grid_thw

        if not self._debug_once:
            print(
                f"[GRU-Qwen][debug] batch gru_features={tuple(gru_features.shape)} "
                f"gru_hidden={tuple(gru_hidden.shape)} projected={tuple(projected.shape)}"
            )
            if input_ids is not None:
                print(f"[GRU-Qwen][debug] motion_token_id={self.motion_token_id}")
                print(f"[GRU-Qwen][debug] placement_stats=(found,use,gru_len) {placement_stats}")
                print(f"[GRU-Qwen][debug] input_ids shape={tuple(input_ids.shape)}")
                print(f"[GRU-Qwen][debug] combined_embeds shape={tuple(combined_embeds.shape)}")
        
        outputs = self.qwen(**model_kwargs)

        if not self._debug_once and hasattr(outputs, "logits"):
            print(f"[GRU-Qwen][debug] logits shape={tuple(outputs.logits.shape)}")
            self._debug_once = True
        
        return {
            "loss": outputs.loss if hasattr(outputs, "loss") else None,
            "logits": outputs.logits if hasattr(outputs, "logits") else None,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            "attentions": outputs.attentions if hasattr(outputs, "attentions") else None,
        }

    def get_trainable_params_count(self) -> Dict[str, int]:
        """Return total and trainable parameter counts for all major modules."""
        qwen_total = sum(p.numel() for p in self.qwen.parameters())
        qwen_trainable = sum(p.numel() for p in self.qwen.parameters() if p.requires_grad)

        gru_total = sum(p.numel() for p in self.trajectory_gru.parameters())
        gru_trainable = sum(
            p.numel() for p in self.trajectory_gru.parameters() if p.requires_grad
        )

        projector_total = sum(p.numel() for p in self.projector.parameters())
        projector_trainable = sum(
            p.numel() for p in self.projector.parameters() if p.requires_grad
        )

        total_params = qwen_total + gru_total + projector_total
        total_trainable = qwen_trainable + gru_trainable + projector_trainable

        return {
            "qwen_total": qwen_total,
            "qwen_trainable": qwen_trainable,
            "trajectory_gru_total": gru_total,
            "trajectory_gru_trainable": gru_trainable,
            "projector_total": projector_total,
            "projector_trainable": projector_trainable,
            "total_params": total_params,
            "total_trainable": total_trainable,
        }

    def validate_alignment_setup(self, strict: bool = True) -> Dict[str, int]:
        """Validate module freezing expectations for alignment-style training."""
        stats = self.get_trainable_params_count()

        expects_qwen_frozen = not (self.tune_qwen_lm or self.tune_qwen_vision)
        if expects_qwen_frozen and stats["qwen_trainable"] != 0 and strict:
            raise RuntimeError(
                "Alignment validation failed: Qwen should be frozen but has "
                f"{stats['qwen_trainable']:,} trainable params"
            )

        if self.tune_projector and stats["projector_trainable"] <= 0 and strict:
            raise RuntimeError(
                "Alignment validation failed: projector is expected trainable but has 0 trainable params"
            )

        if stats["trajectory_gru_trainable"] != 0 and strict:
            raise RuntimeError(
                "Alignment validation failed: trajectory GRU is expected frozen but has "
                f"{stats['trajectory_gru_trainable']:,} trainable params"
            )

        return stats

    def print_trainable_parameters(self):
        """Print summary of trainable parameters."""
        counts = self.get_trainable_params_count()

        print("\n[GRU-Qwen] Trainable Parameters:")
        print(
            f"  - qwen: total={counts['qwen_total']:,} trainable={counts['qwen_trainable']:,}"
        )
        print(
            "  - trajectory_gru: "
            f"total={counts['trajectory_gru_total']:,} "
            f"trainable={counts['trajectory_gru_trainable']:,}"
        )
        print(
            f"  - projector: total={counts['projector_total']:,} trainable={counts['projector_trainable']:,}"
        )
        print(
            f"  - all_modules: total={counts['total_params']:,} trainable={counts['total_trainable']:,}\n"
        )
