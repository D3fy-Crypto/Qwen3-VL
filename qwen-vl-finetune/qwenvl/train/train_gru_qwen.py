"""
Training script for GRU-Qwen model (Trajectory + Language).

Usage:
    torchrun --nproc_per_node=1 train_gru_qwen.py \
        --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
        --gru_checkpoint_path /path/to/best_model.pt \
        --dataset_use trajectory_dataset \
        --output_dir ./output
"""

import os
import logging   
import torch
import transformers
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from qwenvl.models.gru_qwen import GRUQwenModel
from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
from transformers import Trainer, AutoProcessor, TrainerCallback

local_rank = None


class InferenceSnapshotCallback(TrainerCallback):
    """Saves inference-only model weights every N optimizer steps.

    Writes output_dir/inference-step-{N:06d}/model.safetensors (+ config.json).
    Optimizer and scheduler state are NOT saved. Works with DeepSpeed ZeRO.
    """

    def __init__(self, output_dir: str, every_n_steps: int = 100):
        self.output_dir = output_dir
        self.every_n_steps = every_n_steps
        self._trainer = None

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_n_steps <= 0:
            return
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return
        if self._trainer is None:
            return
        snapshot_dir = os.path.join(self.output_dir, f"inference-step-{state.global_step:06d}")
        self._trainer.save_model(snapshot_dir)
        if args.should_save:
            print(f"[InferenceSnapshot] step {state.global_step} -> {snapshot_dir}", flush=True)


def rank0_print(*args):
    if local_rank in (None, -1, 0):
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


@dataclass
class GRUQwenTrainingArguments(TrainingArguments):
    """Extended training arguments for GRU-Qwen model."""
    gru_checkpoint_path: str = field(default=None)
    gru_qwen_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a Trainer-saved GRU-Qwen checkpoint directory or model.safetensors "
                "used to warm-start the full wrapper."
            )
        },
    )
    qwen_base_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Base Qwen model path used for config/processor/model architecture when "
                "--model_name_or_path points at a GRU-Qwen checkpoint directory."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional tokenizer path. Defaults to the GRU-Qwen checkpoint tokenizer if present."},
    )
    alignment_modules_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a GRU-Qwen Trainer checkpoint used to warm-start only "
                "trajectory_gru + projector weights (Qwen backbone is not loaded from it). "
                "Compatible with DeepSpeed ZeRO-3."
            )
        },
    )
    projector_k: int = field(default=1)
    tune_projector: bool = field(default=True)
    tune_qwen_vision: bool = field(default=False)
    tune_qwen_lm: bool = field(default=False)
    qwen_lm_unfreeze_last_n_layers: int = field(default=0)
    qwen_unfreeze_lm_head: bool = field(default=False)
    alignment_strict: bool = field(default=True)


def _looks_like_trainer_checkpoint(path: str) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and (candidate / "model.safetensors").exists()


def _has_hf_config(path: str) -> bool:
    candidate = Path(path)
    return not candidate.exists() or (candidate / "config.json").exists()


def _default_local_qwen_base() -> Optional[str]:
    candidate = project_root.parents[1] / "qwen_models" / "instruct"
    return str(candidate) if (candidate / "config.json").exists() else None


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, GRUQwenTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        if local_rank is not None and local_rank >= 0:
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cuda"
    else:
        device = "cpu"
    dtype = torch.bfloat16 if training_args.bf16 else (
        torch.float16 if training_args.fp16 else None
    )

    qwen_model_path = model_args.model_name_or_path
    gru_qwen_checkpoint_path = training_args.gru_qwen_checkpoint_path
    alignment_modules_checkpoint_path = training_args.alignment_modules_checkpoint_path

    if gru_qwen_checkpoint_path and alignment_modules_checkpoint_path:
        raise ValueError(
            "Pass either --gru_qwen_checkpoint_path (full wrapper warm-start) "
            "or --alignment_modules_checkpoint_path (projector/GRU only), not both."
        )

    if not _has_hf_config(qwen_model_path) and _looks_like_trainer_checkpoint(qwen_model_path):
        gru_qwen_checkpoint_path = gru_qwen_checkpoint_path or qwen_model_path
        qwen_model_path = training_args.qwen_base_model_path or _default_local_qwen_base()
        if qwen_model_path is None:
            raise ValueError(
                "--model_name_or_path points to a GRU-Qwen checkpoint without config.json. "
                "Pass --qwen_base_model_path with the original/base Qwen model directory."
            )

    if gru_qwen_checkpoint_path and getattr(training_args, "deepspeed", None):
        raise ValueError(
            "Loading --gru_qwen_checkpoint_path uses a normal full-state_dict warm-start and "
            "is not compatible with DeepSpeed ZeRO-3 initialization. Set USE_DEEPSPEED=0 "
            "or remove --deepspeed for this checkpoint warm-start."
        )

    tokenizer_path = training_args.tokenizer_name_or_path
    if tokenizer_path is None and gru_qwen_checkpoint_path:
        checkpoint_dir = Path(gru_qwen_checkpoint_path)
        if checkpoint_dir.is_file():
            checkpoint_dir = checkpoint_dir.parent
        if (checkpoint_dir / "tokenizer.json").exists():
            tokenizer_path = str(checkpoint_dir)
    tokenizer_path = tokenizer_path or qwen_model_path

    rank0_print(f"[GRU-Qwen] Base Qwen model: {qwen_model_path}")
    if gru_qwen_checkpoint_path:
        rank0_print(f"[GRU-Qwen] GRU-Qwen checkpoint: {gru_qwen_checkpoint_path}")
    rank0_print(f"[GRU-Qwen] Tokenizer: {tokenizer_path}")

    model_config = transformers.AutoConfig.from_pretrained(
        qwen_model_path, trust_remote_code=True
    )
    model_type = getattr(model_config, "model_type", "") or ""
    model_type_lower = model_type.lower()

    rank0_print(f"\n[GRU-Qwen Training] Initializing model...")
    model = GRUQwenModel(
        qwen_model_id=qwen_model_path,
        gru_checkpoint_path=training_args.gru_checkpoint_path,
        projector_k=training_args.projector_k,
        device=device,
        dtype=dtype,
        tune_qwen_vision=training_args.tune_qwen_vision,
        tune_qwen_lm=training_args.tune_qwen_lm,
        tune_projector=training_args.tune_projector,
        qwen_lm_unfreeze_last_n_layers=training_args.qwen_lm_unfreeze_last_n_layers,
        qwen_unfreeze_lm_head=training_args.qwen_unfreeze_lm_head,
    )

    rank0_print(f"[GRU-Qwen] Model initialized: {model.__class__.__name__}")
    rank0_print(f"[GRU-Qwen] Device: {device}, dtype: {dtype}")
    rank0_print("[GRU-Qwen] Model architecture:")
    rank0_print(model)

    qwen_path_lower = qwen_model_path.lower()
    if "qwen3" in model_type_lower or "qwen3" in qwen_path_lower:
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_type_lower or "qwen2_5" in model_type_lower or "qwen2.5" in qwen_path_lower:
        data_args.model_type = "qwen2.5vl"
    else:
        data_args.model_type = "qwen2vl"

    rank0_print(
        f"[GRU-Qwen] Resolved model_type={model_type!r} -> data_args.model_type={data_args.model_type}"
    )

    processor = AutoProcessor.from_pretrained(qwen_model_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    motion_token_text = getattr(data_args, "motion_token_text", "<motion>")
    added_motion_tokens = 0
    if tokenizer.convert_tokens_to_ids(motion_token_text) == tokenizer.unk_token_id:
        added_motion_tokens = tokenizer.add_special_tokens(
            {"additional_special_tokens": [motion_token_text]}
        )
    motion_token_id = tokenizer.convert_tokens_to_ids(motion_token_text)
    if added_motion_tokens > 0:
        model.qwen.resize_token_embeddings(len(tokenizer))
        rank0_print(
            f"[GRU-Qwen] Added special motion token {motion_token_text} with id={motion_token_id}"
        )
    else:
        rank0_print(
            f"[GRU-Qwen] Using existing motion token {motion_token_text} with id={motion_token_id}"
        )
    model.motion_token_id = int(motion_token_id)

    # Important: resize_token_embeddings can recreate embedding params with
    # requires_grad=True. Re-apply freeze/train policy so projector-only runs
    # stay projector-only.
    model._set_trainable_parameters()

    if gru_qwen_checkpoint_path:
        checkpoint_vocab_size = GRUQwenModel.checkpoint_vocab_size(gru_qwen_checkpoint_path)
        if checkpoint_vocab_size is not None:
            current_vocab_size = int(model.qwen.get_input_embeddings().weight.shape[0])
            if checkpoint_vocab_size != current_vocab_size:
                rank0_print(
                    "[GRU-Qwen] Resizing Qwen token embeddings to checkpoint vocab "
                    f"{checkpoint_vocab_size} (current={current_vocab_size})"
                )
                model.qwen.resize_token_embeddings(checkpoint_vocab_size)

    # Ensure dataset tokenization uses the same tokenizer instance (with motion token added).
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer

    if gru_qwen_checkpoint_path:
        model.load_gru_qwen_checkpoint(gru_qwen_checkpoint_path, strict=False)
    elif alignment_modules_checkpoint_path:
        model.load_alignment_modules_from_checkpoint(
            alignment_modules_checkpoint_path, strict=False
        )

    # Re-apply trainability after any checkpoint load to guarantee the intended
    # freeze/unfreeze policy (and avoid surprises from wrapper reload paths).
    model._set_trainable_parameters()

    # Enable gradient-checkpointing input-grad hook only when Qwen has trainable params.
    # In projector-only alignment runs, forcing embedding grads can accidentally unfreeze
    # a huge embedding matrix.
    qwen_has_trainable = any(p.requires_grad for p in model.qwen.parameters())
    requested_qwen_unfreeze = (
        training_args.tune_qwen_lm
        or training_args.qwen_lm_unfreeze_last_n_layers > 0
        or training_args.qwen_unfreeze_lm_head
        or training_args.tune_qwen_vision
    )
    if requested_qwen_unfreeze and (not qwen_has_trainable):
        raise RuntimeError(
            "Requested Qwen unfreeze, but no Qwen parameters are trainable. "
            "Check model wrapping/path compatibility before launching DeepSpeed."
        )
    if training_args.gradient_checkpointing and getattr(training_args, "deepspeed", None):
        rank0_print(
            "[GRU-Qwen] Disabling gradient checkpointing for DeepSpeed run "
            "to avoid ZeRO recompute metadata mismatch in this wrapper path."
        )
        training_args.gradient_checkpointing = False
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    if training_args.gradient_checkpointing and qwen_has_trainable:
        if hasattr(model.qwen, "enable_input_require_grads"):
            model.qwen.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.qwen.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    elif training_args.gradient_checkpointing and (not qwen_has_trainable):
        rank0_print(
            "[GRU-Qwen] Skipping Qwen input-grad hook for gradient checkpointing "
            "because Qwen params are frozen (projector-only mode)."
        )

    # Use LoRA if enabled
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType

        rank0_print("[GRU-Qwen] Enabling LoRA...")
        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    # Print trainable parameters
    if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        rank0_print()
        model.print_trainable_parameters()
        # Auto-disable strict validation for unified model loading (no gru_qwen_checkpoint_path)
        use_strict = training_args.alignment_strict and (gru_qwen_checkpoint_path or False)
        stats = model.validate_alignment_setup(strict=use_strict)
        rank0_print("[GRU-Qwen][alignment-check]")
        rank0_print(f"  total Qwen params: {stats['qwen_total']:,}")
        rank0_print(f"  trainable Qwen params: {stats['qwen_trainable']:,}")
        rank0_print(f"  projector trainable params: {stats['projector_trainable']:,}")
        rank0_print(f"  GRU trainable params: {stats['trajectory_gru_trainable']:,}")

    rank0_print("[GRU-Qwen] Loading dataset...")
    from qwenvl.data.data_processor_gru import make_supervised_data_module
    data_module = make_supervised_data_module(processor, data_args=data_args)
    if data_module.get("train_dataset") is None or data_module.get("data_collator") is None:
        raise RuntimeError("GRU-Qwen training requires non-empty train_dataset and data_collator")
    rank0_print(f"[GRU-Qwen] Dataset size: {len(data_module['train_dataset'])}")

    # Initialize trainer
    rank0_print("[GRU-Qwen] Initializing Trainer...")
    snapshot_cb = InferenceSnapshotCallback(
        output_dir=training_args.output_dir,
        every_n_steps=training_args.inference_snapshot_steps,
    )
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[snapshot_cb],
        **data_module
    )
    snapshot_cb._trainer = trainer

    # Resume from checkpoint if available
    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("[GRU-Qwen] Found checkpoint, resuming training...")
        trainer.train(resume_from_checkpoint=True)
    else:
        rank0_print("[GRU-Qwen] Starting fresh training...")
        trainer.train()

    trainer.save_state()
    model.qwen.config.use_cache = True

    rank0_print("[GRU-Qwen] Saving final model...")
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)

    rank0_print("[GRU-Qwen] Training complete!")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
