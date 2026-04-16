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

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from qwenvl.models.gru_qwen import GRUQwenModel
from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
from transformers import Trainer, AutoProcessor

local_rank = None


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
    projector_k: int = field(default=1)
    tune_projector: bool = field(default=True)
    tune_qwen_vision: bool = field(default=False)
    tune_qwen_lm: bool = field(default=False)
    alignment_strict: bool = field(default=True)


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, GRUQwenTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if training_args.bf16 else (
        torch.float16 if training_args.fp16 else None
    )

    model_config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    model_type = getattr(model_config, "model_type", "") or ""
    model_type_lower = model_type.lower()

    rank0_print(f"\n[GRU-Qwen Training] Initializing model...")
    model = GRUQwenModel(
        qwen_model_id=model_args.model_name_or_path,
        gru_checkpoint_path=training_args.gru_checkpoint_path,
        projector_k=training_args.projector_k,
        device=device,
        dtype=dtype,
        tune_qwen_vision=training_args.tune_qwen_vision,
        tune_qwen_lm=training_args.tune_qwen_lm,
        tune_projector=training_args.tune_projector,
    )

    rank0_print(f"[GRU-Qwen] Model initialized: {model.__class__.__name__}")
    rank0_print(f"[GRU-Qwen] Device: {device}, dtype: {dtype}")

    if "qwen3" in model_type_lower or "qwen3" in model_args.model_name_or_path.lower():
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_type_lower or "qwen2_5" in model_type_lower or "qwen2.5" in model_args.model_name_or_path.lower():
        data_args.model_type = "qwen2.5vl"
    else:
        data_args.model_type = "qwen2vl"

    rank0_print(
        f"[GRU-Qwen] Resolved model_type={model_type!r} -> data_args.model_type={data_args.model_type}"
    )

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Enable gradient checkpointing if requested
    if training_args.gradient_checkpointing:
        if hasattr(model.qwen, "enable_input_require_grads"):
            model.qwen.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.qwen.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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
        stats = model.validate_alignment_setup(strict=training_args.alignment_strict)
        rank0_print("[GRU-Qwen][alignment-check]")
        rank0_print(f"  total Qwen params: {stats['qwen_total']:,}")
        rank0_print(f"  trainable Qwen params: {stats['qwen_trainable']:,}")
        rank0_print(f"  projector trainable params: {stats['projector_trainable']:,}")
        rank0_print(f"  GRU trainable params: {stats['trajectory_gru_trainable']:,}")

    rank0_print("[GRU-Qwen] Loading dataset...")
    from qwenvl.data.data_processor import make_supervised_data_module
    data_module = make_supervised_data_module(processor, data_args=data_args)
    if data_module.get("train_dataset") is None or data_module.get("data_collator") is None:
        raise RuntimeError("GRU-Qwen training requires non-empty train_dataset and data_collator")
    rank0_print(f"[GRU-Qwen] Dataset size: {len(data_module['train_dataset'])}")

    # Initialize trainer
    rank0_print("[GRU-Qwen] Initializing Trainer...")
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module
    )

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
