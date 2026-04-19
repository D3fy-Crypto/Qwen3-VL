"""
Training script for GRU-SFT: Qwen VL with GRU action-sequence modality.

Adds a "gru" modality (one-hot encoded action sequences from {0,1,2,3}) to the
standard NaVILA SFT pipeline. The GRU encoder output is projected into Qwen's
embedding space and prepended as trajectory-memory tokens.

Usage (see scripts/slurm_sft_gru_sft.sh):
    torchrun --nproc_per_node=N \\
        qwenvl/train/train_gru_sft_qwen.py \\
        --model_name_or_path /path/to/instruct \\
        --gru_warmstart_ckpt /path/to/gru_ckpt/model.safetensors \\
        --dataset_use human \\
        --output_dir /path/to/checkpoints
"""

import logging
import os
import pathlib
import sys
import torch
import transformers
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from qwenvl.models.gru_sft_module import GRUSFTQwenModel
from qwenvl.data.data_processor_gru_sft import make_supervised_data_module_gru_sft
from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
from transformers import AutoProcessor, AutoTokenizer, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank in (None, -1, 0):
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


@dataclass
class GRUSFTTrainingArguments(TrainingArguments):
    """Extended training arguments for GRU-SFT."""
    gru_warmstart_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model.safetensors for warm-starting Qwen backbone."},
    )
    projector_k: int = field(default=1)
    tune_projector: bool = field(default=True)
    tune_qwen_vision: bool = field(default=False)
    tune_qwen_lm: bool = field(default=True)


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, GRUSFTTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if training_args.bf16 else (
        torch.float16 if training_args.fp16 else None
    )

    # Auto-detect model type for data processor
    path_lower = model_args.model_name_or_path.lower()
    if "qwen3" in path_lower:
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in path_lower:
        data_args.model_type = "qwen2.5vl"
    else:
        data_args.model_type = "qwen2vl"

    rank0_print(f"\n[GRU-SFT] Initializing model from {model_args.model_name_or_path}")
    model = GRUSFTQwenModel(
        qwen_model_id=model_args.model_name_or_path,
        projector_k=training_args.projector_k,
        device=device,
        dtype=dtype,
        tune_qwen_vision=training_args.tune_qwen_vision,
        tune_qwen_lm=training_args.tune_qwen_lm,
        tune_projector=training_args.tune_projector,
    )

    # Warm-start Qwen backbone from previously fine-tuned checkpoint
    if training_args.gru_warmstart_ckpt:
        rank0_print(f"[GRU-SFT] Loading warm-start weights from {training_args.gru_warmstart_ckpt}")
        model.load_warmstart_weights(training_args.gru_warmstart_ckpt)

    rank0_print(f"[GRU-SFT] model_type={data_args.model_type}")

    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    ):
        model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Gradient checkpointing — proxy through GRUSFTQwenModel to inner Qwen
    if training_args.gradient_checkpointing:
        if hasattr(model.qwen, "enable_input_require_grads"):
            model.qwen.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.qwen.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Optional LoRA (applied to inner Qwen only)
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        rank0_print("[GRU-SFT] Enabling LoRA on Qwen...")
        for p in model.qwen.parameters():
            p.requires_grad = False
        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model.qwen = get_peft_model(model.qwen, lora_config)

    rank0_print("[GRU-SFT] Building dataset...")
    data_module = make_supervised_data_module_gru_sft(processor, data_args=data_args)
    rank0_print(f"[GRU-SFT] Dataset size: {len(data_module['train_dataset'])}")

    rank0_print("[GRU-SFT] Initializing Trainer...")
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("[GRU-SFT] Checkpoint found, resuming training.")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.qwen.config.use_cache = True

    rank0_print("[GRU-SFT] Saving final model...")
    if training_args.save_strategy != "no":
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    rank0_print("[GRU-SFT] Done.")


if __name__ == "__main__":
    train()
