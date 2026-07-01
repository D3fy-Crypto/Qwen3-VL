"""
ZeRO-3-native GRU-Qwen SFT entry (chang).

Mirrors `qwenvl/train/train_qwen.py` (the zero3-working backbone) but swaps in:
  - model: `Qwen3VLGRUForConditionalGeneration` (PreTrainedModel wrapper; one
    `from_pretrained($BASE_GRU_DIR)` loads backbone + aligned projector + frozen
    GRU together, zero3-aware — no `.to(device)`, no manual warm-start).
  - data: `qwenvl.data.data_processor_gru_chang` (mixed navila; `<gru>`@151669
    placeholder injection; non-nav samples have has_gru=False).
  - freeze mask: GRU frozen; projector + LM(+lm_head) + vision trainable.

BASE_GRU_DIR must be a self-contained dir (model.safetensors with keys
qwen.*/projector.*/trajectory_gru.* + config.json[vocab_size=151670] + the
tokenizer that already carries <gru>@151669). See the plan / model docstring.
"""

import os
import logging
import pathlib
import sys
from pathlib import Path

import torch
import transformers

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from qwenvl.models.gru_qwen_chang import Qwen3VLGRUForConditionalGeneration
from qwenvl.data.data_processor_gru_chang import make_supervised_data_module
from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
from transformers import AutoProcessor, Trainer, TrainerCallback

local_rank = None


class InferenceSnapshotCallback(TrainerCallback):
    """Saves inference-only model weights every N optimizer steps (ZeRO-3 safe)."""

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
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_model(model_args, model):
    """Freeze mask: GRU frozen; projector + LM(+lm_head) + vision per flags."""
    qwen = model.qwen
    inner = qwen.model  # Qwen3VLModel with .visual and .language_model

    # Trajectory GRU is always frozen (the aligned projector is the trainable head).
    for p in model.trajectory_gru.parameters():
        p.requires_grad = False

    # Projector is always trainable in SFT (warm-started, keeps training).
    for p in model.projector.parameters():
        p.requires_grad = True

    # Vision tower.
    for p in inner.visual.parameters():
        p.requires_grad = bool(model_args.tune_mm_vision)

    # Language model + lm_head.
    tune_llm = bool(model_args.tune_mm_llm)
    for p in inner.language_model.parameters():
        p.requires_grad = tune_llm
    qwen.lm_head.weight.requires_grad = tune_llm


def _count_trainable(model):
    total = sum(int(getattr(p, "ds_numel", p.numel())) for p in model.parameters())
    train = sum(int(getattr(p, "ds_numel", p.numel())) for p in model.parameters() if p.requires_grad)
    return total, train


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # The GRU framework targets Qwen3-VL (BASE_GRU_DIR is a Qwen3-VL checkpoint).
    data_args.model_type = "qwen3vl"

    model = Qwen3VLGRUForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    rank0_print(f"[GRU-Qwen-chang] loaded {model.__class__.__name__} from {model_args.model_name_or_path}")

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, inp, out):
                out.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    set_model(model_args, model)

    if local_rank in (None, -1, 0):
        total, train_n = _count_trainable(model)
        rank0_print(f"[GRU-Qwen-chang] trainable params: {train_n:,} / {total:,}")
        rank0_print(
            f"[GRU-Qwen-chang] gru_token_id={model.gru_token_id} | "
            f"projector trainable={any(p.requires_grad for p in model.projector.parameters())} | "
            f"gru trainable={any(p.requires_grad for p in model.trajectory_gru.parameters())}"
        )

    data_module = make_supervised_data_module(processor, data_args=data_args)
    snapshot_cb = InferenceSnapshotCallback(
        output_dir=training_args.output_dir,
        every_n_steps=training_args.inference_snapshot_steps,
    )
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args,
        callbacks=[snapshot_cb],
        **data_module,
    )
    snapshot_cb._trainer = trainer

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    if training_args.save_strategy != "no":
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
