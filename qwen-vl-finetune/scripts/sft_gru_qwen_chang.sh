#!/bin/bash

# ZeRO-3-native GRU-Qwen SFT launcher (chang).
#
# Loads everything from one self-contained BASE_GRU_DIR (backbone + aligned
# projector + frozen GRU + tokenizer with <gru>@151669). See the plan.
#
# Local A6000 single-process offload sanity (tight on memory):
#   USE_DEEPSPEED=1 DEEPSPEED=./scripts/zero3_offload.json NPROC_PER_NODE=1 \
#   MAX_STEPS=2 PER_DEVICE_TRAIN_BATCH_SIZE=1 MODEL_MAX_LENGTH=1024 \
#   REPORT_TO=none DATASETS=r2r bash scripts/sft_gru_qwen_chang.sh
#
# Cluster multi-GPU (real run): set NPROC_PER_NODE=#GPU, DEEPSPEED=./scripts/zero3.json.

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

USE_DEEPSPEED=${USE_DEEPSPEED:-1}
DEEPSPEED=${DEEPSPEED:-./scripts/zero3.json}

# Single self-contained directory = backbone + projector + GRU + tokenizer(<gru>).
# Confirm/override this after the one-time config-file copy (dir is being moved).
BASE_GRU_DIR=${BASE_GRU_DIR:-/home/rithvik/IROS_proj/models_ckpts/trained/gru/base_qwen_with_gru}

# Training hyperparameters
LEARNING_RATE=${LEARNING_RATE:-1e-5}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-}
SAVE_STEPS=${SAVE_STEPS:-500}
INFERENCE_SNAPSHOT_STEPS=${INFERENCE_SNAPSHOT_STEPS:-0}
LOGGING_STEPS=${LOGGING_STEPS:-10}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-8192}
REPORT_TO=${REPORT_TO:-wandb}
WANDB_MODE=${WANDB_MODE:-online}
DATASETS=${DATASETS:-r2r,rxr,envdrop,human,scanqa}
RUN_NAME=${RUN_NAME:-gru-qwen-chang-sft}
OUTPUT_DIR=${OUTPUT_DIR:-./output_gru_qwen_chang}

# Freeze mask: GRU always frozen + projector always trained (handled in code);
# these toggle the backbone (default: unfreeze everything except GRU).
TUNE_MM_VISION=${TUNE_MM_VISION:-True}
TUNE_MM_LLM=${TUNE_MM_LLM:-True}

export WANDB_MODE

ENTRY_FILE=qwenvl/train/train_gru_qwen_chang.py

args=(
    --model_name_or_path "${BASE_GRU_DIR}"
    --dataset_use "${DATASETS}"
    --data_flatten True
    --tune_mm_vision "${TUNE_MM_VISION}"
    --tune_mm_llm "${TUNE_MM_LLM}"
    --bf16
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --max_pixels 50176
    --min_pixels 784
    --eval_strategy no
    --save_strategy steps
    --save_steps "${SAVE_STEPS}"
    --save_total_limit 2
    --inference_snapshot_steps "${INFERENCE_SNAPSHOT_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay 0.01
    --warmup_ratio 0.03
    --max_grad_norm 1.0
    --lr_scheduler_type cosine
    --logging_steps "${LOGGING_STEPS}"
    --model_max_length "${MODEL_MAX_LENGTH}"
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --run_name "${RUN_NAME}"
    --report_to "${REPORT_TO}"
)

if [[ -n "${MAX_STEPS}" ]]; then
    args+=(--max_steps "${MAX_STEPS}")
fi

if [[ "${USE_DEEPSPEED}" == "1" && -n "${DEEPSPEED}" ]]; then
    args=(--deepspeed "${DEEPSPEED}" "${args[@]}")
fi

echo "========================================"
echo "GRU-Qwen (chang) SFT"
echo "  BASE_GRU_DIR : ${BASE_GRU_DIR}"
echo "  Datasets     : ${DATASETS}"
echo "  Output       : ${OUTPUT_DIR}"
echo "  DeepSpeed    : ${USE_DEEPSPEED} (${DEEPSPEED})"
echo "  GPUs/node    : ${NPROC_PER_NODE}"
echo "  Max steps    : ${MAX_STEPS:-<full>}"
echo "========================================"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${ENTRY_FILE} "${args[@]}"
