#!/bin/bash

# GRU-Qwen Finetuning Script
# Combines trajectory GRU encoder with Qwen VL for multimodal language understanding.
#
# Usage:
#   bash scripts/sft_gru_qwen.sh
#
# Smoke-test overrides:
#   MAX_STEPS=1 PER_DEVICE_TRAIN_BATCH_SIZE=1 GRAD_ACCUM_STEPS=1 \
#   NUM_TRAIN_EPOCHS=1 REPORT_TO=none SAVE_STEPS=1 LOGGING_STEPS=1 \
#   MODEL_NAME_OR_PATH=/home/rithvik/IROS_proj/cvpr_proj/qwen_models/instruct \
#   bash scripts/sft_gru_qwen.sh

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# Optional DeepSpeed configuration. Full GRU-Qwen checkpoint warm-start is not
# compatible with ZeRO-3's empty partitioned parameters, so keep this off here.
USE_DEEPSPEED=${USE_DEEPSPEED:-0}
DEEPSPEED=${DEEPSPEED:-./scripts/zero3.json}

# Model configuration
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/home/rithvik/IROS_proj/cvpr_proj/qwen_models/instruct}
GRU_QWEN_CHECKPOINT_PATH=${GRU_QWEN_CHECKPOINT_PATH:-}
ALIGNMENT_MODULES_CHECKPOINT_PATH=${ALIGNMENT_MODULES_CHECKPOINT_PATH:-}

# GRU-specific configuration
GRU_CHECKPOINT_PATH=${GRU_CHECKPOINT_PATH:-}
PROJECTOR_K=${PROJECTOR_K:-1}
QWEN_LM_UNFREEZE_LAST_N_LAYERS=${QWEN_LM_UNFREEZE_LAST_N_LAYERS:-0}
QWEN_UNFREEZE_LM_HEAD=${QWEN_UNFREEZE_LM_HEAD:-False}

# Training hyperparameters
LEARNING_RATE=${LEARNING_RATE:-1e-4}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-8}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
MAX_STEPS=${MAX_STEPS:-}
SAVE_STEPS=${SAVE_STEPS:-500}
INFERENCE_SNAPSHOT_STEPS=${INFERENCE_SNAPSHOT_STEPS:-100}
LOGGING_STEPS=${LOGGING_STEPS:-10}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-8192}
REPORT_TO=${REPORT_TO:-wandb}
WANDB_MODE=${WANDB_MODE:-online}
DATASETS=${DATASETS:-r2r,rxr,envdrop,human,scanqa}
RUN_NAME=${RUN_NAME:-gru-qwen-baseline}
OUTPUT_DIR=${OUTPUT_DIR:-./output_gru_qwen}

export WANDB_MODE

# Training entry point
ENTRY_FILE=qwenvl/train/train_gru_qwen.py

# Build arguments as an array so smoke-test overrides stay easy to read.
args=(
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --projector_k "${PROJECTOR_K}"
    --tune_projector True
    --tune_qwen_vision False
    --tune_qwen_lm False
    --qwen_lm_unfreeze_last_n_layers "${QWEN_LM_UNFREEZE_LAST_N_LAYERS}"
    --qwen_unfreeze_lm_head "${QWEN_UNFREEZE_LM_HEAD}"
    --dataset_use "${DATASETS}"
    --data_flatten False
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
    --warmup_ratio 0.05
    --max_grad_norm 1.0
    --lr_scheduler_type cosine
    --logging_steps "${LOGGING_STEPS}"
    --model_max_length "${MODEL_MAX_LENGTH}"
    --gradient_checkpointing True
    --dataloader_num_workers 0
    --run_name "${RUN_NAME}"
    --report_to "${REPORT_TO}"
)

if [[ -n "${MAX_STEPS}" ]]; then
    args+=(--max_steps "${MAX_STEPS}")
fi

if [[ -n "${GRU_CHECKPOINT_PATH}" ]]; then
    args+=(--gru_checkpoint_path "${GRU_CHECKPOINT_PATH}")
fi

if [[ -n "${GRU_QWEN_CHECKPOINT_PATH}" ]]; then
    args+=(--gru_qwen_checkpoint_path "${GRU_QWEN_CHECKPOINT_PATH}")
fi

if [[ -n "${ALIGNMENT_MODULES_CHECKPOINT_PATH}" ]]; then
    args+=(--alignment_modules_checkpoint_path "${ALIGNMENT_MODULES_CHECKPOINT_PATH}")
fi

if [[ "${USE_DEEPSPEED}" == "1" && -n "${DEEPSPEED}" ]]; then
    args=(--deepspeed "${DEEPSPEED}" "${args[@]}")
fi

# Optional: Enable LoRA for full fine-tuning with less memory.
# args+=(--lora_enable True --lora_r 64 --lora_alpha 128)

# Launch training
echo "========================================"
echo "GRU-Qwen Training"
echo "========================================"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "GRU-Qwen Checkpoint: ${GRU_QWEN_CHECKPOINT_PATH}"
echo "Alignment Modules Checkpoint: ${ALIGNMENT_MODULES_CHECKPOINT_PATH}"
echo "GRU Checkpoint: ${GRU_CHECKPOINT_PATH}"
echo "Qwen LM Last-N Unfreeze: ${QWEN_LM_UNFREEZE_LAST_N_LAYERS}"
echo "Qwen LM Head Unfreeze: ${QWEN_UNFREEZE_LM_HEAD}"
echo "Dataset: ${DATASETS}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Batch Size: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Max Steps: ${MAX_STEPS}"
echo "Report To: ${REPORT_TO}"
echo "DeepSpeed: ${USE_DEEPSPEED} (${DEEPSPEED})"
echo "========================================"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${ENTRY_FILE} "${args[@]}"



