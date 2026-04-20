#!/bin/bash
set -euo pipefail

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/home/rithvik/IROS_proj/cvpr_proj/qwen_models/instruct}
DATASET_USE=${DATASET_USE:-r2r_alignment_qa}
PIPELINE=${PIPELINE:-gru}
MOTION_TOKEN_TEXT=${MOTION_TOKEN_TEXT:-<motion>}
GRU_CHECKPOINT_PATH=${GRU_CHECKPOINT_PATH:-/home/rithvik/IROS_proj/cvpr_proj/traj_model/checkpoints/best_model.pt}
BATCH_SIZE=${BATCH_SIZE:-1}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bf16}
DUMP_JSON=${DUMP_JSON:-./debug/gru_forward_exact.json}

python tools/debug_gru_forward_exact.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset_use "${DATASET_USE}" \
  --pipeline "${PIPELINE}" \
  --motion_token_text "${MOTION_TOKEN_TEXT}" \
  --gru_checkpoint_path "${GRU_CHECKPOINT_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --dump_json "${DUMP_JSON}"
