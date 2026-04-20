#!/bin/bash
set -euo pipefail

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/home/rithvik/IROS_proj/cvpr_proj/qwen_models/instruct}
DATASET_USE=${DATASET_USE:-r2r_alignment_qa}
PIPELINE=${PIPELINE:-gru}
MOTION_TOKEN_TEXT=${MOTION_TOKEN_TEXT:-<motion>}
BATCH_SIZE=${BATCH_SIZE:-1}
DUMP_JSON=${DUMP_JSON:-./debug/gru_dataset_batch.json}

python tools/debug_gru_dataset_batch.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset_use "${DATASET_USE}" \
  --pipeline "${PIPELINE}" \
  --motion_token_text "${MOTION_TOKEN_TEXT}" \
  --batch_size "${BATCH_SIZE}" \
  --dump_json "${DUMP_JSON}"
