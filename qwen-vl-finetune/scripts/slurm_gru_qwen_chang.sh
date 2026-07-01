#!/bin/bash
#SBATCH --job-name=qwen3vl_sft_gru_chang
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --account=tinoosh
#SBATCH --output=/scratch/tinoosh/chang/logs/%j_%x.out
#SBATCH --error=/scratch/tinoosh/chang/logs/%j_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# ZeRO-3-native GRU-Qwen SFT launcher (chang).
#
# Two ways to run:
#   * Cluster:  sbatch scripts/slurm_gru_qwen_chang.sh
#       Uses the #SBATCH block above + the SLURM setup below (modules, conda, cd,
#       master addr, 4 GPUs). ADJUST for your cluster: the #SBATCH account /
#       partition / log paths, and REPO_DIR / conda env / BASE_GRU_DIR below.
#   * Local:    bash scripts/slurm_gru_qwen_chang.sh   (the SLURM block is skipped).
#
# Loads everything from one self-contained BASE_GRU_DIR (backbone + aligned
# projector + frozen GRU + tokenizer with <gru>@151669). See the plan.
#
# Local A6000 single-process offload sanity (tight on memory):
#   USE_DEEPSPEED=1 DEEPSPEED=./scripts/zero3_offload.json NPROC_PER_NODE=1 \
#   MAX_STEPS=2 PER_DEVICE_TRAIN_BATCH_SIZE=1 MODEL_MAX_LENGTH=1024 \
#   REPORT_TO=none DATASETS=r2r_gru bash scripts/slurm_gru_qwen_chang.sh

# --- Environment + path defaults: cluster (sbatch) vs local (bash) -----------
# Each variable gets exactly ONE default, chosen by mode. Export any of them
# before launching to override. BASE_GRU_DIR = the self-contained model dir
# (backbone + projector + GRU + tokenizer<gru>).
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # ---- Cluster: modules, conda, working dir, distributed, cluster paths -----
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    module purge
    module load helpers/0.1.1
    module load cuda/12.6.3
    module load gcc/9.3.0
    module load anaconda3/2024.02-1
    conda activate /scratch/tinoosh/chang/envs/sft_env

    # Repo checkout on the cluster (adjust to yours).
    REPO_DIR=${REPO_DIR:-/weka/home/djonna1/cvpr_proj/Qwen3-VL/qwen-vl-finetune}
    cd "${REPO_DIR}" || exit 1
    mkdir -p /scratch/tinoosh/chang/logs

    export WANDB_ENTITY="${WANDB_ENTITY:-project_llm}"
    export WANDB_PROJECT="${WANDB_PROJECT:-qwen3vl-sft-gru}"

    # Distributed coordinates + GPU count from the SLURM allocation.
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -n 1)
    MASTER_PORT=$(shuf -i 20001-29999 -n 1)
    NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-4}

    BASE_GRU_DIR=${BASE_GRU_DIR:-/home/djonna1/scratchtinoosh/iros_dataset/base_qwen_with_gru}
    RUN_NAME=${RUN_NAME:-gru-qwen-chang-sft-$(date +%Y%m%d-%H%M)}
    OUTPUT_DIR=${OUTPUT_DIR:-/scratch/tinoosh/chang/checkpoints/${RUN_NAME}}
else
    # ---- Local single-process run (vega paths) --------------------------------
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
    NPROC_PER_NODE=${NPROC_PER_NODE:-1}

    BASE_GRU_DIR=${BASE_GRU_DIR:-/home/rithvik/IROS_proj/models_ckpts/trained/gru/base_qwen_with_gru}
    RUN_NAME=${RUN_NAME:-gru-qwen-chang-sft}
    OUTPUT_DIR=${OUTPUT_DIR:-./output_gru_qwen_chang}
fi

USE_DEEPSPEED=${USE_DEEPSPEED:-1}
DEEPSPEED=${DEEPSPEED:-./scripts/zero3.json}

# Training hyperparameters — all values mirror slurm_sft.sh.
LEARNING_RATE=${LEARNING_RATE:-1e-4}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-10}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-}
SAVE_STEPS=${SAVE_STEPS:-500}
INFERENCE_SNAPSHOT_STEPS=${INFERENCE_SNAPSHOT_STEPS:-10000}
LOGGING_STEPS=${LOGGING_STEPS:-10}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
REPORT_TO=${REPORT_TO:-wandb}
WANDB_MODE=${WANDB_MODE:-online}
# Full mix mirrors slurm_sft.sh, with the nav datasets swapped to their GRU variants:
#   r2r_gru/rxr_gru/human_gru -> oversampled + per-row `gru`; envdrop_gru -> inline motion.
#   scanqa/video_chatgpt/sharegptvideo/sharegpt4v have no trajectory -> has_gru=False (no <gru>).
DATASETS=${DATASETS:-r2r_gru,rxr_gru,envdrop_gru,human_gru,scanqa,video_chatgpt,sharegptvideo,sharegpt4v}

# Freeze mask: GRU always frozen + projector always trained (handled in code);
# these toggle the backbone (mirrors slurm_sft.sh: vision + mlp + llm all True).
TUNE_MM_VISION=${TUNE_MM_VISION:-True}
TUNE_MM_MLP=${TUNE_MM_MLP:-True}
TUNE_MM_LLM=${TUNE_MM_LLM:-True}

export WANDB_MODE

ENTRY_FILE=qwenvl/train/train_gru_qwen_chang.py

args=(
    --model_name_or_path "${BASE_GRU_DIR}"
    --dataset_use "${DATASETS}"
    --data_flatten True
    --tune_mm_vision "${TUNE_MM_VISION}"
    --tune_mm_mlp "${TUNE_MM_MLP}"
    --tune_mm_llm "${TUNE_MM_LLM}"
    --bf16
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --max_pixels 50176
    --min_pixels 784
    --eval_strategy no
    --save_strategy steps
    --save_steps "${SAVE_STEPS}"
    --save_total_limit 2
    --inference_snapshot_steps "${INFERENCE_SNAPSHOT_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay 0
    --warmup_ratio 0.03
    --max_grad_norm 1
    --lr_scheduler_type cosine
    --logging_steps "${LOGGING_STEPS}"
    --model_max_length "${MODEL_MAX_LENGTH}"
    --gradient_checkpointing True
    --dataloader_num_workers 8
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
