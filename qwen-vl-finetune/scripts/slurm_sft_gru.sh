#!/bin/bash
#SBATCH --job-name=qwen3vl_sft_gru
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

# ============================================================
# GRU-Qwen SFT on SLURM.
#
# This is a thin SLURM wrapper around scripts/sft_gru_qwen.sh: it sets up the
# environment (modules + conda) and exports the env vars that sft_gru_qwen.sh
# consumes, then hands off to that script which builds and launches torchrun.
#
# Submit a full run:
#   sbatch scripts/slurm_sft_gru.sh
#
# Submit the smoke test (mirrors the manual 3-GPU ZeRO-3 command):
#   SMOKE=1 sbatch scripts/slurm_sft_gru.sh
# ============================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Paths shared by both modes -----------------------------------------
export MODEL_NAME_OR_PATH="/scratch/tinoosh/iros_dataset/Qwen-Model/instruct"
export GRU_CHECKPOINT_PATH="/scratch/tinoosh/iros_dataset/Qwen-Model/gru_ckpt/model.safetensors"

# --- DeepSpeed -----------------------------------------------------------
export USE_DEEPSPEED=1
export DEEPSPEED=./scripts/zero3.json

# --- Distributed: 4 GPUs on one node ------------------------------------
export NPROC_PER_NODE=4

if [[ "${SMOKE:-0}" == "1" ]]; then
    # Reproduces the manual smoke command exactly.
    export MAX_STEPS=2
    export PER_DEVICE_TRAIN_BATCH_SIZE=1
    export GRAD_ACCUM_STEPS=1
    export NUM_TRAIN_EPOCHS=1
    export REPORT_TO=none
    export SAVE_STEPS=10000
    export INFERENCE_SNAPSHOT_STEPS=1
    export LOGGING_STEPS=1
    export DATASETS=r2r
    export RUN_NAME="smoke-gru-qwen-z3-3g"
    export OUTPUT_DIR="./smoke_out_z3_3gpu"
else
    # Full training run.
    # Effective batch = per_device(1) * grad_accum(40) * gpus(4) = 160,
    # matching slurm_sft.sh (4 * 10 * 4 = 160).
    export PER_DEVICE_TRAIN_BATCH_SIZE=1
    export GRAD_ACCUM_STEPS=40
    export NUM_TRAIN_EPOCHS=1
    export REPORT_TO=wandb
    export SAVE_STEPS=500
    export INFERENCE_SNAPSHOT_STEPS=100
    export LOGGING_STEPS=10
    export DATASETS="r2r,envdrop,human,rxr,scanqa,video_chatgpt,sharegptvideo,sharegpt4v"
    export RUN_NAME="gru-qwen-sft-$(date +%Y%m%d-%H%M)"
    export OUTPUT_DIR="/scratch/tinoosh/chang/checkpoints/${RUN_NAME}"
    export WANDB_ENTITY="project_llm"
    export WANDB_PROJECT="qwen3vl-sft-gru"
fi
# ============================================================

module purge
module load helpers/0.1.1
module load cuda/12.6.3
module load gcc/9.3.0
module load anaconda3/2024.02-1
conda activate /scratch/tinoosh/chang/envs/sft_env

cd /weka/scratch/tinoosh/rithvik/qwen_gru_ft/Qwen3-VL/qwen-vl-finetune

mkdir -p "${OUTPUT_DIR}"
mkdir -p /scratch/tinoosh/chang/logs

# Master coordinates for torchrun (single-node, but set explicitly).
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -n 1)
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=$(shuf -i 20001-29999 -n 1)

echo "========================================"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : ${SLURM_NODELIST}"
echo "Mode        : $([[ "${SMOKE:-0}" == "1" ]] && echo SMOKE || echo FULL)"
echo "Model       : ${MODEL_NAME_OR_PATH}"
echo "GRU ckpt    : ${GRU_CHECKPOINT_PATH}"
echo "Dataset     : ${DATASETS}"
echo "Output Dir  : ${OUTPUT_DIR}"
echo "Run Name    : ${RUN_NAME}"
echo "GPUs/node   : ${NPROC_PER_NODE}"
echo "DeepSpeed   : ${USE_DEEPSPEED} (${DEEPSPEED})"
echo "Batch/GPU   : ${PER_DEVICE_TRAIN_BATCH_SIZE}  (effective: $((PER_DEVICE_TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE)))"
echo "========================================"

bash scripts/sft_gru_qwen.sh
