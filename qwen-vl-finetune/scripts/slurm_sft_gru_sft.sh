#!/bin/bash
#SBATCH --job-name=qwen3vl_sft_gru
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/tinoosh/chang/logs/%j_%x.out
#SBATCH --error=/scratch/tinoosh/chang/logs/%j_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# ============================================================
# Configurable — edit these before submitting
# ============================================================
MODEL_PATH="/weka/scratch/tinoosh/iros_dataset/Qwen-Model/instruct"
GRU_WARMSTART="/weka/scratch/tinoosh/iros_dataset/Qwen-Model/gru_ckpt/model.safetensors"
DATASETS="human"
RUN_NAME="qwen3vl-gru-sft-$(date +%Y%m%d-%H%M)"
OUTPUT_DIR="/scratch/tinoosh/chang/checkpoints/${RUN_NAME}"
WANDB_PROJECT="qwen3vl-gru-sft"

LR=1e-4
BATCH_SIZE=2
GRAD_ACCUM=8
EPOCHS=1
MODEL_MAX_LENGTH=8192   # longer context for long-horizon trajectories

if [[ "${SMOKE:-0}" == "1" ]]; then
    BATCH_SIZE=2
    GRAD_ACCUM=2
    NPROC_OVERRIDE=2
    MODEL_MAX_LENGTH=2048
    MAX_STEPS_ARG="--max_steps 10"
    SAVE_STRATEGY_ARG="--save_strategy steps --save_steps 500 --save_total_limit 2"
    REPORT_ARG="--report_to none"
    TUNE_LLM_ARG="--tune_mm_llm True --tune_qwen_lm True"
    PIXELS_ARG="--max_pixels 4096 --min_pixels 784"
    WORKERS_ARG="--dataloader_num_workers 4"
    DEEPSPEED_ARG="--deepspeed ./scripts/zero3.json"
    RUN_NAME="smoke-gru-$(date +%Y%m%d-%H%M)"
    OUTPUT_DIR="/scratch/tinoosh/chang/checkpoints/${RUN_NAME}"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
else
    NPROC_OVERRIDE=8
    MAX_STEPS_ARG=""
    SAVE_STRATEGY_ARG="--save_strategy steps --save_steps 500 --save_total_limit 2"
    REPORT_ARG="--report_to wandb"
    TUNE_LLM_ARG="--tune_mm_llm True --tune_qwen_lm True"
    PIXELS_ARG="--max_pixels 50176 --min_pixels 784"
    WORKERS_ARG="--dataloader_num_workers 4"
    DEEPSPEED_ARG="--deepspeed ./scripts/zero3.json"
fi
# ============================================================

module purge
module load helpers/0.1.1
module load cuda/12.6.3
module load gcc/9.3.0
module load anaconda3/2024.02-1
conda activate /scratch/tinoosh/chang/envs/sft_env

cd /weka/home/djonna1/cvpr_proj/Qwen3-VL/qwen-vl-finetune

mkdir -p "${OUTPUT_DIR}"
mkdir -p /scratch/tinoosh/chang/logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -n 1)
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=$(shuf -i 20001-29999 -n 1)
NPROC_PER_NODE=${NPROC_OVERRIDE}

export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_RUN_ID="${RUN_NAME}"

echo "========================================"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : ${SLURM_NODELIST}"
echo "Model       : ${MODEL_PATH}"
echo "Warmstart   : ${GRU_WARMSTART}"
echo "Dataset     : ${DATASETS}"
echo "Output Dir  : ${OUTPUT_DIR}"
echo "Run Name    : ${RUN_NAME}"
echo "GPUs/node   : ${NPROC_PER_NODE}"
echo "LR          : ${LR}"
echo "Batch/GPU   : ${BATCH_SIZE}  (effective: $((BATCH_SIZE * GRAD_ACCUM * NPROC_PER_NODE)))"
echo "========================================"

torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    qwenvl/train/train_gru_sft_qwen.py \
        ${DEEPSPEED_ARG} \
        --model_name_or_path "${MODEL_PATH}" \
        --gru_warmstart_ckpt "${GRU_WARMSTART}" \
        --model_type qwen3vl \
        --dataset_use "${DATASETS}" \
        --tune_mm_vision True \
        --tune_mm_mlp True \
        ${TUNE_LLM_ARG} \
        --tune_projector True \
        --tune_qwen_vision False \
        --bf16 \
        --output_dir "${OUTPUT_DIR}" \
        --num_train_epochs ${EPOCHS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        ${PIXELS_ARG} \
        --eval_strategy no \
        ${SAVE_STRATEGY_ARG} \
        --learning_rate ${LR} \
        --weight_decay 0 \
        --warmup_ratio 0.03 \
        --max_grad_norm 1 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --model_max_length ${MODEL_MAX_LENGTH} \
        --gradient_checkpointing True \
        ${WORKERS_ARG} \
        --run_name "${RUN_NAME}" \
        ${MAX_STEPS_ARG} \
        ${REPORT_ARG}
