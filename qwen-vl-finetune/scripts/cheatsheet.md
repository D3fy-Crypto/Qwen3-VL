# sanity check on interative node
SMOKE=1 sbatch --partition=interactive --gres=gpu:1 --time=00-00:30 \
    qwen-vl-finetune/scripts/slurm_sft.sh

# sanity check in interative session
srun --partition=interactive --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty bash

cd /weka/home/djonna1/cvpr_proj/Qwen3-VL/qwen-vl-finetune

SMOKE=1 bash scripts/slurm_sft.sh


sprio | head -1 && sprio | sort -k2 -rn | head -20 | grep a100

squeue -j 1299671 -o "%i %u %j %N %T"

time left
squeue -p a100 -t R -o "%i %u %L %N"