#!/bin/bash
#SBATCH --job-name=fc_t2v_ltx
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/t2v_ltx_%j.out
#SBATCH --error=logs/t2v_ltx_%j.err

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

echo "=== T2V LTX-2B Experiment ==="
echo "Node: $(hostname)"
echo "Python: $(which python)"
nvidia-smi -L
echo ""

export HF_HUB_ENABLE_XET=0
export HF_XET_CACHE=/tmp/xet_cache
export XET_LOG_DIR=/tmp/xet_logs

accelerate launch --num_processes 8 --mixed_precision bf16 main.py \
    -c configs/flowception/t2v/openvid_ltx_t2v.yaml \
    -t -n ltx_t2v_10k \
    --append \
        SOLVER.MAX_ITER 10000 \
        SOLVER.LOG_WANDB false \
        SOLVER.CKPT_EVERY 2500 \
        SOLVER.EVAL_FREQ 5000 \
        SOLVER.SNAPSHOT_F 500 \
        SOLVER.COMPILE_MODELS false \
        DATA.CLUSTER local
