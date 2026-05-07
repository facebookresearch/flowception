#!/bin/bash
#SBATCH --job-name=flowception_test
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_run_%j.out
#SBATCH --error=logs/test_run_%j.err

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

echo "=== Node: $(hostname) ==="
nvidia-smi -L
echo ""

# ---------- Step 1: Toy I2V (sanity check) ----------
echo "========== TOY I2V =========="
python main.py \
    -c configs/flowception/toy/toy_coloring_i2v.yaml \
    -t -n toy_i2v_test \
    --append SOLVER.MAX_ITER 20 SOLVER.LOG_WANDB false \
            SOLVER.EVAL_FREQ 999999 SOLVER.SNAPSHOT_F 999999 \
            SOLVER.CKPT_EVERY 999999
echo "TOY I2V: PASSED"
echo ""

# ---------- Step 2: Toy T2V ----------
echo "========== TOY T2V =========="
python main.py \
    -c configs/flowception/toy/toy_coloring_t2v.yaml \
    -t -n toy_t2v_test \
    --append SOLVER.MAX_ITER 20 SOLVER.LOG_WANDB false \
            SOLVER.EVAL_FREQ 999999 SOLVER.SNAPSHOT_F 999999 \
            SOLVER.CKPT_EVERY 999999
echo "TOY T2V: PASSED"
echo ""

# ---------- Step 3: Toy Interpolation ----------
echo "========== TOY INTERPOLATION =========="
python main.py \
    -c configs/flowception/toy/toy_coloring_interpolate.yaml \
    -t -n toy_interp_test \
    --append SOLVER.MAX_ITER 20 SOLVER.LOG_WANDB false \
            SOLVER.EVAL_FREQ 999999 SOLVER.SNAPSHOT_F 999999 \
            SOLVER.CKPT_EVERY 999999
echo "TOY INTERPOLATION: PASSED"
echo ""

# ---------- Step 4: OpenVid-1M I2V (real data, tiny subset) ----------
# Requires OpenVid-1M data. Set OPENVID_DATA_DIR to the dataset root to run this step.
if [ -n "${OPENVID_DATA_DIR:-}" ] && [ -d "$OPENVID_DATA_DIR" ]; then
    echo "========== OPENVID I2V (10 videos) =========="
    python main.py \
        -c configs/flowception/i2v/openvid_i2v.yaml \
        -t -n openvid_i2v_test \
        --append DATA.CLUSTER local SOLVER.LOG_WANDB false \
                SOLVER.COMPILE_MODELS false SOLVER.BATCH_SIZE 1 \
                SOLVER.IM_SIZE 128 SOLVER.MAX_ITER 10 \
                SOLVER.EVAL_FREQ 999999 SOLVER.SNAPSHOT_F 999999 \
                SOLVER.CKPT_EVERY 999999 DATA.WORKERS 0
    echo "OPENVID I2V: PASSED"
    echo ""
else
    echo "========== OPENVID I2V: SKIPPED (set OPENVID_DATA_DIR to enable) =========="
    echo ""
fi

echo "========== ALL TESTS PASSED =========="
