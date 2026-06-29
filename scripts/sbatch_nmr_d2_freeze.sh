#!/bin/bash
#SBATCH --job-name=nmr-d2-freeze
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/nmr_d2_freeze_%j.out
#SBATCH --error=logs/nmr_d2_freeze_%j.err
set -euo pipefail
# Load environment
ml miniconda3/26.1.1
conda activate gpu-byteplane-scan
echo "=== Hardware validation ==="
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "NO_GPU")
echo "GPU: ${GPU_NAME}"
if ! echo "${GPU_NAME}" | grep -qi "H200"; then
    echo "FATAL: Expected H200 GPU, got '${GPU_NAME}'. Exiting."
    exit 2
fi
echo "=== Job info ==="
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "DATE=$(date -Is)"
echo "PWD=$(pwd)"
echo "=== Git info ==="
git rev-parse HEAD || true
git status --short || true
mkdir -p logs
bash scripts/run_nmr_d2_freeze.sh
