#!/bin/bash
#SBATCH -J nmr_r1_comparison
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/u4063895/workspace/gpu-byteplane-reliability-nmr/slurm-%j.out

set -euo pipefail

echo "=== NMR-R1: Graded vs Storage-Matched Uniform ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME"
if ! echo "$GPU_NAME" | grep -qi "H200"; then echo "FATAL: Expected H200"; exit 2; fi

ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
conda activate gpu-byteplane-scan

SCRIPT_DIR="/home/u4063895/workspace/gpu-byteplane-reliability-nmr/scripts"
RESULTS_DIR="/home/u4063895/workspace/gpu-byteplane-reliability-nmr/results/reliability_layer1/phase3/nmr_rescue_r1/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
cd "$SCRIPT_DIR"

echo ""
echo "=== R1 Comparison: 5 seeds, 4 configs, 3 fault modes ==="
python3 run_nmr_r1_comparison.py

echo ""
echo "=== Copy results ==="
cp results/reliability_layer1/phase3/nmr_rescue_r1/job_${SLURM_JOB_ID}/nmr_r1_comparison.csv "$RESULTS_DIR/" 2>/dev/null || true

echo ""
echo "=== R1 complete ==="
date
