#!/bin/bash
#SBATCH -J nmr_r0_cpu_oracle
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/slurm-%j.out
set -euo pipefail
echo "=== NMR-R0: CPU Deterministic Oracle ==="
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
SCRIPT_DIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/scripts"
RESULTS_DIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/results/reliability_layer1/phase3/nmr_rescue_r0/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
cd "$SCRIPT_DIR"
echo ""
echo "=== CPU Oracle: running r=3 byte-level majority vote ==="
python3 run_nmr_r0_cpu_oracle.py
echo ""
echo "=== Copy results ==="
cp results/reliability_layer1/phase3/nmr_rescue_r0/job_${SLURM_JOB_ID}/nmr_r0_cpu_oracle.csv "$RESULTS_DIR/" 2>/dev/null || true
echo ""
echo "=== CPU Oracle complete ==="
date
