#!/bin/bash
#SBATCH -J g4_max_cb
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/slurm-%j.out
set -euo pipefail
echo "=== Phase 3-Z G4: MAX/top-k CB Validation ==="
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
RESULTS_DIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/results/reliability_layer1/phase3/phase3z_ext/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
cd "$SCRIPT_DIR"
HURR_DIR="${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg4096"
CESM_DIR="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"
echo ""
echo "=== G4: hurricane_u ==="
python3 phase3z_ext_g4_max_cb.py \
    --artifact-dir "$HURR_DIR" \
    --dataset hurricane_u \
    --n-rows 1000000
echo ""
echo "=== G4: cesm_atm_cloud ==="
python3 phase3z_ext_g4_max_cb.py \
    --artifact-dir "$CESM_DIR" \
    --dataset cesm_atm_cloud \
    --n-rows 1000000
# Copy CSV
cp results/reliability_layer1/phase3/phase3z_ext/job_${SLURM_JOB_ID}/g4_max_cb.csv "$RESULTS_DIR/" 2>/dev/null || true
echo ""
echo "=== G4 complete ==="
date
