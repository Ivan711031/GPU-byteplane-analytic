#!/usr/bin/env bash
#SBATCH -J nmr_d2_smoke
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-1:00:00
#SBATCH --mail-type=END,FAIL

# Smoke test: sensor 10M, verify calibration differentiates rates and output
# lands on shared filesystem.

echo "=== NMR-D2 Smoke ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)"
GPU_NAME="unknown"
if nvidia-smi --query-gpu=name --format=csv,noheader > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
fi
echo "GPU: $GPU_NAME"
date

ml purge 2>/dev/null || true
ml load miniconda3/26.1.1 cuda/12.6
source /work/envstack/apps/miniconda3/26.1.1/etc/profile.d/conda.sh
conda activate gpu-byteplane-scan

PROJ_ROOT=/home/u4063895/workspace/gpu-byteplane-scan-experiments/.worktrees/issue-198-nmr-d2-uniform-spread
SCRIPT_DIR="$PROJ_ROOT/scripts"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

export DATASET=sensor
export N_ROWS=1000000
export ARTIFACT_DIR=/work/u4063895/datasets/reliability_layer1/artifacts/sensor/n1000000/scale100
export HEADLINE_RATES="1e-07 1e-06 1e-05"
export DIAG_RATE=0.01
export SEEDS="0 1"
export CANDIDATE_RATES="1e-7 5e-7 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3"
export RUN_WW=false

OUTDIR="$PROJ_ROOT/results/reliability_layer1/phase4/nmr_d2/job_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

bash "$SCRIPT_DIR/run_nmr_d2.sh" 2>&1 | tee "$OUTDIR/pipeline.log"

echo ""
echo "=== Output ==="
find "$OUTDIR" -name '*.csv' -exec ls -lh {} \;
echo "=== Smoke done ==="
date
