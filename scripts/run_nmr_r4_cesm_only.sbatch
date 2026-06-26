#!/usr/bin/env bash
#SBATCH -J nmr_r4_cesm
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%j.out

set -euo pipefail

echo "=== NMR-R4: Logical Diversity — cesm_atm_cloud only ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
echo "GPU: $GPU_NAME"
if [ -z "$GPU_NAME" ] || ! echo "$GPU_NAME" | grep -qi "H200"; then
    echo "FATAL: Expected H200 but got '$GPU_NAME'" >&2
    exit 2
fi

ml purge 2>/dev/null || true
ml load miniconda3/26.1.1
source /work/envstack/apps/miniconda3/26.1.1/etc/profile.d/conda.sh
conda activate gpu-byteplane-scan

LOCALITY_DATA_ROOT="${LOCALITY_DATA_ROOT:-/work/${USER}/datasets/locality_sensitivity}"
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$PROJ_ROOT/scripts"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$SCRIPT_DIR"
RESULTS_DIR="$PROJ_ROOT/results/reliability_layer1/phase4/nmr_r4_logical_diversity/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "=== Dataset: cesm_atm_cloud (168480000 rows) ==="
echo "Rates: 2e-6 2e-5 2e-4"
echo "Seeds: 0 1 2"

python3 "$SCRIPT_DIR/run_nmr_r4_logical_diversity.py" \
    --plane-dir "$LOCALITY_DATA_ROOT/cesm_atm_cloud/seg4096" \
    --dataset cesm_atm_cloud \
    --n-rows 168480000 \
    --fault-rates 2e-6 2e-5 2e-4 \
    --seeds 0 1 2 \
    --segment-size 4096 \
    --output "$RESULTS_DIR/nmr_r4_cesm_atm_cloud.csv"

echo ""
echo "=== Cesm complete ==="
ls -lh "$RESULTS_DIR"/
date
