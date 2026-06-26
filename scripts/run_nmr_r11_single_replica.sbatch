#!/usr/bin/env bash
#SBATCH -J nmr_r11_single
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-02:00:00
#SBATCH --mail-type=END,FAIL

# NMR-R11: Single-Replica Detect-and-Bound Contract Freeze for v1.3
# Usage: DATASET=hurricane_u sbatch scripts/run_nmr_r11_single_replica.sbatch
#        DATASET=cesm_atm_cloud sbatch scripts/run_nmr_r11_single_replica.sbatch

set -euo pipefail

DATASET="${DATASET:?must be set to hurricane_u or cesm_atm_cloud}"

# Write handoff marker on exit
PROJ_ROOT="${PROJ_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cleanup() {
    local rc=$?
    mkdir -p "$PROJ_ROOT/handoff/job_done"
    GIT_COMMIT=$(git -C "$PROJ_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")
    MARKER="$PROJ_ROOT/handoff/job_done/job_${SLURM_JOB_ID}.json"
    cat > "$MARKER" <<EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "job_name": "nmr_r11_single",
  "exit_status": $rc,
  "finished_at": "$(date -Iseconds)",
  "dataset": "$DATASET",
  "workdir": "$PROJ_ROOT",
  "git_commit": "$GIT_COMMIT",
  "next_action": "read results, write freeze report"
}
EOF
    echo "Handoff marker: $MARKER (exit=$rc)"
    exit $rc
}
trap cleanup EXIT

echo "=== NMR-R11 Single-Replica Detect-and-Bound Freeze ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  Dataset: $DATASET"
GPU_NAME="unknown"
if nvidia-smi --query-gpu=name --format=csv,noheader > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
fi
echo "GPU: $GPU_NAME"
if ! echo "$GPU_NAME" | grep -qi "H200"; then
    echo "FATAL: Expected H200, found: $GPU_NAME" >&2
    exit 2
fi
date

case "$DATASET" in
cesm_atm_cloud)
    N_ROWS=168480000
    ;;
hurricane_u)
    N_ROWS=25000000
    ;;
*)
    echo "Unsupported DATASET=$DATASET" >&2
    exit 1
    ;;
esac

ARTIFACT_DIR="/work/u4063895/datasets/locality_sensitivity/${DATASET}/seg4096"

ml purge 2>/dev/null || true
ml load miniconda3/26.1.1 cuda/12.6
source /work/envstack/apps/miniconda3/26.1.1/etc/profile.d/conda.sh
conda activate gpu-byteplane-scan

SCRIPT_DIR="$PROJ_ROOT/scripts"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
OUTDIR="$PROJ_ROOT/results/v1_3_freeze/reliability_single/job_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "N_ROWS=$N_ROWS"
echo "ARTIFACT_DIR=$ARTIFACT_DIR"
echo "OUTDIR=$OUTDIR"

# Step 1: Plane-uniform mode (3 rates × 3 seeds)
echo ""
echo "=== Step 1: Plane-uniform fault injection ==="
python3 "$SCRIPT_DIR/nmr_r11_single_replica.py" \
    --artifact-dir "$ARTIFACT_DIR" \
    --dataset "$DATASET" \
    --n-rows "$N_ROWS" \
    --rates 1e-6 1e-5 1e-4 \
    --seeds 0 1 2 \
    --mode plane_uniform \
    --output "$OUTDIR/nmr_r11_plane_uniform.csv"

# Step 2: Same-fault-all negative control (3 seeds)
echo ""
echo "=== Step 2: Same-fault-all negative control ==="
python3 "$SCRIPT_DIR/nmr_r11_single_replica.py" \
    --artifact-dir "$ARTIFACT_DIR" \
    --dataset "$DATASET" \
    --n-rows "$N_ROWS" \
    --rates 1e-5 \
    --seeds 0 1 2 \
    --mode same_fault_all \
    --output "$OUTDIR/nmr_r11_same_fault_all.csv"

echo ""
echo "=== Complete: $DATASET ==="
find "$OUTDIR" -name '*.csv' -exec ls -lh {} \;
date


