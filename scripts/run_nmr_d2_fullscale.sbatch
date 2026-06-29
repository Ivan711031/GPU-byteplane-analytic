#!/usr/bin/env bash
#SBATCH -J nmr_d2_fullscale
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 3:00:00
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
# Full-scale D2 for primary fields.
# Usage: DATASET=cesm_atm_cloud sbatch scripts/run_nmr_d2_fullscale.sbatch
#        DATASET=hurricane_u  sbatch scripts/run_nmr_d2_fullscale.sbatch
DATASET="${DATASET:?must be set (cesm_atm_cloud or hurricane_u)}"
echo "=== NMR-D2 Full-Scale ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  Dataset: $DATASET"
GPU_NAME="unknown"
if nvidia-smi --query-gpu=name --format=csv,noheader > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
fi
echo "GPU: $GPU_NAME"
date
# Dataset config
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
ARTIFACT_DIR="${WORK_DIR}/datasets/locality_sensitivity/${DATASET}/seg4096"
SEGMENT_SIZE=4096
N_SEGMENTS=$(( (N_ROWS + SEGMENT_SIZE - 1) / SEGMENT_SIZE ))
ml purge 2>/dev/null || true
ml load miniconda3/26.1.1 cuda/12.6
source ${ENVSTACK_ROOT}/miniconda3/26.1.1/etc/profile.d/conda.sh
conda activate gpu-byteplane-scan
PROJ_ROOT=${PROJ_DIR}/workspace/gpu-byteplane-scan-experiments/.worktrees/WORKTREE
SCRIPT_DIR="$PROJ_ROOT/scripts"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
OUTDIR="$PROJ_ROOT/results/reliability_layer1/phase4/nmr_d2/${DATASET}_fullscale/job_${SLURM_JOB_ID}"
PMAP_DIR="$OUTDIR/protection_maps"
mkdir -p "$OUTDIR" "$PMAP_DIR"
echo "N_ROWS=$N_ROWS SEGMENT_SIZE=$SEGMENT_SIZE N_SEGMENTS=$N_SEGMENTS"
echo "OUTDIR=$OUTDIR"
# Step 0: Protection maps
echo "--- Step 0: Protection Maps ---"
for policy in graded_seg_B3 uniform_spread_seg_B3; do
    python3 "$SCRIPT_DIR/nmr_d2_protection_map.py" \
        --policy $policy \
        --n-segments $N_SEGMENTS \
        --seed 0 \
        --segment-size $SEGMENT_SIZE \
        --output "$PMAP_DIR/${policy}_seed0.json"
done
# Step 1: Calibration (8 rates)
echo "--- Step 1: Calibration ---"
python3 "$SCRIPT_DIR/nmr_d2_calibration.py" \
    --protection-map-dir "$PMAP_DIR" \
    --clean-plane-dir "$ARTIFACT_DIR" \
    --dataset "$DATASET" \
    --n-rows "$N_ROWS" \
    --seed 0 \
    --candidate-rates 5e-7 1e-6 2e-6 5e-6 1e-5 2e-5 5e-5 1e-4 \
    --output "$OUTDIR/nmr_d2_calibration_matrix.csv"
# Step 2: Diagnostic
echo "--- Step 2: Diagnostic ---"
python3 "$SCRIPT_DIR/nmr_d2_diagnostic.py" \
    --protection-map-dir "$PMAP_DIR" \
    --clean-plane-dir "$ARTIFACT_DIR" \
    --dataset "$DATASET" \
    --n-rows "$N_ROWS" \
    --fault-rate 0.01 \
    --seed 0 \
    --output-dir "$OUTDIR/diagnostic"
# Step 3: Stochastic (3 seeds × 3 rates)
echo "--- Step 3: Stochastic ---"
python3 "$SCRIPT_DIR/nmr_d2_stochastic.py" \
    --protection-map-dir "$PMAP_DIR" \
    --clean-plane-dir "$ARTIFACT_DIR" \
    --dataset "$DATASET" \
    --n-rows "$N_ROWS" \
    --headline-rates 2e-6 2e-5 2e-4 \
    --seeds 0 1 2 \
    --mode plane_uniform \
    --output "$OUTDIR/nmr_d2_stochastic_matrix.csv"
# Step 4: Claim matrix
echo "--- Step 4: Claim Matrix ---"
python3 "$SCRIPT_DIR/nmr_d2_claim_matrix.py" \
    --stochastic-results "$OUTDIR/nmr_d2_stochastic_matrix.csv" \
    --policies graded_seg_B3 uniform_spread_seg_B3 \
    --output-dir "$OUTDIR"
echo ""
echo "=== Full-Scale Complete: $DATASET ==="
find "$OUTDIR" -name '*.csv' -exec ls -lh {} \;
date
