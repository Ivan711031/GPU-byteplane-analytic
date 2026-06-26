#!/usr/bin/env bash
#SBATCH -J nmr_d2_cesm_full
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 3:00:00
#SBATCH --mail-type=END,FAIL

# Cesm full-scale: skip calibration (use slice rates), reduced stochastic cells.
echo "=== NMR-D2 Cesm Full-Scale (light) ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)"
GPU_NAME="unknown"
if nvidia-smi --query-gpu=name --format=csv,noheader > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
fi
echo "GPU: $GPU_NAME"
date

DATASET=cesm_atm_cloud
N_ROWS=168480000
ARTIFACT_DIR="/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"
SEGMENT_SIZE=4096
N_SEGMENTS=$(( (N_ROWS + SEGMENT_SIZE - 1) / SEGMENT_SIZE ))

ml purge 2>/dev/null || true
ml load miniconda3/26.1.1 cuda/12.6
source /work/envstack/apps/miniconda3/26.1.1/etc/profile.d/conda.sh
conda activate gpu-byteplane-scan

PROJ_ROOT=/home/u4063895/workspace/gpu-byteplane-scan-experiments/.worktrees/issue-198-nmr-d2-uniform-spread
SCRIPT_DIR="$PROJ_ROOT/scripts"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
OUTDIR="$PROJ_ROOT/results/reliability_layer1/phase4/nmr_d2/cesm_atm_cloud_fullscale/job_${SLURM_JOB_ID}"
PMAP_DIR="$OUTDIR/protection_maps"
mkdir -p "$OUTDIR" "$PMAP_DIR"

echo "N_ROWS=$N_ROWS N_SEGMENTS=$N_SEGMENTS OUTDIR=$OUTDIR"

# Step 0: Protection maps
echo "--- Step 0 ---"
for policy in graded_seg_B3 uniform_spread_seg_B3; do
    python3 "$SCRIPT_DIR/nmr_d2_protection_map.py" \
        --policy $policy --n-segments $N_SEGMENTS --seed 0 \
        --segment-size $SEGMENT_SIZE \
        --output "$PMAP_DIR/${policy}_seed0.json"
done

# Step 2: Diagnostic
echo "--- Diagnostic ---"
python3 "$SCRIPT_DIR/nmr_d2_diagnostic.py" \
    --protection-map-dir "$PMAP_DIR" \
    --clean-plane-dir "$ARTIFACT_DIR" \
    --dataset "$DATASET" \
    --n-rows "$N_ROWS" \
    --fault-rate 0.01 \
    --seed 0 \
    --output-dir "$OUTDIR/diagnostic"

# Step 3: Reduced stochastic (1 rate, 3 seeds)
echo "--- Stochastic ---"
python3 "$SCRIPT_DIR/nmr_d2_stochastic.py" \
    --protection-map-dir "$PMAP_DIR" \
    --clean-plane-dir "$ARTIFACT_DIR" \
    --dataset "$DATASET" \
    --n-rows "$N_ROWS" \
    --headline-rates 2e-5 \
    --seeds 0 1 2 \
    --mode plane_uniform \
    --output "$OUTDIR/nmr_d2_stochastic_matrix.csv"

# Step 4: Claim matrix
echo "--- Claim Matrix ---"
python3 "$SCRIPT_DIR/nmr_d2_claim_matrix.py" \
    --stochastic-results "$OUTDIR/nmr_d2_stochastic_matrix.csv" \
    --policies graded_seg_B3 uniform_spread_seg_B3 \
    --output-dir "$OUTDIR"

echo "=== Complete ==="
find "$OUTDIR" -name '*.csv' -exec ls -lh {} \;
date
