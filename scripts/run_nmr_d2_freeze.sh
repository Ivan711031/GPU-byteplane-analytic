#!/bin/bash
# NMR-D2 v1.3 freeze runner (CPU-only).
# Usage: sbatch -p ngs32g scripts/run_nmr_d2_freeze.sh
# Or: bash scripts/run_nmr_d2_freeze.sh [--no-submit]
#
# Generates protection maps, runs stochastic evaluation, claim matrix,
# and normalizer sanity for the D2 graded-vs-uniform freeze.
set -euo pipefail
BASE_DIR="${WORK_DIR}/datasets/locality_sensitivity"
OUT_DIR="results/v1_3_freeze/d2_allocation"
SCRIPTS_DIR="scripts"
SEEDS="0 1 2"
FAULT_RATES="2e-5 2e-4"
POLICIES="graded_seg_B3 uniform_spread_seg_B3"
NORMALIZER_BRANCH="main"
mkdir -p "$OUT_DIR/stochastic"
mkdir -p "$OUT_DIR/claim"
mkdir -p "$OUT_DIR/normalizer"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-local}
echo "=== NMR-D2 Freeze Runner ==="
echo "Job ID: $JOB_ID"
echo "Timestamp: $TIMESTAMP"
echo "Host: $(hostname)"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "Git branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
echo ""
# ---- Dataset sweep ----
# Protection maps are auto-generated per-dataset by nmr_d2_stochastic.py
# with the correct n_segments for each dataset's n_rows.
echo "=== Stochastic evaluation ==="
run_dataset() {
    local DATASET=$1
    local N_ROWS=$2
    local PLANE_DIR="$BASE_DIR/$DATASET/seg4096"
    local MANIFEST="$PLANE_DIR/manifest.json"
    local STOCH_OUT="$OUT_DIR/stochastic/${DATASET}_stochastic_matrix.csv"
    local CLAIM_OUT_DIR="$OUT_DIR/claim/$DATASET"
    local NORM_OUT_DIR="$OUT_DIR/normalizer/$DATASET"
    local PMAP_DIR="$OUT_DIR/protection_maps/$DATASET"
    echo "--- Dataset: $DATASET (n_rows=$N_ROWS) ---"
    python3 "$SCRIPTS_DIR/nmr_d2_stochastic.py" \
        --protection-map-dir "$PMAP_DIR" \
        --clean-plane-dir "$PLANE_DIR" \
        --dataset "$DATASET" \
        --n-rows "$N_ROWS" \
        --headline-rates $FAULT_RATES \
        --seeds $SEEDS \
        --mode plane_uniform \
        --output "$STOCH_OUT"
    echo ""
    echo "--- Claim matrix: $DATASET ---"
    python3 "$SCRIPTS_DIR/nmr_d2_claim_matrix.py" \
        --stochastic-results "$STOCH_OUT" \
        --policies ${POLICIES} \
        --output-dir "$CLAIM_OUT_DIR"
    echo ""
    echo "--- Normalizer sanity: $DATASET ---"
    python3 "$SCRIPTS_DIR/validate_nmr_d2_normalizer.py" \
        --stochastic-csv "$STOCH_OUT" \
        --artifact-dir "$PLANE_DIR" \
        --manifest "$MANIFEST" \
        --n-rows "$N_ROWS" \
        --output-dir "$NORM_OUT_DIR"
    echo ""
}
# hurricane_u: 25M rows
run_dataset "hurricane_u" 25000000
# cesm_atm_cloud: 168M rows
run_dataset "cesm_atm_cloud" 168480000
echo "=== Write handoff marker ==="
HANDOFF_DIR="handoff/job_done"
mkdir -p "$HANDOFF_DIR"
cat > "$HANDOFF_DIR/job_${JOB_ID}.json" << EOF
{
  "job_id": "${JOB_ID}",
  "job_name": "nmr_d2_freeze",
  "timestamp": "${TIMESTAMP}",
  "host": "$(hostname)",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "output_dir": "${OUT_DIR}",
  "datasets": "hurricane_u cesm_atm_cloud",
  "policies": "${POLICIES}",
  "fault_rates": "${FAULT_RATES}",
  "seeds": "${SEEDS}",
  "exit_status": 0
}
EOF
echo "  Handoff marker: $HANDOFF_DIR/job_${JOB_ID}.json"
echo ""
echo "=== DONE ==="
ls -lh "$OUT_DIR/stochastic/"
ls -lh "$OUT_DIR/claim/"*/
ls -lh "$OUT_DIR/normalizer/"*/
