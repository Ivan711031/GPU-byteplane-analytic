#!/bin/bash
# CPU-only pilot runner for Claim 1 realistic fault campaign.
# Runs pure-Python evaluator (no CUDA dependency).
set -euo pipefail

SEGMENT_SIZE=4096
OUTPUT_DIR="results/reliability_layer1/claim1_realistic_pilot/job_${SLURM_JOB_ID}"

echo "=== NMR-Claim1 realistic fault campaign pilot (CPU) ==="
echo "JOB_ID=${SLURM_JOB_ID:-local}"

# n_rows = 5000000 gives ~0.5 expected events at 1e-7 rate,
# and ~50 events at 1e-5 rate, which is sufficient for pilot
# labeling (too_sparse / informative / saturated).
PILOT_N_ROWS=5000000

run_policy_sweep() {
    local dataset="$1" artifact_dir="$2" label="$3"
    echo ""
    echo "--- ${label}: ${dataset} (n_rows=${PILOT_N_ROWS}) ---"
    python3 -u scripts/nmr_claim1_realistic_campaign.py \
        --artifact-dir "${artifact_dir}" \
        --dataset "${dataset}" \
        --n-rows "${PILOT_N_ROWS}" \
        --segment-size "${SEGMENT_SIZE}" \
        --fault-families F1 F2 F3 F4 F5 F6 F7 F8 \
        --rate-anchors 1e-7 1e-5 \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --output-dir "${OUTPUT_DIR}/${dataset}"
}

run_policy_sweep "hurricane_u" \
    "/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg4096" \
    "PILOT"

run_policy_sweep "cesm_atm_cloud" \
    "/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg4096" \
    "PILOT"

echo ""
echo "=== Pilot complete ==="
echo "Output: ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"/*/claim1_realistic_*.csv 2>/dev/null || true
