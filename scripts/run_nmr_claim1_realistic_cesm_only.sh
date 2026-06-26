#!/bin/bash
set -euo pipefail

PILOT_N_ROWS=5000000
SEGMENT_SIZE=4096
OUTPUT_DIR="results/reliability_layer1/claim1_realistic_pilot/job_${SLURM_JOB_ID}"

echo "=== NMR-Claim1 cesm_atm_cloud pilot (continuation) ==="
echo "JOB_ID=${SLURM_JOB_ID:-local}"

python3 -u scripts/nmr_claim1_realistic_campaign.py \
    --artifact-dir "/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg4096" \
    --dataset cesm_atm_cloud \
    --n-rows "${PILOT_N_ROWS}" \
    --segment-size "${SEGMENT_SIZE}" \
    --fault-families F1 F2 F3 F4 F5 F6 F7 F8 \
    --rate-anchors 1e-7 1e-5 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "=== Complete ==="
ls -lh "${OUTPUT_DIR}/"claim1_realistic_*.csv
