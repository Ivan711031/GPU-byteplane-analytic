#!/bin/bash
#SBATCH -J z1a_cpu_validation
#SBATCH -p ngs32g
#SBATCH -t 0-02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u4063895@nchc.org.tw

set -euo pipefail

echo "=== Phase 3-Z1-A: Real-Dataset CPU Validation ==="
echo "Job ID: $SLURM_JOB_ID"
date

CESM="/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"
HURR="/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg4096"

RESULTS_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3z_z1a/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
CSV_PATH="$RESULTS_DIR/z1a_cpu_validation.csv"

echo "=== Running Z1-A: full dataset (unfiltered SUM) ==="
python3 /home/u4063895/workspace/gpu-byteplane-reliability-nmr/scripts/run_z1a_cpu_validation.py \
    --dataset "$CESM" \
    --dataset "$HURR" \
    --csv "$CSV_PATH"

echo ""
echo "=== Results ==="
head -30 "$CSV_PATH"
echo "..."
echo "Total rows: $(wc -l < "$CSV_PATH")"

echo ""
echo "=== Z1-A complete ==="
date
