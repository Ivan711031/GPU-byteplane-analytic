#!/usr/bin/env bash
#SBATCH -J nmr_r4_logical_diversity
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%j.out

# NMR-R4: Logical diversity under structured clustered fault injection.
# Compares placement strategies (no-diversity, logical-spatial,
# logical-spatial+logical-temporal) under clustered fault models.
#
# Usage:
#   DATASET=hurricane_u sbatch scripts/run_nmr_r4_logical_diversity.sbatch
#   DATASET=cesm_atm_cloud sbatch scripts/run_nmr_r4_logical_diversity.sbatch
#
# Or run both sequentially:
#   sbatch scripts/run_nmr_r4_logical_diversity.sbatch  # runs both

set -euo pipefail

DATASET="${DATASET:-}"

echo "=== NMR-R4: Logical Diversity Under Structured Clustered Fault Injection ==="
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

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$PROJ_ROOT/scripts"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$SCRIPT_DIR"
RESULTS_DIR="$PROJ_ROOT/results/reliability_layer1/phase4/nmr_r4_logical_diversity/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

run_one_dataset() {
    local dataset=$1
    local n_rows=$2
    local plane_dir=$3
    local rates=$4
    local seeds=$5
    local output="$RESULTS_DIR/nmr_r4_${dataset}.csv"

    echo ""
    echo "=== Dataset: $dataset (${n_rows} rows) ==="
    echo "Plane dir: $plane_dir"
    echo "Rates: $rates"
    echo "Seeds: $seeds"

    python3 "$SCRIPT_DIR/run_nmr_r4_logical_diversity.py" \
        --plane-dir "$plane_dir" \
        --dataset "$dataset" \
        --n-rows "$n_rows" \
        --fault-rates $rates \
        --seeds $seeds \
        --segment-size 4096 \
        --output "$output"

    echo "Wrote: $output"
}

# ── Dataset configurations ──
HURRICANE_ROWS=25000000
CESM_ROWS=168480000
LOCALITY_DATA_ROOT="${LOCALITY_DATA_ROOT:-/work/${USER}/datasets/locality_sensitivity}"

if [ -n "$DATASET" ]; then
    case "$DATASET" in
        hurricane_u)
            run_one_dataset "hurricane_u" "$HURRICANE_ROWS" \
                "$LOCALITY_DATA_ROOT/hurricane_u/seg4096" \
                "1e-5 5e-5 1e-4" "0 1 2"
            ;;
        cesm_atm_cloud)
            run_one_dataset "cesm_atm_cloud" "$CESM_ROWS" \
                "$LOCALITY_DATA_ROOT/cesm_atm_cloud/seg4096" \
                "2e-6 2e-5 2e-4" "0 1 2"
            ;;
        *)
            echo "Unknown dataset: $DATASET" >&2
            exit 1
            ;;
    esac
else
    # Run both datasets
    run_one_dataset "hurricane_u" "$HURRICANE_ROWS" \
        "$LOCALITY_DATA_ROOT/hurricane_u/seg4096" \
        "1e-5 5e-5 1e-4" "0 1 2"

    run_one_dataset "cesm_atm_cloud" "$CESM_ROWS" \
        "$LOCALITY_DATA_ROOT/cesm_atm_cloud/seg4096" \
        "2e-6 2e-5 2e-4" "0 1 2"
fi

echo ""
echo "=== NMR-R4 Complete ==="
echo "Results in: $RESULTS_DIR"
ls -lh "$RESULTS_DIR"/
date
