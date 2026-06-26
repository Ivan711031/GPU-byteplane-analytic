#!/bin/bash
#SBATCH --job-name=rl_15_prep
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=/home/u4063895/gpu-byteplane-scan-experiments/logs/rl_15_prep_%j.log
#SBATCH --error=/home/u4063895/gpu-byteplane-scan-experiments/logs/rl_15_prep_%j.err
set -euo pipefail

# Phase 1.5 data preparation: raw data generation + artifact conversion + fault plans
# All three strategies in one CPU job (should fit under 55 min)

REPO_ROOT="${REPO_ROOT:-/home/u4063895/gpu-byteplane-scan-experiments}"
cd "$REPO_ROOT"
JOB_ID="${SLURM_JOB_ID:-nojob}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

RAW_BIN="$REPO_ROOT/buff_encoder/synth_datasets"
ART_ROOT="${RELIABILITY_ARTIFACT_ROOT:-/work/u4063895/datasets/reliability_layer1}"
PARAMS_JSON="$REPO_ROOT/results/phase1_5_strategy_params.json"
HANDOFF_DIR="$REPO_ROOT/handoff/job_done"
LOGS_DIR="$REPO_ROOT/logs"
mkdir -p "$HANDOFF_DIR" "$LOGS_DIR"

# Use separate raw root for N=1e8 to avoid overwriting existing N=1e6 files
RAW_15="/work/u4063895/datasets/reliability_layer1_phase1_5"
RAW_15_DIR="$RAW_15/reliability_layer1/raw"
mkdir -p "$RAW_15_DIR"

echo "[$TIMESTAMP] Phase 1.5 data preparation (job $JOB_ID)"
echo "Artifact root: $ART_ROOT"
echo "Raw (Phase 1.5): $RAW_15_DIR"

# Step 1: Generate N=1e8 raw files for all datasets
for ds in sensor uniform; do
    raw_file="$RAW_15_DIR/${ds}.f64le.bin"
    if [[ -f "$raw_file" ]]; then
        echo "  Raw $ds N=1e8 already exists, skipping"
    else
        echo "  Generating raw $ds N=1e8..."
        "$RAW_BIN" --dataset "$ds" --count 100000000 --seed 20260413 --out-dir "$RAW_15_DIR"
        echo "  Raw $ds done ($(du -h "$raw_file" | cut -f1))"
    fi
done

# Copy heavy_tailed and zipfian N=1e8 files from existing location
for ds in heavy_tailed zipfian; do
    src="/work/u4063895/datasets/reliability_layer1/raw/${ds}.f64le.bin"
    dst="$RAW_15_DIR/${ds}.f64le.bin"
    if [[ ! -f "$dst" ]]; then
        echo "  Copying raw $ds from $src..."
        cp "$src" "$dst"
        cp "$(dirname "$src")/${ds}.meta.json" "$RAW_15_DIR/" 2>/dev/null || true
        echo "  Raw $ds copied ($(du -h "$dst" | cut -f1))"
    else
        echo "  Raw $ds already exists"
    fi
done

# Step 2: Convert artifacts for all 3 strategies
for sid in native_scale1 per_dataset hybrid_scale100_plus_shifted; do
    echo "[$(date)] Converting artifacts for $sid..."
    SKIP_CSV="$LOGS_DIR/phase1_5_conversion_skipped_${sid}_${JOB_ID}.csv"
    python3 "$REPO_ROOT/scripts/convert_phase1_5_artifacts.py" \
        --strategy-id "$sid" \
        --params-json "$PARAMS_JSON" \
        --datasets sensor uniform heavy_tailed zipfian \
        --n-rows 100000000 \
        --raw-data-root "$RAW_15" \
        --artifact-root "$ART_ROOT" \
        --seed 20260413 \
        --skip-csv "$SKIP_CSV" 2>&1
    echo "  Artifacts for $sid done"
done

# Step 3: Generate fault plans for all 3 strategies
for sid in native_scale1 per_dataset hybrid_scale100_plus_shifted; do
    echo "[$(date)] Generating fault plans for $sid..."
    SKIP_CSV="$LOGS_DIR/phase1_5_conversion_skipped_${sid}_${JOB_ID}.csv"
    # Pass RELIABILITY_ARTIFACT_ROOT via env for the fault plan generator
    RELIABILITY_ARTIFACT_ROOT="$ART_ROOT" \
    python3 "$REPO_ROOT/scripts/generate_phase1_5_fault_plans.py" \
        --strategy-id "$sid" \
        --params-json "$PARAMS_JSON" \
        --datasets sensor uniform heavy_tailed zipfian \
        --n-rows 100000000 \
        --rates 1e-08 1e-07 1e-06 1e-05 1e-04 \
        --seeds 0-29 \
        --skip-csv "$SKIP_CSV" 2>&1
    echo "  Fault plans for $sid done"
done

echo "[$(date)] Data preparation complete"

# Write handoff marker
cat > "$HANDOFF_DIR/job_${JOB_ID}.json" <<EOF
{
  "job_id": $JOB_ID,
  "job_name": "rl_15_prep",
  "exit_status": 0,
  "finished_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "workdir": "$REPO_ROOT",
  "stdout": "/home/u4063895/gpu-byteplane-scan-experiments/logs/rl_15_prep_${JOB_ID}.log",
  "next_action": "Submit 12 canonical matrix GPU jobs (3 strategies x 4 datasets), then combined review"
}
EOF
echo "ALL PASSED"
