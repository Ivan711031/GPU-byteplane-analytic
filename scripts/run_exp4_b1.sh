#!/bin/bash
# run_exp4_b1.sh: Capped-k COUNT sweep for Exp4-B1
#
# This script sweeps k=1..max_plane_count for each (dataset, artifact, selectivity).
# It is designed to be submitted via sbatch on TWCC Nano5.
#
# Usage:
#   sbatch --account=<ACCOUNT> scripts/run_exp4_b1.sh \
#     --datasets uniform,heavy_tailed \
#     --artifacts exact \
#     --selectivities "50 90 99"
#   sbatch --account=<ACCOUNT> scripts/run_exp4_b1.sh \
#     --datasets uniform --artifacts p10 --selectivities 50 \
#     --v2-root /work/$USER/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p \
#     --raw-root /work/$USER/datasets/synthetic/dev
#
# Output structure:
#   results/exp4/b1_${timestamp}_job${job_id}_${gpu_tag}/
#   ├── run_${dataset}_${artifact}_s${selectivity}_k${k}.csv
#   ├── rounds_${dataset}_${artifact}_s${selectivity}_k${k}.csv
#   ├── sweep_summary.csv
#   └── run_meta.txt
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -o results/exp4/b1_%j.out
#SBATCH -e results/exp4/b1_%j.err
set -euo pipefail
# ============================================================================
# Environment setup
# ============================================================================
module purge
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan
# Hardware validation: must be H200
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi
ROOT_DIR="/home/$USER/workspace/gpu-byteplane-scan-experiments"
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: ROOT_DIR not found: $ROOT_DIR" >&2
  exit 2
fi
cd "$ROOT_DIR"
# Build if missing or --rebuild
REBUILD=0
if [[ ! -f "$ROOT_DIR/build/exp4/bench_progressive_filter" ]]; then
  REBUILD=1
fi
# Build helper
do_build() {
  echo "Building bench_progressive_filter..." >&2
  cmake -S benchmarks/experiment4 -B build/exp4 \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
  cmake --build build/exp4 -j
}
# ============================================================================
# Parse arguments
# ============================================================================
DATASETS=""
ARTIFACTS=""
SELECTIVITIES=""
ITERS=200
WARMUP=10
BLOCK=256
V2_ROOT=""
RAW_ROOT=""
FULL_BUDGET_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --artifacts)
      ARTIFACTS="$2"
      shift 2
      ;;
    --selectivities)
      SELECTIVITIES="$2"
      shift 2
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --block)
      BLOCK="$2"
      shift 2
      ;;
    --rebuild)
      REBUILD=1
      shift
      ;;
    --v2-root)
      V2_ROOT="$2"
      shift 2
      ;;
    --raw-root)
      RAW_ROOT="$2"
      shift 2
      ;;
    --full-budget-only)
      FULL_BUDGET_ONLY=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done
if [[ -z "$DATASETS" || -z "$ARTIFACTS" || -z "$SELECTIVITIES" ]]; then
  echo "Usage: $0 --datasets D1,D2 --artifacts exact --selectivities '50 90 99'" >&2
  exit 1
fi
# Rebuild if requested
if [[ "$REBUILD" == 1 ]]; then
  do_build
fi
# ============================================================================
# Setup output directory
# ============================================================================
timestamp="$(date +%Y%m%d_%H%M%S)"
job_id="${SLURM_JOB_ID:-local}"
gpu_tag="${GPU_NAME// /_}"
run_dir="$ROOT_DIR/results/exp4/b1_${timestamp}_job${job_id}_${gpu_tag}"
mkdir -p "$run_dir"
echo "=== Exp4-B1 Capped-k COUNT Sweep ===" >&2
echo "Datasets: $DATASETS" >&2
echo "Artifacts: $ARTIFACTS" >&2
echo "Selectivities: $SELECTIVITIES" >&2
echo "Output: $run_dir" >&2
echo "GPU: $GPU_NAME" >&2
echo "" >&2
# ============================================================================
# Artifact mapping
# ============================================================================
declare -A ARTIFACT_ROOTS=(
  [exact]="/work/$USER/datasets/synthetic/dev_buff_exp3"
  [p3]="/work/$USER/datasets/synthetic/dev_buff_exp4_p3"
  [p6]="/work/$USER/datasets/synthetic/dev_buff_exp4_p6"
)
# ============================================================================
# Utility: compute threshold from raw FP64 quantile
# ============================================================================
compute_threshold() {
  local selectivity=$1
  local raw_path=$2
  local dataset_name=$3
  local cache_var="threshold_${dataset_name}_${selectivity}"
  if [[ -v "$cache_var" ]]; then
    printf '%s\n' "${!cache_var}"
    return 0
  fi
  local quantile threshold
  quantile=$(python3 -c "print(1 - $selectivity/100)")
  threshold=$(python3 -c "
import numpy as np
data = np.fromfile('${raw_path}', dtype=np.float64)
print(np.quantile(data, ${quantile}))
")
  eval "${cache_var}='$threshold'"
  printf '%s\n' "$threshold"
}
# ============================================================================
# Utility: get max_plane_count from manifest
# ============================================================================
get_max_planes() {
  local encoded_root=$1
  python3 -c "
import json
with open('${encoded_root}/manifest.json') as f:
    m = json.load(f)
print(m.get('max_plane_count', 8))
"
}
# ============================================================================
# Main sweep
# ============================================================================
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
IFS=',' read -ra ARTIFACT_ARRAY <<< "$ARTIFACTS"
declare -a CSV_FILES=()
declare -a GATE_FAILURES=()
TOTAL_RUNS=0
PASSED_RUNS=0
FAILED_RUNS=0
for dataset in "${DATASET_ARRAY[@]}"; do
  dataset=$(echo "$dataset" | xargs)
  raw_path="/work/$USER/datasets/synthetic/dev/${dataset}.f64le.bin"
  if [[ ! -f "$raw_path" ]]; then
    echo "ERROR: Raw data not found: $raw_path" >&2
    exit 1
  fi
  # Cache thresholds
  for s in $SELECTIVITIES; do
    s=$(echo "$s" | xargs)
    compute_threshold "$s" "$raw_path" "$dataset" > /dev/null
  done
  for artifact in "${ARTIFACT_ARRAY[@]}"; do
    artifact=$(echo "$artifact" | xargs)
    if [[ -n "$V2_ROOT" ]]; then
      encoded_root="${V2_ROOT}/${dataset}_${artifact}"
    else
      artifact_root="${ARTIFACT_ROOTS[$artifact]:-}"
      if [[ -z "$artifact_root" || ! -d "$artifact_root" ]]; then
        echo "WARNING: Legacy artifact root not found for '$artifact'" >&2
        continue
      fi
      encoded_root="${artifact_root}/${dataset}"
    fi
    if [[ ! -d "$encoded_root" ]]; then
      echo "WARNING: Encoded root not found: $encoded_root" >&2
      continue
    fi
    max_planes=$(get_max_planes "$encoded_root")
    echo "Dataset: $dataset / Artifact: $artifact / Max planes: $max_planes" >&2
    for s in $SELECTIVITIES; do
      s=$(echo "$s" | xargs)
      threshold=$(compute_threshold "$s" "$raw_path" "$dataset")
      k_start=1; [[ "$FULL_BUDGET_ONLY" == 1 ]] && k_start=$max_planes
      for k in $(seq $k_start $max_planes); do
        csv_file="${run_dir}/run_${dataset}_${artifact}_s${s}_k${k}.csv"
        rounds_csv="${run_dir}/rounds_${dataset}_${artifact}_s${s}_k${k}.csv"
        ((++TOTAL_RUNS))
        echo "[Run $TOTAL_RUNS] $dataset / $artifact / s=$s / k=$k" >&2
        cmd=("$ROOT_DIR/build/exp4/bench_progressive_filter"
          --device 0
          --encoded-root "$encoded_root"
          --threshold "$threshold"
          --validate
          --csv "$csv_file"
          --output-rounds-csv "$rounds_csv"
          --max-filter-planes "$k"
          --warmup "$WARMUP"
          --iters "$ITERS"
          --block "$BLOCK")
        if [[ -n "$RAW_ROOT" ]]; then
          cmd+=(--raw-root "$RAW_ROOT")
        fi
        if "${cmd[@]}" 2>&1 | tee "${run_dir}/run_${dataset}_${artifact}_s${s}_k${k}.log" | tail -5; then
          # Check blocking gate
          # NOTE: column offsets must match bench_progressive_filter.cu CSV schema.
          # After 2026-05-01 schema hardening: gpu_count=f12, cpu_encoded_count=f13.
          gpu_count=$(tail -1 "$csv_file" | cut -d',' -f12)
          cpu_encoded=$(tail -1 "$csv_file" | cut -d',' -f13)
          if [[ "$gpu_count" != "$cpu_encoded" ]]; then
            echo "  ✗ GATE FAILURE: gpu=$gpu_count != cpu_enc=$cpu_encoded" >&2
            GATE_FAILURES+=("${dataset}_${artifact}_s${s}_k${k}")
            ((++FAILED_RUNS))
            continue
          fi
          ((++PASSED_RUNS))
          CSV_FILES+=("$csv_file")
          echo "  ✓ Pass" >&2
        else
          echo "  ✗ Benchmark exited with error" >&2
          GATE_FAILURES+=("${dataset}_${artifact}_s${s}_k${k}_ERROR")
          ((++FAILED_RUNS))
        fi
      done
      echo "" >&2
    done
  done
done
# ============================================================================
# Merge CSVs into sweep summary
# ============================================================================
summary_file="${run_dir}/sweep_summary.csv"
if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
  echo "ERROR: No valid CSV files to merge" >&2
  exit 1
fi
head -1 "${CSV_FILES[0]}" > "$summary_file"
for csv in "${CSV_FILES[@]}"; do
  tail -1 "$csv" >> "$summary_file"
done
echo "Summary: $summary_file ($(tail -n +2 "$summary_file" | wc -l) rows)" >&2
# ============================================================================
# Merge rounds sidecars
# ============================================================================
rounds_summary="${run_dir}/rounds_summary.csv"
first_rounds=true
for rounds_csv in "${run_dir}"/rounds_*.csv; do
  if [[ -f "$rounds_csv" ]]; then
    if $first_rounds; then
      head -1 "$rounds_csv" > "$rounds_summary"
      first_rounds=false
    fi
    tail -n +2 "$rounds_csv" >> "$rounds_summary"
  fi
done
if [[ -f "$rounds_summary" ]]; then
  echo "Rounds summary: $rounds_summary ($(tail -n +2 "$rounds_summary" | wc -l) rows)" >&2
fi
# ============================================================================
# Metadata
# ============================================================================
meta_file="${run_dir}/run_meta.txt"
cat > "$meta_file" <<EOF
=== Exp4-B1 Sweep Metadata ===
Job ID: ${SLURM_JOB_ID:-local}
Hostname: $(hostname)
Start Time: $(date -Is)
GPU: $GPU_NAME
=== Parameters ===
Datasets: $DATASETS
Artifacts: $ARTIFACTS
Selectivities: $SELECTIVITIES
Iterations: $ITERS
Warmup: $WARMUP
Block Size: $BLOCK
Raw Root: ${RAW_ROOT:-auto}
=== Results ===
Total Runs: $TOTAL_RUNS
Passed Runs: $PASSED_RUNS
Failed Runs: $FAILED_RUNS
=== Blocking Gate Failures ===
Count: ${#GATE_FAILURES[@]}
EOF
if [[ ${#GATE_FAILURES[@]} -gt 0 ]]; then
  for f in "${GATE_FAILURES[@]}"; do echo "  - $f" >> "$meta_file"; done
else
  echo "  None" >> "$meta_file"
fi
echo "" >&2
echo "=== Sweep Complete ===" >&2
echo "Output directory: $run_dir" >&2
if [[ ${#GATE_FAILURES[@]} -gt 0 ]]; then
  echo "⚠ ${#GATE_FAILURES[@]} gate failures" >&2
  exit 1
else
  echo "✓ All gates passed" >&2
  exit 0
fi
