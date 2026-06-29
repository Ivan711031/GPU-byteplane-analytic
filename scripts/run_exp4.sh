#!/bin/bash
# run_exp4.sh: Selectivity sweep runner for Exp4 progressive filter benchmarks
#
# This script is designed to be submitted directly via sbatch:
#   sbatch scripts/run_exp4.sh \
#     --datasets uniform,heavy_tailed \
#     --artifacts exact,p3,p6 \
#     --selectivities "1 5 10 25 50 75 90 95 99"
#
# Output structure (mirrors scripts/legacy/root_runners/run_exp3.sh pattern):
#   results/exp4/run_${timestamp}_job${job_id}_${gpu_tag}/
#   ├── run_${dataset}_${artifact}_s${selectivity}.csv   # per-run CSV
#   ├── sweep_summary.csv                                 # merged results
#   └── run_meta.txt                                      # metadata + gate stats
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
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
# Repo root. When sbatch copies the script to /var/spool/slurmd/, BASH_SOURCE
# and git rev-parse do not work. Use hardcoded path (same as scripts/exp4_smoke_metrics.sh).
ROOT_DIR="/home/$USER/workspace/gpu-byteplane-scan-experiments"
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: ROOT_DIR not found: $ROOT_DIR" >&2
  exit 2
fi
cd "$ROOT_DIR"
# Build if missing
if [[ ! -f "$ROOT_DIR/build/exp4/bench_progressive_filter" ]]; then
  echo "Building bench_progressive_filter..." >&2
  cmake -S benchmarks/experiment4 -B build/exp4 \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
  cmake --build build/exp4 -j
fi
# ============================================================================
# Parse arguments
# ============================================================================
DATASETS=""
ARTIFACTS=""
SELECTIVITIES=""
ITERS=200
WARMUP=10
BLOCK=256
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
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done
if [[ -z "$DATASETS" || -z "$ARTIFACTS" || -z "$SELECTIVITIES" ]]; then
  echo "Usage: $0 --datasets D1,D2 --artifacts exact,p3,p6 --selectivities '1 5 ...'" >&2
  exit 1
fi
# ============================================================================
# Setup output directory (matches scripts/legacy/root_runners/run_exp3.sh pattern)
# ============================================================================
timestamp="$(date +%Y%m%d_%H%M%S)"
job_id="${SLURM_JOB_ID:-local}"
gpu_tag="${GPU_NAME// /_}"
run_dir="$ROOT_DIR/results/exp4/run_${timestamp}_job${job_id}_${gpu_tag}"
mkdir -p "$run_dir"
echo "=== Exp4 Selectivity Sweep ===" >&2
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
# Main sweep
# ============================================================================
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
IFS=',' read -ra ARTIFACT_ARRAY <<< "$ARTIFACTS"
declare -a CSV_FILES=()
declare -a GATE_FAILURES=()
declare -a FIDELITY_MISMATCHES=()
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
  # Cache thresholds for this dataset
  for s in $SELECTIVITIES; do
    s=$(echo "$s" | xargs)
    compute_threshold "$s" "$raw_path" "$dataset" > /dev/null
  done
  for artifact in "${ARTIFACT_ARRAY[@]}"; do
    artifact=$(echo "$artifact" | xargs)
    artifact_root="${ARTIFACT_ROOTS[$artifact]}"
    if [[ ! -d "$artifact_root" ]]; then
      echo "WARNING: Artifact root not found: $artifact_root" >&2
      continue
    fi
    encoded_root="${artifact_root}/${dataset}"
    if [[ ! -d "$encoded_root" ]]; then
      echo "WARNING: Encoded root not found: $encoded_root" >&2
      continue
    fi
    for s in $SELECTIVITIES; do
      s=$(echo "$s" | xargs)
      threshold=$(compute_threshold "$s" "$raw_path" "$dataset")
      csv_file="${run_dir}/run_${dataset}_${artifact}_s${s}.csv"
      ((++TOTAL_RUNS))
      echo "[Run $TOTAL_RUNS] $dataset / $artifact / s=$s" >&2
      echo "  Threshold: $threshold" >&2
      if "$ROOT_DIR/build/exp4/bench_progressive_filter" \
        --device 0 \
        --encoded-root "$encoded_root" \
        --threshold "$threshold" \
        --validate \
        --csv "$csv_file" \
        --warmup "$WARMUP" \
        --iters "$ITERS" \
        --block "$BLOCK" 2>&1 | tee "${run_dir}/run_${dataset}_${artifact}_s${s}.log" | tail -5; then
        # Check blocking gate
        gpu_count=$(tail -1 "$csv_file" | cut -d',' -f7)
        cpu_encoded=$(tail -1 "$csv_file" | cut -d',' -f8)
        cpu_raw=$(tail -1 "$csv_file" | cut -d',' -f9)
        if [[ "$gpu_count" != "$cpu_encoded" ]]; then
          echo "  ✗ GATE FAILURE: gpu=$gpu_count != cpu_enc=$cpu_encoded" >&2
          GATE_FAILURES+=("${dataset}_${artifact}_s${s}")
          ((++FAILED_RUNS))
          continue
        fi
        if [[ "$cpu_encoded" != "$cpu_raw" ]]; then
          echo "  ⚠ Fidelity mismatch (expected for bounded): cpu_enc=$cpu_encoded != cpu_raw=$cpu_raw" >&2
          FIDELITY_MISMATCHES+=("${dataset}_${artifact}_s${s}")
        fi
        ((++PASSED_RUNS))
        CSV_FILES+=("$csv_file")
        echo "  ✓ Pass" >&2
      else
        echo "  ✗ Benchmark exited with error" >&2
        GATE_FAILURES+=("${dataset}_${artifact}_s${s}_ERROR")
        ((++FAILED_RUNS))
      fi
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
# Metadata
# ============================================================================
meta_file="${run_dir}/run_meta.txt"
cat > "$meta_file" <<EOF
=== Exp4 Sweep Metadata ===
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
cat >> "$meta_file" <<EOF
=== Fidelity Mismatches (non-blocking) ===
Count: ${#FIDELITY_MISMATCHES[@]}
EOF
if [[ ${#FIDELITY_MISMATCHES[@]} -gt 0 ]]; then
  for m in "${FIDELITY_MISMATCHES[@]}"; do echo "  - $m" >> "$meta_file"; done
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
