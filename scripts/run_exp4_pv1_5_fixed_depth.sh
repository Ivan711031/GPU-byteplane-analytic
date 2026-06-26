#!/bin/bash
# run_exp4_pv1_5_fixed_depth.sh: Exp4 fixed-depth encoded COUNT baseline sweep
#
# Usage:
#   sbatch --account=<PROJECT_ID> scripts/run_exp4_pv1_5_fixed_depth.sh \
#     --datasets uniform,heavy_tailed \
#     --artifacts exact \
#     --selectivities "50 90 99"

#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module purge
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan

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

BIN="$ROOT_DIR/build/exp4/bench_progressive_filter"
if [[ ! -x "$BIN" ]]; then
  echo "ERROR: benchmark binary missing: $BIN" >&2
  echo "Build on login node before submission." >&2
  exit 2
fi

DATASETS="uniform,heavy_tailed"
ARTIFACTS="exact"
SELECTIVITIES="50 90 99"
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

timestamp="$(date +%Y%m%d_%H%M%S)"
job_id="${SLURM_JOB_ID:-local}"
gpu_tag="${GPU_NAME// /_}"
run_dir="$ROOT_DIR/results/exp4/fixed_depth_count_${timestamp}_job${job_id}_${gpu_tag}"
mkdir -p "$run_dir"

declare -A ARTIFACT_ROOTS=(
  [exact]="/work/$USER/datasets/synthetic/dev_buff_exp3"
  [p3]="/work/$USER/datasets/synthetic/dev_buff_exp4_p3"
  [p6]="/work/$USER/datasets/synthetic/dev_buff_exp4_p6"
)

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
      log_file="${run_dir}/run_${dataset}_${artifact}_s${s}.log"

      ((++TOTAL_RUNS))
      echo "[Run $TOTAL_RUNS] $dataset / $artifact / s=$s / fixed_depth" >&2

      if "$BIN" \
        --device 0 \
        --encoded-root "$encoded_root" \
        --threshold "$threshold" \
        --validate \
        --baseline-mode fixed_depth \
        --csv "$csv_file" \
        --warmup "$WARMUP" \
        --iters "$ITERS" \
        --block "$BLOCK" 2>&1 | tee "$log_file" | tail -5; then

        gpu_count=$(tail -1 "$csv_file" | cut -d',' -f12)
        cpu_encoded=$(tail -1 "$csv_file" | cut -d',' -f13)

        if [[ "$gpu_count" != "$cpu_encoded" ]]; then
          echo "  ✗ GATE FAILURE: gpu=$gpu_count != cpu_enc=$cpu_encoded" >&2
          GATE_FAILURES+=("${dataset}_${artifact}_s${s}")
          ((++FAILED_RUNS))
          continue
        fi

        ((++PASSED_RUNS))
        CSV_FILES+=("$csv_file")
        echo "  ✓ Pass" >&2
      else
        echo "  ✗ Benchmark exited with error" >&2
        GATE_FAILURES+=("${dataset}_${artifact}_s${s}_ERROR")
        ((++FAILED_RUNS))
      fi
    done
  done
done

summary_file="${run_dir}/fixed_depth_count_summary.csv"
if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
  echo "ERROR: No valid CSV files to merge" >&2
  exit 1
fi

head -1 "${CSV_FILES[0]}" > "$summary_file"
for csv in "${CSV_FILES[@]}"; do
  tail -1 "$csv" >> "$summary_file"
done

meta_file="${run_dir}/run_meta.txt"
cat > "$meta_file" <<EOF
=== Exp4 PV1-5 Fixed-Depth COUNT Baseline ===
Job ID: ${SLURM_JOB_ID:-local}
Hostname: $(hostname)
Start Time: $(date -Is)
GPU: $GPU_NAME
Git Commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)
Git Dirty: $(if [[ -n "$(git status --short 2>/dev/null)" ]]; then echo dirty; else echo clean; fi)

=== Parameters ===
Datasets: $DATASETS
Artifacts: $ARTIFACTS
Selectivities: $SELECTIVITIES
Iterations: $ITERS
Warmup: $WARMUP
Block Size: $BLOCK
Baseline Mode: fixed_depth

=== Results ===
Total Runs: $TOTAL_RUNS
Passed Runs: $PASSED_RUNS
Failed Runs: $FAILED_RUNS
Summary CSV: $summary_file

=== Blocking Gate Failures ===
Count: ${#GATE_FAILURES[@]}
EOF

if [[ ${#GATE_FAILURES[@]} -gt 0 ]]; then
  for f in "${GATE_FAILURES[@]}"; do echo "  - $f" >> "$meta_file"; done
else
  echo "  None" >> "$meta_file"
fi

repro_file="${run_dir}/repro_command.txt"
cat > "$repro_file" <<EOF
sbatch --account=<PROJECT_ID> scripts/run_exp4_pv1_5_fixed_depth.sh \\
  --datasets $DATASETS \\
  --artifacts $ARTIFACTS \\
  --selectivities "$SELECTIVITIES" \\
  --iters $ITERS \\
  --warmup $WARMUP \\
  --block $BLOCK
EOF

readme_file="${run_dir}/README.md"
cat > "$readme_file" <<EOF
# Exp4 PV1-5 Fixed-Depth Encoded COUNT Baseline

This directory contains the fixed-depth encoded COUNT baseline for Exp4.

## Definition

- Kernel path: \`fixed_depth_rowpack16_count\`
- Benchmark: \`fixed_depth_filter\`
- Load strategy: \`rowpack16\`
- Baseline mode: \`fixed_depth\`
- Execution rule: every processed row reads the full active encoded depth for its segment; no row-level or pack-level predicate early-exit is allowed.

## Matrix

- Datasets: $DATASETS
- Artifacts: $ARTIFACTS
- Selectivities: $SELECTIVITIES
- Iterations: $ITERS
- Warmup: $WARMUP

## Files

- \`fixed_depth_count_summary.csv\`: merged baseline summary
- \`run_meta.txt\`: provenance and gate summary
- \`repro_command.txt\`: sbatch reproduction command
- \`run_*.csv\`: per-run benchmark output
- \`run_*.log\`: per-run stderr/stdout tail logs

## Correctness Gate

Every run must satisfy \`gpu_count == cpu_encoded_count\`.
EOF

echo "" >&2
echo "=== Fixed-Depth Baseline Complete ===" >&2
echo "Output directory: $run_dir" >&2

if [[ ${#GATE_FAILURES[@]} -gt 0 ]]; then
  echo "⚠ ${#GATE_FAILURES[@]} gate failures" >&2
  exit 1
fi

echo "✓ All gates passed" >&2
exit 0
