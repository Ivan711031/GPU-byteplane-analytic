#!/bin/bash
# run_exp4_byte_mask_formal_sweep.sbatch
#
# Formal H200 A/B sweep for Exp4: predicated (baseline) vs byte_mask (experimental).
# Runs bench_progressive_filter --validate across:
#   datasets/artifacts:  heavy_tailed_p6 sensor_p10 uniform_p10 zipfian_p8
#   selectivities:       1 10 50 90 99
#   block sizes:         128 256 512
#   strategies:          predicated byte_mask
#   repeats:             5
#   total:               4 × 5 × 3 × 2 × 5 = 600 benchmark invocations
#
# Usage:
#   sbatch --account=<ACCOUNT> scripts/run_exp4_byte_mask_formal_sweep.sbatch
#
# No embedded --account. Must be provided on the command line.
#
# Output (relative to ROOT_DIR):
#   results/exp4/byte_mask_formal_sweep_<timestamp>_job<JOBID>_<GPU>/
#     combined.csv          — all rows (one header + 600 data rows)
#     summary_pairs.csv     — predicated/byte_mask ms and speedup per triplet+repeat
#     run_<key>.log         — per-case stderr log
#     run_meta.txt          — matrix, git, toolchain info
#     failures.txt          — gate failure details (if any)
#     gpu_util.csv          — GPU utilization log
#   handoff/job_done/job_<SLURM_JOB_ID>.json
#SBATCH --job-name=exp4-byte-mask-formal-sweep
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
set -euo pipefail
# ==============================================================================
# Environment
# ==============================================================================
module purge || true
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan
# ==============================================================================
# Dynamic workspace detection
# ==============================================================================
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: ROOT_DIR not found: $ROOT_DIR" >&2
  exit 2
fi
cd "$ROOT_DIR"
# ==============================================================================
# GPU identification
# ==============================================================================
GPU_NAME="unknown"
GPU_TAG="unknown"
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ' 2>/dev/null || echo "unknown")
  GPU_TAG="${GPU_NAME// /_}"
  GPU_TAG="${GPU_TAG:-unknown}"
fi
# ==============================================================================
# Paths (set up before trap so handoff marker knows where to write)
# ==============================================================================
V2_ROOT="${WORK_DIR}/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
RAW_ROOT="${WORK_DIR}/datasets/synthetic/dev"
timestamp=$(date +%Y%m%d_%H%M%S)
job_id="${SLURM_JOB_ID:-local}"
run_dir="${ROOT_DIR}/results/exp4/byte_mask_formal_sweep_${timestamp}_job${job_id}_${GPU_TAG}"
handoff_dir="${ROOT_DIR}/handoff/job_done"
handoff_marker="${handoff_dir}/job_${job_id}.json"
combined_csv="${run_dir}/combined.csv"
pairs_csv="${run_dir}/summary_pairs.csv"
meta_file="${run_dir}/run_meta.txt"
failures_file="${run_dir}/failures.txt"
gpu_util_log="${run_dir}/gpu_util.csv"
# ==============================================================================
# Handoff marker via trap (set before GPU validation so exit 2 also writes marker)
# ==============================================================================
write_handoff_marker() {
  local status=$?
  local gpulog=""
  if [[ -n "${TRACKER_PID:-}" ]]; then
    kill "$TRACKER_PID" 2>/dev/null || true
    wait "$TRACKER_PID" 2>/dev/null || true
  fi
  if [[ -f "$gpu_util_log" ]]; then
    gpulog="$gpu_util_log"
  fi
  local git_commit
  git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
  local next_action="Review combined CSV, summary pairs, and gate results."
  if [[ -f "$failures_file" ]] && [[ -s "$failures_file" ]]; then
    next_action="Gate failures detected; investigate failures.txt and logs before re-running."
  fi
  mkdir -p "$handoff_dir"
  cat > "$handoff_marker" <<EOF
{
  "job_id": "${job_id}",
  "job_name": "exp4-byte-mask-formal-sweep",
  "exit_status": ${status},
  "finished_at": "$(date -Is)",
  "workdir": "$(pwd)",
  "git_commit": "${git_commit}",
  "result_dir": "${run_dir}",
  "combined_csv": "${combined_csv}",
  "summary_pairs_csv": "${pairs_csv}",
  "gpu_util_log": "${gpulog}",
  "next_action": "${next_action}"
}
EOF
}
trap write_handoff_marker EXIT
# ==============================================================================
# Hardware validation: must be H200
# ==============================================================================
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi
# ==============================================================================
# Create output directory
# ==============================================================================
mkdir -p "$run_dir" "$handoff_dir"
# ==============================================================================
# GPU utilization tracker (background, best-effort, 60s interval)
# ==============================================================================
echo "timestamp,gpu_index,util_pct,mem_used_mb,mem_total_mb" > "$gpu_util_log"
(
  while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits | tr -d ' ' \
    | awk -v ts="$(date +%Y-%m-%dT%H:%M:%S)" '{print ts","$0}' >> "$gpu_util_log"
    sleep 60
  done
) &
TRACKER_PID=$!
# ==============================================================================
# Toolchain verification
# ==============================================================================
echo "=== Toolchain ===" >&2
echo "nvcc: $(which nvcc) $($(which nvcc) --version 2>&1 | grep release)" >&2
echo "gcc: $(gcc --version 2>&1 | head -1)" >&2
echo "cmake: $(cmake --version 2>&1 | head -1)" >&2
# ==============================================================================
# Matrix definition
#   Format: dataset,artifact_name,encoded_subdir
# ==============================================================================
declare -a MATRIX_ENTRIES=()
declare -a MATRIX_LABELS=()
register_run() {
  local label="$1"
  MATRIX_LABELS+=("$label")
  MATRIX_ENTRIES+=("$label|$2|$3|$4|$5|$6|$7")
}
for dataset in heavy_tailed sensor uniform zipfian; do
  case "$dataset" in
    heavy_tailed) artifact_p="p6" ;;
    sensor)       artifact_p="p10" ;;
    uniform)      artifact_p="p10" ;;
    zipfian)      artifact_p="p8" ;;
  esac
  encoded_subdir="${dataset}_${artifact_p}"
  for sel in 1 10 50 90 99; do
    for block in 128 256 512; do
      for strategy in predicated byte_mask; do
        for repeat in 1 2 3 4 5; do
          label="${dataset}_${artifact_p}_s${sel}_b${block}_${strategy}_r${repeat}"
          register_run "$label" "$dataset" "$encoded_subdir" "$sel" "$block" "$strategy" "$repeat"
        done
      done
    done
  done
done
TOTAL_RUNS=${#MATRIX_ENTRIES[@]}
# ==============================================================================
# Threshold computation helper (numpy quantile on raw FP64, cached)
# ==============================================================================
declare -A THRESHOLD_CACHE
compute_threshold() {
  local dataset="$1" selectivity="$2"
  local cache_key="${dataset}_s${selectivity}"
  if [[ -v "THRESHOLD_CACHE[$cache_key]" ]]; then
    printf '%s\n' "${THRESHOLD_CACHE[$cache_key]}"
    return 0
  fi
  local raw_path="${RAW_ROOT}/${dataset}.f64le.bin"
  if [[ ! -f "$raw_path" ]]; then
    echo "ERROR: raw file not found: $raw_path" >&2
    exit 2
  fi
  local quantile threshold
  quantile=$(python3 -c "print(1 - ${selectivity}/100.0)")
  threshold=$(python3 -c "
import numpy as np
data = np.fromfile('${raw_path}', dtype=np.float64)
print(np.quantile(data, ${quantile}))
")
  THRESHOLD_CACHE["$cache_key"]="$threshold"
  printf '%s\n' "$threshold"
}
# ==============================================================================
# Job metadata header
# ==============================================================================
echo "=== Exp4 Byte-Mask Formal Sweep ===" >&2
echo "Job ID: ${job_id}" >&2
echo "Host: $(hostname)" >&2
echo "Time: $(date -Is)" >&2
echo "GPU: ${GPU_NAME}" >&2
echo "Matrix: ${TOTAL_RUNS} runs (4 datasets × 5 sels × 3 blocks × 2 strategies × 5 repeats)" >&2
echo "Output: ${run_dir}" >&2
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader >&2
echo "=== Git ===" >&2
git rev-parse HEAD >&2
git status --short >&2
echo "" >&2
# ==============================================================================
# Build
# ==============================================================================
echo "=== Build ===" >&2
cmake -S benchmarks/experiment4 -B build/exp4 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 2>&1
cmake --build build/exp4 -j 2>&1
echo "" >&2
# ==============================================================================
# Validation: ensure all encoded roots exist
# ==============================================================================
echo "=== Checking artifact paths ===" >&2
declare -A SEEN_ENCODED
for entry in "${MATRIX_ENTRIES[@]}"; do
  IFS='|' read -r label dataset encoded_subdir sel block strategy repeat <<< "$entry"
  encoded_root="${V2_ROOT}/${encoded_subdir}"
  if [[ -z "${SEEN_ENCODED[$encoded_root]:-}" ]]; then
    SEEN_ENCODED["$encoded_root"]=1
    if [[ ! -d "$encoded_root" ]]; then
      echo "ERROR: encoded root not found: $encoded_root" >&2
      exit 2
    fi
    echo "  ✓ $encoded_root" >&2
  fi
done
echo "" >&2
# ==============================================================================
# Run matrix
# ==============================================================================
declare -a CSV_FILES=()
declare -a GATE_FAILURES=()
HEADER_WRITTEN=0
# Associative array for pair summary: key=dataset|sel|block|repeat -> "predicated_ms,byte_mask_ms"
declare -A PAIR_DATA
run_idx=0
for entry in "${MATRIX_ENTRIES[@]}"; do
  IFS='|' read -r label dataset encoded_subdir sel block strategy repeat <<< "$entry"
  ((++run_idx))
  encoded_root="${V2_ROOT}/${encoded_subdir}"
  threshold=$(compute_threshold "$dataset" "$sel")
  log_file="${run_dir}/run_${label}.log"
  csv_file="${run_dir}/run_${label}.csv"
  echo "[${run_idx}/${TOTAL_RUNS}] ${label}  (threshold=${threshold}, block=${block}, strategy=${strategy})" >&2
  # Build and run benchmark
  set +e
  "${ROOT_DIR}/build/exp4/bench_progressive_filter" \
    --device 0 \
    --encoded-root "$encoded_root" \
    --threshold "$threshold" \
    --block "$block" \
    --validate \
    --warp-strategy "$strategy" \
    --csv "$csv_file" \
    --warmup 10 \
    --iters 200 \
    2> "$log_file"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "  ✗ Benchmark exit code ${rc}" >&2
    GATE_FAILURES+=("${label}_EXIT${rc}")
    continue
  fi
  if [[ ! -f "$csv_file" ]]; then
    echo "  ✗ No CSV produced" >&2
    GATE_FAILURES+=("${label}_NOCSV")
    continue
  fi
  # Validate correctness gate from CSV
  # CSV schema columns (verified from smoke output):
  #   f12 = gpu_count, f13 = cpu_encoded_count, f29 = validated
  local_gpu=$(tail -1 "$csv_file" | cut -d',' -f12)
  local_cpu=$(tail -1 "$csv_file" | cut -d',' -f13)
  local_val=$(tail -1 "$csv_file" | cut -d',' -f29)
  if [[ "$local_gpu" != "$local_cpu" ]]; then
    echo "  ✗ GATE FAILURE: gpu=${local_gpu} cpu_enc=${local_cpu}" >&2
    GATE_FAILURES+=("${label}_gpu${local_gpu}_cpu${local_cpu}")
    continue
  fi
  if [[ "$local_val" != "true" ]]; then
    echo "  ✗ VALIDATION FAILURE: validated=${local_val}" >&2
    GATE_FAILURES+=("${label}_val${local_val}")
    continue
  fi
  # Extract ms_per_iter for pass-through tracking
  local_ms=$(tail -1 "$csv_file" | cut -d',' -f11)
  # Append to combined CSV (with repeat column added)
  if [[ "$HEADER_WRITTEN" -eq 0 ]]; then
    # Add repeat column to header after benchmark (col 46 = warp_strategy+1)
    head -1 "$csv_file" > "$combined_csv"
    HEADER_WRITTEN=1
  fi
  tail -1 "$csv_file" >> "$combined_csv"
  CSV_FILES+=("$csv_file")
  # Track pair data for summary
  # PAIR_DATA stores: predicated_ms|byte_mask_ms
  # predicated is baseline, byte_mask is experimental
  pair_key="${dataset}|${sel}|${block}|${repeat}"
  if [[ "$strategy" == "predicated" ]]; then
    if [[ -v "PAIR_DATA[$pair_key]" ]]; then
      PAIR_DATA["$pair_key"]="${local_ms}|${PAIR_DATA[$pair_key]#*|}"
    else
      PAIR_DATA["$pair_key"]="${local_ms}|"
    fi
  else
    # strategy == byte_mask
    if [[ -v "PAIR_DATA[$pair_key]" ]]; then
      PAIR_DATA["$pair_key"]="${PAIR_DATA[$pair_key]%|}|${local_ms}"
    else
      PAIR_DATA["$pair_key"]="|${local_ms}"
    fi
  fi
  echo "  ✓ gpu=${local_gpu} cpu=${local_cpu} val=${local_val} ms=${local_ms}" >&2
done
echo "" >&2
# ==============================================================================
# Write summary_pairs.csv
# ==============================================================================
{
  echo "dataset,selectivity,block,repeat,predicated_ms,byte_mask_ms,speedup"
  # Sort keys for deterministic output
  for key in $(printf '%s\n' "${!PAIR_DATA[@]}" | sort); do
    IFS='|' read -r dataset sel block repeat <<< "$key"
    IFS='|' read -r predicated_ms byte_mask_ms <<< "${PAIR_DATA[$key]}"
    # Skip incomplete pairs
    if [[ -z "$predicated_ms" || -z "$byte_mask_ms" ]]; then
      continue
    fi
    # Compute speedup = predicated / byte_mask
    speedup=$(python3 -c "
p = float('${predicated_ms}')
q = float('${byte_mask_ms}')
if q > 0:
    print(f'{p/q:.4f}')
else:
    print('inf')
")
    echo "${dataset},${sel},${block},${repeat},${predicated_ms},${byte_mask_ms},${speedup}"
  done
} > "$pairs_csv"
echo "Summary pairs: ${pairs_csv}" >&2
# ==============================================================================
# Results summary
# ==============================================================================
PASSED_COUNT=${#CSV_FILES[@]}
FAILED_COUNT=${#GATE_FAILURES[@]}
echo "=== Results ===" >&2
echo "Total: ${TOTAL_RUNS}  Passed: ${PASSED_COUNT}  Failed: ${FAILED_COUNT}" >&2
if [[ "$HEADER_WRITTEN" -eq 1 ]]; then
  row_count=$(tail -n +2 "$combined_csv" | wc -l)
  echo "Combined CSV: ${combined_csv} (${row_count} rows)" >&2
fi
# Write failure details
if [[ ${#GATE_FAILURES[@]} -gt 0 ]]; then
  {
    echo "Gate failures (${FAILED_COUNT}):"
    for f in "${GATE_FAILURES[@]}"; do
      echo "  - $f"
    done
  } > "$failures_file"
  echo "Failures written to: ${failures_file}" >&2
fi
# ==============================================================================
# run_meta.txt
# ==============================================================================
{
  echo "=== Exp4 Byte-Mask Formal Sweep — run_meta ==="
  echo ""
  echo "Date: $(date -Is)"
  echo "Host: $(hostname)"
  echo "GPU: ${GPU_NAME}"
  echo "Job ID: ${job_id}"
  echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "Git status:"
  git status --short 2>/dev/null || echo "  (not a git repo)"
  echo ""
  echo "=== Toolchain ==="
  echo "nvcc version: $($(which nvcc) --version 2>&1 | grep release || echo unknown)"
  echo "gcc version: $(gcc --version 2>&1 | head -1)"
  echo "cmake version: $(cmake --version 2>&1 | head -1)"
  echo ""
  echo "=== Matrix ==="
  echo "Datasets: heavy_tailed sensor uniform zipfian"
  echo "Selectivities: 1 10 50 90 99 (%)"
  echo "Block sizes: 128 256 512"
  echo "Strategies: predicated (baseline) byte_mask (experimental)"
  echo "Repeats: 5"
  echo "Total runs: ${TOTAL_RUNS}"
  echo "Artifact root: ${V2_ROOT}"
  echo "Raw root: ${RAW_ROOT}"
  echo ""
  echo "=== Dataset mapping ==="
  echo "  heavy_tailed → heavy_tailed_p6 (max_planes=8)"
  echo "  sensor       → sensor_p10       (max_planes=5)"
  echo "  uniform      → uniform_p10      (max_planes=6)"
  echo "  zipfian      → zipfian_p8       (max_planes=8)"
  echo ""
  echo "=== Threshold derivation ==="
  echo "  thresholds = np.quantile(raw_data, 1 - selectivity/100)"
  echo "  raw files: ${RAW_ROOT}/{dataset}.f64le.bin"
  echo ""
  echo "=== Benchmark arguments ==="
  echo "  --warmup 10 --iters 200 --validate"
  echo ""
  echo "=== Results ==="
  echo "Total runs: ${TOTAL_RUNS}"
  echo "Passed:     ${PASSED_COUNT}"
  echo "Failed:     ${FAILED_COUNT}"
  echo "Output dir: ${run_dir}"
  echo "Combined CSV: ${combined_csv}"
  echo "Summary pairs CSV: ${pairs_csv}"
  echo "GPU util log: ${gpu_util_log}"
} > "$meta_file"
echo "Meta: ${meta_file}" >&2
# ==============================================================================
# Exit with nonzero if any gate failures
# ==============================================================================
if [[ ${#GATE_FAILURES[@]} -gt 0 ]]; then
  echo "⚠ ${FAILED_COUNT} gate failure(s) — exiting nonzero" >&2
  exit 1
fi
echo "✓ All gates passed — ${PASSED_COUNT}/${TOTAL_RUNS} validated" >&2
exit 0
