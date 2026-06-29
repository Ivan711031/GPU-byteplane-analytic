#!/usr/bin/env bash
# run_exp4_ncu_byte_mask_ab.sbatch
#
# Exp4 NCU A/B profiling: byte_mask vs predicated warp strategy comparison.
#
# For each case in a compact matrix (uniform control, heavy_tailed hard,
# zipfian skew, heavy_tailed k=1), this script:
#   1. Runs a correctness baseline (--validate, warmup=0 iters=20)
#      for both predicated and byte_mask.
#   2. Captures NCU --set basic for both strategies.
#   3. Writes per-case CSV/log/NCU report and a summary.csv.
#
# Usage:
#   sbatch --account=<ACCOUNT> scripts/run_exp4_ncu_byte_mask_ab.sbatch
#
# Output:
#   results/exp4/ncu_byte_mask_ab_<timestamp>_job<JOBID>_H200/
#     summary.csv                — combined results matrix
#     case_<ID>_<strategy>/      — per-case dir with CSV, .ncu-rep, logs
#     gpu_util.csv               — GPU utilization log
#     run_meta.txt               — metadata
#   handoff/job_done/job_<SLURM_JOB_ID>.json
#SBATCH --job-name=exp4-ncu-byte-mask-ab
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
set -euo pipefail
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
# Constants and paths
# ==============================================================================
V2_ROOT="${WORK_DIR}/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
RAW_ROOT="${WORK_DIR}/datasets/synthetic/dev"
declare -A DATASET_P
DATASET_P[uniform]="p10"
DATASET_P[heavy_tailed]="p6"
DATASET_P[zipfian]="p8"
timestamp=$(date +%Y%m%d_%H%M%S)
job_id="${SLURM_JOB_ID:-local}"
GPU_TAG="unknown"
handoff_dir="${ROOT_DIR}/handoff/job_done"
handoff_marker="${handoff_dir}/job_${job_id}.json"
# ==============================================================================
# Threshold computation cache (numpy quantile on raw FP64)
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
# Case definitions
# ==============================================================================
# Columns: case_id,dataset,selectivity,k_label,max_filter_planes
#   k_label="k1" when max_filter_planes=1; "full" when unset
#   max_filter_planes = integer or empty string for full depth
#   strategies: predicated, byte_mask (both always run)
declare -a CASE_IDS=()
declare -a CASE_DATASETS=()
declare -a CASE_SELS=()
declare -a CASE_KLABELS=()
declare -a CASE_MFPS=()
register_case() {
  CASE_IDS+=("$1")
  CASE_DATASETS+=("$2")
  CASE_SELS+=("$3")
  CASE_KLABELS+=("$4")
  CASE_MFPS+=("$5")
}
# A. Uniform control (tie baseline, full depth)
register_case "A" "uniform"      50  "full" ""
# B. Heavy-tailed hard (large win case, full depth)
register_case "B" "heavy_tailed" 99  "full" ""
# C. Zipfian skew (large win case, full depth)
register_case "C" "zipfian"      99  "full" ""
# D. Heavy-tailed hard, k=1 (capped depth for additional insight)
register_case "D" "heavy_tailed" 99  "k1"   "1"
STRATEGIES=("predicated" "byte_mask")
TOTAL_CASES=${#CASE_IDS[@]}
# ==============================================================================
# Handoff marker via trap (set before GPU validation so exit 2 also writes)
# ==============================================================================
write_handoff_marker() {
  local status=$?
  local result_dir_str=""
  local summary_csv_path=""
  local gpulog=""
  if [[ -d "$run_dir" ]]; then
    result_dir_str="$run_dir"
    summary_csv_path="${run_dir}/summary.csv"
  fi
  if [[ -n "${TRACKER_PID:-}" ]]; then
    kill "$TRACKER_PID" 2>/dev/null || true
    wait "$TRACKER_PID" 2>/dev/null || true
  fi
  if [[ -f "${run_dir}/gpu_util.csv" ]]; then
    gpulog="${run_dir}/gpu_util.csv"
  fi
  local git_commit
  git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
  local slurm_out=""
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    local candidate="${SLURM_SUBMIT_DIR}/slurm-${SLURM_JOB_ID}.out"
    if [[ -f "$candidate" ]]; then
      slurm_out="$candidate"
    fi
  fi
  local next_action="Review summary.csv for per-case NCU status between byte_mask and predicated."
  if [[ "$status" -ne 0 ]]; then
    next_action="Job failed with exit code ${status}; investigate slurm output."
  fi
  mkdir -p "$handoff_dir"
  cat > "$handoff_marker" <<EOF
{
  "job_id": "${job_id}",
  "job_name": "exp4-ncu-byte-mask-ab",
  "exit_status": ${status},
  "finished_at": "$(date -Is)",
  "workdir": "$(pwd)",
  "git_commit": "${git_commit}",
  "result_dir": "${result_dir_str}",
  "summary_csv": "${summary_csv_path}",
  "slurm_output": "${slurm_out}",
  "gpu_util_log": "${gpulog}",
  "next_action": "${next_action}"
}
EOF
}
trap write_handoff_marker EXIT
# ==============================================================================
# GPU identification
# ==============================================================================
GPU_NAME="unknown"
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ' 2>/dev/null || echo "unknown")
  GPU_TAG="${GPU_NAME// /_}"
  GPU_TAG="${GPU_TAG:-unknown}"
fi
run_dir="${ROOT_DIR}/results/exp4/ncu_byte_mask_ab_${timestamp}_job${job_id}_${GPU_TAG}"
# ==============================================================================
# Hardware validation: must be H200 (after trap so exit 2 still writes marker)
# ==============================================================================
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi
mkdir -p "$run_dir" "$handoff_dir"
# ==============================================================================
# GPU utilization tracker (background, best-effort)
# ==============================================================================
gpu_util_log="${run_dir}/gpu_util.csv"
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
# Environment & build
# ==============================================================================
module purge || true
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan
echo "=== Toolchain ===" >&2
echo "nvcc: $(which nvcc) $($(which nvcc) --version 2>&1 | grep release)" >&2
echo "gcc: $(gcc --version 2>&1 | head -1)" >&2
echo "cmake: $(cmake --version 2>&1 | head -1)" >&2
NCU_BIN=""
if command -v ncu &>/dev/null; then
  NCU_BIN=$(which ncu)
elif [[ -x ${ENVSTACK_ROOT}/cuda/12.6/bin/ncu ]]; then
  NCU_BIN=${ENVSTACK_ROOT}/cuda/12.6/bin/ncu
fi
if [[ -n "$NCU_BIN" ]]; then
  echo "ncu: $NCU_BIN $($NCU_BIN --version 2>&1 | head -1)" >&2
else
  echo "WARNING: ncu not found" >&2
fi
echo "=== Job info ===" >&2
echo "Job ID: ${job_id}  Host: $(hostname)  GPU: ${GPU_NAME}" >&2
echo "Output: ${run_dir}" >&2
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader >&2
git rev-parse HEAD >&2
git status --short >&2
echo "" >&2
echo "=== Build ===" >&2
cmake -S benchmarks/experiment4 -B build/exp4 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 2>&1
cmake --build build/exp4 -j 2>&1
echo "" >&2
BENCH_BIN="${ROOT_DIR}/build/exp4/bench_progressive_filter"
# ==============================================================================
# Summary CSV header
# ==============================================================================
summary_csv="${run_dir}/summary.csv"
echo "case_id,dataset,selectivity,k,strategy,validated,gpu_count,cpu_encoded_count,ms_per_iter,ncu_status,ncu_report" > "$summary_csv"
# ==============================================================================
# Run matrix
# ==============================================================================
total_ok=0
total_fail=0
SUBFAILURES=()
for ((idx=0; idx<TOTAL_CASES; idx++)); do
  case_id="${CASE_IDS[$idx]}"
  dataset="${CASE_DATASETS[$idx]}"
  sel="${CASE_SELS[$idx]}"
  k_label="${CASE_KLABELS[$idx]}"
  mfp="${CASE_MFPS[$idx]}"
  artifact_p="${DATASET_P[$dataset]}"
  encoded_subdir="${dataset}_${artifact_p}"
  encoded_root="${V2_ROOT}/${encoded_subdir}"
  threshold=$(compute_threshold "$dataset" "$sel")
  for strategy in "${STRATEGIES[@]}"; do
    subcase_id="${case_id}_${strategy}"
    case_dir="${run_dir}/case_${subcase_id}"
    mkdir -p "$case_dir"
    echo "" >&2
    echo "============================================================" >&2
    echo "[${subcase_id}] dataset=${dataset} sel=${sel} k=${k_label} strategy=${strategy}" >&2
    echo "============================================================" >&2
    echo "  threshold=${threshold}" >&2
    # ==================================================================
    # Baseline validation
    # ==================================================================
    baseline_csv="${case_dir}/baseline.csv"
    baseline_log="${case_dir}/baseline.log"
    set +e
    if [[ -n "$mfp" ]]; then
      "$BENCH_BIN" \
        --device 0 \
        --encoded-root "$encoded_root" \
        --threshold "$threshold" \
        --block 256 \
        --validate \
        --warp-strategy "$strategy" \
        --warmup 0 \
        --iters 20 \
        --max-filter-planes "$mfp" \
        --csv "$baseline_csv" \
        2> "$baseline_log"
    else
      "$BENCH_BIN" \
        --device 0 \
        --encoded-root "$encoded_root" \
        --threshold "$threshold" \
        --block 256 \
        --validate \
        --warp-strategy "$strategy" \
        --warmup 0 \
        --iters 20 \
        --csv "$baseline_csv" \
        2> "$baseline_log"
    fi
    baseline_rc=$?
    set -e
    # Parse baseline CSV
    validated="false"
    gpu_count="NA"
    cpu_encoded="NA"
    ms_per_iter="NA"
    if [[ -f "$baseline_csv" ]]; then
      gpu_count=$(tail -1 "$baseline_csv" | cut -d',' -f12)
      cpu_encoded=$(tail -1 "$baseline_csv" | cut -d',' -f13)
      validated_col=$(tail -1 "$baseline_csv" | cut -d',' -f29)
      ms_per_iter=$(tail -1 "$baseline_csv" | cut -d',' -f11)
      if [[ "$gpu_count" == "$cpu_encoded" ]] && [[ "$validated_col" == "true" ]]; then
        validated="true"
        echo "  ✓ Baseline: gpu=${gpu_count} cpu=${cpu_encoded} validated" >&2
      else
        echo "  ✗ Baseline FAIL: gpu=${gpu_count} cpu=${cpu_encoded} validated=${validated_col}" >&2
      fi
    else
      echo "  ✗ Baseline CSV not found (rc=${baseline_rc})" >&2
    fi
    # ==================================================================
    # NCU profile
    # ==================================================================
    ncu_report_path=""
    ncu_status="not_run"
    ncu_export_base="${case_dir}/ncu_profile"
    ncu_report="${ncu_export_base}.ncu-rep"
    if [[ -z "$NCU_BIN" ]]; then
      echo "  NCU not available" >&2
      ncu_status="FAIL: no NCU binary"
    else
      ncu_stdout="${case_dir}/ncu_stdout.txt"
      ncu_stderr="${case_dir}/ncu_stderr.txt"
      set +e
      if [[ -n "$mfp" ]]; then
        "$NCU_BIN" \
          --set basic \
          --target-processes all \
          --profile-from-start yes \
          --page raw \
          -f \
          -o "$ncu_export_base" \
          "$BENCH_BIN" \
            --device 0 \
            --encoded-root "$encoded_root" \
            --threshold "$threshold" \
            --block 256 \
            --validate \
            --warp-strategy "$strategy" \
            --warmup 0 \
            --iters 20 \
            --max-filter-planes "$mfp" \
            --csv "${case_dir}/ncu_bench.csv" \
          > "$ncu_stdout" 2> "$ncu_stderr"
      else
        "$NCU_BIN" \
          --set basic \
          --target-processes all \
          --profile-from-start yes \
          --page raw \
          -f \
          -o "$ncu_export_base" \
          "$BENCH_BIN" \
            --device 0 \
            --encoded-root "$encoded_root" \
            --threshold "$threshold" \
            --block 256 \
            --validate \
            --warp-strategy "$strategy" \
            --warmup 0 \
            --iters 20 \
            --csv "${case_dir}/ncu_bench.csv" \
          > "$ncu_stdout" 2> "$ncu_stderr"
      fi
      ncu_rc=$?
      set -e
      echo "  NCU exit code: ${ncu_rc}" >&2
      # Detect success
      if grep -q "No kernels were profiled" "$ncu_stdout" 2>/dev/null; then
        ncu_status="FAIL: No kernels were profiled"
        echo "  ✗ ${ncu_status}" >&2
      elif [[ -f "$ncu_report" ]] && [[ -s "$ncu_report" ]]; then
        ncu_report_size=$(stat --printf='%s' "$ncu_report")
        echo "  ✓ .ncu-rep: ${ncu_report_size} bytes" >&2
        ncu_status="PASS"
        ncu_report_path="$ncu_report"
        # Best-effort CSV export from report
        "$NCU_BIN" --csv --page raw \
          --force-overwrite true \
          -i "$ncu_report" \
          -o "${ncu_export_base}" \
          >> "$ncu_stdout" 2>> "$ncu_stderr" || true
      else
        ncu_status="FAIL: report missing or empty"
        echo "  ✗ ${ncu_status}" >&2
      fi
    fi
    # Write per-case status
    echo "$ncu_status" > "${case_dir}/ncu_status.txt"
    # Track failures
    if [[ "$ncu_status" == "PASS" ]]; then
      total_ok=$((total_ok + 1))
    else
      total_fail=$((total_fail + 1))
      SUBFAILURES+=("${subcase_id}: ${ncu_status}")
    fi
    # ==================================================================
    # Append to summary CSV
    # ==================================================================
    # CSV-quoting: all fields are simple (no commas in values)
    echo "${subcase_id},${dataset},${sel},${k_label},${strategy},${validated},${gpu_count},${cpu_encoded},${ms_per_iter},${ncu_status},${ncu_report_path}" >> "$summary_csv"
  done
done
# ==============================================================================
# run_meta.txt
# ==============================================================================
{
  echo "=== Exp4 NCU Byte-Mask A/B — run_meta ==="
  echo ""
  echo "Date: $(date -Is)"
  echo "Host: $(hostname)"
  echo "GPU: ${GPU_NAME}"
  echo "Job ID: ${job_id}"
  echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo ""
  echo "=== Toolchain ==="
  echo "nvcc: $($(which nvcc) --version 2>&1 | grep release || echo unknown)"
  echo "gcc: $(gcc --version 2>&1 | head -1)"
  echo "cmake: $(cmake --version 2>&1 | head -1)"
  echo "ncu: $($NCU_BIN --version 2>&1 | head -1 || echo unknown)"
  echo ""
  echo "=== NCU command shape ==="
  echo "  --set basic --target-processes all --profile-from-start yes"
  echo "  --page raw -f"
  echo "  NO launch-count, NO launch-skip, NO replay-mode"
  echo "  NO kernel-name filter"
  echo ""
  echo "=== Case matrix ==="
  echo "Strategies: predicated byte_mask"
  for ((idx=0; idx<TOTAL_CASES; idx++)); do
    dataset="${CASE_DATASETS[$idx]}"
    sel="${CASE_SELS[$idx]}"
    k_label="${CASE_KLABELS[$idx]}"
    echo "  ${CASE_IDS[$idx]}: ${dataset} sel=${sel} k=${k_label}"
  done
  echo ""
  echo "=== Results ==="
  echo "  PASS: ${total_ok}   FAIL: ${total_fail}"
  if [[ ${#SUBFAILURES[@]} -gt 0 ]]; then
    echo "  Failures:"
    for f in "${SUBFAILURES[@]}"; do
      echo "    - $f"
    done
  fi
  echo ""
  echo "Summary CSV: ${summary_csv}"
  echo "Output dir: ${run_dir}"
  echo ""
  echo "=== Purpose ==="
  echo "  Compare NCU --set basic metrics between byte_mask and predicated warp"
  echo "  strategies across a compact dataset/selectivity matrix. Runs correctness"
  echo "  baseline (--validate) before each NCU capture. Modest iters (20) for"
  echo "  profiling suitability."
} > "${run_dir}/run_meta.txt"
echo "" >&2
echo "=== Exp4 NCU Byte-Mask A/B Complete ===" >&2
echo "PASS: ${total_ok}  FAIL: ${total_fail}" >&2
echo "Summary: ${summary_csv}" >&2
if [[ ${#SUBFAILURES[@]} -gt 0 ]]; then
  echo "Failures:" >&2
  for f in "${SUBFAILURES[@]}"; do
    echo "  ${f}" >&2
  done
fi
# Exit nonzero only if ALL cases failed
if [[ $total_ok -eq 0 ]] && [[ $total_fail -gt 0 ]]; then
  echo "All cases failed — exiting nonzero" >&2
  exit 1
fi
echo "✓ Done" >&2
exit 0
