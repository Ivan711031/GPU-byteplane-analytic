#!/usr/bin/env bash
# run_prompt5_h200_kdepth_ncu_viability.sbatch
#
# Prompt5: H200 k-depth byte-plane physical traffic vs raw FP64.
# Profiles k=1,2,3,4,max for fused filter+aggregate (plus raw FP64 baseline).
#
# Cases:
#   E: uniform_p10, s≈50%, k=2,3,4 (k=1,6 already in job 53622), raw FP64
#   F: heavy_tailed_p6, s≈50%, k=1,2,3,4,max (k=8 already in job 53622), raw FP64
#   G: heavy_tailed_p6, s≈99%, k=1,2,3,4,max (k=8 already in job 53622), raw FP64
#
# Kernels:
#   fused  = progressive_filter_sum_rowpack16_byte_mask
#   raw_fp64 = raw_fp64_filter_sum
#
# Output:
#   results/prompt5_h200_viability/ncu_kdepth_<ts>_job<JID>_NVIDIAH200/
#SBATCH --job-name=prompt5-kdepth
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=results/prompt5_h200_viability/ncu_kdepth_%j.out
#SBATCH --error=results/prompt5_h200_viability/ncu_kdepth_%j.err
set -euo pipefail
# ==============================================================================
# Dynamic workspace detection & result directory
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
RUN_TS=$(date +%Y%m%d_%H%M%S)
JOB_ID="${SLURM_JOB_ID:-local}"
GPU_NAME="unknown"
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ' 2>/dev/null || echo "unknown")
fi
GPU_TAG="${GPU_NAME// /_}"
RESULT_DIR="results/prompt5_h200_viability/ncu_kdepth_${RUN_TS}_job${JOB_ID}_${GPU_TAG}"
mkdir -p "$RESULT_DIR"
STDOUT_PATH="results/prompt5_h200_viability/ncu_kdepth_${JOB_ID}.out"
STDERR_PATH="results/prompt5_h200_viability/ncu_kdepth_${JOB_ID}.err"
GPU_UTIL_LOG="${RESULT_DIR}/gpu_util.log"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv -l 2 > "$GPU_UTIL_LOG" 2>&1 &
  GPU_UTIL_PID=$!
else
  GPU_UTIL_PID=""
fi
# ==============================================================================
# Handoff marker via trap (registered early for fail-fast coverage)
# ==============================================================================
HANDOFF_DIR="${ROOT_DIR}/handoff/job_done"
HANDOFF_MARKER="${HANDOFF_DIR}/job_${JOB_ID}.json"
write_handoff_marker() {
  local status=$?
  if [[ -n "${GPU_UTIL_PID:-}" ]]; then
    kill "$GPU_UTIL_PID" 2>/dev/null || true
    wait "$GPU_UTIL_PID" 2>/dev/null || true
  fi
  mkdir -p "$HANDOFF_DIR"
  local git_commit
  git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
  local next_action="Extract Prompt5 NCU metrics, merge heavy-tailed evidence into the break-even verdict, and update issue #79."
  if [[ "$status" -ne 0 ]]; then
    next_action="Job failed with exit code ${status}; inspect slurm output and per-case ncu_stderr.txt."
  fi
  cat > "$HANDOFF_MARKER" <<HEREDOC
{
  "exit_status": $status,
  "job_id": "${JOB_ID}",
  "job_name": "prompt5-kdepth",
  "finished_at": "$(date -Iseconds)",
  "workdir": "$ROOT_DIR",
  "git_commit": "${git_commit}",
  "stdout": "$STDOUT_PATH",
  "stderr": "$STDERR_PATH",
  "gpu_util_log": "$GPU_UTIL_LOG",
  "result_dir": "$RESULT_DIR",
  "result_csv": "${RESULT_DIR}/summary.csv",
  "next_action": "${next_action}"
}
HEREDOC
  echo "Handoff marker: ${HANDOFF_MARKER}"
}
trap write_handoff_marker EXIT
# ==============================================================================
# H200 fail-fast
# ==============================================================================
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi
echo "GPU: ${GPU_NAME}"
# ==============================================================================
# Environment & build
# ==============================================================================
module purge
module load miniconda3/26.1.1 2>/dev/null || module load miniconda3
module load cuda/12.6 2>/dev/null || module load cuda
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan 2>/dev/null || conda activate gpu-byteplane-rapids 2>/dev/null || true
echo "=== Toolchain ===" >&2
echo "nvcc: $(which nvcc) $($(which nvcc) --version 2>&1 | grep release || echo unknown)" >&2
echo "gcc: $(gcc --version 2>&1 | head -1)" >&2
echo "cmake: $(cmake --version 2>&1 | head -1)" >&2
NCU_BIN=""
if command -v ncu &>/dev/null; then
  NCU_BIN="$(command -v ncu)"
elif [[ -x ${ENVSTACK_ROOT}/cuda/12.6/bin/ncu ]]; then
  NCU_BIN="${ENVSTACK_ROOT}/cuda/12.6/bin/ncu"
fi
if [[ -n "$NCU_BIN" ]]; then
  echo "ncu: ${NCU_BIN} $($NCU_BIN --version 2>&1 | head -1)" >&2
else
  echo "ERROR: ncu not found after loading CUDA module" >&2
  exit 3
fi
# ==============================================================================
# Case set selection (smoke=default, full=all cases)
# ==============================================================================
PROMPT5_CASE_SET="${PROMPT5_CASE_SET:-smoke}"
echo "PROMPT5_CASE_SET=${PROMPT5_CASE_SET}" >&2
# Validate
if [[ "$PROMPT5_CASE_SET" != "smoke" && "$PROMPT5_CASE_SET" != "remaining" && "$PROMPT5_CASE_SET" != "full" ]]; then
  echo "ERROR: PROMPT5_CASE_SET must be 'smoke', 'remaining', or 'full', got: ${PROMPT5_CASE_SET}" >&2
  exit 2
fi
echo "=== Job info ===" >&2
echo "Job ID: ${JOB_ID}  Host: $(hostname)  GPU: ${GPU_NAME}" >&2
echo "Case set: ${PROMPT5_CASE_SET}" >&2
echo "Output: ${RESULT_DIR}" >&2
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader >&2
git rev-parse HEAD >&2
# ==============================================================================
# Build
# ==============================================================================
echo "" >&2
echo "=== Build ===" >&2
BUILD_DIR="build_filter_aggregate"
mkdir -p "$BUILD_DIR"
cmake -S benchmarks/experiment4_filter_aggregate -B "$BUILD_DIR" \
  -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
  -DCMAKE_BUILD_TYPE=Release
BUILD_JOBS="${SLURM_CPUS_ON_NODE:-4}"
cmake --build "$BUILD_DIR" -j"$BUILD_JOBS"
echo "" >&2
BENCH_BIN="${ROOT_DIR}/${BUILD_DIR}/bench_filter_aggregate"
if [[ ! -x "$BENCH_BIN" ]]; then
  echo "ERROR: bench_filter_aggregate not found at ${BENCH_BIN}" >&2
  exit 2
fi
# ==============================================================================
# Data paths
# ==============================================================================
V2_ROOT="/work/${USER}/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
RAW_ROOT="/work/${USER}/datasets/synthetic/dev"
# ==============================================================================
# Case definitions (controlled by PROMPT5_CASE_SET)
# ==============================================================================
declare -a CASE_IDS=()
declare -a CASE_ARTIFACTS=()
declare -a CASE_SEL_LABELS=()
declare -a CASE_K_VALUES=()
declare -a CASE_THRESHOLDS=()
register_case() {
  CASE_IDS+=("$1")
  CASE_ARTIFACTS+=("$2")
  CASE_SEL_LABELS+=("$3")
  CASE_K_VALUES+=("$4")
  CASE_THRESHOLDS+=("$5")
}
# E: uniform_p10 s50 — k=1,6 already covered by job 53622 cases A/B.
#     Smoke mode has already collected k=2/3/4. Full mode reruns it.
if [[ "$PROMPT5_CASE_SET" == "smoke" ]]; then
  register_case "E" "uniform_p10"      "s50" "2 3 4" "500.0271759315981"
fi
if [[ "$PROMPT5_CASE_SET" == "full" ]]; then
  register_case "E" "uniform_p10"      "s50" "2 3 4" "500.0271759315981"
fi
# F: heavy_tailed_p6 s50 — only k=8 covered by job 53622 case C
# G: heavy_tailed_p6 s99 — only k=8 covered by job 53622 case D
if [[ "$PROMPT5_CASE_SET" == "remaining" || "$PROMPT5_CASE_SET" == "full" ]]; then
  register_case "F" "heavy_tailed_p6"  "s50" "1 2 3 4 6" "1.0001588729752771"
  register_case "G" "heavy_tailed_p6"  "s99" "1 2 3 4 6" "8.9028979745652526e-06"
fi
TOTAL_CASES=${#CASE_IDS[@]}
# Kernel targets
declare -a KERNEL_TARGETS=("fused" "raw_fp64")
declare -A KERNEL_REGEX
KERNEL_REGEX[fused]="progressive_filter_sum_rowpack16_byte_mask"
KERNEL_REGEX[raw_fp64]="raw_fp64_filter_sum"
# ==============================================================================
# Summary CSV
# ==============================================================================
SUMMARY_CSV="${RESULT_DIR}/summary.csv"
echo "case_id,kernel_target,dataset_artifact,dataset,selectivity_label,k,threshold,kernel_regex,status,report_path,bench_csv_path" > "$SUMMARY_CSV"
append_summary_row() {
  echo "$1" >> "$SUMMARY_CSV"
}
# ==============================================================================
# Run matrix
# ==============================================================================
TOTAL_OK=0
TOTAL_FAIL=0
FAILURES=()
for ((ci=0; ci<TOTAL_CASES; ci++)); do
  case_id="${CASE_IDS[$ci]}"
  artifact="${CASE_ARTIFACTS[$ci]}"
  sel_label="${CASE_SEL_LABELS[$ci]}"
  k_list="${CASE_K_VALUES[$ci]}"
  threshold="${CASE_THRESHOLDS[$ci]}"
  dataset="${artifact%_p*}"
  artifact_root="${V2_ROOT}/${artifact}"
  raw_path="${RAW_ROOT}/${dataset}.f64le.bin"
  if [[ ! -d "$artifact_root" ]]; then
    echo "ERROR: artifact root not found: $artifact_root" >&2
    exit 2
  fi
  if [[ ! -f "$raw_path" ]]; then
    echo "ERROR: raw file not found: $raw_path" >&2
    exit 2
  fi
  for kernel_target in "${KERNEL_TARGETS[@]}"; do
    kernel_regex="${KERNEL_REGEX[$kernel_target]}"
    # For raw_fp64, k does not apply. Run once per case.
    if [[ "$kernel_target" == "raw_fp64" ]]; then
      case_dir="${RESULT_DIR}/case_${case_id}_raw_fp64"
      if [[ -d "$case_dir" ]]; then
        echo "[${case_id}/raw_fp64] already captured, skipping" >&2
        continue
      fi
      mkdir -p "$case_dir"
      k_display="raw"
      k_flag="1"
      echo "" >&2
      echo "============================================================" >&2
      echo "[${case_id}/raw_fp64] dataset=${dataset} sel=${sel_label}" >&2
      echo "  kernel_regex=${kernel_regex}" >&2
      echo "============================================================" >&2
      # --- Baseline validation ---
      bench_csv="${case_dir}/bench.csv"
      bench_log="${case_dir}/bench.log"
      set +e
      "$BENCH_BIN" \
        --dataset "$artifact_root" \
        --raw "$raw_path" \
        --threshold "$threshold" \
        --max-filter-planes 1 \
        --iters 1 \
        --warmup 0 \
        --validate \
        --block-threads 256 \
        --csv "$bench_csv" \
        > "$bench_log" 2>&1
      bench_rc=$?
      set -e
      if [[ $bench_rc -ne 0 ]]; then
        echo "  ✗ Benchmark failed (rc=${bench_rc})" >&2
        FAILURES+=("${case_id}_raw_fp64: bench rc=${bench_rc}")
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        append_summary_row "${case_id},raw_fp64,${artifact},${dataset},${sel_label},raw,${threshold},${kernel_regex},FAIL_BENCH,,${bench_csv}"
        continue
      fi
      # --- NCU profile ---
      ncu_stdout="${case_dir}/ncu_stdout.txt"
      ncu_stderr="${case_dir}/ncu_stderr.txt"
      ncu_base="${case_dir}/ncu_profile"
      ncu_report="${ncu_base}.ncu-rep"
      set +e
      "$NCU_BIN" \
        --set full \
        --target-processes all \
        --profile-from-start yes \
        --kernel-name "regex:${kernel_regex}" \
        --page raw \
        -f \
        -o "$ncu_base" \
        "$BENCH_BIN" \
          --dataset "$artifact_root" \
          --raw "$raw_path" \
          --threshold "$threshold" \
          --max-filter-planes 1 \
          --iters 1 \
          --warmup 0 \
          --validate \
          --block-threads 256 \
          --csv "${case_dir}/ncu_bench.csv" \
        > "$ncu_stdout" 2> "$ncu_stderr"
      ncu_rc=$?
      set -e
      if [[ -f "$ncu_report" ]] && [[ -s "$ncu_report" ]]; then
        ncu_status="PASS"
        TOTAL_OK=$((TOTAL_OK + 1))
      else
        ncu_status="FAIL: report missing (rc=${ncu_rc})"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        FAILURES+=("${case_id}_raw_fp64: report missing")
      fi
      append_summary_row "${case_id},raw_fp64,${artifact},${dataset},${sel_label},raw,${threshold},${kernel_regex},${ncu_status},${ncu_report},${bench_csv}"
      continue
    fi
    # === fused kernel ===
    for k in $k_list; do
      case_dir="${RESULT_DIR}/case_${case_id}_fused_k${k}"
      mkdir -p "$case_dir"
      echo "" >&2
      echo "============================================================" >&2
      echo "[${case_id}/fused/k=${k}] dataset=${dataset} sel=${sel_label}" >&2
      echo "  threshold=${threshold} kernel_regex=${kernel_regex}" >&2
      echo "============================================================" >&2
      # --- Baseline validation ---
      bench_csv="${case_dir}/bench.csv"
      bench_log="${case_dir}/bench.log"
      set +e
      "$BENCH_BIN" \
        --dataset "$artifact_root" \
        --raw "$raw_path" \
        --threshold "$threshold" \
        --max-filter-planes "$k" \
        --iters 1 \
        --warmup 0 \
        --validate \
        --block-threads 256 \
        --csv "$bench_csv" \
        > "$bench_log" 2>&1
      bench_rc=$?
      set -e
      if [[ $bench_rc -ne 0 ]]; then
        echo "  ✗ Benchmark failed (rc=${bench_rc})" >&2
        FAILURES+=("${case_id}_fused_k${k}: bench rc=${bench_rc}")
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        append_summary_row "${case_id},fused,${artifact},${dataset},${sel_label},${k},${threshold},${kernel_regex},FAIL_BENCH,,${bench_csv}"
        continue
      fi
      # --- NCU profile ---
      ncu_stdout="${case_dir}/ncu_stdout.txt"
      ncu_stderr="${case_dir}/ncu_stderr.txt"
      ncu_base="${case_dir}/ncu_profile"
      ncu_report="${ncu_base}.ncu-rep"
      set +e
      "$NCU_BIN" \
        --set full \
        --target-processes all \
        --profile-from-start yes \
        --kernel-name "regex:${kernel_regex}" \
        --page raw \
        -f \
        -o "$ncu_base" \
        "$BENCH_BIN" \
          --dataset "$artifact_root" \
          --raw "$raw_path" \
          --threshold "$threshold" \
          --max-filter-planes "$k" \
          --iters 1 \
          --warmup 0 \
          --validate \
          --block-threads 256 \
          --csv "${case_dir}/ncu_bench.csv" \
        > "$ncu_stdout" 2> "$ncu_stderr"
      ncu_rc=$?
      set -e
      if [[ -f "$ncu_report" ]] && [[ -s "$ncu_report" ]]; then
        ncu_status="PASS"
        TOTAL_OK=$((TOTAL_OK + 1))
      else
        ncu_status="FAIL: report missing (rc=${ncu_rc})"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        FAILURES+=("${case_id}_fused_k${k}: report missing")
      fi
      append_summary_row "${case_id},fused,${artifact},${dataset},${sel_label},${k},${threshold},${kernel_regex},${ncu_status},${ncu_report},${bench_csv}"
    done
  done
done
# ==============================================================================
# run_meta.txt
# ==============================================================================
{
  echo "=== Prompt5 H200 k-depth NCU Viability Audit — run_meta ==="
  echo ""
  echo "Date: $(date -Iseconds)"
  echo "Host: $(hostname)"
  echo "GPU: ${GPU_NAME}"
  echo "Job ID: ${JOB_ID}"
  echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "PROMPT5_CASE_SET: ${PROMPT5_CASE_SET}"
  echo ""
  echo "=== Toolchain ==="
  echo "nvcc: $($(which nvcc) --version 2>&1 | grep release || echo unknown)"
  echo "gcc: $(gcc --version 2>&1 | head -1)"
  echo "cmake: $(cmake --version 2>&1 | head -1)"
  echo "ncu: $($NCU_BIN --version 2>&1 | head -1 || echo unknown)"
  echo ""
  echo "=== Cases ==="
  for ((ci=0; ci<TOTAL_CASES; ci++)); do
    echo "  ${CASE_IDS[$ci]}: ${CASE_ARTIFACTS[$ci]} sel=${CASE_SEL_LABELS[$ci]} k=${CASE_K_VALUES[$ci]} threshold=${CASE_THRESHOLDS[$ci]}"
  done
  echo ""
  echo "=== Kernels ==="
  for kt in "${KERNEL_TARGETS[@]}"; do
    echo "  ${kt}: ${KERNEL_REGEX[$kt]}"
  done
  echo ""
  echo "=== Results ==="
  echo "  PASS: ${TOTAL_OK}   FAIL: ${TOTAL_FAIL}"
  if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "  Failures:"
    for f in "${FAILURES[@]}"; do
      echo "    - $f"
    done
  fi
  echo ""
  echo "Summary CSV: ${SUMMARY_CSV}"
  echo "Output dir: ${RESULT_DIR}"
} > "${RESULT_DIR}/run_meta.txt"
echo "" >&2
echo "=== Prompt5 H200 k-depth NCU Viability Audit Complete ===" >&2
echo "Case set: ${PROMPT5_CASE_SET}" >&2
echo "PASS: ${TOTAL_OK}  FAIL: ${TOTAL_FAIL}" >&2
echo "Summary: ${SUMMARY_CSV}" >&2
if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo "Failures:" >&2
  for f in "${FAILURES[@]}"; do
    echo "  ${f}" >&2
  done
fi
exit 0
