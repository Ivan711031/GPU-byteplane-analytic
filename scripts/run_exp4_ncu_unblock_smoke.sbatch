#!/usr/bin/env bash
# run_exp4_ncu_unblock_smoke.sbatch
#
# Exp4 NCU unblock smoke: verify that Nsight Compute captures kernels from
# bench_progressive_filter with warp_strategy=passive.
#
# Known failure mode: Exp4-B1 NCU jobs with warmup=0 iters=1 produced
#   "==WARNING== No kernels were profiled".
# Hypothesis: single fast launch / command shape missed kernels.
#
# This script:
#   1. Builds benchmarks/experiment4 with -DCMAKE_CUDA_ARCHITECTURES=90.
#   2. Runs a baseline correctness run (non-NCU) with --validate.
#   3. Runs NCU on the same binary with enough launches (warmup=5 iters=20)
#      and NCU flags that reliably capture kernels.
#   4. Detects success via non-empty .ncu-rep and absence of "No kernels".
#
# Usage:
#   sbatch --account=<ACCOUNT> scripts/run_exp4_ncu_unblock_smoke.sbatch
#
# Output:
#   results/exp4/ncu_unblock_<timestamp>_job<JOBID>_H200/
#     baseline.csv              — correctness CSV (non-NCU)
#     ncu_stdout.txt            — ncu stdout
#     ncu_stderr.txt            — ncu stderr
#     ncu_unblock.ncu-rep       — Nsight Compute report
#     ncu_unblock.csv           — NCU CSV export (if available)
#     ncu_status.txt            — PASS/FAIL with reason
#     run_meta.txt              — metadata
#   handoff/job_done/job_<SLURM_JOB_ID>.json
#SBATCH --job-name=exp4-ncu-unblock
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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
# Paths (set up before trap so handoff marker knows where to write)
# ==============================================================================
V2_ROOT="${WORK_DIR}/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
RAW_ROOT="${WORK_DIR}/datasets/synthetic/dev"
dataset_name="uniform"
artifact_p="p10"
encoded_subdir="${dataset_name}_${artifact_p}"
encoded_root="${V2_ROOT}/${encoded_subdir}"
selectivity=50
timestamp=$(date +%Y%m%d_%H%M%S)
job_id="${SLURM_JOB_ID:-local}"
GPU_TAG="unknown"
handoff_dir="${ROOT_DIR}/handoff/job_done"
handoff_marker="${handoff_dir}/job_${job_id}.json"
# Global for NCU successful report path (set after NCU runs, read by trap)
NCU_REPORT_SUCCESS=""
# ==============================================================================
# Handoff marker via trap (set before GPU validation so exit 2 also writes marker)
# ==============================================================================
write_handoff_marker() {
  local status=$?
  local exit_reason=""
  local ncu_report_path=""
  local ncu_status_str="not_run"
  local baseline_csv_path=""
  local result_dir_str=""
  if [[ -d "$run_dir" ]]; then
    result_dir_str="$run_dir"
    baseline_csv_path="${run_dir}/baseline.csv"
    ncu_report_path="${NCU_REPORT_SUCCESS}"
    if [[ -f "${run_dir}/ncu_status.txt" ]]; then
      ncu_status_str=$(head -1 "${run_dir}/ncu_status.txt" 2>/dev/null || echo "unknown")
    fi
  fi
  local git_commit
  git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
  local next_action="Check ncu_status.txt and ncu_stdout.txt for kernel capture verification."
  if [[ "$status" -ne 0 ]]; then
    next_action="Job failed with exit code ${status}; investigate slurm output."
  fi
  mkdir -p "$handoff_dir"
  cat > "$handoff_marker" <<EOF
{
  "job_id": "${job_id}",
  "job_name": "exp4-ncu-unblock",
  "exit_status": ${status},
  "finished_at": "$(date -Is)",
  "workdir": "$(pwd)",
  "git_commit": "${git_commit}",
  "result_dir": "${result_dir_str}",
  "baseline_csv": "${baseline_csv_path}",
  "ncu_report": "${ncu_report_path}",
  "ncu_status": "${ncu_status_str}",
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
# Create result dir name with GPU tag (before fail-fast so trap uses it)
run_dir="${ROOT_DIR}/results/exp4/ncu_unblock_${timestamp}_job${job_id}_${GPU_TAG}"
# ==============================================================================
# Hardware validation: must be H200 (after trap so exit 2 still writes marker)
# ==============================================================================
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi
# ==============================================================================
# Create output directories
# ==============================================================================
mkdir -p "$run_dir" "$handoff_dir"
# ==============================================================================
# Environment & modules
# ==============================================================================
module purge || true
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan
# ==============================================================================
# Toolchain verification
# ==============================================================================
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
  echo "WARNING: ncu not found in PATH or known locations" >&2
fi
# ==============================================================================
# Job metadata
# ==============================================================================
echo "=== Exp4 NCU Unblock Smoke ===" >&2
echo "Job ID: ${job_id}" >&2
echo "Host: $(hostname)" >&2
echo "Time: $(date -Is)" >&2
echo "GPU: ${GPU_NAME}" >&2
echo "Dataset: ${dataset_name} (artifact: ${encoded_subdir})" >&2
echo "Selectivity: ${selectivity}" >&2
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
BENCH_BIN="${ROOT_DIR}/build/exp4/bench_progressive_filter"
# ==============================================================================
# Verify artifact path
# ==============================================================================
echo "=== Checking artifact path ===" >&2
if [[ ! -d "$encoded_root" ]]; then
  echo "ERROR: encoded root not found: $encoded_root" >&2
  exit 2
fi
echo "  ✓ $encoded_root" >&2
echo "" >&2
# ==============================================================================
# Compute threshold from raw FP64
# ==============================================================================
echo "=== Computing threshold (selectivity=${selectivity}) ===" >&2
raw_path="${RAW_ROOT}/${dataset_name}.f64le.bin"
if [[ ! -f "$raw_path" ]]; then
  echo "ERROR: raw file not found: $raw_path" >&2
  exit 2
fi
quantile=$(python3 -c "print(1 - ${selectivity}/100.0)")
threshold=$(python3 -c "
import numpy as np
data = np.fromfile('${raw_path}', dtype=np.float64)
print(np.quantile(data, ${quantile}))
")
echo "  threshold=${threshold} (quantile=${quantile})" >&2
echo "" >&2
# ==============================================================================
# RUN 1: Baseline correctness (non-NCU) with --validate
# ==============================================================================
echo "=== Run 1: Baseline correctness (non-NCU) ===" >&2
baseline_csv="${run_dir}/baseline.csv"
baseline_log="${run_dir}/baseline.log"
set +e
"$BENCH_BIN" \
  --device 0 \
  --encoded-root "$encoded_root" \
  --threshold "$threshold" \
  --block 256 \
  --validate \
  --warp-strategy passive \
  --warmup 5 \
  --iters 20 \
  --csv "$baseline_csv" \
  2> "$baseline_log"
baseline_rc=$?
set -e
echo "  Baseline exit code: ${baseline_rc}" >&2
if [[ $baseline_rc -ne 0 ]]; then
  echo "  WARNING: baseline benchmark failed (exit=${baseline_rc})" >&2
  echo "  See: ${baseline_log}" >&2
fi
# Validate correctness gate from baseline CSV
if [[ -f "$baseline_csv" ]]; then
  gpu_count=$(tail -1 "$baseline_csv" | cut -d',' -f12)
  cpu_encoded=$(tail -1 "$baseline_csv" | cut -d',' -f13)
  echo "  gpu_count=${gpu_count}  cpu_encoded_count=${cpu_encoded}" >&2
  if [[ "$gpu_count" == "$cpu_encoded" ]]; then
    echo "  ✓ Baseline validation PASS (gpu_count == cpu_encoded_count)" >&2
  else
    echo "  ✗ Baseline validation FAIL (gpu_count != cpu_encoded_count)" >&2
  fi
else
  echo "  WARNING: baseline CSV not created at ${baseline_csv}" >&2
fi
echo "" >&2
# ==============================================================================
# RUN 2: NCU profile with sufficient launches
# ==============================================================================
echo "=== Run 2: NCU profile ===" >&2
ncu_stdout="${run_dir}/ncu_stdout.txt"
ncu_stderr="${run_dir}/ncu_stderr.txt"
: > "$ncu_stdout"
: > "$ncu_stderr"
ncu_passed=0
ncu_fail_reason=""
# --- Shared benchmark arguments for NCU ---
ncu_bench_args=(
  "$BENCH_BIN"
    --device 0
    --encoded-root "$encoded_root"
    --threshold "$threshold"
    --block 256
    --validate
    --warp-strategy passive
    --warmup 5
    --iters 20
    --csv "${run_dir}/ncu_bench.csv"
)
if [[ -z "$NCU_BIN" ]]; then
  echo "  NCU not available — skipping NCU run" >&2
  ncu_fail_reason="NCU binary not found"
  echo "FAIL: ${ncu_fail_reason}" > "${run_dir}/ncu_status.txt"
else
  # -------------------------------------------------------------------
  # Attempt 1: limited --metrics (lightweight, preferred)
  # -------------------------------------------------------------------
  echo "  Attempt 1: --metrics (limited set)" >&2
  ncu1_export_base="${run_dir}/ncu_unblock"
  ncu1_report="${ncu1_export_base}.ncu-rep"
  set +e
  "$NCU_BIN" \
    --target-processes all \
    --replay-mode kernel \
    --launch-skip 0 \
    --launch-count 5 \
    --metrics \
dram__bytes_read.sum,\
dram__bytes_write.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
smsp__sass_inst_executed_op_global_ld.sum \
    --import-source yes \
    --source-folders "$ROOT_DIR" \
    --force-overwrite true \
    -o "${ncu1_export_base}" \
    "${ncu_bench_args[@]}" \
    >> "$ncu_stdout" 2>> "$ncu_stderr"
  ncu1_rc=$?
  set -e
  # Check attempt 1
  if [[ -f "$ncu1_report" ]] && [[ -s "$ncu1_report" ]]; then
    if ! grep -q "No kernels were profiled" "$ncu_stdout" 2>/dev/null; then
      echo "  ✓ Attempt 1 (--metrics) produced .ncu-rep ($(stat --printf='%s' "$ncu1_report") bytes)" >&2
      ncu_passed=1
      NCU_REPORT_SUCCESS="$ncu1_report"
    fi
  fi
  # -------------------------------------------------------------------
  # Attempt 2 (fallback): --set basic if attempt 1 failed
  # -------------------------------------------------------------------
  if [[ $ncu_passed -eq 0 ]]; then
    ncu2_export_base="${run_dir}/ncu_unblock_basic"
    ncu2_report="${ncu2_export_base}.ncu-rep"
    echo "  Attempt 1 did not capture kernels. Attempt 2: --set basic" >&2
    set +e
    "$NCU_BIN" \
      --target-processes all \
      --replay-mode kernel \
      --launch-skip 0 \
      --launch-count 5 \
      --set basic \
      --import-source yes \
      --source-folders "$ROOT_DIR" \
      --force-overwrite true \
      -o "${ncu2_export_base}" \
      "${ncu_bench_args[@]}" \
      >> "$ncu_stdout" 2>> "$ncu_stderr"
    ncu2_rc=$?
    set -e
    if [[ -f "$ncu2_report" ]] && [[ -s "$ncu2_report" ]]; then
      if ! grep -q "No kernels were profiled" "$ncu_stdout" 2>/dev/null; then
        echo "  ✓ Attempt 2 (--set basic) produced .ncu-rep ($(stat --printf='%s' "$ncu2_report") bytes)" >&2
        ncu_passed=1
        NCU_REPORT_SUCCESS="$ncu2_report"
      fi
    fi
    if [[ $ncu_passed -eq 0 ]]; then
      ncu_fail_reason="Both NCU attempts failed to capture kernels"
      echo "  ✗ ${ncu_fail_reason}" >&2
    fi
  fi
  # CSV export from whichever report succeeded
  if [[ $ncu_passed -eq 1 ]] && [[ -f "$NCU_REPORT_SUCCESS" ]]; then
    echo "  Exporting NCU CSV for inspection..." >&2
    "$NCU_BIN" --csv --page details \
      --force-overwrite true \
      -i "$NCU_REPORT_SUCCESS" \
      -o "${NCU_REPORT_SUCCESS%.ncu-rep}.csv" \
      > /dev/null 2>> "$ncu_stderr" || true
  fi
  if [[ $ncu_passed -eq 1 ]]; then
    echo "PASS: NCU captured kernels (report: ${NCU_REPORT_SUCCESS})" > "${run_dir}/ncu_status.txt"
  else
    echo "FAIL: ${ncu_fail_reason}" > "${run_dir}/ncu_status.txt"
  fi
fi
echo "" >&2
# ==============================================================================
# run_meta.txt
# ==============================================================================
{
  echo "=== Exp4 NCU Unblock Smoke — run_meta ==="
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
  echo "ncu version: $($NCU_BIN --version 2>&1 | head -1 || echo unknown)"
  echo ""
  echo "=== Configuration ==="
  echo "Benchmark: bench_progressive_filter"
  echo "Dataset: ${dataset_name}"
  echo "Encoded artifact: ${encoded_subdir} (root: ${encoded_root})"
  echo "Threshold: ${threshold} (derived from raw FP64, selectivity=${selectivity})"
  echo "Block: 256"
  echo "Warp strategy: passive"
  echo "Warmup: 5  Iters: 20"
  echo ""
  echo "=== NCU flags ==="
  echo "  Attempt 1 (--metrics):"
  echo "    --target-processes all"
  echo "    --replay-mode kernel"
  echo "    --launch-skip 0"
  echo "    --launch-count 5"
  echo "    --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,smsp__sass_inst_executed_op_global_ld.sum"
  echo "    --import-source yes"
  echo "    --force-overwrite true"
  echo "  Attempt 2 (fallback, runs only if attempt 1 fails):"
  echo "    Same as above but --set basic instead of --metrics"
  echo ""
  echo "=== Results ==="
  echo "Baseline CSV: ${baseline_csv}"
  echo "Baseline exit code: ${baseline_rc}"
  if [[ -f "$baseline_csv" ]]; then
    echo "Baseline gpu_count: ${gpu_count:-unknown}"
  fi
  echo "NCU successful report: ${NCU_REPORT_SUCCESS:-none}"
  echo "NCU passed: ${ncu_passed}"
  echo "Output dir: ${run_dir}"
  echo ""
  echo "=== Caveat: NCU metric names ==="
  echo "  The four --metrics names used in attempt 1 are standard NVIDIA NCU"
  echo "  metrics. If any are invalid on this NCU/GPU combination, NCU prints"
  echo "  a warning but still profiles. The fallback (attempt 2) uses --set basic"
  echo "  which is always valid. Primary goal is kernel capture, not metrics."
  echo ""
  echo "=== Purpose ==="
  echo "  Verify that NCU can capture progressive_filter_rowpack16_passive kernels."
  echo "  Not for performance analysis."
} > "${run_dir}/run_meta.txt"
echo "Meta: ${run_dir}/run_meta.txt" >&2
# ==============================================================================
# Final summary
# ==============================================================================
echo "=== Summary ===" >&2
echo "Baseline: ${baseline_rc} (gpu_count=${gpu_count:-N/A})" >&2
echo "NCU status file: ${run_dir}/ncu_status.txt" >&2
if [[ -f "${run_dir}/ncu_status.txt" ]]; then
  cat "${run_dir}/ncu_status.txt" >&2
fi
# Exit with baseline RC if baseline failed; otherwise 0 (NCU failure is
# informational in this smoke — we wrote ncu_status.txt for the handoff)
if [[ $baseline_rc -ne 0 ]]; then
  echo "Baseline benchmark failed — exiting with RC=${baseline_rc}" >&2
  exit "${baseline_rc}"
fi
echo "✓ Exp4 NCU unblock smoke complete" >&2
exit 0
