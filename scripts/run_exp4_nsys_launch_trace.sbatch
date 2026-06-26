#!/usr/bin/env bash
# run_exp4_nsys_launch_trace.sbatch
#
# P1: after P0 confirmed NCU CAN see Exp4 kernels, use Nsight Systems to
# confirm actual CUDA launch count, kernel names, process behavior, and
# timing for bench_progressive_filter with warp_strategy=passive.
#
# Uses --warmup 0 --iters 20 so the timeline has enough samples.
# Does NOT collect NCU metrics.
#
# Usage:
#   sbatch --account=<ACCOUNT> scripts/run_exp4_nsys_launch_trace.sbatch
#
# Output:
#   results/exp4/nsys_launch_trace_<timestamp>_job<JOBID>_H200/
#     baseline.csv                — direct correctness CSV (warmup=0 iters=20)
#     nsys_stdout.txt             — nsys profile stdout
#     nsys_stderr.txt             — nsys profile stderr
#     exp4_launch_trace.nsys-rep  — Nsight Systems report
#     nsys_stats_gpu_kern.csv     — CUDA GPU Kernel Summary (if stats succeeds)
#     nsys_stats_cuda_api.csv     — CUDA API Summary (if stats succeeds)
#     nsys_status.txt             — PASS/FAIL with reason
#     run_meta.txt                — metadata
#   handoff/job_done/job_<SLURM_JOB_ID>.json

#SBATCH --job-name=exp4-nsys-launch-trace
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
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
V2_ROOT="/work/u4063895/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
RAW_ROOT="/work/u4063895/datasets/synthetic/dev"

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

# ==============================================================================
# Handoff marker via trap (set before GPU validation so exit 2 also writes marker)
# ==============================================================================
write_handoff_marker() {
  local status=$?
  local nsys_report_path=""
  local nsys_status_str="not_run"
  local result_dir_str=""

  if [[ -d "$run_dir" ]]; then
    result_dir_str="$run_dir"
    nsys_report_path="${run_dir}/exp4_launch_trace.nsys-rep"
    if [[ -f "${run_dir}/nsys_status.txt" ]]; then
      nsys_status_str=$(head -1 "${run_dir}/nsys_status.txt" 2>/dev/null || echo "unknown")
    fi
  fi

  local git_commit
  git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

  local next_action="Check nsys_status.txt and nsys_stdout.txt for launch count and kernel names."
  if [[ "$status" -ne 0 ]]; then
    next_action="Job failed with exit code ${status}; investigate slurm output."
  fi

  mkdir -p "$handoff_dir"
  cat > "$handoff_marker" <<EOF
{
  "job_id": "${job_id}",
  "job_name": "exp4-nsys-launch-trace",
  "exit_status": ${status},
  "finished_at": "$(date -Is)",
  "workdir": "$(pwd)",
  "git_commit": "${git_commit}",
  "result_dir": "${result_dir_str}",
  "nsys_report": "${nsys_report_path}",
  "nsys_status": "${nsys_status_str}",
  "nsys_stdout": "${run_dir}/nsys_stdout.txt",
  "nsys_stderr": "${run_dir}/nsys_stderr.txt",
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
run_dir="${ROOT_DIR}/results/exp4/nsys_launch_trace_${timestamp}_job${job_id}_${GPU_TAG}"

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

NSYS_BIN=""
if command -v nsys &>/dev/null; then
  NSYS_BIN=$(which nsys)
elif [[ -x /work/envstack/apps/cuda/12.6/bin/nsys ]]; then
  NSYS_BIN=/work/envstack/apps/cuda/12.6/bin/nsys
elif [[ -x /opt/nvidia/nsight-systems/2025.3.2/bin/nsys ]]; then
  NSYS_BIN=/opt/nvidia/nsight-systems/2025.3.2/bin/nsys
fi
if [[ -n "$NSYS_BIN" ]]; then
  echo "nsys: $NSYS_BIN $($NSYS_BIN --version 2>&1 | head -1)" >&2
else
  echo "WARNING: nsys not found in PATH or known locations" >&2
fi

# ==============================================================================
# Job metadata
# ==============================================================================
echo "=== Exp4 NSYS Launch Trace ===" >&2
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
echo "  threshold=${threshold}" >&2
echo "" >&2

# ==============================================================================
# Baseline: direct benchmark (warmup=0 iters=20, with --validate)
# ==============================================================================
echo "=== Baseline: direct benchmark (warmup=0 iters=20) ===" >&2
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
  --warmup 0 \
  --iters 20 \
  --csv "$baseline_csv" \
  2> "$baseline_log"
baseline_rc=$?
set -e

echo "  Baseline exit code: ${baseline_rc}" >&2
if [[ -f "$baseline_csv" ]]; then
  gpu_count=$(tail -1 "$baseline_csv" | cut -d',' -f12)
  cpu_encoded=$(tail -1 "$baseline_csv" | cut -d',' -f13)
  echo "  gpu_count=${gpu_count}  cpu_encoded_count=${cpu_encoded}" >&2
  if [[ "$gpu_count" == "$cpu_encoded" ]]; then
    echo "  ✓ Baseline validation PASS" >&2
  else
    echo "  ✗ Baseline validation FAIL" >&2
  fi
else
  echo "  WARNING: baseline CSV not found" >&2
fi
echo "" >&2

# ==============================================================================
# NSYS: launch trace with CUDA + NVTX
# ==============================================================================
echo "=== NSYS launch trace ===" >&2

nsys_stdout="${run_dir}/nsys_stdout.txt"
nsys_stderr="${run_dir}/nsys_stderr.txt"
nsys_export_base="${run_dir}/exp4_launch_trace"
nsys_report="${nsys_export_base}.nsys-rep"

if [[ -z "$NSYS_BIN" ]]; then
  echo "  NSYS not available — skipping NSYS run" >&2
  echo "FAIL: nsys binary not found" > "${run_dir}/nsys_status.txt"
else
  set +e
  "$NSYS_BIN" profile \
    --trace=cuda,nvtx \
    --force-overwrite=true \
    -o "$nsys_export_base" \
    "$BENCH_BIN" \
      --device 0 \
      --encoded-root "$encoded_root" \
      --threshold "$threshold" \
      --block 256 \
      --validate \
      --warp-strategy passive \
      --warmup 0 \
      --iters 20 \
      --csv "${run_dir}/nsys_bench.csv" \
    > "$nsys_stdout" 2> "$nsys_stderr"
  nsys_rc=$?
  set -e

  echo "  NSYS exit code: ${nsys_rc}" >&2

  # Detect success: non-empty .nsys-rep AND no error messages indicating failure
  nsys_passed=0
  nsys_fail_reason=""

  if [[ -f "$nsys_report" ]] && [[ -s "$nsys_report" ]]; then
    echo "  ✓ .nsys-rep exists and non-empty ($(stat --printf='%s' "$nsys_report") bytes)" >&2
    nsys_passed=1
  else
    nsys_fail_reason=".nsys-rep missing or empty"
    echo "  ✗ ${nsys_fail_reason}" >&2
  fi

  if grep -q "Error\|ERROR\|No kernels were profiled" "$nsys_stdout" 2>/dev/null; then
    nsys_fail_reason="NSYS reported error or no kernels"
    echo "  ✗ ${nsys_fail_reason}" >&2
    nsys_passed=0
  fi

  # ---------------------------------------------------------------
  # NSYS stats: export CUDA kernel and API summaries
  # ---------------------------------------------------------------
  if [[ $nsys_passed -eq 1 ]] && [[ -f "$nsys_report" ]]; then
    echo "  Exporting NSYS stats..." >&2

    gpu_kern_csv="${run_dir}/nsys_stats_gpu_kern.csv"
    cuda_api_csv="${run_dir}/nsys_stats_cuda_api.csv"

    "$NSYS_BIN" stats \
      --report cuda_gpu_kern_sum \
      --format csv \
      --output "$gpu_kern_csv" \
      "$nsys_report" \
      >> "$nsys_stdout" 2>> "$nsys_stderr"
    stats_kern_rc=$?

    "$NSYS_BIN" stats \
      --report cuda_api_sum \
      --format csv \
      --output "$cuda_api_csv" \
      "$nsys_report" \
      >> "$nsys_stdout" 2>> "$nsys_stderr"
    stats_api_rc=$?

    if [[ $stats_kern_rc -eq 0 ]]; then
      echo "  ✓ Kernel summary: ${gpu_kern_csv}" >&2
    else
      echo "  ⚠ Kernel stats export failed (rc=${stats_kern_rc})" >&2
    fi
    if [[ $stats_api_rc -eq 0 ]]; then
      echo "  ✓ CUDA API summary: ${cuda_api_csv}" >&2
    else
      echo "  ⚠ CUDA API stats export failed (rc=${stats_api_rc})" >&2
    fi
  fi

  if [[ $nsys_passed -eq 1 ]]; then
    echo "PASS: NSYS captured trace successfully" > "${run_dir}/nsys_status.txt"
  else
    echo "FAIL: ${nsys_fail_reason}" > "${run_dir}/nsys_status.txt"
  fi
fi

echo "" >&2

# ==============================================================================
# run_meta.txt
# ==============================================================================
{
  echo "=== Exp4 NSYS Launch Trace — run_meta ==="
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
  echo "nsys version: $($NSYS_BIN --version 2>&1 | head -1 || echo unknown)"
  echo ""
  echo "=== Configuration ==="
  echo "Benchmark: bench_progressive_filter"
  echo "Dataset: ${dataset_name}"
  echo "Encoded artifact: ${encoded_subdir}"
  echo "Threshold: ${threshold} (derived from raw FP64, selectivity=${selectivity})"
  echo "Block: 256"
  echo "Warp strategy: passive"
  echo "Warmup: 0  Iters: 20"
  echo ""
  echo "=== NSYS flags ==="
  echo "  --trace=cuda,nvtx"
  echo "  --force-overwrite=true"
  echo "  -o <run_dir>/exp4_launch_trace"
  echo ""
  echo "=== NSYS stats reports requested ==="
  echo "  cuda_gpu_kern_sum  — CUDA GPU Kernel Summary"
  echo "  cuda_api_sum       — CUDA API Summary"
  echo ""
  echo "=== Results ==="
  echo "Baseline CSV: ${baseline_csv}"
  echo "Baseline exit code: ${baseline_rc}"
  if [[ -f "$baseline_csv" ]]; then
    echo "Baseline gpu_count: ${gpu_count:-unknown}"
  fi
  echo "NSYS report: ${nsys_report}"
  echo "NSYS exit code: ${nsys_rc:-N/A}"
  echo "NSYS passed: ${nsys_passed}"
  echo "Output dir: ${run_dir}"
  echo ""
  echo "=== Purpose ==="
  echo "  P1: confirm CUDA launch count, kernel names, process behavior,"
  echo "  and timing for the Exp4 progressive filter benchmark path."
  echo "  Does NOT collect NCU metrics."
} > "${run_dir}/run_meta.txt"

echo "Meta: ${run_dir}/run_meta.txt" >&2

# ==============================================================================
# Final summary
# ==============================================================================
echo "=== Summary ===" >&2
echo "Baseline: exit=${baseline_rc}" >&2
echo "NSYS status file: ${run_dir}/nsys_status.txt" >&2
if [[ -f "${run_dir}/nsys_status.txt" ]]; then
  cat "${run_dir}/nsys_status.txt" >&2
fi

if [[ $baseline_rc -ne 0 ]]; then
  echo "Baseline benchmark failed — exiting with RC=${baseline_rc}" >&2
  exit "${baseline_rc}"
fi

echo "✓ Exp4 NSYS launch trace complete" >&2
exit 0
