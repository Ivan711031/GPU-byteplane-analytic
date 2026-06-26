#!/usr/bin/env bash
# run_exp4_ncu_basic_metrics.sbatch
#
# P2: collect real NCU basic metrics for the Exp4 progressive filter kernel.
# P0 proved NCU can capture with --profile-from-start yes (no replay-mode,
# no launch-count). P1 proved NSYS sees 20 launches at ~116 us each.
# This script uses --set basic for actual memory/compute metrics.
#
# Usage:
#   sbatch --account=<ACCOUNT> scripts/run_exp4_ncu_basic_metrics.sbatch
#
# Output:
#   results/exp4/ncu_basic_metrics_<timestamp>_job<JOBID>_H200/
#     baseline.csv                — direct correctness CSV (warmup=0 iters=20)
#     ncu_stdout.txt              — ncu profile stdout
#     ncu_stderr.txt              — ncu profile stderr
#     exp4_ncu_basic.ncu-rep      — Nsight Compute report (--set basic)
#     metric_summary.txt          — brief summary of metrics collected
#     ncu_status.txt              — PASS/FAIL with reason
#     run_meta.txt                — metadata
#   handoff/job_done/job_<SLURM_JOB_ID>.json

#SBATCH --job-name=exp4-ncu-basic-metrics
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
  local ncu_report_path=""
  local ncu_status_str="not_run"
  local result_dir_str=""

  if [[ -d "$run_dir" ]]; then
    result_dir_str="$run_dir"
    ncu_report_path="${run_dir}/exp4_ncu_basic.ncu-rep"
    if [[ -f "${run_dir}/ncu_status.txt" ]]; then
      ncu_status_str=$(head -1 "${run_dir}/ncu_status.txt" 2>/dev/null || echo "unknown")
    fi
  fi

  local git_commit
  git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

  local next_action="Check ncu_status.txt and metric_summary.txt for --set basic metrics."
  if [[ "$status" -ne 0 ]]; then
    next_action="Job failed with exit code ${status}; investigate slurm output."
  fi

  mkdir -p "$handoff_dir"
  cat > "$handoff_marker" <<EOF
{
  "job_id": "${job_id}",
  "job_name": "exp4-ncu-basic-metrics",
  "exit_status": ${status},
  "finished_at": "$(date -Is)",
  "workdir": "$(pwd)",
  "git_commit": "${git_commit}",
  "result_dir": "${result_dir_str}",
  "ncu_report": "${ncu_report_path}",
  "ncu_status": "${ncu_status_str}",
  "ncu_stdout": "${run_dir}/ncu_stdout.txt",
  "ncu_stderr": "${run_dir}/ncu_stderr.txt",
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
run_dir="${ROOT_DIR}/results/exp4/ncu_basic_metrics_${timestamp}_job${job_id}_${GPU_TAG}"

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
elif [[ -x /work/envstack/apps/cuda/12.6/bin/ncu ]]; then
  NCU_BIN=/work/envstack/apps/cuda/12.6/bin/ncu
fi
if [[ -n "$NCU_BIN" ]]; then
  echo "ncu: $NCU_BIN $($NCU_BIN --version 2>&1 | head -1)" >&2
else
  echo "WARNING: ncu not found in PATH or known locations" >&2
fi

# ==============================================================================
# Job metadata
# ==============================================================================
echo "=== Exp4 NCU Basic Metrics ===" >&2
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
# NCU: basic metrics (--set basic, no replay-mode, no launch-count)
# ==============================================================================
echo "=== NCU basic metrics ===" >&2

ncu_stdout="${run_dir}/ncu_stdout.txt"
ncu_stderr="${run_dir}/ncu_stderr.txt"
ncu_export_base="${run_dir}/exp4_ncu_basic"
ncu_report="${ncu_export_base}.ncu-rep"

ncu_passed=0
ncu_fail_reason=""

if [[ -z "$NCU_BIN" ]]; then
  echo "  NCU not available — skipping NCU run" >&2
  echo "FAIL: NCU binary not found" > "${run_dir}/ncu_status.txt"
else
  set +e
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
      --warp-strategy passive \
      --warmup 0 \
      --iters 20 \
      --csv "${run_dir}/ncu_bench.csv" \
    > "$ncu_stdout" 2> "$ncu_stderr"
  ncu_rc=$?
  set -e

  echo "  NCU exit code: ${ncu_rc}" >&2

  # Detect success: non-empty .ncu-rep AND no "No kernels were profiled"
  if grep -q "No kernels were profiled" "$ncu_stdout" 2>/dev/null; then
    ncu_fail_reason="NCU reported 'No kernels were profiled'"
    echo "  ✗ ${ncu_fail_reason}" >&2
  elif [[ -f "$ncu_report" ]] && [[ -s "$ncu_report" ]]; then
    echo "  ✓ .ncu-rep exists and non-empty ($(stat --printf='%s' "$ncu_report") bytes)" >&2
    ncu_passed=1
  else
    ncu_fail_reason=".ncu-rep missing or empty"
    echo "  ✗ ${ncu_fail_reason}" >&2
  fi

  # ---------------------------------------------------------------
  # Post-profile: metric summary and optional CSV/raw export
  # ---------------------------------------------------------------
  if [[ $ncu_passed -eq 1 ]]; then
    # Write metric summary
    {
      echo "=== Metric Summary ==="
      echo "NCU set: basic"
      echo "Report: ${ncu_report}"
      report_size=$(stat --printf='%s' "$ncu_report" 2>/dev/null || echo "?")
      echo "Report size: ${report_size} bytes"
      echo ""
      echo "--- NCU stdout (last 80 lines) ---"
      tail -80 "$ncu_stdout" 2>/dev/null || echo "(stdout unavailable)"
      echo ""
      echo "--- Grep: metric-related lines from stdout ---"
      grep -i 'dram__\|lts__\|sm__\|smsp__\|gpu__\|launch__\|profiler__' "$ncu_stdout" 2>/dev/null \
        | head -30 || echo "(no metric lines found in stdout)"
    } > "${run_dir}/metric_summary.txt"

    # Try CSV export from report (best-effort, do not fail job)
    echo "  Exporting NCU CSV from report..." >&2
    "$NCU_BIN" --csv --page raw \
      --force-overwrite true \
      -i "$ncu_report" \
      -o "${ncu_export_base}" \
      >> "$ncu_stdout" 2>> "$ncu_stderr" || true

    echo "PASS: NCU captured basic metrics" > "${run_dir}/ncu_status.txt"
  else
    echo "FAIL: ${ncu_fail_reason}" > "${run_dir}/ncu_status.txt"
  fi
fi

echo "" >&2

# ==============================================================================
# run_meta.txt
# ==============================================================================
{
  echo "=== Exp4 NCU Basic Metrics — run_meta ==="
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
  echo "Encoded artifact: ${encoded_subdir}"
  echo "Threshold: ${threshold} (derived from raw FP64, selectivity=${selectivity})"
  echo "Block: 256"
  echo "Warp strategy: passive"
  echo "Warmup: 0  Iters: 20"
  echo ""
  echo "=== NCU flags ==="
  echo "  --set basic"
  echo "  --target-processes all"
  echo "  --profile-from-start yes"
  echo "  --page raw"
  echo "  -f"
  echo "  NO --launch-count, NO --launch-skip, NO --replay-mode"
  echo "  NO kernel-name filter"
  echo ""
  echo "=== Results ==="
  echo "Baseline CSV: ${baseline_csv}"
  echo "Baseline exit code: ${baseline_rc}"
  if [[ -f "$baseline_csv" ]]; then
    echo "Baseline gpu_count: ${gpu_count:-unknown}"
  fi
  echo "NCU report: ${ncu_report}"
  echo "NCU exit code: ${ncu_rc:-N/A}"
  echo "NCU passed: ${ncu_passed}"
  echo "Output dir: ${run_dir}"
  echo ""
  echo "=== Purpose ==="
  echo "  P2: collect real NCU basic metrics for the Exp4 progressive filter."
  echo "  Uses --set basic (memory/compute metrics), no replay-mode, no launch-count."
  echo "  Confirmed working command shape from P0 job 51933."
} > "${run_dir}/run_meta.txt"

echo "Meta: ${run_dir}/run_meta.txt" >&2

# ==============================================================================
# Final summary
# ==============================================================================
echo "=== Summary ===" >&2
echo "Baseline: exit=${baseline_rc}" >&2
echo "NCU status file: ${run_dir}/ncu_status.txt" >&2
if [[ -f "${run_dir}/ncu_status.txt" ]]; then
  cat "${run_dir}/ncu_status.txt" >&2
fi

if [[ $baseline_rc -ne 0 ]]; then
  echo "Baseline benchmark failed — exiting with RC=${baseline_rc}" >&2
  exit "${baseline_rc}"
fi

echo "✓ Exp4 NCU basic metrics complete" >&2
exit 0
