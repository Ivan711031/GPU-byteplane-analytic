#!/usr/bin/env bash
# run_r10_kdepth_freeze.sbatch
#
# R10: v1.3 evidence freeze for k-depth dispatch and raw-fallback boundary.
# Runs bench_filter_aggregate (non-NCU) across required k values.
#
# Required scope (issue #218):
#   Datasets: uniform_p10, heavy_tailed_p6
#   Depths:   k=1,2,3,4,k=max, raw-fused FP64 baseline
#   Metrics:  latency, speedup, SUM error, dispatch verdict
#
# Output: results/v1_3_freeze/k_depth/

#SBATCH --job-name=r10-kdepth
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=results/v1_3_freeze/k_depth/r10_kdepth_%j.out
#SBATCH --error=results/v1_3_freeze/k_depth/r10_kdepth_%j.err

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

RESULT_DIR="results/v1_3_freeze/k_depth"
mkdir -p "$RESULT_DIR"

STDOUT_PATH="${RESULT_DIR}/r10_kdepth_${JOB_ID}.out"
STDERR_PATH="${RESULT_DIR}/r10_kdepth_${JOB_ID}.err"
GPU_UTIL_LOG="${RESULT_DIR}/gpu_util_${JOB_ID}.log"

if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv -l 2 > "$GPU_UTIL_LOG" 2>&1 &
  GPU_UTIL_PID=$!
else
  GPU_UTIL_PID=""
fi

# ==============================================================================
# Handoff marker via trap
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
  local next_action="Run analysis: aggregate bench CSVs into freeze matrix, generate report at research/2026-06-XX_R10_v1_3_k_depth_dispatch_freeze.md"
  if [[ "$status" -ne 0 ]]; then
    next_action="Job failed with exit code ${status}; inspect slurm output."
  fi
  cat > "$HANDOFF_MARKER" <<HEREDOC
{
  "exit_status": $status,
  "job_id": "${JOB_ID}",
  "job_name": "r10-kdepth",
  "finished_at": "$(date -Iseconds)",
  "workdir": "$ROOT_DIR",
  "git_commit": "${git_commit}",
  "stdout": "$STDOUT_PATH",
  "stderr": "$STDERR_PATH",
  "gpu_util_log": "$GPU_UTIL_LOG",
  "result_dir": "$RESULT_DIR",
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

echo "=== Job info ===" >&2
echo "Job ID: ${JOB_ID}  Host: $(hostname)  GPU: ${GPU_NAME}" >&2
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
cmake --build "$BUILD_DIR" -j"$BUILD_JOBS" --target bench_filter_aggregate
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
# Run matrix: each (dataset, k) pair gets one bench_filter_aggregate invocation.
# The binary internally runs both fused and raw FP64 kernels per invocation,
# so we get latency, speedup, and SUM error from a single run.
# ==============================================================================
declare -A THRESHOLD
THRESHOLD["uniform_p10"]="500.0271759315981"
THRESHOLD["heavy_tailed_p6"]="1.0001588729752771"
declare -A MAX_K
MAX_K["uniform_p10"]=6
MAX_K["heavy_tailed_p6"]=8

TOTAL_OK=0
TOTAL_FAIL=0
FAILURES=()

for artifact in uniform_p10 heavy_tailed_p6; do
  dataset="${artifact%_p*}"
  artifact_root="${V2_ROOT}/${artifact}"
  raw_path="${RAW_ROOT}/${dataset}.f64le.bin"
  threshold="${THRESHOLD[$artifact]}"
  max_k="${MAX_K[$artifact]}"

  if [[ ! -d "$artifact_root" ]]; then
    echo "ERROR: artifact root not found: $artifact_root" >&2
    FAILURES+=("${artifact}: missing dataset dir")
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
    continue
  fi
  if [[ ! -f "$raw_path" ]]; then
    echo "ERROR: raw file not found: $raw_path" >&2
    FAILURES+=("${artifact}: missing raw file")
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
    continue
  fi

  for k in 1 2 3 4 $max_k; do
    echo "" >&2
    echo "============================================================" >&2
    echo "[${artifact}/k=${k}] threshold=${threshold}" >&2
    echo "============================================================" >&2

    bench_csv="${RESULT_DIR}/${artifact}_k${k}.csv"
    bench_log="${RESULT_DIR}/${artifact}_k${k}.log"

    set +e
    "$BENCH_BIN" \
      --dataset "$artifact_root" \
      --raw "$raw_path" \
      --threshold "$threshold" \
      --max-filter-planes "$k" \
      --iters 200 \
      --warmup 10 \
      --validate \
      --csv "$bench_csv" \
      > "$bench_log" 2>&1
    bench_rc=$?
    set -e

    if [[ $bench_rc -ne 0 ]]; then
      echo "  ✗ Benchmark failed (rc=${bench_rc})" >&2
      FAILURES+=("${artifact}_k${k}: bench rc=${bench_rc}")
      TOTAL_FAIL=$((TOTAL_FAIL + 1))
      continue
    fi

    echo "  ✓ k=${k} ms_per_iter=$(tail -1 $bench_csv | cut -d, -f12)" >&2
    TOTAL_OK=$((TOTAL_OK + 1))
  done
done

# ==============================================================================
# run_meta.txt
# ==============================================================================
{
  echo "=== R10 v1.3 k-depth dispatch freeze — run_meta ==="
  echo ""
  echo "Date: $(date -Iseconds)"
  echo "Host: $(hostname)"
  echo "GPU: ${GPU_NAME}"
  echo "Job ID: ${JOB_ID}"
  echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "Bench binary: bench_filter_aggregate"
  echo "Iterations: 200  Warmup: 10"
  echo ""
  echo "=== Cases ==="
  echo "  uniform_p10 (max_k=6): k=1 2 3 4 6"
  echo "  heavy_tailed_p6 (max_k=8): k=1 2 3 4 8"
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
  echo "Output dir: ${RESULT_DIR}"
} > "${RESULT_DIR}/run_meta.txt"

echo "" >&2
echo "=== R10 k-depth freeze benchmark complete ===" >&2
echo "PASS: ${TOTAL_OK}  FAIL: ${TOTAL_FAIL}" >&2
if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo "Failures:" >&2
  for f in "${FAILURES[@]}"; do
    echo "  ${f}" >&2
  done
fi
exit 0
