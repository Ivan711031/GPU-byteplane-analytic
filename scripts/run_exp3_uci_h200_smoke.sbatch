#!/usr/bin/env bash
#SBATCH --job-name=exp3_uci_h200_smoke
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/exp3_uci_h200_smoke_%j.log
#SBATCH --error=logs/exp3_uci_h200_smoke_%j.err
#
# Minimal smoke test: Exp3 progressive SUM on UCI Household real data.
# Sweeps all refinement depths (k=0..6) for the encoded UCI dataset.
# Do NOT submit directly. Provide --account on sbatch command line.
#   sbatch --account=<PROJECT> scripts/run_exp3_uci_h200_smoke.sbatch

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

JOB_ID="${SLURM_JOB_ID:-nojob}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_BASE_REL="results/exp3_uci_h200_smoke"
RUN_DIR="${ROOT_DIR}/${RESULTS_BASE_REL}/run_${TIMESTAMP}_job${JOB_ID}_H200"
LOG_DIR="${ROOT_DIR}/logs"
MARKER_DIR="${ROOT_DIR}/handoff/job_done"

ENCODED_ROOT="/work/u4063895/datasets/uci_household_global_active_power/bfp-dec12"
REFINE_MIN=0
REFINE_MAX=6
BUILD_DIR="${ROOT_DIR}/build/exp3"
BIN="${BUILD_DIR}/bench_progressive_aggregation"
CSV_PATH="${RUN_DIR}/exp3_uci_smoke.csv"

# ---------------------------------------------------------------------------
# Completion marker via trap (exit status captured at script end)
# Installed early so marker is written even if mkdir or early command fails.
# ---------------------------------------------------------------------------
write_marker() {
  local exit_code="$1"
  local marker_file="${MARKER_DIR}/job_${JOB_ID}.json"
  cat > "$marker_file" << EOF
{
  "job_id": "${JOB_ID}",
  "job_name": "exp3_uci_h200_smoke",
  "exit_status": ${exit_code},
  "finished_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "workdir": "${ROOT_DIR}",
  "git_commit": "$(cd "${ROOT_DIR}" && git rev-parse HEAD 2>/dev/null || echo unknown)",
  "stdout": "logs/exp3_uci_h200_smoke_${JOB_ID}.log",
  "stderr": "logs/exp3_uci_h200_smoke_${JOB_ID}.err",
  "result_dir": "${RUN_DIR}",
  "result_csv": "${CSV_PATH}",
  "next_action": "Verify correctness CSV columns (gpu_count == cpu_encoded_count) and performance sanity"
}
EOF
}
trap 'write_marker $?' EXIT

echo "[$(date)] [exp3-uci-smoke] job ${JOB_ID} started"

mkdir -p "$LOG_DIR" "$MARKER_DIR" "$RUN_DIR"

# ---------------------------------------------------------------------------
# Hardware gate: abort immediately if not H200
# ---------------------------------------------------------------------------
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
case "$GPU_NAME" in
  *H200*) ;;
  *)
    echo "ERROR: Expected H200 GPU, got: ${GPU_NAME}" >&2
    exit 2
    ;;
esac

# ---------------------------------------------------------------------------
# Environment: modules & conda
# ---------------------------------------------------------------------------
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load miniconda3/26.1.1 || true
  module load cuda/12.6 || true
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate gpu-byteplane-scan
fi

# ---------------------------------------------------------------------------
# Toolchain verification
# ---------------------------------------------------------------------------
echo "[$(date)] toolchain: nvcc=$(which nvcc) cmake=$(cmake --version | head -n1) gcc=$(gcc --version | head -n1)"

# ---------------------------------------------------------------------------
# Build benchmark binary
# ---------------------------------------------------------------------------
echo "[$(date)] building Exp3 benchmark"
cmake -S "${ROOT_DIR}/benchmarks/experiment3" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "$BUILD_DIR" -j

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: binary not found: $BIN" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Run meta
# ---------------------------------------------------------------------------
{
  echo "timestamp=${TIMESTAMP}"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_job_name=${SLURM_JOB_NAME:-}"
  echo "gpu_name=${GPU_NAME}"
  echo "encoded_root=${ENCODED_ROOT}"
  echo "value_count=2049280"
  echo "refine_min=${REFINE_MIN}"
  echo "refine_max=${REFINE_MAX}"
  echo "git_commit=$(cd "${ROOT_DIR}" && git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_status_begin"
  (cd "${ROOT_DIR}" && git status --short 2>/dev/null || true)
  echo "git_status_end"
  echo "cuda_begin"
  which nvcc
  nvcc --version 2>/dev/null || true
  cmake --version 2>/dev/null | head -n 1 || true
  gcc --version 2>/dev/null | head -n 1 || true
  echo "cuda_end"
} > "${RUN_DIR}/run_meta.txt"

# ---------------------------------------------------------------------------
# Run Exp3 progressive SUM sweep across all refinement depths (k=0..6)
# ---------------------------------------------------------------------------
echo "[$(date)] launching Exp3 encoded_dev_subcolumns refine=${REFINE_MIN}..${REFINE_MAX}"

"$BIN" \
  --device 0 \
  --mode encoded_dev_subcolumns \
  --encoded-root "$ENCODED_ROOT" \
  --refine_min "$REFINE_MIN" \
  --refine_max "$REFINE_MAX" \
  --load_strategy rowpack16 \
  --real_kernel_variant runtime \
  --block 256 \
  --items_per_thread 1 \
  --warmup 5 \
  --iters 100 \
  --validate \
  --csv "$CSV_PATH"

echo "[$(date)] smoke complete. Results: ${CSV_PATH}"
ls -lh "$CSV_PATH"
