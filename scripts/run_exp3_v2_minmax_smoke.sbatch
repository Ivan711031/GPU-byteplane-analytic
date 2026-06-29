#!/usr/bin/env bash
#SBATCH --job-name=exp3_v2_minmax_smoke
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/exp3_v2_minmax_smoke_%j.log
#SBATCH --error=logs/exp3_v2_minmax_smoke_%j.err
#
# Smoke test: Exp3 progressive MIN/MAX on v2 byte-plane artifacts.
# Sweeps all refinement depths for 4 representative datasets.
# Do NOT submit directly. Provide --account on sbatch command line.
#   sbatch --account=<PROJECT> scripts/run_exp3_v2_minmax_smoke.sbatch
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
RESULTS_BASE_REL="results/exp3_v2_minmax_smoke"
RUN_DIR="${ROOT_DIR}/${RESULTS_BASE_REL}/run_${TIMESTAMP}_job${JOB_ID}_H200"
LOG_DIR="${ROOT_DIR}/logs"
MARKER_DIR="${ROOT_DIR}/handoff/job_done"
V2_ROOT="${WORK_DIR}/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
BUILD_DIR="${ROOT_DIR}/build/exp3"
BIN="${BUILD_DIR}/bench_progressive_aggregation"
# Smoke set: 4 representative datasets
DATASETS=(
  "heavy_tailed_p6"
  "sensor_p2"
  "uniform_p10"
  "zipfian_p8"
)
# ---------------------------------------------------------------------------
# Completion marker via trap
# ---------------------------------------------------------------------------
write_marker() {
  local exit_code="$1"
  local marker_file="${MARKER_DIR}/job_${JOB_ID}.json"
  cat > "$marker_file" << EOF
{
  "job_id": "${JOB_ID}",
  "job_name": "exp3_v2_minmax_smoke",
  "exit_status": ${exit_code},
  "finished_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "workdir": "${ROOT_DIR}",
  "git_commit": "$(cd "${ROOT_DIR}" && git rev-parse HEAD 2>/dev/null || echo unknown)",
  "result_dir": "${RUN_DIR}",
  "datasets": [$(printf '"%s",' "${DATASETS[@]}" | sed 's/,$//')],
  "next_action": "Verify bound_valid_min and bound_valid_max columns are true for all rows"
}
EOF
}
trap 'write_marker $?' EXIT
echo "[$(date)] [exp3-v2-minmax-smoke] job ${JOB_ID} started"
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
  echo "v2_root=${V2_ROOT}"
  echo "datasets=${DATASETS[*]}"
  echo "git_commit=$(cd "${ROOT_DIR}" && git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_status_begin"
  (cd "${ROOT_DIR}" && git status --short --untracked-files=no 2>/dev/null || true)
  echo "git_status_end"
  echo "cuda_begin"
  which nvcc
  nvcc --version 2>/dev/null || true
  cmake --version 2>/dev/null | head -n 1 || true
  gcc --version 2>/dev/null | head -n 1 || true
  echo "cuda_end"
} > "${RUN_DIR}/run_meta.txt"
# ---------------------------------------------------------------------------
# Run MIN/MAX sweep for each dataset
# ---------------------------------------------------------------------------
for ds in "${DATASETS[@]}"; do
  ENCODED_ROOT="${V2_ROOT}/${ds}"
  CSV_PATH="${RUN_DIR}/exp3_v2_minmax_${ds}.csv"
  if [[ ! -d "$ENCODED_ROOT" ]]; then
    echo "WARNING: dataset directory not found: $ENCODED_ROOT, skipping" >&2
    continue
  fi
  # Determine max_plane_count from manifest
  MAX_PLANES="$(python3 -c "import json; m=json.load(open('${ENCODED_ROOT}/manifest.json')); print(m['max_plane_count'])")"
  REFINE_MAX=$((MAX_PLANES - 1))
  echo "[$(date)] running MIN/MAX sweep for ${ds} (k=0..${REFINE_MAX})"
  "$BIN" \
    --device 0 \
    --mode encoded_dev_subcolumns \
    --encoded-root "$ENCODED_ROOT" \
    --aggregation minmax \
    --refine_min 0 \
    --refine_max "$REFINE_MAX" \
    --load_strategy rowpack16 \
    --real_kernel_variant runtime \
    --block 256 \
    --items_per_thread 1 \
    --warmup 5 \
    --iters 100 \
    --validate \
    --csv "$CSV_PATH"
  echo "[$(date)] ${ds} complete. Results: ${CSV_PATH}"
  ls -lh "$CSV_PATH"
done
echo "[$(date)] all MIN/MAX smoke tests complete. Results in: ${RUN_DIR}"
