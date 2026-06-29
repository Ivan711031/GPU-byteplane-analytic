#!/usr/bin/env bash
#SBATCH --job-name=exp3_v2_minmax_char
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp3_v2_minmax_char_%j.log
#SBATCH --error=logs/exp3_v2_minmax_char_%j.err
#
# Phase 2 characterization: MIN/MAX progressive bound convergence
# + candidate-set collapse analysis across all v2 artifacts.
# Do NOT submit directly. Provide --account on sbatch command line.
#   sbatch --account=<PROJECT> scripts/run_exp3_v2_minmax_characterization.sbatch
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
JOB_ID="${SLURM_JOB_ID:-nojob}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_BASE_REL="results/exp3_v2_minmax_characterization"
RUN_DIR="${ROOT_DIR}/${RESULTS_BASE_REL}/run_${TIMESTAMP}_job${JOB_ID}_H200"
LOG_DIR="${ROOT_DIR}/logs"
MARKER_DIR="${ROOT_DIR}/handoff/job_done"
V2_ROOT="${WORK_DIR}/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
BUILD_DIR="${ROOT_DIR}/build/exp3"
BIN="${BUILD_DIR}/bench_progressive_aggregation"
DATASETS=(
  "heavy_tailed_p2"
  "heavy_tailed_p3"
  "heavy_tailed_p4"
  "heavy_tailed_p5"
  "heavy_tailed_p6"
  "sensor_p2"
  "sensor_p4"
  "sensor_p6"
  "sensor_p8"
  "sensor_p10"
  "uniform_p2"
  "uniform_p4"
  "uniform_p6"
  "uniform_p8"
  "uniform_p10"
  "zipfian_p2"
  "zipfian_p4"
  "zipfian_p6"
  "zipfian_p8"
)
write_marker() {
  local exit_code="$1"
  local marker_file="${MARKER_DIR}/job_${JOB_ID}.json"
  cat > "$marker_file" << EOF
{
  "job_id": "${JOB_ID}",
  "job_name": "exp3_v2_minmax_char",
  "exit_status": ${exit_code},
  "finished_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "workdir": "${ROOT_DIR}",
  "git_commit": "$(cd "${ROOT_DIR}" && git rev-parse HEAD 2>/dev/null || echo unknown)",
  "result_dir": "${RUN_DIR}",
  "datasets_count": ${#DATASETS[@]},
  "next_action": "Run analyze_exp3_v2_minmax_characterization.py on the result directory"
}
EOF
}
trap 'write_marker $?' EXIT
echo "[$(date)] [exp3-v2-minmax-char] job ${JOB_ID} started"
mkdir -p "$LOG_DIR" "$MARKER_DIR" "$RUN_DIR"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
case "$GPU_NAME" in
  *H200*) ;;
  *)
    echo "ERROR: Expected H200 GPU, got: ${GPU_NAME}" >&2
    exit 2
    ;;
esac
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load miniconda3/26.1.1 || true
  module load cuda/12.6 || true
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate gpu-byteplane-scan
fi
echo "[$(date)] toolchain: nvcc=$(which nvcc) cmake=$(cmake --version | head -n1) gcc=$(gcc --version | head -n1)"
echo "[$(date)] building Exp3 benchmark"
cmake -S "${ROOT_DIR}/benchmarks/experiment3" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "$BUILD_DIR" -j
if [[ ! -x "$BIN" ]]; then
  echo "ERROR: binary not found: $BIN" >&2
  exit 1
fi
{
  echo "timestamp=${TIMESTAMP}"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "gpu_name=${GPU_NAME}"
  echo "v2_root=${V2_ROOT}"
  echo "datasets=${DATASETS[*]}"
  echo "git_commit=$(cd "${ROOT_DIR}" && git rev-parse HEAD 2>/dev/null || echo unknown)"
} > "${RUN_DIR}/run_meta.txt"
for ds in "${DATASETS[@]}"; do
  ENCODED_ROOT="${V2_ROOT}/${ds}"
  CSV_PATH="${RUN_DIR}/exp3_v2_minmax_${ds}.csv"
  if [[ ! -d "$ENCODED_ROOT" ]]; then
    echo "WARNING: dataset directory not found: $ENCODED_ROOT, skipping" >&2
    continue
  fi
  MAX_PLANES="$(python3 -c "import json; m=json.load(open('${ENCODED_ROOT}/manifest.json')); print(m['max_plane_count'])")"
  REFINE_MAX=$((MAX_PLANES - 1))
  echo "[$(date)] MIN/MAX sweep ${ds} (k=0..${REFINE_MAX})"
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
  echo "[$(date)] ${ds} complete. Rows: $(wc -l < "$CSV_PATH")"
done
echo "[$(date)] all characterization sweeps complete. Results in: ${RUN_DIR}"
