#!/usr/bin/env bash
#SBATCH --job-name=exp3_real_ab
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=exp3_real_ab_%j.out
#SBATCH --error=exp3_real_ab_%j.err

set -euo pipefail

DEFAULT_ROOT_DIR="/home/u4063895/workspace/gpu-byteplane-scan-experiments"
RESULTS_BASE_REL="results/exp3_real_specialized_ab"
REAL_ENCODED_ROOT="/work/u4063895/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p/uniform_p10"
REAL_REFINE_DEPTH="5"

is_repo_root() {
  local candidate="$1"
  [[ -n "$candidate" &&
     -f "$candidate/scripts/run_exp3.sh" &&
     -d "$candidate/benchmarks/experiment3" ]]
}

require_h200() {
  local gpu_name
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  case "$gpu_name" in
    *H200*) ;;
    *)
      echo "Expected H200, got ${gpu_name}" >&2
      exit 2
      ;;
  esac
}

ROOT_DIR="${EXP3_ROOT_DIR:-}"
if ! is_repo_root "$ROOT_DIR"; then
  ROOT_DIR=""
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && is_repo_root "$SLURM_SUBMIT_DIR"; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
  elif is_repo_root "$DEFAULT_ROOT_DIR"; then
    ROOT_DIR="$DEFAULT_ROOT_DIR"
  else
    echo "could not locate repo root; set EXP3_ROOT_DIR explicitly" >&2
    exit 1
  fi
fi

if command -v module >/dev/null 2>&1; then
  module purge
  module load miniconda3/26.1.1 || true
  module load cuda/12.6 || true
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate gpu-byteplane-scan
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
job_id="${SLURM_JOB_ID:-nojob}"
run_dir="${ROOT_DIR}/${RESULTS_BASE_REL}/run_${timestamp}_job${job_id}_H200"
mkdir -p "$run_dir"

require_h200
cd "$ROOT_DIR"

gpu_log="${run_dir}/gpu_util.csv"
echo "timestamp,gpu_index,util_pct,mem_used_mb,mem_total_mb" > "$gpu_log"
(
  while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits | tr -d ' ' \
      | awk -v ts="$(date +%Y-%m-%dT%H:%M:%S)" '{print ts","$0}' >> "$gpu_log"
    sleep 5
  done
) &
tracker_pid=$!

cleanup() {
  kill "$tracker_pid" 2>/dev/null || true
  wait "$tracker_pid" 2>/dev/null || true
}
trap cleanup EXIT

{
  echo "timestamp=${timestamp}"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_job_name=${SLURM_JOB_NAME:-}"
  echo "slurm_job_partition=${SLURM_JOB_PARTITION:-}"
  echo "hostname=$(hostname)"
  echo "gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  echo "encoded_root=${REAL_ENCODED_ROOT}"
  echo "refine_depth=${REAL_REFINE_DEPTH}"
  echo "git_commit=$(git rev-parse HEAD || true)"
  echo "git_status_begin"
  git status --short || true
  echo "git_status_end"
  echo "cuda_begin"
  which nvcc || true
  nvcc --version || true
  echo "cuda_end"
} > "${run_dir}/run_meta.txt"

cmake -S benchmarks/experiment3 -B build/exp3 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp3 -j

bin="./build/exp3/bench_progressive_aggregation"

run_case() {
  local variant="$1"
  local csv_path="${run_dir}/real_uniform_depth9_${variant}.csv"
  local stdout_path="${run_dir}/real_uniform_depth9_${variant}.stdout"
  local stderr_path="${run_dir}/real_uniform_depth9_${variant}.stderr"

  {
    echo "case=${variant}"
    echo "command_begin"
    printf '%q ' "$bin" \
      --device 0 \
      --mode encoded_dev_subcolumns \
      --encoded-root "$REAL_ENCODED_ROOT" \
      --refine_min "$REAL_REFINE_DEPTH" \
      --refine_max "$REAL_REFINE_DEPTH" \
      --load_strategy rowpack16 \
      --real_kernel_variant "$variant" \
      --block 256 \
      --items_per_thread 1 \
      --warmup 10 \
      --iters 200 \
      --validate \
      --csv "$csv_path"
    echo
    echo "command_end"
  } >> "${run_dir}/run_meta.txt"

  "$bin" \
    --device 0 \
    --mode encoded_dev_subcolumns \
    --encoded-root "$REAL_ENCODED_ROOT" \
    --refine_min "$REAL_REFINE_DEPTH" \
    --refine_max "$REAL_REFINE_DEPTH" \
    --load_strategy rowpack16 \
    --real_kernel_variant "$variant" \
    --block 256 \
    --items_per_thread 1 \
    --warmup 10 \
    --iters 200 \
    --validate \
    --csv "$csv_path" \
    > "$stdout_path" 2> "$stderr_path"
}

run_case runtime
run_case specialized

{
  head -n 1 "${run_dir}/real_uniform_depth9_runtime.csv"
  tail -n 1 "${run_dir}/real_uniform_depth9_runtime.csv"
  tail -n 1 "${run_dir}/real_uniform_depth9_specialized.csv"
} > "${run_dir}/comparison.csv"

python scripts/summarize_exp3_real_ab.py "$run_dir" > "${run_dir}/summary.txt"
