#!/usr/bin/env bash
#SBATCH --job-name=exp3_profdbg
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=exp3_profdbg_%j.out
#SBATCH --error=exp3_profdbg_%j.err

set -euo pipefail

DEFAULT_ROOT_DIR="/home/u4063895/workspace/gpu-byteplane-scan-experiments"
RESULTS_BASE_REL="results/exp3_profiler_debug"
REAL_ENCODED_ROOT="/work/u4063895/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p/uniform_p10"

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

{
  echo "timestamp=${timestamp}"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_job_name=${SLURM_JOB_NAME:-}"
  echo "slurm_job_partition=${SLURM_JOB_PARTITION:-}"
  echo "hostname=$(hostname)"
  echo "gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  echo "ncu_path=$(command -v ncu || true)"
  echo "nsys_path=$(command -v nsys || true)"
  echo "ncu_version_begin"
  ncu --version || true
  echo "ncu_version_end"
  echo "nsys_version_begin"
  nsys --version || true
  echo "nsys_version_end"
} > "${run_dir}/run_meta.txt"

cmake -S benchmarks/experiment3 -B build/exp3 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp3 -j

set +e

ncu --set full \
  --target-processes all \
  --import-source yes \
  --source-folders "$PWD" \
  --force-overwrite true \
  --export "${run_dir}/synthetic_depth7_current" \
  ./build/exp3/bench_progressive_aggregation \
    --device 0 \
    --n 100000000 \
    --segment_rows 4096 \
    --subcolumns 8 \
    --refine_min 7 \
    --refine_max 7 \
    --load_strategy rowpack16 \
    --block 256 \
    --items_per_thread 1 \
    --warmup 0 \
    --iters 1 \
    --csv "${run_dir}/synthetic_depth7_current_bench.csv" \
  > "${run_dir}/synthetic_ncu.stdout" 2> "${run_dir}/synthetic_ncu.stderr"
synthetic_ncu_status=$?

./build/exp3/bench_progressive_aggregation \
  --device 0 \
  --mode encoded_dev_subcolumns \
  --encoded-root "${REAL_ENCODED_ROOT}" \
  --refine_min 9 \
  --refine_max 9 \
  --load_strategy rowpack16 \
  --block 256 \
  --items_per_thread 1 \
  --warmup 0 \
  --iters 1 \
  --csv "${run_dir}/real_uniform_depth9_direct.csv" \
  > "${run_dir}/real_direct.stdout" 2> "${run_dir}/real_direct.stderr"
real_direct_status=$?

nsys profile \
  --trace=cuda \
  --sample=none \
  --force-overwrite=true \
  --output "${run_dir}/real_uniform_depth9_nsys" \
  ./build/exp3/bench_progressive_aggregation \
    --device 0 \
    --mode encoded_dev_subcolumns \
    --encoded-root "${REAL_ENCODED_ROOT}" \
    --refine_min 9 \
    --refine_max 9 \
    --load_strategy rowpack16 \
    --block 256 \
    --items_per_thread 1 \
    --warmup 0 \
    --iters 1 \
    --csv "${run_dir}/real_uniform_depth9_nsys_bench.csv" \
  > "${run_dir}/real_nsys.stdout" 2> "${run_dir}/real_nsys.stderr"
real_nsys_status=$?

nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum \
  --format csv \
  "${run_dir}/real_uniform_depth9_nsys.nsys-rep" \
  > "${run_dir}/real_uniform_depth9_nsys_stats.csv" 2> "${run_dir}/real_uniform_depth9_nsys_stats.stderr"
real_nsys_stats_status=$?

set -e

{
  echo "synthetic_ncu_status=${synthetic_ncu_status}"
  echo "real_direct_status=${real_direct_status}"
  echo "real_nsys_status=${real_nsys_status}"
  echo "real_nsys_stats_status=${real_nsys_stats_status}"
} >> "${run_dir}/run_meta.txt"

exit 0
