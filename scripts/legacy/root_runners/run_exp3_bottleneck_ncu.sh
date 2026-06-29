#!/usr/bin/env bash
#SBATCH --job-name=exp3_bneck_ncu
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=exp3_bneck_ncu_%j.out
#SBATCH --error=exp3_bneck_ncu_%j.err
set -euo pipefail
DEFAULT_ROOT_DIR="${PROJ_DIR}/workspace/gpu-byteplane-scan-experiments"
RESULTS_BASE_REL="results/exp3_bottleneck_attribution"
REAL_DATASET="uniform"
REAL_ENCODED_ROOT="${WORK_DIR}/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p/uniform_p10"
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
if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found after environment bootstrap" >&2
  exit 1
fi
timestamp="$(date +%Y%m%d_%H%M%S)"
job_id="${SLURM_JOB_ID:-nojob}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
run_dir="${ROOT_DIR}/${RESULTS_BASE_REL}/run_${timestamp}_job${job_id}_H200"
mkdir -p "$run_dir"
require_h200
cd "$ROOT_DIR"
cmake -S benchmarks/experiment3 -B build/exp3 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp3 -j
{
  echo "timestamp=${timestamp}"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_job_name=${SLURM_JOB_NAME:-}"
  echo "slurm_job_partition=${SLURM_JOB_PARTITION:-}"
  echo "hostname=$(hostname)"
  echo "gpu_name=${gpu_name}"
  echo "dataset=${REAL_DATASET}"
  echo "encoded_root=${REAL_ENCODED_ROOT}"
  echo "refine_depth=${REAL_REFINE_DEPTH}"
  echo "git_commit=$(git rev-parse HEAD || true)"
  echo "git_status_begin"
  git status --short || true
  echo "git_status_end"
} > "${run_dir}/run_meta.txt"
python scripts/extract_exp3_bottleneck_ncu.py \
  --report results/exp3/ncu_depth7.ncu-rep \
  --label synthetic_depth7 \
  --csv-out "${run_dir}/synthetic_depth7_summary.csv" \
  --json-out "${run_dir}/synthetic_depth7_summary.json" \
  > "${run_dir}/synthetic_depth7_summary.stdout.json"
ncu --set full \
  --target-processes all \
  --import-source yes \
  --source-folders "$PWD" \
  --force-overwrite true \
  --export "${run_dir}/real_uniform_depth9" \
  ./build/exp3/bench_progressive_aggregation \
    --device 0 \
    --mode encoded_dev_subcolumns \
    --encoded-root "${REAL_ENCODED_ROOT}" \
    --refine_min "${REAL_REFINE_DEPTH}" \
    --refine_max "${REAL_REFINE_DEPTH}" \
    --load_strategy rowpack16 \
    --block 256 \
    --items_per_thread 1 \
    --warmup 0 \
    --iters 1 \
    --csv "${run_dir}/real_uniform_depth9_bench.csv"
python scripts/extract_exp3_bottleneck_ncu.py \
  --report "${run_dir}/real_uniform_depth9.ncu-rep" \
  --label real_uniform_depth9 \
  --csv-out "${run_dir}/real_uniform_depth9_summary.csv" \
  --json-out "${run_dir}/real_uniform_depth9_summary.json" \
  > "${run_dir}/real_uniform_depth9_summary.stdout.json"
python - <<'PY' "${run_dir}"
import csv
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
paths = [
    run_dir / "synthetic_depth7_summary.csv",
    run_dir / "real_uniform_depth9_summary.csv",
]
rows = []
fieldnames = None
for path in paths:
    with path.open() as f:
        reader = csv.DictReader(f)
        row = next(reader)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        rows.append(row)
with (run_dir / "comparison_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
