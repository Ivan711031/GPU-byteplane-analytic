#!/usr/bin/env bash
#SBATCH --job-name=exp3_real_sum
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --array=0-3%1
#SBATCH --output=exp3_real_sum_%A_%a.log
#SBATCH --error=exp3_real_sum_%A_%a.err

set -euo pipefail

DEFAULT_ROOT_DIR="/home/u4063895/workspace/gpu-byteplane-scan-experiments"

is_repo_root() {
  local candidate="$1"
  [[ -n "$candidate" &&
     -f "$candidate/scripts/run_exp3.sh" &&
     -d "$candidate/benchmarks/experiment3" ]]
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

case "${SLURM_ARRAY_TASK_ID:-}" in
  0)
    dataset="heavy_tailed"
    artifact_label="p6"
    max_plane_count="8"
    ;;
  1)
    dataset="sensor"
    artifact_label="p10"
    max_plane_count="5"
    ;;
  2)
    dataset="uniform"
    artifact_label="p10"
    max_plane_count="6"
    ;;
  3)
    dataset="zipfian"
    artifact_label="p8"
    max_plane_count="8"
    ;;
  *)
    echo "unsupported SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-unset}" >&2
    exit 2
    ;;
esac

refine_max="$((max_plane_count - 1))"

export MODE="encoded_dev_subcolumns"
export REAL_KERNEL_VARIANT="specialized"
export ENCODED_ROOT="/work/u4063895/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p/${dataset}_${artifact_label}"
export REFINE_MIN="0"
export REFINE_MAX="$refine_max"
export VALIDATE="1"
export WARMUP="10"
export ITERS="200"
export RESULTS_BASE="${ROOT_DIR}/results/exp3_real_specialized_sum"
export CSV_NAME="throughput.csv"
export RUN_DESC="Specialized real-data SUM throughput on ${dataset}; sweep k=1..${max_plane_count}."

cd "$ROOT_DIR"
exec "$ROOT_DIR/scripts/run_exp3.sh"
