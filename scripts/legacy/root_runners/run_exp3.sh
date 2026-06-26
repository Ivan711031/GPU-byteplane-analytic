#!/usr/bin/env bash
#SBATCH --job-name=exp3_runner
#SBATCH --partition=dev
#SBATCH --account=gov108018
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=exp3_runner_%j.log
#SBATCH --error=exp3_runner_%j.err

set -euo pipefail

DEFAULT_ROOT_DIR="/home/ccres1995/workspace_hanyin/gpu-byteplane-scan-experiments"

is_exp3_root() {
  local candidate="$1"
  [[ -n "$candidate" &&
     -d "$candidate/benchmarks/experiment3" &&
     -x "$candidate/scripts/run_exp3.sh" ]]
}

print_toolchain_info() {
  which nvcc
  cmake --version | head -n 1
  gcc --version | head -n 1
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
if ! is_exp3_root "$ROOT_DIR"; then
  ROOT_DIR=""
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && is_exp3_root "$SLURM_SUBMIT_DIR"; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
  else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if is_exp3_root "$script_dir"; then
      ROOT_DIR="$script_dir"
    elif is_exp3_root "$DEFAULT_ROOT_DIR"; then
      ROOT_DIR="$DEFAULT_ROOT_DIR"
    else
      echo "could not locate Exp3 repo root; set EXP3_ROOT_DIR explicitly" >&2
      exit 1
    fi
  fi
fi

CONDA_ROOT="${CONDA_ROOT:-}"
CUDA_ROOT="${CUDA_ROOT:-/work/envstack/apps/cuda/12.6}"

if command -v module >/dev/null 2>&1; then
  module purge
  module load miniconda3/26.1.1 || true
  module load cuda/12.6 || true
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -n "$CONDA_ROOT" && -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda bootstrap not found; set CONDA_ROOT or load miniconda3/26.1.1" >&2
  exit 1
fi

conda activate gpu-byteplane-scan

if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_ROOT="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  else
    echo "CUDA bootstrap not found at: $CUDA_ROOT/bin/nvcc" >&2
    exit 1
  fi
fi

export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="$CUDA_ROOT"
export CUDACXX="$CUDA_ROOT/bin/nvcc"
export CC="${EXP3_HOST_CC:-/usr/bin/gcc}"
export CXX="${EXP3_HOST_CXX:-/usr/bin/g++}"

if [[ ! -x "$CC" || ! -x "$CXX" ]]; then
  echo "host compiler not found: CC=$CC CXX=$CXX" >&2
  exit 1
fi

print_toolchain_info
require_h200

cd "$ROOT_DIR"
"$ROOT_DIR/scripts/run_exp3.sh" "$@"
