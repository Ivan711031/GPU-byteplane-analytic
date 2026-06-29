#!/usr/bin/env bash
#SBATCH --job-name=exp3_export_dev
#SBATCH --partition=ngs8g
#SBATCH --account=gov108018
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=exp3_export_dev_%j.log
#SBATCH --error=exp3_export_dev_%j.err
set -euo pipefail
DEFAULT_ROOT_DIR="${PROJ_DIR}/workspace/gpu-byteplane-scan-experiments"
is_repo_root() {
  local candidate="$1"
  [[ -n "$candidate" &&
     -d "$candidate/benchmarks/experiment3" &&
     -x "$candidate/scripts/run_exp3_export_dev.sh" ]]
}
ROOT_DIR="${EXP3_ROOT_DIR:-}"
if ! is_repo_root "$ROOT_DIR"; then
  ROOT_DIR=""
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && is_repo_root "$SLURM_SUBMIT_DIR"; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
  else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if is_repo_root "$script_dir"; then
      ROOT_DIR="$script_dir"
    elif is_repo_root "$DEFAULT_ROOT_DIR"; then
      ROOT_DIR="$DEFAULT_ROOT_DIR"
    else
      echo "could not locate repo root; set EXP3_ROOT_DIR explicitly" >&2
      exit 1
    fi
  fi
fi
if command -v module >/dev/null 2>&1; then
  module purge
fi
cd "$ROOT_DIR"
"$ROOT_DIR/scripts/run_exp3_export_dev.sh" "$@"
