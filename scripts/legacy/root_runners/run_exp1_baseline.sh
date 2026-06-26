#!/usr/bin/env bash
#SBATCH --job-name=exp1_baseline_runner
#SBATCH --partition=dev
#SBATCH --account=gov108018
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=exp1_baseline_runner_%j.log
#SBATCH --error=exp1_baseline_runner_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

module purge
module load jupyter/miniconda3
module load cuda/12.6

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda not found. Please load miniconda module or fix conda.sh path." >&2
  exit 1
fi

conda activate gpu-byteplane-scan

cd "$ROOT_DIR"
"$ROOT_DIR/scripts/run_exp1_baseline.sh" "$@"
