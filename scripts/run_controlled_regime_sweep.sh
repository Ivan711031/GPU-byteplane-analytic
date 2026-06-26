#!/bin/bash
# run_controlled_regime_sweep.sh: Controlled Synthetic Regime Sweep
#
# Executes the bound-usefulness sweep from PRD 2026-06-05.
# CPU evaluator; GPU node used only for H200 validation & compute environment.
#
# Usage:
#   sbatch --account=<ACCOUNT> scripts/run_controlled_regime_sweep.sh [--mode full|smoke]
#
# Modes:
#   full   - Complete synthetic matrix (4 families × 4 fractions × 4 densities × 4 planes × 3 seeds)
#            + real anchors (cesm_atm_cloud, hurricane_u)
#            Estimated: ~768 synthetic + 24 anchor cells
#   smoke  - Tiny smoke test: sensor, 2 fractions, 2 densities, 2 planes, 1 seed
#            Estimated: 8 cells

#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -o results/reliability_layer1/phase3/controlled_regime_sweep/job_%j/run_%j.out
#SBATCH -e results/reliability_layer1/phase3/controlled_regime_sweep/job_%j/run_%j.err

set -euo pipefail

# ============================================================================
# Environment
# ============================================================================
module purge
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan

# Hardware validation: must be H200
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
if [[ "$GPU_NAME" != *"H200"* ]]; then
    echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
    exit 2
fi
echo "GPU validated: $GPU_NAME"

ROOT_DIR="${RELIABILITY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"
echo "ROOT_DIR=$ROOT_DIR"

# ============================================================================
# Args
# ============================================================================
MODE="${1:-full}"

echo "=== Controlled Regime Sweep ==="
echo "Mode: $MODE"
echo "GPU: $GPU_NAME"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "---"

# ============================================================================
# Run
# ============================================================================
time python3 scripts/run_controlled_regime_sweep.py --mode "$MODE"

echo "=== Sweep complete ==="
