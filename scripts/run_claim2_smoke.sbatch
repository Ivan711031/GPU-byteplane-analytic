#!/bin/bash
#SBATCH -J claim2_smoke
#SBATCH -p dev
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

set -euo pipefail

ml purge
ml load miniconda3
PYTHON="/home/u4063895/.conda/envs/gpu-byteplane-scan/bin/python3"

PROJ_ROOT="/home/u4063895/workspace/gpu-byteplane-claim2-diversity"
cd "$PROJ_ROOT"

# Smoke: 1 dataset, 2 seeds, 1 rate, all suites
$PYTHON scripts/run_claim2_pilot.py \
    --datasets hurricane_u \
    --n-rows 500000 \
    --seeds 0 1 \
    --rates 1e-06 \
    --output-dir results/reliability_layer1/phase3/claim2_smoke

echo "Smoke complete."
