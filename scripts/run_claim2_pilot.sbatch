#!/bin/bash
#SBATCH -J claim2_pilot
#SBATCH -p dev
#SBATCH -t 0-04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

set -euo pipefail

ml purge
ml load miniconda3
PYTHON="/home/u4063895/.conda/envs/gpu-byteplane-scan/bin/python3"

PROJ_ROOT="/home/u4063895/workspace/gpu-byteplane-claim2-diversity"
cd "$PROJ_ROOT"

# Full pilot matrix: 2 datasets, 10 seeds, 7 rates, all suites, 4 policies
$PYTHON scripts/run_claim2_pilot.py \
    --datasets hurricane_u cesm_atm_cloud \
    --n-rows 500000 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --output-dir results/reliability_layer1/phase3/claim2_pilot

echo "Pilot complete."
