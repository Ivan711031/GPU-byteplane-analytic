#!/bin/bash
#SBATCH -J claim2_headline
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

$PYTHON scripts/run_claim2_headline.py \
    --datasets hurricane_u cesm_atm_cloud \
    --headline-rates 1e-06 1e-05 3e-05 \
    --seeds 0 1 2 3 4 5 6 7 \
    --output-dir results/reliability_layer1/phase3/claim2_headline

echo "Headline matrix complete."
