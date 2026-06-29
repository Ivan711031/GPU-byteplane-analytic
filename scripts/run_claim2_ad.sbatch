#!/bin/bash
#SBATCH -J claim2_ad
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
PYTHON="${PROJ_DIR}/.conda/envs/gpu-byteplane-scan/bin/python3"
PROJ_ROOT="${PROJ_DIR}/workspace/gpu-byteplane-claim2-diversity"
cd "$PROJ_ROOT"
# Targeted rerun: Suite A and D only, 2 datasets, 10 seeds, 7 rates
$PYTHON scripts/run_claim2_pilot.py \
    --datasets hurricane_u cesm_atm_cloud \
    --n-rows 500000 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --suite suite_a,suite_d \
    --output-dir results/reliability_layer1/phase3/claim2_ad_rerun
echo "Suite A/D rerun complete."
