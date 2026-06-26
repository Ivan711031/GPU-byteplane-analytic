#!/bin/bash
#SBATCH --job-name=nmr-c1-realistic-pilot
#SBATCH --partition=ngs32g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/nmr_c1_realistic_pilot_%j.out
#SBATCH --error=logs/nmr_c1_realistic_pilot_%j.err
set -euo pipefail

# Load environment
ml miniconda3/26.1.1
conda activate gpu-byteplane-scan

echo "=== Job info ==="
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "DATE=$(date -Is)"
echo "PWD=$(pwd)"

mkdir -p logs

bash scripts/run_nmr_claim1_realistic_pilot.sh
