#!/bin/bash
#SBATCH -J p3x3_pareto
#SBATCH -p dev
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u4063895@gs.ncku.edu.tw

set -euo pipefail

PROJ_DIR="/home/u4063895/workspace/gpu-byteplane-reliability-nmr"
X1_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3x1_accuracy/job_76605"
X2_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3x2_diversity/job_76666"
OUT_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3x3_pareto/job_${SLURM_JOB_ID}"

module purge; ml load miniconda3/26.1.1
conda activate gpu-byteplane-scan

mkdir -p "$OUT_DIR"
python3 "$PROJ_DIR/scripts/build_phase3x3_pareto.py" \
  --x1-dir "$X1_DIR" --x2-dir "$X2_DIR" --out-dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/build.log"

echo "Done: $(date -Iseconds)" | tee -a "$OUT_DIR/build.log"
