#!/bin/bash
#SBATCH -J ext_e3_bound_tightening
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=END,FAIL
set -euo pipefail
echo "=== Phase 3-Z Extension E3: Bound Tightening Smoke ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date
ml purge
ml load miniconda3/26.1.1
conda activate gpu-byteplane-scan
SCRIPT_DIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/scripts"
RESULTS_DIR="${WORK_DIR}/results/reliability_layer1/phase3/phase3z_ext/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
CESM_DIR="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"
HURR_DIR="${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg4096"
cd "$SCRIPT_DIR"
echo ""
echo "=== E3: cesm_atm_cloud @ offset=24756224 (non-zero cloud region) ==="
python3 phase3z_ext_bound_tightening.py \
    --artifact-dir "$CESM_DIR" \
    --dataset cesm_atm_cloud \
    --n-rows 1000000 \
    --scale 1 \
    --threshold 100.0 \
    --offset 24756224
echo ""
echo "=== E3: hurricane_u (control, no offset) ==="
python3 phase3z_ext_bound_tightening.py \
    --artifact-dir "$HURR_DIR" \
    --dataset hurricane_u \
    --n-rows 1000000 \
    --scale 1 \
    --threshold 50.0
echo ""
echo "=== Results ==="
cat "$RESULTS_DIR/e3_bound_tightening.csv"
echo ""
echo "=== E3 smoke complete ==="
date
