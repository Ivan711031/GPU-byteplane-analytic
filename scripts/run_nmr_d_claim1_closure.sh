#!/bin/bash
#SBATCH -J nmr_d_claim1
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=== NMR-D: Claim-1 Closure (Graded vs Uniform Repair) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME"
if ! echo "$GPU_NAME" | grep -qi "H200"; then
  echo "FATAL: Expected H200, got $GPU_NAME"
  exit 2
fi

ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
conda activate gpu-byteplane-scan

echo "=== Toolchain ==="
which python3
python3 --version

PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
SCRIPT_DIR="$PROJECT_DIR/scripts"
RESULTS_DIR="$PROJECT_DIR/results/reliability_layer1/phase4/nmr_d_claim1_closure/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "Project dir: $PROJECT_DIR"
echo "Results dir: $RESULTS_DIR"

echo
echo "=== Smoke: deterministic + stochastic, 1 dataset, 500k rows ==="
python3 "$SCRIPT_DIR/phase4_nmr_d_claim1_evaluator.py" \
  --mode full \
  --dataset cesm_atm_cloud \
  --seeds 5 \
  --n-rows 500000

echo
echo "=== Coverage manifest ==="
COV="$RESULTS_DIR/nmr_d_coverage_manifest.csv"
if [ -f "$COV" ]; then
  python3 -c "
import csv
with open('$COV') as f:
    rows = list(csv.DictReader(f))
fails = [(r['check_id'], r['observed'], r['required']) for r in rows if r['pass'] == 'false']
print(f'Coverage: {len(fails)} failures')
for cid, obs, req in fails:
    print(f'  {cid}: observed={obs} < required={req}')
if not fails:
    print('All coverage gates pass')
"
fi

cat > "$RESULTS_DIR/job_marker.json" <<ENDJSON
{
  "job_id": "$SLURM_JOB_ID",
  "experiment": "nmr_d_claim1_closure",
  "status": "complete",
  "branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
  "commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "date": "$(date -Iseconds)",
  "project_dir": "$PROJECT_DIR",
  "results_dir": "$RESULTS_DIR"
}
ENDJSON

echo
echo "=== NMR-D Claim-1 Complete ==="
echo "Results: $RESULTS_DIR"
date
