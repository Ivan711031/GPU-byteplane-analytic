#!/bin/bash
#SBATCH -J g2_replication
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/u4063895/workspace/gpu-byteplane-reliability-nmr/slurm-%j.out

set -euo pipefail

echo "=== Phase 3-Z G2: Replication Run ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME"
if ! echo "$GPU_NAME" | grep -qi "H200"; then echo "FATAL: Expected H200"; exit 2; fi

ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
conda activate gpu-byteplane-scan

SCRIPT_DIR="/home/u4063895/workspace/gpu-byteplane-reliability-nmr/scripts"
RESULTS_DIR="/home/u4063895/workspace/gpu-byteplane-reliability-nmr/results/reliability_layer1/phase3/phase3z_ext/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
cd "$SCRIPT_DIR"

HURR_DIR="/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg4096"

# R-vector configurations
RV_DEFAULT="1,1,1,1,1,1,1,1"   # r=[1]*8 (baseline, no replication)
RV_GRADED="3,2,1,1,1,1,1,1"    # graded replication (plane 0 ×3, plane 1 ×2)
RV_UNIFORM="2,2,2,2,2,2,2,2"   # uniform 2× (same total storage as graded)

for RV_LABEL in "default" "graded" "uniform"; do
    case "$RV_LABEL" in
        default)  RV_ARGS="--rv 1 1 1 1 1 1 1 1"; RV_STR="1,1,1,1,1,1,1,1" ;;
        graded)   RV_ARGS="--rv 3 2 1 1 1 1 1 1"; RV_STR="3,2,1,1,1,1,1,1" ;;
        uniform)  RV_ARGS="--rv 2 2 2 2 2 2 2 2"; RV_STR="2,2,2,2,2,2,2,2" ;;
    esac

    echo ""
    echo "=== G2: hurricane_u  RV=$RV_LABEL ($RV_STR) ==="
    python3 phase3z_ext_digest_evaluator.py \
        --artifact-dir "$HURR_DIR" \
        --dataset hurricane_u \
        --n-rows 1000000 \
        --fault-rates 1e-06 \
        --seeds 0 1 2 \
        --variants sum32 pos_weighted fletcher_like \
        $RV_ARGS \
        2>&1 | tee -a "$RESULTS_DIR/g2_${RV_LABEL}.log"
done

# Copy escape CSVs (from scripts/ CWD)
cp results/reliability_layer1/phase3/phase3z_ext/job_${SLURM_JOB_ID}/e1_digest_sweep_escape.csv "$RESULTS_DIR/" 2>/dev/null || true

echo ""
echo "=== G2 Analysis ==="
python3 -c "
import csv, sys
from collections import defaultdict

for rv_label, rv_str in [('default','1,1,1,1,1,1,1,1'),('graded','3,2,1,1,1,1,1,1'),('uniform','2,2,2,2,2,2,2,2')]:
    base = '$RESULTS_DIR'
    ecsv = f'{base}/e1_digest_sweep_escape.csv'
    path2 = f'results/reliability_layer1/phase3/phase3z_ext/job_${SLURM_JOB_ID}/e1_digest_sweep_escape.csv'
    path = ecsv if __import__('os').path.exists(ecsv) else path2
    rows = [r for r in csv.DictReader(open(path)) if r['r_vector'] == rv_str and r['dataset'] == 'hurricane_u']
    groups = defaultdict(list)
    for r in rows:
        groups[(r['variant'], r['fault_mode'])].append(r)
    print(f'\\n=== RV={rv_label} ({rv_str}) ===')
    print(f'  {\"Variant\":<18s} {\"Mode\":<25s} {\"escaped\":>10s} {\"rate\":>10s}')
    print('  ' + '-'*65)
    for key in sorted(groups):
        var, mode = key
        g = groups[key]
        ne = sum(int(r['n_escaped']) for r in g)
        ni = sum(int(r['n_injected']) for r in g)
        rate = ne/max(ni,1)
        print(f'  {var:<18s} {mode:<25s} {ne:>3d}/{ni:<3d}   {rate:.6f}')
    storage = sum(int(x) for x in rv_str.split(',') if x)
    print(f'  Storage: {storage} replicas total')
"

echo ""
echo "=== G2 complete ==="
date
