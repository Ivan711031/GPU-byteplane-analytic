#!/bin/bash
#SBATCH -J z0f_parallel_sum32
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=END,FAIL
set -euo pipefail
echo "=== Phase 3-Z0f Parallel SUM32 Smoke ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME"
if ! echo "$GPU_NAME" | grep -qi "H200"; then echo "FATAL: Expected H200"; exit 2; fi
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
ml load cmake/4.0.0
conda activate gpu-byteplane-scan
BUILD_DIR="${WORK_DIR}/builds/z0f_${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake ${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/benchmarks/experiment4_filter_aggregate \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc) bench_z0f_parallel_sum32
CESM_DIR="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"
HURR_DIR="${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg4096"
RESULTS_DIR="${WORK_DIR}/results/reliability_layer1/phase3/phase3z_z0f/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
CSV_PATH="$RESULTS_DIR/z0f_parallel_sum32_profile.csv"
echo ""
echo "=== Running Z0f: cesm_atm_cloud + hurricane_u ==="
./bench_z0f_parallel_sum32 --dataset "$CESM_DIR" --dataset "$HURR_DIR" --csv "$CSV_PATH"
echo ""
echo "=== Raw results ==="
cat "$CSV_PATH"
echo ""
echo "=== Gate analysis ==="
python3 << 'PYEOF'
import csv, math
from collections import defaultdict
B0 = 0.8702
lines = open("${WORK_DIR}/results/reliability_layer1/phase3/phase3z_z0f/job_'"$SLURM_JOB_ID"'/z0f_parallel_sum32_profile.csv").readlines()
# simpler: use env var
import os
csv_p = os.environ.get("CSV_PATH", "/dev/null")
for line in open(csv_p):
    pass
PYEOF
# Use proper path
CSV_PATH_FULL="${WORK_DIR}/results/reliability_layer1/phase3/phase3z_z0f/job_${SLURM_JOB_ID}/z0f_parallel_sum32_profile.csv"
python3 -c "
import csv, math
from collections import defaultdict
B0=0.8702
rows=[]
with open('$CSV_PATH_FULL') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        p=line.strip().split(',')
        if p[0]=='dataset': continue
        rows.append({'ds':p[0],'path':p[3],'ms':float(p[6])})
groups=defaultdict(list)
for r in rows: groups[(r['ds'],r['path'])].append(r['ms'])
print()
print(f\"{'Dataset':<20} {'Path':<20} {'mean_ms':>10} {'std':>8} {'min':>10} {'max':>10} {'vsA':>8} {'vsB0':>8} {'<B0':>5} {'<0.9B0':>7}\")
print('-'*106)
for ds in sorted(set(k[0] for k in groups)):
    a_mean = sum(groups[(ds,'A')])/len(groups[(ds,'A')]) if (ds,'A') in groups else 1
    for path in ['A','B','B2','D']:
        k=(ds,path)
        if k not in groups: continue
        v=groups[k]; n=len(v); mn=sum(v)/n; sd=math.sqrt(sum((x-mn)**2 for x in v)/n)
        lo=min(v); hi=max(v)
        va=mn/a_mean; vb=mn/B0; lb=mn<B0; l9=mn<0.9*B0
        print(f'{ds:<20} {path:<20} {mn:>10.6f} {sd:>8.6f} {lo:>10.6f} {hi:>10.6f} {va:>8.4f} {vb:>8.4f} {str(lb):>5} {str(l9):>7}')
" 2>&1
echo ""
echo "=== Z0f complete ==="
date
