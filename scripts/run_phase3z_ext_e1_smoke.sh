#!/bin/bash
#SBATCH -J ext_e1_digest_sweep
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "=== Phase 3-Z Extension E1: Digest Upgrade Sweep Smoke ==="
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

SCRIPT_DIR="/home/u4063895/workspace/gpu-byteplane-reliability-nmr/scripts"
BUILD_DIR="/work/u4063895/builds/ext_e1_${SLURM_JOB_ID}"
RESULTS_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3z_ext/job_${SLURM_JOB_ID}"
LOCAL_RESULTS="results/reliability_layer1/phase3/phase3z_ext/job_${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR" "$RESULTS_DIR"

HURR_DIR="/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg4096"
CESM_DIR="/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"

# ── Step 1: Build GPU benchmark ──
cd "$BUILD_DIR"
cmake /home/u4063895/workspace/gpu-byteplane-reliability-nmr/benchmarks/experiment4_filter_aggregate \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc) bench_z_ext_digest_sweep

# ── Step 2: GPU latency benchmark (E1 + E2) ──
echo ""
echo "=== GPU Latency Benchmark ==="
CSV_PATH="$RESULTS_DIR/e2_slack_pareto.csv"
./bench_z_ext_digest_sweep \
    --dataset "$CESM_DIR" \
    --dataset "$HURR_DIR" \
    --csv "$CSV_PATH"

echo ""
echo "=== Raw latency results ==="
cat "$CSV_PATH"

# ── Step 3: CPU escape replay (E1) ──
echo ""
echo "=== CPU Escape Replay (hurricane_u: all variants) ==="
cd "$SCRIPT_DIR"
python3 phase3z_ext_digest_evaluator.py \
    --artifact-dir "$HURR_DIR" \
    --dataset hurricane_u \
    --n-rows 1000000 \
    --fault-rates 1e-06 \
    --seeds 0 1 2

# ── Step 4: Cesm escape (adversarial_cancel may be N/A) ──
echo ""
echo "=== CPU Escape Replay (cesm_atm_cloud: all variants) ==="
python3 phase3z_ext_digest_evaluator.py \
    --artifact-dir "$CESM_DIR" \
    --dataset cesm_atm_cloud \
    --n-rows 1000000 \
    --fault-rates 1e-06 \
    --seeds 0 1 2

# ── Step 5: Escape Analysis ──
echo ""
echo "=== Escape Analysis ==="
python3 -c "
import csv, os
from collections import defaultdict
jid = os.environ.get('SLURM_JOB_ID', 'unknown')
csv_path = f'/home/u4063895/workspace/gpu-byteplane-reliability-nmr/results/reliability_layer1/phase3/phase3z_ext/job_{jid}/e1_digest_sweep_escape.csv'
try:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    print(f'  Escape results: {len(rows)} rows')
    groups = defaultdict(list)
    for r in rows:
        groups[(r['dataset'], r['variant'], r['fault_mode'])].append(r)
    for key in sorted(groups):
        ds, variant, mode = key
        g = groups[key]
        n_escaped = sum(int(r['n_escaped']) for r in g)
        n_injected = sum(int(r['n_injected']) for r in g)
        rate = n_escaped / max(n_injected, 1)
        print(f'  {ds:<20s} {variant:<18s} {mode:<25s} escaped={n_escaped}/{n_injected}  rate={rate:.6f}')
except FileNotFoundError:
    print(f'  No escape CSV yet')
"

# ── Step 6: Latency analysis ──
echo ""
echo "=== Latency vs B0 ==="
python3 -c "
import csv, os
from collections import defaultdict
jid = os.environ.get('SLURM_JOB_ID', 'unknown')
csv_path = f'/work/u4063895/results/reliability_layer1/phase3/phase3z_ext/job_{jid}/e2_slack_pareto.csv'
B0 = 0.8702
rows = []
with open(csv_path) as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        p = line.strip().split(',')
        if p[0] == 'dataset': continue
        rows.append({'ds': p[0], 'path': p[3], 'ms': float(p[6])})
groups = defaultdict(list)
for r in rows:
    groups[(r['ds'], r['path'])].append(r['ms'])
print(f\"{'Dataset':<20s} {'Path':<10s} {'mean_ms':>10s} {'vsB0':>8s} {'<B0':>5s}\")
print('-'*45)
for ds in sorted(set(k[0] for k in groups)):
    for path in ['A','B','B2','C','D','E','F','G','H','I','J']:
        k = (ds, path)
        if k not in groups: continue
        v = groups[k]
        mn = sum(v)/len(v)
        vb = mn/B0
        lb = mn < B0
        lbl = {'A':'A(read)','B':'B(s32ser)','B2':'B2(s32par)','C':'C(s64)','D':'D(d32)','E':'E(pwt)','F':'F(xor)','G':'G(flt)','H':'H(v2)','I':'I(v3)','J':'J(v5)'}.get(path, path)
        print(f'{ds:<20s} {lbl:<10s} {mn:>10.6f} {vb:>8.4f} {str(lb):>5s}')
"

echo ""
echo "=== E1 smoke complete ==="
date
