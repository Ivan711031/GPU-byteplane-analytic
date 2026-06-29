#!/bin/bash
#SBATCH -J z0e_sum32_sweep
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-02:00:00
#SBATCH --mail-type=END,FAIL
set -euo pipefail
echo "=== Phase 3-Z0e SUM32 Allocation-Unit Sweep ==="
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
BUILD_DIR="${WORK_DIR}/builds/z0e_${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake ${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/benchmarks/experiment4_filter_aggregate \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc) bench_z0e_sum32_sweep
CESM_DIR="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"
HURR_DIR="${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg4096"
RESULTS_DIR="${WORK_DIR}/results/reliability_layer1/phase3/phase3z_z0e/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
CSV_PATH="$RESULTS_DIR/z0e_sum32_sweep.csv"
echo ""
echo "=== Running Z0e: cesm_atm_cloud ==="
./bench_z0e_sum32_sweep \
    --dataset "$CESM_DIR" \
    --dataset "$HURR_DIR" \
    --csv "$CSV_PATH"
echo ""
echo "=== Raw results ==="
cat "$CSV_PATH"
echo ""
echo "=== Gate analysis ==="
python3 << 'PYEOF'
import csv, os, math
from collections import defaultdict
csv_path = "'"$CSV_PATH"'"
csv_path = csv_path.strip("'")
rows = []
with open(csv_path) as f:
    for line in f:
        if line.startswith("#") or not line.strip(): continue
        parts = line.strip().split(",")
        if len(parts) >= 7:
            rows.append({
                "ds": parts[0], "n_units": int(parts[1]),
                "alloc": int(parts[2]), "path": parts[3],
                "ms": float(parts[6]),
            })
B0 = 0.8702
groups = defaultdict(list)
for r in rows:
    key = (r["ds"], r["alloc"], r["path"])
    groups[key].append(r["ms"])
stats = {}
for key, vals in groups.items():
    vals.sort()
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean)**2 for v in vals) / n
    stats[key] = {
        "mean": mean, "std": math.sqrt(var),
        "min": vals[0], "max": vals[-1], "n": n
    }
print(f"{'Dataset':<20} {'Alloc':>8} {'A_mean':>10} {'B_mean':>10} {'D_mean':>10} {'B/A':>8} {'B/B0':>8} {'B<B0':>6} {'D/A':>8} {'D<B0':>6}")
print("-" * 110)
datasets = sorted(set(k[0] for k in stats.keys()))
for ds in datasets:
    for au in [4096, 16384, 65536, 262144]:
        ka = (ds, au, "A")
        kb = (ds, au, "B")
        kd = (ds, au, "D")
        if ka not in stats or kb not in stats: continue
        a = stats[ka]; b = stats[kb]; d = stats.get(kd)
        ba = b["mean"] / a["mean"] if a["mean"] > 0 else 0
        bb0 = b["mean"] / B0
        b_lt_b0 = "P" if b["mean"] < B0 else "F"
        da_str = f"{d['mean']/a['mean']:.4f}" if d else "N/A"
        d_b0 = "P" if d and d["mean"] < B0 else "N/A"
        print(f"{ds:<20} {au:>8} {a['mean']:>10.6f} {b['mean']:>10.6f} {d['mean'] if d else 0:>10.6f} {ba:>8.4f} {bb0:>8.4f} {b_lt_b0:>6} {da_str:>8} {d_b0:>6}")
print(f"\n=== BW Gate (0.9 * B0 = {0.9*B0:.4f} ms) ===")
for ds in datasets:
    for au in [4096, 16384, 65536, 262144]:
        kb = (ds, au, "B")
        if kb not in stats: continue
        b = stats[kb]
        margin = b["mean"] < 0.9 * B0
        stable = b["max"] < B0
        print(f"  {ds:20s} au={au:>6}:  B={b['mean']:.4f}+-{b['std']:.4f}  [min={b['min']:.4f}, max={b['max']:.4f}]  "
              f"<0.9*B0={margin}  stable<B0={stable}")
PYEOF
echo ""
echo "=== Z0e complete ==="
date
