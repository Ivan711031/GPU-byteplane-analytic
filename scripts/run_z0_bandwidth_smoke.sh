#!/bin/bash
#SBATCH -J z0_bandwidth_smoke
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "=== Phase 3-Z0 Bandwidth De-risk Smoke ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

# Hardware validation
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME"
if ! echo "$GPU_NAME" | grep -qi "H200"; then
    echo "FATAL: Expected H200, got $GPU_NAME"
    exit 2
fi

# Environment
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
ml load cmake/4.0.0
conda activate gpu-byteplane-scan

which nvcc
which cmake
gcc --version

# Build
BUILD_DIR="/work/u4063895/builds/z0_${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake /home/u4063895/workspace/gpu-byteplane-reliability-nmr/benchmarks/experiment4_filter_aggregate \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=$(which nvcc)
make -j$(nproc) bench_z0_bandwidth_derisk

echo "=== Build complete ==="

# Find dataset (use locality_sensitivity which has manifest.json)
DATASET_DIR="/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"
if [ -d "$DATASET_DIR" ]; then
    echo "Using CESM-ATM cloud (locality_sensitivity, seg4096)"
else
    # Fall back to scientific dev_buff_v2
    DATASET_DIR="/work/u4063895/datasets/scientific/dev_buff_v2_scientific/cesm_atm_cloud/bfp-dec12"
    echo "Using CESM-ATM cloud (dev_buff_v2)"
fi

# Output
RESULTS_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3z_z0/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
CSV_PATH="$RESULTS_DIR/z0_bandwidth_profile.csv"

echo "=== Running Z0 microbenchmark ==="
./bench_z0_bandwidth_derisk --dataset "$DATASET_DIR" --csv "$CSV_PATH"

echo "=== Results ==="
cat "$CSV_PATH"

# Compute derived metrics
echo ""
echo "=== Derived metrics ==="
python3 << 'PYEOF'
import csv, sys, math

rows = []
with open("'"$CSV_PATH"'") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split(",")
        if len(parts) >= 5 and parts[1] in ("A","B","C","D"):
            rows.append({
                "k": int(parts[0]),
                "path": parts[1],
                "label": parts[2],
                "ms_per_iter": float(parts[3]),
            })

for k in sorted(set(r["k"] for r in rows)):
    path_a = next((r for r in rows if r["k"] == k and r["path"] == "A"), None)
    path_b = next((r for r in rows if r["k"] == k and r["path"] == "B"), None)
    path_d = next((r for r in rows if r["k"] == k and r["path"] == "D"), None)
    path_c_rows = [r for r in rows if r["k"] == k and r["path"] == "C"]

    if not path_a or not path_b: continue

    crc_overhead = (path_b["ms_per_iter"] - path_a["ms_per_iter"]) / path_a["ms_per_iter"] if path_a["ms_per_iter"] > 0 else 0
    eager_inflation = path_d["ms_per_iter"] / path_a["ms_per_iter"] if path_a["ms_per_iter"] > 0 and path_d else 0

    print(f"k={k}:")
    print(f"  path_A (single-replica):   {path_a['ms_per_iter']:.6f} ms")
    print(f"  path_B (CRC common):       {path_b['ms_per_iter']:.6f} ms")
    print(f"  CRC-only overhead:         {crc_overhead:.4f}  (tau_crc=0.10)")
    print(f"  Eager inflation:           {eager_inflation:.4f}  (tau_lazy=1.10)")

    for pc in path_c_rows:
        fr = pc["label"].split("fr")[-1]
        amortized = (1 - float(fr.replace("_","e-").replace("e-0","e-"))) * path_b["ms_per_iter"] + float(fr.replace("_","e-").replace("e-0","e-")) * pc["ms_per_iter"]
        lazy_inflation = amortized / path_a["ms_per_iter"] if path_a["ms_per_iter"] > 0 else 0
        pass_str = "PASS" if lazy_inflation <= 1.10 else "FAIL"
        print(f"  path_C (fr={fr}): {pc['ms_per_iter']:.6f} ms amortized={amortized:.6f} inflation={lazy_inflation:.4f} [{pass_str}]")
PYEOF

echo ""
echo "=== Z0 smoke complete ==="
date
