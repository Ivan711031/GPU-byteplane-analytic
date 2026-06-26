#!/bin/bash
#SBATCH -J z0b_corrected_smoke
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "=== Phase 3-Z0b Corrected Bandwidth Smoke ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

# HW validation
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME"
if ! echo "$GPU_NAME" | grep -qi "H200"; then
    echo "FATAL: Expected H200, got $GPU_NAME"
    exit 2
fi

ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
ml load cmake/4.0.0
conda activate gpu-byteplane-scan

BUILD_DIR="/work/u4063895/builds/z0b_${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake /home/u4063895/workspace/gpu-byteplane-reliability-nmr/benchmarks/experiment4_filter_aggregate \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc) bench_z0b_bandwidth_derisk

echo "=== Build complete ==="

DATASET_DIR="/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"
echo "Using dataset: $DATASET_DIR"

RESULTS_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3z_z0b/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
CSV_PATH="$RESULTS_DIR/z0b_bandwidth_profile.csv"

echo "=== Running Z0b corrected microbenchmark ==="
./bench_z0b_bandwidth_derisk --dataset "$DATASET_DIR" --csv "$CSV_PATH"

echo "=== Raw results ==="
cat "$CSV_PATH"

echo ""
echo "=== Derived metrics ==="
python3 << 'PYEOF'
import csv, math, sys

rows = []
csv_path = "'"$CSV_PATH"'"
with open(csv_path.replace("'","")) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split(",")
        if len(parts) >= 5 and parts[1] in ("A","B","B2","C","D"):
            rows.append({
                "k": int(parts[0]),
                "path": parts[1],
                "label": parts[2],
                "ms_per_iter": float(parts[3]),
            })

B0_MS = 0.8702

for k in sorted(set(r["k"] for r in rows)):
    pa = next((r for r in rows if r["k"] == k and r["path"] == "A"), None)
    pb = next((r for r in rows if r["k"] == k and r["path"] == "B"), None)
    pb2 = next((r for r in rows if r["k"] == k and r["path"] == "B2"), None)
    pd = next((r for r in rows if r["k"] == k and r["path"] == "D"), None)
    pc_rows = [r for r in rows if r["k"] == k and r["path"] == "C"]

    print(f"\n--- k={k} ---")
    print(f"  A (baseline):          {pa['ms_per_iter']:.6f} ms")
    print(f"  B (CRC 1tpu):          {pb['ms_per_iter']:.6f} ms" if pb else "  B: N/A")
    print(f"  B2 (CRC parallel):     {pb2['ms_per_iter']:.6f} ms" if pb2 else "  B2: N/A")
    print(f"  D (eager vote):        {pd['ms_per_iter']:.6f} ms" if pd else "  D: N/A")

    if pa and pb:
        crc_oh = (pb['ms_per_iter'] - pa['ms_per_iter']) / pa['ms_per_iter']
        print(f"  CRC overhead (B):        {crc_oh:.4f}  (tau=0.10)  {'PASS' if crc_oh<=0.10 else 'FAIL'}")

    if pa and pb2:
        crc_oh2 = (pb2['ms_per_iter'] - pa['ms_per_iter']) / pa['ms_per_iter']
        print(f"  CRC overhead (B2):       {crc_oh2:.4f}  (tau=0.10)  {'PASS' if crc_oh2<=0.10 else 'FAIL'}")

    if pa and pd:
        eager_inf = pd['ms_per_iter'] / pa['ms_per_iter']
        print(f"  Eager inflation (D/A):  {eager_inf:.4f}  (tau=1.10)  {'PASS' if eager_inf<=1.10 else 'FAIL'}")

    for pc in pc_rows:
        ms = pc['ms_per_iter']
        if pa:
            c_inf = ms / pa['ms_per_iter']
            lt_b0 = ms < B0_MS
            a_lt_b0 = pa['ms_per_iter'] < B0_MS
            print(f"  {pc['label']:45s} {ms:.6f} ms  C/A={c_inf:.4f}  <B0={lt_b0}  "
                  f"baseline<B0={a_lt_b0}")

print("\n=== BW Gate Check ===")
print(f"B0 reference: {B0_MS} ms (raw FP64, from X3)")
print("NOTE: This smoke uses table CRC. HW CRC (__crc32b) probe is compile-time only.")
print("The gate verdict is NEEDS_FIXES — not final BW_BANDWIDTH_BOUND_REGRESSION —")
print("because optimization (HW CRC, fused pipeline) may change the result.")
PYEOF

echo ""
echo "=== Z0b smoke complete ==="
date
