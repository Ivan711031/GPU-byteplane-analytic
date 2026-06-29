#!/bin/bash
#SBATCH -J z0d_integrity_smoke
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=END,FAIL
set -euo pipefail
echo "=== Phase 3-Z0d Cheap Integrity Digest Smoke ==="
echo "Job ID: $SLURM_JOB_ID"
date
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME"
if ! echo "$GPU_NAME" | grep -qi "H200"; then echo "FATAL: Expected H200"; exit 2; fi
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
ml load cmake/4.0.0
conda activate gpu-byteplane-scan
BUILD_DIR="${WORK_DIR}/builds/z0d_${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake ${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/benchmarks/experiment4_filter_aggregate \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc) bench_z0d_integrity_derisk
DATASET_DIR="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"
RESULTS_DIR="${WORK_DIR}/results/reliability_layer1/phase3/phase3z_z0d/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
CSV_PATH="$RESULTS_DIR/z0d_integrity_profile.csv"
echo "=== Running Z0d integrity microbenchmark ==="
./bench_z0d_integrity_derisk --dataset "$DATASET_DIR" --csv "$CSV_PATH"
echo "=== Raw results ==="
cat "$CSV_PATH"
echo ""
echo "=== Derived metrics ==="
python3 << 'PYEOF'
import csv, os, math
csv_path = os.environ.get("CSV_PATH", "")
rows = []
with open(csv_path) as f:
    for line in f:
        if line.startswith("#") or not line.strip(): continue
        parts = line.strip().split(",")
        if len(parts) >= 5:
            rows.append({"path": parts[0], "label": parts[1], "ms": float(parts[2])})
baseline = next((r["ms"] for r in rows if r["path"] == "A"), None)
eager = next((r["ms"] for r in rows if r["path"] == "D"), None)
B0 = 0.8702
print(f"  A baseline:           {baseline:.6f} ms" if baseline else "  A: N/A")
print(f"  D eager vote:         {eager:.6f} ms" if eager else "  D: N/A")
if baseline and eager:
    print(f"  D/A inflation:        {eager/baseline:.4f}  (tau=1.10)  {'PASS' if eager/baseline<=1.10 else 'FAIL'}")
for prefix, name in [("B","xor64"), ("B2","sum32"), ("B3","combined")]:
    r = next((r for r in rows if r["path"] == prefix), None)
    if r and baseline:
        oh = (r["ms"] - baseline) / baseline
        lt_b0 = r["ms"] < B0
        print(f"  {prefix} {name:20s} {r['ms']:.6f} ms  overhead={oh:.4f}  "
              f"<B0={lt_b0}  {'PASS' if oh<=0.10 else 'FAIL'}")
for r in rows:
    if r["path"] == "C" and baseline:
        oh = (r["ms"] - baseline) / baseline
        lt_b0 = r["ms"] < B0
        print(f"  C {r['label']:30s} {r['ms']:.6f} ms  overhead={oh:.4f}  "
              f"<B0={lt_b0}")
print(f"\n=== BW Gate Check ===")
print(f"B0 reference: {B0} ms (raw FP64)")
print(f"Gate: amortized_lazy < B0 AND crc_only_overhead <= 0.10 AND lazy_inflation <= 1.10")
PYEOF
echo ""
echo "=== Z0d smoke complete ==="
date
