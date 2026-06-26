#!/bin/bash
#SBATCH -J p3d_h200_preflight
#SBATCH -p dev
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u4063895@gs.ncku.edu.tw

set -euo pipefail

PROJ_DIR="/home/u4063895/workspace/gpu-byteplane-reliability-nmr"
OUT_DIR="/work/u4063895/results/reliability_layer1/phase3/p3d_h200_preflight/job_${SLURM_JOB_ID}"

# Trap writes handoff marker even on early failure
cleanup() {
    local exit_code=$?
    mkdir -p "$PROJ_DIR/handoff/job_done"
    cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{
  "job_id": ${SLURM_JOB_ID},
  "job_name": "p3d_h200_preflight",
  "exit_status": ${exit_code},
  "finished_at": "$(date -Iseconds)",
  "output_dir": "${OUT_DIR:-unset}",
  "next_action": "Analyze timing results and write P3-D H200 preflight report."
}
EOF
}
trap cleanup EXIT

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date -Iseconds)"

module purge
ml load cmake/4.0.0
ml load cuda/12.6

# Validate H200
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME"
if [[ "$GPU_NAME" != *"H200"* ]]; then
    echo "ERROR: Expected H200, got $GPU_NAME"
    exit 2
fi

cd "$PROJ_DIR"

# Build checksum timing benchmark (H200 uses sm_90 architecture)
BUILD_DIR="benchmarks/experiment0/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j p3d_checksum_timing

cd "$PROJ_DIR"

mkdir -p "$OUT_DIR"

# 1. Checksum timing for each active_byte_len
echo ""
echo "=== P3-D H200 Checksum Timing ==="
echo "" | tee -a "$OUT_DIR/timing_summary.txt"

for L in 2 3 5 6; do
    echo "--- active_byte_len=$L (segment_size=1024, rows=100M) ---"
    ./benchmarks/experiment0/build/p3d_checksum_timing \
        --rows 100000000 \
        --active-byte-len "$L" \
        --segment-size 1024 \
        --warmup 5 \
        --iters 20 \
        2>&1 | tee "$OUT_DIR/timing_L${L}.txt"
    echo "" | tee -a "$OUT_DIR/timing_summary.txt"
done

# 2. Raw FP64 timing for speed_margin_H200 baseline
echo "=== Raw FP64 HBM throughput (speed_margin baseline) ==="
./benchmarks/experiment0/build/p3d_checksum_timing \
    --rows 100000000 \
    --active-byte-len 8 \
    --segment-size 1024 \
    --warmup 5 \
    --iters 20 \
    2>&1 | tee "$OUT_DIR/timing_raw_fp64.txt"

# 3. Certification preflight queries (CPU-side, 384 queries: sensor+uniform)
echo ""
echo "=== P3-D H200 Preflight Certification (CPU) ==="
python3 scripts/run_phase3_p3d_h200_preflight.py \
    --run-tag "h200_preflight_${SLURM_JOB_ID}" \
    --out-dir "$OUT_DIR/certification" \
    2>&1 | tee "$OUT_DIR/preflight_certification.txt"

# 4. Compute speed_margin_H200 from timing results
echo ""
echo "=== speed_margin_H200 Computation ===" > "$OUT_DIR/speed_margin_h200.txt"

# Extract bandwidth numbers from timing results
for L in 2 3 5 6 8; do
    BW=$(grep -oP 'Effective bandwidth: \K[0-9.]+' "$OUT_DIR/timing_L${L}.txt" 2>/dev/null || echo "N/A")
    echo "active_byte_len=$L: effective_bandwidth = ${BW} GB/s" >> "$OUT_DIR/speed_margin_h200.txt"
done

# Using L=8 as raw FP64 approximation
PROGRESSIVE_BW=$(grep -oP 'Effective bandwidth: \K[0-9.]+' "$OUT_DIR/timing_L2.txt" 2>/dev/null || echo "N/A")
RAW_BW=$(grep -oP 'Effective bandwidth: \K[0-9.]+' "$OUT_DIR/timing_raw_fp64.txt" 2>/dev/null || echo "N/A")

echo "" >> "$OUT_DIR/speed_margin_h200.txt"
echo "speed_margin_H200 approximate:" >> "$OUT_DIR/speed_margin_h200.txt"
echo "  progressive bandwidth (L=2): $PROGRESSIVE_BW GB/s" >> "$OUT_DIR/speed_margin_h200.txt"
echo "  raw FP64 bandwidth (L=8):   $RAW_BW GB/s" >> "$OUT_DIR/speed_margin_h200.txt"

echo ""
echo "Preflight complete. Output: $OUT_DIR"
