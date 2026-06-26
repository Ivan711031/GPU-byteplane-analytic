#!/bin/bash
#SBATCH -J p3d_h100_timing
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

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date -Iseconds)"

module purge
ml load cmake/3.24.2
ml load cuda/12.6

# Validate hardware
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME"
if [[ "$GPU_NAME" != *"H100"* ]]; then
    echo "ERROR: Expected H100, got $GPU_NAME"
    exit 2
fi

cd /home/u4063895/gpu-byteplane-scan-experiments

# Build checksum timing benchmark
BUILD_DIR="benchmarks/experiment0/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j p3d_checksum_timing

cd /home/u4063895/gpu-byteplane-scan-experiments

OUT_DIR="/work/u4063895/results/reliability_layer1/phase3/p3d_h100_timing/job_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo ""
echo "=== P3-D H100 Timing Benchmark ==="
./benchmarks/experiment0/build/p3d_checksum_timing \
    --rows 100000000 \
    --active-byte-len 6 \
    --segment-size 1024 \
    --warmup 5 \
    --iters 20 \
    2>&1 | tee "$OUT_DIR/timing_results.txt"

echo ""
echo "Results written to $OUT_DIR/timing_results.txt"

# Write handoff marker
HANDOFF_DIR="/home/u4063895/gpu-byteplane-scan-experiments/handoff/job_done"
mkdir -p "$HANDOFF_DIR"

cat > "$HANDOFF_DIR/job_${SLURM_JOB_ID}.json" <<EOF
{
  "job_id": ${SLURM_JOB_ID},
  "job_name": "p3d_h100_timing",
  "exit_status": 0,
  "finished_at": "$(date -Iseconds)",
  "workdir": "$(pwd)",
  "stdout": "/home/u4063895/gpu-byteplane-scan-experiments/slurm-${SLURM_JOB_ID}.out",
  "stderr": "/home/u4063895/gpu-byteplane-scan-experiments/slurm-${SLURM_JOB_ID}.out",
  "output_dir": "$OUT_DIR",
  "next_action": "Review timing_results.txt for T_progressive, T_verification, T_fallback, and latency budget analysis."
}
EOF

echo "Handoff marker written."
