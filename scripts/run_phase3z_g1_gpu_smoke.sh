#!/bin/bash
#SBATCH -J g1_gpu_smoke
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/u4063895/workspace/gpu-byteplane-reliability-nmr/slurm-%j.out

set -euo pipefail

echo "=== Phase 3-Z G1: H200 GPU Injection Smoke ==="
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
BUILD_DIR="/work/u4063895/builds/g1_${SLURM_JOB_ID}"
RESULTS_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3z_z1b/job_${SLURM_JOB_ID}"
LOCAL_RESULTS="/home/u4063895/workspace/gpu-byteplane-reliability-nmr/results/reliability_layer1/phase3/phase3z_z1b/job_${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR" "$RESULTS_DIR"
mkdir -p "$LOCAL_RESULTS"

HURR_DIR="/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg4096"
CESM_DIR="/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"

# Build G1 benchmark
cd "$BUILD_DIR"
cmake /home/u4063895/workspace/gpu-byteplane-reliability-nmr/benchmarks/experiment4_filter_aggregate \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc) bench_z_g1_gpu_smoke

# Run on hurricane_u
echo ""
echo "=== G1: hurricane_u ==="
./bench_z_g1_gpu_smoke --dataset "$HURR_DIR" 2>&1 | tee "$RESULTS_DIR/g1_hurricane.txt"

# Run on cesm_atm_cloud
echo ""
echo "=== G1: cesm_atm_cloud ==="
./bench_z_g1_gpu_smoke --dataset "$CESM_DIR" 2>&1 | tee "$RESULTS_DIR/g1_cesm.txt"

# Collect verdict
echo ""
echo "=== G1 Results ==="
grep -A1 "Verdict:" "$RESULTS_DIR/g1_hurricane.txt" "$RESULTS_DIR/g1_cesm.txt"

# Copy results to local
cp "$RESULTS_DIR/g1_hurricane.txt" "$LOCAL_RESULTS/"
cp "$RESULTS_DIR/g1_cesm.txt" "$LOCAL_RESULTS/"

echo ""
echo "=== G1 complete ==="
date
