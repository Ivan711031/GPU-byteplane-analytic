#!/bin/bash
#SBATCH -J nmr_r0_gpu_smoke
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/slurm-%j.out
set -euo pipefail
echo "=== NMR-R0: GPU Byte-Level Majority Vote Smoke ==="
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
SCRIPT_DIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/scripts"
BUILD_DIR="${WORK_DIR}/builds/nmr_r0_${SLURM_JOB_ID}"
RESULTS_DIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/results/reliability_layer1/phase3/nmr_rescue_r0/job_${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR" "$RESULTS_DIR"
HURR_DIR="${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg4096"
CESM_DIR="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"
# Build
cd "$BUILD_DIR"
cmake ${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/benchmarks/experiment4_filter_aggregate \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc) bench_nmr_r0_gpu_smoke
# Run
echo ""
echo "=== GPU Smoke: hurricane_u ==="
./bench_nmr_r0_gpu_smoke --dataset "$HURR_DIR" 2>&1 | tee "$RESULTS_DIR/nmr_r0_gpu_smoke.txt"
echo ""
echo "=== GPU Smoke: cesm_atm_cloud ==="
./bench_nmr_r0_gpu_smoke --dataset "$CESM_DIR" 2>&1 | tee -a "$RESULTS_DIR/nmr_r0_gpu_smoke.txt"
echo ""
echo "=== GPU Smoke complete ==="
date
