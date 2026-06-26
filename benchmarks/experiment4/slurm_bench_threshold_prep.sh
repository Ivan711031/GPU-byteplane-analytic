#!/bin/bash
#SBATCH -J thresh_prep_decomp
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u4063895@connect.hku.hk
#SBATCH --output=slurm_logs/threshold_prep_decomp_%j.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
echo "=== Job Info ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Job started at: $(date)"

module purge
module load miniconda3/26.1.1
module load cuda/12.6

echo "=== Toolchain ==="
which nvcc
nvcc --version | tail -1
gcc --version | head -1
cmake --version | head -1

# ---------------------------------------------------------------------------
# Hardware validation: H200 fail-fast gate
# ---------------------------------------------------------------------------
echo "=== GPU Verification ==="
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "GPU: ${GPU_NAME}"
if echo "${GPU_NAME}" | grep -q "H200"; then
    echo "H200 detected - good to proceed"
else
    echo "FATAL: Expected H200 GPU, got: ${GPU_NAME}"
    exit 2
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "=== Build ==="
BUILD_DIR="build_exp4"
SRC_DIR="benchmarks/experiment4"
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target bench_threshold_prep_decomp -j8

# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------
echo "=== Benchmark ==="
RESULTS_DIR="results/threshold_prep_decomp"
mkdir -p "${RESULTS_DIR}"

# Clear any previous results
rm -f "${RESULTS_DIR}/threshold_prep_breakdown.csv"
rm -f "${RESULTS_DIR}/threshold_prep_stage_summary.csv"

./build_exp4/bench_threshold_prep_decomp \
    --raw-root results/buff_encoder_v2/raw_scientific

echo "=== Results ==="
ls -la "${RESULTS_DIR}/"

echo "=== Job finished at: $(date) ==="
