#!/bin/bash
# run_nmr_a2_structured_fault_freeze.sbatch
#
# R12 — v1.3 reliability freeze: structured software fault injection through H200 NMR path.
# Runs 7 fault families across 2 datasets × 5 policies × 3 seeds = 210 GPU invocations.
#
# Usage:
#   sbatch scripts/run_nmr_a2_structured_fault_freeze.sbatch
#
# Output:
#   results/v1_3_freeze/structured_faults/job_<JOBID>/
#     nmr_a_structured_fault_matrix.csv  — canonical per-cell results
#     nmr_a_fault_plan_summary.csv       — fault plan metadata
#     nmr_a_policy_summary.csv           — per-policy-per-family aggregation
#     nmr_a_fault_family_summary.csv     — per-family aggregation
#     nmr_a_oracle_comparison.csv        — CPU vs GPU verdict comparison
#     handoff.json                       — structured handoff
#     verdict.txt                        — freeze verdict

#SBATCH --job-name=nmr-a2-structured-fault-freeze
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "=== R12 NMR-A2 Structured Fault Freeze ==="
echo "job_id: ${SLURM_JOB_ID}"
echo "node: $(hostname)"
echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "cwd: $(pwd)"
echo

# Verify H200
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
if [[ "$GPU_NAME" != *"H200"* ]]; then
    echo "FATAL: Expected H200 but got: $GPU_NAME"
    exit 2
fi

ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
conda activate gpu-byteplane-scan

echo "=== Toolchain ==="
which nvcc && nvcc --version | tail -1
which gcc && gcc --version | head -1
python --version
echo

# Build bench_nmr_b_e2e_pipeline
BUILD_DIR="build_exp4"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake ../benchmarks/experiment4_filter_aggregate \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_CUDA_COMPILER="$(which nvcc)"
make -j$(nproc) bench_nmr_b_e2e_pipeline
GPU_BENCH=$(pwd)/bench_nmr_b_e2e_pipeline
echo "GPU benchmark: $GPU_BENCH"
$GPU_BENCH --help 2>&1 | head -5 || true
cd ..

# Run the structured fault experiment
# Modify output path to v1_3_freeze
OUT_ROOT="results/v1_3_freeze/structured_faults/job_${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT"

python scripts/run_nmr_a_structured_fault.py \
    --gpu-bench "$GPU_BENCH" \
    --n-rows 500000 \
    --gpu-timeout-s 600 2>&1 | tee "$OUT_ROOT/structured_fault_freeze.log"

# Copy results from the hardcoded path if different
if [ -d "results/reliability_layer1/phase4/nmr_a_wu_anchored_injection/job_${SLURM_JOB_ID}" ]; then
    cp -rv "results/reliability_layer1/phase4/nmr_a_wu_anchored_injection/job_${SLURM_JOB_ID}/"* "$OUT_ROOT/" 2>/dev/null || true
fi

echo "=== Results in $OUT_ROOT ==="
ls -la "$OUT_ROOT/"
echo "=== DONE ==="
