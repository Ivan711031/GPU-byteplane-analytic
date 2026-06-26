#!/bin/bash
#SBATCH -J nmr_a_gpu_struct
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%j.out

set -euo pipefail

echo "=== NMR-A2: GPU Structured Fault Injection ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME"
if ! echo "$GPU_NAME" | grep -qi "H200"; then
  echo "FATAL: Expected H200"
  exit 2
fi

ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
ml load cmake/4.0.0
conda activate gpu-byteplane-scan

echo "=== Toolchain Verification ==="
which nvcc
cmake --version
gcc --version

PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
BUILD_DIR="/work/u4063895/builds/nmr_a_${SLURM_JOB_ID}"
RESULTS_DIR="$PROJECT_DIR/results/reliability_layer1/phase4/nmr_a_wu_anchored_injection/job_${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR" "$RESULTS_DIR"

echo "Project dir: $PROJECT_DIR"
echo "Build dir:   $BUILD_DIR"
echo "Results dir: $RESULTS_DIR"

cd "$BUILD_DIR"
cmake "$PROJECT_DIR/benchmarks/experiment4_filter_aggregate" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90
make -j"$(nproc)" bench_nmr_b_e2e_pipeline

GPU_BENCH="$BUILD_DIR/bench_nmr_b_e2e_pipeline"

echo
echo "=== Structured fault families through pre-fused GPU NMR path ==="
cd "$PROJECT_DIR"
python3 scripts/run_nmr_a_structured_fault.py \
  --gpu-bench "$GPU_BENCH" \
  --gpu-timeout-s 1800 2>&1 | tee "$RESULTS_DIR/nmr_a_structured_fault.log"

VERDICT=$(cat "$RESULTS_DIR/verdict.txt" 2>/dev/null || echo "UNKNOWN")
cat > "$RESULTS_DIR/job_marker.json" <<ENDJSON
{
  "job_id": "$SLURM_JOB_ID",
  "experiment": "nmr_a_gpu_structured_fault",
  "status": "complete",
  "verdict": "$VERDICT",
  "date": "$(date -Iseconds)",
  "project_dir": "$PROJECT_DIR",
  "results_dir": "$RESULTS_DIR"
}
ENDJSON

echo
echo "=== NMR-A2 GPU Structured Fault Injection Complete ==="
echo "Results: $RESULTS_DIR"
echo "Verdict: $VERDICT"
date
