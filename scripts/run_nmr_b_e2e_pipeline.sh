#!/bin/bash
#SBATCH -J nmr_b_e2e
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%j.out
set -euo pipefail
echo "=== NMR-B: GPU E2E NMR Pipeline ==="
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
SCRIPT_DIR="$PROJECT_DIR/scripts"
BUILD_DIR="${WORK_DIR}/builds/nmr_b_${SLURM_JOB_ID}"
RESULTS_DIR="$PROJECT_DIR/results/reliability_layer1/phase4/nmr_b_gpu_e2e_pipeline/job_${SLURM_JOB_ID}"
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
echo "=== Phase 1: Shared-fault smoke/oracle ==="
echo "=== Phase 2: Representative full-data GPU scale ==="
cd "$PROJECT_DIR"
python3 scripts/run_nmr_b_e2e_pipeline.py \
  --gpu-bench "$GPU_BENCH" \
  --smoke-only \
  --scale-gpu-only \
  --gpu-timeout-s 1800 2>&1 | tee "$RESULTS_DIR/nmr_b_e2e_pipeline.log"
VERDICT=$(cat "$RESULTS_DIR/verdict.txt" 2>/dev/null || echo "UNKNOWN")
cat > "$RESULTS_DIR/job_marker.json" <<ENDJSON
{
  "job_id": "$SLURM_JOB_ID",
  "experiment": "nmr_b_gpu_e2e_pipeline",
  "status": "complete",
  "verdict": "$VERDICT",
  "date": "$(date -Iseconds)",
  "project_dir": "$PROJECT_DIR",
  "results_dir": "$RESULTS_DIR"
}
ENDJSON
echo
echo "=== NMR-B GPU E2E Pipeline Complete ==="
echo "Results: $RESULTS_DIR"
echo "Verdict: $VERDICT"
date
