#!/bin/bash
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=fa_smoke
#SBATCH --output=results/exp4_filter_aggregate/smoke_%j.out
#SBATCH --error=results/exp4_filter_aggregate/smoke_%j.err
set -euo pipefail
# ---------------------------------------------------------------------------
# H200 fail-fast
# ---------------------------------------------------------------------------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi
echo "GPU: $GPU_NAME"
# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module purge
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan
ROOT_DIR="/home/$USER/workspace/gpu-byteplane-scan-experiments"
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: ROOT_DIR not found: $ROOT_DIR" >&2
  exit 2
fi
cd "$ROOT_DIR"
# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
BUILD_DIR="build_filter_aggregate"
mkdir -p "$BUILD_DIR"
cmake -S benchmarks/experiment4_filter_aggregate -B "$BUILD_DIR" \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j$(nproc)
# ---------------------------------------------------------------------------
# Result directory
# ---------------------------------------------------------------------------
RESULT_DIR="results/exp4_filter_aggregate/smoke_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}_${GPU_NAME// /}"
mkdir -p "$RESULT_DIR"
# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DATASET_ROOT="/work/$USER/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
RAW_ROOT="/work/$USER/datasets/synthetic/dev"
DATASETS=("uniform" "heavy_tailed" "sensor" "zipfian")
ARTIFACTS=("p6")
SELECTIVITIES=(50)
BENCH="$BUILD_DIR/bench_filter_aggregate"
# ---------------------------------------------------------------------------
# Smoke test: one dataset, one artifact, one selectivity, k=1..max
# ---------------------------------------------------------------------------
for DATASET in "${DATASETS[@]}"; do
  for ART in "${ARTIFACTS[@]}"; do
    ARTIFACT_ROOT="${DATASET_ROOT}/${DATASET}_${ART}"
    RAW_FILE="${RAW_ROOT}/${DATASET}.f64le.bin"
    if [[ ! -d "$ARTIFACT_ROOT" ]]; then
      echo "SKIP: artifact not found: $ARTIFACT_ROOT"
      continue
    fi
    # Get max plane count from manifest
    MAX_PLANES=$(python3 -c "
import json
with open('${ARTIFACT_ROOT}/manifest.json') as f:
    m = json.load(f)
print(m['max_plane_count'])
")
    for SEL in "${SELECTIVITIES[@]}"; do
      QUANTILE=$(python3 -c "print(1 - ${SEL}/100)")
      THRESHOLD=$(python3 -c "
import numpy as np
data = np.fromfile('${RAW_FILE}', dtype=np.float64)
print(np.quantile(data, ${QUANTILE}))
")
      echo "=== ${DATASET}_${ART} s${SEL} threshold=${THRESHOLD} max_planes=${MAX_PLANES} ==="
      for K in $(seq 1 $MAX_PLANES); do
        CSV="${RESULT_DIR}/run_${DATASET}_${ART}_s${SEL}_k${K}.csv"
        echo "  k=$K ..."
        $BENCH \
          --dataset "$ARTIFACT_ROOT" \
          --raw "$RAW_FILE" \
          --threshold "$THRESHOLD" \
          --max-filter-planes "$K" \
          --iters 10 \
          --warmup 3 \
          --validate \
          --block-threads 256 \
          2>&1 | tee "${CSV%.csv}.log" | tail -1
      done
    done
  done
done
echo ""
echo "=== Smoke test complete ==="
echo "Results in: $RESULT_DIR"
# ---------------------------------------------------------------------------
# Handoff marker
# ---------------------------------------------------------------------------
mkdir -p handoff/job_done
MARKER="handoff/job_done/job_${SLURM_JOB_ID:-local}.json"
cat > "$MARKER" <<EOF
{
  "job_id": "${SLURM_JOB_ID:-local}",
  "job_name": "fa_smoke",
  "exit_status": 0,
  "finished_at": "$(date -Iseconds)",
  "workdir": "$ROOT_DIR",
  "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "stdout": "$RESULT_DIR/smoke_${SLURM_JOB_ID:-local}.out",
  "stderr": "$RESULT_DIR/smoke_${SLURM_JOB_ID:-local}.err",
  "result_dir": "$RESULT_DIR",
  "next_action": "review_smoke_results"
}
EOF
echo "Handoff marker: $MARKER"