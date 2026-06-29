#!/bin/bash
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=fa_sweep
#SBATCH --output=results/exp4_filter_aggregate/sweep_%j.out
#SBATCH --error=results/exp4_filter_aggregate/sweep_%j.err
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
BENCH="$BUILD_DIR/bench_filter_aggregate"
# ---------------------------------------------------------------------------
# Result directory
# ---------------------------------------------------------------------------
RESULT_DIR="results/exp4_filter_aggregate/formal_sweep_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}_${GPU_NAME// /}"
mkdir -p "$RESULT_DIR"
# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
DATASET_ROOT="/work/$USER/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
RAW_ROOT="/work/$USER/datasets/synthetic/dev"
# One artifact per dataset: highest-precision for widest k range
declare -A DATASET_ARTIFACTS
DATASET_ARTIFACTS[uniform]=uniform_p10
DATASET_ARTIFACTS[heavy_tailed]=heavy_tailed_p6
DATASET_ARTIFACTS[sensor]=sensor_p10
DATASET_ARTIFACTS[zipfian]=zipfian_p8
SELECTIVITIES=(1 5 10 25 50 75 90 95 99)
ITERS=200
WARMUP=10
# ---------------------------------------------------------------------------
# Sweep CSV header (will be created by first benchmark run)
# ---------------------------------------------------------------------------
SWEEP_CSV="$RESULT_DIR/sweep_results.csv"
SWEEP_CSV_INITIALIZED=false
# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
TOTAL_POINTS=0
PASS_COUNT=0
FAIL_COUNT=0
for DATASET in uniform heavy_tailed sensor zipfian; do
  ARTIFACT="${DATASET_ARTIFACTS[$DATASET]}"
  ARTIFACT_ROOT="${DATASET_ROOT}/${ARTIFACT}"
  RAW_FILE="${RAW_ROOT}/${DATASET}.f64le.bin"
  if [[ ! -d "$ARTIFACT_ROOT" ]]; then
    echo "SKIP: artifact not found: $ARTIFACT_ROOT"
    continue
  fi
  if [[ ! -f "$RAW_FILE" ]]; then
    echo "SKIP: raw file not found: $RAW_FILE"
    continue
  fi
  # Get max plane count from manifest
  MAX_PLANES=$(python3 -c "
import json
with open('${ARTIFACT_ROOT}/manifest.json') as f:
    m = json.load(f)
print(m['max_plane_count'])
")
  echo ""
  echo "================================================================"
  echo "Dataset: $DATASET  Artifact: $ARTIFACT  max_planes: $MAX_PLANES"
  echo "================================================================"
  for SEL in "${SELECTIVITIES[@]}"; do
    QUANTILE=$(python3 -c "print(1 - ${SEL}/100)")
    THRESHOLD=$(python3 -c "
import numpy as np
data = np.fromfile('${RAW_FILE}', dtype=np.float64)
print(np.quantile(data, ${QUANTILE}))
")
    for K in $(seq 1 "$MAX_PLANES"); do
      TOTAL_POINTS=$((TOTAL_POINTS + 1))
      CSV_OUT="$RESULT_DIR/run_${ARTIFACT}_s${SEL}_k${K}.csv"
      LOG_OUT="$RESULT_DIR/run_${ARTIFACT}_s${SEL}_k${K}.log"
      echo "  [s${SEL} k=${K}/${MAX_PLANES}] threshold=${THRESHOLD}"
      if $BENCH \
          --dataset "$ARTIFACT_ROOT" \
          --raw "$RAW_FILE" \
          --threshold "$THRESHOLD" \
          --max-filter-planes "$K" \
          --iters "$ITERS" \
          --warmup "$WARMUP" \
          --validate \
          --block-threads 256 \
          --csv "$CSV_OUT" \
          > "$LOG_OUT" 2>&1; then
        PASS_COUNT=$((PASS_COUNT + 1))
        # Append to sweep CSV (skip header after first)
        if [[ "$SWEEP_CSV_INITIALIZED" == "false" ]]; then
          head -1 "$CSV_OUT" > "$SWEEP_CSV"
          SWEEP_CSV_INITIALIZED=true
        fi
        # Append data row (skip header)
        tail -n +2 "$CSV_OUT" >> "$SWEEP_CSV"
      else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "    FAIL: exit code $?"
      fi
    done
  done
done
# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Formal sweep complete"
echo "  Total points: $TOTAL_POINTS"
echo "  PASS: $PASS_COUNT"
echo "  FAIL: $FAIL_COUNT"
echo "  Results: $RESULT_DIR"
echo "  Sweep CSV: $SWEEP_CSV"
echo "================================================================"
# ---------------------------------------------------------------------------
# Handoff marker
# ---------------------------------------------------------------------------
mkdir -p handoff/job_done
MARKER="handoff/job_done/job_${SLURM_JOB_ID:-local}.json"
cat > "$MARKER" <<EOF
{
  "job_id": "${SLURM_JOB_ID:-local}",
  "job_name": "fa_formal_sweep",
  "exit_status": 0,
  "finished_at": "$(date -Iseconds)",
  "workdir": "$ROOT_DIR",
  "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "result_dir": "$RESULT_DIR",
  "total_points": $TOTAL_POINTS,
  "pass_count": $PASS_COUNT,
  "fail_count": $FAIL_COUNT,
  "sweep_csv": "$SWEEP_CSV",
  "next_action": "analyze_formal_sweep"
}
EOF
echo "Handoff marker: $MARKER"