#!/bin/bash
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=r7_v1_3_freeze
#SBATCH --output=results/v1_3_freeze/r7_freeze_%j.out
#SBATCH --error=results/v1_3_freeze/r7_freeze_%j.err

set -euo pipefail

START_TIME=$(date +%s)
ROOT_DIR="/home/$USER/workspace/gpu-byteplane-scan-experiments"

# ensure result directories exist
mkdir -p "$ROOT_DIR/results/v1_3_freeze/execution"

# handoff marker trap
cleanup() {
  mkdir -p handoff/job_done
  local end_ts
  end_ts=$(date +%s)
  local elapsed=$((end_ts - START_TIME))
  cat > "handoff/job_done/job_${SLURM_JOB_ID:-local}.json" <<EOF
{
  "job_id": "${SLURM_JOB_ID:-local}",
  "job_name": "r7_v1_3_freeze",
  "exit_status": $?,
  "finished_at": "$(date -Iseconds)",
  "elapsed_seconds": $elapsed,
  "workdir": "$ROOT_DIR",
  "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
}
EOF
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# H200 fail-fast
# ---------------------------------------------------------------------------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi
echo "GPU: $GPU_NAME"
echo "Job ID: ${SLURM_JOB_ID:-local}"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module purge
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan

# ---------------------------------------------------------------------------
# Build (worktree has the complete source set)
# ---------------------------------------------------------------------------
WT_DIR="/home/$USER/workspace/gpu-byteplane-scan-experiments/.worktrees/issue-215-nmr-r7-evidence-freeze"
BUILD_DIR="build_freeze_${SLURM_JOB_ID:-local}"
mkdir -p "$BUILD_DIR"
cmake -S "$WT_DIR/benchmarks/experiment4_filter_aggregate" -B "$BUILD_DIR" \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --target bench_filter_aggregate -j$(nproc)
BENCH="$BUILD_DIR/bench_filter_aggregate"
if [[ ! -f "$BENCH" ]]; then
  echo "ERROR: build failed — bench_filter_aggregate not found" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Result directory
# ---------------------------------------------------------------------------
RESULT_DIR="$ROOT_DIR/results/v1_3_freeze/execution"

# ---------------------------------------------------------------------------
# Configuration: four scientific quasi-global fields
# ---------------------------------------------------------------------------
# cesm_atm_cloud quasi-global (seg_global, segment_count=1, bfp-dec12, 7 planes)
declare -A DATASET_PATHS
DATASET_PATHS[cesm_atm_cloud]="/work/$USER/datasets/locality_sensitivity/cesm_atm_cloud/seg_global"
DATASET_PATHS[hurricane_u]="/work/$USER/datasets/locality_sensitivity/hurricane_u/seg_global"
DATASET_PATHS[cesm_atm_q]="/work/$USER/datasets/scientific/dev_buff_v2_scientific_quasi_global/cesm_atm_q/bfp-dec12"
DATASET_PATHS[hurricane_tc]="/work/$USER/datasets/scientific/dev_buff_v2_scientific_quasi_global/hurricane_tc/bfp-dec11"

declare -A RAW_FILES
RAW_FILES[cesm_atm_cloud]="/home/$USER/workspace/gpu-byteplane-scan-experiments/results/buff_encoder_v2/raw_scientific/cesm_atm_cloud.f64le.bin"
RAW_FILES[hurricane_u]="/home/$USER/workspace/gpu-byteplane-scan-experiments/results/buff_encoder_v2/raw_scientific/hurricane_u.f64le.bin"
RAW_FILES[cesm_atm_q]="/home/$USER/workspace/gpu-byteplane-scan-experiments/results/buff_encoder_v2/raw_scientific/cesm_atm_q.f64le.bin"
RAW_FILES[hurricane_tc]="/home/$USER/workspace/gpu-byteplane-scan-experiments/results/buff_encoder_v2/raw_scientific/hurricane_tc.f64le.bin"

declare -A MAX_PLANES
MAX_PLANES[cesm_atm_cloud]=7
MAX_PLANES[hurricane_u]=8
MAX_PLANES[cesm_atm_q]=6
MAX_PLANES[hurricane_tc]=6

declare -A PRECISION_MODE
PRECISION_MODE[cesm_atm_cloud]="bfp-dec12"
PRECISION_MODE[hurricane_u]="bfp-dec12"
PRECISION_MODE[cesm_atm_q]="bfp-dec12"
PRECISION_MODE[hurricane_tc]="bfp-dec11"

# Standard selectivities for scientific headline
SELECTIVITIES=(10 50 90)

ITERS=200
WARMUP=10

# ---------------------------------------------------------------------------
# Main sweep CSV (one combined file with all rows)
# ---------------------------------------------------------------------------
SWEEP_CSV="$RESULT_DIR/r7_v1_3_freeze_execution.csv"
SWEEP_CSV_INITIALIZED=false
TOTAL_POINTS=0
PASS_COUNT=0
FAIL_COUNT=0

echo "================================================================"
echo "R7 v1.3 Evidence Freeze — Four-Field H200 Execution"
echo "Start: $(date -Iseconds)"
echo "================================================================"

for DATASET in cesm_atm_cloud hurricane_u cesm_atm_q hurricane_tc; do
  ARTIFACT_ROOT="${DATASET_PATHS[$DATASET]}"
  RAW_FILE="${RAW_FILES[$DATASET]}"
  MAX_K="${MAX_PLANES[$DATASET]}"
  PREC="${PRECISION_MODE[$DATASET]}"

  if [[ ! -d "$ARTIFACT_ROOT" ]]; then
    echo "SKIP: artifact not found: $ARTIFACT_ROOT"
    continue
  fi
  if [[ ! -f "$RAW_FILE" ]]; then
    echo "SKIP: raw file not found: $RAW_FILE"
    continue
  fi

  echo ""
  echo "================================================================"
  echo "Dataset: $DATASET  Precision: $PREC  Max planes: $MAX_K"
  echo "Artifact: $ARTIFACT_ROOT"
  echo "================================================================"

  for SEL in "${SELECTIVITIES[@]}"; do
    QUANTILE=$(python -c "print(1 - ${SEL}/100.0)")
    THRESHOLD=$(python -c "
import numpy as np
data = np.fromfile('${RAW_FILE}', dtype=np.float64)
print(np.quantile(data, ${QUANTILE}))
")

    # k=2 shallow execution
    K=2
    TOTAL_POINTS=$((TOTAL_POINTS + 1))
    CSV_OUT="$RESULT_DIR/run_${DATASET}_s${SEL}_k${K}.csv"
    LOG_OUT="$RESULT_DIR/run_${DATASET}_s${SEL}_k${K}.log"

    echo ""
    echo "  [${DATASET} s${SEL} k=${K}/${MAX_K}] threshold=${THRESHOLD}"
    echo "  CSV: $CSV_OUT"

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
      echo "    PASS"
    else
      FAIL_COUNT=$((FAIL_COUNT + 1))
      echo "    FAIL (exit code $?)"
    fi

    # k=max diagnostic/fallback row
    K=$MAX_K
    TOTAL_POINTS=$((TOTAL_POINTS + 1))
    CSV_OUT="$RESULT_DIR/run_${DATASET}_s${SEL}_k${K}.csv"
    LOG_OUT="$RESULT_DIR/run_${DATASET}_s${SEL}_k${K}.log"

    echo ""
    echo "  [${DATASET} s${SEL} k=${K}/${MAX_K} (max)] threshold=${THRESHOLD}"
    echo "  CSV: $CSV_OUT"

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
      echo "    PASS"
    else
      FAIL_COUNT=$((FAIL_COUNT + 1))
      echo "    FAIL (exit code $?)"
    fi
  done
done

# ---------------------------------------------------------------------------
# Assemble combined sweep CSV
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Assembling combined sweep CSV..."
echo "================================================================"

FIRST_CSV=true
for CSV in "$RESULT_DIR"/run_*.csv; do
  if [[ ! -f "$CSV" ]]; then continue; fi
  if $FIRST_CSV; then
    head -1 "$CSV" > "$SWEEP_CSV"
    FIRST_CSV=false
  fi
  tail -n +2 "$CSV" >> "$SWEEP_CSV"
done

echo "Combined CSV: $SWEEP_CSV ($(wc -l < "$SWEEP_CSV") rows)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "R7 v1.3 Evidence Freeze Complete"
echo "  Total points: $TOTAL_POINTS"
echo "  PASS: $PASS_COUNT"
echo "  FAIL: $FAIL_COUNT"
echo "  Results: $RESULT_DIR"
echo "  Sweep CSV: $SWEEP_CSV"
echo "  GPU: $GPU_NAME"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "  Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "================================================================"

# ---------------------------------------------------------------------------
# Handoff marker
# ---------------------------------------------------------------------------
mkdir -p handoff/job_done
MARKER="handoff/job_done/job_${SLURM_JOB_ID:-local}.json"
cat > "$MARKER" <<EOF
{
  "job_id": "${SLURM_JOB_ID:-local}",
  "job_name": "r7_v1_3_freeze_sweep",
  "exit_status": 0,
  "finished_at": "$(date -Iseconds)",
  "workdir": "$ROOT_DIR",
  "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "result_dir": "$RESULT_DIR",
  "sweep_csv": "$SWEEP_CSV",
  "total_points": $TOTAL_POINTS,
  "pass_count": $PASS_COUNT,
  "fail_count": $FAIL_COUNT,
  "next_action": "generate_r7_report"
}
EOF
echo "Handoff marker: $MARKER"
