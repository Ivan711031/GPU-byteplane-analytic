#!/usr/bin/env bash
#SBATCH -p dev
#SBATCH -J scientific_extra_qg
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o results/exp4_filter_aggregate/scientific_extra_fields_quasi_global_%j.out
#SBATCH -e results/exp4_filter_aggregate/scientific_extra_fields_quasi_global_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

JOB_ID="${SLURM_JOB_ID:-manual}"
RESULT_DIR="results/exp4_filter_aggregate/scientific_extra_fields_quasi_global_${JOB_ID}"
mkdir -p "$RESULT_DIR"
STDOUT_PATH="$ROOT_DIR/results/exp4_filter_aggregate/scientific_extra_fields_quasi_global_${JOB_ID}.out"
STDERR_PATH="$ROOT_DIR/results/exp4_filter_aggregate/scientific_extra_fields_quasi_global_${JOB_ID}.err"
GPU_UTIL_LOG="$RESULT_DIR/gpu_util.csv"
COMPLETION_MARKER="$ROOT_DIR/handoff/job_done/job_${JOB_ID}.json"

write_completion_marker() {
  local status="$1"
  mkdir -p "$(dirname "$COMPLETION_MARKER")"
  local git_commit
  git_commit="$(git rev-parse HEAD 2>/dev/null || true)"
  cat > "$COMPLETION_MARKER" <<EOF
{
  "job_id": "$JOB_ID",
  "job_name": "scientific_extra_qg",
  "exit_status": $status,
  "finished_at": "$(date -Is)",
  "workdir": "$ROOT_DIR",
  "git_commit": "$git_commit",
  "stdout": "$STDOUT_PATH",
  "stderr": "$STDERR_PATH",
  "gpu_util_log": "$GPU_UTIL_LOG",
  "result_dir": "$RESULT_DIR",
  "sweep_csv": "$SWEEP_CSV",
  "next_action": "generate research report"
}
EOF
}

trap 'status=$?; write_completion_marker "$status"' EXIT

module load cuda/12.6
module load miniconda3/26.1.1
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan

echo "[$(date -Is)] hardware gate"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "GPU: $gpu_name"
if [[ "$gpu_name" != *"H200"* ]]; then
  echo "ERROR: expected H200 GPU, got: $gpu_name" >&2
  exit 2
fi

echo "[$(date -Is)] toolchain"
which nvcc
cmake --version
gcc --version | head -n 1

echo "[$(date -Is)] Building true quasi-global artifacts"
python3 scripts/build_scientific_v2_artifact.py \
  --dataset cesm_atm_q \
  --segment-size 168480000 \
  --artifact-root datasets/scientific/dev_buff_v2_scientific_quasi_global

python3 scripts/build_scientific_v2_artifact.py \
  --dataset hurricane_tc \
  --segment-size 25000000 \
  --artifact-root datasets/scientific/dev_buff_v2_scientific_quasi_global

BUILD_DIR="$ROOT_DIR/build_filter_aggregate"
mkdir -p "$BUILD_DIR"
cmake -S "$ROOT_DIR/benchmarks/experiment4_filter_aggregate" -B "$BUILD_DIR" \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j$(nproc)

BENCH="$BUILD_DIR/bench_filter_aggregate"
SWEEP_CSV="$RESULT_DIR/scientific_extra_fields_quasi_global.csv"

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 1 > "$GPU_UTIL_LOG" 2>/dev/null &
GPU_MONITOR_PID=$!

SELECTIVITIES=(10 50 90)
ITERS=200
WARMUP=10
SWEEP_CSV_INITIALIZED=false
TOTAL_POINTS=0
PASS_COUNT=0
FAIL_COUNT=0

# ---- CESM Q ----
DATASET_Q="cesm_atm_q"
ARTIFACT_Q=$(ls -d datasets/scientific/dev_buff_v2_scientific_quasi_global/cesm_atm_q/bfp-dec* | head -n 1)
RAW_Q="results/buff_encoder_v2/raw_scientific/cesm_atm_q.f64le.bin"
MAX_Q=$(python3 -c "import json; print(json.load(open('${ARTIFACT_Q}/manifest.json'))['max_plane_count'])")
K_STAR_SUM=2

echo "[$(date -Is)] Running Filter-Aggregate: cesm_atm_q"
for SEL in "${SELECTIVITIES[@]}"; do
  QUANTILE=$(python3 -c "print(1 - ${SEL}/100)")
  THRESHOLD=$(python3 -c "
import numpy as np
data = np.fromfile('${RAW_Q}', dtype=np.float64)
print(np.quantile(data, ${QUANTILE}))
")
  for K in $K_STAR_SUM $MAX_Q; do
    TOTAL_POINTS=$((TOTAL_POINTS + 1))
    CSV_OUT="$RESULT_DIR/run_${DATASET_Q}_s${SEL}_k${K}.csv"
    LOG_OUT="$RESULT_DIR/run_${DATASET_Q}_s${SEL}_k${K}.log"
    echo "  [s${SEL} k=${K}/${MAX_Q}] threshold=${THRESHOLD}"
    if $BENCH \
        --dataset "$ARTIFACT_Q" \
        --raw "$RAW_Q" \
        --threshold "$THRESHOLD" \
        --max-filter-planes "$K" \
        --iters "$ITERS" \
        --warmup "$WARMUP" \
        --validate \
        --block-threads 256 \
        --csv "$CSV_OUT" > "$LOG_OUT" 2>&1; then
      PASS_COUNT=$((PASS_COUNT + 1))
      if [[ "$SWEEP_CSV_INITIALIZED" == "false" ]]; then
        head -1 "$CSV_OUT" > "$SWEEP_CSV"
        SWEEP_CSV_INITIALIZED=true
      fi
      tail -n +2 "$CSV_OUT" >> "$SWEEP_CSV"
    else
      FAIL_COUNT=$((FAIL_COUNT + 1))
      echo "    FAIL: exit code $?"
    fi
  done
done

# ---- Hurricane TC ----
DATASET_TC="hurricane_tc"
ARTIFACT_TC=$(ls -d datasets/scientific/dev_buff_v2_scientific_quasi_global/hurricane_tc/bfp-dec* | head -n 1)
RAW_TC="results/buff_encoder_v2/raw_scientific/hurricane_tc.f64le.bin"
MAX_TC=$(python3 -c "import json; print(json.load(open('${ARTIFACT_TC}/manifest.json'))['max_plane_count'])")
K_STAR_SUM=2

echo "[$(date -Is)] Running Filter-Aggregate: hurricane_tc"
for SEL in "${SELECTIVITIES[@]}"; do
  QUANTILE=$(python3 -c "print(1 - ${SEL}/100)")
  THRESHOLD=$(python3 -c "
import numpy as np
data = np.fromfile('${RAW_TC}', dtype=np.float64)
print(np.quantile(data, ${QUANTILE}))
")
  for K in $K_STAR_SUM $MAX_TC; do
    TOTAL_POINTS=$((TOTAL_POINTS + 1))
    CSV_OUT="$RESULT_DIR/run_${DATASET_TC}_s${SEL}_k${K}.csv"
    LOG_OUT="$RESULT_DIR/run_${DATASET_TC}_s${SEL}_k${K}.log"
    echo "  [s${SEL} k=${K}/${MAX_TC}] threshold=${THRESHOLD}"
    if $BENCH \
        --dataset "$ARTIFACT_TC" \
        --raw "$RAW_TC" \
        --threshold "$THRESHOLD" \
        --max-filter-planes "$K" \
        --iters "$ITERS" \
        --warmup "$WARMUP" \
        --validate \
        --block-threads 256 \
        --csv "$CSV_OUT" > "$LOG_OUT" 2>&1; then
      PASS_COUNT=$((PASS_COUNT + 1))
      if [[ "$SWEEP_CSV_INITIALIZED" == "false" ]]; then
        head -1 "$CSV_OUT" > "$SWEEP_CSV"
        SWEEP_CSV_INITIALIZED=true
      fi
      tail -n +2 "$CSV_OUT" >> "$SWEEP_CSV"
    else
      FAIL_COUNT=$((FAIL_COUNT + 1))
      echo "    FAIL: exit code $?"
    fi
  done
done

kill $GPU_MONITOR_PID || true

echo "[$(date -Is)] Sweep complete. PASS: $PASS_COUNT, FAIL: $FAIL_COUNT, TOTAL: $TOTAL_POINTS"
echo "Results: $SWEEP_CSV"
