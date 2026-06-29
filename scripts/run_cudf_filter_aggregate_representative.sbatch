#!/bin/bash
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=cudf_fa
#SBATCH --output=results/exp4_filter_aggregate/cudf_baseline_%j.out
#SBATCH --error=results/exp4_filter_aggregate/cudf_baseline_%j.err
# cuDF filter+aggregate baseline (raw FP64) for Exp4 representative points.
# Pass the project allocation with sbatch command-line options when submitting.
set -euo pipefail
# ---------------------------------------------------------------------------
# Query GPU name (stored; validated after handoff trap is registered)
# ---------------------------------------------------------------------------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
# ---------------------------------------------------------------------------
# Resolve repository root (needed early for handoff marker)
# ---------------------------------------------------------------------------
ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/scripts/cudf_filter_aggregate_baseline.py" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
if [[ ! -f "$ROOT_DIR/scripts/cudf_filter_aggregate_baseline.py" ]]; then
  echo "ERROR: could not locate repository root from SLURM_SUBMIT_DIR or script path" >&2
  exit 2
fi
cd "$ROOT_DIR"
# ---------------------------------------------------------------------------
# Result directory (must exist before trap so early failures write handoff)
# ---------------------------------------------------------------------------
RUN_TS=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/exp4_filter_aggregate/cudf_baseline_${RUN_TS}_job${SLURM_JOB_ID:-local}_${GPU_NAME// /}"
mkdir -p "$RESULT_DIR"
OUTPUT_CSV="$RESULT_DIR/cudf_filter_aggregate_representative.csv"
STDOUT_PATH="results/exp4_filter_aggregate/cudf_baseline_${SLURM_JOB_ID:-local}.out"
STDERR_PATH="results/exp4_filter_aggregate/cudf_baseline_${SLURM_JOB_ID:-local}.err"
# GPU utilisation log (nvidia-smi loop in background)
GPU_UTIL_LOG="$RESULT_DIR/gpu_util.log"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv -l 2 > "$GPU_UTIL_LOG" 2>&1 &
GPU_UTIL_PID=$!
# ---------------------------------------------------------------------------
# Trap: write handoff marker on exit (registered early for fail-fast coverage)
# ---------------------------------------------------------------------------
_handoff() {
  local exit_code=$?
  kill "$GPU_UTIL_PID" 2>/dev/null || true
  wait "$GPU_UTIL_PID" 2>/dev/null || true
  mkdir -p handoff/job_done
  cat > "handoff/job_done/job_${SLURM_JOB_ID:-local}.json" <<HERE
{
  "exit_status": $exit_code,
  "job_id": "${SLURM_JOB_ID:-local}",
  "job_name": "cudf_filter_aggregate_baseline",
  "finished_at": "$(date -Iseconds)",
  "workdir": "$ROOT_DIR",
  "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "stdout": "$STDOUT_PATH",
  "stderr": "$STDERR_PATH",
  "gpu_util_log": "$GPU_UTIL_LOG",
  "result_dir": "$RESULT_DIR",
  "result_csv": "$OUTPUT_CSV",
  "next_action": "join_cudf_with_formal_sweep"
}
HERE
  echo "Handoff marker: handoff/job_done/job_${SLURM_JOB_ID:-local}.json"
}
trap _handoff EXIT
# ---------------------------------------------------------------------------
# H200 fail-fast (after trap so handoff marker is always written)
# ---------------------------------------------------------------------------
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi
echo "GPU: $GPU_NAME"
# ---------------------------------------------------------------------------
# Environment setup — prefer RAPIDS env, fall back to project env as last resort
# ---------------------------------------------------------------------------
module purge
module load miniconda3/26.1.1 2>/dev/null || module load miniconda3
module load cuda/12.6 2>/dev/null || module load cuda
eval "$(conda shell.bash hook)"
# Prefer the RAPIDS env; only fall back to gpu-byteplane-scan if it is absent.
RAPIDS_ENV="${RAPIDS_CONDA_ENV:-gpu-byteplane-rapids}"
if ! conda activate "$RAPIDS_ENV" 2>/dev/null; then
  echo "Cannot activate $RAPIDS_ENV; falling back to gpu-byteplane-scan"
  conda activate gpu-byteplane-scan 2>/dev/null || true
fi
# Probe for cudf/cupy; retry with gpu-byteplane-rapids once if needed.
_rapids_probe() {
  python3 -c "import cudf, cupy; print('cuDF', cudf.__version__, 'CuPy', cupy.__version__)" 2>/dev/null
}
if ! _rapids_probe; then
  CURRENT_ENV="${CONDA_DEFAULT_ENV:-unknown}"
  if [[ "$CURRENT_ENV" != "gpu-byteplane-rapids" ]]; then
    echo "RAPIDS probe failed in $CURRENT_ENV; trying gpu-byteplane-rapids …"
    conda activate gpu-byteplane-rapids 2>/dev/null || true
    if ! _rapids_probe; then
      echo "ERROR: cudf/cupy not importable.  Ensure a RAPIDS conda env (gpu-byteplane-rapids) is available." >&2
      exit 3
    fi
  else
    echo "ERROR: cudf/cupy not importable even in gpu-byteplane-rapids." >&2
    exit 3
  fi
fi
# ---------------------------------------------------------------------------
# Resolve the formal sweep CSV
# ---------------------------------------------------------------------------
FORMAL_SWEEP_DIR="results/exp4_filter_aggregate/formal_sweep_20260517_125654_job53158_NVIDIAH200"
FORMAL_CSV="${FORMAL_SWEEP_DIR}/sweep_results.csv"
if [[ ! -f "$FORMAL_CSV" ]]; then
  echo "ERROR: Formal sweep CSV not found at $FORMAL_CSV" >&2
  exit 2
fi
# ---------------------------------------------------------------------------
# Raw file root
# ---------------------------------------------------------------------------
RAW_ROOT="${RAW_ROOT:-/work/$USER/datasets/synthetic/dev}"
# ---------------------------------------------------------------------------
# Run cuDF baseline
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "cuDF filter+aggregate baseline (raw FP64)"
echo "  Formal CSV : $FORMAL_CSV"
echo "  Output CSV : $OUTPUT_CSV"
echo "  Raw root   : $RAW_ROOT"
echo "  Result dir : $RESULT_DIR"
echo "  GPU        : $GPU_NAME"
echo "================================================================"
echo ""
python3 scripts/cudf_filter_aggregate_baseline.py \
  --sweep-csv "$FORMAL_CSV" \
  --output "$OUTPUT_CSV" \
  --raw-root "$RAW_ROOT" \
  --iters 200 \
  --warmup 10 \
  --strategy all \
  --timing-policy all
echo ""
echo "================================================================"
echo "cuDF baseline complete"
echo "  Result CSV: $OUTPUT_CSV"
echo "  Rows: $(tail -n +2 "$OUTPUT_CSV" | wc -l)"
echo "================================================================"
