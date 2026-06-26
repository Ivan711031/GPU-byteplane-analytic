#!/bin/bash
#SBATCH --job-name=rl_phase1_5
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u4063895/gpu-byteplane-scan-experiments/logs/rl_phase1_5_%j.log
#SBATCH --error=/home/u4063895/gpu-byteplane-scan-experiments/logs/rl_phase1_5_%j.err
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_ROOT"
JOB_ID="${SLURM_JOB_ID:-nojob}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

STRATEGY_ID="${STRATEGY_ID:?Must set STRATEGY_ID}"
DATASET="${DATASET:?Must set DATASET}"

# Directory name mapping (matches convert_phase1_5_artifacts.py STRATEGY_DIR)
case "$STRATEGY_ID" in
  hybrid_scale100_plus_shifted) SID_DIR="hybrid" ;;
  *) SID_DIR="$STRATEGY_ID" ;;
esac

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
[[ "$GPU_NAME" == *"H100"* ]] || { echo "Not H100: $GPU_NAME"; exit 2; }
echo "[$TIMESTAMP] H100 OK: $GPU_NAME"

module load miniconda3/24.11.1 cmake/3.24.2 cuda/12.6
echo "nvcc: $(which nvcc)"
cmake --version | head -1
gcc --version | head -1

N=100000000
PARAMS_JSON="$REPO_ROOT/results/phase1_5_strategy_params.json"
SCALE=$(python3 -c "
import json, sys
p = json.load(open('$PARAMS_JSON'))
try:
    d = p['strategies']['$STRATEGY_ID']['per_dataset']['$DATASET']
    print(d['scale'])
except KeyError as e:
    print(f'ERROR: missing param: {e}', file=sys.stderr)
    sys.exit(2)
")
SHIFT_K=$(python3 -c "
import json, sys
p = json.load(open('$PARAMS_JSON'))
try:
    d = p['strategies']['$STRATEGY_ID']['per_dataset']['$DATASET']
    print(d.get('shift_k', 0))
except KeyError as e:
    print(f'ERROR: missing param: {e}', file=sys.stderr)
    sys.exit(2)
")
if [[ "$SHIFT_K" -gt 0 ]]; then
  SCALE_DIR="scale${SCALE}_shift${SHIFT_K}"
else
  SCALE_DIR="scale${SCALE}"
fi

ART_ROOT="${RELIABILITY_ARTIFACT_ROOT:-/work/u4063895/datasets/reliability_layer1}"
RESULTS_BASE="${RELIABILITY_RESULTS_ROOT:-/work/u4063895/results/reliability_layer1}"
ART_DIR="$ART_ROOT/artifacts_phase1_5/$SID_DIR/$DATASET/n${N}/$SCALE_DIR"
FAULT_BASE="$ART_ROOT/fault_plans_phase1_5/$STRATEGY_ID/$DATASET/n${N}/$SCALE_DIR"
RESULTS_DIR="$RESULTS_BASE/phase1_5/$STRATEGY_ID/${DATASET}_${TIMESTAMP}_job${JOB_ID}_H100"
TMPDIR="/tmp/rl_phase1_5_${STRATEGY_ID}_${DATASET}_${JOB_ID}"

FAULT_RATES="1e-08 1e-07 1e-06 1e-05 1e-04"
PLANES="0 1 2 3 4 5 6 7"
SEEDS=$(seq 0 29)
RATE_COUNT=$(printf "%s\n" $FAULT_RATES | wc -l)
PLANE_COUNT=$(printf "%s\n" $PLANES | wc -l)
SEED_COUNT=$(printf "%s\n" $SEEDS | wc -l)
EXPECTED_CONFIGS=$((RATE_COUNT * PLANE_COUNT * SEED_COUNT))

mkdir -p "$TMPDIR" "$RESULTS_DIR"

if [[ ! -f "$ART_DIR/artifact.json" ]]; then
  echo "ERROR: missing artifact $ART_DIR"
  exit 2
fi
echo "Artifact: $DATASET ($(du -sh "$ART_DIR" | cut -f1))"

BUILD_DIR="$REPO_ROOT/build/reliability_layer1_job${JOB_ID}"
cmake -S "$REPO_ROOT/benchmarks/reliability_layer1" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "$BUILD_DIR" -j --clean-first
BIN="$BUILD_DIR/bench_reliability_fixture"
test -x "$BIN" || { echo "Build failed"; exit 2; }

echo "[$(date)] Clean gate..."
"$BIN" --device 0 --n "$N" --artifact-dir "$ART_DIR" --run-clean \
  --csv "$TMPDIR/clean_${DATASET}.csv" 2>&1 | grep 'clean gate passed' || {
  echo "CLEAN GATE FAILED: $DATASET"; exit 2; }
echo "OK: $DATASET"

echo "[$(date)] Full matrix..."
ALL_CSV="$TMPDIR/phase1_5_matrix.csv"
FAIL_CSV="$TMPDIR/case_failures.csv"
FAIL_LOG_DIR="$TMPDIR/failure_logs"
mkdir -p "$FAIL_LOG_DIR"
echo "dataset,target_plane,fault_rate,seed,exit_code,reason,fault_plan_id,log_path" > "$FAIL_CSV"
FIRST=1
TOTAL_CONFIGS=0
ATTEMPTED_CONFIGS=0
MISSING_CONFIGS=0
BINARY_FAILURES=0

for PLANE in $PLANES; do
  for RATE in $FAULT_RATES; do
    RATE_DIR="rate$(LC_NUMERIC=C printf "%.0e" "$RATE")"
    for SEED in $SEEDS; do
      FP="$FAULT_BASE/plane${PLANE}/${RATE_DIR}/seed_${SEED}.json"
      FP_ID="fault_plans_phase1_5/$STRATEGY_ID/$DATASET/n${N}/${SCALE_DIR}/plane${PLANE}/${RATE_DIR}/seed_${SEED}.json"
      if [[ ! -f "$FP" ]]; then
        echo "MISSING: $FP"
        echo "$DATASET,$PLANE,$RATE,$SEED,127,missing_fault_plan,$FP_ID," >> "$FAIL_CSV"
        MISSING_CONFIGS=$((MISSING_CONFIGS + 1))
        continue
      fi
      ATTEMPTED_CONFIGS=$((ATTEMPTED_CONFIGS + 1))
      FP_CSV="$TMPDIR/fp_${DATASET}_p${PLANE}_r${RATE}_s${SEED}.csv"
      if CASE_OUT=$("$BIN" --device 0 --n "$N" \
        --artifact-dir "$ART_DIR" --fault-plan "$FP" \
        --csv "$FP_CSV" 2>&1); then
        BIN_RC=0
      else
        BIN_RC=$?
      fi
      if [[ "$BIN_RC" -ne 0 ]]; then
        BINARY_FAILURES=$((BINARY_FAILURES + 1))
        CASE_LOG="$FAIL_LOG_DIR/${DATASET}_p${PLANE}_${RATE_DIR}_s${SEED}.log"
        printf "%s\n" "$CASE_OUT" > "$CASE_LOG"
        echo "WARN: binary exit $BIN_RC for $DATASET plane=$PLANE rate=$RATE seed=$SEED"
        echo "$DATASET,$PLANE,$RATE,$SEED,$BIN_RC,binary_nonzero,$FP_ID,$CASE_LOG" >> "$FAIL_CSV"
      fi
      if [[ ! -s "$FP_CSV" ]]; then
        echo "WARN: no output for $DATASET plane=$PLANE rate=$RATE seed=$SEED"
        echo "$DATASET,$PLANE,$RATE,$SEED,$BIN_RC,no_csv,$FP_ID," >> "$FAIL_CSV"
        continue
      fi
      if [[ "$FIRST" -eq 1 ]]; then
        printf "%s,strategy_id,strategy_scale,strategy_shift_k\n" "$(head -1 "$FP_CSV")" > "$ALL_CSV"
        FIRST=0
      fi
      DATA_LINE=$(tail -n 1 "$FP_CSV")
      if [[ -n "$DATA_LINE" ]]; then
        printf "%s,%s,%s,%s\n" "$DATA_LINE" "$STRATEGY_ID" "$SCALE" "$SHIFT_K" >> "$ALL_CSV"
      fi
      TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))
    done
  done
  echo "  $DATASET plane=$PLANE done"
done

echo "[$(date)] Matrix complete: $TOTAL_CONFIGS configs"

if [[ -f "$ALL_CSV" ]]; then
  MATCH=$(awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $20); if($20 == "true") c++} END{print c+0}' "$ALL_CSV")
  ROWS=$(tail -n+2 "$ALL_CSV" | wc -l)
else
  MATCH=0
  ROWS=0
fi
FAIL_ROWS=$(tail -n+2 "$FAIL_CSV" | wc -l)
echo "Oracle match: $MATCH/$ROWS"
echo "Attempted configs: $ATTEMPTED_CONFIGS"
echo "Missing configs: $MISSING_CONFIGS"
echo "Binary failures: $BINARY_FAILURES"
echo "Failure rows: $FAIL_ROWS"

# Save outputs BEFORE aggregation (aggregation may fail under set -e)
[[ -f "$ALL_CSV" ]] && cp "$ALL_CSV" "$RESULTS_DIR/canonical_matrix.csv"
cp "$FAIL_CSV" "$RESULTS_DIR/case_failures.csv"
{
  echo "strategy_id=$STRATEGY_ID"
  echo "dataset=$DATASET n=$N scale=$SCALE shift_k=$SHIFT_K"
  echo "scale_dir=$SCALE_DIR"
  echo "rates=$FAULT_RATES seeds=0..29"
  echo "expected_configs=$EXPECTED_CONFIGS"
  echo "attempted_configs=$ATTEMPTED_CONFIGS"
  echo "rows=$ROWS"
  echo "oracle_match=$MATCH/$ROWS"
  echo "missing_configs=$MISSING_CONFIGS"
  echo "binary_failures=$BINARY_FAILURES"
  echo "failure_rows=$FAIL_ROWS"
  echo "gpu=$GPU_NAME job=$JOB_ID"
  echo "platform=Nano5 H100"
} > "$RESULTS_DIR/run_meta.txt"

# Aggregation (best-effort, job output is already saved)
if [[ -f "$ALL_CSV" && "$ROWS" -gt 0 ]]; then
  AGG_DIR="$TMPDIR/agg_input"
  mkdir -p "$AGG_DIR"
  cp "$ALL_CSV" "$AGG_DIR/"
  python3 "$REPO_ROOT/scripts/aggregate_reliability_phase1_5.py" \
    --input-dirs "$AGG_DIR" \
    --strategy-id "$STRATEGY_ID" \
    --params-json "$PARAMS_JSON" \
    --output-dir "$TMPDIR/report" || true
fi
if [[ -d "$TMPDIR/report" ]]; then
  cp -r "$TMPDIR/report/"* "$RESULTS_DIR/" 2>/dev/null || true
fi

cat "$RESULTS_DIR/phase1_5_strategy_review_report.txt" 2>/dev/null || true
echo "Results: $RESULTS_DIR"

if [[ "$ROWS" -ne "$EXPECTED_CONFIGS" ]]; then
  echo "MATRIX INCOMPLETE: expected $EXPECTED_CONFIGS rows, got $ROWS"
  exit 2
fi
if [[ "$MATCH" -ne "$ROWS" ]]; then
  echo "ORACLE MISMATCH: $MATCH/$ROWS"
  exit 2
fi
if [[ "$FAIL_ROWS" -ne 0 ]]; then
  echo "CASE FAILURES: $FAIL_ROWS"
  exit 2
fi
echo "ALL PASSED"
