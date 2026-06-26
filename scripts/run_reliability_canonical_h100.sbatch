#!/usr/bin/env bash
#SBATCH --job-name=rl_canonical_h100
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u4063895/gpu-byteplane-scan-experiments/logs/rl_canonical_h100_%j.log
#SBATCH --error=/home/u4063895/gpu-byteplane-scan-experiments/logs/rl_canonical_h100_%j.err
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT_DIR"
JOB_ID="${SLURM_JOB_ID:-nojob}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
[[ "$GPU_NAME" == *"H100"* ]] || { echo "Not H100: $GPU_NAME"; exit 2; }
echo "[$TIMESTAMP] H100 OK: $GPU_NAME"

module load miniconda3/24.11.1 cmake/3.24.2 cuda/12.6
echo "nvcc: $(which nvcc)"
cmake --version | head -1
gcc --version | head -1

N=100000000
SCALE=100
ART_ROOT="/work/u4063895/datasets/reliability_layer1"
RESULTS_BASE="/work/u4063895/results/reliability_layer1"
RESULTS_DIR="$RESULTS_BASE/canonical_h100_${TIMESTAMP}_job${JOB_ID}_H100"
TMPDIR="/tmp/rl_canon_h100_${JOB_ID}"

DATASETS="${DATASETS:-sensor uniform heavy_tailed zipfian}"
FAULT_RATES="1e-08 1e-07 1e-06 1e-05 1e-04"
PLANES="0 1 2 3 4 5 6 7"
SEEDS=$(seq 0 29)
DATASET_TAG="$(printf "%s" "$DATASETS" | tr ' ' '_' | tr -cs 'A-Za-z0-9_' '_')"
RESULTS_DIR="$RESULTS_BASE/canonical_h100_${DATASET_TAG}_${TIMESTAMP}_job${JOB_ID}_H100"
DATASET_COUNT=$(printf "%s\n" $DATASETS | wc -l)
RATE_COUNT=$(printf "%s\n" $FAULT_RATES | wc -l)
PLANE_COUNT=$(printf "%s\n" $PLANES | wc -l)
SEED_COUNT=$(printf "%s\n" $SEEDS | wc -l)
EXPECTED_CONFIGS=$((DATASET_COUNT * RATE_COUNT * PLANE_COUNT * SEED_COUNT))

mkdir -p "$TMPDIR" "$RESULTS_DIR"

echo "Checking canonical artifacts..."
for DS in $DATASETS; do
  ART_DIR="$ART_ROOT/artifacts/$DS/n${N}/scale${SCALE}"
  if [[ ! -f "$ART_DIR/artifact.json" ]]; then
    echo "ERROR: missing artifact $ART_DIR"
    echo "Run scripts/run_reliability_canonical_artifacts_h100.sbatch first"
    exit 2
  fi
  echo "Artifact: $DS ($(du -sh "$ART_DIR" | cut -f1))"
done

BUILD_DIR="$ROOT_DIR/build/reliability_layer1_job${JOB_ID}"
cmake -S "$ROOT_DIR/benchmarks/reliability_layer1" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "$BUILD_DIR" -j --clean-first
BIN="$BUILD_DIR/bench_reliability_fixture"
test -x "$BIN" || { echo "Build failed"; exit 2; }

echo "[$(date)] Clean gates..."
for DS in $DATASETS; do
  ART_DIR="$ART_ROOT/artifacts/$DS/n${N}/scale${SCALE}"
  "$BIN" --device 0 --n "$N" --artifact-dir "$ART_DIR" --run-clean \
    --csv "$TMPDIR/clean_${DS}.csv" 2>&1 | grep 'clean gate passed' || {
    echo "CLEAN GATE FAILED: $DS"; exit 2; }
  echo "OK: $DS"
done

echo "[$(date)] Full matrix..."
ALL_CSV="$TMPDIR/canonical_matrix.csv"
FAIL_CSV="$TMPDIR/case_failures.csv"
FAIL_LOG_DIR="$TMPDIR/failure_logs"
mkdir -p "$FAIL_LOG_DIR"
echo "dataset,target_plane,fault_rate,seed,exit_code,reason,fault_plan_id,log_path" > "$FAIL_CSV"
FIRST=1
TOTAL_CONFIGS=0
ATTEMPTED_CONFIGS=0
MISSING_CONFIGS=0
BINARY_FAILURES=0

for DS in $DATASETS; do
  ART_DIR="$ART_ROOT/artifacts/$DS/n${N}/scale${SCALE}"
  for PLANE in $PLANES; do
    for RATE in $FAULT_RATES; do
      RATE_DIR="rate$(LC_NUMERIC=C printf "%.0e" "$RATE")"
      for SEED in $SEEDS; do
        FP="$ART_ROOT/fault_plans/$DS/n${N}/scale${SCALE}/plane${PLANE}/${RATE_DIR}/seed_${SEED}.json"
        FP_ID="fault_plans/$DS/n${N}/scale${SCALE}/plane${PLANE}/${RATE_DIR}/seed_${SEED}.json"
        if [[ ! -f "$FP" ]]; then
          echo "MISSING: $FP"
          echo "$DS,$PLANE,$RATE,$SEED,127,missing_fault_plan,$FP_ID," >> "$FAIL_CSV"
          MISSING_CONFIGS=$((MISSING_CONFIGS + 1))
          continue
        fi
        ATTEMPTED_CONFIGS=$((ATTEMPTED_CONFIGS + 1))
        FP_CSV="$TMPDIR/fp_${DS}_p${PLANE}_r${RATE}_s${SEED}.csv"
        if CASE_OUT=$("$BIN" --device 0 --n "$N" \
          --artifact-dir "$ART_DIR" --fault-plan "$FP" \
          --csv "$FP_CSV" 2>&1); then
          BIN_RC=0
        else
          BIN_RC=$?
        fi
        if [[ "$BIN_RC" -ne 0 ]]; then
          BINARY_FAILURES=$((BINARY_FAILURES + 1))
          CASE_LOG="$FAIL_LOG_DIR/${DS}_p${PLANE}_${RATE_DIR}_s${SEED}.log"
          printf "%s\n" "$CASE_OUT" > "$CASE_LOG"
          echo "WARN: binary exit $BIN_RC for $DS plane=$PLANE rate=$RATE seed=$SEED"
          echo "$DS,$PLANE,$RATE,$SEED,$BIN_RC,binary_nonzero,$FP_ID,$CASE_LOG" >> "$FAIL_CSV"
        fi
        if [[ ! -s "$FP_CSV" ]]; then
          echo "WARN: no output for $DS plane=$PLANE rate=$RATE seed=$SEED"
          echo "$DS,$PLANE,$RATE,$SEED,$BIN_RC,no_csv,$FP_ID," >> "$FAIL_CSV"
          continue
        fi
        if [[ "$FIRST" -eq 1 ]]; then
          head -1 "$FP_CSV" > "$ALL_CSV"
          FIRST=0
        fi
        tail -1 "$FP_CSV" >> "$ALL_CSV"
        TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))
      done
    done
    echo "  $DS plane=$PLANE done"
  done
  echo "$DS done"
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

if [[ -f "$ALL_CSV" && "$ROWS" -gt 0 ]]; then
  python3 "$ROOT_DIR/scripts/aggregate_reliability.py" \
    --csv "$ALL_CSV" --output-dir "$TMPDIR/report" --plot
else
  echo "WARN: no canonical matrix rows available for aggregation"
fi

[[ -f "$ALL_CSV" ]] && cp "$ALL_CSV" "$RESULTS_DIR/canonical_matrix.csv"
cp "$FAIL_CSV" "$RESULTS_DIR/case_failures.csv"
if [[ -d "$TMPDIR/report" ]]; then
  cp -r "$TMPDIR/report/"* "$RESULTS_DIR/" 2>/dev/null || true
fi
echo "datasets=$DATASETS n=$N rates=$FAULT_RATES seeds=0..29" > "$RESULTS_DIR/run_meta.txt"
echo "expected_configs=$EXPECTED_CONFIGS" >> "$RESULTS_DIR/run_meta.txt"
echo "attempted_configs=$ATTEMPTED_CONFIGS" >> "$RESULTS_DIR/run_meta.txt"
echo "rows=$ROWS" >> "$RESULTS_DIR/run_meta.txt"
echo "oracle_match=$MATCH/$ROWS" >> "$RESULTS_DIR/run_meta.txt"
echo "missing_configs=$MISSING_CONFIGS" >> "$RESULTS_DIR/run_meta.txt"
echo "binary_failures=$BINARY_FAILURES" >> "$RESULTS_DIR/run_meta.txt"
echo "failure_rows=$FAIL_ROWS" >> "$RESULTS_DIR/run_meta.txt"
echo "gpu=$GPU_NAME job=$JOB_ID" >> "$RESULTS_DIR/run_meta.txt"
echo "platform=Nano5 H100" >> "$RESULTS_DIR/run_meta.txt"

cat "$RESULTS_DIR/phase1_review_report.txt" 2>/dev/null || true
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
