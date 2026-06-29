#!/usr/bin/env bash
#SBATCH --job-name=rl_integration
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/logs/rl_integration_%j.log
#SBATCH --error=${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/logs/rl_integration_%j.err
set +e
trap 'echo "EXIT_CODE=$?"' EXIT
ROOT_DIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr"
cd "$ROOT_DIR"
JOB_ID="${SLURM_JOB_ID:-nojob}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
N="${N:-1000000}"
DATASETS="${DATASETS:-sensor uniform}"
SCALE=100
SEED=20260413
FAULT_RATES="${FAULT_RATES:-1e-6 1e-5}"
PLANE_SEEDS="${PLANE_SEEDS:-0 1}"
RELIABILITY_ARTIFACT_ROOT="${RELIABILITY_ARTIFACT_ROOT:-${WORK_DIR}/datasets/reliability_layer1}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-${WORK_DIR}/datasets}"
RESULTS_DIR="$ROOT_DIR/results/reliability_layer1/run_${TIMESTAMP}_job${JOB_ID}_H200"
echo "Job $JOB_ID started on $(hostname)"
echo "N=$N DATASETS=$DATASETS"
# ── H200 gate ────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
[[ "$GPU_NAME" == *"H200"* ]] || { echo "Not H200: $GPU_NAME"; exit 2; }
echo "✓ H200: $GPU_NAME"
# ── Environment ──────────────────────────────────
module load miniconda3/26.1.1 cuda/12.6 >/dev/null 2>&1 || true
echo "nvcc: $(which nvcc 2>/dev/null || echo not-found)"
# ── Build synth_datasets if needed ───────────────
SYNTH_BIN="$ROOT_DIR/buff_encoder/synth_datasets"
if [[ ! -f "$SYNTH_BIN" ]]; then
  echo "Building synth_datasets..."
  gcc -O3 -lm -o "$SYNTH_BIN" "$ROOT_DIR/buff_encoder/synth_datasets.c"
fi
# ── Generate raw data + artifacts ────────────────
for DS in $DATASETS; do
  RAW_DIR="$RAW_DATA_ROOT/reliability_layer1/raw"
  ART_DIR="$RELIABILITY_ARTIFACT_ROOT/artifacts/$DS/n${N}/scale${SCALE}"
  if [[ ! -f "$ART_DIR/artifact.json" ]]; then
    echo "Generating raw $DS..."
    mkdir -p "$RAW_DIR"
    "$SYNTH_BIN" --dataset "$DS" --count "$N" --seed "$SEED" --out-dir "$RAW_DIR"
    echo "Converting $DS -> artifacts..."
    python3 scripts/build_reliability_artifacts.py \
      --raw-data-root "$RAW_DATA_ROOT" \
      --artifact-root "$RELIABILITY_ARTIFACT_ROOT" \
      --datasets "$DS" \
      --n-rows "$N" \
      --scale "$SCALE" \
      --seed "$SEED" \
      --skip-raw
  else
    echo "Artifact exists: $ART_DIR (skip)"
  fi
done
# ── Build GPU binary ─────────────────────────────
echo "Building GPU binary..."
BUILD_DIR="$ROOT_DIR/build/reliability_layer1"
cmake -S "$ROOT_DIR/benchmarks/reliability_layer1" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "$BUILD_DIR" -j --clean-first
BIN="$BUILD_DIR/bench_reliability_fixture"
echo "Binary: $BIN ($(test -x "$BIN" && echo OK || echo MISSING))"
test -x "$BIN" || { echo "Build failed"; exit 2; }
# ── Verify artifact integrity ────────────────────
echo "Verifying artifact integrity..."
for DS in $DATASETS; do
  ART_DIR="$RELIABILITY_ARTIFACT_ROOT/artifacts/$DS/n${N}/scale${SCALE}"
  if [[ ! -f "$ART_DIR/artifact.json" ]]; then
    echo "ERROR: missing artifact $ART_DIR"; exit 2
  fi
  # CPU reconstruction SUM check via Python (streamed)
  PY_OUT=$(python3 -c "
import json
from pathlib import Path
meta = json.loads(Path('$ART_DIR/artifact.json').read_text())
n = meta['n_rows']
total = 0
for i in range(0, n, 1000000):
    chunk = min(1000000, n - i)
    rows = [0] * chunk
    for p in range(8):
        with open(f'$ART_DIR/plane_{p}.bin', 'rb') as f:
            f.seek(i)
            data = f.read(chunk)
        for j in range(chunk):
            rows[j] |= data[j] << (8 * (7 - p))
    total += sum(rows)
meta_sum = int(meta['clean_encoded_sum'])
print('OK' if total == meta_sum else f'MISMATCH: cpu={total} meta={meta_sum}')
" 2>&1)
  if [[ "$PY_OUT" != "OK" ]]; then
    echo "Artifact verify FAILED: $DS ($PY_OUT)"; exit 2
  fi
  echo "  ✓ $DS artifact verified"
done
# ── Run clean gate for each dataset ──────────────
TMPDIR="/tmp/rl_int_${JOB_ID}"
mkdir -p "$TMPDIR"
for DS in $DATASETS; do
  ART_DIR="$RELIABILITY_ARTIFACT_ROOT/artifacts/$DS/n${N}/scale${SCALE}"
  echo "Clean gate: $DS"
  if "$BIN" --device 0 --n "$N" --artifact-dir "$ART_DIR" --run-clean \
    --csv "$TMPDIR/clean_${DS}.csv" 2>&1 | grep -q 'clean gate passed'; then
    echo "  ✓ $DS clean gate"
  else
    echo "  ✗ $DS clean gate FAILED (binary not found or gate failed)"
    exit 2
  fi
done
# ── Run faulted configs ──────────────────────────
ALL_CSV="$TMPDIR/reliability_integration.csv"
FIRST=1
CONFIG_COUNT=0
for DS in $DATASETS; do
  ART_DIR="$RELIABILITY_ARTIFACT_ROOT/artifacts/$DS/n${N}/scale${SCALE}"
  for PLANE in 0 1 2 3 4 5 6 7; do
    for RATE in $FAULT_RATES; do
      RATE_DIR="rate$(LC_NUMERIC=C printf "%.0e" "$RATE")"
      for SEED in $PLANE_SEEDS; do
        FP="$RELIABILITY_ARTIFACT_ROOT/fault_plans/$DS/n${N}/scale${SCALE}/plane${PLANE}/${RATE_DIR}/seed_${SEED}.json"
        if [[ ! -f "$FP" ]]; then
          echo "MISSING: $FP"
          continue
        fi
        FP_CSV="$TMPDIR/fp_${DS}_p${PLANE}_r${RATE}_s${SEED}.csv"
        "$BIN" --device 0 --n "$N" \
          --artifact-dir "$ART_DIR" --fault-plan "$FP" \
          --csv "$FP_CSV" > /dev/null 2>&1
        bin_rc=$?
        if [[ "$bin_rc" -ne 0 ]]; then
          echo "WARN: binary exit $bin_rc for $DS plane=$PLANE rate=$RATE seed=$SEED"
        fi
        if [[ ! -s "$FP_CSV" ]]; then
          echo "WARN: no CSV for $DS plane=$PLANE rate=$RATE seed=$SEED"
          continue
        fi
        if [[ "$FIRST" -eq 1 ]]; then
          head -1 "$FP_CSV" > "$ALL_CSV"
          FIRST=0
        fi
        tail -1 "$FP_CSV" >> "$ALL_CSV"
        CONFIG_COUNT=$((CONFIG_COUNT + 1))
      done
    done
  done
done
TOTAL=$(tail -n+2 "$ALL_CSV" | wc -l)
MATCH=$(awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $20); if($20 == "true") c++} END{print c+0}' "$ALL_CSV")
echo "DONE: $MATCH/$TOTAL oracle match ($CONFIG_COUNT configs)"
# ── Copy results ─────────────────────────────────
RELIABILITY_RESULTS_ROOT="${RELIABILITY_RESULTS_ROOT:-${WORK_DIR}/results/reliability_layer1}"
RESULTS_DIR="$RELIABILITY_RESULTS_ROOT/integration_$(date +%Y%m%d_%H%M%S)_job${JOB_ID}_H200"
mkdir -p "$RESULTS_DIR"
cp "$ALL_CSV" "$RESULTS_DIR/reliability_integration.csv"
echo "Results: $RESULTS_DIR"
echo "datasets=$DATASETS n=$N rates=$FAULT_RATES seeds=$PLANE_SEEDS" > "$RESULTS_DIR/run_meta.txt"
echo "oracle_match=$MATCH/$TOTAL" >> "$RESULTS_DIR/run_meta.txt"
echo "job_id=$JOB_ID hostname=$(hostname) gpu=$GPU_NAME" >> "$RESULTS_DIR/run_meta.txt"
[[ "$MATCH" == "$TOTAL" ]] || { echo "ORACLE MISMATCH"; exit 2; }
# Assert matrix completeness: 2 datasets x 8 planes x 2 rates x 2 seeds = 64
EXPECTED_CONFIGS=64
if [[ "$CONFIG_COUNT" -ne "$EXPECTED_CONFIGS" ]]; then
  echo "MATRIX INCOMPLETE: expected $EXPECTED_CONFIGS configs, got $CONFIG_COUNT"; exit 2
fi
echo "ALL INTEGRATION TESTS PASSED"
