#!/usr/bin/env bash
#SBATCH --job-name=rl_integration_h100
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/u4063895/gpu-byteplane-scan-experiments/logs/rl_integration_h100_%j.log
#SBATCH --error=/home/u4063895/gpu-byteplane-scan-experiments/logs/rl_integration_h100_%j.err
set -euo pipefail
trap 'echo "EXIT_CODE=$?"' EXIT

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT_DIR"

JOB_ID="${SLURM_JOB_ID:-nojob}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
N="${N:-1000000}"
DATASETS="${DATASETS:-sensor uniform}"
SCALE=100
SEED=20260413
FAULT_RATES="${FAULT_RATES:-1e-6 1e-5}"
PLANE_SEEDS="${PLANE_SEEDS:-0 1}"
RELIABILITY_ARTIFACT_ROOT="${RELIABILITY_ARTIFACT_ROOT:-/work/u4063895/datasets/reliability_layer1}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/work/u4063895/datasets}"
RESULTS_DIR="$ROOT_DIR/results/reliability_layer1/run_${TIMESTAMP}_job${JOB_ID}_H100"
BUILD_DIR="$ROOT_DIR/build/reliability_layer1_job${JOB_ID}"

echo "Job $JOB_ID started on $(hostname)"
echo "N=$N DATASETS=$DATASETS"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
[[ "$GPU_NAME" == *"H100"* ]] || { echo "Not H100: $GPU_NAME"; exit 2; }
echo "H100: $GPU_NAME"

module load miniconda3/24.11.1 cmake/3.24.2 cuda/12.6
echo "nvcc: $(which nvcc)"
cmake --version | head -1
gcc --version | head -1

SYNTH_BIN="$BUILD_DIR/synth_datasets"
if [[ ! -f "$SYNTH_BIN" ]]; then
  echo "Building synth_datasets..."
  mkdir -p "$BUILD_DIR"
  gcc -O3 -lm -o "$SYNTH_BIN" "$ROOT_DIR/buff_encoder/synth_datasets.c"
fi

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

echo "Ensuring small integration fault plans..."
for DS in $DATASETS; do
  for PLANE in 0 1 2 3 4 5 6 7; do
    for RATE in $FAULT_RATES; do
      RATE_DIR="rate$(LC_NUMERIC=C printf "%.0e" "$RATE")"
      for SEED_ID in $PLANE_SEEDS; do
        FP_DIR="$RELIABILITY_ARTIFACT_ROOT/fault_plans/$DS/n${N}/scale${SCALE}/plane${PLANE}/${RATE_DIR}"
        FP_PATH="$FP_DIR/seed_${SEED_ID}.json"
        if [[ -f "$FP_PATH" ]]; then
          continue
        fi
        mkdir -p "$FP_DIR"
        python3 -c "
import json, random
from pathlib import Path
rng = random.Random($SEED_ID)
n = $N
rate = float('$RATE')
fc = int(rate * n)
offsets = sorted(rng.sample(range(n), fc)) if fc else []
masks = [rng.randint(1, 255) for _ in range(fc)]
rel = 'fault_plans/$DS/n${N}/scale${SCALE}/plane${PLANE}/${RATE_DIR}/seed_${SEED_ID}.json'
fp = {
    'metadata': {
        'dataset': '$DS',
        'n_rows': n,
        'scale': $SCALE,
        'artifact_id': 'artifacts/$DS/n${N}/scale${SCALE}',
        'fault_plan_id': rel,
        'target_plane': $PLANE,
        'fault_rate': '$RATE',
        'fault_rate_numeric': rate,
        'seed': $SEED_ID,
        'actual_fault_count': fc,
        'plane_size_bytes': n,
    },
    'entries': [{'offset': o, 'mask': m} for o, m in zip(offsets, masks)],
}
Path('$FP_PATH').write_text(json.dumps(fp, indent=2))
"
      done
    done
  done
done

echo "Building GPU binary..."
cmake -S "$ROOT_DIR/benchmarks/reliability_layer1" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "$BUILD_DIR" -j --clean-first
BIN="$BUILD_DIR/bench_reliability_fixture"
echo "Binary: $BIN ($(test -x "$BIN" && echo OK || echo MISSING))"
test -x "$BIN" || { echo "Build failed"; exit 2; }

echo "Verifying artifact integrity..."
for DS in $DATASETS; do
  ART_DIR="$RELIABILITY_ARTIFACT_ROOT/artifacts/$DS/n${N}/scale${SCALE}"
  if [[ ! -f "$ART_DIR/artifact.json" ]]; then
    echo "ERROR: missing artifact $ART_DIR"; exit 2
  fi
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
  echo "OK: $DS artifact verified"
done

TMPDIR="/tmp/rl_int_${JOB_ID}"
mkdir -p "$TMPDIR"

for DS in $DATASETS; do
  ART_DIR="$RELIABILITY_ARTIFACT_ROOT/artifacts/$DS/n${N}/scale${SCALE}"
  echo "Clean gate: $DS"
  if "$BIN" --device 0 --n "$N" --artifact-dir "$ART_DIR" --run-clean \
    --csv "$TMPDIR/clean_${DS}.csv" 2>&1 | grep -q 'clean gate passed'; then
    echo "OK: $DS clean gate"
  else
    echo "FAILED: $DS clean gate"
    exit 2
  fi
done

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

RELIABILITY_RESULTS_ROOT="${RELIABILITY_RESULTS_ROOT:-/work/u4063895/results/reliability_layer1}"
RESULTS_DIR="$RELIABILITY_RESULTS_ROOT/integration_$(date +%Y%m%d_%H%M%S)_job${JOB_ID}_H100"
mkdir -p "$RESULTS_DIR"
cp "$ALL_CSV" "$RESULTS_DIR/reliability_integration.csv"
echo "Results: $RESULTS_DIR"
echo "datasets=$DATASETS n=$N rates=$FAULT_RATES seeds=$PLANE_SEEDS" > "$RESULTS_DIR/run_meta.txt"
echo "oracle_match=$MATCH/$TOTAL" >> "$RESULTS_DIR/run_meta.txt"
echo "job_id=$JOB_ID hostname=$(hostname) gpu=$GPU_NAME" >> "$RESULTS_DIR/run_meta.txt"

[[ "$MATCH" == "$TOTAL" ]] || { echo "ORACLE MISMATCH"; exit 2; }

DATASET_COUNT=$(printf "%s\n" $DATASETS | wc -l)
RATE_COUNT=$(printf "%s\n" $FAULT_RATES | wc -l)
SEED_COUNT=$(printf "%s\n" $PLANE_SEEDS | wc -l)
EXPECTED_CONFIGS=$((DATASET_COUNT * 8 * RATE_COUNT * SEED_COUNT))
if [[ "$CONFIG_COUNT" -ne "$EXPECTED_CONFIGS" ]]; then
  echo "MATRIX INCOMPLETE: expected $EXPECTED_CONFIGS configs, got $CONFIG_COUNT"; exit 2
fi

echo "ALL INTEGRATION TESTS PASSED"
