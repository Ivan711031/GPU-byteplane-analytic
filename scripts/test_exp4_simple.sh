#!/usr/bin/env bash
# Simple test: single dataset, artifact, selectivity combo
# Purpose: validate quantile computation and benchmark flow

set -euo pipefail

# Use SLURM's submission directory as ROOT, fallback to script-based detection
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT_DIR"

# Fixed test parameters
DATASET="uniform"
ARTIFACT="exact"
SELECTIVITY=50
OUTPUT_DIR="results/exp4/test_simple_$$"
ITERS=10
WARMUP=5
BLOCK=256

# Artifact mapping
declare -A ARTIFACT_ROOTS=(
  [exact]="/work/$USER/datasets/synthetic/dev_buff_exp3"
  [p3]="/work/$USER/datasets/synthetic/dev_buff_exp4_p3"
  [p6]="/work/$USER/datasets/synthetic/dev_buff_exp4_p6"
)

RAW_DATA_PATH="/work/$USER/datasets/synthetic/dev/${DATASET}.f64le.bin"
ENCODED_ROOT="${ARTIFACT_ROOTS[$ARTIFACT]}/${DATASET}"

echo "=== Test Parameters ==="
echo "Dataset: $DATASET"
echo "Artifact: $ARTIFACT"
echo "Selectivity: $SELECTIVITY%"
echo "Output dir: $OUTPUT_DIR"
echo "Raw data: $RAW_DATA_PATH"
echo "Encoded root: $ENCODED_ROOT"

# Validation: files exist
if [[ ! -f "$RAW_DATA_PATH" ]]; then
  echo "ERROR: Raw data not found: $RAW_DATA_PATH" >&2
  exit 1
fi

if [[ ! -d "$ENCODED_ROOT" ]]; then
  echo "ERROR: Encoded root not found: $ENCODED_ROOT" >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/build/exp4/bench_progressive_filter" ]]; then
  echo "ERROR: Binary not found: $ROOT_DIR/build/exp4/bench_progressive_filter" >&2
  exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Compute threshold from quantile
echo "=== Computing threshold from quantile ==="
QUANTILE=$(python3 -c "print(1 - $SELECTIVITY/100)")
THRESHOLD=$(python3 -c "
import numpy as np
import struct
data = np.fromfile('${RAW_DATA_PATH}', dtype=np.float64)
print(np.quantile(data, ${QUANTILE}))
")
echo "Quantile: $QUANTILE"
echo "Threshold: $THRESHOLD"

# Step 2: Run benchmark
CSV_FILE="$OUTPUT_DIR/run_${DATASET}_${ARTIFACT}_s${SELECTIVITY}.csv"
echo "=== Running benchmark ==="
echo "CSV output: $CSV_FILE"

"$ROOT_DIR/build/exp4/bench_progressive_filter" \
  --device 0 \
  --encoded-root "$ENCODED_ROOT" \
  --threshold "$THRESHOLD" \
  --validate \
  --csv "$CSV_FILE" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --block "$BLOCK"

# Step 3: Verify output
echo "=== Verifying output ==="
if [[ ! -f "$CSV_FILE" ]]; then
  echo "ERROR: CSV not created: $CSV_FILE" >&2
  exit 1
fi

echo "CSV header:"
head -1 "$CSV_FILE"
echo ""
echo "CSV data row:"
tail -1 "$CSV_FILE"

# Extract validation status
VALIDATED=$(tail -1 "$CSV_FILE" | cut -d',' -f21)
GPU_COUNT=$(tail -1 "$CSV_FILE" | cut -d',' -f7)
CPU_ENCODED=$(tail -1 "$CSV_FILE" | cut -d',' -f8)

echo ""
echo "=== Results ==="
echo "Validated: $VALIDATED"
echo "GPU count: $GPU_COUNT"
echo "CPU encoded count: $CPU_ENCODED"

if [[ "$GPU_COUNT" != "$CPU_ENCODED" ]]; then
  echo "ERROR: Blocking gate failed: gpu_count != cpu_encoded_count" >&2
  exit 2
fi

if [[ "$VALIDATED" != "true" ]]; then
  echo "WARNING: validated=false in CSV" >&2
fi

echo "✓ Test passed!"
echo "Output: $OUTPUT_DIR"
