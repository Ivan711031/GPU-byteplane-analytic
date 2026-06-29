#!/usr/bin/env bash
# NMR-D2 orchestration runner.
# Runs calibration, diagnostic, stochastic, and claim-matrix pipeline.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
# Defaults
DATASET="${DATASET:-sensor}"
N_ROWS="${N_ROWS:-10000000}"
ARTIFACT_DIR="${ARTIFACT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_ROOT/results/reliability_layer1/phase4/nmr_d2/run_$(date +%Y%m%d_%H%M%S)}"
HEADLINE_RATES="${HEADLINE_RATES:-1e-07 1e-06 1e-05}"
SEEDS="${SEEDS:-0 1 2 3 4}"
CANDIDATE_RATES="${CANDIDATE_RATES:-1e-9 3e-9 1e-8 3e-8 1e-7 3e-7 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3}"
DIAG_RATE="${DIAG_RATE:-0.01}"
PMAP_DIR="${PMAP_DIR:-$OUTPUT_DIR/protection_maps}"
if [ -z "$ARTIFACT_DIR" ]; then
    echo "ERROR: ARTIFACT_DIR must be set" >&2
    echo "Usage: DATASET=sensor N_ROWS=1000000 \\" >&2
    echo "       ARTIFACT_DIR=/work/.../artifacts/sensor/n1000000/scale100 \\" >&2
    echo "       $0" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR" "$PMAP_DIR"
echo "=== NMR-D2 Pipeline ==="
echo "Dataset: $DATASET"
echo "N_ROWS: $N_ROWS"
echo "Output: $OUTPUT_DIR"
echo "Headline rates: $HEADLINE_RATES"
echo "Seeds: $SEEDS"
# Step 0: Generate protection maps (once, shared across all steps)
echo ""
echo "--- Step 0: Protection Maps ---"
python3 "$SCRIPT_DIR/nmr_d2_protection_map.py" \
    --policy graded_seg_B3 \
    --n-segments $(( (N_ROWS + 1023) / 1024 )) \
    --seed 0 \
    --output "$PMAP_DIR/graded_seg_B3_seed0.json"
python3 "$SCRIPT_DIR/nmr_d2_protection_map.py" \
    --policy uniform_spread_seg_B3 \
    --n-segments $(( (N_ROWS + 1023) / 1024 )) \
    --seed 0 \
    --output "$PMAP_DIR/uniform_spread_seg_B3_seed0.json"
# Step 1: Calibration sweep
echo ""
echo "--- Step 1: Pilot Calibration ---"
python3 "$SCRIPT_DIR/nmr_d2_calibration.py" \
    --protection-map-dir "$PMAP_DIR" \
    --clean-plane-dir "$ARTIFACT_DIR" \
    --dataset "$DATASET" \
    --n-rows "$N_ROWS" \
    --seed 0 \
    --candidate-rates $CANDIDATE_RATES \
    --output "$OUTPUT_DIR/nmr_d2_calibration_matrix.csv"
# Step 2: Deterministic diagnostic
echo ""
echo "--- Step 2: Deterministic Diagnostic ---"
python3 "$SCRIPT_DIR/nmr_d2_diagnostic.py" \
    --protection-map-dir "$PMAP_DIR" \
    --clean-plane-dir "$ARTIFACT_DIR" \
    --dataset "$DATASET" \
    --n-rows "$N_ROWS" \
    --fault-rate "$DIAG_RATE" \
    --seed 0 \
    --output-dir "$OUTPUT_DIR/diagnostic"
# Step 3: Stochastic evaluation (plane-uniform)
echo ""
echo "--- Step 3a: Stochastic Evaluation (plane-uniform) ---"
python3 "$SCRIPT_DIR/nmr_d2_stochastic.py" \
    --protection-map-dir "$PMAP_DIR" \
    --clean-plane-dir "$ARTIFACT_DIR" \
    --dataset "$DATASET" \
    --n-rows "$N_ROWS" \
    --headline-rates $HEADLINE_RATES \
    --seeds $SEEDS \
    --mode plane_uniform \
    --output "$OUTPUT_DIR/nmr_d2_stochastic_matrix.csv"
# Step 3b: Stochastic evaluation (workload-weighted) — optional
if [ "${RUN_WW}" = "true" ]; then
echo ""
echo "--- Step 3b: Stochastic Evaluation (workload-weighted) ---"
python3 "$SCRIPT_DIR/nmr_d2_stochastic.py" \
    --protection-map-dir "$PMAP_DIR" \
    --clean-plane-dir "$ARTIFACT_DIR" \
    --dataset "$DATASET" \
    --n-rows "$N_ROWS" \
    --headline-rates $HEADLINE_RATES \
    --seeds $SEEDS \
    --mode workload_weighted \
    --output "$OUTPUT_DIR/nmr_d2_stochastic_ww_matrix.csv"
else
echo ""
echo "--- Step 3b: SKIPPED (RUN_WW != true) ---"
fi
# Step 4: Claim matrix + paired deltas (from plane-uniform results)
echo ""
echo "--- Step 4: Claim Matrix ---"
python3 "$SCRIPT_DIR/nmr_d2_claim_matrix.py" \
    --stochastic-results "$OUTPUT_DIR/nmr_d2_stochastic_matrix.csv" \
    --policies graded_seg_B3 uniform_spread_seg_B3 \
    --output-dir "$OUTPUT_DIR"
# Step 5: Provenance manifest
echo ""
echo "--- Step 5: Provenance ---"
python3 "$SCRIPT_DIR/add_artifact_provenance.py" \
    --output-dir "$OUTPUT_DIR" \
    --manifest "$OUTPUT_DIR/provenance_manifest.json" \
    --label nmr_d2 \
    --commit "$(git -C "$PROJ_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)" \
    2>/dev/null || echo "  (provenance skipped)"
echo ""
echo "=== NMR-D2 Pipeline Complete ==="
echo "Results in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null || true
