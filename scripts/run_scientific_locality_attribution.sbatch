#!/bin/bash
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=sci_loc_attr
#SBATCH --output=results/exp4_filter_aggregate/scientific_locality_attribution_%j.out
#SBATCH --error=results/exp4_filter_aggregate/scientific_locality_attribution_%j.err
#
# Scientific Locality Attribution Mini-Sweep
# Measures raw-fused vs byteplane-fused Filter+Aggregate on scientific datasets
# under 65536 and quasi-global locality (plus legacy 4096).
#
# Phase 1: Generate locality artifacts (65536, quasi-global, 4096)
# Phase 2: Run attribution sweep
#
# Usage:
#   sbatch --account=<PROJECT_ID> scripts/run_scientific_locality_attribution.sbatch

set -euo pipefail

# ---------------------------------------------------------------------------
# H200 fail-fast gate
# ---------------------------------------------------------------------------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi
echo "GPU: $GPU_NAME"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan

ROOT_DIR="/home/$USER/workspace/gpu-byteplane-scan-experiments"
if [[ ! -d "$ROOT_DIR" ]]; then exit 2; fi
cd "$ROOT_DIR"

which nvcc
cmake --version | head -1
gcc --version | head -1

# ---------------------------------------------------------------------------
# Phase 1: Generate locality artifacts
# ---------------------------------------------------------------------------
LOCALITY_ROOT="$ROOT_DIR/datasets/locality_sensitivity"
BUFF_TOOL="$ROOT_DIR/bin/buff_tool"

# Raw data paths
RAW_CESM="$ROOT_DIR/results/buff_encoder_v2/raw_scientific/cesm_atm_cloud.f64le.bin"
RAW_HURRICANE="$ROOT_DIR/results/buff_encoder_v2/raw_scientific/hurricane_u.f64le.bin"

# Precision settings (p12/dec12 for scientific, matching Task 3)
PRECISION_POWER=12

echo "[$(date -Is)] Phase 1: Generating locality artifacts"

# Function to generate and export an artifact
generate_artifact() {
  local DATASET="$1"
  local SEG_SIZE="$2"
  local SEG_LABEL="$3"
  local RAW_PATH="$4"
  local VALUE_COUNT="$5"

  local OUT_DIR="$LOCALITY_ROOT/$DATASET/$SEG_LABEL"
  if [[ -d "$OUT_DIR" && -f "$OUT_DIR/summary.json" ]]; then
    echo "  SKIP: $OUT_DIR already exists"
    return 0
  fi

  local ENCODED_DIR="$LOCALITY_ROOT/${DATASET}_${SEG_LABEL}_encoded"
  mkdir -p "$ENCODED_DIR" "$OUT_DIR"

  echo "  Encoding $DATASET/$SEG_LABEL (segment_size=$SEG_SIZE, value_count=$VALUE_COUNT)..."
  if ! $BUFF_TOOL encode \
      --input "$RAW_PATH" \
      --output "$ENCODED_DIR/${DATASET}_${SEG_LABEL}.buff64" \
      --segment-size "$SEG_SIZE" \
      --precision-power "$PRECISION_POWER"; then
    echo "  FAIL: encode failed for $DATASET/$SEG_LABEL"
    return 1
  fi

  echo "  Exporting runtime artifacts for $DATASET/$SEG_LABEL..."
  if ! $BUFF_TOOL export-runtime \
      --input "$ENCODED_DIR/${DATASET}_${SEG_LABEL}.buff64" \
      --raw-input "$RAW_PATH" \
      --out-dir "$OUT_DIR" \
      --dataset "$DATASET"; then
    echo "  FAIL: export-runtime failed for $DATASET/$SEG_LABEL"
    return 1
  fi

  echo "  Done: $OUT_DIR"
  return 0
}

mkdir -p "$LOCALITY_ROOT"

# cesm_atm_cloud: 168,480,000 rows
CESM_ROWS=168480000
generate_artifact "cesm_atm_cloud" "4096" "seg4096" "$RAW_CESM" "$CESM_ROWS"
generate_artifact "cesm_atm_cloud" "65536" "seg65536" "$RAW_CESM" "$CESM_ROWS"
generate_artifact "cesm_atm_cloud" "$CESM_ROWS" "seg_global" "$RAW_CESM" "$CESM_ROWS"

# hurricane_u: 25,000,000 rows
HURRICANE_ROWS=25000000
generate_artifact "hurricane_u" "4096" "seg4096" "$RAW_HURRICANE" "$HURRICANE_ROWS"
generate_artifact "hurricane_u" "65536" "seg65536" "$RAW_HURRICANE" "$HURRICANE_ROWS"
generate_artifact "hurricane_u" "$HURRICANE_ROWS" "seg_global" "$RAW_HURRICANE" "$HURRICANE_ROWS"

echo "[$(date -Is)] Phase 1 complete: Locality artifacts generated"

# ---------------------------------------------------------------------------
# Phase 2: Build benchmark binary
# ---------------------------------------------------------------------------
BUILD_DIR="$ROOT_DIR/build_filter_aggregate"
mkdir -p "$BUILD_DIR"
cmake -S "$ROOT_DIR/benchmarks/experiment4_filter_aggregate" -B "$BUILD_DIR" \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
cmake --build "$BUILD_DIR" -j$(nproc) 2>&1 | tail -3

BENCH="$BUILD_DIR/bench_filter_aggregate"

# ---------------------------------------------------------------------------
# Phase 2: Run attribution sweep
# ---------------------------------------------------------------------------
JOB_ID="${SLURM_JOB_ID:-local}"
RESULT_DIR="$ROOT_DIR/results/exp4_filter_aggregate/scientific_locality_attribution_${JOB_ID}"
mkdir -p "$RESULT_DIR"

# ---------------------------------------------------------------------------
# Output CSV
# ---------------------------------------------------------------------------
ATTRIBUTION_CSV="$RESULT_DIR/scientific_locality_attribution.csv"
echo "dataset,representation,segment_size,segment_count,selectivity,threshold,k,max_planes,"\
"byteplane_fused_ms,raw_fused_ms,bp_vs_raw_speedup,"\
"threshold_prep_ms,warm_e2e_ms,"\
"gpu_count,cpu_raw_count,count_ok,"\
"gpu_sum,cpu_raw_sum,sum_abs_err,sum_rel_err,"\
"raw_baseline_count,raw_baseline_sum,"\
"k_type" > "$ATTRIBUTION_CSV"

# ---------------------------------------------------------------------------
# Threshold prep times from Task 3 locality sweep (ms)
# ---------------------------------------------------------------------------
get_prep_ms() {
  local DS="$1" REP="$2" SEL="$3"
  if [[ "$DS" == "cesm_atm_cloud" ]]; then
    if [[ "$REP" == "seg4096" ]]; then
      case "$SEL" in 10) echo "10.41" ;; 50) echo "17.47" ;; 90) echo "19.42" ;; *) echo "0" ;; esac
    elif [[ "$REP" == "seg65536" ]]; then
      case "$SEL" in 10) echo "0.68" ;; 50) echo "1.10" ;; 90) echo "1.21" ;; *) echo "0" ;; esac
    elif [[ "$REP" == "seg_global" ]]; then
      echo "0.00"
    else echo "0"; fi
  elif [[ "$DS" == "hurricane_u" ]]; then
    if [[ "$REP" == "seg4096" ]]; then
      case "$SEL" in 10) echo "2.50" ;; 50) echo "3.35" ;; 90) echo "2.55" ;; *) echo "0" ;; esac
    elif [[ "$REP" == "seg65536" ]]; then
      case "$SEL" in 10) echo "0.20" ;; 50) echo "0.20" ;; 90) echo "0.19" ;; *) echo "0" ;; esac
    elif [[ "$REP" == "seg_global" ]]; then
      echo "0.00"
    else echo "0"; fi
  else echo "0"; fi
}

get_k_star_sum() {
  # All scientific datasets have SUM k*=2 at epsilon=1e-3 for all representations
  echo "2"
}

# ---------------------------------------------------------------------------
# Sweep function
# ---------------------------------------------------------------------------
ITERS=100
WARMUP=10
TOTAL_POINTS=0
PASS_COUNT=0
FAIL_COUNT=0

run_sweep() {
  local DATASET="$1"
  local RAW_PATH="$2"
  local THR10="$3"
  local THR50="$4"
  local THR90="$5"
  shift 5

  for SEG in "$@"; do
    local ARTIFACT="$LOCALITY_ROOT/$DATASET/$SEG"
    if [[ ! -d "$ARTIFACT" ]]; then
      echo "SKIP: $ARTIFACT not found"
      continue
    fi

    # Get artifact metadata
    local SEG_SIZE=$(python3 -c "
import json
with open('$ARTIFACT/summary.json') as f:
    d = json.load(f)
print(d['segment_size'])
")
    local SEG_COUNT=$(python3 -c "
import json
with open('$ARTIFACT/summary.json') as f:
    d = json.load(f)
print(d['segment_count'])
")
    local MAX_K=$(python3 -c "
import json
with open('$ARTIFACT/summary.json') as f:
    d = json.load(f)
print(d['max_plane_count'])
")

    # Determine representation label
    local REP_LABEL
    case "$SEG" in
      seg4096) REP_LABEL="4096" ;;
      seg16384) REP_LABEL="16384" ;;
      seg65536) REP_LABEL="65536" ;;
      seg_global) REP_LABEL="quasi-global" ;;
      *) REP_LABEL="$SEG" ;;
    esac

    local K_STAR=$(get_k_star_sum "$DATASET" "$SEG")

    echo "=== $DATASET $SEG (rep=$REP_LABEL, seg_size=$SEG_SIZE, seg_count=$SEG_COUNT, max_planes=$MAX_K, k*=$K_STAR) ==="

    for thr_pair in "10:$THR10" "50:$THR50" "90:$THR90"; do
      local SEL="${thr_pair%%:*}"
      local THR="${thr_pair##*:}"
      local PREP_MS=$(get_prep_ms "$DATASET" "$SEG" "$SEL")

      for K in $(seq 1 "$MAX_K"); do
        TOTAL_POINTS=$((TOTAL_POINTS + 1))
        local CSV_OUT="$RESULT_DIR/${DATASET}_${SEG}_s${SEL}_k${K}.csv"
        local LOG_OUT="$RESULT_DIR/${DATASET}_${SEG}_s${SEL}_k${K}.log"

        # Determine k_type
        local K_TYPE="reference"
        if [[ "$K" -eq "$K_STAR" ]]; then
          K_TYPE="k_star"
        elif [[ "$K" -eq "$MAX_K" ]]; then
          K_TYPE="k_max"
        fi

        echo "  s${SEL} k=${K}/${MAX_K} thr=${THR} k_type=${K_TYPE} ..."

        if $BENCH \
            --dataset "$ARTIFACT" \
            --raw "$RAW_PATH" \
            --threshold "$THR" \
            --max-filter-planes "$K" \
            --iters "$ITERS" \
            --warmup "$WARMUP" \
            --validate \
            --block-threads 256 \
            --csv "$CSV_OUT" > "$LOG_OUT" 2>&1; then
          PASS_COUNT=$((PASS_COUNT + 1))

          # Parse CSV and append to attribution CSV
          python3 -c "
import csv, sys

csv_path = '$CSV_OUT'
bp_ms = 'NA'
raw_ms = 'NA'
gpu_count = 'NA'
cpu_raw_count = 'NA'
count_ok = 'NA'
gpu_sum = 'NA'
cpu_raw_sum = 'NA'
sum_abs_err = 'NA'
sum_rel_err = 'NA'
raw_baseline_count = 'NA'
raw_baseline_sum = 'NA'

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        bp_ms = row.get('ms_per_iter', 'NA')
        raw_ms = row.get('raw_baseline_ms_per_iter', 'NA')
        gpu_count = row.get('gpu_count', 'NA')
        cpu_raw_count = row.get('cpu_raw_count', 'NA')
        count_ok = row.get('validated', 'NA')
        gpu_sum = row.get('gpu_sum', 'NA')
        cpu_raw_sum = row.get('cpu_raw_sum', 'NA')
        sum_abs_err = row.get('raw_sum_abs_err', 'NA')
        sum_rel_err = row.get('raw_sum_rel_err', 'NA')
        raw_baseline_count = row.get('raw_baseline_count', 'NA')
        raw_baseline_sum = row.get('raw_baseline_sum', 'NA')

# Compute speedup
try:
    bp_val = float(bp_ms)
    raw_val = float(raw_ms)
    if bp_val > 0:
        speedup = f'{raw_val / bp_val:.4f}'
    else:
        speedup = 'NA'
except (ValueError, ZeroDivisionError):
    speedup = 'NA'

# Compute warm E2E
prep_ms = $PREP_MS
try:
    warm_e2e = f'{prep_ms + float(bp_ms):.4f}'
except ValueError:
    warm_e2e = 'NA'

print(f'$DATASET,$REP_LABEL,$SEG_SIZE,$SEG_COUNT,$SEL,$THR,$K,$MAX_K,'\
f'{bp_ms},{raw_ms},{speedup},'\
f'{prep_ms},{warm_e2e},'\
f'{gpu_count},{cpu_raw_count},{count_ok},'\
f'{gpu_sum},{cpu_raw_sum},{sum_abs_err},{sum_rel_err},'\
f'{raw_baseline_count},{raw_baseline_sum},'\
'$K_TYPE')
" >> "$ATTRIBUTION_CSV" 2>/dev/null || echo "    WARN: parse failed for $CSV_OUT"
        else
          FAIL_COUNT=$((FAIL_COUNT + 1))
          echo "    FAIL: see $LOG_OUT"
        fi
      done
    done
  done
}

# ---------------------------------------------------------------------------
# Run sweeps
# ---------------------------------------------------------------------------

echo "[$(date -Is)] Phase 2: Starting cesm_atm_cloud sweep"
run_sweep "cesm_atm_cloud" \
  "$RAW_CESM" \
  "0.1714855656027794" "0.02812151238322258" "0.0" \
  "seg4096" "seg65536" "seg_global"

echo "[$(date -Is)] Phase 2: Starting hurricane_u sweep"
run_sweep "hurricane_u" \
  "$RAW_HURRICANE" \
  "65.99803924560547" "53.02256774902344" "41.232883071899415" \
  "seg4096" "seg65536" "seg_global"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Scientific Locality Attribution Sweep Complete ==="
echo "PASS: $PASS_COUNT, FAIL: $FAIL_COUNT, TOTAL: $TOTAL_POINTS"
echo "Results: $ATTRIBUTION_CSV"
wc -l "$ATTRIBUTION_CSV"

# ---------------------------------------------------------------------------
# Handoff marker
# ---------------------------------------------------------------------------
mkdir -p handoff/job_done
MARKER="handoff/job_done/job_${JOB_ID}.json"
cat > "$MARKER" <<EOF
{
  "job_id": "$JOB_ID",
  "job_name": "sci_loc_attr",
  "exit_status": 0,
  "finished_at": "$(date -Iseconds)",
  "workdir": "$ROOT_DIR",
  "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "result_csv": "$ATTRIBUTION_CSV",
  "result_dir": "$RESULT_DIR",
  "pass_count": $PASS_COUNT,
  "fail_count": $FAIL_COUNT,
  "total_points": $TOTAL_POINTS,
  "next_action": "review attribution CSV and write report"
}
EOF
echo "Handoff marker: $MARKER"