#!/bin/bash
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=r8_loc_freeze
#SBATCH --output=results/v1_3_freeze/locality/r8_locality_freeze_%j.out
#SBATCH --error=results/v1_3_freeze/locality/r8_locality_freeze_%j.err
#
# R8: v1.3 evidence freeze for locality and threshold-preparation boundary
# Compares true-quasi-global vs fine-local across segment granularities.
#
# Usage:
#   sbatch --account=<PROJECT_ID> scripts/run_r8_locality_freeze.sbatch
set -euo pipefail
# ---- H200 fail-fast gate ----
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi
echo "GPU: $GPU_NAME"
# ---- Environment ----
module purge
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan
ROOT_DIR="/home/$USER/workspace/gpu-byteplane-scan-experiments"
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: ROOT_DIR not found: $ROOT_DIR" >&2
  exit 2
fi
cd "$ROOT_DIR"
which nvcc
nvcc --version | head -2
# ---- Build ----
BUILD_DIR="build_exp4"
cmake -S benchmarks/experiment4 -B "$BUILD_DIR" \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
cmake --build "$BUILD_DIR" --target byteplane_count_gt_cli -j$(nproc) 2>&1 | tail -5
CLI="$ROOT_DIR/build_exp4/byteplane_count_gt_cli"
LOCALITY_ROOT="$ROOT_DIR/datasets/locality_sensitivity"
# ---- Result directory ----
JOB_ID="${SLURM_JOB_ID:-local}"
RESULT_DIR="results/v1_3_freeze/locality"
mkdir -p "$RESULT_DIR"
CSV="$RESULT_DIR/r8_locality_freeze_${JOB_ID}.csv"
echo "dataset,representation,segment_size,segment_count,max_planes,"\
"selectivity_pct,threshold,k,"\
"threshold_prep_ms,gpu_total_ms,warm_e2e_ms,"\
"artifact_load_ms,segment_minmax_ms,gpu_stage_ms,"\
"count,Q,D,U,bytes_read,"\
"job_id,git_commit" > "$CSV"
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
# ---- Helper: read JSON field ----
read_json() {
  python3 -c '
import json, sys
print(json.load(open(sys.argv[1]))[sys.argv[2]])
' "$@"
}
# ---- Run one measurement ----
run_one() {
  local DATASET="$1"
  local SEG_LABEL="$2"
  local ARTIFACT="$3"
  local THRESHOLD="$4"
  local SELECTIVITY="$5"
  local K="$6"
  if [[ ! -d "$ARTIFACT" ]]; then
    echo "SKIP: artifact dir missing: $ARTIFACT"
    return 0
  fi
  # Get metadata
  local MANIFEST="$ARTIFACT/manifest.json"
  local SEG_SIZE=$(read_json "$MANIFEST" segment_size)
  local SEG_COUNT=$(read_json "$MANIFEST" segment_count)
  local MAX_K=$(read_json "$MANIFEST" max_plane_count)
  local REP_LABEL
  case "$SEG_LABEL" in
    seg4096) REP_LABEL="4096" ;;
    seg16384) REP_LABEL="16384" ;;
    seg65536) REP_LABEL="65536" ;;
    seg_global) REP_LABEL="quasi-global" ;;
    *) REP_LABEL="$SEG_LABEL" ;;
  esac
  # Run GPU
  local GPU_OUT="$RESULT_DIR/${DATASET}_${SEG_LABEL}_s${SELECTIVITY}_k${K}.json"
  $CLI \
    --artifact-root "$ARTIFACT" \
    --raw-root "$LOCALITY_ROOT/dev" \
    --threshold "$THRESHOLD" \
    --k "$K" \
    --device 0 \
    > "$GPU_OUT" 2>&1
  local RC=$?
  if [[ $RC -ne 0 ]]; then
    echo "  FAIL: $DATASET $SEG_LABEL s${SELECTIVITY} k=${K} exit=${RC}"
    return 0
  fi
  # Append CSV row using Python with sys.argv args (no bash interpolation in Python code)
  python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
ds_name = sys.argv[2]
rep = sys.argv[3]
seg_size = sys.argv[4]
seg_count = sys.argv[5]
sel = sys.argv[6]
thr = sys.argv[7]
k = sys.argv[8]
job_id = sys.argv[9]
git_commit = sys.argv[10]
prep = d.get("threshold_prep_ms", 0.0)
gpu_total = d.get("gpu_total_ms", 0.0)
warm_e2e = prep + gpu_total
max_planes = d.get("max_planes", 0)
k_actual = d.get("k", int(k))
al_ms = d.get("artifact_load_ms", 0.0)
smm_ms = d.get("segment_minmax_ms", 0.0)
gs_ms = d.get("gpu_stage_ms", 0.0)
cnt = d.get("count", 0)
q = d.get("Q", 0)
dd = d.get("D", 0)
uu = d.get("U", 0)
br = d.get("bytes_read", 0)
print(
    f"{ds_name},{rep},{seg_size},{seg_count},{max_planes},"
    f"{sel},{thr},{k_actual},"
    f"{prep:.4f},{gpu_total:.4f},{warm_e2e:.4f},"
    f"{al_ms:.4f},{smm_ms:.4f},{gs_ms:.4f},"
    f"{cnt},{q},{dd},{uu},{br},"
    f"{job_id},{git_commit}"
)
' "$GPU_OUT" "$DATASET" "$REP_LABEL" "$SEG_SIZE" "$SEG_COUNT" \
  "$SELECTIVITY" "$THRESHOLD" "$K" "$JOB_ID" "$GIT_COMMIT" \
  2>/dev/null >> "$CSV"
  # Get summary values for echo
  local PREP_VAL=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(f"{d.get(sys.argv[2],0):.4f}")' "$GPU_OUT" threshold_prep_ms 2>/dev/null)
  local GPU_VAL=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(f"{d.get(sys.argv[2],0):.4f}")' "$GPU_OUT" gpu_total_ms 2>/dev/null)
  echo "  OK: $DATASET $SEG_LABEL s${SELECTIVITY} k=${K} prep=${PREP_VAL}ms gpu=${GPU_VAL}ms"
}
# ---- Sweep parameters ----
# cesm_atm_cloud: 168,480,000 rows
CESM_ART="$LOCALITY_ROOT/cesm_atm_cloud"
CESM_SEGS="seg4096 seg65536 seg_global"
CESM_THRS="10:0.1714855656027794 50:0.02812151238322258 90:0.0"
# hurricane_u: 25,000,000 rows
HURR_ART="$LOCALITY_ROOT/hurricane_u"
HURR_SEGS="seg4096 seg65536 seg_global"
HURR_THRS="10:65.99803924560547 50:53.02256774902344 90:41.232883071899415"
echo ""
echo "=== R8: v1.3 Locality/Threshold-Prep Freeze ==="
echo "GPU: $GPU_NAME"
echo "CSV: $CSV"
echo ""
# ---- cesm_atm_cloud sweep ----
echo "=== cesm_atm_cloud ==="
for SEG in $CESM_SEGS; do
  ARTIFACT="$CESM_ART/$SEG"
  MAX_K=$(read_json "$ARTIFACT/manifest.json" max_plane_count)
  for THR_PAIR in $CESM_THRS; do
    SEL="${THR_PAIR%%:*}"
    THR="${THR_PAIR##*:}"
    for K in 1 2 "$MAX_K"; do
      run_one "cesm_atm_cloud" "$SEG" "$ARTIFACT" "$THR" "$SEL" "$K"
    done
  done
done
# ---- hurricane_u sweep ----
echo "=== hurricane_u ==="
for SEG in $HURR_SEGS; do
  ARTIFACT="$HURR_ART/$SEG"
  MAX_K=$(read_json "$ARTIFACT/manifest.json" max_plane_count)
  for THR_PAIR in $HURR_THRS; do
    SEL="${THR_PAIR%%:*}"
    THR="${THR_PAIR##*:}"
    for K in 1 2 "$MAX_K"; do
      run_one "hurricane_u" "$SEG" "$ARTIFACT" "$THR" "$SEL" "$K"
    done
  done
done
# ---- Summary ----
echo ""
echo "=== Complete ==="
echo "CSV: $CSV"
ROWS=$(tail -n +2 "$CSV" | wc -l)
echo "Rows: $ROWS"
echo ""
# ---- Handoff marker ----
mkdir -p handoff/job_done
MARKER="handoff/job_done/job_${JOB_ID}.json"
cat > "$MARKER" <<EOF
{
  "job_id": "$JOB_ID",
  "job_name": "r8_loc_freeze",
  "exit_status": 0,
  "finished_at": "$(date -Iseconds)",
  "workdir": "$ROOT_DIR",
  "git_commit": "$GIT_COMMIT",
  "result_csv": "$CSV",
  "result_dir": "$RESULT_DIR",
  "next_action": "analyze CSV and write freeze report"
}
EOF
echo "Handoff marker: $MARKER"
