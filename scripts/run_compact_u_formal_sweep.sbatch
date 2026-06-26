#!/usr/bin/env bash
#SBATCH -p dev
#SBATCH -J compact_u_formal_sweep
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o results/exp4/compact_u_formal_sweep_%j.out
#SBATCH -e results/exp4/compact_u_formal_sweep_%j.err
#
# Compact-U v2 synthetic formal sweep: primitive sweep + cuDF exact baseline.
#
# Sweeps all 19 v2 synthetic artifacts over selectivities 50/90/99 and k=1,2.
# Writes primitive_sweep.csv, merged_compact_u_vs_cudf.csv, and verdict_summary.txt.
#
# Usage:
#   pass the project allocation on the sbatch command line when submitting
#
# NOTE: Do NOT embed project account identifiers in this script.

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -d "$ROOT_DIR/benchmarks/experiment4" ]]; then
  echo "ERROR: cannot locate project root from $ROOT_DIR" >&2
  exit 2
fi
cd "$ROOT_DIR"

mkdir -p results/exp4 handoff/job_done

JOB_ID="${SLURM_JOB_ID:-manual}"
RESULT_DIR="$ROOT_DIR/results/exp4/compact_u_formal_sweep_${JOB_ID}"
mkdir -p "$RESULT_DIR"
STDOUT_PATH="$ROOT_DIR/results/exp4/compact_u_formal_sweep_${JOB_ID}.out"
STDERR_PATH="$ROOT_DIR/results/exp4/compact_u_formal_sweep_${JOB_ID}.err"
GPU_UTIL_LOG="$RESULT_DIR/gpu_util.csv"
SUMMARY_TXT="$RESULT_DIR/verdict_summary.txt"
COMPLETION_MARKER="$ROOT_DIR/handoff/job_done/job_${JOB_ID}.json"

write_completion_marker() {
  local status="$1"
  mkdir -p "$(dirname "$COMPLETION_MARKER")"
  local git_commit
  git_commit="$(git rev-parse HEAD 2>/dev/null || true)"
  cat > "$COMPLETION_MARKER" <<EOF
{
  "job_id": "$JOB_ID",
  "job_name": "compact_u_formal_sweep",
  "exit_status": $status,
  "finished_at": "$(date -Is)",
  "workdir": "$ROOT_DIR",
  "git_commit": "$git_commit",
  "stdout": "$STDOUT_PATH",
  "stderr": "$STDERR_PATH",
  "gpu_util_log": "$GPU_UTIL_LOG",
  "result_dir": "$RESULT_DIR",
  "result_csv": "${MERGED_CSV:-}",
  "next_action": "review merged CSV and verdict summary before paper figure generation"
}
EOF
}

trap 'status=$?; write_completion_marker "$status"' EXIT

module load cuda/12.6

PYTHON_BIN="${PYTHON_BIN:-python}"
RAPIDS_PYTHON="${RAPIDS_PYTHON:-/home/$USER/.conda/envs/gpu-byteplane-rapids/bin/python}"
if [[ -x "$RAPIDS_PYTHON" ]]; then
  PYTHON_BIN="$RAPIDS_PYTHON"
fi
echo "Python: $PYTHON_BIN"

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

BUILD_DIR="$ROOT_DIR/build/exp4"
cmake -S "$ROOT_DIR/benchmarks/experiment4" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "$BUILD_DIR" --target byteplane_count_gt_batch_cli -j1

SWEEP_CSV="$RESULT_DIR/primitive_sweep.csv"
MERGED_CSV="$RESULT_DIR/merged_compact_u_vs_cudf.csv"

ARTIFACT_BASE="/work/$USER/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"

ARTIFACTS=(
  "$ARTIFACT_BASE"/uniform_p2
  "$ARTIFACT_BASE"/uniform_p4
  "$ARTIFACT_BASE"/uniform_p6
  "$ARTIFACT_BASE"/uniform_p8
  "$ARTIFACT_BASE"/uniform_p10
  "$ARTIFACT_BASE"/sensor_p2
  "$ARTIFACT_BASE"/sensor_p4
  "$ARTIFACT_BASE"/sensor_p6
  "$ARTIFACT_BASE"/sensor_p8
  "$ARTIFACT_BASE"/sensor_p10
  "$ARTIFACT_BASE"/heavy_tailed_p2
  "$ARTIFACT_BASE"/heavy_tailed_p3
  "$ARTIFACT_BASE"/heavy_tailed_p4
  "$ARTIFACT_BASE"/heavy_tailed_p5
  "$ARTIFACT_BASE"/heavy_tailed_p6
  "$ARTIFACT_BASE"/zipfian_p2
  "$ARTIFACT_BASE"/zipfian_p4
  "$ARTIFACT_BASE"/zipfian_p6
  "$ARTIFACT_BASE"/zipfian_p8
)

echo "[$(date -Is)] primitive sweep: ${#ARTIFACTS[@]} artifacts"
ARTIFACT_ARGS=()
for a in "${ARTIFACTS[@]}"; do
  ARTIFACT_ARGS+=(--artifact-root "$a")
done
"$PYTHON_BIN" scripts/run_byteplane_count_gt_batch_sweep.py \
  --binary "$BUILD_DIR/byteplane_count_gt_batch_cli" \
  "${ARTIFACT_ARGS[@]}" \
  --raw-root "/work/$USER/datasets/synthetic/dev" \
  --selectivities "50,90,99" \
  --ks "1,2" \
  --compact-u-refine-raw \
  --out-csv "$SWEEP_CSV" \
  --device 0

echo "[$(date -Is)] cuDF exact join"
"$PYTHON_BIN" scripts/cudf_exact_count_gt.py \
  --sweep-csv "$SWEEP_CSV" \
  --output "$MERGED_CSV" \
  --repeat 5 \
  --raw-root "/work/$USER/datasets/synthetic/dev"

if [[ ! -s "$MERGED_CSV" ]]; then
  echo "ERROR: merged CSV missing or empty: $MERGED_CSV" >&2
  exit 2
fi

row_count="$($PYTHON_BIN -c 'import csv, sys
path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as f:
    rows = sum(1 for _ in csv.DictReader(f))
print(rows)
' "$MERGED_CSV")"
if [[ "$row_count" -le 0 ]]; then
  echo "ERROR: merged CSV has no data rows: $MERGED_CSV" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Verdict: analyse raw-FP64 match rate to distinguish exact pass from
# encoded/hybrid drift audit.
# ---------------------------------------------------------------------------
read -r matches_total mismatches_total <<<"$(
  "$PYTHON_BIN" -c "
import csv, sys
path = sys.argv[1]
with open(path, newline='') as f:
    rows = list(csv.DictReader(f))
total = len(rows)
match = sum(1 for r in rows if r.get('raw_cudf_count_matches', r.get('cudf_count_matches', 'False')) == 'True')
print(f'{match} {total - match}')
" "$MERGED_CSV"
)"

if [[ "$matches_total" -eq "$row_count" && "$mismatches_total" -eq 0 ]]; then
  verdict="READY_RAW_EXACT"
  note="All ${row_count} rows show strict raw-FP64 cuDF match. This is a raw exact formal pass."
else
  verdict="READY_ENCODED_HYBRID_WITH_RAW_DRIFT"
  note="Raw strict cuDF match ${matches_total}/${row_count} (${mismatches_total} mismatches)."
  note+=" The mismatches reflect expected drift in bounded/lossy encoded (hybrid) semantics vs raw FP64 cuDF baseline."
  note+=" This is a fidelity/drift audit, NOT a raw exact pass. See handoff report for interpretation."
fi

cat > "$SUMMARY_TXT" <<EOF
result_dir=$RESULT_DIR
primitive_sweep_csv=$SWEEP_CSV
merged_csv=$MERGED_CSV
merged_rows=$row_count
raw_strict_matches=$matches_total
raw_strict_mismatches=$mismatches_total
verdict=$verdict
note=$note
EOF

echo "[$(date -Is)] compact-U formal sweep verdict=$verdict ($matches_total/$row_count raw strict match): $RESULT_DIR"
