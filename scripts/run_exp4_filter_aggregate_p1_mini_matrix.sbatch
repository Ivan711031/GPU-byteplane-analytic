#!/usr/bin/env bash
# run_exp4_filter_aggregate_p1_mini_matrix.sbatch
#
# Exp4 P1 fused-only filter+aggregate mini-matrix (4 cases A-D).
# Fused kernel only: progressive_filter_sum_rowpack16_byte_mask.
#
# Cases:
#   A: uniform_p10, s≈50%, k=1, threshold=500.0271759315981
#   B: uniform_p10, s≈50%, k=6, threshold=500.0271759315981
#   C: heavy_tailed_p6, s≈50%, k=8, threshold=1.0001588729752771
#   D: heavy_tailed_p6, s≈99%, k=8, threshold=8.9028979745652526e-06
#
# Usage: pass the project allocation on the sbatch command line when submitting.
# Do not embed allocation identifiers in this script.
#
# Output:
#   results/exp4_filter_aggregate/p1_mini_matrix_<ts>_job<id>_NVIDIAH200/
#     summary.csv, run_meta.txt, index.md, gpu_util.log
#     case_<id>/ — bench.csv, bench.log
#   handoff/job_done/job_<SLURM_JOB_ID>.json

#SBATCH --job-name=exp4-fa-p1-mini
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=results/exp4_filter_aggregate/p1_mini_matrix_%j.out
#SBATCH --error=results/exp4_filter_aggregate/p1_mini_matrix_%j.err

set -euo pipefail

# ==============================================================================
# Dynamic workspace detection
# ==============================================================================
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: ROOT_DIR not found: $ROOT_DIR" >&2
  exit 2
fi
cd "$ROOT_DIR"

RUN_TS=$(date +%Y%m%d_%H%M%S)
JOB_ID="${SLURM_JOB_ID:-local}"

# ==============================================================================
# GPU detection (before result_dir so tag reflects real hardware)
# ==============================================================================
GPU_NAME="unknown"
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ' 2>/dev/null || echo "unknown")
fi
GPU_TAG="${GPU_NAME// /_}"

RESULT_DIR="results/exp4_filter_aggregate/p1_mini_matrix_${RUN_TS}_job${JOB_ID}_${GPU_TAG}"
mkdir -p "$RESULT_DIR"

STDOUT_PATH="results/exp4_filter_aggregate/p1_mini_matrix_${JOB_ID}.out"
STDERR_PATH="results/exp4_filter_aggregate/p1_mini_matrix_${JOB_ID}.err"
GPU_UTIL_LOG="${RESULT_DIR}/gpu_util.log"

# Start GPU util log in background
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv -l 2 > "$GPU_UTIL_LOG" 2>&1 &
GPU_UTIL_PID=$!

# ==============================================================================
# Handoff marker (registered early for fail-fast coverage)
# ==============================================================================
HANDOFF_DIR="${ROOT_DIR}/handoff/job_done"
HANDOFF_MARKER="${HANDOFF_DIR}/job_${JOB_ID}.json"

write_handoff_marker() {
  local status=$?
  kill "$GPU_UTIL_PID" 2>/dev/null || true
  wait "$GPU_UTIL_PID" 2>/dev/null || true
  mkdir -p "$HANDOFF_DIR"
  local git_commit
  git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
  local next_action="Review summary.csv for correctness and performance across 4 cases A-D."
  if [[ "$status" -ne 0 ]]; then
    next_action="Job failed with exit code ${status}; investigate slurm output."
  fi
  cat > "$HANDOFF_MARKER" <<HEREDOC
{
  "exit_status": $status,
  "job_id": "${JOB_ID}",
  "job_name": "exp4-fa-p1-mini",
  "finished_at": "$(date -Iseconds)",
  "workdir": "$ROOT_DIR",
  "git_commit": "${git_commit}",
  "stdout": "$STDOUT_PATH",
  "stderr": "$STDERR_PATH",
  "gpu_util_log": "$GPU_UTIL_LOG",
  "result_dir": "$RESULT_DIR",
  "result_csv": "${RESULT_DIR}/summary.csv",
  "next_action": "${next_action}"
}
HEREDOC
  echo "Handoff marker: ${HANDOFF_MARKER}"
}
trap write_handoff_marker EXIT

# ==============================================================================
# H200 fail-fast
# ==============================================================================
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi
echo "GPU: ${GPU_NAME}"

# ==============================================================================
# Environment & build
# ==============================================================================
module purge
module load miniconda3/26.1.1 2>/dev/null || module load miniconda3
module load cuda/12.6 2>/dev/null || module load cuda
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan 2>/dev/null || conda activate gpu-byteplane-rapids 2>/dev/null || true

echo "=== Toolchain ===" >&2
echo "nvcc: $(which nvcc) $($(which nvcc) --version 2>&1 | grep release || echo unknown)" >&2
echo "gcc: $(gcc --version 2>&1 | head -1)" >&2
echo "cmake: $(cmake --version 2>&1 | head -1)" >&2

echo "=== Job info ===" >&2
echo "Job ID: ${JOB_ID}  Host: $(hostname)  GPU: ${GPU_NAME}" >&2
echo "Output: ${RESULT_DIR}" >&2
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader >&2
git rev-parse HEAD >&2

echo "" >&2
echo "=== Build ===" >&2
BUILD_DIR="build_filter_aggregate"
mkdir -p "$BUILD_DIR"
cmake -S benchmarks/experiment4_filter_aggregate -B "$BUILD_DIR" \
  -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
  -DCMAKE_BUILD_TYPE=Release
BUILD_JOBS="${SLURM_CPUS_ON_NODE:-4}"
cmake --build "$BUILD_DIR" -j"$BUILD_JOBS"
echo "" >&2

BENCH_BIN="${ROOT_DIR}/${BUILD_DIR}/bench_filter_aggregate"
if [[ ! -x "$BENCH_BIN" ]]; then
  echo "ERROR: bench_filter_aggregate not found at ${BENCH_BIN}" >&2
  exit 2
fi

# ==============================================================================
# Case definitions (4 cases, fused-only)
# ==============================================================================
V2_ROOT="/work/${USER}/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
RAW_ROOT="/work/${USER}/datasets/synthetic/dev"

declare -a CASE_IDS=()
declare -a CASE_ARTIFACTS=()
declare -a CASE_SEL_LABELS=()
declare -a CASE_KS=()
declare -a CASE_THRESHOLDS=()

register_case() {
  CASE_IDS+=("$1")
  CASE_ARTIFACTS+=("$2")
  CASE_SEL_LABELS+=("$3")
  CASE_KS+=("$4")
  CASE_THRESHOLDS+=("$5")
}

register_case "A" "uniform_p10"       "s50"  "1" "500.0271759315981"
register_case "B" "uniform_p10"       "s50"  "6" "500.0271759315981"
register_case "C" "heavy_tailed_p6"   "s50"  "8" "1.0001588729752771"
register_case "D" "heavy_tailed_p6"   "s99"  "8" "8.9028979745652526e-06"

TOTAL_CASES=${#CASE_IDS[@]}

# ==============================================================================
# Summary CSV
# ==============================================================================
SUMMARY_CSV="${RESULT_DIR}/summary.csv"
{
  echo "case_id,dataset_artifact,selectivity_label,k,threshold,status,gpu_count,cpu_encoded_count,validated,ms_per_iter,rows_per_sec,bench_csv"
} > "$SUMMARY_CSV"

append_summary_row() {
  echo "$1" >> "$SUMMARY_CSV"
}

# ==============================================================================
# Run matrix
# ==============================================================================
TOTAL_OK=0
TOTAL_FAIL=0
FAILURES=()

for ((ci=0; ci<TOTAL_CASES; ci++)); do
  case_id="${CASE_IDS[$ci]}"
  artifact="${CASE_ARTIFACTS[$ci]}"
  sel_label="${CASE_SEL_LABELS[$ci]}"
  k="${CASE_KS[$ci]}"
  threshold="${CASE_THRESHOLDS[$ci]}"

  dataset="${artifact%_p*}"  # e.g. uniform_p10 → uniform

  artifact_root="${V2_ROOT}/${artifact}"
  raw_path="${RAW_ROOT}/${dataset}.f64le.bin"

  if [[ ! -d "$artifact_root" ]]; then
    echo "ERROR: artifact root not found: $artifact_root" >&2
    exit 2
  fi
  if [[ ! -f "$raw_path" ]]; then
    echo "ERROR: raw file not found: $raw_path" >&2
    exit 2
  fi

  case_dir="${RESULT_DIR}/case_${case_id}"
  mkdir -p "$case_dir"

  echo "" >&2
  echo "============================================================" >&2
  echo "[${case_id}] dataset=${dataset} artifact=${artifact} sel=${sel_label} k=${k}" >&2
  echo "  threshold=${threshold}" >&2
  echo "============================================================" >&2

  # --- Fused benchmark ---
  bench_csv="${case_dir}/bench.csv"
  bench_log="${case_dir}/bench.log"

  set +e
  "$BENCH_BIN" \
    --dataset "$artifact_root" \
    --raw "$raw_path" \
    --threshold "$threshold" \
    --max-filter-planes "$k" \
    --iters 200 \
    --warmup 10 \
    --validate \
    --block-threads 256 \
    --csv "$bench_csv" \
    > "$bench_log" 2>&1
  bench_rc=$?
  set -e

  if [[ $bench_rc -ne 0 ]]; then
    echo "  ✗ Benchmark failed (rc=${bench_rc})" >&2
    FAILURES+=("${case_id}: bench rc=${bench_rc}")
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
    append_summary_row \
      "${case_id},${artifact},${sel_label},${k},${threshold},FAIL_BENCH,NA,NA,NA,NA,NA,${bench_csv}"
    continue
  fi

  # Parse CSV for validation gate
  gpu_count="NA"
  cpu_enc="NA"
  validated="false"
  ms_per_iter="NA"
  rows_per_sec="NA"
  status="FAIL"

  if [[ -f "$bench_csv" ]]; then
    # Header column indices: col12=gpu_count, col13=cpu_enc_count, col49=validated
    gpu_count=$(tail -1 "$bench_csv" | cut -d',' -f12 2>/dev/null || echo "NA")
    cpu_enc=$(tail -1 "$bench_csv" | cut -d',' -f13 2>/dev/null || echo "NA")
    validated=$(tail -1 "$bench_csv" | cut -d',' -f49 2>/dev/null || echo "false")
    ms_per_iter=$(tail -1 "$bench_csv" | cut -d',' -f11 2>/dev/null || echo "NA")
    rows_per_sec=$(tail -1 "$bench_csv" | cut -d',' -f36 2>/dev/null || echo "NA")

    if [[ "$gpu_count" == "$cpu_enc" ]] && [[ "$validated" == "true" ]]; then
      status="PASS"
      echo "  ✓ PASS: gpu=${gpu_count} cpu=${cpu_enc} validated=${validated} ms=${ms_per_iter} rows/s=${rows_per_sec}" >&2
    else
      echo "  ✗ VALIDATION FAILED: gpu=${gpu_count} cpu=${cpu_enc} validated=${validated}" >&2
      FAILURES+=("${case_id}: gpu=${gpu_count} cpu=${cpu_enc} validated=${validated}")
    fi
  else
    echo "  ✗ CSV not found (rc=${bench_rc})" >&2
    FAILURES+=("${case_id}: CSV not found")
  fi

  append_summary_row \
    "${case_id},${artifact},${sel_label},${k},${threshold},${status},${gpu_count},${cpu_enc},${validated},${ms_per_iter},${rows_per_sec},${bench_csv}"

  if [[ "$status" == "PASS" ]]; then
    TOTAL_OK=$((TOTAL_OK + 1))
  else
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
  fi

  # Per-case metadata
  {
    echo "case_id=${case_id}"
    echo "dataset=${dataset}"
    echo "artifact=${artifact}"
    echo "artifact_root=${artifact_root}"
    echo "sel_label=${sel_label}"
    echo "k=${k}"
    echo "threshold=${threshold}"
    echo "raw_path=${raw_path}"
    echo "status=${status}"
    echo "bench_csv=${bench_csv}"
  } > "${case_dir}/run_meta.txt"
done

# ==============================================================================
# run_meta.txt
# ==============================================================================
{
  echo "=== Exp4 P1 Fused-Only Filter+Aggregate Mini-Matrix — run_meta ==="
  echo ""
  echo "Date: $(date -Iseconds)"
  echo "Host: $(hostname)"
  echo "GPU: ${GPU_NAME}"
  echo "Job ID: ${JOB_ID}"
  echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo ""
  echo "=== Toolchain ==="
  echo "nvcc: $($(which nvcc) --version 2>&1 | grep release || echo unknown)"
  echo "gcc: $(gcc --version 2>&1 | head -1)"
  echo "cmake: $(cmake --version 2>&1 | head -1)"
  echo ""
  echo "=== Kernel ==="
  echo "  fused: progressive_filter_sum_rowpack16_byte_mask"
  echo ""
  echo "=== Benchmark args ==="
  echo "  --iters 200 --warmup 10 --validate --block-threads 256"
  echo ""
  echo "=== Case matrix ==="
  for ((ci=0; ci<TOTAL_CASES; ci++)); do
    echo "  ${CASE_IDS[$ci]}: ${CASE_ARTIFACTS[$ci]} sel=${CASE_SEL_LABELS[$ci]} k=${CASE_KS[$ci]} threshold=${CASE_THRESHOLDS[$ci]}"
  done
  echo ""
  echo "=== Results ==="
  echo "  PASS: ${TOTAL_OK}   FAIL: ${TOTAL_FAIL}"
  if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "  Failures:"
    for f in "${FAILURES[@]}"; do
      echo "    - $f"
    done
  fi
  echo ""
  echo "Summary CSV: ${SUMMARY_CSV}"
  echo "Output dir: ${RESULT_DIR}"
} > "${RESULT_DIR}/run_meta.txt"

# ==============================================================================
# index.md
# ==============================================================================
cat > "${RESULT_DIR}/index.md" <<INDEXEOF
# Exp4 P1 Fused-Only Filter+Aggregate Mini-Matrix — Index

## Kernel

| Target | Kernel Name (regex) |
|--------|---------------------|
| fused | \`progressive_filter_sum_rowpack16_byte_mask\` |

## Cases

| Case | Artifact | Selectivity | k | Threshold | Dataset |
|------|----------|:-----------:|:-:|:---------:|---------|
| A | uniform_p10 | s≈50% | 1 | 500.027... | uniform |
| B | uniform_p10 | s≈50% | 6 | 500.027... | uniform |
| C | heavy_tailed_p6 | s≈50% | 8 | 1.000... | heavy_tailed |
| D | heavy_tailed_p6 | s≈99% | 8 | 8.903e-06 | heavy_tailed |

## Benchmark Args

\`\`\`
bench_filter_aggregate --dataset <artifact_root> --raw <raw> --threshold <t> \\
    --max-filter-planes <k> --iters 200 --warmup 10 --validate \\
    --block-threads 256 --csv <csv>
\`\`\`

## Result Layout

\`\`\`
${RESULT_DIR}/
  case_A/           — bench.csv, bench.log, run_meta.txt
  case_B/
  case_C/
  case_D/
  summary.csv       — 4-row status matrix
  run_meta.txt      — job metadata
  index.md          — this file
  gpu_util.log      — nvidia-smi GPU utilisation
\`\`\`
INDEXEOF

# ==============================================================================
# Final summary
# ==============================================================================
echo "" >&2
echo "=== Exp4 P1 Fused-Only Mini-Matrix Complete ===" >&2
echo "PASS: ${TOTAL_OK}  FAIL: ${TOTAL_FAIL}" >&2
echo "Summary: ${SUMMARY_CSV}" >&2
echo "Output: ${RESULT_DIR}" >&2
if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo "Failures:" >&2
  for f in "${FAILURES[@]}"; do
    echo "  ${f}" >&2
  done
fi

if [[ $TOTAL_FAIL -gt 0 ]]; then
  echo "One or more cases failed — exiting nonzero" >&2
  exit 1
fi

echo "Done" >&2
exit 0
