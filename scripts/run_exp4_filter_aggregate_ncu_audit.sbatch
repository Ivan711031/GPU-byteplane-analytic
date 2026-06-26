#!/usr/bin/env bash
# run_exp4_filter_aggregate_ncu_audit.sbatch
#
# Exp4 Filter+Aggregate NCU bottleneck audit.
# Profiles 4 cases × 3 kernels = 12 .ncu-rep captures with --set full.
#
# Cases:
#   A: uniform_p10, s≈50%, k=1
#   B: uniform_p10, s≈50%, k=6
#   C: heavy_tailed_p6, s≈50%, k=8
#   D: heavy_tailed_p6, s≈99%, k=8
#
# Kernels:
#   fused       = progressive_filter_sum_rowpack16_byte_mask
#   count_only  = progressive_filter_count_rowpack16_byte_mask
#   raw_fp64    = raw_fp64_filter_sum
#
# Usage: pass the allocation on the sbatch command line when submitting.
#
# Output:
#   results/exp4_filter_aggregate/ncu_audit_<timestamp>_job<JOBID>_NVIDIAH200/
#     case_<id>_<kernel>/ — per-capture dir with .ncu-rep, ncu_stdout.txt, ncu_stderr.txt, bench.csv
#     summary.csv, run_meta.txt, index.md, gpu_util.log
#   handoff/job_done/job_<SLURM_JOB_ID>.json

#SBATCH --job-name=exp4-fa-ncu
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=results/exp4_filter_aggregate/ncu_audit_%j.out
#SBATCH --error=results/exp4_filter_aggregate/ncu_audit_%j.err

set -euo pipefail

# ==============================================================================
# Dynamic workspace detection & result directory (before trap for handoff coverage)
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

# GPU name queried early so result_dir uses it
GPU_NAME="unknown"
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ' 2>/dev/null || echo "unknown")
fi
GPU_TAG="${GPU_NAME// /_}"

RESULT_DIR="results/exp4_filter_aggregate/ncu_audit_${RUN_TS}_job${JOB_ID}_${GPU_TAG}"
mkdir -p "$RESULT_DIR"

STDOUT_PATH="results/exp4_filter_aggregate/ncu_audit_${JOB_ID}.out"
STDERR_PATH="results/exp4_filter_aggregate/ncu_audit_${JOB_ID}.err"
GPU_UTIL_LOG="${RESULT_DIR}/gpu_util.log"

# Start GPU util log in background
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv -l 2 > "$GPU_UTIL_LOG" 2>&1 &
GPU_UTIL_PID=$!

# ==============================================================================
# Handoff marker via trap (registered early for fail-fast coverage)
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
  local next_action="Review summary.csv and index.md for per-case NCU status and bottleneck analysis."
  if [[ "$status" -ne 0 ]]; then
    next_action="Job failed with exit code ${status}; investigate slurm output."
  fi
  cat > "$HANDOFF_MARKER" <<HEREDOC
{
  "exit_status": $status,
  "job_id": "${JOB_ID}",
  "job_name": "exp4-fa-ncu-audit",
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
# H200 fail-fast (after trap so handoff is written on mismatch)
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

# Locate ncu
NCU_BIN=""
if command -v ncu &>/dev/null; then
  NCU_BIN="$(command -v ncu)"
elif [[ -x /work/envstack/apps/cuda/12.6/bin/ncu ]]; then
  NCU_BIN="/work/envstack/apps/cuda/12.6/bin/ncu"
fi
if [[ -n "$NCU_BIN" ]]; then
  echo "ncu: ${NCU_BIN} $($NCU_BIN --version 2>&1 | head -1)" >&2
else
  echo "ERROR: ncu not found after loading CUDA module" >&2
  exit 3
fi

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
# Case definitions
# ==============================================================================
# Syntax: case_id|dataset_artifact|selectivity_label|k|artifact_root_suffix|threshold
#
# Raw file paths follow the convention /work/$USER/datasets/synthetic/dev/<dataset>.f64le.bin
# Artifact roots are under /work/$USER/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p/
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

# Kernel targets for NCU --kernel-name regex
declare -a KERNEL_TARGETS=("fused" "count_only" "raw_fp64")
declare -A KERNEL_REGEX
KERNEL_REGEX[fused]="progressive_filter_sum_rowpack16_byte_mask"
KERNEL_REGEX[count_only]="progressive_filter_count_rowpack16_byte_mask"
KERNEL_REGEX[raw_fp64]="raw_fp64_filter_sum"

# ==============================================================================
# Summary CSV
# ==============================================================================
SUMMARY_CSV="${RESULT_DIR}/summary.csv"
{
  echo "case_id,kernel_target,dataset_artifact,dataset,selectivity_label,k,threshold,kernel_regex,status,report_path,bench_csv_path"
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

  for kernel_target in "${KERNEL_TARGETS[@]}"; do
    kernel_regex="${KERNEL_REGEX[$kernel_target]}"
    case_dir="${RESULT_DIR}/case_${case_id}_${kernel_target}"
    mkdir -p "$case_dir"

    echo "" >&2
    echo "============================================================" >&2
    echo "[${case_id}/${kernel_target}] dataset=${dataset} artifact=${artifact} sel=${sel_label} k=${k}" >&2
    echo "  threshold=${threshold}" >&2
    echo "  kernel_regex=${kernel_regex}" >&2
    echo "============================================================" >&2

    # --- Baseline validation (1 iter, no warmup) ---
    bench_csv="${case_dir}/bench.csv"
    bench_log="${case_dir}/bench.log"

    set +e
    "$BENCH_BIN" \
      --dataset "$artifact_root" \
      --raw "$raw_path" \
      --threshold "$threshold" \
      --max-filter-planes "$k" \
      --iters 1 \
      --warmup 0 \
      --validate \
      --block-threads 256 \
      --csv "$bench_csv" \
      > "$bench_log" 2>&1
    bench_rc=$?
    set -e

    if [[ $bench_rc -ne 0 ]]; then
      echo "  ✗ Benchmark failed (rc=${bench_rc})" >&2
      FAILURES+=("${case_id}_${kernel_target}: bench rc=${bench_rc}")
      TOTAL_FAIL=$((TOTAL_FAIL + 1))
      append_summary_row \
        "${case_id},${kernel_target},${artifact},${dataset},${sel_label},${k},${threshold},${kernel_regex},FAIL_BENCH,,${bench_csv}"
      continue
    fi

    # Quick validation check (header-aware column lookup)
    if [[ -f "$bench_csv" ]]; then
      # Parse header to find column indices for gpu_count, cpu_enc_count, validated
      header_cols=$(head -1 "$bench_csv" | awk -F',' '{for(i=1;i<=NF;i++){print i,$i}}')
      col_gpu=$(echo "$header_cols" | awk '$2=="gpu_count"{print $1}')
      col_cpu=$(echo "$header_cols" | awk '$2=="cpu_enc_count"{print $1}')
      col_val=$(echo "$header_cols" | awk '$2=="validated"{print $1}')
      if [[ -n "$col_gpu" ]] && [[ -n "$col_cpu" ]] && [[ -n "$col_val" ]]; then
        gpu_count=$(tail -1 "$bench_csv" | cut -d',' -f"$col_gpu" 2>/dev/null || echo "NA")
        cpu_enc=$(tail -1 "$bench_csv" | cut -d',' -f"$col_cpu" 2>/dev/null || echo "NA")
        validated=$(tail -1 "$bench_csv" | cut -d',' -f"$col_val" 2>/dev/null || echo "false")
        if [[ "$gpu_count" == "$cpu_enc" ]] && [[ "$validated" == "true" ]]; then
          echo "  ✓ Baseline validated: gpu=${gpu_count} cpu=${cpu_enc}" >&2
        else
          echo "  ✗ Baseline validation failed: gpu=${gpu_count} cpu=${cpu_enc} validated=${validated}" >&2
        fi
      fi
    fi

    # --- NCU profile ---
    ncu_stdout="${case_dir}/ncu_stdout.txt"
    ncu_stderr="${case_dir}/ncu_stderr.txt"
    ncu_base="${case_dir}/ncu_profile"
    ncu_report="${ncu_base}.ncu-rep"

    set +e
    "$NCU_BIN" \
      --set full \
      --target-processes all \
      --profile-from-start yes \
      --kernel-name "regex:${kernel_regex}" \
      --page raw \
      -f \
      -o "$ncu_base" \
      "$BENCH_BIN" \
        --dataset "$artifact_root" \
        --raw "$raw_path" \
        --threshold "$threshold" \
        --max-filter-planes "$k" \
        --iters 1 \
        --warmup 0 \
        --validate \
        --block-threads 256 \
        --csv "${case_dir}/ncu_bench.csv" \
      > "$ncu_stdout" 2> "$ncu_stderr"
    ncu_rc=$?
    set -e

    echo "  NCU exit code: ${ncu_rc}" >&2

    if [[ -f "$ncu_report" ]] && [[ -s "$ncu_report" ]]; then
      ncu_report_size=$(stat --printf='%s' "$ncu_report" 2>/dev/null || echo "0")
      echo "  ✓ .ncu-rep: ${ncu_report_size} bytes" >&2
      ncu_status="PASS"
      ncu_report_path="$ncu_report"
      TOTAL_OK=$((TOTAL_OK + 1))
    elif grep -q "No kernels were profiled" "$ncu_stdout" 2>/dev/null; then
      echo "  ✗ No kernels were profiled" >&2
      ncu_status="FAIL: no kernels profiled"
      ncu_report_path=""
      TOTAL_FAIL=$((TOTAL_FAIL + 1))
      FAILURES+=("${case_id}_${kernel_target}: no kernels profiled")
    else
      echo "  ✗ .ncu-rep missing or empty (rc=${ncu_rc})" >&2
      ncu_status="FAIL: report missing (rc=${ncu_rc})"
      ncu_report_path=""
      TOTAL_FAIL=$((TOTAL_FAIL + 1))
      FAILURES+=("${case_id}_${kernel_target}: report missing")
    fi

    append_summary_row \
      "${case_id},${kernel_target},${artifact},${dataset},${sel_label},${k},${threshold},${kernel_regex},${ncu_status},${ncu_report_path},${case_dir}/bench.csv"

    # Write per-case metadata
    {
      echo "case_id=${case_id}"
      echo "kernel_target=${kernel_target}"
      echo "kernel_regex=${kernel_regex}"
      echo "dataset=${dataset}"
      echo "artifact=${artifact}"
      echo "artifact_root=${artifact_root}"
      echo "sel_label=${sel_label}"
      echo "k=${k}"
      echo "threshold=${threshold}"
      echo "raw_path=${raw_path}"
      echo "ncu_status=${ncu_status}"
      echo "ncu_report=${ncu_report_path}"
      echo "bench_csv=${case_dir}/bench.csv"
    } > "${case_dir}/run_meta.txt"
  done
done

# ==============================================================================
# run_meta.txt
# ==============================================================================
{
  echo "=== Exp4 Filter+Aggregate NCU Bottleneck Audit — run_meta ==="
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
  echo "ncu: $($NCU_BIN --version 2>&1 | head -1 || echo unknown)"
  echo ""
  echo "=== NCU command shape ==="
  echo "  --set full --target-processes all --profile-from-start yes"
  echo "  --kernel-name regex:<kernel> --page raw -f"
  echo "  No kernel replay or launch limiter options are used."
  echo ""
  echo "=== Case matrix ==="
  for ((ci=0; ci<TOTAL_CASES; ci++)); do
    echo "  ${CASE_IDS[$ci]}: ${CASE_ARTIFACTS[$ci]} sel=${CASE_SEL_LABELS[$ci]} k=${CASE_KS[$ci]} threshold=${CASE_THRESHOLDS[$ci]}"
  done
  echo ""
  echo "=== Kernel targets ==="
  for kt in "${KERNEL_TARGETS[@]}"; do
    echo "  ${kt}: ${KERNEL_REGEX[$kt]}"
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
# Exp4 Filter+Aggregate NCU Bottleneck Audit — Index

## Cases

| Case | Artifact | Selectivity | k | Threshold | Dataset |
|------|----------|:-----------:|:-:|:---------:|---------|
| A | uniform_p10 | s≈50% | 1 | 500.027... | uniform |
| B | uniform_p10 | s≈50% | 6 | 500.027... | uniform |
| C | heavy_tailed_p6 | s≈50% | 8 | 1.000... | heavy_tailed |
| D | heavy_tailed_p6 | s≈99% | 8 | 8.903e-06 | heavy_tailed |

## Kernels Profiled

| Target | Kernel Name (regex) |
|--------|---------------------|
| fused | \`progressive_filter_sum_rowpack16_byte_mask\` |
| count_only | \`progressive_filter_count_rowpack16_byte_mask\` |
| raw_fp64 | \`raw_fp64_filter_sum\` |

## NCU Command Shape

\`\`\`
ncu --set full --target-processes all --profile-from-start yes \\
    --kernel-name regex:<kernel> --page raw -f \\
    -o <base> <bench> --dataset ... --raw ... --threshold ... \\
    --max-filter-planes <k> --validate --iters 1 --warmup 0 \\
    --block-threads 256 --csv <csv>
\`\`\`

No kernel replay or launch limiter options are used.

## Result Layout

\`\`\`
${RESULT_DIR}/
  case_A_fused/         — .ncu-rep, ncu_stdout.txt, ncu_stderr.txt, bench.csv, run_meta.txt
  case_A_count_only/
  case_A_raw_fp64/
  case_B_fused/
  ... (12 directories total)
  summary.csv           — 12-row status matrix
  run_meta.txt          — job metadata
  index.md              — this file
  gpu_util.log          — nvidia-smi GPU utilisation
\`\`\`

## Analysis Questions

1. What are the NCU bottleneck classifications for each of the three
   kernel strategies (fused, count_only, raw_fp64) on uniform_p10 at
   k=1 (case A)?

2. How does increasing k from 1 to 6 change the bottleneck profile of
   the fused kernel on uniform_p10 (case A → case B)?

3. What are the NCU bottleneck classifications on heavy_tailed_p6 at
   s≈50%, k=8 (case C)?

4. How does moving from s≈50% to s≈99% on heavy_tailed_p6 at k=8
   change the bottleneck profile (case C → case D)?

5. What is the raw_fp64_filter_sum bottleneck classification compared
   to the byte-plane fused kernel across cases?

6. What is the difference in bottleneck patterns between the full
   filter+aggregate (fused) kernel and the count-only kernel?
INDEXEOF

# ==============================================================================
# Final summary
# ==============================================================================
echo "" >&2
echo "=== Exp4 Filter+Aggregate NCU Bottleneck Audit Complete ===" >&2
echo "PASS: ${TOTAL_OK}  FAIL: ${TOTAL_FAIL}" >&2
echo "Summary: ${SUMMARY_CSV}" >&2
echo "Output: ${RESULT_DIR}" >&2
if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo "Failures:" >&2
  for f in "${FAILURES[@]}"; do
    echo "  ${f}" >&2
  done
fi

if [[ $TOTAL_OK -eq 0 ]] && [[ $TOTAL_FAIL -gt 0 ]]; then
  echo "All cases failed — exiting nonzero" >&2
  exit 1
fi

echo "Done" >&2
exit 0
