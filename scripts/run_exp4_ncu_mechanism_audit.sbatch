#!/usr/bin/env bash
# run_exp4_ncu_mechanism_audit.sbatch
#
# Exp4 NCU physical mechanism audit matrix.
#
# After P0/P1/P2 unblocked NCU for Exp4, this script runs 8 benchmark
# cases (with dedup) to answer:
#   1. Does smaller k (max-filter-planes) reduce physical traffic / metrics?
#   2. Is the bottleneck memory traffic, memory latency, scheduler/issue,
#      or ALU/control classification?
#   3. How do dataset difficulty and selectivity change the execution model?
#
# Usage:
#   sbatch --account=<ACCOUNT> scripts/run_exp4_ncu_mechanism_audit.sbatch
#
# Output:
#   results/exp4/ncu_mechanism_audit_<timestamp>_job<JOBID>_H200/
#     case_*/                    — per-case dir with CSV, .ncu-rep, logs
#     summary.csv                — combined results matrix
#     run_meta.txt               — metadata
#   handoff/job_done/job_<SLURM_JOB_ID>.json

#SBATCH --job-name=exp4-ncu-mechanism-audit
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL

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

# ==============================================================================
# Constants and paths
# ==============================================================================
V2_ROOT="/work/u4063895/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
RAW_ROOT="/work/u4063895/datasets/synthetic/dev"

timestamp=$(date +%Y%m%d_%H%M%S)
job_id="${SLURM_JOB_ID:-local}"
GPU_TAG="unknown"

handoff_dir="${ROOT_DIR}/handoff/job_done"
handoff_marker="${handoff_dir}/job_${job_id}.json"

# Artifact suffix per dataset
declare -A DATASET_P
DATASET_P[uniform]="p10"
DATASET_P[heavy_tailed]="p6"
DATASET_P[zipfian]="p8"

# ==============================================================================
# Threshold computation cache (numpy quantile on raw FP64)
# ==============================================================================
declare -A THRESHOLD_CACHE

compute_threshold() {
  local dataset="$1" selectivity="$2"
  local cache_key="${dataset}_s${selectivity}"
  if [[ -v "THRESHOLD_CACHE[$cache_key]" ]]; then
    printf '%s\n' "${THRESHOLD_CACHE[$cache_key]}"
    return 0
  fi
  local raw_path="${RAW_ROOT}/${dataset}.f64le.bin"
  if [[ ! -f "$raw_path" ]]; then
    echo "ERROR: raw file not found: $raw_path" >&2
    exit 2
  fi
  local quantile threshold
  quantile=$(python3 -c "print(1 - ${selectivity}/100.0)")
  threshold=$(python3 -c "
import numpy as np
data = np.fromfile('${raw_path}', dtype=np.float64)
print(np.quantile(data, ${quantile}))
")
  THRESHOLD_CACHE["$cache_key"]="$threshold"
  printf '%s\n' "$threshold"
}

# ==============================================================================
# Case definitions
# ==============================================================================
# Columns: case_id,role,dataset,selectivity,k_label,max_filter_planes
#   k_label="k1" when max_filter_planes=1; "full" when unset
#   max_filter_planes = integer or empty string for full depth
#   role is a human-readable description

declare -a CASE_IDS=()
declare -a CASE_ROLES=()
declare -a CASE_DATASETS=()
declare -a CASE_SELS=()
declare -a CASE_KLABELS=()
declare -a CASE_MFPS=()   # max-filter-planes; empty-string means no limit

register_case() {
  CASE_IDS+=("$1")
  CASE_ROLES+=("$2")
  CASE_DATASETS+=("$3")
  CASE_SELS+=("$4")
  CASE_KLABELS+=("$5")
  CASE_MFPS+=("$6")
}

# A. Depth effect on easy dataset
register_case "A1" "easy_depth_k1"       "uniform"      50 "k1"   "1"
register_case "A2" "easy_depth_full"     "uniform"      50 "full" ""

# B. Depth effect on hard dataset
register_case "B1" "hard_depth_k1"       "heavy_tailed" 99 "k1"   "1"
register_case "B2" "hard_depth_full"     "heavy_tailed" 99 "full" ""

# C. Selectivity effect at full depth on hard dataset
register_case "C1" "hard_sel_1_full"     "heavy_tailed"  1 "full" ""
register_case "C2" "hard_sel_50_full"    "heavy_tailed" 50 "full" ""
# C3 = same as B2 (heavy_tailed sel=99 full) — dedup handled at run time
register_case "C3" "hard_sel_99_full"    "heavy_tailed" 99 "full" ""

# D. Dataset effect at full depth / high selectivity
register_case "D1" "zipfian_sel_99_full" "zipfian"      99 "full" ""

TOTAL_CASES=${#CASE_IDS[@]}  # 8, but C3 is a dedup of B2

# ==============================================================================
# Handoff marker via trap (set before GPU validation so exit 2 also writes)
# ==============================================================================
write_handoff_marker() {
  local status=$?
  local result_dir_str=""
  local summary_csv=""
  if [[ -d "$run_dir" ]]; then
    result_dir_str="$run_dir"
    summary_csv="${run_dir}/summary.csv"
  fi
  local git_commit
  git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
  local next_action="Review summary.csv for per-case NCU status and metrics."
  if [[ "$status" -ne 0 ]]; then
    next_action="Job failed with exit code ${status}; investigate slurm output."
  fi
  mkdir -p "$handoff_dir"
  cat > "$handoff_marker" <<EOF
{
  "job_id": "${job_id}",
  "job_name": "exp4-ncu-mechanism-audit",
  "exit_status": ${status},
  "finished_at": "$(date -Is)",
  "workdir": "$(pwd)",
  "git_commit": "${git_commit}",
  "result_dir": "${result_dir_str}",
  "summary_csv": "${summary_csv}",
  "next_action": "${next_action}"
}
EOF
}
trap write_handoff_marker EXIT

# ==============================================================================
# GPU identification
# ==============================================================================
GPU_NAME="unknown"
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ' 2>/dev/null || echo "unknown")
  GPU_TAG="${GPU_NAME// /_}"
  GPU_TAG="${GPU_TAG:-unknown}"
fi
run_dir="${ROOT_DIR}/results/exp4/ncu_mechanism_audit_${timestamp}_job${job_id}_${GPU_TAG}"

# ==============================================================================
# Hardware validation
# ==============================================================================
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi
mkdir -p "$run_dir" "$handoff_dir"

# ==============================================================================
# Environment & build
# ==============================================================================
module purge || true
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan

echo "=== Toolchain ===" >&2
echo "nvcc: $(which nvcc) $($(which nvcc) --version 2>&1 | grep release)" >&2
echo "gcc: $(gcc --version 2>&1 | head -1)" >&2
echo "cmake: $(cmake --version 2>&1 | head -1)" >&2

NCU_BIN=""
if command -v ncu &>/dev/null; then
  NCU_BIN=$(which ncu)
elif [[ -x /work/envstack/apps/cuda/12.6/bin/ncu ]]; then
  NCU_BIN=/work/envstack/apps/cuda/12.6/bin/ncu
fi
if [[ -n "$NCU_BIN" ]]; then
  echo "ncu: $NCU_BIN $($NCU_BIN --version 2>&1 | head -1)" >&2
else
  echo "WARNING: ncu not found" >&2
fi

echo "=== Job info ===" >&2
echo "Job ID: ${job_id}  Host: $(hostname)  GPU: ${GPU_NAME}" >&2
echo "Output: ${run_dir}" >&2
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader >&2
git rev-parse HEAD >&2
git status --short >&2
echo "" >&2

echo "=== Build ===" >&2
cmake -S benchmarks/experiment4 -B build/exp4 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 2>&1
cmake --build build/exp4 -j 2>&1
echo "" >&2

BENCH_BIN="${ROOT_DIR}/build/exp4/bench_progressive_filter"

# ==============================================================================
# Summary CSV helper
# ==============================================================================
summary_csv="${run_dir}/summary.csv"

write_summary_header() {
  # Write once at the top of the file
  echo "case_id,role,dataset,artifact_subdir,selectivity,k_label,max_filter_planes,threshold,"\
"baseline_validated,gpu_count,cpu_encoded_count,ms_per_iter,logical_bytes,logical_GBps,"\
"estimated_physical_GBps,estimated_pack_load_bytes,rows_per_sec,"\
"ncu_status,report_path,report_size_bytes" > "$summary_csv"
}

append_summary_row() {
  # All arguments are comma-separated values already properly formatted
  echo "$1" >> "$summary_csv"
}

# ==============================================================================
# Run matrix
# ==============================================================================
write_summary_header

# Cache for dedup: key = dataset|subdir|sel|mfp → report info
declare -A NCU_CACHE_REPORT
declare -A NCU_CACHE_STATUS

total_ok=0
total_fail=0
FAILURES=()

shared_bench_args=(
  --device 0
  --block 256
  --validate
  --warp-strategy passive
  --warmup 0
  --iters 20
)

for ((idx=0; idx<TOTAL_CASES; idx++)); do
  case_id="${CASE_IDS[$idx]}"
  role="${CASE_ROLES[$idx]}"
  dataset="${CASE_DATASETS[$idx]}"
  sel="${CASE_SELS[$idx]}"
  k_label="${CASE_KLABELS[$idx]}"
  mfp="${CASE_MFPS[$idx]}"

  artifact_p="${DATASET_P[$dataset]}"
  encoded_subdir="${dataset}_${artifact_p}"
  encoded_root="${V2_ROOT}/${encoded_subdir}"

  # Derive threshold
  threshold=$(compute_threshold "$dataset" "$sel")

  # Build cache key for dedup
  # empty mfp → "full"
  cache_mfp="${mfp:-full}"
  dedup_key="${dataset}|${encoded_subdir}|${sel}|${cache_mfp}"

  case_dir="${run_dir}/case_${case_id}"
  mkdir -p "$case_dir"

  echo "" >&2
  echo "============================================================" >&2
  echo "[${case_id}/${role}] dataset=${dataset} artifact=${encoded_subdir} sel=${sel} k=${k_label}" >&2
  echo "============================================================" >&2
  echo "  threshold=${threshold}" >&2

  # ==================================================================
  # Check dedup cache first
  # ==================================================================
  if [[ -v "NCU_CACHE_REPORT[$dedup_key]" ]]; then
    echo "  DEDUP: reusing results from ${NCU_CACHE_REPORT[$dedup_key]}" >&2
    cached_report="${NCU_CACHE_REPORT[$dedup_key]}"
    cached_status="${NCU_CACHE_STATUS[$dedup_key]}"
    echo "${cached_status}" > "${case_dir}/ncu_status.txt"
  fi

  # ==================================================================
  # Baseline validation
  # ==================================================================
  baseline_csv="${case_dir}/baseline.csv"
  baseline_log="${case_dir}/baseline.log"

  if [[ ! -f "$baseline_csv" ]]; then
    set +e
    if [[ -n "$mfp" ]]; then
      "$BENCH_BIN" \
        "${shared_bench_args[@]}" \
        --encoded-root "$encoded_root" \
        --threshold "$threshold" \
        --max-filter-planes "$mfp" \
        --csv "$baseline_csv" \
        2> "$baseline_log"
    else
      "$BENCH_BIN" \
        "${shared_bench_args[@]}" \
        --encoded-root "$encoded_root" \
        --threshold "$threshold" \
        --csv "$baseline_csv" \
        2> "$baseline_log"
    fi
    baseline_rc=$?
    set -e
  else
    baseline_rc=0
  fi

  # Parse baseline CSV
  baseline_validated="false"
  gpu_count="NA"
  cpu_encoded="NA"
  ms_per_iter="NA"
  logical_bytes="NA"
  logical_gbps="NA"
  estimated_physical_gbps="NA"
  estimated_pack_load_bytes="NA"
  rows_per_sec="NA"

  if [[ -f "$baseline_csv" ]]; then
    gpu_count=$(tail -1 "$baseline_csv" | cut -d',' -f12)
    cpu_encoded=$(tail -1 "$baseline_csv" | cut -d',' -f13)
    validated_col=$(tail -1 "$baseline_csv" | cut -d',' -f29)
    ms_per_iter=$(tail -1 "$baseline_csv" | cut -d',' -f11)
    logical_bytes=$(tail -1 "$baseline_csv" | cut -d',' -f21)
    logical_gbps=$(tail -1 "$baseline_csv" | cut -d',' -f22)
    estimated_physical_gbps=$(tail -1 "$baseline_csv" | cut -d',' -f23)
    estimated_pack_load_bytes=$(tail -1 "$baseline_csv" | cut -d',' -f24)
    rows_per_sec=$(tail -1 "$baseline_csv" | cut -d',' -f25)

    if [[ "$gpu_count" == "$cpu_encoded" ]] && [[ "$validated_col" == "true" ]]; then
      baseline_validated="true"
      echo "  ✓ Baseline: gpu=${gpu_count} cpu=${cpu_encoded} validated" >&2
    else
      echo "  ✗ Baseline FAIL: gpu=${gpu_count} cpu=${cpu_encoded} validated=${validated_col}" >&2
    fi
  else
    echo "  ✗ Baseline CSV not found (rc=${baseline_rc})" >&2
  fi

  # ==================================================================
  # NCU profile (skip if dedup cache hit)
  # ==================================================================
  ncu_report_path=""
  ncu_report_size=""
  ncu_status="not_run"
  ncu_export_base="${case_dir}/ncu_profile"

  if [[ -v "NCU_CACHE_REPORT[$dedup_key]" ]]; then
    # Dedup: reuse cached result
    ncu_report_path="${NCU_CACHE_REPORT[$dedup_key]}"
    ncu_report_size=$(stat --printf='%s' "$ncu_report_path" 2>/dev/null || echo "0")
    ncu_status="${NCU_CACHE_STATUS[$dedup_key]}"
    echo "  DEDUP: ncu_status=${ncu_status}" >&2
  else
    # Actually run NCU
    ncu_stdout="${case_dir}/ncu_stdout.txt"
    ncu_stderr="${case_dir}/ncu_stderr.txt"
    ncu_report="${ncu_export_base}.ncu-rep"

    if [[ -z "$NCU_BIN" ]]; then
      echo "  NCU not available" >&2
      ncu_status="FAIL: no NCU binary"
    else
      set +e
      if [[ -n "$mfp" ]]; then
        "$NCU_BIN" \
          --set basic \
          --target-processes all \
          --profile-from-start yes \
          --page raw \
          -f \
          -o "$ncu_export_base" \
          "$BENCH_BIN" \
            "${shared_bench_args[@]}" \
            --encoded-root "$encoded_root" \
            --threshold "$threshold" \
            --max-filter-planes "$mfp" \
            --csv "${case_dir}/ncu_bench.csv" \
          > "$ncu_stdout" 2> "$ncu_stderr"
      else
        "$NCU_BIN" \
          --set basic \
          --target-processes all \
          --profile-from-start yes \
          --page raw \
          -f \
          -o "$ncu_export_base" \
          "$BENCH_BIN" \
            "${shared_bench_args[@]}" \
            --encoded-root "$encoded_root" \
            --threshold "$threshold" \
            --csv "${case_dir}/ncu_bench.csv" \
          > "$ncu_stdout" 2> "$ncu_stderr"
      fi
      ncu_rc=$?
      set -e

      echo "  NCU exit code: ${ncu_rc}" >&2

      # Detect success
      if grep -q "No kernels were profiled" "$ncu_stdout" 2>/dev/null; then
        ncu_status="FAIL: No kernels were profiled"
        echo "  ✗ ${ncu_status}" >&2
      elif [[ -f "$ncu_report" ]] && [[ -s "$ncu_report" ]]; then
        ncu_report_size=$(stat --printf='%s' "$ncu_report")
        echo "  ✓ .ncu-rep: ${ncu_report_size} bytes" >&2
        ncu_status="PASS"
      else
        ncu_status="FAIL: report missing or empty"
        echo "  ✗ ${ncu_status}" >&2
      fi

      # Best-effort CSV export
      if [[ "$ncu_status" == "PASS" ]] && [[ -f "$ncu_report" ]]; then
        "$NCU_BIN" --csv --page raw \
          --force-overwrite true \
          -i "$ncu_report" \
          -o "${ncu_export_base}" \
          >> "$ncu_stdout" 2>> "$ncu_stderr" || true
      fi

      ncu_report_path="$ncu_report"
    fi

    # Cache for dedup
    NCU_CACHE_REPORT["$dedup_key"]="$ncu_report_path"
    NCU_CACHE_STATUS["$dedup_key"]="$ncu_status"
  fi

  # Write per-case status
  echo "$ncu_status" > "${case_dir}/ncu_status.txt"
  echo "$ncu_report_path" > "${case_dir}/ncu_report_path.txt"

  # Track failures
  if [[ "$ncu_status" == "PASS" ]]; then
    total_ok=$((total_ok + 1))
  else
    total_fail=$((total_fail + 1))
    FAILURES+=("${case_id}: ${ncu_status}")
  fi

  # ==================================================================
  # Append to summary CSV
  # ==================================================================
  artifact_subdir="${encoded_subdir}"
  mfp_display="${mfp:-full}"
  # Shell-safe CSV quoting: fields that might contain commas are already
  # known to be simple values (case_id, role, dataset etc. have no commas).
  # ncu_report_path might have no commas either since it's an absolute path.
  append_summary_row \
"${case_id},${role},${dataset},${artifact_subdir},${sel},${k_label},${mfp_display},${threshold},"\
"${baseline_validated},${gpu_count},${cpu_encoded},${ms_per_iter},${logical_bytes},${logical_gbps},"\
"${estimated_physical_gbps},${estimated_pack_load_bytes},${rows_per_sec},"\
"${ncu_status},${ncu_report_path},${ncu_report_size}"

done

# ==============================================================================
# run_meta.txt
# ==============================================================================
{
  echo "=== Exp4 NCU Mechanism Audit — run_meta ==="
  echo ""
  echo "Date: $(date -Is)"
  echo "Host: $(hostname)"
  echo "GPU: ${GPU_NAME}"
  echo "Job ID: ${job_id}"
  echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo ""
  echo "=== Toolchain ==="
  echo "nvcc: $($(which nvcc) --version 2>&1 | grep release || echo unknown)"
  echo "gcc: $(gcc --version 2>&1 | head -1)"
  echo "cmake: $(cmake --version 2>&1 | head -1)"
  echo "ncu: $($NCU_BIN --version 2>&1 | head -1 || echo unknown)"
  echo ""
  echo "=== NCU command shape ==="
  echo "  --set basic --target-processes all --profile-from-start yes"
  echo "  --page raw -f"
  echo "  NO launch-count, NO launch-skip, NO replay-mode"
  echo ""
  echo "=== Case matrix ==="
  echo "Total cases (incl. dedup): ${TOTAL_CASES}"
  for ((idx=0; idx<TOTAL_CASES; idx++)); do
    dataset="${CASE_DATASETS[$idx]}"
    sel="${CASE_SELS[$idx]}"
    k_label="${CASE_KLABELS[$idx]}"
    echo "  ${CASE_IDS[$idx]} (${CASE_ROLES[$idx]}): ${dataset} sel=${sel} k=${k_label}"
  done
  echo ""
  echo "=== Dedup ==="
  echo "  C3 (heavy_tailed sel=99 full) == B2 — second occurrence reuses same report"
  echo ""
  echo "=== Results ==="
  echo "  PASS: ${total_ok}   FAIL: ${total_fail}"
  if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "  Failures:"
    for f in "${FAILURES[@]}"; do
      echo "    - $f"
    done
  fi
  echo ""
  echo "Summary CSV: ${summary_csv}"
  echo "Output dir: ${run_dir}"
} > "${run_dir}/run_meta.txt"

echo "" >&2
echo "=== Exp4 NCU Mechanism Audit Complete ===" >&2
echo "PASS: ${total_ok}  FAIL: ${total_fail}" >&2
echo "Summary: ${summary_csv}" >&2
if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo "Failures:" >&2
  for f in "${FAILURES[@]}"; do
    echo "  ${f}" >&2
  done
fi

# Exit nonzero only if ALL cases failed
if [[ $total_ok -eq 0 ]] && [[ $total_fail -gt 0 ]]; then
  echo "All cases failed — exiting nonzero" >&2
  exit 1
fi

echo "✓ Done" >&2
exit 0
