#!/usr/bin/env bash
#SBATCH -p dev
#SBATCH -J v1_3_ncu_freeze
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -o results/v1_3_freeze/ncu/v1_3_ncu_freeze_%j.out
#SBATCH -e results/v1_3_freeze/ncu/v1_3_ncu_freeze_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

# Derive main repo root for cross-worktree data access
if MAIN_REPO_DIR="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)"/.. 2>/dev/null && pwd)"; then
  : # derived successfully
else
  MAIN_REPO_DIR="$ROOT_DIR"
fi

# Ensure dataset paths accessible in worktree
[[ -L datasets/scientific ]] || ln -sf /work/${USER:?USER unset}/datasets/scientific datasets/scientific 2>/dev/null || true
[[ -f datasets/scientific/dev_buff_v2_scientific_quasi_global/cesm_atm_q/bfp-dec12/manifest.json ]] || {
  echo "ERROR: dataset manifest not found" >&2; exit 1
}
[[ -d results/buff_encoder_v2/raw_scientific ]] || mkdir -p results/buff_encoder_v2
[[ -f results/buff_encoder_v2/raw_scientific/cesm_atm_q.f64le.bin ]] || \
  ln -sf "$MAIN_REPO_DIR/results/buff_encoder_v2/raw_scientific" results/buff_encoder_v2/raw_scientific 2>/dev/null || true
[[ -f results/buff_encoder_v2/raw_scientific/cesm_atm_q.f64le.bin ]] || {
  echo "ERROR: raw scientific data not found" >&2; exit 1
}

RUN_TS="$(date +%Y%m%d_%H%M%S)"
JOB_ID="${SLURM_JOB_ID:-manual}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | python3 -c 'import sys; print(sys.stdin.readline().strip())')"
GPU_TAG="${GPU_NAME// /}"

case "$GPU_NAME" in
  *H200*) ;;
  *) echo "ERROR: Expected H200 GPU, got: $GPU_NAME" >&2; exit 2 ;;
esac

RESULT_DIR="results/v1_3_freeze/ncu/v1_3_ncu_${RUN_TS}_job${JOB_ID}_${GPU_TAG}"
SUMMARY_CSV="$RESULT_DIR/v1_3_ncu_summary.csv"
mkdir -p "$RESULT_DIR" "$ROOT_DIR/handoff/job_done"

module purge
module load miniconda3/26.1.1 2>/dev/null || module load miniconda3
module load cuda/12.6 2>/dev/null || module load cuda
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan 2>/dev/null || conda activate gpu-byteplane-rapids 2>/dev/null || true

NCU_BIN="$(command -v ncu)"
echo "NCU: $NCU_BIN $("$NCU_BIN" --version 2>&1 | head -1)"
echo "GPU: $GPU_NAME"
echo "Result dir: $RESULT_DIR"

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 1 > "$RESULT_DIR/gpu_util.csv" 2>/dev/null &
GPU_MONITOR_PID=$!

write_completion_marker() {
  local status=$?
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
  local git_commit
  git_commit="$(git rev-parse HEAD 2>/dev/null || true)"
  cat > "$ROOT_DIR/handoff/job_done/job_${JOB_ID}.json" <<EOF
{
  "job_id": "$JOB_ID",
  "job_name": "v1_3_ncu_freeze",
  "exit_status": $status,
  "finished_at": "$(date -Is)",
  "workdir": "$ROOT_DIR",
  "git_commit": "$git_commit",
  "gpu": "$GPU_NAME",
  "result_dir": "$RESULT_DIR",
  "summary_csv": "$SUMMARY_CSV"
}
EOF
}
trap write_completion_marker EXIT

# Create a minimal cmake project in a temp build dir to avoid polluting source tree
BUILD_DIR="$ROOT_DIR/build/filter_aggregate"
FA_SRC="$ROOT_DIR/benchmarks/experiment4_filter_aggregate"
CMAKE_SRC="$ROOT_DIR/build/filter_aggregate_cmake_src"
mkdir -p "$CMAKE_SRC" "$BUILD_DIR"
cat > "$CMAKE_SRC/CMakeLists.txt" << 'CMAKE_EOF'
cmake_minimum_required(VERSION 3.24)
project(gpu_byteplane_scan_filter_aggregate LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(_FA_CUDA_OPTIONS --use_fast_math -O3 -lineinfo --ptxas-options=-v,--warn-on-spills)
add_executable(bench_filter_aggregate
  bench_filter_aggregate.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../benchmarks/experiment3/exp3_real_data_layout.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../buff_encoder/buff_codec.cpp
)
target_include_directories(bench_filter_aggregate PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../benchmarks/experiment4_filter_aggregate
  ${CMAKE_CURRENT_SOURCE_DIR}/../../benchmarks/experiment3
  ${CMAKE_CURRENT_SOURCE_DIR}/../../benchmarks/experiment4
  ${CMAKE_CURRENT_SOURCE_DIR}/../../buff_encoder
)
target_compile_options(bench_filter_aggregate PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
set_target_properties(bench_filter_aggregate PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
CMAKE_EOF
ln -sf "$FA_SRC/bench_filter_aggregate.cu" "$CMAKE_SRC/bench_filter_aggregate.cu"
cmake -S "$CMAKE_SRC" -B "$BUILD_DIR" \
  -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"${SLURM_CPUS_ON_NODE:-4}" --target bench_filter_aggregate
BENCH="$BUILD_DIR/bench_filter_aggregate"

cat > "$SUMMARY_CSV" <<'EOF'
field,artifact_path,kernel_name,k,runtime_ms,logical_bytes_per_row_estimate,dram_bytes_read,l2_sectors,l1tex_sectors,sm_throughput_pct,dram_throughput_pct,long_scoreboard_ratio,eligible_warps_per_cycle,achieved_occupancy,report_path,bench_csv_path
EOF

csv_get() {
  local csv="$1" col="$2"
  python3 - "$csv" "$col" <<'PY'
import csv, sys
path, col = sys.argv[1], sys.argv[2]
with open(path, newline='') as f:
    rows = list(csv.DictReader(f))
if not rows: print("")
else: print(rows[-1].get(col, ""))
PY
}

append_summary() {
  local field="$1" artifact_path="$2" k_label="$3" kernel_name="$4"
  local ncu_stdout="$5" ncu_report="$6" bench_csv="$7"

  # NCU metric extraction — metric names and units are version-specific;
  # validated against ncu --version output (NVIDIA Nsight Compute CLI 2024.x+)
  python3 - "$field" "$artifact_path" "$k_label" "$kernel_name" "$ncu_stdout" "$ncu_report" "$bench_csv" "$SUMMARY_CSV" <<'PY'
import csv, re, sys

field, artifact, k_label, kernel, ncu_stdout, report, bench_csv, summary = sys.argv[1:]
text = open(ncu_stdout, errors="replace").read() if ncu_stdout else ""

def clean_num(s):
    s = s.strip().replace(",", "")
    if s in {"", "no", "data", "no data", "nan", "(!) nan"}: return ""
    m = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", s)
    return m.group(0) if m else ""

def metric_pair(name):
    for line in text.splitlines():
        if line.strip().startswith(name):
            tail = line.split(name, 1)[1].strip()
            parts = tail.split()
            if not parts: return "", ""
            return clean_num(parts[-1]), " ".join(parts[:-1])
    return "", ""

def metric(name):
    v, _ = metric_pair(name); return v

def bench_value(col):
    try:
        with open(bench_csv, newline="") as f: rows = list(csv.DictReader(f))
        return rows[-1].get(col, "") if rows else ""
    except: return ""

dur_val, dur_unit = metric_pair("gpu__time_duration.sum")
if not dur_val: dur_val, dur_unit = metric_pair("gpu__time_duration.avg")
runtime_ms = ""
if dur_val:
    scale = 1.0
    if dur_unit == "us": scale = 1.0 / 1000.0
    elif dur_unit == "ns": scale = 1.0 / 1_000_000.0
    elif dur_unit == "s": scale = 1000.0
    runtime_ms = f"{float(dur_val) * scale:.6f}"

n = bench_value("n")
logical = bench_value("logical_bytes")
if kernel == "raw_fp64_filter_sum": logical_bpr = "8.000000"
elif n and logical: logical_bpr = f"{float(logical) / float(n):.6f}"
else: logical_bpr = ""

dram_val, dram_unit = metric_pair("dram__bytes_read.sum")
dram_val2 = metric("dram__bytes_read.sum")
dram_sectors = metric("dram__sectors_read.sum")
if dram_val and dram_unit:
    dram_display = f"{dram_val} {dram_unit}"
elif dram_val2:
    dram_display = dram_val2
elif dram_sectors:
    dram_display = f"{dram_sectors} sectors"
else:
    dram_display = ""

row = {
    "field": field,
    "artifact_path": artifact,
    "kernel_name": kernel,
    "k": k_label,
    "runtime_ms": runtime_ms,
    "logical_bytes_per_row_estimate": logical_bpr,
    "dram_bytes_read": dram_display,
    "l2_sectors": metric("lts__t_sectors.sum"),
    "l1tex_sectors": metric("l1tex__t_sectors.sum") or metric("SM_B.TriageCompute.l1tex__t_sectors.sum"),
    "sm_throughput_pct": metric("sm__throughput.avg.pct_of_peak_sustained_elapsed") or metric("TPC.TriageCompute.sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed"),
    "dram_throughput_pct": metric("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed") or metric("dram__bytes_read.sum.pct_of_peak_sustained_elapsed"),
    "long_scoreboard_ratio": metric("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
    "eligible_warps_per_cycle": metric("smsp__warps_eligible.avg.per_cycle_active"),
    "achieved_occupancy": metric("sm__warps_active.avg.pct_of_peak_sustained_active"),
    "report_path": report,
    "bench_csv_path": bench_csv,
}

with open(summary, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    writer.writerow(row)
PY
}

run_case() {
  local field="$1" artifact="$2" raw="$3" threshold="$4" k_run="$5" k_label="$6" case_name="$7" kernel_regex="$8"
  local case_dir="$RESULT_DIR/${field}_${case_name}"
  mkdir -p "$case_dir"
  local bench_csv="$case_dir/bench.csv"
  local bench_log="$case_dir/bench.log"
  local ncu_base="$case_dir/ncu_profile"
  local ncu_stdout="$case_dir/ncu_stdout.txt"
  local ncu_stderr="$case_dir/ncu_stderr.txt"

  echo "[$(date -Is)] case=${field}/${case_name} threshold=${threshold} k_run=${k_run} kernel=${kernel_regex}"

  "$BENCH" \
    --dataset "$artifact" \
    --raw "$raw" \
    --threshold "$threshold" \
    --max-filter-planes "$k_run" \
    --iters 1 --warmup 0 --validate \
    --block-threads 256 \
    --csv "$bench_csv" > "$bench_log" 2>&1

  gpu_count="$(csv_get "$bench_csv" gpu_count)"
  cpu_enc="$(csv_get "$bench_csv" cpu_enc_count)"
  validated="$(csv_get "$bench_csv" validated)"
  if [[ "$gpu_count" != "$cpu_enc" || "$validated" != "true" ]]; then
    echo "ERROR: validation failed ${field}/${case_name}: gpu=${gpu_count} cpu=${cpu_enc} validated=${validated}" >&2
    exit 2
  fi

  "$NCU_BIN" \
    --set full \
    --target-processes all \
    --profile-from-start yes \
    --kernel-name "regex:${kernel_regex}" \
    --page raw -f \
    -o "$ncu_base" \
    "$BENCH" \
      --dataset "$artifact" \
      --raw "$raw" \
      --threshold "$threshold" \
      --max-filter-planes "$k_run" \
      --iters 1 --warmup 0 --validate \
      --block-threads 256 \
      --csv "$case_dir/ncu_bench.csv" \
    > "$ncu_stdout" 2> "$ncu_stderr"

  if [[ ! -s "${ncu_base}.ncu-rep" ]]; then
    echo "ERROR: missing NCU report ${field}/${case_name}" >&2; exit 3
  fi
  if grep -q "No kernels were profiled" "$ncu_stdout" "$ncu_stderr" 2>/dev/null; then
    echo "ERROR: NCU captured no kernels ${field}/${case_name}" >&2; exit 3
  fi

  append_summary "$field" "$artifact" "$k_label" "$kernel_regex" "$ncu_stdout" "${ncu_base}.ncu-rep" "$bench_csv"
}

V1_3_FIELDS="${V1_3_FIELDS:-cesm_atm_q}"

run_field_cesm_atm_q() {
  local field="cesm_atm_q"
  local artifact="datasets/scientific/dev_buff_v2_scientific_quasi_global/cesm_atm_q/bfp-dec12"
  local raw="results/buff_encoder_v2/raw_scientific/cesm_atm_q.f64le.bin"
  local threshold="0.0041057588532567024"
  local kmax
  kmax="$(python3 -c "import json; print(json.load(open('${artifact}/manifest.json'))['max_plane_count'])")"
  echo "cesm_atm_q: k2 threshold=${threshold} kmax=${kmax}"
  run_case "$field" "$artifact" "$raw" "$threshold" "2" "2" "s10_k2_fused" "progressive_filter_sum_rowpack16_byte_mask"
  run_case "$field" "$artifact" "$raw" "$threshold" "$kmax" "max" "s10_kmax_fused" "progressive_filter_sum_rowpack16_byte_mask"
  run_case "$field" "$artifact" "$raw" "$threshold" "$kmax" "raw" "s10_raw_fp64" "raw_fp64_filter_sum"
}

run_field_hurricane_tc() {
  local field="hurricane_tc"
  local artifact="datasets/scientific/dev_buff_v2_scientific_quasi_global/hurricane_tc/bfp-dec11"
  local raw="results/buff_encoder_v2/raw_scientific/hurricane_tc.f64le.bin"
  local threshold="89.66339111328125"
  local kmax
  kmax="$(python3 -c "import json; print(json.load(open('${artifact}/manifest.json'))['max_plane_count'])")"
  echo "hurricane_tc: k2 threshold=${threshold} kmax=${kmax}"
  run_case "$field" "$artifact" "$raw" "$threshold" "2" "2" "s10_k2_fused" "progressive_filter_sum_rowpack16_byte_mask"
  run_case "$field" "$artifact" "$raw" "$threshold" "$kmax" "max" "s10_kmax_fused" "progressive_filter_sum_rowpack16_byte_mask"
  run_case "$field" "$artifact" "$raw" "$threshold" "$kmax" "raw" "s10_raw_fp64" "raw_fp64_filter_sum"
}

for field in ${V1_3_FIELDS}; do
  case "$field" in
    cesm_atm_q) run_field_cesm_atm_q ;;
    hurricane_tc) run_field_hurricane_tc ;;
    *)
      echo "Building quasi-global artifact for ${field}..."
      python3 scripts/build_scientific_v2_artifact.py \
        --dataset "$field" \
        --segment-size 168480000 \
        --artifact-root datasets/scientific/dev_buff_v2_scientific_quasi_global
      artifact_local="datasets/scientific/dev_buff_v2_scientific_quasi_global/${field}/bfp-dec12"
      raw_local="results/buff_encoder_v2/raw_scientific/${field}.f64le.bin"
      threshold="0.0041057588532567024"
      kmax="$(python3 -c "import json; print(json.load(open('${artifact_local}/manifest.json'))['max_plane_count'])")"
      run_case "$field" "$artifact_local" "$raw_local" "$threshold" "2" "2" "s10_k2_fused" "progressive_filter_sum_rowpack16_byte_mask"
      run_case "$field" "$artifact_local" "$raw_local" "$threshold" "$kmax" "max" "s10_kmax_fused" "progressive_filter_sum_rowpack16_byte_mask"
      run_case "$field" "$artifact_local" "$raw_local" "$threshold" "$kmax" "raw" "s10_raw_fp64" "raw_fp64_filter_sum"
      ;;
  esac
done

echo "[$(date -Is)] v1.3 NCU freeze complete"
echo "Result dir: $RESULT_DIR"
echo "Summary CSV: $SUMMARY_CSV"
