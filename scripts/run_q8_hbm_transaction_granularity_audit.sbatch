#!/usr/bin/env bash
#SBATCH -p dev
#SBATCH -J q8-hbm-ncu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -o results/exp4_filter_aggregate/q8_hbm_ncu_%j.out
#SBATCH -e results/exp4_filter_aggregate/q8_hbm_ncu_%j.err
set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
JOB_ID="${SLURM_JOB_ID:-manual}"
GPU_NAME="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | python3 -c 'import sys; print(sys.stdin.readline().strip())')"
fi
GPU_TAG="${GPU_NAME// /}"
RESULT_DIR="results/exp4_filter_aggregate/q8_hbm_transaction_${RUN_TS}_job${JOB_ID}_${GPU_TAG}"
SUMMARY_CSV="$RESULT_DIR/q8_hbm_transaction_summary.csv"
STDOUT_PATH="$ROOT_DIR/results/exp4_filter_aggregate/q8_hbm_ncu_${JOB_ID}.out"
STDERR_PATH="$ROOT_DIR/results/exp4_filter_aggregate/q8_hbm_ncu_${JOB_ID}.err"
GPU_UTIL_LOG="$RESULT_DIR/gpu_util.csv"
HANDOFF_MARKER="$ROOT_DIR/handoff/job_done/job_${JOB_ID}.json"
mkdir -p "$RESULT_DIR"
GPU_MONITOR_PID=""
write_completion_marker() {
  local status="$1"
  if [[ -n "$GPU_MONITOR_PID" ]]; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
  mkdir -p "$(dirname "$HANDOFF_MARKER")"
  local git_commit
  git_commit="$(git rev-parse HEAD 2>/dev/null || true)"
  cat > "$HANDOFF_MARKER" <<EOF
{
  "job_id": "$JOB_ID",
  "job_name": "q8-hbm-ncu",
  "exit_status": $status,
  "finished_at": "$(date -Is)",
  "workdir": "$ROOT_DIR",
  "git_commit": "$git_commit",
  "stdout": "$STDOUT_PATH",
  "stderr": "$STDERR_PATH",
  "gpu_util_log": "$GPU_UTIL_LOG",
  "result_dir": "$RESULT_DIR",
  "result_csv": "$SUMMARY_CSV",
  "next_action": "review Q8 NCU summary and write research report"
}
EOF
}
trap 'status=$?; write_completion_marker "$status"' EXIT
echo "[$(date -Is)] hardware gate"
echo "GPU: $GPU_NAME"
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: expected H200 GPU, got: $GPU_NAME" >&2
  exit 2
fi
module purge
module load miniconda3/26.1.1 2>/dev/null || module load miniconda3
module load cuda/12.6 2>/dev/null || module load cuda
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan 2>/dev/null || conda activate gpu-byteplane-rapids 2>/dev/null || true
echo "[$(date -Is)] toolchain"
which nvcc
cmake --version
gcc --version | python3 -c 'import sys; print(sys.stdin.readline().strip())'
NCU_BIN="$(command -v ncu)"
"$NCU_BIN" --version | python3 -c 'import sys; print(sys.stdin.readline().strip())'
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 1 > "$GPU_UTIL_LOG" 2>/dev/null &
GPU_MONITOR_PID=$!
echo "[$(date -Is)] ensuring true quasi-global artifacts"
python3 scripts/build_scientific_v2_artifact.py \
  --dataset cesm_atm_q \
  --segment-size 168480000 \
  --artifact-root datasets/scientific/dev_buff_v2_scientific_quasi_global
if [[ "${Q8_FIELDS:-cesm_atm_q}" == *"hurricane_tc"* ]]; then
  python3 scripts/build_scientific_v2_artifact.py \
    --dataset hurricane_tc \
    --segment-size 25000000 \
    --artifact-root datasets/scientific/dev_buff_v2_scientific_quasi_global
fi
BUILD_DIR="$ROOT_DIR/build_filter_aggregate"
mkdir -p "$BUILD_DIR"
cmake -S "$ROOT_DIR/benchmarks/experiment4_filter_aggregate" -B "$BUILD_DIR" \
  -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"${SLURM_CPUS_ON_NODE:-4}"
BENCH="$BUILD_DIR/bench_filter_aggregate"
cat > "$SUMMARY_CSV" <<'EOF'
field,artifact_path,kernel_name,k,runtime_ms,logical_bytes_per_row_estimate,dram_bytes_or_sector_proxy,l2_sector_proxy,l1tex_sector_proxy,sm_throughput_pct,dram_throughput_pct,long_scoreboard_pct_or_cycles,eligible_warps_per_cycle,achieved_occupancy,report_path,bench_csv_path,command_line
EOF
cat > "$RESULT_DIR/run_meta.txt" <<EOF
date=$(date -Is)
job_id=$JOB_ID
gpu=$GPU_NAME
git_commit=$(git rev-parse HEAD 2>/dev/null || true)
ncu_command_shape=--set full --target-processes all --profile-from-start yes --kernel-name regex:<kernel> --page raw -f
q8_fields=${Q8_FIELDS:-cesm_atm_q}
EOF
csv_get() {
  local csv="$1"
  local col="$2"
  python3 - "$csv" "$col" <<'PY'
import csv, sys
path, col = sys.argv[1], sys.argv[2]
with open(path, newline='') as f:
    rows = list(csv.DictReader(f))
if not rows:
    print("")
else:
    print(rows[-1].get(col, ""))
PY
}
append_summary_from_ncu() {
  local field="$1"
  local artifact_path="$2"
  local k_label="$3"
  local kernel_name="$4"
  local ncu_stdout="$5"
  local ncu_report="$6"
  local bench_csv="$7"
  local command_file="$8"
  python3 - "$field" "$artifact_path" "$k_label" "$kernel_name" "$ncu_stdout" "$ncu_report" "$bench_csv" "$command_file" "$SUMMARY_CSV" <<'PY'
import csv, math, re, sys
field, artifact, k_label, kernel, ncu_stdout, report, bench_csv, command_file, summary = sys.argv[1:]
text = open(ncu_stdout, errors="replace").read() if ncu_stdout else ""
def clean_num(s):
    s = s.strip().replace(",", "")
    if s in {"", "no", "data", "no data", "nan", "(!) nan"}:
        return ""
    m = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", s)
    return m.group(0) if m else ""
def metric_pair(name):
    for line in text.splitlines():
        if line.strip().startswith(name):
            tail = line.split(name, 1)[1].strip()
            parts = tail.split()
            if not parts:
                return "", ""
            return clean_num(parts[-1]), " ".join(parts[:-1])
    return "", ""
def metric(name):
    value, _unit = metric_pair(name)
    return value
def bench_value(col):
    try:
        with open(bench_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        return rows[-1].get(col, "") if rows else ""
    except Exception:
        return ""
duration_value, duration_unit = metric_pair("gpu__time_duration.sum")
if not duration_value:
    duration_value, duration_unit = metric_pair("gpu__time_duration.avg")
runtime_ms = ""
if duration_value:
    scale = 1.0
    if duration_unit == "us":
        scale = 1.0 / 1000.0
    elif duration_unit == "ns":
        scale = 1.0 / 1_000_000.0
    elif duration_unit == "s":
        scale = 1000.0
    runtime_ms = f"{float(duration_value) * scale:.6f}"
n = bench_value("n")
logical = bench_value("logical_bytes")
if kernel == "raw_fp64_filter_sum":
    logical_bpr = "8.000000"
elif n and logical:
    logical_bpr = f"{float(logical) / float(n):.6f}"
else:
    logical_bpr = ""
dram_value, dram_unit = metric_pair("dram__bytes_read.sum")
dram_sectors = metric("dram__sectors_read.sum")
dram_proxy = f"{dram_value} {dram_unit}" if dram_value else (f"{dram_sectors} sectors" if dram_sectors else "")
row = {
    "field": field,
    "artifact_path": artifact,
    "kernel_name": kernel,
    "k": k_label,
    "runtime_ms": runtime_ms,
    "logical_bytes_per_row_estimate": logical_bpr,
    "dram_bytes_or_sector_proxy": dram_proxy,
    "l2_sector_proxy": metric("lts__t_sectors.sum"),
    "l1tex_sector_proxy": metric("l1tex__t_sectors.sum") or metric("SM_B.TriageCompute.l1tex__t_sectors.sum"),
    "sm_throughput_pct": metric("sm__throughput.avg.pct_of_peak_sustained_elapsed") or metric("TPC.TriageCompute.sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed"),
    "dram_throughput_pct": metric("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed") or metric("dram__bytes_read.sum.pct_of_peak_sustained_elapsed"),
    "long_scoreboard_pct_or_cycles": metric("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
    "eligible_warps_per_cycle": metric("smsp__warps_eligible.avg.per_cycle_active"),
    "achieved_occupancy": metric("sm__warps_active.avg.pct_of_peak_sustained_active"),
    "report_path": report,
    "bench_csv_path": bench_csv,
    "command_line": open(command_file).read().strip() if command_file else "",
}
with open(summary, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    writer.writerow(row)
PY
}
run_case() {
  local field="$1"
  local artifact="$2"
  local raw="$3"
  local threshold="$4"
  local k_run="$5"
  local k_label="$6"
  local case_name="$7"
  local kernel_regex="$8"
  local case_dir="$RESULT_DIR/${field}_${case_name}"
  mkdir -p "$case_dir"
  local bench_csv="$case_dir/bench.csv"
  local bench_log="$case_dir/bench.log"
  local command_file="$case_dir/ncu_command.txt"
  local ncu_base="$case_dir/ncu_profile"
  local ncu_stdout="$case_dir/ncu_stdout.txt"
  local ncu_stderr="$case_dir/ncu_stderr.txt"
  echo "[$(date -Is)] case=$field/$case_name threshold=$threshold k_run=$k_run kernel=$kernel_regex"
  "$BENCH" \
    --dataset "$artifact" \
    --raw "$raw" \
    --threshold "$threshold" \
    --max-filter-planes "$k_run" \
    --iters 1 \
    --warmup 0 \
    --validate \
    --block-threads 256 \
    --csv "$bench_csv" > "$bench_log" 2>&1
  local gpu_count cpu_enc validated
  gpu_count="$(csv_get "$bench_csv" gpu_count)"
  cpu_enc="$(csv_get "$bench_csv" cpu_enc_count)"
  validated="$(csv_get "$bench_csv" validated)"
  if [[ "$gpu_count" != "$cpu_enc" || "$validated" != "true" ]]; then
    echo "ERROR: validation failed for $field/$case_name: gpu=$gpu_count cpu=$cpu_enc validated=$validated" >&2
    exit 2
  fi
  printf '%q ' "$NCU_BIN" --set full --target-processes all --profile-from-start yes --kernel-name "regex:${kernel_regex}" --page raw -f -o "$ncu_base" "$BENCH" --dataset "$artifact" --raw "$raw" --threshold "$threshold" --max-filter-planes "$k_run" --iters 1 --warmup 0 --validate --block-threads 256 --csv "$case_dir/ncu_bench.csv" > "$command_file"
  printf '\n' >> "$command_file"
  "$NCU_BIN" \
    --set full \
    --target-processes all \
    --profile-from-start yes \
    --kernel-name "regex:${kernel_regex}" \
    --page raw \
    -f \
    -o "$ncu_base" \
    "$BENCH" \
      --dataset "$artifact" \
      --raw "$raw" \
      --threshold "$threshold" \
      --max-filter-planes "$k_run" \
      --iters 1 \
      --warmup 0 \
      --validate \
      --block-threads 256 \
      --csv "$case_dir/ncu_bench.csv" \
    > "$ncu_stdout" 2> "$ncu_stderr"
  if [[ ! -s "${ncu_base}.ncu-rep" ]]; then
    echo "ERROR: missing NCU report for $field/$case_name" >&2
    exit 3
  fi
  if grep -q "No kernels were profiled" "$ncu_stdout" "$ncu_stderr" 2>/dev/null; then
    echo "ERROR: NCU captured no kernels for $field/$case_name" >&2
    exit 3
  fi
  append_summary_from_ncu "$field" "$artifact" "$k_label" "$kernel_regex" "$ncu_stdout" "${ncu_base}.ncu-rep" "$bench_csv" "$command_file"
}
run_field_cesm_atm_q() {
  local field="cesm_atm_q"
  local artifact="datasets/scientific/dev_buff_v2_scientific_quasi_global/cesm_atm_q/bfp-dec12"
  local raw="results/buff_encoder_v2/raw_scientific/cesm_atm_q.f64le.bin"
  local threshold="0.0041057588532567024"
  local k2="2"
  local kmax
  kmax="$(python3 -c "import json; print(json.load(open('${artifact}/manifest.json'))['max_plane_count'])")"
  run_case "$field" "$artifact" "$raw" "$threshold" "$k2" "2" "s10_k2_fused" "progressive_filter_sum_rowpack16_byte_mask"
  run_case "$field" "$artifact" "$raw" "$threshold" "$kmax" "max" "s10_kmax_fused" "progressive_filter_sum_rowpack16_byte_mask"
  run_case "$field" "$artifact" "$raw" "$threshold" "$kmax" "raw" "s10_raw_fp64" "raw_fp64_filter_sum"
}
run_field_hurricane_tc() {
  local field="hurricane_tc"
  local artifact="datasets/scientific/dev_buff_v2_scientific_quasi_global/hurricane_tc/bfp-dec11"
  local raw="results/buff_encoder_v2/raw_scientific/hurricane_tc.f64le.bin"
  local threshold="89.66339111328125"
  local k2="2"
  local kmax
  kmax="$(python3 -c "import json; print(json.load(open('${artifact}/manifest.json'))['max_plane_count'])")"
  run_case "$field" "$artifact" "$raw" "$threshold" "$k2" "2" "s10_k2_fused" "progressive_filter_sum_rowpack16_byte_mask"
  run_case "$field" "$artifact" "$raw" "$threshold" "$kmax" "max" "s10_kmax_fused" "progressive_filter_sum_rowpack16_byte_mask"
  run_case "$field" "$artifact" "$raw" "$threshold" "$kmax" "raw" "s10_raw_fp64" "raw_fp64_filter_sum"
}
for field in ${Q8_FIELDS:-cesm_atm_q}; do
  case "$field" in
    cesm_atm_q) run_field_cesm_atm_q ;;
    hurricane_tc) run_field_hurricane_tc ;;
    *) echo "ERROR: unsupported Q8 field: $field" >&2; exit 2 ;;
  esac
done
echo "[$(date -Is)] Q8 NCU audit complete"
echo "Result dir: $RESULT_DIR"
echo "Summary CSV: $SUMMARY_CSV"
