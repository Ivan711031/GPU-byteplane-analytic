#!/usr/bin/env bash
# run_exp4_ncu_byte_mask_kspecialized_bcd.sbatch
#
# Exp4 P2-2 NCU profiling: compare byte_mask vs byte_mask_specialized_k{1,4,8}.
#
# Matrix (10 profiling rows):
#   B full:  byte_mask vs byte_mask_specialized_k8
#   C full:  byte_mask vs byte_mask_specialized_k8
#   B k4:    byte_mask --max-filter-planes 4 vs byte_mask_specialized_k4 --max-filter-planes 4
#   C k4:    byte_mask --max-filter-planes 4 vs byte_mask_specialized_k4 --max-filter-planes 4
#   D k1:    byte_mask --max-filter-planes 1 vs byte_mask_specialized_k1 --max-filter-planes 1
#
# Usage:
#   sbatch --account=<ACCOUNT> scripts/run_exp4_ncu_byte_mask_kspecialized_bcd.sbatch
#
# Output:
#   results/exp4/ncu_bm_kspecialized_bcd_<timestamp>_job<JOBID>_H200/
#     summary.csv
#     metric_focus.txt
#     case_<ID>_<strategy>/
#       baseline.csv
#       ncu_profile.ncu-rep
#       ncu_profile.csv
#       ncu_stdout.txt
#       ncu_stderr.txt
#   handoff/job_done/job_<SLURM_JOB_ID>.json
#
# NOTE: extractor is scripts/extract_exp4_ncu_deep_metrics.py (already reads
# the strategy column from summary.csv).
#SBATCH --job-name=exp4-ncu-bm-kspec-bcd
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
set -euo pipefail
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
V2_ROOT="${WORK_DIR}/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
RAW_ROOT="${WORK_DIR}/datasets/synthetic/dev"
declare -A DATASET_P
DATASET_P[heavy_tailed]="p6"
DATASET_P[zipfian]="p8"
timestamp=$(date +%Y%m%d_%H%M%S)
job_id="${SLURM_JOB_ID:-local}"
GPU_NAME="unknown"
GPU_TAG="unknown"
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ' 2>/dev/null || echo "unknown")
  GPU_TAG="${GPU_NAME// /_}"
  GPU_TAG="${GPU_TAG:-unknown}"
fi
run_dir="${ROOT_DIR}/results/exp4/ncu_bm_kspecialized_bcd_${timestamp}_job${job_id}_${GPU_TAG}"
handoff_dir="${ROOT_DIR}/handoff/job_done"
handoff_marker="${handoff_dir}/job_${job_id}.json"
write_handoff_marker() {
  local status=$?
  if [[ -n "${TRACKER_PID:-}" ]]; then
    kill "$TRACKER_PID" 2>/dev/null || true
    wait "$TRACKER_PID" 2>/dev/null || true
  fi
  mkdir -p "$handoff_dir"
  local git_commit
  git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
  local slurm_out=""
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    local candidate="${SLURM_SUBMIT_DIR}/slurm-${SLURM_JOB_ID}.out"
    if [[ -f "$candidate" ]]; then
      slurm_out="$candidate"
    fi
  fi
  local next_action="Extract scheduler/warp-state/memory/instruction metrics from ncu_profile.csv for all 10 profiling rows. Compare byte_mask vs byte_mask_specialized_k*."
  if [[ "$status" -ne 0 ]]; then
    next_action="Job failed with exit code ${status}; inspect slurm output and per-case ncu_stderr.txt."
  fi
  cat > "$handoff_marker" <<EOF
{
  "job_id": "${job_id}",
  "job_name": "exp4-ncu-bm-kspec-bcd",
  "exit_status": ${status},
  "finished_at": "$(date -Is)",
  "workdir": "$(pwd)",
  "git_commit": "${git_commit}",
  "result_dir": "${run_dir}",
  "summary_csv": "${run_dir}/summary.csv",
  "slurm_output": "${slurm_out}",
  "gpu_util_log": "${run_dir}/gpu_util.csv",
  "next_action": "${next_action}"
}
EOF
}
trap write_handoff_marker EXIT
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi
mkdir -p "$run_dir" "$handoff_dir"
gpu_util_log="${run_dir}/gpu_util.csv"
echo "timestamp,gpu_index,util_pct,mem_used_mb,mem_total_mb" > "$gpu_util_log"
(
  while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits | tr -d ' ' \
    | awk -v ts="$(date +%Y-%m-%dT%H:%M:%S)" '{print ts","$0}' >> "$gpu_util_log"
    sleep 60
  done
) &
TRACKER_PID=$!
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
elif [[ -x ${ENVSTACK_ROOT}/cuda/12.6/bin/ncu ]]; then
  NCU_BIN=${ENVSTACK_ROOT}/cuda/12.6/bin/ncu
fi
if [[ -z "$NCU_BIN" ]]; then
  echo "ERROR: ncu not found" >&2
  exit 2
fi
echo "ncu: $NCU_BIN $($NCU_BIN --version 2>&1 | head -1)" >&2
echo "=== Job info ===" >&2
echo "Job ID: ${job_id}  Host: $(hostname)  GPU: ${GPU_NAME}" >&2
echo "Output: ${run_dir}" >&2
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader >&2
git rev-parse HEAD >&2
git status --short >&2
echo "" >&2
cat > "${run_dir}/metric_focus.txt" <<'EOF'
P2-2 specialized-k NCU focus:
- Compare byte_mask (runtime-variable round count) vs byte_mask_specialized_k*
  (compile-time K with #pragma unroll + effective_rounds guard).
- Key metrics: issued/executed instructions, eligible warps, long scoreboard,
  L1/L2/DRAM traffic, instruction mix (ALU/FMA/uniform).
- Assess whether unrolling reduces instruction count or improves ILP.
NCU capture mode:
- --set full
- --launch-count 1
- --target-processes all
- --profile-from-start yes
- .ncu-rep and ncu_stdout.txt are the primary capture artifacts; CSV export
  from .ncu-rep is best-effort.
- ncu_status in summary.csv:
  - PASS: both .ncu-rep and ncu_profile.csv exist.
  - PASS: stdout metrics: .ncu-rep exists and ncu_stdout.txt contains required
    P1 metrics; p2_kspecialized_metrics.csv can be extracted from stdout.
  - PARTIAL: report only.
  - FAIL: NCU capture/report failed.
EOF
echo "=== Build ===" >&2
cmake -S benchmarks/experiment4 -B build/exp4 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 2>&1
cmake --build build/exp4 -j 2>&1
echo "" >&2
BENCH_BIN="${ROOT_DIR}/build/exp4/bench_progressive_filter"
summary_csv="${run_dir}/summary.csv"
echo "case_id,dataset,selectivity,k,strategy,max_filter_planes,validated,gpu_count,cpu_encoded_count,ms_per_iter,ncu_status,ncu_report,ncu_csv" > "$summary_csv"
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
# Each entry: case_id dataset selectivity k_label mfp strategy
declare -a PROFILES
PROFILES+=("Bfull heavy_tailed 99 full NA byte_mask")
PROFILES+=("Bfull heavy_tailed 99 full NA byte_mask_specialized_k8")
PROFILES+=("Cfull zipfian 99 full NA byte_mask")
PROFILES+=("Cfull zipfian 99 full NA byte_mask_specialized_k8")
PROFILES+=("Bk4 heavy_tailed 99 k4 4 byte_mask")
PROFILES+=("Bk4 heavy_tailed 99 k4 4 byte_mask_specialized_k4")
PROFILES+=("Ck4 zipfian 99 k4 4 byte_mask")
PROFILES+=("Ck4 zipfian 99 k4 4 byte_mask_specialized_k4")
PROFILES+=("Dk1 heavy_tailed 99 k1 1 byte_mask")
PROFILES+=("Dk1 heavy_tailed 99 k1 1 byte_mask_specialized_k1")
total_ok=0
total_fail=0
SUBFAILURES=()
for entry in "${PROFILES[@]}"; do
  read -r case_id dataset sel k_label mfp strategy <<< "$entry"
  artifact_p="${DATASET_P[$dataset]}"
  encoded_root="${V2_ROOT}/${dataset}_${artifact_p}"
  threshold=$(compute_threshold "$dataset" "$sel")
  case_dir="${run_dir}/case_${case_id}_${strategy}"
  mkdir -p "$case_dir"
  echo "" >&2
  echo "============================================================" >&2
  echo "[${case_id}_${strategy}] dataset=${dataset} sel=${sel} k=${k_label} mfp=${mfp}" >&2
  echo "============================================================" >&2
  echo "  threshold=${threshold}" >&2
  # --- Baseline timing (with validation) ---
  baseline_csv="${case_dir}/baseline.csv"
  baseline_log="${case_dir}/baseline.log"
  set +e
  if [[ "$mfp" != "NA" ]]; then
    "$BENCH_BIN" --device 0 --encoded-root "$encoded_root" --threshold "$threshold" \
      --block 256 --validate --warp-strategy "$strategy" --warmup 10 --iters 200 \
      --max-filter-planes "$mfp" --csv "$baseline_csv" 2> "$baseline_log"
  else
    "$BENCH_BIN" --device 0 --encoded-root "$encoded_root" --threshold "$threshold" \
      --block 256 --validate --warp-strategy "$strategy" --warmup 10 --iters 200 \
      --csv "$baseline_csv" 2> "$baseline_log"
  fi
  baseline_rc=$?
  set -e
  validated="false"
  gpu_count="NA"
  cpu_encoded="NA"
  ms_per_iter="NA"
  if [[ -f "$baseline_csv" ]]; then
    gpu_count=$(tail -1 "$baseline_csv" | cut -d',' -f12)
    cpu_encoded=$(tail -1 "$baseline_csv" | cut -d',' -f13)
    validated_col=$(tail -1 "$baseline_csv" | cut -d',' -f29)
    ms_per_iter=$(tail -1 "$baseline_csv" | cut -d',' -f11)
    if [[ "$baseline_rc" -eq 0 ]] && [[ "$gpu_count" == "$cpu_encoded" ]] && [[ "$validated_col" == "true" ]]; then
      validated="true"
      echo "  ✓ Baseline: gpu=${gpu_count} cpu=${cpu_encoded} ms=${ms_per_iter}" >&2
    else
      echo "  ✗ Baseline FAIL: rc=${baseline_rc} gpu=${gpu_count} cpu=${cpu_encoded} validated=${validated_col}" >&2
    fi
  else
    echo "  ✗ Baseline CSV not found (rc=${baseline_rc})" >&2
  fi
  # --- NCU capture ---
  ncu_export_base="${case_dir}/ncu_profile"
  ncu_report="${ncu_export_base}.ncu-rep"
  ncu_csv="${ncu_export_base}.csv"
  ncu_stdout="${case_dir}/ncu_stdout.txt"
  ncu_stderr="${case_dir}/ncu_stderr.txt"
  ncu_status="not_run"
  set +e
  if [[ "$mfp" != "NA" ]]; then
    "$NCU_BIN" --set full --target-processes all --profile-from-start yes \
      --launch-count 1 --page raw -f -o "$ncu_export_base" \
      "$BENCH_BIN" --device 0 --encoded-root "$encoded_root" --threshold "$threshold" \
        --block 256 --validate --warp-strategy "$strategy" --warmup 0 --iters 1 \
        --max-filter-planes "$mfp" --csv "${case_dir}/ncu_bench.csv" \
      > "$ncu_stdout" 2> "$ncu_stderr"
  else
    "$NCU_BIN" --set full --target-processes all --profile-from-start yes \
      --launch-count 1 --page raw -f -o "$ncu_export_base" \
      "$BENCH_BIN" --device 0 --encoded-root "$encoded_root" --threshold "$threshold" \
        --block 256 --validate --warp-strategy "$strategy" --warmup 0 --iters 1 \
        --csv "${case_dir}/ncu_bench.csv" \
      > "$ncu_stdout" 2> "$ncu_stderr"
  fi
  ncu_rc=$?
  set -e
  if grep -q "No kernels were profiled" "$ncu_stdout" 2>/dev/null; then
    ncu_status="FAIL: No kernels were profiled"
  elif [[ "$ncu_rc" -eq 0 ]] && [[ -f "$ncu_report" ]] && [[ -s "$ncu_report" ]]; then
    "$NCU_BIN" --csv --page raw --force-overwrite true \
      -i "$ncu_report" -o "$ncu_export_base" \
      >> "$ncu_stdout" 2>> "$ncu_stderr" || true
    if [[ -f "$ncu_csv" ]] && [[ -s "$ncu_csv" ]]; then
      ncu_status="PASS"
    elif grep -q "smsp__warps_eligible.avg.per_cycle_active" "$ncu_stdout" 2>/dev/null \
      && grep -q "dram__bytes_read.sum" "$ncu_stdout" 2>/dev/null; then
      ncu_status="PASS: stdout metrics"
      ncu_csv=""
    else
      ncu_status="PARTIAL: report only"
      ncu_csv=""
    fi
  else
    ncu_status="FAIL: ncu rc=${ncu_rc} or report missing"
  fi
  echo "$ncu_status" > "${case_dir}/ncu_status.txt"
  echo "${case_id},${dataset},${sel},${k_label},${strategy},${mfp:-NA},${validated},${gpu_count},${cpu_encoded},${ms_per_iter},${ncu_status},${ncu_report},${ncu_csv}" >> "$summary_csv"
  if { [[ "$ncu_status" == "PASS" ]] || [[ "$ncu_status" == "PASS: stdout metrics" ]]; } \
    && [[ "$validated" == "true" ]]; then
    total_ok=$((total_ok + 1))
  else
    total_fail=$((total_fail + 1))
    SUBFAILURES+=("${case_id}_${strategy}: validated=${validated} ncu=${ncu_status}")
  fi
done
# Run extractor
metrics_csv="${run_dir}/p2_kspecialized_metrics.csv"
extractor_status="not_run"
if [[ -f "${ROOT_DIR}/scripts/extract_exp4_ncu_deep_metrics.py" ]]; then
  if python "${ROOT_DIR}/scripts/extract_exp4_ncu_deep_metrics.py" "$run_dir" --output "$metrics_csv" \
    > "${run_dir}/extract_metrics_stdout.txt" 2> "${run_dir}/extract_metrics_stderr.txt"; then
    extractor_status="PASS"
  else
    extractor_status="FAIL"
    total_fail=$((total_fail + 1))
    SUBFAILURES+=("extract_metrics: failed; see ${run_dir}/extract_metrics_stderr.txt")
  fi
else
  extractor_status="FAIL: extractor missing"
  total_fail=$((total_fail + 1))
  SUBFAILURES+=("extract_metrics: missing scripts/extract_exp4_ncu_deep_metrics.py")
fi
cat > "${run_dir}/run_meta.txt" <<EOF
job_id=${job_id}
job_name=exp4-ncu-bm-kspec-bcd
gpu=${GPU_NAME}
git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
root_dir=${ROOT_DIR}
strategies=byte_mask,byte_mask_specialized_k1,byte_mask_specialized_k4,byte_mask_specialized_k8
cases=B_full,C_full,B_k4,C_k4,D_k1
ncu_capture=--set full --launch-count 1 --page raw
focus=instruction_count,eligible_warps,long_scoreboard,memory_sectors_per_request,L1_L2_DRAM_traffic,instruction_pipe_utilization
summary_csv=${summary_csv}
metrics_csv=${metrics_csv}
extractor_status=${extractor_status}
EOF
echo "" >&2
echo "=== Summary ===" >&2
echo "ok=${total_ok} fail=${total_fail}" >&2
echo "summary=${summary_csv}" >&2
if (( total_fail > 0 )); then
  printf 'Failures:\n' >&2
  printf '  %s\n' "${SUBFAILURES[@]}" >&2
  exit 1
fi
echo "DONE" >&2
