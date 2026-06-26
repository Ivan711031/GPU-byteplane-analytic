#!/usr/bin/env bash
# run_exp4_byte_mask_kspecialized_smoke.sbatch
#
# Exp4 P2-2 smoke test: byte_mask_specialized_k{1,4,8} correctness + timing.
# Each case compares byte_mask (baseline) against the matching specialized
# strategy under the same --max-filter-planes cap.
#
# Matrix:
#   A) B heavy_tiled s99 full:         byte_mask vs byte_mask_specialized_k8
#   B) C zipfian    s99 full:          byte_mask vs byte_mask_specialized_k8
#   C) B heavy_tiled s99 k4:           byte_mask --max-filter-planes 4 vs byte_mask_specialized_k4 --max-filter-planes 4
#   D) C zipfian    s99 k4:            byte_mask --max-filter-planes 4 vs byte_mask_specialized_k4 --max-filter-planes 4
#   E) D heavy_tiled s99 k1:           byte_mask --max-filter-planes 1 vs byte_mask_specialized_k1 --max-filter-planes 1
#
# Usage:
#   sbatch --account=<ACCOUNT> scripts/run_exp4_byte_mask_kspecialized_smoke.sbatch
#
# Output:
#   results/exp4/byte_mask_kspecialized_smoke_<timestamp>_job<JOBID>_H200/
#     summary.csv
#     case_<ID>_<strategy>/
#       bench.csv
#       bench.log
#   handoff/job_done/job_<SLURM_JOB_ID>.json

#SBATCH --job-name=exp4-bm-kspec-smoke
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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

V2_ROOT="/work/u4063895/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
RAW_ROOT="/work/u4063895/datasets/synthetic/dev"

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

run_dir="${ROOT_DIR}/results/exp4/byte_mask_kspecialized_smoke_${timestamp}_job${job_id}_${GPU_TAG}"
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
  local next_action="Check summary.csv for correctness (gpu_count == cpu_encoded_count) and compare ms_per_iter between byte_mask and byte_mask_specialized_k* pairs."
  if [[ "$status" -ne 0 ]]; then
    next_action="Job failed with exit code ${status}; inspect slurm output and per-case bench.log."
  fi
  cat > "$handoff_marker" <<EOF
{
  "job_id": "${job_id}",
  "job_name": "exp4-bm-kspec-smoke",
  "exit_status": ${status},
  "finished_at": "$(date -Is)",
  "workdir": "$(pwd)",
  "git_commit": "${git_commit}",
  "result_dir": "${run_dir}",
  "summary_csv": "${run_dir}/summary.csv",
  "slurm_output": "${slurm_out}",
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
summary_csv="${run_dir}/summary.csv"
echo "case_id,dataset,selectivity,k,strategy,max_filter_planes,validated,gpu_count,cpu_encoded_count,ms_per_iter,status,csv" > "$summary_csv"

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
declare -a CASES
CASES+=("A heavy_tailed 99 full NA byte_mask")
CASES+=("A heavy_tailed 99 full NA byte_mask_specialized_k8")
CASES+=("B zipfian 99 full NA byte_mask")
CASES+=("B zipfian 99 full NA byte_mask_specialized_k8")
CASES+=("C heavy_tailed 99 k4 4 byte_mask")
CASES+=("C heavy_tailed 99 k4 4 byte_mask_specialized_k4")
CASES+=("D zipfian 99 k4 4 byte_mask")
CASES+=("D zipfian 99 k4 4 byte_mask_specialized_k4")
CASES+=("E heavy_tailed 99 k1 1 byte_mask")
CASES+=("E heavy_tailed 99 k1 1 byte_mask_specialized_k1")

total_ok=0
total_fail=0
SUBFAILURES=()

for entry in "${CASES[@]}"; do
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

  bench_csv="${case_dir}/bench.csv"
  bench_log="${case_dir}/bench.log"
  set +e
  if [[ "$mfp" != "NA" ]]; then
    "$BENCH_BIN" --device 0 --encoded-root "$encoded_root" --threshold "$threshold" \
      --block 256 --validate --warp-strategy "$strategy" --warmup 5 --iters 50 \
      --max-filter-planes "$mfp" --csv "$bench_csv" 2> "$bench_log"
  else
    "$BENCH_BIN" --device 0 --encoded-root "$encoded_root" --threshold "$threshold" \
      --block 256 --validate --warp-strategy "$strategy" --warmup 5 --iters 50 \
      --csv "$bench_csv" 2> "$bench_log"
  fi
  bench_rc=$?
  set -e

  validated="false"
  gpu_count="NA"
  cpu_encoded="NA"
  ms_per_iter="NA"
  status="FAIL"
  if [[ -f "$bench_csv" ]]; then
    gpu_count=$(tail -1 "$bench_csv" | cut -d',' -f12)
    cpu_encoded=$(tail -1 "$bench_csv" | cut -d',' -f13)
    validated_col=$(tail -1 "$bench_csv" | cut -d',' -f29)
    ms_per_iter=$(tail -1 "$bench_csv" | cut -d',' -f11)
    if [[ "$bench_rc" -eq 0 ]] && [[ "$gpu_count" == "$cpu_encoded" ]] && [[ "$validated_col" == "true" ]]; then
      validated="true"
      status="PASS"
      echo "  ✓ gpu=${gpu_count} cpu=${cpu_encoded} ms=${ms_per_iter}" >&2
    else
      echo "  ✗ FAIL: rc=${bench_rc} gpu=${gpu_count} cpu=${cpu_encoded} validated=${validated_col}" >&2
      SUBFAILURES+=("${case_id}_${strategy}: rc=${bench_rc} gpu=${gpu_count} cpu=${cpu_encoded} validated=${validated_col}")
    fi
  else
    echo "  ✗ CSV not found (rc=${bench_rc})" >&2
    SUBFAILURES+=("${case_id}_${strategy}: CSV not found")
  fi

  echo "${case_id},${dataset},${sel},${k_label},${strategy},${mfp},${validated},${gpu_count},${cpu_encoded},${ms_per_iter},${status},${bench_csv}" >> "$summary_csv"

  if [[ "$status" == "PASS" ]]; then
    total_ok=$((total_ok + 1))
  else
    total_fail=$((total_fail + 1))
  fi
done

cat > "${run_dir}/run_meta.txt" <<EOF
job_id=${job_id}
job_name=exp4-bm-kspec-smoke
gpu=${GPU_NAME}
git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
root_dir=${ROOT_DIR}
strategies=byte_mask,byte_mask_specialized_k1,byte_mask_specialized_k4,byte_mask_specialized_k8
cases=A,B,C,D,E
block=256 warmup=5 iters=50 validate=on
summary_csv=${summary_csv}
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
