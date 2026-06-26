#!/bin/bash
#SBATCH --job-name=nmr-c-v2-smoke
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail

WORKDIR=$(pwd)
BUILD_DIR="${WORKDIR}/build_nmr_c_v2"
RESULT_CSV="${WORKDIR}/results/nmr_c_v2_smoke_${SLURM_JOB_ID}.csv"
GPU_LOG="${WORKDIR}/logs/${SLURM_JOB_ID}_gpu_util.csv"
HANDOFF_MARKER="${WORKDIR}/handoff/job_done/job_${SLURM_JOB_ID}.json"

mkdir -p "${WORKDIR}/logs" "${WORKDIR}/results" "${WORKDIR}/handoff/job_done"

echo "=== Job info ==="
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "DATE=$(date -Is)"
echo "PWD=${WORKDIR}"

echo "=== GPU info ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if ! echo "${GPU_NAME}" | grep -qi "H200"; then
  echo "FATAL: GPU is NOT H200: ${GPU_NAME}. Aborting."
  exit 2
fi
echo "GPU verified: ${GPU_NAME}"

echo "=== Module & Toolchain ==="
ml load miniconda3 cuda/12.6 || ml load miniconda3 cuda/13.0 || true
which nvcc && nvcc --version
which gcc && gcc --version

echo "=== Git info ==="
git rev-parse HEAD
git status --short

# ---- GPU utilization tracker ----
echo "timestamp,gpu_index,util_pct,mem_used_mb,mem_total_mb" > "$GPU_LOG"
(
  while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits | tr -d ' ' \
    | awk -v ts="$(date +%Y-%m-%dT%H:%M:%S)" '{print ts","$0}' >> "$GPU_LOG"
    sleep 60
  done
) &
TRACKER_PID=$!

write_handoff() {
  local status=$?
  kill "$TRACKER_PID" 2>/dev/null || true
  wait "$TRACKER_PID" 2>/dev/null || true
  mkdir -p "$(dirname "$HANDOFF_MARKER")"
  git_commit="$(git rev-parse HEAD 2>/dev/null || true)"
  cat > "$HANDOFF_MARKER" <<EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "job_name": "${SLURM_JOB_NAME}",
  "exit_status": ${status},
  "finished_at": "$(date -Is)",
  "workdir": "${WORKDIR}",
  "git_commit": "${git_commit}",
  "stdout": "logs/${SLURM_JOB_ID}.out",
  "stderr": "logs/${SLURM_JOB_ID}.err",
  "gpu_util_log": "${GPU_LOG}",
  "result_csv": "${RESULT_CSV}",
  "next_action": "Review seff, sacct, logs, GPU util, and result CSV for correctness and throughput."
}
EOF
}
trap write_handoff EXIT

# ---- Build ----
echo "=== Build ==="
rm -rf "$BUILD_DIR"
cmake -B "$BUILD_DIR" \
  -S "${WORKDIR}/benchmarks/experiment4_filter_aggregate" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90

cmake --build "$BUILD_DIR" --target bench_nmr_c_v2_k_sweep -j$(nproc)
echo "Build OK."

BIN="$BUILD_DIR/bench_nmr_c_v2_k_sweep"

# ---- Smoke test: hurricane_u (small dataset, fastest) ----
HURRICANE_DS="/work/u4063895/datasets/scientific/dev_buff_v2_scientific/hurricane_u/bfp-dec12"
HURRICANE_RAW="/work/u4063895/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"

echo "=== Smoke: hurricane_u ==="
"$BIN" \
  --dataset "$HURRICANE_DS" \
  --raw "$HURRICANE_RAW" \
  --k "1,2,4,all" \
  --iters 10 \
  --threshold 0.0 \
  --csv "$RESULT_CSV"

# ---- Smoke test: cesm_atm_cloud ----
CESM_DS="/work/u4063895/datasets/scientific/dev_buff_v2_scientific/cesm_atm_cloud/bfp-dec12"
CESM_RAW="/work/u4063895/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"

echo "=== Smoke: cesm_atm_cloud ==="
"$BIN" \
  --dataset "$CESM_DS" \
  --raw "$CESM_RAW" \
  --k "1,2,4,all" \
  --iters 10 \
  --threshold 0.0 \
  --csv "$RESULT_CSV"

# ---- F2 Fault Smoke: localized multi-bit corruption ----
# Corrupt bytes [1000, 2000) in replica 0 of plane 0 (1000 bytes flipped)
# P2 vote (3 replicas, 1 corrupted) should correct the fault.
echo "=== F2 Smoke: localized corruption + vote sanity ==="
"$BIN" \
  --dataset "$HURRICANE_DS" \
  --raw "$HURRICANE_RAW" \
  --path P0 --k 2 --iters 5 --threshold 0.0 --csv "$RESULT_CSV"
"$BIN" \
  --dataset "$HURRICANE_DS" \
  --raw "$HURRICANE_RAW" \
  --path P2 --k 2 --iters 5 --threshold 0.0 --csv "$RESULT_CSV"
"$BIN" \
  --dataset "$HURRICANE_DS" \
  --raw "$HURRICANE_RAW" \
  --path P2 --k 2 --iters 5 --threshold 0.0 --csv "$RESULT_CSV" \
  --fault-family F2 --fault-params 1000,1000

# ---- F4 Fault Smoke: column-like repeated offset ----
# Corrupt every 1000th byte (stride=1000, count=25000) in replica 0 of plane 0.
echo "=== F4 Smoke: column corruption + vote sanity ==="
"$BIN" \
  --dataset "$HURRICANE_DS" \
  --raw "$HURRICANE_RAW" \
  --path P0 --k 2 --iters 5 --threshold 0.0 --csv "$RESULT_CSV"
"$BIN" \
  --dataset "$HURRICANE_DS" \
  --raw "$HURRICANE_RAW" \
  --path P2 --k 2 --iters 5 --threshold 0.0 --csv "$RESULT_CSV"
"$BIN" \
  --dataset "$HURRICANE_DS" \
  --raw "$HURRICANE_RAW" \
  --path P2 --k 2 --iters 5 --threshold 0.0 --csv "$RESULT_CSV" \
  --fault-family F4 --fault-params 1000,25000

echo "=== Done ==="
echo "Results: $RESULT_CSV"
head -1 "$RESULT_CSV"
echo "---"
cat "$RESULT_CSV"
