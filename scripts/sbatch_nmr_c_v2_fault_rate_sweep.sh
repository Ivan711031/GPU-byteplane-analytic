#!/bin/bash
#SBATCH --job-name=nmr-c-v2-fr-sweep
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
WORKDIR=$(pwd)
BUILD_DIR="${WORKDIR}/build_nmr_c_v2_fr"
RESULT_CSV="${WORKDIR}/results/nmr_c_v2_fault_rate_sweep_${SLURM_JOB_ID}.csv"
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
    sleep 15
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
  "next_action": "Review seff, sacct, logs, GPU util, and result CSV for correctness. Check for false_recovery and silent_wrong patterns."
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
# ---- Run sweep ----
echo "=== Sweep ==="
export SLURM_JOB_ID  # pass to runner for CSV naming
bash "${WORKDIR}/scripts/run_nmr_c_v2_fault_rate_sweep.sh" "$BIN"
echo "=== Done ==="
echo "Results: $RESULT_CSV"
wc -l "$RESULT_CSV"
echo "---"
head -1 "$RESULT_CSV"
echo "..."
tail -5 "$RESULT_CSV"
