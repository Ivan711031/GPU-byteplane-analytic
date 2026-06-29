#!/bin/bash
#SBATCH --job-name=nmr-c-v2-fr-smoke
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%j_fr_smoke.out
#SBATCH --error=logs/%j_fr_smoke.err
set -euo pipefail
WORKDIR=$(pwd)
BUILD_DIR="${WORKDIR}/build_nmr_c_v2_fr_smoke"
RESULT_CSV="${WORKDIR}/results/nmr_c_v2_fr_smoke_${SLURM_JOB_ID}.csv"
GPU_LOG="${WORKDIR}/logs/${SLURM_JOB_ID}_fr_smoke_gpu.csv"
HANDOFF_MARKER="${WORKDIR}/handoff/job_done/job_${SLURM_JOB_ID}.json"
mkdir -p "${WORKDIR}/logs" "${WORKDIR}/results" "${WORKDIR}/handoff/job_done"
echo "=== Job info ==="
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "DATE=$(date -Is)"
echo "=== GPU info ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
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
git diff --stat
# GPU tracker
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
  cat > "$HANDOFF_MARKER" <<EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "job_name": "${SLURM_JOB_NAME}",
  "exit_status": ${status},
  "finished_at": "$(date -Is)",
  "workdir": "${WORKDIR}",
  "result_csv": "${RESULT_CSV}",
  "next_action": "Verify build succeeds, CSV has expected columns, and P2/P3 vote mechanism handles fault plans correctly."
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
# ---- Generate fault plan ----
HURRICANE_N=25000000
HURRICANE_PLANES=8
FAULT_DIR="${WORKDIR}/results/fault_plans_nmr_c_v2/hurricane_u"
mkdir -p "$FAULT_DIR"
python3 "${WORKDIR}/scripts/generate_nmr_c_v2_fault_plans.py" \
  --n-rows "$HURRICANE_N" \
  --max-planes "$HURRICANE_PLANES" \
  --families F1 \
  --rates 1e-6 \
  --seeds 0 \
  --output-dir "$FAULT_DIR"
FPLAN=$(ls "$FAULT_DIR"/fault_plan_F1_low_*_seed0.fplan | head -1)
echo "Fault plan: $FPLAN"
head -3 "$FPLAN"
# ---- Smoke: hurricane_u P2 with F1 fault plan ----
HURRICANE_DS="${WORK_DIR}/datasets/scientific/dev_buff_v2_scientific/hurricane_u/bfp-dec12"
HURRICANE_RAW="${WORK_DIR}/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"
echo ""
echo "=== Smoke: P2 with F1 fault plan ==="
"$BIN" \
  --dataset "$HURRICANE_DS" \
  --raw "$HURRICANE_RAW" \
  --path P2 \
  --k "1,2,4,all" \
  --iters 10 \
  --threshold 0.0 \
  --fault-plan "$FPLAN" \
  --csv "$RESULT_CSV"
echo ""
echo "=== Smoke: P3 with F1 fault plan ==="
"$BIN" \
  --dataset "$HURRICANE_DS" \
  --raw "$HURRICANE_RAW" \
  --path P3 \
  --k "1,2,4,all" \
  --iters 10 \
  --threshold 0.0 \
  --fault-plan "$FPLAN" \
  --csv "$RESULT_CSV"
echo ""
echo "=== Results ==="
echo "CSV: $RESULT_CSV"
wc -l "$RESULT_CSV"
echo "---"
head -1 "$RESULT_CSV"
echo "---"
cat "$RESULT_CSV"
echo ""
echo "=== Done ==="
