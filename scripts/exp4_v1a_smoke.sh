#!/bin/bash
#SBATCH --job-name=exp4-v1a-smoke
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

module purge
module load miniconda3/26.1.1
module load cuda/12.6
conda activate gpu-byteplane-scan

cd /home/u4063895/workspace/gpu-byteplane-scan-experiments

mkdir -p logs results/exp4 handoff/job_done

# Hardware validation: must be H200
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi

RESULT_CSV="results/exp4/smoke_uniform_500_${SLURM_JOB_ID}.csv"
GPU_LOG="logs/${SLURM_JOB_ID}_gpu_util.csv"
HANDOFF_DIR="handoff/job_done"
HANDOFF_MARKER="${HANDOFF_DIR}/job_${SLURM_JOB_ID}.json"

echo "=== Job info ==="
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "DATE=$(date -Is)"
echo "PWD=$(pwd)"
echo "=== GPU info ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "=== CUDA info ==="
nvcc --version
echo "=== Git info ==="
git rev-parse HEAD || true
git status --short || true

# GPU utilization tracker
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

write_handoff_marker() {
  status=$?
  kill "$TRACKER_PID" 2>/dev/null || true
  wait "$TRACKER_PID" 2>/dev/null || true
  mkdir -p "$HANDOFF_DIR"
  git_commit="$(git rev-parse HEAD 2>/dev/null || true)"
  cat > "$HANDOFF_MARKER" <<EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "job_name": "${SLURM_JOB_NAME}",
  "exit_status": ${status},
  "finished_at": "$(date -Is)",
  "workdir": "$(pwd)",
  "git_commit": "${git_commit}",
  "stdout": "logs/${SLURM_JOB_ID}.out",
  "stderr": "logs/${SLURM_JOB_ID}.err",
  "gpu_util_log": "${GPU_LOG}",
  "result_csv": "${RESULT_CSV}",
  "next_action": "Review seff, sacct, stdout, stderr, GPU utilization, and result CSV. Then write a short run report."
}
EOF
}
trap write_handoff_marker EXIT

echo "=== Run benchmark ==="
./build/exp4/bench_progressive_filter \
  --device 0 \
  --encoded-root /work/u4063895/datasets/synthetic/dev_buff_exp3/uniform \
  --threshold 500.0 \
  --validate \
  --csv "$RESULT_CSV"
