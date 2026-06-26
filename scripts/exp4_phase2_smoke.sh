#!/bin/bash
#SBATCH --job-name=exp4-phase2-smoke
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/exp4/slurm_phase2_smoke_%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

ROOT_DIR="/home/u4063895/workspace/gpu-byteplane-scan-experiments"
cd "$ROOT_DIR"

module purge
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan

mkdir -p results/exp4 logs handoff/job_done

# Hardware validation: must be H200
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi

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

GPU_LOG="logs/${SLURM_JOB_ID}_gpu_util.csv"
HANDOFF_DIR="handoff/job_done"
HANDOFF_MARKER="${HANDOFF_DIR}/job_${SLURM_JOB_ID}.json"

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
  "stdout": "results/exp4/slurm_phase2_smoke_${SLURM_JOB_ID}.out",
  "stderr": "same_as_stdout",
  "gpu_util_log": "${GPU_LOG}",
  "next_action": "Review seff, sacct, stdout, GPU utilization, and result CSVs. Check gpu_matches_cpu_encoded for p3 and p6."
}
EOF
}
trap write_handoff_marker EXIT

echo "=== Build ==="
cmake -S benchmarks/experiment4 -B build/exp4 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp4 -j

# Run smoke tests for both p3 and p6 bounded-precision artifacts
for precision in p3 p6; do
    ENCODED_ROOT="/work/u4063895/datasets/synthetic/dev_buff_exp4_${precision}/uniform"
    
    if [[ ! -d "$ENCODED_ROOT" ]]; then
        echo "WARNING: $ENCODED_ROOT not found, skipping ${precision}" >&2
        continue
    fi

    echo "=== Smoke Test: ${precision} threshold=500.0 (primary) ==="
    ./build/exp4/bench_progressive_filter \
      --device 0 \
      --encoded-root "$ENCODED_ROOT" \
      --threshold 500.0 \
      --validate \
      --csv "results/exp4/smoke_${precision}_uniform_500_${SLURM_JOB_ID}.csv"

    echo "=== Smoke Test: ${precision} threshold=0.0 (edge, known risk) ==="
    ./build/exp4/bench_progressive_filter \
      --device 0 \
      --encoded-root "$ENCODED_ROOT" \
      --threshold 0.0 \
      --validate \
      --csv "results/exp4/smoke_${precision}_uniform_0_${SLURM_JOB_ID}.csv" || true

    echo "=== Smoke Test: ${precision} threshold=1000.0 (edge, known risk) ==="
    ./build/exp4/bench_progressive_filter \
      --device 0 \
      --encoded-root "$ENCODED_ROOT" \
      --threshold 1000.0 \
      --validate \
      --csv "results/exp4/smoke_${precision}_uniform_1000_${SLURM_JOB_ID}.csv" || true
done

echo "=== All Phase 2 smoke tests complete ==="
