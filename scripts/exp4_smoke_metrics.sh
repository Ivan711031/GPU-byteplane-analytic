#!/bin/bash
#SBATCH --job-name=exp4-metrics-smoke
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

ROOT_DIR="/home/u4063895/workspace/gpu-byteplane-scan-experiments"
cd "$ROOT_DIR"

module purge
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan

mkdir -p logs results/exp4 handoff/job_done

# Hardware validation: must be H200
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ ! "$GPU_NAME" =~ "H200" ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi

RESULT_DIR="results/exp4"
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
    sleep 30
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
  "result_dir": "${RESULT_DIR}",
  "next_action": "Inspect smoke results, check metrics availability, and write final report"
}
EOF
}
trap write_handoff_marker EXIT

echo "=== Smoke Test Setup ==="
echo "Testing uniform dataset with exact (exp3), p3, p6 artifacts"
echo "Thresholds: 0, 500, 1000"

# Define artifacts
declare -A ARTIFACTS=(
  [exact]="/work/u4063895/datasets/synthetic/dev_buff_exp3/uniform"
  [p3]="/work/u4063895/datasets/synthetic/dev_buff_exp4_p3/uniform"
  [p6]="/work/u4063895/datasets/synthetic/dev_buff_exp4_p6/uniform"
)

# Thresholds to test
THRESHOLDS=(0 500 1000)

# Check that build exists
if [[ ! -f "build/exp4/bench_progressive_filter" ]]; then
  echo "Building bench_progressive_filter..."
  cmake -S benchmarks/experiment4 -B build/exp4 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
  cmake --build build/exp4 -j
fi

# Run all combinations
for precision in exact p3 p6; do
  artifact_root="${ARTIFACTS[$precision]}"
  
  if [[ ! -d "$artifact_root" ]]; then
    echo "WARNING: $artifact_root not found, skipping $precision" >&2
    continue
  fi
  
  echo ""
  echo "=== Testing $precision (artifact: $artifact_root) ==="
  
  for threshold in "${THRESHOLDS[@]}"; do
    csv_file="${RESULT_DIR}/smoke_uniform_${precision}_${threshold}_${SLURM_JOB_ID}.csv"
    
    echo "[smoke] Running: precision=$precision threshold=$threshold"
    ./build/exp4/bench_progressive_filter \
      --device 0 \
      --encoded-root "$artifact_root" \
      --threshold "$threshold" \
      --validate \
      --csv "$csv_file" \
      2>&1 | grep -E '\[exp4\]|error:|ERROR:' || true
    
    if [[ -f "$csv_file" ]]; then
      echo "[smoke] Result saved to $csv_file"
      head -2 "$csv_file" | tail -1 | cut -d, -f1-10,24-27
    else
      echo "[smoke] ERROR: CSV file not created" >&2
    fi
  done
done

echo ""
echo "=== Smoke test complete ==="
echo "Results in: $RESULT_DIR/"
ls -lh "$RESULT_DIR"/smoke_uniform_*_${SLURM_JOB_ID}.csv || true
