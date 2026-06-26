#!/bin/bash
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --constraint=H200
#SBATCH --time=00:30:00
#SBATCH --job-name=buff_cpu_proxy_h200
#SBATCH --output=results/buff_cpu_proxy/sweep_%j.out
#SBATCH --error=results/buff_cpu_proxy/sweep_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
JOB_ID="${SLURM_JOB_ID:-local}"
STDOUT_PATH="$ROOT_DIR/results/buff_cpu_proxy/sweep_${JOB_ID}.out"
STDERR_PATH="$ROOT_DIR/results/buff_cpu_proxy/sweep_${JOB_ID}.err"
MARKER_DIR="$ROOT_DIR/handoff/job_done"
MARKER_PATH="$MARKER_DIR/job_${JOB_ID}.json"

write_marker() {
    local status="$1"
    mkdir -p "$MARKER_DIR"
    cat > "$MARKER_PATH" <<EOF
{
  "job_id": "$JOB_ID",
  "job_name": "buff_cpu_proxy_h200",
  "exit_status": $status,
  "finished_at": "$(date -Is)",
  "workdir": "$ROOT_DIR",
  "git_commit": "$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)",
  "stdout": "$STDOUT_PATH",
  "stderr": "$STDERR_PATH",
  "result_csv_generic": "$ROOT_DIR/results/buff_cpu_proxy/buff_cpu_proxy_sweep.csv",
  "result_csv_native": "$ROOT_DIR/results/buff_cpu_proxy/buff_cpu_proxy_sweep_native.csv",
  "result_dir": "$ROOT_DIR/results/buff_cpu_proxy",
  "next_action": "Review the headline-only CPU proxy CSVs and fairness audit"
}
EOF
}

trap 'status=$?; write_marker "$status"' EXIT

# Environment setup
module purge
module load miniconda3/26.1.1 cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan

cd "$ROOT_DIR"

echo "[$(date -Is)] Starting BUFF CPU Proxy Sweep Job"
echo "Host: $(hostname)"
read -r GPU_NAME < <(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "GPU: $GPU_NAME"
if [[ "$GPU_NAME" != *H200* ]]; then
    echo "ERROR: expected H200 node, got: $GPU_NAME" >&2
    exit 2
fi
echo "Conda Env: $CONDA_DEFAULT_ENV"

echo "[$(date -Is)] Toolchain"
which nvcc
cmake --version
gcc --version | head -n 1

BUILD_DIR="$ROOT_DIR/build_filter_aggregate"
cmake -S "$ROOT_DIR/benchmarks/experiment4_filter_aggregate" -B "$BUILD_DIR" \
  -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --target buff_cpu_proxy buff_cpu_proxy_native -j"${SLURM_CPUS_PER_TASK:-1}"

DATASETS="${Q7_DATASETS:-cesm_atm_q,cesm_atm_cloud,hurricane_u,hurricane_tc}"
REPLACE_ARGS=()
if [[ "${Q7_REPLACE_OUTPUT:-1}" == "1" ]]; then
  REPLACE_ARGS+=(--replace-output)
fi
python3 scripts/run_buff_cpu_proxy_sweep.py --datasets "$DATASETS" "${REPLACE_ARGS[@]}"

echo "[$(date -Is)] Job complete"
