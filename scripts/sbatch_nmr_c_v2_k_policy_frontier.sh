#!/bin/bash
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -J nmr-c-v2-k-pol-plt
#SBATCH -t 00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaihc@narlabs.org.tw
#SBATCH -o results/nmr_c_v2_k_policy_frontier/pilot_%j.out
#SBATCH -e results/nmr_c_v2_k_policy_frontier/pilot_%j.err
set -euo pipefail
DS="${1:-all}"
echo "=== NMR-C v2 K-Aware Policy Frontier PILOT ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_JOB_NODELIST"
echo "Dataset: $DS"
echo "Started: $(date)"
# ---- Hardware validation ----
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU"
if ! echo "$GPU" | grep -q "H200"; then
  echo "FATAL: expected H200, got: $GPU"
  exit 2
fi
# ---- Environment ----
WORKDIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/.worktrees/WORKTREE"
cd "$WORKDIR"
source /etc/profile.d/lmod.sh 2>/dev/null
ml load miniconda3 cuda 2>&1
# ---- Build ----
BUILD_DIR="${WORKDIR}/build_nmr_c_v2_k_policy"
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
  benchmarks/experiment4_filter_aggregate 2>&1 | tail -1
cmake --build "$BUILD_DIR" --target bench_nmr_c_v2_k_sweep -j$(nproc) 2>&1 | tail -1
BIN="$BUILD_DIR/bench_nmr_c_v2_k_sweep"
echo "Binary: $BIN"
echo "nvidia-smi:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# ---- Run pilot matrix ----
"${WORKDIR}/scripts/run_nmr_c_v2_k_policy_frontier.sh" "$BIN" "$DS"
echo ""
echo "Done at $(date)"
