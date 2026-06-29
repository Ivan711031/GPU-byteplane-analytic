#!/bin/bash
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -J nmr-c-v2-k-pol-ce
#SBATCH -t 01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaihc@narlabs.org.tw
#SBATCH -o results/nmr_c_v2_k_policy_frontier/pilot_cesm_%j.out
#SBATCH -e results/nmr_c_v2_k_policy_frontier/pilot_cesm_%j.err
set -euo pipefail
echo "=== NMR-C v2 K-Aware Policy Frontier CESM ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_JOB_NODELIST"
echo "Started: $(date)"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU"
if ! echo "$GPU" | grep -q "H200"; then
  echo "FATAL: expected H200, got: $GPU"
  exit 2
fi
WORKDIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/.worktrees/WORKTREE"
cd "$WORKDIR"
source /etc/profile.d/lmod.sh 2>/dev/null
ml load miniconda3 cuda 2>&1
BUILD_DIR="${WORKDIR}/build_nmr_c_v2_k_policy"
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
  benchmarks/experiment4_filter_aggregate 2>&1 | tail -1
cmake --build "$BUILD_DIR" --target bench_nmr_c_v2_k_sweep -j$(nproc) 2>&1 | tail -1
BIN="$BUILD_DIR/bench_nmr_c_v2_k_sweep"
"${WORKDIR}/scripts/run_nmr_c_v2_k_policy_frontier.sh" "$BIN" cesm_atm_cloud
echo ""
echo "Done at $(date)"
