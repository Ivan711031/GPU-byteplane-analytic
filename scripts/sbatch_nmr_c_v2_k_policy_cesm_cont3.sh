#!/bin/bash
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -J nmr-c-v2-k-pol-c3
#SBATCH -t 00:15:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaihc@narlabs.org.tw
#SBATCH -o results/nmr_c_v2_k_policy_frontier/pilot_cesm_c3_%j.out
#SBATCH -e results/nmr_c_v2_k_policy_frontier/pilot_cesm_c3_%j.err
set -euo pipefail
echo "=== CESM continuation 3 (remaining F6-F8 for uniform_full_r3) ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Started: $(date)"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU"
if ! echo "$GPU" | grep -q "H200"; then echo "FATAL: expected H200"; exit 2; fi
WORKDIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/.worktrees/WORKTREE"
cd "$WORKDIR"
source /etc/profile.d/lmod.sh 2>/dev/null
ml load miniconda3 cuda 2>&1
BUILD_DIR="${WORKDIR}/build_nmr_c_v2_k_policy"
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 benchmarks/experiment4_filter_aggregate 2>&1 | tail -1
cmake --build "$BUILD_DIR" --target bench_nmr_c_v2_k_sweep -j$(nproc) 2>&1 | tail -1
BIN="$BUILD_DIR/bench_nmr_c_v2_k_sweep"
RESULT_DIR="${WORKDIR}/results/nmr_c_v2_k_policy_frontier"
PREV_CSV="${RESULT_DIR}/pilot/nmr_c_v2_k_policy_frontier_pilot_139612.csv"
CSV="${RESULT_DIR}/pilot/nmr_c_v2_k_policy_frontier_pilot_${SLURM_JOB_ID}.csv"
if [ -f "$PREV_CSV" ]; then
  cp "$PREV_CSV" "$CSV"
  echo "Continuing from $PREV_CSV ($(wc -l < "$PREV_CSV") rows)"
fi
DATASET_PATH="${WORK_DIR}/datasets/scientific/dev_buff_v2_scientific/cesm_atm_cloud/bfp-dec12"
RAW_PATH="${WORK_DIR}/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"
PLAN_DIR="${RESULT_DIR}/fault_plans/cesm_atm_cloud"
K_VALUES="1,2,4,8"
POLICY="uniform_full_r3"
FAMILIES="F6 F7 F8"
RATES="1e-6 1e-4 1e-3"
SEEDS="0 1 2"
ITERS=3
rate_label() {
  case "$1" in 1e-6) echo "low" ;; 1e-4) echo "mid" ;; 1e-3) echo "high" ;; esac
}
TOTAL=0
for FAMILY in $FAMILIES; do
  for RATE in $RATES; do
    RLABEL=$(rate_label "$RATE")
    for SEED in $SEEDS; do
      TOTAL=$((TOTAL + 1))
      FPLAN=$(ls "${PLAN_DIR}"/fault_plan_${FAMILY}_${RLABEL}_*_seed${SEED}_k8.fplan 2>/dev/null | head -1)
      [ -z "$FPLAN" ] && continue
      echo -n "  [$TOTAL] cesm_atm_cloud policy=$POLICY $FAMILY rate=$RATE seed=$SEED ... "
      OUT=$("$BIN" --dataset "$DATASET_PATH" --raw "$RAW_PATH" \
        --policy "$POLICY" --k "$K_VALUES" --iters $ITERS \
        --fault-plan "$FPLAN" --csv "$CSV" 2>&1)
      RC=$?
      if [ $RC -ne 0 ]; then echo "FAILED (rc=$RC)"; else echo "OK"; fi
    done
  done
done
echo ""
echo "Total configs: $TOTAL"
echo "CSV rows: $(wc -l < "$CSV")"
echo "Done at $(date)"
