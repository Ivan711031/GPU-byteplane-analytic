#!/bin/bash
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -J nmr-c-v2-k-pol-c2
#SBATCH -t 00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaihc@narlabs.org.tw
#SBATCH -o results/nmr_c_v2_k_policy_frontier/pilot_c2_%j.out
#SBATCH -e results/nmr_c_v2_k_policy_frontier/pilot_c2_%j.err

set -euo pipefail

echo "=== NMR-C v2 K-Aware Policy Frontier CESM CONTINUATION ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Started: $(date)"

# ---- Hardware validation ----
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU"
if ! echo "$GPU" | grep -q "H200"; then
  echo "FATAL: expected H200, got: $GPU"
  exit 2
fi

# ---- Environment ----
WORKDIR="/home/u4063895/workspace/gpu-byteplane-reliability-nmr/.worktrees/issue-303-nmr-c-v2-k-policy-frontier"
cd "$WORKDIR"
source /etc/profile.d/lmod.sh 2>/dev/null
ml load miniconda3 cuda 2>&1

# ---- Build ----
BUILD_DIR="${WORKDIR}/build_nmr_c_v2_k_policy"
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
  benchmarks/experiment4_filter_aggregate 2>&1 | tail -1
cmake --build "$BUILD_DIR" --target bench_nmr_c_v2_k_sweep -j$(nproc) 2>&1 | tail -1
BIN="$BUILD_DIR/bench_nmr_c_v2_k_sweep"

# ---- Resume from partial CSV (job 139352 did 330/432 configs) ----
# Completed: graded_B0, graded_B8, graded_B16, uniform_full_r1, uniform_full_r2 (partial: F1-F4, F5 up to seed=1)
# Remaining: uniform_full_r2 (F5 seed=2, F6, F7, F8) + uniform_full_r3 (all)
RESULT_DIR="${WORKDIR}/results/nmr_c_v2_k_policy_frontier"
FAULT_PLAN_DIR="${RESULT_DIR}/fault_plans"
PREV_CSV="${RESULT_DIR}/pilot/nmr_c_v2_k_policy_frontier_pilot_139352.csv"
CSV="${RESULT_DIR}/pilot/nmr_c_v2_k_policy_frontier_pilot_${SLURM_JOB_ID}.csv"

# Copy previous CSV as base
if [ -f "$PREV_CSV" ]; then
  cp "$PREV_CSV" "$CSV"
  echo "Continuing from $PREV_CSV ($(wc -l < "$PREV_CSV") rows)"
else
  echo "WARNING: previous CSV not found, starting fresh"
fi

# ---- Matrix for continuation ----
DATASET_PATH="/work/u4063895/datasets/scientific/dev_buff_v2_scientific/cesm_atm_cloud/bfp-dec12"
RAW_PATH="/work/u4063895/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"
K_VALUES="1,2,4,8"
PLAN_DIR="${FAULT_PLAN_DIR}/cesm_atm_cloud"

# Remaining: uniform_full_r2 (F5 seed=2 only, F6-F8 all) + uniform_full_r3 (all F1-F8)
RATES="1e-6 1e-4 1e-3"
SEEDS="0 1 2"
ITERS=3

rate_label() {
  case "$1" in
    1e-6) echo "low" ;;
    1e-4) echo "mid" ;;
    1e-3) echo "high" ;;
    *) echo "unknown" ;;
  esac
}

TOTAL=0
PASS=0
MISMATCH=0
FAILED=0

run_policy_families() {
  local POLICY="$1"
  shift
  local FAMILIES="$@"

  echo ""
  echo "=== Policy: $POLICY (families: $FAMILIES) ==="

  for FAMILY in $FAMILIES; do
    for RATE in $RATES; do
      RLABEL=$(rate_label "$RATE")
      for SEED in $SEEDS; do
        # Skip already-completed uniform_full_r2 F1-F4, F5 seed=0,1
        if [ "$POLICY" = "uniform_full_r2" ]; then
          if [ "$FAMILY" = "F5" ] && [ "$SEED" = "0" ]; then continue; fi
          if [ "$FAMILY" = "F5" ] && [ "$SEED" = "1" ]; then continue; fi
        fi

        TOTAL=$((TOTAL + 1))

        # Use k=8 fault plan
        FPLAN=$(ls "${PLAN_DIR}"/fault_plan_${FAMILY}_${RLABEL}_*_seed${SEED}_k8.fplan 2>/dev/null | head -1)
        if [ -z "$FPLAN" ]; then
          echo "WARNING: no .fplan for $FAMILY rate=$RATE seed=$SEED"
          continue
        fi

        echo -n "  [$TOTAL] cesm_atm_cloud policy=$POLICY $FAMILY rate=$RATE seed=$SEED ... "

        OUT=$("$BIN" \
          --dataset "$DATASET_PATH" \
          --raw "$RAW_PATH" \
          --policy "$POLICY" \
          --k "$K_VALUES" \
          --iters $ITERS \
          --fault-plan "$FPLAN" \
          --csv "$CSV" 2>&1)
        RC=$?

        if [ $RC -ne 0 ]; then
          FAILED=$((FAILED + 1))
          echo "FAILED (rc=$RC)"
          echo "$OUT" | tail -5
          continue
        fi

        if echo "$OUT" | grep -q "match=MISMATCH"; then
          MISMATCH=$((MISMATCH + 1))
          echo "OK (MISMATCH expected - byteplane vs raw)"
        else
          PASS=$((PASS + 1))
          echo "OK"
        fi
      done
    done
  done
}

# ---- Run remaining configs ----
# uniform_full_r2: completed F1-F4, F5 rate=1e-6, F5 rate=1e-4. Remaining: F5 rate=1e-3 seeds 1,2 + F6-F8
run_policy_families "uniform_full_r2" F5 F6 F7 F8
# uniform_full_r3: none completed, run all F1-F8
run_policy_families "uniform_full_r3" F1 F2 F3 F4 F5 F6 F7 F8

echo ""
echo "=== Continuation Summary ==="
echo "Total configs: $TOTAL"
echo "OK:     $PASS"
echo "MISMATCH: $MISMATCH"
echo "FAILED:   $FAILED"
echo "CSV:   $CSV"
echo "Rows:  $(wc -l < "$CSV")"

# Write completion marker
mkdir -p "${WORKDIR}/handoff/job_done"
cat > "${WORKDIR}/handoff/job_done/job_${SLURM_JOB_ID}.json" << EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "job_name": "${SLURM_JOB_NAME}",
  "exit_status": 0,
  "finished_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "workdir": "$WORKDIR",
  "git_commit": "$(git rev-parse HEAD || echo unknown)",
  "csv": "$CSV",
  "total_configs": $TOTAL,
  "pass": $PASS,
  "mismatch": $MISMATCH,
  "failed": $FAILED,
  "next_action": "merge with job_139352 CSV; run aggregation analysis"
}
EOF
echo "Done at $(date)"
