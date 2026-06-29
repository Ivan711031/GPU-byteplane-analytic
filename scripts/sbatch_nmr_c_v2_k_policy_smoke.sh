#!/bin/bash
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -J nmr-c-v2-k-pol-smoke
#SBATCH -t 00:10:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaihc@narlabs.org.tw
#SBATCH -o results/nmr_c_v2_k_policy_frontier/smoke_%j.out
#SBATCH -e results/nmr_c_v2_k_policy_frontier/smoke_%j.err
set -euo pipefail
echo "=== NMR-C v2 K-Aware Policy Frontier SMOKE TEST ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_JOB_NODELIST"
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
echo "nvcc: $(which nvcc)"
# ---- Build ----
BUILD_DIR="${WORKDIR}/build_nmr_c_v2_k_policy"
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
  benchmarks/experiment4_filter_aggregate 2>&1 | tail -3
cmake --build "$BUILD_DIR" --target bench_nmr_c_v2_k_sweep -j$(nproc) 2>&1 | tail -3
BIN="$BUILD_DIR/bench_nmr_c_v2_k_sweep"
echo "Binary: $BIN"
# ---- Smoke matrix (minimal) ----
RESULT_DIR="${WORKDIR}/results/nmr_c_v2_k_policy_frontier/smoke"
mkdir -p "$RESULT_DIR"
CSV="${RESULT_DIR}/smoke_${SLURM_JOB_ID}.csv"
DATASET_PATH="${WORK_DIR}/datasets/scientific/dev_buff_v2_scientific/hurricane_u/bfp-dec12"
RAW_PATH="${WORK_DIR}/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"
# Generate minimal fault plans
python3 "${WORKDIR}/scripts/generate_nmr_c_v2_fault_plans.py" \
  --n-rows 25000000 --max-planes 8 --k 4 \
  --families F1 --rates 1e-4 --seeds 0 \
  --output-dir "$RESULT_DIR"
FPLAN=$(ls "$RESULT_DIR"/fault_plan_F1_*_seed0_k4.fplan 2>/dev/null | head -1)
echo ""
echo "=== Smoke test: graded_B8 vs uniform_full_r2 at k=4, F1 mid, seed=0 ==="
for POLICY in graded_B8 uniform_full_r2; do
  echo ""
  echo "--- Policy: $POLICY ---"
  "$BIN" \
    --dataset "$DATASET_PATH" \
    --raw "$RAW_PATH" \
    --policy "$POLICY" \
    --k 4 \
    --iters 3 \
    --fault-plan "$FPLAN" \
    --csv "$CSV"
done
echo ""
echo "=== Results ==="
echo "CSV: $CSV"
if [ -f "$CSV" ]; then
  echo "Rows: $(wc -l < "$CSV")"
  echo "Header:"
  head -1 "$CSV"
  echo "Data:"
  cat "$CSV" | column -t -s,
else
  echo "ERROR: no CSV output"
  exit 1
fi
# Verify CSV has expected columns
HEADER=$(head -1 "$CSV")
for col in truth_raw clean_k_answer faulted_k_answer total_materialized_B active_prefix_B; do
  if ! echo "$HEADER" | grep -q "$col"; then
    echo "FATAL: missing column $col in CSV"
    exit 1
  fi
done
echo "All required columns present."
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
  "next_action": "if pass, submit full pilot matrix"
}
EOF
echo ""
echo "Smoke test complete. Check $CSV for results."
