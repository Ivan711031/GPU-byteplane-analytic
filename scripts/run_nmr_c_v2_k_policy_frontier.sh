#!/bin/bash
# NMR-C v2 K-Aware Policy Frontier Runner
#
# Runs the k-conditioned graded-vs-uniform correctness frontier experiment.
# For each (dataset, k, policy, fault_plan) cell, records:
#   truth_raw (P4/CPU oracle), clean_k_answer (P0 no-fault), faulted_k_answer (P2 with policy replicas)
#
# Must be run from within an sbatch allocation or from a GPU node.
# Usage: ./run_nmr_c_v2_k_policy_frontier.sh <bench_nmr_c_v2_k_sweep_binary> [DATASET_NAME]
set -euo pipefail

BIN="${1:-}"
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
  echo "Usage: $0 <bench_nmr_c_v2_k_sweep_binary> [hurricane_u|cesm_atm_cloud|all]"
  exit 1
fi

DATASET_FILTER="${2:-all}"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS_DIR="${WORKDIR}/scripts"

# ---------- Pilot sweep parameters ----------
K_VALUES="1 2 4 8"
POLICIES="${POLICIES:-graded_B0 graded_B7 graded_B14 graded_B16 uniform_full_r1 uniform_full_r2 uniform_full_r3}"
FAMILIES="${FAMILIES:-F1 F2 F3 F4 F5 F6 F7 F8}"
RATES="1e-6 1e-4 1e-3"
SEEDS="0 1 2"
ITERS=3  # minimal iters for correctness (not timing)

# ---------- Dataset config ----------
declare -A DS_ENCODED DS_RAW DS_N DS_PLANES

DS_ENCODED["hurricane_u"]="/work/u4063895/datasets/scientific/dev_buff_v2_scientific/hurricane_u/bfp-dec12"
DS_RAW["hurricane_u"]="/work/u4063895/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"
DS_N["hurricane_u"]=25000000
DS_PLANES["hurricane_u"]=8

DS_ENCODED["cesm_atm_cloud"]="/work/u4063895/datasets/scientific/dev_buff_v2_scientific/cesm_atm_cloud/bfp-dec12"
DS_RAW["cesm_atm_cloud"]="/work/u4063895/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"
DS_N["cesm_atm_cloud"]=168480000
DS_PLANES["cesm_atm_cloud"]=7

# ---------- Output ----------
RESULT_DIR="${WORKDIR}/results/nmr_c_v2_k_policy_frontier"
FAULT_PLAN_DIR="${RESULT_DIR}/fault_plans"
OUT_DIR="${RESULT_DIR}/pilot"
mkdir -p "${OUT_DIR}" "${FAULT_PLAN_DIR}"

CSV="${OUT_DIR}/nmr_c_v2_k_policy_frontier_pilot_${SLURM_JOB_ID:-manual}.csv"

echo "=== NMR-C v2 K-Aware Policy Frontier ==="
echo "Binary:     $BIN"
echo "CSV:        $CSV"
echo "k values:   $K_VALUES"
echo "Policies:   $POLICIES"
echo "Families:   $FAMILIES"
echo "Rates:      $RATES"
echo "Seeds:      $SEEDS"

# ---------- Select datasets ----------
if [ "$DATASET_FILTER" = "all" ]; then
  DATASETS="hurricane_u cesm_atm_cloud"
else
  DATASETS="$DATASET_FILTER"
fi

# ---------- Generate fault plans (shared across all k for fairness) ----------
for DS in $DATASETS; do
  N=${DS_N[$DS]}
  MAXP=${DS_PLANES[$DS]}
  PLAN_DIR="${FAULT_PLAN_DIR}/${DS}"
  echo ""
  echo "=== Generating fault plans for $DS (N=$N, MAXP=$MAXP) ==="
  python3 "${SCRIPTS_DIR}/generate_nmr_c_v2_fault_plans.py" \
    --n-rows "$N" \
    --max-planes "$MAXP" \
    --k "$MAXP" \
    --families $FAMILIES \
    --rates $RATES \
    --seeds $SEEDS \
    --output-dir "$PLAN_DIR"
  echo "Generated $(ls "$PLAN_DIR"/*.fplan 2>/dev/null | wc -l) .fplan files"
done

# ---------- Rate label mapping ----------
rate_label() {
  case "$1" in
    1e-6|1e-06|1.000000e-06) echo "low" ;;
    1e-4|1e-04|1.000000e-04) echo "mid" ;;
    1e-3|1e-03|1.000000e-03) echo "high" ;;
    *) echo "unknown" ;;
  esac
}

# ---------- Run pilot matrix ----------
TOTAL=0
PASS=0
FAILED=0

for DS in $DATASETS; do
  ENC="${DS_ENCODED[$DS]}"
  RAW="${DS_RAW[$DS]}"
  PLAN_DIR="${FAULT_PLAN_DIR}/${DS}"

  echo ""
  echo "=========================================="
  echo "Dataset: $DS"
  echo "=========================================="

  for POLICY in $POLICIES; do
    echo ""
    echo "--- Policy: $POLICY ---"

    for FAMILY in $FAMILIES; do
      for RATE in $RATES; do
        RLABEL=$(rate_label "$RATE")
        for SEED in $SEEDS; do
          TOTAL=$((TOTAL + 1))

          # Single fault plan shared across all k (fair: faults on ALL planes)
          # k=1,2,4,8 all use the same plan; only the read set differs
          K_STR="1,2,4,8"
          FPLAN=$(ls "${PLAN_DIR}"/fault_plan_${FAMILY}_${RLABEL}_*_seed${SEED}_k${DS_PLANES[$DS]}.fplan 2>/dev/null | head -1)
          if [ -z "$FPLAN" ]; then
            echo "WARNING: no .fplan for $DS $FAMILY rate=$RATE seed=$SEED"
            continue
          fi

          echo -n "  [$TOTAL] $DS policy=$POLICY $FAMILY rate=$RATE seed=$SEED ... "

          OUT=$("$BIN" \
            --dataset "$ENC" \
            --raw "$RAW" \
            --policy "$POLICY" \
            --k "$K_STR" \
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

          PASS=$((PASS + 1))
          echo "OK"
        done
      done
    done
  done
done

echo ""
echo "=== Summary ==="
echo "Total configs: $TOTAL"
echo "OK:     $PASS"
echo "FAILED:   $FAILED"
echo "CSV:   $CSV"
echo "Rows:  $(wc -l < "$CSV" 2>/dev/null || echo 0)"

# Write completion marker
MARKER_DIR="${WORKDIR}/handoff/job_done"
mkdir -p "$MARKER_DIR"
cat > "${MARKER_DIR}/job_${SLURM_JOB_ID:-manual}.json" << EOF
{
  "job_id": "${SLURM_JOB_ID:-manual}",
  "job_name": "${SLURM_JOB_NAME:-manual}",
  "exit_status": $([ "$FAILED" -eq 0 ] && echo 0 || echo 1),
  "finished_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "workdir": "$WORKDIR",
  "git_commit": "$(git rev-parse HEAD)",
  "csv": "$CSV",
  "total_configs": $TOTAL,
  "pass": $PASS,
  "failed": $FAILED,
  "next_action": "check CSV for correctness gates; if pass, proceed to aggregate analysis"
}
EOF
echo "Completion marker: ${MARKER_DIR}/job_${SLURM_JOB_ID:-manual}.json"
