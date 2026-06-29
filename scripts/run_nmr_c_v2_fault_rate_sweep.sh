#!/bin/bash
# NMR-C v2 Fault-Rate Sweep Runner
#
# Generates per-k fault plans and runs the sweep matrix:
#   datasets × paths(P2,P3) × k(1,2,4,all) × families(F1-F8) × rates(low,mid,high) × seeds(0,1,2)
#
# Must be run from within an sbatch allocation or from a GPU node.
set -euo pipefail
BIN="${1:-}"
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
  echo "Usage: $0 <bench_nmr_c_v2_k_sweep_binary>"
  exit 1
fi
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS_DIR="${WORKDIR}/scripts"
# ---------- rate anchors ----------
RATES="1e-6 1e-4 1e-3"
# ---------- k values ----------
K_VALUES="1 2 4 all"
# ---------- per-dataset config ----------
declare -A DS_ENCODED DS_RAW DS_N DS_PLANES
DS_ENCODED["hurricane_u"]="${WORK_DIR}/datasets/scientific/dev_buff_v2_scientific/hurricane_u/bfp-dec12"
DS_RAW["hurricane_u"]="${WORK_DIR}/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"
DS_N["hurricane_u"]=25000000
DS_PLANES["hurricane_u"]=8
DS_ENCODED["cesm_atm_cloud"]="${WORK_DIR}/datasets/scientific/dev_buff_v2_scientific/cesm_atm_cloud/bfp-dec12"
DS_RAW["cesm_atm_cloud"]="${WORK_DIR}/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"
DS_N["cesm_atm_cloud"]=168480000
DS_PLANES["cesm_atm_cloud"]=7
DATASETS="hurricane_u cesm_atm_cloud"
FAMILIES="F1 F2 F3 F4 F5 F6 F7 F8"
SEEDS="0 1 2"
PATHS="P2 P3"
ITERS=10
# ---------- output ----------
RESULT_DIR="${WORKDIR}/results"
FAULT_PLAN_DIR="${RESULT_DIR}/fault_plans_nmr_c_v2"
mkdir -p "${RESULT_DIR}" "${FAULT_PLAN_DIR}"
CSV="${RESULT_DIR}/nmr_c_v2_fault_rate_sweep_${SLURM_JOB_ID:-manual}.csv"
echo "=== NMR-C v2 Fault-Rate Sweep ==="
echo "Binary:  $BIN"
echo "CSV:     $CSV"
echo "Fault plans: $FAULT_PLAN_DIR"
# ---------- Generate per-k fault plans per dataset ----------
for DS in $DATASETS; do
  N=${DS_N[$DS]}
  MAXP=${DS_PLANES[$DS]}
  PLAN_DIR="${FAULT_PLAN_DIR}/${DS}"
  echo ""
  echo "=== Generating fault plans for $DS (N=$N, MAXP=$MAXP) ==="
  for K in $K_VALUES; do
    if [ "$K" = "all" ]; then
      K_ARG="$MAXP"
    else
      K_ARG="$K"
    fi
    python3 "${SCRIPTS_DIR}/generate_nmr_c_v2_fault_plans.py" \
      --n-rows "$N" \
      --max-planes "$MAXP" \
      --k "$K_ARG" \
      --families $FAMILIES \
      --rates $RATES \
      --seeds $SEEDS \
      --output-dir "$PLAN_DIR"
  done
  echo "Fault plans generated: $(ls "$PLAN_DIR"/*.fplan 2>/dev/null | wc -l) .fplan files"
done
# ---------- Run sweep ----------
echo ""
echo "=== Running sweep ==="
echo "Matrix: $(echo $DATASETS | wc -w) ds × $(echo $PATHS | wc -w) paths × $(echo $K_VALUES | wc -w) ks (clean baseline) + $(echo $FAMILIES | wc -w) fam × $(echo $RATES | wc -w) rates × $(echo $SEEDS | wc -w) seeds (fault sweep)"
TOTAL=0
PASS=0
MISMATCH=0
# ---------- Clean no-fault baseline ----------
echo ""
echo "=== Clean no-fault baseline (one per dataset × path × k) ==="
for DS in $DATASETS; do
  ENC="${DS_ENCODED[$DS]}"
  RAW="${DS_RAW[$DS]}"
  for K in $K_VALUES; do
    if [ "$K" = "all" ]; then
      K_STR="all"
    else
      K_STR="$K"
    fi
    for PATH_ in $PATHS; do
      TOTAL=$((TOTAL + 1))
      echo -n "  [baseline $TOTAL] $DS $PATH_ k=$K_STR ... "
      OUT=$("$BIN" \
        --dataset "$ENC" \
        --raw "$RAW" \
        --path "$PATH_" \
        --k "$K_STR" \
        --iters $ITERS \
        --csv "$CSV" 2>&1)
      RC=$?
      if [ $RC -ne 0 ]; then
        echo "FAILED (rc=$RC)"
        echo "$OUT" | tail -5
        continue
      fi
      if echo "$OUT" | grep -q "match=MISMATCH"; then
        echo "MISMATCH (see CSV)"
      else
        PASS=$((PASS + 1))
        echo "OK"
      fi
    done
  done
done
# ---------- Fault-rate sweep ----------
echo ""
echo "=== Fault-rate sweep ==="
for DS in $DATASETS; do
  ENC="${DS_ENCODED[$DS]}"
  RAW="${DS_RAW[$DS]}"
  PLAN_DIR="${FAULT_PLAN_DIR}/${DS}"
  for K in $K_VALUES; do
    if [ "$K" = "all" ]; then
      K_STR="all"
    else
      K_STR="$K"
    fi
    for FAMILY in $FAMILIES; do
      for RATE in $RATES; do
        # Map rate to label for filename matching
        case "$RATE" in
          1e-6|1e-06|1.000000e-06) RLABEL="low" ;;
          1e-4|1e-04|1.000000e-04) RLABEL="mid" ;;
          1e-3|1e-03|1.000000e-03) RLABEL="high" ;;
          *) RLABEL="unknown" ;;
        esac
        for SEED in $SEEDS; do
          # Find the per-k .fplan file
          FPLAN=$(ls "${PLAN_DIR}"/fault_plan_${FAMILY}_${RLABEL}_*_seed${SEED}_k${K_STR}.fplan 2>/dev/null | head -1)
          if [ -z "$FPLAN" ]; then
            echo "WARNING: no .fplan for $DS $FAMILY rate=$RATE seed=$SEED k=$K_STR"
            continue
          fi
          for PATH_ in $PATHS; do
            TOTAL=$((TOTAL + 1))
            echo -n "  [$TOTAL] $DS $PATH_ k=$K_STR $FAMILY rate=$RATE seed=$SEED ... "
            OUT=$("$BIN" \
              --dataset "$ENC" \
              --raw "$RAW" \
              --path "$PATH_" \
              --k "$K_STR" \
              --iters $ITERS \
              --fault-plan "$FPLAN" \
              --csv "$CSV" 2>&1)
            RC=$?
            if [ $RC -ne 0 ]; then
              echo "FAILED (rc=$RC)"
              echo "$OUT" | tail -5
              continue
            fi
            # Check for mismatches
            if echo "$OUT" | grep -q "match=MISMATCH"; then
              MISMATCH=$((MISMATCH + 1))
              echo "MISMATCH (see CSV)"
              echo "$OUT" | grep "match=" | tail -4
            else
              PASS=$((PASS + 1))
              echo "OK"
            fi
          done
        done
      done
    done
  done
done
echo ""
echo "=== Summary ==="
echo "Total configs: $TOTAL"
echo "Pass: $PASS"
echo "Mismatch: $MISMATCH"
echo "CSV: $CSV"
echo "Rows: $(wc -l < "$CSV")"
