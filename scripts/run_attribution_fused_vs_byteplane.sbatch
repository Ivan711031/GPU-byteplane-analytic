#!/bin/bash
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=fa_attr
#SBATCH --output=results/exp4_filter_aggregate/attribution_%j.out
#SBATCH --error=results/exp4_filter_aggregate/attribution_%j.err

# ============================================================================
# Exp4 Filter+Aggregate: Raw-Fused vs Byteplane-Fused Attribution Test
# ============================================================================
# Measures all four comparison paths:
#   A. cuDF (reuse external baseline)
#   B. raw-fused CUDA exact (single kernel: filter + count + sum)
#   C. byteplane-fused full-depth (k = max_planes)
#   D. byteplane-fused progressive (k = k* at s≈50%, and k=1)
#
# For each (dataset, selectivity, k), we run the existing bench_filter_aggregate
# which produces BOTH the raw-fused baseline timing AND the byteplane-fused
# timing in a single invocation.
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# H200 fail-fast gate
# ---------------------------------------------------------------------------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi
echo "GPU: $GPU_NAME"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module purge
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan

ROOT_DIR="/home/$USER/workspace/gpu-byteplane-scan-experiments"
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: ROOT_DIR not found: $ROOT_DIR" >&2
  exit 2
fi
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Build (if needed)
# ---------------------------------------------------------------------------
BUILD_DIR="build_filter_aggregate"
BENCH="$BUILD_DIR/bench_filter_aggregate"
if [[ ! -f "$BENCH" ]]; then
  mkdir -p "$BUILD_DIR"
  cmake -S benchmarks/experiment4_filter_aggregate -B "$BUILD_DIR" \
    -DCMAKE_CUDA_COMPILER=$(which nvcc) \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR" -j$(nproc)
fi

# Verify toolchain
which nvcc
cmake --version | head -1
gcc --version | head -1

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_ROOT="/work/$USER/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
RAW_ROOT="/work/$USER/datasets/synthetic/dev"

declare -A DATASET_ARTIFACTS
DATASET_ARTIFACTS[uniform]=uniform_p10
DATASET_ARTIFACTS[heavy_tailed]=heavy_tailed_p6
DATASET_ARTIFACTS[sensor]=sensor_p10
DATASET_ARTIFACTS[zipfian]=zipfian_p8

# k* (minimum k that reaches s≈50%) for each dataset, from formal sweep + cuDF baseline
declare -A K_STAR
K_STAR[sensor]=1
K_STAR[uniform]=1
K_STAR[heavy_tailed]=4
K_STAR[zipfian]=3

# max_planes for each dataset
declare -A MAX_PLANES
MAX_PLANES[sensor]=5
MAX_PLANES[uniform]=6
MAX_PLANES[heavy_tailed]=8
MAX_PLANES[zipfian]=8

SELECTIVITIES=(10 50 90)
WARMUP=20
ITERS=200

# ---------------------------------------------------------------------------
# Result directory
# ---------------------------------------------------------------------------
RESULT_DIR="results/exp4_filter_aggregate/attribution_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}_${GPU_NAME// /}"
mkdir -p "$RESULT_DIR"
ALL_CSV="$RESULT_DIR/all_runs.csv"
echo "Result dir: $RESULT_DIR"

# CSV header
echo "dataset,selectivity_pct,threshold,k,k_star_k,max_planes,ms_per_iter_fused,raw_baseline_ms_per_iter,gpu_count,enc_count,count_ok,iters,warmup" > "$ALL_CSV"

# ---------------------------------------------------------------------------
# Run attribution matrix
# ---------------------------------------------------------------------------
TOTAL=0
PASS=0
FAIL=0

for DATASET in sensor uniform heavy_tailed zipfian; do
  ARTIFACT="${DATASET_ARTIFACTS[$DATASET]}"
  ARTIFACT_ROOT="${DATASET_ROOT}/${ARTIFACT}"
  RAW_FILE="${RAW_ROOT}/${DATASET}.f64le.bin"

  if [[ ! -d "$ARTIFACT_ROOT" ]]; then
    echo "SKIP: artifact not found: $ARTIFACT_ROOT"
    continue
  fi
  if [[ ! -f "$RAW_FILE" ]]; then
    echo "SKIP: raw file not found: $RAW_FILE"
    continue
  fi

  K_STAR_VAL="${K_STAR[$DATASET]}"
  MAX_P_VAL="${MAX_PLANES[$DATASET]}"

  echo ""
  echo "================================================================"
  echo "Dataset: $DATASET  Artifact: $ARTIFACT  k*=$K_STAR_VAL  max=$MAX_P_VAL"
  echo "================================================================"

  for SEL in "${SELECTIVITIES[@]}"; do
    # Compute threshold as quantile(1 - sel/100) of raw data
    QUANTILE=$(python3 -c "print(1 - ${SEL}/100)")
    THRESHOLD=$(python3 -c "
import numpy as np
data = np.fromfile('${RAW_FILE}', dtype=np.float64)
q = np.quantile(data, ${QUANTILE})
print(q)
")

    # For each dataset & selectivity, determine k values to test:
    # - k=1 always
    # - k=k* (representative at s≈50%) 
    # - k=max_planes
    # For sensor and uniform where k*=1, we only test k=1 and k=max_planes (skip duplicate)
    KS=()
    KS+=("1")
    if [[ "$K_STAR_VAL" != "1" ]]; then
      KS+=("$K_STAR_VAL")
    fi
    if [[ "$MAX_P_VAL" != "1" ]]; then
      KS+=("$MAX_P_VAL")
    fi

    for K in "${KS[@]}"; do
      TOTAL=$((TOTAL + 1))
      CSV_OUT="$RESULT_DIR/run_${DATASET}_s${SEL}_k${K}.csv"
      LOG_OUT="$RESULT_DIR/run_${DATASET}_s${SEL}_k${K}.log"
      IS_KSTAR="false"
      if [[ "$K" == "$K_STAR_VAL" ]]; then
        IS_KSTAR="true"
      fi

      echo "  [${DATASET} s=${SEL}% k=${K}] threshold=${THRESHOLD} k*=${IS_KSTAR}"

      if $BENCH \
          --dataset "$ARTIFACT_ROOT" \
          --raw "$RAW_FILE" \
          --threshold "$THRESHOLD" \
          --max-filter-planes "$K" \
          --iters "$ITERS" \
          --warmup "$WARMUP" \
          --validate \
          --block-threads 256 \
          --csv "$CSV_OUT" \
          > "$LOG_OUT" 2>&1; then
        PASS=$((PASS + 1))

        # Extract key fields from CSV and append to summary
        # CSV columns: ms_per_iter, gpu_count, cpu_enc_count, raw_baseline_ms_per_iter, validated
        # Use tail -1 to skip header
        TAIL=$(tail -1 "$CSV_OUT")
        GPU_COUNT=$(echo "$TAIL" | awk -F',' '{print $11}')
        ENC_COUNT=$(echo "$TAIL" | awk -F',' '{print $12}')
        VALIDATED=$(echo "$TAIL" | awk -F',' '{print $56}')
        FUSED_MS=$(echo "$TAIL" | awk -F',' '{print $10}')
        RAW_MS=$(echo "$TAIL" | awk -F',' '{print $38}')
        COUNT_OK="false"
        if [[ "$GPU_COUNT" == "$ENC_COUNT" ]]; then
          COUNT_OK="true"
        fi

        echo "${DATASET},${SEL},${THRESHOLD},${K},${IS_KSTAR},${MAX_P_VAL},${FUSED_MS},${RAW_MS},${GPU_COUNT},${ENC_COUNT},${COUNT_OK},${ITERS},${WARMUP}" >> "$ALL_CSV"

        echo "    OK fused_ms=${FUSED_MS} raw_ms=${RAW_MS} validated=${VALIDATED} count_ok=${COUNT_OK}"
      else
        FAIL=$((FAIL + 1))
        echo "    FAIL"
        tail -5 "$LOG_OUT" 2>/dev/null || true
      fi
    done
  done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Attribution runs complete"
echo "  Total: $TOTAL  PASS: $PASS  FAIL: $FAIL"
echo "  Results: $RESULT_DIR"
echo "  All runs CSV: $ALL_CSV"
echo "================================================================"

# ---------------------------------------------------------------------------
# Post-process: build attribution CSV + analysis
# ---------------------------------------------------------------------------
echo ""
echo "Building attribution analysis..."
python3 - "$ALL_CSV" "$RESULT_DIR" <<'PYEOF'
import csv
import os
import sys

all_csv = sys.argv[1]
result_dir = sys.argv[2]

# Read all runs
rows = []
with open(all_csv) as f:
    reader = csv.DictReader(f)
    for r in reader:
        r['ms_per_iter_fused'] = float(r['ms_per_iter_fused'])
        r['raw_baseline_ms_per_iter'] = float(r['raw_baseline_ms_per_iter'])
        r['selectivity_pct'] = int(r['selectivity_pct'])
        rows.append(r)

# Build attribution CSV (one row per (dataset, selectivity, k) group)
# Fields:
#   dataset, selectivity_pct, threshold, k, is_k_star, max_planes,
#   fused_ms, raw_fused_ms, cudf_ms,
#   speedup_fused_vs_cudf, speedup_raw_vs_cudf,
#   speedup_bp_full_vs_raw, speedup_bp_kstar_vs_raw,
#   count_ok

# cuDF baseline: reuse representative values from existing results
# From cudf_baseline_20260517_183723_job53522_NVIDIAH200 report and the newer cudf_baseline_20260518_212620_job55516
# Using compact_then_reduce, gpu_hot_path numbers (minimal overhead)
CUDF_BASELINE = {
    # (dataset, selectivity_pct_k) -> median_ms
    # At s≈50%:
    ('sensor', 50): 4.643,   # masked_reduce gpu_hot_path: 4.642976
    ('uniform', 50): 4.768,  # masked_reduce gpu_hot_path: 4.768032
    ('heavy_tailed', 50): 4.767,  # masked_reduce gpu_hot_path: 4.766784
    ('zipfian', 50): 4.764,  # masked_reduce gpu_hot_path: 4.768064
    # At s≈10% and s≈90%, we don't have cuDF measurements; note as N/A
}

# Build attribution table
attr_rows = []
for r in rows:
    ds = r['dataset']
    sel = r['selectivity_pct']
    k = int(r['k'])
    is_kstar = r['k_star_k']
    max_p = int(r['max_planes'])
    fused_ms = r['ms_per_iter_fused']
    raw_ms = r['raw_baseline_ms_per_iter']

    # cuDF speedup
    cudf_key = (ds, sel)
    if cudf_key in CUDF_BASELINE:
        cudf_ms = CUDF_BASELINE[cudf_key]
        speedup_fused_vs_cudf = cudf_ms / fused_ms if fused_ms > 0 else 0
        speedup_raw_vs_cudf = cudf_ms / raw_ms if raw_ms > 0 else 0
    else:
        cudf_ms = float('nan')
        speedup_fused_vs_cudf = float('nan')
        speedup_raw_vs_cudf = float('nan')

    # Byteplane vs raw
    if k == max_p:
        speedup_bp_full_vs_raw = raw_ms / fused_ms if fused_ms > 0 else 0
    else:
        speedup_bp_full_vs_raw = float('nan')

    if is_kstar == 'true':
        speedup_bp_kstar_vs_raw = raw_ms / fused_ms if fused_ms > 0 else 0
    else:
        speedup_bp_kstar_vs_raw = float('nan')

    attr_rows.append({
        'dataset': ds,
        'selectivity_pct': sel,
        'threshold': r['threshold'],
        'k': k,
        'is_k_star': is_kstar,
        'max_planes': max_p,
        'fused_ms': fused_ms,
        'raw_fused_ms': raw_ms,
        'cudf_ms': cudf_ms,
        'speedup_fused_vs_cudf': speedup_fused_vs_cudf,
        'speedup_raw_vs_cudf': speedup_raw_vs_cudf,
        'speedup_bp_full_vs_raw': speedup_bp_full_vs_raw,
        'speedup_bp_kstar_vs_raw': speedup_bp_kstar_vs_raw,
        'count_ok': r['count_ok']
    })

# Write attribution CSV
attr_csv = os.path.join(result_dir, 'attribution_raw_fused_vs_byteplane_fused.csv')
fieldnames = [
    'dataset', 'selectivity_pct', 'threshold', 'k', 'is_k_star', 'max_planes',
    'fused_ms', 'raw_fused_ms', 'cudf_ms',
    'speedup_fused_vs_cudf', 'speedup_raw_vs_cudf',
    'speedup_bp_full_vs_raw', 'speedup_bp_kstar_vs_raw',
    'count_ok'
]
with open(attr_csv, 'w') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(attr_rows)

print(f"Attribution CSV: {attr_csv}")

# Print summary table
print()
print("=" * 120)
print("ATTRIBUTION TABLE")
print("=" * 120)
print(f"{'Dataset':<14} {'Sel':>3} {'k':>2} {'k*?':>4} {'max_p':>5} {'Fused ms':>10} {'Raw ms':>10} {'cuDF ms':>10} {'Fused/cuDF':>10} {'Raw/cuDF':>10} {'BP/raw':>8} {'k*/raw':>8} {'CountOK':>8}")
print("-" * 120)
for a in sorted(attr_rows, key=lambda x: (x['dataset'], x['selectivity_pct'], x['k'])):
    cudf_str = f"{a['cudf_ms']:.3f}" if not (isinstance(a['cudf_ms'], float) and (a['cudf_ms'] != a['cudf_ms'])) else "N/A"
    fused_cudf_str = f"{a['speedup_fused_vs_cudf']:.2f}x" if not (isinstance(a['speedup_fused_vs_cudf'], float) and (a['speedup_fused_vs_cudf'] != a['speedup_fused_vs_cudf'])) else "N/A"
    raw_cudf_str = f"{a['speedup_raw_vs_cudf']:.2f}x" if not (isinstance(a['speedup_raw_vs_cudf'], float) and (a['speedup_raw_vs_cudf'] != a['speedup_raw_vs_cudf'])) else "N/A"
    bp_full_str = f"{a['speedup_bp_full_vs_raw']:.3f}x" if not (isinstance(a['speedup_bp_full_vs_raw'], float) and (a['speedup_bp_full_vs_raw'] != a['speedup_bp_full_vs_raw'])) else "N/A"
    bp_kstar_str = f"{a['speedup_bp_kstar_vs_raw']:.3f}x" if not (isinstance(a['speedup_bp_kstar_vs_raw'], float) and (a['speedup_bp_kstar_vs_raw'] != a['speedup_bp_kstar_vs_raw'])) else "N/A"
    print(f"{a['dataset']:<14} {a['selectivity_pct']:>3} {a['k']:>2} {a['is_k_star']:>4} {a['max_planes']:>5} {a['fused_ms']:>10.6f} {a['raw_fused_ms']:>10.6f} {cudf_str:>10} {fused_cudf_str:>10} {raw_cudf_str:>10} {bp_full_str:>8} {bp_kstar_str:>8} {a['count_ok']:>8}")
print("=" * 120)

# Build key conclusions
print()
print("KEY FINDINGS")
print("-" * 80)

# Group by dataset and find rows at s≈50% for conclusions
s50_rows = [a for a in attr_rows if a['selectivity_pct'] == 50]
for a in s50_rows:
    raw_fused = a['raw_fused_ms']
    bp_fused = a['fused_ms']
    cudf = a['cudf_ms']
    is_kstar = a['is_k_star'] == 'true'
    is_full = a['k'] == a['max_planes']

    if is_full:
        fusion_gain = cudf / raw_fused if cudf == cudf else float('nan')
        bp_full_gain = raw_fused / bp_fused
        print(f"  {a['dataset']} (s=50%, k=max={a['k']}):")
        print(f"    Fusion-only (cuDF → raw-fused): {fusion_gain:.2f}x" if fusion_gain == fusion_gain else f"    Fusion-only: N/A (no cuDF)")
        print(f"    BP full-depth vs raw-fused: {bp_full_gain:.3f}x")
    if is_kstar:
        bp_kstar_gain = raw_fused / bp_fused
        print(f"  {a['dataset']} (s=50%, k=k*={a['k']}):")
        print(f"    BP k* vs raw-fused: {bp_kstar_gain:.3f}x")
    if a['k'] == 1:
        bp_k1_gain = raw_fused / bp_fused
        print(f"  {a['dataset']} (s=50%, k=1):")
        print(f"    BP k=1 vs raw-fused: {bp_k1_gain:.3f}x")
    print()

PYEOF

# ---------------------------------------------------------------------------
# Handoff marker
# ---------------------------------------------------------------------------
mkdir -p handoff/job_done
MARKER="handoff/job_done/job_${SLURM_JOB_ID:-local}.json"
cat > "$MARKER" <<EOF
{
  "job_id": "${SLURM_JOB_ID:-local}",
  "job_name": "fa_attribution_raw_vs_bp",
  "exit_status": 0,
  "finished_at": "$(date -Iseconds)",
  "workdir": "$ROOT_DIR",
  "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "result_dir": "$RESULT_DIR",
  "total_points": $TOTAL,
  "pass_count": $PASS,
  "fail_count": $FAIL,
  "attribution_csv": "$RESULT_DIR/attribution_raw_fused_vs_byteplane_fused.csv",
  "all_runs_csv": "$ALL_CSV",
  "next_action": "write_attribution_report"
}
EOF
echo "Handoff marker: $MARKER"
