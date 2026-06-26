#!/bin/bash
# run_exp4_strategy_byte_mask_smoke.sbatch
#
# H200 smoke for Exp4 byte_mask warp strategy vs predicated baseline.
# Runs bench_progressive_filter --validate across a matrix of:
#   datasets/artifacts: heavy_tailed_p6, uniform_p10, zipfian_p8
#   selectivities:      50 99
#   block sizes:        128 256 512
#   strategies:         predicated byte_mask
#   repeats:            3
#
# Usage:
#   sbatch --account=<ACCOUNT> scripts/run_exp4_strategy_byte_mask_smoke.sbatch
#
# Output:
#   results/exp4/byte_mask_smoke_<timestamp>_job<JOBID>/
#     combined.csv              — one header + all rows
#     summary_pairs.csv          — paired (predicated_ms, byte_mask_ms, speedup)
#     run_<key>.log              — per-case stderr log
#     run_meta.txt               — matrix, git, toolchain info
#     failures.txt               — gate failure details (if any)
#     gpu_util.csv               — GPU utilization log
#   handoff/job_done/job_<SLURM_JOB_ID>.json

#SBATCH --job-name=exp4-byte-mask-smoke
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ==============================================================================
# Environment
# ==============================================================================
module purge || true
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan

# ==============================================================================
# Dynamic workspace detection
# ==============================================================================
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: ROOT_DIR not found: $ROOT_DIR" >&2
  exit 2
fi
cd "$ROOT_DIR"

# ==============================================================================
# GPU identification
# ==============================================================================
GPU_NAME="unknown"
GPU_TAG="unknown"
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ' 2>/dev/null || echo "unknown")
  GPU_TAG="${GPU_NAME// /_}"
  GPU_TAG="${GPU_TAG:-unknown}"
fi

# ==============================================================================
# Paths (set up before trap so handoff marker knows where to write)
# ==============================================================================
V2_ROOT="/work/u4063895/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
RAW_ROOT="/work/u4063895/datasets/synthetic/dev"

timestamp=$(date +%Y%m%d_%H%M%S)
job_id="${SLURM_JOB_ID:-local}"
run_dir="${ROOT_DIR}/results/exp4/byte_mask_smoke_${timestamp}_job${job_id}_${GPU_TAG}"
handoff_dir="${ROOT_DIR}/handoff/job_done"
handoff_marker="${handoff_dir}/job_${job_id}.json"
combined_csv="${run_dir}/combined.csv"
summary_csv="${run_dir}/summary_pairs.csv"
meta_file="${run_dir}/run_meta.txt"
failures_file="${run_dir}/failures.txt"
gpu_util_log="${run_dir}/gpu_util.csv"

# ==============================================================================
# Matrix definition
# ==============================================================================
declare -a MATRIX_ENTRIES=()
declare -a MATRIX_LABELS=()

register_run() {
  local label="$1"
  MATRIX_LABELS+=("$label")
  MATRIX_ENTRIES+=("$label|$2|$3|$4|$5|$6|$7")
}

# Format: dataset, artifact_p, encoded_subdir
for dataset_info in "heavy_tailed|p6|heavy_tailed_p6" "uniform|p10|uniform_p10" "zipfian|p8|zipfian_p8"; do
  IFS='|' read -r dataset artifact_p encoded_subdir <<< "$dataset_info"
  for sel in 50 99; do
    for block in 128 256 512; do
      for rep in 1 2 3; do
        for strategy in predicated byte_mask; do
          label="${dataset}_p${artifact_p}_s${sel}_b${block}_r${rep}_${strategy}"
          register_run "$label" "$dataset" "$encoded_subdir" "$sel" "$block" "$rep" "$strategy"
        done
      done
    done
  done
done

TOTAL_RUNS=${#MATRIX_ENTRIES[@]}

# ==============================================================================
# Threshold computation helper
# ==============================================================================
declare -A THRESHOLD_CACHE

compute_threshold() {
  local dataset="$1" selectivity="$2"
  local cache_key="${dataset}_s${selectivity}"
  if [[ -v "THRESHOLD_CACHE[$cache_key]" ]]; then
    printf '%s\n' "${THRESHOLD_CACHE[$cache_key]}"
    return 0
  fi
  local raw_path="${RAW_ROOT}/${dataset}.f64le.bin"
  if [[ ! -f "$raw_path" ]]; then
    echo "ERROR: raw file not found: $raw_path" >&2
    exit 2
  fi
  local quantile threshold
  quantile=$(python3 -c "print(1 - ${selectivity}/100.0)")
  threshold=$(python3 -c "
import numpy as np
data = np.fromfile('${raw_path}', dtype=np.float64)
print(np.quantile(data, ${quantile}))
")
  THRESHOLD_CACHE["$cache_key"]="$threshold"
  printf '%s\n' "$threshold"
}

# ==============================================================================
# Handoff marker via trap
# ==============================================================================
write_handoff_marker() {
  local status=$?
  local gpulog=""
  if [[ -n "${TRACKER_PID:-}" ]]; then
    kill "$TRACKER_PID" 2>/dev/null || true
    wait "$TRACKER_PID" 2>/dev/null || true
  fi
  if [[ -f "$gpu_util_log" ]]; then
    gpulog="$gpu_util_log"
  fi
  local git_commit
  git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
  local next_action="Review combined CSV, summary pairs, and gate results."
  if [[ -f "$failures_file" ]] && [[ -s "$failures_file" ]]; then
    next_action="Gate failures detected; investigate failures.txt and logs before re-running."
  fi
  mkdir -p "$handoff_dir"
  cat > "$handoff_marker" <<EOF
{
  "job_id": "${job_id}",
  "job_name": "exp4-byte-mask-smoke",
  "exit_status": ${status},
  "finished_at": "$(date -Is)",
  "workdir": "$(pwd)",
  "git_commit": "${git_commit}",
  "result_dir": "${run_dir}",
  "combined_csv": "${combined_csv}",
  "summary_csv": "${summary_csv}",
  "gpu_util_log": "${gpulog}",
  "next_action": "${next_action}"
}
EOF
}
trap write_handoff_marker EXIT

# ==============================================================================
# Hardware validation: must be H200
# ==============================================================================
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi

# ==============================================================================
# Create output directories
# ==============================================================================
mkdir -p "$run_dir" "$handoff_dir"

# ==============================================================================
# GPU utilization tracker (background)
# ==============================================================================
echo "timestamp,gpu_index,util_pct,mem_used_mb,mem_total_mb" > "$gpu_util_log"
(
  while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits | tr -d ' ' \
    | awk -v ts="$(date +%Y-%m-%dT%H:%M:%S)" '{print ts","$0}' >> "$gpu_util_log"
    sleep 60
  done
) &
TRACKER_PID=$!

# ==============================================================================
# Toolchain verification
# ==============================================================================
echo "=== Toolchain ===" >&2
echo "nvcc: $(which nvcc) $($(which nvcc) --version 2>&1 | grep release)" >&2
echo "gcc: $(gcc --version 2>&1 | head -1)" >&2
echo "cmake: $(cmake --version 2>&1 | head -1)" >&2

# ==============================================================================
# Job metadata header
# ==============================================================================
echo "=== Exp4 Byte-Mask Strategy Smoke ===" >&2
echo "Job ID: ${job_id}" >&2
echo "Host: $(hostname)" >&2
echo "Time: $(date -Is)" >&2
echo "GPU: ${GPU_NAME}" >&2
echo "Matrix: ${TOTAL_RUNS} runs" >&2
echo "Output: ${run_dir}" >&2
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader >&2
echo "=== Git ===" >&2
git rev-parse HEAD >&2
git status --short >&2
echo "" >&2

# ==============================================================================
# Build
# ==============================================================================
echo "=== Build ===" >&2
cmake -S benchmarks/experiment4 -B build/exp4 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 2>&1
cmake --build build/exp4 -j 2>&1
echo "" >&2

# ==============================================================================
# Validation: ensure all encoded roots exist
# ==============================================================================
echo "=== Checking artifact paths ===" >&2
declare -A SEEN_ENCODED
for entry in "${MATRIX_ENTRIES[@]}"; do
  IFS='|' read -r label dataset encoded_subdir sel block rep strategy <<< "$entry"
  encoded_root="${V2_ROOT}/${encoded_subdir}"
  if [[ -z "${SEEN_ENCODED[$encoded_root]:-}" ]]; then
    SEEN_ENCODED["$encoded_root"]=1
    if [[ ! -d "$encoded_root" ]]; then
      echo "ERROR: encoded root not found: $encoded_root" >&2
      exit 2
    fi
    echo "  ✓ $encoded_root" >&2
  fi
done
echo "" >&2

# ==============================================================================
# Run matrix
# ==============================================================================
declare -a CSV_FILES=()
declare -a GATE_FAILURES=()
HEADER_WRITTEN=0

run_idx=0
for entry in "${MATRIX_ENTRIES[@]}"; do
  IFS='|' read -r label dataset encoded_subdir sel block rep strategy <<< "$entry"
  ((++run_idx))

  encoded_root="${V2_ROOT}/${encoded_subdir}"
  threshold=$(compute_threshold "$dataset" "$sel")
  log_file="${run_dir}/run_${label}.log"
  csv_file="${run_dir}/run_${label}.csv"

  echo "[${run_idx}/${TOTAL_RUNS}] ${label}  (threshold=${threshold}, block=${block}, strategy=${strategy})" >&2

  # Build and run benchmark
  set +e
  "${ROOT_DIR}/build/exp4/bench_progressive_filter" \
    --device 0 \
    --encoded-root "$encoded_root" \
    --threshold "$threshold" \
    --block "$block" \
    --validate \
    --warp-strategy "$strategy" \
    --csv "$csv_file" \
    --warmup 10 \
    --iters 200 \
    2> "$log_file"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "  ✗ Benchmark exit code ${rc}" >&2
    GATE_FAILURES+=("${label}_EXIT${rc}")
    continue
  fi

  if [[ ! -f "$csv_file" ]]; then
    echo "  ✗ No CSV produced" >&2
    GATE_FAILURES+=("${label}_NOCSV")
    continue
  fi

  # Validate correctness gate from CSV:
  #   gpu_count      = f12
  #   cpu_encoded_count = f13
  #   validated      = f29 (must be "true")
  gpu_count=$(tail -1 "$csv_file" | cut -d',' -f12)
  cpu_encoded=$(tail -1 "$csv_file" | cut -d',' -f13)
  validated=$(tail -1 "$csv_file" | cut -d',' -f29)

  if [[ "$gpu_count" != "$cpu_encoded" ]]; then
    echo "  ✗ GATE FAILURE: gpu=${gpu_count} cpu_enc=${cpu_encoded}" >&2
    GATE_FAILURES+=("${label}_gpu${gpu_count}_cpu${cpu_encoded}")
    continue
  fi

  if [[ "$validated" != "true" ]]; then
    echo "  ✗ GATE FAILURE: validated=${validated} (expected true)" >&2
    GATE_FAILURES+=("${label}_val${validated}")
    continue
  fi

  # Append to combined CSV
  if [[ "$HEADER_WRITTEN" -eq 0 ]]; then
    head -1 "$csv_file" > "$combined_csv"
    HEADER_WRITTEN=1
  fi
  tail -1 "$csv_file" >> "$combined_csv"
  CSV_FILES+=("$csv_file")

  echo "  ✓ gpu=${gpu_count} cpu=${cpu_encoded} validated=${validated}" >&2
done

echo "" >&2

# ==============================================================================
# Results summary
# ==============================================================================
PASSED_COUNT=${#CSV_FILES[@]}
FAILED_COUNT=${#GATE_FAILURES[@]}

echo "=== Results ===" >&2
echo "Total: ${TOTAL_RUNS}  Passed: ${PASSED_COUNT}  Failed: ${FAILED_COUNT}" >&2

if [[ "$HEADER_WRITTEN" -eq 1 ]]; then
  row_count=$(tail -n +2 "$combined_csv" | wc -l)
  echo "Combined CSV: ${combined_csv} (${row_count} rows)" >&2
fi

# ==============================================================================
# Build summary_pairs.csv: pair predicated vs byte_mask per (dataset, sel, block, rep)
# CSV columns: dataset,selectivity,block,rep,predicated_ms,byte_mask_ms,speedup
# ==============================================================================
if [[ "$HEADER_WRITTEN" -eq 1 ]]; then
  echo "=== Building summary pairs ===" >&2
  export run_dir
  python3 << 'PYEOF'
import csv, os, re, glob

run_dir = os.environ.get('run_dir', '')
if not run_dir:
    print("ERROR: run_dir not set", file=__import__('sys').stderr)
    __import__('sys').exit(1)

# Pattern: run_{dataset}_p{artifact_p}_s{sel}_b{block}_r{rep}_{strategy}.csv
pat = re.compile(r'run_(\w+)_p(\w+)_s(\d+)_b(\d+)_r(\d+)_(predicated|byte_mask)\.csv$')

# Collect ms_per_iter from each per-run CSV
records = []  # (dataset, artifact_p, sel, block, rep, strategy, ms)
for fname in os.listdir(run_dir):
    m = pat.match(fname)
    if not m:
        continue
    dataset, artifact_p, sel, block, rep, strategy = m.groups()
    fpath = os.path.join(run_dir, fname)
    try:
        with open(fpath, newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) < 2:
                continue
            # ms_per_iter is field 11 (0-indexed: 10)
            ms_str = rows[-1][10].strip()
            ms = float(ms_str)
    except (csv.Error, ValueError, IndexError, OSError):
        continue
    records.append((dataset, artifact_p, int(sel), int(block), int(rep), strategy, ms))

# Group and pair
pairs = {}
for dataset, ap, sel, block, rep, strategy, ms in records:
    key = (dataset, ap, sel, block, rep)
    if key not in pairs:
        pairs[key] = {}
    pairs[key][strategy] = ms

outpath = os.path.join(run_dir, 'summary_pairs.csv')
with open(outpath, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['dataset', 'selectivity', 'block', 'rep', 'predicated_ms', 'byte_mask_ms', 'speedup'])
    for (dataset, ap, sel, block, rep), strat_map in sorted(pairs.items()):
        pred_ms = strat_map.get('predicated')
        byte_ms = strat_map.get('byte_mask')
        if pred_ms is not None and byte_ms is not None and byte_ms > 0:
            speedup = pred_ms / byte_ms
            w.writerow([f'{dataset}_p{ap}', sel, block, rep,
                        f'{pred_ms:.6f}', f'{byte_ms:.6f}', f'{speedup:.6f}'])
        elif pred_ms is not None:
            w.writerow([f'{dataset}_p{ap}', sel, block, rep,
                        f'{pred_ms:.6f}', 'NA', 'NA'])
        elif byte_ms is not None:
            w.writerow([f'{dataset}_p{ap}', sel, block, rep,
                        'NA', f'{byte_ms:.6f}', 'NA'])

print(f"  Wrote {outpath} ({len(pairs)} row(s))")
PYEOF
fi

# Write failure details
if [[ ${#GATE_FAILURES[@]} -gt 0 ]]; then
  {
    echo "Gate failures (${FAILED_COUNT}):"
    for f in "${GATE_FAILURES[@]}"; do
      echo "  - $f"
    done
  } > "$failures_file"
  echo "Failures written to: ${failures_file}" >&2
fi

# ==============================================================================
# run_meta.txt
# ==============================================================================
cat > "$meta_file" <<METAEOF
=== Exp4 Byte-Mask Strategy Smoke — run_meta ===

Date: $(date -Is)
Host: $(hostname)
GPU: ${GPU_NAME}
Job ID: ${job_id}
Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)
Git status:
$(git status --short 2>/dev/null || echo "  (not a git repo)")

=== Toolchain ===
nvcc version: $($(which nvcc) --version 2>&1 | grep release || echo unknown)
gcc version: $(gcc --version 2>&1 | head -1)
cmake version: $(cmake --version 2>&1 | head -1)

=== Matrix ===
Strategies: predicated byte_mask
Repeats per combo: 3
Artifact root: ${V2_ROOT}
Raw root: ${RAW_ROOT}

Dataset entries:
$(for entry in "${MATRIX_ENTRIES[@]}"; do
  IFS='|' read -r label dataset encoded_subdir sel block rep strategy <<< "$entry"
  echo "  ${label}"
done)

=== Dataset mapping ===
  heavy_tailed → heavy_tailed_p6 (max_planes=8)
  uniform      → uniform_p10      (max_planes=6)
  zipfian      → zipfian_p8       (max_planes=8)

=== Threshold derivation ===
  thresholds = np.quantile(raw_data, 1 - selectivity/100)
  raw files: ${RAW_ROOT}/{dataset}.f64le.bin

=== Results ===
Total runs: ${TOTAL_RUNS}
Passed:     ${PASSED_COUNT}
Failed:     ${FAILED_COUNT}
Output dir: ${run_dir}
Combined CSV: ${combined_csv}
Summary CSV: ${summary_csv}
METAEOF

echo "Meta: ${meta_file}" >&2

# ==============================================================================
# Exit with nonzero if any gate failures
# ==============================================================================
if [[ ${#GATE_FAILURES[@]} -gt 0 ]]; then
  echo "⚠ ${FAILED_COUNT} gate failure(s) — exiting nonzero" >&2
  exit 1
fi

echo "✓ All gates passed" >&2
exit 0
