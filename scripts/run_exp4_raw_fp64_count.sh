#!/usr/bin/env bash
#SBATCH --job-name=exp4_raw_fp64_count
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/exp4_raw_fp64_count_%j.log
#SBATCH --error=logs/exp4_raw_fp64_count_%j.err
#SBATCH --mail-type=END,FAIL
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
JOB_ID="${SLURM_JOB_ID:-nojob}"
RUN_BASE="$ROOT_DIR/results/raw_count_baseline"
RUN_DIR="$RUN_BASE/run_$(date +%Y%m%d_%H%M%S)_job${JOB_ID}_H200"
JOB_MARKER_DIR="$ROOT_DIR/handoff/job_done"
mkdir -p "$ROOT_DIR/logs" "$JOB_MARKER_DIR" "$RUN_DIR"
write_marker() {
  local exit_code="$1"
  local marker_file="$JOB_MARKER_DIR/job_${JOB_ID}.json"
  cat > "$marker_file" << EOF
{
  "job_id": "$JOB_ID",
  "job_name": "exp4_raw_fp64_count",
  "exit_status": $exit_code,
  "finished_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "workdir": "$ROOT_DIR",
  "git_commit": "$(cd "$ROOT_DIR" && git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "stdout": "logs/exp4_raw_fp64_count_${JOB_ID}.log",
  "stderr": "logs/exp4_raw_fp64_count_${JOB_ID}.err",
  "result_csv": "$RUN_DIR/raw_fp64_count_baseline.csv",
  "result_dir": "$RUN_DIR",
  "next_action": "Review raw FP64 COUNT comparator results and update the PV1-1a report"
}
EOF
}
trap 'write_marker $?' EXIT
echo "[$(date)] [exp4-raw-count] H200 hardware gate"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q H200 || exit 2
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "[$(date)] [exp4-raw-count] toolchain check"
module purge
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan >/dev/null 2>&1
export CUDACXX="$(command -v nvcc)"
which nvcc
cmake --version
gcc --version | head -n 1
BUILD_DIR="$ROOT_DIR/build/exp4"
cmake -S "$ROOT_DIR/benchmarks/experiment4" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 >/dev/null
cmake --build "$BUILD_DIR" --target bench_raw_fp64_count -j >/dev/null
BIN="$BUILD_DIR/bench_raw_fp64_count"
if [[ ! -x "$BIN" ]]; then
  echo "ERROR: raw FP64 COUNT binary not found: $BIN" >&2
  exit 1
fi
DATA_ROOT="/work/$USER/datasets/synthetic/dev"
datasets=(uniform heavy_tailed sensor zipfian)
selectivities=(50 90 99)
cat > "$RUN_DIR/README.md" << EOF
# PV1-1a Raw FP64 COUNT Comparator
Job ID: ${JOB_ID}
GPU: ${gpu_name}
Binary: ${BIN}
Datasets: ${datasets[*]}
Selectivities: ${selectivities[*]}
This run measures raw FP64 COUNT directly on the dataset-matched synthetic dev
inputs using the strict predicate \`x > threshold\`.
The estimated physical throughput column is set equal to logical throughput by
convention for this comparator. It is not profiler-backed HBM traffic.
EOF
{
  printf 'run_description=raw_fp64_count comparator for dataset-matched synthetic dev inputs\n'
  printf 'job_id=%s\n' "$JOB_ID"
  printf 'run_dir=%s\n' "$RUN_DIR"
  printf 'gpu_name=%s\n' "$gpu_name"
  printf 'git_commit=%s\n' "$(cd "$ROOT_DIR" && git rev-parse HEAD 2>/dev/null || echo unknown)"
  printf 'bin=%s\n' "$BIN"
  printf 'data_root=%s\n' "$DATA_ROOT"
  printf 'datasets=%s\n' "${datasets[*]}"
  printf 'selectivities=%s\n' "${selectivities[*]}"
} > "$RUN_DIR/run_meta.txt"
compute_thresholds() {
  local raw_path="$1"
  shift
  python3 - "$raw_path" "$@" <<'PY'
import sys
import numpy as np
raw_path = sys.argv[1]
selectivities = [int(x) for x in sys.argv[2:]]
data = np.fromfile(raw_path, dtype=np.float64)
for s in selectivities:
    q = 1.0 - (s / 100.0)
    print(f"{s}\t{np.quantile(data, q):.17g}")
PY
}
declare -A THRESHOLDS
per_run_csvs=()
for ds in "${datasets[@]}"; do
  raw_path="$DATA_ROOT/${ds}.f64le.bin"
  if [[ ! -f "$raw_path" ]]; then
    echo "ERROR: raw file not found: $raw_path" >&2
    exit 1
  fi
  while IFS=$'\t' read -r s threshold; do
    THRESHOLDS["${ds},${s}"]="$threshold"
  done < <(compute_thresholds "$raw_path" "${selectivities[@]}")
  for s in "${selectivities[@]}"; do
    threshold="${THRESHOLDS["${ds},${s}"]}"
    csv_file="$RUN_DIR/raw_fp64_count_${ds}_s${s}.csv"
    log_file="$RUN_DIR/raw_fp64_count_${ds}_s${s}.log"
    echo "[$(date)] [exp4-raw-count] running ${ds} s=${s} threshold=${threshold}" >&2
    "$BIN" \
      --device 0 \
      --input "$raw_path" \
      --dataset "$ds" \
      --threshold "$threshold" \
      --target_selectivity "$s" \
      --block 256 \
      --grid_mul 1 \
      --warmup 10 \
      --iters 200 \
      --validate \
      --csv "$csv_file" 2>&1 | tee "$log_file"
    per_run_csvs+=("$csv_file")
  done
done
summary_csv="$RUN_DIR/raw_fp64_count_baseline.csv"
python3 - "$summary_csv" "${per_run_csvs[@]}" <<'PY'
from __future__ import annotations
import csv
import sys
from pathlib import Path
summary_csv = Path(sys.argv[1])
source_csvs = [Path(p) for p in sys.argv[2:]]
rows = []
for source_csv in source_csvs:
    with source_csv.open(newline="") as f:
      reader = csv.DictReader(f)
      row = next(reader)
      rows.append(row)
if not rows:
    raise SystemExit("no rows produced")
fieldnames = list(rows[0].keys())
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
{
  printf 'summary_csv=%s\n' "$summary_csv"
  printf 'per_run_csvs=%s\n' "${per_run_csvs[*]}"
} >> "$RUN_DIR/run_meta.txt"
echo "[$(date)] [exp4-raw-count] summary CSV: $summary_csv"
