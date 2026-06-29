#!/usr/bin/env bash
#SBATCH --job-name=exp3_raw_fp64_sum
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/exp3_raw_fp64_sum_%j.log
#SBATCH --error=logs/exp3_raw_fp64_sum_%j.err
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
JOB_ID="${SLURM_JOB_ID:-nojob}"
RUN_BASE="$ROOT_DIR/results/exp3_raw_fp64_sum_baseline"
RUN_DIR="$RUN_BASE/run_$(date +%Y%m%d_%H%M%S)_job${JOB_ID}_H200"
JOB_MARKER_DIR="$ROOT_DIR/handoff/job_done"
mkdir -p "$ROOT_DIR/logs" "$JOB_MARKER_DIR" "$RUN_DIR"
write_marker() {
  local exit_code="$1"
  local marker_file="$JOB_MARKER_DIR/job_${JOB_ID}.json"
  cat > "$marker_file" << EOF
{
  "job_id": "$JOB_ID",
  "job_name": "exp3_raw_fp64_sum",
  "exit_status": $exit_code,
  "finished_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "workdir": "$ROOT_DIR",
  "git_commit": "$(cd "$ROOT_DIR" && git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "stdout": "logs/exp3_raw_fp64_sum_${JOB_ID}.log",
  "stderr": "logs/exp3_raw_fp64_sum_${JOB_ID}.err",
  "result_csv": "$RUN_DIR/raw_fp64_sum_baseline.csv",
  "result_dir": "$RUN_DIR",
  "next_action": "Review raw FP64 SUM comparator results and update PV1-6 report"
}
EOF
}
trap 'write_marker $?' EXIT
echo "[$(date)] [exp3-raw] H200 hardware gate"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q H200 || exit 2
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "[$(date)] [exp3-raw] toolchain check"
module load miniconda3/26.1.1 cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan >/dev/null 2>&1
export CUDACXX="$(command -v nvcc)"
which nvcc
cmake --version
gcc --version | head -n 1
BUILD_DIR="$ROOT_DIR/build/exp3"
cmake -S "$ROOT_DIR/benchmarks/experiment3" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 >/dev/null
cmake --build "$BUILD_DIR" --target bench_raw_fp64_sum -j >/dev/null
BIN="$BUILD_DIR/bench_raw_fp64_sum"
if [[ ! -x "$BIN" ]]; then
  echo "ERROR: raw FP64 SUM binary not found: $BIN" >&2
  exit 1
fi
DATA_ROOT="/work/$USER/datasets/synthetic/dev"
datasets=(heavy_tailed sensor uniform zipfian)
cat > "$RUN_DIR/README.md" << EOF
# PV1-6 Raw FP64 SUM Comparator
Job ID: ${JOB_ID}
GPU: H200
Binary: ${BIN}
Datasets: ${datasets[*]}
This run measures raw FP64 SUM throughput directly on the dataset-matched
synthetic dev inputs and is used as the raw comparator for fixed-depth encoded
SUM and progressive encoded SUM.
The estimated physical throughput column is set equal to logical throughput by
convention for this comparator. It is not profiler-backed HBM traffic.
EOF
{
  printf 'run_description=raw_fp64_sum comparator for dataset-matched synthetic dev inputs\n'
  printf 'job_id=%s\n' "$JOB_ID"
  printf 'run_dir=%s\n' "$RUN_DIR"
  printf 'gpu_name=%s\n' "$gpu_name"
  printf 'git_commit=%s\n' "$(cd "$ROOT_DIR" && git rev-parse HEAD 2>/dev/null || echo unknown)"
  printf 'bin=%s\n' "$BIN"
  printf 'data_root=%s\n' "$DATA_ROOT"
  printf 'datasets=%s\n' "${datasets[*]}"
} > "$RUN_DIR/run_meta.txt"
cat > "$RUN_DIR/repro_command.txt" << EOF
cd ${ROOT_DIR}
for ds in ${datasets[*]}; do
  ${BIN} --device 0 --input /work/\$USER/datasets/synthetic/dev/\${ds}.f64le.bin --dataset \${ds} --block 256 --grid_mul 1 --warmup 10 --iters 200 --validate --csv ${RUN_DIR}/raw_fp64_sum_\${ds}.csv
done
EOF
per_dataset_csvs=()
for ds in "${datasets[@]}"; do
  input_path="$DATA_ROOT/${ds}.f64le.bin"
  out_csv="$RUN_DIR/raw_fp64_sum_${ds}.csv"
  echo "[$(date)] [exp3-raw] running $ds -> $out_csv"
  "$BIN" \
    --device 0 \
    --input "$input_path" \
    --dataset "$ds" \
    --block 256 \
    --grid_mul 1 \
    --warmup 10 \
    --iters 200 \
    --validate \
    --csv "$out_csv"
  per_dataset_csvs+=("$out_csv")
done
final_csv="$RUN_DIR/raw_fp64_sum_baseline.csv"
python3 - "$RUN_DIR" "$final_csv" "${per_dataset_csvs[@]}" <<'PY'
from __future__ import annotations
import csv
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
final_csv = Path(sys.argv[2])
source_csvs = [Path(p) for p in sys.argv[3:]]
rows = []
for source_csv in source_csvs:
    with source_csv.open(newline="") as f:
        row = next(csv.DictReader(f))
    rows.append({
        "dataset": row["dataset"],
        "raw_path": row["raw_path"],
        "aggregation": row["aggregation"],
        "baseline_type": row["baseline_type"],
        "n": row["n"],
        "logical_bytes": row["logical_bytes"],
        "logical_GBps": row["logical_GBps"],
        "estimated_physical_GBps": row["estimated_physical_GBps"],
        "rows_per_sec": row["rows_per_sec"],
        "cpu_sum": row["cpu_sum"],
        "gpu_sum": row["gpu_sum"],
        "abs_error": row["abs_error"],
        "rel_error": row["rel_error"],
        "validated": row["validated"],
        "device": row["device"],
        "job_id": row["job_id"],
        "kernel_path": row["kernel_path"],
        "source_run": str(run_dir),
        "source_csv": str(source_csv),
        "notes": "raw fp64 sum comparator; estimated_physical_GBps equals logical_GBps by convention only",
    })
fieldnames = list(rows[0].keys())
with final_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
echo "[$(date)] [exp3-raw] final CSV: $final_csv"
