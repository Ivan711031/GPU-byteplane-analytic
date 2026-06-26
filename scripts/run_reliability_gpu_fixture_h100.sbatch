#!/usr/bin/env bash
#SBATCH --job-name=rl_gpu_fixture_h100
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/home/u4063895/gpu-byteplane-scan-experiments/logs/rl_gpu_fixture_h100_%j.log
#SBATCH --error=/home/u4063895/gpu-byteplane-scan-experiments/logs/rl_gpu_fixture_h100_%j.err
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT_DIR"
JOB_ID="${SLURM_JOB_ID:-nojob}"
echo "Job $JOB_ID started on $(hostname)"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"H100"* ]] || { echo "Not H100: $GPU_NAME"; exit 2; }

module load miniconda3/24.11.1 cmake/3.24.2 cuda/12.6
conda activate gpu-byteplane-scan 2>/dev/null || true
echo "nvcc: $(which nvcc)"
cmake --version | head -1
gcc --version | head -1

ARTIFACT_DIR="/tmp/rl_gpu_fixture_${JOB_ID}/artifacts"
FAULT_PLAN_DIR="/tmp/rl_gpu_fixture_${JOB_ID}/fault_plans"
CSV_DIR="/tmp/rl_gpu_fixture_${JOB_ID}/results"
mkdir -p "$ARTIFACT_DIR" "$FAULT_PLAN_DIR" "$CSV_DIR"

echo "Building CPU artifacts..."
python3 scripts/build_reliability_tiny_fixture.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --fault-plan-dir "$FAULT_PLAN_DIR" \
  --csv "$CSV_DIR/cpu_ref.csv"

echo "Building GPU binary..."
BUILD_DIR="$ROOT_DIR/build/reliability_layer1_job${JOB_ID}"
cmake -S benchmarks/reliability_layer1 -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "$BUILD_DIR" -j
BIN="$BUILD_DIR/bench_reliability_fixture"

echo "Clean gate..."
"$BIN" --device 0 --n 1000 --artifact-dir "$ARTIFACT_DIR" --run-clean \
  --csv "$CSV_DIR/clean.csv"

FIRST_FP=$(find "$FAULT_PLAN_DIR" -name 'seed_0.json' | head -1)
echo "Faulted gate (first plan: $FIRST_FP)..."
"$BIN" --device 0 --n 1000 --artifact-dir "$ARTIFACT_DIR" \
  --fault-plan "$FIRST_FP" --csv "$CSV_DIR/faulted.csv"

echo "All 32 fault plans..."
ALL_CSV="$CSV_DIR/all_faults.csv"
FIRST=1
for fp in $(find "$FAULT_PLAN_DIR" -name '*.json' | sort); do
  FP_CSV=$(mktemp /tmp/rl_fp_XXXXXX.csv)
  "$BIN" --device 0 --n 1000 --artifact-dir "$ARTIFACT_DIR" \
    --fault-plan "$fp" --csv "$FP_CSV" > /dev/null 2>&1
  if [[ "$FIRST" -eq 1 ]]; then
    head -1 "$FP_CSV" > "$ALL_CSV"
    FIRST=0
  fi
  tail -1 "$FP_CSV" >> "$ALL_CSV"
done

TOTAL=$(tail -n+2 "$ALL_CSV" | wc -l)
MATCH=$(awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $20); if($20 == "true") c++} END{print c+0}' "$ALL_CSV")
echo "DONE: $MATCH/$TOTAL oracle match (expected 32)"

RELIABILITY_RESULTS_ROOT="${RELIABILITY_RESULTS_ROOT:-/work/u4063895/results/reliability_layer1}"
RESULTS_DIR="$RELIABILITY_RESULTS_ROOT/gpu_fixture_$(date +%Y%m%d_%H%M%S)_job${JOB_ID}_H100"
mkdir -p "$RESULTS_DIR"
cp -r "$CSV_DIR/"* "$RESULTS_DIR/"
{
  echo "job_id=$JOB_ID"
  echo "hostname=$(hostname)"
  echo "gpu=$GPU_NAME"
  echo "oracle_match=$MATCH/$TOTAL"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
} > "$RESULTS_DIR/run_meta.txt"
echo "Results: $RESULTS_DIR"

[[ "$MATCH" == "32" ]] || { echo "MISMATCH: expected 32 matches, got $MATCH"; exit 2; }
echo "ALL PASSED"
