#!/usr/bin/env bash
#SBATCH --job-name=exp3_v2_full
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-18%4
#SBATCH --output=results/exp3_v2_full_specialized_sum/array_%A_%a.out
#SBATCH --error=results/exp3_v2_full_specialized_sum/array_%A_%a.err
#
# Note: the parent directory for --output/--error must exist before sbatch.
# Run: mkdir -p results/exp3_v2_full_specialized_sum
set -euo pipefail
# ------------------------------------------------------------------
# H200 fail-fast
# ------------------------------------------------------------------
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null || true)"
if [[ -z "$gpu_name" ]] || [[ "$gpu_name" != *"H200"* ]]; then
  echo "ERROR: expected H200, got: ${gpu_name:-unknown}" >&2
  exit 2
fi
echo "GPU OK: $gpu_name"
# ------------------------------------------------------------------
# Locate repo root
# ------------------------------------------------------------------
DEFAULT_ROOT_DIR="/home/${USER}/workspace/gpu-byteplane-scan-experiments"
is_repo_root() {
  local candidate="$1"
  [[ -n "$candidate" &&
     -x "$candidate/scripts/run_exp3.sh" &&
     -f "$candidate/scripts/run_exp3.sh" &&
     -d "$candidate/benchmarks/experiment3" ]]
}
ROOT_DIR="${EXP3_ROOT_DIR:-}"
if ! is_repo_root "$ROOT_DIR"; then
  ROOT_DIR=""
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && is_repo_root "$SLURM_SUBMIT_DIR"; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
  elif is_repo_root "$DEFAULT_ROOT_DIR"; then
    ROOT_DIR="$DEFAULT_ROOT_DIR"
  else
    echo "could not locate repo root; set EXP3_ROOT_DIR explicitly" >&2
    exit 1
  fi
fi
# ------------------------------------------------------------------
# Array task mapping: 19 tasks (0–18)
# ------------------------------------------------------------------
ARTIFACT_VERSION="v2_2026-05-10"
SUM_REFERENCE_DOMAIN="encoded_full_depth"
case "${SLURM_ARRAY_TASK_ID:-}" in
  # heavy_tailed: p2 max7, p3 max7, p4 max7, p5 max8, p6 max8 (0-4)
  0)
    dataset="heavy_tailed"
    artifact_label="p2"
    precision_power="2"
    max_plane_count="7"
    ;;
  1)
    dataset="heavy_tailed"
    artifact_label="p3"
    precision_power="3"
    max_plane_count="7"
    ;;
  2)
    dataset="heavy_tailed"
    artifact_label="p4"
    precision_power="4"
    max_plane_count="7"
    ;;
  3)
    dataset="heavy_tailed"
    artifact_label="p5"
    precision_power="5"
    max_plane_count="8"
    ;;
  4)
    dataset="heavy_tailed"
    artifact_label="p6"
    precision_power="6"
    max_plane_count="8"
    ;;
  # sensor: p2 max2, p4 max3, p6 max3, p8 max4, p10 max5 (5-9)
  5)
    dataset="sensor"
    artifact_label="p2"
    precision_power="2"
    max_plane_count="2"
    ;;
  6)
    dataset="sensor"
    artifact_label="p4"
    precision_power="4"
    max_plane_count="3"
    ;;
  7)
    dataset="sensor"
    artifact_label="p6"
    precision_power="6"
    max_plane_count="3"
    ;;
  8)
    dataset="sensor"
    artifact_label="p8"
    precision_power="8"
    max_plane_count="4"
    ;;
  9)
    dataset="sensor"
    artifact_label="p10"
    precision_power="10"
    max_plane_count="5"
    ;;
  # uniform: p2 max3, p4 max4, p6 max4, p8 max5, p10 max6 (10-14)
  10)
    dataset="uniform"
    artifact_label="p2"
    precision_power="2"
    max_plane_count="3"
    ;;
  11)
    dataset="uniform"
    artifact_label="p4"
    precision_power="4"
    max_plane_count="4"
    ;;
  12)
    dataset="uniform"
    artifact_label="p6"
    precision_power="6"
    max_plane_count="4"
    ;;
  13)
    dataset="uniform"
    artifact_label="p8"
    precision_power="8"
    max_plane_count="5"
    ;;
  14)
    dataset="uniform"
    artifact_label="p10"
    precision_power="10"
    max_plane_count="6"
    ;;
  # zipfian: p2 max6, p4 max6, p6 max7, p8 max8 (15-18)
  15)
    dataset="zipfian"
    artifact_label="p2"
    precision_power="2"
    max_plane_count="6"
    ;;
  16)
    dataset="zipfian"
    artifact_label="p4"
    precision_power="4"
    max_plane_count="6"
    ;;
  17)
    dataset="zipfian"
    artifact_label="p6"
    precision_power="6"
    max_plane_count="7"
    ;;
  18)
    dataset="zipfian"
    artifact_label="p8"
    precision_power="8"
    max_plane_count="8"
    ;;
  *)
    echo "unsupported SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-unset}" >&2
    exit 2
    ;;
esac
# ------------------------------------------------------------------
# Verify artifact directory and manifest
# ------------------------------------------------------------------
artifacts_base="/work/${USER}/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p"
encoded_root="${artifacts_base}/${dataset}_${artifact_label}"
if [[ ! -d "$encoded_root" ]]; then
  echo "ERROR: artifact directory not found: $encoded_root" >&2
  exit 2
fi
manifest_json="$encoded_root/manifest.json"
if [[ ! -f "$manifest_json" ]]; then
  echo "ERROR: manifest.json not found: $manifest_json" >&2
  exit 2
fi
# Verify max_plane_count matches expected
actual_max="$(python3 -c "
import json, sys
with open('$manifest_json') as f:
    m = json.load(f)
print(m['max_plane_count'])
" 2>/dev/null || echo "")"
if [[ "$actual_max" != "$max_plane_count" ]]; then
  echo "ERROR: max_plane_count mismatch for ${dataset}_${artifact_label}: expected $max_plane_count, manifest says ${actual_max:-missing}" >&2
  exit 2
fi
echo "Artifact verified: ${dataset}_${artifact_label}, max_plane_count=$actual_max"
# ------------------------------------------------------------------
# Ensure runtime results directory exists
# ------------------------------------------------------------------
mkdir -p "$ROOT_DIR/results/exp3_v2_full_specialized_sum"
# ------------------------------------------------------------------
# Export all settings for scripts/run_exp3.sh
# ------------------------------------------------------------------
export MODE="encoded_dev_subcolumns"
export REAL_KERNEL_VARIANT="specialized"
export ENCODED_ROOT="$encoded_root"
export ARTIFACT_VERSION="$ARTIFACT_VERSION"
export ARTIFACT_LABEL="$artifact_label"
export PRECISION_POWER="$precision_power"
export SUM_REFERENCE_DOMAIN="$SUM_REFERENCE_DOMAIN"
export VALIDATE="1"
export WARMUP="10"
export ITERS="200"
export LOAD_STRATEGY="rowpack16"
export REFINE_MIN="0"
export REFINE_MAX="$((max_plane_count - 1))"
export EXP3_SKIP_BUILD="1"
export RESULTS_BASE="$ROOT_DIR/results/exp3_v2_full_specialized_sum"
export CSV_NAME="throughput.csv"
export RUN_DESC="Exp3 v2 specialized SUM throughput: ${dataset}_${artifact_label}, k=1..${max_plane_count}"
# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
cd "$ROOT_DIR"
exec "$ROOT_DIR/scripts/run_exp3.sh"
