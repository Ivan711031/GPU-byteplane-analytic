#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/reliability_layer1}"
RELIABILITY_RESULTS_ROOT="${RELIABILITY_RESULTS_ROOT:-${WORK_DIR}/results/reliability_layer1}"
RESULTS_BASE="${RESULTS_BASE:-$RELIABILITY_RESULTS_ROOT}"
DEVICE="${DEVICE:-0}"
N="${N:-1000}"
BLOCK="${BLOCK:-256}"
GRID_MUL="${GRID_MUL:-1.0}"
CUDA_ARCH="${CUDA_ARCH:-90}"
ARTIFACT_DIR="${ARTIFACT_DIR:-}"
FAULT_PLAN="${FAULT_PLAN:-}"
RUN_CLEAN="${RUN_CLEAN:-1}"
join_cmd() {
  printf '%q ' "$@"
}
detect_gpu_name() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local name
    if name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$DEVICE" 2>/dev/null)"; then
      name="$(printf '%s\n' "$name" | head -n 1)"
      if [[ -n "$name" ]]; then
        printf '%s\n' "$name"
        return 0
      fi
    fi
  fi
  printf 'unknown_gpu\n'
}
normalize_gpu_tag() {
  local raw="$1"
  local upper cleaned
  upper="$(printf '%s' "$raw" | tr '[:lower:]' '[:upper:]')"
  if [[ "$upper" == *"H200"* ]]; then
    printf 'H200\n'
    return 0
  fi
  cleaned="$(printf '%s' "$upper" | tr -cs 'A-Z0-9' '_' | sed -e 's/^_\+//' -e 's/_\+$//')"
  if [[ -z "$cleaned" ]]; then
    cleaned="UNKNOWN_GPU"
  fi
  printf '%s\n' "$cleaned"
}
h200_fail_fast() {
  local name
  name="$(detect_gpu_name)"
  if [[ "$name" != *"H200"* ]]; then
    echo "error: expected H200, got: $name" >&2
    exit 2
  fi
}
git_branch="unknown"
git_commit="unknown"
git_dirty="unknown"
if git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_branch="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  git_commit="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  if git -C "$ROOT_DIR" diff --quiet && git -C "$ROOT_DIR" diff --cached --quiet; then
    git_dirty="clean"
  else
    git_dirty="dirty"
  fi
fi
timestamp="$(date +%Y%m%d_%H%M%S)"
job_id="${SLURM_JOB_ID:-nojob}"
gpu_name="${GPU_NAME_OVERRIDE:-$(detect_gpu_name)}"
gpu_tag="$(normalize_gpu_tag "$gpu_name")"
run_dir="$RESULTS_BASE/run_${timestamp}_job${job_id}_${gpu_tag}"
run_desc="${RUN_DESC:-"reliability layer1 tiny fixture (N=$N)"}"
cleanup_failed_run() {
  local status=$?
  if [[ $status -ne 0 && -n "${run_dir:-}" && -d "${run_dir}" ]]; then
    printf 'run failed; partial output directory retained: %s\n' "$run_dir" >&2
  fi
}
trap cleanup_failed_run EXIT
mkdir -p "$run_dir"
# Fail fast if not H200
h200_fail_fast
if [[ "$N" != "1000" ]]; then
  echo "error: reliability tiny fixture N must be 1000, got: $N" >&2
  exit 2
fi
if [[ -z "$ARTIFACT_DIR" ]]; then
  echo "error: ARTIFACT_DIR is required" >&2
  exit 2
fi
setup_file="$run_dir/setup_estimate.txt"
{
  printf 'n=%s\n' "$N"
  printf 'block=%s\n' "$BLOCK"
  printf 'grid_mul=%s\n' "$GRID_MUL"
  printf 'artifact_dir=%s\n' "$ARTIFACT_DIR"
  printf 'fault_plan=%s\n' "${FAULT_PLAN:-"(none)"}"
  printf 'run_clean=%s\n' "$RUN_CLEAN"
} > "$setup_file"
cmake_args=(
  -S "$ROOT_DIR/benchmarks/reliability_layer1"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
)
cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" -j
bin="$BUILD_DIR/bench_reliability_fixture"
benchmark_csv="$run_dir/reliability_tiny.csv"
rl_cmd=(
  "$bin"
  --device "$DEVICE"
  --n "$N"
  --artifact-dir "$ARTIFACT_DIR"
  --block "$BLOCK"
  --grid-mul "$GRID_MUL"
  --csv "$benchmark_csv"
)
if [[ "$RUN_CLEAN" == "1" ]]; then
  rl_cmd+=(--run-clean)
fi
if [[ -n "$FAULT_PLAN" ]]; then
  rl_cmd+=(--fault-plan "$FAULT_PLAN")
fi
"${rl_cmd[@]}"
meta_file="$run_dir/run_meta.txt"
{
  printf 'run_description=%s\n' "$run_desc"
  printf 'timestamp=%s\n' "$timestamp"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'pwd=%s\n' "$PWD"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-}"
  printf 'gpu_name=%s\n' "$gpu_name"
  printf 'gpu_tag=%s\n' "$gpu_tag"
  printf 'git_branch=%s\n' "$git_branch"
  printf 'git_commit=%s\n' "$git_commit"
  printf 'git_dirty=%s\n' "$git_dirty"
  printf 'command=%s\n' "$(join_cmd "${rl_cmd[@]}")"
} > "$meta_file"
{
  printf 'cd %q\n' "$ROOT_DIR"
  printf '%s\n' "$(join_cmd "${rl_cmd[@]}")"
} > "$run_dir/repro_command.txt"
printf 'reliability tiny outputs in: %s\n' "$run_dir"
printf 'metadata: %s\n' "$meta_file"
