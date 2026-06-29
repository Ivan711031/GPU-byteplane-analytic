#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/exp1}"
RESULTS_BASE="${RESULTS_BASE:-$ROOT_DIR/results/exp1}"
DEVICE="${DEVICE:-0}"
N="${N:-100000000}"
BLOCK="${BLOCK:-256}"
GRID_MUL="${GRID_MUL:-1}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-1000}"
CUDA_ARCH="${CUDA_ARCH:-90}"
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
  if [[ "$upper" == *"H100"* ]]; then
    printf 'H100\n'
    return 0
  fi
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
run_dir="$RESULTS_BASE/run_${timestamp}_job${job_id}_${gpu_tag}_contiguous64"
run_desc="${RUN_DESC:-"No description provided for this baseline run."}"
cleanup_failed_run() {
  local status=$?
  if [[ $status -ne 0 && -n "${run_dir:-}" && -d "${run_dir}" ]]; then
    rm -rf "$run_dir"
    printf 'run failed; removed partial output directory: %s\n' "$run_dir" >&2
  fi
}
trap cleanup_failed_run EXIT
mkdir -p "$run_dir"
alloc_bytes=$((N * 8))
setup_file="$run_dir/setup_estimate.txt"
{
  printf 'n=%s\n' "$N"
  printf 'element_bytes=8\n'
  printf 'estimated_data_allocation_bytes=%s\n' "$alloc_bytes"
  printf 'notes=d_out_allocation_depends_on_runtime_occupancy_grid\n'
} > "$setup_file"
cmake_args=(
  -S "$ROOT_DIR/benchmarks/experiment1"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
)
cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" -j
bin="$BUILD_DIR/bench_contiguous_baseline"
scalar_csv="$run_dir/contiguous64_scalar.csv"
vec2_csv="$run_dir/contiguous64_vec2.csv"
scalar_cmd=(
  "$bin"
  --variant scalar
  --device "$DEVICE"
  --n "$N"
  --block "$BLOCK" --grid_mul "$GRID_MUL"
  --warmup "$WARMUP" --iters "$ITERS"
  --csv "$scalar_csv"
)
vec2_cmd=(
  "$bin"
  --variant vec2
  --device "$DEVICE"
  --n "$N"
  --block "$BLOCK" --grid_mul "$GRID_MUL"
  --warmup "$WARMUP" --iters "$ITERS"
  --csv "$vec2_csv"
)
"${scalar_cmd[@]}"
"${vec2_cmd[@]}"
meta_file="$run_dir/run_meta.txt"
{
  printf 'run_description=%s\n' "$run_desc"
  printf 'timestamp=%s\n' "$timestamp"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'pwd=%s\n' "$PWD"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-}"
  printf 'slurm_job_name=%s\n' "${SLURM_JOB_NAME:-}"
  printf 'slurm_job_partition=%s\n' "${SLURM_JOB_PARTITION:-}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
  printf 'gpu_name=%s\n' "$gpu_name"
  printf 'gpu_tag=%s\n' "$gpu_tag"
  printf 'device_index=%s\n' "$DEVICE"
  printf 'cuda_arch=%s\n' "$CUDA_ARCH"
  printf 'git_branch=%s\n' "$git_branch"
  printf 'git_commit=%s\n' "$git_commit"
  printf 'git_dirty=%s\n' "$git_dirty"
  printf 'memory_setup_file=%s\n' "$setup_file"
  printf 'command_contiguous64_scalar=%s\n' "$(join_cmd "${scalar_cmd[@]}")"
  printf 'command_contiguous64_vec2=%s\n' "$(join_cmd "${vec2_cmd[@]}")"
} > "$meta_file"
{
  printf 'cd %q\n' "$ROOT_DIR"
  printf '%s\n' "$(join_cmd "${scalar_cmd[@]}")"
  printf '%s\n' "$(join_cmd "${vec2_cmd[@]}")"
} > "$run_dir/repro_command.txt"
{
  printf '# Adjust metrics and output path as needed.\n'
  printf 'cd %q\n' "$ROOT_DIR"
  printf 'ncu --set full --target-processes all --import-source yes --source-folders %q --export %q %s\n' \
    "$ROOT_DIR" \
    "$run_dir/ncu_contiguous64_scalar" "$(join_cmd "$bin" --variant scalar --device "$DEVICE" --n "$N" --block "$BLOCK" --grid_mul "$GRID_MUL" --warmup "$WARMUP" --iters "$ITERS" --csv "$scalar_csv")"
  printf 'ncu --set full --target-processes all --import-source yes --source-folders %q --export %q %s\n' \
    "$ROOT_DIR" \
    "$run_dir/ncu_contiguous64_vec2" "$(join_cmd "$bin" --variant vec2 --device "$DEVICE" --n "$N" --block "$BLOCK" --grid_mul "$GRID_MUL" --warmup "$WARMUP" --iters "$ITERS" --csv "$vec2_csv")"
} > "$run_dir/ncu_command_template.txt"
printf 'contiguous baseline outputs in: %s\n' "$run_dir"
printf 'setup summary: %s\n' "$setup_file"
printf 'metadata: %s\n' "$meta_file"
