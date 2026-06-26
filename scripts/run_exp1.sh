#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/exp1}"
RESULTS_BASE="${RESULTS_BASE:-$ROOT_DIR/results/exp1}"

DEVICE="${DEVICE:-0}"
N="${N:-100000000}"
PLANE_BYTES="${PLANE_BYTES:-1}"
STRATEGY="${STRATEGY:-byte}"
BYTE_VARIANT="${BYTE_VARIANT:-baseline}"
K_MIN="${K_MIN:-1}"
K_MAX="${K_MAX:-8}"
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
run_dir="$RESULTS_BASE/run_${timestamp}_job${job_id}_${gpu_tag}"
run_desc="${RUN_DESC:-"No description provided for this run."}"

cleanup_failed_run() {
  local status=$?
  if [[ $status -ne 0 && -n "${run_dir:-}" && -d "${run_dir}" ]]; then
    rm -rf "$run_dir"
    printf 'run failed; removed partial output directory: %s\n' "$run_dir" >&2
  fi
}
trap cleanup_failed_run EXIT

mkdir -p "$run_dir"

if [[ "$PLANE_BYTES" != "1" && "$PLANE_BYTES" != "2" ]]; then
  echo "error: PLANE_BYTES must be 1 or 2, got: $PLANE_BYTES" >&2
  exit 2
fi

if [[ "$PLANE_BYTES" == "1" ]]; then
  total_planes=8
else
  total_planes=4
fi

byte_variant_args=()
if [[ "$STRATEGY" == "byte" && "$PLANE_BYTES" == "1" ]]; then
  byte_variant_args=(--byte_variant "$BYTE_VARIANT")
elif [[ "$BYTE_VARIANT" != "baseline" ]]; then
  echo "error: BYTE_VARIANT only applies when STRATEGY=byte and PLANE_BYTES=1" >&2
  exit 2
fi

plane_alloc_bytes=$((N * PLANE_BYTES * total_planes))
pointer_array_bytes=$((total_planes * 8))

setup_file="$run_dir/setup_estimate.txt"
{
  printf 'n=%s\n' "$N"
  printf 'plane_bytes=%s\n' "$PLANE_BYTES"
  printf 'total_planes=%s\n' "$total_planes"
  printf 'estimated_plane_allocation_bytes=%s\n' "$plane_alloc_bytes"
  printf 'estimated_pointer_array_bytes=%s\n' "$pointer_array_bytes"
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

bin="$BUILD_DIR/bench_byteplane_scan"
exp1_cmd=(
  "$bin"
  --device "$DEVICE"
  --n "$N"
  --plane_bytes "$PLANE_BYTES"
  --strategy "$STRATEGY"
  "${byte_variant_args[@]}"
  --k_min "$K_MIN" --k_max "$K_MAX"
  --block "$BLOCK" --grid_mul "$GRID_MUL"
  --warmup "$WARMUP" --iters "$ITERS"
  --csv "$run_dir/exp1.csv"
)

"${exp1_cmd[@]}"

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
  printf 'byte_variant=%s\n' "$BYTE_VARIANT"
  printf 'git_branch=%s\n' "$git_branch"
  printf 'git_commit=%s\n' "$git_commit"
  printf 'git_dirty=%s\n' "$git_dirty"
  printf 'memory_setup_file=%s\n' "$setup_file"
  printf 'command_exp1=%s\n' "$(join_cmd "${exp1_cmd[@]}")"
} > "$meta_file"

{
  printf 'cd %q\n' "$ROOT_DIR"
  printf '%s\n' "$(join_cmd "${exp1_cmd[@]}")"
} > "$run_dir/repro_command.txt"

{
  printf '# Adjust metrics and output path as needed.\n'
  printf 'cd %q\n' "$ROOT_DIR"
  printf 'ncu --set full --target-processes all --import-source yes --source-folders %q --export %q %s\n' \
    "$ROOT_DIR" \
    "$run_dir/ncu_exp1" "$(join_cmd "$bin" --device "$DEVICE" --n "$N" --plane_bytes "$PLANE_BYTES" --strategy "$STRATEGY" "${byte_variant_args[@]}" --k_min "$K_MIN" --k_max "$K_MAX" --block "$BLOCK" --grid_mul "$GRID_MUL" --warmup "$WARMUP" --iters "$ITERS" --csv "$run_dir/exp1_ncu.csv")"
} > "$run_dir/ncu_command_template.txt"

printf 'exp1 outputs in: %s\n' "$run_dir"
printf 'setup summary: %s\n' "$setup_file"
printf 'metadata: %s\n' "$meta_file"
