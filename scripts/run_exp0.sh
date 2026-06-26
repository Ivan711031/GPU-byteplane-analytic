#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/exp0}"
RESULTS_BASE="${RESULTS_BASE:-$ROOT_DIR/results/exp0}"

DEVICE="${DEVICE:-0}"
BYTES_MIN="${BYTES_MIN:-1MB}"
BYTES_MAX="${BYTES_MAX:-8GB}"
BYTES_MULT="${BYTES_MULT:-2}"
BLOCK="${BLOCK:-256}"
GRID_MUL="${GRID_MUL:-1}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-1000}"
CUDA_ARCH="${CUDA_ARCH:-90}"

MASK_STRIDE="${MASK_STRIDE:-2}"
MASK_ACTIVE="${MASK_ACTIVE:-1}"
GATHER_SPAN="${GATHER_SPAN:-0}"
GATHER_SEED="${GATHER_SEED:-1}"

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

cleanup_failed_run() {
  local status=$?
  if [[ $status -ne 0 && -n "${run_dir:-}" && -d "${run_dir}" ]]; then
    rm -rf "$run_dir"
    printf 'run failed; removed partial output directory: %s\n' "$run_dir" >&2
  fi
}
trap cleanup_failed_run EXIT

mkdir -p "$run_dir"

cmake_args=(
  -S "$ROOT_DIR/benchmarks/experiment0"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
)

cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" -j

bin="$BUILD_DIR/bench_hbm_bw"
common_args=(
  --device "$DEVICE"
  --bytes_min "$BYTES_MIN" --bytes_max "$BYTES_MAX" --bytes_mult "$BYTES_MULT"
  --block "$BLOCK" --grid_mul "$GRID_MUL"
  --warmup "$WARMUP" --iters "$ITERS"
)

seq_cmd=("$bin" --mode seq "${common_args[@]}" --csv "$run_dir/exp0_seq.csv")
masked_cmd=("$bin" --mode masked "${common_args[@]}" --mask_stride "$MASK_STRIDE" --mask_active "$MASK_ACTIVE" --csv "$run_dir/exp0_masked.csv")
gather_cmd=("$bin" --mode gather "${common_args[@]}" --gather_span "$GATHER_SPAN" --gather_seed "$GATHER_SEED" --csv "$run_dir/exp0_gather.csv")

"${seq_cmd[@]}"
"${masked_cmd[@]}"
"${gather_cmd[@]}"

meta_file="$run_dir/run_meta.txt"
{
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
  printf 'command_seq=%s\n' "$(join_cmd "${seq_cmd[@]}")"
  printf 'command_masked=%s\n' "$(join_cmd "${masked_cmd[@]}")"
  printf 'command_gather=%s\n' "$(join_cmd "${gather_cmd[@]}")"
} > "$meta_file"

{
  printf 'cd %q\n' "$ROOT_DIR"
  printf '%s\n' "$(join_cmd "${seq_cmd[@]}")"
  printf '%s\n' "$(join_cmd "${masked_cmd[@]}")"
  printf '%s\n' "$(join_cmd "${gather_cmd[@]}")"
} > "$run_dir/repro_command.txt"

{
  printf '# Adjust metrics and output path as needed.\n'
  printf 'cd %q\n' "$ROOT_DIR"
  printf 'ncu --set full --target-processes all --export %q %s\n' \
    "$run_dir/ncu_exp0_seq" "$(join_cmd "$bin" --mode seq "${common_args[@]}" --csv "$run_dir/exp0_seq_ncu.csv")"
} > "$run_dir/ncu_command_template.txt"

printf 'exp0 outputs in: %s\n' "$run_dir"
printf 'metadata: %s\n' "$meta_file"
