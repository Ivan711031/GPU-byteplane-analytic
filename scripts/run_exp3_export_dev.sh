#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/exp3_export_dev}"
RESULTS_BASE="${RESULTS_BASE:-$ROOT_DIR/results/exp3_real_export}"
DATASET_ROOT="${DATASET_ROOT:-/work/u4063895/datasets/synthetic/dev}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/work/u4063895/datasets/synthetic/dev_buff_exp3}"
SEGMENT_SIZE="${SEGMENT_SIZE:-4096}"
MAX_VALUES="${MAX_VALUES:-0}"
DATASETS="${DATASETS:-heavy_tailed sensor uniform zipfian}"
HOST_CXX="${HOST_CXX:-/usr/bin/g++}"

join_cmd() {
  printf '%q ' "$@"
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
host_tag="$(hostname | tr '[:lower:]' '[:upper:]' | tr -cs 'A-Z0-9' '_' | sed -e 's/^_\\+//' -e 's/_\\+$//')"
run_dir="$RESULTS_BASE/run_${timestamp}_job${job_id}_${host_tag}"
run_desc="${RUN_DESC:-"Full DEV encoded export for Exp3 real-data mode."}"

mkdir -p "$run_dir"
mkdir -p "$BUILD_DIR"

if [[ ! -x "$HOST_CXX" ]]; then
  echo "error: HOST_CXX not found: $HOST_CXX" >&2
  exit 2
fi

setup_file="$run_dir/setup_estimate.txt"
{
  printf 'dataset_root=%s\n' "$DATASET_ROOT"
  printf 'output_root=%s\n' "$OUTPUT_ROOT"
  printf 'segment_size=%s\n' "$SEGMENT_SIZE"
  printf 'max_values=%s\n' "$MAX_VALUES"
  printf 'datasets=%s\n' "$DATASETS"
  printf 'host_cxx=%s\n' "$HOST_CXX"
} > "$setup_file"

bin="$BUILD_DIR/export_encoded_dev_layout_cpu"
compile_cmd=(
  "$HOST_CXX"
  -std=c++20
  -O3
  -I"$ROOT_DIR/benchmarks/experiment3"
  -I"$ROOT_DIR/buff_encoder"
  "$ROOT_DIR/benchmarks/experiment3/export_encoded_dev_layout.cpp"
  "$ROOT_DIR/benchmarks/experiment3/exp3_real_data_layout.cpp"
  "$ROOT_DIR/buff_encoder/buff_codec.cpp"
  -o "$bin"
)

"${compile_cmd[@]}"

dataset_log="$run_dir/export_commands.txt"
: > "$dataset_log"

for dataset in $DATASETS; do
  input_path="$DATASET_ROOT/${dataset}.f64le.bin"
  export_cmd=(
    "$bin"
    --input "$input_path"
    --output-root "$OUTPUT_ROOT"
    --dataset "$dataset"
    --segment-size "$SEGMENT_SIZE"
  )
  if [[ "$MAX_VALUES" != "0" ]]; then
    export_cmd+=(--max-values "$MAX_VALUES")
  fi

  printf '%s\n' "$(join_cmd "${export_cmd[@]}")" >> "$dataset_log"
  "${export_cmd[@]}" | tee "$run_dir/${dataset}_export.log"
done

meta_file="$run_dir/run_meta.txt"
{
  printf 'run_description=%s\n' "$run_desc"
  printf 'timestamp=%s\n' "$timestamp"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'pwd=%s\n' "$PWD"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-}"
  printf 'slurm_job_name=%s\n' "${SLURM_JOB_NAME:-}"
  printf 'slurm_job_partition=%s\n' "${SLURM_JOB_PARTITION:-}"
  printf 'dataset_root=%s\n' "$DATASET_ROOT"
  printf 'output_root=%s\n' "$OUTPUT_ROOT"
  printf 'segment_size=%s\n' "$SEGMENT_SIZE"
  printf 'max_values=%s\n' "$MAX_VALUES"
  printf 'datasets=%s\n' "$DATASETS"
  printf 'git_branch=%s\n' "$git_branch"
  printf 'git_commit=%s\n' "$git_commit"
  printf 'git_dirty=%s\n' "$git_dirty"
  printf 'build_command=%s\n' "$(join_cmd "${compile_cmd[@]}")"
} > "$meta_file"

{
  printf 'cd %q\n' "$ROOT_DIR"
  printf '%s\n' "$(join_cmd "${compile_cmd[@]}")"
  while IFS= read -r line; do
    printf '%s\n' "$line"
  done < "$dataset_log"
} > "$run_dir/repro_command.txt"

printf 'full DEV export outputs in: %s\n' "$run_dir"
printf 'encoded artifact root: %s\n' "$OUTPUT_ROOT"
