#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/exp3}"
RESULTS_BASE="${RESULTS_BASE:-$ROOT_DIR/results/exp3}"

DEVICE="${DEVICE:-0}"
MODE="${MODE:-synthetic_fixed_point_subcolumns}"
ENCODED_ROOT="${ENCODED_ROOT:-}"
V2_ROOT="${V2_ROOT:-}"
REAL_KERNEL_VARIANT="${REAL_KERNEL_VARIANT:-runtime}"
CSV_NAME="${CSV_NAME:-exp3.csv}"
N="${N:-100000000}"
SEGMENT_ROWS="${SEGMENT_ROWS:-1048576}"
SUBCOLUMNS="${SUBCOLUMNS:-8}"
REFINE_MIN="${REFINE_MIN:-0}"
REFINE_MAX="${REFINE_MAX:-7}"
LOAD_STRATEGY="${LOAD_STRATEGY:-rowpack16}"
BLOCK="${BLOCK:-256}"
ITEMS_PER_THREAD="${ITEMS_PER_THREAD:-1}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-200}"
CUDA_ARCH="${CUDA_ARCH:-90}"
EXP3_SKIP_BUILD="${EXP3_SKIP_BUILD:-0}"
ARTIFACT_VERSION="${ARTIFACT_VERSION:-}"
ARTIFACT_LABEL="${ARTIFACT_LABEL:-}"
PRECISION_POWER="${PRECISION_POWER:-}"
SUM_REFERENCE_DOMAIN="${SUM_REFERENCE_DOMAIN:-}"

join_cmd() {
  printf '%q ' "$@"
}

ceil_div() {
  local x="$1"
  local y="$2"
  printf '%s\n' $(((x + y - 1) / y))
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
benchmark_csv="$run_dir/.exp3_raw.csv"
final_csv="$run_dir/$CSV_NAME"

cleanup_failed_run() {
  local status=$?
  if [[ $status -ne 0 && -n "${run_dir:-}" && -d "${run_dir}" ]]; then
    printf 'run failed; partial output directory retained: %s\n' "$run_dir" >&2
  fi
}
trap cleanup_failed_run EXIT

mkdir -p "$run_dir"

if [[ "$LOAD_STRATEGY" != "rowpack16" ]]; then
  echo "error: LOAD_STRATEGY must be rowpack16 in exp3 v1, got: $LOAD_STRATEGY" >&2
  exit 2
fi

if [[ "$MODE" == "encoded_dev_subcolumns" && -z "$ENCODED_ROOT" ]]; then
  if [[ -n "$V2_ROOT" ]]; then
    ENCODED_ROOT="$V2_ROOT"
  fi
fi

if [[ "$MODE" == "encoded_dev_subcolumns" && -z "$ENCODED_ROOT" ]]; then
  echo "error: ENCODED_ROOT is required when MODE=encoded_dev_subcolumns" >&2
  exit 2
fi

if [[ "$MODE" == "encoded_dev_subcolumns" ]]; then
  case "$REAL_KERNEL_VARIANT" in
    runtime|specialized) ;;
    *)
      echo "error: REAL_KERNEL_VARIANT must be runtime or specialized, got: $REAL_KERNEL_VARIANT" >&2
      exit 2
      ;;
  esac
fi

setup_file="$run_dir/setup_estimate.txt"
if [[ "$MODE" == "synthetic_fixed_point_subcolumns" ]]; then
  tile_rows=$((BLOCK * ITEMS_PER_THREAD * 16))
  tiles_per_segment="$(ceil_div "$SEGMENT_ROWS" "$tile_rows")"
  num_segments="$(ceil_div "$N" "$SEGMENT_ROWS")"
  grid=$((num_segments * tiles_per_segment))
  subcolumn_alloc_bytes=$((N * SUBCOLUMNS))
  segment_base_bytes=$((num_segments * 8))
  basis_bytes=$((num_segments * 8 * 8))
  partial_bytes=$((grid * 8))
  {
    printf 'mode=%s\n' "$MODE"
    printf 'n=%s\n' "$N"
    printf 'segment_rows=%s\n' "$SEGMENT_ROWS"
    printf 'subcolumns=%s\n' "$SUBCOLUMNS"
    printf 'refine_min=%s\n' "$REFINE_MIN"
    printf 'refine_max=%s\n' "$REFINE_MAX"
    printf 'load_strategy=%s\n' "$LOAD_STRATEGY"
    printf 'block=%s\n' "$BLOCK"
    printf 'items_per_thread=%s\n' "$ITEMS_PER_THREAD"
    printf 'tile_rows=%s\n' "$tile_rows"
    printf 'tiles_per_segment=%s\n' "$tiles_per_segment"
    printf 'num_segments=%s\n' "$num_segments"
    printf 'grid=%s\n' "$grid"
    printf 'estimated_subcolumn_allocation_bytes=%s\n' "$subcolumn_alloc_bytes"
    printf 'estimated_segment_base_bytes=%s\n' "$segment_base_bytes"
    printf 'estimated_subcolumn_basis_bytes=%s\n' "$basis_bytes"
    printf 'estimated_partial_output_bytes=%s\n' "$partial_bytes"
  } > "$setup_file"
else
  {
    printf 'mode=%s\n' "$MODE"
    printf 'encoded_root=%s\n' "$ENCODED_ROOT"
    printf 'real_kernel_variant=%s\n' "$REAL_KERNEL_VARIANT"
    printf 'csv_name=%s\n' "$CSV_NAME"
    printf 'refine_min=%s\n' "$REFINE_MIN"
    printf 'refine_max=%s\n' "$REFINE_MAX"
    printf 'load_strategy=%s\n' "$LOAD_STRATEGY"
    printf 'block=%s\n' "$BLOCK"
    printf 'items_per_thread=%s\n' "$ITEMS_PER_THREAD"
    printf 'warmup=%s\n' "$WARMUP"
    printf 'iters=%s\n' "$ITERS"
  } > "$setup_file"
fi

bin="$BUILD_DIR/bench_progressive_aggregation"

if [[ "$EXP3_SKIP_BUILD" == "1" ]]; then
  if [[ ! -f "$bin" || ! -x "$bin" ]]; then
    echo "error: EXP3_SKIP_BUILD=1 but $bin not found or not executable" >&2
    exit 2
  fi
else
  cmake_args=(
    -S "$ROOT_DIR/benchmarks/experiment3"
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
  )
  cmake "${cmake_args[@]}"
  cmake --build "$BUILD_DIR" -j
fi
exp3_cmd=(
  "$bin"
  --device "$DEVICE"
  --mode "$MODE"
  --refine_min "$REFINE_MIN" --refine_max "$REFINE_MAX"
  --load_strategy "$LOAD_STRATEGY"
  --block "$BLOCK"
  --items_per_thread "$ITEMS_PER_THREAD"
  --warmup "$WARMUP" --iters "$ITERS"
  --csv "$benchmark_csv"
)

if [[ "$MODE" == "synthetic_fixed_point_subcolumns" ]]; then
  exp3_cmd+=(
    --n "$N"
    --segment_rows "$SEGMENT_ROWS"
    --subcolumns "$SUBCOLUMNS"
  )
else
  exp3_cmd+=(--encoded-root "$ENCODED_ROOT")
  exp3_cmd+=(--real_kernel_variant "$REAL_KERNEL_VARIANT")
fi

if [[ "${VALIDATE:-0}" == "1" || "${VALIDATE:-false}" == "true" ]]; then
  exp3_cmd+=(--validate)
fi

"${exp3_cmd[@]}"

if [[ "$MODE" == "encoded_dev_subcolumns" ]]; then
  kernel_path="runtime_rowpack16"
  if [[ "$REAL_KERNEL_VARIANT" == "specialized" ]]; then
    kernel_path="specialized_rowpack16"
  fi
  python3 - "$benchmark_csv" "$final_csv" "$gpu_tag" "$kernel_path" "$ARTIFACT_VERSION" "$ARTIFACT_LABEL" "$PRECISION_POWER" "$ENCODED_ROOT" "$SUM_REFERENCE_DOMAIN" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

raw_csv = Path(sys.argv[1])
final_csv = Path(sys.argv[2])
gpu_tag = sys.argv[3]
kernel_path = sys.argv[4]
artifact_version = sys.argv[5] if len(sys.argv) > 5 else ""
artifact_label = sys.argv[6] if len(sys.argv) > 6 else ""
precision_power = sys.argv[7] if len(sys.argv) > 7 else ""
encoded_root = sys.argv[8] if len(sys.argv) > 8 else ""
sum_reference_domain = sys.argv[9] if len(sys.argv) > 9 else ""

with raw_csv.open(newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit(f"no rows found in {raw_csv}")


def normalized_field(row: dict[str, str], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    return value.strip()


def resolved_kernel_path(row: dict[str, str], fallback: str) -> str:
    raw = normalized_field(row, "kernel_path")
    return raw or fallback


def resolved_gpu_tag(row: dict[str, str], fallback: str) -> str:
    raw = normalized_field(row, "gpu_tag")
    return raw or fallback

fieldnames = [
    "experiment",
    "dataset",
    "aggregation",
    "mode",
    "kernel_path",
    "k",
    "logical_subcolumns_read",
    "max_plane_count",
    "segment_rows",
    "n",
    "iters",
    "warmup",
    "ms_per_iter",
    "rows_per_sec",
    "billion_rows_per_sec",
    "logical_bytes_per_iter",
    "logical_GBps",
    "gpu_approx_sum",
    "cpu_approx_sum",
    "abs_cpu_gpu_diff",
    "exact_sum",
    "abs_exact_cpu_diff",
    "abs_exact_gpu_diff",
    "device",
    "gpu_tag",
    "validated",
    "refinement_depth",
    "load_strategy",
    "benchmark",
    "logical_bytes",
    "segment_plane_count_min",
    "segment_plane_count_max",
    "encoded_layout",
    "artifact_version",
    "artifact_label",
    "precision_power",
    "encoded_root",
    "sum_reference_domain",
]

with final_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        logical_subcolumns_read = int(row["logical_subcolumns_read"])
        n = int(row["n"])
        row_kernel_path = resolved_kernel_path(row, kernel_path)
        row_gpu_tag = resolved_gpu_tag(row, gpu_tag)
        out = {
            "experiment": "exp3_real_progressive_sum",
            "dataset": row["dataset"],
            "aggregation": row["aggregation"],
            "mode": row["mode"],
            "kernel_path": row_kernel_path,
            "k": str(logical_subcolumns_read),
            "logical_subcolumns_read": row["logical_subcolumns_read"],
            "max_plane_count": row["max_plane_count"],
            "segment_rows": row["segment_rows"],
            "n": row["n"],
            "iters": row["iters"],
            "warmup": row["warmup"],
            "ms_per_iter": row["ms_per_iter"],
            "rows_per_sec": row["rows_per_sec"],
            "billion_rows_per_sec": row["billion_rows_per_sec"],
            "logical_bytes_per_iter": str(n * logical_subcolumns_read),
            "logical_GBps": row["logical_GBps"],
            "gpu_approx_sum": row["gpu_approximate_sum"],
            "cpu_approx_sum": row["cpu_approximate_sum"],
            "abs_cpu_gpu_diff": row["abs_cpu_gpu_diff"],
            "exact_sum": row["exact_sum"],
            "abs_exact_cpu_diff": row["abs_exact_cpu_diff"],
            "abs_exact_gpu_diff": row["abs_exact_gpu_diff"],
            "device": row["device"],
            "gpu_tag": row_gpu_tag,
            "validated": row["validated"],
            "refinement_depth": row["refinement_depth"],
            "load_strategy": row["load_strategy"],
            "benchmark": row["benchmark"],
            "logical_bytes": row["logical_bytes"],
            "segment_plane_count_min": row["segment_plane_count_min"],
            "segment_plane_count_max": row["segment_plane_count_max"],
            "encoded_layout": row["encoded_layout"],
            "artifact_version": artifact_version,
            "artifact_label": artifact_label,
            "precision_power": precision_power,
            "encoded_root": encoded_root,
            "sum_reference_domain": sum_reference_domain,
        }
        writer.writerow(out)
PY
else
  mv "$benchmark_csv" "$final_csv"
fi

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
  printf 'csv_name=%s\n' "$CSV_NAME"
  printf 'benchmark_csv=%s\n' "$benchmark_csv"
  printf 'final_csv=%s\n' "$final_csv"
  printf 'git_branch=%s\n' "$git_branch"
  printf 'git_commit=%s\n' "$git_commit"
  printf 'git_dirty=%s\n' "$git_dirty"
  printf 'memory_setup_file=%s\n' "$setup_file"
  printf 'mode=%s\n' "$MODE"
  printf 'encoded_root=%s\n' "$ENCODED_ROOT"
  printf 'real_kernel_variant=%s\n' "$REAL_KERNEL_VARIANT"
  printf 'artifact_version=%s\n' "$ARTIFACT_VERSION"
  printf 'artifact_label=%s\n' "$ARTIFACT_LABEL"
  printf 'precision_power=%s\n' "$PRECISION_POWER"
  printf 'sum_reference_domain=%s\n' "$SUM_REFERENCE_DOMAIN"
  printf 'command_exp3=%s\n' "$(join_cmd "${exp3_cmd[@]}")"
} > "$meta_file"

{
  printf 'cd %q\n' "$ROOT_DIR"
  printf '%s\n' "$(join_cmd "${exp3_cmd[@]}")"
} > "$run_dir/repro_command.txt"

{
  printf '# Adjust metrics and output path as needed.\n'
  printf 'cd %q\n' "$ROOT_DIR"
  if [[ "$MODE" == "synthetic_fixed_point_subcolumns" ]]; then
    printf 'ncu --set full --target-processes all --launch-count 1 --launch-skip 0 --import-source yes --source-folders %q --export %q %s\n' \
      "$ROOT_DIR" \
      "$run_dir/ncu_exp3_depths" "$(join_cmd "$bin" --device "$DEVICE" --mode "$MODE" --n "$N" --segment_rows "$SEGMENT_ROWS" --subcolumns "$SUBCOLUMNS" --refine_min "$REFINE_MIN" --refine_max "$REFINE_MAX" --load_strategy "$LOAD_STRATEGY" --block "$BLOCK" --items_per_thread "$ITEMS_PER_THREAD" --warmup 0 --iters 1 --csv "$run_dir/exp3_ncu.csv")"
  else
    printf 'ncu --set full --target-processes all --launch-count 1 --launch-skip 0 --import-source yes --source-folders %q --export %q %s\n' \
      "$ROOT_DIR" \
      "$run_dir/ncu_exp3_depths" "$(join_cmd "$bin" --device "$DEVICE" --mode "$MODE" --encoded-root "$ENCODED_ROOT" --real_kernel_variant "$REAL_KERNEL_VARIANT" --refine_min "$REFINE_MIN" --refine_max "$REFINE_MAX" --load_strategy "$LOAD_STRATEGY" --block "$BLOCK" --items_per_thread "$ITEMS_PER_THREAD" --warmup 0 --iters 1 --csv "$run_dir/exp3_ncu.csv")"
  fi
} > "$run_dir/ncu_command_template.txt"

printf 'exp3 outputs in: %s\n' "$run_dir"
printf 'benchmark csv: %s\n' "$benchmark_csv"
printf 'final csv: %s\n' "$final_csv"
printf 'setup summary: %s\n' "$setup_file"
printf 'metadata: %s\n' "$meta_file"
