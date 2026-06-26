#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/results/ncu_smoke/run_$(date +%Y%m%d_%H%M%S)}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <benchmark-cmd> [args...]" >&2
  exit 2
fi

cmd=("$@")

mkdir -p "$RUN_DIR"

if command -v module >/dev/null 2>&1; then
  module purge
  module load miniconda3/26.1.1 || true
  module load cuda/12.6 || true
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate gpu-byteplane-scan >/dev/null 2>&1 || true
fi

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
case "$gpu_name" in
  *H200*) ;;
  *)
    echo "Expected H200, got ${gpu_name}" >&2
    exit 2
    ;;
esac

{
  printf 'timestamp=%s\n' "$(date +%Y%m%d_%H%M%S)"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'gpu_name=%s\n' "$gpu_name"
  printf 'git_commit=%s\n' "$(cd "$ROOT_DIR" && git rev-parse HEAD 2>/dev/null || echo unknown)"
  printf 'ncu_version_begin\n'
  ncu --version || true
  printf 'ncu_version_end\n'
  printf 'command=%q' "${cmd[0]}"
  for arg in "${cmd[@]:1}"; do
    printf ' %q' "$arg"
  done
  printf '\n'
} > "$RUN_DIR/run_meta.txt"

ncu --set full \
  --target-processes all \
  --launch-count 1 \
  --launch-skip 0 \
  --import-source yes \
  --source-folders "$ROOT_DIR" \
  --force-overwrite true \
  --export "$RUN_DIR/ncu_profile" \
  "${cmd[@]}"
