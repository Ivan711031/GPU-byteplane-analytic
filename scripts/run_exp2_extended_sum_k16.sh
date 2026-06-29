#!/usr/bin/env bash
#SBATCH --job-name=exp2_sum_k16
#SBATCH --partition=ngs8g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=exp2_sum_k16_%j.log
#SBATCH --error=exp2_sum_k16_%j.err
#SBATCH --mail-type=END,FAIL
set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$ROOT_DIR" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
BIN_PATH="/tmp/buff_error_study_k16"
cd "$ROOT_DIR"
g++ -std=c++20 -O2 -Wall -Wextra -pedantic -I buff_encoder \
    buff_encoder/buff_error_study.cpp \
    buff_encoder/buff_codec.cpp \
    -o "$BIN_PATH"
"$BIN_PATH" \
    --segment-size 4096 \
    --max-k 16 \
    --input-dir ${WORK_DIR}/datasets/synthetic/dev \
    --out-dir results/exp2_extended_sum_k16
