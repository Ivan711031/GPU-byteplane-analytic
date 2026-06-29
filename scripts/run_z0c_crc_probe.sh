#!/bin/bash
#SBATCH -J z0c_crc_probe
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00
#SBATCH --mail-type=END,FAIL
set -euo pipefail
echo "=== Phase 3-Z0c CRC Implementation Probe ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME"
if ! echo "$GPU_NAME" | grep -qi "H200"; then echo "FATAL: Expected H200"; exit 2; fi
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
conda activate gpu-byteplane-scan
SRC="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/benchmarks/experiment4_filter_aggregate/z0c_crc_probe.cu"
TMPDIR="${WORK_DIR}/builds/z0c_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
echo "=== Compiling ==="
nvcc -arch=sm_90a -O3 -o "$TMPDIR/z0c_crc_probe" "$SRC" 2>&1
echo "Compilation: SUCCESS"
echo ""
echo "=== Running CRC probe (4096 bytes/unit) ==="
"$TMPDIR/z0c_crc_probe" --bytes 4096
echo ""
echo "=== Running CRC probe (ALLOC_UNIT * 41133 = full dataset) ==="
"$TMPDIR/z0c_crc_probe" --bytes 168480000
echo ""
echo "=== Z0c complete ==="
date
