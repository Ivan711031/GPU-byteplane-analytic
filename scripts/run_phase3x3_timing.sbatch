#!/bin/bash
#SBATCH -J p3x3_timing
#SBATCH -p dev
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
set -euo pipefail
PROJ_DIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr"
OUT_DIR="${WORK_DIR}/results/reliability_layer1/phase3/phase3x3_measured_timing/job_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $(hostname)"; echo "Started: $(date -Iseconds)"
module purge; ml load cmake/4.0.0; ml load cuda/12.6; ml load gcc/11.5
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"H200"* ]] || { echo "ERROR: Expected H200"; exit 2; }
cd "$PROJ_DIR"
BUILD_DIR="benchmarks/experiment4_filter_aggregate/build"
mkdir -p "$BUILD_DIR"; cd "$BUILD_DIR"
rm -f CMakeCache.txt
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc) bench_graded_repair_timing 2>&1 | tee "$OUT_DIR/build.log"
cd "$PROJ_DIR"
TMG="./benchmarks/experiment4_filter_aggregate/build/bench_graded_repair_timing"
CESM_CLOUD="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg_global"
CESM_RAW="${WORK_DIR}/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"
HURRICANE_U="${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg_global"
HURRICANE_RAW="${WORK_DIR}/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"
for FIELD in cesm_cloud hurricane_u; do
  echo "=== Timing: $FIELD ===" | tee -a "$OUT_DIR/results.log"
  DS="$CESM_CLOUD"; RAW="$CESM_RAW"
  [ "$FIELD" = "hurricane_u" ] && { DS="$HURRICANE_U"; RAW="$HURRICANE_RAW"; }
  $TMG --dataset "$DS" --raw "$RAW" \
    --csv "$OUT_DIR/${FIELD}_timing.csv" \
    2>&1 | tee -a "$OUT_DIR/results.log"
done
echo "Done: $(date -Iseconds)" | tee -a "$OUT_DIR/results.log"
