#!/bin/bash
#SBATCH -J p3x4_qtc
#SBATCH -p dev
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
set -euo pipefail
PROJ_DIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr"
OUT_DIR="${WORK_DIR}/results/reliability_layer1/phase3/phase3x4_qtc_generalization/job_${SLURM_JOB_ID}"
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
make -j$(nproc) bench_graded_repair_x1 bench_graded_repair_timing 2>&1 | tee "$OUT_DIR/build.log"
cd "$PROJ_DIR"
X1="./benchmarks/experiment4_filter_aggregate/build/bench_graded_repair_x1"
TMG="./benchmarks/experiment4_filter_aggregate/build/bench_graded_repair_timing"
# q/tc artifacts now available via symlinks from dev_buff_v2_scientific
CESM_Q="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_q/seg_global"
HURRICANE_TC="${WORK_DIR}/datasets/locality_sensitivity/hurricane_tc/seg_global"
for FIELD in cesm_atm_q hurricane_tc; do
  echo "=== X4: $FIELD ===" | tee -a "$OUT_DIR/results.log"
  DS="$CESM_Q"; [ "$FIELD" = "hurricane_tc" ] && DS="$HURRICANE_TC"
  # X1-style paired-seed sweep (B=3 only, per X4 acceptance spec)
  $X1 --dataset "$DS" --n-seeds 30 \
    --csv "$OUT_DIR/${FIELD}_x4_accuracy_canonical.csv" \
    --delta-csv "$OUT_DIR/${FIELD}_x4_paired_delta_summary.csv" \
    --storage-csv "$OUT_DIR/${FIELD}_x4_storage_audit.csv" \
    --handoff "$OUT_DIR/${FIELD}_handoff.json" \
    2>&1 | tee -a "$OUT_DIR/results.log"
  # GPU timing for amortized latency
  $TMG --dataset "$DS" \
    --csv "$OUT_DIR/${FIELD}_timing.csv" \
    2>&1 | tee -a "$OUT_DIR/results.log"
done
echo "Done: $(date -Iseconds)" | tee -a "$OUT_DIR/results.log"
