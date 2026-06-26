#!/bin/bash
#SBATCH -J p3s_fused_nmr
#SBATCH -p dev
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u4063895@gs.ncku.edu.tw

set -euo pipefail

PROJ_DIR="/home/u4063895/workspace/gpu-byteplane-reliability-nmr"
OUT_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3s_fused_nmr/job_${SLURM_JOB_ID}"

cleanup() {
    local exit_code=$?
    mkdir -p "$PROJ_DIR/handoff/job_done"
    cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{
  "job_id": ${SLURM_JOB_ID},
  "job_name": "p3s_fused_nmr",
  "exit_status": ${exit_code},
  "finished_at": "$(date -Iseconds)",
  "output_dir": "${OUT_DIR}",
  "next_action": "Write Phase 3-S fused NMR feasibility report."
}
EOF
}
trap cleanup EXIT
mkdir -p "$OUT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date -Iseconds)"

module purge
ml load cmake/4.0.0
ml load cuda/12.6

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME"
if [[ "$GPU_NAME" != *"H200"* ]]; then
    echo "ERROR: Expected H200, got $GPU_NAME"
    exit 2
fi

cd "$PROJ_DIR"

# Build both benchmarks
BUILD_DIR="benchmarks/experiment4_filter_aggregate/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j bench_filter_aggregate bench_nmr_overhead
cd "$PROJ_DIR"

FA="./benchmarks/experiment4_filter_aggregate/build/bench_filter_aggregate"
NMR="./benchmarks/experiment4_filter_aggregate/build/bench_nmr_overhead"

CESM_CLOUD="/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg_global"
CESM_RAW="/work/u4063895/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"
HURRICANE_U="/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg_global"
HURRICANE_RAW="/work/u4063895/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"

for CASE in cesm_cloud hurricane_u; do
    echo "" | tee -a "$OUT_DIR/results.log"
    echo "==============================================" | tee -a "$OUT_DIR/results.log"
    echo " Case: $CASE" | tee -a "$OUT_DIR/results.log"
    echo "==============================================" | tee -a "$OUT_DIR/results.log"

    if [ "$CASE" = "cesm_cloud" ]; then
        DS="$CESM_CLOUD"; RAW="$CESM_RAW"; THRESH=0.177; SEG_ROWS=168480000
    else
        DS="$HURRICANE_U"; RAW="$HURRICANE_RAW"; THRESH=8.94; SEG_ROWS=25000000
    fi

    # bench_filter_aggregate runs both B0 (raw FP64) and B1 (k=2 progressive) internally.
    # It requires --raw for threshold encoding and --max-filter-planes for k depth.
    echo "--- B0 + B1 (k=2 progressive vs raw-fused baseline) ---" | tee -a "$OUT_DIR/${CASE}.log"
    $FA \
        --dataset "$DS" --raw "$RAW" \
        --threshold "$THRESH" \
        --segment-rows "$SEG_ROWS" \
        --iters 50 --warmup 10 \
        --max-filter-planes 2 \
        --csv "$OUT_DIR/${CASE}_b0_b1.csv" \
        2>&1 | tee -a "$OUT_DIR/${CASE}.log"

    # B2 duplicate-read/compare of highest plane
    echo "" | tee -a "$OUT_DIR/${CASE}.log"
    echo "--- B2: duplicate-read/compare of highest plane ---" | tee -a "$OUT_DIR/${CASE}.log"
    $NMR \
        --dataset "$DS" \
        --iters 50 --warmup 10 \
        2>&1 | tee -a "$OUT_DIR/${CASE}.log"
done

echo ""
echo "All measurements complete. Output: $OUT_DIR"
