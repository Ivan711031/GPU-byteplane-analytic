#!/bin/bash
#SBATCH -J p3u_graded_nmr
#SBATCH -p dev
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
set -euo pipefail
PROJ_DIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr"
OUT_DIR="${WORK_DIR}/results/reliability_layer1/phase3/phase3u_fused_2replica_nmr/job_${SLURM_JOB_ID}"
cleanup() {
    local exit_code=$?
    mkdir -p "$PROJ_DIR/handoff/job_done"
    cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{
  "job_id": ${SLURM_JOB_ID},
  "job_name": "p3u_graded_nmr",
  "exit_status": ${exit_code},
  "finished_at": "$(date -Iseconds)",
  "output_dir": "${OUT_DIR}",
  "next_action": "Write Phase 3-U graded NMR implementation report."
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
BUILD_DIR="benchmarks/experiment4_filter_aggregate/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j bench_filter_aggregate bench_graded_nmr
cd "$PROJ_DIR"
FA="./benchmarks/experiment4_filter_aggregate/build/bench_filter_aggregate"
NMR="./benchmarks/experiment4_filter_aggregate/build/bench_graded_nmr"
CESM_CLOUD="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg_global"
CESM_RAW="${WORK_DIR}/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"
HURRICANE_U="${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg_global"
HURRICANE_RAW="${WORK_DIR}/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"
for CASE in cesm_cloud hurricane_u; do
    echo "" | tee -a "$OUT_DIR/results.log"
    echo "==============================================" | tee -a "$OUT_DIR/results.log"
    echo " $CASE" | tee -a "$OUT_DIR/results.log"
    echo "==============================================" | tee -a "$OUT_DIR/results.log"
    if [ "$CASE" = "cesm_cloud" ]; then
        DS="$CESM_CLOUD"; RAW="$CESM_RAW"; THRESH=0.177; SEG_ROWS=168480000
    else
        DS="$HURRICANE_U"; RAW="$HURRICANE_RAW"; THRESH=8.94; SEG_ROWS=25000000
    fi
    # B0 + B1 via bench_filter_aggregate (proven working)
    echo "--- B0+B1 (bench_filter_aggregate) ---" | tee -a "$OUT_DIR/results.log"
    $FA \
        --dataset "$DS" --raw "$RAW" \
        --threshold "$THRESH" \
        --segment-rows "$SEG_ROWS" \
        --iters 50 --warmup 10 \
        --max-filter-planes 2 \
        --csv "$OUT_DIR/${CASE}_b0_b1.csv" \
        2>&1 | tee -a "$OUT_DIR/results.log"
    # Extract B1 + B0 times from bench_filter_aggregate CSV
    B0_MS=$(python3 -c "
import csv
with open('$OUT_DIR/${CASE}_b0_b1.csv') as f:
    r = csv.DictReader(f)
    row = next(r)
    print(row['raw_baseline_ms_per_iter'])
")
    B1_MS=$(python3 -c "
import csv
with open('$OUT_DIR/${CASE}_b0_b1.csv') as f:
    r = csv.DictReader(f)
    row = next(r)
    print(row['ms_per_iter'])
")
    echo "  B0=${B0_MS}ms B1=${B1_MS}ms" | tee -a "$OUT_DIR/results.log"
    # Add --b1-time to all B2 invocations for gate computation
    B2_ARGS="--dataset $DS --raw $RAW --threshold $THRESH --segment-rows $SEG_ROWS --b0-time $B0_MS --b1-time $B1_MS --iters 30 --warmup 5"
    echo "" | tee -a "$OUT_DIR/results.log"
    echo "--- B2 (byte_compare, no_fault) ---" | tee -a "$OUT_DIR/results.log"
    $NMR $B2_ARGS --compare-mode byte_compare --fault no_fault \
        --csv "$OUT_DIR/${CASE}_b2_byte_nofault.csv" \
        2>&1 | tee -a "$OUT_DIR/results.log"
    echo "" | tee -a "$OUT_DIR/results.log"
    echo "--- B2 (segment_crc32, no_fault) ---" | tee -a "$OUT_DIR/results.log"
    $NMR $B2_ARGS --compare-mode segment_crc32 --fault no_fault \
        --csv "$OUT_DIR/${CASE}_b2_crc_nofault.csv" \
        2>&1 | tee -a "$OUT_DIR/results.log"
    echo "" | tee -a "$OUT_DIR/results.log"
    echo "--- B2 (byte_compare, single_byte fault) ---" | tee -a "$OUT_DIR/results.log"
    $NMR $B2_ARGS --compare-mode byte_compare --fault single_byte \
        --csv "$OUT_DIR/${CASE}_b2_byte_sfault.csv" \
        2>&1 | tee -a "$OUT_DIR/results.log"
    echo "" | tee -a "$OUT_DIR/results.log"
    echo "--- B2 (segment_crc32, multi_seg fault) ---" | tee -a "$OUT_DIR/results.log"
    $NMR $B2_ARGS --compare-mode segment_crc32 --fault multi_seg \
        --csv "$OUT_DIR/${CASE}_b2_crc_mfault.csv" \
        2>&1 | tee -a "$OUT_DIR/results.log"
done
echo ""
echo "All measurements complete. Output: $OUT_DIR"
