#!/bin/bash
#SBATCH -J p3w1_baseline
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
OUT_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3w1_baseline_consolidation/job_${SLURM_JOB_ID}"
CANONICAL_CSV="$OUT_DIR/w1_baseline.csv"

cleanup() { local rc=$?; mkdir -p "$PROJ_DIR/handoff/job_done"
  cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{"job_id":${SLURM_JOB_ID},"job_name":"p3w1_baseline","exit_status":${rc},"finished_at":"$(date -Iseconds)","output_dir":"${OUT_DIR}","next_action":"Write W1 baseline consolidation report."}
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
if [[ "$GPU_NAME" != *"H200"* ]]; then echo "ERROR: Expected H200"; exit 2; fi

cd "$PROJ_DIR"

BUILD_DIR="benchmarks/experiment4_filter_aggregate/build"
mkdir -p "$BUILD_DIR"; cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j bench_filter_aggregate bench_graded_nmr
cd "$PROJ_DIR"

FA="./benchmarks/experiment4_filter_aggregate/build/bench_filter_aggregate"
NMR="./benchmarks/experiment4_filter_aggregate/build/bench_graded_nmr"
ASSEMBLE="./scripts/assemble_w1_baseline.py"

CESM_CLOUD="/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg_global"
CESM_RAW="/work/u4063895/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"
HURRICANE_U="/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg_global"
HURRICANE_RAW="/work/u4063895/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"

run_cell() {
    local label=$1 DS=$2 RAW=$3 THRESH=$4 SEG_ROWS=$5 SEL=$6 PLANES=$7

    for K in 2 $PLANES; do
        echo "--- ${label} sel=${SEL} k=${K} ---"

        # B0+B1 via bench_filter_aggregate
        local fa_csv="$OUT_DIR/_fa_${label}_s${SEL}_k${K}.csv"
        $FA --dataset "$DS" --raw "$RAW" --threshold "$THRESH" \
            --segment-rows "$SEG_ROWS" --iters 50 --warmup 10 \
            --max-filter-planes $K --csv "$fa_csv" \
            2>&1 | rtk grep -E "raw_baseline:|result:|speedup"

        for FAULT in no_fault single_byte single_seg multi_seg; do
            local nmr_csv="$OUT_DIR/_nmr_${label}_s${SEL}_k${K}_${FAULT}.csv"
            $NMR --dataset "$DS" --raw "$RAW" --threshold "$THRESH" \
                --segment-rows "$SEG_ROWS" \
                --iters 30 --warmup 5 \
                --compare-mode segment_crc32 --fault $FAULT \
                --csv "$nmr_csv" \
                2>&1 | rtk grep -E "B2 dup|Mismatch|Integrity|Detection|PASS|FAIL"

            # Assemble canonical row
            python3 "$ASSEMBLE" \
                --label "$label" --selectivity "$SEL" --k "$K" \
                --fault "$FAULT" \
                --max-planes "$PLANES" \
                --segment-rows "$SEG_ROWS" \
                --fa-csv "$fa_csv" \
                --nmr-csv "$nmr_csv" \
                --output-csv "$CANONICAL_CSV"
        done
    done
}

# 2 datasets × 3 selectivities × 2 k × 4 fault cases = 48 rows
run_cell cesm_cloud "$CESM_CLOUD" "$CESM_RAW" 0.177 168480000 10 7
run_cell cesm_cloud "$CESM_CLOUD" "$CESM_RAW" 0.029 168480000 50 7
run_cell cesm_cloud "$CESM_CLOUD" "$CESM_RAW" 0.000 168480000 90 7
run_cell hurricane_u "$HURRICANE_U" "$HURRICANE_RAW" 8.94 25000000 10 8
run_cell hurricane_u "$HURRICANE_U" "$HURRICANE_RAW" -2.31 25000000 50 8
run_cell hurricane_u "$HURRICANE_U" "$HURRICANE_RAW" -12.97 25000000 90 8

echo ""
echo "=== Canonical CSV rows ==="
wc -l "$CANONICAL_CSV"
echo ""
rtk cat "$CANONICAL_CSV"
echo ""
echo "Output: $OUT_DIR"
