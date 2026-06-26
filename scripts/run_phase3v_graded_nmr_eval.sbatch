#!/bin/bash
#SBATCH -J p3v_nmr_eval
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
OUT_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3v_graded_nmr_eval/job_${SLURM_JOB_ID}"

cleanup() { local rc=$?; mkdir -p "$PROJ_DIR/handoff/job_done"
  cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{"job_id":${SLURM_JOB_ID},"job_name":"p3v_nmr_eval","exit_status":${rc},"finished_at":"$(date -Iseconds)","output_dir":"${OUT_DIR}","next_action":"Write graded NMR evaluation report."}
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

CESM_CLOUD="/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg_global"
CESM_RAW="/work/u4063895/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"
HURRICANE_U="/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg_global"
HURRICANE_RAW="/work/u4063895/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"

# CSV header
echo "dataset,selectivity,k,mode,compare_mode,fault_case,b0_ms,b1_ms,b2_ms,overhead_ms,margin_ms,overhead_frac,b2_lt_b0,b2_gate,mismatches" \
  > "$OUT_DIR/evaluation_matrix.csv"

run_case() {
    local label=$1 DS=$2 RAW=$3 THRESH=$4 SEG_ROWS=$5 SEL=$6

    # B0+B1 via bench_filter_aggregate
    local fa_csv="$OUT_DIR/${label}_sel${SEL}_b0_b1.csv"
    $FA --dataset "$DS" --raw "$RAW" --threshold "$THRESH" \
        --segment-rows "$SEG_ROWS" --iters 50 --warmup 10 \
        --max-filter-planes 2 --csv "$fa_csv" \
        2>&1 | tee -a "$OUT_DIR/${label}_sel${SEL}.log"

    local B0_MS=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['raw_baseline_ms_per_iter'])")
    local B1_MS=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['ms_per_iter'])")
    echo "  ${label} sel=${SEL}: B0=${B0_MS}ms B1=${B1_MS}ms" | tee -a "$OUT_DIR/${label}_sel${SEL}.log"

    local B2_ARGS="--dataset $DS --raw $RAW --threshold $THRESH --segment-rows $SEG_ROWS"
    B2_ARGS+=" --b0-time $B0_MS --b1-time $B1_MS --iters 30 --warmup 5"

    for CMODE in byte_compare segment_crc32; do
        for FAULT in no_fault single_byte single_seg multi_seg; do
            local b2_csv="$OUT_DIR/${label}_sel${SEL}_${CMODE}_${FAULT}.csv"
            $NMR $B2_ARGS --compare-mode $CMODE --fault $FAULT --csv "$b2_csv" \
                2>&1 | tee -a "$OUT_DIR/${label}_sel${SEL}.log"

            # Compute gate using values from FA CSV + this B2 CSV
            python3 -c "
import csv
b0 = $B0_MS; b1 = $B1_MS
r = csv.DictReader(open('$b2_csv'))
row = next(r)
b2 = float(row['b2_ms'])
ovh = float(row['overhead_ms'])
mismatch = row['mismatches']
margin = b0 - b1
b2_allowed = 'YES' if (b2 < b0 and ovh < 0.9 * margin and margin > 0) else 'NO'
b2_lt = 'YES' if b2 < b0 else 'NO'
ovh_frac = ovh / margin if margin > 0 else 999.0
print(f'$label,$SEL,2,B2,$CMODE,$FAULT,{b0:.4f},{b1:.4f},{b2:.4f},{ovh:.4f},{margin:.4f},{ovh_frac:.4f},{b2_lt},{b2_allowed},{mismatch}')
" >> "$OUT_DIR/evaluation_matrix.csv"
        done
    done
}

# CESM Cloud, 3 selectivities
run_case cesm_cloud "$CESM_CLOUD" "$CESM_RAW" 0.177 168480000 10
run_case cesm_cloud "$CESM_CLOUD" "$CESM_RAW" 0.029 168480000 50
run_case cesm_cloud "$CESM_CLOUD" "$CESM_RAW" 0.000 168480000 90

# Hurricane U, 3 selectivities
run_case hurricane_u "$HURRICANE_U" "$HURRICANE_RAW" 8.94 25000000 10
run_case hurricane_u "$HURRICANE_U" "$HURRICANE_RAW" -2.31 25000000 50
run_case hurricane_u "$HURRICANE_U" "$HURRICANE_RAW" -12.97 25000000 90

echo ""
echo "Evaluation complete."
echo "Matrix: $OUT_DIR/evaluation_matrix.csv"
rtk cat "$OUT_DIR/evaluation_matrix.csv"
