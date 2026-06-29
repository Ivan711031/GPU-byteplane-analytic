#!/bin/bash
#SBATCH -J p3w5_pareto
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
OUT_DIR="${WORK_DIR}/results/reliability_layer1/phase3/phase3w5_pareto/job_${SLURM_JOB_ID}"
trap_cleanup() { local rc=$?; mkdir -p "$PROJ_DIR/handoff/job_done"
  cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{"job_id":${SLURM_JOB_ID},"job_name":"p3w5_pareto","exit_status":${rc},"finished_at":"$(date -Iseconds)","output_dir":"${OUT_DIR}"}
EOF
}
trap trap_cleanup EXIT
mkdir -p "$OUT_DIR"
module purge; ml load cmake/4.0.0 cuda/12.6
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ "$GPU_NAME" != *"H200"* ]]; then echo "ERROR: Expected H200"; exit 2; fi
cd "$PROJ_DIR"
BUILD_DIR="benchmarks/experiment4_filter_aggregate/build"
mkdir -p "$BUILD_DIR"; cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j bench_filter_aggregate bench_graded_nmr
cd "$PROJ_DIR"
FA="./benchmarks/experiment4_filter_aggregate/build/bench_filter_aggregate"
NMR="./benchmarks/experiment4_filter_aggregate/build/bench_graded_nmr"
CESM_CLOUD="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg_global"
CESM_RAW="${WORK_DIR}/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"
HURRICANE_U="${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg_global"
HURRICANE_RAW="${WORK_DIR}/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"
SEG_ROWS=1024
HEADER="dataset,selectivity,k,protection_strategy,fault_case,fault_intensity_label,"
HEADER+="B0_ms,B1_ms,B2_ms,overhead_ms,margin_ms,overhead_fraction,"
HEADER+="B2_allowed,mismatch_count,certification_status,notes"
echo "$HEADER" > "$OUT_DIR/w5_pareto.csv"
run_cell() {
    local LABEL=$1 DS=$2 RAW=$3 THRESH=$4 PLANES=$5 PS=$6 FI_LABEL=$7 FC=$8
    local fa_csv="$OUT_DIR/_fa_${LABEL}.csv"
    $FA --dataset "$DS" --raw "$RAW" --threshold "$THRESH" \
        --segment-rows $SEG_ROWS --iters 50 --warmup 10 \
        --max-filter-planes 2 --csv "$fa_csv" > /dev/null 2>&1
    local B0=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['raw_baseline_ms_per_iter'])")
    local B1=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['ms_per_iter'])")
    local B2="" OVH="" MIS=""
    if [ "$PS" = "G0" ]; then
        B2=$B1; OVH="0"; MIS="0"
    elif [ "$PS" = "U2" ]; then
        # Sum per-plane duplicate-read for all active planes
        local TB2=$B1; local TMIS=0
        for ((p=0; p<PLANES; p++)); do
            local n_csv="$OUT_DIR/_u2_${LABEL}_p${p}.csv"
            $NMR --dataset "$DS" --raw "$RAW" --threshold "$THRESH" \
                --segment-rows $SEG_ROWS --plane-id $p \
                --b0-time "$B0" --b1-time "$B1" \
                --iters 30 --warmup 5 \
                --compare-mode segment_crc32 --fault $FC \
                --csv "$n_csv" > /dev/null 2>&1
            local PD=$(python3 -c "import csv; r=csv.DictReader(open('$n_csv')); print(next(r)['dup_ms'])")
            local PM=$(python3 -c "import csv; r=csv.DictReader(open('$n_csv')); print(next(r)['mismatches'])")
            TB2=$(python3 -c "print($TB2+$PD)")
            TMIS=$(python3 -c "print($TMIS+int($PM))")
        done
        B2=$TB2; OVH=$(python3 -c "print($TB2-$B1)"); MIS=$TMIS
    else
        # G1 or U1: single plane
        local PID=0; local RPID=1  # G1: S0 with clean ref at plane 1 (D3)
        if [ "$PS" = "U1" ]; then
            PID=$((PLANES/2))
            RPID=$PID  # U1: same-source (no diversity for Pareto sweep)
        fi
        local n_csv="$OUT_DIR/_${PS}_${LABEL}.csv"
        $NMR --dataset "$DS" --raw "$RAW" --threshold "$THRESH" \
            --segment-rows $SEG_ROWS --plane-id $PID --ref-plane-id $RPID \
            --b0-time "$B0" --b1-time "$B1" \
            --iters 30 --warmup 5 \
            --compare-mode segment_crc32 --fault $FC \
            --csv "$n_csv" > /dev/null 2>&1
        B2=$(python3 -c "import csv; r=csv.DictReader(open('$n_csv')); print(next(r)['b2_ms'])")
        OVH=$(python3 -c "import csv; r=csv.DictReader(open('$n_csv')); print(next(r)['overhead_ms'])")
        MIS=$(python3 -c "import csv; r=csv.DictReader(open('$n_csv')); print(next(r)['mismatches'])")
    fi
    local MARGIN=$(python3 -c "print(max(0,$B0-$B1))")
    local OF="NA" ALLOWED="NO"
    python3 -c "
m=$MARGIN; o=$OVH; b0=$B0; b2=$B2
if m>0:
    of=f'{o/m:.4f}'; al='YES' if(b2<b0 and o<0.9*m) else 'NO'
else:
    of='NA'; al='NO'
print(of); print(al)
" > "$OUT_DIR/_lu.txt"
    OF=$(head -1 "$OUT_DIR/_lu.txt")
    ALLOWED=$(tail -1 "$OUT_DIR/_lu.txt")
    python3 -c "
with open('$OUT_DIR/w5_pareto.csv','a') as f:
    f.write('$LABEL,50,2,$PS,$FC,$FI_LABEL,$B0,$B1,$B2,$OVH,$MARGIN,$OF,$ALLOWED,$MIS,CERTIFIED_BOUNDED,W5\n')
"
    echo "  $LABEL $PS $FI_LABEL: B2=$B2 allowed=$ALLOWED"
}
# Map fault intensity → NMR fault case
# For seg_global (1 segment), %-based segments collapse to single_seg
FAULT_MAP=(
    "0:no_fault"
    "1_byte:single_byte"
    "1_seg:single_seg"
    "10_seg:multi_seg"
    "0.001pct:single_seg"
    "0.01pct:single_seg"
    "0.1pct:single_seg"
)
for DS_INFO in "cesm_cloud,$CESM_CLOUD,$CESM_RAW,0.029,7" "hurricane_u,$HURRICANE_U,$HURRICANE_RAW,-2.31,8"; do
    IFS=',' read -r LABEL DS_PATH RAW_PATH THRESH PLANES <<< "$DS_INFO"
    # Prepare clean ref for D3 diversity: copy plane_000 to plane_001
    cp "$DS_PATH/plane_000.bin" "$DS_PATH/plane_001.bin"
    for entry in "${FAULT_MAP[@]}"; do
        IFS=':' read -r FI_LABEL FC <<< "$entry"
        for PS in G1 U1; do
            run_cell "$LABEL" "$DS_PATH" "$RAW_PATH" "$THRESH" "$PLANES" "$PS" "$FI_LABEL" "$FC"
        done
        # U2 at no_fault only (too expensive for all intensities)
        [ "$FI_LABEL" = "0" ] && run_cell "$LABEL" "$DS_PATH" "$RAW_PATH" "$THRESH" "$PLANES" "U2" "$FI_LABEL" "$FC"
    done
done
echo ""
echo "=== W5 Pareto Results ==="
rtk cat "$OUT_DIR/w5_pareto.csv"
