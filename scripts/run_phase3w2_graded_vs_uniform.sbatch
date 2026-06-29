#!/bin/bash
#SBATCH -J p3w2_graded_uniform
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
OUT_DIR="${WORK_DIR}/results/reliability_layer1/phase3/phase3w2_graded_vs_uniform/job_${SLURM_JOB_ID}"
CANONICAL_CSV="$OUT_DIR/w2_canonical.csv"
cleanup() { local rc=$?; mkdir -p "$PROJ_DIR/handoff/job_done"
  cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{"job_id":${SLURM_JOB_ID},"job_name":"p3w2_graded_uniform","exit_status":${rc},"finished_at":"$(date -Iseconds)","output_dir":"${OUT_DIR}","next_action":"Write W2 graded vs uniform report."}
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
CESM_CLOUD="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg_global"
CESM_RAW="${WORK_DIR}/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"
HURRICANE_U="${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg_global"
HURRICANE_RAW="${WORK_DIR}/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"
HEADER="dataset,selectivity,k,protection_strategy,protected_planes,extra_read_bytes,read_amplification,B0_ms,B1_ms,B2_ms,overhead_ms,margin_ms,overhead_fraction,B2_allowed,mismatch_count,certification_status,notes"
echo "$HEADER" > "$CANONICAL_CSV"
measure_protection() {
    local label=$1 DS=$2 RAW=$3 THRESH=$4 SEG_ROWS=$5 SEL=$6 PLANES=$7 PS=$8
    local PP_COUNT=$9 EXTRA_BYTES=${10}
    # B0+B1 via FA
    local fa_csv="$OUT_DIR/_fa_${label}_s${SEL}.csv"
    $FA --dataset "$DS" --raw "$RAW" --threshold "$THRESH" \
        --segment-rows "$SEG_ROWS" --iters 50 --warmup 10 \
        --max-filter-planes 2 --csv "$fa_csv" \
        2>&1 | rtk grep -E "raw_baseline:|result:|speedup"
    local B0_MS=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['raw_baseline_ms_per_iter'])")
    local B1_MS=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['ms_per_iter'])")
    local B2_MS="0.0"
    local MISMATCH_SUM="0"
    if [ "$PS" = "G0" ]; then
        # No NMR: B2 = B1 (no extra read)
        B2_MS=$B1_MS
        MISMATCH_SUM="0"
    elif [ "$PS" = "G1" ]; then
        # Graded S0-only: read plane 0
        local nmr_csv="$OUT_DIR/_nmr_${label}_s${SEL}_g1.csv"
        $NMR --dataset "$DS" --raw "$RAW" --threshold "$THRESH" \
            --segment-rows "$SEG_ROWS" --plane-id 0 \
            --b0-time "$B0_MS" --b1-time "$B1_MS" \
            --iters 30 --warmup 5 \
            --compare-mode segment_crc32 --fault no_fault \
            --csv "$nmr_csv" 2>&1 | rtk grep -E "B2 dup|Mismatch"
        B2_MS=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['b2_ms'])")
        MISMATCH_SUM=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['mismatches'])")
    elif [ "$PS" = "U1" ]; then
        # Uniform same-budget: read a mid-plane (plane 3 or 4)
        local MID_PLANE=3
        [ "$PLANES" -le 4 ] && MID_PLANE=2
        local nmr_csv="$OUT_DIR/_nmr_${label}_s${SEL}_u1.csv"
        $NMR --dataset "$DS" --raw "$RAW" --threshold "$THRESH" \
            --segment-rows "$SEG_ROWS" --plane-id $MID_PLANE \
            --b0-time "$B0_MS" --b1-time "$B1_MS" \
            --iters 30 --warmup 5 \
            --compare-mode segment_crc32 --fault no_fault \
            --csv "$nmr_csv" 2>&1 | rtk grep -E "B2 dup|Mismatch"
        B2_MS=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['b2_ms'])")
        MISMATCH_SUM=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['mismatches'])")
    elif [ "$PS" = "U2" ]; then
        # Uniform full: sum duplicate-read cost for all active planes
        local TOTAL_B2=$B1_MS
        local TOTAL_MISMATCH=0
        for ((p = 0; p < PLANES; p++)); do
            local nmr_csv="$OUT_DIR/_nmr_${label}_s${SEL}_u2_p${p}.csv"
            $NMR --dataset "$DS" --raw "$RAW" --threshold "$THRESH" \
                --segment-rows "$SEG_ROWS" --plane-id $p \
                --b0-time "$B0_MS" --b1-time "$B1_MS" \
                --iters 30 --warmup 5 \
                --compare-mode segment_crc32 --fault no_fault \
                --csv "$nmr_csv" > /dev/null 2>&1
            local P_DUP=$(python3 -c "
import csv
r = csv.DictReader(open('$nmr_csv'))
for row in r:
    print(row['dup_ms'])
    break
")
            local P_MIS=$(python3 -c "
import csv
r = csv.DictReader(open('$nmr_csv'))
for row in r:
    print(row['mismatches'])
    break
")
            TOTAL_B2=$(python3 -c "print($TOTAL_B2 + $P_DUP)")
            TOTAL_MISMATCH=$(python3 -c "print($TOTAL_MISMATCH + int($P_MIS))")
            echo "  plane $p: dup=${P_DUP}ms mismatch=${P_MIS}"
        done
        B2_MS=$TOTAL_B2
        MISMATCH_SUM=$TOTAL_MISMATCH
    fi
    local MARGIN=$(python3 -c "print(max(0, $B0_MS - $B1_MS))")
    local OVH=$(python3 -c "print($B2_MS - $B1_MS)")
    local OVH_FRAC="NA"
    local B2_ALLOWED="NO"
    python3 -c "
b0=$B0_MS; b1=$B1_MS; b2=$B2_MS; ovh=$OVH; margin=$MARGIN
if margin > 0:
    of = ovh / margin
    allowed = 'YES' if (b2 < b0 and ovh < 0.9 * margin) else 'NO'
    print(f'{of:.4f},{allowed}')
else:
    print('NA,NO')
" > "$OUT_DIR/_lu.txt"
    IFS=',' read -r OVH_FRAC B2_ALLOWED < "$OUT_DIR/_lu.txt"
    local RA=$(python3 -c "print(($EXTRA_BYTES + 2) / 8.0)")
    python3 -c "
import csv
row = {'dataset':'$label','selectivity':$SEL,'k':2,'protection_strategy':'$PS',
       'protected_planes':$PP_COUNT,'extra_read_bytes':$EXTRA_BYTES,'read_amplification':$RA,
       'B0_ms':$B0_MS,'B1_ms':$B1_MS,'B2_ms':$B2_MS,'overhead_ms':$OVH,
       'margin_ms':$MARGIN,'overhead_fraction':'$OVH_FRAC',
       'B2_allowed':'$B2_ALLOWED','mismatch_count':$MISMATCH_SUM,
       'certification_status':'CERTIFIED_BOUNDED','notes':'primary'}
with open('$CANONICAL_CSV','a',newline='') as f:
    w = csv.DictWriter(f,fieldnames=list(row.keys()))
    w.writerow(row)
"
    echo "  $label $PS: B2=${B2_MS}ms allowed=$B2_ALLOWED"
}
# Each dataset: run G0, G1, U1, U2
# CESM Cloud: 7 planes, sel=50%
measure_protection cesm_cloud "$CESM_CLOUD" "$CESM_RAW" 0.029 168480000 50 7 G0 0 0
measure_protection cesm_cloud "$CESM_CLOUD" "$CESM_RAW" 0.029 168480000 50 7 G1 1 168480000
measure_protection cesm_cloud "$CESM_CLOUD" "$CESM_RAW" 0.029 168480000 50 7 U1 1 168480000
measure_protection cesm_cloud "$CESM_CLOUD" "$CESM_RAW" 0.029 168480000 50 7 U2 7 1179360000
# Hurricane U: 8 planes, sel=50%
measure_protection hurricane_u "$HURRICANE_U" "$HURRICANE_RAW" -2.31 25000000 50 8 G0 0 0
measure_protection hurricane_u "$HURRICANE_U" "$HURRICANE_RAW" -2.31 25000000 50 8 G1 1 25000000
measure_protection hurricane_u "$HURRICANE_U" "$HURRICANE_RAW" -2.31 25000000 50 8 U1 1 25000000
measure_protection hurricane_u "$HURRICANE_U" "$HURRICANE_RAW" -2.31 25000000 50 8 U2 8 200000000
echo ""
echo "=== Canonical CSV ==="
rtk cat "$CANONICAL_CSV"
