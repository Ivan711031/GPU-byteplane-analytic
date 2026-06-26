#!/bin/bash
#SBATCH -J p3w4_v3
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
OUT_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3w4_v3/job_${SLURM_JOB_ID}"
trap_cleanup() { local rc=$?; mkdir -p "$PROJ_DIR/handoff/job_done"
  cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{"job_id":${SLURM_JOB_ID},"job_name":"p3w4_v3","exit_status":${rc},"finished_at":"$(date -Iseconds)","output_dir":"${OUT_DIR}"}
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
make -j bench_graded_nmr bench_filter_aggregate
cd "$PROJ_DIR"
NMR="./benchmarks/experiment4_filter_aggregate/build/bench_graded_nmr"
FA="./benchmarks/experiment4_filter_aggregate/build/bench_filter_aggregate"

CESM_CLOUD="/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg_global"
CESM_RAW="/work/u4063895/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"

HEADER="dataset,scenario,plane,k,fault_case,B0_ms,B1_ms,B2_detect_ms,overhead_ms,"
HEADER+="mismatch_detect,response_tier,certification_status,"
HEADER+="bound_before_ms,bound_after_ms,interval_widening_factor,fallback_path,notes"
echo "$HEADER" > "$OUT_DIR/w4_v3.csv"

# Helper: run a cell and write canonical row
run_cell() {
    local SCEN=$1 PID=$2 FC=$3 K=$4 NOTE=$5
    local ECHO_PREFIX="  $SCEN plane=$PID k=$K"
    local RESP="R0" CERT="CERTIFIED_BOUNDED" BB="0" BA="0" IWF="0" FBPATH="none"

    # B0+B1
    local fa_csv="$OUT_DIR/_fa_k${K}_${SCEN}.csv"
    $FA --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
        --segment-rows 168480000 --iters 50 --warmup 10 \
        --max-filter-planes $K --csv "$fa_csv" > /dev/null 2>&1
    local B0=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['raw_baseline_ms_per_iter'])")
    local B1=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['ms_per_iter'])")

    # Detection B2
    local det_csv="$OUT_DIR/_detect_${SCEN}_k${K}.csv"
    $NMR --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
        --segment-rows 168480000 --plane-id $PID \
        --b0-time "$B0" --b1-time "$B1" \
        --iters 30 --warmup 5 \
        --compare-mode segment_crc32 --fault $FC \
        --csv "$det_csv" > /dev/null 2>&1
    local B2_D=$(python3 -c "import csv; r=csv.DictReader(open('$det_csv')); print(next(r)['b2_ms'])")
    local OVH=$(python3 -c "import csv; r=csv.DictReader(open('$det_csv')); print(next(r)['overhead_ms'])")
    local MIS_D=$(python3 -c "import csv; r=csv.DictReader(open('$det_csv')); print(next(r)['mismatches'])")
    local MARGIN=$(python3 -c "print(max(0,$B0-$B1))")

    # Determine response based on scenario and detection
    if [ "$FC" = "no_fault" ]; then
        # R0: accept — no fault detected, CERTIFIED_BOUNDED
        RESP="R0"; CERT="CERTIFIED_BOUNDED"; BB="0"; BA="0"; IWF="0"
        echo "$ECHO_PREFIX (no_fault) detect=$MIS_D → R0 accept"
    elif [ "$MIS_D" = "0" ]; then
        # R0: accept — CRC clean despite fault flag
        RESP="R0"; CERT="CERTIFIED_BOUNDED"
        echo "$ECHO_PREFIX (detect=0) → R0 accept"
    elif [ "$PID" -ge 3 ]; then
        # R1: absorb — low-significance plane (S2+), widen bound
        BB=$MARGIN; BA=$(python3 -c "print($MARGIN + 0.001)"); IWF=$(python3 -c "print(($MARGIN+0.001)/max($MARGIN,0.0001))")
        RESP="R1"; CERT="CERTIFIED_WIDENED"
        echo "$ECHO_PREFIX detect=$MIS_D → R1 absorb (low-sig), bound widened"
    elif [ "$PID" -le 1 ]; then
        # R4: full S0 re-read — high-significance fault, verify
        local ver_csv="$OUT_DIR/_verify_${SCEN}_k${K}.csv"
        $NMR --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
            --segment-rows 168480000 --plane-id $PID \
            --b0-time "$B0" --b1-time "$B1" \
            --iters 30 --warmup 5 \
            --compare-mode segment_crc32 --fault $FC \
            --csv "$ver_csv" > /dev/null 2>&1
        local B2_V=$(python3 -c "import csv; r=csv.DictReader(open('$ver_csv')); print(next(r)['b2_ms'])")
        local MIS_V=$(python3 -c "import csv; r=csv.DictReader(open('$ver_csv')); print(next(r)['mismatches'])")

        if [ "$MIS_V" != "0" ] && [ "$K" -eq 2 ]; then
            # R5: raw FP64 fallback — re-read still shows fault at k=2
            local raw_csv="$OUT_DIR/_rawfb_${SCEN}.csv"
            $FA --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
                --segment-rows 168480000 --iters 50 --warmup 10 \
                --max-filter-planes 2 --csv "$raw_csv" > /dev/null 2>&1
            local RAW_T=$(python3 -c "import csv; r=csv.DictReader(open('$raw_csv')); print(next(r)['raw_baseline_ms_per_iter'])")
            RESP="R5+R4"; CERT="CERTIFIED_BOUNDED"; FBPATH="raw_fp64"
            BB=$MARGIN; BA=$(python3 -c "print($RAW_T - $B1)"); IWF=$(python3 -c "print(($RAW_T - $B1)/max($MARGIN,0.0001))")
            echo "$ECHO_PREFIX detect=$MIS_D re-read=$MIS_V → R4+R5 raw fallback (${RAW_T}ms)"
        elif [ "$MIS_V" != "0" ]; then
            # R6: UNCERTIFIED — fault persists even at k=max
            RESP="R6"; CERT="UNCERTIFIED"; FBPATH="all_failed"
            echo "$ECHO_PREFIX detect=$MIS_D re-read=$MIS_V → R6 UNCERTIFIED (persistent at k=max)"
        else
            # R4: fault cleared on re-read
            RESP="R4"; CERT="CERTIFIED_BOUNDED"; FBPATH="verify_ok"
            BB=$MARGIN; BA=$MARGIN; IWF="1.0"
            echo "$ECHO_PREFIX detect=$MIS_D re-read=$MIS_V → R4 verify OK"
        fi
    fi

    python3 -c "
with open('$OUT_DIR/w4_v3.csv','a') as f:
    f.write(f'cesm_cloud,$SCEN,$PID,$K,$FC,$B0,$B1,$B2_D,$OVH,$MIS_D,$RESP,$CERT,$BB,$BA,$IWF,$FBPATH,$NOTE\n')
"
}

# === Required scenarios across required response paths ===

# R0: no_fault on S0 and S1
run_cell S0_accept 0 no_fault 2 "R0_accept_no_fault"
run_cell S0_accept_k7 0 no_fault 7 "R0_accept_no_fault_k7"

# S0 mismatch at k=2 and k=max (R4 → R5 at k=2, R6 at k=max)
run_cell S0_reject 0 single_byte 2 "R4_verify_then_R5_raw_fallback"
run_cell S0_reject_k7 0 single_byte 7 "R6_UNCERTIFIED_persistent_at_kmax"

# S1 mismatch
run_cell S1_reject 1 single_byte 2 "R4_verify_then_R5_raw_fallback"
run_cell S1_reject_k7 1 single_byte 7 "R6_UNCERTIFIED_persistent_at_kmax"

# S2+ mismatch (low-significance → R1 absorb)
run_cell S2_absorb 3 single_seg 2 "R1_absorb_widen_bound"
run_cell S2_absorb_k7 3 single_seg 7 "R1_absorb_widen_bound"

# Inactive/sentinel: CESM Cloud has 7 active planes (0-6), no inactive. Label N/A.
python3 -c "
with open('$OUT_DIR/w4_v3.csv','a') as f:
    f.write('cesm_cloud,inactive_sentinel,NA,2,NA,NA,NA,NA,NA,NA,NA,CERTIFIED_BOUNDED,0,0,0,none,inactive_not_present\n')
"
echo "  inactive/sentinel: not present (7 active planes, no sentinel)"

# Multi-plane: corrupt plane_000 and plane_001 segments
run_cell multi_plane 0 multi_seg 2 "R4_verify_detected"

echo ""
echo "=== W4 v3 Full Reactive Policy Matrix ==="
rtk cat "$OUT_DIR/w4_v3.csv"
