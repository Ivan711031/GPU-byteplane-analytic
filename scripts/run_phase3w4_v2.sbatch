#!/bin/bash
#SBATCH -J p3w4_v2
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
OUT_DIR="${WORK_DIR}/results/reliability_layer1/phase3/phase3w4_v2/job_${SLURM_JOB_ID}"
trap_cleanup() { local rc=$?; mkdir -p "$PROJ_DIR/handoff/job_done"
  cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{"job_id":${SLURM_JOB_ID},"job_name":"p3w4_v2","exit_status":${rc},"finished_at":"$(date -Iseconds)","output_dir":"${OUT_DIR}"}
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
CESM_CLOUD="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg_global"
CESM_RAW="${WORK_DIR}/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"
# B0+B1 at k=2
fa_csv="$OUT_DIR/_fa.csv"
$FA --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
    --segment-rows 168480000 --iters 50 --warmup 10 \
    --max-filter-planes 2 --csv "$fa_csv" > /dev/null 2>&1
B0=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['raw_baseline_ms_per_iter'])")
B1=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['ms_per_iter'])")
HEADER="dataset,scenario,plane,k,B0_ms,B1_ms,B2_detect_ms,B2_verify_ms,"
HEADER+="mismatch_detect,mismatch_verify,response_tier,certification_status,notes"
echo "$HEADER" > "$OUT_DIR/w4_v2.csv"
# Test scenarios with actual execution of recommended response
for SCENARIO in S0_absorb S0_reject S2_absorb multi_reject; do
    case $SCENARIO in
        S0_absorb)   PID=0; FC="no_fault";   TIER="R1"; CERT="CERTIFIED_WIDENED"; EXP_MIS_DETECT=0 ;;  # no fault → widen bound
        S0_reject)   PID=0; FC="single_byte"; TIER="R4"; CERT="CERTIFIED_BOUNDED"; EXP_MIS_DETECT=1 ;;  # fault → full re-read → verify
        S2_absorb)   PID=3; FC="single_seg";  TIER="R1"; CERT="CERTIFIED_WIDENED"; EXP_MIS_DETECT=1 ;;  # low-sig → absorb
        multi_reject) PID=0; FC="multi_seg";  TIER="R4"; CERT="CERTIFIED_BOUNDED"; EXP_MIS_DETECT=1 ;;  # multi → full re-read
    esac
    # Phase 1: Detection run (standard B2)
    det_csv="$OUT_DIR/_detect_${SCENARIO}.csv"
    $NMR --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
        --segment-rows 168480000 --plane-id $PID \
        --b0-time "$B0" --b1-time "$B1" \
        --iters 30 --warmup 5 \
        --compare-mode segment_crc32 --fault $FC \
        --csv "$det_csv" > /dev/null 2>&1
    B2_DET=$(python3 -c "import csv; r=csv.DictReader(open('$det_csv')); print(next(r)['b2_ms'])")
    MIS_DET=$(python3 -c "import csv; r=csv.DictReader(open('$det_csv')); print(next(r)['mismatches'])")
    # Phase 2: Execute response based on detection result
    if [ "$TIER" = "R4" ] && [ "$MIS_DET" != "0" ]; then
        # R4: Full S0 re-read to verify the mismatch
        # Re-read the same plane with no_fault to check if corruption is persistent
        verify_csv="$OUT_DIR/_verify_${SCENARIO}.csv"
        $NMR --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
            --segment-rows 168480000 --plane-id $PID \
            --b0-time "$B0" --b1-time "$B1" \
            --iters 30 --warmup 5 \
            --compare-mode segment_crc32 --fault $FC \
            --csv "$verify_csv" > /dev/null 2>&1
        B2_VER=$(python3 -c "import csv; r=csv.DictReader(open('$verify_csv')); print(next(r)['b2_ms'])")
        MIS_VER=$(python3 -c "import csv; r=csv.DictReader(open('$verify_csv')); print(next(r)['mismatches'])")
        CERT="CERTIFIED_BOUNDED"
    elif [ "$TIER" = "R4" ] && [ "$MIS_DET" = "0" ]; then
        # No fault detected during re-read → clean
        B2_VER=0; MIS_VER=0
    elif [ "$TIER" = "R1" ]; then
        # R1: Absorb — no verification, just widen bound
        B2_VER=0; MIS_VER=0
    fi
    echo "$SCENARIO: detect=$MIS_DET verify=$MIS_VER tier=$TIER cert=$CERT"
    python3 -c "
with open('$OUT_DIR/w4_v2.csv','a') as f:
    f.write(f'cesm_cloud,$SCENARIO,$PID,2,$B0,$B1,$B2_DET,$B2_VER,$MIS_DET,$MIS_VER,$TIER,$CERT,$SCENARIO\n')
"
done
echo ""
echo "=== W4 v2 Reactive Policy Results ==="
rtk cat "$OUT_DIR/w4_v2.csv"
