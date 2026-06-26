#!/bin/bash
#SBATCH -J p3w4_reactive
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
OUT_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3w4_reactive/job_${SLURM_JOB_ID}"
cleanup() { local rc=$?; mkdir -p "$PROJ_DIR/handoff/job_done"
  cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{"job_id":${SLURM_JOB_ID},"job_name":"p3w4_reactive","exit_status":${rc},"finished_at":"$(date -Iseconds)","output_dir":"${OUT_DIR}"}
EOF
}
trap cleanup EXIT
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

CESM_CLOUD="/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg_global"
CESM_RAW="/work/u4063895/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"

# B0+B1 at k=2
fa_csv="$OUT_DIR/_fa_k2.csv"
$FA --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
    --segment-rows 168480000 --iters 50 --warmup 10 \
    --max-filter-planes 2 --csv "$fa_csv" > /dev/null 2>&1
B0_K2=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['raw_baseline_ms_per_iter'])")
B1_K2=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['ms_per_iter'])")

# B0+B1 at k=7 (max)
fa_csv7="$OUT_DIR/_fa_k7.csv"
$FA --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
    --segment-rows 168480000 --iters 50 --warmup 10 \
    --max-filter-planes 7 --csv "$fa_csv7" > /dev/null 2>&1
B0_K7=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv7')); print(next(r)['raw_baseline_ms_per_iter'])")
B1_K7=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv7')); print(next(r)['ms_per_iter'])")

echo "k=2: B0=$B0_K2 B1=$B1_K2"
echo "k=7: B0=$B0_K7 B1=$B1_K7"

# W4: test reactive responses across plane ranks with fault injection
# Scenarios: S0 (plane 0), S1 (plane 1), S2+ (plane 3), inactive (plane 7), multi-plane
HEADER="dataset,scenario,plane,k,fault_case,B0_ms,B1_ms,B2_ms,overhead_ms,margin_ms,overhead_fraction,B2_allowed,mismatch_count,response_tier,certification_status,notes"
echo "$HEADER" > "$OUT_DIR/w4_reactive.csv"

for K in 2 7; do
    if [ "$K" = "2" ]; then B0=$B0_K2; B1=$B1_K2; else B0=$B0_K7; B1=$B1_K7; fi

    for SCENARIO in S0 S1 S2 multi; do
        case $SCENARIO in
            S0) PID=0; FC="single_byte"; TIER="R2" ;;
            S1) PID=1; FC="single_byte"; TIER="R2" ;;
            S2) PID=3; FC="single_seg"; TIER="R1" ;;  # S2+ low-sig absorb
            multi) PID=0; FC="multi_seg"; TIER="R4" ;;
        esac

        # B2 measurement at this plane with fault
        nmr_csv="$OUT_DIR/_k${K}_${SCENARIO}.csv"
        $NMR --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
            --segment-rows 168480000 --plane-id $PID \
            --b0-time "$B0" --b1-time "$B1" \
            --iters 30 --warmup 5 \
            --compare-mode segment_crc32 --fault $FC \
            --csv "$nmr_csv" > /dev/null 2>&1

        B2=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['b2_ms'])")
        OVH=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['overhead_ms'])")
        MIS=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['mismatches'])")

        MARGIN=$(python3 -c "print(max(0,$B0-$B1))")
        if [ "$MIS" -gt 0 ] && [ "$SCENARIO" = "S2" ]; then
            CERT="CERTIFIED_WIDENED"
            RESPONSE="R1"
            NOTES="low_sig_absorbed"
        elif [ "$MIS" -gt 0 ]; then
            CERT="CERTIFIED_BOUNDED"
            RESPONSE=$TIER
            NOTES="detected_escalated"
        else
            CERT="CERTIFIED_BOUNDED"
            RESPONSE="R0"
            NOTES="no_fault"
        fi

        OVHF="NA"; ALLOWED="NO"
        python3 -c "
m=$MARGIN; o=$OVH; b0=$B0; b2=$B2
if m>0:
    of = f'{o/m:.4f}'
    al = 'YES' if (b2<b0 and o<0.9*m) else 'NO'
    print(f'{of},{al}')
else:
    print('NA,NO')
" > "$OUT_DIR/_lu.txt"
        IFS=',' read -r OVHF ALLOWED < "$OUT_DIR/_lu.txt"

        python3 -c "
with open('$OUT_DIR/w4_reactive.csv','a') as f:
    f.write(f'cesm_cloud,$SCENARIO,$PID,$K,$FC,$B0,$B1,$B2,$OVH,$MARGIN,$OVHF,$ALLOWED,$MIS,$RESPONSE,$CERT,$NOTES\n')
" 2>&1
    done
done

echo ""
echo "=== W4 Reactive Degradation Results ==="
rtk cat "$OUT_DIR/w4_reactive.csv"
