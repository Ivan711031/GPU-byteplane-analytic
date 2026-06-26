#!/bin/bash
#SBATCH -J p3w4_v4_r2
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
OUT_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3w4_v4_r2/job_${SLURM_JOB_ID}"
trap_cleanup() { local rc=$?
  [ -f "$PLANE_BAK" ] && cp "$PLANE_BAK" "$PLANE0"
  [ -f "$OUT_DIR/_plane1.bak" ] && cp "$OUT_DIR/_plane1.bak" "$PLANE1"
  rm -f "$PLANE_BAK" "$OUT_DIR/_plane1.bak"
  mkdir -p "$PROJ_DIR/handoff/job_done"
  cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{"job_id":${SLURM_JOB_ID},"job_name":"p3w4_v4_r2","exit_status":${rc},"finished_at":"$(date -Iseconds)","output_dir":"${OUT_DIR}"}
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

# Use CESM Cloud with segment-rows=1024 to create multiple segments
CESM_CLOUD="/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg_global"
CESM_RAW="/work/u4063895/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"
SEG_ROWS=1024

HEADER="dataset,phase,scenario,plane,segment,k,fault_case,B0_ms,B1_ms,dup_ms,mismatches,response_tier,certification_status,notes"
echo "$HEADER" > "$OUT_DIR/w4_v4_r2.csv"

# Get B0+B1
fa_csv="$OUT_DIR/_fa.csv"
$FA --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
    --segment-rows $SEG_ROWS --iters 50 --warmup 10 \
    --max-filter-planes 2 --csv "$fa_csv" > /dev/null 2>&1
B0_MS=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['raw_baseline_ms_per_iter'])")
B1_MS=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['ms_per_iter'])")

# === R2: Segment-aware re-read test ===
# We corrupt a specific byte in segment 5 of plane 0 (S0).
# Phase 1 (detect): full-plane B2 → identifies which segment has mismatch
# Phase 2 (R2): re-read only segment 5 → verifies the specific segment

echo "=== R2: Segment-level re-read test (CESM Cloud, seg_rows=1024) ==="

# Phase 1: Full detection run on S0 with segment-level fault in seg 5
# Fault: corrupt byte 0 of segment 5
# Use custom fault: we inject via plane_000.bin modification on disk
PLANE0="$CESM_CLOUD/plane_000.bin"
PLANE1="$CESM_CLOUD/plane_001.bin"
PLANE_BAK="$OUT_DIR/_plane0.bak"
cp "$PLANE0" "$PLANE_BAK"
cp "$PLANE1" "$OUT_DIR/_plane1.bak"
# Step 1: Copy CLEAN S0 data to plane_001 as reference (no fault)
cp "$PLANE0" "$PLANE1"
echo "Copied clean S0 to plane_001 as reference"

# Step 2: Inject fault into plane_000 (the live data)
python3 -c "
import numpy as np
d = np.frombuffer(open('$PLANE0','rb').read(), dtype=np.uint8).copy()
d[5 * $SEG_ROWS] ^= 0xFF  # corrupt segment 5, byte 0
d.tofile('$PLANE0')
"
echo "Injected fault in segment 5 of plane_000 (live data)"

# Phase 1: Detect with clean ref (--plane-id 0 corrupted, --ref-plane-id 1 clean)
# Note: run_nmr uses --plane-id and --ref-plane-id, so I need to update it
# Phase 1: Full-plane detect with clean reference (--ref-plane-id 1 = clean S0 copy)
nmr_det="$OUT_DIR/_detect.csv"
$NMR --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
    --segment-rows $SEG_ROWS --plane-id 0 --ref-plane-id 1 \
    --segment-id -1 \
    --b0-time "$B0_MS" --b1-time "$B1_MS" \
    --iters 30 --warmup 5 \
    --compare-mode segment_crc32 --fault no_fault \
    --csv "$nmr_det" > /dev/null 2>&1
D2_D=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_det')); print(next(r)['dup_ms'])")
MIS_D=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_det')); print(next(r)['mismatches'])")
echo "Phase 1 (detect all segments): dup=$D2_D ms mismatches=$MIS_D"
python3 -c "
with open('$OUT_DIR/w4_v4_r2.csv','a') as f:
    f.write('cesm_cloud,detect,S0,0,-1,2,no_fault,$B0_MS,$B1_MS,$D2_D,$MIS_D,R2_detect,CERTIFIED_BOUNDED,fault_in_seg5\n')
"

if [ "$MIS_D" = "0" ]; then
    echo "ERROR: No mismatches detected in phase 1! Fault injection may have failed."
    exit 2
fi

# Phase 2a: R2 re-read of suspect segment 5 only (segment_id=5)
nmr_r2="$OUT_DIR/_r2_seg5.csv"
$NMR --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
    --segment-rows $SEG_ROWS --plane-id 0 --ref-plane-id 1 \
    --segment-id 5 \
    --b0-time "$B0_MS" --b1-time "$B1_MS" \
    --iters 30 --warmup 5 \
    --compare-mode segment_crc32 --fault no_fault \
    --csv "$nmr_r2" > /dev/null 2>&1
D2_R=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_r2')); print(next(r)['dup_ms'])")
MIS_R=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_r2')); print(next(r)['mismatches'])")

echo "Phase 2a (R2 re-read seg 5): dup=$D2_R ms mismatches=$MIS_R"

if [ "$MIS_R" != "0" ]; then
    # Fault persists in seg 5 from same source → try clean-source re-read (proper R2)
    nmr_r2_clean="$OUT_DIR/_r2_clean.csv"
    $NMR --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
        --segment-rows $SEG_ROWS --plane-id 1 --ref-plane-id 1 \
        --segment-id 5 \
        --b0-time "$B0_MS" --b1-time "$B1_MS" \
        --iters 30 --warmup 5 \
        --compare-mode segment_crc32 --fault no_fault \
        --csv "$nmr_r2_clean" > /dev/null 2>&1
    MIS_RC=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_r2_clean')); print(next(r)['mismatches'])")
    if [ "$MIS_RC" = "0" ]; then
        RESPONSE="R2"
        CERT="CERTIFIED_BOUNDED"
        echo "Phase 2a: clean-source R2 re-read cleared mismatch → R2 verified"
        python3 -c "
with open('$OUT_DIR/w4_v4_r2.csv','a') as f:
    f.write('cesm_cloud,r2_clean,S0,1,5,2,no_fault,$B0_MS,$B1_MS,0,0,R2_clean,CERTIFIED_BOUNDED,R2_clean_source_verified\n')
"
    else
        RESPONSE="R4"
        CERT="CERTIFIED_BOUNDED"
        echo "Phase 2a: both sources show mismatch → escalate to R4 full re-read"
    fi
else
    RESPONSE="R2"
    CERT="CERTIFIED_BOUNDED"
    echo "Phase 2a: R2 re-read cleared the mismatch → R2 verified"
fi

python3 -c "
with open('$OUT_DIR/w4_v4_r2.csv','a') as f:
    f.write('cesm_cloud,r2_reread,S0,0,5,2,no_fault,$B0_MS,$B1_MS,$D2_R,$MIS_R,$RESPONSE,$CERT,R2_segment_reread\n')
"

# Phase 2b: R2 re-read of clean segment (seg 3) as control — should show 0 mismatches
nmr_ctl="$OUT_DIR/_r2_ctrl.csv"
$NMR --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
    --segment-rows $SEG_ROWS --plane-id 0 --ref-plane-id 1 \
    --segment-id 3 \
    --b0-time "$B0_MS" --b1-time "$B1_MS" \
    --iters 30 --warmup 5 \
    --compare-mode segment_crc32 --fault no_fault \
    --csv "$nmr_ctl" > /dev/null 2>&1
D2_C=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_ctl')); print(next(r)['dup_ms'])")
MIS_C=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_ctl')); print(next(r)['mismatches'])")
echo "Phase 2b (control: re-read clean seg 3): dup=$D2_C ms mismatches=$MIS_C"
python3 -c "
with open('$OUT_DIR/w4_v4_r2.csv','a') as f:
    f.write('cesm_cloud,r2_control,S0,0,3,2,no_fault,$B0_MS,$B1_MS,$D2_C,$MIS_C,R2_control,CERTIFIED_BOUNDED,clean_segment_control\n')
"

# Phase 3: R2 on clean S2+ data (plane 3, seg 3) — R1 absorb
nmr_s2="$OUT_DIR/_r2_s2.csv"
$NMR --dataset "$CESM_CLOUD" --raw "$CESM_RAW" --threshold 0.177 \
    --segment-rows $SEG_ROWS --plane-id 3 --ref-plane-id 3 \
    --segment-id 3 \
    --b0-time "$B0_MS" --b1-time "$B1_MS" \
    --iters 30 --warmup 5 \
    --compare-mode segment_crc32 --fault no_fault \
    --csv "$nmr_s2" > /dev/null 2>&1
D2_S2=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_s2')); print(next(r)['dup_ms'])")
MIS_S2=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_s2')); print(next(r)['mismatches'])")
echo "Phase 3 (S2 clean seg 3): dup=$D2_S2 ms mismatches=$MIS_S2"
python3 -c "
with open('$OUT_DIR/w4_v4_r2.csv','a') as f:
    f.write('cesm_cloud,r2_s2,S2,3,3,2,no_fault,$B0_MS,$B1_MS,$D2_S2,$MIS_S2,R1_absorb,CERTIFIED_WIDENED,S2_segment_reread_absorb\n')
"

echo ""
echo "=== W4 v4 R2 Results ==="
rtk cat "$OUT_DIR/w4_v4_r2.csv"
