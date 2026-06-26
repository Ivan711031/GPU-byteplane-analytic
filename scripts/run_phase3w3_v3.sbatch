#!/bin/bash
#SBATCH -J p3w3_v3
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
OUT_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3w3_v3/job_${SLURM_JOB_ID}"
trap_cleanup() { local rc=$?
  for f in "$PLANE0" "$PLANE1" "$PLANE2"; do
    local base=$(basename $f)
    [ -f "$OUT_DIR/_${base}.orig" ] && cp "$OUT_DIR/_${base}.orig" "$f"
  done
  rm -f "$OUT_DIR/_plane_*.orig"
  mkdir -p "$PROJ_DIR/handoff/job_done"
  cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{"job_id":${SLURM_JOB_ID},"job_name":"p3w3_v3","exit_status":${rc},"finished_at":"$(date -Iseconds)","output_dir":"${OUT_DIR}"}
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

HURRICANE_U="/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg_global"
HURRICANE_RAW="/work/u4063895/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"
PLANE0="$HURRICANE_U/plane_000.bin"
PLANE1="$HURRICANE_U/plane_001.bin"
PLANE2="$HURRICANE_U/plane_002.bin"

fa_csv="$OUT_DIR/_fa.csv"
$FA --dataset "$HURRICANE_U" --raw "$HURRICANE_RAW" \
    --threshold -2.31 --segment-rows 25000000 \
    --iters 50 --warmup 10 --max-filter-planes 2 \
    --csv "$fa_csv" > /dev/null 2>&1
B0_MS=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['raw_baseline_ms_per_iter'])")
B1_MS=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['ms_per_iter'])")

HEADER="dataset,selectivity,k,diversity_strategy,plane_id,ref_plane_id,fault_scenario,B0_ms,B1_ms,B2_ms,overhead_ms,margin_ms,overhead_fraction,B2_allowed,mismatch_count,certification_status,notes,sanity"
echo "$HEADER" > "$OUT_DIR/w3_v3.csv"

# Step 1: Backup originals and create true S0 replicas
# plane_001 and plane_002 will be EXACT COPIES of plane_000 (S0 data)
for f in "$PLANE0" "$PLANE1" "$PLANE2"; do
    cp "$f" "$OUT_DIR/_$(basename $f).orig"
done

# Overwrite plane_001 and plane_002 with S0 data from plane_000
cp "$PLANE0" "$PLANE1"  # Now plane_001 = replica of S0
cp "$PLANE0" "$PLANE2"  # Now plane_002 = replica of S0
S0_CHECKSUM=$(sha256sum "$PLANE0" | cut -c1-16)
R1_CHECKSUM=$(sha256sum "$PLANE1" | cut -c1-16)
R2_CHECKSUM=$(sha256sum "$PLANE2" | cut -c1-16)
echo "S0 checksum: $S0_CHECKSUM" | tee "$OUT_DIR/checksum.log"
echo "Replica1:    $R1_CHECKSUM" | tee -a "$OUT_DIR/checksum.log"  
echo "Replica2:    $R2_CHECKSUM" | tee -a "$OUT_DIR/checksum.log"
if [ "$S0_CHECKSUM" != "$R1_CHECKSUM" ] || [ "$S0_CHECKSUM" != "$R2_CHECKSUM" ]; then
    echo "ERROR: Replica checksums do not match S0! Aborting."
    exit 2
fi
echo "All replicas are true copies of S0 data. Sanity: PASS" | tee -a "$OUT_DIR/checksum.log"

run_cell() {
    local DSTRAT=$1 PID=$2 RPID=$3 FSCEN=$4 LABEL=$5
    local nmr_csv="$OUT_DIR/_${DSTRAT}_${FSCEN}.csv"
    
    $NMR --dataset "$HURRICANE_U" --raw "$HURRICANE_RAW" \
        --threshold -2.31 --segment-rows 25000000 \
        --plane-id $PID --ref-plane-id $RPID \
        --b0-time "$B0_MS" --b1-time "$B1_MS" \
        --iters 30 --warmup 5 \
        --compare-mode segment_crc32 --fault no_fault \
        --csv "$nmr_csv" > /dev/null 2>&1

    local B2=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['b2_ms'])")
    local OVH=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['overhead_ms'])")
    local MIS=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['mismatches'])")
    local MARGIN=$(python3 -c "print(max(0,$B0_MS-$B1_MS))")

    python3 -c "
m=$MARGIN; o=$OVH; b0=$B0_MS; b2=$B2
of='NA'; al='NO'
if m>0: of=f'{o/m:.4f}'; al='YES' if(b2<b0 and o<0.9*m) else 'NO'
with open('$OUT_DIR/w3_v3.csv','a') as f:
    f.write(f'hurricane_u,50,2,$DSTRAT,$PID,$RPID,$FSCEN,$B0_MS,$B1_MS,$B2,$OVH,$MARGIN,{of},{al},$MIS,CERTIFIED_BOUNDED,$LABEL\n')
"
    echo "  $DSTRAT $LABEL: mismatch=$MIS"
}

# Step 2: Sanity — no_fault baseline. ALL strategies must produce 0 mismatches.
for entry in "D0:0:0" "D1:0:1" "D3:0:2"; do
    IFS=':' read -r dst pid rpid <<< "$entry"
    
    # Restore all plane files to clean S0 replicas before each test
    for f in "$PLANE0" "$PLANE1" "$PLANE2"; do
        base=$(basename $f)
        [ -f "$OUT_DIR/_${base}.orig" ] && cp "$OUT_DIR/_${base}.orig" "$f"
    done
    cp "$PLANE0" "$PLANE1"  # Ensure S0 replica
    cp "$PLANE0" "$PLANE2"
    
    echo "  --- $dst sanity (no_fault) ---"
    run_cell "$dst" "$pid" "$rpid" "no_fault" "sanity_no_fault"
done

# Step 3: F1 — single-target fault. Corrupt plane_000 only (the live replica).
for entry in "D0:0:0" "D1:0:1" "D3:0:2"; do
    IFS=':' read -r dst pid rpid <<< "$entry"
    
    for f in "$PLANE0" "$PLANE1" "$PLANE2"; do
        base=$(basename $f)
        [ -f "$OUT_DIR/_${base}.orig" ] && cp "$OUT_DIR/_${base}.orig" "$f"
    done
    cp "$PLANE0" "$PLANE1"
    cp "$PLANE0" "$PLANE2"
    
    # Corrupt plane_000 only
    python3 -c "
import numpy as np
d = np.frombuffer(open('$PLANE0','rb').read(), dtype=np.uint8).copy()
d[0] ^= 0xFF
d.tofile('$PLANE0')
"
    echo "  --- $dst F1 (single-target plane_000) ---"
    run_cell "$dst" "$pid" "$rpid" "F1" "F1_single_target"
done

# Step 4: F2 — common-mode same-file. Corrupt a segment in plane_000.
# D0: both replicas from same corrupted file → 0 mismatches
# D1/D3: live from corrupted file, ref from clean replica → 1 mismatch
for entry in "D0:0:0" "D1:0:1" "D3:0:2"; do
    IFS=':' read -r dst pid rpid <<< "$entry"
    
    for f in "$PLANE0" "$PLANE1" "$PLANE2"; do
        base=$(basename $f)
        [ -f "$OUT_DIR/_${base}.orig" ] && cp "$OUT_DIR/_${base}.orig" "$f"
    done
    cp "$PLANE0" "$PLANE1"
    cp "$PLANE0" "$PLANE2"
    
    # Corrupt plane_000 segment
    python3 -c "
import numpy as np
d = np.frombuffer(open('$PLANE0','rb').read(), dtype=np.uint8).copy()
d[0:1024] = 0xFF
d.tofile('$PLANE0')
"
    echo "  --- $dst F2 (common-mode same-file) ---"
    run_cell "$dst" "$pid" "$rpid" "F2" "F2_cm_samefile"
done

# Step 5: F3 — common-mode cross-file. Corrupt plane_000 AND plane_001.
# D0: both from plane_000 (corrupted) → 0 mismatches
# D1: live from plane_000 (corrupted), ref from plane_001 (ALSO corrupted) → 0 mismatches
# D3: live from plane_000 (corrupted), ref from plane_002 (clean) → 1 mismatch
for entry in "D0:0:0" "D1:0:1" "D3:0:2"; do
    IFS=':' read -r dst pid rpid <<< "$entry"
    
    for f in "$PLANE0" "$PLANE1" "$PLANE2"; do
        base=$(basename $f)
        [ -f "$OUT_DIR/_${base}.orig" ] && cp "$OUT_DIR/_${base}.orig" "$f"
    done
    cp "$PLANE0" "$PLANE1"
    cp "$PLANE0" "$PLANE2"
    
    # Corrupt both plane_000 AND plane_001 (the two adjacent replicas)
    python3 -c "
import numpy as np
for fn in ['$PLANE0','$PLANE1']:
    d = np.frombuffer(open(fn,'rb').read(), dtype=np.uint8).copy()
    d[0:512] = 0xAA
    d.tofile(fn)
"
    echo "  --- $dst F3 (common-mode cross-file) ---"
    run_cell "$dst" "$pid" "$rpid" "F3" "F3_cm_crossfile"
done

echo ""
echo "=== W3 v3 Diversity Results (true S0 replicas) ==="
rtk cat "$OUT_DIR/w3_v3.csv"
