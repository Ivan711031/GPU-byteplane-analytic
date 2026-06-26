#!/bin/bash
#SBATCH -J p3w3_v2
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
OUT_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3w3_v2/job_${SLURM_JOB_ID}"
cleanup() { local rc=$?; mkdir -p "$PROJ_DIR/handoff/job_done"
  cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{"job_id":${SLURM_JOB_ID},"job_name":"p3w3_v2","exit_status":${rc},"finished_at":"$(date -Iseconds)","output_dir":"${OUT_DIR}"}
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
make -j bench_graded_nmr bench_filter_aggregate
cd "$PROJ_DIR"
NMR="./benchmarks/experiment4_filter_aggregate/build/bench_graded_nmr"
FA="./benchmarks/experiment4_filter_aggregate/build/bench_filter_aggregate"

HURRICANE_U="/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg_global"
HURRICANE_RAW="/work/u4063895/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"

fa_csv="$OUT_DIR/_fa.csv"
$FA --dataset "$HURRICANE_U" --raw "$HURRICANE_RAW" \
    --threshold -2.31 --segment-rows 25000000 \
    --iters 50 --warmup 10 --max-filter-planes 2 \
    --csv "$fa_csv" > /dev/null 2>&1
B0_MS=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['raw_baseline_ms_per_iter'])")
B1_MS=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['ms_per_iter'])")

HEADER="dataset,selectivity,k,diversity_strategy,plane_id,ref_plane_id,fault_scenario,B0_ms,B1_ms,B2_ms,overhead_ms,margin_ms,overhead_fraction,B2_allowed,mismatch_count,certification_status,notes"
echo "$HEADER" > "$OUT_DIR/w3_v2.csv"

# Backup original plane files
PLANE0="$HURRICANE_U/plane_000.bin"
PLANE1="$HURRICANE_U/plane_001.bin"
PLANE6="$HURRICANE_U/plane_006.bin"
for f in "$PLANE0" "$PLANE1" "$PLANE6"; do
    cp "$f" "$OUT_DIR/_$(basename $f).orig"
done

trap_cleanup() {
    for f in "$PLANE0" "$PLANE1" "$PLANE6"; do
        base=$(basename $f)
        [ -f "$OUT_DIR/_${base}.orig" ] && cp "$OUT_DIR/_${base}.orig" "$f"
    done
    rm -f "$OUT_DIR/_plane_*.orig"
}
trap trap_cleanup EXIT INT TERM

# Diversity strategies using --ref-plane-id
# D0: same source (plane=0, ref=0) — replicas logically adjacent
# D1: spatial separation (plane=0, ref=1) — adjacent plane for diversity
# D3: max separation (plane=0, ref=6) — maximally distant in logical order

# Fault scenarios:
# F1: single-target — corrupt ONE plane file
# F2: common-mode same-file — D0 both corrupted (same file), D1/D3 one corrupted
# F3: common-mode cross-file — corrupt TWO files, D0 only sees one, D1/D3 sees both

for DIVERSITY in "D0:0:0" "D1:0:1" "D3:0:6"; do
    IFS=':' read -r DSTRAT PID RPID <<< "$DIVERSITY"

    # F1: single-target fault → corrupt plane_000 only
    # Restore all files first
    for f in "$PLANE0" "$PLANE1" "$PLANE6"; do
        base=$(basename $f)
        cp "$OUT_DIR/_${base}.orig" "$f"
    done
    python3 -c "
import numpy as np
d = np.frombuffer(open('$PLANE0','rb').read(), dtype=np.uint8).copy()
d[0] ^= 0xFF
d.tofile('$PLANE0')
"
    echo "  $DSTRAT (p=$PID ref=$RPID): F1 single-target"
    nmr_csv="$OUT_DIR/_${DSTRAT}_F1.csv"
    $NMR --dataset "$HURRICANE_U" --raw "$HURRICANE_RAW" \
        --threshold -2.31 --segment-rows 25000000 \
        --plane-id $PID --ref-plane-id $RPID \
        --b0-time "$B0_MS" --b1-time "$B1_MS" \
        --iters 30 --warmup 5 \
        --compare-mode segment_crc32 --fault no_fault \
        --csv "$nmr_csv" > /dev/null 2>&1
    B2=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['b2_ms'])")
    OVH=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['overhead_ms'])")
    MIS=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['mismatches'])")
    MARGIN=$(python3 -c "print(max(0,$B0_MS-$B1_MS))")
    python3 scripts/assemble_w3_row.py "$OUT_DIR/w3_v2.csv" \
        hurricane_u 50 2 "$DSTRAT" $PID $RPID single_target \
        "$B0_MS" "$B1_MS" "$B2" "$OVH" "$MARGIN" "$MIS" F1_corrupt_plane0
    echo "    mismatch=$MIS"

    for FSCEN_TAG in "F2:common_mode_same" "F3:common_mode_cross"; do
        IFS=':' read -r FTAG FLABEL <<< "$FSCEN_TAG"
        for f in "$PLANE0" "$PLANE1" "$PLANE6"; do
            base=$(basename $f)
            cp "$OUT_DIR/_${base}.orig" "$f"
        done
        if [ "$FTAG" = "F2" ]; then
            python3 -c "import numpy as np; d=np.frombuffer(open('$PLANE0','rb').read(),dtype=np.uint8).copy(); d[0:1024]=0xFF; d.tofile('$PLANE0')"
        else
            python3 -c "import numpy as np; d=np.frombuffer(open('$PLANE0','rb').read(),dtype=np.uint8).copy(); d[0:512]=0xAA; d.tofile('$PLANE0'); d2=np.frombuffer(open('$PLANE1','rb').read(),dtype=np.uint8).copy(); d2[0:512]=0xAA; d2.tofile('$PLANE1')"
        fi
        echo "  $DSTRAT (p=$PID ref=$RPID): $FTAG"
        nmr_csv="$OUT_DIR/_${DSTRAT}_${FTAG}.csv"
        $NMR --dataset "$HURRICANE_U" --raw "$HURRICANE_RAW" \
            --threshold -2.31 --segment-rows 25000000 \
            --plane-id $PID --ref-plane-id $RPID \
            --b0-time "$B0_MS" --b1-time "$B1_MS" \
            --iters 30 --warmup 5 \
            --compare-mode segment_crc32 --fault no_fault \
            --csv "$nmr_csv" > /dev/null 2>&1
        B2=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['b2_ms'])")
        OVH=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['overhead_ms'])")
        MIS=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['mismatches'])")
        MARGIN=$(python3 -c "print(max(0,$B0_MS-$B1_MS))")
        python3 scripts/assemble_w3_row.py "$OUT_DIR/w3_v2.csv" \
            hurricane_u 50 2 "$DSTRAT" "$PID" "$RPID" "$FLABEL" \
            "$B0_MS" "$B1_MS" "$B2" "$OVH" "$MARGIN" "$MIS" "${FTAG}_cm"
        echo "    mismatch=$MIS"
    done
done

echo ""
echo "=== W3 v2 Diversity Results ==="
rtk cat "$OUT_DIR/w3_v2.csv"
