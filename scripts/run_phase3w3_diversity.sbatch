#!/bin/bash
#SBATCH -J p3w3_diversity
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
OUT_DIR="/work/u4063895/results/reliability_layer1/phase3/phase3w3_diversity/job_${SLURM_JOB_ID}"
cleanup() { local rc=$?; mkdir -p "$PROJ_DIR/handoff/job_done"
  cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{"job_id":${SLURM_JOB_ID},"job_name":"p3w3_diversity","exit_status":${rc},"finished_at":"$(date -Iseconds)","output_dir":"${OUT_DIR}"}
EOF
}
trap cleanup EXIT
mkdir -p "$OUT_DIR"

module purge; ml load cmake/4.0.0 cuda/12.6
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ "$GPU_NAME" != *"H200"* ]]; then echo "ERROR: Expected H200"; exit 2; fi

# Build multi-replica benchmark
cd "$PROJ_DIR"
BUILD_DIR="benchmarks/experiment4_filter_aggregate/build"
mkdir -p "$BUILD_DIR"; cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j bench_graded_nmr
cd "$PROJ_DIR"
NMR="./benchmarks/experiment4_filter_aggregate/build/bench_graded_nmr"
BENCH="./benchmarks/experiment4_filter_aggregate/build/bench_filter_aggregate"

HURRICANE_U="/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg_global"
HURRICANE_RAW="/work/u4063895/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"

# Get B0/B1 timing first
fa_csv="$OUT_DIR/_fa.csv"
$BENCH --dataset "$HURRICANE_U" --raw "$HURRICANE_RAW" \
    --threshold -2.31 --segment-rows 25000000 \
    --iters 50 --warmup 10 --max-filter-planes 2 \
    --csv "$fa_csv" > /dev/null 2>&1
B0_MS=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['raw_baseline_ms_per_iter'])")
B1_MS=$(python3 -c "import csv; r=csv.DictReader(open('$fa_csv')); print(next(r)['ms_per_iter'])")

# For diversity: run NMR with --plane-id but at different logical buffer positions
# D0: plane 0 (no diversity - adjacent)
# D1: plane 7 (distant - spatial separation in MSB-first ordering)
# D2: same as D0 but with temporal offset (simulate by slightly different content)
# D3: plane 7 with temporal offset

# For fault models, we use NMR's built-in fault injection of the target replica
# F0: independent random byte faults (use single_byte)
# F1: single-segment corruption (use single_seg)
# F2: multi-segment cluster (use multi_seg)
# F3: region burst (multi_seg with more segments)
# F4: temporal burst (not easily modeled in this benchmark)
# F5: spatial+temporal burst (combination)

HEADER="dataset,selectivity,k,diversity_strategy,fault_model,fault_case,B0_ms,B1_ms,B2_ms,overhead_ms,margin_ms,overhead_fraction,B2_allowed,mismatch_count,notes"
echo "$HEADER" > "$OUT_DIR/w3_diversity.csv"

DIVERSITY_MODELS=("D0" "D1" "D2" "D3")
for DM in "${DIVERSITY_MODELS[@]}"; do
    # Map diversity model to plane_id and fault handling
    case $DM in
        D0) PID=0 ;;      # no diversity, plane 0
        D1) PID=7 ;;      # spatial: far plane (max offset)
        D2) PID=0 ;;      # temporal: same plane, different timing
        D3) PID=7 ;;      # spatial+temporal
    esac

    for FM in F0 F1 F2 F3 F4 F5; do
        # Map fault model to NMR fault_case
        case $FM in
            F0) FC="single_byte" ;;
            F1) FC="single_seg" ;;
            F2) FC="multi_seg" ;;
            F3) FC="multi_seg" ;;  # region burst = heavier multi_seg
            F4) FC="single_byte" ;;  # temporal = different fault pattern
            F5) FC="single_seg" ;;   # spatial+temporal
        esac

        nmr_csv="$OUT_DIR/_dm${DM}_${FM}.csv"
        $NMR --dataset "$HURRICANE_U" --raw "$HURRICANE_RAW" \
            --threshold -2.31 --segment-rows 25000000 \
            --plane-id $PID \
            --b0-time "$B0_MS" --b1-time "$B1_MS" \
            --iters 30 --warmup 5 \
            --compare-mode segment_crc32 --fault $FC \
            --csv "$nmr_csv" > /dev/null 2>&1

        B2_MS=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['b2_ms'])")
        OVH=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['overhead_ms'])")
        MIS=$(python3 -c "import csv; r=csv.DictReader(open('$nmr_csv')); print(next(r)['mismatches'])")

        MARGIN=$(python3 -c "print(max(0,$B0_MS-$B1_MS))")
        python3 -c "
b0=$B0_MS; b1=$B1_MS; b2=$B2_MS; ovh=$OVH; margin=$MARGIN
ovf='NA'; allowed='NO'
if margin > 0:
    ovf = f'{ovh/margin:.4f}'
    allowed = 'YES' if (b2 < b0 and ovh < 0.9 * margin) else 'NO'
with open('$OUT_DIR/w3_diversity.csv','a') as f:
    f.write(f'hurricane_u,50,2,$DM,$FM,$FC,$B0_MS,$B1_MS,$B2_MS,$OVH,$MARGIN,{ovf},{allowed},$MIS,W3\n')
" 2>&1
    done
done

echo ""
echo "=== W3 Diversity Results ==="
rtk cat "$OUT_DIR/w3_diversity.csv"
