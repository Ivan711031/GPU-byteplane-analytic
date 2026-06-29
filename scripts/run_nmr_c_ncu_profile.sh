#!/bin/bash
#SBATCH -J nmr_c_ncu
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/slurm-%j.out
set -euo pipefail
echo "=== NMR-C: NCU Free Redundancy Profiling ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME"
if ! echo "$GPU_NAME" | grep -qi "H200"; then echo "FATAL: Expected H200"; exit 2; fi
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
ml load cmake/4.0.0
conda activate gpu-byteplane-scan
PROJECT_DIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr"
SCRIPT_DIR="$PROJECT_DIR/scripts"
BUILD_DIR="${WORK_DIR}/builds/nmr_c_${SLURM_JOB_ID}"
RESULTS_DIR="$PROJECT_DIR/results/reliability_layer1/phase4/nmr_c_ncu_free_redundancy/job_${SLURM_JOB_ID}"
RAW_NCU_DIR="$RESULTS_DIR/nmr_c_raw_ncu_reports"
mkdir -p "$BUILD_DIR" "$RESULTS_DIR" "$RAW_NCU_DIR"
HURR_DIR="${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg4096"
CESM_DIR="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"
HURR_RAW="${WORK_DIR}/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"
CESM_RAW="${WORK_DIR}/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"
# Verify raw files exist
for f in "$HURR_RAW" "$CESM_RAW"; do
    if [ ! -f "$f" ]; then echo "FATAL: raw file not found: $f"; exit 2; fi
done
# Build
cd "$BUILD_DIR"
cmake "$PROJECT_DIR/benchmarks/experiment4_filter_aggregate" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc) bench_nmr_c_ncu_profile
BENCH="$BUILD_DIR/bench_nmr_c_ncu_profile"
CSV="$RESULTS_DIR/nmr_c_ncu_profile.csv"
: > "$CSV"
# Paths to profile
PATHS=("P0" "P1" "P2" "P3" "P4")
ITERS=100
run_path() {
    local DS_PATH=$1 DS_RAW=$2 DS_LABEL=$3 PATH_ID=$4 NCU_FLAG=$5
    local PATH_LABEL
    case $PATH_ID in
        P0) PATH_LABEL="P0_baseline_byteplane_k2" ;;
        P1) PATH_LABEL="P1_digest_only" ;;
        P2) PATH_LABEL="P2_vote_read_compare" ;;
        P3) PATH_LABEL="P3_digest_plus_vote" ;;
        P4) PATH_LABEL="P4_raw_fused_fp64_reference" ;;
    esac
    local NCU_REPORT="$RAW_NCU_DIR/${DS_LABEL}_${PATH_LABEL}.ncu-rep"
    echo ""
    echo "=== ${DS_LABEL} ${PATH_LABEL} ==="
    if [ "$NCU_FLAG" = "ncu" ]; then
        # Write NCU timing to a separate CSV (not the main latency CSV) to avoid pollution
        local NCU_TIMING_CSV="$RAW_NCU_DIR/${DS_LABEL}_${PATH_LABEL}_ncu_timing.csv"
        echo "  NCU profiling..."
        ncu --set full \
            --target-processes all \
            --csv \
            -o "$NCU_REPORT" \
            "$BENCH" --dataset "$DS_PATH" --raw "$DS_RAW" --path "$PATH_ID" --iters 10 --csv "$NCU_TIMING_CSV" 2>&1 | \
            tee "$RAW_NCU_DIR/${DS_LABEL}_${PATH_LABEL}_ncu_stdout.txt" || true
    else
        echo "  Latency only (no NCU)..."
        "$BENCH" --dataset "$DS_PATH" --raw "$DS_RAW" --path "$PATH_ID" --iters "$ITERS" --csv "$CSV" 2>&1 | \
            tee -a "$RAW_NCU_DIR/${DS_LABEL}_${PATH_LABEL}_latency.txt" || true
    fi
}
# Phase 1: Latency-only runs for all paths (faster, produces main CSV)
DS_PATHS=("$HURR_DIR" "$CESM_DIR")
DS_RAWS=("$HURR_RAW" "$CESM_RAW")
DS_LABELS=(hurricane_u cesm_atm_cloud)
for i in "${!DS_PATHS[@]}"; do
    dp="${DS_PATHS[$i]}"
    dr="${DS_RAWS[$i]}"
    dl="${DS_LABELS[$i]}"
    for pid in "${PATHS[@]}"; do
        run_path "$dp" "$dr" "$dl" "$pid" "no_ncu"
    done
done
# Phase 2: NCU profiling for hurricane_u (smaller) first, then cesm if smoke passes
echo ""
echo "=== Phase 2: NCU profiling (hurricane_u first) ==="
# Smoke NCU: just P0 on hurricane_u to validate profiler capture
echo "  NCU smoke: hurricane_u P0"
ncu --set full --target-processes all --csv \
    -o "$RAW_NCU_DIR/hurricane_u_P0_smoke" \
    "$BENCH" --dataset "$HURR_DIR" --raw "$HURR_RAW" --path "P0" --iters 10 --csv "$CSV" 2>&1 | \
    tee "$RAW_NCU_DIR/hurricane_u_P0_ncu_smoke.txt"
# If smoke passes, profile all paths on hurricane_u
echo ""
echo "  NCU full: hurricane_u all paths"
for pid in "${PATHS[@]}"; do
    run_path "$HURR_DIR" "$HURR_RAW" hurricane_u "$pid" "ncu"
done
# Then cesm
echo ""
echo "  NCU: cesm_atm_cloud all paths"
for pid in "${PATHS[@]}"; do
    run_path "$CESM_DIR" "$CESM_RAW" cesm_atm_cloud "$pid" "ncu"
done
# Copy CSVs from scripts/results/ to canonical results/
CSV_SRC="$SCRIPT_DIR/results/reliability_layer1/phase4/nmr_c_ncu_free_redundancy/job_${SLURM_JOB_ID}"
if [ -d "$CSV_SRC" ]; then
    cp -f "$CSV_SRC"/*.csv "$RESULTS_DIR/" 2>/dev/null || true
fi
# Run aggregator to produce normalized metrics and verdict
echo ""
echo "=== Running NCU aggregator ==="
python3 "$SCRIPT_DIR/aggregate_nmr_c_ncu.py" --results-dir "$RESULTS_DIR" 2>&1 | tee "$RESULTS_DIR/aggregator_output.txt"
# Provenance
cat > "$RESULTS_DIR/provenance_manifest.json" <<ENDJSON
{
    "experiment": "nmr_c_ncu_free_redundancy",
    "job_id": "$SLURM_JOB_ID",
    "hostname": "$(hostname)",
    "gpu": "$GPU_NAME",
    "date": "$(date -Iseconds)",
    "paths": ["P0_baseline_byteplane_k2", "P1_digest_only", "P2_vote_read_compare", "P3_digest_plus_vote", "P4_raw_fused_fp64_reference"],
    "datasets": ["hurricane_u", "cesm_atm_cloud"],
    "iters_latency": $ITERS,
    "iters_ncu": 10
}
ENDJSON
# Completion marker
VERDICT=$(cat "$RESULTS_DIR/verdict.txt" 2>/dev/null || echo "UNKNOWN")
cat > "$RESULTS_DIR/job_marker.json" <<ENDJSON
{
    "job_id": "$SLURM_JOB_ID",
    "experiment": "nmr_c_ncu_free_redundancy",
    "status": "complete",
    "verdict": "$VERDICT",
    "date": "$(date -Iseconds)"
}
ENDJSON
echo ""
echo "=== NMR-C NCU Profile Complete ==="
echo "Results: $RESULTS_DIR"
echo "CSV: $CSV"
echo "Verdict: $VERDICT"
date
