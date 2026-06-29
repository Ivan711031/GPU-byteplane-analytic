#!/bin/bash
#SBATCH -J p3x0_repair_sanity
#SBATCH -p dev
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
set -euo pipefail
PROJ_DIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr"
OUT_DIR="${WORK_DIR}/results/reliability_layer1/phase3/phase3x0_repair_sanity/job_${SLURM_JOB_ID}"
cleanup() {
    local exit_code=$?
    mkdir -p "$PROJ_DIR/handoff/job_done"
    cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{
  "job_id": ${SLURM_JOB_ID},
  "job_name": "p3x0_repair_sanity",
  "exit_status": ${exit_code},
  "finished_at": "$(date -Iseconds)",
  "output_dir": "${OUT_DIR}",
  "next_action": "Write Phase 3-X0 repair sanity report."
}
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
ml load gcc/11.5
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME"
if [[ "$GPU_NAME" != *"H200"* ]]; then
    echo "ERROR: Expected H200, got $GPU_NAME"
    exit 2
fi
cd "$PROJ_DIR"
# ------------------------------------------------------------------
# Verify baseline benchmarks unchanged
# ------------------------------------------------------------------
echo ""
echo "=== Baseline non-regression check ==="
BASELINE_1="benchmarks/experiment4_filter_aggregate/bench_filter_aggregate.cu"
BASELINE_2="benchmarks/experiment4_filter_aggregate/bench_graded_nmr.cu"
BENCHMARK_LOG="$OUT_DIR/git_diff_baseline.log"
git diff --exit-code "$BASELINE_1" "$BASELINE_2" > "$BENCHMARK_LOG" 2>&1 && \
    echo "PASS: Baseline benchmarks unchanged." || \
    echo "FAIL: Baseline benchmarks modified! See $BENCHMARK_LOG"
# ------------------------------------------------------------------
# Build
# ------------------------------------------------------------------
echo ""
echo "=== Building bench_graded_repair ==="
BUILD_DIR="benchmarks/experiment4_filter_aggregate/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
rm -f CMakeCache.txt
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc) bench_graded_repair 2>&1 | tee "$OUT_DIR/build.log"
# Also rebuild other targets for the build dir completeness
make -j$(nproc) bench_filter_aggregate bench_graded_nmr bench_nmr_overhead 2>&1 | tee -a "$OUT_DIR/build.log"
cd "$PROJ_DIR"
X0="./benchmarks/experiment4_filter_aggregate/build/bench_graded_repair"
# ------------------------------------------------------------------
# Dataset paths
# ------------------------------------------------------------------
CESM_CLOUD="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg_global"
CESM_RAW="${WORK_DIR}/datasets/locality_sensitivity/dev/cesm_atm_cloud.f64le.bin"
HURRICANE_U="${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg_global"
HURRICANE_RAW="${WORK_DIR}/datasets/locality_sensitivity/dev/hurricane_u.f64le.bin"
CESM_Q="${WORK_DIR}/datasets/scientific/dev_buff_v2_scientific/cesm_atm_q/bfp-dec12"
HURRICANE_TC="${WORK_DIR}/datasets/scientific/dev_buff_v2_scientific/hurricane_tc/bfp-dec12"
# ------------------------------------------------------------------
# Smoke test: cesm_atm_cloud (seg_global, 168M values, 7 planes, 1 segment)
# ------------------------------------------------------------------
echo "" | tee -a "$OUT_DIR/results.log"
echo "==============================================" | tee -a "$OUT_DIR/results.log"
echo " cesm_atm_cloud seg_global (B=3)" | tee -a "$OUT_DIR/results.log"
echo "==============================================" | tee -a "$OUT_DIR/results.log"
$X0 \
    --dataset "$CESM_CLOUD" \
    --raw "$CESM_RAW" \
    --repair-budget 3 \
    --csv "$OUT_DIR/cesm_atm_cloud_x0_canonical.csv" \
    --coverage-csv "$OUT_DIR/cesm_atm_cloud_coverage_manifest.csv" \
    --handoff "$OUT_DIR/cesm_atm_cloud_handoff.json" \
    2>&1 | tee -a "$OUT_DIR/results.log"
# ------------------------------------------------------------------
# Smoke test: hurricane_u (seg_global, 25M values, 7 planes, 1 segment)
# ------------------------------------------------------------------
echo "" | tee -a "$OUT_DIR/results.log"
echo "==============================================" | tee -a "$OUT_DIR/results.log"
echo " hurricane_u seg_global (B=3)" | tee -a "$OUT_DIR/results.log"
echo "==============================================" | tee -a "$OUT_DIR/results.log"
$X0 \
    --dataset "$HURRICANE_U" \
    --raw "$HURRICANE_RAW" \
    --repair-budget 3 \
    --csv "$OUT_DIR/hurricane_u_x0_canonical.csv" \
    --coverage-csv "$OUT_DIR/hurricane_u_coverage_manifest.csv" \
    --handoff "$OUT_DIR/hurricane_u_handoff.json" \
    2>&1 | tee -a "$OUT_DIR/results.log"
# ------------------------------------------------------------------
# Multi-segment test: cesm_atm_q bfp-dec12 (168M values, 41133 segments)
# (No raw data available → artifact_sanity is expected failure)
# ------------------------------------------------------------------
echo "" | tee -a "$OUT_DIR/results.log"
echo "==============================================" | tee -a "$OUT_DIR/results.log"
echo " cesm_atm_q bfp-dec12 (multi-segment, B=3)" | tee -a "$OUT_DIR/results.log"
echo "==============================================" | tee -a "$OUT_DIR/results.log"
$X0 \
    --dataset "$CESM_Q" \
    --repair-budget 3 \
    --csv "$OUT_DIR/cesm_atm_q_x0_canonical.csv" \
    --coverage-csv "$OUT_DIR/cesm_atm_q_coverage_manifest.csv" \
    --handoff "$OUT_DIR/cesm_atm_q_handoff.json" \
    2>&1 | tee -a "$OUT_DIR/results.log" || echo "[NOTE] cesm_atm_q expected: no raw path available"
# ------------------------------------------------------------------
# Budget sweep: test B=1..4 on hurricane_u
# ------------------------------------------------------------------
for B in 1 2 4; do
    echo "" | tee -a "$OUT_DIR/results.log"
    echo "--- hurricane_u B=${B} ---" | tee -a "$OUT_DIR/results.log"
    $X0 \
        --dataset "$HURRICANE_U" \
        --raw "$HURRICANE_RAW" \
        --repair-budget "$B" \
        --csv "$OUT_DIR/hurricane_u_B${B}_x0_canonical.csv" \
        --coverage-csv "$OUT_DIR/hurricane_u_B${B}_coverage_manifest.csv" \
        --handoff "$OUT_DIR/hurricane_u_B${B}_handoff.json" \
        2>&1 | tee -a "$OUT_DIR/results.log" || echo "[NOTE] B=${B} completed with notes"
done
# ------------------------------------------------------------------
# Aggregate canonical CSV
# ------------------------------------------------------------------
echo "" | tee -a "$OUT_DIR/results.log"
echo "=== Aggregating canonical CSV ===" | tee -a "$OUT_DIR/results.log"
# Merge all canonical CSVs (skip header for subsequent files)
HEADER=$(head -1 "$OUT_DIR/cesm_atm_cloud_x0_canonical.csv")
echo "$HEADER" > "$OUT_DIR/x0_canonical.csv"
for f in "$OUT_DIR"/*_x0_canonical.csv; do
    tail -n +2 "$f" >> "$OUT_DIR/x0_canonical.csv"
done
echo "Merged CSV: $OUT_DIR/x0_canonical.csv" | tee -a "$OUT_DIR/results.log"
wc -l "$OUT_DIR/x0_canonical.csv" | tee -a "$OUT_DIR/results.log"
# ------------------------------------------------------------------
# Final summary
# ------------------------------------------------------------------
echo "" | tee -a "$OUT_DIR/results.log"
echo "==============================================" | tee -a "$OUT_DIR/results.log"
echo " Phase 3-X0 Complete" | tee -a "$OUT_DIR/results.log"
echo "==============================================" | tee -a "$OUT_DIR/results.log"
echo "Output directory: $OUT_DIR" | tee -a "$OUT_DIR/results.log"
echo "X0 canonical CSV: $OUT_DIR/x0_canonical.csv" | tee -a "$OUT_DIR/results.log"
echo "Git branch: $(git branch --show-current)" | tee -a "$OUT_DIR/results.log"
echo "Commit: $(git log -1 --oneline)" | tee -a "$OUT_DIR/results.log"
echo ""
echo "Done."
