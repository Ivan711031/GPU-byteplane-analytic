#!/bin/bash
#SBATCH -J p3r_reconcile
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
OUT_DIR="${WORK_DIR}/results/reliability_layer1/phase3/phase3r_reconcile/job_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"
trap cleanup EXIT
cleanup() {
    local exit_code=$?
    mkdir -p "$PROJ_DIR/handoff/job_done"
    cat > "$PROJ_DIR/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{
  "job_id": ${SLURM_JOB_ID},
  "job_name": "p3r_reconcile",
  "exit_status": ${exit_code},
  "finished_at": "$(date -Iseconds)",
  "output_dir": "${OUT_DIR}",
  "next_action": "Write reconciliation report."
}
EOF
}
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date -Iseconds)"
module purge
ml load cmake/4.0.0
ml load cuda/12.6
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME"
if [[ "$GPU_NAME" != *"H200"* ]]; then
    echo "ERROR: Expected H200, got $GPU_NAME"
    exit 2
fi
cd "$PROJ_DIR"
# -------------------------------------------------------------------------
# Build bench_filter_aggregate (main paper fused Filter+Aggregate kernel)
# -------------------------------------------------------------------------
BUILD_DIR="benchmarks/experiment4_filter_aggregate/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j bench_filter_aggregate
cd "$PROJ_DIR"
BENCH="./benchmarks/experiment4_filter_aggregate/build/bench_filter_aggregate"
# Raw scientific data NOT available on CLUSTER_HOST (resides on CLUSTER_HOST workspace).
# Q1 byte-plane timing is reproducible; raw-fused baseline uses paper source CSV numbers.
# -------------------------------------------------------------------------
# Q1: Reproduce main paper byte-plane fused timing on CESM Cloud seg_global
# -------------------------------------------------------------------------
CESM_CLOUD="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg_global"
HURRICANE_U="${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg_global"
echo "=== Q1: Reproduce main paper H200 byte-plane fused timing ===" > "$OUT_DIR/q1_results.txt"
# CESM Cloud — 3 selectivities at k=2 (main paper shallow-k)
for SEL in 10 50 90; do
    THRESH="$(python3 -c "print(${SEL}/100.0 * 0.964)")"  # scale to [0, 0.964] range
    echo "" | tee -a "$OUT_DIR/q1_results.txt"
    echo "--- cesm_atm_cloud sel=${SEL}% k=2 (byte-plane fused) ---" | tee -a "$OUT_DIR/q1_results.txt"
    $BENCH \
        --dataset "$CESM_CLOUD" \
        --threshold "$THRESH" \
        --segment-rows 168480000 \
        --iters 100 \
        --warmup 10 \
        --max-filter-planes 2 \
        --csv "$OUT_DIR/q1_cesm_cloud_bp.csv" \
        2>&1 | tee -a "$OUT_DIR/q1_results.txt"
done
# Hurricane U (25M rows)
if [ -d "$HURRICANE_U" ]; then
    HURR_MAX=$(python3 -c "print(max(abs(0.0), abs(1.0)))")  # placeholder — adjust threshold
    for SEL in 10 50 90; do
        echo "" | tee -a "$OUT_DIR/q1_results.txt"
        echo "--- hurricane_u sel=${SEL}% k=2 (byte-plane fused) ---" | tee -a "$OUT_DIR/q1_results.txt"
        $BENCH \
            --dataset "$HURRICANE_U" \
            --threshold "$(python3 -c "print(${SEL}/100.0)")" \
            --segment-rows 25000000 \
            --iters 100 \
            --warmup 10 \
            --max-filter-planes 2 \
            --csv "$OUT_DIR/q1_hurricane_u_bp.csv" \
            2>&1 | tee -a "$OUT_DIR/q1_results.txt"
    done
fi
# Hurricane U (25M rows, global segment)
HURRICANE_U="${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg_global"
if [ -d "$HURRICANE_U" ]; then
    for SEL in 10 50 90; do
        echo "" | tee -a "$OUT_DIR/q1_results.txt"
        echo "--- hurricane_u selectivity=${SEL}% k=2 (raw FP64) ---" | tee -a "$OUT_DIR/q1_results.txt"
        $BENCH \
            --dataset "$HURRICANE_U" \
            --threshold "$(python3 -c "print(0.1*${SEL})")" \
            --segment-rows 25000000 \
            --iters 100 \
            --warmup 10 \
            --max-filter-planes 0 \
            --csv "$OUT_DIR/q1_hurricane_u_raw.csv" \
            2>&1 | tee -a "$OUT_DIR/q1_results.txt"
        echo "" | tee -a "$OUT_DIR/q1_results.txt"
        echo "--- hurricane_u selectivity=${SEL}% k=2 (byte-plane fused) ---" | tee -a "$OUT_DIR/q1_results.txt"
        $BENCH \
            --dataset "$HURRICANE_U" \
            --threshold "$(python3 -c "print(0.1*${SEL})")" \
            --segment-rows 25000000 \
            --iters 100 \
            --warmup 10 \
            --max-filter-planes 2 \
            --csv "$OUT_DIR/q1_hurricane_u_bp.csv" \
            2>&1 | tee -a "$OUT_DIR/q1_results.txt"
    done
fi
# -------------------------------------------------------------------------
# Q2: Compare #160 preflight vs main lane on CESM Cloud
# -------------------------------------------------------------------------
echo "" >> "$OUT_DIR/q2_analysis.txt"
echo "=== Q2: Main lane vs #160 preflight lane comparison ===" >> "$OUT_DIR/q2_analysis.txt"
echo "" >> "$OUT_DIR/q2_analysis.txt"
# Compare #160-style no-filter read+sum (proxy: threshold=0 so all qualify, sum is trivial)
# on the same CESM Cloud artifact at different k depths
echo "--- #160-style progressive (k=1, near-zero threshold) on CESM Cloud ---" >> "$OUT_DIR/q2_analysis.txt"
$BENCH \
    --dataset "$CESM_CLOUD" \
    --threshold -1e10 \
    --segment-rows 168480000 \
    --iters 100 \
    --warmup 10 \
    --max-filter-planes 1 \
    --csv "$OUT_DIR/q2_prog_k1.csv" \
    2>&1 | tee -a "$OUT_DIR/q2_analysis.txt"
echo "" >> "$OUT_DIR/q2_analysis.txt"
echo "--- #160-style progressive (k=2, near-zero threshold) on CESM Cloud ---" >> "$OUT_DIR/q2_analysis.txt"
$BENCH \
    --dataset "$CESM_CLOUD" \
    --threshold -1e10 \
    --segment-rows 168480000 \
    --iters 100 \
    --warmup 10 \
    --max-filter-planes 2 \
    --csv "$OUT_DIR/q2_prog_k2.csv" \
    2>&1 | tee -a "$OUT_DIR/q2_analysis.txt"
echo "" >> "$OUT_DIR/q2_analysis.txt"
echo "--- #160-style progressive (k=7, near-zero threshold) on CESM Cloud ---" >> "$OUT_DIR/q2_analysis.txt"
$BENCH \
    --dataset "$CESM_CLOUD" \
    --threshold -1e10 \
    --segment-rows 168480000 \
    --iters 100 \
    --warmup 10 \
    --max-filter-planes 7 \
    --csv "$OUT_DIR/q2_prog_k7.csv" \
    2>&1 | tee -a "$OUT_DIR/q2_analysis.txt"
# -------------------------------------------------------------------------
# Q3: Minimal fused replica/vote overhead probe
# -------------------------------------------------------------------------
echo "" >> "$OUT_DIR/q3_overhead.txt"
echo "=== Q3: Fused replica/vote overhead probe on CESM Cloud ===" >> "$OUT_DIR/q3_overhead.txt"
# We estimate NMR overhead from the benchmark's plane-level read costs.
# The main paper's fused kernel reads planes progressively. NMR voting requires
# reading an additional copy of the highest-significance plane and comparing.
# Highest active plane read (plane_000 = MSB, highest significance for 7-plane CESM)
echo "" >> "$OUT_DIR/q3_overhead.txt"
echo "--- Highest plane read overhead (k=1, plane_000 only, near-zero threshold) ---" >> "$OUT_DIR/q3_overhead.txt"
$BENCH \
    --dataset "$CESM_CLOUD" \
    --threshold -1e10 \
    --segment-rows 168480000 \
    --iters 100 \
    --warmup 10 \
    --max-filter-planes 1 \
    --csv "$OUT_DIR/q3_plane0.csv" \
    2>&1 | tee -a "$OUT_DIR/q3_overhead.txt"
# Full fused at k=2 (plane_000 + plane_001, 10% selectivity)
echo "" >> "$OUT_DIR/q3_overhead.txt"
echo "--- k=2 (2 planes) fused Filter+Aggregate at 10% sel ---" >> "$OUT_DIR/q3_overhead.txt"
THRESH="0.0964"
$BENCH \
    --dataset "$CESM_CLOUD" \
    --threshold "$THRESH" \
    --segment-rows 168480000 \
    --iters 100 \
    --warmup 10 \
    --max-filter-planes 2 \
    --csv "$OUT_DIR/q3_k2.csv" \
    2>&1 | tee -a "$OUT_DIR/q3_overhead.txt"
# 2-replica NMR proxy: k=2 + read highest plane again
echo "" >> "$OUT_DIR/q3_overhead.txt"
echo "NMR overhead estimate (from measured timings):" >> "$OUT_DIR/q3_overhead.txt"
echo "  T_k2_fused = k=2 fused Filter+Aggregate time (from Q1)" >> "$OUT_DIR/q3_overhead.txt"
echo "  T_plane0 = single highest-plane read time (from k=1 no-filter)" >> "$OUT_DIR/q3_overhead.txt"
echo "  T_2replica_approx = T_k2_fused + T_plane0 (read twice, compare once)" >> "$OUT_DIR/q3_overhead.txt"
echo "  Margin = T_raw_fused_paper - T_k2_fused (from paper source CSV)" >> "$OUT_DIR/q3_overhead.txt"
echo "  Overhead_ratio = T_plane0 / margin" >> "$OUT_DIR/q3_overhead.txt"
echo "  2-replica plausible if overhead_ratio < 1.0 (extra read fits within margin)" >> "$OUT_DIR/q3_overhead.txt"
echo "" >> "$OUT_DIR/q3_overhead.txt"
echo "Job complete. All results in: $OUT_DIR"
