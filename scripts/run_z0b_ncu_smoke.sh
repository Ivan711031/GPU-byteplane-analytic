#!/bin/bash
#SBATCH -J z0b_ncu_smoke
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=END,FAIL
set -euo pipefail
echo "=== Phase 3-Z0b Nsight Compute Smoke ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME"
if ! echo "$GPU_NAME" | grep -qi "H200"; then
    echo "FATAL: Expected H200, got $GPU_NAME"
    exit 2
fi
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
ml load cmake/4.0.0
conda activate gpu-byteplane-scan
BUILD_DIR="${WORK_DIR}/builds/z0b_ncu_${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake ${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/benchmarks/experiment4_filter_aggregate \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc) bench_z0b_bandwidth_derisk
echo "=== Build complete ==="
DATASET_DIR="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"
BENCH="./bench_z0b_bandwidth_derisk"
RESULTS_DIR="${WORK_DIR}/results/reliability_layer1/phase3/phase3z_z0b/job_${SLURM_JOB_ID}_ncu"
mkdir -p "$RESULTS_DIR"
# Metrics list
NCU_METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed"
NCU_METRICS="$NCU_METRICS,dram__throughput.avg.pct_of_peak_sustained_elapsed"
NCU_METRICS="$NCU_METRICS,dram__bytes.sum.per_second"
NCU_METRICS="$NCU_METRICS,sm__warps_active.avg.pct_of_peak_sustained_elapsed"
NCU_METRICS="$NCU_METRICS,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
NCU_METRICS="$NCU_METRICS,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"
NCU_METRICS="$NCU_METRICS,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"
NCU_METRICS="$NCU_METRICS,l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
NCU_METRICS="$NCU_METRICS,lts__throughput.avg.pct_of_peak_sustained_elapsed"
NCU_METRICS="$NCU_METRICS,sm__inst_executed.avg.pct_of_peak_sustained_elapsed"
# Profile each path kernel
for KERNEL in "path_a_read" "path_b_crc_1tpu" "path_c_crc_repair" "path_d_eager_vote"; do
    echo ""
    echo "=== Profiling $KERNEL ==="
    CSV_OUT="$RESULTS_DIR/${KERNEL}_ncu.csv"
    ncu --target-processes all \
        --kernel-name "regex:$KERNEL" \
        --replay-mode application \
        --metrics "$NCU_METRICS" \
        --csv \
        --page details \
        -o "$RESULTS_DIR/${KERNEL}_ncu" \
        -- "$BENCH" --dataset "$DATASET_DIR" --k 1 --fault-rate 0 --csv /dev/null \
        2>&1 | tee "$RESULTS_DIR/${KERNEL}_ncu_stdout.txt"
    echo "  $KERNEL done"
done
# Summary
echo ""
echo "=== NCU Summary ==="
RESULTS_DIR_NCU="$RESULTS_DIR" python3 << 'PYEOF'
import re, os
results_dir = os.environ.get("RESULTS_DIR_NCU", "")
for kernel in ["path_a_read", "path_b_crc_1tpu", "path_c_crc_repair", "path_d_eager_vote"]:
    stdout_path = os.path.join(results_dir, f"{kernel}_ncu_stdout.txt")
    if not os.path.exists(stdout_path):
        print(f"  {kernel}: no output")
        continue
    text = open(stdout_path).read()
    # Extract key metrics from ncu stdout
    dram_pct = re.search(r"dram__throughput.*?([\d.]+)\s*%", text)
    sm_pct = re.search(r"sm__throughput.*?([\d.]+)\s*%", text)
    occupancy = re.search(r"sm__warps_active.*?([\d.]+)\s*%", text)
    mem_stall = re.search(r"long_scoreboard.*?([\d.]+)", text)
    math_stall = re.search(r"math_pipe_throttle.*?([\d.]+)", text)
    print(f"  {kernel}:")
    print(f"    DRAM throughput:  {dram_pct.group(1) if dram_pct else 'N/A'}%")
    print(f"    SM throughput:    {sm_pct.group(1) if sm_pct else 'N/A'}%")
    print(f"    Occupancy:        {occupancy.group(1) if occupancy else 'N/A'}%")
    print(f"    Mem stall:        {mem_stall.group(1) if mem_stall else 'N/A'}")
    print(f"    Math stall:       {math_stall.group(1) if math_stall else 'N/A'}")
    if dram_pct and sm_pct:
        if float(dram_pct.group(1)) > float(sm_pct.group(1)):
            print(f"    => MEMORY BOUND (DRAM > SM)")
        else:
            print(f"    => COMPUTE BOUND (SM >= DRAM)")
PYEOF
echo ""
echo "=== NCU smoke complete ==="
date
