#!/bin/bash
#SBATCH --job-name=rl_15_fp
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=${PROJ_DIR}/gpu-byteplane-scan-experiments/logs/rl_15_fp_%j.log
#SBATCH --error=${PROJ_DIR}/gpu-byteplane-scan-experiments/logs/rl_15_fp_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${PROJ_DIR}/gpu-byteplane-scan-experiments}"
cd "$REPO_ROOT"
JOB_ID="${SLURM_JOB_ID:-nojob}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ART_ROOT="${RELIABILITY_ARTIFACT_ROOT:-${WORK_DIR}/datasets/reliability_layer1}"
PARAMS_JSON="$REPO_ROOT/results/phase1_5_strategy_params.json"
HANDOFF_DIR="$REPO_ROOT/handoff/job_done"
mkdir -p "$HANDOFF_DIR"
echo "[$TIMESTAMP] Phase 1.5 fault plan generation (job $JOB_ID)"
for sid in native_scale1 per_dataset hybrid_scale100_plus_shifted; do
    echo "[$(date)] Generating fault plans for $sid..."
    RELIABILITY_ARTIFACT_ROOT="$ART_ROOT" \
    python3 "$REPO_ROOT/scripts/generate_phase1_5_fault_plans.py" \
        --strategy-id "$sid" \
        --params-json "$PARAMS_JSON" \
        --datasets sensor uniform heavy_tailed zipfian \
        --n-rows 100000000 \
        --rates 1e-08 1e-07 1e-06 1e-05 1e-04 \
        --seeds 0-29
    echo "  Fault plans for $sid done"
done
echo "[$(date)] Fault plan generation complete"
cat > "$HANDOFF_DIR/job_${JOB_ID}.json" <<EOF
{
  "job_id": $JOB_ID,
  "job_name": "rl_15_fp",
  "exit_status": 0,
  "finished_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "workdir": "$REPO_ROOT",
  "stdout": "${PROJ_DIR}/gpu-byteplane-scan-experiments/logs/rl_15_fp_${JOB_ID}.log",
  "next_action": "Submit 12 canonical matrix GPU jobs (3 strategies x 4 datasets), then combined review"
}
EOF
echo "ALL PASSED"
