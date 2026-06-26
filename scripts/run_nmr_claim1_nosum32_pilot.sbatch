#!/bin/bash
#SBATCH -J c1ns-full
#SBATCH -p dev
#SBATCH -t 0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u4063895@gs.ncku.edu.tw
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

WORKTREE_ROOT="/home/u4063895/workspace/gpu-byteplane-scan-experiments/.worktrees/issue-285-nmr-claim1-no-sum32"
RESULTS_DIR="$WORKTREE_ROOT/results/reliability_layer1/claim1_nosum32_pilot/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "=== Job $SLURM_JOB_ID: Claim1 No-SUM32 Full Matrix ==="
echo "Worktree: $WORKTREE_ROOT"
echo "Results:  $RESULTS_DIR"
echo "Start:    $(date -Iseconds)"
echo "Host:     $(hostname)"
echo

RUNNER="$WORKTREE_ROOT/scripts/nmr_claim1_realistic_campaign.py"
FAMILIES="${FAMILIES:-F1 F2 F3 F4 F5 F6 F7 F8}"
RATES="${RATES:-1e-9 1e-7 1e-5 1e-3}"
if [ -z "${SEEDS:-}" ]; then SEEDS=$(seq 0 29 | tr '\n' ' '); fi
N_ROWS=5000000

run_dataset() {
    local dataset="$1"
    local artifact_dir="$2"
    echo ">>> Dataset: $dataset ($(date -Iseconds))"
    echo ">>> Families: $FAMILIES"
    echo ">>> Rates: $RATES"
    echo ">>> Seeds: $(echo $SEEDS | wc -w) seeds"
    python "$RUNNER" \
        --artifact-dir "$artifact_dir" \
        --dataset "$dataset" \
        --n-rows "$N_ROWS" \
        --segment-size 4096 \
        --mode replication_only \
        --fault-families $FAMILIES \
        --rate-anchors $RATES \
        --seeds $SEEDS \
        --output-dir "$RESULTS_DIR/$dataset"
    echo "<<< Done: $dataset ($(date -Iseconds))"
    echo
}

if [ -n "${DATASET:-}" ]; then
    case "$DATASET" in
        hurricane_u)   run_dataset hurricane_u   "/work/u4063895/datasets/scientific/dev_buff_v2_scientific/hurricane_u/bfp-dec12/" ;;
        cesm_atm_cloud) run_dataset cesm_atm_cloud "/work/u4063895/datasets/scientific/dev_buff_v2_scientific/cesm_atm_cloud/bfp-dec12/" ;;
        *) echo "Unknown dataset: $DATASET"; exit 1 ;;
    esac
else
    run_dataset hurricane_u   "/work/u4063895/datasets/scientific/dev_buff_v2_scientific/hurricane_u/bfp-dec12/"
    run_dataset cesm_atm_cloud "/work/u4063895/datasets/scientific/dev_buff_v2_scientific/cesm_atm_cloud/bfp-dec12/"
fi

echo "=== All complete ==="
echo "End:      $(date -Iseconds)"

# Handoff marker
SEEDS_STR=$(echo $SEEDS | tr '\n' ' ' | sed 's/ *$//')
mkdir -p "$WORKTREE_ROOT/handoff/job_done"
cat > "$WORKTREE_ROOT/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{
  "job_id": ${SLURM_JOB_ID},
  "job_name": "c1ns-full-${DATASET:-all}",
  "exit_status": 0,
  "finished_at": "$(date -Iseconds)",
  "output_dir": "${RESULTS_DIR}",
  "git_commit": "$(git -C "$WORKTREE_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)",
  "mode": "replication_only",
  "datasets": "${DATASET:-hurricane_u cesm_atm_cloud}",
  "policies": 20,
  "families": "${FAMILIES}",
  "rates": "${RATES}",
  "seeds": "${SEEDS_STR}",
  "next_action": "Review full-matrix results and write verdict."
}
EOF
echo "Handoff marker written."
