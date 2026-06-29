#!/bin/bash
#SBATCH -J c1ns-%x
#SBATCH -p dev
#SBATCH -t 0-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
set -euo pipefail
WORKTREE_ROOT="${PROJ_DIR}/workspace/gpu-byteplane-scan-experiments/.worktrees/WORKTREE"
RESULTS_DIR="$WORKTREE_ROOT/results/reliability_layer1/claim1_nosum32_pilot_combinefix/job_${SLURM_JOB_ID}"
cd "$WORKTREE_ROOT"
echo "=== Job $SLURM_JOB_ID: ${DATASET} rates=${RATES} ==="
echo "Worktree: $WORKTREE_ROOT"
echo "Results:  $RESULTS_DIR"
echo "Git HEAD: $(git rev-parse HEAD)"
echo "Start:    $(date -Iseconds)"
echo "Host:     $(hostname)"
echo
RUNNER="$WORKTREE_ROOT/scripts/nmr_claim1_realistic_campaign.py"
SEGMENT_SIZE=4096
N_ROWS=5000000
case "$DATASET" in
    hurricane_u)   ARTIFACT="${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg4096" ;;
    cesm_atm_cloud) ARTIFACT="${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg4096" ;;
    *) echo "Unknown dataset: $DATASET"; exit 1 ;;
esac
echo ">>> Dataset: $DATASET rates=$RATES ($(date -Iseconds))"
python -u "$RUNNER" \
    --artifact-dir "$ARTIFACT" \
    --dataset "$DATASET" \
    --n-rows "$N_ROWS" \
    --segment-size "$SEGMENT_SIZE" \
    --mode replication_only \
    --fault-families F1 F2 F3 F4 F5 F6 F7 F8 \
    --rate-anchors $RATES \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --output-dir "$RESULTS_DIR/$DATASET"
echo "<<< Done: $DATASET ($(date -Iseconds))"
echo
echo "=== Complete ==="
echo "End:      $(date -Iseconds)"
mkdir -p "$WORKTREE_ROOT/handoff/job_done"
cat > "$WORKTREE_ROOT/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{
  "job_id": ${SLURM_JOB_ID},
  "job_name": "c1ns-${DATASET}-${RATES// /_}",
  "exit_status": 0,
  "finished_at": "$(date -Iseconds)",
  "output_dir": "${RESULTS_DIR}/${DATASET}",
  "git_commit": "$(git -C "$WORKTREE_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)",
  "mode": "replication_only",
  "dataset": "${DATASET}",
  "policies": 20,
  "families": "F1 F2 F3 F4 F5 F6 F7 F8",
  "rates": "${RATES}",
  "seeds": "0 1 2 3 4 5 6 7 8 9",
  "n_rows": ${N_ROWS},
  "next_action": "Aggregate split results and run comparison."
}
EOF
echo "Handoff marker written."
