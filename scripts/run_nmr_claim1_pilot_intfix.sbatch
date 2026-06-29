#!/bin/bash
#SBATCH -J c1ns-pilot-intfix
#SBATCH -p dev
#SBATCH -t 0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
set -euo pipefail
WORKTREE_ROOT="${PROJ_DIR}/workspace/gpu-byteplane-scan-experiments/.worktrees/WORKTREE"
RESULTS_DIR="$WORKTREE_ROOT/results/reliability_layer1/claim1_nosum32_pilot_intfix/job_${SLURM_JOB_ID}"
cd "$WORKTREE_ROOT"
echo "=== Job $SLURM_JOB_ID: Claim1 No-SUM32 Pilot (int-fix) ==="
echo "Worktree: $WORKTREE_ROOT"
echo "Results:  $RESULTS_DIR"
echo "Git HEAD: $(git rev-parse HEAD)"
echo "Start:    $(date -Iseconds)"
echo "Host:     $(hostname)"
echo
RUNNER="$WORKTREE_ROOT/scripts/nmr_claim1_realistic_campaign.py"
SEGMENT_SIZE=4096
N_ROWS=5000000
# ── Canonical P0 rerun matrix ───────────────────────────────────
# Matches the exact 4-rate-anchor sweep from the old pilot
# (jobs 108937-108940 combined).
run_dataset() {
    local dataset="$1" artifact_dir="$2"
    echo ">>> Dataset: $dataset ($(date -Iseconds))"
    python -u "$RUNNER" \
        --artifact-dir "$artifact_dir" \
        --dataset "$dataset" \
        --n-rows "$N_ROWS" \
        --segment-size "$SEGMENT_SIZE" \
        --mode replication_only \
        --fault-families F1 F2 F3 F4 F5 F6 F7 F8 \
        --rate-anchors 1e-9 1e-7 1e-5 1e-3 \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --output-dir "$RESULTS_DIR/$dataset"
    echo "<<< Done: $dataset ($(date -Iseconds))"
    echo
}
run_dataset hurricane_u   "${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg4096"
run_dataset cesm_atm_cloud "${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"
echo "=== All complete ==="
echo "End:      $(date -Iseconds)"
mkdir -p "$WORKTREE_ROOT/handoff/job_done"
cat > "$WORKTREE_ROOT/handoff/job_done/job_${SLURM_JOB_ID}.json" <<EOF
{
  "job_id": ${SLURM_JOB_ID},
  "job_name": "c1ns-pilot-intfix",
  "exit_status": 0,
  "finished_at": "$(date -Iseconds)",
  "output_dir": "${RESULTS_DIR}",
  "git_commit": "$(git -C "$WORKTREE_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)",
  "mode": "replication_only",
  "datasets": "hurricane_u cesm_atm_cloud",
  "policies": 20,
  "families": "F1 F2 F3 F4 F5 F6 F7 F8",
  "rates": "1e-9 1e-7 1e-5 1e-3",
  "seeds": "0 1 2 3 4 5 6 7 8 9",
  "n_rows": ${N_ROWS},
  "next_action": "Run audits 1-3 on fixed int canonical pilot."
}
EOF
echo "Handoff marker written."
