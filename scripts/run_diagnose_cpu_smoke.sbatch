#!/usr/bin/env bash
#SBATCH -p ngs32g
#SBATCH -J diagnose_cpu_smoke
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o results/diagnostics/diagnose_cpu_smoke_%j.out
#SBATCH -e results/diagnostics/diagnose_cpu_smoke_%j.err
#
# CPU-only diagnostics: sensor p10 mismatch + heavy_tailed p6 count path.
# No GPU required. Run on ngs32g (CPU-only) partition.
#
# Usage:
#   pass the project allocation on the sbatch command line when submitting
set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"
RESULT_DIR="$ROOT_DIR/results/diagnostics"
HANDOFF_DIR="$ROOT_DIR/handoff/job_done"
mkdir -p "$RESULT_DIR" "$HANDOFF_DIR"
write_marker() {
  local status="$1"
  local marker="$HANDOFF_DIR/job_${SLURM_JOB_ID}.json"
  python3 - "$marker" "$status" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
marker = Path(sys.argv[1])
status = int(sys.argv[2])
root = Path(os.environ["ROOT_DIR"])
result_dir = Path(os.environ["RESULT_DIR"])
payload = {
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "job_name": os.environ.get("SLURM_JOB_NAME"),
    "exit_status": status,
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "workdir": str(root),
    "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
    "stdout": str(result_dir / f"diagnose_cpu_smoke_{os.environ.get('SLURM_JOB_ID')}.out"),
    "stderr": str(result_dir / f"diagnose_cpu_smoke_{os.environ.get('SLURM_JOB_ID')}.err"),
    "gpu_util_log": None,
    "next_action": "Compare sensor_p10 and heavy_tailed_p6 diagnostic results against expected benchmark counts.",
}
marker.write_text(json.dumps(payload, indent=2) + "\n")
PY
}
trap 'write_marker "$?"' EXIT
GIT_COMMIT="$(git rev-parse HEAD)"
export ROOT_DIR RESULT_DIR GIT_COMMIT
if command -v ml >/dev/null 2>&1; then
  ml purge
  ml load jupyter/miniconda3 >/dev/null 2>&1 || ml load miniconda3 >/dev/null 2>&1 || true
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate gpu-byteplane-scan || true
fi
python3 - <<'PY'
import numpy
print("numpy", numpy.__version__)
PY
echo "[$(date -Is)] === sensor_p10 mismatch diagnosis ==="
python3 scripts/diagnose_sensor_p10_mismatch.py \
  2>&1 | tee "$RESULT_DIR/sensor_p10_mismatch.out"
echo "Exit: $?"
echo "[$(date -Is)] === heavy_tailed_p6 count path diagnosis ==="
python3 scripts/diagnose_heavy_tailed_p6.py \
  2>&1 | tee "$RESULT_DIR/heavy_tailed_p6_diagnose.out"
echo "Exit: $?"
echo "[$(date -Is)] diagnose_cpu_smoke complete: $RESULT_DIR"
