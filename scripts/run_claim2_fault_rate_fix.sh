#!/usr/bin/env bash
#SBATCH -J claim2-fault-rate-fix
#SBATCH -t 0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u4063895@gs.nchc.org.tw
#
# Regenerate Claim 2 results with rate-parametrized fault generation.
# CPU-only pipeline (GPU requested only to satisfy partition MinGRES).
#
# Usage:
#   sbatch --account=gov108018 -p dev scripts/run_claim2_fault_rate_fix.sh
#
# Output:
#   results/reliability_layer1/phase3/claim2_fault_rate_fix/

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
HANDOFF_DIR="$REPO_DIR/handoff/job_done"
mkdir -p "$HANDOFF_DIR"

# Completion marker on exit
cleanup() {
    local ec=$?
    echo "{\"job_id\":${SLURM_JOB_ID:-0},\"job_name\":\"claim2-fault-rate-fix\",\"exit_status\":$ec,\"finished_at\":\"$(date -Iseconds)\",\"workdir\":\"$REPO_DIR\",\"git_commit\":\"$(cd "$REPO_DIR" && git rev-parse HEAD 2>/dev/null || echo unknown)\"}" > "$HANDOFF_DIR/job_${SLURM_JOB_ID:-local}.json"
    exit $ec
}
trap cleanup EXIT

# Hostname & hardware check
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

cd "$REPO_DIR"

# Python env
PYTHON=/home/u4063895/.conda/envs/gpu-byteplane-scan/bin/python3
echo "Python: $($PYTHON --version)"
echo "Numpy: $($PYTHON -c 'import numpy; print(numpy.__version__)')"

# Output base
OUT_BASE="results/reliability_layer1/phase3/claim2_fault_rate_fix"
mkdir -p "$OUT_BASE"

JID="${SLURM_JOB_ID:-local}"

echo "=== 1. Full pilot (all suites A-D) ==="
$PYTHON scripts/run_claim2_pilot.py \
    --datasets hurricane_u cesm_atm_cloud \
    --output-dir "$OUT_BASE/claim2_pilot" \
    --label claim2_pilot \
    --suite suite_a,suite_b,suite_c,suite_d

echo "=== 2. Aggregated verdict ==="
PILOT_CSV="$OUT_BASE/claim2_pilot/job_${JID}/claim2_pilot_matrix.csv"
$PYTHON scripts/claim2_aggregate.py \
    --input-csv "$PILOT_CSV" \
    --output-dir "$OUT_BASE/claim2_final" \
    --label claim2_final

echo "=== 3. Verdict ==="
cat "$OUT_BASE/claim2_final/claim2_final_verdict.md"

echo "=== 4. Check: Suite B/C/D rates differ ==="
$PYTHON -c "
import csv
from collections import defaultdict

with open('$PILOT_CSV') as f:
    rows = list(csv.DictReader(f))

# Group by (suite, family, rate) and check unique fault signatures
# We compare by the number of entries generated (approximated by escape_rate)
groups = defaultdict(set)
for r in rows:
    key = (r['suite'], r['fault_family'], r['rate'])
    groups[key].add(r['escape_rate'])

print('=== Rate differentiation check ===')
all_ok = True
for (suite, family, rate), escapes in sorted(groups):
    if suite == 'suite_a':
        continue  # Suite A is no-fault, expected to be same
    other_rates = []
    for r_test in ['1e-07','3e-07','1e-06','3e-06','1e-05','3e-05','1e-04']:
        test_key = (suite, family, r_test)
        if test_key in groups and test_key != (suite, family, rate):
            other_rates.append(r_test)
    # Check this rate's escape rates differ from others
    for other_rate in other_rates:
        other_escapes = groups[(suite, family, other_rate)]
        if escapes == other_escapes:
            print(f'  ⚠️  {suite}/{family}: rate={rate} IDENTICAL to rate={other_rate}')
            all_ok = False
    
if all_ok:
    print('✅ All Suite B/C/D rates produce different results')
else:
    print('❌ Some rates still not differentiated')
"

echo "=== Done ==="
echo "Output: $OUT_BASE"
echo "Job: ${SLURM_JOB_ID:-local}"
