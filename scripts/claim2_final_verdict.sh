#!/usr/bin/env bash
# Produce the reproducible final Claim 2 combined verdict.
# Combines:
#   - Suite B/C from full pilot (job_97293)
#   - Suite D from A/D rerun   (job_97488)
#   - Suite A from fixed local eval (no-fault null control)
# Then runs claim2_aggregate.py to produce the verdict.
#
# Usage:
#   ./scripts/claim2_final_verdict.sh
#
# Output:
#   results/reliability_layer1/phase3/claim2_final/claim2_final_matrix.csv
#   results/reliability_layer1/phase3/claim2_final/claim2_final_verdict.md
#
# Prerequisites:
#   - results/reliability_layer1/phase3/claim2_pilot/job_97293/claim2_pilot_matrix.csv
#   - results/reliability_layer1/phase3/claim2_ad_rerun/job_97488/claim2_pilot_matrix.csv
#   - results/reliability_layer1/phase3/claim2_suite_a_fixed/job_local/claim2_suite_a_fixed_matrix.csv
#   - scripts/claim2_aggregate.py in PYTHONPATH

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$BASE_DIR/results/reliability_layer1/phase3/claim2_final"
mkdir -p "$OUT_DIR"

COMBINED_CSV="$OUT_DIR/claim2_final_matrix.csv"
PYTHON=/home/u4063895/.conda/envs/gpu-byteplane-scan/bin/python3

"$PYTHON" -c "
import csv, sys

PILOT    = '$BASE_DIR/results/reliability_layer1/phase3/claim2_pilot/job_97293/claim2_pilot_matrix.csv'
AD_RERUN = '$BASE_DIR/results/reliability_layer1/phase3/claim2_ad_rerun/job_97488/claim2_pilot_matrix.csv'
A_FIXED  = '$BASE_DIR/results/reliability_layer1/phase3/claim2_suite_a_fixed/job_local/claim2_suite_a_fixed_matrix.csv'
OUTPUT   = '$COMBINED_CSV'

def load_suites(path, suites):
    with open(path) as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r['suite'] in suites]

bc = load_suites(PILOT,    ['suite_b', 'suite_c'])
d  = load_suites(AD_RERUN, ['suite_d'])
a  = load_suites(A_FIXED,  ['suite_a'])

combined = bc + d + a
fn = list(combined[0].keys())
with open(OUTPUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader()
    w.writerows(combined)
print(f'Combined: {len(bc)} (B/C) + {len(d)} (D) + {len(a)} (A) = {len(combined)} rows')
print(f'Output: {OUTPUT}')
"

echo "--- Running aggregator ---"
cd "$BASE_DIR"
"$PYTHON" scripts/claim2_aggregate.py \
    --input-csv "$COMBINED_CSV" \
    --output-dir "$OUT_DIR" \
    --label claim2_final

echo "--- Verdict ---"
cat "$OUT_DIR/claim2_final_verdict.md"
