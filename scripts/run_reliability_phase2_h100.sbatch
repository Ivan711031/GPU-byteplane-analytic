#!/bin/bash
#SBATCH --job-name=rl_phase2
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=${PROJ_DIR}/gpu-byteplane-scan-experiments/logs/rl_phase2_%j.log
#SBATCH --error=${PROJ_DIR}/gpu-byteplane-scan-experiments/logs/rl_phase2_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_ROOT"
JOB_ID="${SLURM_JOB_ID:-nojob}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_PATH="$REPO_ROOT/logs/rl_phase2_${JOB_ID}.log"
STDERR_PATH="$REPO_ROOT/logs/rl_phase2_${JOB_ID}.err"
DONE_DIR="$REPO_ROOT/handoff/job_done"
mkdir -p "$DONE_DIR"
write_handoff_marker() {
  local status="$1"
  local finished_at
  finished_at="$(date -Is)"
  cat > "$DONE_DIR/job_${JOB_ID}.json" << EOF
{"job_id":"$JOB_ID","job_name":"rl_phase2","exit_status":$status,"finished_at":"$finished_at","workdir":"$REPO_ROOT","stdout":"$STDOUT_PATH","stderr":"$STDERR_PATH","next_action":"Review seff, sacct, stdout, stderr, and Phase 2 output files."}
EOF
}
trap 'rc=$?; write_handoff_marker "$rc"; exit "$rc"' EXIT
STRATEGY_ID="${STRATEGY_ID:?Must set STRATEGY_ID}"
DATASET="${DATASET:?Must set DATASET}"
POLICY_ID="${POLICY_ID:-}"
BUDGET_POINTS="${BUDGET_POINTS:-8 12 16 20 24}"
FAULT_RATES="${FAULT_RATES:-1e-07 1e-06 1e-05}"
SEEDS="${SEEDS:-$(seq 0 14)}"
case "$STRATEGY_ID" in
  hybrid_scale100_plus_shifted) SID_DIR="hybrid" ;;
  *) SID_DIR="$STRATEGY_ID" ;;
esac
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
[[ "$GPU_NAME" == *"H100"* ]] || { echo "Not H100: $GPU_NAME"; exit 2; }
echo "[$TIMESTAMP] H100 OK: $GPU_NAME"
module load miniconda3/24.11.1 cmake/3.24.2 cuda/12.6
echo "nvcc: $(which nvcc)"
cmake --version | head -1
gcc --version | head -1
N=100000000
CATALOGUE="$REPO_ROOT/results/policy_catalogue.json"
PARAMS_JSON="$REPO_ROOT/results/phase1_5_strategy_params.json"
SCALE=$(python3 -c "
import json, sys
p = json.load(open('$PARAMS_JSON'))
try:
    d = p['strategies']['$STRATEGY_ID']['per_dataset']['$DATASET']
    print(d['scale'])
except KeyError as e:
    print(f'ERROR: missing param: {e}', file=sys.stderr)
    sys.exit(2)
")
SHIFT_K=$(python3 -c "
import json, sys
p = json.load(open('$PARAMS_JSON'))
try:
    d = p['strategies']['$STRATEGY_ID']['per_dataset']['$DATASET']
    print(d.get('shift_k', 0))
except KeyError as e:
    print(f'ERROR: missing param: {e}', file=sys.stderr)
    sys.exit(2)
")
if [[ "$SHIFT_K" -gt 0 ]]; then
  SCALE_DIR="scale${SCALE}_shift${SHIFT_K}"
else
  SCALE_DIR="scale${SCALE}"
fi
ART_ROOT="${RELIABILITY_ARTIFACT_ROOT:-${WORK_DIR}/datasets/reliability_layer1}"
RESULTS_BASE="${RELIABILITY_RESULTS_ROOT:-${WORK_DIR}/results/reliability_layer1}"
ART_DIR="$ART_ROOT/artifacts_phase1_5/$SID_DIR/$DATASET/n${N}/$SCALE_DIR"
FAULT_BASE="$ART_ROOT/fault_plans_phase2/$DATASET/n${N}/scale${SCALE}"
RESULTS_DIR="$RESULTS_BASE/phase2/${POLICY_ID:-all}/${DATASET}_${TIMESTAMP}_job${JOB_ID}_H100"
TMPDIR="/tmp/rl_phase2_${DATASET}_${JOB_ID}"
RATE_COUNT=$(printf "%s\n" $FAULT_RATES | wc -l)
SEED_COUNT=$(printf "%s\n" $SEEDS | wc -l)
mkdir -p "$TMPDIR" "$RESULTS_DIR"
if [[ ! -f "$ART_DIR/artifact.json" ]]; then
  echo "ERROR: missing artifact $ART_DIR"; exit 2
fi
echo "Artifact: $DATASET ($(du -sh "$ART_DIR" | cut -f1))"
BUILD_DIR="$REPO_ROOT/build/reliability_layer1_job${JOB_ID}"
cmake -S "$REPO_ROOT/benchmarks/reliability_layer1" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "$BUILD_DIR" -j --clean-first
FIXTURE_BIN="$BUILD_DIR/bench_reliability_fixture"
PHASE2_BIN="$BUILD_DIR/bench_reliability_phase2"
test -x "$FIXTURE_BIN" || { echo "Build failed (fixture)"; exit 2; }
test -x "$PHASE2_BIN" || { echo "Build failed (phase2)"; exit 2; }
echo "[$(date)] Clean gate..."
"$FIXTURE_BIN" --device 0 --n "$N" --artifact-dir "$ART_DIR" --run-clean \
  --csv "$TMPDIR/clean_${DATASET}.csv" 2>&1 | grep 'clean gate passed' || {
  echo "CLEAN GATE FAILED: $DATASET"; exit 2; }
echo "OK: $DATASET"
echo "[$(date)] Pre-flights..."
# ── Pre-flight 1: vacuous-plane classification ──
python3 -c "
import json, sys
cat = json.load(open('$CATALOGUE'))
ds_entries = [e for e in cat['entries'] if e['dataset'] == '$DATASET']
if not ds_entries:
    print(f'ERROR: no catalogue entries for dataset $DATASET', file=sys.stderr)
    sys.exit(2)
vp = ds_entries[0].get('vacuous_planes', [])
print(f'vacuous_planes={vp}')
for e in ds_entries:
    if e['policy'] != 'graded_vacuous_aware':
        continue
    for p in vp:
        if e['r_vector'][p] != 1:
            print(f'ERROR: vacuous plane {p} has r_p={e[\"r_vector\"][p]} != 1 in {e[\"policy\"]} B={e[\"budget_B\"]}', file=sys.stderr)
            sys.exit(2)
print('vacuous-plane classification OK')
"
# ── Pre-flight 2: voting helper correctness ──
python3 -c "
import sys
sys.path.insert(0, '$REPO_ROOT/scripts')
from phase2_vote import vote_byte, vote_plane
assert vote_byte([5, 5, 5, 7, 9]) == 5
assert vote_byte([3, 7, 7, 3]) == 3
assert vote_byte([42, 42, 42]) == 42
assert vote_byte([99]) == 99
clean = bytes([10, 20, 30, 40])
r0 = bytes([10, 99, 30, 99])
r1 = bytes([10, 20, 30, 40])
r2 = bytes([10, 20, 30, 99])
voted, stats = vote_plane([r0, r1, r2], clean)
assert voted == bytes([10, 20, 30, 99]), f'voted={voted}'
assert stats['resolved_correctly'] == 3, f'resolved={stats[\"resolved_correctly\"]}'
assert stats['detected_mismatch'] == 1, f'detected={stats[\"detected_mismatch\"]}'
print('voting-helper correctness OK')
"
# ── Pre-flight 3: independence audit ──
python3 -c "
import json, hashlib
cat = json.load(open('$CATALOGUE'))
ds_entries = [e for e in cat['entries'] if e['dataset'] == '$DATASET']
for e in ds_entries:
    rv = e['r_vector']
    for p in range(8):
        if rv[p] <= 1:
            continue
        seeds = set()
        for j in range(rv[p]):
            key = f'0:{e[\"dataset\"]}:{p}:1e-06:{j}'
            h = hashlib.sha256(key.encode()).hexdigest()
            seeds.add(int(h[:16], 16))
        if len(seeds) != rv[p]:
            print(f'WARN: replica seed collision in {e[\"policy\"]} B={e[\"budget_B\"]} plane {p}')
print('independence audit done')
"
# ── Write helper: apply fault plan via XOR ──
cat > "$TMPDIR/apply_fault_plan.py" << 'PYEOF'
import json, sys
clean_path = sys.argv[1]
fault_plan_path = sys.argv[2]
output_path = sys.argv[3]
with open(clean_path, 'rb') as f:
    data = bytearray(f.read())
fp = json.load(open(fault_plan_path))
for entry in fp['entries']:
    off = entry['offset']
    mask = entry['mask']
    if off < len(data):
        data[off] ^= mask
with open(output_path, 'wb') as f:
    f.write(data)
PYEOF
# ── Write helper: vote replicas and save voted planes ──
cat > "$TMPDIR/phase2_vote_and_save.py" << PYEOF
from pathlib import Path
import csv, sys
sys.path.insert(0, "$REPO_ROOT/scripts")
from phase2_vote import vote_plane, _discover_replicas
replica_dir = Path(sys.argv[1])
clean_dir = Path(sys.argv[2])
n_rows = int(sys.argv[3])
voted_dir = Path(sys.argv[4])
csv_path = Path(sys.argv[5])
voted_dir.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "plane_id", "n_rows", "r_p",
    "resolved_correctly", "detected_mismatch",
    "undetected_corruption", "total",
    "resolved_frac", "detected_frac", "undetected_frac",
]
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for plane_id in range(8):
        clean_file = Path(clean_dir) / f"plane_{plane_id}.bin"
        clean_bytes = clean_file.read_bytes()
        if len(clean_bytes) != n_rows:
            print(f"plane {plane_id}: clean length {len(clean_bytes)} != n_rows {n_rows}", file=sys.stderr)
            sys.exit(1)
        replica_files = _discover_replicas(replica_dir, plane_id)
        if not replica_files:
            # No replicas — copy clean as voted, write CSV row and continue
            (voted_dir / f"plane_{plane_id}_voted.bin").write_bytes(clean_bytes)
            writer.writerow({
                "plane_id": str(plane_id), "n_rows": str(n_rows), "r_p": "1",
                "resolved_correctly": str(n_rows), "detected_mismatch": "0",
                "undetected_corruption": "0", "total": str(n_rows),
                "resolved_frac": "1.000000", "detected_frac": "0.000000",
                "undetected_frac": "0.000000",
            })
            continue
        replica_bytes = [f.read_bytes() for f in replica_files]
        voted, stats = vote_plane(replica_bytes, clean_bytes)
        (voted_dir / f"plane_{plane_id}_voted.bin").write_bytes(voted)
        total = n_rows
        row = {
            "plane_id": str(plane_id),
            "n_rows": str(n_rows),
            "r_p": str(len(replica_files)),
            "resolved_correctly": str(stats["resolved_correctly"]),
            "detected_mismatch": str(stats["detected_mismatch"]),
            "undetected_corruption": str(stats["undetected_corruption"]),
            "total": str(total),
            "resolved_frac": f"{stats['resolved_correctly'] / total:.6f}",
            "detected_frac": f"{stats['detected_mismatch'] / total:.6f}",
            "undetected_frac": f"{stats['undetected_corruption'] / total:.6f}",
        }
        writer.writerow(row)
        print(f"plane {plane_id}: r_p={row['r_p']} resolved={stats['resolved_correctly']} "
              f"detected={stats['detected_mismatch']} undetected={stats['undetected_corruption']}")
print(f"CSV written: {csv_path}")
PYEOF
# ── Write helper: per-config pipeline ──
cat > "$TMPDIR/run_one_config.sh" << 'CONFIGEOF'
#!/bin/bash
set -euo pipefail
CFG_DIR="$1"
POLICY="$2"
BUDGET_B="$3"
R_VEC_CSV="$4"
RATE="$5"
SEED="$6"
TMPDIR="$7"
ART_DIR="$8"
FAULT_BASE="$9"
PHASE2_BIN="${10}"
N="${11}"
STRATEGY_ID="${12}"
SCALE="${13}"
REPO_ROOT="${14}"
DATASET="${15}"
VACUOUS_CSV="${16}"
CFG_TAG="${POLICY}_B${BUDGET_B}_r${RATE}_s${SEED}"
REPLICA_DIR="$CFG_DIR/replicas"
VOTED_DIR="$CFG_DIR/voted"
mkdir -p "$REPLICA_DIR" "$VOTED_DIR"
RATE_DIR="rate$(LC_NUMERIC=C printf "%.0e" "$RATE")"
IFS=',' read -ra RVEC <<< "$R_VEC_CSV"
IFS=',' read -ra VACUOUS <<< "$VACUOUS_CSV"
is_vacuous_plane() {
    local plane="$1"
    local vp
    for vp in "${VACUOUS[@]}"; do
        [[ -n "$vp" && "$vp" -eq "$plane" ]] && return 0
    done
    return 1
}
# Step 1: For occupied planes with r_p >= 1, apply fault plans per replica.
# Vacuous planes remain clean regardless of r_p.
for p in $(seq 0 7); do
    rp="${RVEC[$p]}"
    if [[ "$rp" -le 0 ]] || is_vacuous_plane "$p"; then
        continue
    fi
    for j in $(seq 0 $((rp - 1))); do
        FP="$FAULT_BASE/$POLICY/plane${p}/${RATE_DIR}/replica${j}/seed_${SEED}.json"
        if [[ ! -f "$FP" ]]; then
            echo "MISSING: $FP" >&2
            exit 127
        fi
        python3 "$TMPDIR/apply_fault_plan.py" \
            "$ART_DIR/plane_${p}.bin" \
            "$FP" \
            "$REPLICA_DIR/plane_${p}_replica_${j}.bin"
    done
done
# Step 2: Run voting (produces voted planes + outcomes CSV in one shot)
python3 "$TMPDIR/phase2_vote_and_save.py" \
    "$REPLICA_DIR" \
    "$ART_DIR" \
    "$N" \
    "$VOTED_DIR" \
    "$CFG_DIR/vote_outcomes.csv"
# Step 3: Compute expected voted sum via analytic oracle
ORACLE_CSV="$CFG_DIR/oracle_output.csv"
python3 "$REPO_ROOT/scripts/phase2_oracle.py" \
    --artifact-dir "$ART_DIR" \
    --fault-plan-dir "$FAULT_BASE" \
    --r-vector $(tr ',' ' ' <<< "$R_VEC_CSV") \
    --dataset "$DATASET" \
    --n-rows "$N" \
    --scale "$SCALE" \
    --fault-rate "$RATE" \
    --base-seed "$SEED" \
    --policy-id "$POLICY" \
    --budget-B "$BUDGET_B" \
    --sensitivity-profile "$REPO_ROOT/results/phase2_sensitivity_profile.json" \
    --strategy-id "$STRATEGY_ID" \
    --strategy-scale "$SCALE" \
    --csv "$ORACLE_CSV" 2>&1
# Parse expected_voted_sum from oracle CSV (column 11)
EXPECTED_SUM=$(tail -1 "$ORACLE_CSV" | awk -F',' '{print $11}' | tr -d ' ')
echo "Oracle expected_voted_sum=$EXPECTED_SUM"
# Step 4: Run GPU SUM on voted planes with oracle verification
ROW_CSV="$CFG_DIR/row.csv"
ALLOCATION_R_PIPE="$(tr ',' '|' <<< "$R_VEC_CSV")"
if ! "$PHASE2_BIN" \
    --device 0 --n "$N" \
    --artifact-dir "$ART_DIR" \
    --voted-dir "$VOTED_DIR" \
    --vote-outcomes "$CFG_DIR/vote_outcomes.csv" \
    --csv "$ROW_CSV" \
    --policy-id "$POLICY" \
    --budget-B "$BUDGET_B" \
    --fault-rate "$RATE" \
    --base-seed "$SEED" \
    --strategy-id "$STRATEGY_ID" \
    --strategy-scale "$SCALE" \
    --allocation-r "$ALLOCATION_R_PIPE" \
    --expected-sum "$EXPECTED_SUM" \
    2>&1; then
    echo "BINARY FAIL: $CFG_TAG rc=$?" >&2
    exit 2
fi
# Step 4: Verify vote outcome sanity
python3 -c "
import csv, sys
path = '$CFG_DIR/vote_outcomes.csv'
n_rows = int('$N')
with open(path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = int(row['plane_id'])
        resolved = int(row['resolved_correctly'])
        detected = int(row['detected_mismatch'])
        undetected = int(row['undetected_corruption'])
        total = resolved + detected + undetected
        if total != n_rows:
            print(f'WARN: plane {pid} vote total {total} != n_rows {n_rows}', file=sys.stderr)
    print('vote outcome sanity OK')
"
# Emit data row with appended strategy columns (written to CFG_DIR/data_row.txt)
if [[ -s "$ROW_CSV" ]]; then
    DATA_LINE=$(tail -1 "$ROW_CSV")
    printf "%s\n" "$DATA_LINE" > "$CFG_DIR/data_row.txt"
fi
CONFIGEOF
chmod +x "$TMPDIR/run_one_config.sh"
# ── Extract catalogue entries ──
python3 -c "
import json, sys
cat = json.load(open('$CATALOGUE'))
budget_points = [int(x) for x in '$BUDGET_POINTS'.split()]
selected_rates = {f'{float(x):.0e}' for x in '$FAULT_RATES'.split()}
policy_filter = '$POLICY_ID'
ds_entries = [e for e in cat['entries']
              if e['dataset'] == '$DATASET'
              and e['budget_B'] in budget_points
              and format(float(e['fault_rate']), '.0e') in selected_rates
              and (not policy_filter or e['policy'] == policy_filter)]
ds_entries.sort(key=lambda e: (e['policy'], e['budget_B'], float(e['fault_rate'])))
for e in ds_entries:
    rvec_csv = ','.join(str(x) for x in e['r_vector'])
    vacuous_csv = ','.join(str(x) for x in e.get('vacuous_planes', []))
    rate = format(float(e['fault_rate']), '.0e')
    print(f'{e[\"policy\"]}|{e[\"budget_B\"]}|{rate}|{rvec_csv}|{vacuous_csv}')
" > "$TMPDIR/catalogue_entries.txt"
# ── Compute expected rows (only configs where at least one plane has r_p > 1) ──
EXPECTED_CONFIGS=$(python3 -c "
import json, sys
cat = json.load(open('$CATALOGUE'))
budget_points = [int(x) for x in '$BUDGET_POINTS'.split()]
policy_filter = '$POLICY_ID'
selected_rates = {f'{float(x):.0e}' for x in '$FAULT_RATES'.split()}
seeds = list(range($SEED_COUNT))
total = 0
has_entries = False
for e in cat['entries']:
    if e['dataset'] != '$DATASET': continue
    if e['budget_B'] not in budget_points: continue
    if format(float(e['fault_rate']), '.0e') not in selected_rates: continue
    if policy_filter and e['policy'] != policy_filter: continue
    has_entries = True
    if any(rp > 1 for rp in e['r_vector']):
        total += len(seeds)
if not has_entries:
    print(f'ERROR: no catalogue entries for dataset=$DATASET budget_points=$BUDGET_POINTS policy=$POLICY_ID', file=sys.stderr)
    sys.exit(2)
print(total)
")
echo "Expected configs (with replication): $EXPECTED_CONFIGS"
echo "[$(date)] Full matrix..."
ALL_CSV="$TMPDIR/phase2_matrix.csv"
FAIL_CSV="$TMPDIR/case_failures.csv"
FAIL_LOG_DIR="$TMPDIR/failure_logs"
mkdir -p "$FAIL_LOG_DIR"
echo "policy,budget_B,fault_rate,base_seed,exit_code,reason,r_vector" > "$FAIL_CSV"
FIRST=1
TOTAL_CONFIGS=0
ATTEMPTED_CONFIGS=0
MISSING_CONFIGS=0
BINARY_FAILURES=0
while IFS='|' read -r policy budget_B entry_rate rvec_csv vacuous_csv; do
  echo "[$(date)] Policy=$policy B=$budget_B rate=$entry_rate r_vec=[$rvec_csv]"
  # Pre-generate fault plans for this (dataset, policy, budget_B, rate, r_vector)
  REPLICAS_ARG=$(echo "$rvec_csv" | tr ',' ' ')
  python3 "$REPO_ROOT/scripts/generate_phase2_fault_plans.py" \
    --dataset "$DATASET" \
    --n-rows "$N" \
    --artifact-json "$ART_DIR/artifact.json" \
    --scale "$SCALE" \
    --rates "$entry_rate" \
    --seeds $(echo $SEEDS | awk '{print $1"-"$NF}') \
    --replicas $REPLICAS_ARG \
    --policy-id "$policy" \
    --artifact-root "$ART_ROOT" 2>&1 | tail -1
  # Check if at least one plane has r_p > 1
  HAS_REPLICATION=0
  for rp in $REPLICAS_ARG; do
    if [[ "$rp" -gt 1 ]]; then HAS_REPLICATION=1; break; fi
  done
  RATE="$entry_rate"
  RATE_DIR="rate$(LC_NUMERIC=C printf "%.0e" "$RATE")"
  for SEED in $SEEDS; do
      TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))
      if [[ "$HAS_REPLICATION" -eq 0 ]]; then
        echo "  SKIP (all r_p <= 1): $policy B=$budget_B rate=$RATE seed=$SEED"
        continue
      fi
      # Verify at least one fault plan exists (use rvec_csv for rp)
      PLANE_FOUND=0
      for p in $(seq 0 7); do
        rp=$(echo "$rvec_csv" | cut -d',' -f$((p+1)))
        [[ "$rp" -le 1 ]] && continue
        FP="$FAULT_BASE/$policy/plane${p}/${RATE_DIR}/replica0/seed_${SEED}.json"
        if [[ -f "$FP" ]]; then PLANE_FOUND=1; break; fi
      done
      if [[ "$PLANE_FOUND" -eq 0 ]]; then
        echo "MISSING fault plans: $DATASET policy=$policy B=$budget_B rate=$RATE seed=$SEED"
        echo "$policy,$budget_B,$RATE,$SEED,127,missing_fault_plan,$rvec_csv" >> "$FAIL_CSV"
        MISSING_CONFIGS=$((MISSING_CONFIGS + 1))
        continue
      fi
      ATTEMPTED_CONFIGS=$((ATTEMPTED_CONFIGS + 1))
      CFG_DIR="$TMPDIR/configs/${policy}_B${budget_B}_r${RATE}_s${SEED}"
      mkdir -p "$CFG_DIR"
      set +e
      "$TMPDIR/run_one_config.sh" \
        "$CFG_DIR" "$policy" "$budget_B" "$rvec_csv" "$RATE" "$SEED" \
        "$TMPDIR" "$ART_DIR" "$FAULT_BASE" "$PHASE2_BIN" "$N" \
        "$STRATEGY_ID" "$SCALE" "$REPO_ROOT" "$DATASET" "$vacuous_csv" \
        >"$FAIL_LOG_DIR/${policy}_B${budget_B}_r${RATE}_s${SEED}.log" 2>&1
      RC=$?
      set -e
      DATA_ROW="$CFG_DIR/data_row.txt"
      if [[ -s "$DATA_ROW" && "$RC" -eq 0 ]]; then
        if [[ "$FIRST" -eq 1 ]]; then
          head -1 "$CFG_DIR/row.csv" > "$ALL_CSV"
          FIRST=0
        fi
        cat "$DATA_ROW" >> "$ALL_CSV"
      else
        BINARY_FAILURES=$((BINARY_FAILURES + 1))
        echo "$policy,$budget_B,$RATE,$SEED,$RC,binary_failure,$rvec_csv" >> "$FAIL_CSV"
        FAIL_ART_DIR="$RESULTS_DIR/failure_artifacts/${policy}_B${budget_B}_r${RATE}_s${SEED}"
        mkdir -p "$FAIL_ART_DIR"
        cp "$FAIL_LOG_DIR/${policy}_B${budget_B}_r${RATE}_s${SEED}.log" "$FAIL_ART_DIR/stderr.log"
        for f in vote_outcomes.csv oracle_output.csv row.csv data_row.txt; do
          [[ -f "$CFG_DIR/$f" ]] && cp "$CFG_DIR/$f" "$FAIL_ART_DIR/$f"
        done
      fi
      # Keep per-config intermediates for post-failure inspection.
  done
  echo "  $DATASET policy=$policy B=$budget_B done"
done < "$TMPDIR/catalogue_entries.txt"
echo "[$(date)] Matrix complete: expected=$EXPECTED_CONFIGS attempted=$ATTEMPTED_CONFIGS"
ORACLE_COL=20
if [[ -f "$ALL_CSV" ]]; then
  MATCH=$(awk -F',' -v col="$ORACLE_COL" '{gsub(/^[ \t]+|[ \t]+$/, "", $col); if($col == "true") c++} END{print c+0}' "$ALL_CSV")
  ROWS=$(tail -n+2 "$ALL_CSV" | wc -l)
else
  MATCH=0
  ROWS=0
fi
FAIL_ROWS=$(tail -n+2 "$FAIL_CSV" | wc -l)
echo "Oracle match: $MATCH/$ROWS"
echo "Attempted configs: $ATTEMPTED_CONFIGS"
echo "Missing configs: $MISSING_CONFIGS"
echo "Binary failures: $BINARY_FAILURES"
echo "Failure rows: $FAIL_ROWS"
[[ -f "$ALL_CSV" ]] && cp "$ALL_CSV" "$RESULTS_DIR/canonical_matrix.csv"
cp "$FAIL_CSV" "$RESULTS_DIR/case_failures.csv"
{
  echo "strategy_id=$STRATEGY_ID"
  echo "dataset=$DATASET n=$N scale=$SCALE shift_k=$SHIFT_K"
  echo "scale_dir=$SCALE_DIR"
  echo "budget_points=$BUDGET_POINTS"
  echo "fault_rates=$FAULT_RATES seeds=$(echo $SEEDS | awk '{print $1"-"$NF}')"
  echo "expected_configs=$EXPECTED_CONFIGS"
  echo "attempted_configs=$ATTEMPTED_CONFIGS"
  echo "rows=$ROWS"
  echo "oracle_match=$MATCH/$ROWS"
  echo "missing_configs=$MISSING_CONFIGS"
  echo "binary_failures=$BINARY_FAILURES"
  echo "failure_rows=$FAIL_ROWS"
  echo "gpu=$GPU_NAME job=$JOB_ID"
  echo "platform=Nano5 H100"
} > "$RESULTS_DIR/run_meta.txt"
echo "Aggregation will be handled by phase2 aggregator (issue #130)"
if [[ "$ROWS" -ne "$EXPECTED_CONFIGS" ]]; then
  echo "MATRIX INCOMPLETE: expected $EXPECTED_CONFIGS rows, got $ROWS"
  exit 2
fi
if [[ "$MATCH" -ne "$ROWS" ]]; then
  echo "ORACLE MISMATCH: $MATCH/$ROWS"
  exit 2
fi
if [[ "$FAIL_ROWS" -ne 0 ]]; then
  echo "CASE FAILURES: $FAIL_ROWS"
  exit 2
fi
echo "ALL PASSED"
