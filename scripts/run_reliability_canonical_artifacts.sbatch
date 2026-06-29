#!/usr/bin/env bash
#SBATCH --job-name=rl_canonical_artifacts
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/logs/rl_canonical_artifacts_%j.log
#SBATCH --error=${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr/logs/rl_canonical_artifacts_%j.err
set -euo pipefail
ROOT_DIR="${PROJ_DIR}/workspace/gpu-byteplane-reliability-nmr"
cd "$ROOT_DIR"
JOB_ID="${SLURM_JOB_ID:-nojob}"
echo "Job $JOB_ID started on $(hostname)"
# Build synth_datasets
gcc -O3 -lm -o buff_encoder/synth_datasets buff_encoder/synth_datasets.c
# Check disk space
echo "Disk: $(df -h /work | tail -1 | awk '{print $4}') available"
# Generate all 4 datasets at N=1e8
python3 scripts/build_reliability_artifacts.py \
  --raw-data-root ${WORK_DIR}/datasets \
  --artifact-root ${WORK_DIR}/datasets/reliability_layer1 \
  --datasets sensor uniform heavy_tailed zipfian \
  --n-rows 100000000 \
  --scale 100 \
  --seed 20260413
echo "Artifacts done. Listing sizes:"
du -sh ${WORK_DIR}/datasets/reliability_layer1/artifacts/*/
# Now generate fault plans for the canonical matrix
# datasets=sensor,uniform,heavy_tailed,zipfian
# target_planes=0..7
# fault_rates=1e-8,1e-7,1e-6,1e-5,1e-4
# seeds=0..29
RELIABILITY_ARTIFACT_ROOT="${WORK_DIR}/datasets/reliability_layer1"
N=100000000
SCALE=100
SEED_BASE=20260413
echo "Generating fault plans..."
for DS in sensor uniform heavy_tailed zipfian; do
  for PLANE in 0 1 2 3 4 5 6 7; do
    for RATE in 1e-8 1e-7 1e-6 1e-5 1e-4; do
      RATE_DIR="rate$(LC_NUMERIC=C printf "%.0e" "$RATE")"
      for SEED in $(seq 0 29); do
        FP_DIR="$RELIABILITY_ARTIFACT_ROOT/fault_plans/$DS/n${N}/scale${SCALE}/plane${PLANE}/${RATE_DIR}"
        mkdir -p "$FP_DIR"
        FP_PATH="$FP_DIR/seed_${SEED}.json"
        if [[ -f "$FP_PATH" ]]; then
          continue
        fi
        python3 -c "
import json, os, random
rng = random.Random($SEED)
n = $N
fc = int($RATE * n)
if fc == 0:
    exit(0)
offsets = sorted(rng.sample(range(n), fc))
masks = [rng.randint(1, 255) for _ in range(fc)]
rel = 'fault_plans/$DS/n${N}/scale${SCALE}/plane${PLANE}/${RATE_DIR}/seed_${SEED}.json'
fp = {
    'metadata': {
        'dataset': '$DS', 'n_rows': n, 'scale': $SCALE,
        'artifact_id': 'artifacts/$DS/n${N}/scale${SCALE}',
        'fault_plan_id': rel,
        'target_plane': $PLANE,
        'fault_rate': '$RATE', 'fault_rate_numeric': $RATE,
        'seed': $SEED, 'actual_fault_count': fc,
        'plane_size_bytes': n,
    },
    'entries': [{'offset': o, 'mask': m} for o, m in zip(offsets, masks)],
}
with open('$FP_PATH', 'w') as f:
    json.dump(fp, f, indent=2)
"
      done
      echo "  $DS plane=$PLANE rate=$RATE done"
    done
  done
  echo " $DS done"
done
echo "ALL DONE: $(find $RELIABILITY_ARTIFACT_ROOT/fault_plans -name '*.json' | wc -l) fault plans"
echo "Disk: $(df -h /work | tail -1 | awk '{print $4}') available"
# ── Write inventory ──
python3 -c "
import json
from pathlib import Path
root = Path('$RELIABILITY_ARTIFACT_ROOT')
inv = {'generated': [], 'skipped': [], 'invalid': [], 'exploratory': [], 'structurally_sparse': {'artifacts': [], 'planes': []}}
for art in sorted((root / 'artifacts').rglob('artifact.json')):
    try:
        meta = json.loads(art.read_text())
        inv['generated'].append(str(art.relative_to(root)))
        nzf = meta.get('plane_nonzero_fraction', [])
        for p, frac in enumerate(nzf):
            if p in (0,1,2) and frac < 1e-6:
                inv['structurally_sparse']['planes'].append({'artifact': str(art.relative_to(root)), 'plane': p, 'nonzero_fraction': frac})
        if meta.get('overflow_count',0) > 0:
            inv['exploratory'].append({'path': str(art.relative_to(root)), 'reason': f'overflow={meta[\"overflow_count\"]}'})
    except Exception as e:
        inv['invalid'].append({'path': str(art.relative_to(root)), 'error': str(e)})
for fp in sorted((root / 'fault_plans').rglob('*.json')):
    try:
        data = json.loads(fp.read_text())
        meta = data.get('metadata', {})
        inv['generated'].append(meta.get('fault_plan_id', str(fp.relative_to(root))))
    except Exception as e:
        inv['invalid'].append({'path': str(fp.relative_to(root)), 'error': str(e)})
with open(str(root / 'inventory.json'), 'w') as f:
    json.dump(inv, f, indent=2)
print(f'Inventory: {root / \"inventory.json\"} ({len(inv[\"generated\"])} entries)')
"
