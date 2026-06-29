#!/usr/bin/env bash
#SBATCH -J nmr_d2_mapseed
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH --mail-type=END,FAIL
# Usage: DATASET=hurricane_u sbatch scripts/run_nmr_d2_mapseed_sweep.sbatch
#        DATASET=cesm_atm_cloud sbatch scripts/run_nmr_d2_mapseed_sweep.sbatch
DATASET="${DATASET:?must be hurricane_u or cesm_atm_cloud}"
echo "=== NMR-D2 Map-Seed Sweep ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  Dataset: $DATASET"
GPU_NAME="unknown"
nvidia-smi --query-gpu=name --format=csv,noheader > /dev/null 2>&1 && GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) || true
echo "GPU: $GPU_NAME"
date
case "$DATASET" in
cesm_atm_cloud) N_ROWS=168480000 ;;
hurricane_u) N_ROWS=25000000 ;;
*) echo "bad dataset"; exit 1 ;;
esac
ARTIFACT_DIR="${WORK_DIR}/datasets/locality_sensitivity/${DATASET}/seg4096"
SEGMENT_SIZE=4096
N_SEGMENTS=$(( (N_ROWS + SEGMENT_SIZE - 1) / SEGMENT_SIZE ))
ml purge 2>/dev/null || true
ml load miniconda3/26.1.1 cuda/12.6
source ${ENVSTACK_ROOT}/miniconda3/26.1.1/etc/profile.d/conda.sh
conda activate gpu-byteplane-scan
PROJ_ROOT=${PROJ_DIR}/workspace/gpu-byteplane-scan-experiments/.worktrees/WORKTREE
SCRIPT_DIR="$PROJ_ROOT/scripts"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
OUTDIR="$PROJ_ROOT/results/reliability_layer1/phase4/nmr_d2_mapseed_sweep/${DATASET}/job_${SLURM_JOB_ID}"
PMAP_DIR="$OUTDIR/protection_maps"
mkdir -p "$OUTDIR" "$PMAP_DIR"
MAP_SEEDS="0 1 2 3 4 5 6 7"
HEADLINE_RATES="2e-5 2e-4"
FAULT_SEEDS="0 1"
echo "N_ROWS=$N_ROWS N_SEGMENTS=$N_SEGMENTS MAP_SEEDS=$MAP_SEEDS"
echo "HEADLINE_RATES=$HEADLINE_RATES FAULT_SEEDS=$FAULT_SEEDS"
echo "OUTDIR=$OUTDIR"
for MAP_SEED in $MAP_SEEDS; do
    echo ""
    echo "=== Map seed $MAP_SEED ==="
    SEED_DIR="$OUTDIR/seed_${MAP_SEED}"
    mkdir -p "$SEED_DIR"
    # Generate protection maps for this seed
    for policy in graded_seg_B3 uniform_spread_seg_B3; do
        pmap_path="$PMAP_DIR/${policy}_seed${MAP_SEED}.json"
        if [ ! -f "$pmap_path" ]; then
            python3 "$SCRIPT_DIR/nmr_d2_protection_map.py" \
                --policy $policy --n-segments $N_SEGMENTS --seed $MAP_SEED \
                --segment-size $SEGMENT_SIZE --output "$pmap_path"
        fi
    done
    # Stochastic evaluation with this map seed
    python3 "$SCRIPT_DIR/nmr_d2_stochastic.py" \
        --protection-map-dir "$PMAP_DIR" \
        --clean-plane-dir "$ARTIFACT_DIR" \
        --dataset "$DATASET" \
        --n-rows "$N_ROWS" \
        --headline-rates $HEADLINE_RATES \
        --seeds $FAULT_SEEDS \
        --mode plane_uniform \
        --pmap-seed $MAP_SEED \
        --output "$SEED_DIR/nmr_d2_stochastic_matrix.csv"
    # Claim matrix
    python3 "$SCRIPT_DIR/nmr_d2_claim_matrix.py" \
        --stochastic-results "$SEED_DIR/nmr_d2_stochastic_matrix.csv" \
        --policies graded_seg_B3 uniform_spread_seg_B3 \
        --output-dir "$SEED_DIR"
done
# Aggregate across all map seeds
echo ""
echo "=== Aggregate across map seeds ==="
AGG_PATH="$OUTDIR/nmr_d2_mapseed_aggregate.csv"
python3 -c "
import csv, sys, json
from collections import defaultdict, Counter
from pathlib import Path
rows = []
for ms in $MAP_SEEDS:
    p = Path('$OUTDIR') / f'seed_{ms}' / 'nmr_d2_paired_delta_summary.csv'
    if p.is_file():
        with p.open() as f:
            for r in csv.DictReader(f):
                r['map_seed'] = str(ms)
                rows.append(r)
# Per-map-seed aggregate
by_seed = defaultdict(list)
for r in rows:
    by_seed[r['map_seed']].append(r)
for ms in $MAP_SEEDS:
    subset = by_seed.get(ms, [])
    if not subset:
        continue
    verdicts = Counter(r['claim_verdict'] for r in subset)
    relbw = [float(r['delta_relative_bound_width']) for r in subset]
    msb = [float(r['delta_msb_coverage_rate']) for r in subset]
    print(f'  seed {ms}: GRADED_WINS={verdicts.get(\"GRADED_WINS\", 0)}/{len(subset)}  '
          f'delta_relbw=({min(relbw):.4e}, {sum(relbw)/len(relbw):.4e}, {max(relbw):.4e})  '
          f'delta_msb={sum(msb)/len(msb):.4e}')
# Overall aggregate
verdicts_all = Counter(r['claim_verdict'] for r in rows)
relbw_all = [float(r['delta_relative_bound_width']) for r in rows]
print(f'\\nOverall: GRADED_WINS={verdicts_all.get(\"GRADED_WINS\", 0)}/{len(rows)}  '
      f'delta_relbw_median={sorted(relbw_all)[len(relbw_all)//2]:.4e}  '
      f'min={min(relbw_all):.4e}  max={max(relbw_all):.4e}')
print(f'Verdicts: {dict(verdicts_all)}')
# Write aggregate CSV
with open('$AGG_PATH', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['map_seed', 'n_cells', 'graded_wins', 'uniform_wins', 'tie',
                'delta_relbw_mean', 'delta_relbw_min', 'delta_relbw_max', 'delta_msb_mean'])
    for ms in $MAP_SEEDS:
        subset = by_seed.get(ms, [])
        if not subset:
            continue
        v = Counter(r['claim_verdict'] for r in subset)
        rb = [float(r['delta_relative_bound_width']) for r in subset]
        mb = [float(r['delta_msb_coverage_rate']) for r in subset]
        w.writerow([ms, len(subset), v.get('GRADED_WINS', 0), v.get('UNIFORM_WINS', 0),
                    v.get('TIE', 0), sum(rb)/len(rb), min(rb), max(rb), sum(mb)/len(mb)])
    w.writerow(['OVERALL', len(rows), verdicts_all.get('GRADED_WINS', 0),
                verdicts_all.get('UNIFORM_WINS', 0), verdicts_all.get('TIE', 0),
                sum(relbw_all)/len(relbw_all), min(relbw_all), max(relbw_all),
                sum([float(r['delta_msb_coverage_rate']) for r in rows])/len(rows)])
print(f'Aggregate: $AGG_PATH')
" 2>&1
echo ""
echo "=== Sweep Complete ==="
echo "Results: $OUTDIR"
date
