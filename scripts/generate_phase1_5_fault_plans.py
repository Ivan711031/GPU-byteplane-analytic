#!/usr/bin/env python3
"""Generate per-strategy fault plans for Phase 1.5 reliability experiments.

Usage:
  python3 scripts/generate_phase1_5_fault_plans.py \
    --strategy-id native_scale1 \
    --params-json results/phase1_5_strategy_params.json \
    --datasets sensor uniform heavy_tailed zipfian \
    --n-rows 100000000 \
    --rates 1e-8 1e-7 1e-6 1e-5 1e-4 \
    --seeds 0-29
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path


def make_scale_dir(scale, shift_k):
    """Build scale directory name from scale value and shift_k."""
    if shift_k > 0:
        return f"scale{scale}_shift{shift_k}"
    return f"scale{scale}"


def make_rate_dir(rate_str):
    """Build rate directory name (e.g. '1e-6' -> 'rate1e-06')."""
    rate_val = float(rate_str)
    return f"rate{rate_val:.0e}"


def parse_seed_spec(seeds):
    """Parse seed list accepting ranges like '0-29' or single values."""
    result = []
    for s in seeds:
        if '-' in s:
            parts = s.split('-', 1)
            start, end = int(parts[0]), int(parts[1])
            result.extend(range(start, end + 1))
        else:
            result.append(int(s))
    return result


def read_skip_csv(path):
    """Read skip CSV, return set of (strategy_id, dataset) tuples."""
    skipped = set()
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            skipped.add((row['strategy_id'].strip(), row['dataset'].strip()))
    return skipped


def generate_fault_entries(n_rows, rate_val, seed):
    """Generate fault entries deterministically.

    Returns list of {'offset': int, 'mask': int} dicts.
    """
    rng = random.Random(seed)
    fault_count = int(rate_val * n_rows)
    if fault_count == 0:
        return []
    offsets = sorted(rng.sample(range(n_rows), fault_count))
    masks = [rng.randint(1, 255) for _ in range(fault_count)]
    return [{'offset': o, 'mask': m} for o, m in zip(offsets, masks)]


def validate_artifact(artifact_path, strategy_id, dataset, n_rows):
    """Validate artifact JSON metadata matches expected values."""
    data = json.loads(Path(artifact_path).read_text())
    meta = data.get('metadata', {})
    errors = []
    if meta.get('strategy_id') != strategy_id:
        errors.append(
            f"strategy_id mismatch: expected {strategy_id}, "
            f"got {meta.get('strategy_id')}"
        )
    if meta.get('dataset') != dataset:
        errors.append(
            f"dataset mismatch: expected {dataset}, "
            f"got {meta.get('dataset')}"
        )
    if meta.get('n_rows') != n_rows:
        errors.append(
            f"n_rows mismatch: expected {n_rows}, "
            f"got {meta.get('n_rows')}"
        )
    if errors:
        raise ValueError('; '.join(errors))


def main():
    parser = argparse.ArgumentParser(
        description='Generate per-strategy fault plans for Phase 1.5'
    )
    parser.add_argument('--strategy-id', required=True,
                        help='Strategy identifier (e.g. native_scale1)')
    parser.add_argument('--artifact-json', type=Path, default=None,
                        help='Path to artifact metadata JSON for validation')
    parser.add_argument('--params-json', type=Path, required=True,
                        help='Path to phase1_5_strategy_params.json')
    parser.add_argument('--datasets', nargs='+',
                        default=['sensor', 'uniform', 'heavy_tailed', 'zipfian'])
    parser.add_argument('--n-rows', type=int, default=100000000)
    parser.add_argument('--rates', nargs='+',
                        default=['1e-8', '1e-7', '1e-6', '1e-5', '1e-4'])
    parser.add_argument('--seeds', nargs='+',
                        default=['0-29'])
    parser.add_argument('--skip-csv', type=Path, default=None,
                        help='Path to phase1_5_conversion_skipped.csv')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if files exist')
    args = parser.parse_args()

    roots = Path(
        os.environ.get(
            'RELIABILITY_ARTIFACT_ROOT',
            '/work/u4063895/datasets/reliability_layer1'
        )
    )

    params = json.loads(args.params_json.read_text())
    if args.strategy_id not in params['strategies']:
        print(
            f"ERROR: strategy_id '{args.strategy_id}' not found in params JSON",
            file=sys.stderr
        )
        sys.exit(1)

    strategy_cfg = params['strategies'][args.strategy_id]
    per_dataset_cfg = strategy_cfg['per_dataset']
    plane_size_bytes = args.n_rows

    seeds = parse_seed_spec(args.seeds)

    skip_cells = set()
    if args.skip_csv is not None:
        skip_cells = read_skip_csv(args.skip_csv)

    generated = 0
    skipped_exists = 0
    skipped_inapplicable = 0

    for dataset in args.datasets:
        if dataset not in per_dataset_cfg:
            print(
                f"WARNING: dataset '{dataset}' not in params "
                f"for strategy '{args.strategy_id}', skipping"
            )
            continue

        if (args.strategy_id, dataset) in skip_cells:
            skipped_inapplicable += 1
            print(f"SKIP (inapplicable): {args.strategy_id}/{dataset}")
            continue

        ds_cfg = per_dataset_cfg[dataset]
        scale_val = ds_cfg['scale']
        shift_k = ds_cfg['shift_k']
        scale_dir = make_scale_dir(scale_val, shift_k)

        if args.artifact_json is not None:
            validate_artifact(
                args.artifact_json, args.strategy_id, dataset, args.n_rows
            )

        for plane in range(8):
            for rate_str in args.rates:
                rate_val = float(rate_str)
                rate_dir = make_rate_dir(rate_str)
                fault_count = int(rate_val * plane_size_bytes)

                if fault_count == 0:
                    continue

                for seed in seeds:
                    out_dir = (
                        roots / 'fault_plans_phase1_5'
                        / args.strategy_id / dataset
                        / f'n{args.n_rows}' / scale_dir
                        / f'plane{plane}' / rate_dir
                    )
                    out_path = out_dir / f'seed_{seed}.json'

                    if out_path.exists() and not args.force:
                        skipped_exists += 1
                        continue

                    entries = generate_fault_entries(
                        plane_size_bytes, rate_val, seed
                    )

                    rel = (
                        f'fault_plans_phase1_5/{args.strategy_id}/{dataset}'
                        f'/n{args.n_rows}/{scale_dir}'
                        f'/plane{plane}/{rate_dir}/seed_{seed}.json'
                    )
                    art_rel = (
                        f'artifacts_phase1_5/{args.strategy_id}/{dataset}'
                        f'/n{args.n_rows}/{scale_dir}'
                    )

                    fp = {
                        'metadata': {
                            'dataset': dataset,
                            'n_rows': args.n_rows,
                            'scale': scale_val,
                            'artifact_id': art_rel,
                            'fault_plan_id': rel,
                            'target_plane': plane,
                            'strategy_id': args.strategy_id,
                            'fault_rate': rate_str,
                            'fault_rate_numeric': rate_val,
                            'seed': seed,
                            'actual_fault_count': fault_count,
                            'plane_size_bytes': plane_size_bytes,
                            'mask_distribution': '[1, 255] uniform',
                            'offset_order': 'sorted',
                            'offset_uniqueness': 'enforced',
                        },
                        'entries': entries,
                    }

                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(fp, indent=2) + '\n')
                    generated += 1

    print(
        f"Generated: {generated}, "
        f"Skipped (exists): {skipped_exists}, "
        f"Skipped (inapplicable): {skipped_inapplicable}"
    )


if __name__ == '__main__':
    main()
