#!/usr/bin/env python3
"""Generate per-replica fault plans for Phase 2 reliability experiments.

Usage:
  python3 scripts/generate_phase2_fault_plans.py \
    --dataset sensor \
    --n-rows 100000000 \
    --artifact-json /path/to/artifact.json \
    --scale 8233095970213 \
    --rates 1e-7 1e-6 1e-5 \
    --seeds 0-14 \
    --replicas 4 3 3 3 3 3 3 3 \
    --artifact-root ${WORK_DIR}/datasets/reliability_layer1
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


def derive_replica_seed(base_seed, dataset, plane, fault_rate, replica_index):
    """Derive deterministic replica seed from base parameters.

    replica_seed = SHA-256(base_seed:dataset:plane:fault_rate:replica_index)
    Returns an integer in [0, 2^64).
    """
    key = f"{base_seed}:{dataset}:{plane}:{fault_rate}:{replica_index}"
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:16], 16)


def generate_fault_entries(n_rows, rate_val, seed):
    """Generate fault entries deterministically.

    Returns list of {'offset': int, 'mask': int} dicts.
    Reuses Phase 1 fault model: random-byte XOR, unique sorted offsets,
    masks in [1, 255], fault_count = floor(rate_val * n_rows).
    """
    rng = random.Random(seed)
    fault_count = int(rate_val * n_rows)
    if fault_count == 0:
        return []
    offsets = sorted(rng.sample(range(n_rows), fault_count))
    masks = [rng.randint(1, 255) for _ in range(fault_count)]
    return [{'offset': o, 'mask': m} for o, m in zip(offsets, masks)]


def make_rate_dir(rate_str):
    """Build rate directory name (e.g. '1e-6' -> 'rate1e-06')."""
    rate_val = float(rate_str)
    return f"rate{rate_val:.0e}"


def parse_seed_spec(seeds):
    """Parse seed list accepting ranges like '0-14' or single values."""
    result = []
    for s in seeds:
        if '-' in s:
            parts = s.split('-', 1)
            start, end = int(parts[0]), int(parts[1])
            result.extend(range(start, end + 1))
        else:
            result.append(int(s))
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Generate per-replica fault plans for Phase 2'
    )
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (e.g. sensor)')
    parser.add_argument('--n-rows', type=int, default=100000000)
    parser.add_argument('--artifact-json', type=Path, default=None,
                        help='Path to artifact metadata JSON for validation')
    parser.add_argument('--scale', type=int, required=True,
                        help='Scale value for directory naming')
    parser.add_argument('--rates', nargs='+',
                        default=['1e-7', '1e-6', '1e-5'])
    parser.add_argument('--seeds', nargs='+',
                        default=['0-14'])
    parser.add_argument('--replicas', type=int, nargs=8, required=True,
                        help='r_p for each of 8 planes (e.g. 4 3 3 3 3 3 3 3)')
    parser.add_argument('--artifact-root', type=Path, default=None,
                        help='Override RELIABILITY_ARTIFACT_ROOT')
    parser.add_argument('--policy-id', default='placeholder',
                        help='Policy identifier (placeholder; assigned at runtime)')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if files exist')
    args = parser.parse_args()

    roots = args.artifact_root or Path(
        os.environ.get(
            'RELIABILITY_ARTIFACT_ROOT',
            '${WORK_DIR}/datasets/reliability_layer1'
        )
    )

    plane_size_bytes = args.n_rows
    seeds = parse_seed_spec(args.seeds)

    generated = 0
    skipped_exists = 0
    skipped_no_replication = 0
    skipped_zero_fault = 0

    for plane in range(8):
        r_p = args.replicas[plane]
        if r_p == 0:
            skipped_no_replication += 1
            continue

        for rate_str in args.rates:
            rate_val = float(rate_str)
            rate_dir = make_rate_dir(rate_str)
            fault_count = int(rate_val * plane_size_bytes)

            if fault_count == 0:
                skipped_zero_fault += 1
                continue

            for seed in seeds:
                for replica_j in range(r_p):
                    replica_seed = derive_replica_seed(
                        seed, args.dataset, plane, rate_str, replica_j
                    )

                    out_dir = (
                        roots / 'fault_plans_phase2'
                        / args.dataset
                        / f'n{args.n_rows}'
                        / f'scale{args.scale}'
                        / args.policy_id
                        / f'plane{plane}'
                        / rate_dir
                        / f'replica{replica_j}'
                    )
                    out_path = out_dir / f'seed_{seed}.json'

                    if out_path.exists() and not args.force:
                        skipped_exists += 1
                        continue

                    entries = generate_fault_entries(
                        plane_size_bytes, rate_val, replica_seed
                    )

                    rel = (
                        f'fault_plans_phase2/{args.dataset}'
                        f'/n{args.n_rows}/scale{args.scale}'
                        f'/{args.policy_id}'
                        f'/plane{plane}/{rate_dir}'
                        f'/replica{replica_j}/seed_{seed}.json'
                    )

                    fp = {
                        'metadata': {
                            'dataset': args.dataset,
                            'n_rows': args.n_rows,
                            'scale': args.scale,
                            'policy_id': args.policy_id,
                            'fault_plan_id': rel,
                            'target_plane': plane,
                            'strategy_id': 'per_dataset',
                            'phase': 'phase2',
                            'replica_index': replica_j,
                            'replica_seed': replica_seed,
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
        f"Skipped (no-replication r_p<=1): {skipped_no_replication}, "
        f"Skipped (zero fault): {skipped_zero_fault}"
    )


if __name__ == '__main__':
    main()
