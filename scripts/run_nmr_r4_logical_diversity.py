#!/usr/bin/env python3
"""NMR-R4: Logical diversity under structured clustered fault injection.

Compares replica placement strategies under clustered fault models:
  - Placement: no-diversity, logical-spatial, logical-spatial+logical-temporal
  - Fault models: independent, regional_clustered, burst_window, combined

Metrics:
  - all_replicas_hit_rate (primary): fraction of (plane,segment) where all r=3 are hit
  - recoverable_rate: majority vote recovers clean value given at least one fault
  - unrecoverable_same_domain_rate: all-replicas-hit AND majority vote fails

This is a LOGICAL/SOFTWARE-ONLY fault injection experiment.  It does NOT
validate physical HBM channel/bank/SID fault domains.

Usage:
  python3 scripts/run_nmr_r4_logical_diversity.py \\
      --plane-dir \$LOCALITY_DATA_ROOT/hurricane_u/seg4096 \\
      --dataset hurricane_u --n-rows 25000000 \\
      --fault-rates 2e-6 2e-5 2e-4 \\
      --seeds 0 1 2 \\
      --output results/reliability_layer1/phase4/nmr_r4_logical_diversity/smoke.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import nmr_d2_protection_map as pmap_gen

PLANES = 8
SEGMENT_SIZE = 4096
R = 3

PLACEMENT_STRATEGIES = [
    "no-diversity",
    "logical-spatial",
    "logical-spatial+logical-temporal",
]

FAULT_MODELS = [
    "independent",
    "regional_clustered",
    "burst_window",
    "combined",
]

# Each fault model's cluster lengths (bytes).  independent = len 1.
# fault_rate controls number of CLUSTER EVENTS, not byte corruption rate.
# Each event corrupts cluster_len consecutive bytes.
CLUSTER_LENS: dict[str, list[int]] = {
    "independent": [1],
    "regional_clustered": [64],
    "burst_window": [256],
    "combined": [64, 256],
}


def load_clean_plane(plane_dir: Path, plane: int, n_rows: int) -> bytes:
    for fmt in (f"plane_{plane}.bin", f"plane_{plane:03d}.bin"):
        path = plane_dir / fmt
        if path.is_file():
            data = path.read_bytes()
            if len(data) >= n_rows:
                return data[:n_rows]
            return data + bytes(n_rows - len(data))
    return bytes(n_rows)


def sample_events(
    n_rows: int,
    event_rate: float,
    rng: random.Random,
) -> list[int]:
    """Sample cluster-START positions uniformly from [0, n_rows).

    Always returns >= 1 event when event_rate > 0 and n_rows * event_rate >= 0.5,
    and at least 1 event whenever event_rate > 0 (floor of 0).
    """
    if event_rate <= 0.0:
        return []
    raw = n_rows * event_rate
    n_events = max(1, int(raw + 0.5))
    return sorted(rng.sample(range(n_rows), min(n_events, n_rows)))


def expand_cluster_positions(
    starts: list[int], cluster_len: int, n_rows: int
) -> set[int]:
    """Expand cluster starts into individual byte positions."""
    result: set[int] = set()
    for cs in starts:
        end = min(cs + cluster_len, n_rows)
        for pos in range(cs, end):
            result.add(pos)
    return result


def generate_fault_positions(
    n_rows: int,
    fault_rate: float,
    fault_model: str,
    placement_strategy: str,
    rng: random.Random,
) -> list[list[int]]:
    """Return per-replica fault-position lists.

    fault_rate controls the expected number of CLUSTER EVENTS per byte.
    Each event corrupts cluster_len bytes (1 for independent, 64/256 for clustered).

    Strategy-dependent correlation:
      - no-diversity: ONE set shared across all replicas
      - logical-spatial: INDEPENDENT sets per replica
      - logical-spatial+logical-temporal: INDEPENDENT per replica + temporal masking
    """
    cluster_lens = CLUSTER_LENS[fault_model]
    n_models = len(cluster_lens)

    all_rep_positions: list[set[int]] = [set() for _ in range(R)]

    for model_idx, cluster_len in enumerate(cluster_lens):
        rate_share = fault_rate / n_models

        if placement_strategy == "no-diversity":
            starts = sample_events(
                n_rows, rate_share, random.Random(rng.randint(0, 2**31 - 1))
            )
            positions = expand_cluster_positions(starts, cluster_len, n_rows)
            for rep_idx in range(R):
                all_rep_positions[rep_idx] |= positions
        else:
            for rep_idx in range(R):
                rep_rng = random.Random(rng.randint(0, 2**31 - 1))
                starts = sample_events(n_rows, rate_share, rep_rng)
                all_rep_positions[rep_idx] |= expand_cluster_positions(
                    starts, cluster_len, n_rows
                )

    # Convert sets to unsorted lists (apply_faults doesn't need sorting)
    result = [list(s) for s in all_rep_positions]

    if placement_strategy == "logical-spatial+logical-temporal":
        temporal_rng = random.Random(rng.randint(0, 2**31 - 1))
        for rep_idx in range(R):
            result[rep_idx] = [
                p for p in result[rep_idx] if temporal_rng.random() < 0.5
            ]

    return result


def apply_faults(
    replicas: list[bytearray],
    fault_positions: list[list[int]],
    rng: random.Random,
) -> None:
    for rep_idx in range(R):
        rep_rng = random.Random(rng.randint(0, 2**31 - 1))
        for pos in fault_positions[rep_idx]:
            mask = rep_rng.randint(1, 255)
            replicas[rep_idx][pos] ^= mask


def evaluate_segment(
    clean: bytes,
    replicas: list[bytearray],
    seg_start: int,
    seg_end: int,
) -> tuple[str, int]:
    clean_slice = memoryview(clean)[seg_start:seg_end]

    replicas_hit = 0
    for rep in replicas:
        rep_slice = memoryview(rep)[seg_start:seg_end]
        if rep_slice != clean_slice:
            replicas_hit += 1

    if replicas_hit == 0:
        return "clean", 0

    for i in range(seg_start, seg_end):
        cv = clean[i]
        v0, v1, v2 = replicas[0][i], replicas[1][i], replicas[2][i]
        if v0 != cv or v1 != cv or v2 != cv:
            count_clean = (v0 == cv) + (v1 == cv) + (v2 == cv)
            if count_clean <= 1:
                return "detected", replicas_hit

    return "repaired", replicas_hit


def run_one_cell(
    clean_planes: list[bytes],
    dataset: str,
    n_rows: int,
    pmap: dict[str, Any],
    fault_rate: float,
    base_seed: int,
    placement_strategy: str,
    fault_model: str,
    replicas: list[bytearray] | None = None,
) -> list[dict[str, Any]]:
    pmap_data = pmap.get("map", {})
    segment_size = pmap.get("segment_size", SEGMENT_SIZE)
    n_segments = (n_rows + segment_size - 1) // segment_size

    if replicas is None:
        replicas = [bytearray(n_rows) for _ in range(R)]

    rows: list[dict] = []

    for p in range(PLANES):
        clean = clean_planes[p]
        clean_slice = memoryview(clean)

        for rep_idx in range(R):
            replicas[rep_idx][:] = clean_slice

        rng = random.Random(base_seed * 1000 + p)

        fault_positions = generate_fault_positions(
            n_rows, fault_rate, fault_model, placement_strategy, rng
        )

        apply_faults(replicas, fault_positions, rng)

        for seg_idx in range(n_segments):
            start = seg_idx * segment_size
            end = min(start + segment_size, n_rows)
            key = f"{p}_{seg_idx}"
            r_val = pmap_data.get(key, 1)

            if r_val < R:
                continue

            seg_replicas = replicas[:r_val]
            cr, replicas_hit = evaluate_segment(clean, seg_replicas, start, end)

            rows.append({
                "dataset": dataset,
                "policy": pmap.get("policy", "graded_seg_B3"),
                "placement_strategy": placement_strategy,
                "fault_model": fault_model,
                "plane": p,
                "segment": seg_idx,
                "n_rows": n_rows,
                "fault_rate": str(fault_rate),
                "seed": base_seed,
                "r_value": r_val,
                "replicas_hit": replicas_hit,
                "coverage_relation": cr,
                "segment_start": start,
                "segment_end": end,
            })

    return rows


def compute_metrics(rows: list[dict]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {}

    cr_counts = Counter(r["coverage_relation"] for r in rows)
    hit_counts = Counter(r["replicas_hit"] for r in rows)

    total_faulty = cr_counts.get("repaired", 0) + cr_counts.get("detected", 0)
    total_all_hit = hit_counts.get(R, 0)
    total_all_hit_unrec = sum(
        1 for r in rows if r["replicas_hit"] == R and r["coverage_relation"] == "detected"
    )

    return {
        "all_replicas_hit_rate": hit_counts.get(R, 0) / n if n > 0 else 0.0,
        "recoverable_rate": cr_counts.get("repaired", 0) / total_faulty if total_faulty > 0 else 0.0,
        "unrecoverable_same_domain_rate": total_all_hit_unrec / total_all_hit if total_all_hit > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n-rows", type=int, required=True)
    parser.add_argument("--fault-rates", type=float, nargs="+",
                        default=[2e-6, 2e-5, 2e-4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protection-map", type=Path, default=None)
    parser.add_argument("--segment-size", type=int, default=SEGMENT_SIZE)
    args = parser.parse_args()

    n_segments = (args.n_rows + args.segment_size - 1) // args.segment_size

    if args.protection_map and Path(args.protection_map).is_file():
        pmap = json.loads(Path(args.protection_map).read_text())
    else:
        pmap = pmap_gen.make_graded_seg_B3(n_segments, 0, args.segment_size)

    clean_planes = [load_clean_plane(args.plane_dir, p, args.n_rows) for p in range(PLANES)]
    replicas = [bytearray(args.n_rows) for _ in range(R)]

    all_rows: list[dict] = []

    for strategy in PLACEMENT_STRATEGIES:
        for fault_model in FAULT_MODELS:
            for rate_val in args.fault_rates:
                for seed in args.seeds:
                    seg_rows = run_one_cell(
                        clean_planes,
                        args.dataset,
                        args.n_rows,
                        pmap,
                        rate_val,
                        seed,
                        strategy,
                        fault_model,
                        replicas,
                    )
                    metrics = compute_metrics(seg_rows)
                    summary = {
                        "dataset": args.dataset,
                        "policy": pmap.get("policy", "graded_seg_B3"),
                        "placement_strategy": strategy,
                        "fault_model": fault_model,
                        "fault_rate": str(rate_val),
                        "seed": seed,
                        "n_segments": n_segments,
                        "n_rows": args.n_rows,
                        **metrics,
                    }
                    all_rows.append(summary)

                    print(
                        f"{strategy:>35s} {fault_model:>20s} "
                        f"rate={rate_val:.0e} seed={seed}: "
                        f"all_hit={metrics['all_replicas_hit_rate']:.6f} "
                        f"recoverable={metrics['recoverable_rate']:.6f} "
                        f"unrec_same={metrics['unrecoverable_same_domain_rate']:.6f}",
                        flush=True,
                    )

    fieldnames = [
        "dataset", "policy", "placement_strategy", "fault_model",
        "fault_rate", "seed", "n_segments", "n_rows",
        "all_replicas_hit_rate", "recoverable_rate",
        "unrecoverable_same_domain_rate",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {args.output} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
