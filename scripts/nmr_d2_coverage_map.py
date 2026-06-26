#!/usr/bin/env python3
"""NMR-D2 coverage map evaluator with inline fault injection.

Injects faults programmatically (same model as NMR-R3), no external fault
plan files needed.  Supported fault modes:
  - plane_uniform: each plane independently gets faults at the given rate
  - same_fault_all: all replicas receive the same fault (negative control)
  - workload_weighted: fault probability weighted by plane significance

coverage_relation values:
  - "repaired": r=3, majority vote recovers clean byte
  - "detected":  r=3, majority vote != clean (degraded)
  - "unprotected": r=1 AND a fault is present
  - "clean": no fault in this segment-plane cell

Fault model: single-replica logical fault injection (replica[0] corrupted).
This is a software/logical independent-fault model.  It does NOT represent
physical HBM channel/bank/SID correlated faults.

Usage:
  python3 scripts/nmr_d2_coverage_map.py \\
      --protection-map /tmp/pmap.json \\
      --clean-plane-dir /path/to/planes \\
      --dataset sensor --n-rows 10000 --scale 100 \\
      --fault-rate 1e-06 --seed 0 \\
      --output /tmp/coverage_map.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SEGMENT_SIZE = 1024
PLANES = 8
PLANE_WEIGHTS = [1 << (8 * (7 - p)) for p in range(PLANES)]


def load_protection_map(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_clean_plane(plane_dir: Path, plane: int, n_rows: int) -> bytes:
    """Read plane file, return first n_rows bytes.

    Contract: missing plane files (e.g. datasets with max_plane_count < 8)
    return ``bytes(n_rows)`` (all-zero).  The D2 protection map still assigns
    r=1 to these planes, so any fault injected into an all-zero segment will
    be detected as "unprotected" — zero data does not suppress fault coverage.
    """
    for fmt in (f"plane_{plane}.bin", f"plane_{plane:03d}.bin"):
        path = plane_dir / fmt
        if path.is_file():
            data = path.read_bytes()
            if len(data) >= n_rows:
                return data[:n_rows]
            data = data + bytes(n_rows - len(data))
            return data
    return bytes(n_rows)


def vote_byte(replica_values: list[int]) -> int:
    cnt = Counter(replica_values)
    max_count = max(cnt.values())
    r = len(replica_values)
    if max_count > r / 2:
        return next(v for v, c in cnt.items() if c == max_count)
    tied = sorted(v for v, c in cnt.items() if c == max_count)
    return tied[0]


def inject_plane_faults(
    plane_bytes: bytes,
    n_rows: int,
    fault_rate: float,
    rng: random.Random,
    mode: str = "plane_uniform",
    plane: int = 0,
) -> list[int]:
    """Return list of absolute byte positions to fault in replica 0.

    mode="plane_uniform": n_faults = round(N * rate) at random positions.
    mode="same_fault_all": single random position (used for negative control).
    mode="workload_weighted": rate weighted by plane significance.
    """
    if mode == "same_fault_all":
        return [rng.randint(0, n_rows - 1)]

    if mode == "workload_weighted":
        weight_ratio = PLANE_WEIGHTS[plane] / sum(PLANE_WEIGHTS[:8])
        effective_rate = fault_rate * (1.0 + 7.0 * weight_ratio)
    else:
        effective_rate = fault_rate

    if effective_rate >= 1.0 / n_rows:
        n_faults = max(1, int(n_rows * effective_rate + 0.5))
    else:
        n_faults = int(n_rows * effective_rate + 0.5)

    if n_faults == 0:
        return []

    return sorted(rng.sample(range(n_rows), min(n_faults, n_rows)))


def evaluate_segment_coverage(
    clean: bytes,
    replicas: list[bytearray],
    seg_start: int,
    seg_end: int,
) -> str:
    """Return coverage_relation for one (plane, segment).

    D2 only supports r=1 (no repair) or r=3 (majority vote).
    """
    r_p = len(replicas)
    if r_p not in (1, 3):
        raise ValueError(f"D2 only supports r=1 or r=3, got r={r_p}")

    if r_p == 1:
        for rb in replicas:
            for i in range(seg_start, seg_end):
                if rb[i] != clean[i]:
                    return "unprotected"
        return "clean"

    crc_hit = False
    for rb in replicas:
        for i in range(seg_start, seg_end):
            if rb[i] != clean[i]:
                crc_hit = True
                break
        if crc_hit:
            break

    if not crc_hit:
        return "clean"

    for i in range(seg_start, seg_end):
        values = [rb[i] for rb in replicas]
        vb = vote_byte(values)
        if vb != clean[i]:
            return "detected"

    return "repaired"


def compute_coverage_map(
    protection_map: dict[str, Any],
    clean_plane_dir: Path,
    dataset: str,
    n_rows: int,
    fault_rate: float,
    base_seed: int,
    mode: str = "plane_uniform",
) -> list[dict[str, Any]]:
    pmap = protection_map.get("map", {})
    policy = protection_map.get("policy", "unknown")
    segment_size = protection_map.get("segment_size", SEGMENT_SIZE)
    n_segments = (n_rows + segment_size - 1) // segment_size

    clean_planes = [load_clean_plane(clean_plane_dir, p, n_rows) for p in range(PLANES)]

    rng = random.Random(base_seed)
    rows: list[dict] = []

    # Per-plane fault positions (computed once, shared across segments)
    plane_fault_positions: dict[int, list[int]] = {}
    for p in range(PLANES):
        plane_fault_positions[p] = inject_plane_faults(
            clean_planes[p], n_rows, fault_rate, rng, mode, p,
        )

    for p in range(PLANES):
        clean = clean_planes[p]
        fault_positions = plane_fault_positions[p]

        # Determine max r_val for this plane to create replicas once
        max_r = 1
        for seg_idx in range(n_segments):
            rv = pmap.get(f"{p}_{seg_idx}", 1)
            if rv > max_r:
                max_r = rv

        # Create replicas once per plane (not per segment)
        replicas = [bytearray(clean) for _ in range(max_r)]

        # Apply all fault positions to replica[0] (and all for same_fault_all)
        for pos in fault_positions:
            mask = rng.randint(1, 255)
            if mode == "same_fault_all":
                for rep in replicas:
                    rep[pos] ^= mask
            else:
                replicas[0][pos] ^= mask

        for seg_idx in range(n_segments):
            start = seg_idx * segment_size
            end = min(start + segment_size, n_rows)
            key = f"{p}_{seg_idx}"
            r_val = pmap.get(key, 1)

            # Use only first r_val replicas for this segment's evaluation
            seg_replicas = replicas[:r_val]
            cr = evaluate_segment_coverage(clean, seg_replicas, start, end)

            rows.append({
                "dataset": dataset,
                "policy": policy,
                "plane": p,
                "segment": seg_idx,
                "n_rows": n_rows,
                "fault_rate": str(fault_rate),
                "seed": base_seed,
                "r_value": r_val,
                "n_replicas": len(replicas),
                "coverage_relation": cr,
                "segment_start": start,
                "segment_end": end,
            })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protection-map", type=Path, required=True)
    parser.add_argument("--clean-plane-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="tiny_fixture")
    parser.add_argument("--n-rows", type=int, required=True)
    parser.add_argument("--fault-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="plane_uniform",
                        choices=["plane_uniform", "same_fault_all",
                                 "workload_weighted"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pmap = load_protection_map(args.protection_map)
    rows = compute_coverage_map(
        pmap, args.clean_plane_dir, args.dataset,
        args.n_rows, args.fault_rate, args.seed, args.mode,
    )

    fieldnames = [
        "dataset", "policy", "plane", "segment", "n_rows",
        "fault_rate", "seed", "r_value", "n_replicas",
        "coverage_relation", "segment_start", "segment_end",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"coverage_map: {args.output} ({len(rows)} rows)")

    cr_counts = Counter(r["coverage_relation"] for r in rows)
    print(f"coverage_relation distribution: {dict(cr_counts)}")


if __name__ == "__main__":
    main()
