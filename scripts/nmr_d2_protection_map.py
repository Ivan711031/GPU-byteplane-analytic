#!/usr/bin/env python3
"""NMR-D2 segment-level protection map generator.

Generates ProtectionMap: (plane, segment_idx) -> r_value (3 or 1)
for the two headline policies.

Usage:
  python3 scripts/nmr_d2_protection_map.py \\
      --policy graded_seg_B3 \\
      --n-segments 1000 \\
      --seed 42 \\
      --output /tmp/protection_map.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PLANES = 8
SEGMENT_SIZE = 4096


def make_graded_seg_B3(n_segments: int, seed: int, segment_size: int = SEGMENT_SIZE) -> dict[str, Any]:
    rng = __import__("random").Random(seed)
    pmap: dict[str, int] = {}
    plane1_count = (n_segments + 1) // 2
    plane1_indices = list(range(n_segments))
    rng.shuffle(plane1_indices)
    plane1_protected = set(plane1_indices[:plane1_count])

    for seg in range(n_segments):
        for p in range(PLANES):
            key = f"{p}_{seg}"
            if p == 0:
                pmap[key] = 3
            elif p == 1:
                pmap[key] = 3 if seg in plane1_protected else 1
            else:
                pmap[key] = 1

    per_plane: dict[str, dict] = {}
    for p in range(PLANES):
        vals = [pmap[f"{p}_{seg}"] for seg in range(n_segments)]
        protected = sum(1 for v in vals if v == 3)
        per_plane[str(p)] = {
            "protected_fraction": protected / n_segments if n_segments > 0 else 0.0,
            "r3_segments": protected,
            "total_segments": n_segments,
        }

    total_extra = sum(
        (v - 1) for v in pmap.values()
    )
    expected_extra = 3 * n_segments

    return {
        "policy": "graded_seg_B3",
        "seed": seed,
        "n_segments": n_segments,
        "segment_size": segment_size,
        "total_extra_storage_bytes": total_extra * segment_size,
        "expected_extra_storage_bytes": expected_extra * segment_size,
        "total_extra_storage_ratio": total_extra / n_segments if n_segments > 0 else 0.0,
        "per_plane": per_plane,
        "map": pmap,
    }


def make_uniform_spread_seg_B3(n_segments: int, seed: int, segment_size: int = SEGMENT_SIZE) -> dict[str, Any]:
    rng = __import__("random").Random(seed)
    total_protected_needed = int(1.5 * n_segments)
    base = total_protected_needed // 8
    rem = total_protected_needed % 8
    per_plane_protected = [base + 1 if i < rem else base for i in range(8)]

    pmap: dict[str, int] = {}
    for p in range(PLANES):
        count = per_plane_protected[p]
        indices = list(range(n_segments))
        rng.shuffle(indices)
        protected_set = set(indices[:count])
        for seg in range(n_segments):
            key = f"{p}_{seg}"
            pmap[key] = 3 if seg in protected_set else 1

    per_plane: dict[str, dict] = {}
    for p in range(PLANES):
        vals = [pmap[f"{p}_{seg}"] for seg in range(n_segments)]
        protected = sum(1 for v in vals if v == 3)
        per_plane[str(p)] = {
            "protected_fraction": protected / n_segments if n_segments > 0 else 0.0,
            "r3_segments": protected,
            "total_segments": n_segments,
        }

    total_extra = sum(
        (v - 1) for v in pmap.values()
    )

    return {
        "policy": "uniform_spread_seg_B3",
        "seed": seed,
        "n_segments": n_segments,
        "segment_size": segment_size,
        "total_extra_storage_bytes": total_extra * segment_size,
        "expected_extra_storage_bytes": 3 * n_segments * segment_size,
        "total_extra_storage_ratio": total_extra / n_segments if n_segments > 0 else 0.0,
        "per_plane": per_plane,
        "map": pmap,
    }


def validate_budget(result: dict) -> tuple[bool, float]:
    n_seg = result["n_segments"]
    total_extra = sum((v - 1) for v in result["map"].values())
    expected = 3 * n_seg
    ratio = total_extra / expected if expected > 0 else 0.0
    ok = abs(ratio - 1.0) < 0.001
    if not ok:
        print(
            f"BUDGET MISMATCH: got {total_extra}, expected {expected} "
            f"(ratio={ratio:.6f})",
            file=sys.stderr,
        )
    return ok, ratio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=str, required=True,
                        choices=["graded_seg_B3", "uniform_spread_seg_B3"])
    parser.add_argument("--n-segments", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-size", type=int, default=SEGMENT_SIZE)
    args = parser.parse_args()

    if args.policy == "graded_seg_B3":
        result = make_graded_seg_B3(args.n_segments, args.seed, args.segment_size)
    else:
        result = make_uniform_spread_seg_B3(args.n_segments, args.seed, args.segment_size)

    budget_ok, budget_ratio = validate_budget(result)
    if not budget_ok:
        pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    pp = result["per_plane"]
    print(f"policy={result['policy']} seed={args.seed} n_segments={args.n_segments}")
    print(f"total_extra_storage_ratio={result['total_extra_storage_ratio']:.4f} "
          f"(expected 3.0000)")
    for p in range(PLANES):
        info = pp[str(p)]
        print(f"  plane {p}: protected_fraction={info['protected_fraction']:.4f}")
    print(f"map_path={args.output}")


if __name__ == "__main__":
    main()
