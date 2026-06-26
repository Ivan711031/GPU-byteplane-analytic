#!/usr/bin/env python3
"""Generate fault plans for NMR-C v2 fault-rate sweep experiment.

F1-F8 fault families adapted from Issue #283, producing JSON fault plans
consumable by bench_nmr_c_v2_k_sweep --fault-plan.

Each family generates {plane, replica, offset, mask} entries where:
  - plane: 0..max_planes-1 (only first k planes matter for the fused path)
  - replica: 0,1,2 (F1-F6 target replica 0; F7/F8 target more)
  - offset: byte offset within the plane buffer
  - mask: XOR mask in [1, 255]

Usage:
  python3 scripts/generate_nmr_c_v2_fault_plans.py \
    --n-rows 25000000 --max-planes 8 \
    --families F1 F2 F3 F4 F5 F6 F7 F8 \
    --rates 1e-6 1e-4 1e-3 \
    --seeds 0 1 2 \
    --output-dir fault_plans_nmr_c_v2/hurricane_u
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


def derive_seed(base_seed: int, family: str, rate: float) -> int:
    key = f"{base_seed}:{family}:{rate}"
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:16], 16)


def _fault_family_F1(rng: random.Random, n_rows: int, rate: float,
                     max_planes: int) -> list[dict]:
    n_events = max(1, int(n_rows * rate)) if rate > 0 else 0
    events: list[dict] = []
    for _ in range(n_events):
        plane = rng.randint(0, max_planes - 1)
        offset = rng.randint(0, n_rows - 1)
        mask = rng.randint(1, 255)
        events.append({"plane": plane, "replica": 0, "offset": offset, "mask": mask})
    return events


def _fault_family_F2(rng: random.Random, n_rows: int, rate: float,
                     max_planes: int,
                     burst_min: int = 2, burst_max: int = 4) -> list[dict]:
    burst_max = min(burst_max, n_rows - 1)
    burst_min = min(burst_min, burst_max)
    burst_len = rng.randint(burst_min, burst_max) if burst_min < burst_max else burst_max
    n_events = max(1, int(n_rows * rate)) if rate > 0 else 0
    n_events = min(n_events, max(100, int(n_rows * rate / 20)))
    events: list[dict] = []
    for _ in range(n_events):
        plane = rng.randint(0, max_planes - 1)
        start = rng.randint(0, max(0, n_rows - burst_len - 1))
        for i in range(burst_len):
            mask = rng.randint(1, 255)
            events.append({"plane": plane, "replica": 0, "offset": start + i, "mask": mask})
    return events


def _fault_family_F3(rng: random.Random, n_rows: int, rate: float,
                     max_planes: int,
                     run_length: int = 64) -> list[dict]:
    run_length = min(run_length, n_rows)
    n_bursts = max(1, int(n_rows * rate)) if rate > 0 else 0
    n_bursts = min(n_bursts, max(100, int(n_rows * rate / 20)))
    events: list[dict] = []
    for _ in range(n_bursts):
        plane = rng.randint(0, max_planes - 1)
        start = rng.randint(0, max(0, n_rows - run_length))
        for i in range(run_length):
            mask = rng.randint(1, 255)
            events.append({"plane": plane, "replica": 0, "offset": start + i, "mask": mask})
    return events


def _fault_family_F4(rng: random.Random, n_rows: int, rate: float,
                     max_planes: int,
                     affected_row_fraction: float | None = None) -> list[dict]:
    """Column-like repeated offset corruption.
    Affected row fraction scales with rate: low→0.01, mid→0.1, high→1.0."""
    if affected_row_fraction is None:
        affected_row_fraction = min(1.0, max(0.01, rate * 1000.0))
    segment_size = 4096
    n_segments = (n_rows + segment_size - 1) // segment_size
    n_affected = max(1, int(n_segments * affected_row_fraction))
    n_affected = min(n_affected, n_segments)
    plane = rng.randint(0, max_planes - 1)
    mask = rng.randint(1, 255)
    offset = rng.randint(0, 255)
    events: list[dict] = []
    affected_segments = rng.sample(range(n_segments), n_affected)
    for seg_idx in affected_segments:
        eff_offset = offset + seg_idx * segment_size
        if eff_offset >= n_rows:
            continue
        events.append({"plane": plane, "replica": 0, "offset": eff_offset, "mask": mask})
    return events


def _fault_family_F5(rng: random.Random, n_rows: int, rate: float,
                     max_planes: int) -> list[dict]:
    density = min(0.1, rate * 1000)
    region_size = min(4096, n_rows)
    n_regions = max(1, int(n_rows * rate))
    n_regions = min(n_regions, max(100, int(n_rows * rate / 20)))
    events: list[dict] = []
    for _ in range(n_regions):
        plane = rng.randint(0, max_planes - 1)
        region_start = rng.randint(0, max(0, n_rows - region_size))
        n_corrupt = max(1, int(region_size * density))
        positions = rng.sample(
            range(region_start, region_start + region_size),
            min(n_corrupt, region_size),
        )
        for pos in positions:
            mask = rng.randint(1, 255)
            events.append({"plane": plane, "replica": 0, "offset": pos, "mask": mask})
    return events


def _fault_family_F6(rng: random.Random, n_rows: int, rate: float,
                     max_planes: int) -> list[dict]:
    burst_events = max(1, int(rate * 1e6))
    region_size = min(1024, n_rows)
    events: list[dict] = []
    plane = rng.randint(0, max_planes - 1)
    region_start = rng.randint(0, max(0, n_rows - region_size))
    for _ in range(burst_events):
        pos = rng.randint(region_start, region_start + region_size - 1)
        pos = min(pos, n_rows - 1)
        mask = rng.randint(1, 255)
        events.append({"plane": plane, "replica": 0, "offset": pos, "mask": mask})
    return events


def _fault_family_F7(rng: random.Random, n_rows: int, rate: float,
                     max_planes: int,
                     target_replicas: list[int] | None = None) -> list[dict]:
    if target_replicas is None:
        target_replicas = [0, 1]
    n_events = max(1, int(n_rows * rate)) if rate > 0 else 0
    events: list[dict] = []
    for _ in range(n_events):
        plane = rng.randint(0, max_planes - 1)
        offset = rng.randint(0, n_rows - 1)
        mask = rng.randint(1, 255)
        for rep in target_replicas:
            events.append({"plane": plane, "replica": rep, "offset": offset, "mask": mask})
    return events


def _fault_family_F8(rng: random.Random, n_rows: int, rate: float,
                     max_planes: int) -> list[dict]:
    f1_ratio = 0.5
    f1_rng = random.Random(rng.randint(0, 2**31))
    f7_rng = random.Random(rng.randint(0, 2**31))
    f1_events = _fault_family_F1(f1_rng, n_rows, rate * f1_ratio, max_planes)
    f7_events = _fault_family_F7(f7_rng, n_rows, rate * (1 - f1_ratio), max_planes)
    return f1_events + f7_events


_FAULT_FAMILIES = {
    "F1": _fault_family_F1,
    "F2": _fault_family_F2,
    "F3": _fault_family_F3,
    "F4": _fault_family_F4,
    "F5": _fault_family_F5,
    "F6": _fault_family_F6,
    "F7": _fault_family_F7,
    "F8": _fault_family_F8,
}


def generate_fault_plan(n_rows: int, max_planes: int,
                        family: str, rate: float, seed: int,
                        k: int | None = None) -> dict[str, Any]:
    effective_planes = min(max_planes, k) if k is not None else max_planes
    rng = random.Random(derive_seed(seed, family, rate))
    entries = _FAULT_FAMILIES[family](rng, n_rows, rate, effective_planes)
    result = {
        "fault_family": family,
        "fault_rate": rate,
        "fault_rate_label": _rate_label(rate),
        "seed": seed,
        "n_rows": n_rows,
        "max_planes": effective_planes,
        "entry_count": len(entries),
        "entries": entries,
    }
    if k is not None:
        result["protected_planes"] = k
    return result


def _rate_label(rate: float) -> str:
    if rate <= 1e-6:
        return "low"
    elif rate <= 1e-4:
        return "mid"
    else:
        return "high"


def write_fplan(plan: dict[str, Any], fpath: str) -> None:
    """Write a compact text .fplan file consumable by the GPU binary."""
    entries = plan["entries"]
    with open(fpath, "w") as f:
        header = (f"FAMILY={plan['fault_family']} RATE={plan['fault_rate']:.6e}"
                  f" SEED={plan['seed']} N={plan['n_rows']} MAXP={plan['max_planes']}"
                  f" ENTRIES={len(entries)}")
        if "protected_planes" in plan:
            header += f" K={plan['protected_planes']}"
        header += "\n"
        f.write(header)
        for e in entries:
            f.write(f"{e['plane']} {e['replica']} {e['offset']} {e['mask']}\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate NMR-C v2 fault plans for F1-F8 families")
    ap.add_argument("--n-rows", type=int, required=True,
                    help="Number of rows (== plane bytes for BFP)")
    ap.add_argument("--max-planes", type=int, required=True,
                    help="Maximum plane count")
    ap.add_argument("--k", type=int, default=None,
                    help="Restrict entries to planes < K (default: use max-planes)")
    ap.add_argument("--families", nargs="+", required=True,
                    help="Fault families (F1-F8)")
    ap.add_argument("--rates", type=float, nargs="+", required=True,
                    help="Fault rates (e.g. 1e-6 1e-4 1e-3)")
    ap.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="Seeds (e.g. 0 1 2)")
    ap.add_argument("--output-dir", type=str, required=True,
                    help="Output directory for fault plan JSONs")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for family in args.families:
        if family not in _FAULT_FAMILIES:
            print(f"WARNING: unknown family {family}, skipping")
            continue
        for rate in args.rates:
            for seed in args.seeds:
                plan = generate_fault_plan(
                    args.n_rows, args.max_planes, family, rate, seed, k=args.k)
                base = f"fault_plan_{family}_{_rate_label(rate)}_{rate:.0e}_seed{seed}"
                if args.k is not None:
                    base += f"_k{args.k}"
                jpath = os.path.join(args.output_dir, base + ".json")
                fpath = os.path.join(args.output_dir, base + ".fplan")
                with open(jpath, "w") as f:
                    json.dump(plan, f)
                write_fplan(plan, fpath)
                print(f"  {jpath}  entries={plan['entry_count']}")


if __name__ == "__main__":
    main()
