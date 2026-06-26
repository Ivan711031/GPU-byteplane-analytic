#!/usr/bin/env python3
"""NMR-D2 pilot fault-rate calibration.

Sweeps fault rates from 1e-9 to 1e-3, computing per-policy coverage loss
to identify three informative headline rates:
  - Low: near-zero coverage loss for graded
  - Medium: measurable graded advantage
  - High: both policies stressed

Output: nmr_d2_calibration_matrix.csv

Usage:
  python3 scripts/nmr_d2_calibration.py \\
      --protection-map-dir /tmp/pmaps \\
      --clean-plane-dir /path/to/planes \\
      --dataset sensor --n-rows 10000000 \\
      --output /tmp/nmr_d2_calibration_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import nmr_d2_coverage_map as cov
import nmr_d2_protection_map as pmap_gen

PLANES = 8
CANDIDATE_RATES = [
    1e-9, 3e-9, 1e-8, 3e-8,
    1e-7, 3e-7, 1e-6, 3e-6,
    1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
]
POLICIES = ["graded_seg_B3", "uniform_spread_seg_B3"]


def coverage_loss_rate(rows: list[dict]) -> dict[str, float]:
    n_total = len(rows)
    if n_total == 0:
        return {}
    cr_counts = Counter(r["coverage_relation"] for r in rows)
    return {
        "repair_coverage_rate": cr_counts.get("repaired", 0) / n_total,
        "detected_rate": cr_counts.get("detected", 0) / n_total,
        "unprotected_rate": cr_counts.get("unprotected", 0) / n_total,
        "clean_rate": cr_counts.get("clean", 0) / n_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protection-map-dir", type=Path, required=True)
    parser.add_argument("--clean-plane-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n-rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-rates", type=float, nargs="+",
                        default=CANDIDATE_RATES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_rows: list[dict] = []

    for policy in POLICIES:
        pmap_path = args.protection_map_dir / f"{policy}_seed{args.seed}.json"
        if not pmap_path.is_file():
            seg_size = pmap_gen.SEGMENT_SIZE
            n_segments = (args.n_rows + seg_size - 1) // seg_size
            print(f"Generating protection map: {pmap_path} (n_segments={n_segments}, segment_size={seg_size})", file=sys.stderr)
            if policy == "graded_seg_B3":
                result = pmap_gen.make_graded_seg_B3(n_segments, args.seed, seg_size)
            else:
                result = pmap_gen.make_uniform_spread_seg_B3(n_segments, args.seed, seg_size)
            pmap_path.parent.mkdir(parents=True, exist_ok=True)
            pmap_path.write_text(json.dumps(result, indent=2) + "\n")

        pmap = json.loads(pmap_path.read_text())
        pmap_seg_size = pmap.get("segment_size", pmap_gen.SEGMENT_SIZE)
        n_segments = (args.n_rows + pmap_seg_size - 1) // pmap_seg_size

        for rate_val in args.candidate_rates:
            rows = cov.compute_coverage_map(
                pmap,
                args.clean_plane_dir,
                args.dataset,
                args.n_rows,
                rate_val,
                args.seed,
                mode="plane_uniform",
            )

            loss = coverage_loss_rate(rows)
            row = {
                "dataset": args.dataset,
                "policy": policy,
                "fault_rate": str(rate_val),
                "seed": args.seed,
                "n_segments": n_segments,
                "n_rows": args.n_rows,
                **loss,
            }
            all_rows.append(row)

            print(
                f"{policy} rate={rate_val:.0e}: "
                f"repair={loss['repair_coverage_rate']:.6f} "
                f"detect={loss['detected_rate']:.6f} "
                f"unprotected={loss['unprotected_rate']:.6f}"
            )

    fieldnames = [
        "dataset", "policy", "fault_rate", "seed",
        "n_segments", "n_rows",
        "repair_coverage_rate", "detected_rate",
        "unprotected_rate", "clean_rate",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"calibration_matrix: {args.output} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
