#!/usr/bin/env python3
"""NMR-D2 stochastic evaluator.

Runs plane-uniform and workload-weighted fault injection suites
at the three headline fault rates across multiple seeds.

Output: nmr_d2_stochastic_matrix.csv

Usage:
  python3 scripts/nmr_d2_stochastic.py \\
      --protection-map-dir /tmp/pmaps \\
      --clean-plane-dir /path/to/planes \\
      --dataset sensor --n-rows 10000000 \\
      --headline-rates 1e-07 1e-06 1e-05 \\
      --seeds 0 1 2 3 4 \\
      --output /tmp/nmr_d2_stochastic_matrix.csv
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
PLANE_WEIGHTS = [1 << (8 * (7 - p)) for p in range(PLANES)]


def compute_clean_encoded_sum(clean_plane_dir: Path, n_rows: int) -> int:
    """Sum(row_p * weight[p]) across planes that have files.

    Contract: missing plane files are treated as all-zero.  This matches the
    D2 coverage map loader (``load_clean_plane`` in nmr_d2_coverage_map.py),
    which also returns ``bytes(n_rows)`` for planes without a file.

    Some real scientific datasets (e.g. cesm_atm_cloud) have max_plane_count=7;
    plane 7 is absent and contributes zero to the encoded sum.  Explicitly
    raising an error here would break those datasets without adding scientific
    value, since the zero contribution is the correct behavior.
    """
    total = 0
    for p in range(PLANES):
        for fmt in (f"plane_{p}.bin", f"plane_{p:03d}.bin"):
            path = clean_plane_dir / fmt
            if path.is_file():
                data = path.read_bytes()[:n_rows]
                break
        else:
            continue
        total += sum(data) * PLANE_WEIGHTS[p]
    return total


def compute_metrics(rows: list[dict], n_rows: int, clean_encoded_sum: int = 0) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {}

    cr_counts = Counter(r["coverage_relation"] for r in rows)

    repair_coverage_rate = cr_counts.get("repaired", 0) / n
    detected_rate = cr_counts.get("detected", 0) / n
    unprotected_rate = cr_counts.get("unprotected", 0) / n
    clean_rate = cr_counts.get("clean", 0) / n

    total_crc_hit = cr_counts.get("repaired", 0) + cr_counts.get("detected", 0)
    if total_crc_hit > 0:
        repair_failure_degrade_rate = cr_counts.get("detected", 0) / total_crc_hit
    else:
        repair_failure_degrade_rate = 0.0

    plane0_rows = [r for r in rows if r["plane"] == 0]
    p0_n = len(plane0_rows)
    if p0_n > 0:
        p0_repaired = sum(1 for r in plane0_rows if r["coverage_relation"] == "repaired")
        msb_coverage_rate = p0_repaired / p0_n
    else:
        msb_coverage_rate = 0.0

    expected_bound_width = 0.0
    for r in rows:
        p = r["plane"]
        cr = r["coverage_relation"]
        if cr in ("detected", "unprotected"):
            seg_len = r["segment_end"] - r["segment_start"]
            expected_bound_width += seg_len * 255 * PLANE_WEIGHTS[p]

    return {
        "repair_coverage_rate": repair_coverage_rate,
        "detected_rate": detected_rate,
        "unprotected_rate": unprotected_rate,
        "clean_rate": clean_rate,
        "repair_failure_degrade_rate": repair_failure_degrade_rate,
        "msb_coverage_rate": msb_coverage_rate,
        "expected_bound_width": expected_bound_width,
        "clean_encoded_sum": float(clean_encoded_sum),
        "relative_bound_width": (
            expected_bound_width / max(abs(clean_encoded_sum), 1)
            if clean_encoded_sum else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protection-map-dir", type=Path, required=True)
    parser.add_argument("--clean-plane-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n-rows", type=int, required=True)
    parser.add_argument("--headline-rates", type=float, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--mode", type=str, default="plane_uniform",
                        choices=["plane_uniform", "workload_weighted"])
    parser.add_argument("--pmap-seed", type=int, default=0,
                        help="Protection map seed (stochastic samples different segment layouts)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_rows: list[dict] = []

    clean_encoded_sum = compute_clean_encoded_sum(args.clean_plane_dir, args.n_rows)
    print(f"clean_encoded_sum={clean_encoded_sum}", file=sys.stderr)

    for policy in ["graded_seg_B3", "uniform_spread_seg_B3"]:
        pmap_path = args.protection_map_dir / f"{policy}_seed{args.pmap_seed}.json"
        if not pmap_path.is_file():
            n_segments = (args.n_rows + pmap_gen.SEGMENT_SIZE - 1) // pmap_gen.SEGMENT_SIZE
            if policy == "graded_seg_B3":
                result = pmap_gen.make_graded_seg_B3(n_segments, args.pmap_seed)
            else:
                result = pmap_gen.make_uniform_spread_seg_B3(n_segments, args.pmap_seed)
            pmap_path.parent.mkdir(parents=True, exist_ok=True)
            pmap_path.write_text(json.dumps(result, indent=2) + "\n")

        pmap = json.loads(pmap_path.read_text())
        pmap_seg_size = pmap.get("segment_size", pmap_gen.SEGMENT_SIZE)
        n_segments = (args.n_rows + pmap_seg_size - 1) // pmap_seg_size

        for rate_val in args.headline_rates:
            for seed in args.seeds:
                rows = cov.compute_coverage_map(
                    pmap,
                    args.clean_plane_dir,
                    args.dataset,
                    args.n_rows,
                    rate_val,
                    seed,
                    mode=args.mode,
                )

                metrics = compute_metrics(rows, args.n_rows, clean_encoded_sum)
                row = {
                    "dataset": args.dataset,
                    "policy": policy,
                    "fault_rate": str(rate_val),
                    "seed": seed,
                    "n_segments": n_segments,
                    "n_rows": args.n_rows,
                    **metrics,
                }
                all_rows.append(row)

                print(
                    f"{policy} rate={rate_val:.0e} seed={seed}: "
                    f"repair={metrics['repair_coverage_rate']:.6f} "
                    f"detected={metrics['detected_rate']:.6f} "
                    f"unprotected={metrics['unprotected_rate']:.6f} "
                    f"msb_cov={metrics['msb_coverage_rate']:.6f} "
                    f"bw={metrics['expected_bound_width']:.4e} "
                    f"rel_bw={metrics['relative_bound_width']:.6f}"
                )

    fieldnames = [
        "dataset", "policy", "fault_rate", "seed",
        "n_segments", "n_rows",
        "repair_coverage_rate", "detected_rate",
        "unprotected_rate", "clean_rate",
        "repair_failure_degrade_rate",
        "msb_coverage_rate",
        "expected_bound_width",
        "clean_encoded_sum",
        "relative_bound_width",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"stochastic_matrix: {args.output} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
