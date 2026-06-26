#!/usr/bin/env python3
"""Aggregate NMR-R11 single-replica CSV results into summary report tables."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def compute_metrics(rows: list[dict]) -> dict[str, float]:
    n = len(rows) if rows else 1
    detected = sum(1 for r in rows if r["detected"] == "True")
    contains_truth = sum(1 for r in rows if r["contains_truth"] == "True")
    outcomes = Counter(r["outcome"] for r in rows)
    widths = [float(r.get("bound_width", "0")) for r in rows]
    return {
        "total_cells": n,
        "detected_rate": detected / n,
        "contains_truth": contains_truth / n,
        "expected_bound_width": sum(widths) / n if widths else 0.0,
        "recovered_rate": outcomes.get("recovered", 0) / n,
        "bounded_rate": outcomes.get("bounded", 0) / n,
        "unavailable_rate": outcomes.get("unavailable", 0) / n,
        "silent_wrong_rate": outcomes.get("silent_wrong", 0) / n,
        "recovered_count": outcomes.get("recovered", 0),
        "bounded_count": outcomes.get("bounded", 0),
        "unavailable_count": outcomes.get("unavailable", 0),
        "silent_wrong_count": outcomes.get("silent_wrong", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for d in args.result_dirs:
        if not d.is_dir():
            print(f"SKIP: {d} not found", file=sys.stderr)
            continue
        for csv_path in sorted(d.glob("*.csv")):
            rows = load_csv(csv_path)
            print(f"Loaded {csv_path}: {len(rows)} rows")
            all_rows.extend(rows)

    if not all_rows:
        print("No data loaded", file=sys.stderr)
        sys.exit(1)

    # Aggregate by dataset, mode, rate
    groups = defaultdict(list)
    for r in all_rows:
        groups[(r.get("dataset", "?"), r.get("mode", "?"), r.get("rate", "?"))].append(r)

    print("\n=== Per-group Summary ===")
    fieldnames = ["dataset", "mode", "rate", "total_cells",
                  "detected_rate", "contains_truth",
                  "expected_bound_width",
                  "recovered_count", "bounded_count",
                  "unavailable_count", "silent_wrong_count"]
    out_rows: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        m = compute_metrics(group)
        row = {k: m[k] for k in fieldnames if k in m}
        row["dataset"], row["mode"], row["rate"] = key
        out_rows.append(row)
        print(f"  {key[0]:<20s} {key[1]:<16s} rate={key[2]:>8s}: "
              f"det={m['detected_rate']:.4f} ct={m['contains_truth']:.4f} "
              f"bw={m['expected_bound_width']:.4e} "
              f"recovered={m['recovered_count']} bounded={m['bounded_count']} "
              f"unavail={m['unavailable_count']} silent={m['silent_wrong_count']}")

    # Overall
    print("\n=== Overall Summary ===")
    m = compute_metrics(all_rows)
    print(f"  Total cells:         {m['total_cells']}")
    print(f"  detected_rate:       {m['detected_rate']:.6f}")
    print(f"  contains_truth:      {m['contains_truth']:.6f}")
    print(f"  expected_bound_width:{m['expected_bound_width']:.6e}")
    print(f"  recovered:           {m['recovered_count']}")
    print(f"  bounded:             {m['bounded_count']}")
    print(f"  unavailable:         {m['unavailable_count']}")
    print(f"  silent_wrong:        {m['silent_wrong_count']}")

    # Write output CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {args.output} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
