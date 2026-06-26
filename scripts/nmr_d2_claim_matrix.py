#!/usr/bin/env python3
"""NMR-D2 claim matrix builder.

Reads stochastic results, computes paired deltas (graded - uniform),
and produces claim verdicts.

Outputs:
  - nmr_d2_claim_matrix.csv (main comparison)
  - nmr_d2_paired_delta_summary.csv (delta columns)

Usage:
  python3 scripts/nmr_d2_claim_matrix.py \\
      --stochastic-results /tmp/nmr_d2_stochastic_matrix.csv \\
      --policies graded_seg_B3 uniform_spread_seg_B3 \\
      --output-dir /tmp/nmr_d2_claims
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HEADLINE_METRICS = [
    "repair_coverage_rate",
    "msb_coverage_rate",
    "repair_failure_degrade_rate",
    "expected_bound_width",
    "relative_bound_width",
]

# Verdict voting only uses metrics that actually differentiate at matched B.
# repair_coverage_rate is NOT a differentiator (both policies cover equal area
# at B=3).  repair_failure_degrade_rate is always 0 in D2 (single-replica
# faults are always recovered by r=3 majority vote).  So verdict rests on:
#   msb_coverage_rate  — graded protects MSB, uniform does not
#   relative_bound_width — normalized to clean sum, enables cross-dataset comparison
VERDICT_METRICS = ["msb_coverage_rate", "relative_bound_width"]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def group_by_key(rows: list[dict]) -> dict:
    """Group rows by (dataset, fault_rate, seed)."""
    groups: dict = defaultdict(dict)
    for r in rows:
        key = (r["dataset"], r["fault_rate"], r["seed"])
        groups[key][r["policy"]] = r
    return groups


def paired_delta(graded_row: dict, uniform_row: dict) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for metric in HEADLINE_METRICS:
        g = float(graded_row.get(metric, "nan"))
        u = float(uniform_row.get(metric, "nan"))
        deltas[f"delta_{metric}"] = g - u
    return deltas


def claim_verdict(deltas: dict[str, float]) -> str:
    """Determine claim verdict from paired deltas.

    Only uses VERDICT_METRICS (msb_coverage_rate, relative_bound_width).
    repair_coverage_rate: both policies cover equal area at matched B=3.
    expected_bound_width: included in CSV but verdict uses normalized version.
    """
    if not deltas:
        return "TIE"

    graded_wins = 0
    uniform_wins = 0

    d_msb = deltas.get("delta_msb_coverage_rate", 0.0)
    d_relbw = deltas.get("delta_relative_bound_width", 0.0)

    if d_msb > 0.001:
        graded_wins += 1
    elif d_msb < -0.001:
        uniform_wins += 1

    if d_relbw < -1e-6:
        graded_wins += 1
    elif d_relbw > 1e-6:
        uniform_wins += 1

    if graded_wins == uniform_wins:
        return "TIE"
    return "GRADED_WINS" if graded_wins > uniform_wins else "UNIFORM_WINS"


def aggregate_seed_summary(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    n = len(rows)
    agg: dict[str, float] = {}
    for metric in HEADLINE_METRICS:
        vals = [float(r.get(metric, "nan")) for r in rows]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            agg[f"{metric}_mean"] = sum(vals) / len(vals)
            agg[f"{metric}_min"] = min(vals)
            agg[f"{metric}_max"] = max(vals)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stochastic-results", type=Path, required=True)
    parser.add_argument("--policies", type=str, nargs=2,
                        default=["graded_seg_B3", "uniform_spread_seg_B3"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_csv(args.stochastic_results)
    groups = group_by_key(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    claim_rows: list[dict] = []
    delta_rows: list[dict] = []
    g_policy, u_policy = args.policies

    for key, policy_rows in sorted(groups.items()):
        dataset, fault_rate, seed = key
        graded_row = policy_rows.get(g_policy)
        uniform_row = policy_rows.get(u_policy)

        if graded_row is None or uniform_row is None:
            print(f"WARNING: missing policy for key={key}", file=sys.stderr)
            continue

        deltas = paired_delta(graded_row, uniform_row)
        verdict = claim_verdict(deltas)

        claim_row = {
            "dataset": dataset,
            "fault_rate": fault_rate,
            "seed": seed,
        }
        for metric in HEADLINE_METRICS:
            claim_row[f"{g_policy}_{metric}"] = float(graded_row.get(metric, "nan"))
            claim_row[f"{u_policy}_{metric}"] = float(uniform_row.get(metric, "nan"))
        claim_row.update(deltas)
        claim_row["claim_verdict"] = verdict
        claim_rows.append(claim_row)

        delta_row = {
            "dataset": dataset,
            "fault_rate": fault_rate,
            "seed": seed,
            **deltas,
            "claim_verdict": verdict,
        }
        delta_rows.append(delta_row)

    # Write claim matrix
    claim_fieldnames = [
        "dataset", "fault_rate", "seed",
    ]
    for metric in HEADLINE_METRICS:
        claim_fieldnames.append(f"{g_policy}_{metric}")
        claim_fieldnames.append(f"{u_policy}_{metric}")
    for metric in HEADLINE_METRICS:
        claim_fieldnames.append(f"delta_{metric}")
    claim_fieldnames.append("claim_verdict")

    claim_path = args.output_dir / "nmr_d2_claim_matrix.csv"
    with claim_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=claim_fieldnames)
        writer.writeheader()
        writer.writerows(claim_rows)
    print(f"claim_matrix: {claim_path} ({len(claim_rows)} rows)")

    # Write paired delta summary
    delta_fieldnames = [
        "dataset", "fault_rate", "seed",
    ] + [f"delta_{m}" for m in HEADLINE_METRICS] + ["claim_verdict"]
    delta_path = args.output_dir / "nmr_d2_paired_delta_summary.csv"
    with delta_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=delta_fieldnames)
        writer.writeheader()
        writer.writerows(delta_rows)
    print(f"paired_delta_summary: {delta_path} ({len(delta_rows)} rows)")

    # Per-rate aggregate
    by_rate: dict[str, list[dict]] = defaultdict(list)
    for dr in delta_rows:
        by_rate[dr["fault_rate"]].append(dr)

    for rate, rate_deltas in sorted(by_rate.items()):
        verdicts = Counter(r["claim_verdict"] for r in rate_deltas)
        print(f"\n  rate={rate}: verdicts={dict(verdicts)}")
        for metric in VERDICT_METRICS:
            vals = [float(r.get(f"delta_{metric}", "nan")) for r in rate_deltas]
            vals = [v for v in vals if not math.isnan(v)]
            if vals:
                mean_d = sum(vals) / len(vals)
                print(f"    delta_{metric}: mean={mean_d:.6e}")

    # Write aggregate summary
    agg_path = args.output_dir / "nmr_d2_aggregate_summary.csv"
    agg_fieldnames = ["fault_rate", "verdict", "count"] + [f"delta_{m}" for m in HEADLINE_METRICS]
    agg_rows: list[dict] = []
    for rate, rate_deltas in sorted(by_rate.items()):
        for verdict in set(r["claim_verdict"] for r in rate_deltas):
            subset = [r for r in rate_deltas if r["claim_verdict"] == verdict]
            row: dict[str, Any] = {
                "fault_rate": rate,
                "verdict": verdict,
                "count": len(subset),
            }
            for metric in HEADLINE_METRICS:
                vals = [float(r.get(f"delta_{metric}", "nan")) for r in subset]
                vals = [v for v in vals if not math.isnan(v)]
                row[f"delta_{metric}"] = sum(vals) / len(vals) if vals else 0.0
            agg_rows.append(row)
    with agg_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"aggregate_summary: {agg_path} ({len(agg_rows)} rows)")


if __name__ == "__main__":
    main()
