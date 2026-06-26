#!/usr/bin/env python3
"""Recompute aggregate CSVs from an existing pilot matrix CSV.

Usage:
  python3 scripts/recompute_aggregates.py \\
    --pilot-matrix results/.../claim1_realistic_pilot_matrix.csv \\
    --output-dir /tmp/recomputed
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Import aggregation functions from the campaign module
from nmr_claim1_realistic_campaign import (
    compute_all_metrics,
    compute_all_replication_metrics,
    write_headline_matrix,
    write_policy_frontier,
    write_fault_family_summary,
    write_exposure_summary,
    write_outcome_summary,
    write_replication_headline,
    write_replication_frontier,
)


NUMERIC_FIELDS = {
    "policy_B", "event_count", "bit_flip_count", "affected_plane_count",
    "affected_segment_count", "unique_mutated_byte_count",
    "xor_cancellation_count", "attempted_mutated_byte_count",
    "clean_answer", "delivered_answer", "abs_error", "relative_error",
    "bound_width", "relative_bound_width",
    "significance_weighted_impact_budget",
    "rate",
}
BOOL_FIELDS = {"detected", "contains_truth", "decision_flip"}


def convert_row(r: dict) -> dict:
    """Convert CSV string fields to proper Python types."""
    result = dict(r)
    for k, v in result.items():
        if k in NUMERIC_FIELDS:
            try:
                result[k] = float(v) if "." in str(v) else int(v)
            except (ValueError, TypeError):
                pass
        elif k in BOOL_FIELDS:
            if isinstance(v, str):
                result[k] = v.lower() in ("true", "1")
    return result


def load_pilot_matrix(path: Path) -> list[dict]:
    """Load pilot matrix CSV with type conversion."""
    with path.open(newline="") as f:
        raw = list(csv.DictReader(f))
    return [convert_row(r) for r in raw]


def group_rows(rows: list[dict]) -> dict[tuple, list[dict]]:
    """Group rows by (dataset, family, severity_label, rate, policy_B, policy_type)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (
            r.get("dataset", ""),
            r.get("fault_family", ""),
            r.get("severity_label", ""),
            r.get("rate", ""),
            r.get("policy_B", ""),
            r.get("policy_type", ""),
        )
        groups[key].append(r)
    return groups


def check_faulted_planes(rows: list[dict]) -> bool:
    """Check if the CSV has faulted_planes data."""
    if not rows:
        return False
    fp = rows[0].get("faulted_planes", "")
    return bool(fp) and fp != "[]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", type=str, default="detect_and_bound",
                        choices=["detect_and_bound", "replication_only"])
    args = parser.parse_args()

    rows = load_pilot_matrix(args.pilot_matrix)
    print(f"Loaded {len(rows)} rows from {args.pilot_matrix}")

    has_fp = check_faulted_planes(rows)
    if not has_fp:
        print("NOTE: CSV is missing 'faulted_planes' column.")
        print("  r2_applicable_cells / r3_applicable_cells will be 0")
        print("  r2_disagreement_rate / r3_majority_repair_rate will be NA")
        print("  To get these metrics, re-run pilot with current code.\n")

    is_rep = args.mode == "replication_only"
    _hl_fn = write_replication_headline if is_rep else write_headline_matrix
    _fr_fn = write_replication_frontier if is_rep else write_policy_frontier
    _hl_suffix = "_nosum32" if is_rep else ""

    args.output_dir.mkdir(parents=True, exist_ok=True)

    hgroup: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (
            r.get("dataset", ""),
            r.get("fault_family", ""),
            r.get("severity_label", ""),
            r.get("rate", ""),
            r.get("policy_B", ""),
            r.get("policy_type", ""),
        )
        hgroup[key].append(r)

    hl_path = args.output_dir / f"claim1_realistic_headline_matrix{_hl_suffix}.csv"
    _hl_fn(hgroup, hl_path)
    print(f"headline_matrix{_hl_suffix}: {hl_path} ({len(hgroup)} rows)")

    fr_path = args.output_dir / f"claim1_realistic_policy_frontier{_hl_suffix}.csv"
    _fr_fn(hgroup, fr_path)
    print(f"policy_frontier{_hl_suffix}: {fr_path}")

    ff_path = args.output_dir / "claim1_realistic_fault_family_summary.csv"
    write_fault_family_summary(rows, ff_path, is_rep)
    print(f"fault_family_summary: {ff_path}")

    exp_path = args.output_dir / "claim1_realistic_exposure_summary.csv"
    write_exposure_summary(rows, exp_path)
    print(f"exposure_summary: {exp_path}")

    # Outcome summary
    oc_path = args.output_dir / "claim1_realistic_outcome_summary.csv"
    write_outcome_summary(rows, oc_path, is_rep)
    print(f"outcome_summary: {oc_path}")

    print(f"\nDone. Output written to {args.output_dir}")


if __name__ == "__main__":
    main()
