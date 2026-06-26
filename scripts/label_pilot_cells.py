#!/usr/bin/env python3
"""Label pilot matrix cells as too_sparse / informative / saturated.

Usage:
  python3 scripts/label_pilot_cells.py \
    --pilot-matrix results/.../claim1_realistic_pilot_matrix.csv \
    --output /tmp/matrix_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def load_pilot_matrix(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def group_by_cell(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (
            r["dataset"],
            r["fault_family"],
            r.get("severity_label", ""),
            r["rate"],
        )
        groups[key].append(r)
    return groups


def label_cell(rows: list[dict]) -> str:
    """Label one cell as too_sparse / informative / saturated.

    A cell = (dataset, family, severity, rate) with N seeds × 20 policies.
    """
    seeds = set(r["seed"] for r in rows)

    per_policy: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        # Composite key: r_vector + policy_B uniquely identifies a policy
        key = f"B{r['policy_B']}_{r['r_vector']}"
        per_policy[key].append(r)

    detection_rates: dict[str, float] = {}
    silent_wrong_rates: dict[str, float] = {}
    exact_correct_rates: dict[str, float] = {}

    for pname, prs in per_policy.items():
        n = len(prs)
        if n == 0:
            continue
        detected = sum(1 for r in prs if r.get("detected", "False") == "True")
        sw = sum(1 for r in prs if r.get("outcome", "") == "silent_wrong")
        exact = sum(1 for r in prs if r.get("outcome", "") == "exact_correct")
        detection_rates[pname] = detected / n
        silent_wrong_rates[pname] = sw / n
        exact_correct_rates[pname] = exact / n

    if not detection_rates:
        return "too_sparse"

    # Check saturated: all cells are exact_correct (no faults)
    if all(v == 1.0 for v in exact_correct_rates.values()):
        return "saturated"

    # Check too_sparse: all detection rates near zero
    max_dr = max(detection_rates.values())
    min_dr = min(detection_rates.values())

    if max_dr < 0.05:
        return "too_sparse"

    # Check too_sparse: no variance across policies
    if max_dr - min_dr < 0.001:
        return "too_sparse"

    # Check informative: variance exists, not too noisy
    dr_variance = max_dr - min_dr
    best_sw = min(silent_wrong_rates.values())
    has_mid_range = any(0.10 <= v <= 0.95 for v in detection_rates.values())

    if dr_variance > 0.05 and best_sw <= 0.05 and has_mid_range:
        return "informative"

    # Borderline: has variance but doesn't meet all informative criteria
    if dr_variance > 0.02:
        return "informative"

    return "too_sparse"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_pilot_matrix(args.pilot_matrix)
    cells = group_by_cell(rows)

    labels: list[dict] = []
    for key, cell_rows in sorted(cells.items()):
        dataset, family, severity, rate = key
        label = label_cell(cell_rows)

        # Count policies in different detection ranges
        per_policy: dict[str, list[dict]] = defaultdict(list)
        for r in cell_rows:
            per_policy[r.get("policy_name", "?")].append(r)
        dr_min = 1.0
        dr_max = 0.0
        sw_min = 1.0
        for pname, prs in per_policy.items():
            n = len(prs)
            if n == 0:
                continue
            detected = sum(1 for r in prs if r.get("detected", "False") == "True")
            sw = sum(1 for r in prs if r.get("outcome", "") == "silent_wrong")
            dr = detected / n
            dr_min = min(dr_min, dr)
            dr_max = max(dr_max, dr)
            sw_min = min(sw_min, sw / n)

        n_seeds = len(set(r["seed"] for r in cell_rows))

        labels.append({
            "dataset": dataset,
            "fault_family": family,
            "severity_label": severity,
            "rate": rate,
            "n_seeds": n_seeds,
            "label": label,
            "detection_rate_min": f"{dr_min:.4f}",
            "detection_rate_max": f"{dr_max:.4f}",
            "best_silent_wrong_rate": f"{sw_min:.4f}",
        })

    fieldnames = [
        "dataset", "fault_family", "severity_label", "rate",
        "n_seeds", "label",
        "detection_rate_min", "detection_rate_max",
        "best_silent_wrong_rate",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(labels)

    print(f"Labels: {args.output} ({len(labels)} cells)")
    summary = defaultdict(list)
    for l in labels:
        summary[l["label"]].append(
            f"{l['dataset']}/{l['fault_family']}/{l['severity_label']}@{l['rate']}"
        )
    for label, cells in sorted(summary.items()):
        print(f"  {label}: {len(cells)} cells")
        for c in cells:
            print(f"    {c}")


if __name__ == "__main__":
    main()
