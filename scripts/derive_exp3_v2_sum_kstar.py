#!/usr/bin/env python3
"""Derive k* tables from the joined Exp3 v2 precision-throughput matrix.

Produces two families:
  - empirical k*: smallest k whose OBSERVED error satisfies epsilon
  - bound k*: smallest k whose ANALYTIC BOUND satisfies epsilon

Epsilon grids inherited from prior Exp2 work (see analyze_v2_subcolumn_error.py).
"""

from __future__ import annotations

import csv
from pathlib import Path

JOINED_PATH = Path("results/precision_throughput/exp3_v2_sum_precision_throughput.csv")
OUT_DIR = Path("results/precision_throughput")

ABS_EPSILONS = [
    0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1.0, 10.0,
    100.0, 1e3, 1e4, 1e5, 1e6, 1e8, 1e10, 1e12,
]
REL_EPSILONS = [0.0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1]

KSTAR_COLUMNS = ["epsilon_type", "epsilon", "dataset", "artifact_label", "k_star"]

# 19-artifact matrix for iteration
ARTIFACT_MATRIX = [
    ("heavy_tailed", "p2", 7), ("heavy_tailed", "p3", 7),
    ("heavy_tailed", "p4", 7), ("heavy_tailed", "p5", 8),
    ("heavy_tailed", "p6", 8), ("sensor", "p2", 2),
    ("sensor", "p4", 3), ("sensor", "p6", 3),
    ("sensor", "p8", 4), ("sensor", "p10", 5),
    ("uniform", "p2", 3), ("uniform", "p4", 4),
    ("uniform", "p6", 4), ("uniform", "p8", 5),
    ("uniform", "p10", 6), ("zipfian", "p2", 6),
    ("zipfian", "p4", 6), ("zipfian", "p6", 7),
    ("zipfian", "p8", 8),
]


def _derive_kstar(
    rows: list[dict],
    epsilons: list[float],
    metric: str,
) -> list[dict]:
    out: list[dict] = []
    for ds, al, _ in ARTIFACT_MATRIX:
        matching = sorted(
            [r for r in rows if r["dataset"] == ds and r["artifact_label"] == al],
            key=lambda r: int(r["k"]),
        )
        for eps in epsilons:
            k_star = "NA"
            for r in matching:
                if float(r[metric]) <= eps:
                    k_star = str(r["k"])
                    break
            out.append({
                "epsilon_type": "absolute" if "abs" in metric else "relative",
                "epsilon": eps,
                "dataset": ds,
                "artifact_label": al,
                "k_star": k_star,
            })
    return out


def main() -> int:
    if not JOINED_PATH.is_file():
        print(f"ERROR: joined CSV not found: {JOINED_PATH}")
        return 1

    with JOINED_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    specs = [
        # (suffix, variant, epsilons, metric)
        ("absolute_empirical", "absolute", ABS_EPSILONS, "abs_error_vs_encoded_full_depth"),
        ("relative_empirical", "relative", REL_EPSILONS, "rel_error_vs_encoded_full_depth"),
        ("absolute_bound", "absolute", ABS_EPSILONS, "analytic_abs_bound_vs_encoded_full_depth"),
        ("relative_bound", "relative", REL_EPSILONS, "analytic_rel_bound_vs_encoded_full_depth"),
    ]

    for suffix, variant, epsilons, metric in specs:
        out_rows = _derive_kstar(rows, epsilons, metric)
        out_path = OUT_DIR / f"exp3_v2_sum_kstar_{suffix}.csv"
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=KSTAR_COLUMNS)
            w.writeheader()
            w.writerows(out_rows)
        print(f"Wrote {out_path} ({len(out_rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
