#!/usr/bin/env python3
"""Join Exp3 v2 SUM error/bound matrix with the committed throughput matrix.

Input:
  - results/exp3_v2_sum_precision/exp3_v2_sum_error_bound_all.csv
  - results/exp3_v2_full_specialized_sum/exp3_v2_full_sum_throughput_all.csv

Join keys:
  - dataset
  - artifact_version
  - artifact_label
  - precision_power
  - k

Output:
  - results/precision_throughput/exp3_v2_sum_precision_throughput.csv (103 rows)
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

THROUGHPUT_PATH = Path("results/exp3_v2_full_specialized_sum/exp3_v2_full_sum_throughput_all.csv")
PRECISION_PATH = Path("results/exp3_v2_sum_precision/exp3_v2_sum_error_bound_all.csv")
OUT_DIR = Path("results/precision_throughput")
OUT_PATH = OUT_DIR / "exp3_v2_sum_precision_throughput.csv"

THROUGHPUT_STABLE_KEYS = [
    "dataset", "artifact_version", "artifact_label", "precision_power", "k",
]

THROUGHPUT_JOIN_FIELDS = [
    "max_plane_count",
    "ms_per_iter",
    "rows_per_sec",
    "billion_rows_per_sec",
    "logical_GBps",
    "validated",
    "kernel_path",
    "gpu_tag",
    "sum_reference_domain",
]

PRECISION_JOIN_FIELDS = [
    "progressive_sum",
    "encoded_full_depth_sum",
    "abs_error_vs_encoded_full_depth",
    "rel_error_vs_encoded_full_depth",
    "analytic_abs_bound_vs_encoded_full_depth",
    "analytic_rel_bound_vs_encoded_full_depth",
    "bound_gap_vs_encoded_full_depth",
]

OUTPUT_FIELDS = [
    "dataset",
    "artifact_version",
    "artifact_label",
    "precision_power",
    "k",
    "max_plane_count",
    "progressive_sum",
    "encoded_full_depth_sum",
    "abs_error_vs_encoded_full_depth",
    "rel_error_vs_encoded_full_depth",
    "analytic_abs_bound_vs_encoded_full_depth",
    "analytic_rel_bound_vs_encoded_full_depth",
    "bound_gap_vs_encoded_full_depth",
    "ms_per_iter",
    "rows_per_sec",
    "billion_rows_per_sec",
    "logical_GBps",
    "validated",
    "kernel_path",
    "gpu_tag",
    "sum_reference_domain",
]


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    for p in [THROUGHPUT_PATH, PRECISION_PATH]:
        if not p.is_file():
            print(f"ERROR: input not found: {p}", file=sys.stderr)
            return 1

    tp_rows = load_csv(THROUGHPUT_PATH)
    pr_rows = load_csv(PRECISION_PATH)

    print(f"Throughput rows: {len(tp_rows)}")
    print(f"Precision rows:  {len(pr_rows)}")

    # Index precision rows with duplicate detection
    pr_index: dict[tuple, dict] = {}
    for i, r in enumerate(pr_rows):
        key = (r["dataset"], r["artifact_version"], r["artifact_label"],
               r["precision_power"], int(r["k"]))
        if key in pr_index:
            print(f"ERROR: duplicate key in precision at row {i}: {key}", file=sys.stderr)
            return 1
        pr_index[key] = r

    # Index throughput rows with duplicate detection
    tp_index: dict[tuple, dict] = {}
    for i, r in enumerate(tp_rows):
        key = (r["dataset"], r["artifact_version"], r["artifact_label"],
               r["precision_power"], int(r["k"]))
        if key in tp_index:
            print(f"ERROR: duplicate key in throughput at row {i}: {key}", file=sys.stderr)
            return 1
        tp_index[key] = r

    # Validate counts
    gates_ok = True

    unmatched_pr = set(pr_index.keys()) - set(tp_index.keys())
    unmatched_tp = set(tp_index.keys()) - set(pr_index.keys())

    if unmatched_pr:
        print(f"ERROR: {len(unmatched_pr)} precision rows have no throughput match:", file=sys.stderr)
        for k in sorted(unmatched_pr):
            print(f"  {k}", file=sys.stderr)
        gates_ok = False

    if unmatched_tp:
        print(f"ERROR: {len(unmatched_tp)} throughput rows have no precision match:", file=sys.stderr)
        for k in sorted(unmatched_tp):
            print(f"  {k}", file=sys.stderr)
        gates_ok = False

    # Build joined rows using throughput as driver (should have 103 rows)
    joined: list[dict] = []
    for r in tp_rows:
        key = (r["dataset"], r["artifact_version"], r["artifact_label"],
               r["precision_power"], int(r["k"]))
        pr = pr_index.get(key)
        if pr is None:
            print(f"ERROR: throughput row {key} missing precision match", file=sys.stderr)
            gates_ok = False
            continue

        out = {}
        for f in ["dataset", "artifact_version", "artifact_label", "precision_power", "k"]:
            out[f] = r[f]
        for f in THROUGHPUT_JOIN_FIELDS:
            out[f] = r[f]
        for f in PRECISION_JOIN_FIELDS:
            out[f] = pr[f]
        joined.append(out)

    # Row-level gates on joined output
    expected_rows = 103
    if len(joined) != expected_rows:
        print(f"GATE FAIL: expected {expected_rows} joined rows, got {len(joined)}", file=sys.stderr)
        gates_ok = False
    else:
        print(f"GATE OK: {len(joined)} joined rows")

    # Validate joined fields
    for i, r in enumerate(joined):
        errs = []
        if r.get("validated", "").strip().lower() != "true":
            errs.append(f"validated={r.get('validated')!r}")
        if r.get("kernel_path", "").strip() != "specialized_rowpack16":
            errs.append(f"kernel_path={r.get('kernel_path')!r}")
        if r.get("gpu_tag", "").strip() != "H200":
            errs.append(f"gpu_tag={r.get('gpu_tag')!r}")
        if r.get("sum_reference_domain", "").strip() != "encoded_full_depth":
            errs.append(f"sum_reference_domain={r.get('sum_reference_domain')!r}")
        if errs:
            print(f"GATE FAIL: row {i} ({r['dataset']}_{r['artifact_label']} k={r['k']}): {'; '.join(errs)}", file=sys.stderr)
            gates_ok = False

    if gates_ok:
        print("GATE OK: all joined rows pass field-level validation")

    # Write
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        w.writerows(joined)
    print(f"Wrote {OUT_PATH} ({len(joined)} rows)")

    if not gates_ok:
        print("\nOne or more gates FAILED.", file=sys.stderr)
        return 1

    print("\nAll gates PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
