#!/usr/bin/env python3
"""Aggregate Exp3 v2 full specialized SUM throughput matrix.

Discover run directories under results/exp3_v2_full_specialized_sum,
concatenate all throughput.csv into a single file, and emit
coverage and correctness summaries with hard gate enforcement.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

# ------------------------------------------------------------------
# Expected matrix (dataset, artifact_label, max_plane_count)
# 19 unique combos, 103 total rows (k=1..max_plane_count each)
# ------------------------------------------------------------------
EXPECTED = [
    ("heavy_tailed", "p2", 7),
    ("heavy_tailed", "p3", 7),
    ("heavy_tailed", "p4", 7),
    ("heavy_tailed", "p5", 8),
    ("heavy_tailed", "p6", 8),
    ("sensor", "p2", 2),
    ("sensor", "p4", 3),
    ("sensor", "p6", 3),
    ("sensor", "p8", 4),
    ("sensor", "p10", 5),
    ("uniform", "p2", 3),
    ("uniform", "p4", 4),
    ("uniform", "p6", 4),
    ("uniform", "p8", 5),
    ("uniform", "p10", 6),
    ("zipfian", "p2", 6),
    ("zipfian", "p4", 6),
    ("zipfian", "p6", 7),
    ("zipfian", "p8", 8),
]

EXPECTED_COMBOS = set((d, a) for d, a, _ in EXPECTED)
EXPECTED_ROWS = sum(m for _, _, m in EXPECTED)


def _artifact_numeric(artifact_label: str) -> str:
    """Extract numeric suffix from pN label: 'p2' -> '2', 'p10' -> '10'."""
    return artifact_label[1:] if artifact_label.startswith("p") else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-base",
        type=Path,
        default=Path("results/exp3_v2_full_specialized_sum"),
        help="Base directory containing run directories (default: results/exp3_v2_full_specialized_sum)",
    )
    parser.add_argument(
        "--out-all",
        type=Path,
        default=Path("results/exp3_v2_full_specialized_sum/exp3_v2_full_sum_throughput_all.csv"),
        help="Output concatenated CSV path",
    )
    parser.add_argument(
        "--out-coverage",
        type=Path,
        default=Path("results/exp3_v2_full_specialized_sum/coverage_summary.csv"),
        help="Output coverage summary path",
    )
    parser.add_argument(
        "--out-correctness",
        type=Path,
        default=Path("results/exp3_v2_full_specialized_sum/correctness_summary.csv"),
        help="Output correctness summary path",
    )
    args = parser.parse_args()

    results_base = args.results_base.resolve()
    if not results_base.is_dir():
        print(f"ERROR: results base directory not found: {results_base}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Discover run directories: any dir containing throughput.csv
    # ------------------------------------------------------------------
    all_rows: list[dict] = []
    run_dirs_found: list[Path] = []

    for candidate in sorted(results_base.iterdir()):
        if not candidate.is_dir():
            continue
        csv_path = candidate / "throughput.csv"
        if not csv_path.is_file():
            continue
        run_dirs_found.append(candidate)
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_run_dir"] = str(candidate)
                all_rows.append(row)

    if not all_rows:
        print(f"ERROR: no throughput.csv files found under {results_base}", file=sys.stderr)
        return 2

    print(f"Found {len(run_dirs_found)} run directories, {len(all_rows)} total rows")

    # ------------------------------------------------------------------
    # Concatenate and write all-rows CSV
    # ------------------------------------------------------------------
    if all_rows and "_run_dir" in all_rows[0]:
        # Determine fieldnames: remove internal _run_dir, keep everything else
        sample = all_rows[0]
        fieldnames = [k for k in sample.keys() if k != "_run_dir"]
    else:
        fieldnames = list(all_rows[0].keys())

    args.out_all.parent.mkdir(parents=True, exist_ok=True)
    with args.out_all.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            out_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out_row)
    print(f"Wrote concatenated CSV: {args.out_all} ({len(all_rows)} rows)")

    # ------------------------------------------------------------------
    # Gate checks
    # ------------------------------------------------------------------
    gates_ok = True

    # 1) 19 unique (dataset, artifact_label)
    combos_seen = set()
    for row in all_rows:
        key = (row.get("dataset", "").strip(), row.get("artifact_label", "").strip())
        if key != ("", ""):
            combos_seen.add(key)

    if len(combos_seen) != len(EXPECTED_COMBOS):
        missing = EXPECTED_COMBOS - combos_seen
        extra = combos_seen - EXPECTED_COMBOS
        print(f"GATE FAIL: expected {len(EXPECTED_COMBOS)} unique combos, got {len(combos_seen)}", file=sys.stderr)
        if missing:
            print(f"  missing: {sorted(missing)}", file=sys.stderr)
        if extra:
            print(f"  extra:   {sorted(extra)}", file=sys.stderr)
        gates_ok = False
    else:
        print(f"GATE OK: {len(combos_seen)} unique (dataset, artifact_label) combos")

    # 2) 103 total rows
    if len(all_rows) != EXPECTED_ROWS:
        print(f"GATE FAIL: expected {EXPECTED_ROWS} rows, got {len(all_rows)}", file=sys.stderr)
        gates_ok = False
    else:
        print(f"GATE OK: {EXPECTED_ROWS} total rows")

    # 3) Every expected (dataset, artifact_label, k=1..max_plane_count) present, no duplicates
    observed_ks: dict[tuple[str, str], set[int]] = defaultdict(set)
    duplicate_rows = 0
    for row in all_rows:
        ds = row.get("dataset", "").strip()
        al = row.get("artifact_label", "").strip()
        k_str = row.get("k", "").strip()
        try:
            k_val = int(k_str)
        except (ValueError, TypeError):
            k_val = -1

        key = (ds, al)
        if k_val in observed_ks[key]:
            duplicate_rows += 1
        observed_ks[key].add(k_val)

    if duplicate_rows > 0:
        print(f"GATE FAIL: found {duplicate_rows} duplicate keys", file=sys.stderr)
        gates_ok = False

    for ds, al, mpc in EXPECTED:
        expected_ks = set(range(1, mpc + 1))
        actual_ks = observed_ks.get((ds, al), set())
        if actual_ks != expected_ks:
            print(f"GATE FAIL: {ds}_{al} expected k={sorted(expected_ks)}, got k={sorted(actual_ks)}", file=sys.stderr)
            gates_ok = False

    if gates_ok:
        print("GATE OK: all k-ranges present with no duplicates")

    # 4-10) Row-level gates
    for i, row in enumerate(all_rows):
        ds = row.get("dataset", "").strip()
        al = row.get("artifact_label", "").strip()
        er = row.get("expected_encoding_root", "") or row.get("encoded_root", "")
        er = er.strip()

        if row.get("validated", "").strip().lower() != "true":
            print(f"GATE FAIL: row {i} ({ds}_{al}) validated={row.get('validated')!r}", file=sys.stderr)
            gates_ok = False

        if row.get("kernel_path", "").strip() != "specialized_rowpack16":
            print(f"GATE FAIL: row {i} ({ds}_{al}) kernel_path={row.get('kernel_path')!r}", file=sys.stderr)
            gates_ok = False

        if row.get("mode", "").strip() != "encoded_dev_subcolumns":
            print(f"GATE FAIL: row {i} ({ds}_{al}) mode={row.get('mode')!r}", file=sys.stderr)
            gates_ok = False

        if row.get("gpu_tag", "").strip() != "H200":
            print(f"GATE FAIL: row {i} ({ds}_{al}) gpu_tag={row.get('gpu_tag')!r}", file=sys.stderr)
            gates_ok = False

        if "dev_buff_v2_20260510/exp_runtime_by_p" not in er:
            print(f"GATE FAIL: row {i} ({ds}_{al}) encoded_root missing dev_buff_v2_20260510/exp_runtime_by_p", file=sys.stderr)
            gates_ok = False

        expected_suffix = f"{ds}_{al}"
        if not er.endswith(expected_suffix):
            print(f"GATE FAIL: row {i} ({ds}_{al}) encoded_root does not end with {expected_suffix!r}: {er!r}", file=sys.stderr)
            gates_ok = False

        if row.get("artifact_version", "").strip() != "v2_2026-05-10":
            print(f"GATE FAIL: row {i} ({ds}_{al}) artifact_version={row.get('artifact_version')!r}", file=sys.stderr)
            gates_ok = False

        if row.get("artifact_label", "").strip() != al:
            print(f"GATE FAIL: row {i} ({ds}_{al}) artifact_label={row.get('artifact_label')!r}", file=sys.stderr)
            gates_ok = False

        expected_precision = _artifact_numeric(al)
        if row.get("precision_power", "").strip() != expected_precision:
            print(f"GATE FAIL: row {i} ({ds}_{al}) precision_power={row.get('precision_power')!r}, expected={expected_precision!r}", file=sys.stderr)
            gates_ok = False

        if row.get("sum_reference_domain", "").strip() != "encoded_full_depth":
            print(f"GATE FAIL: row {i} ({ds}_{al}) sum_reference_domain={row.get('sum_reference_domain')!r}", file=sys.stderr)
            gates_ok = False

    # ------------------------------------------------------------------
    # Coverage summary CSV
    # ------------------------------------------------------------------
    cov_rows = []
    for ds, al, mpc in EXPECTED:
        actual_ks = observed_ks.get((ds, al), set())
        expected_ks = set(range(1, mpc + 1))
        missing = sorted(expected_ks - actual_ks)
        extra = sorted(actual_ks - expected_ks)
        cov_rows.append({
            "dataset": ds,
            "artifact_label": al,
            "expected_max_k": mpc,
            "expected_rows": mpc,
            "observed_rows": len(actual_ks),
            "missing_k": ",".join(str(k) for k in missing) if missing else "",
            "extra_k": ",".join(str(k) for k in extra) if extra else "",
            "complete": str(actual_ks == expected_ks).lower(),
        })

    args.out_coverage.parent.mkdir(parents=True, exist_ok=True)
    with args.out_coverage.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "artifact_label", "expected_max_k", "expected_rows",
            "observed_rows", "missing_k", "extra_k", "complete",
        ])
        writer.writeheader()
        writer.writerows(cov_rows)
    print(f"Wrote coverage summary: {args.out_coverage}")

    # ------------------------------------------------------------------
    # Correctness summary CSV
    # ------------------------------------------------------------------
    corr_rows = []
    for ds, al, mpc in EXPECTED:
        subset = [r for r in all_rows
                  if r.get("dataset", "").strip() == ds
                  and r.get("artifact_label", "").strip() == al]
        n = len(subset)
        all_valid = all(r.get("validated", "").strip().lower() == "true" for r in subset)
        corr_rows.append({
            "dataset": ds,
            "artifact_label": al,
            "rows": n,
            "all_validated_true": str(all_valid).lower(),
            "abs_exact_gpu_diff_max": max(
                (r.get("abs_exact_gpu_diff", "") for r in subset),
                key=lambda x: float(x) if x else 0.0,
                default="",
            ),
            "abs_cpu_gpu_diff_max": max(
                (r.get("abs_cpu_gpu_diff", "") for r in subset),
                key=lambda x: float(x) if x else 0.0,
                default="",
            ),
        })

    args.out_correctness.parent.mkdir(parents=True, exist_ok=True)
    with args.out_correctness.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "artifact_label", "rows", "all_validated_true",
            "abs_exact_gpu_diff_max", "abs_cpu_gpu_diff_max",
        ])
        writer.writeheader()
        writer.writerows(corr_rows)
    print(f"Wrote correctness summary: {args.out_correctness}")

    if not gates_ok:
        print("\nOne or more gates FAILED. See errors above.", file=sys.stderr)
        return 3

    print("\nAll gates PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
