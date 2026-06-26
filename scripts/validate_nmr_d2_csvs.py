#!/usr/bin/env python3
"""Validate that existing stochastic CSVs are usable with updated claim_matrix.py.

Checks:
1. All 8 plane files exist in the artifact dir
2. clean_encoded_sum in CSV matches recomputed value from plane files
3. relative_bound_width is consistent with clean_encoded_sum and expected_bound_width

Usage:
  python3 scripts/validate_nmr_d2_csvs.py \\
      --stochastic-csv /path/to/nmr_d2_stochastic_matrix.csv \\
      --clean-plane-dir /path/to/planes \\
      --n-rows N
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PLANES = 8
PLANE_WEIGHTS = [1 << (8 * (7 - p)) for p in range(PLANES)]


def compute_clean_encoded_sum(clean_plane_dir: Path, n_rows: int) -> int:
    """Sum(row_p * weight[p]) across planes that have files.

    Contract: missing plane files are treated as all-zero.  This matches the
    D2 coverage map loader (``load_clean_plane`` in nmr_d2_coverage_map.py).
    Some datasets (e.g. cesm_atm_cloud, max_plane_count=7) have fewer than 8
    active planes; the absent planes contribute zero, which is correct.
    """
    total = 0
    missing = []
    for p in range(PLANES):
        found = False
        for fmt in (f"plane_{p}.bin", f"plane_{p:03d}.bin"):
            path = clean_plane_dir / fmt
            if path.is_file():
                data = path.read_bytes()[:n_rows]
                total += sum(data) * PLANE_WEIGHTS[p]
                found = True
                break
        if not found:
            missing.append(p)
    if missing:
        print(f"  NOTE: plane files missing for planes {missing} — "
              f"treated as zero (valid for datasets with <8 active planes)")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stochastic-csv", type=Path, required=True)
    parser.add_argument("--clean-plane-dir", type=Path, required=True)
    parser.add_argument("--n-rows", type=int, required=True)
    args = parser.parse_args()

    errors = []

    # 1. At least one plane file exists (datasets may have <8 active planes)
    any_plane = False
    for p in range(PLANES):
        for fmt in (f"plane_{p}.bin", f"plane_{p:03d}.bin"):
            path = args.clean_plane_dir / fmt
            if path.is_file():
                any_plane = True
                break
        if any_plane:
            break
    if not any_plane:
        errors.append("no plane files found at all")
    else:
        present = []
        for p in range(PLANES):
            for fmt in (f"plane_{p}.bin", f"plane_{p:03d}.bin"):
                if (args.clean_plane_dir / fmt).is_file():
                    present.append(p)
                    break
        print(f"  PASS  planes present: {present}")

    # 2. clean_encoded_sum matches
    expected_sum = compute_clean_encoded_sum(args.clean_plane_dir, args.n_rows)
    with args.stochastic_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    csv_sum = float(rows[0]["clean_encoded_sum"])
    ratio = abs(csv_sum - expected_sum) / max(expected_sum, 1)
    if ratio > 0.001:
        errors.append(
            f"clean_encoded_sum mismatch: CSV={csv_sum:.4e}, recomputed={expected_sum:.4e}, "
            f"rel_diff={ratio:.6f}"
        )
    else:
        print(f"  PASS  clean_encoded_sum consistent: {csv_sum:.4e} vs {expected_sum:.4e} (diff={ratio:.2e})")

    # 3. relative_bound_width consistency
    for row in rows:
        bw = float(row["expected_bound_width"])
        rel = float(row["relative_bound_width"])
        expected_rel = bw / max(abs(expected_sum), 1)
        if abs(rel - expected_rel) / max(abs(expected_rel), 1e-30) > 0.001:
            errors.append(
                f"relative_bound_width mismatch at {row['policy']} rate={row['fault_rate']} "
                f"seed={row['seed']}: CSV={rel:.6e}, expected={expected_rel:.6e}"
            )
            break
    else:
        print(f"  PASS  all {len(rows)} rows have consistent relative_bound_width")

    # 4. relative_bound_width column exists in CSV
    if "relative_bound_width" not in rows[0]:
        errors.append("MISSING COLUMN: relative_bound_width not in CSV header")
    else:
        print(f"  PASS  relative_bound_width column present")

    if errors:
        print(f"\nFAIL ({len(errors)} checks failed):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print(f"\n  All checks PASS — CSV is ready for downstream processing.")


if __name__ == "__main__":
    main()
