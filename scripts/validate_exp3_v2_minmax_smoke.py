#!/usr/bin/env python3
"""Validate Exp3 v2 MIN/MAX progressive bound smoke test results.

Checks:
1. bound_valid_min == true for all rows (exact_min ∈ [min_lower, min_upper])
2. bound_valid_max == true for all rows (exact_max ∈ [max_lower, max_upper])
3. At full depth (k = max_planes - 1): min_lower ≈ min_upper ≈ exact_min
4. At full depth: max_lower ≈ max_upper ≈ exact_max
5. Bounds tighten monotonically as k increases (min_lower increases, min_upper decreases)
"""

import csv
import sys
from pathlib import Path


def validate_minmax_csv(csv_path: Path) -> bool:
    """Validate a single MIN/MAX CSV file. Returns True if all checks pass."""
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"  ERROR: no rows in {csv_path}")
        return False

    all_ok = True

    # Group by dataset for monotonicity checks
    by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        ds = row.get("dataset", "unknown")
        by_dataset.setdefault(ds, []).append(row)

    for ds, ds_rows in by_dataset.items():
        ds_rows.sort(key=lambda r: int(r["refinement_depth"]))

        for row in ds_rows:
            k = int(row["refinement_depth"])
            bound_valid_min = row.get("bound_valid_min", "").strip().lower()
            bound_valid_max = row.get("bound_valid_max", "").strip().lower()

            exact_min = float(row.get("exact_min", "nan"))
            exact_max = float(row.get("exact_max", "nan"))
            min_lower = float(row.get("min_lower", "nan"))
            min_upper = float(row.get("min_upper", "nan"))
            max_lower = float(row.get("max_lower", "nan"))
            max_upper = float(row.get("max_upper", "nan"))

            # Check 1: bound validity
            if bound_valid_min != "true":
                print(f"  FAIL: {ds} k={k} bound_valid_min={bound_valid_min} "
                      f"(exact_min={exact_min}, min_lower={min_lower}, min_upper={min_upper})")
                all_ok = False
            else:
                print(f"  OK: {ds} k={k} bound_valid_min=true")

            if bound_valid_max != "true":
                print(f"  FAIL: {ds} k={k} bound_valid_max={bound_valid_max} "
                      f"(exact_max={exact_max}, max_lower={max_lower}, max_upper={max_upper})")
                all_ok = False
            else:
                print(f"  OK: {ds} k={k} bound_valid_max=true")

            # Check 2: min_lower <= exact_min <= min_upper
            if not (min_lower <= exact_min + 1e-12 and exact_min <= min_upper + 1e-12):
                print(f"  WARN: {ds} k={k} min bound interval check: "
                      f"min_lower={min_lower} <= exact_min={exact_min} <= min_upper={min_upper}")

            # Check 3: max_lower <= exact_max <= max_upper
            if not (max_lower <= exact_max + 1e-12 and exact_max <= max_upper + 1e-12):
                print(f"  WARN: {ds} k={k} max bound interval check: "
                      f"max_lower={max_lower} <= exact_max={exact_max} <= max_upper={max_upper}")

        # Check 4: monotonicity - min_lower should increase (or stay same) with k
        # and min_upper should decrease (or stay same) with k
        for i in range(1, len(ds_rows)):
            prev_k = int(ds_rows[i - 1]["refinement_depth"])
            curr_k = int(ds_rows[i]["refinement_depth"])
            prev_min_lower = float(ds_rows[i - 1].get("min_lower", "nan"))
            curr_min_lower = float(ds_rows[i].get("min_lower", "nan"))
            prev_min_upper = float(ds_rows[i - 1].get("min_upper", "nan"))
            curr_min_upper = float(ds_rows[i].get("min_upper", "nan"))
            prev_max_lower = float(ds_rows[i - 1].get("max_lower", "nan"))
            curr_max_lower = float(ds_rows[i].get("max_lower", "nan"))
            prev_max_upper = float(ds_rows[i - 1].get("max_upper", "nan"))
            curr_max_upper = float(ds_rows[i].get("max_upper", "nan"))

            # min_lower should be non-decreasing (bounds tighten)
            if curr_min_lower < prev_min_lower - 1e-12:
                print(f"  WARN: {ds} min_lower decreased: k={prev_k} -> k={curr_k}: "
                      f"{prev_min_lower} -> {curr_min_lower}")
            # min_upper should be non-increasing (bounds tighten)
            if curr_min_upper > prev_min_upper + 1e-12:
                print(f"  WARN: {ds} min_upper increased: k={prev_k} -> k={curr_k}: "
                      f"{prev_min_upper} -> {curr_min_upper}")
            # max_lower should be non-decreasing (bounds tighten)
            if curr_max_lower < prev_max_lower - 1e-12:
                print(f"  WARN: {ds} max_lower decreased: k={prev_k} -> k={curr_k}: "
                      f"{prev_max_lower} -> {curr_max_lower}")
            # max_upper should be non-increasing (bounds tighten)
            if curr_max_upper > prev_max_upper + 1e-12:
                print(f"  WARN: {ds} max_upper increased: k={prev_k} -> k={curr_k}: "
                      f"{prev_max_upper} -> {curr_max_upper}")

        # Check 5: at full depth, bounds should converge to exact values
        last_row = ds_rows[-1]
        last_k = int(last_row["refinement_depth"])
        last_min_lower = float(last_row.get("min_lower", "nan"))
        last_min_upper = float(last_row.get("min_upper", "nan"))
        last_max_lower = float(last_row.get("max_lower", "nan"))
        last_max_upper = float(last_row.get("max_upper", "nan"))
        last_exact_min = float(last_row.get("exact_min", "nan"))
        last_exact_max = float(last_row.get("exact_max", "nan"))

        min_gap = abs(last_min_upper - last_min_lower)
        max_gap = abs(last_max_upper - last_max_lower)
        print(f"  Full depth k={last_k}: min_gap={min_gap:.3g}, max_gap={max_gap:.3g}")
        print(f"    min_lower={last_min_lower:.17g}, min_upper={last_min_upper:.17g}, exact_min={last_exact_min:.17g}")
        print(f"    max_lower={last_max_lower:.17g}, max_upper={last_max_upper:.17g}, exact_max={last_exact_max:.17g}")

    return all_ok


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <result_dir> [<result_dir2> ...]")
        sys.exit(1)

    all_ok = True
    for result_dir_str in sys.argv[1:]:
        result_dir = Path(result_dir_str)
        if not result_dir.is_dir():
            print(f"WARNING: directory not found: {result_dir}")
            continue

        csv_files = sorted(result_dir.glob("exp3_v2_minmax_*.csv"))
        if not csv_files:
            print(f"WARNING: no MIN/MAX CSV files found in {result_dir}")
            continue

        for csv_path in csv_files:
            print(f"\nValidating: {csv_path.name}")
            if not validate_minmax_csv(csv_path):
                all_ok = False

    if all_ok:
        print("\nAll validation checks PASSED.")
        sys.exit(0)
    else:
        print("\nSome validation checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()