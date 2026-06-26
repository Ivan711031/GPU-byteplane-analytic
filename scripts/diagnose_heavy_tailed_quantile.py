#!/usr/bin/env python3
"""Compute selectivity thresholds for a heavy-tailed raw f64le binary file.

Usage:
    python scripts/diagnose_heavy_tailed_quantile.py
    python scripts/diagnose_heavy_tailed_quantile.py --limit 100000
    python scripts/diagnose_heavy_tailed_quantile.py --selectivities 10,50,90,99,99.9
"""

from __future__ import annotations

import argparse

import numpy as np

RAW_DEFAULT = "/work/u4063895/datasets/synthetic/dev/heavy_tailed.f64le.bin"


def main():
    parser = argparse.ArgumentParser(
        description="Compute selectivity thresholds for a heavy-tailed f64le binary file"
    )
    parser.add_argument("--raw", default=RAW_DEFAULT, help="Path to f64le binary file")
    parser.add_argument(
        "--selectivities", default="50,90,99",
        help="Comma-separated selectivity percentages (default: 50,90,99)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional: only read first N rows (smoke test)"
    )
    args = parser.parse_args()

    raw_path = args.raw
    selectivities = [float(s.strip()) for s in args.selectivities.split(",")]

    if args.limit is not None:
        raw = np.fromfile(raw_path, dtype="<f8", count=args.limit)
        print(f"Loaded {len(raw)} rows (limit={args.limit})")
    else:
        raw = np.memmap(raw_path, dtype="<f8", mode="r")
        print(f"Memmap {len(raw)} rows (full file)")

    total = len(raw)

    print(f"\n{'sel%':>8s}  {'threshold':>20s}  {'count':>12s}  {'actual_sel%':>12s}")
    print("-" * 60)
    for sel in selectivities:
        q = 1.0 - sel / 100.0
        threshold = float(np.quantile(raw, q, method="linear"))
        cnt = int(np.count_nonzero(raw > threshold))
        actual_sel = cnt / total * 100.0
        print(f"{sel:8.4g}  {threshold:20.10g}  {cnt:12d}  {actual_sel:12.6f}")


if __name__ == "__main__":
    main()
