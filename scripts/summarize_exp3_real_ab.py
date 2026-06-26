#!/usr/bin/env python3
"""Summarize Exp3 real-data runtime vs specialized A/B CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_one(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise SystemExit(f"expected exactly one data row in {path}, found {len(rows)}")
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    runtime = read_one(args.run_dir / "real_uniform_depth9_runtime.csv")
    specialized = read_one(args.run_dir / "real_uniform_depth9_specialized.csv")

    runtime_ms = float(runtime["ms_per_iter"])
    specialized_ms = float(specialized["ms_per_iter"])
    runtime_gbps = float(runtime["logical_GBps"])
    specialized_gbps = float(specialized["logical_GBps"])

    speedup = runtime_ms / specialized_ms
    gbps_ratio = specialized_gbps / runtime_gbps

    lines = [
        f"run_dir,{args.run_dir}",
        f"runtime_ms,{runtime_ms:.6f}",
        f"specialized_ms,{specialized_ms:.6f}",
        f"runtime_logical_GBps,{runtime_gbps:.3f}",
        f"specialized_logical_GBps,{specialized_gbps:.3f}",
        f"speedup_ms_ratio,{speedup:.3f}",
        f"speedup_GBps_ratio,{gbps_ratio:.3f}",
        f"runtime_gpu_sum,{runtime['gpu_approximate_sum']}",
        f"specialized_gpu_sum,{specialized['gpu_approximate_sum']}",
        f"runtime_abs_exact_gpu_diff,{runtime['abs_exact_gpu_diff']}",
        f"specialized_abs_exact_gpu_diff,{specialized['abs_exact_gpu_diff']}",
    ]
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
