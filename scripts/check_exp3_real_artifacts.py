#!/usr/bin/env python3
"""Check Exp3 real-data artifact min/max and rectangular plane layout."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_histogram(text: str) -> dict[int, int]:
    hist: dict[int, int] = {}
    if not text:
        return hist
    for item in text.split(","):
        key, value = item.strip().split(":")
        hist[int(key)] = int(value)
    return hist


def check_dataset(root: Path) -> dict[str, str]:
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    segment_meta_path = root / "segment_meta.csv"

    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())

    with segment_meta_path.open(newline="") as f:
        segments = list(csv.DictReader(f))

    active_counts = [int(row["active_plane_count"]) for row in segments]
    hist = Counter(active_counts)
    observed_min = min(active_counts)
    observed_max = max(active_counts)

    expected_value_count = int(manifest["value_count"])
    max_plane_count = int(manifest["max_plane_count"])
    plane_files = sorted(root.glob("plane_*.bin"))
    plane_sizes = [path.stat().st_size for path in plane_files]
    plane_size_ok = all(size == expected_value_count for size in plane_sizes)

    manifest_ok = (
        manifest["dataset"] == summary["dataset"]
        and int(manifest["segment_count"]) == len(segments)
        and int(manifest["segment_count"]) == int(summary["segment_count"])
        and int(manifest["segment_plane_count_min"]) == observed_min
        and int(manifest["segment_plane_count_max"]) == observed_max
        and int(summary["segment_plane_count_min"]) == observed_min
        and int(summary["segment_plane_count_max"]) == observed_max
        and int(summary["max_plane_count"]) == max_plane_count
        and len(plane_files) == max_plane_count
        and plane_size_ok
        and parse_histogram(summary.get("plane_count_histogram", "")) == dict(sorted(hist.items()))
    )

    return {
        "dataset": str(manifest["dataset"]),
        "value_count": str(expected_value_count),
        "segment_count": str(len(segments)),
        "max_plane_count": str(max_plane_count),
        "observed_segment_plane_count_min": str(observed_min),
        "observed_segment_plane_count_max": str(observed_max),
        "plane_count_histogram": ", ".join(f"{k}:{v}" for k, v in sorted(hist.items())),
        "plane_file_count": str(len(plane_files)),
        "plane_file_bytes_min": str(min(plane_sizes) if plane_sizes else 0),
        "plane_file_bytes_max": str(max(plane_sizes) if plane_sizes else 0),
        "rectangular_plane_layout": str(plane_size_ok).lower(),
        "artifact_consistent": str(manifest_ok).lower(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/work/u4063895/datasets/synthetic/dev_buff_exp3"),
    )
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    rows = [check_dataset(path) for path in sorted(args.root.iterdir()) if path.is_dir()]
    if not rows:
        raise SystemExit(f"no dataset directories found under {args.root}")

    fieldnames = list(rows[0].keys())
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if not all(row["artifact_consistent"] == "true" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
