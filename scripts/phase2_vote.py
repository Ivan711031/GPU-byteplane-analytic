#!/usr/bin/env python3
"""Voting helper with outcome instrumentation for Reliability Phase 2.

Given r_p replicas of each plane p, produces the majority-voted byte array
and per-byte outcome classification (resolved_correctly, detected_mismatch,
undetected_corruption).

Tie-break rule: when no strict majority exists, pick the lexicographically
smallest byte value among the tied candidates.

Usage:
  python3 scripts/phase2_vote.py \\
      --replica-dir /path/to/faulted/planes \\
      --clean-dir /path/to/clean/planes \\
      --n-rows 1000 \\
      --output-csv /tmp/vote_outcomes.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def vote_byte(replica_values: list[int]) -> int:
    """Vote a single byte position across r_p replicas.

    Args:
        replica_values: list of byte values (0-255) from each replica.

    Returns:
        Voted byte value.

    Tie-break: when no value has > r_p/2 votes, pick the lexicographically
    smallest among the tied (most-voted) candidates.
    """
    if not replica_values:
        raise ValueError("cannot vote on empty replica list")
    cnt = Counter(replica_values)
    max_count = max(cnt.values())
    r = len(replica_values)
    if max_count > r / 2:
        return next(v for v, c in cnt.items() if c == max_count)
    tied = sorted(v for v, c in cnt.items() if c == max_count)
    return tied[0]


def vote_plane(
    replica_bytes: list[bytes], clean_bytes: bytes
) -> tuple[bytes, dict[str, int]]:
    """Vote r_p replicas of a plane.

    Args:
        replica_bytes: list of r_p byte arrays, each length = n_rows.
        clean_bytes: original fault-free byte array (for outcome classification).

    Returns:
        voted_bytes: majority-voted byte array (length = n_rows).
        stats: dict with counts of:
            - resolved_correctly
            - detected_mismatch
            - undetected_corruption

    Raises:
        ValueError: if replica_bytes is empty or lengths mismatch.
    """
    if not replica_bytes:
        raise ValueError("replica_bytes must not be empty")
    n = len(clean_bytes)
    for rb in replica_bytes:
        if len(rb) != n:
            raise ValueError(
                f"replica length {len(rb)} != clean length {n}"
            )

    voted = bytearray(n)
    resolved = 0
    detected = 0
    undetected = 0

    for i, values in enumerate(zip(*replica_bytes)):
        cb = clean_bytes[i]
        vb = vote_byte(list(values))
        voted[i] = vb
        if vb == cb:
            resolved += 1
        elif cb in values:
            detected += 1
        else:
            undetected += 1

    stats = {
        "resolved_correctly": resolved,
        "detected_mismatch": detected,
        "undetected_corruption": undetected,
    }
    return bytes(voted), stats


def _discover_planes(clean_dir: Path) -> dict[int, Path]:
    """Discover plane files in clean-dir named plane_{p}.bin."""
    planes: dict[int, Path] = {}
    for f in sorted(clean_dir.glob("plane_*.bin")):
        try:
            p = int(f.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        planes[p] = f
    return planes


def _discover_replicas(
    replica_dir: Path, plane_id: int
) -> list[Path]:
    """Discover replica files for a given plane in replica-dir.

    Files named plane_{plane_id}_replica_*.bin.
    """
    pattern = f"plane_{plane_id}_replica_*.bin"
    return sorted(replica_dir.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replica-dir", type=Path, required=True,
        help="directory containing plane_{p}_replica_{r}.bin files",
    )
    parser.add_argument(
        "--clean-dir", type=Path, required=True,
        help="directory containing plane_{p}.bin files",
    )
    parser.add_argument(
        "--n-rows", type=int, required=True,
        help="number of rows (bytes per plane)",
    )
    parser.add_argument(
        "--output-csv", type=Path,
        default=Path("/tmp/vote_outcomes.csv"),
        help="output CSV path",
    )
    args = parser.parse_args()

    clean_dir: Path = args.clean_dir
    replica_dir: Path = args.replica_dir
    n_rows = args.n_rows
    csv_path: Path = args.output_csv

    if not clean_dir.is_dir():
        print(f"clean-dir not found: {clean_dir}", file=sys.stderr)
        sys.exit(1)
    if not replica_dir.is_dir():
        print(f"replica-dir not found: {replica_dir}", file=sys.stderr)
        sys.exit(1)

    planes = _discover_planes(clean_dir)
    if not planes:
        print("No plane_{p}.bin files found in clean-dir", file=sys.stderr)
        sys.exit(1)

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "plane_id", "n_rows", "r_p",
        "resolved_correctly", "detected_mismatch",
        "undetected_corruption", "total",
        "resolved_frac", "detected_frac", "undetected_frac",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for plane_id in sorted(planes):
            clean_file = planes[plane_id]
            clean_bytes = clean_file.read_bytes()
            if len(clean_bytes) != n_rows:
                print(
                    f"plane {plane_id}: clean length {len(clean_bytes)} "
                    f"!= n_rows {n_rows}",
                    file=sys.stderr,
                )
                sys.exit(1)

            replica_files = _discover_replicas(replica_dir, plane_id)
            if not replica_files:
                print(
                    f"plane {plane_id}: no replica files found",
                    file=sys.stderr,
                )
                continue

            replica_bytes = [f.read_bytes() for f in replica_files]
            r_p = len(replica_bytes)
            for i, rb in enumerate(replica_bytes):
                if len(rb) != n_rows:
                    print(
                        f"plane {plane_id}, replica {i}: length "
                        f"{len(rb)} != n_rows {n_rows}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            voted, stats = vote_plane(replica_bytes, clean_bytes)
            total = n_rows
            row = {
                "plane_id": str(plane_id),
                "n_rows": str(n_rows),
                "r_p": str(r_p),
                "resolved_correctly": str(stats["resolved_correctly"]),
                "detected_mismatch": str(stats["detected_mismatch"]),
                "undetected_corruption": str(stats["undetected_corruption"]),
                "total": str(total),
                "resolved_frac": f"{stats['resolved_correctly'] / total:.6f}",
                "detected_frac": f"{stats['detected_mismatch'] / total:.6f}",
                "undetected_frac": f"{stats['undetected_corruption'] / total:.6f}",
            }
            writer.writerow(row)
            print(
                f"plane {plane_id}: r_p={r_p} "
                f"resolved={stats['resolved_correctly']} "
                f"detected={stats['detected_mismatch']} "
                f"undetected={stats['undetected_corruption']}"
            )

    print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    main()
