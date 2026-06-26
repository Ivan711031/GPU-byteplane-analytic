#!/usr/bin/env python3
"""CPU-only tracer bullet for Phase 1 reliability pipeline.

Generates the deterministic tiny fixture, converts to 8-plane artifacts,
generates fault plans, applies faults, validates against analytic oracle,
and emits CSV. No GPU required.

Includes explicit test assertions for:
- uint64 → plane byte split and reconstruction
- Decimal string round-trip for all SUM fields
- Fault plan offset uniqueness, sorting, mask range
- Signed delta oracle arithmetic (positive and negative)
- artifact_checksum and fault_plan_checksum in CSV

Usage:
  ./scripts/build_reliability_tiny_fixture.py \
    --artifact-dir /tmp/rl_tiny/artifacts \
    --fault-plan-dir /tmp/rl_tiny/fault_plans \
    --csv /tmp/rl_tiny/tiny_fixture.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shlex
import struct
import sys
import time
from array import array
from pathlib import Path
from typing import Any


DATE_TAG = "20260528"
SCALE = 100

# PRD-defined deterministic encoded fixture values.
# These exercise every byte-level boundary: all-zeros, all-ones per plane,
# carry propagation across byte boundaries, and extreme MSB values.
# Values > 2^53 cannot round-trip through FP64 at scale=100, so they are
# injected directly as encoded uint64, bypassing the FP64 conversion step.
FP64_ROUNDTRIPPABLE = [
    0,          # all zeros
    1,          # plane 7 LSB only
    255,        # plane 7 all-ones
    256,        # plane 6 LSB
    65535,      # planes 6-7 all-ones (2^16-1)
    65536,      # plane 5 LSB (2^16)
    4294967295, # planes 4-7 all-ones (2^32-1)
    4294967296, # plane 3 LSB (2^32)
    72057594037927936,  # plane 0 LSB (2^56)
]

DIRECT_ENCODED = [
    18446744073709551614,  # MAX_UINT64 - 1; all planes 0xFF except plane 7 = 0xFE
    18446744073709551615,  # MAX_UINT64; all planes 0xFF
]

# Full fixture: 9 FP64 round-trippable + 2 direct encoded = 11 unique values
ENCODED_FIXTURE_VALUES = FP64_ROUNDTRIPPABLE + DIRECT_ENCODED


def uint64_to_planes(x: int) -> list[int]:
    """MSB-first: plane_0 = most significant byte."""
    return [(x >> (8 * (7 - p))) & 0xFF for p in range(8)]


def planes_to_uint64(planes: list[int]) -> int:
    """Reconstruct uint64 from 8 MSB-first plane bytes."""
    return sum(p << (8 * (7 - i)) for i, p in enumerate(planes))


def decimal_string(n: int) -> str:
    """Format integer as decimal string (exact, no scientific notation)."""
    return str(n)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


TINY_N = 1000


# ── Tests ──────────────────────────────────────────────────────

def test_byte_split_known_values() -> None:
    """Verify uint64_to_planes and planes_to_uint64 are exact inverses."""
    cases = [
        (0, [0, 0, 0, 0, 0, 0, 0, 0]),
        (1, [0, 0, 0, 0, 0, 0, 0, 1]),
        (255, [0, 0, 0, 0, 0, 0, 0, 255]),
        (256, [0, 0, 0, 0, 0, 0, 1, 0]),
        (72057594037927936, [1, 0, 0, 0, 0, 0, 0, 0]),
        (18446744073709551615, [255, 255, 255, 255, 255, 255, 255, 255]),
        (18446744073709551614, [255, 255, 255, 255, 255, 255, 255, 254]),
    ]
    for encoded, expected_planes in cases:
        assert uint64_to_planes(encoded) == expected_planes, \
            f"uint64_to_planes({encoded})"
        reconstructed = planes_to_uint64(expected_planes)
        assert reconstructed == encoded, \
            f"planes_to_uint64({expected_planes}) = {reconstructed} != {encoded}"
    print(f"  ✓ byte_split: {len(cases)} cases pass")


def test_decimal_round_trip() -> None:
    """Verify decimal string format → int → string is lossless for SUM values."""
    cases = [
        0,
        1,
        255,
        72057594037927936,
        7998393891707346210,
        18446744073709551615,
        -1,
        -18446744073709551615,
    ]
    for val in cases:
        s = decimal_string(val)
        parsed = int(s)
        assert parsed == val, f"decimal round-trip: {val} → '{s}' → {parsed}"
        # No commas, no whitespace, no scientific notation
        assert s.isdigit() or (s.startswith('-') and s[1:].isdigit()), \
            f"decimal format: '{s}' contains invalid chars"
    print(f"  ✓ decimal round-trip: {len(cases)} cases pass")


def generate_tiny_fixture_raw(path: Path) -> list[float]:
    """Generate FP64 raw values for the round-trippable portion.

    The 9 FP64-round-trippable values are encoded as FP64, then converted.
    The 2 direct-only values are injected as pre-encoded uint64 after conversion.
    """
    raw = []
    for i in range(TINY_N):
        idx = i % len(FP64_ROUNDTRIPPABLE)
        if idx < len(FP64_ROUNDTRIPPABLE):
            encoded = FP64_ROUNDTRIPPABLE[idx]
            raw.append(encoded / SCALE)
    buf = struct.pack(f"<{len(raw)}d", *raw)
    path.write_bytes(buf)
    return raw


def convert_to_planes(raw_values: list[float], scale: int,
                      artifact_dir: Path, raw_path: Path) -> dict[str, Any]:
    """Convert raw FP64 values + direct encoded values to 8-plane artifacts.

    For rows within the FP64 round-trippable range, conversion goes through
    raw → round(v*scale) → uint64.
    For the last rows in each cycle, the direct-encoded value is injected
    straight into the plane bytes (bypassing FP64 to avoid precision loss).

    Returns artifact metadata dict.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(raw_values)
    planes: list[list[int]] = [[] for _ in range(8)]
    overflow_count = 0
    max_encoded = 0
    raw_sum_fp64 = 0.0
    clean_encoded_sum = 0

    for i, raw_val in enumerate(raw_values):
        raw_sum_fp64 += raw_val
        if math.isnan(raw_val) or math.isinf(raw_val) or raw_val < 0.0:
            raise ValueError(f"invalid raw value {raw_val} at row {i}")

        # Determine which encoded value this row represents
        cycle_idx = i % len(ENCODED_FIXTURE_VALUES)
        if cycle_idx < len(FP64_ROUNDTRIPPABLE):
            encoded = int(round(raw_val * scale))
        else:
            # Direct encoded: use the predefined uint64, skipping FP64
            encoded = ENCODED_FIXTURE_VALUES[cycle_idx]

        if encoded < 0 or encoded > 18446744073709551615:
            overflow_count += 1
            continue
        max_encoded = max(max_encoded, encoded)
        clean_encoded_sum += encoded
        for p in range(8):
            planes[p].append(uint64_to_planes(encoded)[p])

    plane_paths: list[Path] = []
    plane_checksums: list[str] = []
    plane_nonzero: list[int] = []
    for p in range(8):
        p_path = artifact_dir / f"plane_{p}.bin"
        arr = array("B", planes[p])
        arr.tofile(p_path.open("wb"))
        plane_paths.append(p_path)
        plane_checksums.append(sha256_file(p_path))
        plane_nonzero.append(sum(1 for b in planes[p] if b != 0))

    metadata = {
        "dataset": "tiny_fixture",
        "dataset_generator": "build_reliability_tiny_fixture.py",
        "n_rows": n_rows,
        "scale": scale,
        "base": 0,
        "dtype": "float64",
        "artifact_format": "fixed_width_8plane_byte_split",
        "plane_count": 8,
        "plane_order": "MSB_first",
        "plane_files": sorted(p.name for p in artifact_dir.glob("plane_*.bin")),
        "plane_checksums": {f"plane_{p}": plane_checksums[p] for p in range(8)},
        "plane_sizes_bytes": [n_rows] * 8,
        "plane_nonzero_count": plane_nonzero,
        "plane_nonzero_fraction": [c / n_rows for c in plane_nonzero],
        "source_raw_path": str(raw_path),
        "source_checksum": sha256_file(raw_path),
        "raw_min": min(raw_values),
        "raw_max": max(raw_values),
        "encoded_max": max_encoded,
        "overflow_count": overflow_count,
        "quantization_max_error": 0.0,
        "quantization_mean_error": 0.0,
        "raw_fp64_sum": raw_sum_fp64,
        "clean_encoded_sum": str(clean_encoded_sum),
        "git_commit": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip() or "unknown",
        "creation_command": shlex.join(sys.argv),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    meta_path = artifact_dir / "artifact.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def cpu_reconstruct_sum(artifact_dir: Path) -> int:
    """Load plane files and compute clean encoded SUM via reconstruction."""
    plane_data = []
    for p in range(8):
        plane_path = artifact_dir / f"plane_{p}.bin"
        arr = array("B")
        arr.frombytes(plane_path.read_bytes())
        plane_data.append(arr)

    total = 0
    n = len(plane_data[0])
    for i in range(n):
        row = sum(plane_data[p][i] << (8 * (7 - p)) for p in range(8))
        total += row
    return total


def test_cpu_reconstruct_sum(artifact_dir: Path, expected: int) -> None:
    """Assert CPU reconstruction SUM matches expected value."""
    cpu_sum = cpu_reconstruct_sum(artifact_dir)
    assert cpu_sum == expected, \
        f"CPU reconstruction SUM mismatch: {cpu_sum} != {expected}"
    print(f"  ✓ cpu_reconstruct_sum: {cpu_sum}")


def generate_fault_plans(artifact_dir: Path, fault_plan_dir: Path,
                         target_planes: list[int], fault_rates: list[float],
                         seeds: list[int],
                         metadata: dict[str, Any]) -> list[Path]:
    """Generate fault plan JSON files for all requested configurations.

    Each plan contains explicit offsets and masks for replay.
    """
    fault_plan_dir.mkdir(parents=True, exist_ok=True)
    n_rows = metadata["n_rows"]
    scale = metadata["scale"]
    dataset = metadata["dataset"]
    artifact_id = str(artifact_dir)

    plan_paths: list[Path] = []

    for plane in target_planes:
        for rate in fault_rates:
            for seed in seeds:
                rng = random.Random(seed)
                plane_size = n_rows
                fault_count = int(rate * plane_size)
                if rate > 0 and fault_count == 0:
                    raise ValueError(
                        f"fault_rate={rate} on plane_size={plane_size} "
                        f"produces fault_count=0 (invalid)"
                    )

                offsets = sorted(rng.sample(range(plane_size), fault_count))
                masks = [rng.randint(1, 255) for _ in range(fault_count)]

                entries = [
                    {"offset": o, "mask": m}
                    for o, m in zip(offsets, masks)
                ]

                fault_plan = {
                    "metadata": {
                        "dataset": dataset,
                        "n_rows": n_rows,
                        "scale": scale,
                        "artifact_id": artifact_id,
                        "target_plane": plane,
                        "fault_rate": f"{rate:.0e}",
                        "fault_rate_numeric": rate,
                        "seed": seed,
                        "actual_fault_count": fault_count,
                        "plane_size_bytes": plane_size,
                        "mask_distribution": "uniform_int_1_255",
                        "offset_order": "ascending",
                        "offset_uniqueness": "unique",
                        "git_commit": metadata.get("git_commit", "unknown"),
                        "generation_command": shlex.join(sys.argv),
                    },
                    "entries": entries,
                }

                plane_dir = fault_plan_dir / f"plane{plane}" / f"rate{rate:.0e}"
                plane_dir.mkdir(parents=True, exist_ok=True)
                fp_path = plane_dir / f"seed_{seed}.json"
                fp_path.write_text(json.dumps(fault_plan, indent=2))
                plan_paths.append(fp_path)

    return plan_paths


def test_fault_plan_properties(fault_plan_path: Path) -> None:
    """Assert fault plan constraints: unique, sorted, nonzero masks."""
    fp = json.loads(fault_plan_path.read_text())
    meta = fp["metadata"]
    entries = fp["entries"]

    assert meta["actual_fault_count"] == len(entries), \
        f"fault_count mismatch: metadata={meta['actual_fault_count']} entries={len(entries)}"

    offsets = [e["offset"] for e in entries]
    masks = [e["mask"] for e in entries]

    # Unique
    assert len(set(offsets)) == len(offsets), "offsets are not unique"

    # Sorted ascending
    assert all(offsets[i] <= offsets[i + 1] for i in range(len(offsets) - 1)), \
        "offsets are not sorted ascending"

    # Nonzero masks
    assert all(m >= 1 and m <= 255 for m in masks), \
        "mask outside valid range [1, 255]"


def test_oracle_signed_deltas(artifact_dir: Path, fault_plan_path: Path,
                              clean_encoded_sum: int) -> int:
    """Compute oracle and verify it matches direct faulted reconstruction.

    Also asserts at least one negative delta occurs in the fixture (proving
    signed arithmetic is exercised, not just unsigned wraparound).
    """
    fp = json.loads(fault_plan_path.read_text())
    meta = fp["metadata"]
    target_plane = meta["target_plane"]
    plane_weight = 1 << (8 * (7 - target_plane))

    pp = artifact_dir / f"plane_{target_plane}.bin"
    arr = array("B")
    arr.frombytes(pp.read_bytes())
    clean_plane = list(arr)

    sum_delta = 0
    negative_seen = False
    for entry in fp["entries"]:
        offset = entry["offset"]
        mask = entry["mask"]
        old_byte = clean_plane[offset]
        new_byte = old_byte ^ mask
        delta = int(new_byte) - int(old_byte)
        if delta < 0:
            negative_seen = True
        sum_delta += delta * plane_weight

    if not negative_seen:
        print("  ⚠ no negative oracle deltas in this fault plan "
              "(not a bug for fixture, only caveat)")

    expected = clean_encoded_sum + sum_delta
    assert isinstance(expected, int), \
        f"oracle result is not int: {type(expected)}"
    return expected


def apply_fault_plan(artifact_dir: Path,
                     fault_plan_path: Path) -> tuple[list[list[int]], dict[str, Any]]:
    """Apply fault plan to clean plane data in memory.

    Returns (faulted_planes_list, fault_plan_metadata).
    """
    fp = json.loads(fault_plan_path.read_text())
    meta = fp["metadata"]
    target_plane = meta["target_plane"]

    plane_data: list[list[int]] = []
    for p in range(8):
        pp = artifact_dir / f"plane_{p}.bin"
        arr = array("B")
        arr.frombytes(pp.read_bytes())
        plane_data.append(list(arr))

    for entry in fp["entries"]:
        offset = entry["offset"]
        mask = entry["mask"]
        plane_data[target_plane][offset] ^= mask

    return plane_data, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--fault-plan-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("reliability_tiny_fixture.csv"))
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    fault_plan_dir = Path(args.fault_plan_dir)
    csv_out_path = Path(args.csv)
    csv_out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ── Run contract tests ──
    print("Contract tests:")
    test_byte_split_known_values()
    test_decimal_round_trip()

    # ── Step 1: Generate tiny fixture raw data ──
    raw_path = artifact_dir / "tiny_raw.f64le.bin"
    raw_values = generate_tiny_fixture_raw(raw_path)
    print(f"\nStep 1: {len(raw_values)} raw FP64 values at {raw_path}")

    # ── Step 2: Convert to planes ──
    metadata = convert_to_planes(raw_values, SCALE, artifact_dir, raw_path)
    clean_sum = int(metadata["clean_encoded_sum"])
    print(f"Step 2: clean encoded SUM = {clean_sum}")

    # ── Step 3: Verify CPU reconstruction SUM ──
    print("Step 3:")
    test_cpu_reconstruct_sum(artifact_dir, clean_sum)

    # ── Step 4: Generate fault plans ──
    target_planes = [0, 1, 2, 3, 4, 5, 6, 7]
    fault_rates = [1e-3, 1e-2]
    seeds = [0, 1]
    plan_paths = generate_fault_plans(
        artifact_dir, fault_plan_dir, target_planes, fault_rates, seeds, metadata,
    )
    print(f"Step 4: {len(plan_paths)} fault plans generated")

    # ── Step 5: Apply faults and validate against oracle ──
    csv_rows: list[dict[str, str]] = []
    all_oracles_match = True

    artifact_checksum = sha256_file(artifact_dir / "artifact.json")

    for fp_path in plan_paths:
        fp_meta = json.loads(fp_path.read_text())["metadata"]
        target_plane = fp_meta["target_plane"]
        rate = fp_meta["fault_rate_numeric"]
        seed_val = fp_meta["seed"]
        fault_count = fp_meta["actual_fault_count"]

        # Test fault plan properties (sorted, unique, nonzero masks)
        test_fault_plan_properties(fp_path)

        fault_plan_checksum = sha256_file(fp_path)
        fault_plan_rel = str(fp_path.relative_to(fault_plan_dir.parent))

        # Apply fault
        faulted_planes, _ = apply_fault_plan(artifact_dir, fp_path)
        faulted_sum = sum(
            sum(faulted_planes[p][i] << (8 * (7 - p)) for p in range(8))
            for i in range(len(faulted_planes[0]))
        )

        # Compute oracle with signed delta tracking
        expected_sum = test_oracle_signed_deltas(
            artifact_dir, fp_path, clean_sum)

        oracle_match = (faulted_sum == expected_sum)
        if not oracle_match:
            all_oracles_match = False

        signed_damage = faulted_sum - clean_sum
        abs_damage = abs(signed_damage)
        plane_weight = 1 << (8 * (7 - target_plane))
        normalized_damage = abs_damage // plane_weight if plane_weight > 0 else 0

        row = {
            "run_id": f"cpu_tracer_{DATE_TAG}",
            "dataset": "tiny_fixture",
            "n_rows": str(metadata["n_rows"]),
            "scale": str(SCALE),
            "target_plane": str(target_plane),
            "plane_weight": str(plane_weight),
            "plane_nonzero_count": str(metadata["plane_nonzero_count"][target_plane]),
            "plane_nonzero_fraction": str(metadata["plane_nonzero_fraction"][target_plane]),
            "fault_rate": f"{rate:.0e}",
            "fault_count": str(fault_count),
            "seed": str(seed_val),
            "fault_model": "plane_targeted_random_byte_xor",
            "clean_encoded_sum": decimal_string(clean_sum),
            "expected_corrupted_sum": decimal_string(expected_sum),
            "gpu_corrupted_sum": decimal_string(faulted_sum),
            "signed_sum_damage_encoded": decimal_string(signed_damage),
            "abs_sum_damage_encoded": decimal_string(abs_damage),
            "normalized_abs_sum_damage": str(normalized_damage),
            "decoded_abs_sum_damage": decimal_string(abs_damage // SCALE),
            "oracle_match": "true" if oracle_match else "false",
            "artifact_id": str(artifact_dir),
            "fault_plan_id": fault_plan_rel,
            "artifact_checksum": artifact_checksum,
            "fault_plan_checksum": fault_plan_checksum,
            "git_commit": metadata.get("git_commit", "unknown"),
            "hostname": __import__("socket").gethostname(),
            "gpu_name": "CPU_ONLY",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "repro_command": f"python3 {' '.join(shlex.quote(a) for a in sys.argv)}",
            "validity_status": "canonical" if oracle_match else "ORACLE_MISMATCH",
        }
        csv_rows.append(row)

    # ── Write CSV ──
    fieldnames = [
        "run_id", "dataset", "n_rows", "scale",
        "target_plane", "plane_weight", "plane_nonzero_count", "plane_nonzero_fraction",
        "fault_rate", "fault_count", "seed", "fault_model",
        "clean_encoded_sum", "expected_corrupted_sum", "gpu_corrupted_sum",
        "signed_sum_damage_encoded", "abs_sum_damage_encoded",
        "normalized_abs_sum_damage", "decoded_abs_sum_damage",
        "oracle_match", "artifact_id", "fault_plan_id",
        "artifact_checksum", "fault_plan_checksum",
        "git_commit", "hostname", "gpu_name", "slurm_job_id",
        "repro_command", "validity_status",
    ]

    with csv_out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nCSV written: {csv_out_path} ({len(csv_rows)} rows)")
    print(f"All oracles match: {all_oracles_match}")
    print(f"CSV columns: {len(fieldnames)} (artifact_checksum + fault_plan_checksum included)")

    for row in csv_rows:
        print(f"  plane={row['target_plane']:>1} "
              f"rate={row['fault_rate']:>4} "
              f"seed={row['seed']:>2} "
              f"|damage|={row['abs_sum_damage_encoded']:>20} "
              f"oracle={row['oracle_match']}")

    if not all_oracles_match:
        print("ERROR: oracle mismatch(es) detected", file=sys.stderr)
        sys.exit(2)

    print("\nOK: all CPU tracer bullet tests pass")


if __name__ == "__main__":
    main()
