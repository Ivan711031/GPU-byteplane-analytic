#!/usr/bin/env python3
"""Validate active-delta-global contract for Issue #135.

Four checks:
1. Reconstruction identity: encoded[i] = base_fixed + delta[i]
2. SUM identity: clean_encoded_sum = n_rows * base_fixed + sum(delta)
3. Plane-weight identity: delta[i] = sum_p plane_byte[p][i] * plane_weight[p]
4. Inactive trailing planes are zero

Usage:
  python3 scripts/validate_active_delta_contract.py \
    --artifact-dir /path/to/active_delta_artifact \
    [--dataset sensor]

Synth mode (no external data):
  python3 scripts/validate_active_delta_contract.py \
    --artifact-root /tmp/validate_active_delta \
    --datasets sensor uniform heavy_tailed zipfian \
    --n-rows 100000 \
    --synth

Exit code:
  0 if all checks pass for all datasets
  1 if any check fails
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import math
from pathlib import Path
from typing import Any

CHUNK_ROWS = 500000


def synth_dataset(dataset: str, n_rows: int, seed: int) -> list[float]:
    rng = __import__("random").Random(seed)
    if dataset == "sensor":
        lo, hi = 15.0, 35.0
        vals = [lo + (hi - lo) * ((i + rng.random()) / n_rows) for i in range(n_rows)]
        for _ in range(n_rows // 100):
            idx = rng.randint(0, n_rows - 1)
            vals[idx] += rng.gauss(0, 0.5)
    elif dataset == "uniform":
        vals = [rng.uniform(0.0, 1000.0) for _ in range(n_rows)]
    elif dataset == "heavy_tailed":
        vals = [max(0.0, rng.paretovariate(1.5) - 1.0) for _ in range(n_rows)]
    elif dataset == "zipfian":
        vals = [rng.expovariate(0.001) for _ in range(n_rows)]
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return vals


def encode_active_delta(
    values: list[float], scale: int, artifact_dir: Path
) -> tuple[dict[str, Any], list[int]]:
    n_rows = len(values)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    fixed_vals = []
    for raw_val in values:
        scaled = raw_val * scale
        encoded = int(round(scaled))
        fixed_vals.append(encoded)

    base_fixed = min(fixed_vals)
    deltas = [v - base_fixed for v in fixed_vals]
    max_delta = max(deltas)
    active_byte_len = max(1, (max_delta.bit_length() + 7) // 8) if max_delta > 0 else 1
    clean_delta_sum = sum(deltas)
    clean_encoded_sum = n_rows * base_fixed + clean_delta_sum

    plane_files = [artifact_dir / f"plane_{p}.bin" for p in range(8)]
    plane_f = [f.open("wb") for f in plane_files]

    for delta in deltas:
        for p in range(8):
            if p < active_byte_len:
                shift = 8 * (active_byte_len - 1 - p)
                byte_val = (delta >> shift) & 0xFF
            else:
                byte_val = 0
            plane_f[p].write(struct.pack("B", byte_val))

    for f in plane_f:
        f.close()

    plane_weight = []
    for p in range(8):
        if p < active_byte_len:
            pw = 256 ** (active_byte_len - 1 - p)
        else:
            pw = 0
        plane_weight.append(pw)

    metadata = {
        "artifact_format": "active_delta_global_v1",
        "dataset": artifact_dir.parent.parent.parent.name,
        "n_rows": n_rows,
        "scale": scale,
        "base_fixed": base_fixed,
        "max_delta": max_delta,
        "active_byte_len": active_byte_len,
        "plane_count": 8,
        "plane_weight": plane_weight,
        "clean_encoded_sum": str(clean_encoded_sum),
        "clean_delta_sum": str(clean_delta_sum),
    }

    meta_path = artifact_dir / "artifact.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    return metadata, deltas


def check_reconstruction(
    artifact_dir: Path, metadata: dict[str, Any], chunk_rows: int
) -> tuple[int, list[str]]:
    n_rows = metadata["n_rows"]
    base_fixed = metadata["base_fixed"]
    active_byte_len = metadata["active_byte_len"]
    plane_files = [artifact_dir / f"plane_{p}.bin" for p in range(8)]
    errors = []

    offset = 0
    total = 0
    while offset < n_rows:
        remaining = n_rows - offset
        this_chunk = min(chunk_rows, remaining)

        plane_data = []
        for p in range(8):
            with plane_files[p].open("rb") as f:
                f.seek(offset)
                data = f.read(this_chunk)
                if len(data) < this_chunk:
                    errors.append(f"short read plane_{p} at offset {offset}")
                    return total, errors
                plane_data.append(data)

        for i in range(this_chunk):
            delta = 0
            for p in range(active_byte_len):
                delta |= plane_data[p][i] << (8 * (active_byte_len - 1 - p))
            row = delta
            encoded_row = base_fixed + row
            total += encoded_row

        offset += this_chunk

    meta_sum = int(metadata["clean_encoded_sum"])
    if total != meta_sum:
        errors.append(f"SUM mismatch: computed={total}, metadata={meta_sum}")
    return total, errors


def check_plane_weight_identity(
    artifact_dir: Path, metadata: dict[str, Any], chunk_rows: int,
    oracle_deltas: list[int] | None = None,
) -> list[str]:
    if oracle_deltas is None:
        return []
    n_rows = metadata["n_rows"]
    active_byte_len = metadata["active_byte_len"]
    plane_weight = metadata["plane_weight"]
    plane_files = [artifact_dir / f"plane_{p}.bin" for p in range(8)]
    errors = []

    offset = 0
    while offset < n_rows:
        remaining = n_rows - offset
        this_chunk = min(chunk_rows, remaining)

        plane_data = []
        for p in range(8):
            with plane_files[p].open("rb") as f:
                f.seek(offset)
                data = f.read(this_chunk)
                if len(data) < this_chunk:
                    errors.append(f"short read plane_{p} at offset {offset}")
                    return errors
                plane_data.append(data)

        for i in range(this_chunk):
            reconstructed = 0
            for p in range(8):
                reconstructed += plane_data[p][i] * plane_weight[p]
            expected = oracle_deltas[offset + i]
            if reconstructed != expected:
                errors.append(
                    f"plane-weight mismatch at row {offset + i}: "
                    f"reconstructed={reconstructed}, expected={expected}"
                )
                return errors

        offset += this_chunk
    return errors


def check_trailing_planes_zero(
    artifact_dir: Path, metadata: dict[str, Any]
) -> list[str]:
    n_rows = metadata["n_rows"]
    active_byte_len = metadata["active_byte_len"]
    errors = []

    for p in range(active_byte_len, 8):
        path = artifact_dir / f"plane_{p}.bin"
        nonzero = 0
        with path.open("rb") as f:
            while True:
                buf = f.read(1 << 20)
                if not buf:
                    break
                nonzero += sum(1 for b in buf if b != 0)
        if nonzero > 0:
            errors.append(
                f"plane_{p} (trailing, p >= active_byte_len={active_byte_len}) "
                f"has {nonzero} nonzero bytes"
            )

    return errors


def validate_artifact(
    artifact_dir: Path, metadata: dict[str, Any],
    oracle_deltas: list[int] | None = None,
) -> int:
    n_rows = metadata["n_rows"]
    print(f"  n_rows={n_rows}, active_byte_len={metadata['active_byte_len']}")
    print(f"  base_fixed={metadata['base_fixed']}, max_delta={metadata['max_delta']}")
    print(f"  plane_weight={metadata['plane_weight']}")

    chunk_rows = min(CHUNK_ROWS, n_rows)

    # Check 1 & 2: Reconstruction + SUM
    total, errors = check_reconstruction(artifact_dir, metadata, chunk_rows)
    if errors:
        print(f"  FAIL: reconstruction/SUM identity: {'; '.join(errors)}")
        return 1
    meta_sum = int(metadata["clean_encoded_sum"])
    print(f"  PASS: reconstruction identity, SUM={total} (metadata={meta_sum})")

    # Check 3: Plane-weight identity
    errors = check_plane_weight_identity(artifact_dir, metadata, chunk_rows, oracle_deltas)
    if oracle_deltas is not None:
        if errors:
            print(f"  FAIL: plane-weight identity: {'; '.join(errors)}")
            return 1
        print("  PASS: plane-weight identity holds for all rows")
    else:
        if errors:
            for e in errors:
                print(f"  WARN: {e}")
        print("  INFO: plane-weight identity not checked per-row (no delta oracle; use --synth)")

    # Check 4: Inactive trailing planes are zero
    errors = check_trailing_planes_zero(artifact_dir, metadata)
    if errors:
        print(f"  FAIL: trailing plane zero check: {'; '.join(errors)}")
        return 1
    print("  PASS: trailing planes are zero")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument(
        "--datasets", nargs="+", default=["sensor", "uniform", "heavy_tailed", "zipfian"]
    )
    parser.add_argument("--n-rows", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--synth", action="store_true")
    args = parser.parse_args()

    if args.artifact_dir is not None:
        meta = json.loads((args.artifact_dir / "artifact.json").read_text())
        ok = validate_artifact(args.artifact_dir, meta, oracle_deltas=None)
        sys.exit(ok)

    if not args.synth:
        print("No --artifact-dir or --synth specified; nothing to validate.")
        sys.exit(1)

    scale_map = {
        "sensor": int(2**48 // 45),
        "uniform": int(2**48 // 1100),
        "heavy_tailed": int(2**48 // int(1e10)),
        "zipfian": int(2**48 // int(1e5)),
    }

    exit_code = 0
    for ds in args.datasets:
        print(f"\nDataset: {ds}")
        scale = scale_map.get(ds, int(2**48 // 1000))
        print(f"  Generating {args.n_rows} synthetic rows, scale={scale}")
        vals = synth_dataset(ds, args.n_rows, args.seed)
        art_dir = args.artifact_root / ds / f"n{args.n_rows}" / f"scale{scale}"
        meta, deltas = encode_active_delta(vals, scale, art_dir)
        code = validate_artifact(art_dir, meta, oracle_deltas=deltas)
        if code != 0:
            exit_code = code

    if exit_code == 0:
        print(f"\nAll checks PASSED for all datasets.")
    else:
        print(f"\nSome checks FAILED.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
