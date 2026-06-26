#!/usr/bin/env python3
"""Build active_delta_global_v1 artifacts for Issue #137.

Generates canonical 8-plane active-delta artifacts from Phase 1.5 raw data,
with full contract-compliant metadata (checksums, occupancy, entropy).

Usage (nano5 raw data):
  python3 scripts/build_active_delta_global_artifacts.py \
    --raw-data-root /work/u4063895/datasets/reliability_layer1_phase1_5 \
    --strategy-params results/phase1_5_strategy_params.json \
    --artifact-root /path/to/artifacts_phase1_5b \
    --datasets sensor uniform heavy_tailed zipfian \
    --n-rows 100000000

Usage (local synth for testing):
  python3 scripts/build_active_delta_global_artifacts.py \
    --artifact-root /tmp/artifacts_phase1_5b \
    --datasets sensor uniform heavy_tailed zipfian \
    --n-rows 100000 --synth

Verification gates run automatically after each dataset.
Exit code 0 if all gates pass for all datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

CHUNK_ROWS = 500000
VACUOUS_THRESHOLD = 1e-6


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_stream(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_scale_map(strategy_params_path: Path) -> dict[str, int]:
    sp = json.loads(strategy_params_path.read_text())
    per_ds = sp["strategies"]["per_dataset"]["per_dataset"]
    return {ds: info["scale"] for ds, info in per_ds.items()}


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


def build_artifact(
    dataset: str,
    n_rows: int,
    scale: int,
    artifact_dir: Path,
    raw_path: Path | None = None,
    raw_values: list[float] | None = None,
    *,
    input_type: str = "synth",
    strategy_params_path: str | None = None,
    strategy_params_checksum: str | None = None,
    git_commit: str = "unknown",
    cmd: str = "unknown",
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    plane_files: list[Path] = [artifact_dir / f"plane_{p}.bin" for p in range(8)]
    plane_writers = [f.open("wb") for f in plane_files]

    fixed_vals: list[int] = []
    rows_processed = 0
    overflow_count = 0

    if raw_path is not None:
        raw_f = raw_path.open("rb")
        while rows_processed < n_rows:
            remaining = n_rows - rows_processed
            this_chunk = min(CHUNK_ROWS, remaining)
            chunk_bytes = this_chunk * 8
            raw_data = raw_f.read(chunk_bytes)
            if len(raw_data) < chunk_bytes:
                raise RuntimeError(f"short read at row {rows_processed}")
            chunk_vals = struct.unpack(f"<{this_chunk}d", raw_data)
            for raw_val in chunk_vals:
                if math.isnan(raw_val) or math.isinf(raw_val) or raw_val < 0.0:
                    raise ValueError(f"invalid raw at row {rows_processed}: {raw_val}")
                scaled = raw_val * scale
                encoded = int(round(scaled))
                if encoded < 0 or encoded > 18446744073709551615:
                    overflow_count += 1
                    if overflow_count == 1:
                        raise RuntimeError(
                            f"OVERFLOW at row {rows_processed}: raw={raw_val} "
                            f"scaled={scaled} encoded={encoded}"
                        )
                    continue
                fixed_vals.append(encoded)
                rows_processed += 1
        raw_f.close()
    elif raw_values is not None:
        for raw_val in raw_values:
            if math.isnan(raw_val) or math.isinf(raw_val) or raw_val < 0.0:
                raise ValueError(f"invalid raw value: {raw_val}")
            scaled = raw_val * scale
            encoded = int(round(scaled))
            if encoded < 0 or encoded > 18446744073709551615:
                overflow_count += 1
                if overflow_count == 1:
                    raise RuntimeError(
                        f"OVERFLOW at row {rows_processed}: raw={raw_val} "
                        f"scaled={scaled} encoded={encoded}"
                    )
                continue
            fixed_vals.append(encoded)
            rows_processed += 1
    else:
        raise RuntimeError("no input source: provide raw_path or raw_values")

    if len(fixed_vals) == 0:
        raise RuntimeError("no valid encoded values")

    n_actual = len(fixed_vals)
    base_fixed = min(fixed_vals)
    deltas = [v - base_fixed for v in fixed_vals]
    max_delta = max(deltas)
    active_byte_len = max(1, (max_delta.bit_length() + 7) // 8) if max_delta > 0 else 1
    clean_delta_sum = sum(deltas)
    clean_encoded_sum = n_actual * base_fixed + clean_delta_sum

    h_delta = hashlib.sha256()
    for d in deltas:
        h_delta.update(struct.pack("<Q", d))
    delta_checksum = h_delta.hexdigest()

    plane_weights: list[int] = []
    for p in range(8):
        pw = 256 ** (active_byte_len - 1 - p) if p < active_byte_len else 0
        plane_weights.append(pw)

    for i, delta in enumerate(deltas):
        for p in range(8):
            if p < active_byte_len:
                shift = 8 * (active_byte_len - 1 - p)
                byte_val = (delta >> shift) & 0xFF
            else:
                byte_val = 0
            plane_writers[p].write(struct.pack("B", byte_val))

    for w in plane_writers:
        w.close()

    plane_checksums: list[str] = []
    plane_nonzero_count: list[int] = []
    plane_nonzero_fraction: list[float] = []
    plane_unique_count: list[int] = []
    plane_entropy_bits: list[float] = []

    for p in range(8):
        fpath = plane_files[p]
        cksum = sha256_file(fpath)
        plane_checksums.append(cksum)

        nonzero = 0
        byte_counts: dict[int, int] = {}
        with fpath.open("rb") as f:
            while True:
                buf = f.read(1 << 20)
                if not buf:
                    break
                for b in buf:
                    byte_counts[b] = byte_counts.get(b, 0) + 1
                nonzero += sum(1 for b in buf if b != 0)

        total = sum(byte_counts.values())
        uniq = len(byte_counts)
        plane_nonzero_count.append(nonzero)
        plane_nonzero_fraction.append(nonzero / n_actual)
        plane_unique_count.append(uniq)

        if total > 0 and uniq > 1:
            ent = -sum(
                (c / total) * math.log2(c / total) for c in byte_counts.values() if c > 0
            )
            plane_entropy_bits.append(ent)
        else:
            plane_entropy_bits.append(0.0)

    metadata_no_checksum: dict[str, Any] = {
        "artifact_format": "active_delta_global_v1",
        "strategy_id": "active_delta_global",
        "encoded_semantics": "delta_plus_base",
        "dataset": dataset,
        "n_rows": n_actual,
        "scale": scale,
        "base_fixed": base_fixed,
        "max_delta": max_delta,
        "active_byte_len": active_byte_len,
        "plane_count": 8,
        "plane_order": "MSB_first_active_delta",
        "plane_weight": plane_weights,
        "plane_nonzero_count": plane_nonzero_count,
        "plane_nonzero_fraction": plane_nonzero_fraction,
        "plane_unique_count": plane_unique_count,
        "plane_entropy_bits": plane_entropy_bits,
        "clean_delta_sum": str(clean_delta_sum),
        "clean_encoded_sum": str(clean_encoded_sum),
        "delta_checksum": delta_checksum,
        "overflow_count": overflow_count,
        "input_type": input_type,
        "source_path": str(raw_path.resolve()) if raw_path else None,
        "source_size_bytes": raw_path.stat().st_size if raw_path else None,
        "source_checksum": sha256_stream(raw_path) if raw_path else None,
        "strategy_params_path": strategy_params_path,
        "strategy_params_checksum": strategy_params_checksum,
        "generator_git_commit": git_commit,
        "generator_command": cmd,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    exclude_from_checksum = {
        "plane_checksums", "metadata_checksum", "artifact_manifest_checksum",
        "source_checksum", "strategy_params_checksum", "delta_checksum",
    }
    cksum_input = {k: v for k, v in metadata_no_checksum.items()
                   if k not in exclude_from_checksum}
    metadata_canonical_bytes = json.dumps(cksum_input, sort_keys=True, indent=2).encode()
    metadata_checksum = sha256_bytes(metadata_canonical_bytes)

    sorted_plane_checksums = sorted(plane_checksums)
    manifest_input = "".join(sorted_plane_checksums) + metadata_checksum
    artifact_manifest_checksum = sha256_bytes(manifest_input.encode())

    metadata: dict[str, Any] = dict(metadata_no_checksum)
    metadata["plane_checksums"] = plane_checksums
    metadata["metadata_checksum"] = metadata_checksum
    metadata["artifact_manifest_checksum"] = artifact_manifest_checksum

    meta_path = artifact_dir / "artifact.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def verify_gates(artifact_dir: Path, metadata: dict[str, Any]) -> int:
    n_rows = metadata["n_rows"]
    base_fixed = metadata["base_fixed"]
    active_byte_len = metadata["active_byte_len"]
    plane_weight = metadata["plane_weight"]
    plane_files = [artifact_dir / f"plane_{p}.bin" for p in range(8)]
    errors: list[str] = []

    # Gate 1: Plane file existence and length
    for p in range(8):
        if not plane_files[p].exists():
            errors.append(f"plane_{p}.bin missing")
            continue
        sz = plane_files[p].stat().st_size
        if sz != n_rows:
            errors.append(f"plane_{p}.bin size {sz} != n_rows {n_rows}")

    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return 1
    print("  PASS: plane files exist and have correct length")

    # Gate 2: Inactive trailing planes are all zero
    trailing_nonzero = 0
    for p in range(active_byte_len, 8):
        with plane_files[p].open("rb") as f:
            while True:
                buf = f.read(1 << 20)
                if not buf:
                    break
                trailing_nonzero += sum(1 for b in buf if b != 0)
    if trailing_nonzero > 0:
        errors.append(f"trailing planes have {trailing_nonzero} nonzero bytes")
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return 1
    print("  PASS: inactive trailing planes are zero")

    # Gate 3: Reconstruction + SUM identity (full scan)
    total = 0
    offset = 0
    while offset < n_rows:
        remaining = n_rows - offset
        this_chunk = min(CHUNK_ROWS, remaining)
        plane_chunks = []
        for p in range(8):
            with plane_files[p].open("rb") as f:
                f.seek(offset)
                data = f.read(this_chunk)
                if len(data) < this_chunk:
                    raise RuntimeError(f"short read plane_{p} at {offset}")
                plane_chunks.append(data)
        for i in range(this_chunk):
            delta = 0
            for p in range(active_byte_len):
                delta |= plane_chunks[p][i] << (8 * (active_byte_len - 1 - p))
            total += base_fixed + delta
        offset += this_chunk

    meta_sum = int(metadata["clean_encoded_sum"])
    if total != meta_sum:
        errors.append(f"SUM mismatch: computed={total}, metadata={meta_sum}")
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return 1
    print(f"  PASS: reconstruction identity, SUM={total} (metadata={meta_sum})")

    # Gate 4: Metadata weight consistency check (self-consistent;
    # verifies plane_weight entries match the inverse of bit-shift encoding)
    weight_errors = 0
    offset = 0
    while offset < n_rows:
        remaining = n_rows - offset
        this_chunk = min(10000, remaining)
        plane_chunks = []
        for p in range(8):
            with plane_files[p].open("rb") as f:
                f.seek(offset)
                data = f.read(this_chunk)
                plane_chunks.append(data)
        for i in range(this_chunk):
            reconstructed = 0
            for p in range(8):
                reconstructed += plane_chunks[p][i] * plane_weight[p]
            expected = 0
            for p in range(active_byte_len):
                expected |= plane_chunks[p][i] << (8 * (active_byte_len - 1 - p))
            if reconstructed != expected:
                errors.append(
                    f"plane-weight mismatch at row {offset + i}: "
                    f"reconstructed={reconstructed}, expected={expected}"
                )
                weight_errors += 1
                if weight_errors >= 5:
                    break
        if weight_errors >= 5:
            break
        offset += this_chunk

    if weight_errors > 0:
        for e in errors:
            print(f"  FAIL: {e}")
        return 1
    print("  PASS: metadata weight consistency check (self-consistent)")

    # Gate 4b: Delta oracle checksum verification (against original deltas)
    if "delta_checksum" in metadata:
        h = hashlib.sha256()
        offset = 0
        while offset < n_rows:
            remaining = n_rows - offset
            this_chunk = min(CHUNK_ROWS, remaining)
            plane_chunks = []
            for p in range(active_byte_len):
                with plane_files[p].open("rb") as f:
                    f.seek(offset)
                    data = f.read(this_chunk)
                    plane_chunks.append(data)
            for i in range(this_chunk):
                delta = 0
                for p in range(active_byte_len):
                    delta |= plane_chunks[p][i] << (8 * (active_byte_len - 1 - p))
                h.update(struct.pack("<Q", delta))
            offset += this_chunk
        computed = h.hexdigest()
        if computed != metadata["delta_checksum"]:
            errors.append(
                f"delta checksum mismatch: computed={computed}, "
                f"metadata={metadata['delta_checksum']}"
            )
        else:
            print(f"  PASS: delta oracle checksum verification matches metadata")
    else:
        print("  SKIP: delta oracle checksum not available (missing in metadata)")

    # Gate 5: P0/P1 signal check
    for p in [0, 1]:
        frac = metadata["plane_nonzero_fraction"][p]
        uniq = metadata["plane_unique_count"][p]
        ent = metadata["plane_entropy_bits"][p]
        if frac < VACUOUS_THRESHOLD:
            errors.append(
                f"plane_{p} is vacuous (frac={frac:.2e} < {VACUOUS_THRESHOLD})"
            )
        else:
            print(f"  PASS: plane_{p} has signal — frac={frac:.6f}, unique={uniq}, entropy={ent:.2f}")

    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-root", type=Path, default=None)
    parser.add_argument("--strategy-params", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+",
                        default=["sensor", "uniform", "heavy_tailed", "zipfian"])
    parser.add_argument("--n-rows", type=int, default=100000000)
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--synth", action="store_true")
    args = parser.parse_args()

    if args.synth and args.strategy_params is None:
        raw_max_est = {"sensor": 35.0, "uniform": 1000.0,
                       "heavy_tailed": 1e10, "zipfian": 1e5}
        scale_map = {ds: int(2 ** 48 // raw_max_est[ds]) for ds in args.datasets}
    elif args.strategy_params is not None:
        scale_map = load_scale_map(args.strategy_params)
    else:
        print("ERROR: need --strategy-params or --synth")
        sys.exit(1)

    raw_base = args.raw_data_root / "reliability_layer1" / "raw" if args.raw_data_root else None

    git_commit = get_git_commit()
    cmd = "python3 " + " ".join(sys.argv)

    sp_path: str | None = None
    sp_checksum: str | None = None
    if args.strategy_params is not None and args.strategy_params.exists():
        sp_path = str(args.strategy_params.resolve())
        sp_checksum = sha256_stream(args.strategy_params)

    exit_code = 0
    for ds in args.datasets:
        scale = scale_map.get(ds)
        if scale is None:
            print(f"  SKIP: no scale for {ds}")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"  scale={scale}, n_rows={args.n_rows}")
        art_dir = args.artifact_root / ds / f"n{args.n_rows}" / f"scale{scale}"

        if args.synth:
            print("  Generating synthetic raw data...")
            vals = synth_dataset(ds, args.n_rows, args.seed)
            print(f"  Building artifact (synth mode)...")
            meta = build_artifact(ds, args.n_rows, scale, art_dir, raw_values=vals,
                                  input_type="synth",
                                  strategy_params_path=sp_path,
                                  strategy_params_checksum=sp_checksum,
                                  git_commit=git_commit,
                                  cmd=cmd)
        elif raw_base is not None:
            raw_path = raw_base / f"{ds}.f64le.bin"
            if not raw_path.exists():
                print(f"  SKIP: {raw_path} not found")
                continue
            print(f"  Raw data: {raw_path}")
            print(f"  Building artifact (streaming)...")
            meta = build_artifact(ds, args.n_rows, scale, art_dir, raw_path=raw_path,
                                  input_type="phase1_5_raw",
                                  strategy_params_path=sp_path,
                                  strategy_params_checksum=sp_checksum,
                                  git_commit=git_commit,
                                  cmd=cmd)
        else:
            print("  ERROR: no data source (use --synth or --raw-data-root)")
            sys.exit(1)

        print(f"  active_byte_len={meta['active_byte_len']}")
        print(f"  base_fixed={meta['base_fixed']}, max_delta={meta['max_delta']}")
        print(f"  plane_weight={meta['plane_weight']}")
        print(f"  plane_nonzero_fraction: {[round(v, 6) for v in meta['plane_nonzero_fraction']]}")
        print(f"  plane_entropy_bits: {[round(v, 4) for v in meta['plane_entropy_bits']]}")
        print(f"  plane_unique_count: {meta['plane_unique_count']}")

        print("  Running verification gates...")
        code = verify_gates(art_dir, meta)
        if code != 0:
            print(f"  FAIL: verification failed for {ds}")
            exit_code = code
        else:
            print(f"  PASS: all gates passed for {ds}")

    if exit_code == 0:
        print(f"\nAll artifacts built and verified successfully.")
    else:
        print(f"\nSome artifacts or verifications FAILED.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
