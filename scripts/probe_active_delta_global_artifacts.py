#!/usr/bin/env python3
"""Active-delta-global artifact probe for Issue #133.

Builds smoke artifacts using the active-delta-global encoding to test whether
plane 0 / plane 1 can become meaningful active byte planes (rather than
fixed-container headroom planes).

Usage (WSL → CLUSTER_HOST via sshpass):
  python3 scripts/probe_active_delta_global_artifacts.py \
    --raw-data-root ${WORK_DIR}/datasets/reliability_layer1_phase1_5 \
    --artifact-root /tmp/probe_active_delta_global \
    --strategy-params results/phase1_5_strategy_params.json \
    --datasets sensor uniform heavy_tailed zipfian \
    --n-rows 10000000 \

Usage (standalone Python, local synth):
  python3 scripts/probe_active_delta_global_artifacts.py \
    --artifact-root /tmp/probe_active_delta_global \
    --datasets sensor uniform heavy_tailed zipfian \
    --n-rows 100000 \
    --synth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any


CHUNK_ROWS = 500000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synth_dataset(dataset: str, n_rows: int, seed: int) -> tuple[list[float], float, float]:
    rng = __import__("random").Random(seed)
    if dataset == "sensor":
        lo, hi = 15.0, 35.0
        vals = [lo + (hi - lo) * ((i + rng.random()) / n_rows) for i in range(n_rows)]
        for i in range(n_rows // 100):
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
    raw_min = min(vals)
    raw_max = max(vals)
    return vals, raw_min, raw_max


def load_scale_map(strategy_params_path: Path, strategy_id: str = "per_dataset") -> dict[str, int]:
    sp = json.loads(strategy_params_path.read_text())
    per_ds = sp["strategies"][strategy_id]["per_dataset"]
    return {ds: info["scale"] for ds, info in per_ds.items()}


def probe_active_delta(dataset: str,
                       raw_vals_or_path,
                       n_rows: int,
                       scale: int,
                       artifact_dir: Path,
                       is_synth: bool = False) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    reads_raw = isinstance(raw_vals_or_path, (str, Path))
    raw_f: Any = None
    precomputed: Any = None
    if reads_raw:
        raw_path = Path(raw_vals_or_path)
        raw_f = raw_path.open("rb")
    else:
        precomputed = raw_vals_or_path

    plane_tmp: list[Path] = []
    plane_f: list[Any] = []
    for p in range(8):
        tmp = artifact_dir / f".plane_{p}.tmp"
        plane_tmp.append(tmp)
        plane_f.append(tmp.open("wb"))

    raw_min = float("inf")
    raw_max = float("-inf")
    raw_sum_fp64 = 0.0
    overflow_count = 0
    quantization_max_error = 0.0
    quantization_sum_error = 0.0

    fixed_vals: list[int] = []
    rows_processed = 0

    while rows_processed < n_rows:
        remaining = n_rows - rows_processed
        this_chunk = min(CHUNK_ROWS, remaining)

        if reads_raw:
            chunk_bytes = this_chunk * 8
            raw_data = raw_f.read(chunk_bytes)
            if len(raw_data) < chunk_bytes:
                raise RuntimeError(f"short read at row {rows_processed}")
            chunk_values = struct.unpack(f"<{this_chunk}d", raw_data)
        elif precomputed is not None:
            chunk_values = precomputed[rows_processed:rows_processed + this_chunk]
        else:
            raise RuntimeError("no input source")

        for raw_val in chunk_values:
            raw_sum_fp64 += raw_val
            raw_min = min(raw_min, raw_val)
            raw_max = max(raw_max, raw_val)

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

            quant_err = abs(scaled - round(scaled)) / scale
            quantization_max_error = max(quantization_max_error, quant_err)
            quantization_sum_error += quant_err

            fixed_vals.append(encoded)
            rows_processed += 1

    if reads_raw:
        raw_f.close()

    if len(fixed_vals) < 1:
        raise RuntimeError("no valid encoded values")

    base_fixed = min(fixed_vals)
    deltas = [v - base_fixed for v in fixed_vals]
    max_delta = max(deltas)
    active_byte_len = max(1, (max_delta.bit_length() + 7) // 8) if max_delta > 0 else 1
    clean_encoded_sum = n_rows * base_fixed + sum(deltas)

    for i, delta in enumerate(deltas):
        for p in range(8):
            if p < active_byte_len:
                shift = 8 * (active_byte_len - 1 - p)
                byte_val = (delta >> shift) & 0xFF
            else:
                byte_val = 0
            plane_f[p].write(struct.pack("B", byte_val))

    for f in plane_f:
        f.close()

    plane_paths: list[Path] = []
    plane_checksums: list[str] = []
    plane_nonzero: list[int] = []
    plane_unique: list[int] = []
    plane_entropy: list[float] = []
    plane_weight: list[int] = []

    for p in range(8):
        final = artifact_dir / f"plane_{p}.bin"
        if plane_tmp[p].exists():
            plane_tmp[p].rename(final)
        plane_paths.append(final)
        plane_checksums.append(sha256_file(final))

        nonzero = 0
        byte_counts: dict[int, int] = {}
        with final.open("rb") as f:
            while True:
                buf = f.read(1 << 20)
                if not buf:
                    break
                nonzero += sum(1 for b in buf if b != 0)
                for b in buf:
                    byte_counts[b] = byte_counts.get(b, 0) + 1

        plane_nonzero.append(nonzero)
        unique = len(byte_counts)
        plane_unique.append(unique)

        total = sum(byte_counts.values())
        if total > 0 and unique > 1:
            ent = -sum((c / total) * math.log2(c / total) for c in byte_counts.values() if c > 0)
            plane_entropy.append(ent)
        else:
            plane_entropy.append(0.0)

    for p in range(8):
        if p < active_byte_len:
            pw = 256 ** (active_byte_len - 1 - p)
        else:
            pw = 0
        plane_weight.append(pw)

    metadata: dict[str, Any] = {
        "dataset": dataset,
        "n_rows": len(fixed_vals),
        "scale": scale,
        "base_fixed": base_fixed,
        "max_delta": max_delta,
        "active_byte_len": active_byte_len,
        "artifact_format": "active_delta_global_probe_v1",
        "plane_count": 8,
        "plane_order": "MSB_first_active_delta",
        "plane_files": sorted(p.name for p in artifact_dir.glob("plane_*.bin")),
        "plane_checksums": {f"plane_{p}": plane_checksums[p] for p in range(8)},
        "plane_sizes_bytes": [n_rows] * 8,
        "plane_nonzero_count": plane_nonzero,
        "plane_nonzero_fraction": [c / n_rows for c in plane_nonzero],
        "plane_unique_count": plane_unique,
        "plane_entropy_bits": plane_entropy,
        "plane_weight": plane_weight,
        "clean_encoded_sum": str(clean_encoded_sum),
        "clean_delta_sum": str(sum(deltas)),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_sum_fp64": raw_sum_fp64,
        "overflow_count": overflow_count,
        "quantization_max_error": quantization_max_error,
        "quantization_mean_error": quantization_sum_error / rows_processed if rows_processed else 0.0,
        "input_type": "synth" if is_synth else "phase1_5_raw",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    meta_path = artifact_dir / "artifact.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def verify_artifact(artifact_dir: Path, metadata: dict[str, Any]) -> int:
    n_rows = metadata["n_rows"]
    base_fixed = metadata["base_fixed"]
    active_byte_len = metadata["active_byte_len"]
    plane_files = [artifact_dir / f"plane_{p}.bin" for p in range(8)]

    total = 0
    offset = 0
    buf_size = min(CHUNK_ROWS, n_rows)

    while offset < n_rows:
        remaining = n_rows - offset
        this_chunk = min(buf_size, remaining)

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
    assert total == meta_sum, (
        f"CPU reconstruction SUM mismatch: {total} != {meta_sum}"
    )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-root", type=Path, default=None,
                        help="Path to Phase 1.5 raw data root (contains reliability_layer1/raw/)")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--strategy-params", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+",
                        default=["sensor", "uniform", "heavy_tailed", "zipfian"])
    parser.add_argument("--n-rows", type=int, default=10000000)
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--synth", action="store_true",
                        help="Use synthetic data (no CLUSTER_HOST raw data needed)")

    args = parser.parse_args()


    has_raw_data = args.raw_data_root is not None and (args.raw_data_root / "reliability_layer1" / "raw").exists()

    if args.strategy_params:
        scale_map = load_scale_map(args.strategy_params)
    else:
        scale_map = {}

    for ds in args.datasets:
        if ds in scale_map:
            scale = scale_map[ds]
        else:
            raw_max_est = {"sensor": 35.0, "uniform": 1000.0, "heavy_tailed": 1e10, "zipfian": 1e5}.get(ds, 1000.0)
            scale = int(2 ** 48 // raw_max_est)
        print(f"Dataset: {ds}, scale: {scale}")

        if args.synth:
            print(f"  Generating synthetic {ds} data (N={args.n_rows})...")
            vals, raw_min, raw_max = synth_dataset(ds, args.n_rows, args.seed)
            print(f"  raw_min={raw_min}, raw_max={raw_max}")
            art_dir = args.artifact_root / ds / f"n{args.n_rows}" / f"scale{scale}"
            meta = probe_active_delta(ds, vals, args.n_rows, scale, art_dir, is_synth=True)
        elif has_raw_data and args.raw_data_root is not None:
            raw_path = args.raw_data_root / "reliability_layer1" / "raw" / f"{ds}.f64le.bin"
            if not raw_path.exists():
                print(f"  WARN: {raw_path} not found, skipping")
                continue
            print(f"  Reading {raw_path} (N={args.n_rows})...")
            art_dir = args.artifact_root / ds / f"n{args.n_rows}" / f"scale{scale}"
            meta = probe_active_delta(ds, raw_path, args.n_rows, scale, art_dir, is_synth=False)
        else:
            print(f"  No raw data source; use --synth to generate synthetic data")
            sys.exit(1)

        print(f"  active_byte_len={meta['active_byte_len']}, base_fixed={meta['base_fixed']}, max_delta={meta['max_delta']}")
        print(f"  plane_nonzero_fraction: {[round(v, 6) for v in meta['plane_nonzero_fraction']]}")
        print(f"  plane_entropy_bits: {[round(v, 4) for v in meta['plane_entropy_bits']]}")

        cpu_sum = verify_artifact(art_dir, meta)
        meta_sum = int(meta["clean_encoded_sum"])
        status = "OK" if cpu_sum == meta_sum else "MISMATCH"
        print(f"  VERIFY: {status}, cpu_sum={cpu_sum}, meta_sum={meta_sum}")

    print("Done.")


if __name__ == "__main__":
    main()
