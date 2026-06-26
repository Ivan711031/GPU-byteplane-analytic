#!/usr/bin/env python3
"""Build pinned synthetic raw datasets and fixed-width 8-plane artifacts.

Phase 1 pipeline steps 1-3:
1. Generate raw FP64 datasets using synth_datasets.c with pinned seed.
2. Convert to fixed-width 8-plane byte-split artifacts.
3. Verify artifact metadata, plane sizes, reconstruction, and clean encoded SUM.

Chunked processing: reads raw file in streaming fashion, writes plane files
incrementally. Handles N=1e8 without OOM.

Usage:
  python3 scripts/build_reliability_artifacts.py \
    --raw-data-root /work/u4063895/datasets \
    --artifact-root /work/u4063895/datasets/reliability_layer1 \
    --datasets sensor uniform heavy_tailed zipfian \
    --n-rows 100000000 \
    --scale 100 \
    --seed 20260413
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCALE = 100
CHUNK_ROWS = 1000000  # 1M rows per chunk ~ 8 MB raw, 1 MB per plane


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_raw_datasets(raw_dir: Path,
                          datasets: list[str],
                          n_rows: int,
                          seed: int,
                          synth_bin: str) -> dict[str, Path]:
    """Run synth_datasets.c to generate raw FP64 LE binaries."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    ds_map = {"sensor": "sensor", "uniform": "uniform",
              "heavy_tailed": "heavy_tailed", "zipfian": "zipfian"}
    result_paths: dict[str, Path] = {}
    for ds in datasets:
        mapped = ds_map.get(ds, ds)
        cmd = [synth_bin, "--dataset", mapped, "--count", str(n_rows),
               "--seed", str(seed), "--out-dir", str(raw_dir)]
        print(f"Generating {ds}: {' '.join(shlex.quote(a) for a in cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}", file=sys.stderr)
            raise RuntimeError(f"synth_datasets failed for {ds}")
        print(result.stdout)
        data_file = raw_dir / f"{mapped}.f64le.bin"
        if not data_file.exists():
            raise RuntimeError(f"expected data file not found: {data_file}")
        result_paths[ds] = data_file
    return result_paths


def convert_raw_to_planes_chunked(raw_path: Path,
                                   dataset: str,
                                   n_rows: int,
                                   scale: int,
                                   artifact_dir: Path,
                                   canonical_seed: int) -> dict[str, Any]:
    """Convert raw FP64 LE binary to 8-plane artifacts using chunked I/O.

    Reads the raw file in streaming chunks, writes plane files via temp
    files, and accumulates statistics without loading all rows into memory.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Open raw file
    raw_f = raw_path.open("rb")

    # Open temp plane files for writing
    plane_tmp: list[Path] = []
    plane_f: list[Any] = []
    for p in range(8):
        tmp = artifact_dir / f".plane_{p}.tmp"
        plane_tmp.append(tmp)
        plane_f.append(tmp.open("wb"))

    overflow_count = 0
    zero_encoded_count = 0
    max_encoded = 0
    raw_sum_fp64 = 0.0
    clean_encoded_sum = 0
    raw_min = float("inf")
    raw_max = float("-inf")
    quantization_max_error = 0.0
    quantization_sum_error = 0.0

    rows_processed = 0
    chunk_buf_size = CHUNK_ROWS
    chunk_bytes = chunk_buf_size * 8

    while rows_processed < n_rows:
        remaining = n_rows - rows_processed
        this_chunk = min(chunk_buf_size, remaining)
        this_bytes = this_chunk * 8

        raw_data = raw_f.read(this_bytes)
        if len(raw_data) < this_bytes:
            raise RuntimeError(
                f"short read at row {rows_processed}: "
                f"got {len(raw_data)} bytes, expected {this_bytes}"
            )

        chunk_values = struct.unpack(f"<{this_chunk}d", raw_data)

        for raw_val in chunk_values:
            raw_sum_fp64 += raw_val
            raw_min = min(raw_min, raw_val)
            raw_max = max(raw_max, raw_val)

            if math.isnan(raw_val) or math.isinf(raw_val) or raw_val < 0.0:
                raise ValueError(f"invalid raw value at row {rows_processed}: {raw_val}")

            scaled = raw_val * scale
            encoded = int(round(scaled))

            if encoded < 0 or encoded > 18446744073709551615:
                overflow_count += 1
                if overflow_count == 1:
                    raw_f.close()
                    for f in plane_f:
                        f.close()
                    raise RuntimeError(
                        f"OVERFLOW at row {rows_processed}: raw={raw_val} "
                        f"scaled={scaled} encoded={encoded}. "
                        f"Overflow count: {overflow_count}. "
                        f"Artifact generation hard-failed."
                    )
                continue

            max_encoded = max(max_encoded, encoded)
            clean_encoded_sum += encoded
            if encoded == 0:
                zero_encoded_count += 1

            quant_err = abs(scaled - round(scaled)) / scale
            quantization_max_error = max(quantization_max_error, quant_err)
            quantization_sum_error += quant_err

            # Write each plane byte
            for p in range(8):
                byte_val = (encoded >> (8 * (7 - p))) & 0xFF
                plane_f[p].write(struct.pack("B", byte_val))

            rows_processed += 1

    raw_f.close()
    for f in plane_f:
        f.close()

    # Rename temp files to final names
    plane_paths: list[Path] = []
    plane_checksums: list[str] = []
    plane_nonzero: list[int] = []
    for p in range(8):
        final = artifact_dir / f"plane_{p}.bin"
        if plane_tmp[p].exists():
            plane_tmp[p].rename(final)
        plane_paths.append(final)
        plane_checksums.append(sha256_file(final))

        # Count nonzero bytes (stream over file)
        nonzero = 0
        with final.open("rb") as f:
            while True:
                buf = f.read(1 << 20)  # 1MB buffer
                if not buf:
                    break
                nonzero += sum(1 for b in buf if b != 0)
        plane_nonzero.append(nonzero)

    # Record synth_datasets metadata if available
    raw_dir = raw_path.parent
    dataset_meta_path = raw_dir / f"{dataset}.meta.json"
    synth_derived_seed = None
    generator_parameters = None
    generator_distribution = None
    if dataset_meta_path.exists():
        try:
            dm = json.loads(dataset_meta_path.read_text())
            synth_derived_seed = dm.get("seed")
            generator_parameters = dm.get("distribution")
            generator_distribution = dm.get("description")
        except Exception:
            pass

    metadata = {
        "dataset": dataset,
        "canonical_input_seed": canonical_seed,
        "synth_derived_seed": synth_derived_seed,
        "generator_distribution": generator_distribution,
        "generator_parameters": generator_parameters,
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
        "raw_min": raw_min,
        "raw_max": raw_max,
        "encoded_max": max_encoded,
        "overflow_count": overflow_count,
        "quantization_max_error": quantization_max_error,
        "quantization_mean_error": quantization_sum_error / rows_processed if rows_processed else 0.0,
        "quantization_zero_count": zero_encoded_count,
        "raw_fp64_sum": raw_sum_fp64,
        "clean_encoded_sum": str(clean_encoded_sum),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip() or "unknown",
        "generation_command": shlex.join(sys.argv),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    meta_path = artifact_dir / "artifact.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def verify_artifact(artifact_dir: Path, metadata: dict[str, Any]) -> int:
    """Verify artifact integrity via streamed reconstruction.

    Reads plane files in chunks to avoid OOM.
    """
    n_rows = metadata["n_rows"]
    plane_files = [artifact_dir / f"plane_{p}.bin" for p in range(8)]

    total = 0
    offset = 0
    buf_size = min(CHUNK_ROWS, n_rows)

    while offset < n_rows:
        remaining = n_rows - offset
        this_chunk = min(buf_size, remaining)

        # Read one chunk of each plane
        plane_chunks = []
        for p in range(8):
            with plane_files[p].open("rb") as f:
                f.seek(offset)
                data = f.read(this_chunk)
                if len(data) < this_chunk:
                    raise RuntimeError(f"short read plane_{p} at offset {offset}")
                plane_chunks.append(data)

        for i in range(this_chunk):
            row = 0
            for p in range(8):
                row |= plane_chunks[p][i] << (8 * (7 - p))
            total += row

        offset += this_chunk

    meta_sum = int(metadata["clean_encoded_sum"])
    assert total == meta_sum, (
        f"CPU reconstruction SUM mismatch: {total} != {meta_sum}"
    )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+",
                        default=["sensor", "uniform", "heavy_tailed", "zipfian"])
    parser.add_argument("--n-rows", type=int, default=100000000)
    parser.add_argument("--scale", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--synth-bin", default="buff_encoder/synth_datasets")
    parser.add_argument("--skip-raw", action="store_true",
                        help="Skip raw dataset generation (use existing files)")
    args = parser.parse_args()

    raw_root = Path(args.raw_data_root)
    artifact_root = Path(args.artifact_root)

    # Step 1: Generate raw datasets
    if args.skip_raw:
        raw_paths = {
            ds: raw_root / "reliability_layer1" / "raw" / f"{ds}.f64le.bin"
            for ds in args.datasets
        }
        missing = [ds for ds, p in raw_paths.items() if not p.exists()]
        if missing:
            raise SystemExit(f"skip-raw but missing: {missing}")
        print(f"Using existing raw datasets in {raw_root / 'reliability_layer1' / 'raw'}")
    else:
        synth_bin = Path(args.synth_bin)
        if not synth_bin.exists():
            raise SystemExit(
                f"synth_datasets binary not found: {synth_bin}. "
                f"Build first: cd buff_encoder && gcc -O3 -lm -o synth_datasets synth_datasets.c"
            )
        raw_dir = raw_root / "reliability_layer1" / "raw"
        raw_paths = generate_raw_datasets(
            raw_dir, args.datasets, args.n_rows, args.seed, str(synth_bin)
        )

    # Step 2: Convert to fixed-width artifacts (chunked)
    for ds in args.datasets:
        raw_path = raw_paths[ds]
        art_dir = artifact_root / "artifacts" / ds / f"n{args.n_rows}" / f"scale{args.scale}"
        print(f"Converting {ds} -> {art_dir}")

        metadata = convert_raw_to_planes_chunked(
            raw_path, ds, args.n_rows, args.scale, art_dir, args.seed,
        )

        # Step 3: Verify
        cpu_sum = verify_artifact(art_dir, metadata)
        meta_sum = int(metadata["clean_encoded_sum"])
        status = "OK" if cpu_sum == meta_sum else "MISMATCH"
        nz0 = metadata["plane_nonzero_count"][0]
        print(f"  {status}: cpu_sum={cpu_sum} meta_sum={meta_sum} "
              f"nonzero_plane0={nz0}")

    print("All artifacts generated and verified.")


if __name__ == "__main__":
    main()
