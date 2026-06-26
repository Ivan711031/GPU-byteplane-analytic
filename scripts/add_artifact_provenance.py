#!/usr/bin/env python3
"""Add provenance + delta_checksum to existing active-delta artifacts.

Existing artifacts built before the #137 NEEDS_FIXES did not include
provenance fields or delta_checksum in artifact.json.  This script
re-processes the source raw data (one read pass) to compute those
fields and updates artifact.json in-place.

Usage (nano5):
  python3 scripts/add_artifact_provenance.py \\
    --artifact-dir /work/u4063895/datasets/artifacts_phase1_5b/sensor/n100000000/scale8233095970213 \\
    --source-path  /work/u4063895/datasets/reliability_layer1_phase1_5/reliability_layer1/raw/sensor.f64le.bin \\
    --strategy-params results/phase1_5_strategy_params.json \\
    --dataset sensor
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

CHUNK_ROWS = 500000


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--strategy-params", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n-rows", type=int, default=None,
                        help="Override n_rows (default: from artifact.json)")
    args = parser.parse_args()

    meta_path = args.artifact_dir / "artifact.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found")
        sys.exit(1)

    metadata: dict[str, Any] = json.loads(meta_path.read_text())
    scale = metadata["scale"]
    n_rows = args.n_rows or metadata["n_rows"]

    print(f"Artifact: {args.artifact_dir}")
    print(f"  dataset={args.dataset}, scale={scale}, n_rows={n_rows}")
    print(f"  source: {args.source_path}")

    # ---- single pass: source checksum + encode to fixed values ----
    print("  Reading source data and computing source checksum...")
    h_source = hashlib.sha256()
    raw_f = args.source_path.open("rb")
    fixed_vals: list[int] = []
    rows_processed = 0
    while rows_processed < n_rows:
        remaining = n_rows - rows_processed
        this_chunk = min(CHUNK_ROWS, remaining)
        chunk_bytes = this_chunk * 8
        raw_data = raw_f.read(chunk_bytes)
        if len(raw_data) < chunk_bytes:
            raise RuntimeError(f"short read at row {rows_processed}")
        h_source.update(raw_data)
        chunk_vals = struct.unpack(f"<{this_chunk}d", raw_data)
        for raw_val in chunk_vals:
            if math.isnan(raw_val) or math.isinf(raw_val) or raw_val < 0.0:
                raise ValueError(f"invalid raw at row {rows_processed}: {raw_val}")
            scaled = raw_val * scale
            encoded = int(round(scaled))
            if encoded < 0 or encoded > 18446744073709551615:
                continue
            fixed_vals.append(encoded)
            rows_processed += 1
    raw_f.close()
    print(f"  Read {len(fixed_vals)} valid rows")

    source_checksum = h_source.hexdigest()
    source_size = args.source_path.stat().st_size

    # ---- delta checksum ----
    print("  Computing delta checksum...")
    base_fixed = min(fixed_vals)
    deltas = [v - base_fixed for v in fixed_vals]
    h_delta = hashlib.sha256()
    for d in deltas:
        h_delta.update(struct.pack("<Q", d))
    delta_checksum = h_delta.hexdigest()

    # ---- strategy params checksum ----
    print("  Computing strategy params checksum...")
    sp_checksum = sha256_stream(args.strategy_params)
    sp_path = str(args.strategy_params.resolve())

    # ---- git / command ----
    git_commit = get_git_commit()
    cmd = "python3 " + " ".join(sys.argv)

    # ---- update artifact.json ----
    print("  Updating artifact.json...")
    metadata["input_type"] = "phase1_5_raw"
    metadata["source_path"] = str(args.source_path.resolve())
    metadata["source_size_bytes"] = source_size
    metadata["source_checksum"] = source_checksum
    metadata["strategy_params_path"] = sp_path
    metadata["strategy_params_checksum"] = sp_checksum
    metadata["generator_git_commit"] = git_commit
    metadata["generator_command"] = cmd
    metadata["delta_checksum"] = delta_checksum

    # recompute metadata_checksum (exclude checksum fields)
    exclude = {"plane_checksums", "metadata_checksum", "artifact_manifest_checksum",
               "source_checksum", "strategy_params_checksum", "delta_checksum"}
    cksum_input = {k: v for k, v in metadata.items() if k not in exclude}
    canonical = json.dumps(cksum_input, sort_keys=True, indent=2).encode()
    metadata["metadata_checksum"] = hashlib.sha256(canonical).hexdigest()

    sorted_plane = sorted(metadata["plane_checksums"])
    manifest_input = "".join(sorted_plane) + metadata["metadata_checksum"]
    metadata["artifact_manifest_checksum"] = hashlib.sha256(
        manifest_input.encode()
    ).hexdigest()

    meta_path.write_text(json.dumps(metadata, indent=2))
    print("  Done. New fields:")
    for field in ("input_type", "source_path", "source_size_bytes",
                  "source_checksum", "strategy_params_path",
                  "strategy_params_checksum", "generator_git_commit",
                  "generator_command", "delta_checksum",
                  "metadata_checksum", "artifact_manifest_checksum"):
        val = metadata.get(field)
        if isinstance(val, str) and len(val) > 80:
            val = val[:77] + "..."
        print(f"    {field}: {val}")


if __name__ == "__main__":
    main()
