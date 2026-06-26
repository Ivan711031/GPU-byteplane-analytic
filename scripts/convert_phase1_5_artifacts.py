#!/usr/bin/env python3
"""Phase 1.5 per-strategy artifact converter wrapper.

Reuses Phase 1 converter (build_reliability_artifacts.py) for non-hybrid
strategies. For hybrid, encodes at scale=100 then left-shifts by shift_k.

Usage:
  python3 scripts/convert_phase1_5_artifacts.py

Args:
  --strategy-id    Strategy to process (default: all three)
  --params-json    Path to phase1_5_strategy_params.json
  --datasets       Datasets to convert (default: all four)
  --n-rows         Row count (default: 100000000)
  --raw-data-root  RAW_DATA_ROOT override
  --artifact-root  RELIABILITY_ARTIFACT_ROOT override
  --seed           Canonical seed
  --skip-csv       Path for skipped conversion log
"""

from __future__ import annotations

import argparse
import csv
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_reliability_artifacts import (
    CHUNK_ROWS,
    convert_raw_to_planes_chunked,
    sha256_file,
    verify_artifact,
)

UINT64_MAX = 18446744073709551615
CANONICAL_SEED = 20260413

# Params JSON strategy_id -> artifact directory short name
STRATEGY_DIR = {
    "native_scale1": "native_scale1",
    "per_dataset": "per_dataset",
    "hybrid_scale100_plus_shifted": "hybrid",
}

SKIPPED_CSV_FIELDS = [
    "strategy_id", "strategy_scale", "strategy_shift_k",
    "dataset", "n_rows", "reason", "details",
]


def load_params(params_path: Path) -> dict[str, Any]:
    """Load and validate strategy params JSON."""
    data = json.loads(params_path.read_text())
    sdata = data.get("strategies")
    if not sdata:
        raise ValueError("params JSON missing 'strategies' key")
    for sid in ("native_scale1", "per_dataset", "hybrid_scale100_plus_shifted"):
        if sid not in sdata:
            raise ValueError(f"params JSON missing strategy '{sid}'")
        ds_map = sdata[sid].get("per_dataset")
        if not ds_map:
            raise ValueError(f"strategy '{sid}' missing 'per_dataset'")
        for ds in ("sensor", "uniform", "heavy_tailed", "zipfian"):
            if ds not in ds_map:
                raise ValueError(f"strategy '{sid}' per_dataset missing '{ds}'")
            entry = ds_map[ds]
            if "scale" not in entry:
                raise ValueError(f"strategy '{sid}' dataset '{ds}' missing 'scale'")
    return data


def get_strategy_params(
    params: dict[str, Any], strategy_id: str, dataset: str,
) -> tuple[int, int]:
    """Return (scale, shift_k) for the given strategy and dataset."""
    entry = params["strategies"][strategy_id]["per_dataset"][dataset]
    return entry["scale"], entry.get("shift_k", 0)


def make_artifact_dir(
    artifact_root: Path, strategy_id: str, dataset: str,
    n_rows: int, scale: int, shift_k: int,
) -> Path:
    """Build per-strategy artifact directory path."""
    dir_name = STRATEGY_DIR.get(strategy_id, strategy_id)
    if strategy_id == "hybrid_scale100_plus_shifted":
        scale_component = f"scale{scale}_shift{shift_k}"
    else:
        scale_component = f"scale{scale}"
    return artifact_root / "artifacts_phase1_5" / dir_name / dataset / f"n{n_rows}" / scale_component


def convert_hybrid(
    raw_path: Path, dataset: str, n_rows: int,
    shift_k: int, artifact_dir: Path, canonical_seed: int,
) -> dict[str, Any]:
    """Convert raw FP64 to shifted 8-plane artifacts for hybrid strategy.

    Encodes at scale=100, left-shifts by shift_k, splits MSB-first into 8 planes.
    Accumulates same statistics as Phase 1 converter but clean_encoded_sum
    reflects shifted values.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scale = 100

    raw_f = raw_path.open("rb")
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

    while rows_processed < n_rows:
        remaining = n_rows - rows_processed
        this_chunk = min(CHUNK_ROWS, remaining)
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
                raise ValueError(
                    f"invalid raw value at row {rows_processed}: {raw_val}"
                )

            u = int(round(raw_val * scale))
            if u < 0 or u > UINT64_MAX:
                overflow_count += 1
                if overflow_count == 1:
                    raw_f.close()
                    for f in plane_f:
                        f.close()
                    raise RuntimeError(
                        f"OVERFLOW (scale=100) at row {rows_processed}: "
                        f"raw={raw_val} scaled={raw_val * scale} encoded={u}."
                    )
                continue

            u_shifted = u << shift_k
            if u_shifted > UINT64_MAX:
                overflow_count += 1
                if overflow_count == 1:
                    raw_f.close()
                    for f in plane_f:
                        f.close()
                    raise RuntimeError(
                        f"OVERFLOW (shift) at row {rows_processed}: "
                        f"raw={raw_val} u={u} shift_k={shift_k} "
                        f"u_shifted={u_shifted}."
                    )
                continue

            max_encoded = max(max_encoded, u_shifted)
            clean_encoded_sum += u_shifted
            if u_shifted == 0:
                zero_encoded_count += 1

            quant_err = abs(raw_val * scale - round(raw_val * scale)) / scale
            quantization_max_error = max(quantization_max_error, quant_err)
            quantization_sum_error += quant_err

            for p in range(8):
                byte_val = (u_shifted >> (8 * (7 - p))) & 0xFF
                plane_f[p].write(struct.pack("B", byte_val))

            rows_processed += 1

    raw_f.close()
    for f in plane_f:
        f.close()

    plane_paths: list[Path] = []
    plane_checksums: list[str] = []
    plane_nonzero: list[int] = []
    for p in range(8):
        final = artifact_dir / f"plane_{p}.bin"
        if plane_tmp[p].exists():
            plane_tmp[p].rename(final)
        plane_paths.append(final)
        plane_checksums.append(sha256_file(final))
        nonzero = 0
        with final.open("rb") as f:
            while True:
                buf = f.read(1 << 20)
                if not buf:
                    break
                nonzero += sum(1 for b in buf if b != 0)
        plane_nonzero.append(nonzero)

    # Record raw metadata if available
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

    metadata: dict[str, Any] = {
        "dataset": dataset,
        "canonical_input_seed": canonical_seed,
        "synth_derived_seed": synth_derived_seed,
        "generator_distribution": generator_distribution,
        "generator_parameters": generator_parameters,
        "n_rows": n_rows,
        "scale": scale,
        "shift_k": shift_k,
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
        "quantization_mean_error": (
            quantization_sum_error / rows_processed if rows_processed else 0.0
        ),
        "quantization_zero_count": zero_encoded_count,
        "raw_fp64_sum": raw_sum_fp64,
        "clean_encoded_sum": str(clean_encoded_sum),
        "git_commit": subprocess
        .run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        .stdout.strip()
        or "unknown",
        "generation_command": shlex.join(sys.argv),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    (artifact_dir / "artifact.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def log_skipped(csv_path: Path, entry: dict[str, Any]) -> None:
    """Append a row to the skipped CSV, creating header on first write."""
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SKIPPED_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy-id",
        choices=["native_scale1", "per_dataset", "hybrid_scale100_plus_shifted"],
        help="Single strategy to process (default: all three)",
    )
    parser.add_argument(
        "--params-json", type=Path,
        default=Path("results/phase1_5_strategy_params.json"),
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["sensor", "uniform", "heavy_tailed", "zipfian"],
    )
    parser.add_argument("--n-rows", type=int, default=100000000)
    parser.add_argument(
        "--raw-data-root", type=Path,
        default=Path(os.environ.get("RAW_DATA_ROOT", "/work/u4063895/datasets")),
    )
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path(
            os.environ.get(
                "RELIABILITY_ARTIFACT_ROOT",
                "/work/u4063895/datasets/reliability_layer1",
            )
        ),
    )
    parser.add_argument("--seed", type=int, default=CANONICAL_SEED)
    parser.add_argument("--skip-csv", type=Path, help="Overflow skip log path")
    args = parser.parse_args()

    raw_root = args.raw_data_root / "reliability_layer1" / "raw"
    artifact_root = args.artifact_root
    skip_csv_path = args.skip_csv or (
        artifact_root / "phase1_5_conversion_skipped.csv"
    )

    params = load_params(args.params_json)

    strategy_ids = ["native_scale1", "per_dataset", "hybrid_scale100_plus_shifted"]
    if args.strategy_id:
        strategy_ids = [args.strategy_id]

    for sid in strategy_ids:
        for ds in args.datasets:
            scale, shift_k = get_strategy_params(params, sid, ds)
            raw_path = raw_root / f"{ds}.f64le.bin"
            if not raw_path.exists():
                print(f"SKIP {sid}/{ds}: raw file not found {raw_path}")
                log_skipped(skip_csv_path, {
                    "strategy_id": sid,
                    "strategy_scale": scale,
                    "strategy_shift_k": shift_k,
                    "dataset": ds,
                    "n_rows": args.n_rows,
                    "reason": "raw_file_missing",
                    "details": str(raw_path),
                })
                continue

            art_dir = make_artifact_dir(
                artifact_root, sid, ds, args.n_rows, scale, shift_k,
            )

            if art_dir.exists() and (art_dir / "artifact.json").exists():
                plane_files = [art_dir / f"plane_{p}.bin" for p in range(8)]
                if all(p.exists() for p in plane_files):
                    print(f"EXISTS {sid}/{ds} -> {art_dir}")
                    continue

            print(
                f"Converting {sid}/{ds} scale={scale} "
                f"shift_k={shift_k} -> {art_dir}"
            )

            try:
                if sid == "hybrid_scale100_plus_shifted":
                    metadata = convert_hybrid(
                        raw_path, ds, args.n_rows, shift_k,
                        art_dir, args.seed,
                    )
                else:
                    metadata = convert_raw_to_planes_chunked(
                        raw_path, ds, args.n_rows, scale,
                        art_dir, args.seed,
                    )
            except RuntimeError as e:
                if "OVERFLOW" in str(e):
                    print(f"  OVERFLOW: skipping {sid}/{ds}")
                    log_skipped(skip_csv_path, {
                        "strategy_id": sid,
                        "strategy_scale": scale,
                        "strategy_shift_k": shift_k,
                        "dataset": ds,
                        "n_rows": args.n_rows,
                        "reason": "inapplicable_overflow",
                        "details": str(e),
                    })
                    continue
                raise

            # Inject per-strategy metadata
            meta_path = art_dir / "artifact.json"
            meta = json.loads(meta_path.read_text())
            meta["strategy_id"] = sid
            meta["strategy_scale"] = scale
            meta["strategy_shift_k"] = shift_k
            meta_path.write_text(json.dumps(meta, indent=2))

            # Clean gate: CPU reconstruction
            cpu_sum = verify_artifact(art_dir, meta)
            meta_sum = int(meta["clean_encoded_sum"])
            status = "OK" if cpu_sum == meta_sum else "MISMATCH"
            nz0 = meta["plane_nonzero_count"][0]
            print(
                f"  {status}: cpu_sum={cpu_sum} meta_sum={meta_sum} "
                f"nonzero_plane0={nz0}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
