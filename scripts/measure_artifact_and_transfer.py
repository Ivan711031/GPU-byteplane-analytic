#!/usr/bin/env python3
"""Measure encoded artifact size and optional host-to-GPU transfer time."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = [
    "dataset",
    "artifact_version",
    "artifact_label",
    "rows",
    "raw_bytes",
    "main_artifact_bytes",
    "side_artifact_bytes",
    "total_artifact_bytes",
    "bytes_per_row",
    "compression_ratio_vs_raw_fp64",
    "active_plane_count_min",
    "active_plane_count_max",
    "active_plane_count_mean",
    "transfer_mode",
    "transfer_bytes",
    "cudaMemcpy_ms",
    "effective_transfer_GBps",
]

CORE_FILENAMES = {"manifest.json", "summary.json", "segment_meta.csv"}
PLANE_RE = re.compile(r"plane_(\d{3})\.bin$")
SIDE_TABLE_RE = re.compile(
    r"^(?:side|aux|outlier|spill)(?:[_-](?:table|tables|meta|data))?(?:[_-][A-Za-z0-9][A-Za-z0-9._-]*)*\.(?:bin|csv|tsv|txt|json|parquet)$",
    re.IGNORECASE,
)
SIDE_REFERENCE_KEYS = ("table", "side", "aux", "outlier", "spill")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        dest="artifact_roots",
        action="append",
        type=Path,
        required=True,
        help="Artifact root or parent directory containing dataset subdirectories.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Write the measurement CSV to this path instead of stdout.",
    )
    parser.add_argument(
        "--transfer-command",
        help=(
            "Optional command template to time for transfer measurement. "
            "Supported placeholders: {artifact_dir}, {artifact_label}, {artifact_version}, "
            "{dataset}, {rows}, {raw_bytes}, {main_artifact_bytes}, {side_artifact_bytes}, "
            "{total_artifact_bytes}, {transfer_bytes}."
        ),
    )
    parser.add_argument(
        "--transfer-bytes",
        type=int,
        help="Optional explicit transfer byte count to report and time.",
    )
    parser.add_argument(
        "--transfer-mode",
        help="Optional label for the transfer timing mode.",
    )
    parser.add_argument(
        "--provenance-out",
        type=Path,
        help="Optional provenance JSON path. Defaults to <csv-out>.provenance.json when csv-out is set.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_segment_meta(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty segment metadata: {path}") from exc
        rows: list[dict[str, str]] = []
        for line_no, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"segment_meta.csv column count mismatch at {path}:{line_no} "
                    f"(expected {len(header)}, got {len(row)})"
                )
            rows.append(dict(zip(header, row)))
    return rows


def discover_dataset_dirs(artifact_root: Path) -> list[Path]:
    if not artifact_root.exists():
        raise FileNotFoundError(f"artifact root does not exist: {artifact_root}")

    direct = artifact_root if (artifact_root / "manifest.json").is_file() else None
    dataset_dirs = [direct] if direct else []

    for manifest_path in artifact_root.rglob("manifest.json"):
        dataset_dir = manifest_path.parent
        if not (dataset_dir / "summary.json").is_file():
            continue
        if not (dataset_dir / "segment_meta.csv").is_file():
            continue
        if dataset_dir not in dataset_dirs:
            dataset_dirs.append(dataset_dir)

    if not dataset_dirs:
        raise FileNotFoundError(f"no artifact dataset directories found under {artifact_root}")

    return sorted(dataset_dirs)


def derive_artifact_label(artifact_root: Path, dataset_dir: Path) -> str:
    artifact_root = artifact_root.resolve()
    dataset_dir = dataset_dir.resolve()

    if dataset_dir == artifact_root:
        return artifact_root.name
    if dataset_dir.parent == artifact_root:
        # Flat layout: directory name = "{dataset}_{label}" (v2).
        return dataset_dir.name
    if dataset_dir.parent.parent == artifact_root:
        return dataset_dir.name
    return dataset_dir.name


def is_plane_file(path: Path) -> bool:
    return PLANE_RE.fullmatch(path.name) is not None


def plane_index(path: Path) -> int:
    match = PLANE_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"not a plane file: {path}")
    return int(match.group(1))


def is_side_candidate(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name in CORE_FILENAMES:
        return False
    if is_plane_file(path):
        return False
    return SIDE_TABLE_RE.fullmatch(path.name) is not None


def is_referenced_side_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name in CORE_FILENAMES:
        return False
    if is_plane_file(path):
        return False
    return True


def key_mentions_side_reference(key: str) -> bool:
    key_lower = key.lower()
    return any(token in key_lower for token in SIDE_REFERENCE_KEYS)


def collect_candidate_paths(value: Any, *, key_context: bool = False) -> set[str]:
    out: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            out.update(collect_candidate_paths(child, key_context=key_context or key_mentions_side_reference(key)))
    elif isinstance(value, list):
        for item in value:
            out.update(collect_candidate_paths(item, key_context=key_context))
    elif isinstance(value, str):
        if key_context:
            out.add(value)
    return out


def resolve_side_files(dataset_dir: Path, manifest: dict[str, Any], summary: dict[str, Any], segment_rows: list[dict[str, str]]) -> list[Path]:
    discovered: dict[Path, None] = {}
    dataset_root = dataset_dir.resolve()

    for path in dataset_dir.iterdir():
        if is_side_candidate(path):
            discovered[path.resolve()] = None

    for candidate in collect_candidate_paths(manifest) | collect_candidate_paths(summary):
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = (dataset_dir / candidate_path).resolve()
        if candidate_path.is_file():
            resolved = candidate_path.resolve()
            if resolved.parent == dataset_root or dataset_root in resolved.parents:
                if is_referenced_side_file(resolved):
                    discovered[resolved] = None

    for row in segment_rows:
        for key, value in row.items():
            if value is None or value == "":
                continue
            if key_mentions_side_reference(key):
                candidate_path = Path(value)
                if not candidate_path.is_absolute():
                    candidate_path = (dataset_dir / candidate_path).resolve()
                if candidate_path.is_file():
                    resolved = candidate_path.resolve()
                    if resolved.parent == dataset_root or dataset_root in resolved.parents:
                        if is_referenced_side_file(resolved):
                            discovered[resolved] = None

    return sorted(discovered)


def format_number(value: float | int) -> float | int:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def measure_dataset(dataset_dir: Path, artifact_label: str, transfer_command: str | None, transfer_mode: str | None, transfer_bytes_override: int | None) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    summary_path = dataset_dir / "summary.json"
    segment_meta_path = dataset_dir / "segment_meta.csv"

    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    segment_rows = read_segment_meta(segment_meta_path)

    dataset = manifest.get("dataset")
    if not dataset:
        raise ValueError(f"manifest missing dataset field: {manifest_path}")

    # Fix flat-layout label: "uniform_p10" -> "p10"
    prefix = f"{dataset}_"
    if artifact_label.startswith(prefix):
        artifact_label = artifact_label[len(prefix):]

    rows = int(manifest.get("value_count") or sum(int(row["row_count"]) for row in segment_rows))
    segment_row_total = sum(int(row["row_count"]) for row in segment_rows)
    if segment_row_total != rows:
        raise ValueError(
            f"row count mismatch for {dataset_dir}: manifest value_count={rows} "
            f"but segment_meta row_count sum={segment_row_total}"
        )

    source_path = manifest.get("source_path")
    if source_path and Path(source_path).is_file():
        raw_bytes = Path(source_path).stat().st_size
    else:
        raw_bytes = rows * 8

    plane_files = sorted((path for path in dataset_dir.iterdir() if path.is_file() and is_plane_file(path)), key=plane_index)
    max_plane_count = int(manifest.get("max_plane_count") or len(plane_files))
    if len(plane_files) != max_plane_count:
        raise ValueError(
            f"plane file count mismatch for {dataset_dir}: manifest max_plane_count={max_plane_count} "
            f"but found {len(plane_files)} plane_*.bin files"
        )

    main_artifact_bytes = sum(path.stat().st_size for path in plane_files)
    side_files = resolve_side_files(dataset_dir, manifest, summary, segment_rows)
    side_artifact_bytes = sum(path.stat().st_size for path in side_files)
    total_artifact_bytes = main_artifact_bytes + side_artifact_bytes

    active_counts = [int(row["active_plane_count"]) for row in segment_rows]
    active_plane_count_min = min(active_counts)
    active_plane_count_max = max(active_counts)
    active_plane_count_mean = sum(active_counts) / len(active_counts)

    bytes_per_row = total_artifact_bytes / rows if rows else 0.0
    compression_ratio = raw_bytes / total_artifact_bytes if total_artifact_bytes else math.inf

    transfer_mode_value = transfer_mode or ("external_command" if transfer_command else "not_run")
    transfer_bytes = transfer_bytes_override if transfer_bytes_override is not None else total_artifact_bytes
    cuda_memcpy_ms = 0.0
    effective_transfer_gbps = 0.0

    if transfer_command:
        context = {
            "artifact_dir": str(dataset_dir),
            "artifact_label": artifact_label,
            "artifact_version": str(manifest.get("format", "")),
            "dataset": dataset,
            "rows": rows,
            "raw_bytes": raw_bytes,
            "main_artifact_bytes": main_artifact_bytes,
            "side_artifact_bytes": side_artifact_bytes,
            "total_artifact_bytes": total_artifact_bytes,
            "transfer_bytes": transfer_bytes,
        }
        rendered_command = transfer_command.format_map(context)
        completed = subprocess.run(
            shlex.split(rendered_command),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "transfer command failed with exit code "
                f"{completed.returncode}: {rendered_command}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        # Parse helper stdout for "H2D <bytes> <ms>" — CUDA event timing.
        h2d_ms = 0.0
        h2d_bytes = 0
        for line in completed.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[0] == "H2D":
                try:
                    h2d_bytes = int(parts[1])
                    h2d_ms = float(parts[2])
                except ValueError:
                    pass
                break
        if h2d_ms > 0:
            cuda_memcpy_ms = h2d_ms
            transfer_bytes = h2d_bytes
            effective_transfer_gbps = (transfer_bytes / (cuda_memcpy_ms / 1000.0)) / 1e9
        elif completed.stdout.strip():
            # Fallback: use wall-clock timing if helper output not recognized
            # (e.g., generic external command without CUDA event output).
            start = time.perf_counter()
            subprocess.run(shlex.split(rendered_command), capture_output=True, check=False)
            cuda_memcpy_ms = (time.perf_counter() - start) * 1000.0
            if cuda_memcpy_ms > 0:
                effective_transfer_gbps = (transfer_bytes / (cuda_memcpy_ms / 1000.0)) / 1e9

    row = {
        "dataset": dataset,
        "artifact_version": str(manifest.get("format", "")),
        "artifact_label": artifact_label,
        "rows": rows,
        "raw_bytes": raw_bytes,
        "main_artifact_bytes": main_artifact_bytes,
        "side_artifact_bytes": side_artifact_bytes,
        "total_artifact_bytes": total_artifact_bytes,
        "bytes_per_row": bytes_per_row,
        "compression_ratio_vs_raw_fp64": compression_ratio,
        "active_plane_count_min": active_plane_count_min,
        "active_plane_count_max": active_plane_count_max,
        "active_plane_count_mean": active_plane_count_mean,
        "transfer_mode": transfer_mode_value,
        "transfer_bytes": transfer_bytes,
        "cudaMemcpy_ms": cuda_memcpy_ms,
        "effective_transfer_GBps": effective_transfer_gbps,
    }
    return {key: format_number(value) for key, value in row.items()}


def write_csv(rows: list[dict[str, Any]], csv_out: Path | None) -> None:
    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_provenance(
    provenance_out: Path,
    *,
    artifact_roots: list[Path],
    csv_out: Path | None,
    transfer_command: str | None,
    transfer_mode: str | None,
    rows: list[dict[str, Any]],
) -> None:
    provenance = {
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "hostname": socket.gethostname(),
        "gpu_name": None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifact_roots": [str(path) for path in artifact_roots],
        "csv_out": str(csv_out) if csv_out else None,
        "transfer_command": transfer_command,
        "transfer_mode": transfer_mode or ("external_command" if transfer_command else "not_run"),
        "row_count": len(rows),
    }
    provenance_out.parent.mkdir(parents=True, exist_ok=True)
    provenance_out.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()

    measured_rows: list[dict[str, Any]] = []
    for artifact_root in args.artifact_roots:
        for dataset_dir in discover_dataset_dirs(artifact_root):
            measured_rows.append(
                measure_dataset(
                    dataset_dir,
                    artifact_label=derive_artifact_label(artifact_root, dataset_dir),
                    transfer_command=args.transfer_command,
                    transfer_mode=args.transfer_mode,
                    transfer_bytes_override=args.transfer_bytes,
                )
            )

    measured_rows.sort(key=lambda row: (row["artifact_label"], row["dataset"]))
    write_csv(measured_rows, args.csv_out)

    if args.csv_out:
        provenance_out = args.provenance_out or args.csv_out.with_suffix(args.csv_out.suffix + ".provenance.json")
        write_provenance(
            provenance_out,
            artifact_roots=args.artifact_roots,
            csv_out=args.csv_out,
            transfer_command=args.transfer_command,
            transfer_mode=args.transfer_mode,
            rows=measured_rows,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
