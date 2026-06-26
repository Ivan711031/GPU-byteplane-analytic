#!/usr/bin/env python3
"""Run byteplane_count_gt_batch_cli over artifact matrices and write a CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_csv_floats(text: str) -> list[float]:
    return [float(item) for item in text.replace(",", " ").split() if item.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(item) for item in text.replace(",", " ").split() if item.strip()]


def resolve_dataset_name(artifact_dir: Path) -> str:
    """Resolve the dataset slug from an artifact directory.

    For v2 artifacts (named ``dataset_pN``) the raw file is ``dataset.f64le.bin``.
    For exp3 artifacts the directory name *is* the dataset slug.

    Priority:
    1. ``manifest.json`` / ``summary.json`` ``dataset`` field (if present).
    2. Strip ``_p\\d+$`` suffix (v2 naming convention).
    3. Fallback to directory name as-is (old exp3 behaviour).
    """
    name = artifact_dir.name

    for candidate in ("manifest.json", "summary.json"):
        p = artifact_dir / candidate
        if p.exists():
            try:
                ds = json.loads(p.read_text()).get("dataset")
                if ds:
                    return ds
            except Exception:
                continue

    m = re.search(r"(.*)_p\d+$", name)
    if m:
        return m.group(1)

    return name


def thresholds_for_artifact(
    artifact: Path,
    raw_root: Path,
    thresholds: list[float],
    selectivities: list[float],
) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = [("", threshold) for threshold in thresholds]
    if not selectivities:
        return items
    dataset = resolve_dataset_name(artifact)
    raw_path = raw_root / f"{dataset}.f64le.bin"
    values = np.memmap(raw_path, dtype="<f8", mode="r")
    for selectivity in selectivities:
        threshold = float(np.quantile(values, 1.0 - selectivity / 100.0))
        items.append((f"{selectivity:g}", threshold))
    return items


def run_batch(
    *,
    binary: Path,
    artifact: Path,
    raw_root: Path,
    segment_minmax_root: Path | None,
    row_state_dir: Path | None,
    direct_refine_raw: bool,
    compact_u_refine_raw: bool,
    threshold_items: list[tuple[str, float]],
    ks: list[int],
    device: int,
    block: int,
    repeat: int | None = None,
) -> tuple[dict[str, Any], float]:
    thresholds = ",".join(f"{threshold:.17g}" for _, threshold in threshold_items)
    cmd = [
        str(binary),
        "--artifact-root",
        str(artifact),
        "--raw-root",
        str(raw_root),
        "--thresholds",
        thresholds,
        "--ks",
        ",".join(str(k) for k in ks),
        "--device",
        str(device),
        "--block",
        str(block),
    ]
    if direct_refine_raw:
        cmd.append("--direct-refine-raw")
    if compact_u_refine_raw:
        cmd.append("--compact-u-refine-raw")
    if segment_minmax_root is not None:
        cmd.extend(["--segment-minmax", str(segment_minmax_root / f"{artifact.name}.csv")])
    if repeat is not None:
        cmd.extend(["--repeat", str(repeat)])
    if row_state_dir is not None and not direct_refine_raw and not compact_u_refine_raw:
        cmd.extend(["--row-state-dir", str(row_state_dir / artifact.name)])

    start = time.perf_counter()
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    wall_ms = (time.perf_counter() - start) * 1000.0
    return json.loads(completed.stdout), wall_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build/exp4/byteplane_count_gt_batch_cli")
    parser.add_argument("--artifact-root", action="append", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--segment-minmax-root")
    parser.add_argument("--row-state-dir")
    parser.add_argument("--direct-refine-raw", action="store_true")
    parser.add_argument("--compact-u-refine-raw", action="store_true")
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--selectivities", default="")
    parser.add_argument("--ks", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--block", default=256, type=int)
    args = parser.parse_args()

    thresholds = parse_csv_floats(args.thresholds)
    selectivities = parse_csv_floats(args.selectivities)
    if not thresholds and not selectivities:
        parser.error("at least one of --thresholds or --selectivities is required")
    if args.direct_refine_raw and args.compact_u_refine_raw:
        parser.error("--direct-refine-raw and --compact-u-refine-raw are mutually exclusive")
    ks = parse_csv_ints(args.ks)

    binary = Path(args.binary)
    raw_root = Path(args.raw_root)
    segment_minmax_root = Path(args.segment_minmax_root) if args.segment_minmax_root else None
    row_state_dir = Path(args.row_state_dir) if args.row_state_dir else None
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if row_state_dir is not None and not args.direct_refine_raw and not args.compact_u_refine_raw:
        row_state_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "artifact_root",
        "dataset",
        "selectivity",
        "threshold",
        "k",
        "rows",
        "max_planes",
        "count",
        "refined_exact_count",
        "output_mode",
        "Q",
        "D",
        "U",
        "U_count",
        "count_lower",
        "count_upper",
        "bytes_read",
        "compact_U_bytes",
        "U_fraction",
        "artifact_load_ms",
        "segment_minmax_ms",
        "threshold_prep_ms",
        "threshold_classify_ms",
        "threshold_encode_ms",
        "threshold_base_ms",
        "threshold_subtract_extract_ms",
        "threshold_pack_ms",
        "threshold_static_candidate_ms",
        "threshold_dependent_candidate_ms",
        "threshold_mixed_segments",
        "threshold_allq_segments",
        "threshold_alld_segments",
        "gpu_stage_ms",
        "gpu_total_ms",
        "primitive_ms",
        "raw_stage_ms",
        "per_query_ms_median",
        "per_query_ms_p10",
        "per_query_ms_p90",
        "direct_refine_ms",
        "direct_total_ms",
        "compact_u_refine_ms",
        "total_refined_ms",
        "compact_u_ms",
        "batch_wall_ms",
        "batch_wall_external_ms",
        "amortized_batch_wall_ms",
        "row_state_path",
        "used_gpu",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for artifact_text in args.artifact_root:
            artifact = Path(artifact_text)
            threshold_items = thresholds_for_artifact(artifact, raw_root, thresholds, selectivities)
            payload, external_wall_ms = run_batch(
                binary=binary,
                artifact=artifact,
                raw_root=raw_root,
                segment_minmax_root=segment_minmax_root,
                row_state_dir=row_state_dir,
                direct_refine_raw=args.direct_refine_raw,
                compact_u_refine_raw=args.compact_u_refine_raw,
                threshold_items=threshold_items,
                ks=ks,
                device=args.device,
                block=args.block,
                repeat=args.repeat,
            )
            results = payload["results"]
            batch_wall_ms = float(payload["batch_wall_ms"])
            threshold_lookup = {
                f"{threshold:.17g}": selectivity for selectivity, threshold in threshold_items
            }
            for result in results:
                threshold = float(result["threshold"])
                threshold_key = f"{threshold:.17g}"
                exact_count = result["refined_exact_count"] if result.get("output_mode") != "encoded_interval" else ""
                row = {
                    "artifact_root": str(artifact),
                    "dataset": result["dataset"],
                    "selectivity": threshold_lookup.get(threshold_key, ""),
                    "threshold": result["threshold"],
                    "k": result["k"],
                    "rows": result["rows"],
                    "max_planes": result["max_planes"],
                    "count": result["count"],
                    "refined_exact_count": exact_count,
                    "output_mode": result.get("output_mode", "encoded_interval"),
                    "Q": result["Q"],
                    "D": result["D"],
                    "U": result["U"],
                    "U_count": result.get("U_count", result["U"]),
                    "count_lower": result["Q"],
                    "count_upper": result["Q"] + result["U"],
                    "bytes_read": result["bytes_read"],
                    "compact_U_bytes": result.get("compact_U_bytes", 0),
                    "U_fraction": result["U"] / result["rows"] if result["rows"] else 0.0,
                    "artifact_load_ms": result.get("artifact_load_ms", ""),
                    "segment_minmax_ms": result.get("segment_minmax_ms", ""),
                    "threshold_prep_ms": result.get("threshold_prep_ms", ""),
                    "threshold_classify_ms": result.get("threshold_classify_ms", ""),
                    "threshold_encode_ms": result.get("threshold_encode_ms", ""),
                    "threshold_base_ms": result.get("threshold_base_ms", ""),
                    "threshold_subtract_extract_ms": result.get("threshold_subtract_extract_ms", ""),
                    "threshold_pack_ms": result.get("threshold_pack_ms", ""),
                    "threshold_static_candidate_ms": result.get("threshold_static_candidate_ms", ""),
                    "threshold_dependent_candidate_ms": result.get("threshold_dependent_candidate_ms", ""),
                    "threshold_mixed_segments": result.get("threshold_mixed_segments", ""),
                    "threshold_allq_segments": result.get("threshold_allq_segments", ""),
                    "threshold_alld_segments": result.get("threshold_alld_segments", ""),
                    "gpu_stage_ms": result.get("gpu_stage_ms", ""),
                    "gpu_total_ms": result.get("gpu_total_ms", ""),
                    "primitive_ms": result.get("primitive_ms", ""),
                    "raw_stage_ms": result.get("raw_stage_ms", 0.0),
                    "per_query_ms_median": result.get("per_query_ms_median", ""),
                    "per_query_ms_p10": result.get("per_query_ms_p10", ""),
                    "per_query_ms_p90": result.get("per_query_ms_p90", ""),
                    "direct_refine_ms": result.get("direct_refine_ms", 0.0),
                    "direct_total_ms": result.get("direct_total_ms", 0.0),
                    "compact_u_refine_ms": result.get("compact_u_refine_ms", 0.0),
                    "total_refined_ms": result.get("total_refined_ms", 0.0),
                    "compact_u_ms": result.get("compact_u_ms", 0.0),
                    "batch_wall_ms": batch_wall_ms,
                    "batch_wall_external_ms": external_wall_ms,
                    "amortized_batch_wall_ms": batch_wall_ms / len(results) if results else "",
                    "row_state_path": result.get("row_state_path", ""),
                    "used_gpu": result["used_gpu"],
                }
                writer.writerow(row)
                print(json.dumps(row, sort_keys=True), file=sys.stderr)
            f.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
