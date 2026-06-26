#!/usr/bin/env python3
"""Download, preprocess, and encode a large NYC TLC Yellow Taxi trip_distance slice."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pyarrow.parquet as pq


DATE_TAG = "20260515"
DATASET_DISPLAY = "NYC TLC Yellow Taxi"
COLUMN_NAME = "trip_distance"
COLUMN_UNIT = "mile"
SEGMENT_SIZE = 4096
PRECISION_POWERS = (4, 6)
SOURCE_URL_TEMPLATE = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year_month}.parquet"
DEFAULT_START_MONTH = "2009-01"
DEFAULT_END_MONTH = "2025-12"
DEFAULT_TARGET_ROWS = 500_000_000
DEFAULT_DATASET_KEY = f"nyc_tlc_yellow_trip_distance_ge500m_{DATE_TAG}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-month",
        default=DEFAULT_START_MONTH,
        help="Earliest month to consider, inclusive, in YYYY-MM format.",
    )
    parser.add_argument(
        "--end-month",
        default=DEFAULT_END_MONTH,
        help="Latest month to consider, inclusive, in YYYY-MM format.",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=DEFAULT_TARGET_ROWS,
        help="Stop after cleaned row count reaches at least this value.",
    )
    parser.add_argument(
        "--dataset-key",
        default=DEFAULT_DATASET_KEY,
        help="Dataset key used for raw path and exported artifact directory.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(f"results/real_dataset_source/nyc_tlc_yellow_{DATE_TAG}"),
        help="Directory for downloaded monthly parquet files.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(f"results/real_dataset_source/nyc_tlc_yellow_{DATE_TAG}/selected_months_ge500m.json"),
        help="Manifest describing the selected source months.",
    )
    parser.add_argument(
        "--buff-tool",
        type=Path,
        default=Path("bin/buff_tool"),
        help="Path to buff_tool.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(f"results/buff_encoder_v2/raw_real_{DATE_TAG}"),
        help="Directory for the concatenated cleaned FP64 slice.",
    )
    parser.add_argument(
        "--encoded-dir",
        type=Path,
        default=Path(f"results/buff_encoder_v2/containers_real_{DATE_TAG}"),
        help="Directory for .buff64 containers.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(f"datasets/real/dev_buff_v2_tlc_500m_{DATE_TAG}"),
        help="Root directory for exported runtime artifacts.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(f"results/buff_encoder_v2/{DEFAULT_DATASET_KEY}_build_summary.json"),
        help="Build summary JSON path.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def run_checked(cmd: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            + " ".join(shlex.quote(part) for part in cmd)
            + "\nstdout:\n"
            + completed.stdout
            + "\nstderr:\n"
            + completed.stderr
        )


def parse_year_month(year_month: str) -> tuple[int, int]:
    year_text, month_text = year_month.split("-")
    year = int(year_text)
    month = int(month_text)
    if month < 1 or month > 12:
        raise ValueError(f"invalid month: {year_month}")
    return year, month


def iter_year_months_desc(start_month: str, end_month: str) -> list[str]:
    start_year, start_value = parse_year_month(start_month)
    end_year, end_value = parse_year_month(end_month)
    start_total = start_year * 12 + (start_value - 1)
    end_total = end_year * 12 + (end_value - 1)
    if end_total < start_total:
        raise ValueError(f"end month {end_month} is before start month {start_month}")

    months: list[str] = []
    current = end_total
    while current >= start_total:
        year = current // 12
        month = current % 12 + 1
        months.append(f"{year:04d}-{month:02d}")
        current -= 1
    return months


def month_source_url(year_month: str) -> str:
    return SOURCE_URL_TEMPLATE.format(year_month=year_month)


def parquet_path_for_month(source_dir: Path, year_month: str) -> Path:
    return source_dir / f"yellow_tripdata_{year_month}.parquet"


def download_parquet_for_month(source_dir: Path, year_month: str) -> Path:
    path = parquet_path_for_month(source_dir, year_month)
    if path.is_file():
        return path

    ensure_dir(path.parent)
    run_checked(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "3",
            "-A",
            "Mozilla/5.0",
            "--output",
            str(path),
            month_source_url(year_month),
        ],
        cwd=Path.cwd(),
    )
    return path


def materialize_trip_distance_slice(
    *,
    source_dir: Path,
    source_manifest: Path,
    raw_path: Path,
    start_month: str,
    end_month: str,
    target_rows: int,
) -> dict[str, Any]:
    ensure_dir(raw_path.parent)

    months_used: list[dict[str, Any]] = []
    total_before = 0
    total_after = 0
    null_count = 0
    non_finite_count = 0
    negative_count = 0
    zero_count = 0
    exact_sum = 0.0
    raw_min = float("inf")
    raw_max = float("-inf")

    with raw_path.open("wb") as out:
        for year_month in iter_year_months_desc(start_month, end_month):
            parquet_path = download_parquet_for_month(source_dir, year_month)
            pf = pq.ParquetFile(parquet_path)

            month_before = 0
            month_after = 0
            month_zero = 0
            month_min = float("inf")
            month_max = float("-inf")

            for batch in pf.iter_batches(columns=[COLUMN_NAME], batch_size=262144):
                column = batch.column(0)
                month_before += len(column)
                null_count += column.null_count

                values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)
                finite_mask = np.isfinite(values)
                nonneg_mask = values >= 0.0
                valid_mask = finite_mask & nonneg_mask

                non_finite_count += int((~finite_mask).sum())
                negative_count += int((finite_mask & ~nonneg_mask).sum())

                valid = values[valid_mask]
                if valid.size == 0:
                    continue

                month_after += int(valid.size)
                month_zero += int((valid == 0.0).sum())
                exact_sum += float(valid.sum(dtype=np.float64))
                batch_min = float(valid.min())
                batch_max = float(valid.max())
                month_min = min(month_min, batch_min)
                month_max = max(month_max, batch_max)
                raw_min = min(raw_min, batch_min)
                raw_max = max(raw_max, batch_max)
                out.write(np.asarray(valid, dtype="<f8").tobytes())

            total_before += month_before
            total_after += month_after
            zero_count += month_zero

            months_used.append(
                {
                    "year_month": year_month,
                    "parquet_path": str(parquet_path),
                    "source_url": month_source_url(year_month),
                    "row_count_before": month_before,
                    "row_count_after": month_after,
                    "zero_count": month_zero,
                    "raw_min": None if month_after == 0 else month_min,
                    "raw_max": None if month_after == 0 else month_max,
                }
            )

            if total_after >= target_rows:
                break

    if total_after < target_rows:
        raise RuntimeError(
            f"cleaned row count {total_after} did not reach target_rows {target_rows}"
        )

    write_json(
        source_manifest,
        {
            "dataset_display": DATASET_DISPLAY,
            "column_name": COLUMN_NAME,
            "column_unit": COLUMN_UNIT,
            "start_month": start_month,
            "end_month": end_month,
            "target_rows": target_rows,
            "months_used": months_used,
            "row_count_before": total_before,
            "row_count_after": total_after,
        },
    )

    return {
        "start_month": start_month,
        "end_month": end_month,
        "target_rows": target_rows,
        "months_used_count": len(months_used),
        "first_selected_month": months_used[0]["year_month"],
        "last_selected_month": months_used[-1]["year_month"],
        "row_count_before": total_before,
        "row_count_after": total_after,
        "null_count": null_count,
        "non_finite_count": non_finite_count,
        "negative_count": negative_count,
        "zero_count": zero_count,
        "positive_count": total_after - zero_count,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "exact_sum": exact_sum,
        "preprocessing_policy": {
            "drop_null": True,
            "drop_non_finite": True,
            "drop_negative": True,
            "drop_zero": False,
            "anomaly_filter": "none",
            "partial_final_month": False,
        },
    }


def build_artifact(
    *,
    buff_tool: Path,
    raw_path: Path,
    encoded_dir: Path,
    artifact_root: Path,
    source_path: Path,
    dataset_key: str,
    precision_power: int,
) -> dict[str, Any]:
    ensure_dir(encoded_dir)
    dataset_dir = artifact_root / dataset_key
    artifact_label = f"p{precision_power}"
    artifact_dir = dataset_dir / artifact_label
    buff64_path = encoded_dir / f"{dataset_key}_{artifact_label}.buff64"

    run_checked(
        [
            str(buff_tool),
            "encode",
            "--input",
            str(raw_path),
            "--output",
            str(buff64_path),
            "--segment-size",
            str(SEGMENT_SIZE),
            "--precision-power",
            str(precision_power),
        ],
        cwd=Path.cwd(),
    )

    run_checked(
        [
            str(buff_tool),
            "export-runtime",
            "--input",
            str(buff64_path),
            "--raw-input",
            str(raw_path),
            "--out-dir",
            str(artifact_dir),
            "--dataset",
            dataset_key,
            "--source-path",
            str(source_path),
        ],
        cwd=Path.cwd(),
    )

    manifest = read_json(artifact_dir / "manifest.json")
    summary = read_json(artifact_dir / "summary.json")
    value_count = int(summary["value_count"])
    container_bytes = buff64_path.stat().st_size
    return {
        "artifact_label": artifact_label,
        "precision_power": precision_power,
        "buff64_path": str(buff64_path),
        "artifact_dir": str(artifact_dir),
        "max_plane_count": manifest["max_plane_count"],
        "segment_plane_count_min": manifest["segment_plane_count_min"],
        "segment_plane_count_max": manifest["segment_plane_count_max"],
        "mean_plane_count": summary["mean_plane_count"],
        "quantization_bound": summary["quantization_bound"],
        "container_bytes": container_bytes,
        "container_bytes_per_row": container_bytes / value_count,
        "exact_sum": summary["exact_sum"],
        "exact_min": summary["exact_min"],
        "exact_max": summary["exact_max"],
    }


def main() -> int:
    args = parse_args()
    if not args.buff_tool.is_file():
        raise FileNotFoundError(f"buff_tool not found: {args.buff_tool}")

    raw_path = args.raw_dir / f"{args.dataset_key}.f64le.bin"
    preprocess = materialize_trip_distance_slice(
        source_dir=args.source_dir,
        source_manifest=args.source_manifest,
        raw_path=raw_path,
        start_month=args.start_month,
        end_month=args.end_month,
        target_rows=args.target_rows,
    )

    artifacts = {}
    for power in PRECISION_POWERS:
        artifacts[f"p{power}"] = build_artifact(
            buff_tool=args.buff_tool,
            raw_path=raw_path,
            encoded_dir=args.encoded_dir,
            artifact_root=args.artifact_root,
            source_path=args.source_manifest,
            dataset_key=args.dataset_key,
            precision_power=power,
        )

    write_json(
        args.summary_json,
        {
            "dataset_display": DATASET_DISPLAY,
            "dataset_key": args.dataset_key,
            "column_name": COLUMN_NAME,
            "column_unit": COLUMN_UNIT,
            "source_manifest": str(args.source_manifest),
            "raw_path": str(raw_path),
            "segment_size": SEGMENT_SIZE,
            "preprocess": preprocess,
            "artifacts": artifacts,
        },
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        raise
