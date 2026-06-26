#!/usr/bin/env python3
"""Export D1 v2 runtime artifacts for the four synthetic datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DATE_TAG = "20260510"
ARTIFACT_VERSION = "buff_bfp_v2"
ARTIFACT_FAMILY = "buff_bfp"
FIDELITY_CLASS = "precision_bounded"
DATASETS = ("uniform", "heavy_tailed", "sensor", "zipfian")
DATASET_POWERS = {
    "uniform": 4,
    "heavy_tailed": 4,
    "sensor": 4,
    "zipfian": 4,
}
SEGMENT_SIZE = 4096
VALUE_COUNT = 100_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator",
        type=Path,
        default=Path("bin/synth_datasets"),
        help="Path to the synthetic dataset generator binary.",
    )
    parser.add_argument(
        "--buff-tool",
        type=Path,
        default=Path("bin/buff_tool"),
        help="Path to the buff_tool binary.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(f"results/buff_encoder_v2/raw_dev_{DATE_TAG}"),
        help="Directory for raw dev synthetic datasets.",
    )
    parser.add_argument(
        "--encoded-dir",
        type=Path,
        default=Path(f"results/buff_encoder_v2/containers_dev_{DATE_TAG}"),
        help="Directory for intermediate .buff64 containers.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(f"datasets/synthetic/dev_buff_v2_{DATE_TAG}"),
        help="Versioned runtime artifact root.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(f"research/{DATE_TAG[:4]}-{DATE_TAG[4:6]}-{DATE_TAG[6:]}_New_Encoder_Artifact_Export_Report.md"),
        help="Markdown export report path.",
    )
    return parser.parse_args()


def git_commit(cwd: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()


def run_checked(cmd: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=cwd, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            + " ".join(shlex.quote(part) for part in cmd)
        )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def read_segment_meta(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing segment metadata header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"segment metadata has no rows: {path}")
    return rows


def plane_histogram_string(active_counts: list[int]) -> str:
    histogram: dict[int, int] = {}
    for count in active_counts:
        histogram[count] = histogram.get(count, 0) + 1
    return ",".join(f"{plane_count}:{histogram[plane_count]}" for plane_count in sorted(histogram))


def rewrite_artifact_metadata(dataset_dir: Path, *, artifact_label: str) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    summary_path = dataset_dir / "summary.json"
    segment_meta_path = dataset_dir / "segment_meta.csv"

    manifest = read_json(manifest_path)
    summary = read_json(summary_path)
    segment_rows = read_segment_meta(segment_meta_path)

    plane_files = sorted(dataset_dir.glob("plane_*.bin"))
    main_bytes_total = sum(path.stat().st_size for path in plane_files)
    side_table_bytes_total = 0
    active_counts = [int(row["active_plane_count"]) for row in segment_rows]
    raw_fractional_bits = [int(row["raw_fractional_bits"]) for row in segment_rows]
    precision_cap_bits = [int(row["precision_cap_bits"]) for row in segment_rows]
    effective_fractional_bits = [int(row["effective_fractional_bits"]) for row in segment_rows]

    value_count = int(manifest["value_count"])
    precision_power = int(summary["precision_power"])
    decimal_bits = int(summary["decimal_bits"])
    quantization_bound = float(summary["quantization_bound"])

    manifest.update(
        {
            "manifest_scope": "artifact_instance",
            "artifact_version": ARTIFACT_VERSION,
            "artifact_family": ARTIFACT_FAMILY,
            "artifact_label": artifact_label,
            "fidelity_class": FIDELITY_CLASS,
            "raw_roundtrip_exact": False,
            "precision_policy": "fixed",
            "precision_power": precision_power,
            "decimal_bits": decimal_bits,
            "quantization_bound": quantization_bound,
            "side_table_mode": "none",
            "side_table_bytes_total": side_table_bytes_total,
        }
    )

    summary.update(
        {
            "manifest_scope": "artifact_summary",
            "artifact_version": ARTIFACT_VERSION,
            "artifact_family": ARTIFACT_FAMILY,
            "artifact_label": artifact_label,
            "fidelity_class": FIDELITY_CLASS,
            "raw_roundtrip_exact": False,
            "precision_policy": "fixed",
            "raw_fractional_bits_min": min(raw_fractional_bits),
            "raw_fractional_bits_max": max(raw_fractional_bits),
            "precision_cap_bits_min": min(precision_cap_bits),
            "precision_cap_bits_max": max(precision_cap_bits),
            "effective_fractional_bits_min": min(effective_fractional_bits),
            "effective_fractional_bits_max": max(effective_fractional_bits),
            "plane_count_histogram": plane_histogram_string(active_counts),
            "main_bytes_total": main_bytes_total,
            "side_table_bytes_total": side_table_bytes_total,
            "total_bytes_per_row": (main_bytes_total + side_table_bytes_total) / value_count,
            "segment_plane_count_min": min(active_counts),
            "segment_plane_count_max": max(active_counts),
            "mean_plane_count": sum(active_counts) / len(active_counts),
        }
    )

    write_json(manifest_path, manifest)
    write_json(summary_path, summary)
    return {"manifest": manifest, "summary": summary}


def write_root_index(artifact_root: Path, entries: list[dict[str, str]]) -> None:
    payload = {
        "manifest_scope": "root_index",
        "artifact_version": ARTIFACT_VERSION,
        "datasets": sorted({entry["dataset"] for entry in entries}),
        "artifact_labels": sorted({entry["artifact_label"] for entry in entries}),
        "entries": entries,
    }
    write_json(artifact_root / "manifest.json", payload)


def write_provenance(path: Path, *, argv: list[str], extra: dict[str, Any]) -> None:
    payload = {
        "command": " ".join(shlex.quote(part) for part in argv),
        "git_commit": git_commit(Path.cwd()),
        "hostname": socket.gethostname(),
        "gpu_name": None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def artifact_complete(dataset_dir: Path) -> bool:
    if not (dataset_dir / "manifest.json").is_file():
        return False
    if not (dataset_dir / "summary.json").is_file():
        return False
    if not (dataset_dir / "segment_meta.csv").is_file():
        return False
    return any(dataset_dir.glob("plane_*.bin"))


def compatibility_status(manifest: dict[str, Any]) -> str:
    if manifest.get("format") == "exp3_encoded_dev_v1" and manifest.get("encoded_layout") == "plane_major_zero_padded":
        return "readable by existing GPU loader unchanged"
    return "migration note required"


def build_report(
    report_path: Path,
    *,
    artifact_root: Path,
    rows: list[dict[str, Any]],
    local_path_note: bool,
) -> None:
    lines = [
        "# New Encoder Artifact Export Report",
        "",
        "Date: 2026-05-10",
        "",
        "Scope: D1 artifact export only.",
        "",
        "## 1. Summary",
        "",
        f"- artifact root: `{artifact_root}`",
        f"- artifact version: `{ARTIFACT_VERSION}`",
        f"- artifact family: `{ARTIFACT_FAMILY}`",
        "",
        "Target artifact matrix exported in this run:",
        "",
        "| Dataset | Artifact label | Artifact path | Rows | Segment rows | active_plane_count_min | active_plane_count_max | active_plane_count_mean | Main bytes/row | Side bytes/row | Total bytes/row | Compatibility status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| "
            + f"{row['dataset']} | {row['artifact_label']} | `{row['artifact_path']}` | "
            + f"{row['rows']} | {row['segment_rows']} | {row['active_plane_count_min']} | "
            + f"{row['active_plane_count_max']} | {row['active_plane_count_mean']:.6f} | "
            + f"{row['main_bytes_per_row']:.6f} | {row['side_bytes_per_row']:.6f} | "
            + f"{row['total_bytes_per_row']:.6f} | {row['compatibility_status']} |"
        )

    lines.extend(
        [
            "",
            "## 2. Validation",
            "",
            "- Every exported `(dataset, artifact_label)` directory contains `manifest.json`, `summary.json`, `segment_meta.csv`, and plane files.",
            "- All exported artifacts keep `format = exp3_encoded_dev_v1` and `encoded_layout = plane_major_zero_padded` for receiver compatibility.",
            "- No side table is emitted in the current branch, so `side bytes/row = 0` for every exported artifact.",
        ]
    )

    if local_path_note:
        lines.extend(
            [
                "",
                "## 3. Environment Note",
                "",
                "- The issue spec uses `/work/<user>/datasets/...` as the example target root.",
                "- This machine does not provide a writable `/work/...` root, so the versioned artifact root was created under the repo-local path shown above.",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    workspace = Path.cwd()

    generator = args.generator.resolve()
    buff_tool = args.buff_tool.resolve()
    if not generator.is_file():
        raise FileNotFoundError(f"generator binary not found: {generator}")
    if not buff_tool.is_file():
        raise FileNotFoundError(f"buff_tool binary not found: {buff_tool}")

    raw_dir = args.raw_dir.resolve()
    encoded_dir = args.encoded_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    report_path = args.report_path.resolve()

    ensure_dir(raw_dir)
    ensure_dir(encoded_dir)
    ensure_dir(artifact_root)

    artifact_entries: list[dict[str, str]] = []
    report_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        precision_power = DATASET_POWERS[dataset]
        artifact_label = f"bfp-dec{precision_power}"
        raw_path = raw_dir / f"{dataset}.f64le.bin"
        encoded_path = encoded_dir / f"{dataset}_{artifact_label}.buff64"
        artifact_dir = artifact_root / dataset / artifact_label

        if not raw_path.is_file():
            run_checked(
                [
                    str(generator),
                    "--dataset",
                    dataset,
                    "--profile",
                    "dev",
                    "--out-dir",
                    str(raw_dir),
                ],
                cwd=workspace,
            )

        if not encoded_path.is_file():
            run_checked(
                [
                    str(buff_tool),
                    "encode",
                    "--input",
                    str(raw_path),
                    "--output",
                    str(encoded_path),
                    "--segment-size",
                    str(SEGMENT_SIZE),
                    "--precision-power",
                    str(precision_power),
                ],
                cwd=workspace,
            )

        if not artifact_complete(artifact_dir):
            ensure_dir(artifact_dir)
            run_checked(
                [
                    str(buff_tool),
                    "export-runtime",
                    "--input",
                    str(encoded_path),
                    "--raw-input",
                    str(raw_path),
                    "--out-dir",
                    str(artifact_dir),
                    "--dataset",
                    dataset,
                    "--source-path",
                    str(raw_path),
                ],
                cwd=workspace,
            )

        rewritten = rewrite_artifact_metadata(artifact_dir, artifact_label=artifact_label)
        manifest = rewritten["manifest"]
        summary = rewritten["summary"]

        rows = int(manifest["value_count"])
        main_bytes_per_row = float(summary["main_bytes_total"]) / rows
        side_bytes_per_row = float(summary["side_table_bytes_total"]) / rows
        total_bytes_per_row = float(summary["total_bytes_per_row"])
        active_plane_count_max = int(summary["segment_plane_count_max"])

        if active_plane_count_max >= 8:
            raise RuntimeError(
                f"{dataset}/{artifact_label} violated the compactness gate: "
                f"active_plane_count_max={active_plane_count_max}"
            )
        if total_bytes_per_row >= 8.0:
            raise RuntimeError(
                f"{dataset}/{artifact_label} violated the bytes/row gate: "
                f"total_bytes_per_row={total_bytes_per_row}"
            )

        artifact_entries.append(
            {
                "dataset": dataset,
                "artifact_label": artifact_label,
                "relative_path": str(Path(dataset) / artifact_label),
            }
        )
        report_rows.append(
            {
                "dataset": dataset,
                "artifact_label": artifact_label,
                "artifact_path": str(artifact_dir),
                "rows": rows,
                "segment_rows": int(manifest["segment_size"]),
                "active_plane_count_min": int(summary["segment_plane_count_min"]),
                "active_plane_count_max": active_plane_count_max,
                "active_plane_count_mean": float(summary["mean_plane_count"]),
                "main_bytes_per_row": main_bytes_per_row,
                "side_bytes_per_row": side_bytes_per_row,
                "total_bytes_per_row": total_bytes_per_row,
                "compatibility_status": compatibility_status(manifest),
            }
        )

    report_rows.sort(key=lambda row: row["dataset"])
    write_root_index(artifact_root, artifact_entries)
    write_provenance(
        artifact_root / "provenance.json",
        argv=sys.argv,
        extra={
            "artifact_root": str(artifact_root),
            "raw_dir": str(raw_dir),
            "encoded_dir": str(encoded_dir),
            "datasets": list(DATASETS),
            "dataset_powers": DATASET_POWERS,
            "segment_size": SEGMENT_SIZE,
            "value_count_per_dataset": VALUE_COUNT,
        },
    )
    build_report(
        report_path,
        artifact_root=artifact_root,
        rows=report_rows,
        local_path_note=not str(artifact_root).startswith("/work/"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
