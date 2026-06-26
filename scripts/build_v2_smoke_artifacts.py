#!/usr/bin/env python3
"""Build early v2 smoke artifacts and a lightweight fidelity audit.

This script is intentionally narrow:
- datasets: uniform + heavy_tailed
- artifact label: bfp-auto
- row count: 1,000,000
- output: D0/D1-shaped runtime artifacts plus a smoke-only fidelity audit
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import socket
import subprocess
import sys
import time
from array import array
from pathlib import Path
from typing import Any


DATE_TAG = "20260510"
ARTIFACT_VERSION = "buff_bfp_v2"
ARTIFACT_FAMILY = "buff_bfp"
ARTIFACT_LABEL = "bfp-auto"
FIDELITY_CLASS = "precision_bounded"
PRECISION_POLICY = "auto"
TARGET_SELECTIVITIES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
DATASETS = ("uniform", "heavy_tailed")
CSV_COLUMNS = [
    "dataset",
    "artifact_version",
    "artifact_label",
    "artifact_family",
    "fidelity_class",
    "raw_roundtrip_exact",
    "precision_policy",
    "precision_power",
    "target_selectivity",
    "threshold",
    "raw_sum",
    "encoded_sum",
    "sum_abs_error",
    "sum_rel_error",
    "sum_verdict",
    "raw_count",
    "encoded_count",
    "count_abs_error",
    "count_rel_error",
    "observed_selectivity",
    "selectivity_drift_pp",
    "count_verdict",
    "count_mainline_suitable",
    "sum_mainline_suitable",
    "artifact_role",
    "notes",
]
SEGMENT_META_COLUMNS = {
    "segment_index",
    "row_offset",
    "row_count",
    "active_plane_count",
    "fractional_bits",
    "integer_offset_bits",
    "raw_fractional_bits",
    "precision_cap_bits",
    "effective_fractional_bits",
    "integer_base_hex",
    "segment_base",
}


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
        default=Path(f"results/buff_encoder_v2/raw_smoke_{DATE_TAG}"),
        help="Directory for generated raw smoke datasets.",
    )
    parser.add_argument(
        "--encoded-dir",
        type=Path,
        default=Path(f"results/buff_encoder_v2/containers_smoke_{DATE_TAG}"),
        help="Directory for intermediate .buff64 containers.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(f"datasets/synthetic/dev_buff_v2_smoke_{DATE_TAG}"),
        help="Versioned smoke artifact root.",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("results/buff_encoder_v2/smoke_fidelity_audit.csv"),
        help="Output path for the smoke fidelity audit CSV.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1_000_000,
        help="Rows per smoke dataset.",
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=4096,
        help="Rows per encoded segment.",
    )
    return parser.parse_args()


def run_checked(cmd: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=cwd, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(shlex.quote(part) for part in cmd)}")


def git_commit(cwd: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()


def ensure_new_output_dir(path: Path) -> None:
    if path.exists():
        if path.is_dir() and not any(path.iterdir()):
            return
        raise FileExistsError(f"refusing to reuse non-empty output path: {path}")
    path.mkdir(parents=True, exist_ok=False)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def read_segment_meta(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing segment metadata header: {path}")
        header = set(reader.fieldnames)
        if not SEGMENT_META_COLUMNS.issubset(header):
            missing = sorted(SEGMENT_META_COLUMNS - header)
            raise ValueError(f"segment metadata missing required columns {missing}: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"segment metadata has no rows: {path}")
    return rows


def plane_histogram_string(active_counts: list[int]) -> str:
    histogram: dict[int, int] = {}
    for count in active_counts:
        histogram[count] = histogram.get(count, 0) + 1
    return ",".join(f"{plane_count}:{histogram[plane_count]}" for plane_count in sorted(histogram))


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute quantile of empty data")
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = q * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    frac = index - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


def load_fp64_file(path: Path) -> list[float]:
    payload = path.read_bytes()
    if len(payload) % 8 != 0:
        raise ValueError(f"FP64 payload size is not divisible by 8: {path}")
    values = array("d")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return list(values)


def decode_runtime_values(dataset_dir: Path, segment_rows: list[dict[str, str]], value_count: int, max_plane_count: int) -> list[float]:
    plane_bytes = [(dataset_dir / f"plane_{plane:03d}.bin").read_bytes() for plane in range(max_plane_count)]
    for plane_index, payload in enumerate(plane_bytes):
        if len(payload) != value_count:
            raise ValueError(
                f"plane_{plane_index:03d}.bin length mismatch in {dataset_dir}: "
                f"expected {value_count}, got {len(payload)}"
            )

    decoded = [0.0] * value_count
    for row in segment_rows:
        row_offset = int(row["row_offset"])
        row_count = int(row["row_count"])
        active_plane_count = int(row["active_plane_count"])
        segment_base = float(row["segment_base"])
        plane_basis = [float(row[f"plane_basis_{plane}"]) for plane in range(active_plane_count)]

        for local_index in range(row_count):
            global_index = row_offset + local_index
            value = segment_base
            for plane_index in range(active_plane_count):
                value += plane_bytes[plane_index][global_index] * plane_basis[plane_index]
            decoded[global_index] = value

    return decoded


def select_count_verdict(selectivity_drift_pp: float) -> str:
    drift = abs(selectivity_drift_pp)
    if drift <= 0.1:
        return "acceptable"
    if drift <= 1.0:
        return "caution"
    return "catastrophic"


def select_sum_verdict(sum_abs_error: float, rows: int, quantization_bound: float) -> str:
    if rows <= 0:
        return "not_applicable"
    declared_bound = rows * quantization_bound
    if declared_bound <= 0.0:
        return "not_applicable"
    if sum_abs_error <= declared_bound:
        return "acceptable"
    if sum_abs_error <= declared_bound * 10.0:
        return "caution"
    return "catastrophic"


def compute_artifact_role(sum_verdict: str, count_verdicts: list[str]) -> tuple[bool, bool, str]:
    sum_mainline = sum_verdict == "acceptable"
    count_mainline = all(verdict == "acceptable" for verdict in count_verdicts)
    if sum_mainline and count_mainline:
        return count_mainline, sum_mainline, "mainline_candidate"
    if any(verdict == "catastrophic" for verdict in count_verdicts) or sum_verdict == "catastrophic":
        return count_mainline, sum_mainline, "reject"
    return count_mainline, sum_mainline, "side_study_only"


def rewrite_artifact_metadata(dataset_dir: Path, *, artifact_label: str, precision_policy: str) -> dict[str, Any]:
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
    max_plane_count = int(manifest["max_plane_count"])

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
            "precision_policy": precision_policy,
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
            "precision_policy": precision_policy,
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
            "max_plane_count": max_plane_count,
            "mean_plane_count": sum(active_counts) / len(active_counts),
        }
    )

    write_json(manifest_path, manifest)
    write_json(summary_path, summary)
    return {"manifest": manifest, "summary": summary, "segment_rows": segment_rows}


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
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_audit_rows(dataset_dir: Path, raw_path: Path, manifest: dict[str, Any], summary: dict[str, Any], segment_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    raw_values = load_fp64_file(raw_path)
    rows = int(manifest["value_count"])
    if rows != len(raw_values):
        raise ValueError(f"raw row count mismatch for {dataset_dir}: manifest={rows}, raw={len(raw_values)}")
    raw_sorted = sorted(raw_values)
    encoded_values = decode_runtime_values(
        dataset_dir,
        segment_rows,
        value_count=rows,
        max_plane_count=int(manifest["max_plane_count"]),
    )

    raw_sum = math.fsum(raw_values)
    encoded_sum = math.fsum(encoded_values)
    sum_abs_error = abs(encoded_sum - raw_sum)
    sum_rel_error = 0.0 if raw_sum == 0.0 else sum_abs_error / abs(raw_sum)
    sum_verdict = select_sum_verdict(sum_abs_error, rows, float(summary["quantization_bound"]))

    thresholds = {
        target: quantile(raw_sorted, 1.0 - (target / 100.0))
        for target in TARGET_SELECTIVITIES
    }

    count_verdicts: list[str] = []
    row_payloads: list[dict[str, Any]] = []
    for target in TARGET_SELECTIVITIES:
        threshold = thresholds[target]
        raw_count = sum(1 for value in raw_values if value > threshold)
        encoded_count = sum(1 for value in encoded_values if value > threshold)
        count_abs_error = abs(encoded_count - raw_count)
        count_rel_error = 0.0 if raw_count == 0 else count_abs_error / raw_count
        observed_selectivity = (encoded_count / rows) * 100.0
        selectivity_drift_pp = observed_selectivity - float(target)
        count_verdict = select_count_verdict(selectivity_drift_pp)
        count_verdicts.append(count_verdict)

        row_payloads.append(
            {
                "dataset": manifest["dataset"],
                "artifact_version": manifest["artifact_version"],
                "artifact_label": manifest["artifact_label"],
                "artifact_family": manifest["artifact_family"],
                "fidelity_class": manifest["fidelity_class"],
                "raw_roundtrip_exact": manifest["raw_roundtrip_exact"],
                "precision_policy": manifest["precision_policy"],
                "precision_power": summary["precision_power"],
                "target_selectivity": target,
                "threshold": threshold,
                "raw_sum": raw_sum,
                "encoded_sum": encoded_sum,
                "sum_abs_error": sum_abs_error,
                "sum_rel_error": sum_rel_error,
                "sum_verdict": sum_verdict,
                "raw_count": raw_count,
                "encoded_count": encoded_count,
                "count_abs_error": count_abs_error,
                "count_rel_error": count_rel_error,
                "observed_selectivity": observed_selectivity,
                "selectivity_drift_pp": selectivity_drift_pp,
                "count_verdict": count_verdict,
            }
        )

    count_mainline_suitable, sum_mainline_suitable, artifact_role = compute_artifact_role(sum_verdict, count_verdicts)
    for payload in row_payloads:
        payload["count_mainline_suitable"] = count_mainline_suitable
        payload["sum_mainline_suitable"] = sum_mainline_suitable
        payload["artifact_role"] = artifact_role
        payload["notes"] = "smoke-only compatibility artifact; not paper evidence"

    return row_payloads


def write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


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
    audit_csv = args.audit_csv.resolve()

    ensure_new_output_dir(raw_dir)
    ensure_new_output_dir(encoded_dir)
    ensure_new_output_dir(artifact_root)

    artifact_entries: list[dict[str, str]] = []
    audit_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        run_checked(
            [
                str(generator),
                "--dataset",
                dataset,
                "--count",
                str(args.count),
                "--out-dir",
                str(raw_dir),
            ],
            cwd=workspace,
        )

        raw_path = raw_dir / f"{dataset}.f64le.bin"
        encoded_path = encoded_dir / f"{dataset}_{ARTIFACT_LABEL}.buff64"
        artifact_dir = artifact_root / dataset / ARTIFACT_LABEL
        artifact_dir.mkdir(parents=True, exist_ok=False)

        run_checked(
            [
                str(buff_tool),
                "encode",
                "--input",
                str(raw_path),
                "--output",
                str(encoded_path),
                "--segment-size",
                str(args.segment_size),
            ],
            cwd=workspace,
        )
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

        rewritten = rewrite_artifact_metadata(artifact_dir, artifact_label=ARTIFACT_LABEL, precision_policy=PRECISION_POLICY)
        manifest = rewritten["manifest"]
        summary = rewritten["summary"]
        segment_rows = rewritten["segment_rows"]

        artifact_entries.append(
            {
                "dataset": dataset,
                "artifact_label": ARTIFACT_LABEL,
                "relative_path": str(Path(dataset) / ARTIFACT_LABEL),
            }
        )
        audit_rows.extend(build_audit_rows(artifact_dir, raw_path, manifest, summary, segment_rows))

    audit_rows.sort(key=lambda row: (row["dataset"], int(row["target_selectivity"])))
    write_root_index(artifact_root, artifact_entries)
    write_audit_csv(audit_csv, audit_rows)

    write_provenance(
        artifact_root / "provenance.json",
        argv=sys.argv,
        extra={
            "artifact_root": str(artifact_root),
            "raw_dir": str(raw_dir),
            "encoded_dir": str(encoded_dir),
            "datasets": list(DATASETS),
            "artifact_label": ARTIFACT_LABEL,
            "row_count_per_dataset": args.count,
            "segment_size": args.segment_size,
        },
    )
    write_provenance(
        audit_csv.with_suffix(audit_csv.suffix + ".provenance.json"),
        argv=sys.argv,
        extra={
            "artifact_root": str(artifact_root),
            "audit_csv": str(audit_csv),
            "datasets": list(DATASETS),
            "artifact_label": ARTIFACT_LABEL,
            "row_count": len(audit_rows),
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
