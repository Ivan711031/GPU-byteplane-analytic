#!/usr/bin/env python3
"""Build and characterize the approved #14 real-data v2 artifact."""

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
import zipfile
from array import array
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DATE_TAG = "20260510"
DATASET_KEY = "uci_household_global_active_power"
DATASET_DISPLAY = "UCI Household Electric Power Consumption"
COLUMN_NAME = "Global_active_power"
COLUMN_UNIT = "kilowatt"
COLUMN_SEMANTICS = "global minute-averaged household active power"
ARTIFACT_VERSION = "buff_bfp_v2"
ARTIFACT_FAMILY = "buff_bfp"
FIDELITY_CLASS = "precision_bounded"
PRECISION_POLICY = "fixed"
SEGMENT_SIZE = 4096
THRESHOLD_GRID = [1, 5, 10, 25, 50, 75, 90, 95, 99]
REAL_DATA_SOURCE_FILENAME = "real_data_source.json"
SELECTION_DECISION_REF = "research/2026-05-01_PV1-9A_Real_Dataset_Audit_and_Selection_Report.md"
TARGET_TEXT_NAME = "household_power_consumption.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path(
            f"results/real_dataset_source/uci_household_{DATE_TAG}/"
            "individual_household_electric_power_consumption.zip"
        ),
        help="Path to the downloaded UCI dataset zip.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path(f"results/real_dataset_source/uci_household_{DATE_TAG}/extracted"),
        help="Directory where the raw text dataset will be extracted.",
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
        default=Path(f"results/buff_encoder_v2/raw_real_{DATE_TAG}"),
        help="Directory for the cleaned FP64 binary slice.",
    )
    parser.add_argument(
        "--encoded-dir",
        type=Path,
        default=Path(f"results/buff_encoder_v2/containers_real_{DATE_TAG}"),
        help="Directory for intermediate .buff64 containers.",
    )
    parser.add_argument(
        "--decoded-dir",
        type=Path,
        default=Path(f"results/buff_encoder_v2/decoded_real_{DATE_TAG}"),
        help="Directory for full-depth decoded .buff64 output.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(f"datasets/real/dev_buff_v2_real_{DATE_TAG}"),
        help="Versioned real-data runtime artifact root.",
    )
    parser.add_argument(
        "--plane-stats-csv",
        type=Path,
        default=Path(f"results/buff_encoder_v2/{DATASET_KEY}_plane_stats.csv"),
        help="Per-plane stats CSV path.",
    )
    parser.add_argument(
        "--byte-hist-csv",
        type=Path,
        default=Path(f"results/buff_encoder_v2/{DATASET_KEY}_byte_histogram.csv"),
        help="Per-plane byte histogram CSV path.",
    )
    parser.add_argument(
        "--threshold-csv",
        type=Path,
        default=Path(f"results/buff_encoder_v2/{DATASET_KEY}_threshold_behavior.csv"),
        help="Threshold/selectivity behavior CSV path.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(f"research/{DATE_TAG[:4]}-{DATE_TAG[4:6]}-{DATE_TAG[6:]}_New_Encoder_Real_Artifact_Characterization_Report.md"),
        help="Markdown characterization report path.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def git_commit(cwd: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_segment_meta(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing segment metadata header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"segment metadata has no rows: {path}")
    return rows


def run_checked(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
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
    return completed


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
    write_json(path, payload)


def extract_text_file(zip_path: Path, extract_dir: Path) -> Path:
    if not zip_path.is_file():
        raise FileNotFoundError(f"dataset zip not found: {zip_path}")
    ensure_dir(extract_dir)
    target = extract_dir / TARGET_TEXT_NAME
    if target.is_file():
        return target
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.endswith(TARGET_TEXT_NAME)]
        if not members:
            raise FileNotFoundError(f"{TARGET_TEXT_NAME} not found inside {zip_path}")
        member = members[0]
        archive.extract(member, path=extract_dir)
        extracted = extract_dir / member
        if extracted != target:
            target.write_bytes(extracted.read_bytes())
    return target


def materialize_clean_slice(text_path: Path, raw_path: Path) -> dict[str, Any]:
    ensure_dir(raw_path.parent)

    row_count_before = 0
    row_count_after = 0
    finite_count = 0
    missing_count = 0
    nan_count = 0
    infinite_count = 0
    negative_count = 0
    missing_marker_question_count = 0
    missing_marker_empty_count = 0
    raw_min = math.inf
    raw_max = -math.inf
    exact_sum = 0.0
    chunk = array("d")
    chunk_limit = 131072

    with text_path.open("r", encoding="latin-1", newline="") as handle, raw_path.open("wb") as out:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader)
        try:
            column_index = header.index(COLUMN_NAME)
        except ValueError as exc:
            raise ValueError(f"{COLUMN_NAME} not found in {text_path}") from exc

        for row in reader:
            if not row:
                continue
            row_count_before += 1
            field = row[column_index].strip()
            if field == "?":
                missing_count += 1
                missing_marker_question_count += 1
                continue
            if field == "":
                missing_count += 1
                missing_marker_empty_count += 1
                continue
            value = float(field)
            if math.isnan(value):
                nan_count += 1
                continue
            if math.isinf(value):
                infinite_count += 1
                continue
            finite_count += 1
            if value < 0.0:
                negative_count += 1
                continue
            row_count_after += 1
            raw_min = min(raw_min, value)
            raw_max = max(raw_max, value)
            exact_sum += value
            chunk.append(value)
            if len(chunk) >= chunk_limit:
                chunk.tofile(out)
                chunk = array("d")

        if chunk:
            chunk.tofile(out)

    if row_count_after == 0:
        raise RuntimeError("cleaned real-data slice is empty")
    if negative_count != 0:
        raise RuntimeError(f"approved non-negative path violated: negative_count={negative_count}")
    if nan_count != 0 or infinite_count != 0:
        raise RuntimeError(
            "approved non-negative path violated: "
            f"nan_count={nan_count} infinite_count={infinite_count}"
        )

    return {
        "row_count_before_transform": row_count_before,
        "row_count_after_transform": row_count_after,
        "finite_count": finite_count,
        "missing_count": missing_count,
        "nan_count": nan_count,
        "infinite_count": infinite_count,
        "negative_count": negative_count,
        "missing_marker_question_count": missing_marker_question_count,
        "missing_marker_empty_count": missing_marker_empty_count,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "transformed_min": raw_min,
        "transformed_max": raw_max,
        "exact_sum": exact_sum,
        "raw_path": str(raw_path),
    }


def probe_precision_powers(
    buff_tool: Path,
    raw_path: Path,
    encoded_dir: Path,
    *,
    cwd: Path,
) -> tuple[int, str, Path, list[dict[str, Any]]]:
    ensure_dir(encoded_dir)
    attempts: list[dict[str, Any]] = []
    for power in range(12, 0, -1):
        artifact_label = f"bfp-dec{power}"
        encoded_path = encoded_dir / f"{DATASET_KEY}_{artifact_label}.buff64"
        cmd = [
            str(buff_tool),
            "encode",
            "--input",
            str(raw_path),
            "--output",
            str(encoded_path),
            "--segment-size",
            str(SEGMENT_SIZE),
            "--precision-power",
            str(power),
        ]
        completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
        attempts.append(
            {
                "precision_power": power,
                "artifact_label": artifact_label,
                "encoded_path": str(encoded_path),
                "exit_code": completed.returncode,
                "stderr": completed.stderr.strip(),
                "stdout": completed.stdout.strip(),
            }
        )
        if completed.returncode == 0:
            return power, artifact_label, encoded_path, attempts
    raise RuntimeError("no precision tier in bfp-dec1..bfp-dec12 produced a valid <=8-plane artifact")


def plane_histogram_string(active_counts: list[int]) -> str:
    histogram: dict[int, int] = {}
    for count in active_counts:
        histogram[count] = histogram.get(count, 0) + 1
    return ",".join(f"{plane_count}:{histogram[plane_count]}" for plane_count in sorted(histogram))


def rewrite_artifact_metadata(
    artifact_dir: Path,
    *,
    artifact_label: str,
    precision_power: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    manifest_path = artifact_dir / "manifest.json"
    summary_path = artifact_dir / "summary.json"
    segment_meta_path = artifact_dir / "segment_meta.csv"

    manifest = read_json(manifest_path)
    summary = read_json(summary_path)
    segment_rows = read_segment_meta(segment_meta_path)

    plane_files = sorted(artifact_dir.glob("plane_*.bin"))
    main_bytes_total = sum(path.stat().st_size for path in plane_files)
    side_table_bytes_total = 0

    active_counts = [int(row["active_plane_count"]) for row in segment_rows]
    integer_offset_bits = [int(row["integer_offset_bits"]) for row in segment_rows]
    raw_fractional_bits = [int(row["raw_fractional_bits"]) for row in segment_rows]
    precision_cap_bits = [int(row["precision_cap_bits"]) for row in segment_rows]
    effective_fractional_bits = [int(row["effective_fractional_bits"]) for row in segment_rows]
    value_count = int(manifest["value_count"])
    quantization_bound = float(summary["quantization_bound"])

    manifest.update(
        {
            "manifest_scope": "artifact_instance",
            "artifact_version": ARTIFACT_VERSION,
            "artifact_family": ARTIFACT_FAMILY,
            "artifact_label": artifact_label,
            "fidelity_class": FIDELITY_CLASS,
            "raw_roundtrip_exact": False,
            "precision_policy": PRECISION_POLICY,
            "precision_power": precision_power,
            "decimal_bits": int(summary["decimal_bits"]),
            "quantization_bound": quantization_bound,
            "side_table_mode": "none",
            "side_table_bytes_total": side_table_bytes_total,
            "segment_plane_count_min": min(active_counts),
            "segment_plane_count_max": max(active_counts),
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
            "precision_policy": PRECISION_POLICY,
            "precision_power": precision_power,
            "raw_fractional_bits_min": min(raw_fractional_bits),
            "raw_fractional_bits_max": max(raw_fractional_bits),
            "integer_offset_bits_min": min(integer_offset_bits),
            "integer_offset_bits_max": max(integer_offset_bits),
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
    return manifest, summary, segment_rows


def write_root_index(artifact_root: Path, *, artifact_label: str) -> None:
    payload = {
        "manifest_scope": "root_index",
        "artifact_version": ARTIFACT_VERSION,
        "datasets": [DATASET_KEY],
        "artifact_labels": [artifact_label],
        "entries": [
            {
                "dataset": DATASET_KEY,
                "artifact_label": artifact_label,
                "relative_path": str(Path(DATASET_KEY) / artifact_label),
            }
        ],
    }
    write_json(artifact_root / "manifest.json", payload)


def write_real_data_source(
    path: Path,
    *,
    text_path: Path,
    cleaned_stats: dict[str, Any],
    transform_command: str,
) -> None:
    payload = {
        "column_name": COLUMN_NAME,
        "column_semantics": COLUMN_SEMANTICS,
        "column_unit": COLUMN_UNIT,
        "dataset_name": DATASET_DISPLAY,
        "dataset_version_or_snapshot": "UCI dataset id 235, downloaded 2026-05-10",
        "finite_count": cleaned_stats["finite_count"],
        "infinite_count": cleaned_stats["infinite_count"],
        "missing_count": cleaned_stats["missing_count"],
        "nan_count": cleaned_stats["nan_count"],
        "negative_count": cleaned_stats["negative_count"],
        "offset_constant": 0.0,
        "raw_max": cleaned_stats["raw_max"],
        "raw_min": cleaned_stats["raw_min"],
        "raw_threshold_unit": COLUMN_UNIT,
        "row_count_after_transform": cleaned_stats["row_count_after_transform"],
        "row_count_before_transform": cleaned_stats["row_count_before_transform"],
        "selection_decision_ref": SELECTION_DECISION_REF,
        "source_table_or_file": str(text_path),
        "threshold_policy": "identity",
        "transform_command_or_script": transform_command,
        "transform_policy": "non_negative_as_is",
        "transformed_max": cleaned_stats["transformed_max"],
        "transformed_min": cleaned_stats["transformed_min"],
    }
    write_json(path, payload)


def decode_full_depth(
    buff_tool: Path,
    encoded_path: Path,
    decoded_path: Path,
    *,
    cwd: Path,
) -> None:
    if decoded_path.is_file():
        return
    ensure_dir(decoded_path.parent)
    run_checked(
        [
            str(buff_tool),
            "decode",
            "--input",
            str(encoded_path),
            "--output",
            str(decoded_path),
        ],
        cwd=cwd,
    )


def load_fp64_memmap(path: Path) -> np.memmap:
    return np.memmap(path, dtype="<f8", mode="r")


def load_plane_arrays(artifact_dir: Path, max_plane_count: int, value_count: int) -> list[np.ndarray]:
    planes = []
    for plane in range(max_plane_count):
        payload = np.fromfile(artifact_dir / f"plane_{plane:03d}.bin", dtype=np.uint8)
        if payload.size != value_count:
            raise ValueError(
                f"plane_{plane:03d}.bin length mismatch: expected {value_count}, got {payload.size}"
            )
        planes.append(payload)
    return planes


def reconstruct_runtime_values(
    segment_rows: list[dict[str, str]],
    planes: list[np.ndarray],
    value_count: int,
) -> np.ndarray:
    decoded = np.zeros(value_count, dtype=np.float64)
    for row in segment_rows:
        row_offset = int(row["row_offset"])
        row_count = int(row["row_count"])
        active_plane_count = int(row["active_plane_count"])
        segment_base = float(row["segment_base"])
        local = np.full(row_count, segment_base, dtype=np.float64)
        for plane in range(active_plane_count):
            basis = float(row[f"plane_basis_{plane}"])
            local += planes[plane][row_offset : row_offset + row_count].astype(np.float64) * basis
        decoded[row_offset : row_offset + row_count] = local
    return decoded


def select_count_verdict(selectivity_drift_pp: float) -> str:
    drift = abs(selectivity_drift_pp)
    if drift <= 0.1:
        return "acceptable"
    if drift <= 1.0:
        return "caution"
    return "catastrophic"


def entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    probs = counts[counts > 0].astype(np.float64) / float(total)
    return float(-(probs * np.log2(probs)).sum())


def write_plane_stats_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_byte_hist_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_threshold_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def characterize_artifact(
    artifact_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    segment_rows: list[dict[str, str]],
    raw_path: Path,
    decoded_path: Path,
) -> dict[str, Any]:
    value_count = int(manifest["value_count"])
    max_plane_count = int(manifest["max_plane_count"])
    raw_values = load_fp64_memmap(raw_path)
    decoded_values = load_fp64_memmap(decoded_path)
    planes = load_plane_arrays(artifact_dir, max_plane_count, value_count)
    runtime_values = reconstruct_runtime_values(segment_rows, planes, value_count)

    if raw_values.shape[0] != value_count:
        raise ValueError(f"raw row count mismatch: expected {value_count}, got {raw_values.shape[0]}")
    if decoded_values.shape[0] != value_count:
        raise ValueError(
            f"decoded row count mismatch: expected {value_count}, got {decoded_values.shape[0]}"
        )

    active_counts = [int(row["active_plane_count"]) for row in segment_rows]
    observed_hist = dict(sorted(Counter(active_counts).items()))
    summary_hist_text = str(summary["plane_count_histogram"])
    expected_hist = {}
    for item in summary_hist_text.split(","):
        key, value = item.split(":")
        expected_hist[int(key)] = int(value)

    zero_padding_ok = True
    for row in segment_rows:
        row_offset = int(row["row_offset"])
        row_count = int(row["row_count"])
        active_plane_count = int(row["active_plane_count"])
        for plane in range(active_plane_count, max_plane_count):
            if np.any(planes[plane][row_offset : row_offset + row_count] != 0):
                zero_padding_ok = False
                break
        if not zero_padding_ok:
            break

    basis_descending_ok = True
    for row in segment_rows:
        active_plane_count = int(row["active_plane_count"])
        bases = [float(row[f"plane_basis_{plane}"]) for plane in range(active_plane_count)]
        for index in range(len(bases) - 1):
            if not (bases[index] > bases[index + 1] > 0.0):
                basis_descending_ok = False
                break
        if not basis_descending_ok:
            break

    reconstruction_abs_diff_max = float(np.max(np.abs(runtime_values - decoded_values)))
    reconstruction_match = reconstruction_abs_diff_max <= 1e-12

    plane_stats_rows: list[dict[str, Any]] = []
    byte_hist_rows: list[dict[str, Any]] = []
    for plane_index, plane_bytes in enumerate(planes):
        counts = np.bincount(plane_bytes, minlength=256)
        plane_stats_rows.append(
            {
                "dataset": DATASET_KEY,
                "artifact_label": manifest["artifact_label"],
                "plane_index": plane_index,
                "entropy_bits": entropy_from_counts(counts),
                "zero_fraction": float(counts[0]) / float(value_count),
                "nonzero_fraction": float(value_count - counts[0]) / float(value_count),
                "unique_values": int(np.count_nonzero(counts)),
                "byte_min": int(np.min(plane_bytes)),
                "byte_max": int(np.max(plane_bytes)),
            }
        )
        for byte_value, count in enumerate(counts):
            if count == 0:
                continue
            byte_hist_rows.append(
                {
                    "dataset": DATASET_KEY,
                    "artifact_label": manifest["artifact_label"],
                    "plane_index": plane_index,
                    "byte_value": byte_value,
                    "count": int(count),
                }
            )

    raw_sum = float(np.sum(raw_values, dtype=np.float64))
    encoded_sum = float(np.sum(decoded_values, dtype=np.float64))
    sum_abs_error = abs(encoded_sum - raw_sum)
    integer_offset_hist = dict(sorted(Counter(int(row["integer_offset_bits"]) for row in segment_rows).items()))
    effective_fractional_hist = dict(
        sorted(Counter(int(row["effective_fractional_bits"]) for row in segment_rows).items())
    )

    threshold_rows: list[dict[str, Any]] = []
    catastrophic_count_rows = 0
    for target in THRESHOLD_GRID:
        threshold = float(np.quantile(raw_values, 1.0 - (target / 100.0), method="linear"))
        raw_count = int(np.count_nonzero(raw_values > threshold))
        decoded_count = int(np.count_nonzero(decoded_values > threshold))
        runtime_count = int(np.count_nonzero(runtime_values > threshold))
        observed_selectivity = (decoded_count / value_count) * 100.0
        selectivity_drift_pp = observed_selectivity - float(target)
        verdict = select_count_verdict(selectivity_drift_pp)
        if verdict == "catastrophic":
            catastrophic_count_rows += 1
        threshold_rows.append(
            {
                "dataset": DATASET_KEY,
                "artifact_label": manifest["artifact_label"],
                "target_selectivity": target,
                "threshold": threshold,
                "raw_count": raw_count,
                "decoded_count": decoded_count,
                "runtime_count": runtime_count,
                "observed_selectivity": observed_selectivity,
                "selectivity_drift_pp": selectivity_drift_pp,
                "count_verdict": verdict,
                "runtime_decoded_match": decoded_count == runtime_count,
            }
        )

    contract_checks = {
        "format_ok": manifest.get("format") == "exp3_encoded_dev_v1",
        "encoded_layout_ok": manifest.get("encoded_layout") == "plane_major_zero_padded",
        "plane_file_count_ok": len(planes) == max_plane_count,
        "segment_plane_min_ok": int(manifest["segment_plane_count_min"]) == min(active_counts),
        "segment_plane_max_ok": int(manifest["segment_plane_count_max"]) == max(active_counts),
        "summary_plane_min_ok": int(summary["segment_plane_count_min"]) == min(active_counts),
        "summary_plane_max_ok": int(summary["segment_plane_count_max"]) == max(active_counts),
        "plane_histogram_ok": expected_hist == observed_hist,
        "zero_padding_ok": zero_padding_ok,
        "basis_descending_ok": basis_descending_ok,
        "reconstruction_match_ok": reconstruction_match,
        "threshold_runtime_decoded_ok": all(row["runtime_decoded_match"] for row in threshold_rows),
    }
    contract_checks["all_ok"] = all(contract_checks.values())

    return {
        "contract_checks": contract_checks,
        "plane_stats_rows": plane_stats_rows,
        "byte_hist_rows": byte_hist_rows,
        "threshold_rows": threshold_rows,
        "observed_plane_histogram": observed_hist,
        "integer_offset_histogram": integer_offset_hist,
        "effective_fractional_histogram": effective_fractional_hist,
        "raw_sum": raw_sum,
        "encoded_sum": encoded_sum,
        "sum_abs_error": sum_abs_error,
        "reconstruction_abs_diff_max": reconstruction_abs_diff_max,
        "catastrophic_count_rows": catastrophic_count_rows,
    }


def build_report(
    report_path: Path,
    *,
    zip_path: Path,
    text_path: Path,
    raw_path: Path,
    encoded_path: Path,
    artifact_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    cleaned_stats: dict[str, Any],
    chosen_power: int,
    probe_attempts: list[dict[str, Any]],
    characterization: dict[str, Any],
    plane_stats_csv: Path,
    byte_hist_csv: Path,
    threshold_csv: Path,
    real_data_source_path: Path,
) -> None:
    lines = [
        "# New Encoder Real Artifact Characterization Report",
        "",
        "Date: 2026-05-10",
        "",
        "Scope: #14 real-data artifact build and characterization only.",
        "",
        "## 1. Approved Input Path",
        "",
        f"- selection decision: `{SELECTION_DECISION_REF}`",
        f"- dataset: `{DATASET_DISPLAY}`",
        f"- dataset key: `{DATASET_KEY}`",
        f"- numeric column: `{COLUMN_NAME}`",
        f"- transform policy: `non_negative_as_is`",
        "- approved pre-cleaning rule: drop rows whose raw field value is `?` before `#14`",
        f"- downloaded zip: `{zip_path}`",
        f"- extracted text source: `{text_path}`",
        f"- cleaned FP64 slice: `{raw_path}`",
        "",
        "## 2. Cleaning Summary",
        "",
        f"- row_count_before_transform: `{cleaned_stats['row_count_before_transform']}`",
        f"- row_count_after_transform: `{cleaned_stats['row_count_after_transform']}`",
        f"- missing_count: `{cleaned_stats['missing_count']}`",
        f"- missing_marker_question_count: `{cleaned_stats['missing_marker_question_count']}`",
        f"- missing_marker_empty_count: `{cleaned_stats['missing_marker_empty_count']}`",
        f"- negative_count: `{cleaned_stats['negative_count']}`",
        f"- nan_count: `{cleaned_stats['nan_count']}`",
        f"- infinite_count: `{cleaned_stats['infinite_count']}`",
        f"- raw_min/transformed_min: `{cleaned_stats['raw_min']:.17g}`",
        f"- raw_max/transformed_max: `{cleaned_stats['raw_max']:.17g}`",
        "",
        "## 3. Precision Selection",
        "",
        f"- selected artifact label: `{manifest['artifact_label']}`",
        f"- selected precision power: `{chosen_power}`",
        f"- intermediate container: `{encoded_path}`",
        f"- runtime artifact dir: `{artifact_dir}`",
        "",
        "| Artifact label | Precision power | Exit code | Note |",
        "| --- | ---: | ---: | --- |",
    ]

    for attempt in probe_attempts:
        note = "selected" if attempt["exit_code"] == 0 else (attempt["stderr"] or attempt["stdout"] or "failed")
        lines.append(
            "| "
            + f"{attempt['artifact_label']} | {attempt['precision_power']} | {attempt['exit_code']} | "
            + note.replace("|", "/")
            + " |"
        )

    lines.extend(
        [
            "",
            "## 4. Artifact Summary",
            "",
            f"- artifact_version: `{manifest['artifact_version']}`",
            f"- artifact_family: `{manifest['artifact_family']}`",
            f"- fidelity_class: `{manifest['fidelity_class']}`",
            f"- raw_roundtrip_exact: `{manifest['raw_roundtrip_exact']}`",
            f"- format: `{manifest['format']}`",
            f"- encoded_layout: `{manifest['encoded_layout']}`",
            f"- segment_size: `{manifest['segment_size']}`",
            f"- value_count: `{manifest['value_count']}`",
            f"- segment_count: `{manifest['segment_count']}`",
            f"- max_plane_count: `{manifest['max_plane_count']}`",
            f"- active_plane_count_min/max/mean: `{summary['segment_plane_count_min']}` / `{summary['segment_plane_count_max']}` / `{summary['mean_plane_count']:.6f}`",
            f"- total_bytes_per_row: `{summary['total_bytes_per_row']:.6f}`",
            f"- quantization_bound: `{summary['quantization_bound']:.17g}`",
            f"- exact_sum(raw): `{characterization['raw_sum']:.17g}`",
            f"- encoded_sum(full-depth): `{characterization['encoded_sum']:.17g}`",
            f"- sum_abs_error(full-depth): `{characterization['sum_abs_error']:.17g}`",
            "",
            "Equivalent ilen/dlen-style metadata for this branch:",
            "",
            f"- integer_offset_bits histogram: `{characterization['integer_offset_histogram']}`",
            f"- effective_fractional_bits histogram: `{characterization['effective_fractional_histogram']}`",
            "",
            "## 5. Contract Checks",
            "",
            "| Check | Result |",
            "| --- | --- |",
        ]
    )

    for key, value in characterization["contract_checks"].items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(
        [
            "",
            f"- reconstruction_abs_diff_max(runtime vs buff_tool decode): `{characterization['reconstruction_abs_diff_max']:.17g}`",
            "",
            "## 6. Plane Behavior",
            "",
            f"- active_plane_count histogram: `{characterization['observed_plane_histogram']}`",
            f"- per-plane stats CSV: `{plane_stats_csv}`",
            f"- per-plane byte histogram CSV: `{byte_hist_csv}`",
            "",
            "| Plane | Entropy (bits) | Zero fraction | Unique values | Byte min | Byte max |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in characterization["plane_stats_rows"]:
        lines.append(
            "| "
            + f"{row['plane_index']} | {row['entropy_bits']:.6f} | {row['zero_fraction']:.6f} | "
            + f"{row['unique_values']} | {row['byte_min']} | {row['byte_max']} |"
        )

    lines.extend(
        [
            "",
            "## 7. Threshold / Selectivity Behavior",
            "",
            f"- threshold behavior CSV: `{threshold_csv}`",
            "",
            "| Target selectivity | Threshold | Raw count | Decoded count | Drift (pp) | Verdict |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for row in characterization["threshold_rows"]:
        lines.append(
            "| "
            + f"{row['target_selectivity']} | {row['threshold']:.17g} | {row['raw_count']} | "
            + f"{row['decoded_count']} | {row['selectivity_drift_pp']:.6f} | {row['count_verdict']} |"
        )

    lines.extend(
        [
            "",
            "## 8. Why This Artifact Is Safe To Use Next",
            "",
            "- The artifact uses the approved `#13` dataset, column, and transform policy.",
            "- Missing-value handling is explicit and auditable through `row_count_before_transform`, `row_count_after_transform`, and the adjacent real-data sidecar.",
            "- The runtime artifact preserves the current Exp3/Exp4 compatibility contract: `format = exp3_encoded_dev_v1`, `encoded_layout = plane_major_zero_padded`, MSB-first plane basis order, and zero padding beyond each segment's active planes.",
            "- The artifact is a compact v2 artifact because `active_plane_count_max < 8` and `total_bytes_per_row < 8`.",
            "- No GPU benchmark was required for this gate; this report is CPU-side artifact build + characterization only.",
        ]
    )

    if characterization["catastrophic_count_rows"] == 0:
        lines.append(
            "- The characterization table shows no catastrophic threshold/selectivity rows, so the artifact is safe to hand to `#15` for formal SUM/COUNT validation."
        )
    else:
        lines.append(
            "- The characterization table includes catastrophic threshold/selectivity rows, so this artifact should not be advanced to `#15` without revisiting the precision tier."
        )

    lines.extend(
        [
            "",
            "## 9. Sidecar And Provenance",
            "",
            f"- real-data sidecar: `{real_data_source_path}`",
            f"- artifact provenance: `{artifact_dir.parent.parent / 'provenance.json'}`",
            "",
            "## 10. Scope Note",
            "",
            "- This report does not claim paper-ready real-data validation by itself.",
            "- Full raw-vs-encoded SUM/COUNT validation remains future work under `#15`.",
        ]
    )

    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    workspace = Path.cwd()

    zip_path = args.zip_path.resolve()
    extract_dir = args.extract_dir.resolve()
    buff_tool = args.buff_tool.resolve()
    raw_dir = args.raw_dir.resolve()
    encoded_dir = args.encoded_dir.resolve()
    decoded_dir = args.decoded_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    plane_stats_csv = args.plane_stats_csv.resolve()
    byte_hist_csv = args.byte_hist_csv.resolve()
    threshold_csv = args.threshold_csv.resolve()
    report_path = args.report_path.resolve()

    if not buff_tool.is_file():
        raise FileNotFoundError(f"buff_tool binary not found: {buff_tool}")

    ensure_dir(raw_dir)
    ensure_dir(encoded_dir)
    ensure_dir(decoded_dir)
    ensure_dir(artifact_root)

    text_path = extract_text_file(zip_path, extract_dir)
    raw_path = raw_dir / f"{DATASET_KEY}.f64le.bin"
    cleaned_stats = materialize_clean_slice(text_path, raw_path)

    chosen_power, artifact_label, encoded_path, probe_attempts = probe_precision_powers(
        buff_tool,
        raw_path,
        encoded_dir,
        cwd=workspace,
    )

    artifact_dir = artifact_root / DATASET_KEY / artifact_label
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
            DATASET_KEY,
            "--source-path",
            str(raw_path),
        ],
        cwd=workspace,
    )

    manifest, summary, segment_rows = rewrite_artifact_metadata(
        artifact_dir,
        artifact_label=artifact_label,
        precision_power=chosen_power,
    )
    if int(summary["segment_plane_count_max"]) >= 8:
        raise RuntimeError(
            f"compactness gate failed: active_plane_count_max={summary['segment_plane_count_max']}"
        )
    if float(summary["total_bytes_per_row"]) >= 8.0:
        raise RuntimeError(f"bytes/row gate failed: total_bytes_per_row={summary['total_bytes_per_row']}")

    write_root_index(artifact_root, artifact_label=artifact_label)
    write_provenance(
        artifact_root / "provenance.json",
        argv=sys.argv,
        extra={
            "artifact_root": str(artifact_root),
            "raw_dir": str(raw_dir),
            "encoded_dir": str(encoded_dir),
            "selected_dataset": DATASET_KEY,
            "selected_artifact_label": artifact_label,
            "selected_precision_power": chosen_power,
            "segment_size": SEGMENT_SIZE,
        },
    )

    transform_command = (
        f"{Path(sys.argv[0]).name}: extract {zip_path.name}, "
        f"select column {COLUMN_NAME}, drop '?' rows, write {raw_path.name}"
    )
    real_data_source_path = artifact_dir / REAL_DATA_SOURCE_FILENAME
    write_real_data_source(
        real_data_source_path,
        text_path=text_path,
        cleaned_stats=cleaned_stats,
        transform_command=transform_command,
    )

    decoded_path = decoded_dir / f"{DATASET_KEY}_{artifact_label}.decoded.f64le.bin"
    decode_full_depth(buff_tool, encoded_path, decoded_path, cwd=workspace)

    characterization = characterize_artifact(
        artifact_dir,
        manifest,
        summary,
        segment_rows,
        raw_path,
        decoded_path,
    )
    write_plane_stats_csv(plane_stats_csv, characterization["plane_stats_rows"])
    write_byte_hist_csv(byte_hist_csv, characterization["byte_hist_rows"])
    write_threshold_csv(threshold_csv, characterization["threshold_rows"])

    write_provenance(
        plane_stats_csv.with_suffix(plane_stats_csv.suffix + ".provenance.json"),
        argv=sys.argv,
        extra={"artifact_dir": str(artifact_dir), "row_count": len(characterization["plane_stats_rows"])},
    )
    write_provenance(
        byte_hist_csv.with_suffix(byte_hist_csv.suffix + ".provenance.json"),
        argv=sys.argv,
        extra={"artifact_dir": str(artifact_dir), "row_count": len(characterization["byte_hist_rows"])},
    )
    write_provenance(
        threshold_csv.with_suffix(threshold_csv.suffix + ".provenance.json"),
        argv=sys.argv,
        extra={"artifact_dir": str(artifact_dir), "row_count": len(characterization["threshold_rows"])},
    )

    build_report(
        report_path,
        zip_path=zip_path,
        text_path=text_path,
        raw_path=raw_path,
        encoded_path=encoded_path,
        artifact_dir=artifact_dir,
        manifest=manifest,
        summary=summary,
        cleaned_stats=cleaned_stats,
        chosen_power=chosen_power,
        probe_attempts=probe_attempts,
        characterization=characterization,
        plane_stats_csv=plane_stats_csv,
        byte_hist_csv=byte_hist_csv,
        threshold_csv=threshold_csv,
        real_data_source_path=real_data_source_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
