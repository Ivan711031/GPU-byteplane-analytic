#!/usr/bin/env python3
"""Build and characterize the approved scientific dataset v2 runtime artifacts."""

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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

DATE_TAG = "20260520"
ARTIFACT_VERSION = "buff_bfp_v2"
ARTIFACT_FAMILY = "buff_bfp"
FIDELITY_CLASS = "precision_bounded"
PRECISION_POLICY = "fixed"
SEGMENT_SIZE = 4096
THRESHOLD_GRID = [1, 5, 10, 25, 50, 75, 90, 95, 99]
REAL_DATA_SOURCE_FILENAME = "real_data_source.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cesm_atm_cloud", "cesm_atm_q", "cesm_atm_t", "hurricane_u", "hurricane_tc", "hurricane_w"],
        help="Which scientific dataset to process.",
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
        default=Path(f"results/buff_encoder_v2/raw_scientific"),
        help="Directory for the cleaned FP64 binary slice.",
    )
    parser.add_argument(
        "--encoded-dir",
        type=Path,
        default=Path(f"results/buff_encoder_v2/containers_scientific"),
        help="Directory for intermediate .buff64 containers.",
    )
    parser.add_argument(
        "--decoded-dir",
        type=Path,
        default=Path(f"results/buff_encoder_v2/decoded_scientific"),
        help="Directory for full-depth decoded .buff64 output.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(f"datasets/scientific/dev_buff_v2_scientific"),
        help="Versioned scientific-data runtime artifact root.",
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=4096,
        help="Segment size for encoding (default: 4096).",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def git_commit(cwd: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()


def run_checked(cmd: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=cwd, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            f"{' '.join(shlex.quote(part) for part in cmd)}"
        )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_segment_meta(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def write_provenance(path: Path, *, argv: list[str], extra: dict[str, Any] | None = None) -> None:
    payload = {
        "argv": argv,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(Path.cwd()),
        "hostname": socket.gethostname(),
        "username": os.environ.get("USER", "unknown"),
    }
    if extra:
        payload.update(extra)
    write_json(path, payload)


def materialize_clean_slice(
    dataset: str,
    f32_path: Path,
    raw_path: Path,
) -> dict[str, Any]:
    print(f"Reading F32 input: {f32_path}")
    raw_f32 = np.fromfile(f32_path, dtype=np.float32)
    row_count_before = len(raw_f32)
    
    finite_mask = np.isfinite(raw_f32)
    finite_count = int(np.sum(finite_mask))
    nan_count = int(np.sum(np.isnan(raw_f32)))
    infinite_count = int(np.sum(np.isinf(raw_f32)))
    negative_count = int(np.sum(raw_f32 < 0))
    
    raw_min = float(np.min(raw_f32[finite_mask])) if finite_count > 0 else 0.0
    raw_max = float(np.max(raw_f32[finite_mask])) if finite_count > 0 else 0.0

    print(f"Loaded {row_count_before} values. Min: {raw_min}, Max: {raw_max}, Negatives: {negative_count}")

    # Cleaning and offset transform according to rules
    if dataset in ("cesm_atm_cloud", "cesm_atm_q", "cesm_atm_t"):
        # clip negative float-underflow noise to 0.0
        cleaned = np.clip(raw_f32, 0.0, None).astype(np.float64)
        offset_constant = 0.0
    elif dataset in ("hurricane_u", "hurricane_tc", "hurricane_w"):
        # apply global offset transform: offset_constant = -raw_min
        offset_constant = -raw_min
        cleaned = (raw_f32 + offset_constant).astype(np.float64)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    row_count_after = len(cleaned)
    exact_sum = float(np.sum(cleaned))
    
    # Save as raw FP64 binary
    print(f"Writing FP64 cleaned output: {raw_path}")
    cleaned.tofile(raw_path)

    return {
        "row_count_before_transform": row_count_before,
        "row_count_after_transform": row_count_after,
        "finite_count": finite_count,
        "missing_count": 0,
        "nan_count": nan_count,
        "infinite_count": infinite_count,
        "negative_count": negative_count,
        "offset_constant": offset_constant,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "transformed_min": float(np.min(cleaned)),
        "transformed_max": float(np.max(cleaned)),
        "exact_sum": exact_sum,
        "raw_path": str(raw_path),
    }


def probe_precision_powers(
    buff_tool: Path,
    raw_path: Path,
    encoded_dir: Path,
    dataset_key: str,
    *,
    segment_size: int,
    cwd: Path,
) -> tuple[int, str, Path, list[dict[str, Any]]]:
    ensure_dir(encoded_dir)
    attempts: list[dict[str, Any]] = []
    # Try precision levels 12 down to 1
    for power in range(12, 0, -1):
        artifact_label = f"bfp-dec{power}"
        encoded_path = encoded_dir / f"{dataset_key}_{artifact_label}.buff64"
        cmd = [
            str(buff_tool),
            "encode",
            "--input",
            str(raw_path),
            "--output",
            str(encoded_path),
            "--segment-size",
            str(segment_size),
            "--precision-power",
            str(power),
        ]
        completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            print(f"Probing: precision power {power} failed encoding: {completed.stderr.strip()[:100]}")
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
            continue

        # Export runtime layout to a probe-specific directory to verify gates
        probe_out_dir = encoded_dir / f"probe_{dataset_key}_p{power}"
        ensure_dir(probe_out_dir)
        export_cmd = [
            str(buff_tool),
            "export-runtime",
            "--input",
            str(encoded_path),
            "--raw-input",
            str(raw_path),
            "--out-dir",
            str(probe_out_dir),
            "--dataset",
            dataset_key,
            "--source-path",
            str(raw_path),
        ]
        export_res = subprocess.run(export_cmd, cwd=cwd, text=True, capture_output=True, check=False)
        if export_res.returncode != 0:
            print(f"Probing: precision power {power} failed export: {export_res.stderr.strip()[:100]}")
            attempts.append(
                {
                    "precision_power": power,
                    "artifact_label": artifact_label,
                    "encoded_path": str(encoded_path),
                    "exit_code": export_res.returncode,
                    "stderr": export_res.stderr.strip(),
                    "stdout": export_res.stdout.strip(),
                }
            )
            continue

        summary_path = probe_out_dir / "summary.json"
        if not summary_path.is_file():
            print(f"Probing: precision power {power} summary.json not found")
            attempts.append(
                {
                    "precision_power": power,
                    "artifact_label": artifact_label,
                    "encoded_path": str(encoded_path),
                    "exit_code": 999,
                    "stderr": "summary.json missing",
                    "stdout": "",
                }
            )
            continue

        summary_val = read_json(summary_path)
        plane_max = int(summary_val["segment_plane_count_max"])
        value_count = int(summary_val["value_count"])
        plane_files = sorted(probe_out_dir.glob("plane_*.bin"))
        main_bytes_total = sum(path.stat().st_size for path in plane_files)
        bytes_per_row = main_bytes_total / value_count

        print(f"Probing: precision power {power} encoded: plane_max={plane_max}, bytes_per_row={bytes_per_row:.4f}")

        if plane_max < 8 and bytes_per_row < 8.0:
            print(f"Probing: precision power {power} succeeded and passed gates.")
            attempts.append(
                {
                    "precision_power": power,
                    "artifact_label": artifact_label,
                    "encoded_path": str(encoded_path),
                    "exit_code": 0,
                    "stderr": "",
                    "stdout": f"Passed gates: plane_max={plane_max}, bytes_per_row={bytes_per_row}",
                }
            )
            return power, artifact_label, encoded_path, attempts
        else:
            print(f"Probing: precision power {power} violated gates (plane_max={plane_max}, bytes_per_row={bytes_per_row})")
            attempts.append(
                {
                    "precision_power": power,
                    "artifact_label": artifact_label,
                    "encoded_path": str(encoded_path),
                    "exit_code": 998,
                    "stderr": f"Violated gates: plane_max={plane_max}, bytes_per_row={bytes_per_row}",
                    "stdout": "",
                }
            )

    raise RuntimeError("No precision tier in bfp-dec1..bfp-dec12 produced an artifact passing the gates (<8 planes and <8.0 bytes/row)")


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


def write_root_index(artifact_root: Path, dataset_key: str, *, artifact_label: str) -> None:
    payload = {
        "manifest_scope": "root_index",
        "artifact_version": ARTIFACT_VERSION,
        "datasets": [dataset_key],
        "artifact_labels": [artifact_label],
        "entries": [
            {
                "dataset": dataset_key,
                "artifact_label": artifact_label,
                "relative_path": str(Path(dataset_key) / artifact_label),
            }
        ],
    }
    write_json(artifact_root / "manifest.json", payload)


def write_real_data_source(
    path: Path,
    *,
    dataset_display: str,
    column_name: str,
    column_semantics: str,
    column_unit: str,
    f32_path: Path,
    cleaned_stats: dict[str, Any],
) -> None:
    payload = {
        "column_name": column_name,
        "column_semantics": column_semantics,
        "column_unit": column_unit,
        "dataset_name": dataset_display,
        "dataset_version_or_snapshot": "SDRBench Scientific Dataset",
        "finite_count": cleaned_stats["finite_count"],
        "infinite_count": cleaned_stats["infinite_count"],
        "missing_count": cleaned_stats["missing_count"],
        "nan_count": cleaned_stats["nan_count"],
        "negative_count": cleaned_stats["negative_count"],
        "offset_constant": cleaned_stats["offset_constant"],
        "raw_max": cleaned_stats["raw_max"],
        "raw_min": cleaned_stats["raw_min"],
        "raw_threshold_unit": column_unit,
        "row_count_after_transform": cleaned_stats["row_count_after_transform"],
        "row_count_before_transform": cleaned_stats["row_count_before_transform"],
        "selection_decision_ref": "research/2026-05-20_Scientific_Data_Qualification_Implementation_Plan.md",
        "source_table_or_file": str(f32_path),
        "threshold_policy": "offset_transformed" if cleaned_stats["offset_constant"] != 0.0 else "identity",
        "transform_command_or_script": f"build_scientific_v2_artifact.py, offset {cleaned_stats['offset_constant']}",
        "transform_policy": "global_offset_transform" if cleaned_stats["offset_constant"] != 0.0 else "non_negative_as_is",
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
    dataset_key: str,
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
                "dataset": dataset_key,
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
                    "dataset": dataset_key,
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
                "dataset": dataset_key,
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
    dataset_key: str,
    dataset_display: str,
    column_name: str,
    f32_path: Path,
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
        f"# Scientific Artifact Characterization Report: {dataset_display}",
        "",
        f"Date: 2026-05-20",
        "",
        "Scope: Scientific dataset promotion and v2 artifact build.",
        "",
        "## 1. Approved Input Path",
        "",
        f"- dataset: `{dataset_display}`",
        f"- dataset key: `{dataset_key}`",
        f"- numeric column: `{column_name}`",
        f"- source file: `{f32_path}`",
        f"- cleaned FP64 slice: `{raw_path}`",
        "",
        "## 2. Cleaning / Transformation Summary",
        "",
        f"- row_count_before_transform: `{cleaned_stats['row_count_before_transform']}`",
        f"- row_count_after_transform: `{cleaned_stats['row_count_after_transform']}`",
        f"- missing_count: `{cleaned_stats['missing_count']}`",
        f"- negative_count: `{cleaned_stats['negative_count']}`",
        f"- nan_count: `{cleaned_stats['nan_count']}`",
        f"- infinite_count: `{cleaned_stats['infinite_count']}`",
        f"- raw_min: `{cleaned_stats['raw_min']:.17g}`",
        f"- raw_max: `{cleaned_stats['raw_max']:.17g}`",
        f"- offset_constant applied: `{cleaned_stats['offset_constant']:.17g}`",
        f"- transformed_min: `{cleaned_stats['transformed_min']:.17g}`",
        f"- transformed_max: `{cleaned_stats['transformed_max']:.17g}`",
        "",
        "## 3. Precision Selection Probing",
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
            f"- main_bytes_total: `{summary['main_bytes_total']}`",
            f"- total_bytes_per_row: `{summary['total_bytes_per_row']:.6f}`",
            "",
            "## 5. Fidelity Audit",
            "",
            f"- contract_checks: `{'all_ok' if characterization['contract_checks']['all_ok'] else 'failed'}`",
            f"- raw_sum: `{characterization['raw_sum']:.17g}`",
            f"- encoded_sum: `{characterization['encoded_sum']:.17g}`",
            f"- sum_abs_error: `{characterization['sum_abs_error']:.17g}`",
            f"- reconstruction_abs_diff_max: `{characterization['reconstruction_abs_diff_max']:.17g}`",
            f"- catastrophic_count_rows: `{characterization['catastrophic_count_rows']}`",
            "",
            "| Target Selectivity (%) | Threshold | Raw Count | Decoded Count | Drift (pp) | Verdict |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for row in characterization["threshold_rows"]:
        lines.append(
            f"| {row['target_selectivity']} | {row['threshold']:.6g} | "
            f"{row['raw_count']} | {row['decoded_count']} | "
            f"{row['selectivity_drift_pp']:.3f} | {row['count_verdict']} |"
        )

    lines.extend(
        [
            "",
            "## 6. Layout Properties",
            "",
            f"- plane_count_histogram: `{summary['plane_count_histogram']}`",
            f"- integer_offset_histogram: `{characterization['integer_offset_histogram']}`",
            f"- effective_fractional_histogram: `{characterization['effective_fractional_histogram']}`",
            "",
            "### Detailed Verification Checks",
            "",
        ]
    )
    for check_key, check_val in characterization["contract_checks"].items():
        lines.append(f"- `{check_key}`: {'PASS' if check_val else 'FAIL'}")

    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Report written: {report_path}")


def main() -> int:
    args = parse_args()
    workspace = Path.cwd()
    buff_tool = args.buff_tool.resolve()
    
    if args.dataset == "cesm_atm_cloud":
        dataset_key = "cesm_atm_cloud"
        dataset_display = "CESM-ATM CLOUD"
        column_name = "CLOUD"
        column_semantics = "cloud fraction"
        column_unit = "fraction"
        f32_path = Path("${WORK_DIR}/datasets/scientific/SDRBENCH-CESM-ATM-26x1800x3600/CLOUD_1_26_1800_3600.f32")
    elif args.dataset == "cesm_atm_q":
        dataset_key = "cesm_atm_q"
        dataset_display = "CESM-ATM Q (specific humidity)"
        column_name = "Q"
        column_semantics = "specific humidity"
        column_unit = "kg/kg"
        f32_path = Path("${WORK_DIR}/datasets/scientific/SDRBENCH-CESM-ATM-26x1800x3600/Q_1_26_1800_3600.f32")
    elif args.dataset == "cesm_atm_t":
        dataset_key = "cesm_atm_t"
        dataset_display = "CESM-ATM T (temperature)"
        column_name = "T"
        column_semantics = "air temperature"
        column_unit = "K"
        f32_path = Path("${WORK_DIR}/datasets/scientific/SDRBENCH-CESM-ATM-26x1800x3600/T_1_26_1800_3600.f32")
    elif args.dataset == "hurricane_u":
        dataset_key = "hurricane_u"
        dataset_display = "Hurricane Isabel U"
        column_name = "U"
        column_semantics = "wind speed U component"
        column_unit = "m/s"
        f32_path = Path("${WORK_DIR}/datasets/scientific/100x500x500/Uf48.bin.f32")
    elif args.dataset == "hurricane_tc":
        dataset_key = "hurricane_tc"
        dataset_display = "Hurricane Isabel TC (temperature)"
        column_name = "TC"
        column_semantics = "temperature in Celsius"
        column_unit = "degC"
        f32_path = Path("${WORK_DIR}/datasets/scientific/100x500x500/TCf48.bin.f32")
    elif args.dataset == "hurricane_w":
        dataset_key = "hurricane_w"
        dataset_display = "Hurricane Isabel W (vertical velocity)"
        column_name = "W"
        column_semantics = "vertical wind velocity"
        column_unit = "m/s"
        f32_path = Path("${WORK_DIR}/datasets/scientific/100x500x500/Wf48.bin.f32")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if not f32_path.is_file():
        raise FileNotFoundError(f"Input file not found: {f32_path}")

    raw_dir = args.raw_dir.resolve()
    encoded_dir = args.encoded_dir.resolve()
    decoded_dir = args.decoded_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    
    ensure_dir(raw_dir)
    ensure_dir(encoded_dir)
    ensure_dir(decoded_dir)
    ensure_dir(artifact_root)

    raw_path = raw_dir / f"{dataset_key}.f64le.bin"
    cleaned_stats = materialize_clean_slice(args.dataset, f32_path, raw_path)

    chosen_power, artifact_label, encoded_path, probe_attempts = probe_precision_powers(
        buff_tool,
        raw_path,
        encoded_dir,
        dataset_key,
        segment_size=args.segment_size,
        cwd=workspace,
    )

    artifact_dir = artifact_root / dataset_key / artifact_label
    ensure_dir(artifact_dir)
    print(f"Exporting runtime layout to: {artifact_dir}")
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
            dataset_key,
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

    if int(summary["segment_plane_count_max"]) > 8:
        raise RuntimeError(
            f"compactness gate failed: active_plane_count_max={summary['segment_plane_count_max']}"
        )
    if float(summary["total_bytes_per_row"]) >= 8.0:
        raise RuntimeError(f"bytes/row gate failed: total_bytes_per_row={summary['total_bytes_per_row']}")

    write_root_index(artifact_root, dataset_key, artifact_label=artifact_label)
    
    write_provenance(
        artifact_root / "provenance.json",
        argv=sys.argv,
        extra={
            "artifact_root": str(artifact_root),
            "raw_dir": str(raw_dir),
            "encoded_dir": str(encoded_dir),
            "selected_dataset": dataset_key,
            "selected_artifact_label": artifact_label,
            "selected_precision_power": chosen_power,
            "segment_size": args.segment_size,
        },
    )

    real_data_source_path = artifact_dir / REAL_DATA_SOURCE_FILENAME
    write_real_data_source(
        real_data_source_path,
        dataset_display=dataset_display,
        column_name=column_name,
        column_semantics=column_semantics,
        column_unit=column_unit,
        f32_path=f32_path,
        cleaned_stats=cleaned_stats,
    )

    decoded_path = decoded_dir / f"{dataset_key}_{artifact_label}.decoded.f64le.bin"
    decode_full_depth(buff_tool, encoded_path, decoded_path, cwd=workspace)

    print("Characterizing artifact and run verification...")
    characterization = characterize_artifact(
        dataset_key,
        artifact_dir,
        manifest,
        summary,
        segment_rows,
        raw_path,
        decoded_path,
    )

    plane_stats_csv = encoded_dir / f"{dataset_key}_plane_stats.csv"
    byte_hist_csv = encoded_dir / f"{dataset_key}_byte_histogram.csv"
    threshold_csv = encoded_dir / f"{dataset_key}_threshold_behavior.csv"

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

    if args.segment_size == 4096:
        report_path = Path(f"research/{DATE_TAG[:4]}-{DATE_TAG[4:6]}-{DATE_TAG[6:]}_{dataset_key}_v2_Artifact_Characterization_Report.md")
    else:
        report_path = Path(f"research/{DATE_TAG[:4]}-{DATE_TAG[4:6]}-{DATE_TAG[6:]}_{dataset_key}_v2_Artifact_Characterization_Report_global.md")
    build_report(
        report_path,
        dataset_key=dataset_key,
        dataset_display=dataset_display,
        column_name=column_name,
        f32_path=f32_path,
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
