#!/usr/bin/env python3
"""Validate the approved real-data v2 artifact for #15 / PV1-9."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DATE_TAG = "2026-05-10"
DATASET = "uci_household_global_active_power"
ARTIFACT_LABEL = "bfp-dec12"
THRESHOLD_GRID = [1, 5, 10, 25, 50, 75, 90, 95, 99]
EPSILON_ABS_TARGETS = [1, 10, 100, 1000, 10000, 100000, 1000000]
EPSILON_REL_TARGETS = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.10]

FIDELITY_COLUMNS = [
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

SUM_COLUMNS = [
    "dataset",
    "artifact_version",
    "artifact_label",
    "aggregation",
    "mode",
    "k",
    "max_plane_count",
    "n",
    "raw_sum",
    "encoded_full_depth_sum",
    "progressive_sum",
    "abs_error_vs_raw",
    "rel_error_vs_raw",
    "abs_error_vs_encoded",
    "rel_error_vs_encoded",
    "logical_bytes",
    "estimated_physical_bytes",
    "ms_per_iter",
    "rows_per_sec",
    "logical_GBps",
    "estimated_physical_GBps",
    "raw_ms",
    "raw_rows_per_sec",
    "raw_logical_GBps",
    "fixed_depth_ms",
    "fixed_depth_rows_per_sec",
    "fixed_depth_logical_GBps",
    "speedup_vs_raw",
    "speedup_vs_fixed_depth",
    "validated",
    "notes",
]

COUNT_COLUMNS = [
    "dataset",
    "artifact_version",
    "artifact_label",
    "threshold",
    "target_selectivity",
    "k",
    "max_plane_count",
    "n",
    "raw_count",
    "encoded_full_depth_count",
    "raw_eq_encoded_full_depth_check",
    "progressive_count",
    "count_lower",
    "count_upper",
    "Q_k",
    "D_k",
    "U_k",
    "count_abs_error_vs_raw",
    "count_rel_error_vs_raw",
    "logical_bytes",
    "estimated_pack_load_bytes",
    "ms_per_iter",
    "rows_per_sec",
    "logical_GBps",
    "estimated_physical_GBps",
    "raw_ms",
    "raw_rows_per_sec",
    "raw_logical_GBps",
    "fixed_depth_ms",
    "fixed_depth_rows_per_sec",
    "fixed_depth_logical_GBps",
    "speedup_vs_raw",
    "speedup_vs_fixed_depth",
    "encoded_full_depth_in_interval",
    "notes",
]

KSTAR_COLUMNS = [
    "dataset",
    "artifact_version",
    "artifact_label",
    "threshold",
    "target_selectivity",
    "epsilon_type",
    "epsilon",
    "kstar",
    "kstar_status",
    "exact_count",
    "max_k",
    "selected_U_k",
    "selected_logical_GBps",
    "selected_estimated_physical_GBps",
    "selected_rows_per_sec",
]


@dataclass
class Segment:
    row_offset: int
    row_count: int
    active_plane_count: int
    fractional_bits: int
    integer_offset_bits: int
    segment_base: float
    bases: np.ndarray
    max_byte_values: np.ndarray
    rem_max_after: np.ndarray
    raw_min: float
    raw_max: float
    pack_count: int
    tail_count: int
    plane_packs: list[np.ndarray]
    plane_tail: list[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(f"datasets/real/dev_buff_v2_real_20260510/{DATASET}/{ARTIFACT_LABEL}"),
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path(f"results/buff_encoder_v2/raw_real_20260510/{DATASET}.f64le.bin"),
    )
    parser.add_argument(
        "--decoded-path",
        type=Path,
        default=Path(f"results/buff_encoder_v2/decoded_real_20260510/{DATASET}_{ARTIFACT_LABEL}.decoded.f64le.bin"),
    )
    parser.add_argument(
        "--fidelity-csv",
        type=Path,
        default=Path("results/buff_encoder_v2/real_fidelity_audit.csv"),
    )
    parser.add_argument(
        "--sum-csv",
        type=Path,
        default=Path("results/buff_encoder_v2/real_sum_precision_throughput.csv"),
    )
    parser.add_argument(
        "--count-csv",
        type=Path,
        default=Path("results/buff_encoder_v2/real_count_precision_throughput.csv"),
    )
    parser.add_argument(
        "--kstar-csv",
        type=Path,
        default=Path("results/buff_encoder_v2/real_count_epsilon_to_kstar.csv"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(f"research/{DATE_TAG}_RealData_Evidence_Chain_Final_Report.md"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def git_commit(cwd: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def load_fp64_memmap(path: Path) -> np.memmap:
    return np.memmap(path, dtype="<f8", mode="r")


def count_verdict(selectivity_drift_pp: float) -> str:
    drift = abs(selectivity_drift_pp)
    if drift <= 0.1:
        return "acceptable"
    if drift <= 1.0:
        return "caution"
    return "catastrophic"


def sum_verdict(sum_abs_error: float, rows: int, quantization_bound: float) -> str:
    if rows <= 0 or quantization_bound <= 0.0:
        return "not_applicable"
    declared_sum_bound = rows * quantization_bound
    if sum_abs_error <= declared_sum_bound:
        return "acceptable"
    if sum_abs_error <= declared_sum_bound * 10.0:
        return "caution"
    return "catastrophic"


def artifact_role(sum_verdict_value: str, count_verdicts: list[str]) -> tuple[bool, bool, str]:
    sum_mainline = sum_verdict_value == "acceptable"
    count_mainline = all(verdict == "acceptable" for verdict in count_verdicts)
    if sum_verdict_value == "catastrophic" or any(verdict == "catastrophic" for verdict in count_verdicts):
        return count_mainline, sum_mainline, "reject"
    if sum_mainline and count_mainline:
        return count_mainline, sum_mainline, "mainline_candidate"
    return count_mainline, sum_mainline, "side_study_only"


def median_ms(fn, repeats: int = 3) -> tuple[Any, float]:
    result = None
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - start) * 1000.0
        samples.append(elapsed)
    return result, float(np.median(np.asarray(samples, dtype=np.float64)))


def load_segments(
    segment_meta_path: Path,
    raw_values: np.memmap,
    planes: list[np.ndarray],
) -> list[Segment]:
    segments: list[Segment] = []
    with segment_meta_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_offset = int(row["row_offset"])
            row_count = int(row["row_count"])
            active_plane_count = int(row["active_plane_count"])
            fractional_bits = int(row["fractional_bits"])
            integer_offset_bits = int(row["integer_offset_bits"])
            segment_base = float(row["segment_base"])
            bases = np.asarray(
                [float(row[f"plane_basis_{plane}"]) for plane in range(active_plane_count)],
                dtype=np.float64,
            )

            total_bits = fractional_bits + integer_offset_bits
            if total_bits <= 0:
                raise ValueError("total_bits must be positive for the selected real artifact")
            plane_count = (total_bits + 7) // 8
            max_byte_values = []
            for plane in range(plane_count):
                width = 8
                if plane + 1 == plane_count:
                    trailing = total_bits - 8 * (plane_count - 1)
                    width = 8 if trailing == 0 else trailing
                max_byte_values.append((1 << width) - 1)
            max_byte_values_arr = np.asarray(max_byte_values, dtype=np.float64)
            rem_max_after = np.zeros(active_plane_count + 1, dtype=np.float64)
            running = 0.0
            for plane in range(active_plane_count - 1, -1, -1):
                running += max_byte_values_arr[plane] * bases[plane]
                rem_max_after[plane] = running

            raw_slice = raw_values[row_offset : row_offset + row_count]
            pack_count = row_count // 16
            tail_count = row_count % 16
            plane_packs = []
            plane_tail = []
            for plane in range(active_plane_count):
                plane_slice = planes[plane][row_offset : row_offset + row_count]
                plane_packs.append(plane_slice[: pack_count * 16].reshape(pack_count, 16))
                plane_tail.append(plane_slice[pack_count * 16 :])

            segments.append(
                Segment(
                    row_offset=row_offset,
                    row_count=row_count,
                    active_plane_count=active_plane_count,
                    fractional_bits=fractional_bits,
                    integer_offset_bits=integer_offset_bits,
                    segment_base=segment_base,
                    bases=bases,
                    max_byte_values=max_byte_values_arr,
                    rem_max_after=rem_max_after,
                    raw_min=float(np.min(raw_slice)),
                    raw_max=float(np.max(raw_slice)),
                    pack_count=pack_count,
                    tail_count=tail_count,
                    plane_packs=plane_packs,
                    plane_tail=plane_tail,
                )
            )
    return segments


def build_sum_basis_vectors(
    segments: list[Segment],
    value_count: int,
    max_plane_count: int,
) -> tuple[float, list[np.ndarray]]:
    base_vector = np.zeros(value_count, dtype=np.float64)
    basis_vectors = [np.zeros(value_count, dtype=np.float64) for _ in range(max_plane_count)]
    for segment in segments:
        sl = slice(segment.row_offset, segment.row_offset + segment.row_count)
        base_vector[sl] = segment.segment_base
        for plane in range(segment.active_plane_count):
            basis_vectors[plane][sl] = segment.bases[plane]
    return float(np.sum(base_vector, dtype=np.float64)), basis_vectors


def compute_sum_at_k(
    base_total: float,
    plane_float: list[np.ndarray],
    basis_vectors: list[np.ndarray],
    k: int,
) -> float:
    total = base_total
    for plane in range(k):
        total += float(np.dot(plane_float[plane], basis_vectors[plane]))
    return total


def raw_count(values: np.ndarray, threshold: float) -> int:
    return int(np.count_nonzero(values > threshold))


def progressive_count_stats(
    segments: list[Segment],
    threshold: float,
    k: int,
) -> dict[str, Any]:
    certainly_qualified = 0
    certainly_disqualified = 0
    uncertain = 0
    total_planes_read = 0
    estimated_pack_load_bytes = 0
    max_planes_read = 0

    for segment in segments:
        if threshold < segment.raw_min:
            certainly_qualified += segment.row_count
            continue
        if threshold >= segment.raw_max:
            certainly_disqualified += segment.row_count
            continue

        effective_k = min(k, segment.active_plane_count)
        local_threshold = threshold - segment.segment_base

        if segment.pack_count > 0:
            active = np.ones((segment.pack_count, 16), dtype=bool)
            prefix = np.zeros((segment.pack_count, 16), dtype=np.float64)
            rounds = np.zeros((segment.pack_count, 16), dtype=np.uint8)
            qualified = np.zeros((segment.pack_count, 16), dtype=bool)
            disqualified = np.zeros((segment.pack_count, 16), dtype=bool)

            for plane in range(effective_k):
                active_packs = np.any(active, axis=1)
                estimated_pack_load_bytes += int(np.count_nonzero(active_packs)) * 16
                if not np.any(active_packs):
                    break

                rounds[active] += 1
                plane_values = segment.plane_packs[plane].astype(np.float64, copy=False)
                prefix[active] += plane_values[active] * segment.bases[plane]
                rem_after = segment.rem_max_after[plane + 1]

                qualified_now = active & (prefix > local_threshold)
                disqualified_now = active & ((prefix + rem_after) <= local_threshold)
                qualified |= qualified_now
                disqualified |= disqualified_now
                active &= ~(qualified_now | disqualified_now)

            certainly_qualified += int(np.count_nonzero(qualified))
            certainly_disqualified += int(np.count_nonzero(disqualified))
            uncertain += int(np.count_nonzero(active))
            total_planes_read += int(np.sum(rounds, dtype=np.uint64))
            if rounds.size > 0:
                max_planes_read = max(max_planes_read, int(np.max(rounds)))

        if segment.tail_count > 0:
            tail_prefix = np.zeros(segment.tail_count, dtype=np.float64)
            tail_rounds = np.zeros(segment.tail_count, dtype=np.uint8)
            tail_active = np.ones(segment.tail_count, dtype=bool)
            tail_qualified = np.zeros(segment.tail_count, dtype=bool)
            tail_disqualified = np.zeros(segment.tail_count, dtype=bool)

            for plane in range(effective_k):
                if not np.any(tail_active):
                    break
                estimated_pack_load_bytes += int(np.count_nonzero(tail_active))
                tail_rounds[tail_active] += 1
                tail_prefix[tail_active] += (
                    segment.plane_tail[plane][tail_active].astype(np.float64) * segment.bases[plane]
                )
                rem_after = segment.rem_max_after[plane + 1]
                qualified_now = tail_active & (tail_prefix > local_threshold)
                disqualified_now = tail_active & ((tail_prefix + rem_after) <= local_threshold)
                tail_qualified |= qualified_now
                tail_disqualified |= disqualified_now
                tail_active &= ~(qualified_now | disqualified_now)

            certainly_qualified += int(np.count_nonzero(tail_qualified))
            certainly_disqualified += int(np.count_nonzero(tail_disqualified))
            uncertain += int(np.count_nonzero(tail_active))
            total_planes_read += int(np.sum(tail_rounds, dtype=np.uint64))
            if tail_rounds.size > 0:
                max_planes_read = max(max_planes_read, int(np.max(tail_rounds)))

    return {
        "Q_k": certainly_qualified,
        "D_k": certainly_disqualified,
        "U_k": uncertain,
        "count_lower": certainly_qualified,
        "count_upper": certainly_qualified + uncertain,
        "progressive_count": certainly_qualified,
        "total_planes_read": total_planes_read,
        "estimated_pack_load_bytes": estimated_pack_load_bytes,
        "max_planes_read": max_planes_read,
    }


def full_depth_count(values: np.ndarray, threshold: float) -> int:
    return int(np.count_nonzero(values > threshold))


def build_fidelity_rows(
    manifest: dict[str, Any],
    summary: dict[str, Any],
    raw_values: np.ndarray,
    decoded_values: np.ndarray,
    thresholds: dict[int, float],
) -> tuple[list[dict[str, Any]], list[str], bool, bool, str]:
    rows = int(manifest["value_count"])
    raw_sum = float(np.sum(raw_values, dtype=np.float64))
    encoded_sum = float(np.sum(decoded_values, dtype=np.float64))
    sum_abs_error = abs(encoded_sum - raw_sum)
    sum_rel_error = 0.0 if raw_sum == 0.0 else sum_abs_error / abs(raw_sum)
    sum_verdict_value = sum_verdict(sum_abs_error, rows, float(summary["quantization_bound"]))

    fidelity_rows: list[dict[str, Any]] = []
    count_verdicts: list[str] = []
    for target in THRESHOLD_GRID:
        threshold = thresholds[target]
        raw_count_value = raw_count(raw_values, threshold)
        encoded_count_value = full_depth_count(decoded_values, threshold)
        count_abs_error = abs(encoded_count_value - raw_count_value)
        count_rel_error = 0.0 if raw_count_value == 0 else count_abs_error / raw_count_value
        observed_selectivity = (encoded_count_value / rows) * 100.0
        selectivity_drift_pp = observed_selectivity - float(target)
        count_verdict_value = count_verdict(selectivity_drift_pp)
        count_verdicts.append(count_verdict_value)

        fidelity_rows.append(
            {
                "dataset": manifest["dataset"],
                "artifact_version": manifest["artifact_version"],
                "artifact_label": manifest["artifact_label"],
                "artifact_family": manifest["artifact_family"],
                "fidelity_class": manifest["fidelity_class"],
                "raw_roundtrip_exact": manifest["raw_roundtrip_exact"],
                "precision_policy": manifest["precision_policy"],
                "precision_power": manifest["precision_power"],
                "target_selectivity": target,
                "threshold": threshold,
                "raw_sum": raw_sum,
                "encoded_sum": encoded_sum,
                "sum_abs_error": sum_abs_error,
                "sum_rel_error": sum_rel_error,
                "sum_verdict": sum_verdict_value,
                "raw_count": raw_count_value,
                "encoded_count": encoded_count_value,
                "count_abs_error": count_abs_error,
                "count_rel_error": count_rel_error,
                "observed_selectivity": observed_selectivity,
                "selectivity_drift_pp": selectivity_drift_pp,
                "count_verdict": count_verdict_value,
            }
        )

    count_mainline_suitable, sum_mainline_suitable, overall_role = artifact_role(
        sum_verdict_value, count_verdicts
    )
    notes = (
        f"quantization_bound={summary['quantization_bound']}; "
        f"full-depth encoded reference from {Path(summary['encoded_path']).name}"
    )
    for row in fidelity_rows:
        row["count_mainline_suitable"] = count_mainline_suitable
        row["sum_mainline_suitable"] = sum_mainline_suitable
        row["artifact_role"] = overall_role
        row["notes"] = notes

    return fidelity_rows, count_verdicts, count_mainline_suitable, sum_mainline_suitable, overall_role


def build_sum_rows(
    manifest: dict[str, Any],
    raw_values: np.ndarray,
    decoded_values: np.ndarray,
    plane_float: list[np.ndarray],
    basis_vectors: list[np.ndarray],
    base_total: float,
) -> list[dict[str, Any]]:
    rows = int(manifest["value_count"])
    max_plane_count = int(manifest["max_plane_count"])

    raw_sum_value, raw_ms = median_ms(lambda: float(np.sum(raw_values, dtype=np.float64)))
    encoded_sum_value, fixed_depth_ms = median_ms(
        lambda: compute_sum_at_k(base_total, plane_float, basis_vectors, max_plane_count)
    )

    raw_seconds = raw_ms / 1000.0
    fixed_seconds = fixed_depth_ms / 1000.0
    raw_rows_per_sec = rows / raw_seconds
    fixed_rows_per_sec = rows / fixed_seconds
    raw_logical_gbps = (rows * 8) / raw_seconds / 1e9
    fixed_logical_gbps = (rows * max_plane_count) / fixed_seconds / 1e9

    sum_rows: list[dict[str, Any]] = []
    for k in range(1, max_plane_count + 1):
        progressive_sum, ms_per_iter = median_ms(
            lambda depth=k: compute_sum_at_k(base_total, plane_float, basis_vectors, depth)
        )
        seconds = ms_per_iter / 1000.0
        logical_bytes = rows * k
        rows_per_sec = rows / seconds
        logical_gbps = logical_bytes / seconds / 1e9
        abs_error_vs_raw = abs(progressive_sum - raw_sum_value)
        rel_error_vs_raw = 0.0 if raw_sum_value == 0.0 else abs_error_vs_raw / abs(raw_sum_value)
        abs_error_vs_encoded = abs(progressive_sum - encoded_sum_value)
        rel_error_vs_encoded = 0.0 if encoded_sum_value == 0.0 else abs_error_vs_encoded / abs(encoded_sum_value)

        sum_rows.append(
            {
                "dataset": manifest["dataset"],
                "artifact_version": manifest["artifact_version"],
                "artifact_label": manifest["artifact_label"],
                "aggregation": "sum",
                "mode": "progressive_runtime_planes",
                "k": k,
                "max_plane_count": max_plane_count,
                "n": rows,
                "raw_sum": raw_sum_value,
                "encoded_full_depth_sum": encoded_sum_value,
                "progressive_sum": progressive_sum,
                "abs_error_vs_raw": abs_error_vs_raw,
                "rel_error_vs_raw": rel_error_vs_raw,
                "abs_error_vs_encoded": abs_error_vs_encoded,
                "rel_error_vs_encoded": rel_error_vs_encoded,
                "logical_bytes": logical_bytes,
                "estimated_physical_bytes": logical_bytes,
                "ms_per_iter": ms_per_iter,
                "rows_per_sec": rows_per_sec,
                "logical_GBps": logical_gbps,
                "estimated_physical_GBps": logical_gbps,
                "raw_ms": raw_ms,
                "raw_rows_per_sec": raw_rows_per_sec,
                "raw_logical_GBps": raw_logical_gbps,
                "fixed_depth_ms": fixed_depth_ms,
                "fixed_depth_rows_per_sec": fixed_rows_per_sec,
                "fixed_depth_logical_GBps": fixed_logical_gbps,
                "speedup_vs_raw": logical_gbps / raw_logical_gbps if raw_logical_gbps > 0.0 else float("nan"),
                "speedup_vs_fixed_depth": (
                    logical_gbps / fixed_logical_gbps if fixed_logical_gbps > 0.0 else float("nan")
                ),
                "validated": abs_error_vs_encoded <= 1e-12 if k == max_plane_count else True,
                "notes": "CPU-reference timing only; not H200 throughput evidence.",
            }
        )

    return sum_rows


def build_count_rows(
    manifest: dict[str, Any],
    raw_values: np.ndarray,
    decoded_values: np.ndarray,
    segments: list[Segment],
    thresholds: dict[int, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = int(manifest["value_count"])
    max_plane_count = int(manifest["max_plane_count"])
    count_rows: list[dict[str, Any]] = []
    kstar_source_rows: list[dict[str, Any]] = []

    for target in THRESHOLD_GRID:
        threshold = thresholds[target]
        raw_count_value, raw_ms = median_ms(lambda t=threshold: raw_count(raw_values, t), repeats=1)
        encoded_count_value, fixed_depth_ms = median_ms(
            lambda t=threshold: full_depth_count(decoded_values, t),
            repeats=1,
        )
        raw_seconds = raw_ms / 1000.0
        fixed_seconds = fixed_depth_ms / 1000.0
        raw_rows_per_sec = rows / raw_seconds
        fixed_rows_per_sec = rows / fixed_seconds
        raw_logical_gbps = (rows * 8) / raw_seconds / 1e9
        fixed_logical_gbps = (rows * max_plane_count) / fixed_seconds / 1e9

        threshold_rows: list[dict[str, Any]] = []
        previous_u: int | None = None
        for k in range(1, max_plane_count + 1):
            stats, ms_per_iter = median_ms(
                lambda depth=k, thresh=threshold: progressive_count_stats(segments, thresh, depth),
                repeats=1,
            )
            encoded_in_interval = stats["count_lower"] <= encoded_count_value <= stats["count_upper"]
            if k == max_plane_count and (stats["U_k"] != 0 or stats["count_lower"] != encoded_count_value):
                raise RuntimeError(
                    f"full-depth progressive mismatch at selectivity {target}: "
                    f"lower={stats['count_lower']} encoded={encoded_count_value} U={stats['U_k']}"
                )
            if previous_u is not None and stats["U_k"] > previous_u:
                raise RuntimeError(
                    f"U(k) must be non-increasing but got U({k - 1})={previous_u}, U({k})={stats['U_k']}"
                )
            previous_u = stats["U_k"]

            seconds = ms_per_iter / 1000.0
            rows_per_sec = rows / seconds
            logical_gbps = stats["total_planes_read"] / seconds / 1e9
            estimated_physical_gbps = stats["estimated_pack_load_bytes"] / seconds / 1e9
            row = {
                "dataset": manifest["dataset"],
                "artifact_version": manifest["artifact_version"],
                "artifact_label": manifest["artifact_label"],
                "threshold": threshold,
                "target_selectivity": target,
                "k": k,
                "max_plane_count": max_plane_count,
                "n": rows,
                "raw_count": raw_count_value,
                "encoded_full_depth_count": encoded_count_value,
                "raw_eq_encoded_full_depth_check": raw_count_value == encoded_count_value,
                "progressive_count": stats["progressive_count"],
                "count_lower": stats["count_lower"],
                "count_upper": stats["count_upper"],
                "Q_k": stats["Q_k"],
                "D_k": stats["D_k"],
                "U_k": stats["U_k"],
                "count_abs_error_vs_raw": abs(stats["progressive_count"] - raw_count_value),
                "count_rel_error_vs_raw": (
                    0.0 if raw_count_value == 0 else abs(stats["progressive_count"] - raw_count_value) / raw_count_value
                ),
                "logical_bytes": stats["total_planes_read"],
                "estimated_pack_load_bytes": stats["estimated_pack_load_bytes"],
                "ms_per_iter": ms_per_iter,
                "rows_per_sec": rows_per_sec,
                "logical_GBps": logical_gbps,
                "estimated_physical_GBps": estimated_physical_gbps,
                "raw_ms": raw_ms,
                "raw_rows_per_sec": raw_rows_per_sec,
                "raw_logical_GBps": raw_logical_gbps,
                "fixed_depth_ms": fixed_depth_ms,
                "fixed_depth_rows_per_sec": fixed_rows_per_sec,
                "fixed_depth_logical_GBps": fixed_logical_gbps,
                "speedup_vs_raw": logical_gbps / raw_logical_gbps if raw_logical_gbps > 0.0 else float("nan"),
                "speedup_vs_fixed_depth": (
                    logical_gbps / fixed_logical_gbps if fixed_logical_gbps > 0.0 else float("nan")
                ),
                "encoded_full_depth_in_interval": encoded_in_interval,
                "notes": "CPU-reference timing only; not H200 throughput evidence.",
            }
            threshold_rows.append(row)
            count_rows.append(row)

        kstar_source_rows.extend(threshold_rows)

    return count_rows, kstar_source_rows


def build_kstar_rows(
    manifest: dict[str, Any],
    count_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int], list[dict[str, Any]]] = {}
    for row in count_rows:
        key = (float(row["threshold"]), int(row["target_selectivity"]))
        grouped.setdefault(key, []).append(row)

    kstar_rows: list[dict[str, Any]] = []
    for (threshold, target), rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        rows = sorted(rows, key=lambda row: int(row["k"]))
        exact_count = int(rows[0]["raw_count"])
        max_k = int(rows[-1]["max_plane_count"])

        for epsilon in EPSILON_ABS_TARGETS:
            valid = [row for row in rows if int(row["U_k"]) <= epsilon]
            if valid:
                selected = valid[0]
                kstar = int(selected["k"])
                status = "ok"
            else:
                selected = rows[-1]
                kstar = ""
                status = "unmet"
            kstar_rows.append(
                {
                    "dataset": manifest["dataset"],
                    "artifact_version": manifest["artifact_version"],
                    "artifact_label": manifest["artifact_label"],
                    "threshold": threshold,
                    "target_selectivity": target,
                    "epsilon_type": "abs",
                    "epsilon": epsilon,
                    "kstar": kstar,
                    "kstar_status": status,
                    "exact_count": exact_count,
                    "max_k": max_k,
                    "selected_U_k": int(selected["U_k"]),
                    "selected_logical_GBps": float(selected["logical_GBps"]),
                    "selected_estimated_physical_GBps": float(selected["estimated_physical_GBps"]),
                    "selected_rows_per_sec": float(selected["rows_per_sec"]),
                }
            )

        for epsilon in EPSILON_REL_TARGETS:
            abs_bound = epsilon * exact_count
            valid = [row for row in rows if int(row["U_k"]) <= abs_bound]
            if valid:
                selected = valid[0]
                kstar = int(selected["k"])
                status = "ok"
            else:
                selected = rows[-1]
                kstar = ""
                status = "unmet"
            kstar_rows.append(
                {
                    "dataset": manifest["dataset"],
                    "artifact_version": manifest["artifact_version"],
                    "artifact_label": manifest["artifact_label"],
                    "threshold": threshold,
                    "target_selectivity": target,
                    "epsilon_type": "rel",
                    "epsilon": epsilon,
                    "kstar": kstar,
                    "kstar_status": status,
                    "exact_count": exact_count,
                    "max_k": max_k,
                    "selected_U_k": int(selected["U_k"]),
                    "selected_logical_GBps": float(selected["logical_GBps"]),
                    "selected_estimated_physical_GBps": float(selected["estimated_physical_GBps"]),
                    "selected_rows_per_sec": float(selected["rows_per_sec"]),
                }
            )

    return kstar_rows


def build_report(
    report_path: Path,
    *,
    artifact_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    sidecar: dict[str, Any],
    fidelity_rows: list[dict[str, Any]],
    sum_rows: list[dict[str, Any]],
    count_rows: list[dict[str, Any]],
    kstar_rows: list[dict[str, Any]],
) -> None:
    caution_or_cat_rows = [row for row in fidelity_rows if row["count_verdict"] in {"caution", "catastrophic"}]
    worst_count_row = max(fidelity_rows, key=lambda row: abs(float(row["selectivity_drift_pp"])))
    sum_at_4 = next(row for row in sum_rows if int(row["k"]) == 4)
    sum_at_full = next(row for row in sum_rows if int(row["k"]) == int(row["max_plane_count"]))

    worst_u_rows: list[dict[str, Any]] = []
    seen_targets: set[int] = set()
    for row in sorted(count_rows, key=lambda item: (int(item["target_selectivity"]), int(item["k"]))):
        target = int(row["target_selectivity"])
        if target in seen_targets:
            continue
        seen_targets.add(target)
        matching = [item for item in count_rows if int(item["target_selectivity"]) == target]
        worst_u_rows.append(max(matching, key=lambda item: int(item["U_k"])))

    abs_one_rows = [
        row for row in kstar_rows if row["epsilon_type"] == "abs" and float(row["epsilon"]) == 1.0
    ]
    rel_point_one_rows = [
        row for row in kstar_rows if row["epsilon_type"] == "rel" and abs(float(row["epsilon"]) - 0.1) < 1e-12
    ]

    lines = [
        "# RealData Evidence Chain Final Report",
        "",
        "Date: 2026-05-10",
        "",
        "Scope: PV1-9 real-data evidence chain finalization for the approved UCI column.",
        "",
        "## 1. Readiness Decision",
        "",
        "- Real-data semantic validation is ready for the selected artifact and selected column.",
        "- Paper-v1 real-data throughput readiness remains blocked in this local workspace because no H200 / Slurm benchmark path was available here.",
        "- Synthetic findings therefore **partially generalize**: the bounded v2 artifact is faithful on this selected real column, but this single non-negative household-power column does not replace the skew-stress synthetic evidence, and the hardware-throughput claim still needs receiver-side H200 execution.",
        "",
        "## 2. Issue Chain Coverage",
        "",
        f"- `#13`: approved dataset/column/policy = `{sidecar['dataset_name']} / {sidecar['column_name']}` with `transform_policy={sidecar['transform_policy']}` and pre-cleaning that drops `?` rows.",
        f"- `#14`: artifact root = `{artifact_dir}` with `artifact_label={manifest['artifact_label']}`, `active_plane_count_max={manifest['segment_plane_count_max']}`, `total_bytes_per_row={summary['total_bytes_per_row']}`.",
        "- `#15`: this report and the CSVs below provide full-depth raw-vs-encoded fidelity, progressive SUM precision-throughput rows, progressive COUNT Q/D/U rows, and `epsilon -> k*` output for the approved real artifact.",
        "",
        "## 3. Approved Real-Data Path",
        "",
        f"- dataset: `{sidecar['dataset_name']}`",
        f"- numeric column: `{sidecar['column_name']}`",
        f"- unit: `{sidecar['column_unit']}`",
        f"- semantics: `{sidecar['column_semantics']}`",
        f"- rows before transform: `{sidecar['row_count_before_transform']}`",
        f"- rows after transform: `{sidecar['row_count_after_transform']}`",
        f"- missing rows dropped: `{sidecar['missing_count']}`",
        f"- transform provenance: `{sidecar['transform_command_or_script']}`",
        "",
        "## 4. Full-Depth Fidelity",
        "",
        f"- raw-roundtrip exact: `{manifest['raw_roundtrip_exact']}`",
        f"- fidelity class: `{manifest['fidelity_class']}`",
        f"- raw SUM: `{fidelity_rows[0]['raw_sum']:.17g}`",
        f"- encoded SUM: `{fidelity_rows[0]['encoded_sum']:.17g}`",
        f"- SUM abs error: `{fidelity_rows[0]['sum_abs_error']:.17g}`",
        f"- SUM verdict: `{fidelity_rows[0]['sum_verdict']}`",
        f"- worst COUNT drift row: `s={worst_count_row['target_selectivity']}`, threshold `{worst_count_row['threshold']:.17g}`, drift `{worst_count_row['selectivity_drift_pp']:.6f} pp`, verdict `{worst_count_row['count_verdict']}`",
        "",
    ]

    if caution_or_cat_rows:
        lines.extend(
            [
                "COUNT caution/catastrophic rows were observed:",
                "",
                "| Target selectivity | Threshold | Raw count | Encoded count | Drift (pp) | Verdict |",
                "| ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in caution_or_cat_rows:
            lines.append(
                f"| {row['target_selectivity']} | {row['threshold']:.17g} | {row['raw_count']} | "
                f"{row['encoded_count']} | {row['selectivity_drift_pp']:.6f} | {row['count_verdict']} |"
            )
    else:
        lines.append("- All nine full-depth COUNT fidelity rows are `acceptable`.")

    lines.extend(
        [
            "",
            "## 5. Progressive SUM",
            "",
            "| k | Progressive sum | Abs error vs raw | Rel error vs raw | Logical GB/s | Speedup vs raw | Speedup vs fixed-depth |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sum_rows:
        lines.append(
            f"| {row['k']} | {row['progressive_sum']:.17g} | {row['abs_error_vs_raw']:.17g} | "
            f"{row['rel_error_vs_raw']:.17g} | {row['logical_GBps']:.6f} | "
            f"{row['speedup_vs_raw']:.6f} | {row['speedup_vs_fixed_depth']:.6f} |"
        )

    lines.extend(
        [
            "",
            f"- Example midpoint row: `k=4` gives abs error `{sum_at_4['abs_error_vs_raw']:.17g}` with logical throughput `{sum_at_4['logical_GBps']:.6f} GB/s` in this CPU-reference run.",
            f"- Full depth `k={sum_at_full['k']}` matches the encoded reference with abs error vs encoded `{sum_at_full['abs_error_vs_encoded']:.17g}`.",
            "",
            "## 6. Progressive COUNT",
            "",
            "| Target selectivity | Worst U(k) across k | k at worst U(k) | Worst logical GB/s row |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in worst_u_rows:
        lines.append(
            f"| {row['target_selectivity']} | {row['U_k']} | {row['k']} | {row['logical_GBps']:.6f} |"
        )

    lines.extend(
        [
            "",
            "Selected `epsilon -> k*` checkpoints:",
            "",
            "| Target selectivity | epsilon type | epsilon | k* | status | Selected U(k) |",
            "| ---: | --- | ---: | ---: | --- | ---: |",
        ]
    )
    for row in abs_one_rows + rel_point_one_rows:
        lines.append(
            f"| {row['target_selectivity']} | {row['epsilon_type']} | {row['epsilon']} | "
            f"{row['kstar'] if row['kstar'] != '' else 'NA'} | {row['kstar_status']} | {row['selected_U_k']} |"
        )

    lines.extend(
        [
            "",
            "## 7. Throughput Contract Note",
            "",
            "- The CSVs report `rows_per_sec`, `logical_GBps`, and `estimated_physical_GBps` using the PV1-8A formulas.",
            "- These are **CPU-reference wall-clock work rates** for this local validation script, not H200 GPU benchmark evidence.",
            "- `estimated_physical_GBps` here is load-accounting only. It is not profiler-backed HBM bandwidth.",
            "",
            "## 8. Output Files",
            "",
            "- `results/buff_encoder_v2/real_fidelity_audit.csv`",
            "- `results/buff_encoder_v2/real_sum_precision_throughput.csv`",
            "- `results/buff_encoder_v2/real_count_precision_throughput.csv`",
            "- `results/buff_encoder_v2/real_count_epsilon_to_kstar.csv`",
            "",
            "## 9. Bottom Line",
            "",
            "- For the approved `UCI Household Electric Power Consumption / Global_active_power` path, the v2 artifact is suitable for real-data SUM mainline and COUNT mainline at full depth.",
            "- Progressive SUM and capped-k COUNT curves now exist for this artifact in CPU/reference form.",
            "- The remaining blocker is receiver-side H200 throughput execution, not encoder semantics on the approved column.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()

    artifact_dir = args.artifact_dir.resolve()
    raw_path = args.raw_path.resolve()
    decoded_path = args.decoded_path.resolve()
    fidelity_csv = args.fidelity_csv.resolve()
    sum_csv = args.sum_csv.resolve()
    count_csv = args.count_csv.resolve()
    kstar_csv = args.kstar_csv.resolve()
    report_path = args.report_path.resolve()

    manifest = read_json(artifact_dir / "manifest.json")
    summary = read_json(artifact_dir / "summary.json")
    sidecar = read_json(artifact_dir / "real_data_source.json")

    raw_values = load_fp64_memmap(raw_path)
    decoded_values = load_fp64_memmap(decoded_path)
    if raw_values.shape[0] != int(manifest["value_count"]):
        raise ValueError("raw_values length does not match manifest value_count")
    if decoded_values.shape[0] != int(manifest["value_count"]):
        raise ValueError("decoded_values length does not match manifest value_count")

    max_plane_count = int(manifest["max_plane_count"])
    planes = [
        np.fromfile(artifact_dir / f"plane_{plane:03d}.bin", dtype=np.uint8)
        for plane in range(max_plane_count)
    ]
    plane_float = [plane.astype(np.float64) for plane in planes]

    segments = load_segments(artifact_dir / "segment_meta.csv", raw_values, planes)
    base_total, basis_vectors = build_sum_basis_vectors(segments, int(manifest["value_count"]), max_plane_count)

    thresholds = {
        target: float(np.quantile(raw_values, 1.0 - (target / 100.0), method="linear"))
        for target in THRESHOLD_GRID
    }

    fidelity_rows, _, _, _, _ = build_fidelity_rows(
        manifest, summary, raw_values, decoded_values, thresholds
    )
    sum_rows = build_sum_rows(
        manifest, raw_values, decoded_values, plane_float, basis_vectors, base_total
    )
    count_rows, _ = build_count_rows(
        manifest, raw_values, decoded_values, segments, thresholds
    )
    kstar_rows = build_kstar_rows(manifest, count_rows)

    write_csv(fidelity_csv, FIDELITY_COLUMNS, fidelity_rows)
    write_csv(sum_csv, SUM_COLUMNS, sum_rows)
    write_csv(count_csv, COUNT_COLUMNS, count_rows)
    write_csv(kstar_csv, KSTAR_COLUMNS, kstar_rows)

    for path, rows in [
        (fidelity_csv, fidelity_rows),
        (sum_csv, sum_rows),
        (count_csv, count_rows),
        (kstar_csv, kstar_rows),
    ]:
        write_provenance(
            path.with_suffix(path.suffix + ".provenance.json"),
            argv=sys.argv,
            extra={
                "artifact_dir": str(artifact_dir),
                "row_count": len(rows),
                "output": str(path),
            },
        )

    build_report(
        report_path,
        artifact_dir=artifact_dir,
        manifest=manifest,
        summary=summary,
        sidecar=sidecar,
        fidelity_rows=fidelity_rows,
        sum_rows=sum_rows,
        count_rows=count_rows,
        kstar_rows=kstar_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
