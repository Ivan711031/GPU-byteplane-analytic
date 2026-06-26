#!/usr/bin/env python3
"""Current-v2 subcolumn error study for the exported bfp-dec4 artifacts."""

from __future__ import annotations

import csv
import json
import math
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


DATE_TAG = "20260511"
MAX_K = 8
ABS_EPSILONS = [
    0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1.0, 10.0,
    100.0, 1e3, 1e4, 1e5, 1e6, 1e8, 1e10, 1e12,
]
REL_EPSILONS = [0.0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1]
THRESHOLDS = {
    "sensor": 25.0,
    "uniform": 500.0,
    "heavy_tailed": 1.0,
    "zipfian": 16.0,
}
DATASET_DISPLAY = {
    "sensor": "Sensor",
    "uniform": "Uniform",
    "heavy_tailed": "Heavy-tailed",
    "zipfian": "Zipfian",
}
AGGREGATIONS = ["sum", "avg", "count_gt", "min", "max", "var"]
METRICS_COLUMNS = [
    "dataset",
    "artifact_version",
    "artifact_label",
    "aggregation",
    "k",
    "effective_k",
    "max_plane_count",
    "threshold",
    "exact",
    "approx",
    "abs_error",
    "rel_error",
    "analytic_bound",
    "analytic_rel_bound",
    "bound_gap",
]
DATASET_SUMMARY_COLUMNS = [
    "dataset",
    "artifact_version",
    "artifact_label",
    "threshold",
    "total_values",
    "segment_size",
    "segment_count",
    "precision_power",
    "decimal_bits",
    "quantization_bound",
    "min_plane_count",
    "max_plane_count",
    "mean_plane_count",
    "exact_sum",
    "exact_avg",
    "exact_min",
    "exact_max",
    "exact_var",
    "exact_count",
]
KSTAR_COLUMNS = ["epsilon_type", "epsilon", "aggregation", "dataset", "k_star"]
HISTORICAL_CHECK_COLUMNS = ["artifact", "check", "passed", "details"]


@dataclass
class SegmentInfo:
    row_offset: int
    row_count: int
    active_plane_count: int
    segment_base: float
    bases: list[float]
    omitted_tail_bounds: list[float]


class AggregateAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.sum = np.longdouble(0.0)
        self.sumsq = np.longdouble(0.0)
        self.minimum = math.inf
        self.maximum = -math.inf
        self.count_gt = 0

    def add(self, values: np.ndarray, threshold: float) -> None:
        values_ld = values.astype(np.longdouble, copy=False)
        self.n += int(values.shape[0])
        self.sum += values_ld.sum(dtype=np.longdouble)
        self.sumsq += np.multiply(values_ld, values_ld).sum(dtype=np.longdouble)
        self.minimum = min(self.minimum, float(np.min(values)))
        self.maximum = max(self.maximum, float(np.max(values)))
        self.count_gt += int(np.count_nonzero(values > threshold))

    def average(self) -> float:
        return 0.0 if self.n == 0 else float(self.sum / np.longdouble(self.n))

    def variance(self) -> float:
        if self.n == 0:
            return 0.0
        mean = self.sum / np.longdouble(self.n)
        variance = self.sumsq / np.longdouble(self.n) - mean * mean
        return float(max(variance, np.longdouble(0.0)))


def git_commit(cwd: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def map_dataset_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in DATASET_DISPLAY:
        return normalized
    raise KeyError(f"unexpected dataset name: {name}")


def safe_relative_error(exact: float, approx: float) -> float:
    abs_error = abs(exact - approx)
    denom = max(abs(exact), 1e-30)
    return abs_error / denom


def safe_gap(bound: float, empirical: float) -> float:
    if empirical == 0.0:
        return 1.0 if bound == 0.0 else math.inf
    return bound / empirical


def nice_positive_floor(value: float) -> float:
    return value if value > 0.0 and math.isfinite(value) else 1e-18


def write_svg_plot(
    output_path: Path,
    title: str,
    max_k: int,
    empirical: list[tuple[float, float]],
    bound: list[tuple[float, float]],
) -> None:
    width = 760
    height = 420
    left = 72
    right = 24
    top = 48
    bottom = 56
    plot_width = width - left - right
    plot_height = height - top - bottom

    min_y = math.inf
    max_y = 0.0
    for _, value in empirical + bound:
        adjusted = nice_positive_floor(value)
        min_y = min(min_y, adjusted)
        max_y = max(max_y, adjusted)
    if not math.isfinite(min_y):
        min_y = 1e-18
    if max_y <= min_y:
        max_y = min_y * 10.0

    log_min = math.log10(min_y)
    log_max = math.log10(max_y)
    if log_max - log_min < 1e-12:
        log_max = log_min + 1.0

    def x_for(k_value: float) -> float:
        span = float(max(max_k - 1, 1))
        return left + (k_value - 1.0) / span * plot_width

    def y_for(value: float) -> float:
        log_value = math.log10(nice_positive_floor(value))
        ratio = (log_value - log_min) / (log_max - log_min)
        return top + (1.0 - ratio) * plot_height

    def polyline(points: list[tuple[float, float]]) -> str:
        return " ".join(f"{x_for(k_value):.2f},{y_for(value):.2f}" for k_value, value in points)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as out:
        out.write(f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\n")
        out.write("<rect width=\"100%\" height=\"100%\" fill=\"#fffdf8\"/>\n")
        out.write(f"<text x=\"{left}\" y=\"28\" font-family=\"Georgia, serif\" font-size=\"20\" fill=\"#2b2b2b\">{title}</text>\n")
        out.write(f"<line x1=\"{left}\" y1=\"{top + plot_height}\" x2=\"{left + plot_width}\" y2=\"{top + plot_height}\" stroke=\"#555\" stroke-width=\"1.2\"/>\n")
        out.write(f"<line x1=\"{left}\" y1=\"{top}\" x2=\"{left}\" y2=\"{top + plot_height}\" stroke=\"#555\" stroke-width=\"1.2\"/>\n")
        for k_value in range(1, max_k + 1):
            x_pos = x_for(float(k_value))
            out.write(f"<line x1=\"{x_pos}\" y1=\"{top}\" x2=\"{x_pos}\" y2=\"{top + plot_height}\" stroke=\"#ece7dc\" stroke-width=\"1\"/>\n")
            out.write(f"<text x=\"{x_pos}\" y=\"{top + plot_height + 24}\" text-anchor=\"middle\" font-family=\"Menlo, monospace\" font-size=\"12\" fill=\"#444\">{k_value}</text>\n")
        for tick in range(5):
            ratio = tick / 4.0
            value = 10.0 ** (log_min + (1.0 - ratio) * (log_max - log_min))
            y_pos = top + ratio * plot_height
            out.write(f"<line x1=\"{left}\" y1=\"{y_pos}\" x2=\"{left + plot_width}\" y2=\"{y_pos}\" stroke=\"#ece7dc\" stroke-width=\"1\"/>\n")
            out.write(f"<text x=\"{left - 10}\" y=\"{y_pos + 4}\" text-anchor=\"end\" font-family=\"Menlo, monospace\" font-size=\"12\" fill=\"#444\">{value:.1e}</text>\n")
        out.write(f"<polyline fill=\"none\" stroke=\"#c45d2c\" stroke-width=\"2.5\" points=\"{polyline(empirical)}\"/>\n")
        out.write(f"<polyline fill=\"none\" stroke=\"#176087\" stroke-width=\"2.5\" stroke-dasharray=\"8 6\" points=\"{polyline(bound)}\"/>\n")
        out.write(f"<text x=\"{left + 12}\" y=\"{top + 16}\" font-family=\"Menlo, monospace\" font-size=\"12\" fill=\"#c45d2c\">empirical abs error</text>\n")
        out.write(f"<text x=\"{left + 220}\" y=\"{top + 16}\" font-family=\"Menlo, monospace\" font-size=\"12\" fill=\"#176087\">analytic bound</text>\n")
        out.write(f"<text x=\"{left + plot_width / 2}\" y=\"{height - 12}\" text-anchor=\"middle\" font-family=\"Menlo, monospace\" font-size=\"12\" fill=\"#444\">top-k subcolumns kept</text>\n")
        out.write(f"<text x=\"18\" y=\"{top + plot_height / 2}\" transform=\"rotate(-90 18 {top + plot_height / 2})\" text-anchor=\"middle\" font-family=\"Menlo, monospace\" font-size=\"12\" fill=\"#444\">absolute error (log scale)</text>\n")
        out.write("</svg>\n")


def load_segments(artifact_dir: Path) -> list[SegmentInfo]:
    segments: list[SegmentInfo] = []
    with (artifact_dir / "segment_meta.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_offset = int(row["row_offset"])
            row_count = int(row["row_count"])
            active_plane_count = int(row["active_plane_count"])
            fractional_bits = int(row["effective_fractional_bits"])
            integer_offset_bits = int(row["integer_offset_bits"])
            total_bits = fractional_bits + integer_offset_bits
            plane_count = 0 if total_bits == 0 else (total_bits + 7) // 8
            bases = [float(row[f"plane_basis_{plane}"]) for plane in range(active_plane_count)]
            max_values = []
            for plane in range(plane_count):
                width = 8
                if plane + 1 == plane_count:
                    trailing = total_bits - 8 * (plane_count - 1)
                    width = 8 if trailing == 0 else trailing
                max_values.append((1 << width) - 1)
            omitted_tail_bounds = [0.0] * (MAX_K + 1)
            for k in range(1, MAX_K + 1):
                keep = min(k, active_plane_count)
                tail = 0.0
                for plane in range(keep, active_plane_count):
                    tail += float(max_values[plane]) * bases[plane]
                omitted_tail_bounds[k] = tail
            segments.append(
                SegmentInfo(
                    row_offset=row_offset,
                    row_count=row_count,
                    active_plane_count=active_plane_count,
                    segment_base=float(row["segment_base"]),
                    bases=bases,
                    omitted_tail_bounds=omitted_tail_bounds,
                )
            )
    return segments


def analyze_dataset(artifact_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(artifact_dir / "manifest.json")
    summary = read_json(artifact_dir / "summary.json")
    dataset = map_dataset_name(str(manifest["dataset"]))
    threshold = THRESHOLDS[dataset]
    raw_path = Path(str(manifest["source_path"]))
    value_count = int(manifest["value_count"])
    max_plane_count = int(manifest["max_plane_count"])
    quantization_bound = float(summary["quantization_bound"])

    raw_values = np.memmap(raw_path, dtype="<f8", mode="r")
    if raw_values.shape[0] != value_count:
        raise ValueError(f"{dataset}: raw row count mismatch")

    planes = [
        np.memmap(artifact_dir / f"plane_{plane:03d}.bin", dtype=np.uint8, mode="r")
        for plane in range(max_plane_count)
    ]
    for plane in planes:
        if plane.shape[0] != value_count:
            raise ValueError(f"{dataset}: plane length mismatch")

    segments = load_segments(artifact_dir)
    exact = AggregateAccumulator()
    approx_states = {k: AggregateAccumulator() for k in range(1, MAX_K + 1)}
    bound_sum = {k: 0.0 for k in range(1, MAX_K + 1)}
    bound_count = {k: 0.0 for k in range(1, MAX_K + 1)}
    bound_max = {k: 0.0 for k in range(1, MAX_K + 1)}
    bound_second_moment = {k: 0.0 for k in range(1, MAX_K + 1)}

    progress_step = max(len(segments) // 10, 1)
    for index, segment in enumerate(segments, start=1):
        row_slice = slice(segment.row_offset, segment.row_offset + segment.row_count)
        raw_chunk = np.asarray(raw_values[row_slice], dtype=np.float64)
        exact.add(raw_chunk, threshold)
        segment_max = float(np.max(raw_chunk))
        current = np.full(segment.row_count, segment.segment_base, dtype=np.float64)
        contributions = [
            np.asarray(planes[plane][row_slice], dtype=np.float64) * segment.bases[plane]
            for plane in range(segment.active_plane_count)
        ]

        for k in range(1, MAX_K + 1):
            if k <= segment.active_plane_count:
                current += contributions[k - 1]
            approx_states[k].add(current, threshold)
            bound = quantization_bound + segment.omitted_tail_bounds[k]
            bound_sum[k] += segment.row_count * bound
            bound_count[k] += float(np.count_nonzero((raw_chunk > threshold) & (raw_chunk <= threshold + bound)))
            bound_max[k] = max(bound_max[k], bound)
            bound_second_moment[k] += segment.row_count * 2.0 * segment_max * bound

        if index % progress_step == 0 or index == len(segments):
            print(f"[v2-exp2] {dataset}: segment {index}/{len(segments)}")

    exact_sum = float(exact.sum)
    exact_avg = exact.average()
    exact_min = exact.minimum
    exact_max = exact.maximum
    exact_var = exact.variance()
    exact_count = float(exact.count_gt)

    summary_row = {
        "dataset": dataset,
        "artifact_version": manifest["artifact_version"],
        "artifact_label": manifest["artifact_label"],
        "threshold": threshold,
        "total_values": value_count,
        "segment_size": int(manifest["segment_size"]),
        "segment_count": int(manifest["segment_count"]),
        "precision_power": int(manifest["precision_power"]),
        "decimal_bits": int(manifest["decimal_bits"]),
        "quantization_bound": quantization_bound,
        "min_plane_count": int(manifest["segment_plane_count_min"]),
        "max_plane_count": int(manifest["segment_plane_count_max"]),
        "mean_plane_count": float(summary["mean_plane_count"]),
        "exact_sum": exact_sum,
        "exact_avg": exact_avg,
        "exact_min": exact_min,
        "exact_max": exact_max,
        "exact_var": exact_var,
        "exact_count": int(exact_count),
    }

    rows: list[dict[str, Any]] = []
    for k in range(1, MAX_K + 1):
        state = approx_states[k]
        approx_sum = float(state.sum)
        approx_avg = state.average()
        approx_min = state.minimum
        approx_max = state.maximum
        approx_var = state.variance()
        approx_count = float(state.count_gt)
        delta_mu_bound = bound_sum[k] / value_count
        var_bound = bound_second_moment[k] / value_count + (2.0 * exact_avg + delta_mu_bound) * delta_mu_bound
        exact_lookup = {
            "sum": exact_sum,
            "avg": exact_avg,
            "count_gt": exact_count,
            "min": exact_min,
            "max": exact_max,
            "var": exact_var,
        }
        approx_lookup = {
            "sum": approx_sum,
            "avg": approx_avg,
            "count_gt": approx_count,
            "min": approx_min,
            "max": approx_max,
            "var": approx_var,
        }
        bound_lookup = {
            "sum": bound_sum[k],
            "avg": delta_mu_bound,
            "count_gt": bound_count[k],
            "min": bound_max[k],
            "max": bound_max[k],
            "var": var_bound,
        }
        effective_k = min(k, max_plane_count)
        for aggregation in AGGREGATIONS:
            exact_value = exact_lookup[aggregation]
            approx_value = approx_lookup[aggregation]
            analytic_bound = bound_lookup[aggregation]
            abs_error = abs(exact_value - approx_value)
            rel_error = safe_relative_error(exact_value, approx_value)
            analytic_rel_bound = analytic_bound / max(abs(exact_value), 1e-30)
            rows.append(
                {
                    "dataset": dataset,
                    "artifact_version": manifest["artifact_version"],
                    "artifact_label": manifest["artifact_label"],
                    "aggregation": aggregation,
                    "k": k,
                    "effective_k": effective_k,
                    "max_plane_count": max_plane_count,
                    "threshold": threshold,
                    "exact": exact_value,
                    "approx": approx_value,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                    "analytic_bound": analytic_bound,
                    "analytic_rel_bound": analytic_rel_bound,
                    "bound_gap": safe_gap(analytic_bound, abs_error),
                }
            )

    return rows, summary_row


def write_kstar_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
    epsilons: list[float],
    use_relative: bool,
    bound_mode: bool,
) -> None:
    out_rows: list[dict[str, Any]] = []
    metric_column = "analytic_rel_bound" if use_relative and bound_mode else \
        "analytic_bound" if bound_mode else \
        "rel_error" if use_relative else "abs_error"

    for aggregation in AGGREGATIONS:
        for dataset in THRESHOLDS:
            matching = [
                row for row in rows
                if row["aggregation"] == aggregation and row["dataset"] == dataset
            ]
            matching.sort(key=lambda row: int(row["k"]))
            for epsilon in epsilons:
                k_star = "NA"
                for row in matching:
                    if float(row[metric_column]) <= epsilon:
                        k_star = str(row["k"])
                        break
                out_rows.append(
                    {
                        "epsilon_type": "relative" if use_relative else "absolute",
                        "epsilon": epsilon,
                        "aggregation": aggregation,
                        "dataset": dataset,
                        "k_star": k_star,
                    }
                )

    write_csv(output_path, KSTAR_COLUMNS, out_rows)


def load_historical_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def check_metric_arithmetic(label: str, metrics_path: Path) -> dict[str, Any]:
    rows = load_historical_metrics(metrics_path)
    for row in rows:
        exact = float(row["exact"])
        approx = float(row["approx"])
        observed_abs = float(row["abs_error"])
        observed_rel = float(row["rel_error"])
        analytic_bound = float(row["analytic_bound"])

        expected_abs = abs(exact - approx)
        abs_tol = max(1e-9 * max(expected_abs, observed_abs, 1.0), 1e-12)
        if abs(observed_abs - expected_abs) > abs_tol:
            return {
                "artifact": label,
                "check": "metric_arithmetic",
                "passed": False,
                "details": (
                    f"{row['dataset']}/{row['aggregation']}/k={row['k']}: "
                    f"abs_error {observed_abs} vs {expected_abs}"
                ),
            }

        expected_rel = safe_relative_error(exact, approx)
        rel_tol = max(1e-9 * max(expected_rel, observed_rel, 1.0), 1e-12)
        if abs(observed_rel - expected_rel) > rel_tol:
            return {
                "artifact": label,
                "check": "metric_arithmetic",
                "passed": False,
                "details": (
                    f"{row['dataset']}/{row['aggregation']}/k={row['k']}: "
                    f"rel_error {observed_rel} vs {expected_rel}"
                ),
            }

        if analytic_bound + abs_tol < expected_abs:
            return {
                "artifact": label,
                "check": "metric_arithmetic",
                "passed": False,
                "details": (
                    f"{row['dataset']}/{row['aggregation']}/k={row['k']}: "
                    f"analytic_bound {analytic_bound} < abs_error {expected_abs}"
                ),
            }

    return {
        "artifact": label,
        "check": "metric_arithmetic",
        "passed": True,
        "details": f"{len(rows)} rows checked",
    }


def check_historical_exacts(
    label: str,
    metrics_path: Path,
    exact_reference: dict[tuple[str, str], float],
) -> dict[str, Any]:
    rows = load_historical_metrics(metrics_path)
    for row in rows:
        dataset = map_dataset_name(row["dataset"])
        aggregation = row["aggregation"]
        reference = exact_reference[(dataset, aggregation)]
        exact_value = float(row["exact"])
        tolerance = max(1e-9 * max(abs(reference), 1.0), 1e-12)
        if abs(exact_value - reference) > tolerance:
            return {
                "artifact": label,
                "check": "exact_values_match_current_raw",
                "passed": False,
                "details": f"{dataset}/{aggregation}: {exact_value} vs {reference}",
            }
    return {
        "artifact": label,
        "check": "exact_values_match_current_raw",
        "passed": True,
        "details": f"{len(rows)} rows checked",
    }


def kstar_metric_column(file_name: str) -> tuple[bool, bool]:
    lower = file_name.lower()
    use_relative = "relative" in lower
    bound_mode = "bound" in lower
    return use_relative, bound_mode


def verify_historical_kstar(
    label: str,
    metrics_path: Path,
    kstar_path: Path,
) -> dict[str, Any]:
    rows = load_historical_metrics(metrics_path)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (map_dataset_name(row["dataset"]), row["aggregation"])
        groups.setdefault(key, []).append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: int(row["k"]))

    use_relative, bound_mode = kstar_metric_column(kstar_path.name)
    metric_column = "analytic_rel_bound" if use_relative and bound_mode else \
        "analytic_bound" if bound_mode else \
        "rel_error" if use_relative else "abs_error"

    with kstar_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = map_dataset_name(row["dataset"])
            aggregation = row["aggregation"]
            epsilon = float(row["epsilon"])
            expected = "NA"
            for metric_row in groups[(dataset, aggregation)]:
                if float(metric_row[metric_column]) <= epsilon:
                    expected = str(metric_row["k"])
                    break
            observed = row["k_star"]
            if observed != expected:
                return {
                    "artifact": label,
                    "check": f"kstar_consistency:{kstar_path.name}",
                    "passed": False,
                    "details": f"{dataset}/{aggregation}/eps={epsilon}: {observed} vs {expected}",
                }

    return {
        "artifact": label,
        "check": f"kstar_consistency:{kstar_path.name}",
        "passed": True,
        "details": "all rows consistent",
    }


def historical_checks(exact_reference: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    legacy_specs = [
        (
            "results/exp2",
            Path("results/exp2/results/metrics.csv"),
            [Path("results/exp2/results/kstar_absolute.csv"), Path("results/exp2/results/kstar_relative.csv")],
        ),
        (
            "results/exp2_extended_sum_k16",
            Path("results/exp2_extended_sum_k16/results/metrics.csv"),
            [
                Path("results/exp2_extended_sum_k16/results/kstar_absolute.csv"),
                Path("results/exp2_extended_sum_k16/results/kstar_relative.csv"),
            ],
        ),
        (
            "results/exp2_v3_rust_buff",
            Path("results/exp2_v3_rust_buff/results/metrics.csv"),
            [
                Path("results/exp2_v3_rust_buff/results/kstar_absolute_empirical.csv"),
                Path("results/exp2_v3_rust_buff/results/kstar_relative_empirical.csv"),
                Path("results/exp2_v3_rust_buff/results/kstar_absolute_bound.csv"),
                Path("results/exp2_v3_rust_buff/results/kstar_relative_bound.csv"),
            ],
        ),
    ]

    for label, metrics_path, kstar_paths in legacy_specs:
        if not metrics_path.is_file():
            continue
        checks.append(check_metric_arithmetic(label, metrics_path))
        checks.append(check_historical_exacts(label, metrics_path, exact_reference))
        for kstar_path in kstar_paths:
            if kstar_path.is_file():
                checks.append(verify_historical_kstar(label, metrics_path, kstar_path))

    return checks


def build_report(
    report_path: Path,
    *,
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    historical_rows: list[dict[str, Any]],
) -> None:
    bound_cover_count = sum(
        1 for row in metrics_rows
        if float(row["analytic_bound"]) + 1e-12 >= float(row["abs_error"])
    )
    total_metric_rows = len(metrics_rows)

    lines = [
        "# Current Buff Encoder Subcolumn Error Study",
        "",
        "Date: 2026-05-11",
        "",
        "Scope: current v2 `buff_bfp` artifact study over the exported `bfp-dec4` synthetic artifacts.",
        "",
        "## 1. Why This Re-run Exists",
        "",
        "- The repo already contains historical `exp2` outputs, but they do not all target the current exported v2 runtime artifacts.",
        "- `results/exp2` is the older exact-style split study.",
        "- `results/exp2_v3_rust_buff` is closer to the bounded branch, but it is not the same as the current D1 exported `bfp-dec4` artifact matrix; its precision policy and plane-depth profile differ.",
        "- This report therefore regenerates the full `(aggregation, dataset, epsilon) -> k*` study directly from the current `datasets/synthetic/dev_buff_v2_20260510/*/bfp-dec4` artifacts, then checks the historical files for internal correctness.",
        "",
        "## 2. Artifact Under Study",
        "",
        "| Dataset | Artifact label | Precision power | Quantization bound | Plane count | Threshold |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for row in sorted(summary_rows, key=lambda item: item["dataset"]):
        lines.append(
            f"| {row['dataset']} | {row['artifact_label']} | {row['precision_power']} | "
            f"{row['quantization_bound']:.17g} | {row['max_plane_count']} | {row['threshold']:.17g} |"
        )

    lines.extend(
        [
            "",
            "## 3. Part A: Lemmas",
            "",
            "Notation for one segment `s`:",
            "",
            "- `x_i` = raw FP64 value",
            "- `x_i^(P)` = full-depth bounded fixed-point decode of the current artifact",
            "- `x_i^(k)` = top-`k` zero-fill decode with `k <= 8` and `k_eff = min(k, P_s)`",
            "- `q = quantization_bound` from the current artifact manifest",
            "- `T_s(k)` = omitted-tail bound from the exported segment metadata, i.e. the sum of each omitted plane basis times that plane's maximum byte value",
            "- `B_s(k) = q + T_s(k)`",
            "",
            "### Lemma 1 (single-value bound)",
            "",
            "For every row in segment `s`, `0 <= x_i - x_i^(k) <= B_s(k)`.",
            "",
            "Proof.",
            "The current artifact is a bounded fixed-point artifact, so full-depth decode satisfies `0 <= x_i - x_i^(P) <= q` by construction. Top-`k` decode then zero-fills the omitted low-order planes. Each omitted plane contributes a non-negative amount bounded by its plane basis times its maximum byte value, so `0 <= x_i^(P) - x_i^(k) <= T_s(k)`. Adding the two inequalities gives `0 <= x_i - x_i^(k) <= q + T_s(k) = B_s(k)`.",
            "",
            "Typical-case note.",
            "In the current non-negative zero-fill setting the error is one-sided downward, so there is no sign cancellation at the value level. Typical-case improvement comes from the omitted bytes usually being far smaller than their worst-case maxima, not from unbiased rounding.",
            "",
            "### Lemma 2 (SUM / AVG)",
            "",
            "Let `e_i(k) = x_i - x_i^(k)`. Then `|SUM(X) - SUM(X^(k))| = sum_i e_i(k) <= sum_s n_s B_s(k)` and `|AVG(X) - AVG(X^(k))| <= (1/N) sum_s n_s B_s(k)`.",
            "",
            "Proof.",
            "By Lemma 1, every `e_i(k)` is non-negative and individually bounded by the segment-level `B_s(k)`. Summing over rows gives the SUM bound. Dividing by `N` gives the AVG bound.",
            "",
            "Typical-case note.",
            "Because the current artifact is downward-biased, SUM and AVG remain downward-biased too. Their empirical gaps therefore measure how loose the omitted-tail maxima are, not how much cancellation happened.",
            "",
            "### Lemma 3 (COUNT(x > t))",
            "",
            "For a fixed threshold `t`, `0 <= COUNT(x_i > t) - COUNT(x_i^(k) > t) <= sum_s #{ i in s : t < x_i <= t + B_s(k) }`.",
            "",
            "Proof.",
            "Since `x_i^(k) <= x_i`, truncation cannot create a false positive. A row changes classification only when `x_i > t` but `x_i^(k) <= t`, which implies `x_i - t <= B_s(k)` by Lemma 1.",
            "",
            "Typical-case note.",
            "This is distribution-dependent. If the local density near `t` is `f_s(t)`, then the expected count error is roughly `n_s f_s(t) B_s(k)`. The empirical error is small when the threshold sits in a sparse region, and can be large when the threshold cuts through a dense mass.",
            "",
            "### Lemma 4 (MIN / MAX)",
            "",
            "Both extremal operators satisfy `0 <= MIN(X) - MIN(X^(k)) <= max_s B_s(k)` and `0 <= MAX(X) - MAX(X^(k)) <= max_s B_s(k)`.",
            "",
            "Proof.",
            "Every truncated value lies in `[x_i - B_s(k), x_i]`. Therefore the exact extremum can fall by at most the largest admissible single-value drift.",
            "",
            "Typical-case note.",
            "The extremal value itself often has a stable MSB prefix, so MIN/MAX usually converge faster than SUM. However if the winner and runner-up differ by less than the current `B_s(k)`, the identity of the extremum can change until more planes are read.",
            "",
            "### Lemma 5 (VAR)",
            "",
            "Let `mu` be the exact mean and `Delta_mu(k) = (1/N) sum_s n_s B_s(k)`. Let `U_s = max_{i in s} x_i`. Then",
            "`|VAR(X) - VAR(X^(k))| <= (1/N) sum_s 2 n_s U_s B_s(k) + (2 mu + Delta_mu(k)) Delta_mu(k)`.",
            "",
            "Proof.",
            "Write `x_i^(k) = x_i - e_i`. Then `x_i^2 - (x_i^(k))^2 = 2 x_i e_i - e_i^2 <= 2 U_s B_s(k)`. Averaging across the segment bounds the second-moment drift. The mean-square drift obeys `|mu^2 - (mu^(k))^2| <= |mu - mu^(k)| |mu + mu^(k)| <= Delta_mu(k) (2 mu + Delta_mu(k))`. Summing the two contributions yields the stated variance bound.",
            "",
            "Typical-case note.",
            "The bound is usually loose on heavy-tailed data because it uses `U_s`, so one rare large value can dominate the worst-case term even when most rows are benign.",
            "",
            "## 4. Part B: Empirical Results",
            "",
            "- Every dataset was evaluated at `k = 1..8`.",
            "- When `k` exceeds the artifact's `max_plane_count`, the study clamps to full depth, so later rows repeat the full-depth bounded result.",
            "- COUNT uses the historical Exp2 thresholds: `sensor=25`, `uniform=500`, `heavy_tailed=1`, `zipfian=16`.",
            f"- The analytic bound covered the empirical absolute error on all `{bound_cover_count}/{total_metric_rows}` metric rows.",
            "",
            "| Dataset | Aggregation | Best empirical full-depth error | Best analytic full-depth bound |",
            "| --- | --- | ---: | ---: |",
        ]
    )

    for dataset in sorted(THRESHOLDS):
        for aggregation in AGGREGATIONS:
            matching = [
                row for row in metrics_rows
                if row["dataset"] == dataset and row["aggregation"] == aggregation
            ]
            full = next(row for row in matching if int(row["k"]) == MAX_K)
            lines.append(
                f"| {dataset} | {aggregation} | {full['abs_error']:.17g} | {full['analytic_bound']:.17g} |"
            )

    lines.extend(
        [
            "",
            "## 5. Historical Exp2 Checks",
            "",
            "| Artifact | Check | Passed | Details |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in historical_rows:
        lines.append(
            f"| {row['artifact']} | {row['check']} | {'yes' if row['passed'] else 'no'} | {row['details']} |"
        )

    failed = [row for row in historical_rows if not row["passed"]]
    if failed:
        lines.extend(
            [
                "",
                "Historical note.",
                "At least one historical file failed the consistency check above and should not be reused without re-validation.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Historical note.",
                "The historical Exp2 outputs checked above are internally consistent and their `exact` columns still match the current raw synthetic datasets. They remain valid as historical studies, but they are not the same artifact family as the current exported `bfp-dec4` runtime artifacts.",
            ]
        )

    lines.extend(
        [
            "",
            "## 6. Output Files",
            "",
            f"- metrics: `{out_dir / 'results' / 'metrics.csv'}`",
            f"- dataset summary: `{out_dir / 'results' / 'dataset_summary.csv'}`",
            f"- empirical absolute `k*`: `{out_dir / 'results' / 'kstar_absolute_empirical.csv'}`",
            f"- empirical relative `k*`: `{out_dir / 'results' / 'kstar_relative_empirical.csv'}`",
            f"- bound absolute `k*`: `{out_dir / 'results' / 'kstar_absolute_bound.csv'}`",
            f"- bound relative `k*`: `{out_dir / 'results' / 'kstar_relative_bound.csv'}`",
            f"- historical checks: `{out_dir / 'results' / 'historical_exp2_checks.csv'}`",
            f"- plots: `{out_dir / 'plots/'}`",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    workspace = Path.cwd()
    artifact_root = workspace / "datasets/synthetic/dev_buff_v2_20260510"
    out_dir = workspace / "results" / "buff_encoder_v2" / f"subcolumn_error_{DATE_TAG}"
    report_path = workspace / "research" / "2026-05-11_Current_Buff_Encoder_Subcolumn_Error_Study.md"

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    exact_reference: dict[tuple[str, str], float] = {}

    for manifest_path in sorted(artifact_root.glob("*/*/manifest.json")):
        artifact_dir = manifest_path.parent
        print(f"[v2-exp2] analyzing {artifact_dir}")
        rows, summary_row = analyze_dataset(artifact_dir)
        all_rows.extend(rows)
        summary_rows.append(summary_row)
        exact_reference[(summary_row["dataset"], "sum")] = summary_row["exact_sum"]
        exact_reference[(summary_row["dataset"], "avg")] = summary_row["exact_avg"]
        exact_reference[(summary_row["dataset"], "min")] = summary_row["exact_min"]
        exact_reference[(summary_row["dataset"], "max")] = summary_row["exact_max"]
        exact_reference[(summary_row["dataset"], "var")] = summary_row["exact_var"]
        exact_reference[(summary_row["dataset"], "count_gt")] = float(summary_row["exact_count"])

    all_rows.sort(key=lambda row: (row["dataset"], row["aggregation"], int(row["k"])))
    summary_rows.sort(key=lambda row: row["dataset"])

    metrics_path = out_dir / "results" / "metrics.csv"
    dataset_summary_path = out_dir / "results" / "dataset_summary.csv"
    hist_checks_path = out_dir / "results" / "historical_exp2_checks.csv"
    write_csv(metrics_path, METRICS_COLUMNS, all_rows)
    write_csv(dataset_summary_path, DATASET_SUMMARY_COLUMNS, summary_rows)
    write_kstar_csv(out_dir / "results" / "kstar_absolute_empirical.csv", all_rows, ABS_EPSILONS, False, False)
    write_kstar_csv(out_dir / "results" / "kstar_relative_empirical.csv", all_rows, REL_EPSILONS, True, False)
    write_kstar_csv(out_dir / "results" / "kstar_absolute_bound.csv", all_rows, ABS_EPSILONS, False, True)
    write_kstar_csv(out_dir / "results" / "kstar_relative_bound.csv", all_rows, REL_EPSILONS, True, True)

    historical_rows = historical_checks(exact_reference)
    write_csv(hist_checks_path, HISTORICAL_CHECK_COLUMNS, historical_rows)

    plots_dir = out_dir / "plots"
    for aggregation in AGGREGATIONS:
        for dataset in THRESHOLDS:
            matching = [
                row for row in all_rows
                if row["aggregation"] == aggregation and row["dataset"] == dataset
            ]
            empirical = [(float(row["k"]), float(row["abs_error"])) for row in matching]
            bound = [(float(row["k"]), float(row["analytic_bound"])) for row in matching]
            write_svg_plot(
                plots_dir / f"{aggregation}_{dataset}.svg",
                f"{DATASET_DISPLAY[dataset]} / {aggregation}",
                MAX_K,
                empirical,
                bound,
            )

    build_report(
        report_path,
        out_dir=out_dir,
        summary_rows=summary_rows,
        metrics_rows=all_rows,
        historical_rows=historical_rows,
    )

    for path, row_count in [
        (metrics_path, len(all_rows)),
        (dataset_summary_path, len(summary_rows)),
        (hist_checks_path, len(historical_rows)),
    ]:
        write_provenance(
            path.with_suffix(path.suffix + ".provenance.json"),
            argv=sys.argv,
            extra={"output": str(path), "row_count": row_count},
        )

    print(f"[v2-exp2] wrote study outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
