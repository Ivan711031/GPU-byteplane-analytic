#!/usr/bin/env python3
"""Phase 3-Z1 Filtered-Aggregate Evaluator.

Evaluates filtered-SUM and COUNT functionals on byteplane data:
- classify_decoded_intervals: predicate membership classification per P3D
- compute_filtered_delivered_result: full pipeline with fault injection
- compute_filtered_ui_prediction: bound width prediction per Z1a §2
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _ensure_scripts_path() -> None:
    _dir = str(Path(__file__).resolve().parent)
    if _dir not in sys.path:
        sys.path.insert(0, _dir)


_ensure_scripts_path()
from phase3_y0_evaluator import (
    PLANE_WEIGHTS,
    SEGMENT_SIZE,
    compute_clean_sum,
    compute_voted_planes,
    compute_segment_outcomes,
    decode_value,
    bound_width_prediction_err,
)


@dataclass
class FilteredResult:
    dataset: str
    n_rows: int
    scale: int
    k: int
    threshold: float
    functional: str  # "filtered_sum" or "count"

    # Fault-free baseline
    ff_qualified_count: int
    ff_filtered_sum: float

    # With fault
    delivered_qualified_count: int
    delivered_filtered_sum: float
    bound_qualified_count: int
    bound_disqualified_count: int
    uncertainty_count: int

    # Y metrics
    contains_truth: bool
    bound_width: float
    bound_width_predicted: float
    bound_width_prediction_err: float


def select_decoded_thresholds(
    clean_planes: list[bytes],
    scale: int,
    n_rows: int,
    k: int = 8,
) -> dict[str, float]:
    """Compute decoded value distribution and return dense-zone thresholds.

    Returns p25/p50/p75 of the decoded midpoint distribution.
    These fall INSIDE the dense data range, guaranteeing non-empty borderline
    populations for plane corruption tests.
    """
    import numpy as np
    values = np.zeros(n_rows, dtype=np.float64)
    for i in range(n_rows):
        clean_bytes = [clean_planes[p][i] for p in range(8)]
        x_low, x_high = decode_value(clean_bytes, k, scale)
        values[i] = (x_low + x_high) / 2.0
    p25 = float(np.percentile(values, 25))
    p50 = float(np.percentile(values, 50))
    p75 = float(np.percentile(values, 75))
    return {"p25": p25, "p50": p50, "p75": p75}


def compute_max_flip_rows(
    clean_planes: list[bytes],
    plane: int,
    threshold: float,
    scale: int,
    n_rows: int,
    k: int = 8,
) -> int:
    """Count rows whose decoded interval straddles threshold when plane i is corrupted.

    A row can flip membership if its faulted decoded interval [x_low, x_high]
    (with plane i bytes maximally swung) crosses the threshold.
    Returns max_flip_rows = number of rows that WOULD change membership status
    if plane i is fully corrupted (mask=0xFF).
    """
    q_err = 0.5 / scale
    max_swing = 255 * PLANE_WEIGHTS[plane] / scale if plane < 8 else 0.0

    flip_count = 0
    for i in range(n_rows):
        clean_bytes = [clean_planes[p][i] for p in range(8)]
        x_low, x_high = decode_value(clean_bytes, k, scale)

        # If clean row is qualified (x_low >= threshold): corrupted can drop out
        # if x_high - max_swing < threshold (worst-case drop).
        # If clean row is disqualified (x_high < threshold): corrupted can add in
        # if x_low + max_swing >= threshold (worst-case add).
        clean_Q = x_low >= threshold
        clean_D = x_high < threshold

        if clean_Q:
            # Possible drop: fully corrupted value could drop below threshold
            if x_high - max_swing < threshold:
                flip_count += 1
        elif clean_D:
            # Possible add: fully corrupted value could rise above threshold
            if x_low + max_swing >= threshold:
                flip_count += 1
        else:
            # Already uncertain — any corruption changes nothing
            pass

    return flip_count


def assert_borderline_nonempty(
    clean_planes: list[bytes],
    plane: int,
    threshold: float,
    scale: int,
    n_rows: int,
    k: int = 8,
    label: str = "",
) -> int:
    """Assert and return max_flip_rows. Raises ValueError if zero."""
    mfr = compute_max_flip_rows(clean_planes, plane, threshold, scale, n_rows, k)
    if mfr == 0:
        raise ValueError(
            f"borderline population EMPTY: {label} plane={plane} threshold={threshold:.6e}. "
            "Test trivially passes — choose a different threshold in the dense data range."
        )
    return mfr


def contains_truth_value(
    clean_answer: float, delivered_answer: float, bound_width: float
) -> bool:
    lo = delivered_answer - bound_width / 2.0
    hi = delivered_answer + bound_width / 2.0
    return lo <= clean_answer <= hi


def classify_decoded_intervals(
    x_low: float,
    x_high: float,
    threshold: float,
    x_full_low: float | None = None,
    x_full_high: float | None = None,
) -> tuple[bool, bool, bool, bool]:
    """Classify a single decoded interval against threshold.

    Returns (Q, D, U_quant, U_depth).
    Q:        x_low >= threshold (definitely qualified)
    D:        x_high < threshold (definitely disqualified)
    U_quant:  uncertain, not resolved by full-depth decode
    U_depth:  uncertain, resolved by full-depth decode
    """
    if x_low >= threshold:
        return True, False, False, False
    if x_high < threshold:
        return False, True, False, False
    if x_full_low is not None and x_full_high is not None:
        if x_full_low >= threshold or x_full_high < threshold:
            return False, False, False, True
    return False, False, True, False


def _row_midpoint(delivered_bytes: list[int], k: int, scale: int) -> float:
    x_low, x_high = decode_value(delivered_bytes, k, scale)
    return (x_low + x_high) / 2.0


def compute_filtered_delivered_result(
    clean_planes: list[bytes],
    fault_plan_paths: dict[int, list[str]],
    r_vector: list[int],
    scale: int,
    n_rows: int,
    threshold: float,
    functional: str = "filtered_sum",
    dataset: str = "tiny_fixture",
    k: int = 8,
    policy: str = "graded",
    allocation_r: str = "",
    seed: int = 0,
    fault_rate: str = "1e-06",
    segment_size: int = SEGMENT_SIZE,
    fallback_lane: str = "bounded_degradation",
) -> FilteredResult:
    if functional not in ("filtered_sum", "count"):
        raise ValueError(f"unknown functional: {functional}")

    q_err = 0.5 / scale

    # --- Fault-free baseline ---
    ff_qualified_count = 0
    ff_filtered_sum = 0.0

    for i in range(n_rows):
        clean_bytes = [clean_planes[p][i] for p in range(8)]
        ff_x_low, ff_x_high = decode_value(clean_bytes, k, scale)
        ff_xf_low, ff_xf_high = decode_value(clean_bytes, 8, scale)
        Q, D, _, _ = classify_decoded_intervals(
            ff_x_low, ff_x_high, threshold, ff_xf_low, ff_xf_high
        )
        if Q:
            ff_qualified_count += 1
            if functional == "filtered_sum":
                midpoint = _row_midpoint(clean_bytes, k, scale)
                ff_filtered_sum += midpoint
            else:
                ff_filtered_sum += 1.0

    # --- Faulted data ---
    voted_planes = compute_voted_planes(
        clean_planes, fault_plan_paths, r_vector
    )
    outcomes = compute_segment_outcomes(
        clean_planes, fault_plan_paths, r_vector, segment_size
    )

    q_total = 0
    d_total = 0
    u_depth_total = 0
    u_quant_total = 0
    delivered_filtered_sum = 0.0

    for i in range(n_rows):
        seg_idx = i // segment_size
        delivered_bytes = [0] * 8
        for p in range(8):
            st = outcomes.get((seg_idx, p), "clean")
            if st in ("clean", "repaired"):
                delivered_bytes[p] = clean_planes[p][i]
            else:
                delivered_bytes[p] = voted_planes[p][i]

        x_low, x_high = decode_value(delivered_bytes, k, scale)
        xf_low, xf_high = decode_value(delivered_bytes, 8, scale)
        Q, D, Uq, Ud = classify_decoded_intervals(
            x_low, x_high, threshold, xf_low, xf_high
        )

        if Q:
            q_total += 1
            if functional == "filtered_sum":
                midpoint = _row_midpoint(delivered_bytes, k, scale)
                delivered_filtered_sum += midpoint
            else:
                delivered_filtered_sum += 1.0
        elif D:
            d_total += 1
        elif Ud:
            u_depth_total += 1
        else:
            u_quant_total += 1

    # --- Bound width from U_i ---
    n_segments = (n_rows + segment_size - 1) // segment_size

    if functional == "filtered_sum":
        bound_width_fault_free = n_rows / scale
    else:
        bound_width_fault_free = 0.0

    bound_widen = 0.0
    for p in range(8):
        for seg_idx in range(n_segments):
            st = outcomes.get((seg_idx, p), "clean")
            if st in ("degraded", "unprotected"):
                start = seg_idx * segment_size
                end = min(start + segment_size, n_rows)
                count = end - start
                if functional == "filtered_sum":
                    contrib = count * 255 * PLANE_WEIGHTS[p] / scale
                else:
                    contrib = float(count)
                bound_widen += contrib

    bound_width = bound_width_fault_free + bound_widen

    # --- contains_truth ---
    ct = contains_truth_value(
        ff_filtered_sum, delivered_filtered_sum, bound_width
    )

    # --- Predicted bound ---
    pred = compute_filtered_ui_prediction(
        outcomes=outcomes,
        scale=scale,
        n_rows=n_rows,
        functional=functional,
        segment_size=segment_size,
    )
    bound_width_predicted = pred["bound_width_predicted"]

    err = bound_width_prediction_err(bound_width, bound_width_predicted)

    return FilteredResult(
        dataset=dataset,
        n_rows=n_rows,
        scale=scale,
        k=k,
        threshold=threshold,
        functional=functional,
        ff_qualified_count=ff_qualified_count,
        ff_filtered_sum=ff_filtered_sum,
        delivered_qualified_count=q_total,
        delivered_filtered_sum=delivered_filtered_sum,
        bound_qualified_count=q_total + u_depth_total,
        bound_disqualified_count=d_total + u_depth_total,
        uncertainty_count=u_quant_total,
        contains_truth=ct,
        bound_width=bound_width,
        bound_width_predicted=bound_width_predicted,
        bound_width_prediction_err=err,
    )


def compute_filtered_ui_prediction(
    outcomes: dict[tuple[int, int], str],
    scale: int,
    n_rows: int,
    functional: str = "filtered_sum",
    segment_size: int = SEGMENT_SIZE,
) -> dict[str, Any]:
    if functional not in ("filtered_sum", "count"):
        raise ValueError(f"unknown functional: {functional}")

    n_segments = (n_rows + segment_size - 1) // segment_size

    if functional == "filtered_sum":
        bound_width_fault_free = n_rows / scale
    else:
        bound_width_fault_free = 0.0

    predicted_widen = 0.0
    ui_per_plane: dict[str, float] = {}
    for p in range(8):
        plane_total = 0.0
        for seg_idx in range(n_segments):
            st = outcomes.get((seg_idx, p), "clean")
            if st in ("degraded", "unprotected"):
                start = seg_idx * segment_size
                end = min(start + segment_size, n_rows)
                count = end - start
                if functional == "filtered_sum":
                    contrib = count * 255 * PLANE_WEIGHTS[p] / scale
                else:
                    contrib = float(count)
                predicted_widen += contrib
                plane_total += contrib
        ui_per_plane[str(p)] = plane_total

    return {
        "bound_width_fault_free": bound_width_fault_free,
        "bound_width_predicted": bound_width_fault_free + predicted_widen,
        "ui_per_plane": ui_per_plane,
    }
