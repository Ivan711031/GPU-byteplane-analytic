#!/usr/bin/env python3
"""Tests for Phase 3-Z1 Filtered-Aggregate Evaluator.

Tests:
- classify_decoded: Q/D/U classification for known interval-truth values
- filtered_clean_vs_faulted: contains_truth==1.0 (like unfiltered SUM)
- predicate_envelope_conservative: active_count_max=n_rows is valid ceiling
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from phase3_z1_filtered_evaluator import (
    FilteredResult,
    classify_decoded_intervals,
    compute_filtered_delivered_result,
    compute_filtered_ui_prediction,
    contains_truth_value,
)

CLEAN_BYTE: int = 0x42
N_ROWS: int = 8
SCALE: int = 100
SEGMENT_SIZE: int = 1024

TESTS_RUN: list[str] = []
TESTS_PASSED: list[str] = []
TESTS_FAILED: list[str] = []


def _make_fault_plan(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = {
        "metadata": {
            "phase": "phase2",
            "target_plane": 0,
            "replica_index": 0,
            "fault_rate_numeric": 0.0,
            "seed": 0,
            "actual_fault_count": len(entries),
            "plane_size_bytes": 0,
        },
        "entries": list(entries),
    }
    path.write_text(json.dumps(fp) + "\n")


def _make_clean_planes(n_rows: int, byte_val: int = CLEAN_BYTE) -> list[bytes]:
    return [bytes([byte_val] * n_rows) for _ in range(8)]


def _run_test(fn) -> None:
    name = fn.__name__
    TESTS_RUN.append(name)
    try:
        fn()
        print(f"  \u2713 {name}")
        TESTS_PASSED.append(name)
    except Exception as e:
        print(f"  \u2717 {name}: {e}")
        import traceback
        traceback.print_exc()
        TESTS_FAILED.append(name)


#
# -- test_classify_decoded ------------------------------------------------
#


def test_classify_qualified() -> None:
    Q, D, Uq, Ud = classify_decoded_intervals(10.0, 12.0, 5.0)
    assert Q and not D and not Uq and not Ud


def test_classify_disqualified() -> None:
    Q, D, Uq, Ud = classify_decoded_intervals(1.0, 3.0, 5.0)
    assert not Q and D and not Uq and not Ud


def test_classify_uncertain() -> None:
    Q, D, Uq, Ud = classify_decoded_intervals(4.0, 6.0, 5.0)
    assert not Q and not D
    assert Uq or Ud


def test_classify_depth_resolvable() -> None:
    Q, D, Uq, Ud = classify_decoded_intervals(
        4.0, 8.0, 5.0,
        x_full_low=5.5, x_full_high=5.8,
    )
    assert not Q and not D
    assert not Uq and Ud


def test_classify_quantization_uncertain() -> None:
    Q, D, Uq, Ud = classify_decoded_intervals(
        4.5, 5.5, 5.0,
        x_full_low=4.6, x_full_high=5.4,
    )
    assert not Q and not D
    assert Uq and not Ud


def test_classify_threshold_touching() -> None:
    Q, D, Uq, Ud = classify_decoded_intervals(5.0, 7.0, 5.0)
    assert Q and not D and not Uq and not Ud


#
# -- test_filtered_clean_vs_faulted ---------------------------------------
#


def _run_filtered_case(
    plane: int,
    entries: list[dict],
    threshold: float,
    functional: str,
) -> FilteredResult:
    clean_planes = _make_clean_planes(N_ROWS)
    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        _make_fault_plan(
            plan_dir / f"plane{plane}/replica0/seed_0.json",
            entries,
        )
        fpp = {plane: [str(plan_dir / f"plane{plane}/replica0/seed_0.json")]}
        r_vector = [1] * 8
        return compute_filtered_delivered_result(
            clean_planes, fpp, r_vector,
            scale=SCALE, n_rows=N_ROWS,
            threshold=threshold, functional=functional,
            dataset="test_filtered", policy="uniform_repair_fraction",
            allocation_r="|".join(str(1) for _ in range(8)),
            segment_size=SEGMENT_SIZE,
        )


def test_filtered_contains_truth_sum() -> None:
    threshold = float(CLEAN_BYTE) * 256**7 / SCALE * 0.8
    for plane in range(8):
        result = _run_filtered_case(
            plane, [{"offset": 3, "mask": 0xFF}],
            threshold, "filtered_sum",
        )
        assert result.contains_truth, (
            f"filtered_sum: plane {plane} contains_truth=False"
        )


def test_filtered_contains_truth_count() -> None:
    threshold = float(CLEAN_BYTE) * 256**7 / SCALE * 0.8
    for plane in range(8):
        result = _run_filtered_case(
            plane, [{"offset": 3, "mask": 0xFF}],
            threshold, "count",
        )
        assert result.contains_truth, (
            f"count: plane {plane} contains_truth=False"
        )


def test_filtered_ff_positive_count() -> None:
    threshold = 0.0
    result = _run_filtered_case(0, [{"offset": 3, "mask": 0x01}], threshold, "count")
    assert result.ff_qualified_count == N_ROWS, (
        f"all rows should qualify at threshold=0, got {result.ff_qualified_count}"
    )


def test_filtered_all_disqualified_ff() -> None:
    threshold = 1e30
    result = _run_filtered_case(0, [{"offset": 3, "mask": 0x01}], threshold, "count")
    assert result.ff_qualified_count == 0, (
        f"no rows should qualify at extreme threshold, got {result.ff_qualified_count}"
    )


#
# -- test_predicate_envelope_conservative ---------------------------------
#


def test_prediction_valid_ceiling_filtered_sum() -> None:
    tau_pred = 0.10
    for plane in range(8):
        clean_planes = _make_clean_planes(N_ROWS)
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir = Path(tmp) / "fault_plans"
            _make_fault_plan(
                plan_dir / f"plane{plane}/replica0/seed_0.json",
                [{"offset": i, "mask": 0xFF} for i in range(N_ROWS)],
            )
            fpp = {plane: [str(plan_dir / f"plane{plane}/replica0/seed_0.json")]}
            r_vector = [1] * 8
            result = compute_filtered_delivered_result(
                clean_planes, fpp, r_vector,
                scale=SCALE, n_rows=N_ROWS,
                threshold=0.0, functional="filtered_sum",
                dataset="test_envelope", policy="uniform_repair_fraction",
                allocation_r="|".join(str(1) for _ in range(8)),
                segment_size=SEGMENT_SIZE,
            )

        assert result.bound_width_predicted >= result.bound_width * (1.0 - 1e-12), (
            f"plane {plane}: predicted {result.bound_width_predicted} < observed {result.bound_width}"
        )
        assert result.bound_width_prediction_err <= tau_pred, (
            f"plane {plane}: err={result.bound_width_prediction_err:.6e} > tau_pred"
        )


def test_prediction_valid_ceiling_count() -> None:
    tau_pred = 0.10
    for plane in range(8):
        clean_planes = _make_clean_planes(N_ROWS)
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir = Path(tmp) / "fault_plans"
            _make_fault_plan(
                plan_dir / f"plane{plane}/replica0/seed_0.json",
                [{"offset": i, "mask": 0xFF} for i in range(N_ROWS)],
            )
            fpp = {plane: [str(plan_dir / f"plane{plane}/replica0/seed_0.json")]}
            r_vector = [1] * 8
            result = compute_filtered_delivered_result(
                clean_planes, fpp, r_vector,
                scale=SCALE, n_rows=N_ROWS,
                threshold=0.0, functional="count",
                dataset="test_envelope_count", policy="uniform_repair_fraction",
                allocation_r="|".join(str(1) for _ in range(8)),
                segment_size=SEGMENT_SIZE,
            )

        assert result.bound_width_predicted >= result.bound_width * (1.0 - 1e-12), (
            f"plane {plane}: predicted {result.bound_width_predicted} < observed {result.bound_width}"
        )
        assert result.bound_width_prediction_err <= tau_pred, (
            f"plane {plane}: err={result.bound_width_prediction_err:.6e} > tau_pred"
        )


def test_active_count_max_is_n_rows() -> None:
    outcomes: dict[tuple[int, int], str] = {}
    for seg_idx in range(1):
        for p in range(8):
            outcomes[(seg_idx, p)] = "unprotected"

    pred = compute_filtered_ui_prediction(
        outcomes=outcomes, scale=SCALE, n_rows=N_ROWS,
        functional="filtered_sum", segment_size=SEGMENT_SIZE,
    )

    expected_sum = N_ROWS / SCALE
    for p in range(8):
        expected_sum += N_ROWS * 255 * (1 << (8 * (7 - p))) / SCALE

    assert math.isclose(
        pred["bound_width_predicted"], expected_sum, rel_tol=1e-9
    ), f"predicted {pred['bound_width_predicted']} != expected {expected_sum}"


def test_active_count_max_count_no_scale() -> None:
    outcomes: dict[tuple[int, int], str] = {}
    for seg_idx in range(1):
        for p in range(8):
            outcomes[(seg_idx, p)] = "unprotected"

    pred = compute_filtered_ui_prediction(
        outcomes=outcomes, scale=SCALE, n_rows=N_ROWS,
        functional="count", segment_size=SEGMENT_SIZE,
    )

    assert pred["bound_width_fault_free"] == 0.0, (
        "COUNT fault-free bound should be 0"
    )
    expected = 8.0 * N_ROWS
    assert math.isclose(
        pred["bound_width_predicted"], expected, rel_tol=1e-9
    ), f"COUNT predicted {pred['bound_width_predicted']} != expected {expected}"


#
# -- Z1-C: Predicate-envelope with borderline-population guard -------------
#


def test_z1c_borderline_nonempty() -> None:
    """Z1-C guard: verify max_flip_rows > 0 for exercised (plane, threshold) cells.

    On the constant-byte tiny fixture (0x42), plane 7 at p25 has no borderline rows
    because all decoded values are identical and plane 7's swing is too small to
    cross the p25 threshold. This is expected — the guard exists to catch exactly
    this situation on real data where it WOULD indicate a trivial pass.
    """
    from phase3_z1_filtered_evaluator import (
        select_decoded_thresholds, compute_max_flip_rows,
    )
    clean_planes = _make_clean_planes(N_ROWS, byte_val=0x42)

    thresholds = select_decoded_thresholds(clean_planes, SCALE, N_ROWS)
    total_cells = 0
    nonempty_cells = 0

    for label, thr in thresholds.items():
        for plane in range(8):
            total_cells += 1
            mfr = compute_max_flip_rows(clean_planes, plane, thr, SCALE, N_ROWS)
            if mfr > 0:
                nonempty_cells += 1

    # At least some cells must be nonempty (proving the guard can fire)
    assert nonempty_cells > 0, (
        f"All {total_cells} cells have max_flip_rows=0 — guard is vacuously satisfied."
    )
    # Report how many are exercised
    print(f"  borderline guard: {nonempty_cells}/{total_cells} plane×threshold cells exercised")
    for label, thr in thresholds.items():
        for plane in range(8):
            mfr = compute_max_flip_rows(clean_planes, plane, thr, SCALE, N_ROWS)
            print(f"    {label} plane={plane}: max_flip_rows={mfr}" if mfr == 0 else
                  f"    {label} plane={plane}: max_flip_rows={mfr}")


def test_z1c_count_contains_truth_all_planes() -> None:
    """Z1-C primary: COUNT contains_truth for every plane at dense-zone thresholds."""
    from phase3_z1_filtered_evaluator import select_decoded_thresholds, compute_max_flip_rows
    clean_planes = _make_clean_planes(N_ROWS, byte_val=0x42)

    thresholds = select_decoded_thresholds(clean_planes, SCALE, N_ROWS)
    failures: list[str] = []

    for label, thr in thresholds.items():
        for plane in range(8):
            # Guard: skip cells with no borderline population
            if compute_max_flip_rows(clean_planes, plane, thr, SCALE, N_ROWS) == 0:
                continue

            with tempfile.TemporaryDirectory() as tmp:
                plan_dir = Path(tmp) / "fault_plans"
                _make_fault_plan(
                    plan_dir / f"plane{plane}/replica0/seed_0.json",
                    [{"offset": i, "mask": 0xFF} for i in range(N_ROWS)],
                )
                fpp = {plane: [str(plan_dir / f"plane{plane}/replica0/seed_0.json")]}
                rv = [1] * 8

                result = compute_filtered_delivered_result(
                    clean_planes, fpp, rv,
                    scale=SCALE, n_rows=N_ROWS,
                    threshold=thr, functional="count",
                    dataset="z1c_count", policy="uniform_repair_fraction",
                )

            if not result.contains_truth:
                failures.append(f"{label} plane={plane}")

    assert not failures, (
        f"COUNT contains_truth FAILED for: {failures}"
    )


def test_z1c_filtered_sum_contains_truth_all_planes() -> None:
    """Z1-C secondary: filtered-SUM contains_truth."""
    from phase3_z1_filtered_evaluator import select_decoded_thresholds, compute_max_flip_rows
    clean_planes = _make_clean_planes(N_ROWS, byte_val=0x42)

    thresholds = select_decoded_thresholds(clean_planes, SCALE, N_ROWS)
    failures: list[str] = []

    for label, thr in thresholds.items():
        for plane in range(8):
            if compute_max_flip_rows(clean_planes, plane, thr, SCALE, N_ROWS) == 0:
                continue

            with tempfile.TemporaryDirectory() as tmp:
                plan_dir = Path(tmp) / "fault_plans"
                _make_fault_plan(
                    plan_dir / f"plane{plane}/replica0/seed_0.json",
                    [{"offset": i, "mask": 0xFF} for i in range(N_ROWS)],
                )
                fpp = {plane: [str(plan_dir / f"plane{plane}/replica0/seed_0.json")]}
                rv = [1] * 8

                result = compute_filtered_delivered_result(
                    clean_planes, fpp, rv,
                    scale=SCALE, n_rows=N_ROWS,
                    threshold=thr, functional="filtered_sum",
                    dataset="z1c_fsum", policy="uniform_repair_fraction",
                )

            if not result.contains_truth:
                failures.append(f"{label} plane={plane}")

    assert not failures, (
        f"filtered-SUM contains_truth FAILED for: {failures}"
    )


#
# -- run ------------------------------------------------------------------
#


def main() -> None:
    print("Phase 3-Z1 Filtered-Aggregate Evaluator Tests")
    print("=" * 50)

    print("\n--- test_classify_decoded ---")
    _run_test(test_classify_qualified)
    _run_test(test_classify_disqualified)
    _run_test(test_classify_uncertain)
    _run_test(test_classify_depth_resolvable)
    _run_test(test_classify_quantization_uncertain)
    _run_test(test_classify_threshold_touching)

    print("\n--- test_filtered_clean_vs_faulted ---")
    _run_test(test_filtered_contains_truth_sum)
    _run_test(test_filtered_contains_truth_count)
    _run_test(test_filtered_ff_positive_count)
    _run_test(test_filtered_all_disqualified_ff)

    print("\n--- test_predicate_envelope_conservative ---")
    _run_test(test_prediction_valid_ceiling_filtered_sum)
    _run_test(test_prediction_valid_ceiling_count)
    _run_test(test_active_count_max_is_n_rows)
    _run_test(test_active_count_max_count_no_scale)

    print("\n--- Z1-C: Predicate envelope with borderline guard ---")
    _run_test(test_z1c_borderline_nonempty)
    _run_test(test_z1c_count_contains_truth_all_planes)
    _run_test(test_z1c_filtered_sum_contains_truth_all_planes)

    total = len(TESTS_RUN)
    passed = len(TESTS_PASSED)
    failed = len(TESTS_FAILED)
    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed:
        for name in TESTS_FAILED:
            print(f"  FAILED: {name}")
        sys.exit(1)
    else:
        print("  ALL PASSED")


if __name__ == "__main__":
    main()
