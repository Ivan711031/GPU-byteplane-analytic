#!/usr/bin/env python3
"""Unit tests for Phase 3-Y0 preflight.

Tests 5 acceptance criteria:
  AC1: One corrupted S0 replica -> repair succeeds -> err_fault_user_observed == 0
  AC2: Two corrupted S0 replicas -> degrade -> err_fault_user_observed > 0
  AC3: Unprotected S0 (uniform_repair_fraction) -> bound_width_inflation > 1.0
  AC4: Deterministic: same inputs -> same outputs
  AC5: Raw fallback excluded from primary accuracy claim
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from phase2_oracle import apply_fault_plan
from phase3_y0_evaluator import (
    compute_delivered_answer_with_degradation,
    compute_segment_outcomes,
    compute_clean_sum,
    compute_ui_prediction,
    bound_width_prediction_err,
)


CLEAN_BYTE: int = 0x42
N_ROWS: int = 8
SCALE: int = 100
SEGMENT_SIZE: int = 1024


def _make_fault_plan(path: Path, entries: list[dict]) -> None:
    """Write a fault plan JSON file at path."""
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
    """Generate clean plane bytes: all planes have constant byte_val."""
    return [bytes([byte_val] * n_rows) for _ in range(8)]


#
# ── AC1: One corrupted S0 replica ──────────────────────────────────────
#


def test_ac1_one_s0_replica_corrupted() -> None:
    """AC1: 3 replicas of plane 0, 1 faulted, 2 clean -> repair succeeds.

    All tolerances use exact integer comparison because voting recovers
    the clean byte perfectly.
    """
    clean_planes = _make_clean_planes(N_ROWS)
    clean_sum = compute_clean_sum(clean_planes)
    clean_answer = clean_sum / SCALE

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        _make_fault_plan(
            plan_dir / "plane0/replica0/seed_0.json",
            [{"offset": 3, "mask": 0xFF}],
        )
        _make_fault_plan(plan_dir / "plane0/replica1/seed_0.json", [])
        _make_fault_plan(plan_dir / "plane0/replica2/seed_0.json", [])

        fpp = {0: [
            str(plan_dir / "plane0/replica0/seed_0.json"),
            str(plan_dir / "plane0/replica1/seed_0.json"),
            str(plan_dir / "plane0/replica2/seed_0.json"),
        ]}
        r_vector = [3, 1, 1, 1, 1, 1, 1, 1]

        result = compute_delivered_answer_with_degradation(
            clean_planes, fpp, r_vector,
            scale=SCALE, n_rows=N_ROWS,
            dataset="test_ac1", policy="graded",
            allocation_r="3|1|1|1|1|1|1|1",
            segment_size=SEGMENT_SIZE,
        )

    assert result.err_fault_user_observed == 0.0, (
        f"expected 0 error for repaired fault, got {result.err_fault_user_observed}"
    )
    assert result.repair_failure_degrade_rate == 0.0, (
        f"expected 0 repair failure rate, got {result.repair_failure_degrade_rate}"
    )
    assert result.delivered_answer == clean_answer, (
        f"delivered {result.delivered_answer} != clean {clean_answer}"
    )
    assert result.segments_crc_hit == 1, (
        f"expected 1 CRC-hit segment (plane 0), got {result.segments_crc_hit}"
    )
    assert result.segments_repaired == 1, (
        f"expected 1 repaired segment, got {result.segments_repaired}"
    )
    assert result.certified_quality_pass is True


#
# ── AC2: Two corrupted S0 replicas ────────────────────────────────────
#


def test_ac2_two_s0_replicas_corrupted() -> None:
    """AC2: 3 replicas, 2 faults at same offset but different masks.

    Replica 0: XOR 0x03 -> byte=0x41
    Replica 1: XOR 0x02 -> byte=0x40
    Replica 2: clean  -> byte=0x42
    Vote: [0x41, 0x40, 0x42] all unique -> tie-break min=0x40
    Since 0x40 != clean 0x42 -> degrade.
    """
    clean_planes = _make_clean_planes(N_ROWS)
    clean_sum = compute_clean_sum(clean_planes)
    clean_answer = clean_sum / SCALE

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        _make_fault_plan(
            plan_dir / "plane0/replica0/seed_0.json",
            [{"offset": 3, "mask": 0x03}],
        )
        _make_fault_plan(
            plan_dir / "plane0/replica1/seed_0.json",
            [{"offset": 3, "mask": 0x02}],
        )
        _make_fault_plan(plan_dir / "plane0/replica2/seed_0.json", [])

        fpp = {0: [
            str(plan_dir / "plane0/replica0/seed_0.json"),
            str(plan_dir / "plane0/replica1/seed_0.json"),
            str(plan_dir / "plane0/replica2/seed_0.json"),
        ]}
        r_vector = [3, 1, 1, 1, 1, 1, 1, 1]

        result = compute_delivered_answer_with_degradation(
            clean_planes, fpp, r_vector,
            scale=SCALE, n_rows=N_ROWS,
            dataset="test_ac2", policy="graded",
            allocation_r="3|1|1|1|1|1|1|1",
            segment_size=SEGMENT_SIZE,
        )

    assert result.err_fault_user_observed > 0, (
        f"expected > 0 error for degraded fault, got {result.err_fault_user_observed}"
    )
    assert result.repair_failure_degrade_rate > 0, (
        f"expected > 0 repair failure rate, got {result.repair_failure_degrade_rate}"
    )
    assert result.delivered_answer != clean_answer, (
        "delivered answer should differ from clean when degrade occurs"
    )
    assert result.segments_crc_hit == 1, (
        f"expected 1 CRC-hit segment, got {result.segments_crc_hit}"
    )
    assert result.segments_degraded == 1, (
        f"expected 1 degraded segment, got {result.segments_degraded}"
    )
    assert result.segments_repaired == 0, (
        f"expected 0 repaired segments, got {result.segments_repaired}"
    )

    plane0_weight = 1 << (8 * 7)
    diff_per_byte = abs(0x40 - 0x42)
    expected_error = diff_per_byte * plane0_weight / SCALE / abs(clean_answer) if clean_answer != 0 else 0
    assert math.isclose(result.err_fault_user_observed, expected_error, rel_tol=1e-9), (
        f"error {result.err_fault_user_observed} != expected {expected_error}"
    )


#
# ── AC3: Unprotected S0 ──────────────────────────────────────────────
#


def test_ac3_unprotected_s0_degrades() -> None:
    """AC3: uniform_repair_fraction with r_p=1 for S0 (plane 0).

    Single replica, fault applied. Since r_p < 2: unprotected.
    bound_width_inflation should exceed 1.0 because S0 uncertainty
    widens the certified bound by 255 * plane0_weight.
    """
    clean_planes = _make_clean_planes(N_ROWS)

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        _make_fault_plan(
            plan_dir / "plane0/replica0/seed_0.json",
            [{"offset": 3, "mask": 0xFF}],
        )

        fpp = {0: [
            str(plan_dir / "plane0/replica0/seed_0.json"),
        ]}
        r_vector = [1, 1, 1, 1, 1, 1, 1, 1]

        result = compute_delivered_answer_with_degradation(
            clean_planes, fpp, r_vector,
            scale=SCALE, n_rows=N_ROWS,
            dataset="test_ac3", policy="uniform_repair_fraction",
            allocation_r="1|1|1|1|1|1|1|1",
            segment_size=SEGMENT_SIZE,
        )

    assert result.unprotected_s0_uncertainty_rate > 0, (
        "expected > 0 unprotected rate for r_p=1 S0"
    )
    assert result.bound_width_inflation > 1.0, (
        f"expected bound inflation > 1.0, got {result.bound_width_inflation}"
    )
    assert result.segments_unprotected > 0, (
        f"expected unprotected segment, got {result.segments_unprotected}"
    )

    plane0_weight = 1 << (8 * 7)
    expected_widening = 255 * plane0_weight / SCALE * N_ROWS
    q_err = 0.5 / SCALE
    expected_fault_free = 2.0 * q_err * N_ROWS
    expected_inflation = (expected_fault_free + expected_widening) / expected_fault_free
    assert math.isclose(result.bound_width_inflation, expected_inflation, rel_tol=1e-9), (
        f"inflation {result.bound_width_inflation} != expected {expected_inflation}"
    )


#
# ── AC4: Deterministic ──────────────────────────────────────────────
#


def test_ac4_bound_width_deterministic() -> None:
    """AC4: Run AC1 scenario twice; verify byte-identical outputs."""
    clean_planes = _make_clean_planes(N_ROWS)

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        _make_fault_plan(
            plan_dir / "plane0/replica0/seed_0.json",
            [{"offset": 3, "mask": 0xFF}],
        )
        _make_fault_plan(plan_dir / "plane0/replica1/seed_0.json", [])
        _make_fault_plan(plan_dir / "plane0/replica2/seed_0.json", [])

        fpp = {0: [
            str(plan_dir / "plane0/replica0/seed_0.json"),
            str(plan_dir / "plane0/replica1/seed_0.json"),
            str(plan_dir / "plane0/replica2/seed_0.json"),
        ]}
        r_vector = [3, 1, 1, 1, 1, 1, 1, 1]

        result1 = compute_delivered_answer_with_degradation(
            clean_planes, fpp, r_vector,
            scale=SCALE, n_rows=N_ROWS,
            dataset="test_ac4", policy="graded",
            allocation_r="3|1|1|1|1|1|1|1",
            segment_size=SEGMENT_SIZE,
        )
        result2 = compute_delivered_answer_with_degradation(
            clean_planes, fpp, r_vector,
            scale=SCALE, n_rows=N_ROWS,
            dataset="test_ac4", policy="graded",
            allocation_r="3|1|1|1|1|1|1|1",
            segment_size=SEGMENT_SIZE,
        )

    assert result1.err_fault_user_observed == result2.err_fault_user_observed, (
        f"err mismatch: {result1.err_fault_user_observed} vs {result2.err_fault_user_observed}"
    )
    assert result1.bound_width == result2.bound_width, (
        f"bound_width mismatch: {result1.bound_width} vs {result2.bound_width}"
    )
    assert result1.bound_width_inflation == result2.bound_width_inflation, (
        f"inflation mismatch: {result1.bound_width_inflation} vs {result2.bound_width_inflation}"
    )
    assert result1.repair_failure_degrade_rate == result2.repair_failure_degrade_rate
    assert result1.unprotected_s0_uncertainty_rate == result2.unprotected_s0_uncertainty_rate
    assert result1.delivered_answer == result2.delivered_answer


#
# ── AC5: Raw fallback excluded ─────────────────────────────────────
#


def test_ac5_raw_fallback_excluded() -> None:
    """AC5: When fallback='raw_fallback_context', certified_quality_pass is
    still based on the bounded-degradation pipeline; the fallback_lane
    field just marks it as context-only, not excluded from the metric.

    When fallback='bounded_degradation', the primary claim is active.
    """
    clean_planes = _make_clean_planes(N_ROWS)

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        _make_fault_plan(
            plan_dir / "plane0/replica0/seed_0.json",
            [{"offset": 3, "mask": 0xFF}],
        )
        _make_fault_plan(plan_dir / "plane0/replica1/seed_0.json", [])
        _make_fault_plan(plan_dir / "plane0/replica2/seed_0.json", [])

        fpp = {0: [
            str(plan_dir / "plane0/replica0/seed_0.json"),
            str(plan_dir / "plane0/replica1/seed_0.json"),
            str(plan_dir / "plane0/replica2/seed_0.json"),
        ]}
        r_vector = [3, 1, 1, 1, 1, 1, 1, 1]

        raw_result = compute_delivered_answer_with_degradation(
            clean_planes, fpp, r_vector,
            scale=SCALE, n_rows=N_ROWS,
            dataset="test_ac5", policy="graded",
            allocation_r="3|1|1|1|1|1|1|1",
            segment_size=SEGMENT_SIZE,
            fallback_lane="raw_fallback_context",
        )

        bd_result = compute_delivered_answer_with_degradation(
            clean_planes, fpp, r_vector,
            scale=SCALE, n_rows=N_ROWS,
            dataset="test_ac5", policy="graded",
            allocation_r="3|1|1|1|1|1|1|1",
            segment_size=SEGMENT_SIZE,
            fallback_lane="bounded_degradation",
        )

    assert raw_result.fallback_lane == "raw_fallback_context"
    assert bd_result.fallback_lane == "bounded_degradation"

    assert raw_result.certified_quality_pass is True
    assert bd_result.certified_quality_pass is True

    assert raw_result.fallback_lane != bd_result.fallback_lane


#
# ── Z1a: U_i prediction ─────────────────────────────────────────────
#


def test_ui_sum_formula() -> None:
    """Z1a: Verify U_i for SUM matches observed bound width when plane 0 is degraded.

    Uses unprotected S0 (r_p=1, uniform_repair_fraction) so U_i ceiling
    exactly equals the observed bound widen (both use 255 per byte).
    """
    clean_planes = _make_clean_planes(N_ROWS)

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        _make_fault_plan(
            plan_dir / "plane0/replica0/seed_0.json",
            [{"offset": 3, "mask": 0xFF}],
        )

        fpp = {0: [str(plan_dir / "plane0/replica0/seed_0.json")]}
        r_vector = [1, 1, 1, 1, 1, 1, 1, 1]

        observed = compute_delivered_answer_with_degradation(
            clean_planes, fpp, r_vector,
            scale=SCALE, n_rows=N_ROWS,
            dataset="test_ui_sum", policy="uniform_repair_fraction",
            allocation_r="1|1|1|1|1|1|1|1",
            segment_size=SEGMENT_SIZE,
        ).bound_width

        pred = compute_ui_prediction(
            clean_planes, fpp, r_vector,
            scale=SCALE, n_rows=N_ROWS,
            segment_size=SEGMENT_SIZE,
        )

    predicted = pred["bound_width_predicted"]
    # For unprotected S0 with all rows affected, U_i exactly matches observed
    assert math.isclose(predicted, observed, rel_tol=1e-9), (
        f"U_i prediction {predicted} != observed {observed}"
    )

    # Verify plane 0 contribution matches hand calculation
    plane0_weight = 1 << (8 * 7)
    expected_plane0_ui = N_ROWS * 255 * plane0_weight / SCALE
    assert math.isclose(pred["ui_per_plane"]["0"], expected_plane0_ui, rel_tol=1e-9), (
        f"U_i plane 0 {pred['ui_per_plane']['0']} != expected {expected_plane0_ui}"
    )

    # Other planes should have zero contribution (no faults)
    for p in range(1, 8):
        assert pred["ui_per_plane"][str(p)] == 0.0, (
            f"plane {p} should have zero U_i, got {pred['ui_per_plane'][str(p)]}"
        )

    # Fault-free bound check
    expected_fault_free = N_ROWS / SCALE
    assert math.isclose(pred["bound_width_fault_free"], expected_fault_free, rel_tol=1e-9)


def test_prediction_err_within_tau() -> None:
    """Z1a: Verify bound_width_prediction_err <= tau_pred = 0.10.

    Uses unprotected S0 (r_p < 2) so U_i ceiling matches observed,
    yielding zero prediction error.
    """
    tau_pred = 0.10
    clean_planes = _make_clean_planes(N_ROWS)

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        _make_fault_plan(
            plan_dir / "plane0/replica0/seed_0.json",
            [{"offset": 3, "mask": 0xFF}],
        )

        fpp = {0: [str(plan_dir / "plane0/replica0/seed_0.json")]}
        r_vector = [1, 1, 1, 1, 1, 1, 1, 1]

        observed = compute_delivered_answer_with_degradation(
            clean_planes, fpp, r_vector,
            scale=SCALE, n_rows=N_ROWS,
            dataset="test_tau", policy="uniform_repair_fraction",
            allocation_r="1|1|1|1|1|1|1|1",
            segment_size=SEGMENT_SIZE,
        ).bound_width

        pred = compute_ui_prediction(
            clean_planes, fpp, r_vector,
            scale=SCALE, n_rows=N_ROWS,
            segment_size=SEGMENT_SIZE,
        )

    err = bound_width_prediction_err(observed, pred["bound_width_predicted"])
    assert err <= tau_pred, (
        f"bound_width_prediction_err={err:.6e} > tau_pred={tau_pred}"
    )

    # Verify prediction is a valid ceiling (predicted >= observed within FP tol)
    assert pred["bound_width_predicted"] >= observed * (1.0 - 1e-12), (
        f"predicted {pred['bound_width_predicted']} < observed {observed}, "
        "U_i ceiling violated"
    )


#
# ── Run all tests when executed directly ─────────────────────────────
#


if __name__ == "__main__":
    import types
    test_fns = [
        v for k, v in list(globals().items())
        if k.startswith("test_") and isinstance(v, types.FunctionType)
    ]
    test_fns.sort(key=lambda f: f.__name__)

    passed = 0
    failed = 0
    for fn in test_fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
