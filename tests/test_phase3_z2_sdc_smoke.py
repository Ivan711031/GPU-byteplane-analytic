#!/usr/bin/env python3
"""Phase 3-Z2: SDC Containment CPU Smoke.

Compares three lanes — raw_fp64_unchecked, raw_fp64_digest_hard_fail,
byteplane_sum32 — on the Y0 tiny fixture (1000 rows, 11 known values).

Acceptance (PRD §4):
  AC-Z2-1  byteplane_sum32 sdc_rate == 0
  AC-Z2-2  byteplane_sum32 cert_bound_failure_rate == 0
  AC-Z2-3  raw_fp64_unchecked sdc_rate > 0 (baseline)
  AC-Z2-4  raw_fp64_digest_hard_fail hard_fail_rate > 0 (detection works)
  AC-Z2-5  Deterministic: same inputs → same outputs
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_reliability_tiny_fixture import (
    generate_tiny_fixture_raw, convert_to_planes, generate_fault_plans,
    SCALE, TINY_N, ENCODED_FIXTURE_VALUES, FP64_ROUNDTRIPPABLE,
)
from phase3_z2_sdc_evaluator import (
    run_smoke,
    aggregate_events,
    LANE_RAW_UNCHECKED, LANE_RAW_DIGEST, LANE_BYTEPLANE,
    EVENT_SDC, EVENT_HARD_FAIL, EVENT_BOUNDED_DEGRADED,
    EVENT_CERT_BOUND_FAILURE, EVENT_CLEAN,
)

TAU_SDC = 0.0  # byteplane sdc_rate MUST be 0
R_VECTOR = [3, 2, 1, 1, 1, 1, 1, 1]
SEEDS = [0, 1]
FAULT_RATES = ["1e-03", "1e-02"]

TESTS_RUN: list[str] = []
TESTS_PASSED: list[str] = []
TESTS_FAILED: list[str] = []


def _setup_fixture(artifact_dir: Path, fault_plan_dir: Path,
                   fault_rates: list[str] | None = None) -> tuple[int, int]:
    if fault_rates is None:
        fault_rates = FAULT_RATES
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_path = artifact_dir / "tiny_raw.f64le.bin"
    raw_vals = generate_tiny_fixture_raw(raw_path)
    meta = convert_to_planes(raw_vals, SCALE, artifact_dir, raw_path)
    target_planes = list(range(8))
    generate_fault_plans(artifact_dir, fault_plan_dir,
                         target_planes, [float(fr) for fr in fault_rates],
                         SEEDS, meta)
    return meta["n_rows"], SCALE


# ── AC-Z2-1: byteplane_sum32 sdc_rate == 0 ──────────────────────────

def test_z2_1_byteplane_sdc_zero() -> None:
    """AC-Z2-1: byteplane_sum32 sdc_rate must be 0 (certified interval is never silent)."""
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "artifacts"
        f = Path(tmp) / "fault_plans"
        _setup_fixture(a, f)

        for fr in FAULT_RATES:
            result = run_smoke(a, f, "tiny_fixture", 1000, SCALE,
                               R_VECTOR, "graded", SEEDS, fr)
            m = result["lane_metrics"][LANE_BYTEPLANE]
            assert m["sdc_rate"] == TAU_SDC, (
                f"byteplane sdc_rate={m['sdc_rate']:.10e} != {TAU_SDC} "
                f"at fault_rate={fr}"
            )


def test_z2_1_byteplane_sdc_zero_uniform() -> None:
    """AC-Z2-1 variant: uniform_repair_fraction policy."""
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "artifacts"
        f = Path(tmp) / "fault_plans"
        _setup_fixture(a, f)

        for fr in FAULT_RATES:
            rv = [1, 1, 1, 1, 1, 1, 1, 1]
            result = run_smoke(a, f, "tiny_fixture", 1000, SCALE,
                               rv, "uniform_repair_fraction", SEEDS, fr)
            m = result["lane_metrics"][LANE_BYTEPLANE]
            assert m["sdc_rate"] == TAU_SDC, (
                f"byteplane sdc_rate={m['sdc_rate']:.10e} != {TAU_SDC} "
                f"(uniform) at fault_rate={fr}"
            )


# ── AC-Z2-2: byteplane_sum32 cert_bound_failure_rate == 0 ───────────

def test_z2_2_cert_bound_failure_zero() -> None:
    """AC-Z2-2: byteplane cert_bound_failure_rate must be 0 (CB invariant)."""
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "artifacts"
        f = Path(tmp) / "fault_plans"
        _setup_fixture(a, f)

        for fr in FAULT_RATES:
            result = run_smoke(a, f, "tiny_fixture", 1000, SCALE,
                               R_VECTOR, "graded", SEEDS, fr)
            m = result["lane_metrics"][LANE_BYTEPLANE]
            assert m["cert_bound_failure_rate"] == 0.0, (
                f"cert_bound_failure_rate={m['cert_bound_failure_rate']:.10e} "
                f"at fault_rate={fr}"
            )


# ── AC-Z2-3: raw_fp64_unchecked has SDC events ─────────────────────

def test_z2_3_raw_unchecked_has_sdc() -> None:
    """AC-Z2-3: raw_fp64_unchecked sdc_rate > 0 (baseline)."""
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "artifacts"
        f = Path(tmp) / "fault_plans"
        _setup_fixture(a, f)

        for fr in FAULT_RATES:
            result = run_smoke(a, f, "tiny_fixture", 1000, SCALE,
                               R_VECTOR, "graded", SEEDS, fr)
            m = result["lane_metrics"][LANE_RAW_UNCHECKED]
            # At least some faults produce noticeable SDC
            assert m["sdc_rate"] > 0, (
                f"raw_fp64_unchecked sdc_rate={m['sdc_rate']:.10e} == 0 "
                f"at fault_rate={fr}"
            )


# ── AC-Z2-4: raw_fp64_digest detects faults ────────────────────────

def test_z2_4_digest_detects_all() -> None:
    """AC-Z2-4: raw_fp64_digest_hard_fail detects all faults (hard_fail_rate > 0)."""
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "artifacts"
        f = Path(tmp) / "fault_plans"
        _setup_fixture(a, f)

        for fr in FAULT_RATES:
            result = run_smoke(a, f, "tiny_fixture", 1000, SCALE,
                               R_VECTOR, "graded", SEEDS, fr)
            m = result["lane_metrics"][LANE_RAW_DIGEST]
            assert m["hard_fail_rate"] > 0, (
                f"raw_digest hard_fail_rate={m['hard_fail_rate']:.10e} == 0 "
                f"at fault_rate={fr}"
            )


# ── AC-Z2-5: Determinism ──────────────────────────────────────────

def test_z2_5_deterministic() -> None:
    """AC-Z2-5: Same inputs → same outputs for all lanes."""
    class _Result:
        pass

    results: list[dict] = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "artifacts"
            f = Path(tmp) / "fault_plans"
            _setup_fixture(a, f)
            results.append(run_smoke(a, f, "tiny_fixture", 1000, SCALE,
                                     R_VECTOR, "graded", SEEDS, "1e-03"))

    r0, r1 = results
    for lane in [LANE_RAW_UNCHECKED, LANE_RAW_DIGEST, LANE_BYTEPLANE]:
        m0 = r0["lane_metrics"][lane]
        m1 = r1["lane_metrics"][lane]
        for k in m0:
            assert m0[k] == m1[k], (
                f"{lane} metric {k}: {m0[k]} != {m1[k]}"
            )


# ── Test runner ─────────────────────────────────────────────────────

def _run_test(fn) -> None:
    name = fn.__name__
    TESTS_RUN.append(name)
    try:
        fn()
        print(f"  ✓ {name}")
        TESTS_PASSED.append(name)
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        traceback.print_exc()
        TESTS_FAILED.append(name)


def main() -> None:
    print("Phase 3-Z2: SDC Containment CPU Smoke")
    print("=" * 50)

    print("\n--- AC-Z2-1: byteplane sdc_rate == 0 ---")
    _run_test(test_z2_1_byteplane_sdc_zero)
    _run_test(test_z2_1_byteplane_sdc_zero_uniform)

    print("\n--- AC-Z2-2: cert_bound_failure_rate == 0 ---")
    _run_test(test_z2_2_cert_bound_failure_zero)

    print("\n--- AC-Z2-3: raw_fp64_unchecked has SDC ---")
    _run_test(test_z2_3_raw_unchecked_has_sdc)

    print("\n--- AC-Z2-4: raw_fp64_digest detects ---")
    _run_test(test_z2_4_digest_detects_all)

    print("\n--- AC-Z2-5: Deterministic ---")
    _run_test(test_z2_5_deterministic)

    total = len(TESTS_RUN)
    passed = len(TESTS_PASSED)
    failed = len(TESTS_FAILED)
    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed:
        for name in TESTS_FAILED:
            print(f"  FAILED: {name}")
    else:
        print("  ALL PASSED")
        print("\nZ2 CPU smoke: SDC containment validated on tiny fixture")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
