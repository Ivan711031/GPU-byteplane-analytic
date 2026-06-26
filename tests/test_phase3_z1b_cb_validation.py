#!/usr/bin/env python3
"""Phase 3-Z1b: Certified-Bound Injection Validation (CPU smoke).

Validates the CB claim for SUM (unfiltered):
  AC-Z1b-1  contains_truth == 1.0 for each plane i corrupted unrepaired
  AC-Z1b-2  bound_width_prediction_err <= tau_pred (0.10) for each plane i
  AC-Z1b-3  Monotonicity: deeper-plane fault yields narrower bound than shallower
  AC-Z1b-4  Degraded predicate-membership envelope (unfiltered SUM: trivially satisfied)
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from phase2_oracle import apply_fault_plan
from phase3_y0_evaluator import (
    compute_delivered_answer_with_degradation,
    compute_ui_prediction,
    bound_width_prediction_err,
)

TAU_PRED = 0.10
N_ROWS = 8
SCALE = 100
SEGMENT_SIZE = 1024

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
            "fault_rate_numeric": float(len(entries)) / N_ROWS if N_ROWS > 0 else 0,
            "seed": 0,
            "actual_fault_count": len(entries),
            "plane_size_bytes": 0,
        },
        "entries": list(entries),
    }
    path.write_text(json.dumps(fp) + "\n")


def _make_clean_planes(n_rows: int, byte_val: int = 0x42) -> list[bytes]:
    return [bytes([byte_val] * n_rows) for _ in range(8)]


def contains_truth(
    clean_answer: float, delivered_answer: float, bound_width: float
) -> bool:
    lo = delivered_answer - bound_width / 2.0
    hi = delivered_answer + bound_width / 2.0
    return lo <= clean_answer <= hi


#
# ── AC-Z1b-1: contains_truth == 1.0 for each plane i ──────────────────
#


def test_z1b1_contains_truth_each_plane() -> None:
    """AC-Z1b-1: For each plane i, corrupt unrepaired and verify interval contains clean answer."""
    clean_planes = _make_clean_planes(N_ROWS)
    clean_sum = sum(
        sum(clean_planes[p]) * (1 << (8 * (7 - p))) for p in range(8)
    )
    clean_answer = clean_sum / SCALE

    results: dict[int, bool] = {}

    for plane in range(8):
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir = Path(tmp) / "fault_plans"
            _make_fault_plan(
                plan_dir / f"plane{plane}/replica0/seed_0.json",
                [{"offset": 3, "mask": 0xFF}],
            )

            fpp = {plane: [str(plan_dir / f"plane{plane}/replica0/seed_0.json")]}
            r_vector = [1 if p == plane else 1 for p in range(8)]

            result = compute_delivered_answer_with_degradation(
                clean_planes, fpp, r_vector,
                scale=SCALE, n_rows=N_ROWS,
                dataset="z1b_contains_truth", policy="uniform_repair_fraction",
                allocation_r="|".join(str(r) for r in r_vector),
                segment_size=SEGMENT_SIZE,
            )

        ct = contains_truth(clean_answer, result.delivered_answer, result.bound_width)
        results[plane] = ct

    all_pass = all(results.values())
    failures = [p for p, v in results.items() if not v]
    assert all_pass, (
        f"contains_truth failed for planes: {failures}"
    )


def test_z1b1_contains_truth_saturating_mask() -> None:
    """AC-Z1b-1 variant: saturating mask (0xFF) on each plane."""
    clean_planes = _make_clean_planes(N_ROWS)
    clean_sum = sum(
        sum(clean_planes[p]) * (1 << (8 * (7 - p))) for p in range(8)
    )
    clean_answer = clean_sum / SCALE

    results: dict[int, bool] = {}

    for plane in range(8):
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir = Path(tmp) / "fault_plans"
            _make_fault_plan(
                plan_dir / f"plane{plane}/replica0/seed_0.json",
                [{"offset": 0, "mask": 0xFF}],
            )

            fpp = {plane: [str(plan_dir / f"plane{plane}/replica0/seed_0.json")]}
            r_vector = [1 if p == plane else 1 for p in range(8)]

            result = compute_delivered_answer_with_degradation(
                clean_planes, fpp, r_vector,
                scale=SCALE, n_rows=N_ROWS,
                dataset="z1b_saturating", policy="uniform_repair_fraction",
                allocation_r="|".join(str(r) for r in r_vector),
                segment_size=SEGMENT_SIZE,
            )

        ct = contains_truth(clean_answer, result.delivered_answer, result.bound_width)
        results[plane] = ct

    all_pass = all(results.values())
    failures = [p for p, v in results.items() if not v]
    assert all_pass, (
        f"contains_truth (saturating) failed for planes: {failures}"
    )


#
# ── AC-Z1b-2: bound_width_prediction_err <= tau_pred ─────────────────
#


def test_z1b2_prediction_err_each_plane() -> None:
    """AC-Z1b-2: U_i prediction error within tau_pred for each corrupted plane."""
    clean_planes = _make_clean_planes(N_ROWS)
    errs: dict[int, float] = {}

    for plane in range(8):
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir = Path(tmp) / "fault_plans"
            _make_fault_plan(
                plan_dir / f"plane{plane}/replica0/seed_0.json",
                [{"offset": 0, "mask": 0xFF}],
            )

            fpp = {plane: [str(plan_dir / f"plane{plane}/replica0/seed_0.json")]}
            r_vector = [1 if p == plane else 1 for p in range(8)]

            observed = compute_delivered_answer_with_degradation(
                clean_planes, fpp, r_vector,
                scale=SCALE, n_rows=N_ROWS,
                dataset="z1b_pred_err", policy="uniform_repair_fraction",
                allocation_r="|".join(str(r) for r in r_vector),
                segment_size=SEGMENT_SIZE,
            ).bound_width

            pred = compute_ui_prediction(
                clean_planes, fpp, r_vector,
                scale=SCALE, n_rows=N_ROWS,
                segment_size=SEGMENT_SIZE,
            )

        err = bound_width_prediction_err(observed, pred["bound_width_predicted"])
        errs[plane] = err

        # Predicted must be >= observed (valid ceiling)
        assert pred["bound_width_predicted"] >= observed * (1.0 - 1e-12), (
            f"plane {plane}: predicted {pred['bound_width_predicted']} < observed {observed}"
        )

    for plane, err in errs.items():
        assert err <= TAU_PRED, (
            f"plane {plane}: err={err:.6e} > tau_pred={TAU_PRED}"
        )


def test_z1b2_prediction_err_multiple_offsets() -> None:
    """AC-Z1b-2: prediction error with multiple faulted bytes per plane."""
    clean_planes = _make_clean_planes(N_ROWS)
    errs: dict[int, float] = {}

    for plane in range(8):
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir = Path(tmp) / "fault_plans"
            _make_fault_plan(
                plan_dir / f"plane{plane}/replica0/seed_0.json",
                [{"offset": 0, "mask": 0x01},
                 {"offset": 1, "mask": 0x80},
                 {"offset": 2, "mask": 0xFF}],
            )

            fpp = {plane: [str(plan_dir / f"plane{plane}/replica0/seed_0.json")]}
            r_vector = [1 if p == plane else 1 for p in range(8)]

            observed = compute_delivered_answer_with_degradation(
                clean_planes, fpp, r_vector,
                scale=SCALE, n_rows=N_ROWS,
                dataset="z1b_multi_err", policy="uniform_repair_fraction",
                allocation_r="|".join(str(r) for r in r_vector),
                segment_size=SEGMENT_SIZE,
            ).bound_width

            pred = compute_ui_prediction(
                clean_planes, fpp, r_vector,
                scale=SCALE, n_rows=N_ROWS,
                segment_size=SEGMENT_SIZE,
            )

        err = bound_width_prediction_err(observed, pred["bound_width_predicted"])
        errs[plane] = err

    for plane, err in errs.items():
        assert err <= TAU_PRED, (
            f"plane {plane} (multi-offset): err={err:.6e} > tau_pred={TAU_PRED}"
        )


#
# ── AC-Z1b-3: Monotonicity ─────────────────────────────────────────
#


def test_z1b3_monotonicity() -> None:
    """AC-Z1b-3: Deeper-plane fault yields narrower bound than shallower-plane fault.

    Plane 0 (MSB, weight=2^56) should produce the widest bound,
    Plane 7 (LSB, weight=2^0) should produce the narrowest bound.
    """
    clean_planes = _make_clean_planes(N_ROWS)
    bound_widths: dict[int, float] = {}

    for plane in range(8):
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir = Path(tmp) / "fault_plans"
            _make_fault_plan(
                plan_dir / f"plane{plane}/replica0/seed_0.json",
                [{"offset": 0, "mask": 0xFF}],
            )

            fpp = {plane: [str(plan_dir / f"plane{plane}/replica0/seed_0.json")]}
            r_vector = [1 if p == plane else 1 for p in range(8)]

            result = compute_delivered_answer_with_degradation(
                clean_planes, fpp, r_vector,
                scale=SCALE, n_rows=N_ROWS,
                dataset="z1b_mono", policy="uniform_repair_fraction",
                allocation_r="|".join(str(r) for r in r_vector),
                segment_size=SEGMENT_SIZE,
            )

        bound_widths[plane] = result.bound_width

    for i in range(7):
        assert bound_widths[i] > bound_widths[i + 1], (
            f"monotonicity violation: plane {i} width={bound_widths[i]:.10e} "
            f"<= plane {i+1} width={bound_widths[i+1]:.10e}"
        )


#
# ── AC-Z1b-4: Degraded predicate-membership envelope ─────────────────
#


def test_z1b4_predicate_envelope_unfiltered_sum() -> None:
    """AC-Z1b-4: For unfiltered SUM only — predicate envelope trivial.

    SCOPE NOTE: This validates the predicate-membership envelope for unfiltered SUM
    on a tiny deterministic fixture only. The following are NOT yet validated:
    - filtered-SUM / COUNT_GT with threshold predicate
    - MAX / top-k functionals
    - k < 8 depth truncation
    - degraded-repair mode (replica exhaustion without repair)
    - real scientific datasets (cesm, hurricane)

    For unfiltered SUM, every row unconditionally contributes; the conservative
    envelope (active_count = n_rows) trivially bounds all membership uncertainty.
    Verified by checking contains_truth holds even when the corrupted plane
    significantly changes the decoded value (mask=0xFF on all bytes).
    """
    clean_planes = _make_clean_planes(N_ROWS)
    clean_sum = sum(
        sum(clean_planes[p]) * (1 << (8 * (7 - p))) for p in range(8)
    )
    clean_answer = clean_sum / SCALE

    results: dict[int, bool] = {}

    for plane in range(8):
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir = Path(tmp) / "fault_plans"

            _make_fault_plan(
                plan_dir / f"plane{plane}/replica0/seed_0.json",
                [{"offset": i, "mask": 0xFF} for i in range(N_ROWS)],
            )

            fpp = {plane: [str(plan_dir / f"plane{plane}/replica0/seed_0.json")]}
            r_vector = [1 if p == plane else 1 for p in range(8)]

            result = compute_delivered_answer_with_degradation(
                clean_planes, fpp, r_vector,
                scale=SCALE, n_rows=N_ROWS,
                dataset="z1b_pred_env", policy="uniform_repair_fraction",
                allocation_r="|".join(str(r) for r in r_vector),
                segment_size=SEGMENT_SIZE,
            )

        ct = contains_truth(clean_answer, result.delivered_answer, result.bound_width)
        results[plane] = ct

    all_pass = all(results.values())
    failures = [p for p, v in results.items() if not v]
    assert all_pass, (
        f"predicate envelope (all-bytes corrupted) failed for planes: {failures}"
    )


#
# ── Test runner ─────────────────────────────────────────────────────
#


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


def write_csv(
    csv_path: Path,
    clean_planes: list[bytes],
    fault_configs: list[dict],
) -> None:
    """Write Z1b bound validation CSV."""
    clean_sum = sum(
        sum(clean_planes[p]) * (1 << (8 * (7 - p))) for p in range(8)
    )
    clean_answer = clean_sum / SCALE

    fieldnames = [
        "dataset", "n_rows", "scale", "plane", "fault_offsets", "mask",
        "policy", "allocation_r",
        "clean_answer", "delivered_answer",
        "bound_width_observed", "bound_width_predicted",
        "bound_width_prediction_err",
        "contains_truth",
        "clean_encoded_sum",
    ]

    rows: list[dict[str, str]] = []

    for cfg in fault_configs:
        plane = cfg["plane"]
        entries = cfg["entries"]
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir = Path(tmp) / "fault_plans"
            _make_fault_plan(
                plan_dir / f"plane{plane}/replica0/seed_0.json",
                entries,
            )

            fpp = {plane: [str(plan_dir / f"plane{plane}/replica0/seed_0.json")]}
            r_vector = [1 if p == plane else 1 for p in range(8)]

            result = compute_delivered_answer_with_degradation(
                clean_planes, fpp, r_vector,
                scale=SCALE, n_rows=N_ROWS,
                dataset="tiny_fixture", policy="uniform_repair_fraction",
                allocation_r="|".join(str(r) for r in r_vector),
                segment_size=SEGMENT_SIZE,
            )

            pred = compute_ui_prediction(
                clean_planes, fpp, r_vector,
                scale=SCALE, n_rows=N_ROWS,
                segment_size=SEGMENT_SIZE,
            )

        err = bound_width_prediction_err(
            result.bound_width, pred["bound_width_predicted"]
        )
        ct = contains_truth(
            clean_answer, result.delivered_answer, result.bound_width
        )

        rows.append({
            "dataset": "tiny_fixture",
            "n_rows": str(N_ROWS),
            "scale": str(SCALE),
            "plane": str(plane),
            "fault_offsets": ",".join(str(e["offset"]) for e in entries),
            "mask": hex(entries[0]["mask"]) if len(entries) == 1 else "multi",
            "policy": "uniform_repair_fraction",
            "allocation_r": "|".join(str(1) for _ in range(8)),
            "clean_answer": str(clean_answer),
            "delivered_answer": str(result.delivered_answer),
            "bound_width_observed": str(result.bound_width),
            "bound_width_predicted": str(pred["bound_width_predicted"]),
            "bound_width_prediction_err": f"{err:.10e}",
            "contains_truth": str(ct).lower(),
            "clean_encoded_sum": str(clean_sum),
        })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written: {csv_path} ({len(rows)} rows)")


def main() -> None:
    print("Phase 3-Z1b: Certified-Bound Injection Validation (CPU smoke)")
    print("=" * 60)

    #
    # ── Run unit tests ──
    #
    print("\n--- AC-Z1b-1: contains_truth == 1.0 ---")
    _run_test(test_z1b1_contains_truth_each_plane)
    _run_test(test_z1b1_contains_truth_saturating_mask)

    print("\n--- AC-Z1b-2: bound_width_prediction_err <= tau_pred ---")
    _run_test(test_z1b2_prediction_err_each_plane)
    _run_test(test_z1b2_prediction_err_multiple_offsets)

    print("\n--- AC-Z1b-3: Monotonicity ---")
    _run_test(test_z1b3_monotonicity)

    print("\n--- AC-Z1b-4: Predicate envelope (unfiltered SUM) ---")
    _run_test(test_z1b4_predicate_envelope_unfiltered_sum)

    #
    # ── Summary ──
    #
    total = len(TESTS_RUN)
    passed = len(TESTS_PASSED)
    failed = len(TESTS_FAILED)
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed:
        for name in TESTS_FAILED:
            print(f"  FAILED: {name}")
    else:
        print("  ALL PASSED")

    #
    # ── Write CSV ──
    #
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "cpu_smoke")
    csv_dir = Path("results/reliability_layer1/phase3/phase3z_z1b") / f"job_{slurm_job_id}"
    csv_path = csv_dir / "z1b_bound_validation.csv"

    clean_planes = _make_clean_planes(N_ROWS)
    fault_configs = []
    for plane in range(8):
        fault_configs.append({
            "plane": plane,
            "entries": [{"offset": 3, "mask": 0xFF}],
        })
    for plane in range(8):
        fault_configs.append({
            "plane": plane,
            "entries": [{"offset": 0, "mask": 0xFF}],
        })

    write_csv(csv_path, clean_planes, fault_configs)

    if failed:
        sys.exit(1)

    print("\nZ1b CPU smoke: CB claim validated on tiny fixture (SUM, unfiltered)")
    print("Verdict: CB_SUPPORTED (subject to GPU validation on full fields)")
    print()
    print("Open validation items (not yet tested):")
    print("  - filtered-SUM / COUNT_GT with threshold predicate")
    print("  - MAX / top-k functionals")
    print("  - k < 8 depth truncation")
    print("  - degraded-repair mode (replica exhaustion)")
    print("  - real scientific datasets (cesm, hurricane)")


if __name__ == "__main__":
    main()
