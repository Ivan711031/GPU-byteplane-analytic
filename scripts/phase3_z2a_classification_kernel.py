#!/usr/bin/env python3
"""Phase 3-Z2-A: Four-class SDC classification kernel prototype (CPU).

Provides classify_fault() primitive and run_classification() for
reproducing the Z2 evaluator's four-class taxonomy. Designed as parity
reference for the CUDA kernel (Z2-B/Z2-C).

Per PRD §4 taxonomy:
  sdc                  silent wrong answer (no detection, or SUM32 escape)
  cert_bound_failure   certified interval misses truth
  hard_fail            detected availability loss (no usable answer)
  bounded_degraded     detected + interval contains truth
  undetected           byteplane SUM32 escape (separate tracking metric)
  correct              no error
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase2_oracle import apply_fault_plan
from phase3_y0_evaluator import (
    compute_clean_sum,
    compute_delivered_answer_with_degradation,
    compute_voted_planes,
)
from phase3_z2_sdc_evaluator import (
    sum32,
    LANE_RAW_UNCHECKED,
    LANE_RAW_DIGEST,
    LANE_BYTEPLANE,
    EVENT_SDC,
    EVENT_CERT_BOUND_FAILURE,
    EVENT_HARD_FAIL,
    EVENT_BOUNDED_DEGRADED,
    EVENT_UNDETECTED,
    EVENT_CLEAN,
)


@dataclass
class ClassificationResult:
    lane: str
    n_injected: int
    sdc_count: int = 0
    cert_bound_failure_count: int = 0
    hard_fail_count: int = 0
    bounded_degraded_count: int = 0
    detected_count: int = 0
    undetected_count: int = 0

    @property
    def sdc_rate(self) -> float:
        return self.sdc_count / self.n_injected if self.n_injected else 0.0

    @property
    def cert_bound_failure_rate(self) -> float:
        return self.cert_bound_failure_count / self.n_injected if self.n_injected else 0.0

    @property
    def hard_fail_rate(self) -> float:
        return self.hard_fail_count / self.n_injected if self.n_injected else 0.0

    @property
    def bounded_degraded_rate(self) -> float:
        return self.bounded_degraded_count / self.n_injected if self.n_injected else 0.0

    @property
    def detected_rate(self) -> float:
        return self.detected_count / self.n_injected if self.n_injected else 0.0

    @property
    def undetected_rate(self) -> float:
        return self.undetected_count / self.n_injected if self.n_injected else 0.0

    @property
    def clean_rate(self) -> float:
        total_sdc = self.sdc_count + self.cert_bound_failure_count
        total_hf = self.hard_fail_count
        total_bd = self.bounded_degraded_count
        total_und = self.undetected_count
        accounted = total_sdc + total_hf + total_bd + total_und
        clean = self.n_injected - accounted
        return clean / self.n_injected if self.n_injected else 0.0

    def add(self, event: str) -> None:
        if event == EVENT_SDC:
            self.sdc_count += 1
        elif event == EVENT_CERT_BOUND_FAILURE:
            self.cert_bound_failure_count += 1
        elif event == EVENT_HARD_FAIL:
            self.hard_fail_count += 1
        elif event == EVENT_BOUNDED_DEGRADED:
            self.bounded_degraded_count += 1
        elif event == EVENT_UNDETECTED:
            self.undetected_count += 1

    def mark_detected(self, detected: bool) -> None:
        if detected:
            self.detected_count += 1

    def metrics(self) -> dict[str, float]:
        return {
            "sdc_rate": self.sdc_rate,
            "cert_bound_failure_rate": self.cert_bound_failure_rate,
            "hard_fail_rate": self.hard_fail_rate,
            "detected_rate": self.detected_rate,
            "bounded_degraded_rate": self.bounded_degraded_rate,
            "undetected_rate": self.undetected_rate,
            "clean_rate": self.clean_rate,
        }


def classify_fault(
    lane: str,
    clean_answer: float,
    delivered_answer: float,
    bound_width: float,
    certified: bool,
    detected: bool,
) -> tuple[str, str]:
    """Return (event_class, detail) for one injected fault.

    Per PRD §4 taxonomy, matching phase3_z2_sdc_evaluator per-lane
    classifier logic exactly.

    Returns one of:
      ("sdc", ...)                 silent wrong answer
      ("cert_bound_failure", ...)  certified interval misses truth
      ("hard_fail", ...)           detected availability loss
      ("bounded_degraded", ...)    detected + interval contains truth
      ("undetected", ...)          byteplane SUM32 escape
      ("correct", ...)             no error
    """
    if lane == LANE_RAW_UNCHECKED:
        if delivered_answer != clean_answer:
            return (EVENT_SDC, "silent wrong")
        return (EVENT_CLEAN, "no error")

    if lane == LANE_RAW_DIGEST:
        if detected:
            return (EVENT_HARD_FAIL, "availability loss")
        if delivered_answer != clean_answer:
            return (EVENT_SDC, "escape: undetected wrong answer")
        return (EVENT_CLEAN, "no error")

    if not detected:
        return (EVENT_UNDETECTED, "sum32 escape")

    lo = delivered_answer - bound_width / 2.0
    hi = delivered_answer + bound_width / 2.0
    if lo <= clean_answer <= hi:
        return (EVENT_BOUNDED_DEGRADED, "interval contains truth")
    return (EVENT_CERT_BOUND_FAILURE, "interval misses truth")


def per_plane_sum32(planes: list[bytes]) -> list[int]:
    """Per-plane SUM32 digest matching phase3_z2_sdc_evaluator.per_plane_sum32."""
    return [sum32(p) for p in planes]


def compute_sum32_digest(
    plane_bytes: list[bytes], segment_size: int = 4096
) -> dict[tuple[int, int], int]:
    """Compute per-segment SUM32 digest.

    Returns dict[(plane_idx, seg_idx)] -> sum32 value.
    Matches Z0f parallel warp-block reduction: bytewise accumulation
    into uint32, segmented at voting segment boundaries.
    """
    n_rows = len(plane_bytes[0])
    n_segments = (n_rows + segment_size - 1) // segment_size
    digest: dict[tuple[int, int], int] = {}
    for p, plane in enumerate(plane_bytes):
        for seg_idx in range(n_segments):
            start = seg_idx * segment_size
            end = min(start + segment_size, n_rows)
            digest[(p, seg_idx)] = sum32(plane[start:end])
    return digest


def _classify_one_injection(
    lane: str,
    clean_planes: list[bytes],
    fault_plan_paths: dict[int, list[str]],
    r_vector: list[int],
    scale: int,
    n_rows: int,
    ref_sum32: list[int],
    dataset: str = "tiny_fixture",
    policy: str = "graded",
    allocation_r: str = "",
    seed: int = 0,
    fault_rate: str = "1e-03",
    segment_size: int = 1024,
    **y0_kw: Any,
) -> tuple[str, float | None]:
    """Classify one fault injection across a single lane.

    Mirrors the three per-lane classify_* functions from
    phase3_z2_sdc_evaluator exactly.  Returns (event_class, delivered_answer).
    """
    if lane == LANE_RAW_UNCHECKED:
        faulted = []
        for p in range(8):
            paths = fault_plan_paths.get(p, [])
            faulted.append(
                apply_fault_plan(clean_planes[p], paths[0]) if paths else clean_planes[p]
            )
        clean_ans = compute_clean_sum(clean_planes) / scale
        fault_ans = compute_clean_sum(faulted) / scale
        return (EVENT_SDC, fault_ans) if fault_ans != clean_ans else (EVENT_CLEAN, fault_ans)

    if lane == LANE_RAW_DIGEST:
        faulted = []
        for p in range(8):
            paths = fault_plan_paths.get(p, [])
            faulted.append(
                apply_fault_plan(clean_planes[p], paths[0]) if paths else clean_planes[p]
            )
        if any(f != r for f, r in zip(per_plane_sum32(faulted), ref_sum32)):
            return (EVENT_HARD_FAIL, None)
        clean_ans = compute_clean_sum(clean_planes) / scale
        fault_ans = compute_clean_sum(faulted) / scale
        return (EVENT_SDC, fault_ans) if fault_ans != clean_ans else (EVENT_CLEAN, fault_ans)

    delivered = compute_delivered_answer_with_degradation(
        clean_planes=clean_planes,
        fault_plan_paths=fault_plan_paths,
        r_vector=r_vector,
        scale=scale,
        n_rows=n_rows,
        segment_size=segment_size,
        dataset=dataset,
        policy=policy,
        allocation_r=allocation_r,
        seed=seed,
        fault_rate=fault_rate,
        **y0_kw,
    )

    faulted_voted = compute_voted_planes(clean_planes, fault_plan_paths, r_vector)
    detected = any(f != r for f, r in zip(per_plane_sum32(faulted_voted), ref_sum32))
    clean_ans = compute_clean_sum(clean_planes) / scale
    return classify_fault(
        lane=LANE_BYTEPLANE,
        clean_answer=clean_ans,
        delivered_answer=delivered.delivered_answer,
        bound_width=delivered.bound_width,
        certified=delivered.certified,
        detected=detected,
    )


def run_classification(
    clean_planes: list[bytes],
    fault_plan_paths: dict[int, list[str]],
    r_vector: list[int],
    scale: int,
    n_rows: int,
    lanes: list[str] | None = None,
    ref_sum32: list[int] | None = None,
    dataset: str = "tiny_fixture",
    policy: str = "graded",
    allocation_r: str = "",
    seed: int = 0,
    fault_rate: str = "1e-03",
    segment_size: int = 1024,
    **y0_kw: Any,
) -> dict[str, ClassificationResult]:
    """Run four-class classification for all specified lanes on one injection.

    Returns dict[lane] -> ClassificationResult with counts for this single
    injection.  Caller aggregates across injections.
    """
    if lanes is None:
        lanes = [LANE_RAW_UNCHECKED, LANE_RAW_DIGEST, LANE_BYTEPLANE]
    if ref_sum32 is None:
        ref_sum32 = per_plane_sum32(clean_planes)

    results: dict[str, ClassificationResult] = {}
    for lane in lanes:
        event, _ = _classify_one_injection(
            lane=lane,
            clean_planes=clean_planes,
            fault_plan_paths=fault_plan_paths,
            r_vector=r_vector,
            scale=scale,
            n_rows=n_rows,
            ref_sum32=ref_sum32,
            dataset=dataset,
            policy=policy,
            allocation_r=allocation_r,
            seed=seed,
            fault_rate=fault_rate,
            segment_size=segment_size,
            **y0_kw,
        )
        cr = ClassificationResult(lane=lane, n_injected=1)
        cr.add(event)
        if lane in (LANE_RAW_DIGEST, LANE_BYTEPLANE):
            detected_event = event in (EVENT_HARD_FAIL, EVENT_BOUNDED_DEGRADED, EVENT_CERT_BOUND_FAILURE)
            cr.mark_detected(detected_event)
        results[lane] = cr

    return results


def run_classification_matrix(
    clean_planes: list[bytes],
    fault_plan_dir: Path,
    r_vector: list[int],
    scale: int,
    n_rows: int,
    lanes: list[str] | None = None,
    seeds: list[int] | None = None,
    fault_rates: list[str] | None = None,
    target_planes: list[int] | None = None,
    dataset: str = "tiny_fixture",
    policy: str = "graded",
    segment_size: int = 1024,
) -> dict[str, ClassificationResult]:
    """Run classification across all (plane, seed, fault_rate) injections.

    Aggregate four-class counts per lane across the full matrix,
    reproducing the existing evaluator's per-lane event aggregation.

    Returns dict[lane] -> ClassificationResult with aggregated counts.
    """
    if lanes is None:
        lanes = [LANE_RAW_UNCHECKED, LANE_RAW_DIGEST, LANE_BYTEPLANE]
    if seeds is None:
        seeds = [0, 1]
    if fault_rates is None:
        fault_rates = ["1e-03"]
    if target_planes is None:
        target_planes = list(range(8))

    ref_sum32 = per_plane_sum32(clean_planes)
    allocation_r = "|".join(str(r) for r in r_vector)

    aggregated = {lane: ClassificationResult(lane=lane, n_injected=0) for lane in lanes}

    for plane, seed, fr in itertools.product(target_planes, seeds, fault_rates):
        paths: dict[int, list[str]] = {}
        for rep in range(r_vector[plane]):
            fp = (
                fault_plan_dir / f"plane{plane}" / f"rate{fr}"
                / f"seed_{seed}.json"
            )
            if fp.is_file():
                paths.setdefault(plane, []).append(str(fp))
        if not paths:
            continue

        per_lane = run_classification(
            clean_planes=clean_planes,
            fault_plan_paths=paths,
            r_vector=r_vector,
            scale=scale,
            n_rows=n_rows,
            lanes=lanes,
            ref_sum32=ref_sum32,
            dataset=dataset,
            policy=policy,
            allocation_r=allocation_r,
            seed=seed,
            fault_rate=fr,
            segment_size=segment_size,
        )

        for lane in lanes:
            cr = per_lane[lane]
            aggregated[lane].n_injected += cr.n_injected
            aggregated[lane].sdc_count += cr.sdc_count
            aggregated[lane].cert_bound_failure_count += cr.cert_bound_failure_count
            aggregated[lane].hard_fail_count += cr.hard_fail_count
            aggregated[lane].bounded_degraded_count += cr.bounded_degraded_count
            aggregated[lane].detected_count += cr.detected_count
            aggregated[lane].undetected_count += cr.undetected_count

    return aggregated
