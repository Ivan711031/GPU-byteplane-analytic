#!/usr/bin/env python3
"""NMR-Claim1 realistic fault campaign.

Implements the complete Claim 1 campaign pipeline:
  8 fault families (F1-F8) × 20 NMR policies × rate anchors × seeds

Outputs 6 CSV files:
  - claim1_realistic_pilot_matrix.csv
  - claim1_realistic_headline_matrix.csv
  - claim1_realistic_policy_frontier.csv
  - claim1_realistic_fault_family_summary.csv
  - claim1_realistic_exposure_summary.csv
  - claim1_realistic_outcome_summary.csv

Audit #285 findings (2026-06-16):
  (1) Float precision: compute_encoded_sum now returns Python `int`
      (unbounded) instead of `float`.  At n_rows>=5M the float64
      ulp (~1e10) exceeds the delta from low-significance planes
      (5-7), causing silent_wrong → exact_correct misclassification.
      All answer comparisons use exact integer equality.
  (2) has_r2 ANY-gate: classify_replication_outcome uses ANY not ALL
      for protection detection.  One protected plane gates the whole
      cell.  Documented in the function docstring (not changed).

Usage:
  python3 scripts/nmr_claim1_realistic_campaign.py \
    --artifact-dir /path/to/seg4096 \
    --dataset cesm_atm_cloud \
    --n-rows 25000000 \
    --segment-size 4096 \
    --mode detect_and_bound | replication_only \
    --fault-families F1 F2 F3 F4 F5 F6 F7 F8 \
    --rate-anchors 1e-9 1e-7 1e-5 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --output-dir /tmp/claim1_realistic
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import struct
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PLANE_COUNT = 8
PLANE_WEIGHTS = [1 << (8 * (7 - p)) for p in range(PLANE_COUNT)]
SEGMENT_SIZE = 4096
SIGNIFICANCE_RANKING = [0, 1, 2, 3, 4, 5, 6, 7]

# ---------------------------------------------------------------------------
# Pure-Python utilities (no numpy/GPU)
# ---------------------------------------------------------------------------

def popcount(x: int) -> int:
    return bin(x).count("1")


def sum32(data: bytes) -> int:
    total = 0
    aligned = len(data) & ~3
    for val in struct.iter_unpack('<I', data[:aligned]):
        total += val[0]
    r = len(data) - aligned
    if r:
        last = data[aligned:] + b'\x00' * (4 - r)
        total += struct.unpack('<I', last)[0]
    return total & 0xFFFFFFFF


def per_plane_sum32(planes: list[bytes]) -> list[int]:
    return [sum32(p) for p in planes]


def load_planes(artifact_dir: Path, n_rows: int) -> list[bytes]:
    planes: list[bytes] = []
    for p in range(PLANE_COUNT):
        path = artifact_dir / f"plane_{p:03d}.bin"
        if path.is_file():
            data = path.read_bytes()[:n_rows]
        else:
            data = bytes(n_rows)
        if len(data) < n_rows:
            data = data + bytes(n_rows - len(data))
        planes.append(data)
    return planes


def compute_encoded_sum(planes: list[bytes] | list[bytearray]) -> int:
    """Return exact integer encoded sum.  Python int is unbounded,
    so there is NO precision loss regardless of n_rows or plane weight.
    (Fix for audit #285: was previously float(sum(...)), which loses
    the lowest 3-4 planes when n_rows >= 5M.)"""
    return sum(sum(p) * PLANE_WEIGHTS[pi] for pi, p in enumerate(planes))


def compute_per_segment_bound(
    detected_planes: list[bool],
    n_rows: int,
    segment_size: int,
) -> float:
    n_segments = (n_rows + segment_size - 1) // segment_size
    total = 0.0
    for seg in range(n_segments):
        seg_start = seg * segment_size
        seg_end = min(seg_start + segment_size, n_rows)
        seg_len = seg_end - seg_start
        for p in range(PLANE_COUNT):
            if detected_planes[p]:
                total += 255.0 * PLANE_WEIGHTS[p] * seg_len
    return total


# ---------------------------------------------------------------------------
# Policy generators
# ---------------------------------------------------------------------------

def make_uniform_policy(B: int) -> list[int]:
    """Return extra copies per plane (sum = B).
    Total replicas = 1 + extra per plane.
    """
    base_extra = B // 8
    rem = B % 8
    return [base_extra + (1 if i < rem else 0) for i in range(8)]


def make_graded_policy(B: int) -> list[int]:
    """Return extra copies per plane for graded allocation at budget B.

    Algorithm: high-significance-first.  Plane 0 is filled to r=3
    (2 extra copies) before plane 1 receives any extra.  Max 2 extra
    copies per plane, so max total replicas per plane = 3.

    Returns extras list (sum = B).  Total replicas = 1 + extras[p].
    """
    extra = [0] * 8
    remaining = B
    for p in range(8):
        give = remaining if remaining < 2 else 2
        extra[p] = give
        remaining -= give
        if remaining == 0:
            break
    return extra


# ---------------------------------------------------------------------------
# Fault plan data model
# ---------------------------------------------------------------------------

@dataclass
class FaultEvent:
    plane: int
    offset: int
    mask: int
    target_replicas: list[int] | None = None


@dataclass
class RealizedFaultPlan:
    events: list[FaultEvent]
    fault_family: str
    rate: float
    seed: int
    event_count: int
    attempted_mutated_byte_count: int
    unique_mutated_byte_count: int
    bit_flip_count: int
    affected_segment_count: int
    affected_plane_count: int
    replica_overlap_count: int
    same_plane_multi_replica_count: int
    significance_weighted_impact_budget: float
    xor_cancellation_count: int


# ---------------------------------------------------------------------------
# Fault family implementations (F1-F8)
# All CPU-only, seeded random.Random for determinism.
# ---------------------------------------------------------------------------

def _fault_family_F1(
    rng: random.Random, n_rows: int, rate: float, severity_params: dict,
) -> list[FaultEvent]:
    n_events = max(1, int(n_rows * rate)) if rate > 0 else 0
    events: list[FaultEvent] = []
    for _ in range(n_events):
        plane = rng.randint(0, PLANE_COUNT - 1)
        offset = rng.randint(0, n_rows - 1)
        mask = rng.randint(1, 255)
        events.append(FaultEvent(plane=plane, offset=offset, mask=mask))
    return events


def _fault_family_F2(
    rng: random.Random, n_rows: int, rate: float, severity_params: dict,
) -> list[FaultEvent]:
    burst_min = severity_params.get("burst_min", 2)
    burst_max = severity_params.get("burst_max", 4)
    burst_max = min(burst_max, n_rows - 1)
    burst_min = min(burst_min, burst_max)
    burst_len = rng.randint(burst_min, burst_max) if burst_min < burst_max else burst_max
    n_events = max(1, int(n_rows * rate)) if rate > 0 else 0
    n_events = min(n_events, 100)
    events: list[FaultEvent] = []
    for _ in range(n_events):
        plane = rng.randint(0, PLANE_COUNT - 1)
        start = rng.randint(0, max(0, n_rows - burst_len - 1))
        for i in range(burst_len):
            mask = rng.randint(1, 255)
            events.append(FaultEvent(plane=plane, offset=start + i, mask=mask))
    return events


def _fault_family_F3(
    rng: random.Random, n_rows: int, rate: float, severity_params: dict,
) -> list[FaultEvent]:
    run_length = severity_params.get("run_length", min(64, n_rows))
    run_length = min(run_length, n_rows)
    n_bursts = max(1, int(n_rows * rate)) if rate > 0 else 0
    n_bursts = min(n_bursts, 100)
    events: list[FaultEvent] = []
    for _ in range(n_bursts):
        plane = rng.randint(0, PLANE_COUNT - 1)
        start = rng.randint(0, max(0, n_rows - run_length))
        for i in range(run_length):
            mask = rng.randint(1, 255)
            events.append(FaultEvent(plane=plane, offset=start + i, mask=mask))
    return events


def _fault_family_F4(
    rng: random.Random, n_rows: int, rate: float, severity_params: dict,
) -> list[FaultEvent]:
    """Column-like repeated offset corruption.

    Same byte offset repeated across multiple 4096-byte segments,
    simulating a column-like structured fault.  Uses segment-level
    iteration instead of per-row sampling to avoid O(n_rows) cost.
    """
    affected_fraction = severity_params.get("affected_row_fraction", 0.1)
    segment_size = 4096
    n_segments = (n_rows + segment_size - 1) // segment_size
    n_affected = max(1, int(n_segments * affected_fraction))
    n_affected = min(n_affected, n_segments)
    plane = severity_params.get("plane", rng.randint(0, PLANE_COUNT - 1))
    mask = severity_params.get("mask", rng.randint(1, 255))
    offset = severity_params.get("offset", rng.randint(0, 255))
    events: list[FaultEvent] = []
    affected_segments = rng.sample(range(n_segments), n_affected)
    for seg_idx in affected_segments:
        row = seg_idx * segment_size
        eff_offset = offset + seg_idx * segment_size
        if eff_offset >= n_rows:
            continue
        events.append(FaultEvent(plane=plane, offset=eff_offset, mask=mask))
    return events


def _fault_family_F5(
    rng: random.Random, n_rows: int, rate: float, severity_params: dict,
) -> list[FaultEvent]:
    density = severity_params.get("density", min(0.1, rate * 1000))
    region_size = severity_params.get("region_size", min(4096, n_rows))
    region_size = min(region_size, n_rows)
    n_regions = severity_params.get("n_regions", max(1, int(n_rows * rate)))
    n_regions = min(n_regions, 100)
    events: list[FaultEvent] = []
    for _ in range(n_regions):
        plane = rng.randint(0, PLANE_COUNT - 1)
        region_start = rng.randint(0, max(0, n_rows - region_size))
        n_corrupt = max(1, int(region_size * density))
        positions = rng.sample(
            range(region_start, region_start + region_size),
            min(n_corrupt, region_size),
        )
        for pos in positions:
            mask = rng.randint(1, 255)
            events.append(FaultEvent(plane=plane, offset=pos, mask=mask))
    return events


def _fault_family_F6(
    rng: random.Random, n_rows: int, rate: float, severity_params: dict,
) -> list[FaultEvent]:
    burst_events = severity_params.get("burst_events", max(1, int(rate * 1e6)))
    region_size = severity_params.get("region_size", min(1024, n_rows))
    region_size = min(region_size, n_rows)
    events: list[FaultEvent] = []
    plane = rng.randint(0, PLANE_COUNT - 1)
    region_start = rng.randint(0, max(0, n_rows - region_size))
    for _ in range(burst_events):
        pos = rng.randint(region_start, region_start + region_size - 1)
        pos = min(pos, n_rows - 1)
        mask = rng.randint(1, 255)
        events.append(FaultEvent(plane=plane, offset=pos, mask=mask))
    return events


def _fault_family_F7(
    rng: random.Random, n_rows: int, rate: float, severity_params: dict,
) -> list[FaultEvent]:
    target_replicas = severity_params.get("target_replicas", [0, 1])
    n_events = max(1, int(n_rows * rate)) if rate > 0 else 0
    events: list[FaultEvent] = []
    for _ in range(n_events):
        plane = rng.randint(0, PLANE_COUNT - 1)
        offset = rng.randint(0, n_rows - 1)
        mask = rng.randint(1, 255)
        events.append(FaultEvent(
            plane=plane, offset=offset, mask=mask,
            target_replicas=list(target_replicas),
        ))
    return events


def _fault_family_F8(
    rng: random.Random, n_rows: int, rate: float, severity_params: dict,
) -> list[FaultEvent]:
    f1_ratio = severity_params.get("f1_ratio", 0.5)
    f7_params = severity_params.get("f7_params", {"target_replicas": [0, 1]})
    f1_rng = random.Random(rng.randint(0, 2**31))
    f7_rng = random.Random(rng.randint(0, 2**31))
    f1_events = _fault_family_F1(f1_rng, n_rows, rate * f1_ratio, {})
    f7_events = _fault_family_F7(f7_rng, n_rows, rate * (1 - f1_ratio), f7_params)
    return f1_events + f7_events


_FAULT_FAMILIES: dict[str, Any] = {
    "F1": _fault_family_F1,
    "F2": _fault_family_F2,
    "F3": _fault_family_F3,
    "F4": _fault_family_F4,
    "F5": _fault_family_F5,
    "F6": _fault_family_F6,
    "F7": _fault_family_F7,
    "F8": _fault_family_F8,
}

_SEVERITY_DEFAULTS: dict[str, dict[str, Any]] = {
    "F1": {},
    "F2": {"burst_min": 2, "burst_max": 4},
    "F3": {"run_length": 64},
    "F4": {},
    "F5": {"density": 0.1, "region_size": 4096, "n_regions": 1},
    "F6": {"burst_events": 10, "region_size": 1024},
    "F7": {"target_replicas": [0, 1]},
    "F8": {"f1_ratio": 0.5, "f7_params": {"target_replicas": [0, 1]}},
}

_SEVERITY_SWEEP: dict[str, list[dict[str, Any]]] = {
    "F1": [{}],
    "F2": [
        {"burst_min": 2, "burst_max": 2},
        {"burst_min": 4, "burst_max": 4},
    ],
    "F3": [
        {"run_length": 8},
        {"run_length": 512},
    ],
    "F4": [
        {"affected_row_fraction": 0.1},
        {"affected_row_fraction": 1.0},
    ],
    "F5": [
        {"density": 0.5, "region_size": 4096, "n_regions": 1},
    ],
    "F6": [
        {"burst_events": 10, "region_size": 1024},
    ],
    "F7": [
        {"target_replicas": [0, 1]},
    ],
    "F8": [
        {"f1_ratio": 0.5, "f7_params": {"target_replicas": [0, 1]}},
    ],
}


# ---------------------------------------------------------------------------
# Fault plan generator
# ---------------------------------------------------------------------------

def generate_fault_plan(
    n_rows: int,
    segment_size: int,
    fault_family: str,
    rate: float,
    severity_params: dict | None,
    seed: int,
    replica_count: int = 3,
) -> RealizedFaultPlan:
    rng = random.Random(seed)
    if severity_params is None:
        severity_params = _SEVERITY_DEFAULTS.get(fault_family, {})
    merged = dict(_SEVERITY_DEFAULTS.get(fault_family, {}))
    merged.update(severity_params)

    fn = _FAULT_FAMILIES.get(fault_family)
    if fn is None:
        raise ValueError(f"Unknown fault family: {fault_family}")

    events = fn(rng, n_rows, rate, merged)

    unique_positions = set()
    xor_accum: dict[tuple[int, int], int] = {}
    for e in events:
        key = (e.plane, e.offset)
        unique_positions.add(key)
        xor_accum[key] = xor_accum.get(key, 0) ^ e.mask

    bit_flip_count = sum(popcount(e.mask) for e in events)
    affected_planes = set(e.plane for e in events)

    n_segments = (n_rows + segment_size - 1) // segment_size
    affected_segments: set[tuple[int, int]] = set()
    for e in events:
        seg_idx = e.offset // segment_size
        affected_segments.add((e.plane, seg_idx))

    replica_overlap = 0
    same_plane_multi_replica = 0
    for e in events:
        if e.target_replicas is not None:
            if len(e.target_replicas) > 1:
                replica_overlap += 1
                same_plane_multi_replica += 1

    sig_weighted = sum(
        popcount(e.mask) * PLANE_WEIGHTS[e.plane] for e in events
    )
    xor_cancel = sum(1 for v in xor_accum.values() if v == 0)

    return RealizedFaultPlan(
        events=events,
        fault_family=fault_family,
        rate=rate,
        seed=seed,
        event_count=len(events),
        attempted_mutated_byte_count=len(events),
        unique_mutated_byte_count=len(unique_positions),
        bit_flip_count=bit_flip_count,
        affected_segment_count=len(affected_segments),
        affected_plane_count=len(affected_planes),
        replica_overlap_count=replica_overlap,
        same_plane_multi_replica_count=same_plane_multi_replica,
        significance_weighted_impact_budget=float(sig_weighted),
        xor_cancellation_count=xor_cancel,
    )


# ---------------------------------------------------------------------------
# Fault plan application
# ---------------------------------------------------------------------------

def vote_byte(replica_values: list[int]) -> int:
    cnt = Counter(replica_values)
    max_count = max(cnt.values())
    r = len(replica_values)
    if max_count > r / 2:
        return next(v for v, c in cnt.items() if c == max_count)
    tied = sorted(v for v, c in cnt.items() if c == max_count)
    return tied[0]


def apply_fault_plan(
    clean_planes: list[bytes],
    plan: RealizedFaultPlan,
    r_vector: list[int],
    segment_size: int,
    n_rows: int,
) -> tuple[list[bytearray], dict]:
    n_planes = len(clean_planes)

    dirty_positions: dict[int, set[int]] = {}
    for e in plan.events:
        p = e.plane
        if p >= n_planes:
            continue
        if p not in dirty_positions:
            dirty_positions[p] = set()
        dirty_positions[p].add(e.offset)

    replicas: list[list[bytearray]] = []
    for p in range(n_planes):
        r = max(1, r_vector[p])
        plane_reps: list[bytearray] = [bytearray(clean_planes[p])]
        for _ in range(1, r):
            plane_reps.append(bytearray(clean_planes[p]))
        replicas.append(plane_reps)

    for e in plan.events:
        p = e.plane
        if p >= n_planes:
            continue
        reps = replicas[p]
        target_reps = e.target_replicas if e.target_replicas is not None else [0]
        for ri in target_reps:
            if ri < len(reps):
                reps[ri][e.offset] ^= e.mask

    delivered_planes: list[bytearray] = []
    per_plane_outcomes: list[dict[str, Any]] = []

    for p in range(n_planes):
        reps = replicas[p]
        r = len(reps)
        clean = clean_planes[p]
        dp = dirty_positions.get(p, set())

        if r == 1:
            delivered_planes.append(reps[0])
            diff = sum(1 for off in dp if reps[0][off] != clean[off])
            per_plane_outcomes.append({
                "r": 1, "diff_bytes": diff, "detected": diff > 0,
            })
        elif r == 2:
            diff = sum(1 for off in dp if reps[0][off] != reps[1][off])
            delivered_planes.append(reps[0])
            per_plane_outcomes.append({
                "r": 2, "diff_bytes": diff, "detected": diff > 0,
                "disagreement": diff > 0,
            })
        else:
            resolved = 0
            detected = 0
            undetected = 0
            for off in dp:
                vals = [reps[ri][off] for ri in range(r)]
                vb = vote_byte(vals)
                reps[0][off] = vb
                if vb == clean[off]:
                    resolved += 1
                elif clean[off] in vals:
                    detected += 1
                else:
                    undetected += 1
            delivered_planes.append(reps[0])
            per_plane_outcomes.append({
                "r": r, "diff_bytes": detected + undetected,
                "detected": (detected + undetected) > 0,
                "resolved": resolved, "detected_mismatch": detected,
                "undetected_corruption": undetected,
            })

    outcome_dict = {
        "per_plane": per_plane_outcomes,
        "any_detected": any(ppo["detected"] for ppo in per_plane_outcomes),
        "detected_planes": [p for p, ppo in enumerate(per_plane_outcomes) if ppo["detected"]],
        "disagreement_count": sum(
            ppo.get("disagreement", 0) for ppo in per_plane_outcomes
        ),
        "repair_count": sum(
            ppo.get("resolved", 0) for ppo in per_plane_outcomes
        ),
        "repair_failure_count": sum(
            ppo.get("detected_mismatch", 0) for ppo in per_plane_outcomes
        ),
    }

    return delivered_planes, outcome_dict


# ---------------------------------------------------------------------------
# Outcome space accounting
# ---------------------------------------------------------------------------

def classify_outcome(
    clean_answer: float,
    delivered_answer: float,
    detected: bool,
    bound_width: float,
    contains_truth: bool,
    fault_events: list[FaultEvent],
) -> str:
    if not detected and delivered_answer == clean_answer:
        return "exact_correct"
    if not detected and delivered_answer != clean_answer:
        return "silent_wrong"
    if detected and contains_truth:
        return "certified_degraded"
    if detected and not contains_truth:
        return "uncertified"
    return "unknown"


def compute_per_plane_outcome(
    plane: int,
    r: int,
    clean_data: bytes,
    delivered_data: bytearray,
    per_plane_details: dict,
) -> str:
    """Classify outcome for ONE faulted plane.

    Rules (Issue #285 audit):

      r=1:
        delivered != clean → silent_wrong
        delivered == clean → exact_correct

      r=2:
        replicas disagree (diff_bytes > 0) → detected_unavailable
        replicas agree AND correct → exact_correct
        replicas agree BUT wrong (twin corruption) → silent_wrong

      r>=3:
        majority vote delivers clean → majority_recovered
        majority vote delivers wrong → silent_wrong
        tie / no majority → detected_unavailable (theoretical)
    """
    clean_sum = sum(clean_data)
    delivered_sum = sum(delivered_data)
    clean_match = (clean_sum == delivered_sum)
    diff_bytes = per_plane_details.get("diff_bytes", 0)

    if r == 1:
        return "exact_correct" if clean_match else "silent_wrong"

    if r == 2:
        if diff_bytes > 0:
            return "detected_unavailable"
        return "exact_correct" if clean_match else "silent_wrong"

    if r >= 3:
        if clean_match:
            return "majority_recovered"
        return "silent_wrong"

    return "silent_wrong"


def classify_replication_outcome(
    clean_answer: int,
    delivered_answer: int,
    r_vector: list[int],
    fault_events: list[FaultEvent],
    per_plane_outcomes: list[dict],
    per_plane_outcomes_classified: list[dict] | None = None,
) -> str:
    """Two-stage per-cell replication outcome.

    Stage 1 — per-plane:
      Each faulted plane is classified independently by
      compute_per_plane_outcome.

    Stage 2 — query combine:
      Priority (highest→lowest):
        1. Any plane silent_wrong → whole query silent_wrong
        2. Any plane detected_unavailable → whole query detected_unavailable
        3. Any plane majority_recovered → whole query majority_recovered
        4. Otherwise → exact_correct

      The old ANY-gate logic (has_r2/has_r3 hiding per-plane SW) is
      intentionally replaced.  A single plane's silent_wrong is now
      visible at the query level regardless of r>=2/3 on other planes.

    Auxiliary metadata (returned in result dict, not used for outcome):
      has_r2_any, has_r3_any, detected_planes, per_plane_classification
    """
    if not fault_events:
        return "exact_correct"

    faulted_planes = sorted(set(e.plane for e in fault_events))

    if per_plane_outcomes_classified is None:
        per_plane_outcomes_classified = []

    # Stage 1: per-plane outcomes
    per_plane_verdicts: list[dict] = []
    for p in faulted_planes:
        if p >= len(r_vector):
            continue
        r = r_vector[p]
        if r < 1:
            r = 1

        ppo = per_plane_outcomes[p] if p < len(per_plane_outcomes) else {}
        # Find matching classified entry if available
        verdict = "silent_wrong"
        for entry in per_plane_outcomes_classified:
            if entry.get("plane") == p:
                verdict = entry.get("per_plane_outcome", "silent_wrong")
                break
        per_plane_verdicts.append({"plane": p, "r": r, "outcome": verdict})

    # Stage 2: query combine (priority: SW > DU > MR > EC)
    any_sw = any(v["outcome"] == "silent_wrong" for v in per_plane_verdicts)
    any_du = any(v["outcome"] == "detected_unavailable" for v in per_plane_verdicts)
    any_mr = any(v["outcome"] == "majority_recovered" for v in per_plane_verdicts)

    if any_sw:
        return "silent_wrong"
    if any_du:
        return "detected_unavailable"
    if any_mr:
        return "majority_recovered"
    return "exact_correct"


# ---------------------------------------------------------------------------
# Per-cell metrics computation
# ---------------------------------------------------------------------------

def compute_cell_result(
    clean_planes: list[bytes],
    delivered_planes: list[bytearray],
    plan: RealizedFaultPlan,
    r_vector: list[int],
    n_rows: int,
    segment_size: int,
    clean_answer: int,
    clean_digests: list[int],
    tau: float = 0.05,
) -> dict[str, Any]:
    delivered_answer = compute_encoded_sum(delivered_planes)

    delivered_digests = [sum32(bytes(d)) for d in delivered_planes]
    detected_planes_list = [d != c for d, c in zip(delivered_digests, clean_digests)]
    detected = any(detected_planes_list)

    bound_width = compute_per_segment_bound(
        detected_planes_list, n_rows, segment_size,
    )

    lo = delivered_answer - bound_width
    hi = delivered_answer + bound_width
    contains_truth = lo <= clean_answer <= hi

    outcome = classify_outcome(
        clean_answer, delivered_answer, detected, bound_width,
        contains_truth, plan.events,
    )

    abs_err = abs(delivered_answer - clean_answer)
    rel_err = abs_err / max(abs(clean_answer), 1.0)
    bound_width_ratio = bound_width / max(abs(clean_answer), 1.0)

    extra_B = sum(max(0, r - 1) for r in r_vector)

    faulted_planes = sorted(set(e.plane for e in plan.events))

    return {
        "dataset": "",
        "fault_family": plan.fault_family,
        "rate": plan.rate,
        "seed": plan.seed,
        "policy_B": extra_B,
        "total_replicas": sum(r_vector),
        "r_vector": str(r_vector),
        "faulted_planes": str(faulted_planes),
        "outcome": outcome,
        "detected": detected,
        "contains_truth": contains_truth,
        "clean_answer": clean_answer,
        "delivered_answer": delivered_answer,
        "abs_error": abs_err,
        "relative_error": rel_err,
        "bound_width": bound_width,
        "relative_bound_width": bound_width_ratio,
        "event_count": plan.event_count,
        "bit_flip_count": plan.bit_flip_count,
        "affected_plane_count": plan.affected_plane_count,
        "affected_segment_count": plan.affected_segment_count,
        "xor_cancellation_count": plan.xor_cancellation_count,
        "significance_weighted_impact_budget": plan.significance_weighted_impact_budget,
        "unique_mutated_byte_count": plan.unique_mutated_byte_count,
        "attempted_mutated_byte_count": plan.attempted_mutated_byte_count,
        "decision_flip": abs_err > tau * abs(clean_answer),
    }


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_primary_metrics(rows: list[dict]) -> dict[str, float]:
    n = len(rows) if rows else 1
    fault_rows = [r for r in rows if r["event_count"] > 0]
    n_fault = len(fault_rows) if fault_rows else 1

    err_fault_conditioned = sum(
        r["abs_error"] / max(abs(r["clean_answer"]), 1.0)
        for r in fault_rows
    ) / n_fault

    rel_errors = sorted(r["relative_error"] for r in rows)
    rel_p50 = rel_errors[len(rel_errors) // 2] if rel_errors else 0.0
    rel_p99 = rel_errors[int(len(rel_errors) * 0.99)] if rel_errors else 0.0
    rel_p999 = rel_errors[int(len(rel_errors) * 0.999)] if rel_errors else 0.0
    max_rel = max(rel_errors) if rel_errors else 0.0

    df_rate = sum(1 for r in rows if r["decision_flip"]) / n
    cat_rate = sum(1 for r in rows if r["relative_error"] > 1.0) / n
    sw_rate = sum(1 for r in rows if r["outcome"] == "silent_wrong") / n
    aa_rate = sum(
        1 for r in rows if r["outcome"] in ("exact_correct", "certified_degraded")
    ) / n

    return {
        "err_fault_conditioned_user_observed": err_fault_conditioned,
        "relative_error_p50": rel_p50,
        "relative_error_p99": rel_p99,
        "relative_error_p999": rel_p999,
        "max_relative_error": max_rel,
        "decision_flip_rate_at_tau": df_rate,
        "catastrophic_error_rate": cat_rate,
        "silent_wrong_rate": sw_rate,
        "accepted_answer_rate": aa_rate,
    }


def compute_secondary_metrics(rows: list[dict]) -> dict[str, float]:
    n = len(rows) if rows else 1
    widths = sorted(r.get("bound_width", 0) for r in rows)
    rel_widths = sorted(r.get("relative_bound_width", 0) for r in rows)

    bw_med = widths[len(widths) // 2] if widths else 0.0
    bw_p99 = widths[int(len(widths) * 0.99)] if widths else 0.0
    rbw_med = rel_widths[len(rel_widths) // 2] if rel_widths else 0.0
    rbw_p99 = rel_widths[int(len(rel_widths) * 0.99)] if rel_widths else 0.0

    contains_truth_rate = sum(1 for r in rows if r["contains_truth"]) / n
    fallback_rate = sum(
        1 for r in rows if r["delivered_answer"] != r["clean_answer"]
    ) / n
    uncertified_rate = sum(1 for r in rows if r["outcome"] == "uncertified") / n

    repair_coverage_rate = sum(
        1 for r in rows if r["outcome"] in ("exact_correct", "certified_degraded")
    ) / n

    faulted_cells = [r for r in rows
                     if int(r.get("event_count", 0)) > 0
                     or int(r.get("affected_plane_count", 0)) > 0]
    fn = len(faulted_cells) if faulted_cells else 1
    faulted_cell_accepted_rate = sum(
        1 for r in faulted_cells
        if r["outcome"] in ("exact_correct", "certified_degraded")
    ) / fn

    return {
        "certified_bound_width_median": bw_med,
        "certified_bound_width_p99": bw_p99,
        "relative_bound_width_median": rbw_med,
        "relative_bound_width_p99": rbw_p99,
        "contains_truth_rate": contains_truth_rate,
        "fallback_rate": fallback_rate,
        "uncertified_rate": uncertified_rate,
        "repair_coverage_rate": repair_coverage_rate,
        "faulted_cell_accepted_rate": faulted_cell_accepted_rate,
    }


def _parse_r_vector(s: str) -> list[int]:
    stripped = s.strip("[]").strip()
    if not stripped:
        return []
    return [int(x.strip()) for x in stripped.split(",")]


def compute_detector_metrics(rows: list[dict]) -> dict[str, Any]:
    n = len(rows) if rows else 1

    def _is_detected(r):
        v = r.get("detected", False)
        if isinstance(v, str):
            return v.lower() in ("true", "1")
        return bool(v)

    r2_rows: list[dict] = []
    r3_rows: list[dict] = []
    for r in rows:
        r_vec = _parse_r_vector(r.get("r_vector", "[]"))
        fp_str = r.get("faulted_planes", "")
        if not fp_str or fp_str == "[]":
            continue
        try:
            fps = [int(x.strip()) for x in fp_str.strip("[]").split(",") if x.strip()]
        except (ValueError, AttributeError):
            continue
        has_r2_on_faulted = any(
            r_vec[p] >= 2 for p in fps if 0 <= p < len(r_vec)
        )
        has_r3_on_faulted = any(
            r_vec[p] >= 3 for p in fps if 0 <= p < len(r_vec)
        )
        if has_r2_on_faulted:
            r2_rows.append(r)
        if has_r3_on_faulted:
            r3_rows.append(r)

    r2_n = len(r2_rows)
    r3_n = len(r3_rows)

    return {
        "r2_applicable_cells": r2_n,
        "r2_disagreement_rate": (
            sum(1 for r in r2_rows if _is_detected(r)) / r2_n
        ) if r2_n > 0 else None,
        "r3_applicable_cells": r3_n,
        "r3_majority_repair_rate": (
            sum(1 for r in r3_rows if r.get("outcome") == "exact_correct") / r3_n
        ) if r3_n > 0 else None,
        "detector_escape_rate": sum(
            1 for r in rows
            if r.get("outcome") == "silent_wrong"
            and int(r.get("event_count", 0)) > 0
        ) / max(sum(1 for r in rows if int(r.get("event_count", 0)) > 0), 1),
        "false_certification_rate": sum(
            1 for r in rows
            if r.get("outcome") == "uncertified"
        ) / n,
    }


def compute_all_metrics(rows: list[dict]) -> dict[str, float]:
    m = {}
    m.update(compute_primary_metrics(rows))
    m.update(compute_secondary_metrics(rows))
    m.update(compute_detector_metrics(rows))
    return m


# ---------------------------------------------------------------------------
# Replication-only mode functions
# ---------------------------------------------------------------------------

def compute_replication_cell_result(
    clean_planes: list[bytes],
    delivered_planes: list[bytearray],
    plan: RealizedFaultPlan,
    r_vector: list[int],
    clean_answer: int,
    per_plane_outcomes: list[dict],
    tau: float = 0.05,
) -> dict[str, Any]:
    delivered_answer = compute_encoded_sum(delivered_planes)

    # Stage 1: per-plane classification
    faulted_planes_list = sorted(set(e.plane for e in plan.events))
    per_plane_classified: list[dict] = []
    for p in faulted_planes_list:
        if p >= len(r_vector):
            continue
        r = r_vector[p]
        ppo = per_plane_outcomes[p] if p < len(per_plane_outcomes) else {}
        p_outcome = compute_per_plane_outcome(
            p, r, clean_planes[p], delivered_planes[p], ppo,
        )
        per_plane_classified.append({
            "plane": p, "r": r, "diff_bytes": ppo.get("diff_bytes", 0),
            "per_plane_outcome": p_outcome,
        })

    # Stage 2: query combine
    outcome = classify_replication_outcome(
        clean_answer, delivered_answer, r_vector,
        plan.events, per_plane_outcomes, per_plane_classified,
    )

    # Auxiliary metadata
    has_r2_any = any(v["per_plane_outcome"] in ("detected_unavailable",)
                     for v in per_plane_classified)
    has_r3_any = any(v["r"] >= 3 for v in per_plane_classified)
    detected_planes = [v["plane"] for v in per_plane_classified
                       if v["per_plane_outcome"] in ("detected_unavailable",)]

    abs_err = abs(delivered_answer - clean_answer)
    rel_err = abs_err / max(abs(clean_answer), 1.0)
    extra_B = sum(max(0, r - 1) for r in r_vector)
    faulted_planes = sorted(set(e.plane for e in plan.events))
    return {
        "dataset": "", "fault_family": plan.fault_family,
        "rate": plan.rate, "seed": plan.seed,
        "policy_B": extra_B, "total_replicas": sum(r_vector),
        "r_vector": str(r_vector),
        "faulted_planes": str(faulted_planes),
        "outcome": outcome,
        "per_plane_outcomes": str(per_plane_classified),
        "has_r2_any": has_r2_any,
        "has_r3_any": has_r3_any,
        "detected_planes": str(detected_planes),
        "clean_answer": clean_answer,
        "delivered_answer": delivered_answer,
        "abs_error": abs_err, "relative_error": rel_err,
        "event_count": plan.event_count,
        "bit_flip_count": plan.bit_flip_count,
        "affected_plane_count": plan.affected_plane_count,
        "affected_segment_count": plan.affected_segment_count,
        "xor_cancellation_count": plan.xor_cancellation_count,
        "unique_mutated_byte_count": plan.unique_mutated_byte_count,
        "attempted_mutated_byte_count": plan.attempted_mutated_byte_count,
        "significance_weighted_impact_budget": plan.significance_weighted_impact_budget,
        "decision_flip": abs_err > tau * abs(clean_answer),
    }


def compute_replication_primary_metrics(rows: list[dict]) -> dict[str, float]:
    n = len(rows) if rows else 1
    point_rows = [r for r in rows if r.get("outcome") != "detected_unavailable"]
    np_ = len(point_rows) if point_rows else 1

    err_fault = sum(
        r["relative_error"] for r in point_rows
    ) / np_

    rel_errors = sorted(r["relative_error"] for r in point_rows)
    rel_p50 = rel_errors[len(rel_errors) // 2] if rel_errors else 0.0
    rel_p99 = rel_errors[int(len(rel_errors) * 0.99)] if rel_errors else 0.0
    rel_p999 = rel_errors[int(len(rel_errors) * 0.999)] if rel_errors else 0.0
    max_rel = max(rel_errors) if rel_errors else 0.0

    df_rate = sum(1 for r in point_rows if r.get("decision_flip", False)) / np_
    cat_rate = sum(1 for r in point_rows if r.get("relative_error", 0) > 1.0) / np_

    cnt: dict[str, int] = {}
    for oc in ("exact_correct", "majority_recovered", "detected_unavailable", "silent_wrong"):
        cnt[oc] = sum(1 for r in rows if r.get("outcome") == oc)

    return {
        "err_fault_conditioned_user_observed": err_fault,
        "relative_error_p50": rel_p50,
        "relative_error_p99": rel_p99,
        "relative_error_p999": rel_p999,
        "max_relative_error": max_rel,
        "decision_flip_rate_at_tau": df_rate,
        "catastrophic_error_rate": cat_rate,
        "exact_correct_rate": cnt["exact_correct"] / n,
        "majority_recovered_rate": cnt["majority_recovered"] / n,
        "detected_unavailable_rate": cnt["detected_unavailable"] / n,
        "silent_wrong_rate": cnt["silent_wrong"] / n,
    }


def compute_replication_composite_metrics(rows: list[dict]) -> dict[str, float]:
    n = len(rows) if rows else 1
    ec = sum(1 for r in rows if r.get("outcome") == "exact_correct")
    mr = sum(1 for r in rows if r.get("outcome") == "majority_recovered")
    sw = sum(1 for r in rows if r.get("outcome") == "silent_wrong")
    return {
        "correct_answer_rate": (ec + mr) / n,
        "safe_answer_rate": 1.0 - (sw / n),
    }


def compute_replication_secondary_metrics(rows: list[dict]) -> dict[str, float]:
    n = len(rows) if rows else 1
    r2_rows: list[dict] = []
    r3_rows: list[dict] = []
    for r in rows:
        r_vec = _parse_r_vector(r.get("r_vector", "[]"))
        fp_str = r.get("faulted_planes", "")
        if not fp_str or fp_str == "[]":
            continue
        try:
            fps = [int(x.strip()) for x in fp_str.strip("[]").split(",") if x.strip()]
        except (ValueError, AttributeError):
            continue
        has_r2_on_faulted = any(r_vec[p] >= 2 for p in fps if 0 <= p < len(r_vec))
        has_r3_on_faulted = any(r_vec[p] >= 3 for p in fps if 0 <= p < len(r_vec))
        if has_r2_on_faulted:
            r2_rows.append(r)
        if has_r3_on_faulted:
            r3_rows.append(r)
    r2_n = len(r2_rows)
    r3_n = len(r3_rows)
    return {
        "r2_disagreement_rate": (
            sum(1 for r in r2_rows if r.get("outcome") == "detected_unavailable") / r2_n
        ) if r2_n > 0 else None,
        "r3_vote_recovery_rate": (
            sum(1 for r in r3_rows
                if r.get("outcome") in ("exact_correct", "majority_recovered")) / r3_n
        ) if r3_n > 0 else None,
        "r3_vote_failure_rate": (
            sum(1 for r in r3_rows if r.get("outcome") == "detected_unavailable") / r3_n
        ) if r3_n > 0 else None,
    }


def compute_all_replication_metrics(rows: list[dict]) -> dict[str, float]:
    m = {}
    m.update(compute_replication_primary_metrics(rows))
    m.update(compute_replication_composite_metrics(rows))
    m.update(compute_replication_secondary_metrics(rows))
    return m


def write_replication_headline(
    grouped: dict[tuple, list[dict]],
    path: Path,
) -> None:
    METRICS = [
        "correct_answer_rate", "safe_answer_rate",
        "exact_correct_rate", "majority_recovered_rate",
        "detected_unavailable_rate", "silent_wrong_rate",
        "err_fault_conditioned_user_observed",
        "relative_error_p50", "relative_error_p99",
        "decision_flip_rate_at_tau", "catastrophic_error_rate",
        "r2_disagreement_rate", "r3_vote_recovery_rate", "r3_vote_failure_rate",
    ]
    fields = ["dataset", "fault_family", "severity_label", "rate",
              "policy_B", "policy_type"] + METRICS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for key, grp in sorted(grouped.items()):
            dataset, family, severity_label, rate, policy_B, policy_type = key
            m = compute_all_replication_metrics(grp)
            row: dict[str, Any] = {
                "dataset": dataset, "fault_family": family,
                "severity_label": severity_label,
                "rate": f"{rate:.1e}",
                "policy_B": policy_B, "policy_type": policy_type,
            }
            row.update(m)
            w.writerow(row)


def write_replication_frontier(
    grouped: dict[tuple, list[dict]],
    path: Path,
) -> None:
    fields = [
        "dataset", "fault_family", "severity_label", "rate",
        "policy_B", "policy_type",
        "correct_answer_rate", "safe_answer_rate",
        "silent_wrong_rate",
        "total_extra_storage_B",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for key, grp in sorted(grouped.items()):
            dataset, family, severity_label, rate, policy_B, policy_type = key
            m = compute_all_replication_metrics(grp)
            row: dict[str, Any] = {
                "dataset": dataset, "fault_family": family,
                "severity_label": severity_label,
                "rate": f"{rate:.1e}",
                "policy_B": policy_B, "policy_type": policy_type,
                "correct_answer_rate": m.get("correct_answer_rate", 0),
                "safe_answer_rate": m.get("safe_answer_rate", 0),
                "silent_wrong_rate": m.get("silent_wrong_rate", 0),
                "total_extra_storage_B": sum(r["policy_B"] for r in grp) // max(len(grp), 1),
            }
            w.writerow(row)


# ---------------------------------------------------------------------------
# CSV exporters
# ---------------------------------------------------------------------------

def write_pilot_matrix(rows: list[dict], path: Path, replication: bool = False) -> None:
    if replication:
        fields = [
            "dataset", "fault_family", "severity_label", "rate", "seed",
            "policy_name", "policy_type", "policy_B", "r_vector",
            "faulted_planes", "outcome",
            "has_r2_any", "has_r3_any", "detected_planes",
            "clean_answer", "delivered_answer",
            "abs_error", "relative_error",
            "event_count", "bit_flip_count",
            "affected_plane_count", "affected_segment_count",
            "unique_mutated_byte_count", "attempted_mutated_byte_count",
            "xor_cancellation_count", "significance_weighted_impact_budget",
            "decision_flip",
        ]
    else:
        fields = [
            "dataset", "fault_family", "severity_label", "rate", "seed",
            "policy_name", "policy_type", "policy_B", "r_vector",
            "faulted_planes", "outcome",
            "detected", "contains_truth",
            "clean_answer", "delivered_answer",
            "abs_error", "relative_error",
            "bound_width", "relative_bound_width",
            "event_count", "bit_flip_count",
            "affected_plane_count", "affected_segment_count",
            "unique_mutated_byte_count", "xor_cancellation_count",
            "decision_flip",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            safe = {k: r.get(k, "") for k in fields}
            safe["rate"] = f"{r.get('rate', 0):.1e}"
            safe["outcome"] = r.get("outcome", "")
            safe["r_vector"] = r.get("r_vector", "")
            w.writerow(safe)


def write_headline_matrix(
    grouped: dict[tuple, list[dict]],
    path: Path,
) -> None:
    METRICS = [
        "err_fault_conditioned_user_observed",
        "silent_wrong_rate",
        "accepted_answer_rate",
        "catastrophic_error_rate",
        "relative_error_p99",
        "max_relative_error",
        "decision_flip_rate_at_tau",
        "contains_truth_rate",
        "certified_bound_width_median",
        "relative_bound_width_median",
        "repair_coverage_rate",
        "faulted_cell_accepted_rate",
        "r2_applicable_cells",
        "r2_disagreement_rate",
        "r3_applicable_cells",
        "r3_majority_repair_rate",
        "detector_escape_rate",
        "false_certification_rate",
    ]
    fields = ["dataset", "fault_family", "severity_label", "rate", "policy_B", "policy_type"] + METRICS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for key, grp in sorted(grouped.items()):
            dataset, family, severity_label, rate, policy_B, policy_type = key
            m = compute_all_metrics(grp)
            row: dict[str, Any] = {
                "dataset": dataset,
                "fault_family": family,
                "severity_label": severity_label,
                "rate": f"{rate:.1e}",
                "policy_B": policy_B,
                "policy_type": policy_type,
            }
            row.update(m)
            w.writerow(row)


def write_policy_frontier(
    grouped: dict[tuple, list[dict]],
    path: Path,
) -> None:
    fields = [
        "dataset", "fault_family", "severity_label", "rate",
        "policy_B", "policy_type",
        "silent_wrong_rate",
        "accepted_answer_rate",
        "relative_error_p99",
        "certified_bound_width_median",
        "repair_coverage_rate",
        "faulted_cell_accepted_rate",
        "total_extra_storage_B",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for key, grp in sorted(grouped.items()):
            dataset, family, severity_label, rate, policy_B, policy_type = key
            m = compute_all_metrics(grp)
            row: dict[str, Any] = {
                "dataset": dataset,
                "fault_family": family,
                "severity_label": severity_label,
                "rate": f"{rate:.1e}",
                "policy_B": policy_B,
                "policy_type": policy_type,
                "silent_wrong_rate": m.get("silent_wrong_rate", 0),
                "accepted_answer_rate": m.get("accepted_answer_rate", 0),
                "relative_error_p99": m.get("relative_error_p99", 0),
                "certified_bound_width_median": m.get("certified_bound_width_median", 0),
                "repair_coverage_rate": m.get("repair_coverage_rate", 0),
                "faulted_cell_accepted_rate": m.get("faulted_cell_accepted_rate", 0),
                "total_extra_storage_B": sum(r["policy_B"] for r in grp) // max(len(grp), 1),
            }
            w.writerow(row)


def write_fault_family_summary(
    rows: list[dict],
    path: Path,
    replication: bool = False,
) -> None:
    families = set(r["fault_family"] for r in rows)
    _metrics_fn = compute_all_replication_metrics if replication else compute_all_metrics
    if replication:
        fields = [
            "fault_family", "num_cells",
            "exact_correct_rate", "majority_recovered_rate",
            "detected_unavailable_rate", "silent_wrong_rate",
            "correct_answer_rate", "safe_answer_rate",
            "catastrophic_error_rate",
            "mean_relative_error",
            "max_relative_error",
            "mean_event_count",
            "mean_bit_flip_count",
            "mean_affected_planes",
            "mean_xor_cancellation",
        ]
    else:
        fields = [
            "fault_family", "num_cells",
            "silent_wrong_rate", "accepted_answer_rate",
            "catastrophic_error_rate",
            "mean_relative_error",
            "max_relative_error",
            "mean_event_count",
            "mean_bit_flip_count",
            "mean_affected_planes",
            "mean_xor_cancellation",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for family in sorted(families):
            fr = [r for r in rows if r["fault_family"] == family]
            n = len(fr)
            if n == 0:
                continue
            m = _metrics_fn(fr)
            row: dict[str, Any] = {
                "fault_family": family,
                "num_cells": n,
                "silent_wrong_rate": m.get("silent_wrong_rate", 0),
                "catastrophic_error_rate": m.get("catastrophic_error_rate", 0),
                "mean_relative_error": m.get("relative_error_p50", 0),
                "max_relative_error": m.get("max_relative_error", 0),
                "mean_event_count": sum(r["event_count"] for r in fr) / n,
                "mean_bit_flip_count": sum(r["bit_flip_count"] for r in fr) / n,
                "mean_affected_planes": sum(r["affected_plane_count"] for r in fr) / n,
                "mean_xor_cancellation": sum(r["xor_cancellation_count"] for r in fr) / n,
            }
            if replication:
                row["exact_correct_rate"] = m.get("exact_correct_rate", 0)
                row["majority_recovered_rate"] = m.get("majority_recovered_rate", 0)
                row["detected_unavailable_rate"] = m.get("detected_unavailable_rate", 0)
                row["correct_answer_rate"] = m.get("correct_answer_rate", 0)
                row["safe_answer_rate"] = m.get("safe_answer_rate", 0)
            else:
                row["accepted_answer_rate"] = m.get("accepted_answer_rate", 0)
            w.writerow(row)


def write_exposure_summary(rows: list[dict], path: Path) -> None:
    fields = [
        "dataset", "fault_family", "rate",
        "total_event_count", "total_bit_flip_count",
        "total_unique_bytes", "total_attempted_bytes",
        "mean_event_count", "mean_bit_flip_count",
        "mean_affected_planes", "mean_affected_segments",
        "mean_sig_weighted_impact",
        "mean_xor_cancellation",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        keys = set((r["dataset"], r["fault_family"], r["rate"]) for r in rows)
        for (ds, family, rate) in sorted(keys):
            fr = [r for r in rows
                  if r["dataset"] == ds
                  and r["fault_family"] == family
                  and r["rate"] == rate]
            n = len(fr)
            if n == 0:
                continue
            row: dict[str, Any] = {
                "dataset": ds,
                "fault_family": family,
                "rate": f"{rate:.1e}",
                "total_event_count": sum(r["event_count"] for r in fr),
                "total_bit_flip_count": sum(r["bit_flip_count"] for r in fr),
                "total_unique_bytes": sum(r["unique_mutated_byte_count"] for r in fr),
                "total_attempted_bytes": sum(r.get("attempted_mutated_byte_count", r.get("unique_mutated_byte_count", 0)) for r in fr),
                "mean_event_count": sum(r["event_count"] for r in fr) / n,
                "mean_bit_flip_count": sum(r["bit_flip_count"] for r in fr) / n,
                "mean_affected_planes": sum(r["affected_plane_count"] for r in fr) / n,
                "mean_affected_segments": sum(r["affected_segment_count"] for r in fr) / n,
                "mean_sig_weighted_impact": sum(r.get("significance_weighted_impact_budget", 0) for r in fr) / n,
                "mean_xor_cancellation": sum(r["xor_cancellation_count"] for r in fr) / n,
            }
            w.writerow(row)


def write_outcome_summary(rows: list[dict], path: Path, replication: bool = False) -> None:
    if replication:
        outcome_fields = [
            "exact_correct", "majority_recovered",
            "detected_unavailable", "silent_wrong",
        ]
    else:
        outcome_fields = [
            "exact_correct", "certified_degraded",
            "uncertified", "silent_wrong",
        ]
    fields = ["dataset", "fault_family", "rate", "total_cells"]
    for oc in outcome_fields:
        fields.append(oc)
        fields.append(f"{oc}_rate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        keys = set((r["dataset"], r["fault_family"], r["rate"]) for r in rows)
        for (ds, family, rate) in sorted(keys):
            fr = [r for r in rows
                  if r["dataset"] == ds
                  and r["fault_family"] == family
                  and r["rate"] == rate]
            n = len(fr)
            if n == 0:
                continue
            cnt = Counter(r["outcome"] for r in fr)
            row: dict[str, Any] = {
                "dataset": ds, "fault_family": family,
                "rate": f"{rate:.1e}", "total_cells": n,
            }
            for oc in outcome_fields:
                row[oc] = cnt.get(oc, 0)
                row[f"{oc}_rate"] = cnt.get(oc, 0) / n
            w.writerow(row)


# ---------------------------------------------------------------------------
# Policy definitions (20 entries, generated)
# ---------------------------------------------------------------------------

def build_policy_catalogue() -> list[tuple[str, str, list[int]]]:
    policies: list[tuple[str, str, list[int]]] = [
        ("uniform_full_r1", "uniform", [0, 0, 0, 0, 0, 0, 0, 0]),
        ("uniform_full_r2", "uniform", [1, 1, 1, 1, 1, 1, 1, 1]),
        ("uniform_full_r3", "uniform", [2, 2, 2, 2, 2, 2, 2, 2]),
    ]
    for B in range(0, 17):
        policies.append((f"graded_B{B}", "graded", make_graded_policy(B)))
    return policies

_POLICIES = build_policy_catalogue()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n-rows", type=int, required=True)
    parser.add_argument("--segment-size", type=int, default=SEGMENT_SIZE)
    parser.add_argument("--mode", type=str, default="detect_and_bound",
                        choices=["detect_and_bound", "replication_only"])
    parser.add_argument("--fault-families", type=str, nargs="+",
                        default=list(_FAULT_FAMILIES.keys()))
    parser.add_argument("--rate-anchors", type=float, nargs="+",
                        default=[1e-9, 1e-7, 1e-5])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(10)))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    jid = __import__("os").environ.get("SLURM_JOB_ID", "cpu_claim1")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== NMR-Claim1 Realistic Fault Campaign ===")
    print(f"dataset={args.dataset} n_rows={args.n_rows}")
    print(f"families={args.fault_families} rates={args.rate_anchors}")
    print(f"seeds={args.seeds} output={output_dir}")
    print(f"policies={len(_POLICIES)}")

    print(f"\nLoading {args.dataset} ({args.n_rows} rows) ...")
    t0 = time.perf_counter()
    clean_planes = load_planes(args.artifact_dir, args.n_rows)
    print(f"  loaded {len(clean_planes)} planes, "
          f"{len(clean_planes[0])} rows each")

    clean_answer = compute_encoded_sum(clean_planes)

    is_rep = args.mode == "replication_only"
    if not is_rep:
        clean_digests = per_plane_sum32(clean_planes)

    all_rows: list[dict] = []

    for family in args.fault_families:
        if family not in _FAULT_FAMILIES:
            print(f"WARNING: unknown fault family '{family}', skipping",
                  file=sys.stderr)
            continue

        sweeps = _SEVERITY_SWEEP.get(family, [{}])
        for sev_idx, severity_params in enumerate(sweeps):
            for rate in args.rate_anchors:
                for seed in args.seeds:
                    plan = generate_fault_plan(
                        n_rows=args.n_rows,
                        segment_size=args.segment_size,
                        fault_family=family,
                        rate=rate,
                        severity_params=severity_params,
                        seed=seed,
                    )

                    for pol_name, pol_type, pol_extras in _POLICIES:
                        r_vector = [1 + e for e in pol_extras]
                        delivered_planes, outcome_dict = apply_fault_plan(
                            clean_planes, plan, r_vector,
                            args.segment_size, args.n_rows,
                        )

                        if is_rep:
                            result = compute_replication_cell_result(
                                clean_planes, delivered_planes, plan,
                                r_vector, clean_answer,
                                outcome_dict.get("per_plane", []),
                            )
                        else:
                            result = compute_cell_result(
                                clean_planes, delivered_planes, plan,
                                r_vector, args.n_rows, args.segment_size,
                                clean_answer, clean_digests,
                            )
                        result["dataset"] = args.dataset
                        result["policy_name"] = pol_name
                        result["policy_type"] = pol_type
                        result["total_replicas"] = sum(r_vector)
                        result["severity_idx"] = sev_idx
                        result["severity_label"] = str(severity_params) if severity_params else "default"
                        all_rows.append(result)

                n_cells = len(_POLICIES)
                sw = sum(1 for r in all_rows[-n_cells:]
                         if r["outcome"] == "silent_wrong")
                print(f"  {family}[{sev_idx}] rate={rate:.1e} seed={seed:2d}  "
                      f"events={plan.event_count:5d}  "
                      f"cells={n_cells}  silent_wrong={sw:2d}")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal cells: {len(all_rows)} in {elapsed:.2f}s")

    pilot_path = output_dir / "claim1_realistic_pilot_matrix.csv"
    write_pilot_matrix(all_rows, pilot_path, is_rep)
    print(f"pilot_matrix: {pilot_path} ({len(all_rows)} rows)")

    hgroup: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        key = (r["dataset"], r["fault_family"], r.get("severity_label", ""),
               r["rate"],
               r["policy_B"], r.get("policy_type", ""))
        hgroup[key].append(r)

    if is_rep:
        headline_path = output_dir / "claim1_realistic_headline_matrix_nosum32.csv"
        write_replication_headline(hgroup, headline_path)
        print(f"headline_matrix_nosum32: {headline_path} ({len(hgroup)} rows)")
    else:
        headline_path = output_dir / "claim1_realistic_headline_matrix.csv"
        write_headline_matrix(hgroup, headline_path)
        print(f"headline_matrix: {headline_path} ({len(hgroup)} rows)")

    if is_rep:
        frontier_path = output_dir / "claim1_realistic_policy_frontier_nosum32.csv"
        write_replication_frontier(hgroup, frontier_path)
        print(f"policy_frontier_nosum32: {frontier_path}")
    else:
        frontier_path = output_dir / "claim1_realistic_policy_frontier.csv"
        write_policy_frontier(hgroup, frontier_path)
        print(f"policy_frontier: {frontier_path}")

    ff_path = output_dir / "claim1_realistic_fault_family_summary.csv"
    write_fault_family_summary(all_rows, ff_path, is_rep)
    print(f"fault_family_summary: {ff_path}")

    exp_path = output_dir / "claim1_realistic_exposure_summary.csv"
    write_exposure_summary(all_rows, exp_path)
    print(f"exposure_summary: {exp_path}")

    oc_path = output_dir / "claim1_realistic_outcome_summary.csv"
    write_outcome_summary(all_rows, oc_path, is_rep)
    print(f"outcome_summary: {oc_path}")

    print(f"\nDone. Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
