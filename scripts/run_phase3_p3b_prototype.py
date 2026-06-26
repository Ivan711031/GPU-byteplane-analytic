#!/usr/bin/env python3
"""Phase 3 P3-B CPU sparse reactive prototype.

Implements the reactive significance-aware bounded-error execution mechanism
for active_delta_global_v1 artifacts on CPU.

Usage:
    python3 scripts/run_phase3_p3b_prototype.py

Outputs:
    /work/u4063895/results/reliability_layer1/phase3/p3b_cpu_prototype/<run_id>/*.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path("/work/u4063895/results/reliability_layer1/phase3/p3b_cpu_prototype")
ARTIFACT_ROOT = Path("/work/u4063895/datasets/artifacts_phase1_5b")

N_TOTAL_PLANES = 8
PLANE_WEIGHTS = [1 << (8 * (5 - p)) for p in range(6)]  # [2^40, 2^32, 2^24, 2^16, 256, 1]

# Normalized cost model (T_progressive_query = 1.0)
C_VERIFY = 0.15
C_ABSORB = 0.00
C_R3 = 0.05
C_RE_READ = 1.0
C_RECOMPUTE = 3.0
C_FALLBACK = 5.0
C_UNCERTIFIED = 0.0

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SegmentResult:
    segment_id: int = 0
    Q_count: int = 0
    D_count: int = 0
    U_depth: int = 0
    U_quant: int = 0
    U_fault: int = 0
    S_core_low: float = 0.0
    S_core_high: float = 0.0
    swing_low: float = 0.0
    swing_high: float = 0.0
    E_quant_value: float = 0.0
    E_fault_value: float = 0.0
    action_taken: int = 0  # R0..R6
    cost: float = 0.0
    faults_detected: int = 0
    bytes_verified: int = 0


@dataclass
class QueryResult:
    query_idx: int = 0
    dataset: str = ""
    n_rows: int = 0
    segment_size: int = 0
    k: int = 0
    threshold: float = 0.0
    epsilon: float = 0.0
    corruption_type: str = "none"
    corruption_planes: str = ""
    corruption_segment: int = -1
    Q_total: int = 0
    U_depth_total: int = 0
    U_quant_total: int = 0
    U_fault_total: int = 0
    S_low: float = 0.0
    S_high: float = 0.0
    E_quant_total: float = 0.0
    E_fault_total: float = 0.0
    certification: str = ""
    segments_verified: int = 0
    segments_failed: int = 0
    faults_detected: int = 0
    actions: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0])  # R0..R6
    total_cost: float = 0.0
    width: float = 0.0
    relative_width: float = 0.0


@dataclass
class FallbackEvent:
    query_idx: int = 0
    segment_id: int = 0
    plane_id: int = 0
    suspect_count: int = 0
    cause: str = ""
    action_taken: int = 0
    action_cost: float = 0.0
    E_fault_value: float = 0.0
    certification_after: str = ""


@dataclass
class ArtifactData:
    path: Path
    dataset: str
    n_rows: int
    scale: int
    base_fixed: int
    max_delta: int
    active_byte_len: int
    plane_weight: list[int]
    plane_data: list[np.ndarray]  # length 8, each ndarray of uint8 shape (n_rows,)


# ---------------------------------------------------------------------------
# Artifact loader
# ---------------------------------------------------------------------------


def load_artifact(dataset: str, n_rows: int = 100_000_000) -> ArtifactData:
    """Load active_delta_global_v1 artifact from disk."""
    # Find the artifact directory (one level deeper with scale dir)
    ds_root = ARTIFACT_ROOT / dataset / f"n{n_rows}"
    scale_dirs = sorted(ds_root.iterdir())
    if not scale_dirs:
        raise FileNotFoundError(f"No artifact directories found in {ds_root}")
    art_dir = scale_dirs[-1]  # latest scale
    meta_path = art_dir / "artifact.json"
    meta = json.loads(meta_path.read_text())

    plane_data = []
    for p in range(N_TOTAL_PLANES):
        plane_path = art_dir / f"plane_{p}.bin"
        arr = np.fromfile(plane_path, dtype=np.uint8)
        assert len(arr) == n_rows, f"plane_{p}: expected {n_rows} rows, got {len(arr)}"
        plane_data.append(arr)

    return ArtifactData(
        path=art_dir,
        dataset=meta.get("dataset", dataset),
        n_rows=int(meta["n_rows"]),
        scale=int(meta["scale"]),
        base_fixed=int(meta["base_fixed"]),
        max_delta=int(meta["max_delta"]),
        active_byte_len=int(meta["active_byte_len"]),
        plane_weight=meta["plane_weight"],
        plane_data=plane_data,
    )


# ---------------------------------------------------------------------------
# Checksum engine
# ---------------------------------------------------------------------------


def compute_checksums(plane_data: list[np.ndarray], segment_size: int, active_byte_len: int = 6) -> dict:
    """Compute reference CRC32 checksums for every (segment, active plane) pair."""
    ref = {}
    n_rows = len(plane_data[0])
    n_segments = (n_rows + segment_size - 1) // segment_size

    for seg_idx in range(n_segments):
        start = seg_idx * segment_size
        end = min(start + segment_size, n_rows)
        for p in range(active_byte_len):
            chunk = plane_data[p][start:end].tobytes()
            ref[(seg_idx, p)] = zlib.crc32(chunk) & 0xFFFFFFFF
    return ref

def verify_checksums(
    plane_data: list[np.ndarray],
    segment_size: int,
    reference: dict,
    segment_indices: list[int],
    active_byte_len: int = 6,
) -> list[tuple[int, int, int, int]]:
    """Verify checksums for specified segments.

    Returns: list of (segment_id, plane_id, expected_crc, actual_crc) for failures.
    """
    n_rows = len(plane_data[0])
    failures = []

    for seg_idx in segment_indices:
        start = seg_idx * segment_size
        end = min(start + segment_size, n_rows)
        for p in range(active_byte_len):
            chunk = plane_data[p][start:end].tobytes()
            actual = zlib.crc32(chunk) & 0xFFFFFFFF
            expected = reference.get((seg_idx, p), actual)
            if actual != expected:
                failures.append((seg_idx, p, expected, actual))
    return failures


# ---------------------------------------------------------------------------
# Corruption injector
# ---------------------------------------------------------------------------


def inject_corruption(
    plane_data: list[np.ndarray],
    corrupt_type: str,
    segment_size: int,
    seed: int = 42,
    segment_idx: int = 0,
    planes: list[int] | None = None,
) -> tuple[list[np.ndarray], str, int, list[int]]:
    """Inject deterministic corruption into a copy of plane data.

    Returns: (corrupted_plane_data, corrupt_type_str, segment_idx, affected_planes)
    """
    rng = random.Random(seed)
    n_rows = len(plane_data[0])

    result = [arr.copy() for arr in plane_data]  # defensive copy
    affected_planes = planes or [4]  # default P4
    start = segment_idx * segment_size
    end = min(start + segment_size, n_rows)
    seg_len = end - start

    if corrupt_type == "none":
        return result, "none", -1, []

    if corrupt_type == "single_plane":
        for p in affected_planes:
            for i in range(start, end):
                result[p][i] ^= 0xFF  # mask=0xFF flips all bits
        detail = f"P{affected_planes[0]}_xorFF_seg{segment_idx}"
        return result, detail, segment_idx, affected_planes

    if corrupt_type == "multi_plane":
        for p in affected_planes:
            for i in range(start, end):
                result[p][i] ^= 0xFF
        detail = f"P{'_'.join(str(ap) for ap in affected_planes)}_xorFF_seg{segment_idx}"
        return result, detail, segment_idx, affected_planes

    if corrupt_type == "zero_invariant":
        p = 6  # P6
        for i in range(start, min(start + 10, end)):
            result[p][i] = 0x01  # flip P6 bytes to non-zero
        detail = f"P6_zero_inv_violation_seg{segment_idx}"
        return result, detail, segment_idx, [6]

    raise ValueError(f"unknown corruption type: {corrupt_type}")


# ---------------------------------------------------------------------------
# Reactive policy engine
# ---------------------------------------------------------------------------


def apply_policy(
    segment_size: int,
    k: int,
    plane: int,
    suspect_count: int,
    epsilon: float,
    active_byte_len: int,
    u_depth_current: int,
    scale: float = 1.0,
) -> tuple[int, str]:
    """Apply reactive policy rules (R0-R6) based on design report §5.

    E_fault is computed in raw FP64 domain for absorption comparison with epsilon.

    Returns: (action_code, reason)
    """
    if suspect_count == 0:
        return (0, "no anomaly")

    weight = PLANE_WEIGHTS[plane] if plane < 6 else (256 if plane == 6 else 1)
    max_E_fault_delta = suspect_count * 255 * weight
    max_E_fault_raw = max_E_fault_delta / scale  # convert to raw FP64 domain

    # P0/P1: always escalate (R2 -> R4) — E_fault huge in any domain
    if plane <= 1:
        return (4, f"P{plane} high-sig, E_fault_raw={max_E_fault_raw:.2e} > any ε")

    # P2/P3: mid-significance — try R3 then decide
    if plane <= 3:
        if u_depth_current > 0:
            return (3, f"P{plane} mid-sig, R3 to reduce U_depth={u_depth_current}")
        if max_E_fault_raw <= epsilon:
            return (1, f"P{plane} absorbable, E_fault_raw={max_E_fault_raw:.2e} ≤ ε={epsilon:.2e}")
        return (4, f"P{plane} mid-sig, E_fault_raw={max_E_fault_raw:.2e} > ε={epsilon:.2e}")

    # P4/P5: low-significance — prefer R1
    if plane <= 5:
        if max_E_fault_raw <= epsilon:
            return (1, f"P{plane} low-sig, absorb, E_fault_raw={max_E_fault_raw:.2e} ≤ ε={epsilon:.2e}")
        return (4, f"P{plane} low-sig but E_fault_raw={max_E_fault_raw:.2e} > ε={epsilon:.2e}")

    # P6/P7: zero-invariant violation — R1
    if plane >= 6:
        return (1, f"P{plane} zero-inv, absorb, E_fault_raw={max_E_fault_raw:.2e} small")

    return (6, "UNCERTIFIED fallback")


# ---------------------------------------------------------------------------
# Query executor
# ---------------------------------------------------------------------------


def run_query(
    artifact: ArtifactData,
    plane_data: list[np.ndarray],
    reference_checksums: dict,
    k: int,
    threshold: float,
    epsilon: float,
    segment_size: int,
    query_idx: int,
    corrupt_type_label: str = "none",
    corrupt_segment: int = -1,
    corrupt_planes_list: list[int] | None = None,
) -> tuple[QueryResult, list[FallbackEvent]]:
    """Execute a progressive COUNT/SUM query with reactive certification."""
    n_rows = artifact.n_rows
    scale = artifact.scale
    base_fixed = artifact.base_fixed
    active_byte_len = artifact.active_byte_len
    q_err = 0.5 / scale
    n_segments = (n_rows + segment_size - 1) // segment_size
    max_undecoded = [256 ** (active_byte_len - kk) - 1 for kk in range(active_byte_len + 1)]

    # Use θ = ≥ (threshold)
    theta = ">="

    result = QueryResult(
        query_idx=query_idx,
        dataset=artifact.dataset,
        n_rows=n_rows,
        segment_size=segment_size,
        k=k,
        threshold=threshold,
        epsilon=epsilon,
        corruption_type=corrupt_type_label,
        corruption_planes=",".join(str(p) for p in (corrupt_planes_list or [])),
        corruption_segment=corrupt_segment,
    )

    fallback_events: list[FallbackEvent] = []

    Q_total = 0
    U_depth_total = 0
    U_quant_total = 0
    U_fault_total = 0
    S_core_low_total = 0.0
    S_core_high_total = 0.0
    swing_low_total = 0.0
    swing_high_total = 0.0
    E_quant_value_total = 0.0
    E_fault_value_total = 0.0
    total_cost = 0.0
    actions_count = [0, 0, 0, 0, 0, 0, 0]  # R0..R6
    segments_failed = 0
    faults_detected = 0

    for seg_idx in range(n_segments):
        start = seg_idx * segment_size
        end = min(start + segment_size, n_rows)
        seg_len = end - start

        # ---- P6/P7 zero-invariant check (before checksums) ----
        zero_inv_violation = False
        for zp in (6, 7):
            if np.any(plane_data[zp][start:end] != 0):
                zero_inv_violation = True

        # ---- Decode at depth k ----
        partial_delta = np.zeros(seg_len, dtype=np.uint64)
        for p in range(k):
            partial_delta += plane_data[p][start:end].astype(np.uint64) * PLANE_WEIGHTS[p]

        # Full decode for U_quant split
        full_delta = np.zeros(seg_len, dtype=np.uint64)
        for p in range(active_byte_len):
            full_delta += plane_data[p][start:end].astype(np.uint64) * PLANE_WEIGHTS[p]

        # Raw FP64 interval at depth k
        x_low = (base_fixed + partial_delta).astype(np.float64) / scale - q_err
        x_high = (base_fixed + partial_delta + max_undecoded[k]).astype(np.float64) / scale + q_err

        # Raw FP64 interval at full depth
        x_full = (base_fixed + full_delta).astype(np.float64) / scale
        x_full_low = x_full - q_err
        x_full_high = x_full + q_err

        # ---- Classification ----
        Q_mask = x_low >= threshold
        D_mask = x_high < threshold
        U_mask = ~(Q_mask | D_mask)

        # Full-depth classification for U split
        Q_full_mask = x_full_low >= threshold
        D_full_mask = x_full_high < threshold
        U_full_mask = ~(Q_full_mask | D_full_mask)

        U_depth_rows = U_mask & ~U_full_mask
        U_quant_rows = U_mask & U_full_mask

        Q_count = int(np.sum(Q_mask))
        U_depth_count = int(np.sum(U_depth_rows))
        U_quant_count = int(np.sum(U_quant_rows))
        U_fault_count = 0

        # SUM core
        S_core_low = float(np.sum(x_low[Q_mask]))
        S_core_high = float(np.sum(x_high[Q_mask]))

        # SUM swing (for U rows)
        # Sensor: base >= 0, all values positive → swing_low = 0, swing_high = sum(x_high of U)
        swing_low = float(np.sum(np.minimum(0, x_low[U_mask]))) if np.any(U_mask) else 0.0
        swing_high = float(np.sum(np.maximum(0, x_high[U_mask]))) if np.any(U_mask) else 0.0

        # E_quant value
        E_quant_seg = (Q_count + U_depth_count + U_quant_count) * q_err

        # ---- Verification (checksums) ----
        # Verify checksums for this segment against reference
        failures = verify_checksums(plane_data, segment_size, reference_checksums, [seg_idx], active_byte_len)
        seg_cost = C_VERIFY  # every segment pays verification

        E_fault_seg = 0.0
        action = 0
        seg_faults_detected = 0
        seg_repaired = False  # true if R4/R5 fully repaired this segment

        if failures:
            seg_faults_detected = len(failures)
            faults_detected += seg_faults_detected
            segments_failed += 1

            # Aggregate failures by plane
            plane_failures: dict[int, int] = {}
            for (_, p, _, _) in failures:
                plane_failures[p] = plane_failures.get(p, 0) + 1

            # Apply policy per affected plane (take highest-severity action)
            worst_action = 0
            worst_reason = ""
            for sus_plane, sus_count in plane_failures.items():
                act, reason = apply_policy(
                    segment_size, k, sus_plane, sus_count,
                    epsilon, active_byte_len, U_depth_count,
                    scale=scale,
                )
                if act > worst_action:
                    worst_action = act
                    worst_reason = reason

                fe = FallbackEvent(
                    query_idx=query_idx,
                    segment_id=seg_idx,
                    plane_id=sus_plane,
                    suspect_count=sus_count,
                    cause=f"checksum_mismatch_{reason}",
                    action_taken=act,
                    action_cost=0.0,
                    E_fault_value=sus_count * 255 * PLANE_WEIGHTS[sus_plane] / scale,
                    certification_after="",
                )
                fallback_events.append(fe)

            action = worst_action

            if action == 1:
                # R1: absorb — widen bound by E_fault
                for sus_plane, sus_count in plane_failures.items():
                    contribution = sus_count * 255 * PLANE_WEIGHTS[sus_plane] / scale
                    E_fault_seg += contribution
                    U_fault_count += 1  # conservative: at least 1 row uncertain
                seg_cost += C_ABSORB

            elif action in (2, 4):
                # R2: re-read or R4: recompute — full repair
                # Cost depends on action
                if action == 2:
                    seg_cost += C_RE_READ
                else:
                    seg_cost += C_RECOMPUTE

                # R4 repairs: use full delta, E_fault = 0
                # Use the full-depth classification as repair
                Q_count = int(np.sum(Q_full_mask))
                U_depth_count = 0
                U_quant_count = int(np.sum(U_full_mask))
                U_fault_count = 0
                E_fault_seg = 0.0
                seg_repaired = True
                # Recompute SUM with full precision
                S_core_low = float(np.sum(x_full_low[Q_full_mask]))
                S_core_high = float(np.sum(x_full_high[Q_full_mask]))
                swing_low = float(np.sum(np.minimum(0, x_full_low[U_full_mask]))) if np.any(U_full_mask) else 0.0
                swing_high = float(np.sum(np.maximum(0, x_full_high[U_full_mask]))) if np.any(U_full_mask) else 0.0
                E_quant_seg = (Q_count + U_quant_count) * q_err

            elif action == 3:
                # R3: scan deeper — reduces U_depth only
                # Simulate scanning to full depth for this segment
                seg_cost += C_R3
                # After R3, check if remaining E_fault is absorbable
                total_E_fault_check = 0.0
                for sus_plane, sus_count in plane_failures.items():
                    total_E_fault_check += sus_count * 255 * PLANE_WEIGHTS[sus_plane] / scale
                if total_E_fault_check <= epsilon:
                    action = 1  # can now absorb
                    Q_count = int(np.sum(Q_full_mask))
                    U_depth_count = 0
                    U_quant_count = int(np.sum(U_full_mask))
                    U_fault_count = len(plane_failures)
                    E_fault_seg = total_E_fault_check
                    swing_low = float(np.sum(np.minimum(0, x_full_low[U_full_mask]))) if np.any(U_full_mask) else 0.0
                    swing_high = float(np.sum(np.maximum(0, x_full_high[U_full_mask]))) if np.any(U_full_mask) else 0.0
                else:
                    action = 4  # upgrade to R4
                    seg_cost += C_RECOMPUTE
                    Q_count = int(np.sum(Q_full_mask))
                    U_depth_count = 0
                    U_quant_count = int(np.sum(U_full_mask))
                    U_fault_count = 0
                    E_fault_seg = 0.0
                    seg_repaired = True
                    swing_low = float(np.sum(np.minimum(0, x_full_low[U_full_mask]))) if np.any(U_full_mask) else 0.0
                    swing_high = float(np.sum(np.maximum(0, x_full_high[U_full_mask]))) if np.any(U_full_mask) else 0.0

            elif action == 5:
                # R5: full fallback — repair all
                seg_cost += C_FALLBACK
                Q_count = int(np.sum(Q_full_mask))
                U_depth_count = 0
                U_quant_count = int(np.sum(U_full_mask))
                U_fault_count = 0
                E_fault_seg = 0.0
                seg_repaired = True
                S_core_low = float(np.sum(x_full_low[Q_full_mask]))
                S_core_high = float(np.sum(x_full_high[Q_full_mask]))
                swing_low = float(np.sum(np.minimum(0, x_full_low[U_full_mask]))) if np.any(U_full_mask) else 0.0
                swing_high = float(np.sum(np.maximum(0, x_full_high[U_full_mask]))) if np.any(U_full_mask) else 0.0
                E_quant_seg = (Q_count + U_quant_count) * q_err

            elif action == 6:
                seg_cost += C_UNCERTIFIED
                result.certification = "UNCERTIRED"

            # Update fallback event costs
            for fe in fallback_events:
                if fe.query_idx == query_idx and fe.segment_id == seg_idx:
                    fe.action_cost = seg_cost - C_VERIFY
                    fe.certification_after = result.certification or "CERTIFIED_WIDENED"

        elif zero_inv_violation:
            # P6/P7 zero-invariant violation — R1 absorb (E_fault is tiny)
            faults_detected += 1
            segments_failed += 1
            for zp in (6, 7):
                nonzero_count = int(np.sum(plane_data[zp][start:end] != 0))
                if nonzero_count > 0:
                    E_fault_seg += nonzero_count * 255 * (256 if zp == 6 else 1) / scale
                    fe = FallbackEvent(
                        query_idx=query_idx, segment_id=seg_idx,
                        plane_id=zp, suspect_count=nonzero_count,
                        cause=f"P{zp}_zero_invariant_violation",
                        action_taken=1, action_cost=0.0,
                        E_fault_value=nonzero_count * 255 * (256 if zp == 6 else 1) / scale,
                        certification_after="CERTIFIED_WIDENED",
                    )
                    fallback_events.append(fe)
            action = 1
            seg_cost += C_ABSORB

        actions_count[action] += 1
        total_cost += seg_cost
        total_cost += 1.0  # T_progressive per segment

        # Accumulate segment results
        Q_total += Q_count
        U_depth_total += U_depth_count
        U_quant_total += U_quant_count
        U_fault_total += U_fault_count

        S_core_low_total += S_core_low
        S_core_high_total += S_core_high
        swing_low_total += swing_low
        swing_high_total += swing_high
        E_quant_value_total += E_quant_seg
        E_fault_value_total += E_fault_seg

        # Track verified bytes
        seg_bytes = seg_len * len(plane_data)

    # ---- Compose query-level result ----
    S_low_total = S_core_low_total + swing_low_total - E_quant_value_total
    S_high_total = S_core_high_total + swing_high_total + E_quant_value_total

    # COUNT bound
    U_total = U_depth_total + U_quant_total + U_fault_total
    width_count = U_total  # raw COUNT width
    width_sum = S_high_total - S_low_total

    # Certification
    # R4/R5 repair eliminates E_fault → CERTIFIED_BOUNDED (no widening)
    all_repaired = (E_fault_value_total == 0.0)

    if result.certification == "UNCERTIRED":
        result.certification = "UNCERTIFIED"
    elif faults_detected == 0 or all_repaired:
        result.certification = "CERTIFIED_BOUNDED"
    else:
        # Fault detected and NOT fully repaired → check if bound fits within ε
        if width_count <= epsilon or width_sum <= epsilon:
            result.certification = "CERTIFIED_WIDENED"
        else:
            result.certification = "UNCERTIFIED"

    result.Q_total = Q_total
    result.U_depth_total = U_depth_total
    result.U_quant_total = U_quant_total
    result.U_fault_total = U_fault_total
    result.S_low = S_low_total
    result.S_high = S_high_total
    result.E_quant_total = E_quant_value_total
    result.E_fault_total = E_fault_value_total
    result.segments_verified = n_segments
    result.segments_failed = segments_failed
    result.faults_detected = faults_detected
    result.actions = actions_count
    result.total_cost = total_cost
    result.width = width_sum
    result.relative_width = width_sum / max(abs(S_high_total), 1e-300)

    return result, fallback_events


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


def run_scenarios(
    artifact: ArtifactData,
    segment_size: int,
    k_values: list[int],
    thresholds: list[float],
    epsilons: list[float],
) -> tuple[list[QueryResult], list[FallbackEvent]]:
    """Run all evaluation scenarios."""
    reference_checksums = compute_checksums(artifact.plane_data, segment_size, artifact.active_byte_len)

    queries: list[QueryResult] = []
    all_events: list[FallbackEvent] = []
    query_idx = 0

    corruption_scenarios = [
        ("none", -1, []),
        ("single_plane", 500, [4]),    # P4 corruption on segment 500
        ("single_plane", 500, [0]),    # P0 corruption on segment 500
        ("multi_plane", 500, [2, 3]),   # Multi-plane P2+P3
        ("zero_invariant", 500, [6]),   # P6 zero-invariant violation
    ]

    for corr_type, corr_seg, corr_planes in corruption_scenarios:
        for k in k_values:
            for thresh in thresholds:
                for eps in epsilons:
                    # Inject corruption (or none)
                    corrupted_data, label, actual_seg, actual_planes = inject_corruption(
                        artifact.plane_data,
                        corr_type,
                        segment_size,
                        seed=20260602,
                        segment_idx=corr_seg,
                        planes=corr_planes,
                    )

                    result, events = run_query(
                        artifact,
                        corrupted_data,
                        reference_checksums,
                        k,
                        thresh,
                        eps,
                        segment_size,
                        query_idx,
                        corrupt_type_label=label,
                        corrupt_segment=actual_seg,
                        corrupt_planes_list=actual_planes,
                    )

                    queries.append(result)
                    all_events.extend(events)

                    # Determine row label
                    corr_label = label
                    print(
                        f"  Q{query_idx}: k={k} T={thresh} ε={eps} "
                        f"corr={corr_label} "
                        f"→ cert={result.certification} "
                        f"U={result.U_depth_total}+{result.U_quant_total}+{result.U_fault_total}"
                        f" faults={result.faults_detected}"
                        f" cost={result.total_cost:.2f}"
                    )
                    query_idx += 1

    return queries, all_events


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def write_matrix_csv(path: Path, queries: list[QueryResult]) -> None:
    """Write reactive_query_matrix.csv."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id", "dataset", "n_rows", "segment_size", "k", "threshold", "epsilon",
            "corruption_type", "corruption_planes", "corruption_segment",
            "Q_total", "U_depth", "U_quant", "U_fault",
            "S_low", "S_high",
            "E_quant_value", "E_fault_value",
            "width", "relative_width",
            "certification",
            "segments_verified", "segments_failed", "faults_detected",
            "R0", "R1", "R2", "R3", "R4", "R5", "R6",
            "total_cost",
        ])
        for q in queries:
            w.writerow([
                "", q.dataset, q.n_rows, q.segment_size, q.k, q.threshold, q.epsilon,
                q.corruption_type, q.corruption_planes, q.corruption_segment,
                q.Q_total, q.U_depth_total, q.U_quant_total, q.U_fault_total,
                f"{q.S_low:.6e}", f"{q.S_high:.6e}",
                f"{q.E_quant_total:.6e}", f"{q.E_fault_total:.6e}",
                f"{q.width:.6e}", f"{q.relative_width:.6e}",
                q.certification,
                q.segments_verified, q.segments_failed, q.faults_detected,
                q.actions[0], q.actions[1], q.actions[2], q.actions[3],
                q.actions[4], q.actions[5], q.actions[6],
                f"{q.total_cost:.4f}",
            ])


def write_cert_summary_csv(path: Path, queries: list[QueryResult]) -> None:
    """Write certification_summary.csv."""
    total = len(queries)
    bounded = sum(1 for q in queries if q.certification == "CERTIFIED_BOUNDED")
    widened = sum(1 for q in queries if q.certification == "CERTIFIED_WIDENED")
    uncert = sum(1 for q in queries if q.certification == "UNCERTIFIED")

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["certification", "count", "total", "rate"])
        w.writerow(["CERTIFIED_BOUNDED", bounded, total, f"{bounded / max(total, 1):.4f}"])
        w.writerow(["CERTIFIED_WIDENED", widened, total, f"{widened / max(total, 1):.4f}"])
        w.writerow(["UNCERTIFIED", uncert, total, f"{uncert / max(total, 1):.4f}"])
        w.writerow(["CERTIFIED_COMBINED", bounded + widened, total, f"{(bounded + widened) / max(total, 1):.4f}"])

    return bounded, widened, uncert


def write_fallback_trace_csv(path: Path, events: list[FallbackEvent]) -> None:
    """Write fallback_trace.csv."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "query_idx", "segment_id", "plane_id", "suspect_count",
            "cause", "action_taken", "action_cost",
            "E_fault_value", "certification_after",
        ])
        for e in events:
            w.writerow([
                e.query_idx, e.segment_id, e.plane_id, e.suspect_count,
                e.cause, e.action_taken, f"{e.action_cost:.4f}",
                f"{e.E_fault_value:.6e}", e.certification_after,
            ])


def write_run_meta(path: Path, queries: list[QueryResult], elapsed: float) -> None:
    """Write run_meta.txt."""
    bounded = sum(1 for q in queries if q.certification == "CERTIFIED_BOUNDED")
    widened = sum(1 for q in queries if q.certification == "CERTIFIED_WIDENED")
    uncert = sum(1 for q in queries if q.certification == "UNCERTIFIED")
    total = len(queries)

    fallbacks = sum(q.actions[4] + q.actions[5] for q in queries)

    # Compute verification overhead as fraction of speed margin
    # speed_margin = fraction * T_raw_fused
    # T_progressive = T_raw_fused - speed_margin
    # If T_progressive = 1.0 (normalized):
    #   T_raw_fused = 1.0 / (1 - fraction)
    #   speed_margin = fraction / (1 - fraction)
    #   verification_pct_of_margin = C_VERIFY / speed_margin * 100
    cons_margin = 0.16 / (1 - 0.16)  # speed margin when T_prog = 1.0
    opt_margin = 0.83 / (1 - 0.83)
    cons_pct = C_VERIFY / cons_margin * 100
    opt_pct = C_VERIFY / opt_margin * 100
    risk_flag = "YELLOW" if cons_pct > 50 else "GREEN"

    vf_total = sum(q.faults_detected for q in queries)
    actions_r4 = sum(q.actions[4] for q in queries)
    actions_r5 = sum(q.actions[5] for q in queries)
    actions_r3 = sum(q.actions[3] for q in queries)

    lines = [
        f"Run ID: p3b_cpu_prototype_{time.strftime('%Y%m%d_%H%M%S')}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"",
        f"=== Configuration ===",
        f"Dataset: {queries[0].dataset if queries else 'N/A'}",
        f"N rows: {queries[0].n_rows if queries else 0}",
        f"Segment size: {queries[0].segment_size if queries else 0}",
        f"Total queries: {total}",
        f"",
        f"=== Certification Summary ===",
        f"CERTIFIED_BOUNDED: {bounded} ({bounded / max(total, 1) * 100:.1f}%)",
        f"CERTIFIED_WIDENED: {widened} ({widened / max(total, 1) * 100:.1f}%)",
        f"UNCERTIFIED: {uncert} ({uncert / max(total, 1) * 100:.1f}%)",
        f"Combined certification rate: {(bounded + widened) / max(total, 1) * 100:.1f}%",
        f"Certified BOUNDED: {bounded} (no widening from E_fault)",
        f"Certified WIDENED: {widened} (E_fault absorbed within ε)",
        f"",
        f"=== Fallback Stats ===",
        f"Total anomalies detected: {vf_total}",
        f"R1 (absorb) actions: {int(sum(q.actions[1] for q in queries))}",
        f"R3 (scan deeper) actions: {actions_r3}",
        f"R4 (recompute) actions: {actions_r4}",
        f"R5 (full fallback) actions: {actions_r5}",
        f"Total fallback actions (R4+R5): {fallbacks}",
        f"Avg fallback per query: {fallbacks / max(total, 1):.2f}",
        f"",
        f"=== Cost Model (normalized, T_progressive=1.0 per segment) ===",
        f"Average total cost per query: {sum(q.total_cost for q in queries) / max(total, 1):.2f}",
        f"C_verify per segment: {C_VERIFY} (15% of T_progressive)",
        f"C_absorb (R1): {C_ABSORB}",
        f"C_R3 (scan deeper): {C_R3}",
        f"C_re-read (R2): {C_RE_READ}",
        f"C_recompute (R4): {C_RECOMPUTE}",
        f"C_fallback (R5): {C_FALLBACK}",
        f"",
        f"=== Verification Overhead Budget Analysis ===",
        f"Verification overhead: 15% of baseline query (T_progressive)",
        f"Speed margin (paper H200 warm E2E): 16-83% of raw-fused CUDA",
        f"Speed margin at T_progressive=1.0 (normalized): conservative={cons_margin:.2f}, optimistic={opt_margin:.2f}",
        f"Verification overhead as % of speed margin:",
        f"  Conservative (16% margin): {cons_pct:.1f}% of margin consumed by verification alone",
        f"  Optimistic (83% margin): {opt_pct:.1f}% of margin consumed by verification alone",
        f"Risk flag: {risk_flag} — verification overhead "
        f"{'MAY EXCEED' if cons_pct > 100 else 'fits within'} speed margin in conservative scenario",
        f"",
        f"Note: P3-B is CPU-only. No GPU latency verdict. Binding hardware latency",
        f"STOP at P3-C (H100) against re-baselined numbers.",
        f"",
        f"=== Execution ===",
        f"Wall time: {elapsed:.1f}s",
        f"Seed: 20260602",
    ]

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="P3-B CPU sparse reactive prototype")
    parser.add_argument("--dataset", default="sensor", help="Dataset name")
    parser.add_argument("--n-rows", type=int, default=100_000_000, help="Number of rows")
    parser.add_argument("--segment-size", type=int, default=1024, help="Segment size (rows)")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 4, 6], help="Depth values")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[20.0, 25.0], help="Thresholds")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1e10, 1e12], help="Error budgets")
    parser.add_argument("--seed", type=int, default=20260602, help="Random seed")
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 60)
    print("Phase 3 P3-B CPU Sparse Reactive Prototype")
    print("=" * 60)

    # Load artifact
    print(f"\nLoading artifact: {args.dataset} N={args.n_rows}")
    artifact = load_artifact(args.dataset, args.n_rows)
    print(f"  scale={artifact.scale}, base_fixed={artifact.base_fixed}")
    print(f"  active_byte_len={artifact.active_byte_len}")
    print(f"  plane_weights={artifact.plane_weight}")

    # Run scenarios
    print(f"\nRunning {len(args.k)} × {len(args.thresholds)} × {len(args.epsilons)} "
          f"× 5 corruption scenarios = {len(args.k) * len(args.thresholds) * len(args.epsilons) * 5} queries")
    print(f"  k = {args.k}")
    print(f"  thresholds = {args.thresholds}")
    print(f"  epsilons = {args.epsilons}")
    print(f"  segment_size = {args.segment_size}")

    queries, events = run_scenarios(
        artifact,
        args.segment_size,
        args.k,
        args.thresholds,
        args.epsilons,
    )

    # Determine run ID
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = RESULTS_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write outputs
    print(f"\nWriting outputs to {out_dir}")
    write_matrix_csv(out_dir / "reactive_query_matrix.csv", queries)

    bounded, widened, uncert = write_cert_summary_csv(out_dir / "certification_summary.csv", queries)

    write_fallback_trace_csv(out_dir / "fallback_trace.csv", events)

    elapsed = time.time() - t0
    write_run_meta(out_dir / "run_meta.txt", queries, elapsed)

    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  CERTIFIED_BOUNDED: {bounded}")
    print(f"  CERTIFIED_WIDENED: {widened}")
    print(f"  UNCERTIFIED: {uncert}")
    print(f"  Certification rate: {(bounded + widened) / max(len(queries), 1) * 100:.1f}%")
    print(f"  Total fallback actions: {sum(q.actions[4] + q.actions[5] for q in queries)}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
