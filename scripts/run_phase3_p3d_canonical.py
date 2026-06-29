#!/usr/bin/env python3
"""Phase 3 P3-D Canonical Reactive Matrix (CPU certification pass).

Runs the full canonical matrix (B0-B4) for 4 synthetic datasets on CPU,
generating certified answers for all 77,376 queries (without CESM-ATM Q).

Usage:
    python3 scripts/run_phase3_p3d_canonical.py [--run-tag TAG]

Output:
    ${WORK_DIR}/results/reliability_layer1/phase3/p3d_canonical/<run_id>/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

RESULTS_ROOT = Path("${WORK_DIR}/results/reliability_layer1/phase3/p3d_canonical")
ARTIFACT_ROOT = Path("${WORK_DIR}/datasets/artifacts_phase1_5b")

N_TOTAL_PLANES = 8
PLANE_WEIGHTS = [1 << (8 * (5 - p)) for p in range(6)]

# Normalized cost model (T_progressive = 1.0)
C_VERIFY = 0.15
C_ABSORB = 0.00
C_R3 = 0.05
C_RE_READ = 1.0
C_RECOMPUTE = 3.0
C_FALLBACK = 5.0
C_UNCERTIFIED = 0.0

# Sweep dimensions
DATASETS = ["sensor", "uniform", "heavy_tailed", "zipfian"]
K_VALUES = [1, 2, 4, 6]
EPSILONS = [1e6, 1e8, 1e10, 1e12]
CORRUPTION_SCENARIOS = ["none", "low_sig_p4", "high_sig_p0", "multi_plane_p2_p3", "zero_invariant_p6"]
FAULT_FRACTIONS = [0.00001, 0.001, 0.01, 0.1]  # 0.001%, 0.1%, 1%, 10%
SEEDS = [0, 1, 2, 3, 4]

# Uniform dataset thresholds (known)
UNIFORM_THRESHOLDS = {"low": 30.0, "medium": 50.0, "high": 70.0}

# Speed margins (from paper H200 warm E2E, re-baseline on H100)
SPEED_MARGIN_CONSERVATIVE = 0.16 / (1 - 0.16)  # 16% margin → normalized
SPEED_MARGIN_OPTIMISTIC = 0.83 / (1 - 0.83)  # 83% margin → normalized

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FaultPlan:
    segment_id: int
    plane_id: int
    suspect_count: int
    injection_type: str  # "xor_ff" or "set_nonzero"


@dataclass
class QueryResult:
    run_id: str = ""
    dataset: str = ""
    n_rows: int = 0
    segment_size: int = 0
    baseline: str = ""
    k: int = 0
    threshold: float = 0.0
    epsilon: float = 0.0
    operator: str = ""
    scenario: str = ""
    fault_fraction: float = 0.0
    seed: int = 0
    Q_total: int = 0
    U_depth: int = 0
    U_quant: int = 0
    U_fault: int = 0
    S_low: float = 0.0
    S_high: float = 0.0
    E_quant_value: float = 0.0
    E_fault_value: float = 0.0
    width: float = 0.0
    relative_width: float = 0.0
    certification: str = ""
    segments_verified: int = 0
    segments_failed: int = 0
    faults_detected: int = 0
    total_cost: float = 0.0
    R0: int = 0
    R1: int = 0
    R2: int = 0
    R3: int = 0
    R4: int = 0
    R5: int = 0
    R6: int = 0
    override_driven: bool = False
    override_active: bool = False
    r4_override_forced: int = 0
    r4_math_forced: int = 0


@dataclass
class ArtifactData:
    path: Path
    dataset: str
    n_rows: int
    scale: int
    base_fixed: int
    active_byte_len: int
    plane_weight: list[int]
    plane_data: list[np.ndarray]


# ---------------------------------------------------------------------------
# Artifact loader
# ---------------------------------------------------------------------------


def find_artifact_dir(dataset: str, n_rows: int = 100_000_000) -> Path:
    ds_root = ARTIFACT_ROOT / dataset / f"n{n_rows}"
    scale_dirs = sorted(ds_root.iterdir())
    if not scale_dirs:
        raise FileNotFoundError(f"No artifact directories in {ds_root}")
    return scale_dirs[-1]


def load_artifact(dataset: str, n_rows: int = 100_000_000) -> ArtifactData:
    art_dir = find_artifact_dir(dataset, n_rows)
    meta = json.loads((art_dir / "artifact.json").read_text())

    plane_data = []
    for p in range(N_TOTAL_PLANES):
        arr = np.fromfile(art_dir / f"plane_{p}.bin", dtype=np.uint8)
        assert len(arr) == n_rows, f"plane_{p}: expected {n_rows}, got {len(arr)}"
        plane_data.append(arr)

    return ArtifactData(
        path=art_dir,
        dataset=meta.get("dataset", dataset),
        n_rows=int(meta["n_rows"]),
        scale=int(meta["scale"]),
        base_fixed=int(meta["base_fixed"]),
        active_byte_len=int(meta["active_byte_len"]),
        plane_weight=meta["plane_weight"],
        plane_data=plane_data,
    )


# ---------------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------------


def compute_thresholds(artifact: ArtifactData) -> dict[str, float]:
    if artifact.dataset == "uniform":
        return UNIFORM_THRESHOLDS

    full_delta = np.zeros(artifact.n_rows, dtype=np.uint64)
    for p in range(artifact.active_byte_len):
        full_delta += artifact.plane_data[p].astype(np.uint64) * PLANE_WEIGHTS[p]
    values = (artifact.base_fixed + full_delta).astype(np.float64) / artifact.scale

    sample = values[::100]  # sample every 100th row to reduce cost
    p30 = float(np.percentile(sample, 30))
    p50 = float(np.percentile(sample, 50))
    p70 = float(np.percentile(sample, 70))

    return {"low": p30, "medium": p50, "high": p70}


# ---------------------------------------------------------------------------
# Checksum engine
# ---------------------------------------------------------------------------


def compute_reference_checksums(plane_data: list[np.ndarray], segment_size: int, active_byte_len: int = 6) -> dict:
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


# ---------------------------------------------------------------------------
# Fault plan generation
# ---------------------------------------------------------------------------


def generate_fault_plan(
    scenario: str,
    fault_fraction: float,
    seed: int,
    n_segments: int,
    segment_size: int,
) -> list[FaultPlan]:
    """Generate deterministic fault plan for a given (scenario, fraction, seed)."""
    plans: list[FaultPlan] = []
    rng = random.Random(seed)

    if scenario == "none":
        return plans

    n_faulted = max(1, int(n_segments * fault_fraction))
    spacing = max(1, n_segments // n_faulted)
    base_offset = seed * 2

    if scenario == "low_sig_p4":
        plane_id = 4
        suspect_count = segment_size
        for i in range(n_faulted):
            seg_id = (base_offset + i * spacing) % n_segments
            plans.append(FaultPlan(seg_id, plane_id, suspect_count, "xor_ff"))

    elif scenario == "high_sig_p0":
        plane_id = 0
        suspect_count = segment_size
        for i in range(n_faulted):
            seg_id = (base_offset + i * spacing) % n_segments
            plans.append(FaultPlan(seg_id, plane_id, suspect_count, "xor_ff"))

    elif scenario == "multi_plane_p2_p3":
        suspect_count = segment_size
        for i in range(n_faulted):
            seg_id = (base_offset + i * spacing) % n_segments
            plans.append(FaultPlan(seg_id, 2, suspect_count, "xor_ff"))
            plans.append(FaultPlan(seg_id, 3, suspect_count, "xor_ff"))

    elif scenario == "zero_invariant_p6":
        plane_id = 6
        suspect_count = 10  # flip 10 bytes, not all
        for i in range(n_faulted):
            seg_id = (base_offset + i * spacing) % n_segments
            plans.append(FaultPlan(seg_id, plane_id, suspect_count, "set_nonzero"))

    return plans


def apply_fault_plan(plane_data: list[np.ndarray], plans: list[FaultPlan]) -> list[np.ndarray]:
    """Return NEW plane data with faults applied."""
    result = [arr.copy() for arr in plane_data]
    n_rows = len(plane_data[0])
    for plan in plans:
        start = plan.segment_id * 1024
        end = min(start + 1024, n_rows)
        if plan.injection_type == "xor_ff":
            result[plan.plane_id][start:end] ^= 0xFF
        elif plan.injection_type == "set_nonzero":
            result[plan.plane_id][start:min(start + plan.suspect_count, end)] = 0x01
    return result


# ---------------------------------------------------------------------------
# Reactive policy engine
# ---------------------------------------------------------------------------


def apply_policy(
    plane: int,
    suspect_count: int,
    epsilon: float,
    scale: float,
    active_byte_len: int = 6,
    u_depth_current: int = 0,
) -> tuple[int, str]:
    """Returns (action_code, reason)."""
    if suspect_count == 0:
        return (0, "no anomaly")

    weight = PLANE_WEIGHTS[plane] if plane < 6 else (256 if plane == 6 else 1)
    max_E_fault_raw = suspect_count * 255 * weight / scale

    if plane <= 1:
        return (4, f"P{plane} high-sig, E_fault={max_E_fault_raw:.2e} > any ε")

    if plane <= 3:
        if u_depth_current > 0:
            return (3, f"P{plane} mid-sig, R3 to reduce U_depth={u_depth_current}")
        if max_E_fault_raw <= epsilon:
            return (1, f"P{plane} absorb, E_fault={max_E_fault_raw:.2e} ≤ ε={epsilon:.2e}")
        return (4, f"P{plane} mid-sig, E_fault={max_E_fault_raw:.2e} > ε={epsilon:.2e}")

    if plane <= 5:
        if max_E_fault_raw <= epsilon:
            return (1, f"P{plane} low-sig, absorb, E_fault={max_E_fault_raw:.2e} ≤ ε={epsilon:.2e}")
        return (4, f"P{plane} low-sig, E_fault={max_E_fault_raw:.2e} > ε={epsilon:.2e}")

    if plane >= 6:
        return (1, f"P{plane} zero-inv, absorb, E_fault={max_E_fault_raw:.2e} small")

    return (6, "UNCERTIFIED fallback")


# ---------------------------------------------------------------------------
# Query executor
# ---------------------------------------------------------------------------


def prepare_decoded(
    artifact: ArtifactData,
    plane_data: list[np.ndarray],
    k: int,
    segment_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-decode entire dataset at depth k and full depth.

    Uses vectorized full-array operations for speed.
    Returns (x_low, x_high, x_full_low, x_full_high).
    """
    n_rows = artifact.n_rows
    scale = artifact.scale
    base_fixed = artifact.base_fixed
    active_byte_len = artifact.active_byte_len
    q_err = 0.5 / scale
    max_undecoded = [256 ** (active_byte_len - kk) - 1 for kk in range(active_byte_len + 1)]

    # Full-array decode at depth k
    partial = np.zeros(n_rows, dtype=np.uint64)
    for p in range(k):
        partial += plane_data[p].astype(np.uint64) * PLANE_WEIGHTS[p]

    # Full-array decode at depth 6
    full = np.zeros(n_rows, dtype=np.uint64)
    for p in range(active_byte_len):
        full += plane_data[p].astype(np.uint64) * PLANE_WEIGHTS[p]

    # Float64 conversion (vectorized)
    fp64_scale = float(scale)
    fp64_base = float(base_fixed)
    fp64_q = float(q_err)
    partial_f = partial.astype(np.float64)
    full_f = full.astype(np.float64)

    x_low = (fp64_base + partial_f) / fp64_scale - fp64_q
    x_high = (fp64_base + partial_f + max_undecoded[k]) / fp64_scale + fp64_q
    x_full = (fp64_base + full_f) / fp64_scale
    x_full_low = x_full - fp64_q
    x_full_high = x_full + fp64_q

    return x_low, x_high, x_full_low, x_full_high


def classify_and_accum(
    x_low: np.ndarray,
    x_high: np.ndarray,
    x_full_low: np.ndarray,
    x_full_high: np.ndarray,
    threshold: float,
    q_err: float,
    n_segments: int,
    segment_size: int,
    faulted_segments: set[int],
    seg_fault_planes: dict[int, list[tuple[int, int]]],
    epsilon: float,
    scale: float,
    baseline: str,
    active_byte_len: int = 6,
) -> QueryResult:
    """Classify entire arrays at once, then handle faulted segments individually."""
    n_rows = len(x_low)

    # ---- Full-array classification (vectorized, no Python loop) ----
    Q_all = x_low >= threshold
    D_all = x_high < threshold
    U_all = ~(Q_all | D_all)
    Qf_all = x_full_low >= threshold
    Df_all = x_full_high < threshold
    Uf_all = ~(Qf_all | Df_all)

    U_depth_mask = U_all & ~Uf_all
    U_quant_mask = U_all & Uf_all

    Q_total = int(np.sum(Q_all))
    U_depth_total = int(np.sum(U_depth_mask))
    U_quant_total = int(np.sum(U_quant_mask))
    U_fault_total = 0

    S_core_low_total = float(np.sum(x_low[Q_all]))
    S_core_high_total = float(np.sum(x_high[Q_all]))

    has_u = bool(np.any(U_all))
    swing_low_total = float(np.sum(np.minimum(0, x_low[U_all]))) if has_u else 0.0
    swing_high_total = float(np.sum(np.maximum(0, x_high[U_all]))) if has_u else 0.0

    E_quant_value_total = (Q_total + U_depth_total + U_quant_total) * q_err
    E_fault_value_total = 0.0

    total_cost = n_segments * (1.0 + C_VERIFY)  # Default: all segments pay 1.0 + C_VERIFY
    actions_count = [0, 0, 0, 0, 0, 0, 0]
    segments_failed = 0
    faults_detected = 0
    override_driven = False
    r4_override_forced = 0
    r4_math_forced = 0

    # Handle baselines
    if baseline in ("B0", "B3"):
        # No verification cost
        total_cost = n_segments * 1.0
        rs = QueryResult(
            Q_total=Q_total, U_depth=U_depth_total, U_quant=U_quant_total,
            S_low=S_core_low_total + swing_low_total - E_quant_value_total,
            S_high=S_core_high_total + swing_high_total + E_quant_value_total,
            E_quant_value=E_quant_value_total,
            width=(S_core_high_total + swing_high_total + E_quant_value_total) -
                  (S_core_low_total + swing_low_total - E_quant_value_total),
            certification="CERTIFIED_BOUNDED",
            segments_verified=n_segments,
            total_cost=total_cost,
        )
        rs.relative_width = rs.width / max(abs(rs.S_high), 1e-300)
        return rs

    if not faulted_segments:
        # No faults: all segments certified
        rs = QueryResult(
            Q_total=Q_total, U_depth=U_depth_total, U_quant=U_quant_total,
            S_low=S_core_low_total + swing_low_total - E_quant_value_total,
            S_high=S_core_high_total + swing_high_total + E_quant_value_total,
            E_quant_value=E_quant_value_total,
            width=(S_core_high_total + swing_high_total + E_quant_value_total) -
                  (S_core_low_total + swing_low_total - E_quant_value_total),
            certification="CERTIFIED_BOUNDED",
            segments_verified=n_segments,
            total_cost=total_cost,
            R0=n_segments,
        )
        rs.relative_width = rs.width / max(abs(rs.S_high), 1e-300)
        return rs

    # ---- Handle faulted segments (small minority) ----
    # For non-faulted segments, the full-array results are correct.
    # For faulted segments, we need to subtract the clean contribution and
    # add the post-fault-handling contribution.

    faulted_sorted = sorted(faulted_segments)
    n_faulted = len(faulted_sorted)
    actions_count[0] = n_segments - n_faulted  # R0 for clean segments

    for seg_idx in faulted_sorted:
        start = seg_idx * segment_size
        end = min(start + segment_size, n_rows)

        # Remove clean segment contribution
        Q_mask = Q_all[start:end]
        U_mask = U_all[start:end]
        Qf_mask = Qf_all[start:end]
        Uf_mask = Uf_all[start:end]

        seg_Q = int(np.sum(Q_mask))
        seg_UD = int(np.sum(U_depth_mask[start:end]))
        seg_UQ = int(np.sum(U_quant_mask[start:end]))
        seg_SL = float(np.sum(x_low[start:end][Q_mask]))
        seg_SH = float(np.sum(x_high[start:end][Q_mask]))
        seg_has_u = bool(np.any(U_mask))
        seg_swing_l = float(np.sum(np.minimum(0, x_low[start:end][U_mask]))) if seg_has_u else 0.0
        seg_swing_h = float(np.sum(np.maximum(0, x_high[start:end][U_mask]))) if seg_has_u else 0.0
        seg_Eq = (seg_Q + seg_UD + seg_UQ) * q_err

        Q_total -= seg_Q
        U_depth_total -= seg_UD
        U_quant_total -= seg_UQ
        S_core_low_total -= seg_SL
        S_core_high_total -= seg_SH
        swing_low_total -= seg_swing_l
        swing_high_total -= seg_swing_h
        E_quant_value_total -= seg_Eq

        # Remove default cost for this segment
        total_cost -= 1.0 + C_VERIFY
        actions_count[0] -= 1

        # Now apply fault handling
        fp_list = seg_fault_planes[seg_idx]
        faults_detected += len(fp_list)
        segments_failed += 1
        seg_cost = C_VERIFY
        action = 0
        E_fault_seg = 0.0
        Q_count_r = 0
        U_depth_r = 0
        U_quant_r = 0
        U_fault_r = 0
        S_core_low_r = 0.0
        S_core_high_r = 0.0
        swing_low_r = 0.0
        swing_high_r = 0.0
        E_quant_r = 0.0

        if baseline == "B1":
            action = 4
            seg_cost += C_RECOMPUTE
            Q_count_r = int(np.sum(Qf_mask))
            U_quant_r = int(np.sum(Uf_mask))
            S_core_low_r = float(np.sum(x_full_low[start:end][Qf_mask]))
            S_core_high_r = float(np.sum(x_full_high[start:end][Qf_mask]))
            has_uf = bool(np.any(Uf_mask))
            swing_low_r = float(np.sum(np.minimum(0, x_full_low[start:end][Uf_mask]))) if has_uf else 0.0
            swing_high_r = float(np.sum(np.maximum(0, x_full_high[start:end][Uf_mask]))) if has_uf else 0.0
            E_quant_r = (Q_count_r + U_quant_r) * q_err

        elif baseline == "B2":
            action = 4
            seg_cost += C_RECOMPUTE
            Q_count_r = int(np.sum(Qf_mask))
            U_quant_r = int(np.sum(Uf_mask))
            S_core_low_r = float(np.sum(x_full_low[start:end][Qf_mask]))
            S_core_high_r = float(np.sum(x_full_high[start:end][Qf_mask]))
            has_uf = bool(np.any(Uf_mask))
            swing_low_r = float(np.sum(np.minimum(0, x_full_low[start:end][Uf_mask]))) if has_uf else 0.0
            swing_high_r = float(np.sum(np.maximum(0, x_full_high[start:end][Uf_mask]))) if has_uf else 0.0
            E_quant_r = (Q_count_r + U_quant_r) * q_err

        elif baseline == "B4":
            worst_action = 0
            for plane_id, sus_count in fp_list:
                act, _ = apply_policy(plane_id, sus_count, epsilon, scale, active_byte_len, U_depth_total)
                if act > worst_action:
                    worst_action = act
            action = worst_action

            if action == 1:
                for plane_id, sus_count in fp_list:
                    weight = PLANE_WEIGHTS[plane_id] if plane_id < 6 else (256 if plane_id == 6 else 1)
                    E_fault_seg += sus_count * 255 * weight / scale
                    U_fault_r += 1
                seg_cost += C_ABSORB
                Q_count_r = seg_Q
                U_depth_r = seg_UD
                U_quant_r = seg_UQ
                S_core_low_r = seg_SL
                S_core_high_r = seg_SH
                swing_low_r = seg_swing_l
                swing_high_r = seg_swing_h
                E_quant_r = seg_Eq

            elif action in (2, 4):
                seg_cost += C_RECOMPUTE
                Q_count_r = int(np.sum(Qf_mask))
                U_quant_r = int(np.sum(Uf_mask))
                S_core_low_r = float(np.sum(x_full_low[start:end][Qf_mask]))
                S_core_high_r = float(np.sum(x_full_high[start:end][Qf_mask]))
                has_uf = bool(np.any(Uf_mask))
                swing_low_r = float(np.sum(np.minimum(0, x_full_low[start:end][Uf_mask]))) if has_uf else 0.0
                swing_high_r = float(np.sum(np.maximum(0, x_full_high[start:end][Uf_mask]))) if has_uf else 0.0
                E_quant_r = (Q_count_r + U_quant_r) * q_err
                for plane_id, _ in fp_list:
                    if plane_id <= 1:
                        weight = PLANE_WEIGHTS[plane_id]
                        max_ef = sus_count * 255 * weight / scale
                        if max_ef <= epsilon:
                            r4_override_forced += 1
                            override_driven = True
                        else:
                            r4_math_forced += 1

            elif action == 3:
                seg_cost += C_R3
                total_ef = 0.0
                for plane_id, sus_count in fp_list:
                    weight = PLANE_WEIGHTS[plane_id] if plane_id < 6 else (256 if plane_id == 6 else 1)
                    total_ef += sus_count * 255 * weight / scale
                if total_ef <= epsilon:
                    action = 1
                    Q_count_r = int(np.sum(Qf_mask))
                    U_quant_r = int(np.sum(Uf_mask))
                    U_fault_r = len(fp_list)
                    E_fault_seg = total_ef
                else:
                    action = 4
                    seg_cost += C_RECOMPUTE
                    Q_count_r = int(np.sum(Qf_mask))
                    U_quant_r = int(np.sum(Uf_mask))
                    S_core_low_r = float(np.sum(x_full_low[start:end][Qf_mask]))
                    S_core_high_r = float(np.sum(x_full_high[start:end][Qf_mask]))
                    has_uf = bool(np.any(Uf_mask))
                    swing_low_r = float(np.sum(np.minimum(0, x_full_low[start:end][Uf_mask]))) if has_uf else 0.0
                    swing_high_r = float(np.sum(np.maximum(0, x_full_high[start:end][Uf_mask]))) if has_uf else 0.0
                    E_quant_r = (Q_count_r + U_quant_r) * q_err

            elif action == 5:
                seg_cost += C_FALLBACK
                Q_count_r = int(np.sum(Qf_mask))
                U_quant_r = int(np.sum(Uf_mask))
                S_core_low_r = float(np.sum(x_full_low[start:end][Qf_mask]))
                S_core_high_r = float(np.sum(x_full_high[start:end][Qf_mask]))
                has_uf = bool(np.any(Uf_mask))
                swing_low_r = float(np.sum(np.minimum(0, x_full_low[start:end][Uf_mask]))) if has_uf else 0.0
                swing_high_r = float(np.sum(np.maximum(0, x_full_high[start:end][Uf_mask]))) if has_uf else 0.0
                E_quant_r = (Q_count_r + U_quant_r) * q_err

            elif action == 6:
                seg_cost += C_UNCERTIFIED

        # Apply faulted segment result
        actions_count[action] += 1
        total_cost += 1.0 + seg_cost
        Q_total += Q_count_r
        U_depth_total += U_depth_r
        U_quant_total += U_quant_r
        U_fault_total += U_fault_r
        S_core_low_total += S_core_low_r
        S_core_high_total += S_core_high_r
        swing_low_total += swing_low_r
        swing_high_total += swing_high_r
        E_quant_value_total += E_quant_r
        E_fault_value_total += E_fault_seg

    S_low = S_core_low_total + swing_low_total - E_quant_value_total
    S_high = S_core_high_total + swing_high_total + E_quant_value_total
    U_total = U_depth_total + U_quant_total + U_fault_total
    width_sum = S_high - S_low

    cert = "CERTIFIED_BOUNDED"
    if faults_detected > 0 and E_fault_value_total > 0:
        if width_sum <= epsilon and U_total <= epsilon:
            cert = "CERTIFIED_WIDENED"
        else:
            cert = "UNCERTIFIED"

    rs = QueryResult(
        Q_total=Q_total,
        U_depth=U_depth_total,
        U_quant=U_quant_total,
        U_fault=U_fault_total,
        S_low=S_low,
        S_high=S_high,
        E_quant_value=E_quant_value_total,
        E_fault_value=E_fault_value_total,
        width=width_sum,
        relative_width=width_sum / max(abs(S_high), 1e-300),
        certification=cert,
        segments_verified=n_segments,
        segments_failed=segments_failed,
        faults_detected=faults_detected,
        total_cost=total_cost,
        R0=actions_count[0],
        R1=actions_count[1],
        R2=actions_count[2],
        R3=actions_count[3],
        R4=actions_count[4],
        R5=actions_count[5],
        R6=actions_count[6],
        override_driven=override_driven,
        override_active=override_driven,
        r4_override_forced=r4_override_forced,
        r4_math_forced=r4_math_forced,
    )
    return rs


def execute_query(
    artifact: ArtifactData,
    plane_data: list[np.ndarray],
    ref_checksums: dict,
    fault_plans: list[FaultPlan],
    k: int,
    threshold: float,
    epsilon: float,
    baseline: str,
    segment_size: int = 1024,
) -> tuple[QueryResult, float]:
    """Execute a single query. For repeated queries on same (data, k), call
    classify_and_accum directly with pre-decoded arrays."""
    t0 = time.perf_counter()
    n_rows = artifact.n_rows
    scale = artifact.scale
    active_byte_len = artifact.active_byte_len
    q_err = 0.5 / scale
    n_segments = (n_rows + segment_size - 1) // segment_size

    x_low, x_high, x_full_low, x_full_high = prepare_decoded(
        artifact, plane_data, k, segment_size)

    faulted_segs: set[int] = set()
    seg_faults: dict[int, list[tuple[int, int]]] = {}
    for fp in fault_plans:
        faulted_segs.add(fp.segment_id)
        seg_faults.setdefault(fp.segment_id, []).append((fp.plane_id, fp.suspect_count))

    rs = classify_and_accum(
        x_low, x_high, x_full_low, x_full_high,
        threshold, q_err, n_segments, segment_size,
        faulted_segs, seg_faults, epsilon, scale, baseline, active_byte_len,
    )
    rs.n_rows = n_rows
    rs.segment_size = segment_size
    rs.k = k
    rs.threshold = threshold
    rs.epsilon = epsilon
    rs.baseline = baseline
    rs.dataset = artifact.dataset
    elapsed = time.perf_counter() - t0
    return rs, elapsed


# ---------------------------------------------------------------------------
# Full sweep runner
# ---------------------------------------------------------------------------


def run_sweep(
    artifact: ArtifactData,
    thresholds: dict[str, float],
    run_tag: str = "",
    segment_size: int = 1024,
    smoke: bool = False,
) -> tuple[list[QueryResult], dict[str, Any]]:
    """Run the full sweep for one dataset.

    Returns (results, stats).
    """
    n_segments = (artifact.n_rows + segment_size - 1) // segment_size

    results: list[QueryResult] = []
    query_count = 0
    total_elapsed = 0.0
    print(f"\nDataset: {artifact.dataset}")
    print(f"  n_rows={artifact.n_rows}, n_segments={n_segments}")
    print(f"  thresholds={thresholds}")
    print(f"  scale={artifact.scale}")

    n_rows = artifact.n_rows
    scale = artifact.scale
    active_byte_len = artifact.active_byte_len
    q_err = 0.5 / scale

    def _build_fault_lookup(fp_list: list[FaultPlan]) -> tuple[set[int], dict[int, list[tuple[int, int]]]]:
        faulted: set[int] = set()
        seg_fp: dict[int, list[tuple[int, int]]] = {}
        for fp in fp_list:
            faulted.add(fp.segment_id)
            seg_fp.setdefault(fp.segment_id, []).append((fp.plane_id, fp.suspect_count))
        return faulted, seg_fp

    t_start = time.perf_counter()

    if smoke:
        print(f"  SMOKE MODE: 3 scenarios (none,low_sig_p4,high_sig_p0), "
              f"k=[1,4], 1 threshold, ε=1e10")
        for k in [1, 4]:
            # Decode once at depth k
            x_low, x_high, x_full_low, x_full_high = prepare_decoded(
                artifact, artifact.plane_data, k, segment_size)

            for tlabel in [list(thresholds.keys())[0]]:
                thresh = thresholds[tlabel]
                for scenario in ["none", "low_sig_p4", "high_sig_p0"]:
                    if scenario == "none":
                        fp_list = []
                        faulted_segs, seg_fp = set(), {}
                    else:
                        fp_list = generate_fault_plan(scenario, 0.001, 0, n_segments, segment_size)
                        corrupted = apply_fault_plan(artifact.plane_data, fp_list)
                        x_low, x_high, x_full_low, x_full_high = prepare_decoded(
                            artifact, corrupted, k, segment_size)
                        faulted_segs, seg_fp = _build_fault_lookup(fp_list)

                    rs = classify_and_accum(
                        x_low, x_high, x_full_low, x_full_high,
                        thresh, q_err, n_segments, segment_size,
                        faulted_segs, seg_fp, 1e10, scale, "B4", active_byte_len,
                    )
                    rs.query_idx = query_count
                    rs.run_id = run_tag
                    rs.dataset = artifact.dataset
                    rs.n_rows = n_rows
                    rs.segment_size = segment_size
                    rs.k = k
                    rs.threshold = thresh
                    rs.epsilon = 1e10
                    rs.baseline = "B4"
                    rs.scenario = scenario
                    rs.fault_fraction = 0.001
                    rs.seed = 0
                    results.append(rs)
                    query_count += 1
                    elapsed = time.perf_counter() - t_start - sum(r.total_cost for r in results[:-1])
                    print(f"  Q{rs.query_idx}: k={k} T={thresh} ε=1e10 "
                          f"scenario={scenario} → {rs.certification}")
    else:
        # ---- Baselines B0, B1, B3 (no corruption dependency) ----
        for baseline in ["B0", "B1", "B3"]:
            for k in K_VALUES:
                # Decode clean data once per k for all baselines
                x_low, x_high, x_full_low, x_full_high = prepare_decoded(
                    artifact, artifact.plane_data, k, segment_size)

                for tlabel in thresholds:
                    thresh = thresholds[tlabel]
                    eps_list = [EPSILONS[0]] if baseline in ("B0", "B3") else EPSILONS

                    for eps in eps_list:
                        if baseline == "B1":
                            fp_list = [FaultPlan(s, p, segment_size, "xor_ff")
                                       for s in range(min(10, n_segments)) for p in range(6)]
                            faulted_segs, seg_fp = _build_fault_lookup(fp_list)
                        else:
                            faulted_segs, seg_fp = set(), {}

                        rs = classify_and_accum(
                            x_low, x_high, x_full_low, x_full_high,
                            thresh, q_err, n_segments, segment_size,
                            faulted_segs, seg_fp, eps, scale, baseline, active_byte_len,
                        )
                        rs.query_idx = query_count
                        rs.run_id = run_tag
                        rs.dataset = artifact.dataset
                        rs.n_rows = n_rows
                        rs.segment_size = segment_size
                        rs.k = k
                        rs.threshold = thresh
                        rs.epsilon = eps
                        rs.baseline = baseline
                        rs.scenario = "none"
                        results.append(rs)
                        query_count += 1
                        print(f"  [{baseline}] Q{rs.query_idx}: k={k} T={thresh} ε={eps} → {rs.certification}")

        # ---- B2 + B4 (corruption variants) ----
        scenarios = [s for s in CORRUPTION_SCENARIOS if s != "none"]
        total_variants = len(scenarios) * len(FAULT_FRACTIONS) * len(SEEDS)
        v = 0
        t_variant = time.perf_counter()
        for scenario in scenarios:
            for fraction in FAULT_FRACTIONS:
                for seed in SEEDS:
                    v += 1
                    fp_list = generate_fault_plan(scenario, fraction, seed, n_segments, segment_size)
                    if not fp_list:
                        continue
                    corrupted = apply_fault_plan(artifact.plane_data, fp_list)
                    faulted_segs, seg_fp = _build_fault_lookup(fp_list)

                    if v % 10 == 0:
                        elapsed_v = time.perf_counter() - t_variant
                        print(f"  B2/B4 variant {v}/{total_variants} [{elapsed_v:.0f}s]", flush=True)
                    elif v == 1:
                        print(f"  Starting B2/B4: {total_variants} variants, 48 queries each", flush=True)

                    for k in K_VALUES:
                        # Decode corrupted data once per k
                        x_low, x_high, x_full_low, x_full_high = prepare_decoded(
                            artifact, corrupted, k, segment_size)

                        for tlabel in thresholds:
                            thresh = thresholds[tlabel]
                            for eps in EPSILONS:
                                for baseline in ["B2", "B4"]:
                                    rs = classify_and_accum(
                                        x_low, x_high, x_full_low, x_full_high,
                                        thresh, q_err, n_segments, segment_size,
                                        faulted_segs, seg_fp, eps, scale, baseline, active_byte_len,
                                    )
                                    rs.query_idx = query_count
                                    rs.run_id = run_tag
                                    rs.dataset = artifact.dataset
                                    rs.n_rows = n_rows
                                    rs.segment_size = segment_size
                                    rs.k = k
                                    rs.threshold = thresh
                                    rs.epsilon = eps
                                    rs.baseline = baseline
                                    rs.scenario = scenario
                                    rs.fault_fraction = fraction
                                    rs.seed = seed
                                    results.append(rs)
                                    query_count += 1

    elapsed_total = time.perf_counter() - t_start
    stats = {
        "dataset": artifact.dataset,
        "n_rows": artifact.n_rows,
        "n_segments": n_segments,
        "total_queries": len(results),
        "elapsed": elapsed_total,
        "queries_per_second": len(results) / max(elapsed_total, 0.001),
    }
    print(f"  Completed {len(results)} queries in {elapsed_total:.1f}s ({stats['queries_per_second']:.1f} q/s)")
    return results, stats


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def write_matrix_csv(path: Path, all_results: list[QueryResult]) -> None:
    fieldnames = [
        "run_id", "query_idx", "dataset", "n_rows", "segment_size", "baseline",
        "k", "threshold", "epsilon", "operator", "scenario", "fault_fraction", "seed",
        "Q_total", "U_depth", "U_quant", "U_fault",
        "S_low", "S_high", "E_quant_value", "E_fault_value",
        "width", "relative_width", "certification",
        "segments_verified", "segments_failed", "faults_detected",
        "total_cost",
        "R0", "R1", "R2", "R3", "R4", "R5", "R6",
        "override_driven", "override_active",
        "r4_override_forced", "r4_math_forced",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_results:
            for op in ["COUNT", "SUM"]:
                d = {
                    "run_id": row.run_id,
                    "query_idx": row.query_idx,
                    "dataset": row.dataset,
                    "n_rows": row.n_rows,
                    "segment_size": row.segment_size,
                    "baseline": row.baseline,
                    "k": row.k,
                    "threshold": row.threshold,
                    "epsilon": row.epsilon,
                    "operator": op,
                    "scenario": row.scenario,
                    "fault_fraction": row.fault_fraction,
                    "seed": row.seed,
                    "Q_total": row.Q_total,
                    "U_depth": row.U_depth,
                    "U_quant": row.U_quant,
                    "U_fault": row.U_fault,
                    "S_low": f"{row.S_low:.6e}",
                    "S_high": f"{row.S_high:.6e}",
                    "E_quant_value": f"{row.E_quant_value:.6e}",
                    "E_fault_value": f"{row.E_fault_value:.6e}",
                    "width": f"{row.width:.6e}",
                    "relative_width": f"{row.relative_width:.6e}",
                    "certification": row.certification,
                    "segments_verified": row.segments_verified,
                    "segments_failed": row.segments_failed,
                    "faults_detected": row.faults_detected,
                    "total_cost": f"{row.total_cost:.4f}",
                    "R0": row.R0, "R1": row.R1, "R2": row.R2,
                    "R3": row.R3, "R4": row.R4, "R5": row.R5, "R6": row.R6,
                    "override_driven": int(row.override_driven),
                    "override_active": int(row.override_active),
                    "r4_override_forced": row.r4_override_forced,
                    "r4_math_forced": row.r4_math_forced,
                }
                w.writerow(d)


def write_run_meta(path: Path, all_results: list[QueryResult], stats: dict[str, Any], run_tag: str) -> None:
    total = len(all_results)
    bounded = sum(1 for r in all_results if r.certification == "CERTIFIED_BOUNDED")
    widened = sum(1 for r in all_results if r.certification == "CERTIFIED_WIDENED")
    uncert = sum(1 for r in all_results if r.certification == "UNCERTIFIED")
    r4_total = sum(r.R4 for r in all_results)
    r5_total = sum(r.R5 for r in all_results)
    r1_total = sum(r.R1 for r in all_results)
    faults_detected = sum(r.faults_detected for r in all_results)
    override_r4 = sum(r.r4_override_forced for r in all_results)
    math_r4 = sum(r.r4_math_forced for r in all_results)

    cons_margin = 0.16 / (1 - 0.16)
    opt_margin = 0.83 / (1 - 0.83)
    avg_cost = sum(r.total_cost for r in all_results) / max(total, 1)

    lines = [
        f"Run ID: {run_tag}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"Host: {os.uname().nodename}",
        f"",
        f"=== Configuration ===",
        f"SDG: P3-D Canonical Reactive Matrix",
        f"Datasets: sensor, uniform, heavy_tailed, zipfian",
        f"CESM-ATM Q: NOT AVAILABLE (data not found on system)",
        f"N rows per dataset: 100,000,000",
        f"Segment size: 1024",
        f"k values: 1, 2, 4, 6",
        f"ε values: 1e6, 1e8, 1e10, 1e12",
        f"Corruption scenarios: none, low_sig_p4, high_sig_p0, multi_plane_p2_p3, zero_invariant_p6",
        f"Fault fractions: 0.001%, 0.1%, 1%, 10%",
        f"Seeds: 0, 1, 2, 3, 4",
        f"Baselines: B0, B1, B2, B3, B4",
        f"Total queries (expected with 4 datasets): 77,376",
        f"Total queries (actual): {total}",
        f"",
        f"=== Certification Summary ===",
        f"CERTIFIED_BOUNDED: {bounded} ({bounded/max(total,1)*100:.1f}%)",
        f"CERTIFIED_WIDENED: {widened} ({widened/max(total,1)*100:.1f}%)",
        f"UNCERTIFIED: {uncert} ({uncert/max(total,1)*100:.1f}%)",
        f"Combined certification rate: {(bounded+widened)/max(total,1)*100:.1f}%",
        f"",
        f"=== Fallback Stats ===",
        f"Total faults detected: {faults_detected}",
        f"R1 (absorb) actions: {r1_total}",
        f"R4 (recompute) actions: {r4_total}",
        f"R5 (full fallback) actions: {r5_total}",
        f"Override-driven R4: {override_r4}",
        f"Math-driven R4: {math_r4}",
        f"",
        f"=== Cost Model (normalized, T_progressive=1.0 per segment) ===",
        f"Average cost per query: {avg_cost:.2f}",
        f"Speed margin conservative: {cons_margin:.2f}",
        f"Speed margin optimistic: {opt_margin:.2f}",
        f"",
        f"=== Verification Overhead ===",
        f"C_verify per segment: {C_VERIFY}",
        f"Verification as % of conservative margin: {C_VERIFY/cons_margin*100:.1f}%",
        f"Verification as % of optimistic margin: {C_VERIFY/opt_margin*100:.1f}%",
        f"",
        f"=== Execution ===",
        f"Run tag: {run_tag}",
        f"Queries per dataset: {stats.get('queries_per_dataset', {})}",
        f"Wall time: {stats.get('total_elapsed', 0):.1f}s",
        f"",
        f"=== Artifact Checksums ===",
    ]

    # Add artifact SHA256
    for ds in DATASETS:
        art_dir = find_artifact_dir(ds)
        for fname in ["artifact.json"]:
            fpath = art_dir / fname
            if fpath.exists():
                sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
                lines.append(f"  {ds}/{fname}: sha256={sha}")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="P3-D Canonical Reactive Matrix")
    parser.add_argument("--run-tag", default="", help="Run identifier tag")
    parser.add_argument("--segment-size", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true", help="Run 12-query smoke test only")
    parser.add_argument("--dataset", default=None, help="Single dataset (for smoke/sharding)")
    args = parser.parse_args()

    run_tag = args.run_tag or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    datasets = [args.dataset] if args.dataset else DATASETS

    all_results: list[QueryResult] = []
    cumulative_stats: dict[str, Any] = {
        "queries_per_dataset": {},
        "total_elapsed": 0.0,
    }

    t0 = time.perf_counter()

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Loading artifact: {ds}")
        artifact = load_artifact(ds)
        print(f"  scale={artifact.scale}, base_fixed={artifact.base_fixed}")

        thresholds = compute_thresholds(artifact)
        print(f"  thresholds: {thresholds}")

        ds_results, ds_stats = run_sweep(
            artifact, thresholds,
            run_tag=run_tag,
            segment_size=args.segment_size,
            smoke=args.smoke,
        )

        cumulative_stats["queries_per_dataset"][ds] = len(ds_results)
        cumulative_stats["total_elapsed"] += ds_stats["elapsed"]
        all_results.extend(ds_results)

        if args.smoke:
            break

    elapsed = time.perf_counter() - t0
    cumulative_stats["total_elapsed"] = elapsed

    if not all_results:
        print("No results generated!")
        sys.exit(1)

    out_dir = RESULTS_ROOT / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Writing outputs to {out_dir}")
    print(f"Total results: {len(all_results)}")

    write_matrix_csv(out_dir / "reactive_query_matrix.csv", all_results)
    write_run_meta(out_dir / "run_meta.txt", all_results, cumulative_stats, run_tag)

    print(f"Results written to {out_dir}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
