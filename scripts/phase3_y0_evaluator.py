#!/usr/bin/env python3
"""Phase 3-Y0 Evaluator: delivered-answer metrics for bounded-degradation repair.

Computes delivered-answer Sum and certified bound from:
- Clean artifact planes (8-plane MSB-first)
- Replica fault plans (XOR model)
- r_vector allocation

Y0 adds:
- err_fault_user_observed (delivered answer error vs fault-free same-k answer)
- bound_width, bound_width_inflation, certified_quality_pass
- repair_failure_degrade_rate, unprotected_s0_uncertainty_rate
- slo_valid, slo_pass_rate (Y0: always True/1.0)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _ensure_scripts_path() -> None:
    _dir = str(Path(__file__).resolve().parent)
    if _dir not in sys.path:
        sys.path.insert(0, _dir)


_ensure_scripts_path()
from phase2_vote import vote_byte
from phase2_oracle import apply_fault_plan, discover_fault_plan_paths


SEGMENT_SIZE = 1024
PLANE_WEIGHTS = [1 << (8 * (7 - p)) for p in range(8)]


@dataclass
class DeliveredAnswer:
    dataset: str
    n_rows: int
    scale: int
    k: int
    policy: str
    allocation_r: str
    seed: int
    fault_rate: str

    err_fault_user_observed: float
    bound_width: float
    bound_width_inflation: float
    certified_quality_pass: bool
    repair_failure_degrade_rate: float
    unprotected_s0_uncertainty_rate: float
    slo_valid: bool
    slo_pass_rate: float

    segments_total: int
    segments_crc_hit: int
    segments_repaired: int
    segments_degraded: int
    segments_unprotected: int
    clean_answer: float
    delivered_answer: float
    bound_width_fault_free: float
    certified: bool
    fallback_lane: str


def load_clean_planes(artifact_dir: Path, n_rows: int) -> list[bytes]:
    """Load 8 clean plane byte arrays from artifact directory."""
    planes: list[bytes] = []
    for p in range(8):
        path = artifact_dir / f"plane_{p}.bin"
        if not path.is_file():
            raise FileNotFoundError(f"clean plane file not found: {path}")
        data = path.read_bytes()
        if len(data) != n_rows:
            raise ValueError(
                f"plane {p}: expected {n_rows} bytes, got {len(data)}"
            )
        planes.append(data)
    return planes


def load_artifact_metadata(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "artifact.json"
    if not path.is_file():
        raise FileNotFoundError(f"artifact metadata not found: {path}")
    return json.loads(path.read_text())


def compute_clean_sum(planes: list[bytes]) -> int:
    """Compute clean encoded sum from 8 plane byte arrays."""
    total = 0
    for p in range(8):
        total += sum(planes[p]) * PLANE_WEIGHTS[p]
    return total


def compute_segment_outcomes(
    clean_planes: list[bytes],
    fault_plan_paths: dict[int, list[str]],
    r_vector: list[int],
    segment_size: int = SEGMENT_SIZE,
) -> dict[tuple[int, int], str]:
    """Determine per-segment per-plane outcome status.

    CRC detection is per-replica: if ANY replica's segment data differs from
    clean, CRC fires.  Then majority voting decides:
      - "repaired": CRC hit AND voted == clean (voting recovered clean)
      - "degraded":  CRC hit AND voted != clean (voting gave wrong value)
      - "unprotected": r_p < 2 (no repair possible)
      - "clean":      no CRC hit (no replica has fault in this segment)

    Returns dict[(seg_idx, plane)] -> status string.
    """
    n_rows = len(clean_planes[0])
    n_segments = (n_rows + segment_size - 1) // segment_size
    outcome: dict[tuple[int, int], str] = {}

    for seg_idx in range(n_segments):
        start = seg_idx * segment_size
        end = min(start + segment_size, n_rows)

        for p in range(8):
            r_p = r_vector[p]
            plane_paths = fault_plan_paths.get(p, [])

            if r_p < 2:
                if plane_paths:
                    outcome[(seg_idx, p)] = "unprotected"
                else:
                    outcome[(seg_idx, p)] = "clean"
                continue

            if len(plane_paths) != r_p:
                outcome[(seg_idx, p)] = "unprotected"
                continue

            replica_bytes = [
                apply_fault_plan(clean_planes[p], path)
                for path in plane_paths
            ]

            crc_hit = False
            voted_differs = False

            for i in range(start, end):
                clean_b = clean_planes[p][i]
                for rb in replica_bytes:
                    if rb[i] != clean_b:
                        crc_hit = True
                        break
                if crc_hit:
                    break

            if not crc_hit:
                outcome[(seg_idx, p)] = "clean"
                continue

            for i in range(start, end):
                values = [rb[i] for rb in replica_bytes]
                vb = vote_byte(values)
                if vb != clean_planes[p][i]:
                    voted_differs = True
                    break

            if voted_differs:
                outcome[(seg_idx, p)] = "degraded"
            else:
                outcome[(seg_idx, p)] = "repaired"

    return outcome


def compute_voted_planes(
    clean_planes: list[bytes],
    fault_plan_paths: dict[int, list[str]],
    r_vector: list[int],
) -> list[bytes]:
    """Compute majority-voted bytes for each plane.

    For r_p >= 2: apply fault plans and majority-vote per byte.
    For r_p < 2: use the faulted single-replica bytes (if any), or clean.
    """
    n_rows = len(clean_planes[0])
    voted_planes: list[bytes] = []

    for p in range(8):
        r_p = r_vector[p]
        plane_paths = fault_plan_paths.get(p, [])
        if r_p >= 2 and len(plane_paths) >= 2:
            replica_bytes = [
                apply_fault_plan(clean_planes[p], path)
                for path in plane_paths[:r_p]
            ]
            voted = bytearray(n_rows)
            for i in range(n_rows):
                values = [rb[i] for rb in replica_bytes]
                voted[i] = vote_byte(values)
            voted_planes.append(bytes(voted))
        elif r_p >= 1 and len(plane_paths) >= 1:
            faulted = apply_fault_plan(clean_planes[p], plane_paths[0])
            voted_planes.append(faulted)
        else:
            voted_planes.append(clean_planes[p])

    return voted_planes


def decode_value(
    plane_bytes: list[int], k: int, scale: int
) -> tuple[float, float]:
    """Decode a single row's 8-plane representation to float range [x_low, x_high].

    Uses partial decode at depth k + max_undecoded uncertainty.
    """
    partial = sum(
        plane_bytes[p] * PLANE_WEIGHTS[p] for p in range(min(k, 8))
    )
    max_undecoded = 0
    if k < 8:
        max_undecoded = (1 << (8 * (8 - k))) - 1
    q_err = 0.5 / scale
    fp64_scale = float(scale)
    x_low = (partial - q_err * fp64_scale) / fp64_scale
    x_high = (partial + max_undecoded + q_err * fp64_scale) / fp64_scale
    return x_low, x_high


def compute_delivered_answer_with_degradation(
    clean_planes: list[bytes],
    fault_plan_paths: dict[int, list[str]],
    r_vector: list[int],
    scale: int,
    n_rows: int,
    dataset: str = "tiny_fixture",
    k: int = 8,
    policy: str = "graded",
    allocation_r: str = "",
    seed: int = 0,
    fault_rate: str = "1e-06",
    epsilon_bound: float = 1e12,
    segment_size: int = SEGMENT_SIZE,
    fallback_lane: str = "bounded_degradation",
) -> DeliveredAnswer:
    """Main pipeline: compute all delivered-answer metrics."""
    clean_sum = compute_clean_sum(clean_planes)
    clean_answer = clean_sum / scale

    q_err = 0.5 / scale

    bound_width_fault_free = 2.0 * q_err * n_rows

    n_segments = (n_rows + segment_size - 1) // segment_size

    voted_planes = compute_voted_planes(
        clean_planes, fault_plan_paths, r_vector
    )

    outcomes = compute_segment_outcomes(
        clean_planes, fault_plan_paths, r_vector, segment_size
    )

    delivered_sum = 0
    bound_widen = 0.0
    segments_total = n_segments
    segments_crc_hit = 0
    segments_repaired = 0
    segments_degraded = 0
    segments_unprotected = 0

    for i in range(n_rows):
        seg_idx = i // segment_size
        delivered_bytes = [0] * 8
        for p in range(8):
            st = outcomes.get((seg_idx, p), "clean")
            if st == "clean" or st == "repaired":
                delivered_bytes[p] = clean_planes[p][i]
            elif st == "degraded":
                delivered_bytes[p] = voted_planes[p][i]
            elif st == "unprotected":
                delivered_bytes[p] = voted_planes[p][i]

        delivered_uint64 = sum(
            delivered_bytes[p] * PLANE_WEIGHTS[p] for p in range(8)
        )
        delivered_sum += delivered_uint64

    for p in range(8):
        for seg_idx in range(n_segments):
            st = outcomes.get((seg_idx, p), "clean")
            if st == "clean" or st == "repaired":
                continue
            start = seg_idx * segment_size
            end = min(start + segment_size, n_rows)
            if st == "degraded":
                for i in range(start, end):
                    w = PLANE_WEIGHTS[p]
                    diff = abs(voted_planes[p][i] - clean_planes[p][i])
                    bound_widen += diff * w / scale
            elif st == "unprotected":
                for i in range(start, end):
                    w = PLANE_WEIGHTS[p]
                    bound_widen += 255 * w / scale

    for p in range(8):
        for seg_idx in range(n_segments):
            st = outcomes.get((seg_idx, p), "clean")
            if st == "clean":
                continue
            if st == "repaired":
                segments_repaired += 1
            elif st == "degraded":
                segments_degraded += 1
            elif st == "unprotected":
                segments_unprotected += 1
            segments_crc_hit += 1

    delivered_answer = delivered_sum / scale

    raw_err = delivered_answer - clean_answer
    if clean_answer != 0.0:
        err_fault_user_observed = abs(raw_err) / abs(clean_answer)
    else:
        err_fault_user_observed = abs(raw_err)

    bound_width = bound_width_fault_free + bound_widen
    bound_width_inflation = (
        bound_width / bound_width_fault_free if bound_width_fault_free > 0 else 1.0
    )

    certified = True
    certified_quality_pass = certified and bound_width <= epsilon_bound

    repair_failure_degrade_rate = (
        segments_degraded / segments_crc_hit if segments_crc_hit > 0 else 0.0
    )

    s0_unprotected = 0
    s0_total = 0
    for seg_idx in range(n_segments):
        st = outcomes.get((seg_idx, 0), "clean")
        if st == "unprotected" or st == "degraded":
            s0_total += 1
            if st == "unprotected":
                s0_unprotected += 1

    unprotected_s0_uncertainty_rate = (
        s0_unprotected / s0_total if s0_total > 0 else 0.0
    )

    slo_valid = True
    slo_pass_rate = 1.0

    return DeliveredAnswer(
        dataset=dataset,
        n_rows=n_rows,
        scale=scale,
        k=k,
        policy=policy,
        allocation_r=allocation_r,
        seed=seed,
        fault_rate=fault_rate,
        err_fault_user_observed=err_fault_user_observed,
        bound_width=bound_width,
        bound_width_inflation=bound_width_inflation,
        certified_quality_pass=certified_quality_pass,
        repair_failure_degrade_rate=repair_failure_degrade_rate,
        unprotected_s0_uncertainty_rate=unprotected_s0_uncertainty_rate,
        slo_valid=slo_valid,
        slo_pass_rate=slo_pass_rate,
        segments_total=segments_total,
        segments_crc_hit=segments_crc_hit,
        segments_repaired=segments_repaired,
        segments_degraded=segments_degraded,
        segments_unprotected=segments_unprotected,
        clean_answer=clean_answer,
        delivered_answer=delivered_answer,
        bound_width_fault_free=bound_width_fault_free,
        certified=certified,
        fallback_lane=fallback_lane,
    )


def compute_ui_prediction(
    clean_planes: list[bytes],
    fault_plan_paths: dict[int, list[str]],
    r_vector: list[int],
    scale: int,
    n_rows: int,
    segment_size: int = SEGMENT_SIZE,
    k: int = 8,
    functional: str = "sum",
) -> dict:
    """Compute predicted bound width from U_i per-plane contribution ceilings.

    U_i (SUM, unfiltered) = active_count * 255 * weight[i] / scale
    For each degraded/unprotected segment of plane i:
      contribution = segment_byte_count * 255 * weight[i] / scale

    Returns dict with fields:
      bound_width_fault_free  fault-free quantization bound
      bound_width_predicted   fault-free + predicted bound widen from U_i
      ui_per_plane            per-plane total U_i contributions
    """
    outcomes = compute_segment_outcomes(
        clean_planes, fault_plan_paths, r_vector, segment_size
    )
    n_segments = (n_rows + segment_size - 1) // segment_size

    bound_width_fault_free = n_rows / scale

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
                contrib = count * 255 * PLANE_WEIGHTS[p] / scale
                predicted_widen += contrib
                plane_total += contrib
        ui_per_plane[str(p)] = plane_total

    return {
        "bound_width_fault_free": bound_width_fault_free,
        "bound_width_predicted": bound_width_fault_free + predicted_widen,
        "ui_per_plane": ui_per_plane,
    }


def bound_width_prediction_err(
    observed: float, predicted: float, eps: float = 1e-30
) -> float:
    """Relative prediction error: |observed - predicted| / max(predicted, eps)."""
    return abs(observed - predicted) / max(predicted, eps)


def format_csv_row(answer: DeliveredAnswer) -> dict[str, str]:
    """Format DeliveredAnswer as CSV dict per paired-seed schema."""
    return {
        "dataset": answer.dataset,
        "n_rows": str(answer.n_rows),
        "scale": str(answer.scale),
        "k": str(answer.k),
        "policy": answer.policy,
        "allocation_r": answer.allocation_r,
        "seed": str(answer.seed),
        "fault_rate": answer.fault_rate,
        "err_fault_user_observed": str(answer.err_fault_user_observed),
        "bound_width": str(answer.bound_width),
        "bound_width_inflation": str(answer.bound_width_inflation),
        "certified_quality_pass": str(answer.certified_quality_pass).lower(),
        "repair_failure_degrade_rate": str(answer.repair_failure_degrade_rate),
        "unprotected_s0_uncertainty_rate": str(answer.unprotected_s0_uncertainty_rate),
        "slo_valid": str(answer.slo_valid).lower(),
        "slo_pass_rate": str(answer.slo_pass_rate),
        "segments_total": str(answer.segments_total),
        "segments_crc_hit": str(answer.segments_crc_hit),
        "segments_repaired": str(answer.segments_repaired),
        "segments_degraded": str(answer.segments_degraded),
        "segments_unprotected": str(answer.segments_unprotected),
        "clean_answer": str(answer.clean_answer),
        "delivered_answer": str(answer.delivered_answer),
        "bound_width_fault_free": str(answer.bound_width_fault_free),
        "certified": str(answer.certified).lower(),
        "fallback_lane": answer.fallback_lane,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="Directory containing plane_*.bin and artifact.json")
    parser.add_argument("--fault-plan-dir", type=Path, required=True,
                        help="Base directory with fault plan subdirectories")
    parser.add_argument("--r-vector", type=int, nargs=8, required=True,
                        help="Replication vector for planes 0-7")
    parser.add_argument("--dataset", type=str, default="tiny_fixture")
    parser.add_argument("--n-rows", type=int, required=True)
    parser.add_argument("--scale", type=int, default=100)
    parser.add_argument("--fault-rate", type=str, default="1e-06")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--policy", type=str, default="graded",
                        choices=["graded", "uniform_repair_fraction"])
    parser.add_argument("--budget-B", type=int, default=16)
    parser.add_argument("--epsilon-bound", type=float, default=1e12)
    parser.add_argument("--segment-size", type=int, default=SEGMENT_SIZE)
    parser.add_argument("--fallback-lane", type=str, default="bounded_degradation",
                        choices=["bounded_degradation", "raw_fallback_context"])
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--sensitivity-profile", type=Path, default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--strategy-id", type=str, default="per_dataset")
    parser.add_argument("--strategy-scale", type=str, default="")
    args = parser.parse_args()

    clean_planes = load_clean_planes(args.artifact_dir, args.n_rows)

    if args.sensitivity_profile:
        sp = json.loads(args.sensitivity_profile.read_text())
        ds_p = sp["datasets"].get(args.dataset, {})
        th = ds_p.get(args.fault_rate, {})
        vacuous_planes = th.get("vacuous_planes", [])
    else:
        vacuous_planes = [p for p, r in enumerate(args.r_vector) if r <= 1]

    fault_plan_paths = discover_fault_plan_paths(
        args.fault_plan_dir,
        args.dataset,
        args.n_rows,
        args.scale,
        args.policy,
        args.fault_rate,
        args.base_seed,
        args.r_vector,
        vacuous_planes,
    )

    allocation_r_str = "|".join(str(r) for r in args.r_vector)

    result = compute_delivered_answer_with_degradation(
        clean_planes=clean_planes,
        fault_plan_paths=fault_plan_paths,
        r_vector=args.r_vector,
        scale=args.scale,
        n_rows=args.n_rows,
        dataset=args.dataset,
        k=args.k,
        policy=args.policy,
        allocation_r=allocation_r_str,
        seed=args.base_seed,
        fault_rate=args.fault_rate,
        epsilon_bound=args.epsilon_bound,
        segment_size=args.segment_size,
        fallback_lane=args.fallback_lane,
    )

    csv_row = format_csv_row(result)
    csv_row["budget_B"] = str(args.budget_B)
    csv_row["strategy_id"] = args.strategy_id
    csv_row["strategy_scale"] = args.strategy_scale

    fieldnames = [
        "dataset", "n_rows", "scale", "k", "policy", "allocation_r",
        "seed", "fault_rate",
        "err_fault_user_observed", "bound_width", "bound_width_inflation",
        "certified_quality_pass", "repair_failure_degrade_rate",
        "unprotected_s0_uncertainty_rate",
        "slo_valid", "slo_pass_rate",
        "segments_total", "segments_crc_hit", "segments_repaired",
        "segments_degraded", "segments_unprotected",
        "clean_answer", "delivered_answer", "bound_width_fault_free",
        "certified", "fallback_lane",
        "budget_B", "strategy_id", "strategy_scale",
    ]

    print(
        f"clean_answer={result.clean_answer:.10f} "
        f"delivered_answer={result.delivered_answer:.10f} "
        f"err={result.err_fault_user_observed:.10e} "
        f"bound_width={result.bound_width:.10e} "
        f"bound_inflation={result.bound_width_inflation:.6f} "
        f"repair_fail_rate={result.repair_failure_degrade_rate:.6f} "
        f"unprotected_rate={result.unprotected_s0_uncertainty_rate:.6f} "
        f"certified={result.certified_quality_pass}"
    )

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(csv_row)
        print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
