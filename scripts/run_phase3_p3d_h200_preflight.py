#!/usr/bin/env python3
#!/usr/bin/env python3
"""Phase 3 P3-D H200 Preflight (CPU certification + metadata audit).

Runs the H200 preflight subset defined in PRD §6.6:
  3 datasets × 2 ops × 3 k × 2 ε × 4 scenarios × 2 fractions × 2 seeds
  = 576 queries (384 without CESM-ATM Q)

Output:
  ${WORK_DIR}/results/reliability_layer1/phase3/p3d_h200_preflight/<run_id>/
"""

from __future__ import annotations

import argparse
import csv
import functools
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

RESULTS_ROOT = Path("${WORK_DIR}/results/reliability_layer1/phase3/p3d_h200_preflight")
ARTIFACT_ROOT = Path("${WORK_DIR}/datasets/reliability_layer1/artifacts")
RAW_ROOT = Path("${WORK_DIR}/datasets/reliability_layer1/raw")

N_TOTAL_PLANES = 8
PLANE_WEIGHTS_MSB = [1 << (56 - 8 * p) for p in range(N_TOTAL_PLANES)]

SCALE = 100
SEGMENT_ROWS = 1024

C_VERIFY = 0.15
C_ABSORB = 0.00
C_RE_READ = 1.0
C_RECOMPUTE = 6.0
C_FALLBACK = 8.0
C_UNCERTIFIED = 0.0

# Preflight subset per PRD §6.6
DATASETS = ["sensor", "uniform"]
DATASET_META = {
    "sensor": {"active_byte_len": 2, "highest_active": 6, "zero_plane": 0, "multi_plane": [6, 7]},
    "uniform": {"active_byte_len": 3, "highest_active": 5, "zero_plane": 0, "multi_plane": [5, 6]},
}
K_VALUES = [1, 4, 6]
EPSILONS = [1e6, 1e10]
SCENARIOS = ["none", "high_sig", "low_sig", "inactive_probe"]
FAULT_FRACTIONS = [0.00001, 0.01]
SEEDS = [0, 1]


@dataclass
class FaultPlan:
    segment_id: int
    plane_id: int
    suspect_count: int
    injection_type: str


@dataclass
class QueryResult:
    dataset: str = ""
    operator: str = ""
    threshold: float = 0.0
    k: int = 0
    epsilon: float = 0.0
    scenario: str = ""
    fault_fraction: float = 0.0
    seed: int = 0

    answer_float: float = 0.0
    answer_low: float = 0.0
    answer_high: float = 0.0
    certification_status: str = "UNCERTIFIED"
    u_depth: float = 0.0
    u_quant: float = 0.0
    u_fault: float = 0.0
    e_quant_magnitude: float = 0.0
    e_fault_magnitude: float = 0.0

    n_segments: int = 0
    active_byte_len: int = 0
    effective_active_planes_read: int = 0
    zero_plane_reads: int = 0
    wasted_zero_plane_fraction: float = 0.0

    fallback_count: int = 0
    uncertified_count: int = 0

    segment_count_B4: int = 0
    r0_count: int = 0
    r1_count: int = 0
    r4_count: int = 0
    r6_count: int = 0


def load_artifact_meta(dataset: str) -> dict:
    path = ARTIFACT_ROOT / dataset / "n100000000" / "scale100" / "artifact.json"
    return json.loads(path.read_text())


_PLANE_CACHE: dict[tuple[str, int], np.ndarray] = {}
def load_plane(dataset: str, plane: int) -> np.ndarray:
    key = (dataset, plane)
    if key not in _PLANE_CACHE:
        path = ARTIFACT_ROOT / dataset / "n100000000" / "scale100" / f"plane_{plane}.bin"
        _PLANE_CACHE[key] = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return _PLANE_CACHE[key]


_CRC_CACHE: dict[tuple[str, int], list[int]] = {}
def compute_segment_checksums(dataset: str, plane: int, plane_data: np.ndarray, seg_rows: int):
    key = (dataset, plane)
    if key not in _CRC_CACHE:
        n_seg = len(plane_data) // seg_rows
        checksums = []
        for s in range(n_seg):
            seg = plane_data[s * seg_rows:(s + 1) * seg_rows]
            checksums.append(zlib.crc32(seg.tobytes()) & 0xFFFFFFFF)
        _CRC_CACHE[key] = checksums
    return _CRC_CACHE[key]


def decode_row(row_planes: list[int], k: int, active_byte_len: int) -> int:
    value = 0
    start_plane = N_TOTAL_PLANES - active_byte_len
    for i in range(k):
        p = start_plane + active_byte_len - 1 - i
        if 0 <= p < len(row_planes):
            value = (value << 8) | row_planes[p]
    return value


def decode_all(planes: list[np.ndarray], k: int, active_byte_len: int) -> np.ndarray:
    n = len(planes[0])
    result = np.zeros(n, dtype=np.int64)
    start_plane = N_TOTAL_PLANES - active_byte_len
    for i in range(k):
        p = start_plane + active_byte_len - 1 - i
        if 0 <= p < len(planes):
            shift = 8 * i
            result += planes[p].astype(np.int64) << shift
    return result


def compute_thresholds(meta: dict) -> dict:
    # Use metadata percentiles or compute from raw
    return {"low": 30, "medium": 50, "high": 70}


def build_fault_plan(
    dataset: str,
    scenario: str,
    fault_fraction: float,
    seed: int,
    n_segments: int,
    meta: dict,
) -> list[FaultPlan]:
    rng = random.Random(seed)
    plans = []
    if scenario == "none":
        return plans
    active = DATASET_META[dataset]
    if scenario == "high_sig":
        plane = active["highest_active"]
    elif scenario == "low_sig":
        plane = N_TOTAL_PLANES - 1  # P7
    elif scenario == "inactive_probe":
        plane = active["zero_plane"]
    else:
        return plans
    count = max(1, int(n_segments * fault_fraction))
    for i in range(count):
        sid = rng.randint(0, n_segments - 1)
        plans.append(FaultPlan(
            segment_id=sid,
            plane_id=plane,
            suspect_count=SEGMENT_ROWS,
            injection_type="xor_ff" if scenario != "inactive_probe" else "set_nonzero",
        ))
    return plans


def run_query(
    dataset: str,
    operator: str,
    threshold: float,
    k: int,
    epsilon: float,
    scenario: str,
    fault_fraction: float,
    seed: int,
    meta: dict,
) -> QueryResult:
    active_byte_len = meta["active_byte_len"]
    start_plane = N_TOTAL_PLANES - active_byte_len
    # Load from cache — plane files are read once per dataset
    planes_clean = [load_plane(dataset, p) for p in range(N_TOTAL_PLANES)]
    n = len(planes_clean[0])
    n_segments = n // SEGMENT_ROWS
    effective_active = min(k, active_byte_len)
    zero_reads = max(0, k - active_byte_len)
    wasted_frac = zero_reads / k if k > 0 else 0.0

    # Decode at depth k (clean)
    decoded = decode_all(planes_clean, k, active_byte_len)
    decoded_f64 = decoded.astype(np.float64) / SCALE

    fault_plans = build_fault_plan(dataset, scenario, fault_fraction, seed, n_segments, meta)

    # Planes to verify (significance-ordered: MSB-first within active range)
    verify_plane_order = list(range(start_plane, N_TOTAL_PLANES))

    # Per-segment checksums (unperturbed — cached)
    clean_seg_crc: dict[int, dict[int, int]] = {}
    for p in verify_plane_order:
        seg_crcs = compute_segment_checksums(dataset, p, planes_clean[p], SEGMENT_ROWS)
        for s, crc in enumerate(seg_crcs):
            clean_seg_crc.setdefault(s, {})[p] = crc

    if not fault_plans:
        # No fault → clean = fault
        fault_decoded_f64 = decoded_f64.copy()
        faulted_planes = set()
        fault_seg_crc = dict(clean_seg_crc)
    else:
        # Copy planes for fault injection (writable)
        planes = [p.copy() for p in planes_clean]
        faulted_planes: set[int] = set()
        for fp in fault_plans:
            seg = planes[fp.plane_id][fp.segment_id * SEGMENT_ROWS:(fp.segment_id + 1) * SEGMENT_ROWS]
            if fp.injection_type == "xor_ff":
                planes[fp.plane_id][fp.segment_id * SEGMENT_ROWS:(fp.segment_id + 1) * SEGMENT_ROWS] = seg ^ 0xFF
            else:
                planes[fp.plane_id][fp.segment_id * SEGMENT_ROWS] ^= 0x01
            faulted_planes.add(fp.plane_id)

        # Re-decode after fault
        fault_decoded = decode_all(planes, k, active_byte_len)
        fault_decoded_f64 = fault_decoded.astype(np.float64) / SCALE
        # Compute checksums only on faulted planes (others match clean)
        fault_seg_crc = dict(clean_seg_crc)
        for p in faulted_planes:
            seg_crcs = compute_segment_checksums(dataset, p, planes[p], SEGMENT_ROWS)
            for s, crc in enumerate(seg_crcs):
                fault_seg_crc.setdefault(s, {})[p] = crc

    # Reactive policy
    r1_count = 0
    r4_count = 0
    r6_count = 0
    fallback_count = 0
    total_e_fault_raw = 0.0
    total_e_quant = 0.0

    for s in range(n_segments):
        for p in verify_plane_order:
            clean = clean_seg_crc[s][p]
            fault = fault_seg_crc[s][p]
            if clean != fault:
                e_fault_raw = SEGMENT_ROWS * 255 * PLANE_WEIGHTS_MSB[p] / SCALE
                total_e_fault_raw += e_fault_raw

                # Policy: override for highest active plane
                highest_active = DATASET_META[dataset]["highest_active"]
                if p == highest_active and p in faulted_planes:
                    r4_count += 1
                    fallback_count += 1
                elif e_fault_raw <= epsilon:
                    r1_count += 1
                else:
                    r4_count += 1
                    fallback_count += 1

    # Quantization error
    q_err = 0.5 / SCALE
    total_e_quant = n * q_err

    # Answer computation
    if operator == "COUNT":
        mask = fault_decoded_f64 > threshold
        count = np.sum(mask)
        answer_float = float(count)
        # Uncertainty due to quantization straddle
        straddle_mask = (decoded_f64 - q_err <= threshold) & (decoded_f64 + q_err >= threshold)
        u_quant = float(np.sum(straddle_mask))
        # Uncertainty due to depth
        if k < active_byte_len:
            max_additional = (1 << (8 * (active_byte_len - k))) - 1
            depth_straddle = (decoded_f64 + max_additional / SCALE >= threshold) & (decoded_f64 <= threshold)
            u_depth = float(np.sum(depth_straddle))
        else:
            u_depth = 0.0
        u_fault = 0.0
        for fp in fault_plans:
            w = PLANE_WEIGHTS_MSB[fp.plane_id]
            u_fault += fp.suspect_count * 255 * w / SCALE
        u_total = u_quant + u_depth + u_fault
        answer_low = max(0, count - u_total)
        answer_high = count + u_total
    else:
        mask = fault_decoded_f64 > threshold
        answer_float = float(np.sum(fault_decoded_f64[mask]))
        answer_low = answer_float - total_e_quant - total_e_fault_raw
        answer_high = answer_float + total_e_quant + total_e_fault_raw

    # Certification status
    if fallback_count == 0 and total_e_fault_raw == 0:
        cert = "CERTIFIED_BOUNDED"
    elif total_e_fault_raw <= epsilon:
        cert = "CERTIFIED_BOUNDED"
    elif r4_count > 0:
        cert = "CERTIFIED_BOUNDED"
    else:
        cert = "UNCERTIFIED"

    result = QueryResult(
        dataset=dataset, operator=operator, threshold=threshold,
        k=k, epsilon=epsilon, scenario=scenario,
        fault_fraction=fault_fraction, seed=seed,
        answer_float=answer_float, answer_low=answer_low, answer_high=answer_high,
        certification_status=cert,
        u_depth=u_depth if operator == "COUNT" else 0,
        u_quant=u_quant if operator == "COUNT" else 0,
        u_fault=u_fault if operator == "COUNT" else 0,
        e_quant_magnitude=total_e_quant,
        e_fault_magnitude=total_e_fault_raw,
        n_segments=n_segments,
        active_byte_len=active_byte_len,
        effective_active_planes_read=effective_active,
        zero_plane_reads=zero_reads,
        wasted_zero_plane_fraction=wasted_frac,
        fallback_count=fallback_count,
        uncertified_count=1 if cert == "UNCERTIFIED" else 0,
        segment_count_B4=n_segments,
        r1_count=r1_count, r4_count=r4_count, r6_count=r6_count,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    run_tag = args.run_tag or f"h200_preflight_{int(time.time())}"
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_ROOT / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_file = out_dir / "run_meta.txt"
    csv_file = out_dir / "h200_preflight_matrix.csv"

    all_results: list[QueryResult] = []

    total = len(DATASETS) * 2 * len(K_VALUES) * len(EPSILONS) * len(SCENARIOS) * len(FAULT_FRACTIONS) * len(SEEDS)
    done = 0
    for dataset in DATASETS:
        meta = load_artifact_meta(dataset)
        ds_meta = {
            "n_rows": meta["n_rows"],
            "scale": meta["scale"],
            "active_byte_len": DATASET_META[dataset]["active_byte_len"],
            "encoded_max": meta["encoded_max"],
            "raw_min": meta["raw_min"],
            "raw_max": meta["raw_max"],
            "source_checksum": meta["source_checksum"],
        }
        thresholds = compute_thresholds(meta)
        for operator in ["COUNT", "SUM"]:
            for k in K_VALUES:
                for epsilon in EPSILONS:
                    for scenario in SCENARIOS:
                        for ff in FAULT_FRACTIONS:
                            for seed in SEEDS:
                                r = run_query(dataset, operator, thresholds["medium"], k, epsilon, scenario, ff, seed, ds_meta)
                                all_results.append(r)
                                done += 1
                                if done % 48 == 0:
                                    print(f"  [{done}/{total}] {dataset} {operator} k={k} ε={epsilon} {scenario}")
                                    sys.stdout.flush()

    print(f"\nCompleted {done} queries. Writing results...")

    # Write CSV
    fields = [
        "dataset", "operator", "threshold", "k", "epsilon", "scenario",
        "fault_fraction", "seed", "answer_float", "answer_low", "answer_high",
        "certification_status", "u_depth", "u_quant", "u_fault",
        "e_quant_magnitude", "e_fault_magnitude",
        "n_segments", "active_byte_len",
        "effective_active_planes_read", "zero_plane_reads", "wasted_zero_plane_fraction",
        "fallback_count", "uncertified_count",
        "r1_count", "r4_count",
    ]
    with open(csv_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow({k: getattr(r, k) for k in fields})
    print(f"CSV: {csv_file}")

    # Summary
    certified = sum(1 for r in all_results if r.certification_status.startswith("CERTIFIED"))
    uncertified = sum(1 for r in all_results if r.certification_status == "UNCERTIFIED")
    avg_wasted = np.mean([r.wasted_zero_plane_fraction for r in all_results])
    print(f"\nSummary:")
    print(f"  Total queries: {len(all_results)}")
    print(f"  Certified:     {certified} ({100 * certified / len(all_results):.1f}%)")
    print(f"  UNCERTIFIED:   {uncertified}")

    meta_lines = [
        f"run_tag: {run_tag}",
        f"datasets: {DATASETS}",
        f"k_values: {K_VALUES}",
        f"epsilon: {EPSILONS}",
        f"scenarios: {SCENARIOS}",
        f"fault_fractions: {FAULT_FRACTIONS}",
        f"seeds: {SEEDS}",
        f"total_queries: {len(all_results)}",
        f"certified: {certified}",
        f"uncertified: {uncertified}",
        f"avg_wasted_zero_plane_fraction: {avg_wasted:.4f}",
        f"artifact_root: {ARTIFACT_ROOT}",
        f"scale: {SCALE}",
    ]
    for d in DATASETS:
        meta = load_artifact_meta(d)
        ab = DATASET_META[d]["active_byte_len"]
        meta_lines.append(f"dataset:{d}:active_byte_len={ab},encoded_max={meta['encoded_max']},source_checksum={meta['source_checksum']}")

    meta_file.write_text("\n".join(meta_lines) + "\n")
    print(f"Meta: {meta_file}")
    print("Preflight certification complete.")


if __name__ == "__main__":
    main()
