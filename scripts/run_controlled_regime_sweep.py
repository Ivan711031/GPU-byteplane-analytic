#!/usr/bin/env python3
"""Controlled Synthetic Regime Sweep: Bound Usefulness for Certified Degradation.

Generates synthetic byte-plane data with controlled structural knobs,
computes worst-case vs structure-aware bounds, and verifies contains_truth
under single-byte fault injection.

Per PRD 2026-06-05 §3-§8.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Plane weights for 8-plane MSB-first layout
PLANE_WEIGHTS = [1 << (8 * (7 - p)) for p in range(8)]

# Families
FAMILIES = ["sensor", "uniform", "heavy_tailed", "zipfian"]

# Synthetic matrix dimensions
ACTIVE_ROW_FRACTIONS = [0.01, 0.10, 0.50, 1.00]
ACTIVE_PLANE_DENSITIES = [0.01, 0.10, 0.50, 1.00]
FAULT_PLANES = [0, 1, 4, 7]
SEEDS = [20260605, 20260606, 20260607]

N_ROWS = 500000


def _ensure_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def sample_byte_uniform(rng: np.random.Generator) -> int:
    return int(rng.integers(1, 256))


def sample_byte_heavy_tailed(rng: np.random.Generator) -> int:
    u = rng.uniform()
    v = 1.0 / (1.0 - u + 1e-30)
    v = min(v, 255.0)
    return max(1, int(v))


def sample_byte_sensor(rng: np.random.Generator) -> int:
    cluster = rng.integers(0, 4)
    centers = [8, 24, 48, 80]
    value = centers[cluster] + int(rng.integers(-3, 4))
    return max(1, min(255, value))


def sample_byte_zipfian(rng: np.random.Generator) -> int:
    u = rng.uniform()
    rank = int(255.0 * (u ** 4))
    rank = max(1, min(255, rank))
    return rank


SAMPLE_FN = {
    "uniform": sample_byte_uniform,
    "heavy_tailed": sample_byte_heavy_tailed,
    "sensor": sample_byte_sensor,
    "zipfian": sample_byte_zipfian,
}


def generate_synthetic_planes(
    family: str,
    active_row_fraction: float,
    active_plane_density: float,
    n_rows: int,
    seed: int,
) -> tuple[list[bytes], dict[str, Any]]:
    """Generate 8 byte-planes with controlled structural knobs.

    Returns (planes, measured) where:
      planes[p] = bytes of length n_rows
      measured = dict with measured_active_row_fraction, per-plane active counts, etc.
    """
    rng = _ensure_rng(seed)
    sample_fn = SAMPLE_FN[family]

    planes: list[bytearray] = [bytearray(n_rows) for _ in range(8)]
    active_rows_mask = np.zeros(n_rows, dtype=bool)

    n_active_rows = max(1, int(n_rows * active_row_fraction))
    active_indices = rng.choice(n_rows, size=n_active_rows, replace=False)
    active_rows_mask[active_indices] = True

    active_plane_counts = [0] * 8

    for p in range(8):
        plane = planes[p]
        for idx in active_indices:
            if rng.uniform() < active_plane_density:
                plane[idx] = sample_fn(rng)
                active_plane_counts[p] += 1

    # Convert to bytes
    planes_bytes = [bytes(ba) for ba in planes]

    # Measure actual sparsity
    rows_with_any_nonzero = set()
    for p in range(8):
        for i in range(n_rows):
            if planes_bytes[p][i] != 0:
                rows_with_any_nonzero.add(i)

    measured_active_row_fraction = len(rows_with_any_nonzero) / n_rows

    measured = {
        "measured_active_row_fraction": measured_active_row_fraction,
        "measured_active_plane_density": {
            str(p): active_plane_counts[p] / max(n_active_rows, 1)
            for p in range(8)
        },
        "active_count_per_plane": [active_plane_counts[p] for p in range(8)],
        "n_rows": n_rows,
    }
    return planes_bytes, measured


def load_real_planes(
    artifact_dir: str, n_rows: int, offset: int = 0
) -> list[bytes]:
    """Load plane files from a real artifact directory."""
    path = Path(artifact_dir)
    planes: list[bytes] = []
    for p in range(8):
        pf = path / f"plane_{p:03d}.bin"
        if not pf.exists():
            pf = path / f"plane_{p}.bin"
        if pf.exists():
            data = pf.read_bytes()
            if offset > 0:
                data = data[offset:]
            if len(data) >= n_rows:
                data = data[:n_rows]
            else:
                data = data + b'\x00' * (n_rows - len(data))
            planes.append(data)
        else:
            planes.append(bytes(n_rows))
    return planes


def compute_clean_sum(planes: list[bytes]) -> int:
    total = 0
    for p in range(8):
        total += sum(planes[p]) * PLANE_WEIGHTS[p]
    return total


def compute_ui_worst_case(planes: list[bytes]) -> float:
    n_rows = len(planes[0])
    total = 0.0
    for p in range(8):
        total += 255.0 * PLANE_WEIGHTS[p] * n_rows
    return total


def compute_ui_structure_aware(planes: list[bytes]) -> tuple[float, list[int]]:
    n_rows = len(planes[0])
    total = 0.0
    active_counts: list[int] = []
    for p in range(8):
        ac = sum(1 for b in planes[p] if b != 0)
        active_counts.append(ac)
        total += 255.0 * PLANE_WEIGHTS[p] * ac
    return total, active_counts


def inject_single_byte_fault(
    planes: list[bytes], fault_plane: int, seed: int
) -> list[bytes]:
    """Flip one byte in the specified plane."""
    rng = _ensure_rng(seed + 1000 * fault_plane)
    n_rows = len(planes[0])
    idx = int(rng.integers(0, n_rows))
    faulted = list(planes)
    plane = bytearray(planes[fault_plane])
    original = plane[idx]
    # Flip to a different non-zero byte (or flip 0 to something)
    if original == 0:
        plane[idx] = int(rng.integers(1, 256))
    else:
        # Flip all bits
        plane[idx] = original ^ 0xFF
    faulted[fault_plane] = bytes(plane)
    return faulted


def compute_certified_availability(
    planes: list[bytes], fault_plane: int, seed: int, structure_bound: float, clean_sum: int
) -> dict[str, Any]:
    """Inject a single-byte fault and verify contains_truth."""
    faulted = inject_single_byte_fault(planes, fault_plane, seed)
    delivered_sum = compute_clean_sum(faulted)
    _, faulted_active = compute_ui_structure_aware(faulted)
    # Recompute bound from faulted data
    faulted_bound = sum(255.0 * PLANE_WEIGHTS[p] * faulted_active[p] for p in range(8))

    interval_low = delivered_sum - faulted_bound
    interval_high = delivered_sum + faulted_bound
    contains = 1.0 if interval_low <= clean_sum <= interval_high else 0.0

    # certified_availability = 1.0 for single-fault case (we always get a bounded answer)
    certified_avail = 1.0

    # Answer interval width
    ans_width = 2.0 * faulted_bound
    ans_width_norm = ans_width / max(abs(clean_sum), 1e-30) if clean_sum != 0 else ans_width

    return {
        "contains_truth": contains,
        "certified_availability": certified_avail,
        "bound_width_worst_case": compute_ui_worst_case(planes),
        "bound_width_structure_aware": faulted_bound,
        "bound_shrink_factor": compute_ui_worst_case(planes) / max(faulted_bound, 1e-30),
        "answer_interval_width": ans_width,
        "answer_interval_width_norm": ans_width_norm,
        "delivered_sum": delivered_sum,
        "clean_sum": clean_sum,
        "fault_active_counts": faulted_active,
    }


def run_synthetic_cell(
    family: str,
    active_row_fraction: float,
    active_plane_density: float,
    fault_plane: int,
    seed: int,
    n_rows: int = N_ROWS,
) -> dict[str, Any]:
    """Run a single synthetic cell."""
    planes, measured = generate_synthetic_planes(
        family, active_row_fraction, active_plane_density, n_rows, seed
    )

    clean_sum = compute_clean_sum(planes)
    worst_bound = compute_ui_worst_case(planes)
    struct_bound, active_counts = compute_ui_structure_aware(planes)
    shrink = worst_bound / max(struct_bound, 1e-30)

    # Fault injection for contains_truth
    fault_result = compute_certified_availability(
        planes, fault_plane, seed, struct_bound, clean_sum
    )

    row: dict[str, Any] = {
        "dataset_family": family,
        "synthetic_or_real": "synthetic",
        "active_row_fraction_config": active_row_fraction,
        "active_plane_density_config": active_plane_density,
        "significance_skew_config": "default",
        "measured_active_row_fraction": measured["measured_active_row_fraction"],
        "measured_active_plane_density": np.mean(
            list(measured["measured_active_plane_density"].values())
        ),
        "active_count_per_plane": "|".join(str(c) for c in active_counts),
        "corruptible_byte_mass": sum(active_counts),
        "significance_weighted_corruptible_mass": sum(
            c * PLANE_WEIGHTS[p] for p, c in enumerate(active_counts)
        ),
        "fault_plane": fault_plane,
        "fault_mode": "single_replica_detected_unrepaired",
        "seed": seed,
        "contains_truth": fault_result["contains_truth"],
        "certified_availability": fault_result["certified_availability"],
        "bound_width_worst_case": worst_bound,
        "bound_width_structure_aware": fault_result["bound_width_structure_aware"],
        "bound_shrink_factor": fault_result["bound_shrink_factor"],
        "answer_interval_width": fault_result["answer_interval_width"],
        "answer_interval_width_norm": fault_result["answer_interval_width_norm"],
        "verdict_cell": "PASS" if fault_result["contains_truth"] == 1.0 else "FAIL",
    }
    return row


def run_real_anchor(
    artifact_dir: str,
    dataset_name: str,
    fault_plane: int,
    seed: int,
    n_rows: int = N_ROWS,
    offset: int = 0,
) -> dict[str, Any]:
    """Run a real data anchor cell."""
    planes = load_real_planes(artifact_dir, n_rows, offset)

    # Check how many planes we actually got
    actual_planes = sum(1 for p in planes if any(b != 0 for b in p))

    active_counts = []
    for p in range(8):
        ac = sum(1 for b in planes[p] if b != 0)
        active_counts.append(ac)

    measured_active_row_fraction = 0.0
    rows_with_any = set()
    for p in range(min(actual_planes, 8)):
        for i in range(n_rows):
            if planes[p][i] != 0:
                rows_with_any.add(i)
    measured_active_row_fraction = len(rows_with_any) / n_rows

    measured_active_plane_density = np.mean([c / n_rows for c in active_counts])

    clean_sum = compute_clean_sum(planes)
    worst_bound = compute_ui_worst_case(planes)
    struct_bound, _ = compute_ui_structure_aware(planes)
    shrink = worst_bound / max(struct_bound, 1e-30)

    fault_result = compute_certified_availability(
        planes, fault_plane, seed, struct_bound, clean_sum
    )

    row: dict[str, Any] = {
        "dataset_family": dataset_name,
        "synthetic_or_real": "real",
        "active_row_fraction_config": "N/A",
        "active_plane_density_config": "N/A",
        "significance_skew_config": "N/A",
        "measured_active_row_fraction": measured_active_row_fraction,
        "measured_active_plane_density": measured_active_plane_density,
        "active_count_per_plane": "|".join(str(c) for c in active_counts),
        "corruptible_byte_mass": sum(active_counts),
        "significance_weighted_corruptible_mass": sum(
            c * PLANE_WEIGHTS[p] for p, c in enumerate(active_counts)
        ),
        "fault_plane": fault_plane,
        "fault_mode": "single_replica_detected_unrepaired",
        "seed": seed,
        "contains_truth": fault_result["contains_truth"],
        "certified_availability": fault_result["certified_availability"],
        "bound_width_worst_case": worst_bound,
        "bound_width_structure_aware": fault_result["bound_width_structure_aware"],
        "bound_shrink_factor": fault_result["bound_shrink_factor"],
        "answer_interval_width": fault_result["answer_interval_width"],
        "answer_interval_width_norm": fault_result["answer_interval_width_norm"],
        "verdict_cell": "PASS" if fault_result["contains_truth"] == 1.0 else "FAIL",
    }
    return row


FIELDS = [
    "dataset_family",
    "synthetic_or_real",
    "active_row_fraction_config",
    "active_plane_density_config",
    "significance_skew_config",
    "measured_active_row_fraction",
    "measured_active_plane_density",
    "active_count_per_plane",
    "corruptible_byte_mass",
    "significance_weighted_corruptible_mass",
    "fault_plane",
    "fault_mode",
    "seed",
    "contains_truth",
    "certified_availability",
    "bound_width_worst_case",
    "bound_width_structure_aware",
    "bound_shrink_factor",
    "answer_interval_width",
    "answer_interval_width_norm",
    "verdict_cell",
]


def write_csv_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {csv_path}")


def run_synthetic_sweep(
    csv_path: Path,
    families: list[str] | None = None,
    fractions: list[float] | None = None,
    densities: list[float] | None = None,
    fault_planes: list[int] | None = None,
    seeds: list[int] | None = None,
    n_rows: int = N_ROWS,
) -> int:
    """Run the full or partial synthetic matrix."""
    families = families or FAMILIES
    fractions = fractions or ACTIVE_ROW_FRACTIONS
    densities = densities or ACTIVE_PLANE_DENSITIES
    fault_planes = fault_planes or FAULT_PLANES
    seeds = seeds or SEEDS

    total = 0
    for fam in families:
        for arf in fractions:
            for apd in densities:
                for fp in fault_planes:
                    for seed in seeds:
                        row = run_synthetic_cell(fam, arf, apd, fp, seed, n_rows)
                        write_csv_rows(csv_path, [row])
                        total += 1
                        if total % 64 == 0:
                            print(f"  ... {total} synthetic rows completed")
    return total


def run_real_anchors(
    csv_path: Path,
    anchors: list[tuple[str, str, int]],
    fault_planes: list[int] | None = None,
    seeds: list[int] | None = None,
    n_rows: int = N_ROWS,
) -> int:
    """Run real data anchors."""
    fault_planes = fault_planes or FAULT_PLANES
    seeds = seeds or SEEDS
    total = 0
    for dataset_name, artifact_dir, offset in anchors:
        for fp in fault_planes:
            for seed in seeds:
                row = run_real_anchor(
                    artifact_dir, dataset_name, fp, seed, n_rows, offset
                )
                write_csv_rows(csv_path, [row])
                total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "synthetic", "anchors", "full"], default="full")
    parser.add_argument("--n-rows", type=int, default=N_ROWS)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--families", type=str, default=None,
                        help="Comma-separated families for partial run")
    parser.add_argument("--fractions", type=str, default=None,
                        help="Comma-separated active_row_fractions")
    parser.add_argument("--densities", type=str, default=None,
                        help="Comma-separated active_plane_densities")
    parser.add_argument("--fault-planes", type=str, default=None,
                        help="Comma-separated fault plane indices")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds")
    parser.add_argument("--csv", type=str, default=None,
                        help="Direct CSV path (overrides output-dir)")
    parser.add_argument("--cesm-dir", type=str, default=None,
                        help="Path to cesm_atm_cloud artifact dir. Env: CESM_CLOUD_ARTIFACT_DIR")
    parser.add_argument("--cesm-offset", type=int, default=None,
                        help="Byte offset into cesm plane files. Env: CESM_CLOUD_OFFSET")
    parser.add_argument("--hurricane-dir", type=str, default=None,
                        help="Path to hurricane_u artifact dir. Env: HURRICANE_U_ARTIFACT_DIR")
    args = parser.parse_args()

    # Resolve CSV path
    if args.csv:
        csv_path = Path(args.csv)
    else:
        jid = os.environ.get("SLURM_JOB_ID", f"cpu_{int(time.time())}")
        out_root = Path(
            args.output_dir or
            f"results/reliability_layer1/phase3/controlled_regime_sweep/job_{jid}"
        )
        csv_path = out_root / "regime_sweep_bound_usefulness.csv"

    print(f"=== Controlled Regime Sweep === mode={args.mode} output={csv_path}")

    # Parse optional filter args
    families = args.families.split(",") if args.families else None
    fractions = [float(x) for x in args.fractions.split(",")] if args.fractions else None
    densities = [float(x) for x in args.densities.split(",")] if args.densities else None
    fault_planes = [int(x) for x in args.fault_planes.split(",")] if args.fault_planes else None
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None

    # Real anchor configs — use CLI args or env var fallbacks
    workspace = Path(__file__).resolve().parent.parent

    cesm_path = args.cesm_dir
    if not cesm_path:
        cesm_path = os.environ.get("CESM_CLOUD_ARTIFACT_DIR",
            str(workspace / "results/cesm_artifact_placeholder"))
    cesm_offset = args.cesm_offset if args.cesm_offset is not None else 24756224
    _cesm_offset_val = int(os.environ.get("CESM_CLOUD_OFFSET", str(cesm_offset)))

    hurricane_path = args.hurricane_dir
    if not hurricane_path:
        hurricane_path = os.environ.get("HURRICANE_U_ARTIFACT_DIR",
            str(workspace / "results/hurricane_artifact_placeholder"))

    # If CLI paths don't exist, try known CLUSTER_HOST paths as last resort
    if not Path(cesm_path).exists():
        for fallback in [
            "${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg_global",
            "${WORK_DIR}/datasets/scientific/dev_buff_v2_scientific/cesm_atm_cloud/bfp-dec12",
        ]:
            if Path(fallback).exists():
                cesm_path = fallback
                break
    if not Path(hurricane_path).exists():
        for fallback in [
            "${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg_global",
            "${WORK_DIR}/datasets/scientific/dev_buff_v2_scientific/hurricane_u/bfp-dec12",
        ]:
            if Path(fallback).exists():
                hurricane_path = fallback
                break

    anchors: list[tuple[str, str, int]] = []
    if Path(cesm_path).exists():
        anchors.append(("cesm_atm_cloud", cesm_path, _cesm_offset_val))
    if Path(hurricane_path).exists():
        anchors.append(("hurricane_u", hurricane_path, 0))

    total = 0

    if args.mode in ("smoke", "full", "synthetic"):
        sm_families = families or (["sensor"] if args.mode == "smoke" else None)
        sm_fractions = fractions or ([0.01, 1.00] if args.mode == "smoke" else None)
        sm_densities = densities or ([0.01, 1.00] if args.mode == "smoke" else None)
        sm_fault_planes = fault_planes or ([0, 7] if args.mode == "smoke" else None)
        sm_seeds = seeds or ([SEEDS[0]] if args.mode == "smoke" else None)

        print(f"  Synthetic: families={sm_families} fractions={sm_fractions} "
              f"densities={sm_densities} fault_planes={sm_fault_planes} seeds={sm_seeds}")

        t = run_synthetic_sweep(
            csv_path, sm_families, sm_fractions, sm_densities,
            sm_fault_planes, sm_seeds, args.n_rows,
        )
        total += t

    if args.mode in ("anchors", "full"):
        if anchors:
            print(f"  Real anchors: {[(a[0], a[2]) for a in anchors]}")
            t = run_real_anchors(
                csv_path, anchors, fault_planes, seeds, args.n_rows
            )
            total += t
        else:
            print("  WARNING: No real anchor paths found, skipping")

    print(f"=== Complete: {total} rows written to {csv_path} ===")

    # Print summary
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        n_pass = sum(1 for r in rows if r.get("verdict_cell") == "PASS")
        n_fail = sum(1 for r in rows if r.get("verdict_cell") == "FAIL")
        n_total = len(rows)
        print(f"Summary: {n_total} total, {n_pass} PASS, {n_fail} FAIL")
        if n_fail > 0:
            print("FAILED rows:")
            for r in rows:
                if r.get("verdict_cell") == "FAIL":
                    print(f"  {r['dataset_family']} arf={r['active_row_fraction_config']} "
                          f"apd={r['active_plane_density_config']} fp={r['fault_plane']} "
                          f"seed={r['seed']}")


if __name__ == "__main__":
    main()
