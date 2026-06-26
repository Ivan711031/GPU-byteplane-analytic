#!/usr/bin/env python3
"""Z1-A: Real-dataset CPU validation (unfiltered SUM).

Lifts the Z1b tiny-fixture CB validation to real encoded artifacts.

Usage:
    python3 scripts/run_z1a_cpu_validation.py \\
        --dataset /path/to/cesm_atm_cloud/seg4096 \\
        [--dataset /path/to/hurricane_u/seg4096] \\
        --csv /path/to/z1a_cpu_validation.csv
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_y0_evaluator import (
    compute_delivered_answer_with_degradation,
    compute_ui_prediction,
    bound_width_prediction_err,
)


TAU_PRED = 0.10
PLANE_WEIGHTS = [1 << (8 * (7 - p)) for p in range(8)]


def load_dataset(artifact_dir: Path) -> dict[str, Any]:
    """Load 8-plane bytes from artifact directory (handles both naming conventions)."""
    # Try manifest.json (real datasets)
    manifest = None
    meta_path = artifact_dir / "manifest.json"
    if meta_path.exists():
        manifest = json.loads(meta_path.read_text())

    # Determine n_rows and plane naming
    plane_files = sorted(artifact_dir.glob("plane_*.bin"))
    if not plane_files:
        raise FileNotFoundError(f"No plane_*.bin files in {artifact_dir}")

    # Detect naming: plane_0.bin (1-digit) vs plane_000.bin (3-digit)
    n_digits = len(plane_files[0].stem.split("_")[-1])
    plane_map: dict[int, Path] = {}
    for pf in plane_files:
        try:
            p_id = int(pf.stem.split("_")[-1])
            plane_map[p_id] = pf
        except (IndexError, ValueError):
            continue

    # Load 8 planes
    planes: list[bytes] = []
    for p in range(8):
        if p in plane_map:
            planes.append(plane_map[p].read_bytes())
        else:
            # Missing plane (empty for some real datasets)
            n_rows = len(planes[0]) if planes else 0
            planes.append(bytes(n_rows))

    n_rows = len(planes[0])
    scale = 1
    dataset_name = manifest.get("dataset", artifact_dir.name) if manifest else artifact_dir.name
    exact_sum = str(manifest.get("exact_sum", 0)) if manifest else "0"

    return {
        "dataset": dataset_name,
        "n_rows": n_rows,
        "scale": scale,
        "planes": planes,
        "encoded_sum": exact_sum,
    }


def compute_clean_sum(planes: list[bytes]) -> int:
    """Compute clean encoded sum from 8 plane byte arrays."""
    total = 0
    for p in range(8):
        total += sum(planes[p]) * PLANE_WEIGHTS[p]
    return total


def contains_truth(clean_answer: float, delivered_answer: float, bound_width: float) -> bool:
    lo = delivered_answer - bound_width / 2.0
    hi = delivered_answer + bound_width / 2.0
    return lo <= clean_answer <= hi


def run_matrix(dataset: dict[str, Any], fault_modes: list[dict],
               k_values: list[int], limit_rows: int = 0) -> list[dict[str, str]]:
    planes = dataset["planes"]
    n_rows = limit_rows if limit_rows > 0 else dataset["n_rows"]
    scale = dataset["scale"]
    ds_name = dataset["dataset"]

    # Truncate to subset if needed
    if limit_rows > 0:
        planes = [p[:limit_rows] for p in planes]
        n_rows = limit_rows

    clean_sum = compute_clean_sum(planes)
    clean_answer = clean_sum / scale

    print(f"  {ds_name}: {n_rows} rows, clean_answer={clean_answer:.6e}", flush=True)

    rows: list[dict[str, str]] = []

    for cfg in fault_modes:
        plane = cfg["plane"]
        mask = cfg["mask"]
        offsets = cfg["offsets"]
        label = cfg["label"]
        policy = cfg.get("policy", "uniform_repair_fraction")

        with __import__("tempfile").TemporaryDirectory() as tmp:
            plan_dir = Path(tmp) / "fault_plans"
            # Single replica with fault
            fp_path = plan_dir / f"plane{plane}/replica0/seed_0.json"
            fp_path.parent.mkdir(parents=True, exist_ok=True)
            entries = [{"offset": o, "mask": mask} for o in offsets[:n_rows]]
            json.dump({"metadata": {}, "entries": entries}, fp_path.open("w"))

            fpp = {plane: [str(fp_path)]}
            r_vector = [1] * 8

            result = compute_delivered_answer_with_degradation(
                clean_planes=planes,
                fault_plan_paths=fpp,
                r_vector=r_vector,
                scale=scale,
                n_rows=n_rows,
                dataset=ds_name,
                policy=policy,
                allocation_r="|".join(str(r) for r in r_vector),
                segment_size=1024,
            )

            pred = compute_ui_prediction(
                clean_planes=planes,
                fault_plan_paths=fpp,
                r_vector=r_vector,
                scale=scale,
                n_rows=n_rows,
                segment_size=1024,
            )

        err = bound_width_prediction_err(result.bound_width, pred["bound_width_predicted"])
        ct = contains_truth(clean_answer, result.delivered_answer, result.bound_width)
        valid_ceiling = pred["bound_width_predicted"] >= result.bound_width * (1.0 - 1e-12)

        rows.append({
            "dataset": ds_name,
            "n_rows": str(n_rows),
            "scale": str(scale),
            "fault_config": label,
            "plane": str(plane),
            "mask": hex(mask),
            "fault_count": str(len(offsets)),
            "policy": policy,
            "clean_answer": f"{clean_answer:.10e}",
            "delivered_answer": f"{result.delivered_answer:.10e}",
            "bound_width_observed": f"{result.bound_width:.10e}",
            "bound_width_predicted": f"{pred['bound_width_predicted']:.10e}",
            "bound_width_prediction_err": f"{err:.10e}",
            "contains_truth": str(ct).lower(),
            "valid_ceiling": str(valid_ceiling).lower(),
        })

        print(f"    plane={plane} {label:20s} ct={ct} err={err:.4e}" +
              (" CEIL" if not valid_ceiling else ""), flush=True)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--csv", type=Path, default=Path("/tmp/z1a_cpu_validation.csv"))
    parser.add_argument("--limit-rows", type=int, default=0,
                        help="Limit rows for quick smoke (0 = full dataset)")
    parser.add_argument("--fault-mode", choices=["single", "saturating", "all_corrupted"],
                        default="all", help="Fault mode (default: all)")
    args = parser.parse_args()

    fault_modes = []
    for plane in range(8):
        fault_modes.append({
            "plane": plane,
            "mask": 0xFF,
            "offsets": [3],
            "label": f"single_p{plane}",
        })
        fault_modes.append({
            "plane": plane,
            "mask": 0xFF,
            "offsets": [0],
            "label": f"saturating_p{plane}",
        })

    print("=" * 60)
    print("Z1-A: Real-dataset CPU validation (unfiltered SUM)")
    print("=" * 60)

    all_rows: list[dict[str, str]] = []
    summary: dict[str, Any] = {}

    for ds_path in args.dataset:
        dp = Path(ds_path)
        print(f"\nLoading: {dp}")
        dataset = load_dataset(dp)
        rows = run_matrix(dataset, fault_modes, k_values=[8], limit_rows=args.limit_rows)
        all_rows.extend(rows)

        # Summary per dataset
        ct_pass = all(r["contains_truth"] == "true" for r in rows)
        ceiling_pass = all(r["valid_ceiling"] == "true" for r in rows)
        errs = [float(r["bound_width_prediction_err"]) for r in rows]
        median_err = sorted(errs)[len(errs) // 2]

        summary[dataset["dataset"]] = {
            "n_configs": len(rows),
            "contains_truth_all": ct_pass,
            "valid_ceiling_all": ceiling_pass,
            "median_prediction_err": median_err,
            "max_prediction_err": max(errs),
        }
        print(f"  => ct_all={ct_pass} ceiling_all={ceiling_pass} "
              f"median_err={median_err:.4e}", flush=True)

    # Write CSV
    fieldnames = [
        "dataset", "n_rows", "scale", "fault_config", "plane", "mask",
        "fault_count", "policy",
        "clean_answer", "delivered_answer",
        "bound_width_observed", "bound_width_predicted",
        "bound_width_prediction_err",
        "contains_truth", "valid_ceiling",
    ]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nCSV: {args.csv} ({len(all_rows)} rows)")

    # Gate check
    print("\n--- Gate ---")
    for ds_name, s in summary.items():
        gate = "PASS" if (s["contains_truth_all"] and s["valid_ceiling_all"]) else "FAIL"
        print(f"  {ds_name}: {gate}  ct={s['contains_truth_all']} "
              f"ceiling={s['valid_ceiling_all']} median_err={s['median_prediction_err']:.4e}")


if __name__ == "__main__":
    main()
