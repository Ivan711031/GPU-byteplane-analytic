#!/usr/bin/env python3
"""Z1-C: Predicate-membership envelope validation on real datasets.

Runs COUNT and filtered-SUM on cesm/hurricane with decoded-domain
p25/p50/p75 thresholds and borderline-population guard.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_z1_filtered_evaluator import (
    compute_filtered_delivered_result,
    select_decoded_thresholds,
    compute_max_flip_rows,
)
from phase3_y0_evaluator import SEGMENT_SIZE, PLANE_WEIGHTS
from run_z1a_cpu_validation import load_dataset

TAU_PRED = 0.10
SUBSET_ROWS = 1_000_000


def is_bfp_format(dataset: dict) -> bool:
    """Detect BFP format by checking for segment_meta.csv in the source path."""
    from pathlib import Path
    src = dataset.get("_source_path", "")
    if src and (Path(src) / "segment_meta.csv").exists():
        return True
    return False


def load_segment_meta(artifact_dir: Path) -> list[dict[str, Any]]:
    """Load segment_meta.csv and return list of dicts."""
    seg_path = artifact_dir / "segment_meta.csv"
    if not seg_path.exists():
        raise FileNotFoundError(f"segment_meta.csv not found: {seg_path}")
    segs = []
    with open(seg_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            apc = int(row["active_plane_count"])
            segs.append({
                "seg_idx": int(row["segment_index"]),
                "row_offset": int(row["row_offset"]),
                "row_count": int(row["row_count"]),
                "active_plane_count": apc,
                "fractional_bits": int(row["fractional_bits"]),
                "plane_basis": [
                    float(row.get(f"plane_basis_{p}", "0"))
                    for p in range(max(1, apc))
                ],
            })
    return segs


def build_seg_lookup(segs: list[dict], n_rows: int) -> list[int]:
    """Build row->segment index lookup array."""
    lookup = np.full(n_rows, -1, dtype=np.int32)
    for seg in segs:
        start = seg["row_offset"]
        end = start + seg["row_count"]
        lookup[start:end] = seg["seg_idx"]
    return lookup


def decode_bfp_row(
    planes: list[bytes],
    seg: dict[str, Any],
    row_idx: int,
) -> float:
    """Decode one BFP row: value = sum(plane_bytes[p][row] * plane_basis[p])."""
    if seg["active_plane_count"] == 0:
        return 0.0
    val = 0.0
    for p in range(seg["active_plane_count"]):
        byte_val = planes[p][row_idx] if row_idx < len(planes[p]) else 0
        val += byte_val * seg["plane_basis"][p]
    return val


def compute_decoded_values_bfp(
    planes: list[bytes],
    seg_lookup: list[int],
    segs: dict[int, Any],
    n_rows: int,
) -> np.ndarray:
    """Compute decoded midpoint for all BFP rows."""
    values = np.zeros(n_rows, dtype=np.float64)
    for i in range(n_rows):
        seg_idx = seg_lookup[i]
        if seg_idx < 0:
            values[i] = 0.0
        else:
            values[i] = decode_bfp_row(planes, segs[seg_idx], i)
    return values


def select_decoded_thresholds_bfp(
    planes: list[bytes],
    seg_lookup: list[int],
    segs: dict[int, Any],
    n_rows: int,
) -> dict[str, float]:
    """Compute decoded thresholds from BFP data, using non-zero values only.

    Returns p25/p50/p75 of the {x : x > 0} distribution (zero-inflated data
    would push p25/p50/p75 to 0 if zeros were included).
    """
    values = compute_decoded_values_bfp(planes, seg_lookup, segs, n_rows)
    non_zero = values[values > 0]
    if len(non_zero) < 100:
        print(f"  WARNING: only {len(non_zero)} non-zero values — using all values")
        non_zero = values
    p25 = float(np.percentile(non_zero, 25))
    p50 = float(np.percentile(non_zero, 50))
    p75 = float(np.percentile(non_zero, 75))
    return {"p25": p25, "p50": p50, "p75": p75}


def compute_max_flip_rows_bfp(
    planes: list[bytes],
    plane: int,
    threshold: float,
    seg_lookup: list[int],
    segs: dict[int, Any],
    n_rows: int,
) -> int:
    """Count BFP rows that WOULD flip membership if plane i is fully corrupted (swing=255*plane_basis[i])."""
    flip_count = 0
    for i in range(n_rows):
        seg_idx = seg_lookup[i]
        if seg_idx < 0:
            continue
        seg = segs[seg_idx]
        if plane >= seg["active_plane_count"]:
            continue
        basis = seg["plane_basis"][plane] if plane < len(seg["plane_basis"]) else 0.0
        max_swing = 255.0 * basis

        val = decode_bfp_row(planes, seg, i)
        clean_Q = val >= threshold
        clean_D = val < threshold - 1e-30  # approximate

        if clean_Q:
            if val - max_swing < threshold:
                flip_count += 1
        elif clean_D:
            if val + max_swing >= threshold:
                flip_count += 1

    return flip_count


def load_subset(path: Path, n: int, offset: int = 0) -> dict[str, Any]:
    ds = load_dataset(path)
    ds["planes"] = [p[offset:offset + n] for p in ds["planes"]]
    ds["n_rows"] = n
    ds["_source_path"] = str(path)
    return ds


def run_cell(
    ds: dict[str, Any],
    functional: str,
    plane: int,
    threshold: float,
    fault_mode: str,
    csv_writer: Any,
    csv_file: Any,
    max_flip_rows: int = 0,
) -> dict[str, Any]:
    n_rows = ds["n_rows"]
    scale = ds["scale"]
    planes = ds["planes"]

    # Build fault entries
    if fault_mode == "single":
        entries = [{"offset": min(3, n_rows - 1), "mask": 0xFF}]
    elif fault_mode == "saturating":
        entries = [{"offset": i, "mask": 0xFF} for i in range(min(8, n_rows))]
    else:
        raise ValueError(f"unknown fault_mode: {fault_mode}")

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        fp = plan_dir / f"plane{plane}/replica0/seed_0.json"
        fp.parent.mkdir(parents=True)
        json.dump({"metadata": {}, "entries": entries}, fp.open("w"))
        fpp = {plane: [str(fp)]}
        rv = [1] * 8

        result = compute_filtered_delivered_result(
            planes, fpp, rv,
            scale=scale, n_rows=n_rows,
            threshold=threshold, functional=functional,
            dataset=ds["dataset"], policy="uniform_repair_fraction",
            allocation_r="|".join(str(r) for r in rv),
        )

    row = {
        "functional": functional,
        "dataset": ds["dataset"],
        "n_rows": str(n_rows),
        "scale": str(scale),
        "plane": str(plane),
        "threshold": f"{threshold:.10e}",
        "fault_mode": fault_mode,
        "ff_qualified_count": str(result.ff_qualified_count),
        "delivered_qualified_count": str(result.delivered_qualified_count),
        "uncertainty_count": str(result.uncertainty_count),
        "contains_truth": str(result.contains_truth).lower(),
        "bound_width_observed": f"{result.bound_width:.10e}",
        "bound_width_predicted": f"{result.bound_width_predicted:.10e}",
        "bound_width_prediction_err": f"{result.bound_width_prediction_err:.10e}",
        "valid_ceiling": str(
            result.bound_width_predicted >= result.bound_width * (1.0 - 1e-12)
        ).lower(),
        "gate1_pass": str(result.contains_truth).lower(),
        "ff_filtered_sum": f"{result.ff_filtered_sum:.10e}",
        "delivered_filtered_sum": f"{result.delivered_filtered_sum:.10e}",
        "max_flip_rows": str(max_flip_rows),
        "envelope_exercised": "true",
    }

    csv_writer.writerow(row)
    csv_file.flush()
    return row


def main() -> None:
    datasets = [
        ("cesm_atm_cloud", Path("/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"), 24756224),
        ("hurricane_u", Path("/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg4096"), 0),
    ]
    functionals = ["count", "filtered_sum"]
    fault_modes = ["single", "saturating"]

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "z1c_login")
    results_dir = Path("results/reliability_layer1/phase3/phase3z_z1") / f"job_{slurm_job_id}"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "z1c_envelope_validation.csv"

    fieldnames = [
        "functional", "dataset", "n_rows", "scale", "plane", "threshold",
        "fault_mode", "ff_qualified_count", "delivered_qualified_count",
        "uncertainty_count", "contains_truth", "bound_width_observed",
        "bound_width_predicted", "bound_width_prediction_err", "valid_ceiling",
        "gate1_pass", "ff_filtered_sum", "delivered_filtered_sum",
        "max_flip_rows", "envelope_exercised",
    ]

    print("=" * 60)
    print("Z1-C: Predicate-membership envelope validation")
    print("=" * 60)

    all_rows: list[dict[str, str]] = []
    summary: dict[str, Any] = {}

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for ds_name, ds_path, ds_offset in datasets:
            print(f"\n--- {ds_name} ---")
            ds = load_subset(ds_path, SUBSET_ROWS, offset=ds_offset)

            # Detect BFP format vs fixed-width
            bfp = is_bfp_format(ds)
            if bfp:
                segs = load_segment_meta(ds_path)
                seg_map = {s["seg_idx"]: s for s in segs}
                seg_lookup = build_seg_lookup(segs, ds_offset + ds["n_rows"])
                # Slice lookup to our subset
                seg_lookup = seg_lookup[ds_offset:ds_offset + ds["n_rows"]]
                thresholds = select_decoded_thresholds_bfp(
                    ds["planes"], seg_lookup, seg_map, ds["n_rows"]
                )
                print(f"  loaded {ds['n_rows']} rows (BFP format, {len(segs)} segments)")
            else:
                thresholds = select_decoded_thresholds(
                    ds["planes"], ds["scale"], ds["n_rows"]
                )
                print(f"  loaded {ds['n_rows']} rows, scale={ds['scale']}")

            print(f"  decoded thresholds: { {k: f'{v:.6e}' for k, v in thresholds.items()} }")

            ds_summary = {
                "thresholds": {k: float(v) for k, v in thresholds.items()},
                "cells_total": 0,
                "cells_exercised": 0,
                "cells_passed": 0,
                "cells_not_exercised": [],
                "cells_failed": [],
            }

            for functional in functionals:
                for thr_label, thr_val in thresholds.items():
                    for fault_mode in fault_modes:
                        for plane in range(8):
                            ds_summary["cells_total"] += 1

                            # Guard (BFP or fixed-width)
                            if bfp:
                                mfr = compute_max_flip_rows_bfp(
                                    ds["planes"], plane, thr_val,
                                    seg_lookup, seg_map, ds["n_rows"]
                                )
                            else:
                                mfr = compute_max_flip_rows(
                                    ds["planes"], plane, thr_val,
                                    ds["scale"], ds["n_rows"]
                                )
                            if mfr == 0:
                                row = {
                                    "functional": functional,
                                    "dataset": ds_name,
                                    "plane": str(plane),
                                    "threshold": f"{thr_val:.10e}",
                                    "fault_mode": fault_mode,
                                    "max_flip_rows": "0",
                                    "envelope_exercised": "false",
                                    "contains_truth": "n/a",
                                    "gate1_pass": "n/a",
                                }
                                writer.writerow(row)
                                f.flush()
                                ds_summary["cells_not_exercised"].append(
                                    f"{functional}/{thr_label}/p{plane}/{fault_mode}"
                                )
                                continue

                            ds_summary["cells_exercised"] += 1

                            # Run
                            t0 = time.time()
                            res = run_cell(
                                ds, functional, plane, thr_val, fault_mode,
                                writer, f, max_flip_rows=mfr,
                            )
                            elapsed = time.time() - t0

                            gt1 = res["contains_truth"] == "true"
                            gt1_name = "OK" if gt1 else "FAIL"
                            print(f"  {functional:12s} {thr_label:>4s} {fault_mode:10s} "
                                  f"p{plane} ct={gt1_name} mfr={mfr} ({elapsed:.1f}s)", flush=True)

                            if gt1:
                                ds_summary["cells_passed"] += 1
                            else:
                                ds_summary["cells_failed"].append(
                                    f"{functional}/{thr_label}/p{plane}/{fault_mode}"
                                )

            summary[ds_name] = ds_summary
            print(f"  => {ds_summary['cells_exercised']}/{ds_summary['cells_total']} exercised, "
                  f"{ds_summary['cells_passed']} passed, {len(ds_summary['cells_failed'])} failed")

    print(f"\nCSV: {csv_path}")

    # Gate check
    print("\n--- Gate 1 (hard) ---")
    all_ok = True
    for ds_name, s in summary.items():
        if s["cells_failed"]:
            print(f"  {ds_name}: FAIL — {s['cells_failed']}")
            all_ok = False
        else:
            print(f"  {ds_name}: PASS ({s['cells_passed']}/{s['cells_exercised']} exercised, "
                  f"{len(s['cells_not_exercised'])} not exercised)")

    if all_ok:
        print("\n  Verdict: Z1C_ENVELOPE_SUPPORTED")
    else:
        print("\n  Verdict: CB_FAILS_CONTAINMENT")

    if any(len(s["cells_not_exercised"]) > 0 for s in summary.values()):
        print("  Note: some cells had envelope_NOT_exercised (max_flip_rows=0)")
        for ds_name, s in summary.items():
            if s["cells_not_exercised"]:
                print(f"    {ds_name}: {len(s['cells_not_exercised'])} cells — {s['cells_not_exercised'][:3]}...")


if __name__ == "__main__":
    main()
