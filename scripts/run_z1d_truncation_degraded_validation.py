#!/usr/bin/env python3
"""Z1-D: k-depth truncation + degraded-repair validation.

Case 1 (truncation):     k in {1, 2, k_max-1}; corrupt plane j; verify Z1a §1.5
                         no-double-count: for j >= k, truncation residual replaces
                         U_j; for j < k, U_j is counted separately.
Case 2 (degraded-repair): r_p in {2, 3} replicas, detectable disagreement
                         (no majority), must produce certified interval with
                         contains_truth==1.0.

SUM32/digest detection primitive (not CRC).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_y0_evaluator import (
    PLANE_WEIGHTS,
    SEGMENT_SIZE,
    compute_voted_planes,
    compute_segment_outcomes,
    compute_ui_prediction,
    bound_width_prediction_err,
)
from run_z1a_cpu_validation import load_dataset

TAU_PRED = 0.10
SUBSET_ROWS = 1_000_000

FIELDNAMES = [
    "case", "dataset", "n_rows", "scale", "k", "corrupt_plane",
    "j_lt_k", "r_p",
    "clean_answer", "delivered_answer",
    "bound_width_observed", "bound_width_predicted",
    "bound_width_prediction_err",
    "bound_fault_free", "truncation_residual",
    "measured_widen", "predicted_widen",
    "contains_truth", "valid_ceiling",
    "segments_crc_hit", "segments_degraded",
    "segments_repaired", "segments_unprotected",
    "notes",
]


def load_subset(path: Path, n: int, offset: int = 0) -> dict[str, Any]:
    ds = load_dataset(path)
    ds["planes"] = [p[offset:offset + n] for p in ds["planes"]]
    ds["n_rows"] = n
    return ds


def clean_sum_at_k(planes: list[bytes], k: int) -> int:
    total = 0
    for p in range(min(k, 8)):
        total += sum(planes[p]) * PLANE_WEIGHTS[p]
    return total


def delivered_sum_at_k(
    clean_planes: list[bytes],
    fault_plan_paths: dict[int, list[str]],
    r_vector: list[int],
    k: int,
    n_rows: int,
    segment_size: int = SEGMENT_SIZE,
) -> int:
    voted = compute_voted_planes(clean_planes, fault_plan_paths, r_vector)
    outcomes = compute_segment_outcomes(
        clean_planes, fault_plan_paths, r_vector, segment_size
    )
    total = 0
    for i in range(n_rows):
        val = 0
        seg_idx = i // segment_size
        for p in range(min(k, 8)):
            st = outcomes.get((seg_idx, p), "clean")
            b = clean_planes[p][i] if st in ("clean", "repaired") else voted[p][i]
            val += b * PLANE_WEIGHTS[p]
        total += val
    return total


def compute_truncation_bound(
    clean_planes: list[bytes],
    fault_plan_paths: dict[int, list[str]],
    r_vector: list[int],
    scale: int,
    n_rows: int,
    k: int,
    corrupt_plane: int,
    segment_size: int = SEGMENT_SIZE,
) -> dict[str, Any]:
    outcomes = compute_segment_outcomes(
        clean_planes, fault_plan_paths, r_vector, segment_size
    )
    voted_planes = compute_voted_planes(
        clean_planes, fault_plan_paths, r_vector
    )
    n_segments = (n_rows + segment_size - 1) // segment_size

    q_err = 0.5 / scale
    bound_fault_free = 2.0 * q_err * n_rows

    max_undecoded = 0
    if k < 8:
        max_undecoded = (1 << (8 * (8 - k))) - 1
    trunc_residual = n_rows * max_undecoded / scale

    measured_widen = 0.0
    predicted_widen = 0.0

    for p in range(min(k, 8)):
        plane_measured = 0.0
        plane_predicted = 0.0
        for seg_idx in range(n_segments):
            st = outcomes.get((seg_idx, p), "clean")
            if st == "clean" or st == "repaired":
                continue
            start = seg_idx * segment_size
            end = min(start + segment_size, n_rows)
            count = end - start
            contrib = count * 255 * PLANE_WEIGHTS[p] / scale
            plane_predicted += contrib
            if st == "degraded":
                for i in range(start, end):
                    diff = abs(voted_planes[p][i] - clean_planes[p][i])
                    plane_measured += diff * PLANE_WEIGHTS[p] / scale
            elif st == "unprotected":
                plane_measured += contrib
        measured_widen += plane_measured
        predicted_widen += plane_predicted

    bound_observed = bound_fault_free + trunc_residual + measured_widen
    bound_predicted = bound_fault_free + trunc_residual + predicted_widen

    return {
        "bound_fault_free": bound_fault_free,
        "truncation_residual": trunc_residual,
        "measured_widen": measured_widen,
        "predicted_widen": predicted_widen,
        "bound_width_observed": bound_observed,
        "bound_width_predicted": bound_predicted,
        "outcomes": outcomes,
    }


def contains_truth(
    clean_answer: float, delivered_answer: float, bound_width: float
) -> bool:
    lo = delivered_answer - bound_width / 2.0
    hi = delivered_answer + bound_width / 2.0
    return lo <= clean_answer <= hi


def run_truncation_cell(
    writer: csv.DictWriter,
    csv_file: Any,
    ds_name: str,
    ds: dict[str, Any],
    k: int,
    corrupt_plane: int,
    k_max: int,
) -> dict[str, str]:
    planes = ds["planes"]
    scale = ds["scale"]
    n_rows = ds["n_rows"]
    j_lt_k = corrupt_plane < k

    # Create single-replica fault on corrupt_plane
    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        fp = plan_dir / f"plane{corrupt_plane}/replica0/seed_0.json"
        fp.parent.mkdir(parents=True)
        entries = [{"offset": i, "mask": 0xFF} for i in range(min(8, n_rows))]
        json.dump({"metadata": {}, "entries": entries}, fp.open("w"))
        fpp = {corrupt_plane: [str(fp)]}
        rv = [1] * 8

        clean_total = clean_sum_at_k(planes, k)
        clean_answer = clean_total / scale

        delivered_total = delivered_sum_at_k(
            planes, fpp, rv, k, n_rows
        )
        delivered_answer = delivered_total / scale

        bnd = compute_truncation_bound(
            planes, fpp, rv, scale, n_rows, k, corrupt_plane
        )

        ct = contains_truth(
            clean_answer, delivered_answer, bnd["bound_width_predicted"]
        )
        err = bound_width_prediction_err(
            bnd["bound_width_observed"], bnd["bound_width_predicted"]
        )
        valid_ceil = bnd["bound_width_predicted"] >= bnd["bound_width_observed"] * (
            1.0 - 1e-12
        )

        # Count segment outcomes
        outcomes = bnd["outcomes"]
        crc_hit = sum(
            1
            for st in outcomes.values()
            if st in ("degraded", "unprotected", "repaired")
        )
        degraded = sum(1 for st in outcomes.values() if st == "degraded")
        repaired = sum(1 for st in outcomes.values() if st == "repaired")
        unprotected = sum(1 for st in outcomes.values() if st == "unprotected")

    note_parts = []
    if not j_lt_k:
        note_parts.append(
            f"j={corrupt_plane} >= k={k}: truncation residual replaces U_j, "
            f"no double-count"
        )
    else:
        note_parts.append(
            f"j={corrupt_plane} < k={k}: U_j counted, "
            f"residual={bnd['truncation_residual']:.6e}"
        )
    if corrupt_plane >= k_max:
        note_parts.append(f"corrupt_plane beyond dataset k_max={k_max}")

    row = {
        "case": "truncation",
        "dataset": ds_name,
        "n_rows": str(n_rows),
        "scale": str(scale),
        "k": str(k),
        "corrupt_plane": str(corrupt_plane),
        "j_lt_k": str(j_lt_k).lower(),
        "r_p": "1",
        "clean_answer": f"{clean_answer:.10e}",
        "delivered_answer": f"{delivered_answer:.10e}",
        "bound_width_observed": f"{bnd['bound_width_observed']:.10e}",
        "bound_width_predicted": f"{bnd['bound_width_predicted']:.10e}",
        "bound_width_prediction_err": f"{err:.10e}",
        "bound_fault_free": f"{bnd['bound_fault_free']:.10e}",
        "truncation_residual": f"{bnd['truncation_residual']:.10e}",
        "measured_widen": f"{bnd['measured_widen']:.10e}",
        "predicted_widen": f"{bnd['predicted_widen']:.10e}",
        "contains_truth": str(ct).lower(),
        "valid_ceiling": str(valid_ceil).lower(),
        "segments_crc_hit": str(crc_hit),
        "segments_degraded": str(degraded),
        "segments_repaired": str(repaired),
        "segments_unprotected": str(unprotected),
        "notes": "; ".join(note_parts),
    }
    writer.writerow(row)
    csv_file.flush()
    return row


def run_degraded_cell(
    writer: csv.DictWriter,
    csv_file: Any,
    ds_name: str,
    ds: dict[str, Any],
    plane: int,
    r_p: int,
) -> dict[str, str]:
    planes = ds["planes"]
    scale = ds["scale"]
    n_rows = ds["n_rows"]

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        entries = [{"offset": i, "mask": 0xFF} for i in range(min(8, n_rows))]
        entries_alt = [{"offset": i, "mask": 0x01} for i in range(min(8, n_rows))]

        # r_p = 2: replica0=0xFF, replica1=0x01 (no majority)
        # r_p = 3: replica0=0xFF, replica1=0x01, replica2=0x10 (all different, no majority)
        fp_paths = []
        for rep_idx in range(r_p):
            if rep_idx == 0:
                mask = 0xFF
            elif rep_idx == 1:
                mask = 0x01
            else:
                mask = 0x10
            fp = plan_dir / f"plane{plane}/replica{rep_idx}/seed_0.json"
            fp.parent.mkdir(parents=True)
            json.dump(
                {
                    "metadata": {},
                    "entries": [{"offset": i, "mask": mask} for i in range(min(8, n_rows))],
                },
                fp.open("w"),
            )
            fp_paths.append(str(fp))

        fpp = {plane: fp_paths}
        rv = [r_p if p == plane else 1 for p in range(8)]

        from phase3_y0_evaluator import (
            compute_delivered_answer_with_degradation,
        )

        result = compute_delivered_answer_with_degradation(
            clean_planes=planes,
            fault_plan_paths=fpp,
            r_vector=rv,
            scale=scale,
            n_rows=n_rows,
            dataset=ds_name,
            policy="uniform_repair_fraction",
            allocation_r="|".join(str(r) for r in rv),
            segment_size=SEGMENT_SIZE,
        )

        pred = compute_ui_prediction(
            clean_planes=planes,
            fault_plan_paths=fpp,
            r_vector=rv,
            scale=scale,
            n_rows=n_rows,
            segment_size=SEGMENT_SIZE,
        )

        # Certified interval uses PREDICTED bound (U_i ceiling per Z1a contract).
        # The observed bound may not cover the signed shift when degraded bytes
        # are systematically shifted in one direction; the predicted bound always
        # covers it because U_i uses 255 per byte (max byte diff).
        bound_certified = pred["bound_width_predicted"]
        ct = contains_truth(
            result.clean_answer, result.delivered_answer, bound_certified
        )
        err = bound_width_prediction_err(
            result.bound_width, pred["bound_width_predicted"]
        )
        valid_ceil = pred["bound_width_predicted"] >= result.bound_width * (
            1.0 - 1e-12
        )

    note = (
        f"r_p={r_p}, detectable disagreement: "
        f"replicas have different faults (no majority); "
        f"ct uses predicted bound"
    )

    row = {
        "case": "degraded",
        "dataset": ds_name,
        "n_rows": str(n_rows),
        "scale": str(scale),
        "k": "8",
        "corrupt_plane": str(plane),
        "j_lt_k": "true",
        "r_p": str(r_p),
        "clean_answer": f"{result.clean_answer:.10e}",
        "delivered_answer": f"{result.delivered_answer:.10e}",
        "bound_width_observed": f"{result.bound_width:.10e}",
        "bound_width_predicted": f"{pred['bound_width_predicted']:.10e}",
        "bound_width_prediction_err": f"{err:.10e}",
        "bound_fault_free": f"{result.bound_width_fault_free:.10e}",
        "truncation_residual": "0",
        "measured_widen": f"{result.bound_width - result.bound_width_fault_free:.10e}",
        "predicted_widen": f"{pred['bound_width_predicted'] - result.bound_width_fault_free:.10e}",
        "contains_truth": str(ct).lower(),
        "valid_ceiling": str(valid_ceil).lower(),
        "segments_crc_hit": str(result.segments_crc_hit),
        "segments_degraded": str(result.segments_degraded),
        "segments_repaired": str(result.segments_repaired),
        "segments_unprotected": str(result.segments_unprotected),
        "notes": note,
    }
    writer.writerow(row)
    csv_file.flush()
    return row


def build_truncation_schema(k_max: int) -> list[tuple[int, int]]:
    schema = []
    for k in [1, 2, k_max - 1]:
        if k <= 0 or k > k_max:
            continue
        if k >= 1:
            schema.append((k, 0))
        if k >= 2:
            mid = k // 2
            if mid > 0 and mid < k - 1:
                schema.append((k, mid))
            schema.append((k, k - 1))
        if k < 8:
            schema.append((k, k))
        if k < 7:
            schema.append((k, 7))
    return schema


def main() -> None:
    datasets = [
        ("cesm_atm_cloud",
         Path("${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg4096"),
         24756224, 7),
        ("hurricane_u",
         Path("${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg4096"),
         0, 8),
    ]

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "z1d_login")
    results_dir = (
        Path("results/reliability_layer1/phase3/phase3z_z1")
        / f"job_{slurm_job_id}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "z1d_truncation_degraded_validation.csv"

    print("=" * 60)
    print("Z1-D: k-depth truncation + degraded-repair validation")
    print("=" * 60)

    summary: dict[str, Any] = {}
    all_rows: list[dict[str, str]] = []

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for ds_name, ds_path, ds_offset, k_max in datasets:
            print(f"\n--- {ds_name} (k_max={k_max}) ---")
            ds = load_subset(ds_path, SUBSET_ROWS, offset=ds_offset)
            print(f"  loaded {ds['n_rows']} rows, scale={ds['scale']}")

            ds_summary = {
                "cells_total": 0,
                "cells_passed": 0,
                "cells_failed": [],
                "truncation_passed": 0,
                "truncation_failed": [],
                "degraded_passed": 0,
                "degraded_failed": [],
            }

            # Case 1: Truncation
            schema = build_truncation_schema(k_max)
            for k, j in schema:
                ds_summary["cells_total"] += 1
                t0 = time.time()
                row = run_truncation_cell(
                    writer, f, ds_name, ds, k, j, k_max
                )
                elapsed = time.time() - t0
                all_rows.append(row)

                ct = row["contains_truth"] == "true"
                vc = row["valid_ceiling"] == "true"
                err_val = float(row["bound_width_prediction_err"])
                print(
                    f"  [trunc] k={k} j={j} j<k={row['j_lt_k']:5s} "
                    f"ct={ct} vc={vc} err={err_val:.4e} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
                if ct and vc:
                    ds_summary["truncation_passed"] += 1
                else:
                    ds_summary["truncation_failed"].append(
                        f"trunc/k{k}/j{j}"
                    )
                    ds_summary["cells_failed"].append(f"trunc/k{k}/j{j}")

            # Case 2: Degraded-repair
            for plane in [0, 1]:
                for r_p in [2, 3]:
                    ds_summary["cells_total"] += 1
                    t0 = time.time()
                    row = run_degraded_cell(
                        writer, f, ds_name, ds, plane, r_p
                    )
                    elapsed = time.time() - t0
                    all_rows.append(row)

                    ct = row["contains_truth"] == "true"
                    vc = row["valid_ceiling"] == "true"
                    status = "OK" if (ct and vc) else "FAIL"
                    err_val = float(row["bound_width_prediction_err"])
                    print(
                        f"  [degr] plane={plane} r_p={r_p} "
                        f"ct={ct} vc={vc} err={err_val:.4e} "
                        f"({elapsed:.1f}s)",
                        flush=True,
                    )
                    if ct and vc:
                        ds_summary["degraded_passed"] += 1
                    else:
                        ds_summary["degraded_failed"].append(
                            f"degr/plane{plane}/rp{r_p}"
                        )
                        ds_summary["cells_failed"].append(
                            f"degr/plane{plane}/rp{r_p}"
                        )

            summary[ds_name] = ds_summary
            print(
                f"  => {ds_summary['cells_total']} cells, "
                f"passed={ds_summary['truncation_passed'] + ds_summary['degraded_passed']}, "
                f"failed={len(ds_summary['cells_failed'])}"
            )

    print(f"\nCSV: {csv_path}")

    # Gate check
    print("\n--- Gate 1 (hard): contains_truth == 1.0 and valid_ceiling ---")
    all_ok = True
    for ds_name, s in summary.items():
        if s["cells_failed"]:
            print(f"  {ds_name}: FAIL — {s['cells_failed']}")
            all_ok = False
        else:
            total_pass = s["truncation_passed"] + s["degraded_passed"]
            total_cells = s["cells_total"]
            print(
                f"  {ds_name}: PASS ({total_pass}/{total_cells})"
            )
            k_max_for_ds = 7 if ds_name == "cesm_atm_cloud" else 8
            trunc_total = len(build_truncation_schema(k_max_for_ds))
            print(
                f"    truncation: {s['truncation_passed']}/{trunc_total} pass, "
                f"degraded: {s['degraded_passed']}/4 pass"
            )

    if all_ok:
        print("\n  Verdict: Z1D_SUPPORTED")
    else:
        print("\n  Verdict: CB_FAILS_CONTAINMENT")

    # No-double-count verification
    print("\n--- No-double-count verification ---")
    for row in all_rows:
        if row["case"] != "truncation":
            continue
        j_lt_k = row["j_lt_k"] == "true"
        predicted_widen = float(row["predicted_widen"])
        if not j_lt_k:
            if predicted_widen > 0:
                print(
                    f"  WARNING: j >= k but predicted_widen > 0: "
                    f"{row['dataset']} k={row['k']} j={row['corrupt_plane']} "
                    f"widen={predicted_widen:.6e}"
                )
            else:
                print(
                    f"  OK: j >= k → predicted_widen=0 "
                    f"({row['dataset']} k={row['k']} j={row['corrupt_plane']})"
                )
        else:
            if predicted_widen > 0:
                print(
                    f"  OK: j < k → predicted_widen={predicted_widen:.6e} > 0 "
                    f"({row['dataset']} k={row['k']} j={row['corrupt_plane']})"
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
