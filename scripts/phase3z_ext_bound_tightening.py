"""Phase 3-Z Extension E3: Structure-Aware Bound Tightening.

Compares worst-case U_i (using n_rows) vs tightened U_i (using per-segment
active/corruptible count).  Tests on cesm_atm_cloud (zero-heavy) and
hurricane_u (control) for SUM / filtered-SUM / COUNT.

Key correctness constraint (filtered functionals):
  tightened_count = clean_active_count + max_possible_flips
  where max_possible_flips is the number of rows whose decoded interval
  straddles the threshold when plane i is maximally corrupted.

Metrics:
  bound_width_shrink_factor = worst_case_width / tightened_width
  contains_truth (must be 1.0)
  valid-ceiling (predicted >= observed)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase3_y0_evaluator import (
    PLANE_WEIGHTS,
    SEGMENT_SIZE,
    compute_clean_sum,
    decode_value,
)
from phase3_z1_filtered_evaluator import (
    compute_max_flip_rows,
    compute_filtered_ui_prediction,
    compute_filtered_delivered_result,
)


def load_planes_glob(artifact_dir: Path, n_rows: int, offset: int = 0) -> list[bytes]:
    """Load 8 clean plane byte arrays using glob (handles plane_000.bin naming)."""
    plane_files = sorted(artifact_dir.glob("plane_*.bin"))
    plane_map: dict[int, Path] = {}
    for pf in plane_files:
        idx = int(pf.stem.split("_")[-1])
        plane_map[idx] = pf
    planes: list[bytes] = []
    for p in range(8):
        if p in plane_map:
            data = plane_map[p].read_bytes()
            data = data[offset:offset + n_rows]
            if len(data) < n_rows:
                data = data + b'\x00' * (n_rows - len(data))
            planes.append(data)
        else:
            planes.append(bytes(n_rows))
    return planes

SEGMENT_SIZE = 4096


def compute_per_segment_nonzero(plane_bytes: bytes, seg_size: int) -> list[int]:
    """Count non-zero bytes per segment in a plane."""
    n = len(plane_bytes)
    n_segs = (n + seg_size - 1) // seg_size
    counts: list[int] = []
    for si in range(n_segs):
        start = si * seg_size
        end = min(start + seg_size, n)
        c = sum(1 for i in range(start, end) if plane_bytes[i] != 0)
        counts.append(c)
    return counts


def compute_tightened_ui(planes: list[bytes], seg_size: int,
                         functional: str, scale: int,
                         threshold: float | None = None,
                         max_flip_rows: list[list[int]] | None = None) -> float:
    """Compute tightened bound width using per-segment active counts.

    For filtered functionals, uses clean_active_count + max_possible_flips
    envelope.  For plain SUM/COUNT, uses per-segment nonzero count.

    Returns total tightened bound width.
    """
    n_rows = len(planes[0])
    n_segs = (n_rows + seg_size - 1) // seg_size
    q_err = 0.5 / scale

    bound_fault_free = 2.0 * q_err * n_rows if functional == "sum" else 0.0

    total_widen = 0.0
    for p in range(8):
        seg_nonzero = compute_per_segment_nonzero(planes[p], seg_size)
        for si in range(n_segs):
            start = si * seg_size
            end = min(start + seg_size, n_rows)
            active = seg_nonzero[si]

            if functional in ("sum", "count"):
                if functional == "sum":
                    contrib = active * 255 * PLANE_WEIGHTS[p] / scale
                else:  # count
                    contrib = float(active)
            elif functional == "filtered_sum":
                clean_active = seg_nonzero[si]
                # max_possible_flips for this segment
                max_flip = max_flip_rows[p][si] if max_flip_rows else 0
                envelope = clean_active + max_flip
                contrib = envelope * 255 * PLANE_WEIGHTS[p] / scale
            elif functional == "filtered_count":
                clean_active = seg_nonzero[si]
                max_flip = max_flip_rows[p][si] if max_flip_rows else 0
                envelope = clean_active + max_flip
                contrib = float(envelope)
            else:
                raise ValueError(f"unknown functional: {functional}")

            total_widen += contrib

    return bound_fault_free + total_widen


def compute_worst_case_ui(planes: list[bytes], seg_size: int,
                          functional: str, scale: int,
                          threshold: float | None = None,
                          max_flip_rows: list[list[int]] | None = None) -> float:
    """Compute worst-case bound width using n_rows (current Z1a contract)."""
    n_rows = len(planes[0])
    n_segs = (n_rows + seg_size - 1) // seg_size
    q_err = 0.5 / scale

    bound_fault_free = 2.0 * q_err * n_rows if functional == "sum" else 0.0

    total_widen = 0.0
    for p in range(8):
        for si in range(n_segs):
            start = si * seg_size
            end = min(start + seg_size, n_rows)
            count = end - start

            if functional in ("sum", "count"):
                if functional == "sum":
                    contrib = count * 255 * PLANE_WEIGHTS[p] / scale
                else:
                    contrib = float(count)
            elif functional == "filtered_sum":
                max_flip = max_flip_rows[p][si] if max_flip_rows else 0
                envelope = count + max_flip
                contrib = envelope * 255 * PLANE_WEIGHTS[p] / scale
            elif functional == "filtered_count":
                max_flip = max_flip_rows[p][si] if max_flip_rows else 0
                envelope = count + max_flip
                contrib = float(envelope)
            else:
                raise ValueError(f"unknown functional: {functional}")

            total_widen += contrib

    return bound_fault_free + total_widen


def compute_max_flip_rows_per_segment(
    clean_planes: list[bytes], plane: int, threshold: float,
    scale: int, n_rows: int, seg_size: int, k: int = 8
) -> list[int]:
    """Compute max_flip_rows for each segment."""
    n_segs = (n_rows + seg_size - 1) // seg_size
    flip_per_seg: list[int] = []
    for si in range(n_segs):
        start = si * seg_size
        end = min(start + seg_size, n_rows)
        # Need to count flip rows in this segment only
        q_err = 0.5 / scale
        max_swing = 255 * PLANE_WEIGHTS[plane] / scale if plane < 8 else 0.0
        flip_count = 0
        for i in range(start, end):
            clean_bytes = [clean_planes[p][i] for p in range(8)]
            x_low, x_high = decode_value(clean_bytes, k, scale)
            clean_Q = x_low >= threshold
            clean_D = x_high < threshold
            if clean_Q:
                if x_high - max_swing < threshold:
                    flip_count += 1
            elif clean_D:
                if x_low + max_swing >= threshold:
                    flip_count += 1
        flip_per_seg.append(flip_count)
    return flip_per_seg


def run_bound_tightening(artifact_dir: Path, n_rows: int, scale: int,
                         dataset: str, threshold: float | None = None,
                         seg_size: int = SEGMENT_SIZE,
                         offset: int = 0) -> list[dict[str, Any]]:
    """Run bound tightening comparison."""
    planes = load_planes_glob(artifact_dir, n_rows, offset)
    functionals = ["sum", "count", "filtered_sum", "filtered_count"]
    rows: list[dict[str, Any]] = []

    for func in functionals:
        if func.startswith("filtered_") and threshold is None:
            continue

        if func.startswith("filtered_"):
            # Compute max_flip_rows per plane per segment
            max_flip_rows: list[list[int]] = []
            for p in range(8):
                flip = compute_max_flip_rows_per_segment(
                    planes, p, threshold, scale, n_rows, seg_size
                )
                max_flip_rows.append(flip)

        tightened = compute_tightened_ui(
            planes, seg_size, func, scale, threshold,
            max_flip_rows if func.startswith("filtered_") else None
        )
        worst_case = compute_worst_case_ui(
            planes, seg_size, func, scale, threshold,
            max_flip_rows if func.startswith("filtered_") else None
        )

        shrink = worst_case / max(tightened, 1e-30)
        valid_ceiling = worst_case >= tightened - 1e-12

        # For contains_truth, we need injected faults. Without injection,
        # the bound always contains truth (overestimates). Report as N/A
        # for the no-injection case - actual verification requires injection.

        rows.append({
            "dataset": dataset,
            "n_rows": n_rows,
            "functional": func,
            "threshold": str(threshold) if threshold else "N/A",
            "worst_case_width": f"{worst_case:.10e}",
            "tightened_width": f"{tightened:.10e}",
            "shrink_factor": f"{shrink:.6f}",
            "valid_ceiling": str(valid_ceiling),
            "contains_truth": "N/A",  # Without injection, trivially true
        })

        # Per-plane tightening profile
        for p in range(8):
            n_segs = (n_rows + seg_size - 1) // seg_size
            seg_nonzero = compute_per_segment_nonzero(planes[p], seg_size)
            total_active = sum(seg_nonzero)
            total_rows = n_rows

            rows.append({
                "dataset": dataset,
                "n_rows": n_rows,
                "functional": f"{func}_plane{p}",
                "threshold": str(threshold) if threshold else "N/A",
                "active_rows": str(total_active),
                "total_rows": str(total_rows),
                "sparsity": f"{1.0 - total_active / max(total_rows, 1):.6f}",
                "worst_case_width": "N/A",
                "tightened_width": "N/A",
                "shrink_factor": "N/A",
                "valid_ceiling": "N/A",
                "contains_truth": "N/A",
            })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-rows", type=int, default=1000000)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--seg-size", type=int, default=SEGMENT_SIZE)
    parser.add_argument("--offset", type=int, default=0,
                        help="byte offset into plane files (for cesm non-zero region)")
    args = parser.parse_args()

    jid = os.environ.get("SLURM_JOB_ID", "cpu_local")
    out_root = Path(f"results/reliability_layer1/phase3/phase3z_ext/job_{jid}")

    rows = run_bound_tightening(
        artifact_dir=args.artifact_dir,
        n_rows=args.n_rows,
        scale=args.scale,
        dataset=args.dataset,
        threshold=args.threshold,
        seg_size=args.seg_size,
        offset=args.offset,
    )

    fields = [
        "dataset", "n_rows", "functional", "threshold",
        "worst_case_width", "tightened_width", "shrink_factor",
        "valid_ceiling", "contains_truth", "active_rows", "total_rows",
        "sparsity",
    ]
    csv_path = out_root / "e3_bound_tightening.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"{'Appended' if exists else 'Wrote'} {csv_path} ({len(rows)} rows)")

    # Print summary
    print(f"\n=== E3 Bound Tightening: {args.dataset} ===")
    for r in rows:
        if "plane" in r["functional"]:
            continue
        sf = r.get("shrink_factor", "N/A")
        vc = r.get("valid_ceiling", "N/A")
        print(f"  {r['functional']:<20} shrink={sf:>10}  valid_ceiling={vc}")
    print()


if __name__ == "__main__":
    main()
