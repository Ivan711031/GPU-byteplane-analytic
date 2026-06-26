"""Phase 3-Z G4: MAX / top-k Certified-Bound Validation.

Validates the Z1a §4 closed-form U_i bound for MAX and top-k on real
datasets with single-plane byte faults, including identity-shift.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase3_y0_evaluator import PLANE_WEIGHTS, decode_value


def load_planes_glob(artifact_dir: Path, n_rows: int) -> list[bytes]:
    plane_files = sorted(artifact_dir.glob("plane_*.bin"))
    plane_map: dict[int, Path] = {}
    for pf in plane_files:
        idx = int(pf.stem.split("_")[-1])
        plane_map[idx] = pf
    planes: list[bytes] = []
    for p in range(8):
        if p in plane_map:
            data = plane_map[p].read_bytes()[:n_rows]
            if len(data) < n_rows:
                data = data + b'\x00' * (n_rows - len(data))
            planes.append(data)
        else:
            planes.append(bytes(n_rows))
    return planes


def find_max_row(planes: list[bytes], k: int, scale: int) -> tuple[int, float, list[int]]:
    """Find the row with maximum decoded value (using k planes)."""
    best_val = -1e100
    best_row = 0
    best_bytes: list[int] = []
    n = len(planes[0])
    for i in range(n):
        bytes_i = [planes[p][i] for p in range(k)]
        val, _ = decode_value(bytes_i, k, scale)
        if val > best_val:
            best_val = val
            best_row = i
            best_bytes = bytes_i
    return best_row, best_val, best_bytes


def run_g4_validation(artifact_dir: Path, dataset: str, n_rows: int,
                      scale: int, k: int) -> list[dict[str, Any]]:
    planes = load_planes_glob(artifact_dir, n_rows)
    rows: list[dict[str, Any]] = []

    # Find clean MAX and top-10
    max_row, max_val, _ = find_max_row(planes, k, scale)

    # For each plane, inject fault at both the max row and a non-max row
    test_planes = [0, 1, min(7, k - 1), k - 1]

    for plane in test_planes:
        for test_type in ["direct", "identity_shift"]:
            test_planes_copy = [list(planes[p]) for p in range(k)]

            if test_type == "direct":
                # Fault the max row's plane byte: XOR 0xFF
                orig = test_planes_copy[plane][max_row]
                test_planes_copy[plane][max_row] = orig ^ 0xFF
            else:
                # Identity shift: fault a different row to promote it
                target = (max_row + 1000) % n_rows
                if target == max_row:
                    target = (target + 1) % n_rows
                orig = test_planes_copy[plane][target]
                test_planes_copy[plane][target] = orig ^ 0xFF

            # Compute faulted max
            faulted_bytes = [bytes(test_planes_copy[p]) for p in range(k)]
            faulted_max_row, faulted_max_val, _ = find_max_row(faulted_bytes, k, scale)

            # U_i bound per Z1a §4: MAX = 255 * weight[plane] / scale
            u_i = 255.0 * PLANE_WEIGHTS[plane] / scale

            # Certified interval
            lo = faulted_max_val - u_i
            hi = faulted_max_val + u_i

            # For top-1 (MAX): active_count = 1
            contains_truth = lo <= max_val <= hi
            valid_ceiling = u_i >= abs(faulted_max_val - max_val)

            rows.append({
                "dataset": dataset, "n_rows": n_rows,
                "functional": f"MAX_plane{plane}", "test": test_type,
                "plane": plane, "k": k,
                "clean_max_row": str(max_row), "clean_max_val": f"{max_val:.10e}",
                "faulted_max_row": str(faulted_max_row),
                "faulted_max_val": f"{faulted_max_val:.10e}",
                "u_i": f"{u_i:.10e}", "lo": f"{lo:.10e}", "hi": f"{hi:.10e}",
                "contains_truth": str(contains_truth),
                "valid_ceiling": str(valid_ceiling),
            })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-rows", type=int, default=1000000)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    jid = os.environ.get("SLURM_JOB_ID", "cpu_local")
    out_root = Path(f"results/reliability_layer1/phase3/phase3z_ext/job_{jid}")

    rows = run_g4_validation(
        artifact_dir=args.artifact_dir,
        dataset=args.dataset,
        n_rows=args.n_rows,
        scale=args.scale,
        k=args.k,
    )

    fields = [
        "dataset", "n_rows", "functional", "test", "plane", "k",
        "clean_max_row", "clean_max_val", "faulted_max_row", "faulted_max_val",
        "u_i", "lo", "hi", "contains_truth", "valid_ceiling",
    ]
    csv_path = out_root / "g4_max_cb.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")

    print(f"\n=== G4 MAX CB: {args.dataset} ===")
    for r in rows:
        ct = r["contains_truth"]
        vc = r["valid_ceiling"]
        icon = "✅" if ct == "True" and vc == "True" else "❌"
        print(f"  plane={r['plane']} {r['test']:<18s} contains={ct} ceiling={vc} {icon}")


if __name__ == "__main__":
    main()
