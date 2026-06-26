"""NMR-R1: Graded vs Storage-Matched Uniform Comparison.

Five metrics (no point accuracy):
  escape_rate
  certified_availability
  expected_bound_width
  S0_coverage
  containment_per_replica

5% tie tolerance. Graded must beat or match uniform-ish on >=3 of 5.
"""

from __future__ import annotations

import csv
import itertools
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase3_y0_evaluator import PLANE_WEIGHTS

PLANE_COUNT = 8
SEGMENT_SIZE = 4096


def make_replicas(plane_bytes: bytes, r: int) -> list[bytearray]:
    return [bytearray(plane_bytes) for _ in range(r)]


def byte_majority_vote(replicas: list[bytearray]) -> bytearray:
    r = len(replicas)
    n = len(replicas[0])
    result = bytearray(n)
    for i in range(n):
        counts: dict[int, int] = {}
        for rep in replicas:
            b = rep[i]
            counts[b] = counts.get(b, 0) + 1
        majority = max(counts, key=lambda k: (counts[k], k))
        result[i] = majority
    return result


def apply_fault(plane: bytearray, offset: int, mode: str, seed: int) -> None:
    rng = random.Random(seed + offset)
    if mode == "single_flip":
        plane[offset] ^= rng.randint(1, 255)
    elif mode == "saturating":
        plane[offset] = 0xFF
    elif mode == "adversarial_cancel":
        if offset + 4 < len(plane):
            plane[offset] ^= 1
            plane[offset + 4] ^= 1


def compute_metrics(results: list[dict]) -> dict[str, float]:
    n = len(results) if results else 1

    escapes = sum(1 for r in results if not r["recovered"])
    escape_rate = escapes / n

    available = sum(1 for r in results if r["recovered"])
    certified_availability = available / n

    widths = [r["bound_width"] for r in results]
    expected_bound_width = sum(widths) / n

    s0_covered = sum(1 for r in results if not r.get("s0_unprotected", False))
    s0_coverage = s0_covered / n

    # containment_per_replica = (1 - escape_rate) / total_replicas
    # Computed per-config, not here

    return {
        "escape_rate": escape_rate,
        "certified_availability": certified_availability,
        "expected_bound_width": expected_bound_width,
        "s0_coverage": s0_coverage,
    }


def run_r1_comparison(n_rows: int = 1000000, seeds: list[int] | None = None) -> list[dict[str, Any]]:
    if seeds is None:
        seeds = list(range(5))  # 5 seeds for statistical power

    # Generate synthetic plane data
    rng_global = random.Random(42)
    clean_plane = bytes(rng_global.randint(0, 255) for _ in range(n_rows))

    # R-vector configs
    r_configs = [
        ("graded_B11",   [3, 2, 1, 1, 1, 1, 1, 1]),   # 11 replicas
        ("uniformish_B11", [2, 1, 2, 1, 2, 1, 1, 1]),   # best uniform-ish B=11
        ("random_B11_a",   [1, 2, 1, 3, 1, 1, 1, 1]),   # random B=11
        ("random_B11_b",   [2, 1, 1, 1, 1, 2, 2, 1]),   # another random B=11
    ]

    # Optional extra-budget reference
    r_extra = [("uniform_B16", [2, 2, 2, 2, 2, 2, 2, 2])]  # 16 replicas

    fault_modes = ["single_flip", "saturating", "adversarial_cancel"]

    rows: list[dict[str, Any]] = []

    for r_label, r_vec in r_configs:
        total_b = sum(r_vec)
        for fm in fault_modes:
            for seed in seeds:
                # Inject fault into exactly ONE replica per test
                # (logical independent-fault model from R0)
                # Choose which replica to corrupt based on seed
                fault_replica = seed % 3  # cycle through replicas 0,1,2

                # Create replicas per r-vector
                replicas = []
                for p in range(PLANE_COUNT):
                    rep_count = r_vec[p] if p < len(r_vec) else 1
                    rep = make_replicas(clean_plane, rep_count)
                    # Inject fault into fault_replica within this plane's replicas
                    if fault_replica < rep_count:
                        offset = seed * 64 % n_rows
                        apply_fault(rep[fault_replica], offset, fm, seed)
                    replicas.append(rep)

                # Majority vote per plane
                voted_planes: list[bytearray] = []
                for p in range(PLANE_COUNT):
                    voted = byte_majority_vote(replicas[p])
                    voted_planes.append(voted)

                # Compute metrics
                # Determine if S0 (plane 0) is unprotected
                s0_unprotected = r_vec[0] < 2 if len(r_vec) > 0 else True

                # Recovery check per plane: does voted plane match clean?
                plane_mismatches = []
                for p in range(PLANE_COUNT):
                    mism = sum(1 for i in range(n_rows) if voted_planes[p][i] != clean_plane[i])
                    plane_mismatches.append(mism)

                total_mismatches = sum(plane_mismatches)
                recovered = (total_mismatches == 0)
                escape_rate_this = 0.0 if recovered else 1.0

                # Bound width: estimated U_i for the most significant corrupted plane
                corrupted_planes = [p for p in range(PLANE_COUNT) if plane_mismatches[p] > 0]
                bound_width = 0.0
                for p in corrupted_planes:
                    bound_width += 255.0 * PLANE_WEIGHTS[p] * r_vec[p] / max(n_rows, 1)

                rows.append({
                    "config": r_label,
                    "r_vector": "|".join(str(r) for r in r_vec),
                    "total_replicas": total_b,
                    "fault_mode": fm,
                    "seed": str(seed),
                    "n_rows": n_rows,
                    "total_mismatches": total_mismatches,
                    "recovered": str(recovered),
                    "escape_rate_cell": f"{escape_rate_this:.6f}",
                    "bound_width": f"{bound_width:.10e}",
                    "s0_unprotected": str(s0_unprotected),
                    "corrupted_planes": str(corrupted_planes),
                })

    return rows


def summarize(rows: list[dict]) -> None:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["config"], r["r_vector"], int(r["total_replicas"]))].append(r)

    print(f"\n{'Config':<25s} {'B':>3s} {'EscapeRate':>12s} {'CertAvail':>12s} "
          f"{'ExpBndWdth':>14s} {'S0Cover':>10s}")
    print("-" * 80)

    all_metrics: dict[str, dict[str, float]] = {}

    for key in sorted(groups):
        label, rv_str, total_b = key
        g = groups[key]
        n = len(g)
        escapes = sum(1 for r in g if r["recovered"] == "False")
        escape_rate = escapes / max(n, 1)
        available = sum(1 for r in g if r["recovered"] == "True")
        cert_avail = available / max(n, 1)
        widths = [float(r["bound_width"]) for r in g]
        exp_bw = sum(widths) / max(n, 1)
        s0_ok = sum(1 for r in g if r["s0_unprotected"] == "False")
        s0_cov = s0_ok / max(n, 1)
        cpr = (1.0 - escape_rate) / total_b if total_b > 0 else 0.0

        all_metrics[label] = {
            "escape_rate": escape_rate,
            "certified_availability": cert_avail,
            "expected_bound_width": exp_bw,
            "s0_coverage": s0_cov,
            "containment_per_replica": cpr,
        }

        print(f"{label:<25s} {total_b:>3d} {escape_rate:>12.6f} {cert_avail:>12.6f} "
              f"{exp_bw:>14.6e} {s0_cov:>10.6f}  (cpr={cpr:.6f})")

    # Comparison: graded vs uniform-ish
    if "graded_B11" in all_metrics and "uniformish_B11" in all_metrics:
        g = all_metrics["graded_B11"]
        u = all_metrics["uniformish_B11"]
        tiers = ["escape_rate", "certified_availability", "expected_bound_width", "s0_coverage", "containment_per_replica"]
        # Direction: lower is better for escape_rate, expected_bound_width;
        # higher is better for certified_availability, s0_coverage, containment_per_replica
        directions = {
            "escape_rate": "lower",
            "certified_availability": "higher",
            "expected_bound_width": "lower",
            "s0_coverage": "higher",
            "containment_per_replica": "higher",
        }
        graded_wins = 0
        graded_loses = 0
        ties = 0

        print(f"\n{'':>25s} {'Graded':>12s} {'Uniform':>12s} {'Winner':>10s}")
        print("-" * 65)
        for m in tiers:
            gv = g[m]
            uv = u[m]
            dir = directions[m]
            # 5% tie tolerance
            if dir == "lower":
                threshold = gv * 0.95  # graded must be at least 5% lower to win
                if gv < uv * 0.95:
                    winner = "graded"
                    graded_wins += 1
                elif uv < gv * 0.95:
                    winner = "uniform"
                    graded_loses += 1
                else:
                    winner = "tie"
                    ties += 1
            else:
                if gv > uv * 1.05:
                    winner = "graded"
                    graded_wins += 1
                elif uv > gv * 1.05:
                    winner = "uniform"
                    graded_loses += 1
                else:
                    winner = "tie"
                    ties += 1
            print(f"{m:<25s} {gv:>12.6f} {uv:>12.6f} {winner:>10s}")

        wins_plus_ties = graded_wins + ties
        print(f"\n  Graded wins: {graded_wins}/{len(tiers)}  losses: {graded_loses}/{len(tiers)}  ties: {ties}")
        print(f"  Graded matches or beats uniform: {wins_plus_ties}/{len(tiers)}")
        if graded_loses >= 3:
            print(f"  → NMR_R1_NO_ADVANTAGE (graded worse on {graded_loses} metrics, stop)")
        elif wins_plus_ties >= 3:
            print(f"  → NMR_R1_GRADED_ADVANTAGE_DEMONSTRATED (matches or beats on >=3/5 metrics)")
        else:
            print(f"  → NMR_R1_INDISTINGUISHABLE")


def main() -> None:
    jid = os.environ.get("SLURM_JOB_ID", "cpu_r1")
    out_root = Path(f"results/reliability_layer1/phase3/nmr_rescue_r1/job_{jid}")
    out_root.mkdir(parents=True, exist_ok=True)

    rows = run_r1_comparison(n_rows=500000, seeds=list(range(5)))

    fields = [
        "config", "r_vector", "total_replicas", "fault_mode", "seed",
        "n_rows", "total_mismatches", "recovered", "escape_rate_cell",
        "bound_width", "s0_unprotected", "corrupted_planes",
    ]
    csv_path = out_root / "nmr_r1_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")

    summarize(rows)


if __name__ == "__main__":
    main()
