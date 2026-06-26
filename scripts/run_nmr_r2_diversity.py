"""NMR-R2: Diversity Model — logical separation vs simple replication under fault correlation.

Simple replication: same fault hits all r replicas with prob ρ (correlated).
Logical diversity: each replica gets independently rolled faults (separate buffers).
Temporal diversity: not testable in CPU model (requires actual HBM read-path transients).
"""

from __future__ import annotations

import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))


def make_replicas(plane_bytes: bytes, r: int) -> list[bytearray]:
    return [bytearray(plane_bytes) for _ in range(r)]


def byte_majority_vote(replicas: list[bytearray]) -> bytearray:
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


def count_mismatches(voted: bytearray, clean: bytes) -> int:
    return sum(1 for i in range(len(voted)) if voted[i] != clean[i])


def run_r2(n_rows: int = 500000, seeds: list[int] | None = None,
           rho_values: list[float] | None = None) -> list[dict[str, Any]]:
    if seeds is None:
        seeds = list(range(5))
    if rho_values is None:
        rho_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    rng_global = random.Random(42)
    clean = bytes(rng_global.randint(0, 255) for _ in range(n_rows))
    r = 3
    fault_rate = 1e-4  # 1 fault per 10K bytes avg

    rows: list[dict[str, Any]] = []

    for rho in rho_values:
        for seed in seeds:
            rng = random.Random(seed + int(rho * 100))

            # ── Strategy 1: Simple replication — correlated faults ──
            # With prob ρ, a fault hits ALL replicas (shared domain).
            # With prob 1-ρ, it hits exactly 1 replica (independent domain).
            reps_simple = make_replicas(clean, r)
            n_faults_simple = 0
            for i in range(n_rows):
                if rng.random() < fault_rate:
                    n_faults_simple += 1
                    if rng.random() < rho:
                        for j in range(r):
                            reps_simple[j][i] ^= rng.randint(1, 255)
                    else:
                        j = rng.randint(0, r - 1)
                        reps_simple[j][i] ^= rng.randint(1, 255)
            voted_simple = byte_majority_vote(reps_simple)
            mm_simple = count_mismatches(voted_simple, clean)

            # ── Strategy 2: Logical diversity — independent faults ──
            # Each replica gets independent fault rolls (separate buffers).
            reps_div = make_replicas(clean, r)
            n_faults_div = 0
            for j in range(r):
                rng_j = random.Random(seed + int(rho * 100) + j * 1000)
                for i in range(n_rows):
                    if rng_j.random() < fault_rate:
                        n_faults_div += 1
                        reps_div[j][i] ^= rng_j.randint(1, 255)
            voted_div = byte_majority_vote(reps_div)
            mm_div = count_mismatches(voted_div, clean)

            rows.append({
                "rho": f"{rho:.2f}",
                "seed": str(seed),
                "r": r,
                "fault_rate": f"{fault_rate:.0e}",
                "n_faults_simple": n_faults_simple,
                "simple_mismatches": mm_simple,
                "simple_escaped": str(mm_simple > 0),
                "n_faults_diverse": n_faults_div,
                "logical_div_mismatches": mm_div,
                "logical_div_escaped": str(mm_div > 0),
            })

    return rows


def main() -> None:
    jid = os.environ.get("SLURM_JOB_ID", "cpu_r2")
    out_root = Path(f"results/reliability_layer1/phase3/nmr_rescue_r2/job_{jid}")
    out_root.mkdir(parents=True, exist_ok=True)

    rows = run_r2(n_rows=500000, seeds=list(range(5)))

    fields = [
        "rho", "seed", "r", "fault_rate",
        "n_faults_simple", "simple_mismatches", "simple_escaped",
        "n_faults_diverse", "logical_div_mismatches", "logical_div_escaped",
    ]
    csv_path = out_root / "nmr_r2_diversity.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")

    # Summary
    print(f"\n{'ρ':>5s} {'SimpleEsc':>10s} {'DivEsc':>10s} {'DivHelps':>10s}")
    print("-" * 40)
    by_rho = defaultdict(list)
    for r in rows:
        by_rho[r["rho"]].append(r)

    diversity_adds = False
    for rho_str in sorted(by_rho):
        g = by_rho[rho_str]
        n = len(g)
        se = sum(1 for r in g if r["simple_escaped"] == "True")
        de = sum(1 for r in g if r["logical_div_escaped"] == "True")
        div_helps = de < se
        if div_helps:
            diversity_adds = True
        print(f"{rho_str:>5s} {se:>3d}/{n:<3d}      {de:>3d}/{n:<3d}      {'✅' if div_helps else '❌'}")

    print(f"\n  Temporal diversity: NOT tested — requires actual HBM read-path transient noise model")
    print(f"  (CPU model cannot distinguish transient from persistent corruption)")
    print(f"\n  Diversity adds value over replication: {diversity_adds}")
    if diversity_adds:
        print(f"  → NMR_R2_DIVERSITY_ADDS_VALUE (logical diversity)")
    else:
        print(f"  → NMR_R2_REPLICATION_ENOUGH")


if __name__ == "__main__":
    main()
