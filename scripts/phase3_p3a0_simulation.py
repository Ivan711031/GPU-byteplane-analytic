"""P3-A0 analytical win-regime check: synthetic simulation.

Evaluates whether the reactive significance-aware mechanism (B4) can
simultaneously outperform no-verification (B0), full-fallback (B1), and
uniform-active-plane verification (B2) in a non-empty, plausibly-occupied
parameter regime.

Outputs CSV under results/phase3_p3a0/ for post-processing.
"""

import csv
import itertools
import math
import os
import sys

RESULTS_DIR = "results/phase3_p3a0"

# --- active-delta-global parameters (active_byte_len = 6) ---
PLANE_WEIGHTS = [1 << (8 * (5 - p)) for p in range(6)]  # [2^40, 2^32, 2^24, 2^16, 2^8, 2^0]
MAX_BYTE = 255
ACTIVE_PLANES = 6

# Normalized cost model (all costs relative to T_query = 1.0)
C_VERIFY = 0.15       # checksum all active planes (per segment)
C_ABSORB = 0.00       # R1: widen bound, no material cost beyond verification
C_R3 = 0.05           # R3: scan deeper (cheap, CPU/GPU-local)
C_REPAIR = 3.0        # R4/R5: recompute/fallback (per segment)
C_UNCERTIFIED = 0.05  # R6: emit UNCERTIFIED (no repair cost, but answer is lost)


def max_fault_error(plane: int, suspect_count: int) -> int:
    """Worst-case E_fault for suspect_count bytes in a given plane."""
    return suspect_count * MAX_BYTE * PLANE_WEIGHTS[plane]


def simulate_segment(
    fault_plane: int,
    suspect_count: int,
    epsilon: float,
    k_depth: int,
    segment_rows: int,
) -> dict:
    """Simulate reactive decisions for a single segment given a fault scenario.

    Returns cost components and certification outcome for each baseline.
    """
    err = max_fault_error(fault_plane, suspect_count)
    is_high_sig = fault_plane <= 1       # P0/P1: always escalate
    is_mid_sig = fault_plane == 2        # P2: borderline
    is_low_sig = fault_plane >= 3        # P3-P5: absorb candidate

    # --- B0: no verification ---
    # Zero verification cost. Answer has full uncertainty. No certification.
    b0_cost = 1.0  # T_progressive_query only (normalized)

    # --- B1: full fallback ---
    b1_cost = 1.0 + C_VERIFY + C_REPAIR

    # --- B2: uniform verification, always repair on fault ---
    b2_cost = 1.0 + C_VERIFY
    if suspect_count > 0:
        b2_cost += C_REPAIR

    # --- B4: reactive significance-aware ---
    b4_cost = 1.0 + C_VERIFY  # baseline: checksum all active planes
    b4_certified = True
    b4_widened = False
    b4_fallback = False
    b4_absorbed = False
    b4_r3_used = False

    if suspect_count > 0:
        if is_high_sig:
            # P0/P1: must escalate
            b4_cost += C_REPAIR
            b4_certified = True   # repair restores certification
            b4_fallback = True
        elif is_mid_sig or is_low_sig:
            # Can we absorb?
            if err <= epsilon:
                # R1: absorb (widen bound)
                b4_absorbed = True
                b4_widened = True
                # R3 might also be used to tighten U_depth before checking
                if k_depth < ACTIVE_PLANES and is_mid_sig:
                    b4_cost += C_R3
                    b4_r3_used = True
            else:
                # Cannot absorb — do R3 if it helps, then fallback
                if is_mid_sig and k_depth < ACTIVE_PLANES:
                    b4_cost += C_R3
                    b4_r3_used = True
                    b4_cost += C_REPAIR
                    b4_fallback = True
                else:
                    b4_cost += C_REPAIR
                    b4_fallback = True
        # P6/P7 zero-invariant case handled by the same logic (weight = 0 or negligible)
    else:
        # No fault — B4 stays at B2 cost level but is certified
        pass

    return {
        "epsilon": epsilon,
        "fault_plane": fault_plane,
        "suspect_count": suspect_count,
        "k_depth": k_depth,
        "segment_rows": segment_rows,
        "fault_error": err,
        "absorbable": err <= epsilon,
        "b0_cost": b0_cost,
        "b1_cost": b1_cost,
        "b2_cost": b2_cost,
        "b4_cost": b4_cost,
        "b4_certified": b4_certified,
        "b4_widened": b4_widened,
        "b4_fallback": b4_fallback,
        "b4_absorbed": b4_absorbed,
        "b4_r3_used": b4_r3_used,
        "b4_vs_b2_savings": b2_cost - b4_cost,
        "b4_vs_b0_premium": b4_cost - b0_cost,
        "b4_vs_b1_savings": b1_cost - b4_cost,
    }


def run_sweep():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Sweep parameters
    epsilons = [1e2, 1e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7,
                1e8, 1e9, 1e10, 1e11, 1e12, 1e14]
    planes = list(range(ACTIVE_PLANES))
    suspect_range = [1, 5, 10, 50, 100, 500]
    k_range = [2, 3, 4, 5, 6]
    segment_sizes = [512, 1024, 4096]

    rows = []
    for eps in epsilons:
        for plane in planes:
            for sc in suspect_range:
                if sc > 4096:
                    continue
                for k in k_range:
                    for seg in segment_sizes:
                        r = simulate_segment(plane, sc, eps, k, seg)
                        rows.append(r)

    path = os.path.join(RESULTS_DIR, "sweep_cells.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} cells to {path}")

    # --- Aggregation: find win regime ---
    win_cells = [r for r in rows
                 if r["b4_vs_b2_savings"] > 0
                 and r["b4_vs_b0_premium"] > 0
                 and r["b4_vs_b1_savings"] > 0
                 and r["b4_certified"]]

    # Also track near-win (B4 == B2 but certified, cheaper than B1)
    tie_cells = [r for r in rows
                 if r["b4_vs_b2_savings"] >= -0.01
                 and r["b4_vs_b0_premium"] > 0
                 and r["b4_vs_b1_savings"] > 0
                 and r["b4_certified"]]

    print(f"\n=== Win-regime analysis ===")
    print(f"Total cells: {len(rows)}")
    print(f"Strict win cells (B4 < B2, B4 < B1, B4 > B0, certified): {len(win_cells)} ({100*len(win_cells)/len(rows):.1f}%)")
    print(f"Tie+win cells: {len(tie_cells)} ({100*len(tie_cells)/len(rows):.1f}%)")

    # Write aggregated summaries
    _write_summary(rows, win_cells, "summary.csv")

    return win_cells, rows


def _write_summary(all_rows, win_rows, filename):
    """Write aggregated summaries by dimension."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)

        # Per-plane win fraction
        w.writerow(["=== Win fraction by fault plane ==="])
        w.writerow(["plane", "plane_weight", "total_cells", "win_cells", "win_frac", "avg_savings_vs_B2"])
        for p in range(ACTIVE_PLANES):
            sub = [r for r in all_rows if r["fault_plane"] == p]
            wins = [r for r in sub if r in win_rows]
            avg_sav = sum(r["b4_vs_b2_savings"] for r in wins) / max(len(wins), 1)
            w.writerow([p, PLANE_WEIGHTS[p], len(sub), len(wins),
                        f"{len(wins)/max(len(sub),1):.4f}", f"{avg_sav:.4f}"])
        w.writerow([])

        # Per-epsilon win fraction
        w.writerow(["=== Win fraction by epsilon ==="])
        w.writerow(["epsilon", "total_cells", "win_cells", "win_frac"])
        epsilons_sorted = sorted(set(r["epsilon"] for r in all_rows))
        for eps in epsilons_sorted:
            sub = [r for r in all_rows if r["epsilon"] == eps]
            wins = [r for r in sub if r in win_rows]
            w.writerow([eps, len(sub), len(wins), f"{len(wins)/max(len(sub),1):.4f}"])
        w.writerow([])

        # Per-k win fraction
        w.writerow(["=== Win fraction by k_depth ==="])
        w.writerow(["k_depth", "total_cells", "win_cells", "win_frac"])
        for k in sorted(set(r["k_depth"] for r in all_rows)):
            sub = [r for r in all_rows if r["k_depth"] == k]
            wins = [r for r in sub if r in win_rows]
            w.writerow([k, len(sub), len(wins), f"{len(wins)/max(len(sub),1):.4f}"])
        w.writerow([])

        # Per-segment-size win fraction
        w.writerow(["=== Win fraction by segment_rows ==="])
        w.writerow(["segment_rows", "total_cells", "win_cells", "win_frac"])
        for seg in sorted(set(r["segment_rows"] for r in all_rows)):
            sub = [r for r in all_rows if r["segment_rows"] == seg]
            wins = [r for r in sub if r in win_rows]
            w.writerow([seg, len(sub), len(wins), f"{len(wins)/max(len(sub),1):.4f}"])

    print(f"Wrote aggregated summary to {path}")


def compute_mixture_advantage(rows, mixture_planes, eps, k, seg):
    """Compute B4 vs B2 expected savings under a plane-probability mixture.

    mixture_planes: list of (plane, probability) pairs summing to 1.
    """
    total_b2 = 0.0
    total_b4 = 0.0
    certified_frac = 0.0
    absorbed_frac = 0.0
    fallback_frac = 0.0

    for plane, prob in mixture_planes:
        # Average over suspect counts 1..100
        for sc in [1, 5, 10, 50, 100]:
            r = simulate_segment(plane, sc, eps, k, seg)
            total_b2 += prob * r["b2_cost"] / 5
            total_b4 += prob * r["b4_cost"] / 5
            if r["b4_certified"]:
                certified_frac += prob / 5
            if r["b4_absorbed"]:
                absorbed_frac += prob / 5
            if r["b4_fallback"]:
                fallback_frac += prob / 5

    return {
        "epsilon": eps,
        "k_depth": k,
        "segment_rows": seg,
        "expected_b2_cost": total_b2,
        "expected_b4_cost": total_b4,
        "savings_vs_b2": total_b2 - total_b4,
        "savings_pct": (total_b2 - total_b4) / total_b2 * 100 if total_b2 > 0 else 0,
        "certified_frac": certified_frac,
        "absorbed_frac": absorbed_frac,
        "fallback_frac": fallback_frac,
    }


def run_mixture_analysis():
    """Evaluate expected B4 advantage under real-distribution fault mixtures."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Three fault mixture scenarios:
    mixtures = {
        "uniform_plane":                # Equal probability per plane (sensor/uniform-like)
            [(0, 1/6), (1, 1/6), (2, 1/6), (3, 1/6), (4, 1/6), (5, 1/6)],
        "low_skewed":                   # 70% of faults hit P3-P5 (CESM Q plausible)
            [(0, 0.05), (1, 0.10), (2, 0.15), (3, 0.25), (4, 0.25), (5, 0.20)],
        "high_skewed":                  # 80% of faults hit P0-P2 (burst-dominated)
            [(0, 0.40), (1, 0.25), (2, 0.15), (3, 0.10), (4, 0.05), (5, 0.05)],
    }

    epsilons = [1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 1e8, 1e9]
    k_range = [3, 4, 5, 6]
    seg = 1024

    rows = []
    for name, mixture in mixtures.items():
        for eps in epsilons:
            for k in k_range:
                r = compute_mixture_advantage(None, mixture, eps, k, seg)
                r["mixture"] = name
                rows.append(r)

    path = os.path.join(RESULTS_DIR, "mixture_analysis.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} mixture cells to {path}")

    # Print highlights
    print("\n=== Mixture analysis highlights ===")
    for name in mixtures:
        sub = [r for r in rows if r["mixture"] == name]
        wins = [r for r in sub if r["savings_vs_b2"] > 0 and r["certified_frac"] > 0]
        print(f"  {name}: {len(wins)}/{len(sub)} win cells "
              f"(max savings={max(r['savings_pct'] for r in sub):.1f}%, "
              f"max cert={max(r['certified_frac'] for r in sub)*100:.0f}%)")

    return rows


if __name__ == "__main__":
    print("P3-A0 Synthetic Simulation")
    print("=" * 60)
    win_cells, all_rows = run_sweep()
    mixture_rows = run_mixture_analysis()
    print("\nDone.")
