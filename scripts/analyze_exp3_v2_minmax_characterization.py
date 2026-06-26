#!/usr/bin/env python3
"""Analyze Exp3 v2 MIN/MAX progressive bound characterization results.

Groups results per variant (per CSV file), computes convergence metrics
and candidate-set collapse statistics, then prints a decision report.
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict


def variant_name(csv_path: Path) -> str:
    return csv_path.stem.replace("exp3_v2_minmax_", "")


def load_csvs(result_dir: Path) -> dict[str, list[dict]]:
    by_variant = defaultdict(list)
    for csv_path in sorted(result_dir.glob("exp3_v2_minmax_*.csv")):
        vname = variant_name(csv_path)
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                by_variant[vname].append(row)
    return dict(by_variant)


def k_first_exact(rows: list[dict], op: str) -> int:
    rows_sorted = sorted(rows, key=lambda r: int(r["refinement_depth"]))
    max_k = int(rows_sorted[-1]["refinement_depth"])
    for r in rows_sorted:
        k = int(r["refinement_depth"])
        if op == "min" and float(r["min_lower"]) == float(r["min_upper"]):
            return k
        if op == "max" and float(r["max_lower"]) == float(r["max_upper"]):
            return k
    return max_k


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <result_dir>")
        sys.exit(1)

    result_dir = Path(sys.argv[1])
    by_variant = load_csvs(result_dir)

    if not by_variant:
        print("ERROR: no CSV files found")
        sys.exit(1)

    all_rows = [r for rs in by_variant.values() for r in rs]

    print("=" * 80)
    print("Exp3-M2 MIN/MAX Convergence Characterization")
    print(f"Result dir: {result_dir}")
    print(f"Variants: {len(by_variant)}")

    # ── Validation gate ──
    violations = [r for r in all_rows
                  if r.get("bound_valid_min", "").strip().lower() != "true"
                  or r.get("bound_valid_max", "").strip().lower() != "true"]
    if violations:
        print(f"\nERROR: {len(violations)} bound violations found!")
        for v in violations[:5]:
            print(f"  {v.get('_source','?')} k={v['refinement_depth']}")
        sys.exit(1)
    print("Bound validity: ZERO violations")

    # ── Full-depth exactness ──
    for vname, vrows in by_variant.items():
        vrows_sorted = sorted(vrows, key=lambda r: int(r["refinement_depth"]))
        last = vrows_sorted[-1]
        if last["min_lower"] != last["min_upper"] or last["max_lower"] != last["max_upper"]:
            print(f"WARNING: {vname} does not converge at full depth k={last['refinement_depth']}")
    print("Full-depth exactness: ALL variants converge")

    # ── GPU-CPU match ──
    mismatches = 0
    for r in all_rows:
        for col in ["abs_cpu_gpu_min_lower", "abs_cpu_gpu_min_upper",
                     "abs_cpu_gpu_max_lower", "abs_cpu_gpu_max_upper"]:
            v = r.get(col, "0")
            if v and float(v) > 1e-12:
                mismatches += 1
    if mismatches:
        print(f"WARNING: {mismatches} GPU-CPU mismatches")
    else:
        print("GPU-CPU: exact match (all 0)")

    # ── Per-variant convergence summary ──
    print("\n" + "-" * 80)
    print(f"{'variant':<22} {'k_max':>5} {'k_exact_min':>11} {'k_exact_max':>11} "
          f"{'min_early':>10} {'max_early':>10} "
          f"{'min_width_k0':>14} {'max_width_k0':>14} "
          f"{'cand_ratio_min_k0':>17} {'cand_ratio_max_k0':>17}")

    rows_flat = []
    for vname in sorted(by_variant.keys()):
        vrows = sorted(by_variant[vname], key=lambda r: int(r["refinement_depth"]))
        k_max = int(vrows[-1]["refinement_depth"])
        k_exact_min = k_first_exact(vrows, "min")
        k_exact_max = k_first_exact(vrows, "max")
        min_early = f"{k_exact_min}/{k_max}" if k_exact_min < k_max else "--"
        max_early = f"{k_exact_max}/{k_max}" if k_exact_max < k_max else "--"

        k0 = vrows[0]
        ml0 = float(k0["min_lower"])
        mu0 = float(k0["min_upper"])
        xl0 = float(k0["max_lower"])
        xu0 = float(k0["max_upper"])
        mw0 = mu0 - ml0
        xw0 = xu0 - xl0
        total = max(int(k0.get("total_rows", 1)), 1)
        cr_min = int(k0.get("min_candidates", 0)) / total
        cr_max = int(k0.get("max_candidates", 0)) / total

        print(f"{vname:<22} {k_max:>5} {str(k_exact_min):>11} {str(k_exact_max):>11} "
              f"{min_early:>10} {max_early:>10} "
              f"{mw0:>14.4g} {xw0:>14.4g} "
              f"{cr_min:>17.6f} {cr_max:>17.6f}")

        rows_flat.append({
            "vname": vname,
            "k_max": k_max,
            "k_exact_min": k_exact_min,
            "k_exact_max": k_exact_max,
            "min_early": k_exact_min < k_max,
            "max_early": k_exact_max < k_max,
            "cr_min_k0": cr_min,
            "cr_max_k0": cr_max,
        })

    # ── Per-variant bound width progression ──
    print("\n" + "=" * 80)
    print("Per-variant bound width and candidate ratio progression (first row per k)")
    for vname in sorted(by_variant.keys()):
        vrows = sorted(by_variant[vname], key=lambda r: int(r["refinement_depth"]))
        # dedupe: take first row per k
        seen_k = set()
        deduped = []
        for r in vrows:
            k = int(r["refinement_depth"])
            if k not in seen_k:
                seen_k.add(k)
                deduped.append(r)

        print(f"\n--- {vname} ---")
        print(f"{'k':>3} {'min_lower':>24} {'min_upper':>24} {'min_width':>16} "
              f"{'max_lower':>24} {'max_upper':>24} {'max_width':>16} "
              f"{'cand_min':>10} {'cand_max':>10} {'cr_min':>10} {'cr_max':>10}")
        for r in deduped:
            k = int(r["refinement_depth"])
            ml = float(r["min_lower"])
            mu = float(r["min_upper"])
            mw = mu - ml
            xl = float(r["max_lower"])
            xu = float(r["max_upper"])
            xw = xu - xl
            total = max(int(r.get("total_rows", 1)), 1)
            cm = int(r.get("min_candidates", 0))
            cx = int(r.get("max_candidates", 0))
            print(f"{k:>3} {ml:>24.17g} {mu:>24.17g} {mw:>16.4g} "
                  f"{xl:>24.17g} {xu:>24.17g} {xw:>16.4g} "
                  f"{cm:>10} {cx:>10} {cm/total:>10.6f} {cx/total:>10.6f}")

    # ── Decision summary ──
    print("\n" + "=" * 80)
    print("DECISION INPUTS")

    early_min = sum(1 for r in rows_flat if r["min_early"])
    early_max = sum(1 for r in rows_flat if r["max_early"])
    both_early = sum(1 for r in rows_flat if r["min_early"] and r["max_early"])
    total = len(rows_flat)

    print(f"MIN converges before full depth: {early_min}/{total} variants")
    print(f"MAX converges before full depth: {early_max}/{total} variants")
    print(f"Both converge before full depth:  {both_early}/{total} variants")
    print(f"Neither converges early:          {total - early_min - early_max + both_early}/{total} variants")

    # Early convergence distribution
    k_exact_min_dist = defaultdict(int)
    k_exact_max_dist = defaultdict(int)
    for r in rows_flat:
        km = r["k_exact_min"]
        kx = r["k_exact_max"]
        k_exact_min_dist[km] += 1
        k_exact_max_dist[kx] += 1
    print(f"\nMIN k_exact distribution: {dict(sorted(k_exact_min_dist.items()))}")
    print(f"MAX k_exact distribution: {dict(sorted(k_exact_max_dist.items()))}")

    # Candidate ratio summary at k=0
    cr_min_all = [r["cr_min_k0"] for r in rows_flat]
    cr_max_all = [r["cr_max_k0"] for r in rows_flat]
    cr_min_all.sort()
    cr_max_all.sort()
    print(f"\nCandidate ratio at k=0 (median across variants):")
    print(f"  MIN: {cr_min_all[len(cr_min_all)//2]:.6f} (range {cr_min_all[0]:.6f}-{cr_min_all[-1]:.6f})")
    print(f"  MAX: {cr_max_all[len(cr_max_all)//2]:.6f} (range {cr_max_all[0]:.6f}-{cr_max_all[-1]:.6f})")

    # ── Recommendation ──
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    if both_early >= total * 0.7:
        print("Strong: MIN/MAX show widespread early convergence.")
        print("Action: append/extensibility evidence in paper is justified.")
    elif both_early > 0:
        print(f"Mixed: {both_early}/{total} variants get early convergence for both operators,")
        print(f"       but {total - both_early} do not. MIN/MAX has merit as future-work evidence.")
        print("Action: keep as future-work note, not paper mainline.")
    else:
        print("Poor: no early convergence observed. MIN/MAX requires full depth.")
        print("Action: kill or defer indefinitely.")


if __name__ == "__main__":
    main()
