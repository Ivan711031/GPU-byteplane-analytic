#!/usr/bin/env python3
"""Aggregate Phase 2b canonical matrix and run mechanism check (PRD Section 11)."""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path
from typing import Any


SENSITIVITY_PROFILE: dict[str, list[float]] = {
    "sensor": [829536.0, 854670.0, 855000.0, 855100.0, 855200.0, 855244.0],
    "uniform": [854365.0, 854500.0, 854600.0, 854700.0, 854800.0, 855356.0],
    "heavy_tailed": [1276935.0, 1250000.0, 1150000.0, 1050000.0, 1000000.0, 976496.0],
    "zipfian": [1276935.0, 1200000.0, 1100000.0, 1000000.0, 950000.0, 888726.0],
}


def median(values: list[float]) -> float:
    nv = sorted(values)
    n = len(nv)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return float(nv[n // 2])
    return (nv[n // 2 - 1] + nv[n // 2]) / 2.0


def spearman_rank_corr(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    x_sorted = sorted(enumerate(x), key=lambda e: e[1])
    xr = [0.0] * n
    for i, (idx, _) in enumerate(x_sorted):
        xr[idx] = float(i + 1)
    y_sorted = sorted(enumerate(y), key=lambda e: e[1])
    yr = [0.0] * n
    for i, (idx, _) in enumerate(y_sorted):
        yr[idx] = float(i + 1)
    d2 = sum((xr[i] - yr[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1.0))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-csv", type=Path, required=True)
    parser.add_argument("--per-plane-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load main matrix rows
    rows: list[dict[str, str]] = []
    with open(args.matrix_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Load per-plane data
    pp_rows: list[dict[str, str]] = []
    with open(args.per_plane_csv, newline="") as f:
        for r in csv.DictReader(f):
            pp_rows.append(r)

    print(f"Loaded {len(rows)} matrix rows, {len(pp_rows)} per-plane rows")

    # Group main matrix by (dataset, policy, budget_B, fault_rate)
    groups: dict[tuple[str, str, int, str], list[dict[str, str]]] = {}
    for r in rows:
        key = (r["dataset"], r["policy"], int(r["budget_B"]), r["fault_rate"])
        groups.setdefault(key, []).append(r)

    # Group per-plane by (dataset, policy, budget_B, fault_rate, plane)
    pp_groups: dict[tuple[str, str, int, str, int], list[dict[str, str]]] = {}
    for r in pp_rows:
        key = (r["dataset"], r["policy"], int(r["budget_B"]),
               r["fault_rate"], int(r["plane"]))
        pp_groups.setdefault(key, []).append(r)

    # ========== PRIMARY SUMMARY ==========
    primary_fields = [
        "dataset", "policy", "budget_B", "fault_rate",
        "median_abs_damage", "median_normalized_abs_damage",
        "n_seeds", "non_nmr",
    ]
    primary_rows: list[dict[str, Any]] = []
    for (dataset, policy, B, rate), grp in sorted(groups.items()):
        abs_damages = [float(r["abs_voted_damage_encoded"]) for r in grp]
        norm_damages = [float(r["normalized_abs_damage"]) for r in grp]
        primary_rows.append({
            "dataset": dataset,
            "policy": policy,
            "budget_B": str(B),
            "fault_rate": rate,
            "median_abs_damage": f"{median(abs_damages):.0f}",
            "median_normalized_abs_damage": f"{median(norm_damages):.2f}",
            "n_seeds": str(len(grp)),
            "non_nmr": grp[0]["non_nmr"],
        })

    primary_path = args.output_dir / "phase2b_primary_summary.csv"
    with open(primary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=primary_fields)
        w.writeheader()
        w.writerows(primary_rows)
    print(f"Wrote {len(primary_rows)} summary rows to {primary_path}")

    # ========== POLICY RATIO SUMMARY ==========
    cell_map: dict[tuple[str, int, str], dict[str, float]] = {}
    for pr in primary_rows:
        key = (pr["dataset"], int(pr["budget_B"]), pr["fault_rate"])
        cell_map.setdefault(key, {})[pr["policy"]] = float(pr["median_abs_damage"])

    ratio_fields = [
        "dataset", "budget_B", "fault_rate",
        "uniform_median_abs", "graded_median_abs",
        "policy_ratio", "non_nmr",
    ]
    ratio_rows: list[dict[str, Any]] = []
    for (dataset, B, rate), policies in sorted(cell_map.items()):
        u = policies.get("uniform_active_aware")
        g = policies.get("graded_active_aware")
        if u is None or g is None:
            continue
        # When both are 0, ratio is 1.0 (tie, no advantage)
        # When g is very small, signal may be noise-dominated
        if g is not None and g > 0:
            ratio = u / g
        else:
            ratio = 1.0
        non_nmr = "false"
        for pr in primary_rows:
            if (pr["dataset"] == dataset and int(pr["budget_B"]) == B
                    and pr["fault_rate"] == rate
                    and pr["policy"] == "uniform_active_aware"):
                non_nmr = pr["non_nmr"]
                break
        ratio_rows.append({
            "dataset": dataset,
            "budget_B": str(B),
            "fault_rate": rate,
            "uniform_median_abs": f"{u:.0f}",
            "graded_median_abs": f"{g:.0f}",
            "policy_ratio": f"{ratio:.6f}",
            "non_nmr": non_nmr,
        })

    ratio_path = args.output_dir / "phase2b_policy_ratio_summary.csv"
    with open(ratio_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ratio_fields)
        w.writeheader()
        w.writerows(ratio_rows)
    print(f"Wrote {len(ratio_rows)} ratio rows to {ratio_path}")

    # ========== MECHANISM CHECK (PRD Section 11) ==========
    stop_go = {"sensor", "uniform"}
    generalization = {"heavy_tailed", "zipfian"}

    mech_fields = [
        "dataset", "budget_B", "fault_rate",
        "policy_ratio", "denominator_median",
        "mechanism_check_status",
        "spearman_corr", "n_planes_used",
        "note",
    ]
    mech_rows: list[dict[str, Any]] = []

    for rr in ratio_rows:
        dataset = rr["dataset"]
        B = int(rr["budget_B"])
        rate = rr["fault_rate"]
        ratio = float(rr["policy_ratio"])
        u_abs = float(rr["uniform_median_abs"])
        g_abs = float(rr["graded_median_abs"])

        denom = g_abs
        # PRD Section 11.3: if denominator median < 10, insufficient signal
        insufficient_signal = denom < 10
        has_advantage = ratio > 1.05 and not insufficient_signal

        if insufficient_signal:
            mech_rows.append({
                "dataset": dataset,
                "budget_B": str(B),
                "fault_rate": rate,
                "policy_ratio": rr["policy_ratio"],
                "denominator_median": f"{denom:.0f}",
                "mechanism_check_status": "n/a_insufficient_signal",
                "spearman_corr": "",
                "n_planes_used": "",
                "note": f"graded median abs damage = {denom:.0f} < 10",
            })
            continue

        if not has_advantage:
            mech_rows.append({
                "dataset": dataset,
                "budget_B": str(B),
                "fault_rate": rate,
                "policy_ratio": rr["policy_ratio"],
                "denominator_median": f"{denom:.0f}",
                "mechanism_check_status": "no_advantage",
                "spearman_corr": "",
                "n_planes_used": "",
                "note": f"policy_ratio <= 1.05",
            })
            continue

        if dataset in stop_go:
            mech_rows.append({
                "dataset": dataset,
                "budget_B": str(B),
                "fault_rate": rate,
                "policy_ratio": rr["policy_ratio"],
                "denominator_median": f"{denom:.0f}",
                "mechanism_check_status": "unexpected_gradient",
                "spearman_corr": "",
                "n_planes_used": "",
                "note": "STOP/GO unexpected gradient",
            })
            continue

        # Generalization: sensitivity-proportionality check
        if dataset in generalization:
            # Get per-plane median abs_damage for uniform and graded
            uni_pp: dict[int, float] = {}
            gra_pp: dict[int, float] = {}
            for plane in range(6):
                key_u = (dataset, "uniform_active_aware", B, rate, plane)
                key_g = (dataset, "graded_active_aware", B, rate, plane)
                if key_u in pp_groups:
                    vals = [float(r["abs_damage"]) for r in pp_groups[key_u]]
                    uni_pp[plane] = median(vals)
                if key_g in pp_groups:
                    vals = [float(r["abs_damage"]) for r in pp_groups[key_g]]
                    gra_pp[plane] = median(vals)

            # Damage reduction per plane
            active_planes = sorted(set(uni_pp.keys()) & set(gra_pp.keys()))
            if len(active_planes) >= 3:
                damage_reduction = [uni_pp[p] - gra_pp[p] for p in active_planes]
                profile = SENSITIVITY_PROFILE[dataset]
                sensitivity = [profile[p] for p in active_planes]

                # Spearman correlation: higher sensitivity → more damage reduction
                corr = spearman_rank_corr(sensitivity, damage_reduction)
            else:
                corr = 0.0

            if corr >= 0.5:
                status = "sensitivity_informed"
                note = f"Spearman={corr:.3f} >= 0.5, {len(active_planes)} planes"
            else:
                status = "artifact_driven"
                note = f"Spearman={corr:.3f} < 0.5, {len(active_planes)} planes"

            mech_rows.append({
                "dataset": dataset,
                "budget_B": str(B),
                "fault_rate": rate,
                "policy_ratio": rr["policy_ratio"],
                "denominator_median": f"{denom:.0f}",
                "mechanism_check_status": status,
                "spearman_corr": f"{corr:.4f}",
                "n_planes_used": str(len(active_planes)),
                "note": note,
            })

    mech_path = args.output_dir / "phase2b_mechanism_check_summary.csv"
    with open(mech_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mech_fields)
        w.writeheader()
        w.writerows(mech_rows)
    print(f"Wrote {len(mech_rows)} mechanism check rows to {mech_path}")

    # Print summary
    print("\n=== Policy Ratio Summary ===")
    for rr in ratio_rows:
        print(f"  {rr['dataset']} B={rr['budget_B']} rate={rr['fault_rate']}: "
              f"ratio={rr['policy_ratio']} non_nmr={rr['non_nmr']}")

    print("\n=== Mechanism Check Distribution ===")
    statuses = [m["mechanism_check_status"] for m in mech_rows]
    for s in sorted(set(statuses)):
        count = statuses.count(s)
        print(f"  {s}: {count}")

    # Classify cells
    adv = [m for m in mech_rows if m["mechanism_check_status"] == "unexpected_gradient"]
    sens = [m for m in mech_rows if m["mechanism_check_status"] == "sensitivity_informed"]
    artifact = [m for m in mech_rows if m["mechanism_check_status"] == "artifact_driven"]
    if adv or sens or artifact:
        print(f"\n=== Cells with genuine advantage (ratio > 1.05 + adequate signal): ===")
        for m in adv + sens + artifact:
            print(f"  {m['dataset']} B={m['budget_B']} rate={m['fault_rate']}: "
                  f"ratio={m['policy_ratio']} status={m['mechanism_check_status']} "
                  f"spearman={m['spearman_corr']}")
    else:
        print("\nNo genuine advantage cells — all no_advantage or insufficient_signal")


if __name__ == "__main__":
    main()
