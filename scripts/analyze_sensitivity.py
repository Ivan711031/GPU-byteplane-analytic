#!/usr/bin/env python3
"""Analyze active-delta sensitivity sweep results."""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main(result_dir: Path) -> None:
    rows = []
    with open(result_dir / "canonical_matrix.csv") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["_signed"] = int(r["signed_damage_encoded"])
            r["_abs"] = int(r["abs_damage_encoded"])
            r["_norm"] = int(r["normalized_abs_damage"])
            r["_plane"] = int(r["plane"])
            r["_seed"] = int(r["seed"])
            r["_fault_count"] = int(r["fault_count"])
            rows.append(r)

    sensitivity = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["_plane"], r["fault_rate"])
        sensitivity[key].append(r)

    # sensitivity_summary.csv
    sens_fields = [
        "dataset", "plane", "plane_weight", "plane_nonzero_fraction",
        "plane_unique_count", "plane_entropy_bits",
        "fault_rate", "n_seeds",
        "mean_signed_damage", "mean_abs_damage", "mean_normalized_abs_damage",
        "std_normalized_abs_damage", "cv_normalized_abs_damage",
        "mean_fault_count", "active_byte_len",
    ]
    with open(result_dir / "sensitivity_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sens_fields)
        w.writeheader()
        for key in sorted(sensitivity):
            ds, p, rate = key
            vals = sensitivity[key]
            norm_vals = [v["_norm"] for v in vals]
            signed_vals = [v["_signed"] for v in vals]
            abs_vals = [v["_abs"] for v in vals]
            mean_norm = statistics.mean(norm_vals) if norm_vals else 0
            mean_signed = statistics.mean(signed_vals) if signed_vals else 0
            mean_abs = statistics.mean(abs_vals) if abs_vals else 0
            std_norm = statistics.stdev(norm_vals) if len(norm_vals) > 1 else 0
            cv_norm = (std_norm / mean_norm) if mean_norm > 0 else 0
            r0 = vals[0]
            w.writerow({
                "dataset": ds,
                "plane": p,
                "plane_weight": r0["plane_weight"],
                "plane_nonzero_fraction": r0["plane_nonzero_fraction"],
                "plane_unique_count": r0["plane_unique_count"],
                "plane_entropy_bits": r0["plane_entropy_bits"],
                "fault_rate": rate,
                "n_seeds": len(vals),
                "mean_signed_damage": f"{mean_signed:.0f}",
                "mean_abs_damage": f"{mean_abs:.0f}",
                "mean_normalized_abs_damage": f"{mean_norm:.4f}",
                "std_normalized_abs_damage": f"{std_norm:.4f}",
                "cv_normalized_abs_damage": f"{cv_norm:.4f}",
                "mean_fault_count": f"{statistics.mean([v['_fault_count'] for v in vals]):.1f}",
                "active_byte_len": r0["active_byte_len"],
            })
    print(f"Wrote {result_dir / 'sensitivity_summary.csv'} ({len(sensitivity)} rows)")

    # plane_entropy_summary.csv
    ent_fields = [
        "dataset", "active_byte_len", "plane", "plane_weight",
        "plane_nonzero_fraction", "plane_unique_count", "plane_entropy_bits",
    ]
    ent_rows = {}
    for r in rows:
        key = (r["dataset"], r["_plane"])
        if key not in ent_rows:
            ent_rows[key] = {
                "dataset": r["dataset"],
                "active_byte_len": r["active_byte_len"],
                "plane": r["_plane"],
                "plane_weight": r["plane_weight"],
                "plane_nonzero_fraction": r["plane_nonzero_fraction"],
                "plane_unique_count": r["plane_unique_count"],
                "plane_entropy_bits": r["plane_entropy_bits"],
            }
    with open(result_dir / "plane_entropy_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ent_fields)
        w.writeheader()
        for key in sorted(ent_rows):
            w.writerow(ent_rows[key])
    print(f"Wrote {result_dir / 'plane_entropy_summary.csv'} ({len(ent_rows)} rows)")

    # Analysis printout
    print("\n===== SENSITIVITY GRADIENT =====")
    for ds in ["sensor", "uniform", "heavy_tailed", "zipfian"]:
        print(f"\n--- {ds} ---")
        print(f"{'P':>3} {'Rate':>10} {'MeanNorm':>14} {'StdDev':>14} {'CV':>10} {'Entropy':>10}")
        for key in sorted(sensitivity):
            if key[0] != ds:
                continue
            _, p, rate = key
            vals = sensitivity[key]
            norm_vals = [v["_norm"] for v in vals]
            mean_n = statistics.mean(norm_vals)
            std_n = statistics.stdev(norm_vals) if len(norm_vals) > 1 else 0
            cv_n = std_n / mean_n if mean_n > 0 else 0
            ent = float(vals[0]["plane_entropy_bits"])
            print(f"{p:>3} {rate:>10} {mean_n:>14.2f} {std_n:>14.2f} {cv_n:>10.4f} {ent:>10.4f}")

    # Gradients: P0 vs P5 ratio
    print("\n===== P0/P5 RATIO (high-plane vs low-plane sensitivity) =====")
    for ds in ["sensor", "uniform", "heavy_tailed", "zipfian"]:
        print(f"\n--- {ds} ---")
        print(f"{'Rate':>10} {'P0_mean_norm':>14} {'P5_mean_norm':>14} {'P0/P5':>10}")
        for rate in sorted(set(r["fault_rate"] for r in rows)):
            p0_key = (ds, 0, rate)
            p5_key = (ds, 5, rate)
            if p0_key not in sensitivity or p5_key not in sensitivity:
                continue
            p0_mean = statistics.mean([v["_norm"] for v in sensitivity[p0_key]])
            p5_mean = statistics.mean([v["_norm"] for v in sensitivity[p5_key]])
            ratio = p0_mean / p5_mean if p5_mean > 0 else float("inf")
            print(f"{rate:>10} {p0_mean:>14.2f} {p5_mean:>14.2f} {ratio:>10.4f}")

    # Normalized damage vs entropy
    print("\n===== ENTROPY VS SENSITIVITY (normalized mean across rates) =====")
    for ds in ["sensor", "uniform", "heavy_tailed", "zipfian"]:
        print(f"\n--- {ds} ---")
        print(f"{'P':>3} {'Entropy':>10} {'MeanNorm@1e-4':>16} {'MeanNorm@1e-8':>15}")
        for p in range(6):
            key_high = (ds, p, "1e-04")
            key_low = (ds, p, "1e-08")
            if key_high not in sensitivity:
                continue
            high_mean = statistics.mean([v["_norm"] for v in sensitivity[key_high]])
            low_mean = statistics.mean([v["_norm"] for v in sensitivity.get(key_low, [])])
            ent = float(sensitivity[key_high][0]["plane_entropy_bits"])
            print(f"{p:>3} {ent:>10.4f} {high_mean:>16.2f} {low_mean:>15.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/analyze_sensitivity.py <result_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]))
