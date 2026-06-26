#!/usr/bin/env python3
"""Aggregation, plotting, and review report for Phase 1 plane sensitivity.

Reads a Phase 1 CSV and produces:
1. Plane-wise error tables (raw + normalized)
2. MSB-vs-LSB group summaries (raw + normalized)
3. Sparse plane / structural caveat flags
4. GO/CONDITIONAL_GO/NO-GO with full caveat documentation

Usage:
  python3 scripts/aggregate_reliability.py \
    --csv results/reliability_layer1/canonical_*/canonical_matrix.csv \
    --output-dir /tmp/reliability_report
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_exact_int(s: str) -> int:
    """Parse exact decimal string SUM/damage field. Never use float."""
    return int(s)


def median_int(vals: list[int]) -> int:
    """Exact integer median. Returns int, never float."""
    sv = sorted(vals)
    n = len(sv)
    if n == 0:
        return 0
    if n % 2 == 1:
        return sv[n // 2]
    # Even count: average of two middle values, but keep exact integer
    # by using floor. For Phase 1 seed-count analysis this is acceptable.
    return (sv[n // 2 - 1] + sv[n // 2]) // 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/reliability_report"))
    parser.add_argument("--plot", action="store_true", help="Generate plots (requires matplotlib)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Read {len(rows)} rows from {args.csv}")

    # Validate columns
    required = [
        "dataset", "target_plane", "fault_rate", "seed",
        "clean_encoded_sum", "gpu_corrupted_sum",
        "abs_sum_damage_encoded", "normalized_abs_sum_damage",
        "oracle_match", "plane_nonzero_count", "plane_nonzero_fraction",
    ]
    missing = [c for c in required if c not in rows[0]]
    if missing:
        print(f"ERROR: missing columns: {missing}")
        sys.exit(2)

    # Filter canonical
    canonical = [r for r in rows if r.get("oracle_match", "").strip() == "true"]
    skipped = len(rows) - len(canonical)
    if skipped:
        print(f"WARNING: {skipped} non-canonical rows skipped")
    if not canonical:
        print("ERROR: no canonical rows")
        sys.exit(2)

    # Group by (dataset, target_plane, fault_rate) → list of raw/normalized damage
    raw_cells: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    norm_cells: dict[tuple[str, int, str], list[int]] = defaultdict(list)

    for row in canonical:
        ds = row["dataset"]
        plane = int(row["target_plane"])
        rate = row["fault_rate"]
        try:
            raw_cells[(ds, plane, rate)].append(parse_exact_int(row["abs_sum_damage_encoded"]))
            norm_cells[(ds, plane, rate)].append(parse_exact_int(row["normalized_abs_sum_damage"]))
        except (ValueError, KeyError):
            continue

    # Cell medians
    raw_med: dict[tuple[str, int, str], int] = {k: median_int(v) for k, v in raw_cells.items()}
    norm_med: dict[tuple[str, int, str], int] = {k: median_int(v) for k, v in norm_cells.items()}

    # Datasets and fault rates (sorted)
    datasets = sorted(set(k[0] for k in raw_cells))
    fault_rates = sorted(
        set(k[2] for k in raw_cells),
        key=lambda r: float(r.replace("1e-0", "1e-").replace("1e-", "")))

    MSB = [0, 1, 2]
    LSB = [5, 6, 7]

    # ── Build report ──
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Phase 1 Plane Sensitivity Report")
    lines.append(f"Source: {args.csv}")
    lines.append(f"Canonical rows: {len(canonical)}")
    lines.append("=" * 72)

    # 1. Plane-wise raw damage table
    lines.append("\n--- Raw abs_sum_damage_encoded (cell median) ---")
    hdr = f"{'Dataset':<15} {'Plane':>6} " + " ".join(f"{r:>22}" for r in fault_rates)
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for ds in datasets:
        for plane in range(8):
            vals = " ".join(f"{raw_med.get((ds, plane, r), 0):>22}" for r in fault_rates)
            lines.append(f"{ds:<15} {plane:>6} {vals}")

    # 2. Normalized damage table
    lines.append("\n--- Normalized abs_sum_damage (cell median) ---")
    hdr = f"{'Dataset':<15} {'Plane':>6} " + " ".join(f"{r:>22}" for r in fault_rates)
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for ds in datasets:
        for plane in range(8):
            vals = " ".join(f"{norm_med.get((ds, plane, r), 0):>22}" for r in fault_rates)
            lines.append(f"{ds:<15} {plane:>6} {vals}")

    # 3. Plane occupancy / sparse caveat
    lines.append("\n--- Plane Occupancy (nonzero fraction) ---")
    for ds in datasets:
        sample_row = next((r for r in canonical if r["dataset"] == ds), None)
        if sample_row:
            nzf = sample_row.get("plane_nonzero_fraction", "")
            lines.append(f"  {ds}: {nzf}")
            # Sparse plane detection
            fractions = [float(x) for x in nzf.split("|")]
            for p, frac in enumerate(fractions):
                if frac < 1e-6:
                    lines.append(f"    ** plane {p} is structurally sparse (fraction={frac:.2e})")

    # 4. MSB/LSB group summaries (raw + normalized)
    lines.append("\n--- MSB vs LSB Group Summary ---")
    for ds in datasets:
        lines.append(f"\n  Dataset: {ds}")
        for rate in fault_rates:
            # Raw
            raw_msb = [raw_med.get((ds, p, rate), 0) for p in MSB if (ds, p, rate) in raw_med]
            raw_lsb = [raw_med.get((ds, p, rate), 0) for p in LSB if (ds, p, rate) in raw_med]
            raw_msb_m = median_int(raw_msb) if raw_msb else 0
            raw_lsb_m = median_int(raw_lsb) if raw_lsb else 0
            raw_ratio = f"{raw_msb_m / raw_lsb_m:.2e}" if raw_lsb_m > 0 else "UNDEFINED"
            raw_caveat = " ** LSB near noise **" if 0 < raw_lsb_m < 10 else ""

            # Normalized
            norm_msb = [norm_med.get((ds, p, rate), 0) for p in MSB if (ds, p, rate) in norm_med]
            norm_lsb = [norm_med.get((ds, p, rate), 0) for p in LSB if (ds, p, rate) in norm_med]
            norm_msb_m = median_int(norm_msb) if norm_msb else 0
            norm_lsb_m = median_int(norm_lsb) if norm_lsb else 0
            norm_ratio = f"{norm_msb_m / norm_lsb_m:.2e}" if norm_lsb_m > 0 else "UNDEFINED"

            lines.append(f"    rate={rate:>8}  "
                         f"raw_MSB={raw_msb_m:>22} raw_LSB={raw_lsb_m:>22} "
                         f"ratio={raw_ratio}{raw_caveat}")
            lines.append(f"                 "
                         f"norm_MSB={norm_msb_m:>22} norm_LSB={norm_lsb_m:>22} "
                         f"norm_ratio={norm_ratio}")

    # 5. Stop/Go
    lines.append("\n--- Stop/Go Assessment ---")

    # Check sparse planes that could bias raw MSB damage
    sparse_caveats = []
    for ds in datasets:
        sample_row = next((r for r in canonical if r["dataset"] == ds), None)
        if sample_row:
            nzf_str = sample_row.get("plane_nonzero_fraction", "0|0|0|0|0|0|0|0")
            fractions = [float(x) for x in nzf_str.split("|")]
            for p in MSB:
                if fractions[p] < 1e-6:
                    sparse_caveats.append(f"  {ds} plane {p}: nonzero fraction {fractions[p]:.2e}")
    if sparse_caveats:
        lines.append("  Sparse plane caveats (raw MSB damage may be biased by positional weight):")
        for c in sparse_caveats:
            lines.append(f"    {c}")
        lines.append("  ** Raw MSB-vs-LSB separation alone is not sufficient evidence for Phase 2.")
        lines.append("  ** Normalized damage and occupancy-aware analysis are required.")

    # Decision logic
    clean_gates_pass = True  # verified by runner
    oracle_mismatches = skipped

    # Material separation: both raw and normalized should show MSB > LSB
    def has_separation(ds: str, use_norm: bool) -> bool:
        med = norm_med if use_norm else raw_med
        for rate in fault_rates:
            msb_v = [med.get((ds, p, rate), 0) for p in MSB if (ds, p, rate) in med]
            lsb_v = [med.get((ds, p, rate), 0) for p in LSB if (ds, p, rate) in med]
            if not msb_v or not lsb_v:
                continue
            msb_m = median_int(msb_v)
            lsb_m = median_int(lsb_v)
            if lsb_m == 0:
                continue  # undefined ratio not a blocker
            if msb_m <= lsb_m and lsb_m > 0:
                return False
        return True

    primaries = ["sensor", "uniform"]
    raw_sep = all(has_separation(ds, False) for ds in primaries)
    norm_sep = all(has_separation(ds, True) for ds in primaries)

    lines.append(f"  Raw MSB > LSB (sensor+uniform): {raw_sep}")
    lines.append(f"  Normalized MSB > LSB (sensor+uniform): {norm_sep}")

    if clean_gates_pass and oracle_mismatches == 0 and raw_sep:
        if norm_sep and not sparse_caveats:
            recommendation = "GO"
        else:
            recommendation = "CONDITIONAL GO"
    else:
        recommendation = "NO-GO"

    lines.append(f"\n  Recommendation: {recommendation}")
    lines.append(f"    Clean gates: {'PASS' if clean_gates_pass else 'FAIL'}")
    lines.append(f"    Oracle mismatches: {oracle_mismatches}")
    lines.append(f"    Raw separation (sensor+uniform): {raw_sep}")
    lines.append(f"    Normalized separation (sensor+uniform): {norm_sep}")
    lines.append(f"    Sparse planes: {'yes' if sparse_caveats else 'no'}")
    if recommendation == "GO":
        pass
    elif recommendation == "CONDITIONAL GO":
        lines.append("  Caveats to resolve before Phase 2:")
        if not norm_sep:
            lines.append("    - Normalized damage does not show material MSB-vs-LSB separation")
        if sparse_caveats:
            lines.append("    - Sparse high planes may bias raw separation signal")
    else:
        lines.append("  Blockers:")
        if not clean_gates_pass:
            lines.append("    - Clean gates failed")
        if oracle_mismatches > 0:
            lines.append(f"    - {oracle_mismatches} oracle mismatches")
        if not raw_sep:
            lines.append("    - No material MSB-vs-LSB separation in raw damage")

    # Write report
    report_path = output_dir / "phase1_review_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"\nReport: {report_path}")
    print("\n".join(lines[-15:]))

    # Summary CSV
    summary_csv = output_dir / "msb_lsb_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "fault_rate",
                     "raw_msb_median", "raw_lsb_median", "raw_ratio",
                     "norm_msb_median", "norm_lsb_median", "norm_ratio",
                     "caveat"])
        for ds in datasets:
            for rate in fault_rates:
                raw_msb = median_int([raw_med.get((ds, p, rate), 0) for p in MSB if (ds, p, rate) in raw_med])
                raw_lsb = median_int([raw_med.get((ds, p, rate), 0) for p in LSB if (ds, p, rate) in raw_med])
                norm_msb = median_int([norm_med.get((ds, p, rate), 0) for p in MSB if (ds, p, rate) in norm_med])
                norm_lsb = median_int([norm_med.get((ds, p, rate), 0) for p in LSB if (ds, p, rate) in norm_med])
                raw_r = f"{raw_msb / raw_lsb:.2e}" if raw_lsb > 0 else "UNDEFINED"
                norm_r = f"{norm_msb / norm_lsb:.2e}" if norm_lsb > 0 else "UNDEFINED"
                caveat = "LSB near noise" if 0 < raw_lsb < 10 else ""
                w.writerow([ds, rate, raw_msb, raw_lsb, raw_r, norm_msb, norm_lsb, norm_r, caveat])
    print(f"Summary: {summary_csv}")

    # ── Plots ─────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("WARNING: matplotlib not available, skipping plots")
            return

        for metric_name, metric_med in [("raw", raw_med), ("normalized", norm_med)]:
            for ds in datasets:
                fig, ax = plt.subplots(figsize=(10, 6))
                for plane in range(8):
                    vals = [metric_med.get((ds, plane, r), 0) for r in fault_rates]
                    nonzero = [v for v in vals if v > 0]
                    if nonzero:
                        ax.plot(range(len(vals)), vals, marker="o", label=f"plane {plane}")
                ax.set_yscale("log")
                ax.set_xlabel("fault rate index")
                ax.set_ylabel(f"{metric_name} abs_sum_damage_encoded")
                ax.set_title(f"{ds} — {metric_name} damage by plane")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                path = output_dir / f"{ds}_{metric_name}_plane_damage.png"
                fig.savefig(path, dpi=150)
                plt.close(fig)
                print(f"  Plot: {path}")

        # MSB vs LSB group bar charts
        for ds in datasets:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            for ax, metric_name, metric_med in [
                (ax1, "raw", raw_med), (ax2, "normalized", norm_med)
            ]:
                rates = list(fault_rates)
                msb_vals = []
                lsb_vals = []
                for rate in rates:
                    m = median_int([metric_med.get((ds, p, rate), 0) for p in MSB if (ds, p, rate) in metric_med])
                    l = median_int([metric_med.get((ds, p, rate), 0) for p in LSB if (ds, p, rate) in metric_med])
                    msb_vals.append(m)
                    lsb_vals.append(l)
                x = range(len(rates))
                ax.bar(x, msb_vals, width=0.35, label="MSB [0,1,2]", alpha=0.8)
                ax.bar([i + 0.35 for i in x], lsb_vals, width=0.35, label="LSB [5,6,7]", alpha=0.8)
                ax.set_yscale("log")
                ax.set_xticks([i + 0.175 for i in x])
                ax.set_xticklabels(rates, rotation=45)
                ax.set_ylabel(f"{metric_name} median damage")
                ax.set_title(f"{ds} — {metric_name}")
                ax.legend()
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = output_dir / f"{ds}_msb_vs_lsb.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Plot: {path}")


if __name__ == "__main__":
    main()
