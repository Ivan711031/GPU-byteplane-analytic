#!/usr/bin/env python3
"""Phase 2 primary aggregation and policy-ratio plots.

Reads per-dataset per-policy result CSVs and the policy catalogue, then
produces merged matrix, policy-ratio summary, review report, and optional plots.

Usage:
  python3 scripts/aggregate_reliability_phase2.py \
    --input-dirs <per-dataset per-policy result dirs> \
    --policy-catalogue results/policy_catalogue.json \
    --output-dir /tmp/phase2/aggregated \
    --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


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
    return (sv[n // 2 - 1] + sv[n // 2]) // 2


def rate_sort_key(r: str) -> float:
    s = r.strip()
    if not s:
        return 0.0
    return float(s)


def load_input_csvs(input_dirs: list[Path]) -> list[dict[str, str]]:
    all_rows: list[dict[str, str]] = []
    for d in input_dirs:
        if not d.is_dir():
            print(f"WARNING: input dir not found: {d}", file=sys.stderr)
            continue
        for csv_path in sorted(d.glob("*.csv")):
            with csv_path.open(newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    continue
                for row in reader:
                    all_rows.append(row)
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--policy-catalogue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots (requires matplotlib)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy catalogue
    catalogue = json.loads(args.policy_catalogue.read_text())
    catalogue_entries = catalogue.get("entries", [])
    print(f"Loaded {len(catalogue_entries)} catalogue entries")

    # Load all CSV rows
    all_rows = load_input_csvs(args.input_dirs)
    if not all_rows:
        print("ERROR: no CSV rows loaded")
        sys.exit(2)
    print(f"Loaded {len(all_rows)} rows from {len(args.input_dirs)} directories")

    # Separate canonical vs non-canonical
    canonical = [r for r in all_rows
                 if r.get("oracle_match", "").strip() == "true"]
    non_canonical = [r for r in all_rows
                     if r.get("oracle_match", "").strip() != "true"]
    print(f"Canonical: {len(canonical)}, failures: {len(non_canonical)}")

    # Discover unique dimensions
    datasets = sorted(set(r.get("dataset", "").strip() for r in canonical))
    fault_rates = sorted(
        set(r.get("fault_rate", "").strip() for r in canonical),
        key=rate_sort_key)
    budget_vals = sorted(
        set(int(r["budget_B"]) for r in canonical if "budget_B" in r))
    policies = sorted(set(
        r.get("policy", "").strip() for r in canonical))

    print(f"Datasets: {datasets}")
    print(f"Fault rates: {fault_rates}")
    print(f"Budget points: {budget_vals}")
    print(f"Policies: {policies}")

    # ── 1. phase2_matrix.csv ──
    all_fieldnames = list(all_rows[0].keys()) if all_rows else []
    matrix_path = output_dir / "phase2_matrix.csv"
    with matrix_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Phase 2 matrix: {matrix_path} ({len(all_rows)} rows)")

    # ── Build damage groups for policy-ratio computation ──
    # Group by (dataset, budget_B, fault_rate, policy) -> list of abs damage
    damage_cells: dict[tuple[str, int, str, str], list[int]] = defaultdict(list)
    for row in canonical:
        ds = row.get("dataset", "").strip()
        try:
            b = int(row["budget_B"])
            rate = row.get("fault_rate", "").strip()
            policy = row.get("policy", "").strip()
            damage_cells[(ds, b, rate, policy)].append(
                parse_exact_int(row["abs_voted_sum_damage_encoded"]))
        except (ValueError, KeyError):
            continue

    # Cell medians
    cell_med: dict[tuple[str, int, str, str], int] = {
        k: median_int(v) for k, v in damage_cells.items()
    }

    # ── 2. policy_ratio_summary.csv ──
    uniform_policy = "uniform"
    graded_policy = "graded_vacuous_aware"

    summary_header = [
        "dataset", "budget_B", "fault_rate",
        "cell_uniform", "cell_graded", "policy_ratio",
        "noise_dominated_flag", "undefined_flag", "primary_win",
    ]
    summary_rows: list[list] = []

    for ds in datasets:
        for b in budget_vals:
            for rate in fault_rates:
                u_key = (ds, b, rate, uniform_policy)
                g_key = (ds, b, rate, graded_policy)

                cell_u = cell_med.get(u_key, 0)
                cell_g = cell_med.get(g_key, 0)

                undefined_flag = ""
                noise_flag = ""
                primary_win = ""
                ratio = ""

                if cell_g == 0:
                    undefined_flag = "true"
                    ratio = "UNDEFINED"
                else:
                    ratio = cell_u / cell_g
                    if cell_g < 10:
                        noise_flag = "true"

                if isinstance(ratio, float) and ratio > 1.10 and cell_g >= 10:
                    primary_win = "true"

                summary_rows.append([
                    ds, str(b), rate,
                    str(cell_u), str(cell_g),
                    f"{ratio}" if isinstance(ratio, float) else ratio,
                    noise_flag, undefined_flag, primary_win,
                ])

    summary_path = output_dir / "policy_ratio_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(summary_header)
        w.writerows(summary_rows)
    print(f"Policy ratio summary: {summary_path} ({len(summary_rows)} rows)")

    primary_summary_path = output_dir / "phase2_primary_summary.csv"
    with primary_summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(summary_header)
        w.writerows(summary_rows)
    print(
        f"Phase 2 primary summary: {primary_summary_path} "
        f"({len(summary_rows)} rows)"
    )

    # Count primary wins
    primary_win_count = sum(
        1 for r in summary_rows if r[summary_header.index("primary_win")] == "true"
    )
    undefined_count = sum(
        1 for r in summary_rows if r[summary_header.index("undefined_flag")] == "true"
    )
    noise_count = sum(
        1 for r in summary_rows if r[summary_header.index("noise_dominated_flag")] == "true"
    )

    # ── 3. phase2_review_report.txt ──
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Phase 2 Policy-Ratio Review Report")
    lines.append(f"Source directories: {len(args.input_dirs)}")
    lines.append(f"Canonical rows: {len(canonical)}")
    lines.append(f"Failure rows: {len(non_canonical)}")
    lines.append("=" * 72)

    lines.append("\n--- Experiment Dimensions ---")
    lines.append(f"  Datasets: {', '.join(datasets)}")
    lines.append(f"  Fault rates: {', '.join(fault_rates)}")
    lines.append(f"  Budget points B: {', '.join(str(b) for b in budget_vals)}")
    lines.append(f"  Policies: {', '.join(policies)}")

    lines.append(f"\n--- Policy Ratio Summary ---")
    lines.append(f"  Primary-win cells: {primary_win_count}")
    lines.append(f"  Undefined-ratio cells: {undefined_count}")
    lines.append(f"  Noise-dominated cells: {noise_count}")

    lines.append("\n--- Policy Ratio Table (cell_uniform / cell_graded) ---")
    for ds in datasets:
        lines.append(f"\n  Dataset: {ds}")
        hdr = f"{'B':>6}  {'fault_rate':>10}"
        for rate in fault_rates:
            hdr += f"  {rate:>12}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for b in budget_vals:
            row_str = f"{b:>6}"
            for rate in fault_rates:
                r_key = (ds, b, rate, uniform_policy)
                g_key = (ds, b, rate, graded_policy)
                cell_u = cell_med.get(r_key, 0)
                cell_g = cell_med.get(g_key, 0)
                if cell_g == 0:
                    val = "  UNDEFINED  "
                elif cell_g < 10:
                    val = f"{cell_u / cell_g:>9.3f}* "
                else:
                    val = f"{cell_u / cell_g:>12.3f}"
                row_str += f"  {val}"
            lines.append(row_str)

    lines.append("\n--- Vote Outcome Summary ---")
    # Aggregate outcome counts across all canonical rows
    total_resolved = 0
    total_detected = 0
    total_undetected = 0
    total_outcome_rows = 0
    for row in canonical:
        try:
            total_resolved += int(row.get("resolved_correctly_count", "0"))
            total_detected += int(row.get("detected_mismatch_count", "0"))
            total_undetected += int(row.get("undetected_corruption_count", "0"))
            total_outcome_rows += 1
        except (ValueError, KeyError):
            continue
    lines.append(f"  Total rows with outcome data: {total_outcome_rows}")
    lines.append(f"  Resolved correctly: {total_resolved}")
    lines.append(f"  Detected mismatch:  {total_detected}")
    lines.append(f"  Undetected corruption: {total_undetected}")
    total_votes = total_resolved + total_detected + total_undetected
    if total_votes > 0:
        lines.append(f"  Resolved fraction: {total_resolved / total_votes:.6f}")
        lines.append(f"  Detected fraction:  {total_detected / total_votes:.6f}")
        lines.append(f"  Undetected fraction: {total_undetected / total_votes:.6f}")

    lines.append("\n--- Oracle & Clean Gate Summary ---")
    for ds in datasets:
        ds_canon = [r for r in canonical
                    if r.get("dataset", "").strip() == ds]
        ds_total = [r for r in all_rows
                    if r.get("dataset", "").strip() == ds]
        ds_mismatch = len(ds_total) - len(ds_canon)
        lines.append(f"  {ds}: {len(ds_canon)} canonical, "
                     f"{ds_mismatch} mismatches")

    oracle_mismatches = len(non_canonical)
    oracle_gate_pass = (oracle_mismatches == 0)

    lines.append(f"\n  Oracle mismatches: {oracle_mismatches}")
    lines.append(f"  Oracle gate: {'PASS' if oracle_gate_pass else 'FAIL'}")

    lines.append("\n--- Primary-Win Cells ---")
    win_rows = [r for r in summary_rows
                if r[summary_header.index("primary_win")] == "true"]
    primary_win_path = output_dir / "primary_win_cells.csv"
    with primary_win_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(summary_header)
        w.writerows(win_rows)
    print(f"Primary-win cells: {primary_win_path} ({len(win_rows)} rows)")

    if win_rows:
        lines.append(f"  {len(win_rows)} primary-win cells found:")
        for r in win_rows:
            ds, b, rate, cell_u, cell_g, ratio = r[:6]
            lines.append(f"    {ds} B={b} rate={rate}: "
                         f"uniform={cell_u} graded={cell_g} "
                         f"ratio={ratio}")
    else:
        lines.append("  No primary-win cells (policy_ratio <= 1.10 or graded < 10)")

    lines.append("\n--- Verdict ---")
    if oracle_gate_pass:
        if primary_win_count > 0:
            lines.append("  Graded policy shows measurable benefit over uniform "
                         f"({primary_win_count} primary-win cells).")
        else:
            lines.append("  No primary-win cells detected. "
                         "Graded policy does not meaningfully outperform uniform.")
        lines.append("  Recommendation: DATA_QUALIFIED_PASS")
    else:
        lines.append("  Oracle mismatches present. Review case_failures.csv.")
        lines.append("  Recommendation: INCONCLUSIVE")

    report_path = output_dir / "phase2_review_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"Review report: {report_path}")

    # ── 4. case_failures.csv ──
    failures_path = output_dir / "case_failures.csv"
    if non_canonical:
        with failures_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(non_canonical)
    else:
        with failures_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "budget_B", "fault_rate", "policy",
                         "oracle_match", "notes"])
    print(f"Case failures: {failures_path} ({len(non_canonical)} rows)")

    # ── 5. run_meta.txt ──
    meta_lines = [
        f"rows={len(all_rows)}",
        f"canonical={len(canonical)}",
        f"oracle_match={len(canonical)}/{len(all_rows)}",
        f"failure_rows={len(non_canonical)}",
        f"datasets={','.join(datasets)}",
        f"fault_rates={','.join(fault_rates)}",
        f"budget_points={','.join(str(b) for b in budget_vals)}",
        f"policies={','.join(policies)}",
        f"primary_win_cells={primary_win_count}",
        f"undefined_ratio_cells={undefined_count}",
        f"noise_dominated_cells={noise_count}",
    ]
    meta_path = output_dir / "run_meta.txt"
    meta_path.write_text("\n".join(meta_lines) + "\n")
    print(f"Run meta: {meta_path}")

    # ── Plots ──
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("WARNING: matplotlib not available, skipping plots")
            return

        # Plot 1: policy_ratio_curves.png
        for ds in datasets:
            fig, ax = plt.subplots(figsize=(10, 6))
            for rate in fault_rates:
                xs = []
                ys = []
                for b in budget_vals:
                    u_key = (ds, b, rate, uniform_policy)
                    g_key = (ds, b, rate, graded_policy)
                    cell_u = cell_med.get(u_key, 0)
                    cell_g = cell_med.get(g_key, 0)
                    if cell_g > 0 and cell_g >= 10:
                        xs.append(b)
                        ys.append(cell_u / cell_g)
                if xs:
                    ax.plot(xs, ys, marker="o", label=f"rate={rate}")
            ax.axhline(y=1.10, color="red", linestyle="--", alpha=0.6,
                       label="primary-win threshold")
            ax.axhline(y=1.0, color="black", linestyle=":", alpha=0.4)
            ax.set_xlabel("Budget B")
            ax.set_ylabel("Policy ratio (uniform / graded)")
            ax.set_title(f"{ds} — Policy ratio vs Budget")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = output_dir / f"{ds}_policy_ratio_curves.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Plot: {path}")

        # Plot 2: voted_damage_by_plane.png
        # For each dataset, pick the median budget B and show per-plane damage
        # under uniform vs graded at matched budget.
        # Group rows by (dataset, policy, budget_B, plane) -> list of abs damage
        plane_damage: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
        for row in canonical:
            ds = row.get("dataset", "").strip()
            policy = row.get("policy", "").strip()
            try:
                b = int(row["budget_B"])
                # allocation_r encodes per-plane r values as "1|3|3|3|2|2|2|2"
                r_parts = [int(x) for x in row.get("allocation_r", "1|1|1|1|1|1|1|1").split("|")]
                for p in range(8):
                    if r_parts[p] > 0:
                        # Map single per-config damage to per-plane damage.
                        # We store the aggregated per-config abs damage — per-plane
                        # breakdown is not directly in the CSV. Use per-plane
                        # stats from the oracle fields. We approximate by
                        # dividing proportionally to occupied planes.
                        pass
            except (ValueError, KeyError):
                continue

        # Per-plane damage is not directly available in the flat CSV.
        # Instead, show bar chart of abs_voted_sum_damage_encoded per policy
        # grouped by dataset at each budget point.
        for ds in datasets:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = range(len(budget_vals))
            width = 0.35
            for idx, policy in enumerate([uniform_policy, graded_policy]):
                vals = []
                for b in budget_vals:
                    key = (ds, b, "", policy)
                    # Aggregate all rates for this dataset+budget+policy
                    p_vals = [cell_med.get((ds, b, rate, policy), 0)
                              for rate in fault_rates]
                    if p_vals:
                        vals.append(median_int(p_vals))
                    else:
                        vals.append(0)
                offset = idx * width
                ax.bar([i + offset for i in x], vals, width,
                       label=policy, alpha=0.8)
            ax.set_xticks([i + width / 2 for i in x])
            ax.set_xticklabels([str(b) for b in budget_vals])
            ax.set_yscale("log")
            ax.set_xlabel("Budget B")
            ax.set_ylabel("Median abs_voted_sum_damage_encoded")
            ax.set_title(f"{ds} — Voted damage by policy")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = output_dir / f"{ds}_voted_damage_by_policy.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Plot: {path}")

        # Plot 3: vote_outcome_breakdown.png
        # Stacked bar of resolved/detected/undetected per (dataset, B, rate)
        outcome_rows: list[dict] = []
        for row in canonical:
            ds = row.get("dataset", "").strip()
            try:
                b = int(row["budget_B"])
                rate = row.get("fault_rate", "").strip()
                resolved = int(row.get("resolved_correctly_count", "0"))
                detected = int(row.get("detected_mismatch_count", "0"))
                undetected = int(row.get("undetected_corruption_count", "0"))
                outcome_rows.append({
                    "dataset": ds, "budget_B": b, "fault_rate": rate,
                    "resolved": resolved, "detected": detected,
                    "undetected": undetected,
                })
            except (ValueError, KeyError):
                continue

        # Aggregate by (dataset, budget_B, fault_rate) across policies
        outcome_agg: dict[tuple[str, int, str], dict[str, int]] = defaultdict(
            lambda: {"resolved": 0, "detected": 0, "undetected": 0})
        for o in outcome_rows:
            key = (o["dataset"], o["budget_B"], o["fault_rate"])
            outcome_agg[key]["resolved"] += o["resolved"]
            outcome_agg[key]["detected"] += o["detected"]
            outcome_agg[key]["undetected"] += o["undetected"]

        for ds in datasets:
            fig, axes = plt.subplots(
                len(budget_vals), 1, figsize=(10, 3 * len(budget_vals)),
                squeeze=False)
            for bi, b in enumerate(budget_vals):
                ax = axes[bi][0]
                rates = list(fault_rates)
                resolved_vals = []
                detected_vals = []
                undetected_vals = []
                for rate in rates:
                    key = (ds, b, rate)
                    agg = outcome_agg.get(key, {"resolved": 0, "detected": 0,
                                                "undetected": 0})
                    total = agg["resolved"] + agg["detected"] + agg["undetected"]
                    if total > 0:
                        resolved_vals.append(agg["resolved"])
                        detected_vals.append(agg["detected"])
                        undetected_vals.append(agg["undetected"])
                    else:
                        resolved_vals.append(0)
                        detected_vals.append(0)
                        undetected_vals.append(0)
                x = range(len(rates))
                ax.bar(x, resolved_vals, label="Resolved", alpha=0.8)
                ax.bar(x, detected_vals, bottom=resolved_vals,
                       label="Detected", alpha=0.8)
                ax.bar(x, undetected_vals,
                       bottom=[r + d for r, d in zip(resolved_vals, detected_vals)],
                       label="Undetected", alpha=0.8)
                ax.set_xticks(list(x))
                ax.set_xticklabels(rates, rotation=45)
                ax.set_ylabel("Byte count")
                ax.set_title(f"B={b}")
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            fig.suptitle(f"{ds} — Vote outcome breakdown", fontsize=14)
            fig.tight_layout()
            path = output_dir / f"{ds}_vote_outcome_breakdown.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Plot: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
