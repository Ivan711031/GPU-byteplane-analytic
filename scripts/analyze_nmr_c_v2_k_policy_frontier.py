#!/usr/bin/env python3
"""NMR-C v2 K-Aware Policy Frontier Analysis

Reads the pilot CSV, aggregates per (dataset, family, rate, k, policy) cell,
computes k-scoped verdict via matched-budget comparison on informative cells,
and writes summary CSV + verdict markdown.

Required CSV columns:
  dataset, n_rows, max_planes, k_values, policy_name, policy_type,
  truth_raw, clean_k_answer, faulted_k_answer,
  error_to_truth_abs, error_to_truth_rel,
  error_vs_clean_k_abs, error_vs_clean_k_rel,
  total_materialized_B, active_prefix_B, r_vector,
  fault_family, fault_rate, seed, gpu_count, latency_ms, effective_bytes
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_r_vector(s: str) -> list[int]:
    s = s.strip('"').strip()
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1]
        if not inner.strip():
            return []
        return [int(x.strip()) for x in inner.split(',')]
    return []

def severity_label(rate: float) -> str:
    if rate <= 1e-6:
        return "low"
    elif rate <= 1e-4:
        return "mid"
    else:
        return "high"

# ---------------------------------------------------------------------------
# Per-cell aggregation
# ---------------------------------------------------------------------------

def compute_aggregated_metrics(rows: list[dict]) -> dict:
    n = len(rows) if rows else 1
    errors_to_truth = sorted(r["error_to_truth_rel"] for r in rows)
    errors_vs_clean = sorted(r["error_vs_clean_k_rel"] for r in rows)
    p50_truth = errors_to_truth[len(errors_to_truth) // 2] if errors_to_truth else 0.0
    p99_idx = min(int(n * 0.99), n - 1)
    p99_truth = errors_to_truth[p99_idx] if errors_to_truth else 0.0
    max_truth = max(errors_to_truth) if errors_to_truth else 0.0
    p50_clean = errors_vs_clean[len(errors_vs_clean) // 2] if errors_vs_clean else 0.0
    p99_clean = errors_vs_clean[p99_idx] if errors_vs_clean else 0.0
    return {
        "n_cells": n,
        "relative_error_p50_to_truth": p50_truth,
        "relative_error_p99_to_truth": p99_truth,
        "max_relative_error_to_truth": max_truth,
        "relative_error_p50_vs_clean_k": p50_clean,
        "relative_error_p99_vs_clean_k": p99_clean,
    }

# ---------------------------------------------------------------------------
# Informative cell label
# ---------------------------------------------------------------------------

def label_cell(metrics: dict, rows: list[dict]) -> str:
    if metrics["n_cells"] == 0:
        return "too_sparse"
    truths = [r["error_to_truth_rel"] for r in rows]
    min_t = min(truths)
    max_t = max(truths)
    if max_t - min_t < 1e-16:
        return "non_separating"
    if max_t > 100.0:
        return "saturated"
    return "informative"

# ---------------------------------------------------------------------------
# Matched-budget verdict per (dataset, family, rate, k)
# ---------------------------------------------------------------------------

def verdict_per_cell(rows: list[dict]) -> dict:
    """For a set of rows sharing (dataset, family, rate, k), compare graded vs
    uniform at matching total_materialized_B levels."""
    # Group by total_materialized_B -> list of (policy_name, policy_type, rows)
    by_budget: dict[int, dict[str, Any]] = defaultdict(lambda: {"graded": [], "uniform": []})
    for r in rows:
        b = r["total_materialized_B"]
        ptype = r["policy_type"]
        if ptype not in ("graded", "uniform"):
            continue
        by_budget[b][ptype].append(r)

    comparisons = []
    for b, groups in sorted(by_budget.items()):
        g_rows = groups["graded"]
        u_rows = groups["uniform"]
        if not g_rows or not u_rows:
            continue  # no pair at this budget level

        # Collect all (g_policy, u_policy) pairs at this budget
        g_by_name = defaultdict(list)
        for r in g_rows:
            g_by_name[r["policy_name"]].append(r)
        u_by_name = defaultdict(list)
        for r in u_rows:
            u_by_name[r["policy_name"]].append(r)

        for g_name, g_grp in g_by_name.items():
            gm = compute_aggregated_metrics(g_grp)
            for u_name, u_grp in u_by_name.items():
                um = compute_aggregated_metrics(u_grp)
                g_err = gm["relative_error_p50_to_truth"]
                u_err = um["relative_error_p50_to_truth"]
                if abs(g_err - u_err) < 1e-16:
                    verdict = "tie"
                elif g_err < u_err:
                    verdict = "graded_wins"
                else:
                    verdict = "uniform_wins"
                comparisons.append({
                    "budget_B": b,
                    "graded_policy": g_name,
                    "uniform_policy": u_name,
                    "graded_error_p50": g_err,
                    "uniform_error_p50": u_err,
                    "graded_error_p99": gm["relative_error_p99_to_truth"],
                    "uniform_error_p99": um["relative_error_p99_to_truth"],
                    "graded_total_B": b,
                    "uniform_total_B": b,
                    "verdict": verdict,
                })

    # Aggregate verdict: overall support per cell
    wins_g = sum(1 for c in comparisons if c["verdict"] == "graded_wins")
    wins_u = sum(1 for c in comparisons if c["verdict"] == "uniform_wins")
    ties = sum(1 for c in comparisons if c["verdict"] == "tie")

    return {
        "comparisons": comparisons,
        "graded_wins": wins_g,
        "uniform_wins": wins_u,
        "ties": ties,
        "n_budgets": len(set(c["budget_B"] for c in comparisons)),
    }

# ---------------------------------------------------------------------------
# High-level k-scoped verdict rollup
# ---------------------------------------------------------------------------

def rollup_k_verdict(dataset: str, k: int, all_rows: list[dict]) -> dict:
    """Roll up verdicts across all (family, rate) cells for given dataset×k."""
    k_rows = [r for r in all_rows if r["dataset"] == dataset and r["k"] == k]

    # Group by (family, rate) — each is one "informative regime" cell
    cell_groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in k_rows:
        key = (r["fault_family"], float(r["fault_rate"]))
        cell_groups[key].append(r)

    # Also group by policy for per-policy pooled metrics (reference only)
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for r in k_rows:
        by_policy[r["policy_name"]].append(r)

    policy_metrics = {}
    for name, rows in by_policy.items():
        m = compute_aggregated_metrics(rows)
        m["total_materialized_B"] = rows[0]["total_materialized_B"]
        m["active_prefix_B"] = rows[0]["active_prefix_B"]
        m["policy_type"] = rows[0]["policy_type"]
        policy_metrics[name] = m

    # Per-cell analysis
    cell_results = []
    total_g = 0
    total_u = 0
    total_tie = 0
    informative_count = 0

    for (family, rate), cell_rows in sorted(cell_groups.items()):
        # Label this cell
        m = compute_aggregated_metrics(cell_rows)
        label = label_cell(m, cell_rows)

        if label != "informative":
            cell_results.append({
                "family": family, "rate": rate,
                "label": label,
                "verdict": None,
            })
            continue

        informative_count += 1
        v = verdict_per_cell(cell_rows)
        cell_results.append({
            "family": family, "rate": rate,
            "label": label,
            "verdict": v,
        })
        total_g += v["graded_wins"]
        total_u += v["uniform_wins"]
        total_tie += v["ties"]

    # Pooled comparison (reference: all cells together, for display)
    pooled_comparisons = []
    # Use actual matched budgets from any cell
    all_budgets = set()
    for cr in cell_results:
        if cr["verdict"]:
            for c in cr["verdict"]["comparisons"]:
                all_budgets.add(c["budget_B"])
    # For each budget level found, show representative pairs
    seen_pairs = set()
    for b in sorted(all_budgets):
        for cr in cell_results:
            if cr["verdict"]:
                for c in cr["verdict"]["comparisons"]:
                    if c["budget_B"] == b:
                        pair = (c["graded_policy"], c["uniform_policy"])
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            pooled_comparisons.append(c)

    return {
        "dataset": dataset,
        "k": k,
        "pooled_comparisons": pooled_comparisons,
        "policy_metrics": {k: {kk: vv for kk, vv in v.items() if kk != "n_cells"}
                           for k, v in policy_metrics.items()},
        "n_fault_families": len(set(r["fault_family"] for r in k_rows)),
        "cell_results": cell_results,
        "informative_cells": informative_count,
        "total_cells": len(cell_results),
        "graded_wins_total": total_g,
        "uniform_wins_total": total_u,
        "ties_total": total_tie,
    }

# ---------------------------------------------------------------------------
# CSV readers / writers
# ---------------------------------------------------------------------------

def load_pilot_csv(path: Path) -> list[dict]:
    rows = []
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row_clean: dict[str, Any] = {
                    "dataset": row["dataset"],
                    "n_rows": int(row["n_rows"]),
                    "max_planes": int(row["max_planes"]),
                    "k": int(row["k_values"]),
                    "policy_name": row["policy_name"],
                    "policy_type": row["policy_type"],
                    "truth_raw": float(row["truth_raw"]),
                    "clean_k_answer": float(row["clean_k_answer"]),
                    "faulted_k_answer": float(row["faulted_k_answer"]),
                    "error_to_truth_abs": float(row["error_to_truth_abs"]),
                    "error_to_truth_rel": float(row["error_to_truth_rel"]),
                    "error_vs_clean_k_abs": float(row["error_vs_clean_k_abs"]),
                    "error_vs_clean_k_rel": float(row["error_vs_clean_k_rel"]),
                    "total_materialized_B": int(row["total_materialized_B"]),
                    "active_prefix_B": int(row["active_prefix_B"]),
                    "r_vector": row["r_vector"],
                    "fault_family": row["fault_family"],
                    "fault_rate": float(row["fault_rate"]),
                    "seed": int(row["seed"]),
                    "gpu_count": int(row["gpu_count"]),
                }
                rows.append(row_clean)
            except (ValueError, KeyError) as e:
                print(f"WARNING: skipping row: {e}", file=sys.stderr)
                continue
    return rows

def write_summary(rows: list[dict], path: Path):
    """Group by (dataset, family, rate, k, policy) and write aggregated summary."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["fault_family"], r["fault_rate"],
               r["k"], r["policy_name"], r["policy_type"])
        groups[key].append(r)

    summary_rows: list[dict] = []
    for key, grp in sorted(groups.items()):
        dataset, family, rate, k, policy_name, policy_type = key
        m = compute_aggregated_metrics(grp)
        cell_type = label_cell(m, grp)
        r0 = grp[0]
        summary_rows.append({
            "dataset": dataset,
            "fault_family": family,
            "rate": f"{rate:.1e}",
            "severity": severity_label(rate),
            "k": k,
            "policy_name": policy_name,
            "policy_type": policy_type,
            "total_materialized_B": r0["total_materialized_B"],
            "active_prefix_B": r0["active_prefix_B"],
            "r_vector": r0["r_vector"],
            **m,
            "cell_label": cell_type,
        })

    fields = [
        "dataset", "fault_family", "rate", "severity", "k",
        "policy_name", "policy_type",
        "total_materialized_B", "active_prefix_B",
        "r_vector",
        "n_cells",
        "relative_error_p50_to_truth",
        "relative_error_p99_to_truth",
        "max_relative_error_to_truth",
        "relative_error_p50_vs_clean_k",
        "relative_error_p99_vs_clean_k",
        "cell_label",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in sorted(summary_rows, key=lambda r: (r["dataset"], r["k"], r["policy_name"], r["fault_family"], r["rate"])):
            w.writerow(row)
    return summary_rows

def write_verdict_md(datasets: list[str], k_values: list[int],
                     all_rows: list[dict], path: Path):
    """Write k-scoped verdict markdown with per-cell matched-budget analysis."""
    lines = []
    lines.append("# NMR-C v2 K-Aware Graded vs Uniform Correctness Frontier — Verdict")
    lines.append("")
    lines.append(f"**Pilot datasets**: {', '.join(datasets)}")
    lines.append(f"**k sweep**: {', '.join(str(k) for k in k_values)}")
    lines.append(f"**Policies**: graded_B0..B16, uniform_full_r1/r2/r3")
    lines.append(f"**Fault families**: F1-F8")
    lines.append(f"**Total rows**: {len(all_rows)}")
    lines.append("")
    lines.append("**Methodology**: For each informative (dataset, family, rate, k) cell, "
                 "graded and uniform policies are compared at matching "
                 "`total_materialized_B` (read from CSV, not hardcoded). "
                 "Cells labeled `non_separating` or `saturated` are excluded "
                 "from support/no-support counting.")
    lines.append("")

    for dataset in datasets:
        ds_rows = [r for r in all_rows if r["dataset"] == dataset]
        lines.append(f"## Dataset: {dataset}")
        lines.append("")

        for k in k_values:
            v = rollup_k_verdict(dataset, k, all_rows)
            lines.append(f"### k={k}")
            lines.append("")
            lines.append(f"**Informative cells**: {v['informative_cells']}/{v['total_cells']}")
            lines.append("")

            if v["pooled_comparisons"]:
                lines.append("**Representative budget pairs** (at equal `total_materialized_B`):")
                lines.append("")
                lines.append("| Budget B | Graded | Uniform | Graded err p50 | Uniform err p50 |")
                lines.append("|---|---|---|---|---|")
                seen = set()
                for c in v["pooled_comparisons"]:
                    key = (c["budget_B"], c["graded_policy"], c["uniform_policy"])
                    if key in seen:
                        continue
                    seen.add(key)
                    g_str = f"{c['graded_error_p50']:.6e}"
                    u_str = f"{c['uniform_error_p50']:.6e}"
                    lines.append(f"| B={c['budget_B']} | {c['graded_policy']} | {c['uniform_policy']} | {g_str} | {u_str} |")
                lines.append("")

            # Per-cell verdict breakdown
            lines.append("**Per-cell verdict** (dataset × family × rate × k):")
            lines.append("")
            lines.append("| Family | Rate | Label | Graded wins | Uniform wins | Ties | Budgets |")
            lines.append("|---|---|---|---|---|---|---|")
            for cr in v["cell_results"]:
                if cr["verdict"] is None:
                    lines.append(f"| {cr['family']} | {cr['rate']:.0e} | {cr['label']} | — | — | — | — |")
                else:
                    lines.append(f"| {cr['family']} | {cr['rate']:.0e} | {cr['label']} | "
                                 f"{cr['verdict']['graded_wins']} | {cr['verdict']['uniform_wins']} | "
                                 f"{cr['verdict']['ties']} | {cr['verdict']['n_budgets']} |")
            lines.append("")

            # Policy metrics table
            lines.append("**Per-policy pooled metrics** (reference):")
            lines.append("")
            lines.append("| Policy | Type | Total B | err p50 truth | err p99 truth | err p50 clean | err p99 clean |")
            lines.append("|---|---|---|---|---|---|---|")
            for pname, pm in sorted(v["policy_metrics"].items()):
                lines.append(
                    f"| {pname} | {pm.get('policy_type','')} | {pm.get('total_materialized_B','')} | "
                    f"{pm.get('relative_error_p50_to_truth',0):.6e} | "
                    f"{pm.get('relative_error_p99_to_truth',0):.6e} | "
                    f"{pm.get('relative_error_p50_vs_clean_k',0):.6e} | "
                    f"{pm.get('relative_error_p99_vs_clean_k',0):.6e} |"
                )
            lines.append("")

        # Headline per-dataset
        lines.append(f"### {dataset} — Headline Verdict Summary")
        lines.append("")
        for k in k_values:
            v = rollup_k_verdict(dataset, k, all_rows)
            g = v["graded_wins_total"]
            u = v["uniform_wins_total"]
            t = v["ties_total"]
            info = v["informative_cells"]
            lines.append(f"- **k={k}**: {info} informative cells → "
                         f"graded wins={g}, uniform wins={u}, ties={t}")
        lines.append("")

    lines.append("## Notes")
    lines.append("- Comparisons are made at equal `total_materialized_B` (matched from CSV, not hardcoded)")
    lines.append("- Only `informative` cells contribute to support/no-support counting")
    lines.append("- `non_separating` / `saturated` cells are excluded from headline verdict")
    lines.append("- `error_to_truth_rel` includes byteplane encoding approximation error")
    lines.append("- `error_vs_clean_k_rel` isolates reliability effect (faulted vs clean byteplane sum)")
    lines.append("- `total_materialized_B` = sum of extra replica copies across all planes")
    lines.append("- Faults targeting non-existent replicas are silently dropped (no r0 fallback)")

    path.write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="NMR-C v2 K-Aware Policy Frontier Analysis")
    ap.add_argument("--pilot-csv", type=str, required=True, nargs="+",
                    help="Pilot CSV files (multiple for merge)")
    ap.add_argument("--output-dir", type=str, required=True,
                    help="Output directory for summary + verdict")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and merge
    all_rows: list[dict] = []
    for csv_path in args.pilot_csv:
        p = Path(csv_path)
        if not p.is_file():
            print(f"WARNING: {p} not found, skipping")
            continue
        rows = load_pilot_csv(p)
        print(f"Loaded {len(rows)} rows from {p.name}")
        all_rows.extend(rows)

    if not all_rows:
        print("ERROR: no data loaded")
        sys.exit(1)

    # Deduplicate by (dataset, family, rate, k, policy, seed)
    seen = set()
    deduped: list[dict] = []
    for r in all_rows:
        key = (r["dataset"], r["fault_family"], r["fault_rate"],
               r["k"], r["policy_name"], r["seed"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    print(f"Total rows: {len(all_rows)} → deduped: {len(deduped)}")

    # Write combined pilot CSV
    pilot_path = out_dir / "nmr_c_v2_k_policy_frontier_pilot.csv"
    pilot_fields = [
        "dataset", "n_rows", "max_planes", "k",
        "policy_name", "policy_type",
        "truth_raw", "clean_k_answer", "faulted_k_answer",
        "error_to_truth_abs", "error_to_truth_rel",
        "error_vs_clean_k_abs", "error_vs_clean_k_rel",
        "total_materialized_B", "active_prefix_B",
        "r_vector",
        "fault_family", "fault_rate", "seed",
        "gpu_count",
    ]
    with pilot_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pilot_fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(deduped, key=lambda x: (x["dataset"], x["k"], x["policy_name"], x["fault_family"], x["fault_rate"], x["seed"])):
            r["k"] = r["k"]
            w.writerow(r)
    print(f"Pilot CSV: {pilot_path} ({len(deduped)} rows)")

    # Write summary CSV
    datasets = sorted(set(r["dataset"] for r in deduped))
    k_values = sorted(set(r["k"] for r in deduped))
    summary_path = out_dir / "nmr_c_v2_k_policy_frontier_summary.csv"
    summary_rows = write_summary(deduped, summary_path)
    print(f"Summary CSV: {summary_path} ({len(summary_rows)} rows)")

    # Write verdict markdown
    verdict_path = out_dir / "nmr_c_v2_k_policy_frontier_verdict.md"
    write_verdict_md(datasets, k_values, deduped, verdict_path)
    print(f"Verdict: {verdict_path}")

    # Quick stats
    for ds in datasets:
        ds_rows = [r for r in deduped if r["dataset"] == ds]
        print(f"\n{ds}: {len(ds_rows)} rows, {len(set(r['k'] for r in ds_rows))} k values, "
              f"{len(set(r['policy_name'] for r in ds_rows))} policies, "
              f"{len(set(r['fault_family'] for r in ds_rows))} fault families")

if __name__ == "__main__":
    main()
