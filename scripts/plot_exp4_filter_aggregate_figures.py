#!/usr/bin/env python3
"""Exp4 Filter+Aggregate — paper-facing figure generation.

Generates three figures and one cuDF external-baseline table from the
formal sweep CSV and the cuDF baseline CSV.

Usage:
  python scripts/plot_exp4_filter_aggregate_figures.py \
    --sweep-csv results/exp4_filter_aggregate/formal_sweep_20260517_125654_job53158_NVIDIAH200/sweep_results.csv \
    --cudf-csv results/exp4_filter_aggregate/cudf_baseline_20260517_183723_job53522_NVIDIAH200/cudf_filter_aggregate_representative.csv \
    --out-dir results/paper_v1/plots_exp4_filter_aggregate
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASETS = ["uniform", "heavy_tailed", "sensor", "zipfian"]

# Representative k ranges at s≈50%
K_RANGES: dict[str, tuple[int, int]] = {
    "sensor":       (1, 5),
    "uniform":      (1, 6),
    "heavy_tailed": (4, 8),
    "zipfian":      (3, 8),
}

DISPLAY_NAMES: dict[str, str] = {
    "uniform":       "Uniform",
    "heavy_tailed":  "Heavy-Tailed",
    "sensor":        "Sensor",
    "zipfian":       "Zipfian",
}

COLORS: dict[str, str] = {
    "uniform":       "#1f77b4",
    "heavy_tailed":  "#ff7f0e",
    "sensor":        "#2ca02c",
    "zipfian":       "#d62728",
}

MARKERS: dict[str, str] = {
    "uniform":       "o",
    "heavy_tailed":  "s",
    "sensor":        "^",
    "zipfian":       "D",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tag_selectivity(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'target_sel' column: nominal selectivity label closest to actual."""
    NOMINAL = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    targets = []
    for _, row in df.iterrows():
        s = float(row["selectivity"])
        closest = min(NOMINAL, key=lambda t: abs(s - t))
        targets.append(closest)
    df = df.copy()
    df["target_sel"] = targets
    return df


def _select_s50_rows(df: pd.DataFrame) -> pd.DataFrame:
    """For each dataset, pick the threshold whose selectivity is closest to 0.50,
    then return all rows at that (dataset, threshold)."""
    df = _tag_selectivity(df)
    frames = []
    for ds in DATASETS:
        ds_df = df[df["dataset"] == ds].copy()
        if ds_df.empty:
            continue
        # find threshold closest to sel=0.50
        ds_df["sel_dist"] = ds_df["selectivity"].astype(float).apply(
            lambda s: abs(s - 0.50))
        best = ds_df.loc[ds_df["sel_dist"].idxmin()]
        thresh = best["threshold"]
        s50 = ds_df[ds_df["threshold"] == thresh].copy()
        frames.append(s50)
    return pd.concat(frames, ignore_index=True)


def _filter_k_range(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows within the dataset-specific k range."""
    frames = []
    for ds in DATASETS:
        k_lo, k_hi = K_RANGES[ds]
        sub = df[(df["dataset"] == ds) &
                 (df["max_filter_planes"] >= k_lo) &
                 (df["max_filter_planes"] <= k_hi)].copy()
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def _ensure_float_cols(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """Convert columns to float, coercing errors."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Figure 1 — Fused Throughput vs k at s≈50%
# ---------------------------------------------------------------------------

def fig1_throughput_vs_k(df: pd.DataFrame, out_path: str) -> None:
    """Fused rows/sec vs k with raw-CUDA reference line."""
    df = _ensure_float_cols(df, "rows_per_sec", "raw_baseline_rows_per_sec",
                            "max_filter_planes")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for ds in DATASETS:
        sub = df[df["dataset"] == ds].sort_values("max_filter_planes")
        if sub.empty:
            continue
        x = sub["max_filter_planes"].values
        y = sub["rows_per_sec"].values / 1e9
        ax.plot(x, y, marker=MARKERS.get(ds, "o"), color=COLORS.get(ds),
                label=DISPLAY_NAMES.get(ds, ds), linewidth=1.5, markersize=7)

    # Raw CUDA baseline (horizontal)
    raw_rows = df["raw_baseline_rows_per_sec"].dropna()
    if not raw_rows.empty:
        raw_avg = raw_rows.mean() / 1e9
        ax.axhline(y=raw_avg, color="grey", linestyle="--", linewidth=1.0,
                   label=f"Raw CUDA ({raw_avg:.1f}B)")

    ax.set_xlabel("Byte-planes filtered (k)", fontsize=11)
    ax.set_ylabel("Throughput (B rows/s)", fontsize=11)
    ax.set_title("Fused Filter+Aggregate Throughput vs k at s ≈ 50%", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Fused vs Deferred-Gather Speedup at s≈50%
# ---------------------------------------------------------------------------

def fig2_speedup_fused_vs_dg(df: pd.DataFrame, out_path: str) -> None:
    """speedup_fused_vs_dg vs k."""
    df = _ensure_float_cols(df, "speedup_fused_vs_dg", "max_filter_planes")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for ds in DATASETS:
        sub = df[df["dataset"] == ds].sort_values("max_filter_planes")
        if sub.empty:
            continue
        x = sub["max_filter_planes"].values
        y = sub["speedup_fused_vs_dg"].values
        ax.plot(x, y, marker=MARKERS.get(ds, "o"), color=COLORS.get(ds),
                label=DISPLAY_NAMES.get(ds, ds), linewidth=1.5, markersize=7)

    # Parity line
    ymax = df["speedup_fused_vs_dg"].max() * 1.05
    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=1.0, alpha=0.7,
               label="Parity (1×)")

    ax.set_xlabel("Byte-planes filtered (k)", fontsize=11)
    ax.set_ylabel("Speedup: Fused / Deferred-Gather", fontsize=11)
    ax.set_title("Fused vs Deferred-Gather Speedup at s ≈ 50%", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=ymax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Accuracy Drift vs k at s≈50%
# ---------------------------------------------------------------------------

def fig3_raw_drift_vs_k(df: pd.DataFrame, out_path: str) -> None:
    """raw_count_rel_err and raw_sum_rel_err vs k — symlog scale so zero
    errors plot honestly at y=0 instead of being clamped to a floor."""
    df = _ensure_float_cols(df, "raw_count_rel_err", "raw_sum_rel_err",
                            "max_filter_planes")

    fig, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    # symlog: linear below linthresh, logarithmic above.
    # linthresh=1e-15 is below every non-zero error in our data,
    # so the visual is effectively a log plot with zero at the axis.
    LINTHRESH = 1e-15

    for ax, col, title in [
        (ax_c, "raw_count_rel_err", "Count Rel. Error"),
        (ax_s, "raw_sum_rel_err", "Sum Rel. Error"),
    ]:
        for ds in DATASETS:
            sub = df[df["dataset"] == ds].sort_values("max_filter_planes")
            if sub.empty:
                continue
            x = sub["max_filter_planes"].values
            y = sub[col].values  # zero is zero — no clamping needed
            ax.plot(x, y, marker=MARKERS.get(ds, "o"), color=COLORS.get(ds),
                    label=DISPLAY_NAMES.get(ds, ds), linewidth=1.5, markersize=7)

        ax.set_yscale("symlog", linthresh=LINTHRESH)
        ax.set_xlabel("Byte-planes filtered (k)", fontsize=10)
        ax.set_ylabel("Relative Error (symlog scale)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim(left=0)

    handles, labels = ax_c.get_legend_handles_labels()
    # Legend below both subplots so it never overlaps axis labels
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               framealpha=0.8, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Encoded-vs-Raw Accuracy Drift at s ≈ 50%", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# cuDF external-baseline table (Markdown + optional CSV)
# ---------------------------------------------------------------------------

def write_cudf_table(cudf_csv_path: str, out_dir: str,
                     cudf_strategy: str = "compact_then_reduce",
                     cudf_timing_policy: str = "gpu_hot_path") -> None:
    """Read the cuDF CSV and write a paper-facing Markdown table (manual fmt).

    Filters rows to *cudf_strategy* and *cudf_timing_policy* when those columns
    are present, so multi-variant CSVs only show the canonical baseline.
    """
    cudf = pd.read_csv(cudf_csv_path, dtype={"raw_cpu_or_raw_cuda_count_match": str})

    # Filter to canonical baseline when strategy/timing_policy columns exist
    if "strategy" in cudf.columns:
        cudf = cudf[cudf["strategy"] == cudf_strategy]
    if "timing_policy" in cudf.columns:
        cudf = cudf[cudf["timing_policy"] == cudf_timing_policy]

    if cudf.empty:
        print("  WARNING: cuDF CSV empty after strategy/timing_policy filter.  "
              "Skipping table generation.")
        return

    cudf = _ensure_float_cols(
        cudf, "cudf_filter_aggregate_ms_median", "fused_ms_per_iter",
        "raw_cuda_ms_per_iter", "speedup_fused_vs_cudf",
        "speedup_raw_cuda_vs_cudf", "raw_cpu_or_raw_cuda_sum_rel_err",
        "target_selectivity")

    rows_data = []
    for _, r in cudf.iterrows():
        rows_data.append({
            "ds": DISPLAY_NAMES.get(r["dataset"], r["dataset"]),
            "k": int(r["max_filter_planes"]),
            "sel": float(r["target_selectivity"]),
            "cudf_ms": float(r["cudf_filter_aggregate_ms_median"]),
            "fused_ms": float(r["fused_ms_per_iter"]),
            "raw_ms": float(r["raw_cuda_ms_per_iter"]),
            "sp_fused": float(r["speedup_fused_vs_cudf"]),
            "sp_raw": float(r["speedup_raw_cuda_vs_cudf"]),
            "cnt": str(r.get("raw_cpu_or_raw_cuda_count_match", "")),
            "sum_err": float(r["raw_cpu_or_raw_cuda_sum_rel_err"]),
        })

    # Sort by dataset name then k
    rows_data.sort(key=lambda x: (x["ds"], x["k"]))

    header = ("| Dataset | k | Selectivity | cuDF median (ms) | Fused (ms) | "
              "Raw CUDA (ms) | Fused / cuDF | Raw / cuDF | Count match | Sum rel err |")
    sep =    ("|---------|---|:----------:|:-----------------:|:----------:|"
              ":-------------:|:-------------:|:----------:|:-----------:|:-----------:|")

    lines = [
        "# cuDF External Baseline — Filter+Aggregate at s≈50%\n",
        header,
        sep,
    ]
    for r in rows_data:
        lines.append(
            f"| {r['ds']} | {r['k']} | {r['sel']:.3f} | {r['cudf_ms']:.3f} | "
            f"{r['fused_ms']:.3f} | {r['raw_ms']:.3f} | "
            f"{r['sp_fused']:.1f}× | {r['sp_raw']:.1f}× | "
            f"{r['cnt']} | {r['sum_err']:.1e} |"
        )

    lines.append(
        "\n*Count match: true = exact integer match with raw CUDA baseline. "
        "Sum rel err: |cuDF sum − raw baseline sum| / |raw baseline sum|.*\n"
    )

    md_path = os.path.join(out_dir, "table_cudf_external_baseline.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # CSV
    csv_path = os.path.join(out_dir, "table_cudf_external_baseline.csv")
    cudf_out = pd.DataFrame(rows_data)
    cudf_out.to_csv(csv_path, index=False)

    print(f"  Wrote {md_path}")
    print(f"  Wrote {csv_path}")


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def write_readme(out_dir: str, sweep_csv: str, cudf_csv: str) -> None:
    readme = os.path.join(out_dir, "README.md")
    with open(readme, "w") as fh:
        fh.write(f"""# Exp4 Filter+Aggregate — Paper-Facing Figures

**Generated by:** `scripts/plot_exp4_filter_aggregate_figures.py`

## Source Data

| Artifact | Path |
|----------|------|
| Formal sweep CSV | `{sweep_csv}` |
| cuDF baseline CSV | `{cudf_csv}` |

## Figures

| File | Content |
|------|---------|
| `fig1_throughput_vs_k_s50.png` | Fused throughput (B rows/s) vs k at s≈50%, with raw-CUDA reference |
| `fig2_speedup_fused_vs_dg_s50.png` | Fused / deferred-gather speedup vs k at s≈50% |
| `fig3_raw_drift_vs_k_s50.png` | Encoded-vs-raw count & sum relative error vs k (symlog scale) |

## Tables

| File | Content |
|------|---------|
| `table_cudf_external_baseline.md` | cuDF external baseline — paper-facing table |
| `table_cudf_external_baseline.csv` | Same data as machine-readable CSV |

## Regeneration

```bash
python3 scripts/plot_exp4_filter_aggregate_figures.py \\
  --sweep-csv {sweep_csv} \\
  --cudf-csv {cudf_csv} \\
  --out-dir {out_dir}
```
""")
    print(f"  Wrote {readme}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp4 Filter+Aggregate paper-facing figures")
    parser.add_argument("--sweep-csv", required=True,
                        help="Path to formal sweep sweep_results.csv")
    parser.add_argument("--cudf-csv", required=True,
                        help="Path to cuDF baseline CSV")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for PNGs and tables")
    parser.add_argument(
        "--cudf-strategy", default="compact_then_reduce",
        choices=["compact_then_reduce", "masked_reduce"],
        help="cuDF strategy to select for the external-baseline table "
             "(default: compact_then_reduce)")
    parser.add_argument(
        "--cudf-timing-policy", default="gpu_hot_path",
        choices=["gpu_hot_path", "query_result_latency"],
        help="cuDF timing policy to select for the external-baseline table "
             "(default: gpu_hot_path)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -- Load formal sweep ----------------------------------------------------
    df = pd.read_csv(args.sweep_csv)
    print(f"Loaded {len(df)} rows from sweep CSV")

    # Required numeric columns check
    required = ["rows_per_sec", "raw_baseline_rows_per_sec",
                "speedup_fused_vs_dg", "raw_count_rel_err", "raw_sum_rel_err",
                "max_filter_planes", "selectivity", "dataset", "threshold",
                "ms_per_iter", "raw_baseline_ms_per_iter"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: sweep CSV missing columns: {missing}")

    # -- Select s ≈ 50% rows -------------------------------------------------
    s50 = _select_s50_rows(df)
    s50 = _filter_k_range(s50)
    print(f"Selected {len(s50)} rows at s≈50% across {s50['dataset'].nunique()} datasets")

    if s50.empty:
        raise SystemExit("ERROR: no rows selected for s≈50%. Check sweep CSV.")

    # -- Generate figures ----------------------------------------------------
    print("\nGenerating figures …")
    fig1_throughput_vs_k(s50, os.path.join(args.out_dir, "fig1_throughput_vs_k_s50.png"))
    fig2_speedup_fused_vs_dg(s50, os.path.join(args.out_dir, "fig2_speedup_fused_vs_dg_s50.png"))
    fig3_raw_drift_vs_k(s50, os.path.join(args.out_dir, "fig3_raw_drift_vs_k_s50.png"))

    # -- cuDF table ----------------------------------------------------------
    print("\nGenerating cuDF table …")
    write_cudf_table(args.cudf_csv, args.out_dir,
                     cudf_strategy=args.cudf_strategy,
                     cudf_timing_policy=args.cudf_timing_policy)

    # -- README --------------------------------------------------------------
    print("\nGenerating README …")
    write_readme(args.out_dir, args.sweep_csv, args.cudf_csv)

    print(f"\nDone.  All outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
