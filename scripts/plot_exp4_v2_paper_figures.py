#!/usr/bin/env python3
"""Generate Exp4-B1 v2 paper-facing analysis plots from existing CSVs.

Usage:
  python scripts/plot_exp4_v2_paper_figures.py \
    --b1-dir results/exp4/b1_20260510_174435_job44743_NVIDIAH200 \
    --raw-anchor-dir results/exp4/b1_20260510_190958_job45163_NVIDIAH200 \
    --paper-table results/paper_v1/artifact_size_fidelity_transfer.csv \
    --raw-transfer results/paper_v1/raw_fp64_transfer_baseline.csv \
    --out-dir results/paper_v1/plots_exp4_v2
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

# ===========================================================================
# Artifact taxonomy
# ===========================================================================
MAINLINE = {
    ("uniform", "p2"), ("uniform", "p4"), ("uniform", "p6"),
    ("uniform", "p8"), ("uniform", "p10"),
    ("sensor", "p2"), ("sensor", "p4"), ("sensor", "p6"),
    ("sensor", "p8"), ("sensor", "p10"),
    ("heavy_tailed", "p6"),
    ("zipfian", "p6"), ("zipfian", "p8"),
}

REJECT = {
    ("heavy_tailed", "p2"), ("heavy_tailed", "p3"),
    ("zipfian", "p2"),
}

SIDE_STUDY = {
    ("heavy_tailed", "p4"), ("heavy_tailed", "p5"),
    ("zipfian", "p4"),
}

def artifact_label_from_root(dataset: str, artifact_root: str) -> str:
    name = Path(artifact_root).name
    prefix = f"{dataset}_"
    if name.startswith(prefix):
        return name[len(prefix):]
    return name

def taxon(dataset: str, label: str) -> str:
    key = (dataset, label)
    if key in MAINLINE:
        return "mainline"
    if key in REJECT:
        return "reject"
    if key in SIDE_STUDY:
        return "side_study"
    return "other"


# ===========================================================================
# CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--b1-dir", required=True, type=Path,
                   help="Job 44743 output directory")
    p.add_argument("--raw-anchor-dir", required=True, type=Path,
                   help="Job 45163 output directory")
    p.add_argument("--paper-table", required=True, type=Path,
                   help="results/paper_v1/artifact_size_fidelity_transfer.csv")
    p.add_argument("--raw-transfer", required=True, type=Path,
                   help="results/paper_v1/raw_fp64_transfer_baseline.csv")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output directory for plots")
    return p.parse_args()


# ===========================================================================
# Validation helpers
# ===========================================================================
def check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required input not found: {path}")

def check_cols(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


# ===========================================================================
# Mainline filter
# ===========================================================================
def add_label_and_taxon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["artifact_label"] = df.apply(
        lambda r: artifact_label_from_root(r["dataset"], r["artifact_root"]), axis=1)
    df["taxon"] = df.apply(lambda r: taxon(r["dataset"], r["artifact_label"]), axis=1)
    return df

def filter_mainline(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["taxon"] == "mainline"].copy()


# ===========================================================================
# Plot 1: U(k) / n vs k
# ===========================================================================
def plot_u_fraction_vs_k(sweep: pd.DataFrame, out_dir: Path) -> None:
    df = filter_mainline(add_label_and_taxon(sweep))
    df["u_frac"] = df["uncertain"] / df["n"]
    df["k"] = df["max_filter_planes"]

    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, ds in zip(axes, sorted(datasets)):
        sub = df[df["dataset"] == ds]
        for label, grp in sub.groupby("artifact_label"):
            grp = grp.sort_values("k")
            ax.plot(grp["k"], grp["u_frac"], marker="o", label=label)
        ax.set_title(ds)
        ax.set_xlabel("k (planes read)")
        ax.set_ylabel("U / n")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for i in range(len(datasets), len(axes)):
        axes[i].set_visible(False)
    fig.suptitle("Uncertain Fraction vs Progressive Depth", fontsize=14)
    fig.tight_layout()
    path = out_dir / "u_fraction_vs_k_mainline.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


# ===========================================================================
# Plot 2: rows_per_sec vs k
# ===========================================================================
def plot_throughput_vs_k(sweep: pd.DataFrame, out_dir: Path) -> None:
    df = filter_mainline(add_label_and_taxon(sweep))
    df["k"] = df["max_filter_planes"]

    if (df["rows_per_sec"] <= 0).any():
        print("  [warn] some rows_per_sec <= 0, filtering them out")
        df = df[df["rows_per_sec"] > 0].copy()

    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for ax, ds in zip(axes, sorted(datasets)):
        sub = df[df["dataset"] == ds]
        for label, grp in sub.groupby("artifact_label"):
            grp = grp.sort_values("k")
            ax.plot(grp["k"], grp["rows_per_sec"] / 1e9, marker="o", label=label)
        ax.set_title(ds)
        ax.set_xlabel("k (planes read)")
        ax.set_ylabel("rows/sec (billions)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for i in range(len(datasets), len(axes)):
        axes[i].set_visible(False)
    fig.suptitle("Throughput vs Progressive Depth", fontsize=14)
    fig.tight_layout()
    path = out_dir / "throughput_vs_k_mainline.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


# ===========================================================================
# Plot 3: U(k) / n vs rows_per_sec
# ===========================================================================
def plot_u_vs_throughput(sweep: pd.DataFrame, out_dir: Path) -> None:
    df = filter_mainline(add_label_and_taxon(sweep))
    df["u_frac"] = df["uncertain"] / df["n"]
    df = df[df["rows_per_sec"] > 0].copy()

    markers = {"p2": "o", "p4": "s", "p6": "D", "p8": "^", "p10": "v"}
    colors = {"uniform": "C0", "sensor": "C1", "heavy_tailed": "C2", "zipfian": "C3"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for (ds, label), grp in df.groupby(["dataset", "artifact_label"]):
        ax.scatter(grp["u_frac"], grp["rows_per_sec"] / 1e9,
                   marker=markers.get(label, "o"),
                   color=colors.get(ds, "gray"),
                   label=f"{ds} {label}", s=20, alpha=0.7)
    ax.set_xlabel("U / n (uncertain fraction)")
    ax.set_ylabel("rows/sec (billions)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Uncertain Fraction vs Throughput", fontsize=14)
    fig.tight_layout()
    path = out_dir / "u_fraction_vs_throughput_mainline.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


# ===========================================================================
# Plot 4: epsilon -> k* heatmap
# ===========================================================================
def plot_epsilon_kstar_heatmap(ek_csv: Path, out_dir: Path) -> None:
    ek = pd.read_csv(ek_csv)
    check_cols(ek, ["dataset", "artifact", "selectivity", "epsilon", "kstar"], "epsilon_kstar")
    # Filter mainline
    ek["key"] = ek.apply(lambda r: (r["dataset"], r["artifact"]), axis=1)
    ek = ek[ek["key"].isin(MAINLINE)].copy()

    # Create a label: "dataset/artifact s=sel"
    ek["row_label"] = ek.apply(
        lambda r: f"{r['dataset'][:4]}/{r['artifact']} s={r['selectivity']:.2f}", axis=1)

    pivot = ek.pivot_table(index="row_label", columns="epsilon", values="kstar", aggfunc="first")
    # Sort epsilons
    eps_order = sorted(ek["epsilon"].unique())
    pivot = pivot[eps_order]

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.3)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r", interpolation="nearest")
    ax.set_xticks(range(len(eps_order)))
    ax.set_xticklabels([str(e) for e in eps_order], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    ax.set_xlabel("ε (error tolerance)")
    ax.set_ylabel("dataset/artifact/selectivity")
    fig.colorbar(im, ax=ax, label="k*")
    fig.suptitle("Epsilon -> k* (planes needed)", fontsize=14)
    fig.tight_layout()
    path = out_dir / "epsilon_to_kstar_heatmap_mainline.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


# ===========================================================================
# Plot 5: COUNT drift vs selectivity
# ===========================================================================
def plot_count_drift(pt: pd.DataFrame, out_dir: Path) -> None:
    check_cols(pt, ["dataset", "artifact_label", "target_selectivity",
                     "selectivity_drift_pp", "count_verdict"], "paper_table")
    df = pt.copy()
    df = df[df.apply(lambda r: (r["dataset"], r["artifact_label"]) in MAINLINE, axis=1)]

    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    verdict_colors = {"acceptable": "green", "caution": "orange", "catastrophic": "red"}
    for ax, ds in zip(axes, sorted(datasets)):
        sub = df[df["dataset"] == ds]
        for label, grp in sub.groupby("artifact_label"):
            grp = grp.sort_values("target_selectivity")
            colors = [verdict_colors.get(v, "gray") for v in grp["count_verdict"]]
            ax.scatter(grp["target_selectivity"], grp["selectivity_drift_pp"],
                       c=colors, label=label, s=30)
            ax.plot(grp["target_selectivity"], grp["selectivity_drift_pp"],
                    alpha=0.3)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(ds)
        ax.set_xlabel("target selectivity (%)")
        ax.set_ylabel("drift (pp)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for i in range(len(datasets), len(axes)):
        axes[i].set_visible(False)
    fig.suptitle("Encoded COUNT Drift vs Selectivity (mainline)", fontsize=14)
    fig.tight_layout()
    path = out_dir / "count_drift_vs_selectivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


# ===========================================================================
# Plot 6: bytes_per_row vs H2D time
# ===========================================================================
def plot_bytes_vs_h2d(pt: pd.DataFrame, raw_baseline: pd.DataFrame, out_dir: Path) -> None:
    check_cols(pt, ["dataset", "artifact_label", "bytes_per_row",
                     "cudaMemcpy_ms", "count_verdict"], "paper_table")
    # One row per artifact (deduplicate)
    art_df = pt[["dataset", "artifact_label", "bytes_per_row",
                  "cudaMemcpy_ms", "count_verdict"]].drop_duplicates(
        subset=["dataset", "artifact_label"])

    colors = {"uniform": "C0", "sensor": "C1", "heavy_tailed": "C2", "zipfian": "C3"}
    marker_map = {"acceptable": "o", "caution": "s", "catastrophic": "X"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in art_df.iterrows():
        ds = row["dataset"]
        ax.scatter(row["bytes_per_row"], row["cudaMemcpy_ms"],
                   marker=marker_map.get(row["count_verdict"], "o"),
                   color=colors.get(ds, "gray"),
                   s=60, alpha=0.8,
                   label=f"{ds} {row['artifact_label']}")

    # Raw baseline: one point per dataset (all ~8 B/row)
    for _, r in raw_baseline.iterrows():
        ax.scatter(8.0, r["cudaMemcpy_ms"],
                   marker="*", color="red", s=150,
                   label=f"raw {r['dataset']}" if _ == 0 else "")

    # Horizontal guide at ~82 ms
    ax.axhline(82.2, color="red", linestyle="--", alpha=0.4, label="raw baseline ~82 ms")

    ax.set_xlabel("bytes_per_row")
    ax.set_ylabel("cudaMemcpy_ms (HtoD)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Artifact Size vs H2D Transfer Time", fontsize=14)
    fig.tight_layout()
    path = out_dir / "bytes_per_row_vs_h2d_time.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


# ===========================================================================
# Plot 7: artifact tradeoff scatter
# ===========================================================================
def plot_tradeoff(pt: pd.DataFrame, raw_baseline: pd.DataFrame, out_dir: Path) -> None:
    check_cols(pt, ["dataset", "artifact_label", "bytes_per_row",
                     "cudaMemcpy_ms", "selectivity_drift_pp",
                     "count_verdict"], "paper_table")

    # Aggregate per artifact: max drift, representative transfer, speedup
    rows = []
    for (ds, label), grp in pt.groupby(["dataset", "artifact_label"]):
        bpr = grp["bytes_per_row"].iloc[0]
        ms = grp["cudaMemcpy_ms"].iloc[0]
        max_drift = grp["selectivity_drift_pp"].abs().max()
        worst_verdict = grp["count_verdict"].iloc[
            np.argmax(grp["selectivity_drift_pp"].abs().values)]
        raw_ms = raw_baseline["cudaMemcpy_ms"].mean()
        speedup = raw_ms / ms if ms > 0 else 0
        t = taxon(ds, label)
        rows.append({
            "dataset": ds, "artifact_label": label,
            "bytes_per_row": bpr, "max_drift_pp": max_drift,
            "worst_verdict": worst_verdict, "transfer_ms": ms,
            "speedup": speedup, "taxon": t,
        })
    agg = pd.DataFrame(rows)

    colors = {"uniform": "C0", "sensor": "C1", "heavy_tailed": "C2", "zipfian": "C3"}
    taxon_marker = {"mainline": "o", "side_study": "s", "reject": "X", "other": "d"}

    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in agg.iterrows():
        ax.scatter(row["bytes_per_row"], row["max_drift_pp"],
                   s=row["speedup"] * 40,
                   marker=taxon_marker.get(row["taxon"], "o"),
                   color=colors.get(row["dataset"], "gray"),
                   alpha=0.7,
                   label=f"{row['dataset'][:4]} {row['artifact_label']}")

    # Legend for size
    handles = []
    for ds in sorted(colors):
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=colors[ds], markersize=8, label=ds))
    for tax, m in taxon_marker.items():
        handles.append(Line2D([0], [0], marker=m, color="gray",
                              markersize=8, label=tax, linestyle="none"))
    # Speedup scale note
    handles.append(Line2D([0], [0], marker="o", color="gray",
                          markersize=12, label="speedup ~ marker size",
                          markerfacecolor="lightgray", linestyle="none"))
    ax.legend(handles=handles, fontsize=8, ncol=2)

    ax.set_xlabel("bytes_per_row")
    ax.set_ylabel("max |COUNT drift| (pp)")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.3, label="catastrophic threshold")
    ax.axhline(0.1, color="orange", linestyle="--", alpha=0.3, label="caution threshold")
    ax.grid(True, alpha=0.3)
    fig.suptitle("Artifact Size / Fidelity / Transfer Tradeoff", fontsize=14)
    fig.tight_layout()
    path = out_dir / "artifact_size_fidelity_transfer_tradeoff.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    args = parse_args()

    # Validate inputs
    for p in [args.b1_dir, args.raw_anchor_dir, args.paper_table, args.raw_transfer]:
        check_file(p)

    b1_sweep = args.b1_dir / "sweep_summary.csv"
    ek_csv = args.b1_dir / "count_epsilon_to_kstar.csv"
    for p in [b1_sweep, ek_csv]:
        check_file(p)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    sweep = pd.read_csv(b1_sweep)
    pt = pd.read_csv(args.paper_table)
    raw_base = pd.read_csv(args.raw_transfer)

    # Validate required columns
    check_cols(sweep, ["dataset", "artifact_root", "max_filter_planes",
                        "max_plane_count", "uncertain", "n", "rows_per_sec",
                        "selectivity"], "b1_sweep")
    check_cols(raw_base, ["dataset", "transfer_bytes", "cudaMemcpy_ms",
                           "effective_transfer_GBps"], "raw_transfer")

    # Validate raw baseline has 4 datasets
    if len(raw_base) != 4:
        raise ValueError(f"raw baseline has {len(raw_base)} rows, expected 4")

    print("Generating plots...")
    plot_u_fraction_vs_k(sweep, args.out_dir)
    plot_throughput_vs_k(sweep, args.out_dir)
    plot_u_vs_throughput(sweep, args.out_dir)
    plot_epsilon_kstar_heatmap(ek_csv, args.out_dir)
    plot_count_drift(pt, args.out_dir)
    plot_bytes_vs_h2d(pt, raw_base, args.out_dir)
    plot_tradeoff(pt, raw_base, args.out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
