#!/usr/bin/env python3
"""Exp4-B1 v2 story-first paper figures.

Usage:
  python scripts/plot_exp4_v2_story_figures.py \
    --b1-dir results/exp4/b1_20260510_174435_job44743_NVIDIAH200 \
    --raw-anchor-dir results/exp4/b1_20260510_190958_job45163_NVIDIAH200 \
    --paper-table results/paper_v1/artifact_size_fidelity_transfer.csv \
    --raw-transfer results/paper_v1/raw_fp64_transfer_baseline.csv \
    --out-dir results/paper_v1/plots_exp4_v2_story
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

# ── rc ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
})

# ── Taxonomy ─────────────────────────────────────────────────────────────
MAINLINE = {
    ("uniform", "p2"), ("uniform", "p4"), ("uniform", "p6"),
    ("uniform", "p8"), ("uniform", "p10"),
    ("sensor", "p2"), ("sensor", "p4"), ("sensor", "p6"),
    ("sensor", "p8"), ("sensor", "p10"),
    ("heavy_tailed", "p6"),
    ("zipfian", "p6"), ("zipfian", "p8"),
}
REJECT = {("heavy_tailed", "p2"), ("heavy_tailed", "p3"), ("zipfian", "p2")}
SIDE = {("heavy_tailed", "p4"), ("heavy_tailed", "p5"), ("zipfian", "p4")}
DS_ORDER = ["uniform", "sensor", "heavy_tailed", "zipfian"]
COLORS = {"uniform": "#1f77b4", "sensor": "#ff7f0e",
          "heavy_tailed": "#2ca02c", "zipfian": "#d62728"}
REP = [("uniform", "p6"), ("sensor", "p6"), ("heavy_tailed", "p6"), ("zipfian", "p6")]

def art_label(r):
    name = Path(r["artifact_root"]).name
    p = f"{r['dataset']}_"
    return name[len(p):] if name.startswith(p) else name

def taxon(ds, lab):
    k = (ds, lab)
    if k in MAINLINE: return "mainline"
    if k in REJECT: return "reject"
    if k in SIDE: return "side_study"
    return "other"

# ── Helpers ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--b1-dir", required=True, type=Path)
    p.add_argument("--raw-anchor-dir", required=True, type=Path)
    p.add_argument("--paper-table", required=True, type=Path)
    p.add_argument("--raw-transfer", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--k1-dispatch-csv", required=False, type=Path, default=None,
                   help="Path to count_precision_throughput_with_k1_dispatch.csv")
    return p.parse_args()

def chk(f):
    if not f.exists():
        raise FileNotFoundError(f"missing: {f}")

def chk_cols(df, cols, name):
    m = [c for c in cols if c not in df.columns]
    if m:
        raise ValueError(f"{name} missing columns: {m}")

def add_taxon(df):
    df = df.copy()
    df["al"] = df.apply(art_label, axis=1)
    df["tax"] = df.apply(lambda r: taxon(r["dataset"], r["al"]), axis=1)
    return df

# ═════════════════════════════════════════════════════════════════════════
# Fig 1: transfer size scaling
# ═════════════════════════════════════════════════════════════════════════
def fig1_transfer_size_scaling(pt, raw, out):
    chk_cols(pt, ["dataset","artifact_label","bytes_per_row","cudaMemcpy_ms"], "pt")
    art = pt[["dataset","artifact_label","bytes_per_row","cudaMemcpy_ms"]].drop_duplicates(
        subset=["dataset","artifact_label"])
    raw_ms = raw["cudaMemcpy_ms"].mean()  # ~82.2

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # All artifacts light gray
    for _, r in art.iterrows():
        ax.scatter(r["bytes_per_row"], r["cudaMemcpy_ms"], marker="o",
                   color="lightgray", s=40, alpha=0.6, zorder=1)
    # Label representative points
    reps = [("sensor","p2"), ("uniform","p6"), ("heavy_tailed","p6"),
            ("zipfian","p6"), ("zipfian","p8")]
    for ds, lab in reps:
        row = art[(art["dataset"]==ds) & (art["artifact_label"]==lab)]
        if row.empty:
            continue
        r = row.iloc[0]
        ax.scatter(r["bytes_per_row"], r["cudaMemcpy_ms"], marker="o",
                   color=COLORS[ds], s=80, edgecolors="black", zorder=3)
        ax.annotate(f"{ds} {lab}", (r["bytes_per_row"], r["cudaMemcpy_ms"]),
                    fontsize=9, xytext=(6, 4), textcoords="offset points")

    # Raw baseline
    for _, r in raw.iterrows():
        ax.scatter(8.0, r["cudaMemcpy_ms"], marker="*", color="red", s=200, zorder=4)
    ax.annotate("raw FP64 (~82 ms)", (8.0, raw_ms + 1), fontsize=9,
                color="red", ha="center")
    ax.axhline(raw_ms, color="red", ls="--", alpha=0.3)

    ax.set_xlabel("bytes per row")
    ax.set_ylabel("cudaMemcpy HtoD (ms)")
    ax.grid(True, alpha=0.3)
    fig.suptitle("Fig 1: Does smaller encoded payload reduce H2D transfer time?",
                 fontsize=12)
    fig.tight_layout()
    p = out / "fig1_transfer_size_scaling.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{len(art)} artifacts + 4 raw] wrote {p}")

# ═════════════════════════════════════════════════════════════════════════
# Fig 2: mainline COUNT drift (tight y)
# ═════════════════════════════════════════════════════════════════════════
def fig2_mainline_count_drift_zoom(pt, out):
    chk_cols(pt, ["dataset","artifact_label","target_selectivity",
                   "selectivity_drift_pp","count_verdict"], "pt")
    df = pt[pt.apply(lambda r: (r["dataset"], r["artifact_label"]) in MAINLINE, axis=1)].copy()
    print(f"  Fig2: {len(df)} rows for mainline drift")

    vc = {"acceptable":"green","caution":"orange","catastrophic":"red"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, ds in zip(axes, DS_ORDER):
        sub = df[df["dataset"]==ds]
        for lab, grp in sub.groupby("artifact_label"):
            grp = grp.sort_values("target_selectivity")
            c = [vc.get(v,"gray") for v in grp["count_verdict"]]
            ax.scatter(grp["target_selectivity"], grp["selectivity_drift_pp"],
                       c=c, label=lab, s=30)
            ax.plot(grp["target_selectivity"], grp["selectivity_drift_pp"],
                    alpha=0.2, color="gray")
        ax.axhline(0, color="gray", ls="--", alpha=0.4)
        ax.set_title(ds)
        ax.set_xlabel("selectivity (%)")
        ax.set_ylabel("drift (pp)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Fig 2: Are mainline artifacts faithful enough for COUNT?",
                 fontsize=12)
    fig.tight_layout()
    p = out / "fig2_mainline_count_drift_zoom.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")

# ═════════════════════════════════════════════════════════════════════════
# Fig 3: reject COUNT drift (wide y)
# ═════════════════════════════════════════════════════════════════════════
def fig3_reject_count_drift_failure(pt, out):
    chk_cols(pt, ["dataset","artifact_label","target_selectivity",
                   "selectivity_drift_pp","count_verdict"], "pt")
    targets = set(REJECT) | {("heavy_tailed","p4"), ("zipfian","p4")}
    df = pt[pt.apply(lambda r: (r["dataset"], r["artifact_label"]) in targets, axis=1)].copy()
    print(f"  Fig3: {len(df)} rows for reject/side-study drift")

    vc = {"acceptable":"green","caution":"orange","catastrophic":"red"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for (ds, lab), grp in df.groupby(["dataset","artifact_label"]):
        grp = grp.sort_values("target_selectivity")
        c = [vc.get(v,"gray") for v in grp["count_verdict"]]
        ax.scatter(grp["target_selectivity"], grp["selectivity_drift_pp"],
                   c=c, label=f"{ds} {lab}", s=40)
        ax.plot(grp["target_selectivity"], grp["selectivity_drift_pp"],
                alpha=0.3, color="gray")
    ax.axhline(0, color="gray", ls="--", alpha=0.4)
    ax.axhspan(-0.1, 0.1, alpha=0.05, color="green", label="acceptable")
    ax.axhspan(0.1, 1.0, alpha=0.05, color="orange", label="caution")
    ax.axhspan(1.0, 30, alpha=0.05, color="red", label="catastrophic")
    ax.set_xlabel("selectivity (%)")
    ax.set_ylabel("drift (pp)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Fig 3: Which low-precision artifacts fail COUNT fidelity?",
                 fontsize=12)
    fig.tight_layout()
    p = out / "fig3_reject_count_drift_failure.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")

# ═════════════════════════════════════════════════════════════════════════
# Fig 4: U(k)/n for representative artifacts, 3 selectivities each
# ═════════════════════════════════════════════════════════════════════════
def fig4_progressive_resolution(sweep, out):
    chk_cols(sweep, ["dataset","artifact_root","max_filter_planes",
                      "uncertain","n","selectivity"], "sweep")
    df = add_taxon(sweep)
    sel_targets = [0.10, 0.50, 0.95]  # approximate for the sweep selectivity values
    df["u_frac"] = df["uncertain"] / df["n"]
    df["k"] = df["max_filter_planes"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    colors_sel = {0.10: "#1f77b4", 0.50: "#ff7f0e", 0.95: "#d62728"}
    for ax, (ds, lab) in zip(axes, REP):
        sub = df[(df["dataset"]==ds) & (df["al"]==lab)].copy()
        # Find closest selectivities to 0.10, 0.50, 0.95
        picked = []
        for target in sel_targets:
            best = sub.iloc[(sub["selectivity"] - target).abs().argsort()[:1]]
            picked.append(best)
        plot_df = pd.concat(picked).drop_duplicates(subset=["selectivity","k"])
        for _, row in plot_df.iterrows():
            sel = row["selectivity"]
            subk = sub[(sub["selectivity"]==sel) & (sub["k"]<=sub["max_plane_count"])].sort_values("k")
            if len(subk) == 0:
                continue
            label = f"s={sel:.2f}"
            color = colors_sel.get(round(sel, 2), "gray")
            ax.plot(subk["k"], subk["u_frac"], marker="o", color=color, label=label)
        ax.set_title(f"{ds} {lab}")
        ax.set_xlabel("k")
        ax.set_ylabel("U / n")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Fig 4: How quickly does U(k) shrink as more planes are read?",
                 fontsize=12)
    fig.tight_layout()
    p = out / "fig4_progressive_resolution_representative.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")

# ═════════════════════════════════════════════════════════════════════════
# Fig 5: throughput vs k at fixed selectivity (with optional k1 dispatch overlay)
# ═════════════════════════════════════════════════════════════════════════
def fig5_throughput_vs_k_fixed_sel(sweep, out, k1_dispatch=None):
    chk_cols(sweep, ["dataset","artifact_root","max_filter_planes",
                      "rows_per_sec","selectivity"], "sweep")
    df = add_taxon(sweep)
    df = df[df["rows_per_sec"] > 0].copy()
    df["k"] = df["max_filter_planes"]

    # Precompute k1 dispatch lookup: (dataset, artifact_label, selectivity, k=1) -> dispatch throughput
    k1_lookup = {}
    if k1_dispatch is not None:
        k1_k1 = k1_dispatch[k1_dispatch["max_filter_planes"] == 1]
        for _, row in k1_k1.iterrows():
            key = (row["dataset"], row["artifact_label"], row["selectivity"])
            k1_lookup[key] = row["k1_dispatch_rows_per_sec"]

    for sel_target, suffix in [(0.50, "s50"), (0.95, "s95")]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
        axes = axes.flatten()
        for ax, (ds, lab) in zip(axes, REP):
            sub = df[(df["dataset"]==ds) & (df["al"]==lab)].copy()
            best = sub.iloc[(sub["selectivity"] - sel_target).abs().argsort()[:1]]
            if best.empty:
                continue
            closest_sel = best["selectivity"].iloc[0]
            subk = sub[(sub["selectivity"]==closest_sel)].sort_values("k")
            # Generic byte_mask line
            ax.plot(subk["k"], subk["rows_per_sec"]/1e9, marker="s",
                    color=COLORS[ds], linewidth=1.5, label="byte_mask (generic)")
            # k1 dispatch overlay at k=1
            if k1_lookup:
                key = (ds, lab, closest_sel)
                if key in k1_lookup:
                    dispatch_tp = k1_lookup[key] / 1e9
                    generic_k1 = subk[subk["k"]==1]["rows_per_sec"].values
                    if len(generic_k1) > 0:
                        generic_tp = generic_k1[0] / 1e9
                        ax.scatter([1], [dispatch_tp], marker="^", s=80,
                                   color=COLORS[ds], edgecolors="black",
                                   zorder=5, label=f"k1 dispatch ({dispatch_tp/generic_tp:.2f}×)")
            ax.set_title(f"{ds} {lab} (s={closest_sel:.3f})")
            ax.set_xlabel("k")
            ax.set_ylabel("rows/s (B)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Fig 5: Does smaller k improve throughput? (s~{sel_target:.0%})",
                     fontsize=12)
        fig.tight_layout()
        p = out / f"fig5_throughput_vs_k_fixed_selectivity_{suffix}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p}")

# ═════════════════════════════════════════════════════════════════════════
# Fig 6: epsilon → k* heatmap per representative artifact
# ═════════════════════════════════════════════════════════════════════════
def fig6_epsilon_to_kstar(ek_csv, out):
    ek = pd.read_csv(ek_csv)
    chk_cols(ek, ["dataset","artifact","selectivity","epsilon","kstar","max_plane_count"], "ek")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for ax, (ds, lab) in zip(axes, REP):
        sub = ek[(ek["dataset"]==ds) & (ek["artifact"]==lab)].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub["row"] = sub["selectivity"].apply(lambda x: f"s={x:.4f}")
        # Handle NA: kstar == -1 means unmet
        sub["kstar_finite"] = sub["kstar"].replace(-1, np.nan)
        pivot = sub.pivot_table(index="row", columns="epsilon",
                                values="kstar_finite", aggfunc="first")
        eps = sorted(sub["epsilon"].unique())
        pivot = pivot[[c for c in eps if c in pivot.columns]]
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r",
                       interpolation="nearest",
                       norm=Normalize(vmin=1, vmax=sub["max_plane_count"].max()))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{e:.0e}" for e in pivot.columns], fontsize=8, rotation=30)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_title(f"{ds} {lab}")
        ax.set_xlabel("ε")
        ax.set_ylabel("selectivity")
    fig.colorbar(im, ax=axes, label="k*", shrink=0.5, aspect=40)
    fig.suptitle("Fig 6: How many planes needed for error tolerance ε?",
                 fontsize=12)
    fig.tight_layout()
    p = out / "fig6_epsilon_to_kstar_by_dataset.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")

# ═════════════════════════════════════════════════════════════════════════
# Fig 7: artifact decision map (two versions)
# ═════════════════════════════════════════════════════════════════════════
def fig7_decision_map(pt, raw, out):
    chk_cols(pt, ["dataset","artifact_label","bytes_per_row",
                   "cudaMemcpy_ms","selectivity_drift_pp"], "pt")
    rows = []
    for (ds, lab), grp in pt.groupby(["dataset","artifact_label"]):
        bpr = grp["bytes_per_row"].iloc[0]
        ms = grp["cudaMemcpy_ms"].iloc[0]
        md = grp["selectivity_drift_pp"].abs().max()
        raw_ms = raw["cudaMemcpy_ms"].mean()
        sp = raw_ms / ms if ms > 0 else 0
        t = taxon(ds, lab)
        worst_v = grp.loc[grp["selectivity_drift_pp"].abs().idxmax(), "count_verdict"]
        rows.append(dict(ds=ds, lab=lab, bpr=bpr, md=md, sp=sp,
                         tax=t, wv=worst_v))
    agg = pd.DataFrame(rows)

    tax_m = {"mainline":"o", "side_study":"s", "reject":"X", "other":"d"}

    for version, ylim, suf in [
        ("full", None, "full"),
        ("zoom_mainline", (-0.5, 1.5), "zoom_mainline"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for _, r in agg.iterrows():
            is_main = r["tax"] == "mainline"
            ax.scatter(r["bpr"], r["md"],
                       s=r["sp"] * 35,
                       marker=tax_m.get(r["tax"], "o"),
                       color=COLORS[r["ds"]] if is_main else "gray",
                       edgecolors="black" if not is_main else "none",
                       alpha=0.8, zorder=3)
            # Label non-mainline
            if r["tax"] != "mainline":
                ax.annotate(f"  {r['ds'][:4]} {r['lab']}",
                            (r["bpr"], r["md"]), fontsize=7, alpha=0.7)

        # Mainline annotations
        for _, r in agg.iterrows():
            if r["tax"] == "mainline" and r["lab"] in ("p2","p10") or r["ds"]=="heavy_tailed":
                ax.annotate(f"  {r['ds'][:4]} {r['lab']}",
                            (r["bpr"], r["md"]), fontsize=7, alpha=0.8)

        ax.axhline(1.0, color="red", ls="--", alpha=0.3)
        ax.text(1, 1.05, "catastrophic", fontsize=8, color="red", alpha=0.5)
        ax.axhline(0.1, color="orange", ls="--", alpha=0.3)
        ax.text(1, 0.15, "caution", fontsize=8, color="orange", alpha=0.5)
        if ylim:
            ax.set_ylim(ylim)

        from matplotlib.lines import Line2D
        h = [Line2D([0],[0], marker="o", color="w", markerfacecolor=COLORS[d],
                    markersize=8, label=d) for d in COLORS]
        h += [Line2D([0],[0], marker=m, color="gray", markersize=8, label=t,
                     linestyle="none") for t, m in [("mainline","o"),("side_study","s"),("reject","X")]]
        h.append(Line2D([0],[0], marker="o", color="gray", markersize=14,
                        markerfacecolor="lightgray", label="size ~ speedup"))
        ax.legend(handles=h, fontsize=7, ncol=2)

        ax.set_xlabel("bytes per row")
        ax.set_ylabel("max |COUNT drift| (pp)")
        ax.grid(True, alpha=0.3)
        fig.suptitle(f"Fig 7: Which artifacts are mainline, side-study, or reject?",
                     fontsize=12)
        fig.tight_layout()
        p = out / f"fig7_artifact_decision_map_{suf}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{len(agg)} artifacts] wrote {p}")

# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    for p in [args.b1_dir, args.raw_anchor_dir, args.paper_table, args.raw_transfer]:
        chk(p)
    for p in [args.b1_dir / "sweep_summary.csv", args.b1_dir / "count_epsilon_to_kstar.csv"]:
        chk(p)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    sweep = pd.read_csv(args.b1_dir / "sweep_summary.csv")
    pt = pd.read_csv(args.paper_table)
    raw = pd.read_csv(args.raw_transfer)
    chk_cols(sweep, ["dataset","artifact_root","max_filter_planes",
                      "max_plane_count","uncertain","n","rows_per_sec","selectivity"], "sweep")
    chk_cols(raw, ["dataset","transfer_bytes","cudaMemcpy_ms"], "raw")
    if len(raw) != 4:
        raise ValueError(f"raw baseline has {len(raw)} rows, expected 4")

    # Load k1 dispatch data if provided
    k1_dispatch = None
    if args.k1_dispatch_csv:
        if args.k1_dispatch_csv.exists():
            k1_dispatch = pd.read_csv(args.k1_dispatch_csv)
            chk_cols(k1_dispatch, ["dataset","artifact_label","selectivity",
                                    "max_filter_planes","rows_per_sec",
                                    "k1_dispatch_rows_per_sec"], "k1_dispatch")
            print(f"  k1 dispatch: {len(k1_dispatch)} rows, "
                  f"{(k1_dispatch['max_filter_planes']==1).sum()} k=1 rows")
        else:
            print(f"  WARNING: k1 dispatch CSV not found: {args.k1_dispatch_csv}")

    print("Generating story figures...")
    fig1_transfer_size_scaling(pt, raw, args.out_dir)
    fig2_mainline_count_drift_zoom(pt, args.out_dir)
    fig3_reject_count_drift_failure(pt, args.out_dir)
    fig4_progressive_resolution(sweep, args.out_dir)
    fig5_throughput_vs_k_fixed_sel(sweep, args.out_dir, k1_dispatch=k1_dispatch)
    fig6_epsilon_to_kstar(args.b1_dir / "count_epsilon_to_kstar.csv", args.out_dir)
    fig7_decision_map(pt, raw, args.out_dir)
    print("Done.")

if __name__ == "__main__":
    main()
