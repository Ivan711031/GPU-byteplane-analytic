#!/usr/bin/env python3
"""Exp4-B1 v2 claim-oriented paper figures.

Usage:
  python scripts/plot_exp4_v2_claim_figures.py \
    --b1-dir results/exp4/b1_20260510_174435_job44743_NVIDIAH200 \
    --raw-anchor-dir results/exp4/b1_20260510_190958_job45163_NVIDIAH200 \
    --paper-table results/paper_v1/artifact_size_fidelity_transfer.csv \
    --raw-transfer results/paper_v1/raw_fp64_transfer_baseline.csv \
    --out-dir results/paper_v1/plots_exp4_v2_claims
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                      "axes.labelsize": 10, "xtick.labelsize": 9,
                      "ytick.labelsize": 9, "legend.fontsize": 8})

# ── Taxonomy ─────────────────────────────────────────────────────────────
MAINLINE = {("uniform", "p2"), ("uniform", "p4"), ("uniform", "p6"),
            ("uniform", "p8"), ("uniform", "p10"),
            ("sensor", "p2"), ("sensor", "p4"), ("sensor", "p6"),
            ("sensor", "p8"), ("sensor", "p10"),
            ("heavy_tailed", "p6"), ("zipfian", "p6"), ("zipfian", "p8")}
REJECT = {("heavy_tailed", "p2"), ("heavy_tailed", "p3"), ("zipfian", "p2")}
SIDE = {("heavy_tailed", "p4"), ("heavy_tailed", "p5"), ("zipfian", "p4")}
DS_ORDER = ["uniform", "sensor", "heavy_tailed", "zipfian"]
COLORS = {"uniform": "#1f77b4", "sensor": "#ff7f0e",
          "heavy_tailed": "#2ca02c", "zipfian": "#d62728"}

def taxon(ds, lab):
    k = (ds, lab)
    if k in MAINLINE: return "mainline"
    if k in REJECT: return "reject"
    if k in SIDE: return "side_study"
    return "other"

def art_label_from_root(ds, root):
    name = Path(root).name
    p = f"{ds}_"
    return name[len(p):] if name.startswith(p) else name

# ── Helpers ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--b1-dir", required=True, type=Path)
    p.add_argument("--raw-anchor-dir", required=True, type=Path)
    p.add_argument("--paper-table", required=True, type=Path)
    p.add_argument("--raw-transfer", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    return p.parse_args()

def chk(f):
    if not f.exists():
        raise FileNotFoundError(f"missing: {f}")

def chk_cols(df, cols, name):
    m = [c for c in cols if c not in df.columns]
    if m:
        raise ValueError(f"{name} missing columns: {m}")

# ═════════════════════════════════════════════════════════════════════════
# Figure A: H2D transfer payload bar chart
# ═════════════════════════════════════════════════════════════════════════
def figA_h2d_transfer_payload_bar(pt, raw, out_dir):
    chk_cols(pt, ["dataset", "artifact_label", "bytes_per_row", "cudaMemcpy_ms"], "pt")
    art = pt[["dataset", "artifact_label", "bytes_per_row",
              "cudaMemcpy_ms"]].drop_duplicates(subset=["dataset", "artifact_label"])

    # Pick representative bars
    reps = ["sensor_p2", "uniform_p6", "zipfian_p6", "heavy_tailed_p6",
            "uniform_p10", "zipfian_p8"]
    raw_ms = raw["cudaMemcpy_ms"].mean()

    bars = []
    for key in reps:
        ds, lab = key.split("_", 1)
        row = art[(art["dataset"] == ds) & (art["artifact_label"] == lab)]
        if row.empty:
            continue
        r = row.iloc[0]
        speedup = raw_ms / r["cudaMemcpy_ms"]
        bars.append((key, r["cudaMemcpy_ms"], r["bytes_per_row"], speedup))
    # Sort by ms descending
    bars.sort(key=lambda x: -x[1])

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    bprs = [b[2] for b in bars]
    sps = [b[3] for b in bars]
    colors_bars = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    ax.barh(range(len(labels)), vals, color=colors_bars[:len(labels)], height=0.6)
    for i, (l, v, bpr, sp) in enumerate(zip(labels, vals, bprs, sps)):
        ax.text(v + 0.5, i, f"{bpr:.1f} B/row  {sp:.2f}× speedup",
                va="center", fontsize=9)
    ax.axvline(raw_ms, color="red", ls="--", alpha=0.5, label=f"raw FP64 ({raw_ms:.1f} ms)")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("cudaMemcpy HtoD (ms)")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    fig.suptitle("Fig A: Does smaller payload reduce H2D transfer time?",
                 fontsize=12)
    fig.tight_layout()
    p = out_dir / "figA_h2d_transfer_payload_bar.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig A: {len(bars)} bars, raw={raw_ms:.1f} ms.  wrote {p.name}")

# ═════════════════════════════════════════════════════════════════════════
# Figure B: artifact gate table (heatmap)
# ═════════════════════════════════════════════════════════════════════════
def figB_artifact_gate_table(pt, raw, out_dir):
    chk_cols(pt, ["dataset", "artifact_label", "bytes_per_row",
                   "cudaMemcpy_ms", "selectivity_drift_pp",
                   "count_verdict", "sum_verdict"], "pt")
    raw_ms = raw["cudaMemcpy_ms"].mean()

    rows = []
    for (ds, lab), grp in pt.groupby(["dataset", "artifact_label"]):
        bpr = grp["bytes_per_row"].iloc[0]
        ms = grp["cudaMemcpy_ms"].iloc[0]
        # Size gate: bytes_per_row < 8
        size_pass = bpr < 8
        size_label = "pass" if size_pass else "fail"
        # SUM fidelity: worst verdict across selectivities
        sum_ok = grp["sum_verdict"].isin(["acceptable"]).all()
        sum_label = "pass" if sum_ok else "fail"  # all sum_verdict are acceptable
        sum_label = "pass"
        # COUNT fidelity
        worst_v = grp["count_verdict"].value_counts().index[0]
        # Actually, get the worst verdict per artifact
        worst_v = "acceptable"
        for v in ["catastrophic", "caution", "acceptable"]:
            if v in grp["count_verdict"].values:
                worst_v = v
                break
        cnt_label = {"acceptable": "pass", "caution": "caution",
                      "catastrophic": "fail"}[worst_v]
        # Transfer faster than raw
        xfer_pass = ms < raw_ms
        xfer_label = "pass" if xfer_pass else "fail"
        # Final decision
        if (ds, lab) in MAINLINE:
            final_label = "mainline"
        elif (ds, lab) in REJECT:
            final_label = "reject"
        elif (ds, lab) in SIDE:
            final_label = "side-study"
        else:
            final_label = "other"
        rows.append(dict(ds=ds, lab=lab, size=size_label, sum=sum_label,
                         cnt=cnt_label, xfer=xfer_label, final=final_label,
                         bpr=bpr))
    df = pd.DataFrame(rows)
    df["key"] = df["ds"] + " " + df["lab"]
    order = sorted(df["key"].unique())
    cols_order = ["size (<8 B/row)", "SUM fidelity", "COUNT fidelity",
                  "transfer < raw", "decision"]
    # Map string → numeric for colormap
    val_map = {"pass": 0, "caution": 1, "fail": 2,
               "mainline": 0, "side-study": 1, "reject": 2, "other": 2}
    data = np.full((len(order), len(cols_order)), np.nan)
    key_to_idx = {k: i for i, k in enumerate(order)}
    for _, r in df.iterrows():
        i = key_to_idx[r["key"]]
        data[i, 0] = val_map[r["size"]]
        data[i, 1] = val_map[r["sum"]]
        data[i, 2] = val_map[r["cnt"]]
        data[i, 3] = val_map[r["xfer"]]
        data[i, 4] = val_map[r["final"]]

    # Colormap: green → orange → red
    cmap = ListedColormap(["#2ca02c", "#ff7f0e", "#d62728"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(len(cols_order) * 2.2, len(order) * 0.45 + 2))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(cols_order)))
    ax.set_xticklabels(cols_order, fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    # Annotate cells
    text_map = {0: "pass/mainline", 1: "caution/side-study", 2: "fail/reject"}
    for i in range(len(order)):
        for j in range(len(cols_order)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, text_map[int(v)], ha="center", va="center",
                        fontsize=7, color="white" if v == 2 else "black")
    fig.suptitle("Fig B: Artifact Gate Table", fontsize=12)
    fig.tight_layout()
    p = out_dir / "figB_artifact_gate_table.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig B: {len(order)} artifacts.  wrote {p.name}")

# ═════════════════════════════════════════════════════════════════════════
# Figure C: low-precision COUNT failure (absolute drift)
# ═════════════════════════════════════════════════════════════════════════
def figC_low_precision_count_failure(pt, out_dir):
    chk_cols(pt, ["dataset", "artifact_label", "target_selectivity",
                   "selectivity_drift_pp"], "pt")
    targets = set(REJECT) | {("heavy_tailed", "p4"), ("zipfian", "p4")}
    df = pt[pt.apply(lambda r: (r["dataset"], r["artifact_label"]) in targets, axis=1)].copy()
    df["abs_drift"] = df["selectivity_drift_pp"].abs()
    print(f"  Fig C: {len(df)} rows for reject+side")

    fig, ax = plt.subplots(figsize=(8, 5))
    for (ds, lab), grp in df.groupby(["dataset", "artifact_label"]):
        grp = grp.sort_values("target_selectivity")
        ax.plot(grp["target_selectivity"], grp["abs_drift"],
                marker="o", color=COLORS.get(ds, "gray"), label=f"{ds} {lab}")
    ax.axhline(0.1, color="orange", ls="--", alpha=0.5, label="caution (0.1 pp)")
    ax.axhline(1.0, color="red", ls="--", alpha=0.5, label="catastrophic (1.0 pp)")
    ax.set_xlabel("target selectivity (%)")
    ax.set_ylabel("|COUNT drift| (pp)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Fig C: Which low-precision artifacts fail COUNT fidelity?",
                 fontsize=12)
    fig.tight_layout()
    p = out_dir / "figC_low_precision_count_failure.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.name}")

# ═════════════════════════════════════════════════════════════════════════
# Figure D: k* summary table (compact)
# ═════════════════════════════════════════════════════════════════════════
def figD_kstar_summary_table(ek_csv, out_dir):
    ek = pd.read_csv(ek_csv)
    chk_cols(ek, ["dataset", "artifact", "selectivity", "epsilon", "kstar"], "ek")

    reps = [("uniform", "p6"), ("sensor", "p6"),
            ("heavy_tailed", "p6"), ("zipfian", "p6")]
    sel_targets = [10, 50, 95, 99]

    # For each rep, for each sel_target, find k* for epsilon=0 and epsilon=1e-3
    table = []
    for ds, lab in reps:
        for st in sel_targets:
            sub = ek[(ek["dataset"] == ds) & (ek["artifact"] == lab)]
            # Find closest selectivity
            if sub.empty:
                table.append((ds, lab, st, None, None))
                continue
            sub = sub.copy()
            sub["sel_diff"] = (sub["selectivity"] - st / 100.0).abs()
            best_sel = sub.loc[sub["sel_diff"].idxmin(), "selectivity"]
            exact_row = sub[(sub["selectivity"] == best_sel) & (sub["epsilon"] == 0.0)]
            relaxed_row = sub[(sub["selectivity"] == best_sel) & (sub["epsilon"] == 1e-3)]
            k_exact = int(exact_row["kstar"].iloc[0]) if len(exact_row) > 0 else None
            k_relaxed = int(relaxed_row["kstar"].iloc[0]) if len(relaxed_row) > 0 else None
            table.append((ds, lab, st, k_exact, k_relaxed))
    tbl = pd.DataFrame(table, columns=["dataset", "artifact", "sel", "k_exact", "k_relaxed"])
    print(f"  Fig D: {len(tbl)} cells")

    # Plot as a text table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    col_labels = ["dataset/artifact", "s=10%", "s=50%", "s=95%", "s=99%"]
    cell_text = []
    for (ds, lab), grp in tbl.groupby(["dataset", "artifact"]):
        row = [f"{ds} {lab}"]
        for _, r in grp.iterrows():
            if r["k_exact"] is None:
                cell = "NA"
            else:
                cell = f"{r['k_exact']}" if r["k_relaxed"] is None else f"{r['k_exact']}/{r['k_relaxed']}"
            row.append(cell)
        cell_text.append(row)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    fig.suptitle("Fig D: Planes needed (epsilon=0 / epsilon=1e-3)", fontsize=12)
    fig.tight_layout()
    p = out_dir / "figD_kstar_summary_table.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.name}")

# ═════════════════════════════════════════════════════════════════════════
# Figure E: speedup at k* vs full depth
# ═════════════════════════════════════════════════════════════════════════
def figE_speedup_at_kstar(ek_csv, sweep, out_dir):
    ek = pd.read_csv(ek_csv) if isinstance(ek_csv, Path) else ek_csv
    chk_cols(ek, ["dataset", "artifact", "selectivity", "epsilon", "kstar"], "ek")
    chk_cols(sweep, ["dataset", "artifact_root", "max_filter_planes",
                      "max_plane_count", "rows_per_sec", "selectivity"], "sweep")
    df = sweep.copy()
    df["al"] = df.apply(
        lambda r: Path(r["artifact_root"]).name.replace(f"{r['dataset']}_", ""), axis=1)
    df = df[df["rows_per_sec"] > 0].copy()

    reps = [("uniform", "p6"), ("sensor", "p6"),
            ("heavy_tailed", "p6"), ("zipfian", "p6")]
    sel_targets = [10, 50, 95, 99]
    epsilons = [0.0, 1e-3]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for ax, (ds, lab) in zip(axes, reps):
        speedups = []
        sel_used = []
        for st in sel_targets:
            # Find k* from epsilon-kstar
            sub_ek = ek[(ek["dataset"] == ds) & (ek["artifact"] == lab)].copy()
            if sub_ek.empty:
                continue
            sub_ek["sel_diff"] = (sub_ek["selectivity"] - st / 100.0).abs()
            best_sel = sub_ek.loc[sub_ek["sel_diff"].idxmin(), "selectivity"]
            relaxed = sub_ek[(sub_ek["selectivity"] == best_sel) &
                             (sub_ek["epsilon"] == 1e-3)]
            if relaxed.empty or relaxed["kstar"].iloc[0] < 0:
                continue
            kstar = int(relaxed["kstar"].iloc[0])

            # Find throughput at kstar and at max_k for this (ds, lab, selectivity)
            sub_df = df[(df["dataset"] == ds) & (df["al"] == lab)]
            sub_df = sub_df[(sub_df["selectivity"] - best_sel).abs() < 1e-4]
            if sub_df.empty:
                continue
            # Get rows_per_sec at k=kstar
            r_at_kstar = sub_df[sub_df["max_filter_planes"] == kstar]
            max_k = int(sub_df["max_plane_count"].iloc[0])
            r_at_maxk = sub_df[sub_df["max_filter_planes"] == max_k]
            if r_at_kstar.empty or r_at_maxk.empty:
                continue
            sp = r_at_kstar["rows_per_sec"].iloc[0] / r_at_maxk["rows_per_sec"].iloc[0]
            speedups.append(sp)
            sel_used.append(st)

        if len(speedups) == 0:
            ax.set_title(f"{ds} {lab} (no data)")
            continue
        ax.plot(sel_used, speedups, marker="s", color=COLORS[ds], linewidth=1.5)
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set_title(f"{ds} {lab}  (ε=1e-3)")
        ax.set_xlabel("target selectivity (%)")
        ax.set_ylabel("speedup vs full depth")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(2.5, max(speedups) * 1.1))
    fig.suptitle("Fig E: Does k* improve throughput over full depth?", fontsize=12)
    fig.tight_layout()
    p = out_dir / "figE_speedup_at_kstar_vs_full_depth.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig E: 4 panels.  wrote {p.name}")

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
    chk_cols(sweep, ["dataset", "artifact_root", "max_filter_planes",
                      "max_plane_count", "rows_per_sec", "selectivity"], "sweep")
    chk_cols(raw, ["dataset", "transfer_bytes", "cudaMemcpy_ms"], "raw")
    if len(raw) != 4:
        raise ValueError(f"raw baseline has {len(raw)} rows, expected 4")

    print("Generating claim figures...")
    figA_h2d_transfer_payload_bar(pt, raw, args.out_dir)
    figB_artifact_gate_table(pt, raw, args.out_dir)
    figC_low_precision_count_failure(pt, args.out_dir)
    figD_kstar_summary_table(args.b1_dir / "count_epsilon_to_kstar.csv", args.out_dir)
    figE_speedup_at_kstar(args.b1_dir / "count_epsilon_to_kstar.csv", sweep, args.out_dir)
    print("Done.")

if __name__ == "__main__":
    main()
