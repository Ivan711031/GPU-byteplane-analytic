#!/usr/bin/env python3
"""Generate the claim-facing H200 paper figure packet v2.

This script only reads committed CSV outputs. It does not run experiments.
The output packet is written to `results/paper_v1/figures_scoped_h200_v2/`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("results/paper_v1")
DEFAULT_OUT_DIR = ROOT / "figures_scoped_h200_v2"

FIG1_LOCALITY_CSV = Path(
    "results/exp4_filter_aggregate/scientific_locality_attribution_57768/"
    "scientific_locality_attribution.csv"
)
FIG1_EXTRA_CSV = Path(
    "results/exp4_filter_aggregate/scientific_extra_fields_quasi_global_58440/"
    "scientific_extra_fields_quasi_global.csv"
)
FIG2_JOINED_CSV = Path("results/precision_throughput/exp3_v2_sum_precision_throughput.csv")
FIG2_EPS_CSV = Path("results/precision_throughput/exp3_v2_sum_kstar_relative_empirical.csv")
FIG2_BOUND_CSV = Path("results/precision_throughput/exp3_v2_sum_kstar_relative_bound.csv")
FIG3_Q8_CSV = Path(
    "results/exp4_filter_aggregate/q8_hbm_transaction_20260523_003039_job59096_NVIDIAH200/"
    "q8_hbm_transaction_summary.csv"
)
FIG4_PROMPT5_UNIFORM = Path(
    "results/prompt5_h200_viability/ncu_kdepth_20260523_135553_job59368_NVIDIAH200/"
    "k_depth_ncu_matrix.csv"
)
FIG4_PROMPT5_HEAVY = Path(
    "results/prompt5_h200_viability/ncu_kdepth_20260523_165051_job59418_NVIDIAH200/"
    "k_depth_ncu_matrix.csv"
)

FIELD_ORDER = ["cesm_atm_cloud", "hurricane_u", "cesm_atm_q", "hurricane_tc"]
FIELD_DISPLAY = {
    "cesm_atm_cloud": "CESM Cloud Fraction",
    "hurricane_u": "Hurricane U-Wind",
    "cesm_atm_q": "CESM Specific Humidity",
    "hurricane_tc": "Hurricane Temp",
}

EPSILON_ORDER = [0.0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1]
EPSILON_TICK_LABELS = ["0", "1e-12", "1e-9", "1e-6", "1e-4", "1e-3", "1e-2", "5e-2", "1e-1", "2e-1"]

DATASET_ORDER = ["sensor", "heavy_tailed"]
PRECISION_COLORS = {
    2: "#1f77b4",
    3: "#ff7f0e",
    4: "#2ca02c",
    5: "#d62728",
    6: "#9467bd",
    8: "#8c564b",
    10: "#e377c2",
}

COLOR_RAW = "#6b6b6b"
COLOR_K2 = "#2b5c8f"
COLOR_KMAX = "#d95f02"
BACKGROUND_FALLBACK = "#f4e8e4"


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 220) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def precision_power(label: str) -> int:
    if not str(label).startswith("p"):
        raise ValueError(f"unexpected precision label: {label}")
    return int(str(label)[1:])


def extra_selectivity_bucket(value: float) -> int:
    """Map the drifted scientific extra-field rows to the three headline buckets."""

    val = float(value)
    if val < 0.15:
        return 10
    if val < 0.55:
        return 50
    return 90


def format_epsilon(value: float) -> str:
    if float(value) == 0.0:
        return "0"
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def field_panel_order(df: pd.DataFrame) -> list[str]:
    present = [f for f in FIELD_ORDER if f in set(df["field"].astype(str))]
    return present


def parse_bytes_to_mb(value: object) -> float:
    text = str(value).strip()
    if text.endswith("Gbyte"):
        return float(text[:-5].strip()) * 1024.0
    if text.endswith("Mbyte"):
        return float(text[:-5].strip())
    return float(text)


def add_bar_labels(ax: plt.Axes, bars: list[plt.Rectangle], fmt: str = "{:.2f}") -> None:
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def build_fig1_frame() -> pd.DataFrame:
    local = load_csv(FIG1_LOCALITY_CSV)
    extra = load_csv(FIG1_EXTRA_CSV)

    local = local[
        (local["representation"] == "quasi-global")
        & (local["dataset"].isin(["cesm_atm_cloud", "hurricane_u"]))
        & (local["k_type"].isin(["k_star", "k_max"]))
    ].copy()
    local["field"] = local["dataset"]
    local["selectivity_bucket"] = local["selectivity"].astype(float).round().astype(int)
    local["selectivity_reported_pct"] = local["selectivity"].astype(float)
    local["k_role"] = local["k_type"].map({"k_star": "primary", "k_max": "fallback"})
    local["speedup_vs_raw"] = local["bp_vs_raw_speedup"].astype(float)
    local["warm_e2e_ms"] = local["warm_e2e_ms"].astype(float)
    local["raw_fused_ms"] = local["raw_fused_ms"].astype(float)
    local["source_file"] = FIG1_LOCALITY_CSV.as_posix()
    local["bucket_note"] = ""

    extra = extra[extra["dataset"].isin(["cesm_atm_q", "hurricane_tc"])].copy()
    extra["field"] = extra["dataset"]
    extra["selectivity_bucket"] = extra["selectivity"].apply(extra_selectivity_bucket)
    extra["selectivity_reported_pct"] = extra["selectivity"].astype(float) * 100.0
    extra["k"] = extra["max_filter_planes"].astype(int)
    extra["k_role"] = np.where(extra["k"].astype(int) == 2, "primary", "fallback")
    extra["speedup_vs_raw"] = extra["speedup_vs_raw"].astype(float)
    extra["warm_e2e_ms"] = extra["ms_per_iter"].astype(float)
    extra["raw_fused_ms"] = extra["raw_baseline_ms_per_iter"].astype(float)
    extra["source_file"] = FIG1_EXTRA_CSV.as_posix()
    extra["bucket_note"] = np.where(
        (extra["field"] == "cesm_atm_q") & (extra["selectivity_bucket"] == 90) & (extra["selectivity_reported_pct"] < 80.0),
        "high-selectivity bucket",
        "",
    )

    cols = [
        "field",
        "dataset",
        "selectivity_bucket",
        "selectivity_reported_pct",
        "k",
        "k_role",
        "speedup_vs_raw",
        "warm_e2e_ms",
        "raw_fused_ms",
        "bucket_note",
        "source_file",
    ]
    combined = pd.concat([local[cols], extra[cols]], ignore_index=True)
    combined = combined.sort_values(["field", "selectivity_bucket", "k_role"]).reset_index(drop=True)
    return combined


def plot_fig1_scientific_headline(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharey=True)
    axes = axes.flatten()
    bucket_order = [10, 50, 90]
    x_positions = np.arange(len(bucket_order))
    width = 0.34

    for idx, field in enumerate(field_panel_order(df)):
        ax = axes[idx]
        sub = df[df["field"].eq(field)].copy()
        if sub.empty:
            ax.set_axis_off()
            continue

        pivot = sub.pivot_table(
            index="selectivity_bucket",
            columns="k_role",
            values="speedup_vs_raw",
            aggfunc="first",
        ).reindex(bucket_order)
        primary = pivot["primary"].to_numpy(dtype=float)
        fallback = pivot["fallback"].to_numpy(dtype=float)
        max_y = float(np.nanmax(np.concatenate([primary, fallback])))

        bars_primary = ax.bar(
            x_positions - width / 2,
            primary,
            width=width,
            color=COLOR_K2,
            edgecolor="black",
            linewidth=0.8,
            label="k=2 primary",
        )
        bars_fallback = ax.bar(
            x_positions + width / 2,
            fallback,
            width=width,
            color=COLOR_KMAX,
            edgecolor="black",
            linewidth=0.8,
            hatch="//",
            label="k=max diagnostic/fallback",
        )

        ax.axhline(1.0, color=COLOR_RAW, linestyle="--", linewidth=1.0)
        ax.set_title(FIELD_DISPLAY[field])
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["10%", "50%", "90%"])
        ax.set_ylim(0.9, max(7.6, max_y * 1.15))
        ax.grid(True, axis="y", alpha=0.22)
        if idx % 2 == 0:
            ax.set_ylabel("Warm E2E speedup vs raw FP64")
        if idx >= 2:
            ax.set_xlabel("Selectivity bucket")
        add_bar_labels(ax, list(bars_primary), fmt="{:.2f}x")
        add_bar_labels(ax, list(bars_fallback), fmt="{:.2f}x")
        ax.text(
            0.5,
            0.03,
            "raw FP64 = 1x baseline",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLOR_RAW,
        )

        if field == "cesm_atm_q":
            ax.text(
                0.02,
                0.98,
                "90% bucket includes a 65.02% reported-selectivity k=2 row",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )

        if idx == 0:
            ax.legend(frameon=False, loc="upper right")

    for j in range(len(field_panel_order(df)), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Scientific headline speedup on H200", y=0.99, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    save_figure(fig, out_dir, "fig1_scientific_headline")


def build_fig2_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eps = load_csv(FIG2_EPS_CSV)
    throughput = load_csv(FIG2_JOINED_CSV)

    eps = eps.copy()
    eps["precision_power"] = eps["artifact_label"].map(precision_power)
    eps["epsilon"] = eps["epsilon"].astype(float)
    eps["k_star"] = eps["k_star"].astype(int)
    eps["epsilon_rank"] = eps["epsilon"].map({v: i for i, v in enumerate(EPSILON_ORDER)})
    eps = eps[eps["epsilon"].isin(EPSILON_ORDER)].copy()

    points = eps.merge(
        throughput[[
            "dataset",
            "artifact_label",
            "k",
            "billion_rows_per_sec",
            "rows_per_sec",
            "logical_GBps",
            "ms_per_iter",
            "rel_error_vs_encoded_full_depth",
            "validated",
        ]],
        left_on=["dataset", "artifact_label", "k_star"],
        right_on=["dataset", "artifact_label", "k"],
        how="left",
        validate="many_to_one",
    )
    points["selected_throughput_billion_rows_per_sec"] = points["billion_rows_per_sec"].astype(float)
    points["selected_rows_per_sec"] = points["rows_per_sec"].astype(float)
    points["selected_k"] = points["k_star"].astype(int)
    points["source_file"] = FIG2_EPS_CSV.as_posix() + ";" + FIG2_JOINED_CSV.as_posix()

    throughput = throughput.copy()
    throughput["precision_power"] = throughput["artifact_label"].map(precision_power)
    throughput["billion_rows_per_sec"] = throughput["billion_rows_per_sec"].astype(float)
    throughput["rel_error_vs_encoded_full_depth"] = throughput["rel_error_vs_encoded_full_depth"].astype(float)
    throughput["k"] = throughput["k"].astype(int)

    return eps, points, throughput


def plot_fig2_sum_mechanism(eps: pd.DataFrame, throughput: pd.DataFrame, out_dir: Path) -> None:
    # Set up GridSpec: 1 row, 2 columns with different widths
    # Left column contains the heatmap (spanning both rows vertically)
    # Right column contains two vertically stacked subplots (Throughput vs k, Error vs k)
    fig = plt.figure(figsize=(14.0, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], wspace=0.28, hspace=0.38)
    
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_tp = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, 1])

    # 1. Heatmap plotting (Left panel)
    heatmap_rows = [
        ("sensor", "p2", "sensor (p2)"),
        ("sensor", "p6", "sensor (p6)"),
        ("sensor", "p10", "sensor (p10)"),
        ("heavy_tailed", "p2", "heavy_tailed (p2)"),
        ("heavy_tailed", "p6", "heavy_tailed (p6)"),
    ]
    
    matrix = np.zeros((len(heatmap_rows), len(EPSILON_ORDER)))
    
    for r_idx, (ds, art, _) in enumerate(heatmap_rows):
        for c_idx, _ in enumerate(EPSILON_ORDER):
            match = eps[(eps["dataset"] == ds) & (eps["artifact_label"] == art) & (eps["epsilon_rank"] == c_idx)]
            if not match.empty:
                matrix[r_idx, c_idx] = match["k_star"].values[0]
            else:
                matrix[r_idx, c_idx] = np.nan
                
    im = ax_heat.imshow(matrix, cmap=plt.cm.Blues, aspect="auto")
    
    # Grid lines separating cells
    ax_heat.set_xticks(np.arange(-.5, len(EPSILON_ORDER), 1), minor=True)
    ax_heat.set_yticks(np.arange(-.5, len(heatmap_rows), 1), minor=True)
    ax_heat.grid(which="minor", color="w", linestyle="-", linewidth=2.5)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    
    # Text annotations in each cell
    for r_idx in range(len(heatmap_rows)):
        for c_idx in range(len(EPSILON_ORDER)):
            val = matrix[r_idx, c_idx]
            if not np.isnan(val):
                text_color = "white" if val >= 5 else "black"
                ax_heat.text(
                    c_idx,
                    r_idx,
                    f"{int(val)}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=10,
                )
                
    # Labels & colorbar
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.75, aspect=18)
    cbar.set_label("Selected depth $k^*$", fontweight="bold", fontsize=10)
    cbar.set_ticks(np.arange(1, 9))
    
    ax_heat.set_title("Selected depth $k^*$ vs. relative error tolerance ($\epsilon$)", fontsize=11, fontweight="bold", pad=10)
    ax_heat.set_xticks(range(len(EPSILON_ORDER)))
    ax_heat.set_xticklabels(EPSILON_TICK_LABELS, rotation=35, ha="right", fontsize=9)
    ax_heat.set_xlabel("Relative error tolerance ($\epsilon$)", fontweight="bold", fontsize=10)
    
    ax_heat.set_yticks(range(len(heatmap_rows)))
    ax_heat.set_yticklabels([label for _, _, label in heatmap_rows], fontsize=9)
    
    # 2. Side tradeoffs for p6 (Right panels)
    # Filter throughput data for p6
    sensor_tp = throughput[(throughput["dataset"] == "sensor") & (throughput["artifact_label"] == "p6")].sort_values("k")
    heavy_tp = throughput[(throughput["dataset"] == "heavy_tailed") & (throughput["artifact_label"] == "p6")].sort_values("k")
    
    # Top-right: Throughput vs k
    ax_tp.plot(sensor_tp["k"], sensor_tp["billion_rows_per_sec"], marker="o", linewidth=2.0, color="#1f77b4", label="sensor (p6)")
    ax_tp.plot(heavy_tp["k"], heavy_tp["billion_rows_per_sec"], marker="s", linewidth=2.0, color="#ff7f0e", label="heavy_tailed (p6)")
    ax_tp.set_title("Throughput vs. execution depth $k$ ($p=6$)", fontsize=11, fontweight="bold")
    ax_tp.set_ylabel("Throughput (billion rows/s)", fontweight="bold", fontsize=10)
    ax_tp.set_xticks(range(1, 9))
    ax_tp.grid(True, axis="both", alpha=0.22)
    ax_tp.legend(frameon=False, fontsize=9)
    
    # Bottom-right: Relative error vs k
    ax_err.plot(sensor_tp["k"], sensor_tp["rel_error_vs_encoded_full_depth"], marker="o", linewidth=2.0, color="#1f77b4", label="sensor (p6)")
    ax_err.plot(heavy_tp["k"], heavy_tp["rel_error_vs_encoded_full_depth"], marker="s", linewidth=2.0, color="#ff7f0e", label="heavy_tailed (p6)")
    ax_err.set_title("Relative error vs. execution depth $k$ ($p=6$)", fontsize=11, fontweight="bold")
    ax_err.set_xlabel("Execution depth $k$", fontweight="bold", fontsize=10)
    ax_err.set_ylabel("Relative error", fontweight="bold", fontsize=10)
    ax_err.set_yscale("symlog", linthresh=1e-15)
    ax_err.set_xticks(range(1, 9))
    ax_err.grid(True, axis="both", alpha=0.22)
    ax_err.legend(frameon=False, fontsize=9)

    fig.suptitle("SUM mechanism: tolerance-depth-throughput tradeoff on H200", y=0.98, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, out_dir, "fig2_sum_mechanism")


def build_fig3_frame() -> pd.DataFrame:
    df = load_csv(FIG3_Q8_CSV)
    df = df[df["field"].eq("cesm_atm_q")].copy()
    df["dram_mb"] = df["dram_bytes_or_sector_proxy"].apply(parse_bytes_to_mb)
    raw = df[df["k"].astype(str).eq("raw")].iloc[0]
    k2 = df[df["k"].astype(str).eq("2")].iloc[0]
    kmax = df[df["k"].astype(str).eq("max")].iloc[0]

    rows = []
    for metric, raw_value, vals in [
        ("runtime_ms", float(raw["runtime_ms"]), [float(raw["runtime_ms"]), float(k2["runtime_ms"]), float(kmax["runtime_ms"])]),
        ("dram_mb", float(raw["dram_mb"]), [float(raw["dram_mb"]), float(k2["dram_mb"]), float(kmax["dram_mb"])]),
        ("l2_sector_proxy", float(raw["l2_sector_proxy"]), [float(raw["l2_sector_proxy"]), float(k2["l2_sector_proxy"]), float(kmax["l2_sector_proxy"])]),
        ("l1tex_sector_proxy", float(raw["l1tex_sector_proxy"]), [float(raw["l1tex_sector_proxy"]), float(k2["l1tex_sector_proxy"]), float(kmax["l1tex_sector_proxy"])]),
    ]:
        for label, value in zip(["raw", "k=2", "k=max"], vals):
            rows.append(
                {
                    "metric": metric,
                    "metric_display": {
                        "runtime_ms": "Runtime (ms)",
                        "dram_mb": "DRAM read (MB)",
                        "l2_sector_proxy": "L2 sectors",
                        "l1tex_sector_proxy": "L1/TEX sectors",
                    }[metric],
                    "label": label,
                    "value": value,
                    "raw_value": raw_value,
                    "ratio_to_raw": value / raw_value if raw_value else np.nan,
                    "speedup_or_reduction": raw_value / value if value else np.nan,
                    "source_file": FIG3_Q8_CSV.as_posix(),
                }
            )
    return pd.DataFrame(rows)


def plot_fig3_q8_attribution(df: pd.DataFrame, out_dir: Path) -> None:
    # We want a single grouped bar chart comparing the three cases (raw, k=2, k=max)
    # across the four metrics (Runtime, DRAM Read, L2 Sectors, L1/TEX Sectors).
    metrics = ["runtime_ms", "dram_mb", "l2_sector_proxy", "l1tex_sector_proxy"]
    metric_labels = ["Runtime", "DRAM Read", "L2 Sectors", "L1/TEX Sectors"]
    
    # We will compute the x positions
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    
    # Get values for each case
    vals_raw = []
    vals_k2 = []
    vals_kmax = []
    
    for metric in metrics:
        vals_raw.append(df[(df["metric"] == metric) & (df["label"] == "raw")]["ratio_to_raw"].values[0])
        vals_k2.append(df[(df["metric"] == metric) & (df["label"] == "k=2")]["ratio_to_raw"].values[0])
        vals_kmax.append(df[(df["metric"] == metric) & (df["label"] == "k=max")]["ratio_to_raw"].values[0])
        
    bars_raw = ax.bar(x - width, vals_raw, width, color=COLOR_RAW, edgecolor="black", linewidth=0.8, label="raw FP64")
    bars_k2 = ax.bar(x, vals_k2, width, color=COLOR_K2, edgecolor="black", linewidth=0.8, label="k=2 primary")
    bars_kmax = ax.bar(x + width, vals_kmax, width, color=COLOR_KMAX, edgecolor="black", linewidth=0.8, hatch="//", label="k=max fallback")
    
    ax.axhline(1.0, color=COLOR_RAW, linestyle="--", linewidth=1.0)
    ax.text(len(metrics) - 0.6, 1.02, "raw FP64 baseline", va="bottom", ha="right", fontsize=9, color=COLOR_RAW)
    
    ax.set_ylabel("Relative to raw FP64", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.22)
    
    # Add values on top of bars
    add_bar_labels(ax, list(bars_raw), fmt="{:.2f}")
    add_bar_labels(ax, list(bars_k2), fmt="{:.2f}")
    add_bar_labels(ax, list(bars_kmax), fmt="{:.2f}")
    
    ax.legend(frameon=False, loc="upper right")
    
    # Update title and save
    fig.suptitle("Q8 physical-traffic attribution consistency of direction (cesm_atm_q s10)", y=0.98, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(fig, out_dir, "fig3_q8_traffic_attribution")


def build_fig4_frame() -> pd.DataFrame:
    frames = []
    for dataset, path in [("uniform_p10", FIG4_PROMPT5_UNIFORM), ("heavy_tailed_p6", FIG4_PROMPT5_HEAVY)]:
        df = load_csv(path)
        sub = df[(df["selectivity_label"] == "s50") & (df["kernel"] == "fused")].copy()
        sub["dataset_label"] = dataset
        sub["k"] = sub["k"].astype(int)
        sub["speedup_vs_raw"] = sub["speedup_vs_raw"].astype(float)
        sub["dram_ratio_vs_raw"] = sub["dram_ratio_vs_raw"].astype(float)
        sub["dram_reduction_vs_raw"] = 1.0 / sub["dram_ratio_vs_raw"]
        sub["policy_region"] = np.where(sub["k"] >= 3, "fallback", "dispatch_window")
        sub["source_file"] = path.as_posix()
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def plot_fig4_prompt5_break_even(df: pd.DataFrame, out_dir: Path) -> None:
    # 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), sharex=False)
    
    datasets = ["uniform_p10", "heavy_tailed_p6"]
    colors = {"speedup_vs_raw": COLOR_K2, "dram_reduction_vs_raw": COLOR_KMAX}
    
    for col_idx, dataset in enumerate(datasets):
        sub = df[df["dataset_label"].eq(dataset)].copy().sort_values("k")
        ks = sub["k"].to_numpy(dtype=float)
        speedup = sub["speedup_vs_raw"].to_numpy(dtype=float)
        dram_red = sub["dram_reduction_vs_raw"].to_numpy(dtype=float)
        
        # Row 0: Latency Speedup
        ax_lat = axes[0, col_idx]
        ax_lat.axhline(1.0, color=COLOR_RAW, linestyle="--", linewidth=1.0, label="raw FP64 baseline")
        ax_lat.axvspan(2.5, max(ks.max(), 6.5), color=BACKGROUND_FALLBACK, alpha=0.9)
        
        ax_lat.plot(ks, speedup, marker="o", linewidth=2.0, color=colors["speedup_vs_raw"], label="latency speedup")
        ax_lat.set_title(dataset.replace("_", " ") + " - Latency Speedup", fontsize=11, fontweight="bold")
        ax_lat.set_xticks(ks)
        ax_lat.set_ylim(0.65, 1.25)
        ax_lat.grid(True, axis="y", alpha=0.22)
        
        # Labels for dispatch vs fallback
        ax_lat.text(1.5 if dataset == "heavy_tailed_p6" else 2.0, 1.18, "dispatch window", ha="center", fontsize=8, color=COLOR_RAW, fontweight="bold")
        ax_lat.text(4.2 if dataset == "heavy_tailed_p6" else 3.2, 0.72, "latency win disappears", ha="center", fontsize=8, color=COLOR_RAW, fontweight="bold")
        
        if col_idx == 0:
            ax_lat.set_ylabel("Warm E2E Speedup vs raw FP64", fontweight="bold")
            ax_lat.legend(frameon=False, loc="upper right")
            
        # Row 1: DRAM Reduction
        ax_dram = axes[1, col_idx]
        ax_dram.axvspan(2.5, max(ks.max(), 6.5), color=BACKGROUND_FALLBACK, alpha=0.9)
        
        ax_dram.plot(ks, dram_red, marker="s", linewidth=1.8, color=colors["dram_reduction_vs_raw"], label="DRAM reduction")
        ax_dram.set_title(dataset.replace("_", " ") + " - DRAM Reduction", fontsize=11, fontweight="bold")
        ax_dram.set_xlabel("execution depth k", fontweight="bold")
        ax_dram.set_xticks(ks)
        ax_dram.set_ylim(0, max(8.5, float(dram_red.max()) * 1.1))
        ax_dram.grid(True, axis="y", alpha=0.22)
        
        ax_dram.text(4.2 if dataset == "heavy_tailed_p6" else 3.2, float(dram_red.max()) * 0.4, "DRAM savings persist", ha="center", fontsize=8, color=COLOR_RAW, fontweight="bold")
        
        if col_idx == 0:
            ax_dram.set_ylabel("DRAM Reduction Factor vs raw FP64", fontweight="bold")
            ax_dram.legend(frameon=False, loc="upper right")
            
    fig.suptitle("Prompt5 break-even: DRAM savings persist while latency win disappears on NVIDIA H200", y=0.98, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    save_figure(fig, out_dir, "fig4_prompt5_break_even")


def write_readme(out_dir: Path) -> None:
    readme = "\n".join(
        [
            "# Figure Packet v2 for Scoped H200 Byte-Plane Filter+Aggregate",
            "",
            "This packet is the claim-facing redesign requested for issue #103.",
            "It keeps the scientific headline narrow, keeps synthetic mechanism evidence separate, and demotes the threshold-preparation locality story to appendix/deployment-note status.",
            "",
            "## Reproducibility",
            "",
            "Regenerate all figures and CSV snapshots with:",
            "",
            "```bash",
            "python3 scripts/plot_paper_v1_scoped_h200_figures_v2.py --out-dir results/paper_v1/figures_scoped_h200_v2",
            "```",
            "",
            "## Figure 1 - Scientific headline speedup",
            "",
            "- Artifacts: `fig1_scientific_headline.png`, `fig1_scientific_headline.pdf`",
            "- Snapshot: `fig1_scientific_headline_snapshot.csv`",
            "- Sources: `results/exp4_filter_aggregate/scientific_locality_attribution_57768/scientific_locality_attribution.csv` and `results/exp4_filter_aggregate/scientific_extra_fields_quasi_global_58440/scientific_extra_fields_quasi_global.csv`.",
            "- Caption draft: Four true-quasi-global scientific fields, normalized against raw FP64 = 1x. The k=2 shallow approximate operating point represents the primary progressive approximate path, while the k=max fallback depth represents the full encoded-depth diagnostic or fallback path. Note that hurricane_tc uses the bfp-dec11 precision variant. For cesm_atm_q high-selectivity (90%) bucket caveat: the k=2 row is the 65.02% reported-selectivity bucket row (not a paired 90% threshold match).",
            "- Claim boundary: Scientific headline only; no cold-start or universal speedup claim.",
            "",
            "## Figure 2 - SUM tolerance-depth-throughput tradeoff mechanism",
            "",
            "- Artifacts: `fig2_sum_mechanism.png`, `fig2_sum_mechanism.pdf`",
            "- Snapshot: `fig2_sum_mechanism_snapshot.csv`",
            "- Sources: `results/precision_throughput/exp3_v2_sum_kstar_relative_empirical.csv` and `results/precision_throughput/exp3_v2_sum_precision_throughput.csv`.",
            "- Caption draft: SUM tolerance-depth-throughput tradeoff mechanism on H200. Left panel shows epsilon tolerance -> selected depth k* heatmap for representative datasets (sensor, heavy_tailed) and precisions. Right panels show the direct throughput (top) and relative error (bottom) vs execution depth k tradeoff for a representative precision p6. Sensor is the fast-converging control case, and heavy_tailed is the limitation case demonstrating precision/tradeoff collapse.",
            "- Claim boundary: Synthetic mechanism evidence only; not dispatch policy. The full four-dataset source snapshot is retained, but the rendered panels focus on the clearest contrast pair to keep the axes readable.",
            "",
            "## Figure 3 - Q8 physical-traffic attribution",
            "",
            "- Artifacts: `fig3_q8_traffic_attribution.png`, `fig3_q8_traffic_attribution.pdf`",
            "- Snapshot: `fig3_q8_traffic_attribution_snapshot.csv`",
            "- Source: `results/exp4_filter_aggregate/q8_hbm_transaction_20260523_003039_job59096_NVIDIAH200/q8_hbm_transaction_summary.csv`.",
            "- Caption draft: Grouped normalized bar chart demonstrating consistency-of-direction across runtime, DRAM read, L2 sectors, and L1/TEX sectors compared to raw FP64 = 1x, for one scoped scientific case (cesm_atm_q s10).",
            "- Claim boundary: One scientific-field NCU case only; not universal physical-traffic proof. Do not state this proves universal traffic reduction.",
            "",
            "## Figure 4 - Prompt5 break-even / raw-FP64 fallback",
            "",
            "- Artifacts: `fig4_prompt5_break_even.png`, `fig4_prompt5_break_even.pdf`",
            "- Snapshot: `fig4_prompt5_break_even_snapshot.csv`",
            "- Sources: `results/prompt5_h200_viability/ncu_kdepth_20260523_135553_job59368_NVIDIAH200/k_depth_ncu_matrix.csv` and `results/prompt5_h200_viability/ncu_kdepth_20260523_165051_job59418_NVIDIAH200/k_depth_ncu_matrix.csv`.",
            "- Caption draft: Split E2E latency speedup (top row) and DRAM reduction (bottom row) across execution depths (k) for synthetic workloads on H200. Shaded region indicates fallback to raw FP64. DRAM savings persist, but latency wins disappear past k>=3.",
            "- Claim boundary: Synthetic dispatch evidence only; no scientific-field generalization.",
            "",
            "## Explicitly demoted from the main figure set",
            "",
            "- Threshold-preparation locality remains useful context, but it is not a main-text contribution figure in v2.",
            "- Keep that story in appendix or deployment-note material only if needed for reviewer defense.",
            "",
        ]
    )
    (out_dir / "README.md").write_text(readme + "\n", encoding="utf-8")


def write_snapshot(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    ensure_out_dir(out_dir)

    fig1 = build_fig1_frame()
    plot_fig1_scientific_headline(fig1, out_dir)
    write_snapshot(fig1, out_dir, "fig1_scientific_headline_snapshot")

    eps, points, throughput = build_fig2_frame()
    plot_fig2_sum_mechanism(
        eps[eps["dataset"].isin(DATASET_ORDER)].copy(),
        throughput[throughput["dataset"].isin(DATASET_ORDER)].copy(),
        out_dir
    )
    write_snapshot(points, out_dir, "fig2_sum_mechanism_snapshot")

    fig3 = build_fig3_frame()
    plot_fig3_q8_attribution(fig3, out_dir)
    write_snapshot(fig3, out_dir, "fig3_q8_traffic_attribution_snapshot")

    fig4 = build_fig4_frame()
    plot_fig4_prompt5_break_even(fig4, out_dir)
    write_snapshot(fig4, out_dir, "fig4_prompt5_break_even_snapshot")

    write_readme(out_dir)


if __name__ == "__main__":
    main()
