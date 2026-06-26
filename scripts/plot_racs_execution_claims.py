#!/usr/bin/env python3
"""Generate separate execution-result figures for the RACS execution story."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "research" / "figures"

FIG1 = ROOT / "results" / "paper_v1" / "figures_scoped_h200" / "src_fig1_scientific_headline.csv"
FIG3 = ROOT / "results" / "paper_v1" / "figures_scoped_h200" / "src_fig3_threshold_locality.csv"
FIG4 = ROOT / "results" / "paper_v1" / "figures_scoped_h200_v2" / "fig3_q8_traffic_attribution_snapshot.csv"
FIG5 = ROOT / "results" / "paper_v1" / "figures_scoped_h200_v2" / "fig4_prompt5_break_even_snapshot.csv"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def save(fig: plt.Figure, stem: str) -> None:
    svg_path = OUT_DIR / f"{stem}.svg"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {svg_path}")
    print(f"wrote {png_path}")


def plot_a_single_dataset_latency(rows: list[dict[str, str]]) -> None:
    # Representative execution dataset used here: CESM cloud.
    dataset = "cesm_atm_cloud"
    subset = [row for row in rows if row["dataset"] == dataset and row["k_type"] == "k_star"]
    subset.sort(key=lambda row: int(row["selectivity"]))

    xs = [int(row["selectivity"]) for row in subset]
    k2_ms = [float(row["warm_e2e_ms"]) for row in subset]
    raw_ms = [float(row["raw_fused_ms"]) for row in subset]
    speedups = [float(row["speedup"]) for row in subset]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(xs, k2_ms, marker="o", linewidth=2.4, color="#1f77b4", label="Byte-plane fused (k=2)")
    ax.plot(xs, raw_ms, marker="s", linewidth=2.2, color="#d62728", label="Raw-fused FP64")

    for x, y, s in zip(xs, k2_ms, speedups, strict=True):
        ax.text(x, y + 0.01, f"{s:.2f}x", ha="center", va="bottom", fontsize=9, color="#1f77b4")

    ax.set_title("Execution A. CESM cloud latency vs. selectivity")
    ax.set_xlabel("Selectivity bucket (%)")
    ax.set_ylabel("Warm latency (ms)")
    ax.set_xticks(xs)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "racs_execution_figA_cesm_cloud_latency")


def plot_b_locality_boundary(rows: list[dict[str, str]]) -> None:
    # Use a single dataset to avoid the broken mixed-dataset view.
    dataset = "CESM Cloud Fraction"
    selectivity_order = ["10%", "50%", "90%"]
    locality_order = ["168.48M (Quasi-Global)", "65536 (Mid)", "4096 (Local)"]
    locality_short = {
        "168.48M (Quasi-Global)": "Quasi-global",
        "65536 (Mid)": "65536",
        "4096 (Local)": "4096",
    }
    locality_color = {
        "168.48M (Quasi-Global)": "#1f77b4",
        "65536 (Mid)": "#ff7f0e",
        "4096 (Local)": "#d62728",
    }

    subset = [row for row in rows if row["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=(8.8, 4.4))

    width = 0.22
    group_gap = 0.55

    x_positions: list[float] = []
    x_labels: list[str] = []

    for group_idx, sel in enumerate(selectivity_order):
        base = group_idx * (len(locality_order) * width + group_gap)
        for loc_idx, locality in enumerate(locality_order):
            row = next(r for r in subset if r["selectivity"] == sel and r["segment_size"] == locality)
            x = base + loc_idx * width
            total = float(row["warm_e2e_ms"])
            scan = float(row["scan_only_ms"])
            ax.bar(
                x,
                total,
                width=width,
                color=locality_color[locality],
                alpha=0.9,
                label=locality_short[locality] if group_idx == 0 else "",
            )
            ax.plot(
                x,
                scan,
                marker="_",
                markersize=18,
                markeredgewidth=2.2,
                color="#111111",
                linestyle="None",
                label="scan-only" if (group_idx, loc_idx) == (0, 0) else "",
            )
            x_positions.append(x)
            x_labels.append(locality_short[locality])

        center = base + width
        ax.text(center, -0.18, sel, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)

    ax.set_title("Execution B. Locality boundary on CESM cloud")
    ax.set_ylabel("Warm E2E latency (ms)")
    ax.set_xticks(x_positions, x_labels, rotation=0)
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    fig.tight_layout()
    save(fig, "racs_execution_figB_locality_boundary")


def plot_c_scoped_traffic(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    labels = ["raw", "k=2", "k=max"]
    runtime_ms = [float(next(r for r in rows if r["metric"] == "runtime_ms" and r["label"] == label)["value"]) for label in labels]
    dram_mb = [float(next(r for r in rows if r["metric"] == "dram_mb" and r["label"] == label)["value"]) for label in labels]

    x = [0, 1, 2]
    width = 0.34
    ax.bar([p - width / 2 for p in x], runtime_ms, width=width, color="#1f77b4", label="Runtime (ms)")
    ax.bar([p + width / 2 for p in x], dram_mb, width=width, color="#ff7f0e", label="DRAM read (MB)")

    ax.set_title("Execution C. Scoped Q8 traffic attribution (`cesm_atm_q`, s10)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Measured value")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "racs_execution_figC_q8_traffic")


def plot_d_depth_calibration(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    styles = {
        "uniform_p10": {"color": "#1f77b4", "label": "uniform_p10"},
        "heavy_tailed_p6": {"color": "#d62728", "label": "heavy_tailed_p6"},
    }

    for artifact_label, style in styles.items():
        subset = [row for row in rows if row["artifact_label"] == artifact_label and row["kernel"] == "fused"]
        subset.sort(key=lambda row: int(row["k"]))
        xs = [int(row["k"]) for row in subset]
        ys = [float(row["speedup_vs_raw"]) for row in subset]
        ax.plot(xs, ys, marker="o", linewidth=2.3, markersize=5, color=style["color"], label=style["label"])

    ax.axhline(1.0, color="#666666", linewidth=1.0, linestyle=":")
    ax.set_title("Execution D. Synthetic depth calibration")
    ax.set_xlabel("Execution depth k")
    ax.set_ylabel("Speedup vs raw FP64")
    ax.set_xticks([1, 2, 3, 4, 6])
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "racs_execution_figD_depth_calibration")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scientific_rows = load_csv(FIG1)
    locality_rows = load_csv(FIG3)
    traffic_rows = load_csv(FIG4)
    depth_rows = load_csv(FIG5)

    plot_a_single_dataset_latency(scientific_rows)
    plot_b_locality_boundary(locality_rows)
    plot_c_scoped_traffic(traffic_rows)
    plot_d_depth_calibration(depth_rows)


if __name__ == "__main__":
    main()
