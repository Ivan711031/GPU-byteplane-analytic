#!/usr/bin/env python3
"""Plot NMR-D2 expected bound width vs fault rate from full-scale outputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "reliability_layer1" / "phase4" / "nmr_d2" / "figures"

DATASET_FILES = {
    "hurricane_u": ROOT
    / "results"
    / "reliability_layer1"
    / "phase4"
    / "nmr_d2"
    / "hurricane_u_fullscale"
    / "job_88081"
    / "nmr_d2_stochastic_matrix.csv",
    "cesm_atm_cloud": ROOT
    / "results"
    / "reliability_layer1"
    / "phase4"
    / "nmr_d2"
    / "cesm_atm_cloud_fullscale"
    / "job_88206"
    / "nmr_d2_stochastic_matrix.csv",
}

POLICY_STYLE = {
    "graded_seg_B3": {
        "label": "graded_seg_B3",
        "color": "#1f77b4",
        "marker": "o",
    },
    "uniform_spread_seg_B3": {
        "label": "uniform_spread_seg_B3",
        "color": "#d62728",
        "marker": "s",
    },
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, list[float] | float]]]:
    grouped: dict[str, dict[str, dict[float, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        grouped[row["dataset"]][row["policy"]][float(row["fault_rate"])].append(float(row["expected_bound_width"]))

    summary: dict[str, dict[str, dict[str, list[float] | float]]] = {}
    for dataset, by_policy in grouped.items():
        summary[dataset] = {}
        for policy, by_rate in by_policy.items():
            rates = sorted(by_rate)
            means = [sum(by_rate[rate]) / len(by_rate[rate]) for rate in rates]
            mins = [min(by_rate[rate]) for rate in rates]
            maxs = [max(by_rate[rate]) for rate in rates]
            summary[dataset][policy] = {
                "rates": rates,
                "means": means,
                "mins": mins,
                "maxs": maxs,
            }
    return summary


def plot_dataset(ax: plt.Axes, dataset: str, dataset_summary: dict[str, dict[str, list[float] | float]]) -> None:
    for policy, series in dataset_summary.items():
        style = POLICY_STYLE[policy]
        rates = series["rates"]
        means = series["means"]
        mins = series["mins"]
        maxs = series["maxs"]

        ax.plot(
            rates,
            means,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=7,
            label=style["label"],
        )
        ax.fill_between(rates, mins, maxs, color=style["color"], alpha=0.16)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(dataset)
    ax.set_xlabel("fault rate")
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.35)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    merged_summary: dict[str, dict[str, dict[str, list[float] | float]]] = {}
    for _, csv_path in DATASET_FILES.items():
        merged_summary.update(summarize(load_rows(csv_path)))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), sharey=True)
    dataset_order = ["hurricane_u", "cesm_atm_cloud"]
    for ax, dataset in zip(axes, dataset_order, strict=True):
        plot_dataset(ax, dataset, merged_summary[dataset])

    axes[0].set_ylabel("expected bound width")
    axes[1].text(
        0.03,
        0.06,
        "Cesm full-scale currently has one\nheadline rate (2e-5) only.",
        transform=axes[1].transAxes,
        fontsize=10,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "boxstyle": "round,pad=0.3", "alpha": 0.85},
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("NMR-D2: Fault Rate vs Expected Bound Width", fontsize=15, y=1.08)
    fig.text(
        0.5,
        -0.02,
        "Full-scale D2 outputs under plane-uniform single-replica logical fault injection.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout()

    png_path = OUT_DIR / "nmr_d2_fault_rate_vs_expected_bound_width.png"
    svg_path = OUT_DIR / "nmr_d2_fault_rate_vs_expected_bound_width.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
