#!/usr/bin/env python3
"""Generate the four main Exp1 figures."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "results" / "exp1" / "run_20260415_rowpack_full_after_impl"
NCU_DIR = ROOT / "results" / "exp1" / "ncu_rowpack_after_impl_20260416"
OUT_DIR = ROOT / "results" / "exp1" / "plots"

CAPTION = "Synthetic constant byte-plane data initialized with 0xAB, N=100M rows."

SERIES = [
    ("byte_baseline", "byte baseline", "#1f77b4", "o"),
    ("byte_ilp4", "byte ilp4", "#ff7f0e", "s"),
    ("rowpack4", "rowpack4", "#2ca02c", "^"),
    ("rowpack16", "rowpack16", "#d62728", "D"),
]

CONTIG_LABEL = "contiguous64"
CONTIG_COLOR = "#4d4d4d"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_benchmark_data() -> tuple[dict[str, list[dict[str, float]]], dict[str, float]]:
    data = {}
    for key, _, _, _ in SERIES:
        rows = read_rows(BENCH_DIR / f"{key}.csv")
        data[key] = sorted(
            (
                {
                    "k": float(row["k"]),
                    "logical_GBps": float(row["logical_GBps"]),
                    "ms_per_iter": float(row["ms_per_iter"]),
                }
                for row in rows
            ),
            key=lambda row: row["k"],
        )

    contig = read_rows(BENCH_DIR / "contiguous64.csv")[0]
    return data, {
        "logical_GBps": float(contig["logical_GBps"]),
        "ms_per_iter": float(contig["ms_per_iter"]),
    }


def load_sass_widths() -> tuple[dict[str, int], dict[str, str]]:
    rows = read_rows(NCU_DIR / "sass_load_summary.csv")
    keys = {
        "byte_ilp4": "byte ilp4",
        "rowpack4": "rowpack4",
        "rowpack16": "rowpack16",
        "contiguous64": "contiguous64",
    }
    widths = {}
    opcodes = {}
    for key, label in keys.items():
        matched = [row for row in rows if row["report"] == key or row["report"].startswith(f"{key}_")]
        found_widths = set()
        found_opcodes = []
        for row in matched:
            for opcode, bits in re.findall(r"([^;:]+):(\d+)b", row["load_opcodes"]):
                found_widths.add(int(bits))
                if opcode not in found_opcodes:
                    found_opcodes.append(opcode)
        if not found_widths:
            raise RuntimeError(f"No SASS load width found for {key}")
        if len(found_widths) != 1:
            raise RuntimeError(f"Mixed SASS load widths for {key}: {sorted(found_widths)}")
        widths[label] = found_widths.pop()
        opcodes[label] = ", ".join(found_opcodes)
    return widths, opcodes


def style_axis(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8)
    ax.grid(True, axis="x", color="#eeeeee", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def add_caption(fig) -> None:
    fig.text(0.5, 0.015, CAPTION, ha="center", va="bottom", fontsize=9, color="#555555")


def save_line_chart(
    title: str,
    ylabel: str,
    out_path: Path,
    series: list[dict[str, object]],
    hline_label: str,
    hline_value: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=180)
    for item in series:
        points = item["points"]
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            label=str(item["label"]),
            color=str(item["color"]),
            marker=str(item["marker"]),
            linewidth=2.2,
            markersize=5,
        )

    ax.axhline(
        hline_value,
        label=hline_label,
        color=CONTIG_COLOR,
        linewidth=1.8,
        linestyle=(0, (5, 4)),
    )
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_xlim(0.75, 8.25)
    ax.set_xticks(range(1, 9))
    ax.set_ylim(bottom=0)
    style_axis(ax, "k = number of byte-planes read", ylabel)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    add_caption(fig)
    fig.tight_layout(rect=(0, 0.04, 0.86, 1))
    fig.savefig(out_path)
    plt.close(fig)


def save_sass_load_width_chart(out_path: Path) -> None:
    widths, opcodes = load_sass_widths()
    labels = ["byte ilp4", "rowpack4", "rowpack16", "contiguous64"]
    colors = ["#ff7f0e", "#2ca02c", "#d62728", CONTIG_COLOR]

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=180)
    values = [widths[label] for label in labels]
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_title("Exp1 SASS Load Width Evidence", loc="left", fontsize=13, fontweight="bold")
    style_axis(ax, "strategy", "load width bits")
    ax.set_ylim(0, max(values) * 1.25)
    for label, value, bar in zip(labels, values, bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.035,
            f"{value}-bit\n{opcodes[label]}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.text(
        0.5,
        0.05,
        "NCU source view confirms rowpack16 emits 128-bit global loads, not scalar byte loads.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    add_caption(fig)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data, contig = load_benchmark_data()

    throughput_series = [
        {
            "label": label,
            "color": color,
            "marker": marker,
            "points": [(row["k"], row["logical_GBps"]) for row in data[key]],
        }
        for key, label, color, marker in SERIES
    ]
    save_line_chart(
        "Exp1 Main Throughput vs k",
        "Logical throughput (GB/s)",
        OUT_DIR / "exp1_throughput_vs_k.png",
        throughput_series,
        CONTIG_LABEL,
        contig["logical_GBps"],
    )

    speedup_series = [
        {
            "label": label,
            "color": color,
            "marker": marker,
            "points": [(row["k"], row["logical_GBps"] / contig["logical_GBps"]) for row in data[key]],
        }
        for key, label, color, marker in SERIES
    ]
    save_line_chart(
        "Exp1 Speedup over contiguous64",
        "speedup vs contiguous64",
        OUT_DIR / "exp1_speedup_vs_contiguous64.png",
        speedup_series,
        "contiguous64 = 1.0x",
        1.0,
    )

    time_series = [
        {
            "label": label,
            "color": color,
            "marker": marker,
            "points": [(row["k"], row["ms_per_iter"]) for row in data[key]],
        }
        for key, label, color, marker in SERIES
    ]
    save_line_chart(
        "Exp1 Time per Iteration vs k",
        "ms per iteration",
        OUT_DIR / "exp1_time_vs_k.png",
        time_series,
        CONTIG_LABEL,
        contig["ms_per_iter"],
    )

    save_sass_load_width_chart(OUT_DIR / "exp1_sass_load_width.png")

    print("Wrote:")
    for name in [
        "exp1_throughput_vs_k.png",
        "exp1_speedup_vs_contiguous64.png",
        "exp1_time_vs_k.png",
        "exp1_sass_load_width.png",
    ]:
        print(OUT_DIR / name)


if __name__ == "__main__":
    main()
