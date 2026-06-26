#!/usr/bin/env python3
"""Combined comparative review for Reliability Phase 1.5.

Reads per-strategy msb_lsb_summary.csv and phase1_5_strategy_review_report.txt,
applies PRD combined verdict rules, writes final Phase 1.5 review packet.

Usage:
  python3 scripts/aggregate_reliability_phase1_5_combined.py \
    --strategy-dirs <dir1> <dir2> <dir3> \
    --params-json results/phase1_5_strategy_params.json \
    --output-dir /tmp/phase1_5/combined_comparative_20260531
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path

DATASETS = ["sensor", "uniform", "heavy_tailed", "zipfian"]
FAULT_RATES = ["1e-8", "1e-7", "1e-6", "1e-5", "1e-4"]
STOP_GO_DATASETS = ["sensor", "uniform"]
ALL_STRATEGY_IDS = [
    "native_scale1", "per_dataset", "hybrid_scale100_plus_shifted"]
TIE_BREAK_RATES = ["1e-7", "1e-6", "1e-5"]


def load_strategy_data(strategy_dir: Path) -> dict:
    """Load per-strategy data from one strategy's combined output directory."""
    run_meta = strategy_dir / "run_meta.txt"
    summary_csv = strategy_dir / "msb_lsb_summary.csv"
    report_txt = strategy_dir / "phase1_5_strategy_review_report.txt"

    for f in [run_meta, summary_csv, report_txt]:
        if not f.is_file():
            raise FileNotFoundError(f"Required file not found: {f}")

    strategy_id = None
    for line in run_meta.read_text().splitlines():
        if line.startswith("strategy_id="):
            strategy_id = line.split("=", 1)[1].strip()
            break
    if not strategy_id:
        raise ValueError(f"strategy_id not found in {run_meta}")

    report_text = report_txt.read_text()
    verdict = None
    for line in report_text.splitlines():
        if "Verdict:" in line:
            verdict = line.split("Verdict:")[-1].strip()
            break
    if not verdict:
        raise ValueError(f"Verdict not found in {report_txt}")
    if verdict not in ("PASS", "MARGINAL", "FAIL"):
        raise ValueError(f"Unknown verdict '{verdict}' in {report_txt}")

    summary_rows = []
    with summary_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        summary_rows = list(reader)

    norm_ratios: dict[tuple[str, str], str] = {}
    raw_ratios: dict[tuple[str, str], str] = {}
    occ_map: dict[str, list[float]] = {}
    for row in summary_rows:
        ds = row["dataset"]
        rate = row["fault_rate"]
        norm_ratios[(ds, rate)] = row["normalized_ratio"]
        raw_ratios[(ds, rate)] = row["primary_ratio_raw"]
        if ds not in occ_map:
            occ = []
            for p in range(8):
                occ.append(float(row.get(f"plane_{p}_occupancy", "0.0")))
            occ_map[ds] = occ

    oracle_gates = _parse_oracle_gates(report_text)

    return {
        "strategy_id": strategy_id,
        "verdict": verdict,
        "summary_rows": summary_rows,
        "norm_ratios": norm_ratios,
        "raw_ratios": raw_ratios,
        "occupancy": occ_map,
        "oracle_gates": oracle_gates,
    }


def _parse_oracle_gates(report_text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    lines = report_text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        for ds in DATASETS:
            if s.startswith(ds) and "canonical," in s:
                if i + 1 < len(lines):
                    nxt = lines[i + 1].strip()
                    if "Oracle gate:" in nxt:
                        result[ds] = nxt.split("Oracle gate:")[-1].strip()
                break
    return result


def parse_normalized_ratio(val: str) -> float | None:
    if val == "UNDEFINED":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def pick_winner(strategies: dict[str, dict]) -> str | None:
    """Multi-PASS tie-break: strategy with highest min sensor+uniform ratio."""
    pick = None
    best_min_ratio = -1.0
    for sid, data in strategies.items():
        if data["verdict"] != "PASS":
            continue
        ratios = []
        for ds in STOP_GO_DATASETS:
            for rate in TIE_BREAK_RATES:
                r = parse_normalized_ratio(
                    data["norm_ratios"].get((ds, rate), "UNDEFINED"))
                if r is not None:
                    ratios.append(r)
        if not ratios:
            continue
        min_ratio = min(ratios)
        if min_ratio > best_min_ratio:
            best_min_ratio = min_ratio
            pick = sid
    return pick


def determine_combined_verdict(
    strategies: dict[str, dict],
) -> tuple[str, str | None, object]:
    """Returns (verdict, winner, details)."""
    pass_sids = [s for s, d in strategies.items() if d["verdict"] == "PASS"]
    marginal_sids = [
        s for s, d in strategies.items() if d["verdict"] == "MARGINAL"]

    if pass_sids:
        if len(pass_sids) == 1:
            return ("GO", pass_sids[0], "single PASS")
        winner = pick_winner(strategies)
        return ("GO", winner, "multi-PASS tie-break")

    if marginal_sids:
        best_marginal = None
        best_min_ratio = -1.0
        for sid in marginal_sids:
            data = strategies[sid]
            ratios = []
            for ds in STOP_GO_DATASETS:
                for rate in TIE_BREAK_RATES:
                    r = parse_normalized_ratio(
                        data["norm_ratios"].get((ds, rate), "UNDEFINED"))
                    if r is not None:
                        ratios.append(r)
            if ratios:
                mr = min(ratios)
                if mr > best_min_ratio:
                    best_min_ratio = mr
                    best_marginal = sid

        marginal_rows = []
        for sid in marginal_sids:
            data = strategies[sid]
            for ds in STOP_GO_DATASETS:
                for rate in TIE_BREAK_RATES:
                    r = data["norm_ratios"].get((ds, rate), "UNDEFINED")
                    marginal_rows.append((sid, ds, rate, r))

        return ("CONDITIONAL GO", best_marginal, marginal_rows)

    return ("NO-GO", None, None)


def generate_summary_csv(strategies: dict[str, dict],
                         output_dir: Path) -> None:
    header = [
        "strategy_id", "dataset", "fault_rate",
        "msb_group_median_raw", "lsb_group_median_raw", "primary_ratio_raw",
        "normalized_msb_group_median", "normalized_lsb_group_median",
        "normalized_ratio",
        "undefined_ratio_flag", "noise_denominator_flag",
        "per_strategy_verdict",
    ]
    rows = []
    for sid in ALL_STRATEGY_IDS:
        if sid not in strategies:
            continue
        data = strategies[sid]
        for sr in data["summary_rows"]:
            rows.append({
                "strategy_id": sid,
                "dataset": sr["dataset"],
                "fault_rate": sr["fault_rate"],
                "msb_group_median_raw": sr["msb_group_median_raw"],
                "lsb_group_median_raw": sr["lsb_group_median_raw"],
                "primary_ratio_raw": sr["primary_ratio_raw"],
                "normalized_msb_group_median":
                    sr["normalized_msb_group_median"],
                "normalized_lsb_group_median":
                    sr["normalized_lsb_group_median"],
                "normalized_ratio": sr["normalized_ratio"],
                "undefined_ratio_flag": sr["undefined_ratio_flag"],
                "noise_denominator_flag": sr["noise_denominator_flag"],
                "per_strategy_verdict": data["verdict"],
            })

    path = output_dir / "strategies_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Strategies summary: {path}  ({len(rows)} rows)")


def format_ratio_cell(sid: str, ds: str, rate: str,
                      strategies: dict[str, dict]) -> str:
    val = strategies[sid]["norm_ratios"].get((ds, rate), "N/A")
    return str(val)


def generate_report(strategies: dict[str, dict], params: dict,
                    output_dir: Path) -> None:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Phase 1.5 Combined Comparative Review")
    lines.append(f"Date: {date.today().isoformat()}")
    lines.append("=" * 72)

    verdict, winner, details = determine_combined_verdict(strategies)

    # ── Per-Strategy Verdict Table ──
    lines.append("\n=== Per-Strategy Verdict Table ===")
    hdr = (
        f"{'Strategy':<35} {'sensor':>10} {'uniform':>10} "
        f"{'heavy_tailed':>14} {'zipfian':>10} {'Verdict':>12}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for sid in ALL_STRATEGY_IDS:
        if sid not in strategies:
            continue
        data = strategies[sid]
        og = data.get("oracle_gates", {})
        s_cell = og.get("sensor", "-")
        u_cell = og.get("uniform", "-")
        h_cell = og.get("heavy_tailed", "-")
        z_cell = og.get("zipfian", "-")
        row = (
            f"{sid:<35} {s_cell:>10} {u_cell:>10} "
            f"{h_cell:>14} {z_cell:>10} {data['verdict']:>12}")
        lines.append(row)

    # ── Strategy Applicability Table ──
    lines.append("\n=== Strategy Applicability Table ===")
    hdr = (
        f"{'Strategy':<35} {'sensor':>12} {'uniform':>12} "
        f"{'heavy_tailed':>16} {'zipfian':>12}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for sid in ALL_STRATEGY_IDS:
        if sid not in strategies:
            continue
        data = strategies[sid]
        datasets_present = set(r["dataset"] for r in data["summary_rows"])
        s_params = params.get("strategies", {}).get(sid,
                                                     {}).get("per_dataset", {})
        cells = []
        for ds in DATASETS:
            ds_params = s_params.get(ds, {})
            if ds_params.get("inapplicable", False):
                cells.append("inapplicable")
            elif ds in datasets_present:
                cells.append("applicable")
            else:
                cells.append("inapplicable")
        row = (
            f"{sid:<35} {cells[0]:>12} {cells[1]:>12} "
            f"{cells[2]:>16} {cells[3]:>12}")
        lines.append(row)

    # ── Side-by-side Normalized MSB-vs-LSB Ratio (sensor + uniform) ──
    lines.append(
        "\n=== Side-by-side Normalized MSB-vs-LSB Ratio "
        "(sensor + uniform) ===")
    for ds in STOP_GO_DATASETS:
        lines.append(f"\n  Dataset: {ds}")
        hdr = f"{'fault_rate':<12}"
        for sid in ALL_STRATEGY_IDS:
            if sid in strategies:
                hdr += f"  {sid:>35}"
        lines.append(hdr)
        sep = f"{'':-<12}"
        for sid in ALL_STRATEGY_IDS:
            if sid in strategies:
                sep += f"  {'':->35}"
        lines.append(sep)
        for rate in FAULT_RATES:
            rl = f"{rate:<12}"
            for sid in ALL_STRATEGY_IDS:
                if sid in strategies:
                    val = format_ratio_cell(sid, ds, rate, strategies)
                    rl += f"  {val:>35}"
            lines.append(rl)

    # ── Side-by-side Normalized MSB-vs-LSB Ratio (heavy_tailed + zipfian) ──
    lines.append(
        "\n=== Side-by-side Normalized MSB-vs-LSB Ratio "
        "(heavy_tailed + zipfian) ===")
    lines.append("(non-blocking, for reference only)")
    for ds in ["heavy_tailed", "zipfian"]:
        lines.append(f"\n  Dataset: {ds}")
        hdr = f"{'fault_rate':<12}"
        for sid in ALL_STRATEGY_IDS:
            if sid in strategies:
                hdr += f"  {sid:>35}"
        lines.append(hdr)
        sep = f"{'':-<12}"
        for sid in ALL_STRATEGY_IDS:
            if sid in strategies:
                sep += f"  {'':->35}"
        lines.append(sep)
        for rate in FAULT_RATES:
            rl = f"{rate:<12}"
            for sid in ALL_STRATEGY_IDS:
                if sid in strategies:
                    val = format_ratio_cell(sid, ds, rate, strategies)
                    rl += f"  {val:>35}"
            lines.append(rl)

    # ── Plane Occupancy Summary ──
    lines.append("\n=== Plane Occupancy Summary ===")
    lines.append("(planes with nonzero_fraction < 1e-6 per strategy)")
    for sid in ALL_STRATEGY_IDS:
        if sid not in strategies:
            continue
        data = strategies[sid]
        lines.append(f"\n  Strategy: {sid}")
        for ds in DATASETS:
            occ = data["occupancy"].get(ds)
            if occ is None:
                lines.append(f"    {ds}: no data")
                continue
            sparse = [str(p) for p in range(8) if occ[p] < 1e-6]
            count = len(sparse)
            occ_str = "  ".join(
                f"P{p}={occ[p]:.8f}" for p in range(8))
            lines.append(
                f"    {ds}: planes with <1e-6: {count}"
                f" ({', '.join(sparse) if sparse else 'none'})")
            lines.append(f"      [{occ_str}]")

    # ── Winner Determination ──
    lines.append("\n=== Winner Determination ===")
    if verdict == "GO":
        lines.append(f"Named winner: {winner}")
        lines.append(f"Determination: {details}")
        pass_count = len(
            [s for s, d in strategies.items() if d["verdict"] == "PASS"])
        if pass_count > 1:
            lines.append(
                "Tie-break: highest minimum normalized ratio on "
                "sensor+uniform at rates 1e-7, 1e-6, 1e-5")
    elif verdict == "CONDITIONAL GO":
        lines.append(
            f"Named winner: {winner}")
        lines.append(
            "  (highest-normalized MARGINAL strategy)")
        lines.append("MARGINAL ratios on sensor+uniform at 1e-7/1e-6/1e-5:")
        if details:
            for sid, ds, rate, val in details:
                lines.append(f"  {sid}  {ds}  {rate}: {val}")
        lines.append(
            "User acceptance required before Phase 2 proceeds.")
    else:
        lines.append("Named winner: none")
        lines.append("All strategies FAIL.")

    # ── Phase 1.5 Verdict ──
    lines.append("\n=== Phase 1.5 Verdict ===")
    lines.append(verdict)

    # ── Recommendations ──
    lines.append("\n=== Recommendations ===")
    if verdict == "GO" and winner:
        w_params = params.get("strategies", {}).get(winner, {})
        w_desc = w_params.get("description", "")
        lines.append(
            f"Winner strategy '{winner}' ({w_desc}) "
            f"recommended for Phase 2 sensitivity profile.")
        lines.append("Parameters:")
        for k, v in w_params.items():
            if k != "per_dataset":
                lines.append(f"  {k}: {v}")
        lines.append(
            "Artifact path: "
            f"{params['metadata'].get('source_raw_root', 'N/A')}"
            f"/artifacts_phase1_5/{winner}/")
    elif verdict == "CONDITIONAL GO" and winner:
        lines.append(
            "Phase 2 may proceed only if the user explicitly "
            "accepts the marginal signal.")
        lines.append(
            "The specific marginal normalized ratios are listed "
            "above under Winner Determination.")
        lines.append(
            f"If accepted, use winner '{winner}' for "
            "Phase 2 sensitivity profile.")
    else:
        lines.append(
            "ALL STRATEGIES FAIL: rescaling does not resolve "
            "the Phase 1 weak normalized signal.")
        lines.append(
            "Phase 2 protection policy direction is not supported "
            "by encoded sensitivity evidence.")
        lines.append(
            "Next step: mechanism redesign, not another "
            "rescaling candidate.")
        lines.append(
            "A separate research note must propose the next "
            "mechanism before any further reliability-aware "
            "byte-plane work.")

    report_path = output_dir / "phase1_5_combined_review_report.txt"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Combined review report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--params-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    params = json.loads(args.params_json.read_text())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies: dict[str, dict] = OrderedDict()
    for sd in args.strategy_dirs:
        data = load_strategy_data(sd)
        sid = data["strategy_id"]
        strategies[sid] = data
        print(f"Loaded strategy '{sid}' from {sd}")

    missing = [s for s in ALL_STRATEGY_IDS if s not in strategies]
    if missing:
        print(
            f"WARNING: missing strategies: {missing}",
            file=sys.stderr)

    generate_summary_csv(strategies, output_dir)
    generate_report(strategies, params, output_dir)


if __name__ == "__main__":
    main()
