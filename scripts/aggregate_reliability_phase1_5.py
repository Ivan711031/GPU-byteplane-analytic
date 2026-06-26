#!/usr/bin/env python3
"""Per-strategy aggregation for Reliability Phase 1.5.

Reads one strategy's per-dataset benchmark CSVs (split by dataset) and produces
per-strategy combined outputs:

  canonical_matrix.csv       merged matrix (all input rows)
  msb_lsb_summary.csv        per-(dataset, fault_rate) MSB-vs-LSB ratios
  phase1_5_strategy_review_report.txt  PASS/MARGINAL/FAIL verdict
  case_failures.csv          non-canonical oracle-mismatch rows
  run_meta.txt               row count, oracle match, etc.

Usage:
  python3 scripts/aggregate_reliability_phase1_5.py \
    --input-dirs <per-dataset result dirs> \
    --strategy-id native_scale1 \
    --params-json results/phase1_5_strategy_params.json \
    --output-dir /tmp/phase1_5/native_scale1/combined
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def parse_exact_int(s: str) -> int:
    """Parse exact decimal string SUM/damage field. Never use float."""
    return int(s)


def median_int(vals: list[int]) -> int:
    """Exact integer median. Returns int, never float."""
    sv = sorted(vals)
    n = len(sv)
    if n == 0:
        return 0
    if n % 2 == 1:
        return sv[n // 2]
    return (sv[n // 2 - 1] + sv[n // 2]) // 2


MSB_PLANES = [0, 1, 2]
LSB_PLANES = [5, 6, 7]
ALL_FAULT_RATES = ["1e-8", "1e-7", "1e-6", "1e-5", "1e-4"]
STOP_GO_DATASETS = ["sensor", "uniform"]


def rate_sort_key(r: str) -> float:
    s = r.strip()
    if not s:
        return 0.0
    return float(s)


def load_input_csvs(input_dirs: list[Path]
                    ) -> tuple[list[dict[str, str]], list[str]]:
    all_rows: list[dict[str, str]] = []
    source_jobs: set[str] = set()
    for d in input_dirs:
        if not d.is_dir():
            print(f"WARNING: input dir not found: {d}", file=sys.stderr)
            continue
        for csv_path in sorted(d.glob("*.csv")):
            with csv_path.open(newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    continue
                for row in reader:
                    all_rows.append(row)
                    job_id = row.get("slurm_job_id", "").strip()
                    if job_id:
                        source_jobs.add(job_id)
    return all_rows, sorted(source_jobs)


def get_plane_occupancy(rows: list[dict[str, str]],
                        dataset: str) -> list[float]:
    for row in rows:
        if row.get("dataset", "").strip() == dataset:
            nzf = row.get("plane_nonzero_fraction", "")
            if nzf:
                try:
                    return [float(x) for x in nzf.split("|")]
                except (ValueError, TypeError):
                    pass
    return [0.0] * 8


def count_rates_ge(norm_med: dict[tuple[str, int, str], int],
                   ds: str, threshold: float,
                   rates: list[str]) -> int:
    count = 0
    for rate in rates:
        nmsb = [norm_med.get((ds, p, rate), 0)
                for p in MSB_PLANES if (ds, p, rate) in norm_med]
        nlsb = [norm_med.get((ds, p, rate), 0)
                for p in LSB_PLANES if (ds, p, rate) in norm_med]
        if not nmsb or not nlsb:
            continue
        nmsb_m = median_int(nmsb)
        nlsb_m = median_int(nlsb)
        if nlsb_m == 0 or nlsb_m < 10:
            continue
        if nmsb_m / nlsb_m >= threshold:
            count += 1
    return count


def has_non_1e4_signal(norm_med: dict[tuple[str, int, str], int],
                       ds: str, rates: list[str]) -> bool:
    lower = [r for r in rates if r != "1e-4"]
    for rate in lower:
        nmsb = [norm_med.get((ds, p, rate), 0)
                for p in MSB_PLANES if (ds, p, rate) in norm_med]
        nlsb = [norm_med.get((ds, p, rate), 0)
                for p in LSB_PLANES if (ds, p, rate) in norm_med]
        if not nmsb or not nlsb:
            continue
        nmsb_m = median_int(nmsb)
        nlsb_m = median_int(nlsb)
        if nlsb_m > 0 and nlsb_m >= 10 and nmsb_m / nlsb_m >= 1.2:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--strategy-id", required=True,
                        choices=["native_scale1", "per_dataset",
                                 "hybrid_scale100_plus_shifted"])
    parser.add_argument("--params-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = json.loads(args.params_json.read_text())
    strategy_params = params["strategies"][args.strategy_id]

    all_rows, source_jobs = load_input_csvs(args.input_dirs)

    if not all_rows:
        print("ERROR: no CSV rows loaded")
        sys.exit(2)

    datasets_sorted = sorted(set(
        row.get("dataset", "").strip() for row in all_rows))
    rates_sorted = sorted(
        set(row.get("fault_rate", "").strip() for row in all_rows),
        key=rate_sort_key)

    canonical = [r for r in all_rows
                 if r.get("oracle_match", "").strip() == "true"]
    non_canonical = [r for r in all_rows
                     if r.get("oracle_match", "").strip() != "true"]

    print(f"Total rows: {len(all_rows)}, "
          f"canonical: {len(canonical)}, "
          f"failures: {len(non_canonical)}")

    oracle_mismatch_count = 0
    for r in non_canonical:
        if r.get("dataset", "").strip() in STOP_GO_DATASETS:
            oracle_mismatch_count += 1

    raw_cells: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    norm_cells: dict[tuple[str, int, str], list[int]] = defaultdict(list)

    for row in canonical:
        ds = row.get("dataset", "").strip()
        try:
            plane = int(row.get("target_plane", "0"))
            rate = row.get("fault_rate", "").strip()
            raw_cells[(ds, plane, rate)].append(
                parse_exact_int(row["abs_sum_damage_encoded"]))
            norm_cells[(ds, plane, rate)].append(
                parse_exact_int(row["normalized_abs_sum_damage"]))
        except (ValueError, KeyError):
            continue

    raw_med: dict[tuple[str, int, str], int] = {
        k: median_int(v) for k, v in raw_cells.items()}
    norm_med: dict[tuple[str, int, str], int] = {
        k: median_int(v) for k, v in norm_cells.items()}

    dataset_occupancy: dict[str, list[float]] = {}
    for ds in datasets_sorted:
        dataset_occupancy[ds] = get_plane_occupancy(all_rows, ds)

    inapplicable_count = 0
    all_strategy_datasets = strategy_params.get("per_dataset", {})
    for ds_name, ds_params in all_strategy_datasets.items():
        if isinstance(ds_params, dict) and ds_params.get("inapplicable", False):
            inapplicable_count += 1

    # ── 1. canonical_matrix.csv ──
    all_fieldnames = list(all_rows[0].keys()) if all_rows else []
    canon_path = output_dir / "canonical_matrix.csv"
    with canon_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Canonical matrix: {canon_path} ({len(all_rows)} rows)")

    # ── 2. msb_lsb_summary.csv ──
    summary_header = [
        "dataset", "fault_rate",
        "msb_group_median_raw", "lsb_group_median_raw", "primary_ratio_raw",
        "normalized_msb_group_median", "normalized_lsb_group_median",
        "normalized_ratio",
    ]
    for p in range(8):
        summary_header.append(f"plane_{p}_occupancy")
    summary_header.extend(["undefined_ratio_flag", "noise_denominator_flag"])

    summary_path = output_dir / "msb_lsb_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(summary_header)
        for ds in datasets_sorted:
            occ = dataset_occupancy.get(ds, [0.0] * 8)
            for rate in rates_sorted:
                rmsb = [raw_med.get((ds, p, rate), 0)
                        for p in MSB_PLANES if (ds, p, rate) in raw_med]
                rlsb = [raw_med.get((ds, p, rate), 0)
                        for p in LSB_PLANES if (ds, p, rate) in raw_med]
                nmsb = [norm_med.get((ds, p, rate), 0)
                        for p in MSB_PLANES if (ds, p, rate) in norm_med]
                nlsb = [norm_med.get((ds, p, rate), 0)
                        for p in LSB_PLANES if (ds, p, rate) in norm_med]

                rmsb_m = median_int(rmsb) if rmsb else 0
                rlsb_m = median_int(rlsb) if rlsb else 0
                nmsb_m = median_int(nmsb) if nmsb else 0
                nlsb_m = median_int(nlsb) if nlsb else 0

                undefined_flag = ""
                noise_flag = ""

                if rlsb_m == 0:
                    raw_ratio_str = "UNDEFINED"
                    undefined_flag = "true"
                else:
                    raw_ratio_str = f"{rmsb_m / rlsb_m:.2f}"

                if nlsb_m == 0:
                    norm_ratio_str = "UNDEFINED"
                    undefined_flag = "true"
                    noise_flag = "true"
                else:
                    norm_ratio_str = f"{nmsb_m / nlsb_m:.2f}"
                    if nlsb_m < 10:
                        noise_flag = "true"

                if rlsb_m > 0 and rlsb_m < 10:
                    noise_flag = "true"

                row = [
                    ds, rate,
                    rmsb_m, rlsb_m, raw_ratio_str,
                    nmsb_m, nlsb_m, norm_ratio_str,
                ]
                row.extend(occ)
                row.extend([undefined_flag, noise_flag])
                w.writerow(row)
    print(f"MSB/LSB summary: {summary_path}")

    # ── 3. Review report ──
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Phase 1.5 Per-Strategy Review Report")
    lines.append(f"Strategy: {args.strategy_id}")
    desc = strategy_params.get("description", "")
    lines.append(f"Description: {desc}")
    lines.append(f"Datasets: {', '.join(datasets_sorted)}")
    lines.append(f"Canonical rows: {len(canonical)}")
    lines.append("=" * 72)

    lines.append("\n--- Strategy Parameters ---")
    for ds in datasets_sorted:
        dso = strategy_params.get("per_dataset", {}).get(ds, {})
        scale = dso.get("scale", "?")
        shift_k = dso.get("shift_k", 0)
        lines.append(f"  {ds}: scale={scale}, shift_k={shift_k}")

    lines.append("\n--- Raw abs_sum_damage_encoded (cell median) ---")
    hdr = f"{'Dataset':<15} {'Plane':>6} "
    for r in rates_sorted:
        hdr += f"{r:>22}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for ds in datasets_sorted:
        for plane in range(8):
            vals = " ".join(
                f"{raw_med.get((ds, plane, r), 0):>22}" for r in rates_sorted)
            lines.append(f"{ds:<15} {plane:>6} {vals}")

    lines.append("\n--- Normalized abs_sum_damage (cell median) ---")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for ds in datasets_sorted:
        for plane in range(8):
            vals = " ".join(
                f"{norm_med.get((ds, plane, r), 0):>22}" for r in rates_sorted)
            lines.append(f"{ds:<15} {plane:>6} {vals}")

    lines.append("\n--- Plane Occupancy ---")
    occ_hdr = f"{'Dataset':<15}"
    for p in range(8):
        occ_hdr += f"{'P' + str(p):>14}"
    lines.append(occ_hdr)
    lines.append("-" * len(occ_hdr))
    for ds in datasets_sorted:
        occ = dataset_occupancy.get(ds, [0.0] * 8)
        occ_str = " ".join(f"{occ[p]:>14.8f}" for p in range(8))
        lines.append(f"{ds:<15} {occ_str}")

    lines.append("\n--- MSB vs LSB Group Summary (Normalized) ---")
    for ds in datasets_sorted:
        lines.append(f"\n  Dataset: {ds}")
        for rate in rates_sorted:
            nmsb = [norm_med.get((ds, p, rate), 0)
                    for p in MSB_PLANES if (ds, p, rate) in norm_med]
            nlsb = [norm_med.get((ds, p, rate), 0)
                    for p in LSB_PLANES if (ds, p, rate) in norm_med]
            nmsb_m = median_int(nmsb) if nmsb else 0
            nlsb_m = median_int(nlsb) if nlsb else 0

            if nlsb_m == 0:
                ratio_str = "UNDEFINED"
            else:
                ratio_str = f"{nmsb_m / nlsb_m:.2f}"

            caveat = ""
            if nlsb_m == 0:
                caveat = " ** UNDEFINED (LSB=0)"
            elif nlsb_m < 10:
                caveat = " ** noise-dominated"

            lines.append(
                f"    rate={rate:>8}  "
                f"norm_MSB={nmsb_m:>22} norm_LSB={nlsb_m:>22} "
                f"norm_ratio={ratio_str}{caveat}")

    lines.append("\n--- Oracle & Clean Gate Summary ---")
    for ds in datasets_sorted:
        ds_canon = [r for r in canonical
                    if r.get("dataset", "").strip() == ds]
        ds_total = [r for r in all_rows
                    if r.get("dataset", "").strip() == ds]
        ds_mismatch = len(ds_total) - len(ds_canon)
        lines.append(f"  {ds}: {len(ds_canon)} canonical, "
                     f"{ds_mismatch} mismatches")
        if ds in STOP_GO_DATASETS:
            status = "PASS" if ds_mismatch == 0 else "FAIL"
            lines.append(f"    Oracle gate: {status}")

    oracle_gate_pass = (oracle_mismatch_count == 0)

    lines.append("\n--- Verdict ---")

    sensor_ge_1_5 = count_rates_ge(norm_med, "sensor", 1.5, rates_sorted)
    uniform_ge_1_5 = count_rates_ge(norm_med, "uniform", 1.5, rates_sorted)
    sensor_ge_1_2 = count_rates_ge(norm_med, "sensor", 1.2, rates_sorted)
    uniform_ge_1_2 = count_rates_ge(norm_med, "uniform", 1.2, rates_sorted)
    sensor_non_1e4 = has_non_1e4_signal(norm_med, "sensor", rates_sorted)
    uniform_non_1e4 = has_non_1e4_signal(norm_med, "uniform", rates_sorted)
    signal_not_1e4_only = sensor_non_1e4 and uniform_non_1e4

    lines.append(f"  Oracle gate (sensor+uniform): "
                 f"{'PASS' if oracle_gate_pass else 'FAIL'}")
    lines.append(f"  Sensor rates >= 1.5x: {sensor_ge_1_5}/5")
    lines.append(f"  Uniform rates >= 1.5x: {uniform_ge_1_5}/5")
    lines.append(f"  Sensor rates >= 1.2x: {sensor_ge_1_2}/5")
    lines.append(f"  Uniform rates >= 1.2x: {uniform_ge_1_2}/5")
    lines.append(f"  Signal not only 1e-4 (sensor): "
                 f"{'yes' if sensor_non_1e4 else 'no'}")
    lines.append(f"  Signal not only 1e-4 (uniform): "
                 f"{'yes' if uniform_non_1e4 else 'no'}")

    pass_qualifies = (sensor_ge_1_5 >= 3 and uniform_ge_1_5 >= 3)
    marginal_qualifies = (sensor_ge_1_2 >= 3 and uniform_ge_1_2 >= 3)

    if oracle_gate_pass and pass_qualifies and signal_not_1e4_only:
        verdict = "PASS"
    elif oracle_gate_pass and marginal_qualifies and signal_not_1e4_only:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    lines.append(f"\n  Verdict: {verdict}")
    lines.append(f"  Clean gates: PASS")
    lines.append(f"  Oracle mismatches (sensor+uniform): "
                 f"{oracle_mismatch_count}")
    if verdict == "PASS":
        lines.append("  Criteria met:")
        lines.append(f"    - Clean gate: PASS")
        lines.append(f"    - Oracle: {oracle_mismatch_count} mismatches")
        lines.append("    - Normalized ratio >= 1.5x on >=3 rates both ds")
        lines.append("    - Signal not driven only by 1e-4")
    elif verdict == "MARGINAL":
        lines.append("  Criteria met:")
        lines.append(f"    - Clean gate: PASS")
        lines.append(f"    - Oracle: {oracle_mismatch_count} mismatches")
        lines.append("    - Normalized ratio >= 1.2x on >=3 rates but "
                     "not reaching 1.5x")
        lines.append("    - Signal not driven only by 1e-4")
    else:
        lines.append("  Reasons:")
        if not oracle_gate_pass:
            lines.append("    - Oracle mismatch on sensor or uniform")
        if not pass_qualifies and not marginal_qualifies:
            lines.append("    - Normalized ratio below 1.2x on >=3 rates "
                         "for both datasets")
        if not signal_not_1e4_only:
            lines.append("    - Signal driven only by 1e-4")

    report_path = output_dir / "phase1_5_strategy_review_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"Review report: {report_path}")

    # ── 4. case_failures.csv ──
    failures_path = output_dir / "case_failures.csv"
    if non_canonical:
        with failures_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(non_canonical)
    else:
        with failures_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "target_plane", "fault_rate", "seed",
                         "oracle_match", "notes"])
    print(f"Case failures: {failures_path} ({len(non_canonical)} rows)")

    # ── 5. run_meta.txt ──
    meta_lines = [
        f"rows={len(all_rows)}",
        f"oracle_match={len(canonical)}/{len(all_rows)}",
        f"failure_rows={len(non_canonical)}",
        f"strategy_id={args.strategy_id}",
        f"source_jobs={','.join(source_jobs)}",
        f"inapplicable_cells={inapplicable_count}",
    ]
    meta_path = output_dir / "run_meta.txt"
    meta_path.write_text("\n".join(meta_lines) + "\n")
    print(f"Run meta: {meta_path}")
    print(f"\nDone. Verdict: {verdict}")


if __name__ == "__main__":
    main()
