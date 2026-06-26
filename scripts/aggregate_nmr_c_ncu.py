#!/usr/bin/env python3
"""Aggregate NCU raw reports and latency CSV into unified nmr_c_ncu_summary.csv.

Usage:
  python3 aggregate_nmr_c_ncu.py --results-dir PATH/TO/job_<JOBID>

Output:
  nmr_c_ncu_profile.csv          (latency + bandwidth from benchmark)
  nmr_c_latency_summary.csv      (per-dataset per-path latency summary)
  nmr_c_raw_ncu_reports/         (per-path NCU .ncu-rep files)
  nmr_c_ncu_metrics.csv          (extracted NCU metrics)

Expected NCU metric names (CUDA 12.x):
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  dram__bytes.sum
  lts__t_bytes.sum
  sm__warps_active.avg.pct_of_peak_sustained_elapsed
  sm__issue_active.avg.pct_of_peak_sustained_elapsed
  launcher__average_kernel_time.ns
  sm__inst_executed.sum
  sm__inst_executed.sum.per_cycle_elapsed
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


REQUIRED_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum",
    "lts__t_bytes.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
    "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed.sum",
    "sm__inst_executed.sum.per_cycle_elapsed",
    "launcher__average_kernel_time.ns",
]

STALL_METRICS = [
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_memory_dependency_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_inst_fetch_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
]

ALL_METRICS = REQUIRED_METRICS + STALL_METRICS


def parse_ncu_csv(ncu_csv: Path) -> dict[str, str]:
    """Parse NCU CSV output and return metric_name -> value mapping."""
    metrics: dict[str, str] = {}
    if not ncu_csv.exists():
        return metrics
    with open(ncu_csv) as f:
        reader = csv.reader(f)
        headers = None
        for row in reader:
            if not row or row[0].startswith("#") or row[0].startswith("=="):
                continue
            if headers is None:
                headers = row
                continue
            if len(row) < 2:
                continue
            # NCU CSV: [MetricName, Value, Unit, ...]
            metric_name = row[0].strip()
            value = row[1].strip() if len(row) > 1 else ""
            metrics[metric_name] = value
    return metrics


def run_ncu_export(ncu_rep: Path) -> dict[str, str]:
    """Use ncu --csv --import to extract metrics from .ncu-rep file."""
    metrics: dict[str, str] = {}
    if not ncu_rep.exists():
        print(f"  SKIP: {ncu_rep} not found")
        return metrics

    all_metric_names = ALL_METRICS + ["gpu__time_duration.sum"]

    try:
        result = subprocess.run(
            ["ncu", "--csv", "--import", str(ncu_rep),
             "--metrics", ",".join(all_metric_names)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  WARN: ncu --import failed for {ncu_rep.name}: {result.stderr[:200]}")
            return metrics

        for line in result.stdout.splitlines():
            if line.startswith("#") or line.startswith("=="):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                metric_name = parts[0].strip()
                value = parts[1].strip()
                if value and value != "N/A":
                    metrics[metric_name] = value
    except FileNotFoundError:
        print("  WARN: ncu not found on PATH; cannot extract metrics")
    except subprocess.TimeoutExpired:
        print(f"  WARN: ncu --import timeout for {ncu_rep.name}")

    return metrics


def normalize_metric_name(raw: str) -> str:
    """Map raw NCU metric names to stable short names."""
    mapping = {
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_utilization_pct",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
        "dram__bytes.sum": "dram_bytes",
        "lts__t_bytes.sum": "l2_bytes",
        "sm__warps_active.avg.pct_of_peak_sustained_elapsed": "achieved_occupancy_pct",
        "sm__issue_active.avg.pct_of_peak_sustained_elapsed": "issue_slots_utilization",
        "sm__inst_executed.sum": "instruction_count",
        "sm__inst_executed.sum.per_cycle_elapsed": "ipc",
        "launcher__average_kernel_time.ns": "kernel_time_ns",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "stall_long_scoreboard",
        "smsp__average_warps_issue_stalled_memory_dependency_per_issue_active.ratio": "stall_memory_dependency",
        "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio": "stall_not_selected",
        "smsp__average_warps_issue_stalled_inst_fetch_per_issue_active.ratio": "stall_inst_fetch",
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio": "stall_wait",
        "gpu__time_duration.sum": "gpu_time_ns",
    }
    return mapping.get(raw, raw)


def classify_bottleneck(metrics: dict[str, str]) -> str:
    """Classify dominant bottleneck based on metric heuristics."""
    sm_util = _safe_float(metrics.get("sm_utilization_pct", ""))
    dram_pct = _safe_float(metrics.get("dram_throughput_pct", ""))
    stall_mem = _safe_float(metrics.get("stall_memory_dependency", ""))
    stall_ls = _safe_float(metrics.get("stall_long_scoreboard", ""))

    if sm_util < 30 and dram_pct > 60:
        return "memory_bound"
    elif sm_util < 30 and dram_pct < 30:
        return "latency_bound"
    elif sm_util > 60 and dram_pct < 30:
        return "compute_bound"
    elif stall_mem > 0.3 or stall_ls > 0.3:
        return "memory_latency_bound"
    else:
        return "balanced"


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate NMR-C NCU metrics")
    parser.add_argument("--results-dir", required=True,
                        help="Path to job_<JOBID> results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"FATAL: {results_dir} not found")
        sys.exit(2)

    raw_reports = results_dir / "nmr_c_raw_ncu_reports"
    latency_csv = results_dir / "nmr_c_ncu_profile.csv"

    # ── Step 1: Build latency summary ──
    latency_summary: list[dict[str, str | float]] = []
    if latency_csv.exists():
        with open(latency_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                latency_summary.append({
                    "dataset": row.get("dataset", ""),
                    "n_rows": int(row.get("n_rows", 0)),
                    "path": row.get("path", ""),
                    "latency_ms": float(row.get("latency_ms", 0)),
                    "effective_bandwidth_gb_s": float(row.get("effective_bandwidth_gb_s", 0)),
                })

    # Write latency summary
    lat_fields = ["dataset", "n_rows", "path", "latency_ms", "effective_bandwidth_gb_s"]
    lat_out = results_dir / "nmr_c_latency_summary.csv"
    with open(lat_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lat_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(latency_summary)
    print(f"Wrote {lat_out} ({len(latency_summary)} rows)")

    # ── Step 2: Extract NCU metrics from .ncu-rep files ──
    ncu_rows: list[dict[str, str | float]] = []
    if raw_reports.is_dir():
        for fpath in sorted(raw_reports.iterdir()):
            if fpath.suffix == ".ncu-rep":
                # Parse filename for dataset + path
                stem = fpath.stem
                # Remove _ncu suffix if present
                stem = stem.replace("_ncu", "")
                # Try to split as dataset_path
                parts = stem.split("_", 1) if "_" in stem else [stem, ""]
                ds_label = parts[0] if len(parts) > 0 else "unknown"
                path_label = parts[1] if len(parts) > 1 else stem

                path_label_clean = path_label
                for p in ["P0_baseline_byteplane_k2", "P1_digest_only",
                          "P2_vote_read_compare", "P3_digest_plus_vote",
                          "P4_raw_fused_fp64_reference"]:
                    if p in stem:
                        path_label_clean = p
                        ds_label = stem.replace(f"_{p}", "").replace(f"{p}_", "").replace(p, "")
                        break

                print(f"Extracting NCU metrics: {fpath.name}")
                metrics = run_ncu_export(fpath)
                if not metrics:
                    print(f"  No metrics extracted")
                    continue

                row: dict[str, str | float] = {
                    "dataset": ds_label,
                    "path": path_label_clean,
                    "ncu_metric_version": "cuda12_ncu",
                }

                for raw_name, value in metrics.items():
                    norm = normalize_metric_name(raw_name)
                    row[norm] = _safe_float(value)

                # Derive bottleneck classification
                bw_metric = row.get("dram_throughput_gb_s", 0.0)
                dram_pct = _safe_float(metrics.get(
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed", ""))
                dram_bytes = _safe_float(metrics.get("dram__bytes.sum", ""))
                l2_bytes = _safe_float(metrics.get("lts__t_bytes.sum", ""))
                sm_util = _safe_float(metrics.get(
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed", ""))

                # Convert DRAM bytes to GB/s using kernel time
                kernel_ns = _safe_float(metrics.get("gpu__time_duration.sum", ""))
                if kernel_ns > 0 and dram_bytes > 0:
                    bw_gb_s = dram_bytes / kernel_ns
                    row["dram_throughput_gb_s"] = round(bw_gb_s, 4)
                else:
                    row["dram_throughput_gb_s"] = 0.0

                if l2_bytes > 0 and kernel_ns > 0:
                    l2_bw = l2_bytes / kernel_ns
                    row["l2_throughput_gb_s"] = round(l2_bw, 4)
                else:
                    row["l2_throughput_gb_s"] = 0.0

                row["dominant_bottleneck"] = classify_bottleneck(dict(row))

                # Stall ratios as percentages
                for stall_key in ["stall_long_scoreboard", "stall_memory_dependency",
                                  "stall_not_selected", "stall_inst_fetch", "stall_wait"]:
                    val = _safe_float(str(row.get(stall_key, "0")))
                    row[f"issue_stall_{stall_key}_pct"] = round(val * 100, 2)

                ncu_rows.append(row)

    # Write NCU metrics CSV
    ncu_fields = [
        "dataset", "path", "ncu_metric_version",
        "sm_utilization_pct", "dram_throughput_pct", "dram_throughput_gb_s",
        "l2_throughput_gb_s", "achieved_occupancy_pct",
        "dominant_bottleneck",
        "issue_stall_long_scoreboard_pct", "issue_stall_memory_dependency_pct",
        "issue_stall_not_selected_pct", "issue_stall_inst_fetch_pct",
        "dram_bytes", "l2_bytes", "instruction_count", "ipc",
    ]
    ncu_out = results_dir / "nmr_c_ncu_metrics.csv"
    with open(ncu_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ncu_fields, extrasaction="ignore")
        w.writeheader()
        for r in ncu_rows:
            w.writerow({k: r.get(k, "NA") for k in ncu_fields})
    print(f"Wrote {ncu_out} ({len(ncu_rows)} rows)")

    # ── Step 3: Generate honest verdict ──
    print("\n=== NMR-C Verdict ===")
    if latency_summary:
        # Per-dataset breakdown (NCU-profiled entries are in separate CSVs now)
        by_ds_path = defaultdict(list)
        for r in latency_summary:
            by_ds_path[(r["dataset"], r["path"])].append(r)

        ds_set = sorted(set(r["dataset"] for r in latency_summary))
        print(f"  Datasets: {ds_set}")

        for ds in ds_set:
            print(f"\n  --- {ds} ---")
            p4_rows = [r for r in latency_summary if r["dataset"] == ds and r["path"] == "P4_raw_fused_fp64_reference"]
            if not p4_rows:
                print(f"  No P4 baseline for {ds}, skipping budget check")
                continue
            p4_avg = sum(r["latency_ms"] for r in p4_rows) / len(p4_rows)
            print(f"    P4 (raw FP64): {p4_avg:.4f} ms")

            for p_label in ["P0_baseline_byteplane_k2", "P1_digest_only",
                            "P2_vote_read_compare", "P3_digest_plus_vote"]:
                p_rows = [r for r in latency_summary if r["dataset"] == ds and r["path"] == p_label]
                if not p_rows:
                    continue
                p_avg = sum(r["latency_ms"] for r in p_rows) / len(p_rows)
                ratio = p_avg / max(p4_avg, 0.001)
                print(f"    {p_label}: {p_avg:.4f} ms  ({ratio:.2f}x P4)  "
                      f"{'✅' if p_avg < p4_avg else '⚠️ >P4'}")

        # Honest verdict: check if P1/P2/P3 < P4 per dataset
        all_within_budget = True
        for ds in ds_set:
            p4_rows_ds = [r for r in latency_summary
                          if r["dataset"] == ds and r["path"] == "P4_raw_fused_fp64_reference"]
            if not p4_rows_ds:
                continue
            p4_avg_ds = sum(r["latency_ms"] for r in p4_rows_ds) / len(p4_rows_ds)
            for p in ["P1_digest_only", "P2_vote_read_compare", "P3_digest_plus_vote"]:
                p_vals = [r["latency_ms"] for r in latency_summary
                         if r["dataset"] == ds and r["path"] == p]
                if p_vals:
                    if sum(p_vals) / len(p_vals) >= p4_avg_ds:
                        all_within_budget = False

        if ncu_rows:
            mem_bound = sum(1 for r in ncu_rows
                          if r.get("dominant_bottleneck", "") in ("memory_bound", "memory_latency_bound"))
            compute_bound = sum(1 for r in ncu_rows
                              if r.get("dominant_bottleneck", "") == "compute_bound")
            print(f"\n  NCU bottleneck: memory_bound={mem_bound} compute_bound={compute_bound}")
        else:
            print(f"\n  NCU metrics: not available (metric extraction needs debugging)")

        if all_within_budget:
            print(f"\n  Budget check: ALL P1/P2/P3 < P4 per dataset")
            verdict = "NMR_C_LATENCY_WITHIN_BUDGET_BUT_BOTTLENECK_SHIFTED"
        else:
            print(f"\n  Budget check: P1/P2/P3 EXCEEDS P4 on one or more datasets")
            print(f"  Low-marginal-cost redundancy NOT supported with current implementation")
            print(f"  Root causes: (1) all-plane digest adds cost vs single-pass raw FP64")
            print(f"               (2) vote kernel on full planes adds overhead")
            print(f"               (3) raw FP64 is a simpler kernel (no byte-plane)")
            print(f"  To improve: C2 optimization (fused vote+digest, protected-planes-only)")
            verdict = "NMR_C_NEEDS_FIXES"

        print(f"\n  Verdict: {verdict}")
        with open(results_dir / "verdict.txt", "w") as f:
            f.write(f"{verdict}\n")


if __name__ == "__main__":
    main()
