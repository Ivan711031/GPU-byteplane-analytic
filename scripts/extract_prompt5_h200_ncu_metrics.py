#!/usr/bin/env python3
"""
extract_prompt5_h200_ncu_metrics.py

Extract NCU metrics from ncu_stdout.txt files produced by
run_prompt5_h200_kdepth_ncu_viability.sbatch and produce
k_depth_ncu_matrix.csv.

Usage:
    python3 scripts/extract_prompt5_h200_ncu_metrics.py <result_dir>

The result_dir should have the structure:
    result_dir/
      case_<CASE>_fused_k<K>/
        ncu_stdout.txt
        bench.csv
      case_<CASE>_raw_fp64/
        ncu_stdout.txt
        bench.csv
      summary.csv

Output:
    result_dir/k_depth_ncu_matrix.csv
"""

import argparse
import csv
import os
import re
import sys


METRIC_PATTERNS = {
    "gpu__time_duration.sum": re.compile(
        r"gpu__time_duration\.sum\s+(ns|us|ms|s)\s+([\d.]+)"
    ),
    "dram__bytes_read.sum": re.compile(
        r"dram__bytes_read\.sum\s+(byte|KB|MB|GB|Mbyte|Gbyte)\s+([\d,.]+)"
    ),
    "dram__bytes_write.sum": re.compile(
        r"dram__bytes_write\.sum\s+(byte|KB|MB|GB|Mbyte|Gbyte)\s+([\d,.]+)"
    ),
    "lts__t_sectors_srcunit_tex_op_read.sum": re.compile(
        r"lts__t_sectors_srcunit_tex_op_read\.sum\s+sector\s+([\d,.]+)"
    ),
    "lts__t_sectors_srcunit_tex_op_write.sum": re.compile(
        r"lts__t_sectors_srcunit_tex_op_write\.sum\s+sector\s+([\d,.]+)"
    ),
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": re.compile(
        r"l1tex__t_sectors_pipe_lsu_mem_global_op_ld\.sum\s+sector\s+([\d,.]+)"
    ),
    "smsp__inst_executed.sum": re.compile(
        r"smsp__inst_executed\.sum\s+inst\s+([\d,.]+)"
    ),
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": re.compile(
        r"smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active\.ratio\s+inst\s+([\d.]+)"
    ),
    "smsp__warps_active.avg.per_cycle_active": re.compile(
        r"smsp__warps_active\.avg\.per_cycle_active\s+warp\s+([\d.]+)"
    ),
    "smsp__warps_eligible.avg.per_cycle_active": re.compile(
        r"smsp__warps_eligible\.avg\.per_cycle_active\s+warp\s+([\d.]+)"
    ),
}

FALLBACK_PATTERNS = {
    "gpu__time_duration.sum": [
        re.compile(r"gpu__time_duration\.sum\s+(ns|us|ms|s)\s+([\d.]+)"),
        re.compile(r"Duration\s+(ns|us|ms|s)\s+([\d.]+)"),
        re.compile(r"gpu__time_duration\s+(ns|us|ms|s)\s+([\d.]+)"),
    ],
    "dram__bytes_read.sum": [
        re.compile(r"dram__bytes_read\.sum\s+(byte|KB|MB|GB|Mbyte|Gbyte)\s+([\d,.]+)"),
        re.compile(r"DRAM Read\s+(byte|KB|MB|GB)\s+([\d,.]+)"),
    ],
    "lts__t_sectors_srcunit_tex_op_read.sum": [
        re.compile(r"lts__t_sectors_srcunit_tex_op_read\.sum\s+sector\s+([\d,.]+)"),
        re.compile(r"L2 Read Sectors\s+([\d,.]+)"),
    ],
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": [
        re.compile(r"l1tex__t_sectors_pipe_lsu_mem_global_op_ld\.sum\s+sector\s+([\d,.]+)"),
        re.compile(r"L1\/Tex Read Sectors\s+([\d,.]+)"),
    ],
    "smsp__inst_executed.sum": [
        re.compile(r"smsp__inst_executed\.sum\s+inst\s+([\d,.]+)"),
        re.compile(r"Instructions\s+inst\s+([\d,.]+)"),
    ],
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": [
        re.compile(r"smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active\.ratio\s+inst\s+([\d.]+)"),
        re.compile(r"Long Scoreboard\s+([\d.]+)"),
    ],
    "smsp__warps_eligible.avg.per_cycle_active": [
        re.compile(r"smsp__warps_eligible\.avg\.per_cycle_active\s+warp\s+([\d.]+)"),
        re.compile(r"Eligible Warps\s+warp\s+([\d.]+)"),
    ],
}

REQUIRED_METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_write.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "smsp__inst_executed.sum",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
]


def parse_metric_value(text: str) -> float:
    text = text.strip()
    text = text.replace(",", "")
    return float(text)


def convert_to_bytes(value: float, unit: str) -> float:
    unit = unit.lower().strip()
    if unit in ("byte", "bytes"):
        return value
    elif unit == "kb":
        return value * 1024
    elif unit in ("mb", "mbyte"):
        return value * 1024 * 1024
    elif unit in ("gb", "gbyte"):
        return value * 1024 * 1024 * 1024
    return value


def convert_time_to_us(value: float, unit: str) -> float:
    unit = unit.lower().strip()
    if unit == "ns":
        return value / 1000.0
    elif unit == "us":
        return value
    elif unit == "ms":
        return value * 1000.0
    elif unit == "s":
        return value * 1_000_000.0
    return value


def extract_metrics(ncu_stdout_path: str):
    if not os.path.isfile(ncu_stdout_path):
        return None, []
    with open(ncu_stdout_path, "r") as f:
        text = f.read()

    metrics = {}
    found_keys = set()

    for key, pattern in METRIC_PATTERNS.items():
        m = pattern.search(text)
        if m:
            groups = m.groups()
            if key in ("gpu__time_duration.sum",):
                unit = groups[0]
                value = parse_metric_value(groups[1])
                metrics[key] = convert_time_to_us(value, unit)
                found_keys.add(key)
            elif key in ("dram__bytes_read.sum", "dram__bytes_write.sum"):
                unit = groups[0]
                value = parse_metric_value(groups[1])
                metrics[key] = convert_to_bytes(value, unit)
                found_keys.add(key)
            else:
                metrics[key] = parse_metric_value(groups[-1])
                found_keys.add(key)

    for key, patterns in FALLBACK_PATTERNS.items():
        if key in found_keys:
            continue
        for fallback_re in patterns:
            m = fallback_re.search(text)
            if m:
                groups = m.groups()
                if not groups:
                    continue
                if key in ("dram__bytes_read.sum", "dram__bytes_write.sum"):
                    if len(groups) >= 2:
                        val = parse_metric_value(groups[1])
                        val = convert_to_bytes(val, groups[0])  # type: ignore
                    else:
                        val = parse_metric_value(groups[0])
                    metrics[key] = val
                    found_keys.add(key)
                elif key == "gpu__time_duration.sum":
                    if len(groups) >= 2:
                        val = parse_metric_value(groups[1])
                        val = convert_time_to_us(val, groups[0])  # type: ignore
                    else:
                        val = parse_metric_value(groups[0])
                    metrics[key] = val
                    found_keys.add(key)
                else:
                    metrics[key] = parse_metric_value(groups[-1])
                    found_keys.add(key)
                break

    # Report missing required metrics
    missing = [m for m in REQUIRED_METRICS if m not in found_keys]
    if missing:
        print(
            f"  Warning: {os.path.basename(ncu_stdout_path)} missing metrics: {missing}",
            file=sys.stderr,
        )

    return metrics, missing


def parse_case_dir(case_dir_name: str):
    """Return (case_id, kernel, k) or None."""
    m = re.match(r"case_(\w+)_fused_k(\d+)", case_dir_name)
    if m:
        return m.group(1), "fused", int(m.group(2))
    m = re.match(r"case_(\w+)_raw_fp64", case_dir_name)
    if m:
        return m.group(1), "raw_fp64", None
    return None


def read_bench_csv(bench_csv_path: str) -> dict:
    if not os.path.isfile(bench_csv_path):
        return {}
    with open(bench_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = {}
            if "ms_per_iter" in row and row["ms_per_iter"]:
                result["latency_us"] = float(row["ms_per_iter"]) * 1000.0
            if "raw_baseline_ms_per_iter" in row and row["raw_baseline_ms_per_iter"]:
                result["raw_latency_us"] = float(row["raw_baseline_ms_per_iter"]) * 1000.0
            if "speedup_vs_raw" in row and row["speedup_vs_raw"]:
                result["speedup_vs_raw"] = float(row["speedup_vs_raw"])
            return result
    return {}


def normalize_metric_name(name: str) -> str:
    mapping = {
        "gpu__time_duration.sum": "latency_us",
        "dram__bytes_read.sum": "dram_read_bytes",
        "dram__bytes_write.sum": "dram_write_bytes",
        "lts__t_sectors_srcunit_tex_op_read.sum": "l2_read_sectors",
        "lts__t_sectors_srcunit_tex_op_write.sum": "l2_write_sectors",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "l1_global_load_sectors",
        "smsp__inst_executed.sum": "instructions",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "long_scoreboard",
        "smsp__warps_active.avg.per_cycle_active": "achieved_warps_active",
        "smsp__warps_eligible.avg.per_cycle_active": "eligible_warps_avg",
    }
    return mapping.get(name, name)


def parse_summary_csv(summary_csv_path: str) -> dict:
    """
    Read summary.csv and return a dict mapping (case_id, kernel_target) -> row.
    For fused kernels, the key also includes k: (case_id, "fused", k).
    """
    result = {}
    if not os.path.isfile(summary_csv_path):
        return result
    with open(summary_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row.get("case_id", "")
            kernel_target = row.get("kernel_target", "")
            k = row.get("k", "")
            if kernel_target == "fused" and k:
                key = (case_id, "fused", k)
            elif kernel_target == "raw_fp64":
                key = (case_id, "raw_fp64", "raw")
            else:
                key = (case_id, kernel_target)
            result[key] = row
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract Prompt5 NCU metrics into k_depth_ncu_matrix.csv"
    )
    parser.add_argument("result_dir", help="Path to the NCU run result directory")
    args = parser.parse_args()

    result_dir = args.result_dir
    if not os.path.isdir(result_dir):
        print(f"Error: result_dir not found: {result_dir}", file=sys.stderr)
        sys.exit(1)

    summary_csv_path = os.path.join(result_dir, "summary.csv")
    summary_data = parse_summary_csv(summary_csv_path)

    rows = []
    for entry in sorted(os.listdir(result_dir)):
        case_dir_path = os.path.join(result_dir, entry)
        if not os.path.isdir(case_dir_path):
            continue

        parsed = parse_case_dir(entry)
        if parsed is None:
            continue
        case_id, kernel, k = parsed

        ncu_stdout_path = os.path.join(case_dir_path, "ncu_stdout.txt")
        metrics, missing_list = extract_metrics(ncu_stdout_path)

        if metrics is None:
            print(f"Warning: no ncu_stdout.txt found in {case_dir_path}", file=sys.stderr)
            continue

        bench_csv_path = os.path.join(case_dir_path, "bench.csv")
        bench_data = read_bench_csv(bench_csv_path)

        row = {
            "case_id": case_id,
            "dataset": "",
            "artifact_label": "",
            "selectivity_label": "",
            "kernel": kernel,
            "k": str(k) if k is not None else "raw",
        }

        # Populate metadata from summary.csv
        if kernel == "fused":
            lookup_key = (case_id, "fused", row["k"])
        else:
            lookup_key = (case_id, "raw_fp64", "raw")
        if lookup_key in summary_data:
            srow = summary_data[lookup_key]
            row["dataset"] = srow.get("dataset", "")
            row["artifact_label"] = srow.get("dataset_artifact", "")
            row["selectivity_label"] = srow.get("selectivity_label", "")

        # Add benchmark timing
        if "latency_us" in bench_data:
            row["latency_us"] = bench_data["latency_us"]
        if "raw_latency_us" in bench_data:
            row["raw_latency_us"] = bench_data["raw_latency_us"]
        if kernel == "raw_fp64":
            row["speedup_vs_raw"] = 1.0
        elif "speedup_vs_raw" in bench_data:
            row["speedup_vs_raw"] = bench_data["speedup_vs_raw"]

        # Add NCU metrics
        for metric_key, metric_value in metrics.items():
            col_name = normalize_metric_name(metric_key)
            row[col_name] = metric_value

        row["dram_read_bytes"] = metrics.get("dram__bytes_read.sum", "")
        row["l2_read_sectors"] = metrics.get("lts__t_sectors_srcunit_tex_op_read.sum", "")
        row["l1_global_load_sectors"] = metrics.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "")
        row["instructions"] = metrics.get("smsp__inst_executed.sum", "")
        row["long_scoreboard"] = metrics.get(
            "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", ""
        )
        row["achieved_warps_active"] = metrics.get(
            "smsp__warps_active.avg.per_cycle_active", ""
        )
        row["eligible_warps_avg"] = metrics.get(
            "smsp__warps_eligible.avg.per_cycle_active", ""
        )

        latency_us = metrics.get("gpu__time_duration.sum", 0)
        dram_bytes = metrics.get("dram__bytes_read.sum", 0)
        if latency_us > 0 and dram_bytes > 0:
            row["achieved_dram_read_bw_TBps"] = (dram_bytes / 1e12) / (latency_us / 1e6)
        else:
            row["achieved_dram_read_bw_TBps"] = ""

        if missing_list:
            row["missing_metrics"] = "; ".join(missing_list)

        row["verdict"] = ""
        row["main_bottleneck"] = ""
        row["notes"] = ""

        rows.append(row)

    # Second pass: compute dram_ratio_vs_raw for fused k cases
    raw_dram = {}
    for row in rows:
        if row["kernel"] == "raw_fp64":
            raw_dram[row["case_id"]] = row.get("dram_read_bytes", "")

    for row in rows:
        if row["kernel"] == "fused" and row["case_id"] in raw_dram:
            raw_val = raw_dram[row["case_id"]]
            fused_val = row.get("dram_read_bytes", "")
            if raw_val != "" and fused_val != "" and float(raw_val) > 0:
                row["dram_ratio_vs_raw"] = float(fused_val) / float(raw_val)
            else:
                row["dram_ratio_vs_raw"] = ""

    fieldnames = [
        "case_id", "dataset", "artifact_label", "selectivity_label",
        "kernel", "k",
        "latency_us", "speedup_vs_raw",
        "dram_read_bytes", "dram_ratio_vs_raw",
        "l2_read_sectors", "l1_global_load_sectors",
        "instructions", "long_scoreboard",
        "achieved_warps_active", "eligible_warps_avg", "achieved_dram_read_bw_TBps",
        "main_bottleneck", "verdict",
        "missing_metrics", "notes",
    ]

    output_path = os.path.join(result_dir, "k_depth_ncu_matrix.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    total_missing = sum(1 for r in rows if r.get("missing_metrics"))
    print(f"Written: {output_path} ({len(rows)} rows, {total_missing} with missing metrics)")


if __name__ == "__main__":
    main()
