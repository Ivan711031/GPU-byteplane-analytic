#!/usr/bin/env python3

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import an Nsight Compute report and extract the Exp3 bottleneck metrics."
    )
    parser.add_argument("--report", required=True, help="Path to the .ncu-rep file")
    parser.add_argument("--label", required=True, help="Short label for this profile")
    parser.add_argument("--csv-out", help="Optional output CSV path")
    parser.add_argument("--json-out", help="Optional output JSON path")
    return parser.parse_args()


def parse_number(text: str):
    text = text.strip()
    if not text or text == "no data":
        return None
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def choose_data_row(header, rows):
    if not rows:
        raise RuntimeError("no data rows found in imported ncu report")
    if len(rows) == 1:
        return rows[0]

    kernel_idx = header.index("Kernel Name")
    inst_idx = header.index("smsp__inst_executed.sum") if "smsp__inst_executed.sum" in header else -1

    matching = [row for row in rows if "progressive_sum_rowpack16" in row[kernel_idx]]
    candidates = matching or rows
    if inst_idx >= 0:
        def inst_value(row):
            value = parse_number(row[inst_idx])
            if isinstance(value, (int, float)):
                return float(value)
            return -1.0

        return max(candidates, key=inst_value)
    return candidates[0]


def import_raw_csv(report: str):
    out = subprocess.check_output(
        ["ncu", "--import", report, "--csv", "--page", "raw"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    rows = list(csv.reader(io.StringIO(out)))
    if len(rows) < 3:
        raise RuntimeError(f"unexpected ncu --page raw output for {report!r}")
    header, units = rows[0], rows[1]
    values = choose_data_row(header, rows[2:])
    metrics = {}
    for i, name in enumerate(header):
        metrics[name] = {
            "unit": units[i] if i < len(units) else "",
            "raw": values[i] if i < len(values) else "",
            "value": parse_number(values[i]) if i < len(values) else None,
        }
    return metrics


def pick(metrics, *names):
    for name in names:
        metric = metrics.get(name)
        if metric and metric["value"] is not None:
            return {
                "name": name,
                "unit": metric["unit"],
                "raw": metric["raw"],
                "value": metric["value"],
            }
    return {
        "name": "",
        "unit": "",
        "raw": "",
        "value": None,
    }


def build_summary(label: str, report: str, metrics):
    kernel_name = metrics.get("Kernel Name", {}).get("raw", "")
    device = metrics.get("Device", {}).get("raw", "")
    grid_size = metrics.get("Grid Size", {}).get("raw", "")
    block_size = metrics.get("Block Size", {}).get("raw", "")

    summary = {
        "label": label,
        "report": report,
        "kernel_name": kernel_name,
        "device": device,
        "grid_size": grid_size,
        "block_size": block_size,
        "dram_pct": pick(
            metrics,
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__cycles_active.avg.pct_of_peak_sustained_elapsed",
            "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
        ),
        "inst_executed_sum": pick(metrics, "smsp__inst_executed.sum", "inst_executed"),
        "stall_long_scoreboard": pick(
            metrics,
            "smsp__stall_long_scoreboard",
            "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        ),
        "stall_exec_dependency": pick(
            metrics,
            "smsp__stall_exec_dependency",
            "smsp__average_warps_issue_stalled_exec_dependency_per_issue_active.ratio",
        ),
        "stall_branch_resolving": pick(
            metrics,
            "smsp__stall_branch_resolving",
            "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
        ),
        "warps_active_pct": pick(metrics, "sm__warps_active.avg.pct_of_peak_sustained_active"),
        "stall_short_scoreboard": pick(
            metrics,
            "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        ),
        "stall_wait": pick(metrics, "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"),
        "stall_dispatch": pick(
            metrics,
            "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
        ),
        "local_loads": pick(metrics, "sass__inst_executed_local_loads", "smsp__sass_inst_executed_op_local_ld.sum"),
        "local_stores": pick(
            metrics, "sass__inst_executed_local_stores", "smsp__sass_inst_executed_op_local_st.sum"
        ),
        "global_loads": pick(metrics, "sass__inst_executed_global_loads", "smsp__sass_inst_executed_op_global_ld.sum"),
        "global_stores": pick(metrics, "sass__inst_executed_global_stores", "smsp__sass_inst_executed_op_global_st.sum"),
        "branch_inst_pct": pick(metrics, "derived__smsp__inst_executed_op_branch_pct"),
    }
    return summary


def csv_row(summary):
    row = {
        "label": summary["label"],
        "report": summary["report"],
        "kernel_name": summary["kernel_name"],
        "device": summary["device"],
        "grid_size": summary["grid_size"],
        "block_size": summary["block_size"],
    }
    for key in (
        "dram_pct",
        "inst_executed_sum",
        "stall_long_scoreboard",
        "stall_exec_dependency",
        "stall_branch_resolving",
        "warps_active_pct",
        "stall_short_scoreboard",
        "stall_wait",
        "stall_dispatch",
        "local_loads",
        "local_stores",
        "global_loads",
        "global_stores",
        "branch_inst_pct",
    ):
        row[f"{key}_name"] = summary[key]["name"]
        row[f"{key}_unit"] = summary[key]["unit"]
        row[f"{key}_value"] = summary[key]["raw"]
    return row


def write_csv(path: Path, row):
    fieldnames = list(row.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    metrics = import_raw_csv(args.report)
    summary = build_summary(args.label, args.report, metrics)

    if args.csv_out:
      write_csv(Path(args.csv_out), csv_row(summary))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
