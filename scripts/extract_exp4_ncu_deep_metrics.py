#!/usr/bin/env python3
"""Extract Exp4 deep-NCU metrics from ncu_stdout.txt files.

Nsight Compute CSV export may fail for imported reports on this cluster, while
the raw stdout table still contains the metrics needed by the Exp4 P1/P2
roadmap. This script parses those stdout tables and writes one compact CSV row
per case/strategy.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


METRICS = {
    "sm_issue_active_pct": "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
    "smsp_issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "eligible_warps_per_cycle": "smsp__warps_eligible.avg.per_cycle_active",
    "long_scoreboard_stall_inst": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "math_pipe_stall_inst": "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "not_selected_stall_inst": "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "wait_stall_inst": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "barrier_stall_inst": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "inst_issued_per_cycle": "sm__inst_issued.avg.per_cycle_active",
    "inst_executed_per_cycle": "sm__inst_executed.avg.per_cycle_active",
    "l1tex_throughput_pct": "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex_sectors_total": "l1tex__t_sectors.sum",
    "l1tex_global_ld_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex_global_ld_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex_global_st_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l2_sectors_total": "lts__t_sectors.sum",
    "l2_sectors_pct": "lts__t_sectors.sum.pct_of_peak_sustained_elapsed",
    "dram_read_mbyte": "dram__bytes_read.sum",
    "dram_write_mbyte": "dram__bytes_write.sum",
    "dram_read_tbps": "dram__bytes_read.sum.per_second",
    "dram_write_gbps": "dram__bytes_write.sum.per_second",
    "dram_read_pct": "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram_read_sectors": "dram__sectors_read.sum",
    "dram_write_sectors": "dram__sectors_write.sum",
    "inst_mix_alu_pct": "sm__inst_executed_pipe_alu_realtime.avg.pct_of_peak_sustained_elapsed",
    "inst_mix_fma_pct": "sm__inst_executed_pipe_fma_realtime.avg.pct_of_peak_sustained_elapsed",
    "inst_mix_fmaheavy_pct": "sm__inst_executed_pipe_fmaheavy_realtime.avg.pct_of_peak_sustained_elapsed",
    "inst_mix_uniform_pct": "sm__inst_executed_pipe_uniform_realtime.avg.pct_of_peak_sustained_elapsed",
}

REQUIRED_METRICS = (
    "smsp_issue_active_pct",
    "eligible_warps_per_cycle",
    "long_scoreboard_stall_inst",
    "l1tex_global_ld_requests",
    "l1tex_global_ld_sectors",
    "dram_read_mbyte",
    "inst_issued_per_cycle",
)


def parse_value(raw: str) -> float | None:
    raw = raw.strip().replace(",", "")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_stdout(path: Path) -> dict[str, float | None]:
    values: dict[str, float | None] = {key: None for key in METRICS}
    wanted_by_suffix = {metric: key for key, metric in METRICS.items()}

    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = re.split(r"\s{2,}", stripped)
        if len(parts) < 2:
            continue
        metric_name = parts[0]
        value = parse_value(parts[-1])
        if value is None:
            continue
        for suffix, key in wanted_by_suffix.items():
            if metric_name == suffix or metric_name.endswith("." + suffix):
                values[key] = value
                break
    return values


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result_dir = args.result_dir
    summary_path = result_dir / "summary.csv"
    output_path = args.output or result_dir / "p1_deep_metrics.csv"

    rows: list[dict[str, str]] = []
    with summary_path.open(newline="") as f:
        for row in csv.DictReader(f):
            case_id = row["case_id"]
            strategy = row.get("strategy") or "byte_mask"
            stdout_path = result_dir / f"case_{case_id}_{strategy}" / "ncu_stdout.txt"
            metrics = parse_stdout(stdout_path)
            missing = [key for key in REQUIRED_METRICS if metrics.get(key) is None]
            if missing:
                missing_text = ", ".join(missing)
                raise RuntimeError(f"missing required metrics for case={case_id} strategy={strategy}: {missing_text}")

            smsp_issue = metrics["smsp_issue_active_pct"]
            skipped_issue = None if smsp_issue is None else 100.0 - smsp_issue
            ld_requests = metrics["l1tex_global_ld_requests"]
            ld_sectors = metrics["l1tex_global_ld_sectors"]
            sectors_per_request = None
            if ld_requests and ld_sectors is not None:
                sectors_per_request = ld_sectors / ld_requests

            out = {
                "case_id": case_id,
                "dataset": row["dataset"],
                "selectivity": row["selectivity"],
                "k": row["k"],
                "strategy": strategy,
                "validated": row["validated"],
                "ms_per_iter": row["ms_per_iter"],
                "ncu_status": row["ncu_status"],
                "sm_issue_active_pct": fmt(metrics["sm_issue_active_pct"]),
                "smsp_issue_active_pct": fmt(smsp_issue),
                "skipped_issue_slots_pct": fmt(skipped_issue),
                "eligible_warps_per_cycle": fmt(metrics["eligible_warps_per_cycle"]),
                "long_scoreboard_stall_inst": fmt(metrics["long_scoreboard_stall_inst"]),
                "math_pipe_stall_inst": fmt(metrics["math_pipe_stall_inst"]),
                "not_selected_stall_inst": fmt(metrics["not_selected_stall_inst"]),
                "wait_stall_inst": fmt(metrics["wait_stall_inst"]),
                "barrier_stall_inst": fmt(metrics["barrier_stall_inst"]),
                "global_ld_sectors_per_request": fmt(sectors_per_request),
                "l1tex_throughput_pct": fmt(metrics["l1tex_throughput_pct"]),
                "l1tex_sectors_total": fmt(metrics["l1tex_sectors_total"]),
                "l1tex_global_ld_requests": fmt(ld_requests),
                "l1tex_global_ld_sectors": fmt(ld_sectors),
                "l1tex_global_st_sectors": fmt(metrics["l1tex_global_st_sectors"]),
                "l2_sectors_total": fmt(metrics["l2_sectors_total"]),
                "l2_sectors_pct": fmt(metrics["l2_sectors_pct"]),
                "dram_read_mbyte": fmt(metrics["dram_read_mbyte"]),
                "dram_write_mbyte": fmt(metrics["dram_write_mbyte"]),
                "dram_read_TBps": fmt(metrics["dram_read_tbps"]),
                "dram_write_GBps": fmt(metrics["dram_write_gbps"]),
                "dram_read_pct": fmt(metrics["dram_read_pct"]),
                "dram_read_sectors": fmt(metrics["dram_read_sectors"]),
                "dram_write_sectors": fmt(metrics["dram_write_sectors"]),
                "inst_issued_per_cycle": fmt(metrics["inst_issued_per_cycle"]),
                "inst_executed_per_cycle": fmt(metrics["inst_executed_per_cycle"]),
                "inst_pipe_alu_pct": fmt(metrics["inst_mix_alu_pct"]),
                "inst_pipe_fma_pct": fmt(metrics["inst_mix_fma_pct"]),
                "inst_pipe_fmaheavy_pct": fmt(metrics["inst_mix_fmaheavy_pct"]),
                "inst_pipe_uniform_pct": fmt(metrics["inst_mix_uniform_pct"]),
            }
            rows.append(out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
