#!/usr/bin/env python3
"""Analyze byteplane_count_gt + cuDF refinement safety and timing results."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def artifact_label(path_text: str) -> str:
    path = Path(path_text)
    return f"{path.parent.name}/{path.name}"


def enrich_row(row: dict[str, str]) -> dict[str, Any]:
    q = int(row["Q"])
    u = int(row["U"])
    count_lower = int(row.get("count_lower", q))
    count_upper = int(row.get("count_upper", q + u))
    cudf_exact = int(row["cuDF_exact_count"])
    refined = int(row["cudf_refined_exact_count"])
    primitive_ms = float(row.get("primitive_ms") or 0.0)
    primitive_wall_ms = float(row.get("primitive_wall_ms") or primitive_ms)
    cudf_full_ms = float(row.get("cudf_full_ms") or 0.0)
    compact_u_refine_ms = float(row.get("compact_u_refine_ms") or 0.0)
    cudf_refine_ms = float(row.get("cudf_refine_ms") or 0.0)
    refine_ms = compact_u_refine_ms if compact_u_refine_ms > 0.0 else cudf_refine_ms
    total_refined_ms = float(row.get("total_refined_ms") or (primitive_ms + refine_ms))
    total_refined_wall_ms = float(
        row.get("total_refined_wall_ms") or (primitive_wall_ms + refine_ms)
    )

    exact_matches = refined == cudf_exact
    raw_exact_in_interval = count_lower <= cudf_exact <= count_upper
    if exact_matches:
        safety_status = "exact_refinement_safe"
    elif not raw_exact_in_interval:
        safety_status = "raw_outside_encoded_interval"
    else:
        safety_status = "q_d_raw_drift"

    refined_abs_error = refined - cudf_exact
    out: dict[str, Any] = dict(row)
    out.update(
        {
            "artifact_label": artifact_label(row["artifact_root"]),
            "raw_exact_in_interval": raw_exact_in_interval,
            "refined_abs_error": refined_abs_error,
            "refined_rel_error": refined_abs_error / cudf_exact if cudf_exact else 0.0,
            "interval_width": count_upper - count_lower,
            "speedup_vs_cudf_full": (
                cudf_full_ms / total_refined_ms if total_refined_ms > 0.0 else ""
            ),
            "speedup_wall_vs_cudf_full": (
                cudf_full_ms / total_refined_wall_ms if total_refined_wall_ms > 0.0 else ""
            ),
            "total_refined_wall_ms": total_refined_wall_ms,
            "safety_status": safety_status,
        }
    )
    return out


def write_enriched_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["artifact_label"], row["dataset"], row.get("selectivity", ""))].append(row)

    best: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        safe = [r for r in group if r["safety_status"] == "exact_refinement_safe"]
        if not safe:
            continue
        safe.sort(key=lambda r: float(r.get("total_refined_ms") or 0.0))
        chosen = safe[0]
        best.append(
            {
                "artifact_label": key[0],
                "dataset": key[1],
                "selectivity": key[2],
                "best_safe_k": chosen["k"],
                "best_total_refined_ms": chosen.get("total_refined_ms", ""),
                "best_total_refined_wall_ms": chosen.get("total_refined_wall_ms", ""),
                "cudf_full_ms": chosen.get("cudf_full_ms", ""),
                "speedup_vs_cudf_full": chosen.get("speedup_vs_cudf_full", ""),
                "speedup_wall_vs_cudf_full": chosen.get("speedup_wall_vs_cudf_full", ""),
                "U_fraction": chosen.get("U_fraction", ""),
                "interval_width": chosen.get("interval_width", ""),
            }
        )
    return best


def write_markdown(rows: list[dict[str, Any]], best_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(row["safety_status"] for row in rows)
    group_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        group_counts[(row["artifact_label"], row["dataset"])][row["safety_status"]] += 1

    top_bad = sorted(
        [r for r in rows if r["safety_status"] != "exact_refinement_safe"],
        key=lambda r: abs(int(r["refined_abs_error"])),
        reverse=True,
    )[:12]

    lines = [
        "# byteplane_count_gt cuDF Pipeline Safety Analysis",
        "",
        "## Summary",
        "",
        f"- Rows analyzed: {len(rows)}",
        f"- exact_refinement_safe: {status_counts['exact_refinement_safe']}",
        f"- raw_outside_encoded_interval: {status_counts['raw_outside_encoded_interval']}",
        f"- q_d_raw_drift: {status_counts['q_d_raw_drift']}",
        "",
        "Interpretation:",
        "",
        "- `exact_refinement_safe`: `Q + cuDF(raw[U] > threshold)` equals full cuDF raw exact count.",
        "- `raw_outside_encoded_interval`: raw exact count is outside `[Q, Q + U]`; U-only refinement cannot recover SQL exactness.",
        "- `q_d_raw_drift`: raw exact count is inside the interval, but rows classified as Q/D under bounded encoding differ from raw FP64 truth.",
        "",
        "## Status by Artifact",
        "",
        "| artifact | dataset | safe | raw outside interval | Q/D raw drift |",
        "|---|---:|---:|---:|---:|",
    ]
    for (artifact, dataset), counter in sorted(group_counts.items()):
        lines.append(
            f"| {artifact} | {dataset} | {counter['exact_refinement_safe']} | "
            f"{counter['raw_outside_encoded_interval']} | {counter['q_d_raw_drift']} |"
        )

    lines.extend(
        [
            "",
            "## Best Safe K per Artifact/Selectivity",
            "",
            "| artifact | dataset | selectivity | best_safe_k | total_refined_ms | total_refined_wall_ms | cudf_full_ms | speedup | wall_speedup | U_fraction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best_rows:
        lines.append(
            f"| {row['artifact_label']} | {row['dataset']} | {row['selectivity']} | "
            f"{row['best_safe_k']} | {row['best_total_refined_ms']} | "
            f"{row['best_total_refined_wall_ms']} | {row['cudf_full_ms']} | "
            f"{row['speedup_vs_cudf_full']} | {row['speedup_wall_vs_cudf_full']} | "
            f"{row['U_fraction']} |"
        )

    lines.extend(
        [
            "",
            "## Largest Unsafe Drifts",
            "",
            "| artifact | selectivity | k | U | raw exact | refined exact | drift | safety_status |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in top_bad:
        lines.append(
            f"| {row['artifact_label']} | {row.get('selectivity', '')} | {row['k']} | {row['U']} | "
            f"{row['cuDF_exact_count']} | {row['cudf_refined_exact_count']} | "
            f"{row['refined_abs_error']} | {row['safety_status']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--best-safe-csv", required=True)
    parser.add_argument("--report-md", required=True)
    args = parser.parse_args()

    rows = [enrich_row(row) for row in csv.DictReader(Path(args.metrics_csv).open(newline="", encoding="utf-8"))]
    best_rows = best_safe_rows(rows)
    write_enriched_csv(rows, Path(args.out_csv))
    write_enriched_csv(best_rows, Path(args.best_safe_csv))
    write_markdown(rows, best_rows, Path(args.report_md))
    print(
        {
            "rows": len(rows),
            "best_safe_rows": len(best_rows),
            "report_md": args.report_md,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
