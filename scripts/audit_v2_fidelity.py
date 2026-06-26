#!/usr/bin/env python3
"""Produce the D3 v2 artifact fidelity audit table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


THRESHOLD_GRID = [1, 5, 10, 25, 50, 75, 90, 95, 99]
CSV_COLUMNS = [
    "dataset",
    "artifact_version",
    "artifact_label",
    "artifact_family",
    "fidelity_class",
    "raw_roundtrip_exact",
    "precision_policy",
    "precision_power",
    "target_selectivity",
    "threshold",
    "raw_sum",
    "encoded_sum",
    "sum_abs_error",
    "sum_rel_error",
    "sum_verdict",
    "raw_count",
    "encoded_count",
    "count_abs_error",
    "count_rel_error",
    "observed_selectivity",
    "selectivity_drift_pp",
    "count_verdict",
    "count_mainline_suitable",
    "sum_mainline_suitable",
    "artifact_role",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("datasets/synthetic/dev_buff_v2_20260510"),
        help="Versioned v2 artifact root.",
    )
    parser.add_argument(
        "--buff-tool",
        type=Path,
        default=Path("bin/buff_tool"),
        help="Path to the buff_tool binary.",
    )
    parser.add_argument(
        "--decoded-dir",
        type=Path,
        default=Path("results/buff_encoder_v2/decoded_dev_20260510"),
        help="Directory for full-depth decoded .buff64 payloads.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("results/buff_encoder_v2/fidelity_audit.csv"),
        help="Canonical D3 fidelity audit CSV path.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("research/2026-05-10_New_Encoder_Fidelity_Audit_Report.md"),
        help="Markdown report path.",
    )
    return parser.parse_args()


def run_checked(cmd: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=cwd, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            + " ".join(shlex.quote(part) for part in cmd)
        )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def git_commit(cwd: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()


def write_provenance(path: Path, *, argv: list[str], extra: dict[str, Any]) -> None:
    payload = {
        "command": " ".join(shlex.quote(part) for part in argv),
        "git_commit": git_commit(Path.cwd()),
        "hostname": socket.gethostname(),
        "gpu_name": None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def discover_artifacts(artifact_root: Path) -> list[Path]:
    root_manifest = artifact_root / "manifest.json"
    if root_manifest.is_file():
        payload = read_json(root_manifest)
        entries = payload.get("entries", [])
        discovered = [artifact_root / entry["relative_path"] for entry in entries]
    else:
        discovered = [path.parent for path in artifact_root.rglob("manifest.json") if path.parent != artifact_root]

    artifacts = []
    for dataset_dir in discovered:
        if (dataset_dir / "manifest.json").is_file() and (dataset_dir / "summary.json").is_file():
            artifacts.append(dataset_dir)
    return sorted(set(artifacts))


def ensure_decoded_file(buff_tool: Path, summary: dict[str, Any], decoded_path: Path, *, cwd: Path) -> None:
    if decoded_path.is_file():
        return
    encoded_path = Path(summary["encoded_path"])
    if not encoded_path.is_file():
        raise FileNotFoundError(f"encoded_path does not exist: {encoded_path}")
    decoded_path.parent.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            str(buff_tool),
            "decode",
            "--input",
            str(encoded_path),
            "--output",
            str(decoded_path),
        ],
        cwd=cwd,
    )


def load_memmap(path: Path) -> np.memmap:
    return np.memmap(path, dtype="<f8", mode="r")


def count_greater(values: np.ndarray, threshold: float) -> int:
    return int(np.count_nonzero(values > threshold))


def count_verdict(selectivity_drift_pp: float) -> str:
    drift = abs(selectivity_drift_pp)
    if drift <= 0.1:
        return "acceptable"
    if drift <= 1.0:
        return "caution"
    return "catastrophic"


def sum_verdict(sum_abs_error: float, rows: int, quantization_bound: float) -> str:
    if rows <= 0 or quantization_bound <= 0.0:
        return "not_applicable"
    declared_sum_bound = rows * quantization_bound
    if sum_abs_error <= declared_sum_bound:
        return "acceptable"
    if sum_abs_error <= declared_sum_bound * 10.0:
        return "caution"
    return "catastrophic"


def artifact_role(sum_verdict_value: str, count_verdicts: list[str]) -> tuple[bool, bool, str]:
    sum_mainline = sum_verdict_value == "acceptable"
    count_mainline = all(verdict == "acceptable" for verdict in count_verdicts)
    if sum_verdict_value == "catastrophic" or any(verdict == "catastrophic" for verdict in count_verdicts):
        return count_mainline, sum_mainline, "reject"
    if sum_mainline and count_mainline:
        return count_mainline, sum_mainline, "mainline_candidate"
    return count_mainline, sum_mainline, "side_study_only"


def audit_artifact(dataset_dir: Path, decoded_dir: Path, buff_tool: Path, *, cwd: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(dataset_dir / "manifest.json")
    summary = read_json(dataset_dir / "summary.json")

    dataset = str(manifest["dataset"])
    artifact_label = str(manifest["artifact_label"])
    rows = int(manifest["value_count"])
    raw_path = Path(manifest["source_path"])
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw source_path does not exist: {raw_path}")

    decoded_path = decoded_dir / f"{dataset}_{artifact_label}.decoded.f64le.bin"
    ensure_decoded_file(buff_tool, summary, decoded_path, cwd=cwd)

    raw_values = load_memmap(raw_path)
    encoded_values = load_memmap(decoded_path)
    if raw_values.shape[0] != rows:
        raise ValueError(f"raw row count mismatch for {dataset_dir}: expected {rows}, got {raw_values.shape[0]}")
    if encoded_values.shape[0] != rows:
        raise ValueError(
            f"decoded row count mismatch for {dataset_dir}: expected {rows}, got {encoded_values.shape[0]}"
        )

    raw_sum = float(np.sum(raw_values, dtype=np.float64))
    encoded_sum = float(np.sum(encoded_values, dtype=np.float64))
    sum_abs_error = abs(encoded_sum - raw_sum)
    sum_rel_error = 0.0 if raw_sum == 0.0 else sum_abs_error / abs(raw_sum)
    sum_verdict_value = sum_verdict(sum_abs_error, rows, float(summary["quantization_bound"]))

    quantiles = {
        target: float(np.quantile(raw_values, 1.0 - (target / 100.0), method="linear"))
        for target in THRESHOLD_GRID
    }

    count_verdicts: list[str] = []
    rows_out: list[dict[str, Any]] = []
    for target in THRESHOLD_GRID:
        threshold = quantiles[target]
        raw_count = count_greater(raw_values, threshold)
        encoded_count = count_greater(encoded_values, threshold)
        count_abs_error = abs(encoded_count - raw_count)
        count_rel_error = 0.0 if raw_count == 0 else count_abs_error / raw_count
        observed_selectivity = (encoded_count / rows) * 100.0
        selectivity_drift_pp = observed_selectivity - float(target)
        count_verdict_value = count_verdict(selectivity_drift_pp)
        count_verdicts.append(count_verdict_value)

        rows_out.append(
            {
                "dataset": dataset,
                "artifact_version": manifest["artifact_version"],
                "artifact_label": artifact_label,
                "artifact_family": manifest["artifact_family"],
                "fidelity_class": manifest["fidelity_class"],
                "raw_roundtrip_exact": manifest["raw_roundtrip_exact"],
                "precision_policy": manifest["precision_policy"],
                "precision_power": manifest["precision_power"],
                "target_selectivity": target,
                "threshold": threshold,
                "raw_sum": raw_sum,
                "encoded_sum": encoded_sum,
                "sum_abs_error": sum_abs_error,
                "sum_rel_error": sum_rel_error,
                "sum_verdict": sum_verdict_value,
                "raw_count": raw_count,
                "encoded_count": encoded_count,
                "count_abs_error": count_abs_error,
                "count_rel_error": count_rel_error,
                "observed_selectivity": observed_selectivity,
                "selectivity_drift_pp": selectivity_drift_pp,
                "count_verdict": count_verdict_value,
            }
        )

    count_mainline_suitable, sum_mainline_suitable, overall_role = artifact_role(sum_verdict_value, count_verdicts)
    notes = (
        f"quantization_bound={summary['quantization_bound']}; "
        f"full-depth encoded reference from {decoded_path.name}"
    )
    for row in rows_out:
        row["count_mainline_suitable"] = count_mainline_suitable
        row["sum_mainline_suitable"] = sum_mainline_suitable
        row["artifact_role"] = overall_role
        row["notes"] = notes

    artifact_summary = {
        "dataset": dataset,
        "artifact_label": artifact_label,
        "precision_power": manifest["precision_power"],
        "sum_verdict": sum_verdict_value,
        "sum_mainline_suitable": sum_mainline_suitable,
        "count_mainline_suitable": count_mainline_suitable,
        "artifact_role": overall_role,
        "worst_count_verdict": (
            "catastrophic"
            if "catastrophic" in count_verdicts
            else "caution" if "caution" in count_verdicts else "acceptable"
        ),
        "sum_abs_error": sum_abs_error,
        "sum_rel_error": sum_rel_error,
    }
    return rows_out, artifact_summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def bool_word(value: bool) -> str:
    return "yes" if value else "no"


def build_report(
    report_path: Path,
    *,
    artifact_root: Path,
    artifact_summaries: list[dict[str, Any]],
    fidelity_rows: list[dict[str, Any]],
    local_path_note: bool,
) -> None:
    caution_or_cat_count_rows = [
        row
        for row in fidelity_rows
        if row["count_verdict"] in {"caution", "catastrophic"}
    ]
    caution_or_cat_sum_rows = [
        row
        for row in fidelity_rows
        if row["sum_verdict"] in {"caution", "catastrophic"}
    ]

    lines = [
        "# New Encoder Fidelity Audit Report",
        "",
        "Date: 2026-05-10",
        "",
        "Scope: D3 CPU/reference fidelity audit only.",
        "",
        "## 1. Inputs",
        "",
        f"- artifact root: `{artifact_root}`",
        "- threshold grid: `1, 5, 10, 25, 50, 75, 90, 95, 99`",
        "- COUNT verdict thresholds follow `research/2026-05-04_p3p6_Predicate_Drift_Audit.md`:",
        "  - acceptable: `|selectivity_drift_pp| <= 0.1`",
        "  - caution: `0.1 < |selectivity_drift_pp| <= 1.0`",
        "  - catastrophic: `|selectivity_drift_pp| > 1.0`",
        "- SUM verdict is independent from COUNT verdict and is evaluated against the declared aggregate quantization budget `rows * quantization_bound`.",
        "",
        "## 2. Artifact Suitability",
        "",
        "| Dataset | Artifact label | Precision power | SUM verdict | Worst COUNT verdict | SUM mainline suitable | COUNT mainline suitable | Overall role |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]

    for item in artifact_summaries:
        lines.append(
            "| "
            + f"{item['dataset']} | {item['artifact_label']} | {item['precision_power']} | "
            + f"{item['sum_verdict']} | {item['worst_count_verdict']} | "
            + f"{bool_word(item['sum_mainline_suitable'])} | {bool_word(item['count_mainline_suitable'])} | "
            + f"{item['artifact_role']} |"
        )

    lines.extend(["", "## 3. COUNT Caution / Catastrophic Rows", ""])
    if caution_or_cat_count_rows:
        lines.extend(
            [
                "| Dataset | Artifact label | Target selectivity | Threshold | Raw count | Encoded count | Count abs error | Observed selectivity | Selectivity drift (pp) | Verdict |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in caution_or_cat_count_rows:
            lines.append(
                "| "
                + f"{row['dataset']} | {row['artifact_label']} | {row['target_selectivity']} | "
                + f"{row['threshold']:.17g} | {row['raw_count']} | {row['encoded_count']} | "
                + f"{row['count_abs_error']} | {row['observed_selectivity']:.6f} | "
                + f"{row['selectivity_drift_pp']:.6f} | {row['count_verdict']} |"
            )
    else:
        lines.append("No COUNT caution/catastrophic rows were observed.")

    lines.extend(["", "## 4. SUM Caution / Catastrophic Rows", ""])
    if caution_or_cat_sum_rows:
        lines.extend(
            [
                "| Dataset | Artifact label | SUM abs error | SUM rel error | SUM verdict |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        seen: set[tuple[str, str]] = set()
        for row in caution_or_cat_sum_rows:
            key = (str(row["dataset"]), str(row["artifact_label"]))
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                "| "
                + f"{row['dataset']} | {row['artifact_label']} | {row['sum_abs_error']:.17g} | "
                + f"{row['sum_rel_error']:.17g} | {row['sum_verdict']} |"
            )
    else:
        lines.append("No SUM caution/catastrophic rows were observed.")

    lines.extend(["", "## 5. Bottom Line", ""])
    for item in artifact_summaries:
        lines.append(
            "- "
            + f"`{item['dataset']} / {item['artifact_label']}`: "
            + f"SUM mainline={bool_word(item['sum_mainline_suitable'])}, "
            + f"COUNT mainline={bool_word(item['count_mainline_suitable'])}, "
            + f"overall role=`{item['artifact_role']}`."
        )

    if local_path_note:
        lines.extend(
            [
                "",
                "## 6. Environment Note",
                "",
                "- The D1/D3 issue specs use `/work/<user>/datasets/...` as the example artifact root.",
                "- This machine uses the repo-local versioned root shown above because no writable `/work/...` path was available in the current workspace.",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    workspace = Path.cwd()

    artifact_root = args.artifact_root.resolve()
    buff_tool = args.buff_tool.resolve()
    decoded_dir = args.decoded_dir.resolve()
    csv_out = args.csv_out.resolve()
    report_path = args.report_path.resolve()

    if not artifact_root.is_dir():
        raise FileNotFoundError(f"artifact root does not exist: {artifact_root}")
    if not buff_tool.is_file():
        raise FileNotFoundError(f"buff_tool binary not found: {buff_tool}")

    artifact_dirs = discover_artifacts(artifact_root)
    if not artifact_dirs:
        raise FileNotFoundError(f"no artifact directories found under {artifact_root}")

    all_rows: list[dict[str, Any]] = []
    artifact_summaries: list[dict[str, Any]] = []
    for dataset_dir in artifact_dirs:
        rows, artifact_summary = audit_artifact(dataset_dir, decoded_dir, buff_tool, cwd=workspace)
        all_rows.extend(rows)
        artifact_summaries.append(artifact_summary)

    all_rows.sort(key=lambda row: (row["dataset"], row["artifact_label"], int(row["target_selectivity"])))
    artifact_summaries.sort(key=lambda row: (row["dataset"], row["artifact_label"]))

    write_csv(csv_out, all_rows)
    write_provenance(
        csv_out.with_suffix(csv_out.suffix + ".provenance.json"),
        argv=sys.argv,
        extra={
            "artifact_root": str(artifact_root),
            "decoded_dir": str(decoded_dir),
            "csv_out": str(csv_out),
            "row_count": len(all_rows),
        },
    )
    build_report(
        report_path,
        artifact_root=artifact_root,
        artifact_summaries=artifact_summaries,
        fidelity_rows=all_rows,
        local_path_note=not str(artifact_root).startswith("/work/"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
