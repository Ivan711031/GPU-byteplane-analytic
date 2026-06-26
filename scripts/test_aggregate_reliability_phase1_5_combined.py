#!/usr/bin/env python3
"""Unit tests for aggregate_reliability_phase1_5_combined.py.

Tests every acceptance criterion:
  - single-PASS GO
  - multi-PASS tie-break GO
  - all-MARGINAL CONDITIONAL GO
  - all-FAIL NO-GO
  - mixed PASS/FAIL/MARGINAL
  - strategies_summary.csv format
  - combined report sections
  - handle undefined ratios in tie-break

Usage:
  python3 scripts/test_aggregate_reliability_phase1_5_combined.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_reliability_phase1_5_combined import (
    load_strategy_data,
    pick_winner,
    determine_combined_verdict,
    parse_normalized_ratio,
    ALL_STRATEGY_IDS,
    DATASETS,
    FAULT_RATES,
    TIE_BREAK_RATES,
)

PASS = "PASS"
MARGINAL = "MARGINAL"
FAIL = "FAIL"
GO = "GO"
CONDITIONAL_GO = "CONDITIONAL GO"
NO_GO = "NO-GO"

SUMMARY_FIELDS = [
    "dataset", "fault_rate",
    "msb_group_median_raw", "lsb_group_median_raw", "primary_ratio_raw",
    "normalized_msb_group_median", "normalized_lsb_group_median",
    "normalized_ratio",
]
for p in range(8):
    SUMMARY_FIELDS.append(f"plane_{p}_occupancy")
SUMMARY_FIELDS.extend(["undefined_ratio_flag", "noise_denominator_flag"])

PARAMS_JSON = {
    "metadata": {
        "strategy_ids": ["native_scale1", "per_dataset",
                         "hybrid_scale100_plus_shifted"],
        "source_raw_root": "/work/u4063895/datasets/reliability_layer1",
    },
    "strategies": {
        "native_scale1": {
            "description": "Scale = 1",
            "per_dataset": {
                ds: {"scale": 1, "shift_k": 0}
                for ds in DATASETS
            },
        },
        "per_dataset": {
            "description": "Per-dataset scale",
            "per_dataset": {
                ds: {"scale": 8233095970213, "shift_k": 0}
                for ds in DATASETS
            },
        },
        "hybrid_scale100_plus_shifted": {
            "description": "Scale 100 + shift",
            "per_dataset": {
                ds: {"scale": 100, "shift_k": 50}
                for ds in DATASETS
            },
        },
    },
}


def make_summary_rows(
    dataset: str,
    norm_ratios: dict[str, str],
) -> list[dict[str, str]]:
    """Build summary CSV rows for one dataset with given normalized ratios."""
    rows = []
    for rate in FAULT_RATES:
        nr = norm_ratios.get(rate, "1.00")
        rr = nr  # simplified: same as normalized for test
        nmsb = 300 if nr != "UNDEFINED" else 0
        nlsb = 200 if nr != "UNDEFINED" else 0
        if nr != "UNDEFINED":
            try:
                ratio_val = float(nr)
                nmsb = int(ratio_val * 200)
                nlsb = 200
            except ValueError:
                nmsb = 300
                nlsb = 200

        undef_flag = "true" if nr == "UNDEFINED" else ""
        noise_flag = ""

        rows.append({
            "dataset": dataset,
            "fault_rate": rate,
            "msb_group_median_raw": str(nmsb * 1000000),
            "lsb_group_median_raw": str(nlsb * 1000000),
            "primary_ratio_raw": nr,
            "normalized_msb_group_median": str(nmsb),
            "normalized_lsb_group_median": str(nlsb),
            "normalized_ratio": nr,
            "plane_0_occupancy": "1.0",
            "plane_1_occupancy": "1.0",
            "plane_2_occupancy": "1.0",
            "plane_3_occupancy": "0.99999999",
            "plane_4_occupancy": "0.99999990",
            "plane_5_occupancy": "0.5",
            "plane_6_occupancy": "0.1",
            "plane_7_occupancy": "0.01",
            "undefined_ratio_flag": undef_flag,
            "noise_denominator_flag": noise_flag,
        })
    return rows


def make_review_report(
    strategy_id: str,
    verdict: str,
    oracle_status: dict[str, str] | None = None,
) -> str:
    """Build a per-strategy review report with given verdict."""
    if oracle_status is None:
        oracle_status = {"sensor": "PASS", "uniform": "PASS"}

    lines = [
        "=" * 72,
        f"Phase 1.5 Per-Strategy Review Report",
        f"Strategy: {strategy_id}",
        "=" * 72,
        "",
        "--- Oracle & Clean Gate Summary ---",
    ]
    for ds in DATASETS:
        mismatch = 0
        lines.append(f"  {ds}: 1200 canonical, {mismatch} mismatches")
        if ds in oracle_status:
            lines.append(f"    Oracle gate: {oracle_status[ds]}")
    lines.append("")
    lines.append("--- Verdict ---")
    lines.append(f"  Verdict: {verdict}")
    return "\n".join(lines)


def write_strategy_dir(
    base_dir: Path,
    strategy_id: str,
    verdict: str,
    norm_ratios_per_ds: dict[str, dict[str, str]],
    oracle_status: dict[str, str] | None = None,
) -> Path:
    """Write a fake per-strategy combined output directory."""
    sd = base_dir / strategy_id
    sd.mkdir(parents=True, exist_ok=True)

    # run_meta.txt
    meta_lines = [
        f"rows=4800",
        f"oracle_match=4800/4800",
        f"failure_rows=0",
        f"strategy_id={strategy_id}",
    ]
    (sd / "run_meta.txt").write_text("\n".join(meta_lines) + "\n")

    # msb_lsb_summary.csv
    all_rows = []
    for ds, ratios in norm_ratios_per_ds.items():
        all_rows.extend(make_summary_rows(ds, ratios))
    with (sd / "msb_lsb_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    # phase1_5_strategy_review_report.txt
    report = make_review_report(strategy_id, verdict, oracle_status)
    (sd / "phase1_5_strategy_review_report.txt").write_text(report)
    return sd


def run_combined(
    strategy_dirs: list[Path],
    output_dir: Path,
    params: dict | None = None,
) -> None:
    """Run the combined script via subprocess."""
    import subprocess
    output_dir.mkdir(parents=True, exist_ok=True)

    if params is None:
        params = PARAMS_JSON

    params_path = output_dir / "params.json"
    json.dump(params, params_path.open("w"))

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent
            / "aggregate_reliability_phase1_5_combined.py"),
        "--strategy-dirs",
    ] + [str(d) for d in strategy_dirs] + [
        "--params-json", str(params_path),
        "--output-dir", str(output_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}", file=sys.stderr)
        print(f"STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(
            f"Combined script exited code {result.returncode}")
    print(result.stdout)


def read_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def extract_combined_verdict(report_text: str) -> str:
    for line in report_text.splitlines():
        if line.strip() in (GO, CONDITIONAL_GO, NO_GO):
            return line.strip()
    raise ValueError("combined verdict not found")


def extract_winner(report_text: str) -> str | None:
    for line in report_text.splitlines():
        if "Named winner:" in line:
            w = line.split("Named winner:")[-1].strip()
            return None if w == "none" else w
    return None


# ── Test helpers ───────────────────────────────────────────────

def _test_single_pass_go():
    """GO with single PASS strategy."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        base = td_p / "strategies"

        # native_scale1: PASS (all ratios >= 2.00)
        nrm = {
            "sensor": {r: "2.50" for r in FAULT_RATES},
            "uniform": {r: "3.00" for r in FAULT_RATES},
            "heavy_tailed": {r: "1.50" for r in FAULT_RATES},
            "zipfian": {r: "1.20" for r in FAULT_RATES},
        }
        sd1 = write_strategy_dir(base, "native_scale1", PASS, nrm)

        # per_dataset: FAIL (weak ratio)
        nrm2 = {
            "sensor": {r: "1.05" for r in FAULT_RATES},
            "uniform": {r: "1.03" for r in FAULT_RATES},
            "heavy_tailed": {r: "1.01" for r in FAULT_RATES},
            "zipfian": {r: "1.01" for r in FAULT_RATES},
        }
        sd2 = write_strategy_dir(base, "per_dataset", FAIL, nrm2)

        # hybrid: FAIL
        nrm3 = {
            "sensor": {r: "1.05" for r in FAULT_RATES},
            "uniform": {r: "1.03" for r in FAULT_RATES},
            "heavy_tailed": {r: "1.01" for r in FAULT_RATES},
            "zipfian": {r: "1.01" for r in FAULT_RATES},
        }
        sd3 = write_strategy_dir(base, "hybrid_scale100_plus_shifted",
                                 FAIL, nrm3)

        out = td_p / "output"
        run_combined([sd1, sd2, sd3], out)

        report = (out / "phase1_5_combined_review_report.txt").read_text()
        v = extract_combined_verdict(report)
        assert v == GO, f"Expected GO, got {v}"
        w = extract_winner(report)
        assert w == "native_scale1", f"Expected native_scale1 winner, got {w}"

        summary = read_summary_csv(out / "strategies_summary.csv")
        assert len(summary) > 0
        cols = list(summary[0].keys())
        for req in [
            "strategy_id", "dataset", "fault_rate",
            "msb_group_median_raw", "lsb_group_median_raw",
            "normalized_ratio", "per_strategy_verdict",
        ]:
            assert req in cols, f"Missing column: {req}"

        # Check per_strategy_verdict values
        native_rows = [r for r in summary
                       if r["strategy_id"] == "native_scale1"]
        assert all(r["per_strategy_verdict"] == PASS for r in native_rows)
        per_rows = [r for r in summary
                    if r["strategy_id"] == "per_dataset"]
        assert all(r["per_strategy_verdict"] == FAIL for r in per_rows)

    print("  ✓ single-PASS GO")


def _test_multi_pass_tiebreak():
    """GO with multi-PASS tie-break."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        base = td_p / "strategies"

        # Both PASS, but different min ratios
        # native_scale1: ratios range from 1.60 to 3.00 -> min = 1.60
        nrm1 = {
            "sensor": {
                "1e-8": "3.00", "1e-7": "1.60",
                "1e-6": "1.80", "1e-5": "2.00", "1e-4": "2.50",
            },
            "uniform": {
                "1e-8": "3.00", "1e-7": "1.70",
                "1e-6": "1.90", "1e-5": "2.00", "1e-4": "2.50",
            },
            "heavy_tailed": {r: "1.50" for r in FAULT_RATES},
            "zipfian": {r: "1.20" for r in FAULT_RATES},
        }
        sd1 = write_strategy_dir(base, "native_scale1", PASS, nrm1)

        # per_dataset: higher min -> should win
        nrm2 = {
            "sensor": {
                "1e-8": "3.00", "1e-7": "2.10",
                "1e-6": "2.30", "1e-5": "2.00", "1e-4": "2.50",
            },
            "uniform": {
                "1e-8": "3.00", "1e-7": "2.20",
                "1e-6": "2.40", "1e-5": "2.00", "1e-4": "2.50",
            },
            "heavy_tailed": {r: "1.50" for r in FAULT_RATES},
            "zipfian": {r: "1.20" for r in FAULT_RATES},
        }
        sd2 = write_strategy_dir(base, "per_dataset", PASS, nrm2)

        # hybrid: also PASS, min = 1.50
        nrm3 = {
            "sensor": {
                "1e-8": "2.50", "1e-7": "1.50",
                "1e-6": "1.70", "1e-5": "2.00", "1e-4": "2.50",
            },
            "uniform": {
                "1e-8": "2.50", "1e-7": "1.55",
                "1e-6": "1.75", "1e-5": "2.00", "1e-4": "2.50",
            },
            "heavy_tailed": {r: "1.50" for r in FAULT_RATES},
            "zipfian": {r: "1.20" for r in FAULT_RATES},
        }
        sd3 = write_strategy_dir(base, "hybrid_scale100_plus_shifted",
                                 PASS, nrm3)

        out = td_p / "output"
        run_combined([sd1, sd2, sd3], out)

        report = (out / "phase1_5_combined_review_report.txt").read_text()
        v = extract_combined_verdict(report)
        assert v == GO, f"Expected GO, got {v}"
        w = extract_winner(report)
        # per_dataset has highest min (sensor 1e-7=2.10, uniform 1e-7=2.20)
        # min of {2.10, 2.30, 2.20, 2.40} = 2.10
        # native_scale1 min = 1.60, hybrid min = 1.50
        assert w == "per_dataset", (
            f"Expected per_dataset winner, got {w}")
    print("  ✓ multi-PASS tie-break GO")


def _test_all_marginal_conditional_go():
    """CONDITIONAL GO when all are MARGINAL."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        base = td_p / "strategies"

        nrm = {
            "sensor": {
                "1e-8": "1.40", "1e-7": "1.30",
                "1e-6": "1.35", "1e-5": "1.40", "1e-4": "1.45",
            },
            "uniform": {
                "1e-8": "1.40", "1e-7": "1.30",
                "1e-6": "1.35", "1e-5": "1.40", "1e-4": "1.45",
            },
            "heavy_tailed": {r: "1.10" for r in FAULT_RATES},
            "zipfian": {r: "1.05" for r in FAULT_RATES},
        }
        sd1 = write_strategy_dir(base, "native_scale1", MARGINAL, nrm)

        # Higher marginal
        nrm2 = {
            "sensor": {
                "1e-8": "1.45", "1e-7": "1.40",
                "1e-6": "1.42", "1e-5": "1.44", "1e-4": "1.48",
            },
            "uniform": {
                "1e-8": "1.45", "1e-7": "1.40",
                "1e-6": "1.42", "1e-5": "1.44", "1e-4": "1.48",
            },
            "heavy_tailed": {r: "1.10" for r in FAULT_RATES},
            "zipfian": {r: "1.05" for r in FAULT_RATES},
        }
        sd2 = write_strategy_dir(base, "per_dataset", MARGINAL, nrm2)

        # Lower marginal
        nrm3 = {
            "sensor": {
                "1e-8": "1.25", "1e-7": "1.20",
                "1e-6": "1.22", "1e-5": "1.28", "1e-4": "1.30",
            },
            "uniform": {
                "1e-8": "1.25", "1e-7": "1.22",
                "1e-6": "1.25", "1e-5": "1.28", "1e-4": "1.30",
            },
            "heavy_tailed": {r: "1.10" for r in FAULT_RATES},
            "zipfian": {r: "1.05" for r in FAULT_RATES},
        }
        sd3 = write_strategy_dir(base, "hybrid_scale100_plus_shifted",
                                 MARGINAL, nrm3)

        out = td_p / "output"
        run_combined([sd1, sd2, sd3], out)

        report = (out / "phase1_5_combined_review_report.txt").read_text()
        v = extract_combined_verdict(report)
        assert v == CONDITIONAL_GO, f"Expected CONDITIONAL GO, got {v}"
        w = extract_winner(report)
        # per_dataset has highest min = 1.40
        # native_scale1 min = 1.30
        # hybrid min = 1.20
        assert w == "per_dataset", (
            f"Expected per_dataset winner, got {w}")

        # Verify marginal ratios listed
        assert "MARGINAL" in report
        assert "1.40" in report

    print("  ✓ all-MARGINAL CONDITIONAL GO")


def _test_all_fail_no_go():
    """NO-GO when all strategies FAIL."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        base = td_p / "strategies"

        nrm = {
            "sensor": {r: "1.02" for r in FAULT_RATES},
            "uniform": {r: "1.01" for r in FAULT_RATES},
            "heavy_tailed": {r: "1.00" for r in FAULT_RATES},
            "zipfian": {r: "1.00" for r in FAULT_RATES},
        }
        sd1 = write_strategy_dir(base, "native_scale1", FAIL, nrm)
        sd2 = write_strategy_dir(base, "per_dataset", FAIL, nrm)
        sd3 = write_strategy_dir(base, "hybrid_scale100_plus_shifted",
                                 FAIL, nrm)

        out = td_p / "output"
        run_combined([sd1, sd2, sd3], out)

        report = (out / "phase1_5_combined_review_report.txt").read_text()
        v = extract_combined_verdict(report)
        assert v == NO_GO, f"Expected NO-GO, got {v}"
        w = extract_winner(report)
        assert w is None, f"Expected no winner, got {w}"
        assert "mechanism redesign" in report, \
            "Expected 'mechanism redesign' recommendation"
        assert "not another rescaling candidate" in report, \
            "Expected 'not another rescaling candidate'"

    print("  ✓ all-FAIL NO-GO")


def _test_mixed_pass_fail():
    """Mixed: one PASS, one FAIL, one MARGINAL → GO with PASS winner."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        base = td_p / "strategies"

        nrm_pass = {
            "sensor": {r: "2.00" for r in FAULT_RATES},
            "uniform": {r: "2.50" for r in FAULT_RATES},
            "heavy_tailed": {r: "1.50" for r in FAULT_RATES},
            "zipfian": {r: "1.20" for r in FAULT_RATES},
        }
        sd1 = write_strategy_dir(base, "native_scale1", PASS, nrm_pass)

        nrm_fail = {
            "sensor": {r: "1.05" for r in FAULT_RATES},
            "uniform": {r: "1.03" for r in FAULT_RATES},
            "heavy_tailed": {r: "1.01" for r in FAULT_RATES},
            "zipfian": {r: "1.01" for r in FAULT_RATES},
        }
        sd2 = write_strategy_dir(base, "per_dataset", FAIL, nrm_fail)

        nrm_marg = {
            "sensor": {r: "1.30" for r in FAULT_RATES},
            "uniform": {r: "1.30" for r in FAULT_RATES},
            "heavy_tailed": {r: "1.10" for r in FAULT_RATES},
            "zipfian": {r: "1.05" for r in FAULT_RATES},
        }
        sd3 = write_strategy_dir(base, "hybrid_scale100_plus_shifted",
                                 MARGINAL, nrm_marg)

        out = td_p / "output"
        run_combined([sd1, sd2, sd3], out)

        report = (out / "phase1_5_combined_review_report.txt").read_text()
        v = extract_combined_verdict(report)
        assert v == GO, f"Expected GO, got {v}"
        w = extract_winner(report)
        assert w == "native_scale1", f"Expected native_scale1, got {w}"

    print("  ✓ mixed PASS/FAIL/MARGINAL")


def _test_tiebreak_undefined_ratios():
    """Tie-break handles UNDEFINED ratios gracefully."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        base = td_p / "strategies"

        # native_scale1: some UNDEFINED at 1e-7 for sensor, 1e-6 for uniform
        nrm1 = {
            "sensor": {
                "1e-8": "UNDEFINED", "1e-7": "UNDEFINED",
                "1e-6": "1.90", "1e-5": "2.00", "1e-4": "2.50",
            },
            "uniform": {
                "1e-8": "UNDEFINED", "1e-7": "1.70",
                "1e-6": "UNDEFINED", "1e-5": "2.00", "1e-4": "2.50",
            },
            "heavy_tailed": {r: "1.50" for r in FAULT_RATES},
            "zipfian": {r: "1.20" for r in FAULT_RATES},
        }
        sd1 = write_strategy_dir(base, "native_scale1", PASS, nrm1)
        # Only defined tie-break ratios: sensor 1e-6=1.90, uniform 1e-7=1.70
        # min = 1.70

        # per_dataset: all defined, min = 1.80 -> should win
        nrm2 = {
            "sensor": {
                "1e-8": "3.00", "1e-7": "1.80",
                "1e-6": "2.10", "1e-5": "2.00", "1e-4": "2.50",
            },
            "uniform": {
                "1e-8": "3.00", "1e-7": "2.00",
                "1e-6": "2.30", "1e-5": "2.00", "1e-4": "2.50",
            },
            "heavy_tailed": {r: "1.50" for r in FAULT_RATES},
            "zipfian": {r: "1.20" for r in FAULT_RATES},
        }
        sd2 = write_strategy_dir(base, "per_dataset", PASS, nrm2)

        # hybrid: all defined, min = 1.50
        nrm3 = {
            "sensor": {
                "1e-8": "2.00", "1e-7": "1.50",
                "1e-6": "1.70", "1e-5": "2.00", "1e-4": "2.50",
            },
            "uniform": {
                "1e-8": "2.00", "1e-7": "1.55",
                "1e-6": "1.75", "1e-5": "2.00", "1e-4": "2.50",
            },
            "heavy_tailed": {r: "1.50" for r in FAULT_RATES},
            "zipfian": {r: "1.20" for r in FAULT_RATES},
        }
        sd3 = write_strategy_dir(base, "hybrid_scale100_plus_shifted",
                                 PASS, nrm3)

        out = td_p / "output"
        run_combined([sd1, sd2, sd3], out)

        report = (out / "phase1_5_combined_review_report.txt").read_text()
        v = extract_combined_verdict(report)
        assert v == GO, f"Expected GO, got {v}"
        w = extract_winner(report)
        assert w == "per_dataset", (
            f"Expected per_dataset (higher min), got {w}")

        summary = read_summary_csv(out / "strategies_summary.csv")
        undef = [r for r in summary
                 if r["undefined_ratio_flag"] == "true"]
        assert len(undef) >= 1, (
            "Expected some undefined ratio rows in summary")

    print("  ✓ tie-break with undefined ratios")


def _test_summary_csv_format():
    """Verify strategies_summary.csv has correct columns."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        base = td_p / "strategies"

        nrm = {
            "sensor": {r: "2.00" for r in FAULT_RATES},
            "uniform": {r: "2.50" for r in FAULT_RATES},
            "heavy_tailed": {r: "1.50" for r in FAULT_RATES},
            "zipfian": {r: "1.20" for r in FAULT_RATES},
        }
        sd1 = write_strategy_dir(base, "native_scale1", PASS, nrm)
        sd2 = write_strategy_dir(base, "per_dataset", FAIL, nrm)
        sd3 = write_strategy_dir(base, "hybrid_scale100_plus_shifted",
                                 MARGINAL, nrm)

        out = td_p / "output"
        run_combined([sd1, sd2, sd3], out)

        summary = read_summary_csv(out / "strategies_summary.csv")
        assert len(summary) > 0
        cols = list(summary[0].keys())
        required = [
            "strategy_id", "dataset", "fault_rate",
            "msb_group_median_raw", "lsb_group_median_raw",
            "primary_ratio_raw",
            "normalized_msb_group_median",
            "normalized_lsb_group_median",
            "normalized_ratio",
            "undefined_ratio_flag", "noise_denominator_flag",
            "per_strategy_verdict",
        ]
        for c in required:
            assert c in cols, f"Missing required column: {c}"

    print("  ✓ strategies_summary.csv format")


def _test_load_strategy_data():
    """Test load_strategy_data directly."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        nrm = {
            "sensor": {r: "2.00" for r in FAULT_RATES},
            "uniform": {r: "2.50" for r in FAULT_RATES},
            "heavy_tailed": {r: "1.50" for r in FAULT_RATES},
            "zipfian": {r: "1.20" for r in FAULT_RATES},
        }
        sd = write_strategy_dir(td_p, "native_scale1", PASS, nrm)
        data = load_strategy_data(sd)

        assert data["strategy_id"] == "native_scale1"
        assert data["verdict"] == PASS
        assert len(data["summary_rows"]) == len(DATASETS) * len(FAULT_RATES)
        assert data["norm_ratios"][("sensor", "1e-8")] == "2.00"
        assert data["oracle_gates"].get("sensor") == "PASS"
        assert data["oracle_gates"].get("uniform") == "PASS"

    print("  ✓ load_strategy_data")


def _test_pick_winner():
    """Test pick_winner function directly."""
    strategies = {
        "s1": {
            "verdict": PASS,
            "norm_ratios": {
                ("sensor", "1e-7"): "1.60",
                ("sensor", "1e-6"): "1.80",
                ("sensor", "1e-5"): "2.00",
                ("uniform", "1e-7"): "1.70",
                ("uniform", "1e-6"): "1.90",
                ("uniform", "1e-5"): "2.00",
            },
        },
        "s2": {
            "verdict": PASS,
            "norm_ratios": {
                ("sensor", "1e-7"): "2.10",
                ("sensor", "1e-6"): "2.30",
                ("sensor", "1e-5"): "2.00",
                ("uniform", "1e-7"): "2.20",
                ("uniform", "1e-6"): "2.40",
                ("uniform", "1e-5"): "2.00",
            },
        },
    }
    winner = pick_winner(strategies)
    assert winner == "s2", f"Expected s2 (higher min), got {winner}"

    # s2 min = min(2.10, 2.30, 2.00, 2.20, 2.40, 2.00) = 2.00
    # s1 min = min(1.60, 1.80, 2.00, 1.70, 1.90, 2.00) = 1.60

    print("  ✓ pick_winner")


def _test_pick_winner_undefined():
    """pick_winner with some UNDEFINED ratios."""
    strategies = {
        "s1": {
            "verdict": PASS,
            "norm_ratios": {
                ("sensor", "1e-7"): "UNDEFINED",
                ("sensor", "1e-6"): "UNDEFINED",
                ("sensor", "1e-5"): "2.00",
                ("uniform", "1e-7"): "1.70",
                ("uniform", "1e-6"): "UNDEFINED",
                ("uniform", "1e-5"): "2.00",
            },
        },
        "s2": {
            "verdict": PASS,
            "norm_ratios": {
                ("sensor", "1e-7"): "1.80",
                ("sensor", "1e-6"): "2.10",
                ("sensor", "1e-5"): "2.00",
                ("uniform", "1e-7"): "2.00",
                ("uniform", "1e-6"): "2.30",
                ("uniform", "1e-5"): "2.00",
            },
        },
    }
    winner = pick_winner(strategies)
    assert winner == "s2", f"Expected s2, got {winner}"

    # s1 valid ratios: sensor 1e-5=2.00, uniform 1e-7=1.70, uniform 1e-5=2.00
    # s1 min = 1.70
    # s2 min = 1.80

    print("  ✓ pick_winner with undefined")


def _test_determine_combined_verdict():
    """Test determine_combined_verdict directly."""
    make_data = lambda verdict, norm_ratios: {
        "verdict": verdict,
        "norm_ratios": norm_ratios,
    }

    # Single PASS → GO
    strategies = {
        "s1": make_data(PASS, {
            ("sensor", r): "2.00" for r in TIE_BREAK_RATES +
            ["1e-8", "1e-4"]
        } | {
            ("uniform", r): "2.50" for r in TIE_BREAK_RATES +
            ["1e-8", "1e-4"]
        }),
        "s2": make_data(FAIL, {}),
        "s3": make_data(FAIL, {}),
    }
    v, w, _ = determine_combined_verdict(strategies)
    assert v == GO
    assert w == "s1"

    # All MARGINAL → CONDITIONAL GO
    strategies = {
        "s1": make_data(MARGINAL, {
            ("sensor", r): "1.30" for r in TIE_BREAK_RATES
        } | {
            ("uniform", r): "1.30" for r in TIE_BREAK_RATES
        }),
        "s2": make_data(MARGINAL, {
            ("sensor", r): "1.40" for r in TIE_BREAK_RATES
        } | {
            ("uniform", r): "1.40" for r in TIE_BREAK_RATES
        }),
        "s3": make_data(MARGINAL, {
            ("sensor", r): "1.20" for r in TIE_BREAK_RATES
        } | {
            ("uniform", r): "1.20" for r in TIE_BREAK_RATES
        }),
    }
    v, w, details = determine_combined_verdict(strategies)
    assert v == CONDITIONAL_GO
    assert w == "s2"
    assert details is not None
    assert len(details) > 0

    # All FAIL → NO-GO
    strategies = {
        "s1": make_data(FAIL, {}),
        "s2": make_data(FAIL, {}),
        "s3": make_data(FAIL, {}),
    }
    v, w, _ = determine_combined_verdict(strategies)
    assert v == NO_GO
    assert w is None

    print("  ✓ determine_combined_verdict")


def _test_parse_normalized_ratio():
    assert parse_normalized_ratio("1.50") == 1.5
    assert parse_normalized_ratio("3.00") == 3.0
    assert parse_normalized_ratio("UNDEFINED") is None
    assert parse_normalized_ratio("") is None
    assert parse_normalized_ratio("N/A") is None
    print("  ✓ parse_normalized_ratio")


def _test_output_sections():
    """Verify all required sections in the combined report."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        base = td_p / "strategies"

        nrm = {
            "sensor": {r: "2.00" for r in FAULT_RATES},
            "uniform": {r: "2.50" for r in FAULT_RATES},
            "heavy_tailed": {r: "1.50" for r in FAULT_RATES},
            "zipfian": {r: "1.20" for r in FAULT_RATES},
        }
        sd1 = write_strategy_dir(base, "native_scale1", PASS, nrm)
        sd2 = write_strategy_dir(base, "per_dataset", FAIL, nrm)
        sd3 = write_strategy_dir(base, "hybrid_scale100_plus_shifted",
                                 MARGINAL, nrm)

        out = td_p / "output"
        run_combined([sd1, sd2, sd3], out)

        report = (out / "phase1_5_combined_review_report.txt").read_text()

        sections = [
            "Phase 1.5 Combined Comparative Review",
            "Per-Strategy Verdict Table",
            "Strategy Applicability Table",
            "Side-by-side Normalized MSB-vs-LSB Ratio (sensor + uniform)",
            "Side-by-side Normalized MSB-vs-LSB Ratio (heavy_tailed + zipfian)",
            "Plane Occupancy Summary",
            "Winner Determination",
            "Phase 1.5 Verdict",
            "Recommendations",
        ]
        for sec in sections:
            assert sec in report, f"Missing section: {sec}"

        expected_outputs = [
            "strategies_summary.csv",
            "phase1_5_combined_review_report.txt",
        ]
        for name in expected_outputs:
            assert (out / name).exists(), f"Missing output: {name}"

    print("  ✓ output sections present")


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    tests = [
        ("load_strategy_data", _test_load_strategy_data),
        ("parse_normalized_ratio", _test_parse_normalized_ratio),
        ("pick_winner", _test_pick_winner),
        ("pick_winner undefined", _test_pick_winner_undefined),
        ("determine_combined_verdict", _test_determine_combined_verdict),
        ("single-PASS GO", _test_single_pass_go),
        ("multi-PASS tie-break GO", _test_multi_pass_tiebreak),
        ("all-MARGINAL CONDITIONAL GO",
         _test_all_marginal_conditional_go),
        ("all-FAIL NO-GO", _test_all_fail_no_go),
        ("mixed PASS/FAIL/MARGINAL", _test_mixed_pass_fail),
        ("tie-break undefined ratios", _test_tiebreak_undefined_ratios),
        ("summary CSV format", _test_summary_csv_format),
        ("output sections", _test_output_sections),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  \u2717 {name}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed, "
          f"{len(tests)} total")
    if failed:
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
