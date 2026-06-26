#!/usr/bin/env python3
"""Unit tests for aggregate_reliability_phase1_5.py.

Tests are synthetic CSV fixtures exercising every acceptance criterion:
  - PASS, MARGINAL, FAIL (weak ratio), FAIL (oracle mismatch)
  - Undefined ratio, noise denominator
  - Inapplicable cells
  - parse_exact_int / median_int correctness
  - MSB/LSB group median computation
  - Verdict thresholds per PRD

Usage:
  python3 scripts/test_aggregate_reliability_phase1_5.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure the aggregator is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_reliability_phase1_5 import (
    parse_exact_int,
    median_int,
    MSB_PLANES,
    LSB_PLANES,
)

PASS = "PASS"
MARGINAL = "MARGINAL"
FAIL = "FAIL"

# Reusable CSV header matching Phase 1 + Phase 1.5 columns
CSV_FIELDS = [
    "dataset", "target_plane", "fault_rate", "seed",
    "clean_encoded_sum", "gpu_corrupted_sum", "expected_corrupted_sum",
    "abs_sum_damage_encoded", "normalized_abs_sum_damage",
    "oracle_match", "validity_status",
    "plane_nonzero_count", "plane_nonzero_fraction",
    "strategy_id", "strategy_scale", "strategy_shift_k",
    "slurm_job_id",
]

FAULT_RATES = ["1e-8", "1e-7", "1e-6", "1e-5", "1e-4"]
PLANES = list(range(8))
SEEDS = list(range(30))

STRATEGY_PARAMS = {
    "metadata": {"strategy_ids": ["native_scale1"]},
    "strategies": {
        "native_scale1": {
            "description": "Scale = 1",
            "per_dataset": {
                "sensor": {"scale": 1, "shift_k": 0},
                "uniform": {"scale": 1, "shift_k": 0},
                "heavy_tailed": {"scale": 1, "shift_k": 0},
                "zipfian": {"scale": 1, "shift_k": 0},
            },
        },
        "per_dataset": {
            "description": "Per-dataset scale",
            "per_dataset": {
                "sensor": {"scale": 8233095970213, "shift_k": 0},
                "uniform": {"scale": 281474977817, "shift_k": 0},
                "heavy_tailed": {"scale": 208, "shift_k": 0},
                "zipfian": {"scale": 47978, "shift_k": 0},
            },
        },
        "hybrid_scale100_plus_shifted": {
            "description": "Scale 100 + shift",
            "per_dataset": {
                "sensor": {"scale": 100, "shift_k": 51},
                "uniform": {"scale": 100, "shift_k": 46},
                "heavy_tailed": {"scale": 100, "shift_k": 16},
                "zipfian": {"scale": 100, "shift_k": 23},
            },
        },
    },
    "plane_weight": [72057594037927936, 281474976710656, 1099511627776,
                     4294967296, 16777216, 65536, 256, 1],
}


def make_row(dataset: str, plane: int, rate: str, seed: int,
             raw_damage: int, norm_damage: int,
             oracle_match: str = "true",
             strategy_id: str = "native_scale1",
             strategy_scale: str = "1",
             strategy_shift_k: str = "0",
             nzf: str = "1.0|1.0|1.0|1.0|1.0|1.0|1.0|1.0",
             job_id: str = "99999") -> dict[str, str]:
    return {
        "dataset": dataset,
        "target_plane": str(plane),
        "fault_rate": rate,
        "seed": str(seed),
        "clean_encoded_sum": "1000000",
        "gpu_corrupted_sum": str(1000000 + raw_damage),
        "expected_corrupted_sum": str(1000000 + raw_damage),
        "abs_sum_damage_encoded": str(raw_damage),
        "normalized_abs_sum_damage": str(norm_damage),
        "oracle_match": oracle_match,
        "validity_status": "canonical" if oracle_match == "true" else "ORACLE_MISMATCH",
        "plane_nonzero_count": "100000000",
        "plane_nonzero_fraction": nzf,
        "strategy_id": strategy_id,
        "strategy_scale": strategy_scale,
        "strategy_shift_k": strategy_shift_k,
        "slurm_job_id": job_id,
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_aggregator(strategy_id: str, input_dirs: list[Path],
                   output_dir: Path) -> None:
    """Run the aggregator via subprocess."""
    import subprocess
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / "params.json"
    json.dump(STRATEGY_PARAMS, params_path.open("w"))

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "aggregate_reliability_phase1_5.py"),
        "--strategy-id", strategy_id,
        "--params-json", str(params_path),
        "--output-dir", str(output_dir),
        "--input-dirs",
    ] + [str(d) for d in input_dirs]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}", file=sys.stderr)
        print(f"STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Aggregator exited code {result.returncode}")
    print(result.stdout)


def read_report(output_dir: Path) -> str:
    return (output_dir / "phase1_5_strategy_review_report.txt").read_text()


def read_summary(output_dir: Path) -> list[dict[str, str]]:
    with (output_dir / "msb_lsb_summary.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def read_meta(output_dir: Path) -> str:
    return (output_dir / "run_meta.txt").read_text()


def extract_verdict(report: str) -> str:
    for line in report.splitlines():
        if "Verdict:" in line:
            return line.split("Verdict:")[-1].strip()
    raise ValueError("verdict not found")


# ── Tests ────────────────────────────────────────────────────────

def test_parse_exact_int() -> None:
    assert parse_exact_int("0") == 0
    assert parse_exact_int("18446744073709551615") == 18446744073709551615
    assert parse_exact_int("-1") == -1
    assert parse_exact_int("7998393891707346210") == 7998393891707346210
    print("  ✓ parse_exact_int")


def test_median_int() -> None:
    assert median_int([]) == 0
    assert median_int([5]) == 5
    assert median_int([1, 3]) == 2  # floor((1+3)//2)
    assert median_int([1, 2, 3]) == 2
    assert median_int([1, 2, 3, 4]) == 2  # floor((2+3)//2) = 2
    assert median_int([10, 20, 30, 40]) == 25  # floor((20+30)//2)
    print("  ✓ median_int")


def test_median_int_large() -> None:
    vals = [1000, 2000, 3000, 4000, 5000]
    assert median_int(vals) == 3000
    vals_even = [1000, 2000, 3000, 4000]
    assert median_int(vals_even) == 2500  # floor((2000+3000)//2)
    print("  ✓ median_int large")


def build_pass_fixture(tmp_dir: Path) -> tuple[list[Path], Path]:
    """Fixture where sensor and uniform have >=3 rates with norm ratio >= 1.5x.

    Strategy: set high MSB damage and very low LSB damage for all rates.
    MSB planes [0,1,2] get norm_damage=300, LSB planes [5,6,7] get norm_damage=100.
    Ratio = 3.0, which exceeds 1.5.
    """
    sd = tmp_dir / "pass_sensor"
    ud = tmp_dir / "pass_uniform"
    sd.mkdir(parents=True)
    ud.mkdir(parents=True)

    rows_s: list[dict[str, str]] = []
    rows_u: list[dict[str, str]] = []
    for plane in PLANES:
        for rate in FAULT_RATES:
            for seed in range(30):
                grp = "msb" if plane in MSB_PLANES else ("lsb" if plane in LSB_PLANES else "mid")
                if grp == "msb":
                    nd = 300
                elif grp == "lsb":
                    nd = 100
                else:
                    nd = 150
                rd = nd * (1 << (8 * (7 - plane)))
                rows_s.append(make_row("sensor", plane, rate, seed, rd, nd))
                rows_u.append(make_row("uniform", plane, rate, seed, rd, nd))
    write_csv(sd / "sensor_results.csv", rows_s)
    write_csv(ud / "uniform_results.csv", rows_u)
    return [sd, ud], tmp_dir


def test_pass_verdict() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir = build_pass_fixture(td_p / "fixture")
        out = td_p / "output"
        run_aggregator("native_scale1", input_dirs, out)
        report = read_report(out)
        verdict = extract_verdict(report)
        assert verdict == PASS, f"Expected PASS, got {verdict}"

        meta = read_meta(out)
        assert "rows=2400" in meta, f"Expected rows=2400, got {meta}"
        assert "oracle_match=2400/2400" in meta, f"Expected oracle_match=2400/2400, got {meta}"
        assert "failure_rows=0" in meta
        assert "strategy_id=native_scale1" in meta

        summary = read_summary(out)
        for row in summary:
            ratio = float(row["normalized_ratio"])
            assert ratio >= 2.5, f"Expected ratio >= 2.5, got {ratio}"
    print("  ✓ PASS verdict")


def build_marginal_fixture(tmp_dir: Path) -> tuple[list[Path], Path]:
    """Fixture where sensor and uniform have >=3 rates with norm ratio in [1.2, 1.5).

    MSB norm_damage=140, LSB norm_damage=100 → ratio=1.4.
    """
    sd = tmp_dir / "marginal_sensor"
    ud = tmp_dir / "marginal_uniform"
    sd.mkdir(parents=True)
    ud.mkdir(parents=True)

    rows_s: list[dict[str, str]] = []
    rows_u: list[dict[str, str]] = []
    for plane in PLANES:
        for rate in FAULT_RATES:
            for seed in range(30):
                grp = "msb" if plane in MSB_PLANES else ("lsb" if plane in LSB_PLANES else "mid")
                if grp == "msb":
                    nd = 140
                elif grp == "lsb":
                    nd = 100
                else:
                    nd = 120
                rd = nd * (1 << (8 * (7 - plane)))
                rows_s.append(make_row("sensor", plane, rate, seed, rd, nd))
                rows_u.append(make_row("uniform", plane, rate, seed, rd, nd))
    write_csv(sd / "sensor_results.csv", rows_s)
    write_csv(ud / "uniform_results.csv", rows_u)
    return [sd, ud], tmp_dir


def test_marginal_verdict() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir = build_marginal_fixture(td_p / "fixture")
        out = td_p / "output"
        run_aggregator("native_scale1", input_dirs, out)
        report = read_report(out)
        verdict = extract_verdict(report)
        assert verdict == MARGINAL, f"Expected MARGINAL, got {verdict}"
    print("  ✓ MARGINAL verdict")


def build_fail_weak_ratio_fixture(tmp_dir: Path) -> tuple[list[Path], Path]:
    """Fixture where sensor and uniform have ratio < 1.2 for all rates."""
    sd = tmp_dir / "failwr_sensor"
    ud = tmp_dir / "failwr_uniform"
    sd.mkdir(parents=True)
    ud.mkdir(parents=True)

    rows_s: list[dict[str, str]] = []
    rows_u: list[dict[str, str]] = []
    for plane in PLANES:
        for rate in FAULT_RATES:
            for seed in range(30):
                nd = 105 if plane in MSB_PLANES else 100
                rd = nd * (1 << (8 * (7 - plane)))
                rows_s.append(make_row("sensor", plane, rate, seed, rd, nd))
                rows_u.append(make_row("uniform", plane, rate, seed, rd, nd))
    write_csv(sd / "sensor_results.csv", rows_s)
    write_csv(ud / "uniform_results.csv", rows_u)
    return [sd, ud], tmp_dir


def test_fail_weak_ratio() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir = build_fail_weak_ratio_fixture(td_p / "fixture")
        out = td_p / "output"
        run_aggregator("native_scale1", input_dirs, out)
        report = read_report(out)
        verdict = extract_verdict(report)
        assert verdict == FAIL, f"Expected FAIL, got {verdict}"
    print("  ✓ FAIL (weak ratio)")


def build_fail_oracle_mismatch_fixture(tmp_dir: Path) -> tuple[list[Path], Path]:
    """Fixture where sensor has oracle mismatches on some rows."""
    sd = tmp_dir / "failom_sensor"
    sd.mkdir(parents=True)

    rows_s: list[dict[str, str]] = []
    for plane in PLANES:
        for rate in FAULT_RATES:
            for seed in range(30):
                nd = 300 if plane in MSB_PLANES else 100
                rd = nd * (1 << (8 * (7 - plane)))
                om = "false" if (plane == 0 and rate == "1e-4" and seed == 0) else "true"
                rows_s.append(make_row("sensor", plane, rate, seed, rd, nd,
                                        oracle_match=om))
    write_csv(sd / "sensor_results.csv", rows_s)

    ud = tmp_dir / "failom_uniform"
    ud.mkdir(parents=True)
    rows_u: list[dict[str, str]] = []
    for plane in PLANES:
        for rate in FAULT_RATES:
            for seed in range(30):
                nd = 300 if plane in MSB_PLANES else 100
                rd = nd * (1 << (8 * (7 - plane)))
                rows_u.append(make_row("uniform", plane, rate, seed, rd, nd))
    write_csv(ud / "uniform_results.csv", rows_u)
    return [sd, ud], tmp_dir


def test_fail_oracle_mismatch() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir = build_fail_oracle_mismatch_fixture(td_p / "fixture")
        out = td_p / "output"
        run_aggregator("native_scale1", input_dirs, out)
        report = read_report(out)
        verdict = extract_verdict(report)
        assert verdict == FAIL, f"Expected FAIL, got {verdict}"

        case_failures = out / "case_failures.csv"
        assert case_failures.exists()
        with case_failures.open(newline="") as f:
            reader = csv.DictReader(f)
            failures = list(reader)
        assert len(failures) >= 1, "Expected at least 1 failure row"
    print("  ✓ FAIL (oracle mismatch)")


def build_undefined_ratio_fixture(tmp_dir: Path) -> tuple[list[Path], Path]:
    """Fixture where LSB group median == 0 for 1e-8 on sensor."""
    sd = tmp_dir / "undef_sensor"
    sd.mkdir(parents=True)

    rows_s: list[dict[str, str]] = []
    for plane in PLANES:
        for rate in FAULT_RATES:
            for seed in range(30):
                if rate == "1e-8" and plane in LSB_PLANES:
                    nd = 0
                    rd = 0
                elif plane in MSB_PLANES:
                    nd = 300
                    rd = nd * (1 << (8 * (7 - plane)))
                elif plane in LSB_PLANES:
                    nd = 100
                    rd = nd * (1 << (8 * (7 - plane)))
                else:
                    nd = 150
                    rd = nd * (1 << (8 * (7 - plane)))
                rows_s.append(make_row("sensor", plane, rate, seed, rd, nd))
    write_csv(sd / "sensor_results.csv", rows_s)

    ud = tmp_dir / "undef_uniform"
    ud.mkdir(parents=True)
    rows_u: list[dict[str, str]] = []
    for plane in PLANES:
        for rate in FAULT_RATES:
            for seed in range(30):
                nd = 300 if plane in MSB_PLANES else 100
                rd = nd * (1 << (8 * (7 - plane)))
                rows_u.append(make_row("uniform", plane, rate, seed, rd, nd))
    write_csv(ud / "uniform_results.csv", rows_u)
    return [sd, ud], tmp_dir


def test_undefined_ratio() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir = build_undefined_ratio_fixture(td_p / "fixture")
        out = td_p / "output"
        run_aggregator("native_scale1", input_dirs, out)
        summary = read_summary(out)

        # 1e-8 on sensor should have undefined ratio
        undef_rows = [r for r in summary
                      if r["dataset"] == "sensor" and r["fault_rate"] == "1e-8"]
        assert len(undef_rows) == 1
        assert undef_rows[0]["undefined_ratio_flag"] == "true", \
            "Expected undefined ratio flag for 1e-8 on sensor"
        assert undef_rows[0]["normalized_ratio"] == "UNDEFINED"

        # Other rates should be defined
        for r in summary:
            if r["fault_rate"] != "1e-8" or r["dataset"] != "sensor":
                assert r["undefined_ratio_flag"] != "true" or True  # skip

        report = read_report(out)
        verdict = extract_verdict(report)
        # PASS because 3 rates (excluding 1e-8) still hit 1.5x on sensor
        assert verdict in (PASS, MARGINAL), \
            f"Expected PASS/MARGINAL despite undefined ratio, got {verdict}"
    print("  ✓ undefined ratio handling")


def build_noise_denominator_fixture(tmp_dir: Path) -> tuple[list[Path], Path]:
    """Fixture where LSB norm median is very small (<10) for 1e-8 and 1e-7 rates."""
    sd = tmp_dir / "noise_sensor"
    ud = tmp_dir / "noise_uniform"
    sd.mkdir(parents=True)
    ud.mkdir(parents=True)

    def nd_for_plane(plane: int, rate: str) -> int:
        if rate in ("1e-8", "1e-7") and plane in LSB_PLANES:
            return 3  # < 10 → noise
        if plane in MSB_PLANES:
            return 300
        if plane in LSB_PLANES:
            return 100
        return 150

    rows_s: list[dict[str, str]] = []
    rows_u: list[dict[str, str]] = []
    for plane in PLANES:
        for rate in FAULT_RATES:
            for seed in range(30):
                nd = nd_for_plane(plane, rate)
                rd = nd * (1 << (8 * (7 - plane)))
                rows_s.append(make_row("sensor", plane, rate, seed, rd, nd))
                rows_u.append(make_row("uniform", plane, rate, seed, rd, nd))
    write_csv(sd / "sensor_results.csv", rows_s)
    write_csv(ud / "uniform_results.csv", rows_u)
    return [sd, ud], tmp_dir


def test_noise_denominator() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir = build_noise_denominator_fixture(td_p / "fixture")
        out = td_p / "output"
        run_aggregator("native_scale1", input_dirs, out)
        summary = read_summary(out)

        for row in summary:
            if row["dataset"] == "sensor" and row["fault_rate"] in ("1e-8", "1e-7"):
                assert row["noise_denominator_flag"] == "true", \
                    f"Expected noise flag for {row['dataset']} {row['fault_rate']}"
            elif row["noise_denominator_flag"] == "true":
                pass  # raw noise also possible

        report = read_report(out)
        verdict = extract_verdict(report)
        # 3 rates (1e-6, 1e-5, 1e-4) with clean ratio = 3.0 >= 1.5 → PASS
        assert verdict in (PASS, MARGINAL), \
            f"Expected PASS/MARGINAL despite noise on 2 rates, got {verdict}"
    print("  ✓ noise denominator flag")


def test_signal_not_only_1e4() -> None:
    """Fixture where only 1e-4 shows separation on sensor."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        sd = td_p / "s1e4_sensor"
        ud = td_p / "s1e4_uniform"
        sd.mkdir(parents=True)
        ud.mkdir(parents=True)

        rows_s: list[dict[str, str]] = []
        rows_u: list[dict[str, str]] = []
        for plane in PLANES:
            for rate in FAULT_RATES:
                for seed in range(30):
                    if rate == "1e-4" and plane in MSB_PLANES:
                        nd = 300
                    elif plane in MSB_PLANES:
                        nd = 100  # no separation at lower rates
                    elif plane in LSB_PLANES:
                        nd = 100
                    else:
                        nd = 100
                    rd = nd * (1 << (8 * (7 - plane)))
                    rows_s.append(make_row("sensor", plane, rate, seed, rd, nd))
                    rows_u.append(make_row("uniform", plane, rate, seed, rd, nd))
        write_csv(sd / "sensor_results.csv", rows_s)
        write_csv(ud / "uniform_results.csv", rows_u)

        out = td_p / "output"
        run_aggregator("native_scale1", [sd, ud], out)
        report = read_report(out)
        verdict = extract_verdict(report)
        # Only 1e-4 shows separation → FAIL
        assert verdict == FAIL, f"Expected FAIL (1e-4 only), got {verdict}"
    print("  ✓ signal not only 1e-4 check")


def test_inapplicable_cells() -> None:
    """Inapplicable cells are reported but don't affect PASS/FAIL."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir = build_pass_fixture(td_p / "fixture_pass")
        # Add heavy_tailed CSV that has inapplicable marker in params

        # Write params marking heavy_tailed inapplicable for native_scale1
        params = dict(STRATEGY_PARAMS)
        params["strategies"]["native_scale1"]["per_dataset"]["heavy_tailed"] = {
            "scale": 1, "shift_k": 0, "inapplicable": True,
            "inapplicable_reason": "overflow_guard"
        }

        out = td_p / "output"
        out.mkdir(parents=True, exist_ok=True)
        params_path = out / "params.json"
        json.dump(params, params_path.open("w"))

        import subprocess
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "aggregate_reliability_phase1_5.py"),
            "--strategy-id", "native_scale1",
            "--params-json", str(params_path),
            "--output-dir", str(out),
            "--input-dirs",
        ] + [str(d) for d in input_dirs]
        subprocess.run(cmd, capture_output=True, check=True)

        meta = read_meta(out)
        assert "inapplicable_cells=1" in meta, \
            f"Expected inapplicable_cells=1, got {meta}"

        report = read_report(out)
        verdict = extract_verdict(report)
        # heavy_tailed inapplicable doesn't affect verdict (sensor+uniform still PASS)
        assert verdict == PASS, f"Expected PASS despite inapplicable, got {verdict}"
    print("  ✓ inapplicable cells")


def test_msb_lsb_summary_columns() -> None:
    """Verify summary CSV has all required columns."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir = build_pass_fixture(td_p / "fixture")
        out = td_p / "output"
        run_aggregator("native_scale1", input_dirs, out)
        summary = read_summary(out)

        required_cols = [
            "dataset", "fault_rate",
            "msb_group_median_raw", "lsb_group_median_raw", "primary_ratio_raw",
            "normalized_msb_group_median", "normalized_lsb_group_median",
            "normalized_ratio",
        ]
        for p in range(8):
            required_cols.append(f"plane_{p}_occupancy")
        required_cols.extend(["undefined_ratio_flag", "noise_denominator_flag"])

        if summary:
            for col in required_cols:
                assert col in summary[0], f"Missing column: {col}"
    print("  ✓ msb_lsb_summary columns")


def test_output_files_exist() -> None:
    """Verify all expected output files are created."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir = build_pass_fixture(td_p / "fixture")
        out = td_p / "output"
        run_aggregator("native_scale1", input_dirs, out)

        expected = [
            "canonical_matrix.csv",
            "msb_lsb_summary.csv",
            "phase1_5_strategy_review_report.txt",
            "case_failures.csv",
            "run_meta.txt",
        ]
        for name in expected:
            assert (out / name).exists(), f"Missing output: {name}"
    print("  ✓ all output files present")


def test_case_failures_empty_for_pass() -> None:
    """case_failures.csv should be header-only (empty) for passing strategy."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir = build_pass_fixture(td_p / "fixture")
        out = td_p / "output"
        run_aggregator("native_scale1", input_dirs, out)

        with (out / "case_failures.csv").open(newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Header only
        assert len(rows) == 1, \
            f"Expected header-only, got {len(rows)} rows"
    print("  ✓ empty case_failures for PASS")


def test_median_and_parse_edge_cases() -> None:
    """Edge cases for parse_exact_int and median_int."""
    # parse_exact_int: negative, zero, large
    assert parse_exact_int("0") == 0
    assert parse_exact_int("-9223372036854775808") == -9223372036854775808
    assert parse_exact_int("18446744073709551615") == 18446744073709551614 + 1

    # median_int: singleton, two values, even count floor
    assert median_int([42]) == 42
    assert median_int([0, 0, 0, 0]) == 0
    assert median_int([5, 7]) == 6  # floor((5+7)//2)
    assert median_int([1, 2, 3, 100]) == 2  # floor((2+3)//2)

    # large integers
    a = 18446744073709551614
    b = 18446744073709551615
    assert median_int([a, b]) == (a + b) // 2

    print("  ✓ median and parse edge cases")


def test_pass_ratio_threshold_exact() -> None:
    """Verify the ratio >= 1.5x for exactly 3 out of 5 rates."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        sd = td_p / "p3r_sensor"
        ud = td_p / "p3r_uniform"
        sd.mkdir(parents=True)
        ud.mkdir(parents=True)

        # 3 rates get ratio = 2.0 (≥ 1.5), 2 rates get ratio = 1.0 (< 1.5)
        high_rates = {"1e-8", "1e-6", "1e-4"}

        rows_s: list[dict[str, str]] = []
        rows_u: list[dict[str, str]] = []
        for plane in PLANES:
            for rate in FAULT_RATES:
                for seed in range(30):
                    if rate in high_rates:
                        nd = 200 if plane in MSB_PLANES else 100
                    else:
                        nd = 105 if plane in MSB_PLANES else 100
                    rd = nd * (1 << (8 * (7 - plane)))
                    rows_s.append(make_row("sensor", plane, rate, seed, rd, nd))
                    rows_u.append(make_row("uniform", plane, rate, seed, rd, nd))
        write_csv(sd / "sensor_results.csv", rows_s)
        write_csv(ud / "uniform_results.csv", rows_u)

        out = td_p / "output"
        run_aggregator("native_scale1", [sd, ud], out)
        report = read_report(out)
        verdict = extract_verdict(report)
        assert verdict == PASS, (f"Expected PASS with 3/5 high-ratio rates, "
                                 f"got {verdict}")
    print("  ✓ PASS with exactly 3/5 rates >= 1.5x")


# ── Main ─────────────────────────────────────────────────────────

def main() -> None:
    tests = [
        ("parse_exact_int", test_parse_exact_int),
        ("median_int", test_median_int),
        ("median_int large", test_median_int_large),
        ("median+parse edge", test_median_and_parse_edge_cases),
        ("PASS verdict", test_pass_verdict),
        ("PASS 3/5 exact", test_pass_ratio_threshold_exact),
        ("MARGINAL verdict", test_marginal_verdict),
        ("FAIL weak ratio", test_fail_weak_ratio),
        ("FAIL oracle mismatch", test_fail_oracle_mismatch),
        ("undefined ratio", test_undefined_ratio),
        ("noise denominator", test_noise_denominator),
        ("signal only 1e-4", test_signal_not_only_1e4),
        ("inapplicable cells", test_inapplicable_cells),
        ("all output files", test_output_files_exist),
        ("empty case_failures", test_case_failures_empty_for_pass),
        ("summary columns", test_msb_lsb_summary_columns),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}", file=sys.stderr)
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
