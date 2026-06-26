#!/usr/bin/env python3
"""Unit tests for aggregate_reliability_phase2.py (Issue #130 P2-8).

Tests every acceptance criterion:
  AC1: Median computation uses exact integer arithmetic
  AC2: Policy ratio is float (rational), not int
  AC3: Undefined-ratio handling when graded median = 0
  AC4: Noise-denominator flagging when graded median < 10
  AC5: Primary-win cell identification
  AC6: All output files created
  AC7: Empty case_failures when no mismatches
  AC8: Synthetic fixture CSVs produce correct summary

Usage:
  python3 scripts/test_aggregate_reliability_phase2.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_reliability_phase2 import (
    parse_exact_int,
    median_int,
)

# ── Oracle CSV fieldnames (matches phase2_oracle.py) ──
CSV_FIELDS = [
    "run_id", "dataset", "n_rows", "scale",
    "policy", "budget_B", "allocation_r", "base_seed",
    "fault_rate", "clean_encoded_sum", "expected_voted_sum", "gpu_voted_sum",
    "signed_voted_sum_damage_encoded", "abs_voted_sum_damage_encoded",
    "normalized_abs_voted_damage", "decoded_abs_voted_damage",
    "resolved_correctly_count", "detected_mismatch_count",
    "undetected_corruption_count",
    "oracle_match", "vacuous_planes", "occupied_count",
    "artifact_id", "git_commit", "hostname", "gpu_name",
    "slurm_job_id", "repro_command",
    "strategy_id", "strategy_scale", "validity_status",
]

DATASETS = ["sensor", "uniform"]
FAULT_RATES = ["1e-8", "1e-7", "1e-6", "1e-5", "1e-4"]
BUDGET_POINTS = [8, 12, 16, 20, 24]
POLICIES = ["uniform", "graded_vacuous_aware"]
SEEDS = list(range(30))

# Fake policy catalogue for test
POLICY_CATALOGUE = {
    "metadata": {
        "created_at": "2026-06-01",
        "budget_points": BUDGET_POINTS,
        "policies": POLICIES,
    },
    "entries": [
        {"dataset": ds, "budget_B": b, "policy": p,
         "r_vector": [1] * 8, "vacuous_planes": [], "occupied_count": 8}
        for ds in DATASETS
        for b in BUDGET_POINTS
        for p in POLICIES
    ],
}


def make_row(dataset: str, policy: str, budget_B: int,
             fault_rate: str, seed: int,
             abs_damage: int, resolved: int, detected: int,
             undetected: int,
             oracle_match: str = "true",
             allocation_r: str = "1|3|3|3|2|2|2|2",
             occupied: str = "5") -> dict[str, str]:
    total = resolved + detected + undetected
    return {
        "run_id": "test",
        "dataset": dataset,
        "n_rows": str(total),
        "scale": "1",
        "policy": policy,
        "budget_B": str(budget_B),
        "allocation_r": allocation_r,
        "base_seed": str(seed),
        "fault_rate": fault_rate,
        "clean_encoded_sum": "100000000",
        "expected_voted_sum": str(100000000 + abs_damage),
        "gpu_voted_sum": str(100000000 + abs_damage),
        "signed_voted_sum_damage_encoded": str(abs_damage),
        "abs_voted_sum_damage_encoded": str(abs_damage),
        "normalized_abs_voted_damage": str(abs_damage // 1),
        "decoded_abs_voted_damage": str(abs_damage),
        "resolved_correctly_count": str(resolved),
        "detected_mismatch_count": str(detected),
        "undetected_corruption_count": str(undetected),
        "oracle_match": oracle_match,
        "vacuous_planes": "",
        "occupied_count": occupied,
        "artifact_id": "/tmp/test",
        "git_commit": "test",
        "hostname": "test",
        "gpu_name": "test",
        "slurm_job_id": "99999",
        "repro_command": "test",
        "strategy_id": "per_dataset",
        "strategy_scale": "1",
        "validity_status": "canonical" if oracle_match == "true" else "ORACLE_MISMATCH",
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_policy_catalogue(path: Path) -> None:
    path.write_text(json.dumps(POLICY_CATALOGUE) + "\n")


def run_aggregator(input_dirs: list[Path], output_dir: Path,
                   catalogue_path: Path) -> None:
    import subprocess
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "aggregate_reliability_phase2.py"),
        "--policy-catalogue", str(catalogue_path),
        "--output-dir", str(output_dir),
        "--input-dirs",
    ] + [str(d) for d in input_dirs]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}", file=sys.stderr)
        print(f"STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Aggregator exited code {result.returncode}")
    print(result.stdout)


def read_summary(output_dir: Path) -> list[dict[str, str]]:
    with (output_dir / "policy_ratio_summary.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def read_meta(output_dir: Path) -> str:
    return (output_dir / "run_meta.txt").read_text()


def read_report(output_dir: Path) -> str:
    return (output_dir / "phase2_review_report.txt").read_text()


# ── AC1: Exact integer arithmetic ──

def test_ac1_parse_exact_int() -> None:
    assert parse_exact_int("0") == 0
    assert parse_exact_int("18446744073709551615") == 18446744073709551615
    assert parse_exact_int("-1") == -1
    print("  ✓ AC1 parse_exact_int")


def test_ac1_median_int() -> None:
    assert median_int([]) == 0
    assert median_int([5]) == 5
    assert median_int([1, 3]) == 2
    assert median_int([1, 2, 3]) == 2
    assert median_int([1, 2, 3, 4]) == 2
    assert median_int([10, 20, 30, 40]) == 25
    print("  ✓ AC1 median_int")


def test_ac1_large_integers() -> None:
    a = 18446744073709551614
    b = 18446744073709551615
    assert median_int([a, b]) == (a + b) // 2
    assert parse_exact_int("9223372036854775807") == 9223372036854775807
    print("  ✓ AC1 large integer arithmetic")


# ── AC2: Policy ratio is float, not int ──

def build_graded_better_fixture(tmp_dir: Path) -> tuple[list[Path], Path, Path]:
    """Uniform damage > graded damage → policy_ratio > 1.0."""
    sd = tmp_dir / "sensor_csvs"
    ud = tmp_dir / "uniform_csvs"
    sd.mkdir(parents=True)
    ud.mkdir(parents=True)

    rows_s: list[dict[str, str]] = []
    rows_u: list[dict[str, str]] = []
    for policy in POLICIES:
        for b in BUDGET_POINTS:
            for rate in FAULT_RATES:
                for seed in SEEDS:
                    if policy == "uniform":
                        abs_d = 5000
                    else:
                        abs_d = 2000
                    rows_s.append(make_row(
                        "sensor", policy, b, rate, seed, abs_d,
                        resolved=800, detected=150, undetected=50))
                    rows_u.append(make_row(
                        "uniform", policy, b, rate, seed, abs_d,
                        resolved=800, detected=150, undetected=50))
    write_csv(sd / "sensor_results.csv", rows_s)
    write_csv(ud / "uniform_results.csv", rows_u)

    catalogue_path = tmp_dir / "policy_catalogue.json"
    write_policy_catalogue(catalogue_path)
    return [sd, ud], tmp_dir / "output", catalogue_path


def test_ac2_policy_ratio_is_float() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir, cat_path = build_graded_better_fixture(td_p)
        run_aggregator(input_dirs, out_dir, cat_path)
        summary = read_summary(out_dir)

        for row in summary:
            ratio_str = row["policy_ratio"]
            if ratio_str != "UNDEFINED":
                ratio = float(ratio_str)
                assert isinstance(ratio, float), f"Expected float, got {type(ratio)}"
                assert "." in ratio_str or "e" in ratio_str.lower(), (
                    f"Ratio {ratio_str} does not look like float")
    print("  ✓ AC2 policy ratio is float")


def test_ac2_policy_ratio_above_1() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir, cat_path = build_graded_better_fixture(td_p)
        run_aggregator(input_dirs, out_dir, cat_path)
        summary = read_summary(out_dir)

        for row in summary:
            if row["policy_ratio"] != "UNDEFINED":
                ratio = float(row["policy_ratio"])
                assert ratio == 2.5, (
                    f"Expected ratio 5000/2000=2.5, got {ratio}")
    print("  ✓ AC2 policy_ratio = 2.5 (uniform 5000 / graded 2000)")


# ── AC3: Undefined-ratio handling when graded median = 0 ──

def build_zero_graded_fixture(tmp_dir: Path) -> tuple[list[Path], Path, Path]:
    """Graded damage = 0 for some cells → undefined ratio."""
    sd = tmp_dir / "zero_sensor"
    ud = tmp_dir / "zero_uniform"
    sd.mkdir(parents=True)
    ud.mkdir(parents=True)

    rows_s: list[dict[str, str]] = []
    rows_u: list[dict[str, str]] = []
    for policy in POLICIES:
        for b in BUDGET_POINTS:
            for rate in FAULT_RATES:
                for seed in SEEDS:
                    if policy == "graded_vacuous_aware" and rate == "1e-8":
                        abs_d = 0
                    elif policy == "uniform":
                        abs_d = 3000
                    else:
                        abs_d = 1000
                    rows_s.append(make_row(
                        "sensor", policy, b, rate, seed, abs_d,
                        resolved=900, detected=80, undetected=20))
                    rows_u.append(make_row(
                        "uniform", policy, b, rate, seed, abs_d,
                        resolved=900, detected=80, undetected=20))
    write_csv(sd / "sensor_results.csv", rows_s)
    write_csv(ud / "uniform_results.csv", rows_u)

    catalogue_path = tmp_dir / "policy_catalogue.json"
    write_policy_catalogue(catalogue_path)
    return [sd, ud], tmp_dir / "output", catalogue_path


def test_ac3_undefined_ratio() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir, cat_path = build_zero_graded_fixture(td_p)
        run_aggregator(input_dirs, out_dir, cat_path)
        summary = read_summary(out_dir)

        for row in summary:
            if row["policy_ratio"] == "UNDEFINED":
                assert row["undefined_flag"] == "true", (
                    f"Expected undefined_flag=true for UNDEFINED ratio")
                assert int(row["cell_graded"]) == 0, (
                    f"Expected cell_graded=0 for UNDEFINED, got {row['cell_graded']}")
            else:
                assert row["undefined_flag"] != "true", (
                    f"Unexpected undefined_flag=true for defined ratio")
    print("  ✓ AC3 undefined ratio when graded median = 0")


# ── AC4: Noise-denominator flagging when graded median < 10 ──

def build_noise_graded_fixture(tmp_dir: Path) -> tuple[list[Path], Path, Path]:
    """Graded damage < 10 for some cells → noise-dominated flag."""
    sd = tmp_dir / "noise_sensor"
    ud = tmp_dir / "noise_uniform"
    sd.mkdir(parents=True)
    ud.mkdir(parents=True)

    rows_s: list[dict[str, str]] = []
    rows_u: list[dict[str, str]] = []
    for policy in POLICIES:
        for b in BUDGET_POINTS:
            for rate in FAULT_RATES:
                for seed in SEEDS:
                    if policy == "graded_vacuous_aware" and rate in ("1e-8", "1e-7"):
                        abs_d = 3  # < 10 → noise
                    elif policy == "uniform":
                        abs_d = 3000
                    else:
                        abs_d = 1000
                    rows_s.append(make_row(
                        "sensor", policy, b, rate, seed, abs_d,
                        resolved=900, detected=80, undetected=20))
                    rows_u.append(make_row(
                        "uniform", policy, b, rate, seed, abs_d,
                        resolved=900, detected=80, undetected=20))
    write_csv(sd / "sensor_results.csv", rows_s)
    write_csv(ud / "uniform_results.csv", rows_u)

    catalogue_path = tmp_dir / "policy_catalogue.json"
    write_policy_catalogue(catalogue_path)
    return [sd, ud], tmp_dir / "output", catalogue_path


def test_ac4_noise_dominated_flag() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir, cat_path = build_noise_graded_fixture(td_p)
        run_aggregator(input_dirs, out_dir, cat_path)
        summary = read_summary(out_dir)

        for row in summary:
            if row["policy_ratio"] != "UNDEFINED":
                cell_g = int(row["cell_graded"])
                if cell_g < 10:
                    assert row["noise_dominated_flag"] == "true", (
                        f"Expected noise_flag=true for cell_graded={cell_g}")
                else:
                    assert row["noise_dominated_flag"] != "true", (
                        f"Unexpected noise_flag=true for cell_graded={cell_g}")
    print("  ✓ AC4 noise-dominated flag when graded median < 10")


# ── AC5: Primary-win cell identification ──

def test_ac5_primary_win_identification() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir, cat_path = build_graded_better_fixture(td_p)
        run_aggregator(input_dirs, out_dir, cat_path)
        summary = read_summary(out_dir)

        # policy_ratio = 2.5 > 1.10, cell_graded = 2000 >= 10 → primary_win = true
        for row in summary:
            if row["policy_ratio"] != "UNDEFINED" and row["undefined_flag"] != "true":
                ratio = float(row["policy_ratio"])
                cell_g = int(row["cell_graded"])
                if ratio > 1.10 and cell_g >= 10:
                    assert row["primary_win"] == "true", (
                        f"Expected primary_win=true, got {row['primary_win']}")
                else:
                    assert row["primary_win"] != "true", (
                        f"Unexpected primary_win=true for ratio={ratio} graded={cell_g}")
    print("  ✓ AC5 primary-win cell identification")


def test_ac5_primary_win_not_noise() -> None:
    """Primary win requires cell_graded >= 10 (not noise-dominated)."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir, cat_path = build_noise_graded_fixture(td_p)
        run_aggregator(input_dirs, out_dir, cat_path)
        summary = read_summary(out_dir)

        for row in summary:
            if row["undefined_flag"] == "true":
                continue
            cell_g = int(row["cell_graded"])
            if cell_g < 10:
                assert row["primary_win"] != "true", (
                    f"Expected no primary_win when graded={cell_g} < 10")
    print("  ✓ AC5 primary-win not flagged for noise-dominated cells")


# ── AC6: All output files created ──

def test_ac6_output_files_exist() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir, cat_path = build_graded_better_fixture(td_p)
        run_aggregator(input_dirs, out_dir, cat_path)

        expected = [
            "phase2_matrix.csv",
            "policy_ratio_summary.csv",
            "phase2_review_report.txt",
            "case_failures.csv",
            "run_meta.txt",
        ]
        for name in expected:
            assert (out_dir / name).exists(), f"Missing output: {name}"
    print("  ✓ AC6 all output files present")


# ── AC7: Empty case_failures when no mismatches ──

def test_ac7_case_failures_empty() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir, cat_path = build_graded_better_fixture(td_p)
        run_aggregator(input_dirs, out_dir, cat_path)

        with (out_dir / "case_failures.csv").open(newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 1, \
            f"Expected header-only, got {len(rows)} rows"
    print("  ✓ AC7 empty case_failures for clean run")


def test_ac7_case_failures_with_mismatches() -> None:
    """When oracle mismatches exist, case_failures has rows."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        sd = td_p / "fail_sensor"
        sd.mkdir(parents=True)
        rows_s: list[dict[str, str]] = []
        for policy in POLICIES:
            for b in BUDGET_POINTS:
                for rate in FAULT_RATES:
                    for seed in SEEDS:
                        om = "false" if (seed == 0 and rate == "1e-4"
                                         and policy == "uniform") else "true"
                        rows_s.append(make_row(
                            "sensor", policy, b, rate, seed, 5000,
                            resolved=800, detected=150, undetected=50,
                            oracle_match=om))
        write_csv(sd / "sensor_results.csv", rows_s)

        ud = td_p / "fail_uniform"
        ud.mkdir(parents=True)
        rows_u: list[dict[str, str]] = []
        for policy in POLICIES:
            for b in BUDGET_POINTS:
                for rate in FAULT_RATES:
                    for seed in SEEDS:
                        rows_u.append(make_row(
                            "uniform", policy, b, rate, seed, 5000,
                            resolved=800, detected=150, undetected=50))
        write_csv(ud / "uniform_results.csv", rows_u)

        cat_path = td_p / "policy_catalogue.json"
        write_policy_catalogue(cat_path)

        out_dir = td_p / "output"
        run_aggregator([sd, ud], out_dir, cat_path)

        with (out_dir / "case_failures.csv").open(newline="") as f:
            reader = csv.DictReader(f)
            failures = list(reader)
        assert len(failures) >= 1, "Expected at least 1 failure row"
    print("  ✓ AC7 case_failures populated with mismatches")


# ── AC8: End-to-end synthetic fixture correctness ──

def test_ac8_matrix_row_count() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir, cat_path = build_graded_better_fixture(td_p)
        run_aggregator(input_dirs, out_dir, cat_path)

        meta = read_meta(out_dir)
        assert "rows=" in meta

        with (out_dir / "phase2_matrix.csv").open(newline="") as f:
            reader = csv.reader(f)
            matrix_rows = list(reader)
        # Header + data rows
        expected_rows = 1 + 2 * 2 * len(BUDGET_POINTS) * len(FAULT_RATES) * len(SEEDS)
        assert len(matrix_rows) == expected_rows, (
            f"Expected {expected_rows} rows, got {len(matrix_rows)}")
    print("  ✓ AC8 matrix row count")


def test_ac8_summary_columns() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir, cat_path = build_graded_better_fixture(td_p)
        run_aggregator(input_dirs, out_dir, cat_path)
        summary = read_summary(out_dir)

        required = [
            "dataset", "budget_B", "fault_rate",
            "cell_uniform", "cell_graded", "policy_ratio",
            "noise_dominated_flag", "undefined_flag", "primary_win",
        ]
        if summary:
            for col in required:
                assert col in summary[0], f"Missing column: {col}"
    print("  ✓ AC8 summary columns")


def test_ac8_meta_integrity() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        input_dirs, out_dir, cat_path = build_graded_better_fixture(td_p)
        run_aggregator(input_dirs, out_dir, cat_path)
        meta = read_meta(out_dir)

        assert "oracle_match=" in meta
        assert "failure_rows=0" in meta
        assert "primary_win_cells=" in meta
        assert "datasets=" in meta
    print("  ✓ AC8 meta integrity")


def test_ac8_ratio_precise_values() -> None:
    """Verify precise ratio values in a controlled fixture."""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        sd = td_p / "precise_sensor"
        sd.mkdir(parents=True)
        ud = td_p / "precise_uniform"
        ud.mkdir(parents=True)

        rows_s: list[dict[str, str]] = []
        rows_u: list[dict[str, str]] = []
        for policy in POLICIES:
            for b in BUDGET_POINTS:
                for rate in FAULT_RATES:
                    for seed in SEEDS:
                        if policy == "uniform":
                            d = 100
                        elif rate == "1e-8":
                            d = 0
                        else:
                            d = 40
                        rows_s.append(make_row(
                            "sensor", policy, b, rate, seed,
                            abs_damage=d,
                            resolved=800, detected=150, undetected=50))
                        rows_u.append(make_row(
                            "uniform", policy, b, rate, seed,
                            abs_damage=d,
                            resolved=800, detected=150, undetected=50))
        write_csv(sd / "sensor_results.csv", rows_s)
        write_csv(ud / "uniform_results.csv", rows_u)

        cat_path = td_p / "policy_catalogue.json"
        write_policy_catalogue(cat_path)
        out_dir = td_p / "output"
        run_aggregator([sd, ud], out_dir, cat_path)

        summary = read_summary(out_dir)
        for row in summary:
            rate = row["fault_rate"]
            if row["undefined_flag"] == "true":
                assert row["policy_ratio"] == "UNDEFINED"
                assert int(row["cell_graded"]) == 0
            else:
                cell_u = int(row["cell_uniform"])
                cell_g = int(row["cell_graded"])
                ratio = float(row["policy_ratio"])
                expected = cell_u / cell_g
                assert abs(ratio - expected) < 1e-9, (
                    f"Ratio mismatch: {ratio} vs {expected}")
    print("  ✓ AC8 precise ratio values")


# ── Run all tests ──

def main() -> None:
    tests = [
        ("AC1 parse_exact_int", test_ac1_parse_exact_int),
        ("AC1 median_int", test_ac1_median_int),
        ("AC1 large integers", test_ac1_large_integers),
        ("AC2 ratio is float", test_ac2_policy_ratio_is_float),
        ("AC2 ratio above 1", test_ac2_policy_ratio_above_1),
        ("AC3 undefined ratio", test_ac3_undefined_ratio),
        ("AC4 noise flag", test_ac4_noise_dominated_flag),
        ("AC5 primary-win", test_ac5_primary_win_identification),
        ("AC5 primary-win no noise", test_ac5_primary_win_not_noise),
        ("AC6 output files", test_ac6_output_files_exist),
        ("AC7 empty case_failures", test_ac7_case_failures_empty),
        ("AC7 with mismatches", test_ac7_case_failures_with_mismatches),
        ("AC8 matrix row count", test_ac8_matrix_row_count),
        ("AC8 summary columns", test_ac8_summary_columns),
        ("AC8 meta integrity", test_ac8_meta_integrity),
        ("AC8 precise ratios", test_ac8_ratio_precise_values),
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
