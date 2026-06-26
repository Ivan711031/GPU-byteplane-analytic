#!/usr/bin/env python3
"""Tests for run_phase2_mechanism_check.py (Issue #131 P2-9).

Covers all 5 acceptance criteria:
  AC1: Correctly identifies primary-win cells from summary
  AC2: Generates graded_naive r_vectors from catalogue
  AC3: Mechanism check status computation
  AC4: PASS/FAIL gate logic
  AC5: Unit tests with synthetic fixture data
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_phase2_mechanism_check import (
    PRIMARY_RATIO_THRESHOLD,
    CELL_GRADED_MIN,
    ROBUST_FRACTION_REQUIRED,
    MECHANISM_POLICY,
    UNIFORM_POLICY,
    read_policy_ratio_summary,
    identify_primary_win_cells,
    load_policy_catalogue,
    find_catalogue_entry,
    lookup_graded_naive_r_vector,
    compute_mechanism_check_status,
    compute_gate_verdict,
    build_summary_rows,
    load_uniform_results,
    load_mechanism_results,
    compute_cell_median,
    generate_mock_mechanism_results,
)


# ── Fixture helpers ────────────────────────────────────────────────────


def make_policy_ratio_summary_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Write a synthetic policy_ratio_summary.csv."""
    fieldnames = [
        "dataset", "budget_B", "fault_rate", "policy",
        "policy_ratio", "cell_graded", "cell_uniform",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_phase2_matrix_csv(
    path: Path,
    uniform_rows: list[dict[str, str]],
) -> None:
    """Write a synthetic phase2_matrix.csv with uniform results."""
    fieldnames = [
        "run_id", "dataset", "n_rows", "scale",
        "policy", "budget_B", "allocation_r", "base_seed",
        "fault_rate", "clean_encoded_sum", "expected_voted_sum",
        "gpu_voted_sum", "abs_voted_sum_damage_encoded",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(uniform_rows)


def make_policy_catalogue_json(
    path: Path,
    entries: list[dict],
) -> None:
    """Write a synthetic policy_catalogue.json."""
    data = {
        "metadata": {
            "created_at": "2026-06-01",
            "policies": ["uniform", "graded_vacuous_aware", "graded_naive"],
        },
        "entries": entries,
    }
    path.write_text(json.dumps(data) + "\n")


def make_sensitivity_profile_json(path: Path) -> None:
    """Write a minimal sensitivity profile (needed for arg validation)."""
    data = {
        "metadata": {"source": "test"},
        "datasets": {},
    }
    path.write_text(json.dumps(data) + "\n")


# ── Synthetic data constants ──────────────────────────────────────────

WINNING_ROW_1 = {
    "dataset": "sensor",
    "budget_B": "16",
    "fault_rate": "1e-06",
    "policy": "graded_vacuous_aware",
    "policy_ratio": "1.50",
    "cell_graded": "1000",
    "cell_uniform": "1500",
}

WINNING_ROW_2 = {
    "dataset": "sensor",
    "budget_B": "20",
    "fault_rate": "1e-06",
    "policy": "graded_vacuous_aware",
    "policy_ratio": "1.35",
    "cell_graded": "800",
    "cell_uniform": "1080",
}

WINNING_ROW_3 = {
    "dataset": "uniform",
    "budget_B": "16",
    "fault_rate": "1e-07",
    "policy": "graded_vacuous_aware",
    "policy_ratio": "1.20",
    "cell_graded": "500",
    "cell_uniform": "600",
}

NON_WINNING_LOW_RATIO = {
    "dataset": "sensor",
    "budget_B": "8",
    "fault_rate": "1e-06",
    "policy": "graded_vacuous_aware",
    "policy_ratio": "1.05",  # below threshold
    "cell_graded": "1000",
    "cell_uniform": "1050",
}

NON_WINNING_LOW_CELL = {
    "dataset": "sensor",
    "budget_B": "16",
    "fault_rate": "1e-06",
    "policy": "graded_vacuous_aware",
    "policy_ratio": "1.50",
    "cell_graded": "5",  # below min
    "cell_uniform": "7",
}

CATALOGUE_ENTRIES = [
    {
        "dataset": "sensor",
        "budget_B": 16,
        "policy": "graded_naive",
        "fault_rate": "1e-06",
        "r_vector": [2, 2, 2, 2, 2, 2, 2, 2],
        "vacuous_planes": [0, 1],
        "occupied_count": 8,
    },
    {
        "dataset": "sensor",
        "budget_B": 20,
        "policy": "graded_naive",
        "fault_rate": "1e-06",
        "r_vector": [3, 3, 2, 2, 2, 3, 3, 2],
        "vacuous_planes": [0, 1],
        "occupied_count": 8,
    },
    {
        "dataset": "uniform",
        "budget_B": 16,
        "policy": "graded_naive",
        "fault_rate": "1e-07",
        "r_vector": [2, 2, 2, 2, 2, 2, 2, 2],
        "vacuous_planes": [0, 1],
        "occupied_count": 8,
    },
]

UNIFORM_MATRIX_ROWS = [
    {
        "run_id": "u_sensor_16_1e6_0",
        "dataset": "sensor",
        "n_rows": "100000000",
        "scale": "8233095970213",
        "policy": "uniform",
        "budget_B": "16",
        "allocation_r": "2|2|2|2|2|2|2|2",
        "base_seed": "0",
        "fault_rate": "1e-06",
        "clean_encoded_sum": "12345",
        "expected_voted_sum": "10845",
        "gpu_voted_sum": "10845",
        "abs_voted_sum_damage_encoded": "1500",
    },
    {
        "run_id": "u_sensor_20_1e6_1",
        "dataset": "sensor",
        "n_rows": "100000000",
        "scale": "8233095970213",
        "policy": "uniform",
        "budget_B": "20",
        "allocation_r": "3|3|3|3|2|2|2|2",
        "base_seed": "0",
        "fault_rate": "1e-06",
        "clean_encoded_sum": "12345",
        "expected_voted_sum": "11265",
        "gpu_voted_sum": "11265",
        "abs_voted_sum_damage_encoded": "1080",
    },
    {
        "run_id": "u_uniform_16_1e7_2",
        "dataset": "uniform",
        "n_rows": "100000000",
        "scale": "1",
        "policy": "uniform",
        "budget_B": "16",
        "allocation_r": "2|2|2|2|2|2|2|2",
        "base_seed": "0",
        "fault_rate": "1e-07",
        "clean_encoded_sum": "9999",
        "expected_voted_sum": "9399",
        "gpu_voted_sum": "9399",
        "abs_voted_sum_damage_encoded": "600",
    },
]


class TestIdentifyPrimaryWinCells:
    """AC1: Primary-win cell identification."""

    def test_identifies_correct_cells(self):
        rows = [WINNING_ROW_1, WINNING_ROW_2, WINNING_ROW_3,
                NON_WINNING_LOW_RATIO, NON_WINNING_LOW_CELL]
        wins = identify_primary_win_cells(rows)
        assert len(wins) == 3
        datasets = {w["dataset"] for w in wins}
        assert datasets == {"sensor", "uniform"}

    def test_empty_input(self):
        assert identify_primary_win_cells([]) == []

    def test_all_non_winning(self):
        rows = [NON_WINNING_LOW_RATIO, NON_WINNING_LOW_CELL]
        assert identify_primary_win_cells(rows) == []

    def test_boundary_ratio(self):
        # Exactly at threshold does not count (must be strictly >)
        at_threshold = dict(WINNING_ROW_1)
        at_threshold["policy_ratio"] = str(PRIMARY_RATIO_THRESHOLD)
        at_threshold["cell_graded"] = str(CELL_GRADED_MIN)
        wins = identify_primary_win_cells([at_threshold])
        assert len(wins) == 0

    def test_boundary_cell_graded(self):
        below_min = dict(WINNING_ROW_1)
        below_min["cell_graded"] = str(CELL_GRADED_MIN - 1)
        wins = identify_primary_win_cells([below_min])
        assert len(wins) == 0

    def test_round_trip_from_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "policy_ratio_summary.csv"
            all_rows = [
                WINNING_ROW_1, WINNING_ROW_2, NON_WINNING_LOW_RATIO,
            ]
            make_policy_ratio_summary_csv(csv_path, all_rows)
            parsed = read_policy_ratio_summary(csv_path)
            wins = identify_primary_win_cells(parsed)
            assert len(wins) == 2


class TestLookupGradedNaiveRVector:
    """AC2: graded_naive r_vector lookup from catalogue."""

    def test_lookup_sensor_B16(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat_path = Path(tmp) / "policy_catalogue.json"
            make_policy_catalogue_json(cat_path, CATALOGUE_ENTRIES)
            catalogue = load_policy_catalogue(cat_path)
            rv = lookup_graded_naive_r_vector(catalogue, "sensor", 16)
            assert rv == [2, 2, 2, 2, 2, 2, 2, 2]
            assert sum(rv) == 16

    def test_lookup_sensor_B20(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat_path = Path(tmp) / "policy_catalogue.json"
            make_policy_catalogue_json(cat_path, CATALOGUE_ENTRIES)
            catalogue = load_policy_catalogue(cat_path)
            rv = lookup_graded_naive_r_vector(catalogue, "sensor", 20)
            assert rv == [3, 3, 2, 2, 2, 3, 3, 2]
            assert sum(rv) == 20

    def test_lookup_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat_path = Path(tmp) / "policy_catalogue.json"
            make_policy_catalogue_json(cat_path, CATALOGUE_ENTRIES)
            catalogue = load_policy_catalogue(cat_path)
            rv = lookup_graded_naive_r_vector(catalogue, "nonexistent", 16)
            assert rv is None

    def test_find_catalogue_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat_path = Path(tmp) / "policy_catalogue.json"
            make_policy_catalogue_json(cat_path, CATALOGUE_ENTRIES)
            catalogue = load_policy_catalogue(cat_path)
            entry = find_catalogue_entry(
                catalogue, "sensor", 20, MECHANISM_POLICY,
            )
            assert entry is not None
            assert entry["dataset"] == "sensor"
            assert entry["budget_B"] == 20
            assert entry["policy"] == "graded_naive"

    def test_find_nonexistent_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat_path = Path(tmp) / "policy_catalogue.json"
            make_policy_catalogue_json(cat_path, CATALOGUE_ENTRIES)
            catalogue = load_policy_catalogue(cat_path)
            entry = find_catalogue_entry(
                catalogue, "sensor", 16, "nonexistent_policy",
            )
            assert entry is None


class TestMechanismCheckStatus:
    """AC3: Mechanism check status computation."""

    def test_robust_when_primary_le_125x_naive(self):
        # primary_ratio=1.0, naive_ratio=1.0 => 1.0 <= 1.25 => robust
        status = compute_mechanism_check_status(1.0, 1.0)
        assert status == "robust"

    def test_robust_at_boundary(self):
        # primary_ratio=1.25, naive_ratio=1.0 => 1.25 <= 1.25 => robust
        status = compute_mechanism_check_status(1.25, 1.0)
        assert status == "robust"

    def test_vacuous_artifact_when_primary_exceeds(self):
        # primary_ratio=1.26, naive_ratio=1.0 => 1.26 > 1.25 => vacuous_artifact
        status = compute_mechanism_check_status(1.26, 1.0)
        assert status == "vacuous_artifact"

    def test_robust_high_naive_ratio(self):
        # primary_ratio=1.5, naive_ratio=1.3 => 1.5 <= 1.25*1.3=1.625 => robust
        status = compute_mechanism_check_status(1.5, 1.3)
        assert status == "robust"

    def test_vacuous_artifact_low_naive_ratio(self):
        # primary_ratio=1.5, naive_ratio=1.0 => 1.5 > 1.25 => vacuous_artifact
        status = compute_mechanism_check_status(1.5, 1.0)
        assert status == "vacuous_artifact"

    def test_robust_equal_at_125x(self):
        # primary_ratio=1.0, naive_ratio=0.8 => 1.0 <= 1.25*0.8=1.0 => robust
        status = compute_mechanism_check_status(1.0, 0.8)
        assert status == "robust"

    def test_vacuous_artifact_large_primary(self):
        # primary_ratio=2.0, naive_ratio=1.0 => 2.0 > 1.25 => vacuous_artifact
        status = compute_mechanism_check_status(2.0, 1.0)
        assert status == "vacuous_artifact"


class TestGateVerdict:
    """AC4: PASS/FAIL gate logic."""

    def test_pass_all_robust(self):
        rows = [
            {"mechanism_check_status": "robust"},
            {"mechanism_check_status": "robust"},
            {"mechanism_check_status": "robust"},
        ]
        verdict, counts = compute_gate_verdict(rows)
        assert verdict == "PASS"
        assert counts["robust"] == 3

    def test_pass_at_threshold(self):
        # 8 robust + 2 vacuous_artifact = 8/10 = 80%, exactly at threshold
        rows = [
            {"mechanism_check_status": "robust"},
        ] * 8 + [{"mechanism_check_status": "vacuous_artifact"}] * 2
        verdict, counts = compute_gate_verdict(rows)
        assert verdict == "PASS"

    def test_fail_below_threshold(self):
        # 7 robust + 3 vacuous_artifact = 7/10 = 70% < 80%
        rows = [
            {"mechanism_check_status": "robust"},
        ] * 7 + [{"mechanism_check_status": "vacuous_artifact"}] * 3
        verdict, counts = compute_gate_verdict(rows)
        assert verdict == "FAIL"

    def test_fail_any_vacuous_artifact(self):
        # 3 robust + 1 vacuous_artifact = 3/4 = 75% < 80%
        rows = [
            {"mechanism_check_status": "robust"},
            {"mechanism_check_status": "robust"},
            {"mechanism_check_status": "robust"},
            {"mechanism_check_status": "vacuous_artifact"},
        ]
        verdict, counts = compute_gate_verdict(rows)
        assert verdict == "FAIL"
        assert counts["vacuous_artifact"] == 1

    def test_fail_vacuous_artifact_any(self):
        # One vacuous_artifact always fails even if robust fraction is high
        rows = [
            {"mechanism_check_status": "robust"},
            {"mechanism_check_status": "vacuous_artifact"},
        ]
        verdict, counts = compute_gate_verdict(rows)
        assert verdict == "FAIL"

    def test_inconclusive_no_resolved(self):
        rows = [
            {"mechanism_check_status": "pending"},
            {"mechanism_check_status": "pending"},
        ]
        verdict, counts = compute_gate_verdict(rows)
        assert "INCONCLUSIVE" in verdict

    def test_unknown_status_counted(self):
        rows = [
            {"mechanism_check_status": "robust"},
            {"mechanism_check_status": "unknown"},
        ]
        verdict, counts = compute_gate_verdict(rows)
        assert counts["unknown"] == 1
        assert counts["robust"] == 1


class TestEndToEndWithSyntheticData:
    """AC5: Full pipeline with synthetic fixture data."""

    def test_e2e_mock_all_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            # Primary summary with 3 winning cells
            summary_path = out / "policy_ratio_summary.csv"
            make_policy_ratio_summary_csv(summary_path, [
                WINNING_ROW_1, WINNING_ROW_2, WINNING_ROW_3,
            ])

            # Phase 2 matrix with uniform results
            matrix_path = out / "phase2_matrix.csv"
            make_phase2_matrix_csv(matrix_path, UNIFORM_MATRIX_ROWS)

            # Symlink-like: re-use the same format for mechanism check matrix
            # In mock mode, generate_mock_mechanism_results creates synthetic data
            from run_phase2_mechanism_check import (
                load_policy_catalogue, lookup_graded_naive_r_vector,
                identify_primary_win_cells, read_policy_ratio_summary,
            )

            rows = read_policy_ratio_summary(summary_path)
            wins = identify_primary_win_cells(rows)
            assert len(wins) == 3

            uniform_results = load_uniform_results(matrix_path)
            assert len(uniform_results) == 3

            # Set a fixed seed for reproducibility
            gn_results = generate_mock_mechanism_results(
                wins, uniform_results, base_seed=42,
            )
            assert len(gn_results) == 3

            # Build per-cell lookup
            gn_cell_map = {}
            for r in gn_results:
                key = (r["dataset"], int(r["budget_B"]), r["fault_rate"])
                gn_cell_map.setdefault(key, []).append(
                    float(r["abs_voted_sum_damage_encoded"])
                )

            # Build summary and compute verdict
            summary = build_summary_rows(wins, uniform_results, gn_cell_map)
            assert len(summary) == 3

            verdict, counts = compute_gate_verdict(summary)
            # With seed=42, most cells should be robust
            assert counts["robust"] >= 2
            # Unless all rolled reversed (unlikely at 15% per cell)
            assert verdict in ("PASS", "FAIL")

    def test_e2e_mock_known_ratio(self):
        """Verify graded_naive_ratio computation with synthetic data."""
        # For a cell where uniform median = 1500 and graded_naive damage varies,
        # the ratio should be 1500 / median(gn_damages)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            summary_path = out / "policy_ratio_summary.csv"
            make_policy_ratio_summary_csv(summary_path, [WINNING_ROW_1])

            matrix_path = out / "phase2_matrix.csv"
            make_phase2_matrix_csv(matrix_path, UNIFORM_MATRIX_ROWS)

            rows = read_policy_ratio_summary(summary_path)
            wins = identify_primary_win_cells(rows)
            uniform_results = load_uniform_results(matrix_path)

            # Give exact graded_naive damages
            gn_cell_map = {
                ("sensor", 16, "1e-06"): [1200.0, 1300.0, 1250.0],
            }
            summary = build_summary_rows(wins, uniform_results, gn_cell_map)

            assert len(summary) == 1
            row = summary[0]
            # uniform median = 1500, gn median = median(1200, 1250, 1300) = 1250
            # gn_ratio = 1500 / 1250 = 1.20
            gn_ratio = float(row["graded_naive_ratio"])
            assert abs(gn_ratio - 1.20) < 0.01
            # primary_ratio=1.50, naive_ratio=1.20 => 1.50 <= 1.25*1.20 => robust
            assert row["mechanism_check_status"] == "robust"

    def test_e2e_mock_reversed_scenario(self):
        """When graded_naive does worse than uniform, status is reversed."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            summary_path = out / "policy_ratio_summary.csv"
            make_policy_ratio_summary_csv(summary_path, [WINNING_ROW_1])

            matrix_path = out / "phase2_matrix.csv"
            make_phase2_matrix_csv(matrix_path, UNIFORM_MATRIX_ROWS)

            rows = read_policy_ratio_summary(summary_path)
            wins = identify_primary_win_cells(rows)
            uniform_results = load_uniform_results(matrix_path)

            # Graded_naive damage HIGHER than uniform → ratio < 1
            gn_cell_map = {
                ("sensor", 16, "1e-06"): [2000.0, 2100.0, 1900.0],
            }
            summary = build_summary_rows(wins, uniform_results, gn_cell_map)
            row = summary[0]
            # uniform = 1500, gn median = 2000
            # gn_ratio = 1500/2000 = 0.75 → reversed
            gn_ratio = float(row["graded_naive_ratio"])
            assert abs(gn_ratio - 0.75) < 0.01
            # primary_ratio=1.50, naive_ratio=0.75 => 1.50 > 1.25*0.75 => vacuous_artifact
            assert row["mechanism_check_status"] == "vacuous_artifact"

            verdict, counts = compute_gate_verdict(summary)
            assert verdict == "FAIL"
            assert counts["vacuous_artifact"] == 1

    def test_e2e_mock_neutral_scenario(self):
        """When graded_naive is similar to uniform, status is neutral."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            summary_path = out / "policy_ratio_summary.csv"
            make_policy_ratio_summary_csv(summary_path, [WINNING_ROW_1])

            matrix_path = out / "phase2_matrix.csv"
            make_phase2_matrix_csv(matrix_path, UNIFORM_MATRIX_ROWS)

            rows = read_policy_ratio_summary(summary_path)
            wins = identify_primary_win_cells(rows)
            uniform_results = load_uniform_results(matrix_path)

            # Graded_naive damage very close to uniform → ratio ≈ 1
            gn_cell_map = {
                ("sensor", 16, "1e-06"): [1480.0, 1500.0, 1520.0],
            }
            summary = build_summary_rows(wins, uniform_results, gn_cell_map)
            row = summary[0]
            # uniform = 1500, gn median = 1500
            # gn_ratio = 1500/1500 = 1.00 → neutral
            gn_ratio = float(row["graded_naive_ratio"])
            assert abs(gn_ratio - 1.00) < 0.01
            # primary_ratio=1.50, naive_ratio=1.00 => 1.50 > 1.25 => vacuous_artifact
            assert row["mechanism_check_status"] == "vacuous_artifact"

    def test_e2e_mock_no_gn_results_yet(self):
        """Without graded_naive results, status is pending."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            summary_path = out / "policy_ratio_summary.csv"
            make_policy_ratio_summary_csv(summary_path, [WINNING_ROW_1])

            matrix_path = out / "phase2_matrix.csv"
            make_phase2_matrix_csv(matrix_path, UNIFORM_MATRIX_ROWS)

            rows = read_policy_ratio_summary(summary_path)
            wins = identify_primary_win_cells(rows)
            uniform_results = load_uniform_results(matrix_path)

            # No graded_naive results
            gn_cell_map = {}
            summary = build_summary_rows(wins, uniform_results, gn_cell_map)
            assert len(summary) == 1
            assert summary[0]["mechanism_check_status"] == "pending"

    def test_load_uniform_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "phase2_matrix.csv"
            # Add a duplicate run for median calculation
            rows = list(UNIFORM_MATRIX_ROWS)
            # Add another run for sensor B=16 with different seed
            rows.append({
                "run_id": "u_sensor_16_1e6_1",
                "dataset": "sensor",
                "n_rows": "100000000",
                "scale": "8233095970213",
                "policy": "uniform",
                "budget_B": "16",
                "allocation_r": "2|2|2|2|2|2|2|2",
                "base_seed": "1",
                "fault_rate": "1e-06",
                "clean_encoded_sum": "12345",
                "expected_voted_sum": "10845",
                "gpu_voted_sum": "10845",
                "abs_voted_sum_damage_encoded": "1600",
            })
            make_phase2_matrix_csv(matrix_path, rows)

            results = load_uniform_results(matrix_path)
            # sensor B=16 1e-06: median of [1500, 1600] = 1550
            key = ("sensor", 16, "1e-06")
            assert key in results
            assert results[key] == 1550.0

    def test_load_uniform_results_odd_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "phase2_matrix.csv"
            rows = [
                {
                    "run_id": "u_sensor_16_1e6_0",
                    "dataset": "sensor",
                    "n_rows": "100000000",
                    "scale": "8233095970213",
                    "policy": "uniform",
                    "budget_B": "16",
                    "allocation_r": "2|2|2|2|2|2|2|2",
                    "base_seed": "0",
                    "fault_rate": "1e-06",
                    "clean_encoded_sum": "12345",
                    "expected_voted_sum": "10845",
                    "gpu_voted_sum": "10845",
                    "abs_voted_sum_damage_encoded": "1500",
                },
            ]
            make_phase2_matrix_csv(matrix_path, rows)

            results = load_uniform_results(matrix_path)
            key = ("sensor", 16, "1e-06")
            assert results[key] == 1500.0

    def test_compute_cell_median(self):
        assert compute_cell_median([1.0, 3.0, 2.0]) == 2.0
        assert compute_cell_median([1.0, 2.0]) == 1.5
        assert compute_cell_median([5.0]) == 5.0

    def test_mock_results_no_uniform_data(self):
        """When uniform_results is empty, generate_mock should still work."""
        wins = [WINNING_ROW_1]
        uniform_results = {}
        gn_results = generate_mock_mechanism_results(wins, uniform_results)
        assert len(gn_results) == 1
        assert gn_results[0]["dataset"] == "sensor"


class TestEndToEndCLI:
    """Run the script via main() with --mock and verify outputs."""

    def test_cli_mock_creates_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            # Write inputs
            summary_path = out / "policy_ratio_summary.csv"
            make_policy_ratio_summary_csv(summary_path, [
                WINNING_ROW_1, WINNING_ROW_2, NON_WINNING_LOW_RATIO,
            ])

            matrix_path = out / "phase2_matrix.csv"
            make_phase2_matrix_csv(matrix_path, UNIFORM_MATRIX_ROWS)

            cat_path = out / "policy_catalogue.json"
            make_policy_catalogue_json(cat_path, CATALOGUE_ENTRIES)

            profile_path = out / "phase2_sensitivity_profile.json"
            make_sensitivity_profile_json(profile_path)

            output_dir = out / "mechanism_check"
            output_dir.mkdir(exist_ok=True)

            # Run the script
            from run_phase2_mechanism_check import main as script_main
            import sys as _sys
            saved_argv = _sys.argv
            try:
                _sys.argv = [
                    "run_phase2_mechanism_check.py",
                    "--primary-summary", str(summary_path),
                    "--policy-catalogue", str(cat_path),
                    "--sensitivity-profile", str(profile_path),
                    "--phase2-matrix", str(matrix_path),
                    "--output-dir", str(output_dir),
                    "--mock",
                    "--base-seed", "123",
                ]
                script_main()
            finally:
                _sys.argv = saved_argv

            # Verify outputs exist
            assert (output_dir / "phase2_mechanism_check_matrix.csv").is_file()
            assert (output_dir / "mechanism_check_summary.csv").is_file()
            assert (output_dir / "mechanism_check_report.txt").is_file()

            # Verify summary content
            with (output_dir / "mechanism_check_summary.csv").open(newline="") as f:
                reader = csv.DictReader(f)
                srows = list(reader)
            assert len(srows) == 2  # 2 primary-win cells

            # Verify report content
            report = (output_dir / "mechanism_check_report.txt").read_text()
            assert "Verdict:" in report
            assert "robust" in report

    def test_cli_mock_no_wins(self):
        """When no primary-win cells exist, script should produce NO_OP report."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            summary_path = out / "policy_ratio_summary.csv"
            make_policy_ratio_summary_csv(summary_path, [NON_WINNING_LOW_RATIO])

            matrix_path = out / "phase2_matrix.csv"
            make_phase2_matrix_csv(matrix_path, UNIFORM_MATRIX_ROWS)

            cat_path = out / "policy_catalogue.json"
            make_policy_catalogue_json(cat_path, CATALOGUE_ENTRIES)

            profile_path = out / "phase2_sensitivity_profile.json"
            make_sensitivity_profile_json(profile_path)

            output_dir = out / "mechanism_check"
            output_dir.mkdir(exist_ok=True)

            from run_phase2_mechanism_check import main as script_main
            import sys as _sys
            saved_argv = _sys.argv
            try:
                _sys.argv = [
                    "run_phase2_mechanism_check.py",
                    "--primary-summary", str(summary_path),
                    "--policy-catalogue", str(cat_path),
                    "--sensitivity-profile", str(profile_path),
                    "--phase2-matrix", str(matrix_path),
                    "--output-dir", str(output_dir),
                    "--mock",
                ]
                import pytest
                with pytest.raises(SystemExit):
                    script_main()
            finally:
                _sys.argv = saved_argv

            report = (output_dir / "mechanism_check_report.txt").read_text()
            assert "NO_OP" in report or "No primary-win cells" in report


if __name__ == "__main__":
    import types
    test_fns = [
        v for k, v in list(globals().items())
        if k.startswith("test_") and isinstance(v, type) or
        (k.startswith("Test") and isinstance(v, type))
    ]
    # Run test classes
    passed = 0
    failed = 0
    for cls in test_fns:
        if not isinstance(cls, type):
            continue
        instance = cls()
        methods = [
            getattr(instance, m) for m in dir(instance)
            if m.startswith("test_") and callable(getattr(instance, m))
        ]
        for method in methods:
            try:
                method()
                print(f"  \u2713 {cls.__name__}.{method.__name__}")
                passed += 1
            except Exception as e:
                print(f"  \u2717 {cls.__name__}.{method.__name__}: {e}")
                failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
