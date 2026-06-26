import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_nmr_a_structured_fault.py"
    spec = spec_from_file_location("run_nmr_a_structured_fault", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def scenario_by_name(module, fault_family):
    for scenario in module.FAULT_SCENARIOS:
        if scenario["fault_family"] == fault_family:
            return scenario
    raise AssertionError(f"missing scenario: {fault_family}")


class RunNmrAStructuredFaultTests(unittest.TestCase):
    def test_sum32_is_bytewise(self):
        mod = load_module()
        self.assertEqual(mod.sum32(bytes([1, 2, 3, 4])), 10)

    def test_single_domain_burst_is_policy_aware(self):
        mod = load_module()
        scenario = scenario_by_name(mod, "single_domain_burst")

        naive_plan = mod.generate_fault_plan(
            "naive_same_region_replication_r3",
            scenario,
            seed=0,
            r_vector=[3, 1, 1, 1, 1, 1, 1, 1],
            n_rows=64,
            lsb_plane=7,
        )
        logical_plan = mod.generate_fault_plan(
            "logical_striping_diversity_r3",
            scenario,
            seed=0,
            r_vector=[3, 1, 1, 1, 1, 1, 1, 1],
            n_rows=64,
            lsb_plane=7,
        )

        self.assertEqual({entry["replica"] for entry in naive_plan}, {0, 1, 2})
        self.assertEqual(len({entry["offset"] for entry in naive_plan}), 4)
        self.assertEqual(len({entry["replica"] for entry in logical_plan}), 1)
        self.assertEqual(len({entry["offset"] for entry in logical_plan}), 4)

    def test_column_like_offsets_follow_stride(self):
        mod = load_module()
        scenario = {
            "scenario_id": 99,
            "fault_family": "column_like_repeated_offset",
            "plane_label": "msb",
            "plane": 0,
            "column_like_stride": 4,
            "repeat_count": 4,
            "replica_mode": "policy_domain_aware",
        }
        plan = mod.generate_fault_plan(
            "logical_striping_diversity_r3",
            scenario,
            seed=0,
            r_vector=[3, 1, 1, 1, 1, 1, 1, 1],
            n_rows=32,
            lsb_plane=7,
        )
        offsets = sorted({entry["offset"] for entry in plan})
        self.assertEqual(offsets, [offsets[0] + 4 * i for i in range(4)])

    def test_same_fault_all_hits_all_replicas(self):
        mod = load_module()
        scenario = scenario_by_name(mod, "same_fault_all_correlated_control")
        plan = mod.generate_fault_plan(
            "graded_nmr_B11",
            scenario,
            seed=0,
            r_vector=[3, 2, 1, 1, 1, 1, 1, 1],
            n_rows=64,
            lsb_plane=7,
        )
        self.assertEqual({entry["replica"] for entry in plan}, {0, 1, 2})
        self.assertEqual(len({entry["offset"] for entry in plan}), 4)
        meta = mod.summarize_fault_plan(plan, [3, 2, 1, 1, 1, 1, 1, 1])
        self.assertEqual(meta["replicas_hit_count"], 3)
        self.assertAlmostEqual(meta["replica_loss_correlation"], 1.0)

    def test_lsb_plane_uses_last_available_plane(self):
        mod = load_module()
        scenario = scenario_by_name(mod, "plane_localized_lsb")
        plan = mod.generate_fault_plan(
            "certified_degradation_fallback",
            scenario,
            seed=0,
            r_vector=[1, 1, 1, 1, 1, 1, 1, 1],
            n_rows=64,
            lsb_plane=6,
        )
        self.assertEqual({entry["plane"] for entry in plan}, {6})

    def test_cpu_oracle_same_fault_all_is_bounded_degraded(self):
        mod = load_module()
        clean_planes = [bytes([0] * 16) for _ in range(mod.PLANE_COUNT)]
        plan = [{"plane": 0, "replica": replica, "offset": 3, "mask": 0xFF} for replica in range(3)]
        result = mod.cpu_oracle(clean_planes, [3, 1, 1, 1, 1, 1, 1, 1], plan, 16)
        self.assertEqual(result["outcome"], "bounded_degraded")
        self.assertTrue(result["contains_truth"])

    def test_cpu_oracle_rejects_out_of_range_offset(self):
        mod = load_module()
        clean_planes = [bytes([0] * 16) for _ in range(mod.PLANE_COUNT)]
        bad_plan = [{"plane": 0, "replica": 0, "offset": 16, "mask": 0xFF}]
        with self.assertRaisesRegex(ValueError, "invalid fault plan"):
            mod.cpu_oracle(clean_planes, [3, 1, 1, 1, 1, 1, 1, 1], bad_plan, 16)

    def test_grouped_summary_rows_reports_recovered_and_bounded_rates(self):
        mod = load_module()
        rows = [
            {
                "dataset": "hurricane_u",
                "policy": "logical_striping_diversity_r3",
                "fault_family": "single_domain_burst",
                "cpu_gpu_classification_match": "true",
                "contains_truth": "1.0",
                "verdict_cell": "vote_recovered",
                "same_fault_false_recovery": "false",
                "cert_bound_failure": "false",
                "hard_fail": "false",
            },
            {
                "dataset": "hurricane_u",
                "policy": "logical_striping_diversity_r3",
                "fault_family": "single_domain_burst",
                "cpu_gpu_classification_match": "true",
                "contains_truth": "1.0",
                "verdict_cell": "bounded_degraded",
                "same_fault_false_recovery": "false",
                "cert_bound_failure": "false",
                "hard_fail": "false",
            },
        ]
        summary_rows = mod.grouped_summary_rows(
            rows,
            key_fields=["dataset", "policy", "fault_family"],
        )
        self.assertEqual(len(summary_rows), 1)
        summary = summary_rows[0]
        self.assertAlmostEqual(summary["recovered_rate"], 0.5)
        self.assertAlmostEqual(summary["certified_bounded_rate"], 0.5)
        self.assertAlmostEqual(summary["uncertified_or_unrecoverable_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
