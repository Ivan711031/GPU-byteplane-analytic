import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_nmr_b_e2e_pipeline.py"
    spec = spec_from_file_location("run_nmr_b_e2e_pipeline", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RunNmrBE2EPipelineTests(unittest.TestCase):
    def test_sum32_is_bytewise(self):
        mod = load_module()
        self.assertEqual(mod.sum32(bytes([1, 2, 3, 4])), 10)

    def test_same_fault_plan_hits_all_plane0_replicas(self):
        mod = load_module()
        plan = mod.generate_fault_plan(
            "same_fault_all_replicas_plane0",
            seed=0,
            r_vector=[3, 1, 1, 1, 1, 1, 1, 1],
            n_rows=16,
        )
        self.assertEqual(len(plan), 3)
        self.assertEqual({entry["plane"] for entry in plan}, {0})
        self.assertEqual({entry["replica"] for entry in plan}, {0, 1, 2})
        self.assertEqual(len({entry["offset"] for entry in plan}), 1)

    def test_cpu_oracle_majority_recovers_single_plane0_fault(self):
        mod = load_module()
        clean_planes = [bytes([0] * 8) for _ in range(mod.PLANE_COUNT)]
        result = mod.cpu_oracle(
            clean_planes,
            [3, 1, 1, 1, 1, 1, 1, 1],
            [{"plane": 0, "replica": 0, "offset": 3, "mask": 0xFF}],
            8,
        )
        self.assertEqual(result["outcome"], "vote_recovered")
        self.assertTrue(result["contains_truth"])

    def test_cpu_oracle_same_fault_all_is_bounded_degraded(self):
        mod = load_module()
        clean_planes = [bytes([0] * 8) for _ in range(mod.PLANE_COUNT)]
        plan = [{"plane": 0, "replica": rep, "offset": 3, "mask": 0xFF} for rep in range(3)]
        result = mod.cpu_oracle(clean_planes, [3, 1, 1, 1, 1, 1, 1, 1], plan, 8)
        self.assertEqual(result["outcome"], "bounded_degraded")
        self.assertTrue(result["contains_truth"])


if __name__ == "__main__":
    unittest.main()
