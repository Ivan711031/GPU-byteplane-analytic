#!/usr/bin/env python3
"""Tests for phase2_policy_catalogue.py."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from phase2_policy_catalogue import (
    build_catalogue,
    policy_graded_naive,
    policy_graded_vacuous_aware,
    policy_uniform,
)

SENSOR_VA_RANKING = {
    "5": {"rank": 0, "normalized_damage_median": 76},
    "6": {"rank": 1, "normalized_damage_median": 73},
    "7": {"rank": 2, "normalized_damage_median": 72},
    "2": {"rank": 3, "normalized_damage_median": 66},
    "3": {"rank": 4, "normalized_damage_median": 66},
    "4": {"rank": 5, "normalized_damage_median": 58},
}

SENSOR_NAIVE_RANKING = {
    "0": {"rank": 0, "normalized_damage_median": 114},
    "1": {"rank": 1, "normalized_damage_median": 114},
    "5": {"rank": 2, "normalized_damage_median": 76},
    "6": {"rank": 3, "normalized_damage_median": 73},
    "7": {"rank": 4, "normalized_damage_median": 72},
    "2": {"rank": 5, "normalized_damage_median": 66},
    "3": {"rank": 6, "normalized_damage_median": 66},
    "4": {"rank": 7, "normalized_damage_median": 58},
}

VACUOUS_PLANES = [0, 1]


class TestPolicyUniform(unittest.TestCase):

    def test_B8(self):
        r = policy_uniform(8)
        self.assertEqual(r, [1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(sum(r), 8)

    def test_B12(self):
        r = policy_uniform(12)
        self.assertEqual(r, [2, 2, 2, 2, 1, 1, 1, 1])
        self.assertEqual(sum(r), 12)

    def test_B16(self):
        r = policy_uniform(16)
        self.assertEqual(r, [2, 2, 2, 2, 2, 2, 2, 2])
        self.assertEqual(sum(r), 16)

    def test_B20(self):
        r = policy_uniform(20)
        self.assertEqual(r, [3, 3, 3, 3, 2, 2, 2, 2])
        self.assertEqual(sum(r), 20)

    def test_B24(self):
        r = policy_uniform(24)
        self.assertEqual(r, [3, 3, 3, 3, 3, 3, 3, 3])
        self.assertEqual(sum(r), 24)


class TestPolicyGradedVacuousAware(unittest.TestCase):

    def test_B8(self):
        r = policy_graded_vacuous_aware(8, VACUOUS_PLANES, SENSOR_VA_RANKING)
        expected = [1, 1, 1, 1, 1, 1, 1, 1]
        self.assertEqual(r, expected)
        self.assertEqual(sum(r), 8)

    def test_B12(self):
        r = policy_graded_vacuous_aware(12, VACUOUS_PLANES, SENSOR_VA_RANKING)
        # vacuous [0,1] get 1, B_avail=10, occ=6, base=1, rem=4
        # ranked: 5(r0),6(r1),7(r2),2(r3) get 2; 3(r4),4(r5) get 1
        expected = [1, 1, 2, 1, 1, 2, 2, 2]
        self.assertEqual(r, expected)
        self.assertEqual(sum(r), 12)

    def test_B16(self):
        r = policy_graded_vacuous_aware(16, VACUOUS_PLANES, SENSOR_VA_RANKING)
        # B_avail=14, occ=6, base=2, rem=2 -> top 2 ranked (5,6) get 3
        expected = [1, 1, 2, 2, 2, 3, 3, 2]
        self.assertEqual(r, expected)
        self.assertEqual(sum(r), 16)

    def test_B20(self):
        r = policy_graded_vacuous_aware(20, VACUOUS_PLANES, SENSOR_VA_RANKING)
        # B_avail=18, occ=6, base=3, rem=0 -> all occupied get 3
        expected = [1, 1, 3, 3, 3, 3, 3, 3]
        self.assertEqual(r, expected)
        self.assertEqual(sum(r), 20)

    def test_B24(self):
        r = policy_graded_vacuous_aware(24, VACUOUS_PLANES, SENSOR_VA_RANKING)
        # B_avail=22, occ=6, base=3, rem=4
        # top 4 ranked (5,6,7,2) get 4, bottom 2 (3,4) get 3
        expected = [1, 1, 4, 3, 3, 4, 4, 4]
        self.assertEqual(r, expected)
        self.assertEqual(sum(r), 24)

    def test_vacuous_planes_always_1(self):
        for B in [12, 16, 20, 24]:
            r = policy_graded_vacuous_aware(B, VACUOUS_PLANES, SENSOR_VA_RANKING)
            self.assertEqual(r[0], 1, f"B={B} plane 0")
            self.assertEqual(r[1], 1, f"B={B} plane 1")

    def test_sum_equals_B(self):
        for B in [8, 12, 16, 20, 24]:
            r = policy_graded_vacuous_aware(B, VACUOUS_PLANES, SENSOR_VA_RANKING)
            self.assertEqual(sum(r), B, f"B={B}")


class TestPolicyGradedNaive(unittest.TestCase):

    def test_B8(self):
        r = policy_graded_naive(8, SENSOR_NAIVE_RANKING)
        self.assertEqual(sum(r), 8)

    def test_B12(self):
        r = policy_graded_naive(12, SENSOR_NAIVE_RANKING)
        # all 8 ranked, base=1, rem=4
        # top 4: 0(r0),1(r1),5(r2),6(r3) get 2; rest get 1
        expected = [2, 2, 1, 1, 1, 2, 2, 1]
        self.assertEqual(r, expected)
        self.assertEqual(sum(r), 12)

    def test_B24(self):
        r = policy_graded_naive(24, SENSOR_NAIVE_RANKING)
        # all 8 ranked, base=3, rem=0 -> all get 3
        expected = [3, 3, 3, 3, 3, 3, 3, 3]
        self.assertEqual(r, expected)
        self.assertEqual(sum(r), 24)

    def test_sum_equals_B(self):
        for B in [8, 12, 16, 20, 24]:
            r = policy_graded_naive(B, SENSOR_NAIVE_RANKING)
            self.assertEqual(sum(r), B, f"B={B}")


class TestAllPoliciesCrossCheck(unittest.TestCase):

    def test_all_sum_equals_B(self):
        for B in [8, 12, 16, 20, 24]:
            r_u = policy_uniform(B)
            self.assertEqual(sum(r_u), B, f"uniform B={B}")
            r_va = policy_graded_vacuous_aware(
                B, VACUOUS_PLANES, SENSOR_VA_RANKING,
            )
            self.assertEqual(sum(r_va), B, f"graded_vacuous_aware B={B}")
            r_gn = policy_graded_naive(B, SENSOR_NAIVE_RANKING)
            self.assertEqual(sum(r_gn), B, f"graded_naive B={B}")

    def test_B8_uniform_equals_vacuous_aware(self):
        r_u = policy_uniform(8)
        r_va = policy_graded_vacuous_aware(8, VACUOUS_PLANES, SENSOR_VA_RANKING)
        self.assertEqual(r_u, r_va)

    def test_B24_vacuous_aware_nontrivially_different(self):
        r_u = policy_uniform(24)
        r_va = policy_graded_vacuous_aware(24, VACUOUS_PLANES, SENSOR_VA_RANKING)
        self.assertNotEqual(r_u, r_va)
        self.assertEqual(r_va[0], 1)
        self.assertEqual(r_va[1], 1)
        self.assertIn(4, r_va)
        self.assertIn(3, r_va)


class TestCatalogueBuilding(unittest.TestCase):

    PROFILE_DATA = {
        "metadata": {
            "source": "test",
            "created_at": "2026-06-01",
            "vacuous_threshold": 1e-06,
        },
        "datasets": {
            "sensor": {
                "1e-07": {
                    "naive_ranking": SENSOR_NAIVE_RANKING,
                    "vacuous_aware_ranking": SENSOR_VA_RANKING,
                    "vacuous_planes": VACUOUS_PLANES,
                    "vacuous_excluded": {
                        "0": {
                            "reason": "plane_nonzero_fraction < 1e-6",
                            "plane_nonzero_fraction": 0.0,
                        },
                        "1": {
                            "reason": "plane_nonzero_fraction < 1e-6",
                            "plane_nonzero_fraction": 0.0,
                        },
                    },
                },
                "1e-06": {
                    "naive_ranking": SENSOR_NAIVE_RANKING,
                    "vacuous_aware_ranking": SENSOR_VA_RANKING,
                    "vacuous_planes": VACUOUS_PLANES,
                    "vacuous_excluded": {
                        "0": {
                            "reason": "plane_nonzero_fraction < 1e-6",
                            "plane_nonzero_fraction": 0.0,
                        },
                        "1": {
                            "reason": "plane_nonzero_fraction < 1e-6",
                            "plane_nonzero_fraction": 0.0,
                        },
                    },
                },
                "1e-05": {
                    "naive_ranking": SENSOR_NAIVE_RANKING,
                    "vacuous_aware_ranking": SENSOR_VA_RANKING,
                    "vacuous_planes": VACUOUS_PLANES,
                    "vacuous_excluded": {
                        "0": {
                            "reason": "plane_nonzero_fraction < 1e-6",
                            "plane_nonzero_fraction": 0.0,
                        },
                        "1": {
                            "reason": "plane_nonzero_fraction < 1e-6",
                            "plane_nonzero_fraction": 0.0,
                        },
                    },
                },
            },
        },
    }

    def setUp(self):
        self.profile_path = Path("/tmp/test_policy_catalogue_profile.json")
        self.profile_path.write_text(json.dumps(self.PROFILE_DATA))

    def tearDown(self):
        if self.profile_path.exists():
            self.profile_path.unlink()

    def test_entry_count_with_degeneracy(self):
        cat = build_catalogue(self.profile_path, [8, 12, 16], ["sensor"])
        # 3 fault_rates × (2 + 3 + 3) = 24
        self.assertEqual(len(cat["entries"]), 24)

    def test_all_entries_sum_correct(self):
        cat = build_catalogue(
            self.profile_path, [8, 12, 16, 20, 24], ["sensor"],
        )
        for entry in cat["entries"]:
            B = entry["budget_B"]
            r = entry["r_vector"]
            self.assertEqual(
                sum(r), B,
                f"{entry['policy']} B={B}: sum={sum(r)} != {B}",
            )

    def test_B8_no_graded_vacuous_aware_entry(self):
        cat = build_catalogue(self.profile_path, [8], ["sensor"])
        policies = [e["policy"] for e in cat["entries"] if e["fault_rate"] == "1e-06"]
        self.assertListEqual(policies, ["uniform", "graded_naive"])

    def test_B12_has_graded_vacuous_aware(self):
        cat = build_catalogue(self.profile_path, [12], ["sensor"])
        policies = [e["policy"] for e in cat["entries"] if e["fault_rate"] == "1e-06"]
        self.assertIn("graded_vacuous_aware", policies)

    def test_B8_uniform_note(self):
        cat = build_catalogue(self.profile_path, [8], ["sensor"])
        u_entry = next(e for e in cat["entries"] if e["policy"] == "uniform" and e["fault_rate"] == "1e-06")
        self.assertEqual(
            u_entry["notes"],
            "degeneracy: uniform == graded_vacuous_aware at B=8",
        )

    def test_metadata_fields(self):
        cat = build_catalogue(self.profile_path, [8, 12, 16], ["sensor"])
        meta = cat["metadata"]
        self.assertIn("created_at", meta)
        self.assertIn("source_profile", meta)
        self.assertEqual(meta["budget_points"], [8, 12, 16])
        self.assertIn("uniform", meta["policies"])

    def test_missing_dataset_skipped(self):
        cat = build_catalogue(
            self.profile_path, [8], ["sensor", "nonexistent"],
        )
        datasets_in = {e["dataset"] for e in cat["entries"]}
        self.assertEqual(datasets_in, {"sensor"})

    def test_vacuous_fields_in_entries(self):
        cat = build_catalogue(self.profile_path, [12], ["sensor"])
        for entry in cat["entries"]:
            self.assertEqual(entry["vacuous_planes"], [0, 1])

    def test_graded_naive_occupied_count_8(self):
        cat = build_catalogue(self.profile_path, [12], ["sensor"])
        gn = next(
            e for e in cat["entries"] if e["policy"] == "graded_naive" and e["fault_rate"] == "1e-06"
        )
        self.assertEqual(gn["occupied_count"], 8)
        self.assertEqual(gn["budget_avail"], 12)

    def test_graded_vacuous_aware_occupied_count_6(self):
        cat = build_catalogue(self.profile_path, [12], ["sensor"])
        gva = next(
            e for e in cat["entries"] if e["policy"] == "graded_vacuous_aware" and e["fault_rate"] == "1e-06"
        )
        self.assertEqual(gva["occupied_count"], 6)
        self.assertEqual(gva["budget_avail"], 10)

    def test_uniform_occupied_count_6(self):
        cat = build_catalogue(self.profile_path, [12], ["sensor"])
        u = next(
            e for e in cat["entries"] if e["policy"] == "uniform" and e["fault_rate"] == "1e-06"
        )
        self.assertEqual(u["occupied_count"], 6)
        self.assertEqual(u["budget_avail"], 12)


if __name__ == "__main__":
    unittest.main()
