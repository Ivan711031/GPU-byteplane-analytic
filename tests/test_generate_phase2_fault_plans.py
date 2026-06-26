#!/usr/bin/env python3
"""Tests for generate_phase2_fault_plans.py."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from generate_phase2_fault_plans import (
    derive_replica_seed,
    generate_fault_entries,
    make_rate_dir,
    parse_seed_spec,
)


class TestSeedDerivation(unittest.TestCase):
    """Phase 2 AC1: replica seed determinism and independence."""

    def test_deterministic(self):
        s1 = derive_replica_seed(42, 'sensor', 2, '1e-6', 0)
        s2 = derive_replica_seed(42, 'sensor', 2, '1e-6', 0)
        self.assertEqual(s1, s2)

    def test_different_replica_index(self):
        s0 = derive_replica_seed(42, 'sensor', 2, '1e-6', 0)
        s1 = derive_replica_seed(42, 'sensor', 2, '1e-6', 1)
        self.assertNotEqual(s0, s1)

    def test_different_plane(self):
        s0 = derive_replica_seed(42, 'sensor', 2, '1e-6', 0)
        s3 = derive_replica_seed(42, 'sensor', 3, '1e-6', 0)
        self.assertNotEqual(s0, s3)

    def test_different_dataset(self):
        s0 = derive_replica_seed(42, 'sensor', 2, '1e-6', 0)
        s1 = derive_replica_seed(42, 'uniform', 2, '1e-6', 0)
        self.assertNotEqual(s0, s1)

    def test_different_rate(self):
        s0 = derive_replica_seed(42, 'sensor', 2, '1e-6', 0)
        s1 = derive_replica_seed(42, 'sensor', 2, '1e-5', 0)
        self.assertNotEqual(s0, s1)

    def test_different_base_seed(self):
        s0 = derive_replica_seed(42, 'sensor', 2, '1e-6', 0)
        s1 = derive_replica_seed(99, 'sensor', 2, '1e-6', 0)
        self.assertNotEqual(s0, s1)

    def test_replica_seed_in_range(self):
        s = derive_replica_seed(42, 'sensor', 2, '1e-6', 0)
        self.assertGreaterEqual(s, 0)
        self.assertLess(s, 2**64)


class TestFaultModel(unittest.TestCase):
    """Phase 2 AC1: reuses Phase 1 fault model invariants."""

    def test_offset_uniqueness(self):
        entries = generate_fault_entries(100000, 1e-4, 42)
        offsets = [e['offset'] for e in entries]
        self.assertEqual(len(offsets), len(set(offsets)))

    def test_offset_sorted(self):
        entries = generate_fault_entries(100000, 1e-4, 42)
        offsets = [e['offset'] for e in entries]
        self.assertEqual(offsets, sorted(offsets))

    def test_mask_range(self):
        entries = generate_fault_entries(100000, 1e-4, 42)
        for e in entries:
            self.assertGreaterEqual(e['mask'], 1)
            self.assertLessEqual(e['mask'], 255)

    def test_fault_count_matches_floor(self):
        n_rows = 100000
        rate_val = 1e-4
        expected = int(rate_val * n_rows)
        entries = generate_fault_entries(n_rows, rate_val, 42)
        self.assertEqual(len(entries), expected)

    def test_zero_fault_count_returns_empty(self):
        entries = generate_fault_entries(100000, 1e-12, 42)
        self.assertEqual(len(entries), 0)

    def test_deterministic_identical(self):
        a = generate_fault_entries(100000, 1e-4, 42)
        b = generate_fault_entries(100000, 1e-4, 42)
        self.assertEqual(a, b)

    def test_different_seed_different_entries(self):
        a = generate_fault_entries(100000, 1e-4, 42)
        b = generate_fault_entries(100000, 1e-4, 99)
        self.assertNotEqual(a, b)

    def test_large_n_rows_high_rate(self):
        n_rows = 100_000_000
        rate_val = 1e-4
        entries = generate_fault_entries(n_rows, rate_val, 0)
        self.assertEqual(len(entries), 10_000)
        offsets = [e['offset'] for e in entries]
        self.assertEqual(len(offsets), len(set(offsets)))
        self.assertEqual(offsets, sorted(offsets))


class TestCrossReplicaIndependence(unittest.TestCase):
    """Phase 2 AC3: offset sets from different replicas have minimal overlap."""

    def _offset_overlap(self, entries_a, entries_b):
        offsets_a = {e['offset'] for e in entries_a}
        offsets_b = {e['offset'] for e in entries_b}
        return len(offsets_a & offsets_b)

    def test_replica_pair_low_overlap(self):
        n_rows = 1_000_000
        rate_val = 1e-4
        base_seed = 42
        dataset = 'sensor'
        plane = 2
        rate_str = '1e-4'

        entries_0 = generate_fault_entries(
            n_rows, rate_val,
            derive_replica_seed(base_seed, dataset, plane, rate_str, 0)
        )
        entries_1 = generate_fault_entries(
            n_rows, rate_val,
            derive_replica_seed(base_seed, dataset, plane, rate_str, 1)
        )

        overlap = self._offset_overlap(entries_0, entries_1)
        # Expected ~ fault_count^2 / n_rows = (100)^2 / 1e6 = 0.01
        # Allow generous bound: at most 5 overlaps
        self.assertLessEqual(overlap, 5)

    def test_replica_pair_zero_rate_low_overlap(self):
        """At rate=1e-7 (0.1 faults), entries are nearly empty."""
        n_rows = 100_000
        rate_val = 1e-7
        base_seed = 42
        dataset = 'sensor'
        plane = 2
        rate_str = '1e-7'

        entries_0 = generate_fault_entries(
            n_rows, rate_val,
            derive_replica_seed(base_seed, dataset, plane, rate_str, 0)
        )
        entries_1 = generate_fault_entries(
            n_rows, rate_val,
            derive_replica_seed(base_seed, dataset, plane, rate_str, 1)
        )

        overlap = self._offset_overlap(entries_0, entries_1)
        # fault_count = 0 → empty entries, overlap must be 0
        self.assertEqual(overlap, 0)

    def test_three_replicas_all_pair_overlaps_low(self):
        """With r_p = 3, all 3 choose 2 = 3 pairwise overlaps are low."""
        n_rows = 1_000_000
        rate_val = 1e-4
        base_seed = 42
        dataset = 'sensor'
        plane = 2
        rate_str = '1e-4'

        entries = []
        for j in range(3):
            rs = derive_replica_seed(base_seed, dataset, plane, rate_str, j)
            entries.append(generate_fault_entries(n_rows, rate_val, rs))

        for i in range(3):
            for k in range(i + 1, 3):
                overlap = self._offset_overlap(entries[i], entries[k])
                self.assertLessEqual(overlap, 5,
                                     f"Overlap between replica {i} and {k} too high")


class TestFullGeneration(unittest.TestCase):
    """Phase 2 AC2/AC4: end-to-end generation with temp directory."""

    def test_vacuous_skip_no_files(self):
        """AC4: r_p = 1 for all planes → no fault plan files generated."""
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_generation(
                tmp, replicas=[1, 1, 1, 1, 1, 1, 1, 1]
            )
            # Expect no fault_plans_phase2 directory
            phase2_dir = Path(tmp) / 'fault_plans_phase2'
            self.assertFalse(phase2_dir.exists(),
                             msg="No fault plans should exist for r_p=1")

    def test_single_occupied_plane_generates_plans(self):
        """Only plane 2 has r_p=2 → plans generated only for plane 2."""
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_generation(
                tmp, n_rows=1_000_000,
                replicas=[1, 1, 2, 1, 1, 1, 1, 1],
                rates=['1e-4'],
                seeds=['0-1'],
            )
            phase2_dir = Path(tmp) / 'fault_plans_phase2'
            self.assertTrue(phase2_dir.exists())

            # Expect 2 replicas × 1 rate × 2 seeds = 4 files
            files = list(phase2_dir.rglob('*.json'))
            self.assertEqual(len(files), 4)

            # All files should be under plane2
            for f in files:
                self.assertIn('plane2', str(f))

    def test_metadata_fields(self):
        """AC2: metadata includes phase, replica_index, replica_seed, policy_id."""
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_generation(
                tmp, n_rows=1_000_000,
                replicas=[1, 1, 2, 3, 1, 1, 1, 1],
                rates=['1e-4'],
                seeds=['0'],
            )
            files = list((Path(tmp) / 'fault_plans_phase2').rglob('*.json'))
            self.assertGreater(len(files), 0)

            for f in files:
                fp = json.loads(f.read_text())
                meta = fp['metadata']
                self.assertEqual(meta['phase'], 'phase2')
                self.assertIn('replica_index', meta)
                self.assertIn('replica_seed', meta)
                self.assertIn('policy_id', meta)
                self.assertIsInstance(meta['replica_index'], int)
                self.assertIsInstance(meta['replica_seed'], int)
                self.assertGreaterEqual(meta['replica_seed'], 0)

    def test_path_structure(self):
        """AC1: Output files at correct path structure."""
        dataset = 'test_ds'
        n_rows = 1_000_000
        scale = 12345
        policy_id = 'graded_vacuous_aware'

        with tempfile.TemporaryDirectory() as tmp:
            self._run_generation(
                tmp, dataset=dataset, n_rows=n_rows,
                scale=scale, policy_id=policy_id,
                replicas=[1, 2, 1, 1, 1, 1, 1, 1],
                rates=['1e-4'],
                seeds=['0'],
            )
            expected = (
                Path(tmp) / 'fault_plans_phase2'
                / dataset / f'n{n_rows}' / f'scale{scale}'
                / policy_id / 'plane1' / 'rate1e-04'
                / 'replica0' / 'seed_0.json'
            )
            self.assertTrue(expected.exists(),
                            f"Expected path {expected} not found")

    def test_replica_seed_deterministic_across_runs(self):
        """Same parameters produce identical fault plans across runs."""
        with tempfile.TemporaryDirectory() as tmp:
            result1 = self._run_generation(
                tmp, n_rows=1_000_000,
                replicas=[1, 1, 2, 1, 1, 1, 1, 1],
                rates=['1e-4'], seeds=['0'],
            )
            files1 = sorted(
                (Path(tmp) / 'fault_plans_phase2').rglob('*.json')
            )
            content1 = [f.read_text() for f in files1]

        with tempfile.TemporaryDirectory() as tmp:
            result2 = self._run_generation(
                tmp, n_rows=1_000_000,
                replicas=[1, 1, 2, 1, 1, 1, 1, 1],
                rates=['1e-4'], seeds=['0'],
            )
            files2 = sorted(
                (Path(tmp) / 'fault_plans_phase2').rglob('*.json')
            )
            content2 = [f.read_text() for f in files2]

        self.assertEqual(content1, content2)

    def _run_generation(self, tmp_dir, dataset='sensor', n_rows=100000,
                        scale=8233095970213, replicas=None,
                        rates=None, seeds=None, policy_id='placeholder',
                        artifact_json=None):
        """Helper: run main() with given args in temp artifact root."""
        if replicas is None:
            replicas = [1, 1, 1, 1, 1, 1, 1, 1]
        if rates is None:
            rates = ['1e-6']
        if seeds is None:
            seeds = ['0']

        from generate_phase2_fault_plans import main
        # We need to run main with specific args. Since main() uses argparse,
        # we need to patch sys.argv.
        old_argv = sys.argv
        args_list = [
            'generate_phase2_fault_plans.py',
            '--dataset', dataset,
            '--n-rows', str(n_rows),
            '--scale', str(scale),
        ]
        for r in rates:
            args_list.extend(['--rates', r])
        for s in seeds:
            args_list.extend(['--seeds', s])
        args_list.extend(['--replicas'] + [str(r) for r in replicas])
        args_list.extend(['--artifact-root', tmp_dir])
        args_list.extend(['--policy-id', policy_id])
        if artifact_json:
            args_list.extend(['--artifact-json', str(artifact_json)])
        try:
            sys.argv = args_list
            main()
        finally:
            sys.argv = old_argv


class TestUtilities(unittest.TestCase):

    def test_parse_seed_range(self):
        self.assertEqual(parse_seed_spec(['0-4']), [0, 1, 2, 3, 4])

    def test_parse_seed_single(self):
        self.assertEqual(parse_seed_spec(['42']), [42])

    def test_parse_seed_mixed(self):
        self.assertEqual(
            parse_seed_spec(['0-2', '5', '7-8']), [0, 1, 2, 5, 7, 8]
        )

    def test_make_rate_dir(self):
        self.assertEqual(make_rate_dir('1e-6'), 'rate1e-06')
        self.assertEqual(make_rate_dir('1e-7'), 'rate1e-07')
        self.assertEqual(make_rate_dir('1e-5'), 'rate1e-05')


if __name__ == '__main__':
    unittest.main()
