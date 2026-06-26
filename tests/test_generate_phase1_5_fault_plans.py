#!/usr/bin/env python3
"""Tests for generate_phase1_5_fault_plans.py."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from generate_phase1_5_fault_plans import (
    generate_fault_entries,
    make_scale_dir,
    make_rate_dir,
    parse_seed_spec,
    read_skip_csv,
)


class TestFaultPlanGenerator(unittest.TestCase):

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

    def test_strategy_id_round_trip(self):
        """Full plan JSON round-trip preserves strategy_id."""
        n_rows = 100000
        rate_str = '1e-4'
        rate_val = float(rate_str)
        seed = 42
        strategy_id = 'test_strategy'
        plane = 3
        dataset = 'test_dataset'
        scale_dir = make_scale_dir(100, 0)
        rate_dir = make_rate_dir(rate_str)

        art_rel = (
            f'artifacts_phase1_5/{strategy_id}/{dataset}'
            f'/n{n_rows}/{scale_dir}'
        )
        rel = (
            f'fault_plans_phase1_5/{strategy_id}/{dataset}'
            f'/n{n_rows}/{scale_dir}'
            f'/plane{plane}/{rate_dir}/seed_{seed}.json'
        )

        entries = generate_fault_entries(n_rows, rate_val, seed)
        fp = {
            'metadata': {
                'dataset': dataset,
                'n_rows': n_rows,
                'scale': 100,
                'artifact_id': art_rel,
                'fault_plan_id': rel,
                'target_plane': plane,
                'strategy_id': strategy_id,
                'fault_rate': rate_str,
                'fault_rate_numeric': rate_val,
                'seed': seed,
                'actual_fault_count': len(entries),
                'plane_size_bytes': n_rows,
                'mask_distribution': '[1, 255] uniform',
                'offset_order': 'sorted',
                'offset_uniqueness': 'enforced',
            },
            'entries': entries,
        }

        serialized = json.dumps(fp, indent=2)
        deserialized = json.loads(serialized)

        self.assertEqual(
            deserialized['metadata']['strategy_id'], strategy_id
        )
        self.assertEqual(len(deserialized['entries']), len(entries))

    def test_cross_strategy_identical_offsets(self):
        """Same (n_rows, rate, seed) yields identical entries regardless of strategy."""
        n_rows = 100000
        rate_val = 1e-4
        seed = 42
        entries_a = generate_fault_entries(n_rows, rate_val, seed)
        entries_b = generate_fault_entries(n_rows, rate_val, seed)
        self.assertEqual(entries_a, entries_b)

        entries_c = generate_fault_entries(n_rows, rate_val, 99)
        self.assertNotEqual(entries_a, entries_c)

    def test_fault_count_matches_floor(self):
        n_rows = 100000
        rate_val = 1e-4
        expected = int(rate_val * n_rows)
        entries = generate_fault_entries(n_rows, rate_val, 42)
        self.assertEqual(len(entries), expected)

    def test_zero_fault_count_returns_empty(self):
        entries = generate_fault_entries(100000, 1e-12, 42)
        self.assertEqual(len(entries), 0)

    def test_inapplicable_cell_skipping(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                         delete=False) as f:
            f.write('strategy_id,dataset\n')
            f.write('strategy_a,sensor\n')
            f.write('strategy_a,uniform\n')
            f.write('strategy_b,heavy_tailed\n')
            tmp = f.name
        try:
            skipped = read_skip_csv(tmp)
            self.assertIn(('strategy_a', 'sensor'), skipped)
            self.assertIn(('strategy_a', 'uniform'), skipped)
            self.assertIn(('strategy_b', 'heavy_tailed'), skipped)
            self.assertNotIn(('strategy_a', 'heavy_tailed'), skipped)
        finally:
            os.unlink(tmp)

    def test_parse_seed_range(self):
        self.assertEqual(parse_seed_spec(['0-4']), [0, 1, 2, 3, 4])

    def test_parse_seed_single(self):
        self.assertEqual(parse_seed_spec(['42']), [42])

    def test_parse_seed_mixed(self):
        self.assertEqual(
            parse_seed_spec(['0-2', '5', '7-8']), [0, 1, 2, 5, 7, 8]
        )

    def test_make_scale_dir_no_shift(self):
        self.assertEqual(make_scale_dir(100, 0), 'scale100')

    def test_make_scale_dir_with_shift(self):
        self.assertEqual(make_scale_dir(100, 51), 'scale100_shift51')

    def test_make_rate_dir(self):
        self.assertEqual(make_rate_dir('1e-6'), 'rate1e-06')

    def test_large_n_rows_high_rate(self):
        """Stress test: N=1e8, rate=1e-4 => 10000 entries."""
        n_rows = 100_000_000
        rate_val = 1e-4
        entries = generate_fault_entries(n_rows, rate_val, 0)
        self.assertEqual(len(entries), 10_000)
        offsets = [e['offset'] for e in entries]
        self.assertEqual(len(offsets), len(set(offsets)))
        self.assertEqual(offsets, sorted(offsets))


if __name__ == '__main__':
    unittest.main()
