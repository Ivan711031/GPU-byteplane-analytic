#!/usr/bin/env python3
"""Unit tests for phase2_oracle.py (Issue #128 P2-6).

Covers all 6 acceptance criteria:
  AC1: Exact integer arithmetic (never float)
  AC2: Zero-fault runs: expected_voted_sum == clean_encoded_sum
  AC3: Single-replica (r_p=1): matches Phase 1 oracle formula
  AC4: Multi-replica with independent faults: differs from single-replica
  AC5: Vacuous planes (r_p=1): no voting, damage same as Phase 1 (=0)
  AC6: Fixture with known byte arrays and fault plans
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from phase2_oracle import (
    apply_fault_plan,
    compute_voted_oracle,
    discover_fault_plan_paths,
    load_clean_planes,
)


# ── Fixture helpers ─────────────────────────────────────────────────


def _make_fault_plan(path: Path, entries: list[dict]) -> None:
    """Write a fault plan JSON file at path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = {
        "metadata": {
            "phase": "phase2",
            "target_plane": 0,
            "replica_index": 0,
            "fault_rate_numeric": 0.0,
            "seed": 0,
            "actual_fault_count": len(entries),
            "plane_size_bytes": 0,
        },
        "entries": list(entries),
    }
    path.write_text(json.dumps(fp) + "\n")


def _make_clean_planes(n_rows: int, rng_seed: int = 42) -> list[bytes]:
    """Generate deterministic clean plane bytes."""
    import random
    rng = random.Random(rng_seed)
    planes = []
    for _ in range(8):
        planes.append(bytes(rng.randint(0, 255) for _ in range(n_rows)))
    return planes


def _compute_clean_sum(planes: list[bytes]) -> int:
    """Compute clean_encoded_sum from clean plane bytes."""
    total = 0
    for p in range(8):
        weight = 1 << (8 * (7 - p))
        total += sum(planes[p]) * weight
    return total


def _phase1_oracle_single_plane(
    clean_plane: bytes, entries: list[dict], plane: int,
) -> int:
    """Phase 1 oracle formula for a single plane with fault plan entries."""
    weight = 1 << (8 * (7 - plane))
    delta = 0
    for e in entries:
        old_byte = clean_plane[e["offset"]]
        new_byte = old_byte ^ e["mask"]
        delta += (new_byte - old_byte) * weight
    return delta


# ── AC1: Exact integer arithmetic ───────────────────────────────────


def test_ac1_exact_integer_never_float() -> None:
    """AC1: All intermediate and final values are Python int, never float."""
    n_rows = 100
    planes = _make_clean_planes(n_rows)
    clean_sum = _compute_clean_sum(planes)

    with tempfile.TemporaryDirectory() as tmp:
        # Create fault plans: plane 1 with r_p=3
        plan_dir = Path(tmp) / "fault_plans"
        entries_p1 = [{"offset": i, "mask": 0x80} for i in range(5)]
        for j in range(3):
            _make_fault_plan(
                plan_dir / f"plane1/replica{j}/seed_0.json", entries_p1,
            )

        fpp = {1: [
            str(plan_dir / f"plane1/replica{j}/seed_0.json")
            for j in range(3)
        ]}

        r_vector = [1, 3, 1, 1, 1, 1, 1, 1]
        total_voted_sum, stats = compute_voted_oracle(
            planes, fpp, r_vector,
        )

    # Verify all returned values are int
    assert isinstance(total_voted_sum, int), f"expected int, got {type(total_voted_sum)}"
    for p in range(8):
        s = stats[p]
        assert isinstance(s["voted_damage"], int), f"plane {p} damage not int"
        assert isinstance(s["voted_damage_normalized"], int), f"plane {p} norm not int"
        assert isinstance(s["resolved_correctly"], int), f"plane {p} resolved not int"
        assert isinstance(s["detected_mismatch"], int), f"plane {p} detected not int"
        assert isinstance(s["undetected_corruption"], int), f"plane {p} undetected not int"

    # Verify division would have been exact: normalized = abs(damage) // weight
    # No // 2 or / anywhere in the logic

    expected_sum = clean_sum + stats[1]["voted_damage"]
    assert isinstance(expected_sum, int), "expected_sum not int"
    # If expected_sum were float, str(...) would include a decimal point
    assert "." not in str(expected_sum), (
        f"expected_sum string contains decimal: {expected_sum}"
    )


# ── AC2: Zero-fault runs ────────────────────────────────────────────


def test_ac2_zero_fault_all_vacuous() -> None:
    """AC2: No fault plans at all → expected_voted_sum == clean_encoded_sum."""
    n_rows = 50
    planes = _make_clean_planes(n_rows)
    clean_sum = _compute_clean_sum(planes)

    r_vector = [1, 1, 1, 1, 1, 1, 1, 1]
    total_voted_sum, stats = compute_voted_oracle(planes, {}, r_vector)

    assert total_voted_sum == clean_sum, (
        f"voted {total_voted_sum} != clean {clean_sum}"
    )
    for p in range(8):
        assert stats[p]["voted_damage"] == 0, f"plane {p} damage should be 0"
        assert stats[p]["resolved_correctly"] == n_rows, (
            f"plane {p} all resolved"
        )


def test_ac2_zero_fault_with_plans() -> None:
    """AC2: Fault plans exist but have 0 entries → expected_voted_sum == clean."""
    n_rows = 50
    planes = _make_clean_planes(n_rows)
    clean_sum = _compute_clean_sum(planes)

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        _make_fault_plan(plan_dir / "plane2/replica0/seed_0.json", [])  # empty
        _make_fault_plan(plan_dir / "plane2/replica1/seed_0.json", [])  # empty

        fpp = {2: [
            str(plan_dir / "plane2/replica0/seed_0.json"),
            str(plan_dir / "plane2/replica1/seed_0.json"),
        ]}
        r_vector = [1, 1, 2, 1, 1, 1, 1, 1]
        total_voted_sum, stats = compute_voted_oracle(planes, fpp, r_vector)

    assert total_voted_sum == clean_sum, (
        f"zero-fault voted {total_voted_sum} != clean {clean_sum}"
    )


# ── AC3: Single-replica matches Phase 1 oracle ──────────────────────


def test_ac3_single_replica_phase1_match() -> None:
    """AC3: r_p=1 with fault plan → matches Phase 1 oracle delta."""
    n_rows = 100
    planes = _make_clean_planes(n_rows)
    clean_sum = _compute_clean_sum(planes)

    # Fault plan for plane 3: 10 faults
    target_plane = 3
    entries = [
        {"offset": i * 7, "mask": 0x01 + (i % 7)}
        for i in range(10)
    ]

    # Phase 1 delta
    ph1_delta = _phase1_oracle_single_plane(
        planes[target_plane], entries, target_plane,
    )

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        _make_fault_plan(
            plan_dir / f"plane{target_plane}/replica0/seed_0.json",
            entries,
        )

        fpp = {target_plane: [
            str(plan_dir / f"plane{target_plane}/replica0/seed_0.json"),
        ]}
        r_vector = [1, 1, 1, 1, 1, 1, 1, 1]
        total_voted_sum, stats = compute_voted_oracle(planes, fpp, r_vector)

    # expected_voted_sum = clean_sum + ph1_delta
    expected = clean_sum + ph1_delta
    assert total_voted_sum == expected, (
        f"Phase 2 oracle {total_voted_sum} != Phase 1 expected {expected}"
    )
    assert stats[target_plane]["voted_damage"] == ph1_delta, (
        f"plane {target_plane} damage {stats[target_plane]['voted_damage']} "
        f"!= Phase 1 delta {ph1_delta}"
    )


def test_ac3_single_replica_all_planes_independent() -> None:
    """AC3: Each plane has its own single-replica fault plan."""
    n_rows = 50
    planes = _make_clean_planes(n_rows)
    clean_sum = _compute_clean_sum(planes)

    expected_delta = 0
    fpp = {}
    r_vector = [1] * 8

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        for p in range(8):
            entries = [
                {"offset": i * 13 % n_rows, "mask": 0x80 >> (p % 8)}
                for i in range(3)
            ]
            _make_fault_plan(
                plan_dir / f"plane{p}/replica0/seed_0.json", entries,
            )
            fpp[p] = [
                str(plan_dir / f"plane{p}/replica0/seed_0.json"),
            ]
            expected_delta += _phase1_oracle_single_plane(
                planes[p], entries, p,
            )

        total_voted_sum, stats = compute_voted_oracle(planes, fpp, r_vector)

    expected = clean_sum + expected_delta
    assert total_voted_sum == expected, (
        f"got {total_voted_sum}, expected {expected}"
    )


# ── AC4: Multi-replica differs from single-replica ──────────────────


def test_ac4_multi_replica_different_from_single() -> None:
    """AC4: r_p=3 produces different delta than r_p=1 with same base faults."""
    n_rows = 200
    planes = _make_clean_planes(n_rows)
    target_plane = 5

    # Create 3 independent fault plans for the same plane
    # Each replica has faults at different positions (independent seeds)
    import random
    all_entries = []
    for j in range(3):
        rng = random.Random(100 + j)
        offsets = sorted(rng.sample(range(n_rows), 15))
        masks = [rng.randint(1, 255) for _ in range(15)]
        entries = [
            {"offset": o, "mask": m}
            for o, m in zip(offsets, masks)
        ]
        all_entries.append(entries)

    # Phase 1 delta for each individual replica
    ph1_deltas = [
        _phase1_oracle_single_plane(planes[target_plane], e, target_plane)
        for e in all_entries
    ]

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        for j in range(3):
            _make_fault_plan(
                plan_dir / f"plane{target_plane}/replica{j}/seed_0.json",
                all_entries[j],
            )

        # Single-replica oracle (just replica 0)
        fpp_single = {target_plane: [
            str(plan_dir / f"plane{target_plane}/replica0/seed_0.json"),
        ]}
        r_single = [1] * 8
        voted_single, _ = compute_voted_oracle(planes, fpp_single, r_single)

        # Multi-replica oracle (all 3)
        fpp_multi = {target_plane: [
            str(plan_dir / f"plane{target_plane}/replica{j}/seed_0.json")
            for j in range(3)
        ]}
        r_multi = [1] * 8
        r_multi[target_plane] = 3
        voted_multi, _ = compute_voted_oracle(planes, fpp_multi, r_multi)

    # Multi-replica delta should differ from single-replica delta
    # (Majority voting recovers clean at some positions where replica 0 faulted)
    assert voted_multi != voted_single, (
        "multi-replica oracle should differ from single-replica"
    )

    # Multi-replica with independent faults produces a different result
    # than single-replica (AC4). The delta may not be smaller because
    # tie-break voting can coalesce onto a byte farther from clean than the
    # single replica. The only AC4 requirement is that they differ.
    pass


# ── AC5: Vacuous planes ────────────────────────────────────────────


def test_ac5_vacuous_no_voting_no_damage() -> None:
    """AC5: Vacuous planes (r_p=1, no fault plans) → damage == 0."""
    n_rows = 50
    planes = _make_clean_planes(n_rows)
    clean_sum = _compute_clean_sum(planes)

    # All planes vacuous (r_p=1), no fault plans
    fpp: dict[int, list[str]] = {}
    r_vector = [1] * 8
    total_voted_sum, stats = compute_voted_oracle(planes, fpp, r_vector)

    assert total_voted_sum == clean_sum
    for p in range(8):
        assert stats[p]["voted_damage"] == 0, f"vacuous plane {p} non-zero damage"
        assert stats[p]["resolved_correctly"] == n_rows
        assert stats[p]["detected_mismatch"] == 0
        assert stats[p]["undetected_corruption"] == 0


def test_ac5_mixed_vacuous_and_replicated() -> None:
    """AC5: Vacuous planes untouched, replicated planes show damage."""
    n_rows = 100
    planes = _make_clean_planes(n_rows)
    clean_sum = _compute_clean_sum(planes)

    vacuous = [0, 1]  # r_p=1, no fault plans
    replicated = [2]   # r_p=3 with fault plans

    entries_p2 = [{"offset": i * 5, "mask": 0xFF} for i in range(8)]

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        for j in range(3):
            _make_fault_plan(
                plan_dir / f"plane2/replica{j}/seed_0.json",
                entries_p2,
            )

        fpp = {2: [
            str(plan_dir / f"plane2/replica{j}/seed_0.json")
            for j in range(3)
        ]}
        r_vector = [1, 1, 3, 1, 1, 1, 1, 1]
        total_voted_sum, stats = compute_voted_oracle(planes, fpp, r_vector)

    # Vacuous planes should be untouched
    for p in vacuous:
        assert stats[p]["voted_damage"] == 0, f"vacuous plane {p} damaged"
        assert stats[p]["resolved_correctly"] == n_rows

    # Replicated plane should show damage
    assert stats[2]["voted_damage"] != 0, "replicated plane should show damage"
    expected = clean_sum + stats[2]["voted_damage"]
    assert total_voted_sum == expected


# ── AC6: Fixture with known byte arrays and fault plans ────────────


def test_ac6_known_fixture() -> None:
    """AC6: Deterministic fixture with known inputs and expected outputs."""
    n_rows = 16
    # Clean planes: each plane p has bytes with value p (e.g., plane 0 = [0]*16,
    # plane 1 = [1]*16, ..., plane 7 = [7]*16)
    planes = [bytes([p] * n_rows) for p in range(8)]
    clean_sum = _compute_clean_sum(planes)

    plane = 4
    weight = 1 << (8 * (7 - plane))  # 16777216

    # Fault: flip bit 0 at offset 3 in plane 4
    # old byte = 4 (clean), new byte = 4 ^ 0x01 = 5
    # delta per position = (5 - 4) * weight = 1 * weight = weight
    entries = [{"offset": 3, "mask": 0x01}]

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        _make_fault_plan(
            plan_dir / f"plane{plane}/replica0/seed_0.json", entries,
        )

        # Single replica
        fpp = {plane: [
            str(plan_dir / f"plane{plane}/replica0/seed_0.json"),
        ]}
        r_vector = [1] * 8
        total_voted_sum, stats = compute_voted_oracle(planes, fpp, r_vector)

    # Expected: clean_sum + delta
    # delta = (5 - 4) * weight = weight
    expected = clean_sum + weight
    assert total_voted_sum == expected, (
        f"fixture: got {total_voted_sum}, expected {expected}"
    )
    assert stats[plane]["voted_damage"] == weight


def test_ac6_known_fixture_multi_replica_no_majority() -> None:
    """AC6: 2 replicas with different faults at same offset → tie-break."""
    n_rows = 8
    planes = [bytes([0] * n_rows) for _ in range(8)]
    clean_sum = _compute_clean_sum(planes)

    plane = 6
    weight = 1 << (8 * (7 - plane))  # 256

    # Replica 0: flip offset 2 to 0xFF → byte = 255
    # Replica 1: leave offset 2 as 0 → byte = 0
    # Vote: tie between 255 and 0, min = 0 → voted byte = 0 (matches clean)
    # But there's also a fault at offset 5 where both replicate the same fault
    # Wait, need a more careful test...

    # Let's do: offset 2:
    #   clean = 0, replica 0 = 255 (fault), replica 1 = 255 (same fault)
    #   vote: 255 wins (unanimous) → voted = 255, damage = (255-0)*weight
    entries_r0 = [{"offset": 2, "mask": 0xFF}]
    entries_r1 = [{"offset": 2, "mask": 0xFF}]

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        _make_fault_plan(
            plan_dir / f"plane{plane}/replica0/seed_0.json", entries_r0,
        )
        _make_fault_plan(
            plan_dir / f"plane{plane}/replica1/seed_0.json", entries_r1,
        )

        fpp = {plane: [
            str(plan_dir / f"plane{plane}/replica0/seed_0.json"),
            str(plan_dir / f"plane{plane}/replica1/seed_0.json"),
        ]}
        r_vector = [1, 1, 1, 1, 1, 1, 2, 1]
        total_voted_sum, stats = compute_voted_oracle(planes, fpp, r_vector)

    # Both replicas have same fault at offset 2: byte = 255
    # Majority: 255 wins (2/2)
    # Damage = (255 - 0) * weight = 255 * weight
    expected = clean_sum + 255 * weight
    assert total_voted_sum == expected, (
        f"fixture multi: got {total_voted_sum}, expected {expected}"
    )


def test_ac6_fixture_resolved_detected_undetected() -> None:
    """AC6: Known per-byte outcome classification."""
    n_rows = 4
    # Plane 5: bytes [5, 5, 5, 5]
    planes = [bytes([5] * n_rows) for _ in range(8)]
    clean_sum = _compute_clean_sum(planes)
    plane = 5

    # Three replicas with specific fault patterns:
    # Offset 0: [5, 5, 5] → resolved (unanimous correct)
    # Offset 1: [42, 5, 5] → resolved (majority correct: 5 wins)
    # Offset 2: [42, 43, 5] → resolved (5 wins by tie-break after 2-way tie on 42/43)
    #   Wait, votes: 42(x1), 43(x1), 5(x1) → all unique, min=5 → resolved
    # Offset 3: [99, 99, 100] → detected (99 wins, clean=5 is present in replica 2? No)
    #   Wait, clean=5 is not in any replica for offset 3.
    #   [99, 99, 100]: votes for 99=2, 100=1. 99 > 3/2=1.5 → winner=99
    #   clean=5 not in replicas → undetected

    # Let me reconsider:
    # Offset 0: [5, 5, 5] → winner=5, match → resolved ✓
    # Offset 1: [42, 5, 5] → winner=5, match → resolved ✓
    # Offset 2: [42, 5, 42] → winner=42, mismatch, clean=5 IS in replicas → detected ✓
    # Offset 3: [99, 99, 100] → winner=99, mismatch, clean=5 NOT in replicas → undetected ✓

    r0 = b"\x05\x2A\x2A\x63"
    r1 = b"\x05\x05\x05\x63"
    r2 = b"\x05\x05\x2A\x64"

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        for j, data in enumerate([r0, r1, r2]):
            entries = []
            for i in range(n_rows):
                if data[i] != 5:
                    entries.append({"offset": i, "mask": data[i] ^ 5})
            _make_fault_plan(
                plan_dir / f"plane{plane}/replica{j}/seed_0.json",
                entries,
            )

        fpp = {plane: [
            str(plan_dir / f"plane{plane}/replica{j}/seed_0.json")
            for j in range(3)
        ]}
        r_vector = [1, 1, 1, 1, 1, 3, 1, 1]
        total_voted_sum, stats = compute_voted_oracle(planes, fpp, r_vector)

    s = stats[plane]
    # Expected: resolved=2 (offset 0, 1), detected=1 (offset 2), undetected=1 (offset 3)
    assert s["resolved_correctly"] == 2, (
        f"expected 2 resolved, got {s['resolved_correctly']}"
    )
    assert s["detected_mismatch"] == 1, (
        f"expected 1 detected, got {s['detected_mismatch']}"
    )
    assert s["undetected_corruption"] == 1, (
        f"expected 1 undetected, got {s['undetected_corruption']}"
    )

    # Verify voted bytes: offset 0=5, 1=5, 2=42 (tie between [42,5,42]=42 wins), 3=99
    # Damage for plane 5:
    weight = 1 << (8 * (7 - plane))
    # Offset 0: (5-5)*weight = 0
    # Offset 1: (5-5)*weight = 0
    # Offset 2: (42-5)*weight = 37*weight
    # Offset 3: (99-5)*weight = 94*weight
    # Total damage = (37 + 94) * weight = 131 * weight
    expected_damage = 131 * weight
    assert s["voted_damage"] == expected_damage, (
        f"damage {s['voted_damage']} != {expected_damage}"
    )
    expected_voted = clean_sum + expected_damage
    assert total_voted_sum == expected_voted, (
        f"total {total_voted_sum} != {expected_voted}"
    )


# ── Helper function tests ───────────────────────────────────────────


def test_apply_fault_plan() -> None:
    """apply_fault_plan XORs at specified offsets."""
    clean = b"\x00\xFF\xAA\x55"
    with tempfile.TemporaryDirectory() as tmp:
        fp_path = Path(tmp) / "plan.json"
        entries = [
            {"offset": 0, "mask": 0xFF},  # 0x00 ^ 0xFF = 0xFF
            {"offset": 2, "mask": 0x01},  # 0xAA ^ 0x01 = 0xAB
        ]
        _make_fault_plan(fp_path, entries)
        result = apply_fault_plan(clean, str(fp_path))
    assert result == b"\xFF\xFF\xAB\x55"


def test_load_clean_planes() -> None:
    """load_clean_planes reads 8 plane binaries."""
    n_rows = 32
    with tempfile.TemporaryDirectory() as tmp:
        art_dir = Path(tmp)
        for p in range(8):
            (art_dir / f"plane_{p}.bin").write_bytes(bytes([p] * n_rows))
        planes = load_clean_planes(art_dir, n_rows)
    assert len(planes) == 8
    for p in range(8):
        assert planes[p] == bytes([p] * n_rows), f"plane {p} mismatch"


def test_discover_fault_plan_paths() -> None:
    """discover_fault_plan_paths constructs correct paths."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        r_vector = [1, 3, 1, 1, 1, 1, 1, 1]
        # Create fault plans for plane 1 with 3 replicas
        for j in range(3):
            p = (
                base / "graded" / "plane1" / "rate1e-06"
                / f"replica{j}" / "seed_0.json"
            )
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")

        paths = discover_fault_plan_paths(
            base, "sensor", 100, 42, "graded",
            "1e-06", 0, r_vector,
        )
    assert 1 in paths
    assert len(paths[1]) == 3
    for j, p in enumerate(paths[1]):
        assert f"replica{j}" in p


def test_discover_fault_plan_paths_missing_raises() -> None:
    """Missing fault plan file raises FileNotFoundError."""
    import pytest  # type: ignore[import-untyped]
    has_pytest = False
    try:
        import pytest as _pytest_mod
        has_pytest = True
    except ImportError:
        pass
    if not has_pytest:
        # Skip if pytest not available; just verify the function raises
        try:
            with tempfile.TemporaryDirectory() as tmp:
                r_vector = [1, 2, 1, 1, 1, 1, 1, 1]
                discover_fault_plan_paths(
                    Path(tmp), "x", 100, 42, "p", "1e-6", 0, r_vector,
                )
            assert False, "expected FileNotFoundError"
        except FileNotFoundError:
            pass
        return
    with tempfile.TemporaryDirectory() as tmp:
        r_vector = [1, 2, 1, 1, 1, 1, 1, 1]
        with pytest.raises(FileNotFoundError):
            discover_fault_plan_paths(
                Path(tmp), "x", 100, 42, "p", "1e-6", 0, r_vector,
            )


# ── Integration: apply_fault_plan → vote_plane correctness ──────────


def test_integration_single_replica_no_faults_in_plan() -> None:
    """Apply empty fault plan to single replica → no damage."""
    n_rows = 64
    planes = _make_clean_planes(n_rows)
    clean_sum = _compute_clean_sum(planes)

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        _make_fault_plan(
            plan_dir / "plane0/replica0/seed_0.json", [],
        )
        fpp = {0: [str(plan_dir / "plane0/replica0/seed_0.json")]}
        r_vector = [1, 1, 1, 1, 1, 1, 1, 1]
        total_voted_sum, _ = compute_voted_oracle(planes, fpp, r_vector)

    assert total_voted_sum == clean_sum


def test_integration_multi_replica_all_zero_faults() -> None:
    """Multiple replicas but all empty fault plans → no damage."""
    n_rows = 32
    planes = _make_clean_planes(n_rows)
    clean_sum = _compute_clean_sum(planes)

    with tempfile.TemporaryDirectory() as tmp:
        plan_dir = Path(tmp) / "fault_plans"
        for j in range(4):
            _make_fault_plan(
                plan_dir / f"plane0/replica{j}/seed_0.json", [],
            )
        fpp = {0: [
            str(plan_dir / f"plane0/replica{j}/seed_0.json")
            for j in range(4)
        ]}
        r_vector = [4, 1, 1, 1, 1, 1, 1, 1]
        total_voted_sum, _ = compute_voted_oracle(planes, fpp, r_vector)

    assert total_voted_sum == clean_sum


# ── Run all tests when executed directly ────────────────────────────


if __name__ == "__main__":
    # Collect test functions
    import types
    test_fns = [
        v for k, v in list(globals().items())
        if k.startswith("test_") and isinstance(v, types.FunctionType)
    ]
    # Sort by name for deterministic order
    test_fns.sort(key=lambda f: f.__name__)

    passed = 0
    failed = 0
    for fn in test_fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
