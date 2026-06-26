#!/usr/bin/env python3
"""Unit tests for phase2_vote.py (Issue #125 P2-3)."""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from phase2_vote import vote_byte, vote_plane


# ── vote_byte: single-byte voting ──────────────────────────────────


def test_vote_byte_rp1() -> None:
    """r_p = 1: single value always wins."""
    assert vote_byte([0x42]) == 0x42
    assert vote_byte([0x00]) == 0x00
    assert vote_byte([0xFF]) == 0xFF


def test_vote_byte_rp2_majority() -> None:
    """r_p = 2: unanimous picks the value."""
    assert vote_byte([0x42, 0x42]) == 0x42
    assert vote_byte([0xAB, 0xAB]) == 0xAB


def test_vote_byte_rp2_tie() -> None:
    """r_p = 2 tie: pick lexicographically smallest."""
    assert vote_byte([0x42, 0xAB]) == 0x42
    assert vote_byte([0xAB, 0x42]) == 0x42
    assert vote_byte([0xFF, 0x00]) == 0x00


def test_vote_byte_rp3_majority() -> None:
    """r_p = 3: clear majority wins."""
    assert vote_byte([0x42, 0x42, 0xAB]) == 0x42
    assert vote_byte([0xAB, 0x42, 0x42]) == 0x42


def test_vote_byte_rp3_no_majority() -> None:
    """r_p = 3: all different, pick smallest."""
    assert vote_byte([0x42, 0xAB, 0xCD]) == 0x42
    assert vote_byte([0xFF, 0x00, 0x80]) == 0x00


def test_vote_byte_rp4_tie() -> None:
    """r_p = 4: 2v2 tie picks smallest among tied."""
    assert vote_byte([0x42, 0x42, 0xAB, 0xAB]) == 0x42
    assert vote_byte([0xAB, 0xAB, 0x42, 0x42]) == 0x42


def test_vote_byte_rp4_no_majority() -> None:
    """r_p = 4: 2-1-1 split, max_count=2 not > 2, tied={2nd place}."""
    # 2 votes for 0x42, 1 for 0xAB, 1 for 0xCD
    # max_count=2, not > 4/2=2, tie among values with count 2 = {0x42}
    assert vote_byte([0x42, 0x42, 0xAB, 0xCD]) == 0x42


def test_vote_byte_rp4_all_unique() -> None:
    """r_p = 4: all 4 different, pick smallest."""
    assert vote_byte([0x42, 0xAB, 0xCD, 0x01]) == 0x01


def test_vote_byte_empty_raises() -> None:
    """Empty list raises ValueError."""
    raised = False
    try:
        vote_byte([])
    except ValueError:
        raised = True
    assert raised


# ── vote_plane: full-plane voting ──────────────────────────────────


def test_vote_plane_rp1() -> None:
    """r_p = 1: voted == replica, outcome = resolved (match) or undetected (mismatch)."""
    clean = b"\x00\x01\x02\x03"
    replica = [b"\x00\x01\xFF\x03"]
    voted, stats = vote_plane(replica, clean)
    assert voted == b"\x00\x01\xFF\x03"
    # Position 2: replica=0xFF != clean=0x02, clean not in replicas → undetected
    assert stats["resolved_correctly"] == 3
    assert stats["detected_mismatch"] == 0
    assert stats["undetected_corruption"] == 1


def test_vote_plane_rp2_all_good() -> None:
    """r_p = 2: both replicas match clean."""
    clean = b"\x00\x01\x02\x03"
    replicas = [b"\x00\x01\x02\x03", b"\x00\x01\x02\x03"]
    voted, stats = vote_plane(replicas, clean)
    assert voted == clean
    assert stats["resolved_correctly"] == 4
    assert stats["detected_mismatch"] == 0
    assert stats["undetected_corruption"] == 0


def test_vote_plane_rp2_one_fault() -> None:
    """r_p = 2: one replica faulted, other matches clean."""
    clean = b"\x00\x01\x02\x03"
    replicas = [b"\xFF\x01\x02\x03", b"\x00\x01\x02\x03"]
    voted, stats = vote_plane(replicas, clean)
    # Position 0: tie 0xFF vs 0x00, min=0x00 → matches clean
    assert voted == clean
    assert stats["resolved_correctly"] == 4
    assert stats["detected_mismatch"] == 0
    assert stats["undetected_corruption"] == 0


def test_vote_plane_detected_mismatch() -> None:
    """detected_mismatch: voted != clean but correct value existed in a replica."""
    clean = b"\x00"
    # replica 0 = 0xFF (faulted), replica 1 = 0x00 (correct), replica 2 = 0xFF (faulted)
    # votes: {0xFF: 2}, max=2 > 3/2=1.5 → winner=0xFF
    # 0xFF != 0x00, and clean=0x00 IS in replicas → detected_mismatch
    replicas = [b"\xFF", b"\x00", b"\xFF"]
    voted, stats = vote_plane(replicas, clean)
    assert voted == b"\xFF"
    assert stats["resolved_correctly"] == 0
    assert stats["detected_mismatch"] == 1
    assert stats["undetected_corruption"] == 0


def test_vote_plane_undetected_corruption() -> None:
    """undetected_corruption: all replicas share same wrong byte."""
    clean = b"\x00"
    replicas = [b"\xFF", b"\xFF", b"\xFF"]
    voted, stats = vote_plane(replicas, clean)
    assert voted == b"\xFF"
    assert stats["resolved_correctly"] == 0
    assert stats["detected_mismatch"] == 0
    assert stats["undetected_corruption"] == 1


def test_vote_plane_mixed_outcomes() -> None:
    """Multiple positions with different outcomes."""
    clean = b"\x00\x01\x02\x03\x04"
    # Position 0: 0xFF, 0x00, 0x00 → min=0x00 = clean → resolved
    # Position 1: 0xFF, 0xFF, 0x01 → 0xFF != 0x01, clean=0x01 IN replicas → detected
    # Position 2: 0xFF, 0xFF, 0xFF → 0xFF != 0x02, clean=0x02 NOT in → undetected
    # Position 3: 0x01, 0x01, 0x01 → 0x01 != 0x03, clean=0x03 NOT in → undetected
    # Position 4: 0x04, 0x04, 0x04 → 0x04 = clean → resolved
    replicas = [b"\xFF\xFF\xFF\x01\x04",
                b"\x00\xFF\xFF\x01\x04",
                b"\x00\x01\xFF\x01\x04"]
    voted, stats = vote_plane(replicas, clean)
    assert voted == b"\x00\xFF\xFF\x01\x04"
    assert stats["resolved_correctly"] == 2
    assert stats["detected_mismatch"] == 1
    assert stats["undetected_corruption"] == 2


def test_vote_plane_length_mismatch() -> None:
    """Replica length mismatch raises ValueError."""
    raised = False
    try:
        vote_plane([b"\x00\x01", b"\x00"], b"\x00\x01")
    except ValueError:
        raised = True
    assert raised


def test_vote_plane_empty_replicas() -> None:
    """Empty replica list raises ValueError."""
    raised = False
    try:
        vote_plane([], b"\x00")
    except ValueError:
        raised = True
    assert raised


# ── Tie-break integration ──────────────────────────────────────────


def test_tie_break_even_split() -> None:
    """Even split across multiple positions picks smallest each time."""
    clean = b"\xFF\x00\xAB"
    # Position 0: [0x42, 0xFF] → tie, min=0x42 → detected (0x42 != 0xFF, clean=0xFF in)
    # Position 1: [0xFF, 0x00] → tie, min=0x00 → resolved (0x00 == clean)
    # Position 2: [0x01, 0xAB] → tie, min=0x01 → detected (0x01 != 0xAB, clean=0xAB in)
    replicas = [b"\x42\xFF\x01", b"\xFF\x00\xAB"]
    voted, stats = vote_plane(replicas, clean)
    assert voted == b"\x42\x00\x01"
    assert stats["resolved_correctly"] == 1
    assert stats["detected_mismatch"] == 2
    assert stats["undetected_corruption"] == 0


# ── Scale test ─────────────────────────────────────────────────────


def test_scale_1e5() -> None:
    """Handle 100K-byte planes efficiently (proves streaming pattern)."""
    n = 100000
    clean = bytes(random.randint(0, 255) for _ in range(n))
    rng = random.Random(42)
    # Create r_p = 3 replicas with independent faults at 1% rate
    replicas: list[bytearray] = [bytearray(clean) for _ in range(3)]
    for r_idx in range(3):
        for i in range(n):
            if rng.random() < 0.01:
                replicas[r_idx][i] = rng.randint(0, 255)
    replica_bytes = [bytes(rb) for rb in replicas]

    voted, stats = vote_plane(replica_bytes, clean)
    assert len(voted) == n
    assert sum(stats.values()) == n
    assert stats["resolved_correctly"] > stats["detected_mismatch"]
    assert stats["undetected_corruption"] < 5  # near zero under independent faults


# ── Statistical test: random independent faults ────────────────────


def test_statistical_independent_faults() -> None:
    """Under random independent faults, resolved_correctly dominates
    and undetected_corruption is near zero for r_p >= 3."""
    n = 50000
    fault_rate = 0.02
    r_p = 3

    clean = bytes(random.randint(0, 255) for _ in range(n))
    rng = random.Random(12345)
    replicas: list[bytearray] = [bytearray(clean) for _ in range(r_p)]

    for r_idx in range(r_p):
        for i in range(n):
            if rng.random() < fault_rate:
                # Fault: flip to a different random byte
                new_byte = rng.randint(0, 255)
                # Ensure it's different from clean
                while new_byte == clean[i]:
                    new_byte = rng.randint(0, 255)
                replicas[r_idx][i] = new_byte

    replica_bytes = [bytes(rb) for rb in replicas]
    voted, stats = vote_plane(replica_bytes, clean)

    total_fault_positions = n - stats["resolved_correctly"]
    resolved_frac = stats["resolved_correctly"] / n

    # With r_p=3 and 2% independent faults, majority voting should
    # recover correctly at nearly all positions
    assert resolved_frac > 0.98, (
        f"resolved_frac={resolved_frac:.4f} too low"
    )
    # undetected_corruption requires all 3 replicas to fault at the
    # same position AND land on the same byte value (rare)
    assert stats["undetected_corruption"] < 5, (
        f"undetected_corruption={stats['undetected_corruption']} "
        f"should be near zero"
    )
    assert stats["detected_mismatch"] + stats["undetected_corruption"] == total_fault_positions


# ── Main entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    test_vote_byte_rp1()
    test_vote_byte_rp2_majority()
    test_vote_byte_rp2_tie()
    test_vote_byte_rp3_majority()
    test_vote_byte_rp3_no_majority()
    test_vote_byte_rp4_tie()
    test_vote_byte_rp4_no_majority()
    test_vote_byte_rp4_all_unique()
    test_vote_byte_empty_raises()
    test_vote_plane_rp1()
    test_vote_plane_rp2_all_good()
    test_vote_plane_rp2_one_fault()
    test_vote_plane_detected_mismatch()
    test_vote_plane_undetected_corruption()
    test_vote_plane_mixed_outcomes()
    test_vote_plane_length_mismatch()
    test_vote_plane_empty_replicas()
    test_tie_break_even_split()
    test_scale_1e5()
    test_statistical_independent_faults()
    print("All tests pass.")
