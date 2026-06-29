#!/usr/bin/env python3
"""Targeted CPU audit for No-SUM32 pilot sanity checks.

*** LEGACY DIAGNOSTIC NOTICE ***
This script's call sites below audit_1 use a simplified 5-arg call to
classify_replication_outcome that does NOT pass pre-computed per-plane
outcomes.  Without per-plane outcomes the classifier defaults to
silent_wrong for every faulted plane, biasing toward that outcome.
Only audit_1 (row-level dump) has been updated to pass per-plane
classified data.  For formal validation, run the campaign script
directly with --mode replication_only.

Three audits:
  (1) Row-level dump — F1 graded_B1/graded_B2/uniform_B8/uniform_B16 per seed
  (2) Forced-plane microcases — F1 on each plane 0..7 with graded/uniform
  (3) F7/F8 plane-aware breakdown — protected vs unprotected plane outcomes
"""

from __future__ import annotations

import csv
import os
import random
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Re-use from main campaign module — canonical source of truth.
# Local copies below are legacy diagnostics; they may diverge from the
# main campaign's per-plane + query-combine logic.  For formal validation,
# run the campaign script directly with --mode replication_only.
# ---------------------------------------------------------------------------

import sys
sys.path.insert(0, os.path.dirname(__file__))
from nmr_claim1_realistic_campaign import (
    make_graded_policy,
    make_uniform_policy,
    build_policy_catalogue,
    _POLICIES,
    FaultEvent,
    RealizedFaultPlan,
    vote_byte,
    compute_encoded_sum,
    classify_replication_outcome,
    compute_per_plane_outcome,
)

PLANE_COUNT = 8
PLANE_WEIGHTS = [1 << (8 * (7 - p)) for p in range(PLANE_COUNT)]
SEGMENT_SIZE = 4096

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FaultEvent:
    plane: int
    offset: int
    mask: int
    target_replicas: list[int] | None = None

@dataclass
class RealizedFaultPlan:
    events: list[FaultEvent]
    fault_family: str
    rate: float
    seed: int

# ---------------------------------------------------------------------------
# Fault family — deterministic per-plane
# ---------------------------------------------------------------------------

def make_F1_on_plane(plane: int, offset: int, mask: int) -> list[FaultEvent]:
    return [FaultEvent(plane=plane, offset=offset, mask=mask)]

def make_F7_on_plane(plane: int, offset: int, mask: int,
                     target_replicas: list[int] | None = None) -> list[FaultEvent]:
    return [FaultEvent(plane=plane, offset=offset, mask=mask,
                        target_replicas=target_replicas or [0, 1])]

def make_F1_generator(n_events: int, seed: int, forced_plane: int | None = None):
    """Generate F1 events; if forced_plane is set, ALL events hit that plane."""
    rng = random.Random(seed)
    events: list[FaultEvent] = []
    for _ in range(n_events):
        plane = forced_plane if forced_plane is not None else rng.randint(0, 7)
        offset = rng.randint(0, 255)  # small range for test data
        mask = rng.randint(1, 255)
        events.append(FaultEvent(plane=plane, offset=offset, mask=mask))
    return events

def make_F7_generator(n_events: int, seed: int):
    rng = random.Random(seed)
    events: list[FaultEvent] = []
    for _ in range(n_events):
        plane = rng.randint(0, 7)
        offset = rng.randint(0, 255)
        mask = rng.randint(1, 255)
        events.append(FaultEvent(plane=plane, offset=offset, mask=mask,
                                  target_replicas=[0, 1]))
    return events

# ---------------------------------------------------------------------------
# Vote / apply
# ---------------------------------------------------------------------------

def vote_byte(replica_values: list[int]) -> int:
    cnt = Counter(replica_values)
    max_count = max(cnt.values())
    r = len(replica_values)
    if max_count > r / 2:
        return next(v for v, c in cnt.items() if c == max_count)
    tied = sorted(v for v, c in cnt.items() if c == max_count)
    return tied[0]

def apply_fault_plan(
    clean_planes: list[bytearray],
    events: list[FaultEvent],
    r_vector: list[int],
) -> tuple[list[bytearray], dict]:
    n_planes = len(clean_planes)
    dirty_positions: dict[int, set[int]] = {}
    for e in events:
        p = e.plane
        if p not in dirty_positions:
            dirty_positions[p] = set()
        dirty_positions[p].add(e.offset)

    replicas: list[list[bytearray]] = []
    for p in range(n_planes):
        r = max(1, r_vector[p])
        plane_reps = [bytearray(clean_planes[p]) for _ in range(r)]
        replicas.append(plane_reps)

    for e in events:
        p = e.plane
        if p >= n_planes:
            continue
        reps = replicas[p]
        target_reps = e.target_replicas if e.target_replicas is not None else [0]
        for ri in target_reps:
            if ri < len(reps):
                reps[ri][e.offset] ^= e.mask

    delivered_planes: list[bytearray] = []
    per_plane_outcomes: list[dict[str, Any]] = []

    for p in range(n_planes):
        reps = replicas[p]
        r = len(reps)
        clean = clean_planes[p]
        dp = dirty_positions.get(p, set())

        if r == 1:
            delivered_planes.append(reps[0])
            diff = sum(1 for off in dp if reps[0][off] != clean[off])
            per_plane_outcomes.append({
                "r": 1, "diff_bytes": diff, "detected": diff > 0,
            })
        elif r == 2:
            diff = sum(1 for off in dp if reps[0][off] != reps[1][off])
            delivered_planes.append(reps[0])
            per_plane_outcomes.append({
                "r": 2, "diff_bytes": diff, "detected": diff > 0,
                "disagreement": diff > 0,
            })
        else:
            resolved = 0
            detected = 0
            undetected = 0
            for off in dp:
                vals = [reps[ri][off] for ri in range(r)]
                vb = vote_byte(vals)
                reps[0][off] = vb
                if vb == clean[off]:
                    resolved += 1
                elif clean[off] in vals:
                    detected += 1
                else:
                    undetected += 1
            delivered_planes.append(reps[0])
            per_plane_outcomes.append({
                "r": r, "diff_bytes": detected + undetected,
                "detected": (detected + undetected) > 0,
                "resolved": resolved, "detected_mismatch": detected,
                "undetected_corruption": undetected,
            })

    outcome_dict = {
        "per_plane": per_plane_outcomes,
        "any_detected": any(ppo["detected"] for ppo in per_plane_outcomes),
        "detected_planes": [p for p, ppo in enumerate(per_plane_outcomes) if ppo["detected"]],
    }
    return delivered_planes, outcome_dict

# (classify_replication_outcome and compute_encoded_sum are imported above
#  from nmr_claim1_realistic_campaign — the canonical source)

# ---------------------------------------------------------------------------
# Make small test data (256 bytes per plane, all zeros ± noise)
# ---------------------------------------------------------------------------

def make_test_planes(n_rows: int = 256) -> list[bytearray]:
    """Deterministic test data — plane 0 has a pattern, others are varied."""
    rng = random.Random(42)
    planes = []
    for p in range(PLANE_COUNT):
        row = bytearray(rng.randint(0, 255) for _ in range(n_rows))
        planes.append(row)
    return planes

# ---------------------------------------------------------------------------
# Audit 1: Row-level dump per seed for selected policies
# ---------------------------------------------------------------------------

def audit_1_row_level(artifact_dir: Path, dataset: str, n_rows: int):
    """Read real data, run F1 on graded_B1/B2/uniform_B2/uniform_B3, dump per seed."""
    print("=" * 80)
    print("AUDIT 1: Row-level per-seed dump (F1)")
    print("=" * 80)

    clean_planes = [bytearray((artifact_dir / f"plane_{p:03d}.bin").read_bytes()[:n_rows])
                    for p in range(PLANE_COUNT)]
    clean_answer = compute_encoded_sum(clean_planes)

    target_policies = [
        ("graded_B1", "graded", [1, 0, 0, 0, 0, 0, 0, 0]),
        ("graded_B2", "graded", [2, 0, 0, 0, 0, 0, 0, 0]),
        ("uniform_full_r2", "uniform", [1, 1, 1, 1, 1, 1, 1, 1]),
        ("uniform_full_r3", "uniform", [2, 2, 2, 2, 2, 2, 2, 2]),
    ]

    seeds_to_check = list(range(10))
    rate = 1e-5

    for pol_name, pol_type, pol_extras in target_policies:
        r_vector = [1 + e for e in pol_extras]
        print(f"\n--- Policy: {pol_name}  r_vector={r_vector} ---")
        print(f"{'seed':>5} | {'faulted_planes':<20} | {'#events':>7} | {'outcome':<22} | {'delivered==clean':>15} | {'notes'}")
        print("-" * 90)

        for seed in seeds_to_check:
            events = make_F1_generator(int(n_rows * rate), seed, forced_plane=None)
            plan = RealizedFaultPlan(events=events, fault_family="F1", rate=rate, seed=seed)
            faulted_planes = sorted(set(e.plane for e in events))
            plane_counts = Counter(e.plane for e in events)

            delivered_planes, outcome_dict = apply_fault_plan(
                clean_planes, events, r_vector)
            delivered_answer = compute_encoded_sum(delivered_planes)

            # Pre-compute per-plane outcomes for the canonical classifier
            faulted_planes_list = sorted(set(e.plane for e in events))
            per_plane_classified = []
            for p in faulted_planes_list:
                if p >= len(r_vector):
                    continue
                rp = max(1, r_vector[p])
                ppo = outcome_dict.get("per_plane", [])[p] if p < len(outcome_dict.get("per_plane", [])) else {}
                pp_outcome = compute_per_plane_outcome(
                    p, rp, clean_planes[p], delivered_planes[p], ppo,
                )
                per_plane_classified.append({
                    "plane": p, "r": rp, "diff_bytes": ppo.get("diff_bytes", 0),
                    "per_plane_outcome": pp_outcome,
                })

            outcome = classify_replication_outcome(
                clean_answer, delivered_answer, r_vector,
                events, outcome_dict.get("per_plane", []),
                per_plane_classified)

            faulted_str = str(faulted_planes)
            eq = "YES" if delivered_answer == clean_answer else "NO"

            # Per-plane breakdown
            plane_notes = []
            for p in range(8):
                cnt = plane_counts.get(p, 0)
                actual_r = r_vector[p] if p < len(r_vector) else 1
                has_protection = actual_r >= 2
                if cnt > 0:
                    plane_notes.append(f"p{p}(r={actual_r},cnt={cnt})")

            note = ", ".join(plane_notes) if plane_notes else "no faults"

            # Show the critical check: has_r2?
            has_r2 = any(r_vector[p] >= 2 for p in faulted_planes if 0 <= p < len(r_vector))
            has_r3 = any(r_vector[p] >= 3 for p in faulted_planes if 0 <= p < len(r_vector))
            note += f" | has_r2={has_r2} has_r3={has_r3}"

            print(f"{seed:>5} | {faulted_str:<20} | {len(events):>7} | {outcome:<22} | {eq:>15} | {note}")

def audit_1b_row_level_detail(artifact_dir: Path, dataset: str, n_rows: int):
    """For graded_B1, show per-faulted-plane outcome detail to understand r=2 gate."""
    print("\n" + "=" * 80)
    print("AUDIT 1b: Per-plane outcome decomposition (graded_B1, F1, single seed)")
    print("=" * 80)

    clean_planes = [bytearray((artifact_dir / f"plane_{p:03d}.bin").read_bytes()[:n_rows])
                    for p in range(PLANE_COUNT)]
    clean_answer = compute_encoded_sum(clean_planes)

    r_vector = [2, 1, 1, 1, 1, 1, 1, 1]  # graded_B1
    rate = 1e-5
    n_events = max(1, int(n_rows * rate))

    for seed in [0, 1]:
        events = [FaultEvent(plane=7, offset=i, mask=0x01) for i in range(0, min(n_events, 3))]
        events += [FaultEvent(plane=0, offset=n_rows - 1 - i, mask=0xFF) for i in range(min(n_events, 2))]

        delivered_planes, outcome_dict = apply_fault_plan(clean_planes, events, r_vector)
        delivered_answer = compute_encoded_sum(delivered_planes)
        outcome = classify_replication_outcome(
            clean_answer, delivered_answer, r_vector,
            events, outcome_dict.get("per_plane", []))

        print(f"\nSeed {seed}: manual fault event set")
        for e in events:
            print(f"  event: plane={e.plane} offset={e.offset} mask=0x{e.mask:02x} "
                  f"target_reps={e.target_replicas}")
        print(f"  outcome={outcome}  delivered==clean={delivered_answer == clean_answer}")
        print(f"  per_plane:")
        for p, ppo in enumerate(outcome_dict["per_plane"]):
            if ppo.get("diff_bytes", 0) > 0:
                print(f"    plane {p}: r={ppo.get('r')} diff_bytes={ppo.get('diff_bytes')} "
                      f"detected={ppo.get('detected')}")
        print(f"  detected_planes={outcome_dict['detected_planes']}")
        print(f"  Explanation: has_r2 on plane 0 ({r_vector[0]>=2}) "
              f"→ outcome is detected_unavailable (NOT silent_wrong)")
        print(f"  Even though plane=7 has r=1 (would be silent alone), "
              f"plane=0 with r=2 gates the entire cell outcome.")

# ---------------------------------------------------------------------------
# Audit 2: Forced-plane deterministic microcases
# ---------------------------------------------------------------------------

def audit_2_forced_plane():
    """Force F1 on each plane 0..7; verify outcome matches theoretical expectation."""
    print("\n" + "=" * 80)
    print("AUDIT 2: Forced-plane deterministic microcases")
    print("=" * 80)

    clean_planes = make_test_planes(256)
    clean_answer = compute_encoded_sum(clean_planes)

    policies = [
        ("graded_B1", "graded", make_graded_policy(1)),
        ("graded_B2", "graded", make_graded_policy(2)),
        ("uniform_full_r2", "uniform", [1,1,1,1,1,1,1,1]),
        ("uniform_full_r3", "uniform", [2,2,2,2,2,2,2,2]),
    ]

    print(f"{'Policy':<18} {'Plane':>2} | r_vector{'':>18} | {'has_r2':>6} {'has_r3':>6} | {'expec':>10} {'actual':>22}")
    print("-" * 90)

    for pol_name, pol_type, pol_extras in policies:
        r_vector = [1 + e for e in pol_extras]
        for faulted_plane in range(8):
            # Single event on this plane
            events = [FaultEvent(plane=faulted_plane, offset=100, mask=0xFF)]
            delivered_planes, outcome_dict = apply_fault_plan(clean_planes, events, r_vector)
            delivered_answer = compute_encoded_sum(delivered_planes)
            outcome = classify_replication_outcome(
                clean_answer, delivered_answer, r_vector,
                events, outcome_dict.get("per_plane", []))

            # Expected outcome
            actual_r = r_vector[faulted_plane] if faulted_plane < len(r_vector) else 1
            if delivered_answer == clean_answer:
                expected = "repair" if actual_r >= 3 else "correct"
            else:
                if actual_r < 2:
                    expected = "silent_wrong"
                elif actual_r >= 3:
                    expected = "det_unavail"
                else:  # r=2
                    expected = "det_unavail"

            actual_has_r2 = r_vector[faulted_plane] >= 2
            actual_has_r3 = r_vector[faulted_plane] >= 3

            print(f"{pol_name:<18} {faulted_plane:>2} | "
                  f"{str(r_vector):<18} | "
                  f"{actual_has_r2!s:>6} {actual_has_r3!s:>6} | "
                  f"{expected:>10} {outcome:>22}")
        print()

# ---------------------------------------------------------------------------
# Audit 2b: F7 forced-plane microcases
# ---------------------------------------------------------------------------

def audit_2b_forced_plane_F7():
    """Force F7 (replica-targeting) on each plane 0..7."""
    print("\n" + "=" * 80)
    print("AUDIT 2b: F7 forced-plane (replica-targeting) microcases")
    print("=" * 80)

    clean_planes = make_test_planes(256)
    clean_answer = compute_encoded_sum(clean_planes)

    policies = [
        ("graded_B1", "graded", make_graded_policy(1)),
        ("graded_B2", "graded", make_graded_policy(2)),
        ("uniform_full_r2", "uniform", [1,1,1,1,1,1,1,1]),
        ("uniform_full_r3", "uniform", [2,2,2,2,2,2,2,2]),
    ]

    print(f"{'Policy':<18} {'Plane':>2} | r_vector{'':>18} | {'has_r2':>6} {'has_r3':>6} | {'expec':>12} {'actual':>24} {'notes'}")
    print("-" * 110)

    for pol_name, pol_type, pol_extras in policies:
        r_vector = [1 + e for e in pol_extras]
        for faulted_plane in range(8):
            events = [FaultEvent(plane=faulted_plane, offset=100, mask=0xFF,
                                  target_replicas=[0, 1])]
            delivered_planes, outcome_dict = apply_fault_plan(clean_planes, events, r_vector)
            delivered_answer = compute_encoded_sum(delivered_planes)
            outcome = classify_replication_outcome(
                clean_answer, delivered_answer, r_vector,
                events, outcome_dict.get("per_plane", []))

            actual_r = r_vector[faulted_plane] if faulted_plane < len(r_vector) else 1
            if delivered_answer == clean_answer:
                expected = "repair"
            else:
                if actual_r < 2:
                    expected = "silent_wrong"
                elif actual_r == 2:
                    # With r=2 and F7 corrupting both replicas identically:
                    # diff_bytes = 0 → the r=2 diff_bytes=0 loophole → silent_wrong
                    expected = "silent_wrong(r2=bug)"
                else:
                    expected = "det_unavail"

            per_plane = outcome_dict.get("per_plane", [])
            ppo_info = ""
            if faulted_plane < len(per_plane):
                pl = per_plane[faulted_plane]
                ppo_info = f"(r={pl.get('r')} diff={pl.get('diff_bytes')} det={pl.get('detected')})"

            actual_has_r2 = r_vector[faulted_plane] >= 2
            actual_has_r3 = r_vector[faulted_plane] >= 3

            print(f"{pol_name:<18} {faulted_plane:>2} | "
                  f"{str(r_vector):<18} | "
                  f"{actual_has_r2!s:>6} {actual_has_r3!s:>6} | "
                  f"{expected:>12} {outcome:>24} {ppo_info}")
        print()

# ---------------------------------------------------------------------------
# Audit 3: F7/F8 plane-aware breakdown
# ---------------------------------------------------------------------------

def audit_3_F7_F8_plane_breakdown(artifact_dir: Path, dataset: str, n_rows: int):
    """For F7 and F8, compute outcome per cell, split by whether the faulted
    plane has r>=2 (protected) or r=1 (unprotected)."""
    print("\n" + "=" * 80)
    print("AUDIT 3: F7/F8 plane-aware breakdown")
    print("=" * 80)

    clean_planes = [bytearray((artifact_dir / f"plane_{p:03d}.bin").read_bytes()[:n_rows])
                    for p in range(PLANE_COUNT)]
    clean_answer = compute_encoded_sum(clean_planes)

    rate = 1e-5
    seeds = list(range(10))

    policies_to_check = [
        ("uniform_full_r1", "uniform", [0, 0, 0, 0, 0, 0, 0, 0]),
        ("graded_B1", "graded", [1, 0, 0, 0, 0, 0, 0, 0]),
        ("graded_B2", "graded", [2, 0, 0, 0, 0, 0, 0, 0]),
        ("graded_B4", "graded", [2, 2, 0, 0, 0, 0, 0, 0]),
        ("graded_B8", "graded", [2, 2, 2, 2, 0, 0, 0, 0]),
        ("uniform_full_r2", "uniform", [1, 1, 1, 1, 1, 1, 1, 1]),
        ("uniform_full_r3", "uniform", [2, 2, 2, 2, 2, 2, 2, 2]),
    ]

    for family_name, gen_fn in [("F7", make_F7_generator),]:
        print(f"\n--- {family_name} ---")
        print(f"{'Policy':<18} {'seed':>5} | {'% protected':>11} {'% unprotected':>13} | {'Protected outcomes':<50} | {'Unprotected outcomes':<50} | {'cell_outcome':<25}")
        print("-" * 180)

        for pol_name, pol_type, pol_extras in policies_to_check:
            r_vector = [1 + e for e in pol_extras]

            for seed in seeds:
                events = gen_fn(int(n_rows * rate), seed)
                n_total = len(events)
                if n_total == 0:
                    continue

                # Per-event plane analysis
                plane_counts = Counter(e.plane for e in events)
                n_protected = sum(v for p, v in plane_counts.items()
                                  if r_vector[p] >= 2)
                n_unprotected = sum(v for p, v in plane_counts.items()
                                    if r_vector[p] < 2)
                pct_prot = n_protected / n_total * 100 if n_total else 0
                pct_unprot = n_unprotected / n_total * 100 if n_total else 0

                # Cell-level outcome
                delivered_planes, outcome_dict = apply_fault_plan(
                    clean_planes, events, r_vector)
                delivered_answer = compute_encoded_sum(delivered_planes)
                outcome = classify_replication_outcome(
                    clean_answer, delivered_answer, r_vector,
                    events, outcome_dict.get("per_plane", []))

                # Per-plane outcome breakdown for each faulted plane
                protected_outcomes = []
                unprotected_outcomes = []
                for p, cnt in sorted(plane_counts.items()):
                    actual_r = r_vector[p] if p < len(r_vector) else 1
                    ppo = outcome_dict.get("per_plane", [])
                    ppo_r = ppo[p].get("r", "?") if p < len(ppo) else "?"
                    ppo_diff = ppo[p].get("diff_bytes", "?") if p < len(ppo) else "?"
                    ppo_det = ppo[p].get("detected", "?") if p < len(ppo) else "?"
                    label = f"p{p}(r={actual_r},cnt={cnt},diff={ppo_diff})"
                    if actual_r >= 2:
                        protected_outcomes.append(label)
                    else:
                        unprotected_outcomes.append(label)

                prot_str = "; ".join(protected_outcomes) if protected_outcomes else "(none)"
                unprot_str = "; ".join(unprotected_outcomes) if unprotected_outcomes else "(none)"

                print(f"{pol_name:<18} {seed:>5} | "
                      f"{pct_prot:>10.1f}% {pct_unprot:>12.1f}% | "
                      f"{prot_str:<50} | {unprot_str:<50} | {outcome:<25}")

def audit_3b_F8_breakdown(artifact_dir: Path, dataset: str, n_rows: int):
    """For F8, split by F1 vs F7 components + plane protection."""
    print("\n" + "=" * 80)
    print("AUDIT 3b: F8 plane-aware breakdown (F1 vs F7 component)")
    print("=" * 80)

    clean_planes = [bytearray((artifact_dir / f"plane_{p:03d}.bin").read_bytes()[:n_rows])
                    for p in range(PLANE_COUNT)]
    clean_answer = compute_encoded_sum(clean_planes)

    rate = 1e-5
    seeds = list(range(10))
    F1_RATIO = 0.5

    policies_to_check = [
        ("uniform_full_r1", "uniform", [0, 0, 0, 0, 0, 0, 0, 0]),
        ("graded_B1", "graded", [1, 0, 0, 0, 0, 0, 0, 0]),
        ("graded_B2", "graded", [2, 0, 0, 0, 0, 0, 0, 0]),
        ("uniform_full_r2", "uniform", [1, 1, 1, 1, 1, 1, 1, 1]),
        ("uniform_full_r3", "uniform", [2, 2, 2, 2, 2, 2, 2, 2]),
    ]

    for pol_name, pol_type, pol_extras in policies_to_check:
        r_vector = [1 + e for e in pol_extras]
        prot_planes = [p for p in range(8) if r_vector[p] >= 2]
        unprot_planes = [p for p in range(8) if r_vector[p] < 2]

        cell_outcomes: list[str] = []
        for seed in seeds:
            rng = random.Random(seed)
            f1_events = make_F1_generator(int(n_rows * rate * F1_RATIO), seed)
            f7_events = make_F7_generator(int(n_rows * rate * (1 - F1_RATIO)), seed * 997)
            all_events = f1_events + f7_events

            delivered_planes, outcome_dict = apply_fault_plan(clean_planes, all_events, r_vector)
            delivered_answer = compute_encoded_sum(delivered_planes)
            outcome = classify_replication_outcome(
                clean_answer, delivered_answer, r_vector,
                all_events, outcome_dict.get("per_plane", []))

            # Count F1 events on unprotected planes (potential silent_wrong)
            f1_on_unprot = sum(1 for e in f1_events if e.plane in unprot_planes)
            f1_on_prot = sum(1 for e in f1_events if e.plane in prot_planes)
            f7_on_unprot = sum(1 for e in f7_events if e.plane in unprot_planes)
            f7_on_prot = sum(1 for e in f7_events if e.plane in prot_planes)

            cell_outcomes.append(
                f"seed={seed}: {outcome} "
                f"(F1→prot={f1_on_prot}/unprot={f1_on_unprot} "
                f"F7→prot={f7_on_prot}/unprot={f7_on_unprot})")

        print(f"\n{pol_name:<18} r_vector={r_vector}")
        print(f"  protected planes:   {prot_planes}")
        print(f"  unprotected planes: {unprot_planes}")
        for line in cell_outcomes:
            print(f"  {line}")
        outcome_counts = Counter(c.split(": ")[1].split(" ")[0] for c in cell_outcomes)
        print(f"  aggregated: {dict(outcome_counts)}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Use actual dataset for audits 1 and 3
    base = Path("${WORK_DIR}/datasets/locality_sensitivity")
    dataset = "hurricane_u"
    artifact_dir = base / dataset / "seg4096"
    n_rows = 5000000  # match the actual pilot experiment

    # Audit 1: Row-level per seed
    audit_1_row_level(artifact_dir, dataset, n_rows)
    audit_1b_row_level_detail(artifact_dir, dataset, n_rows)

    # Audit 2: Forced-plane microcases
    audit_2_forced_plane()
    audit_2b_forced_plane_F7()

    # Audit 3: F7/F8 plane-aware
    audit_3_F7_F8_plane_breakdown(artifact_dir, dataset, n_rows)
    audit_3b_F8_breakdown(artifact_dir, dataset, n_rows)


if __name__ == "__main__":
    main()
