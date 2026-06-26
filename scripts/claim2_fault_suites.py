"""Fault suite definitions for Claim 2 diversity answer-quality closure.

Each suite contains fault families with:
- fault_family: unique string identifier
- description: human-readable explanation
- generator: callable(seed, n_rows, rate) -> list of fault entries
  where rate is a float controlling fault count / window size / burst intensity
- replica_mode: how faults distribute across replicas for each policy
"""

from __future__ import annotations

import random
from typing import Any, Callable

PLANE_WEIGHTS = [1 << (8 * (7 - p)) for p in range(8)]
R = 3  # fixed replica budget for all policies

# ── Helpers ──────────────────────────────────────────────────


def fault_count_from_rate(
    rate: float, n_rows: int, min_count: int = 1, max_count: int | None = None
) -> int:
    """Map a fault rate to an integer fault count for n_rows.

    Linear scaling: count = rate * n_rows, clamped to [min_count, max_count].
    This ensures different rates produce meaningfully different fault counts
    so that "across fault rates" claims are non-vacuous.
    """
    count = max(min_count, int(rate * n_rows))
    if max_count is not None:
        count = min(count, max_count)
    return count


def _rate_seed(rate: float) -> int:
    """Deterministic integer seed component derived from rate.

    Used to seed per-generator RNGs so that even when two rates produce
    the same fault count (e.g. 1 at very low rates), the fault positions
    and masks still differ.
    """
    return int(rate * 1e9 + 0.5)


def contiguous_offsets(rng: random.Random, n_rows: int, length: int) -> list[int]:
    if length <= 0:
        return []
    base = rng.randint(0, max(n_rows - length, 0))
    return [base + delta for delta in range(length)]


def column_offsets(
    rng: random.Random, n_rows: int, stride: int, repeat_count: int
) -> list[int]:
    stride = max(stride, 1)
    if repeat_count <= 0:
        return []
    base = rng.randint(0, min(stride - 1, max(n_rows - 1, 0)))
    offsets: list[int] = []
    for idx in range(repeat_count):
        offset = base + idx * stride
        if offset >= n_rows:
            break
        offsets.append(offset)
    return offsets


def random_offsets(rng: random.Random, n_rows: int, count: int) -> list[int]:
    if count <= 0:
        return []
    return sorted(rng.sample(range(n_rows), min(count, n_rows)))


# ── Replica mode specifiers ──────────────────────────────────
# Each returns list of replica indices (0..R-1) for the fault plan entry.

def same_all() -> list[int]:
    return list(range(R))


def single_random(rng: random.Random) -> list[int]:
    return [rng.randint(0, R - 1)]


# ── Suite A: Null / Sanity Regime ────────────────────────────
# Suite A is a no-fault null control: rate is intentionally ignored.

def gen_null_no_fault(seed: int, n_rows: int, rate: float = 0.0) -> list[dict[str, Any]]:
    return []


def gen_null_no_fault_alt(seed: int, n_rows: int, rate: float = 0.0) -> list[dict[str, Any]]:
    return []


# ── Suite B: Spatial Regime ──────────────────────────────────
# Rate controls burst length, repeat count, or fault count.

def gen_regional_single_domain_burst(seed: int, n_rows: int, rate: float) -> list[dict[str, Any]]:
    rng = random.Random(seed * 1000 + 20 + _rate_seed(rate))
    burst_len = fault_count_from_rate(rate, n_rows, min_count=1, max_count=256)
    offsets = contiguous_offsets(rng, n_rows, burst_len)
    rep = rng.randint(0, R - 1)
    return [{"plane": 0, "replicas": [rep], "offset": o, "mask": rng.randint(1, 255)} for o in offsets]


def gen_column_like_repeated_offset(seed: int, n_rows: int, rate: float) -> list[dict[str, Any]]:
    rng = random.Random(seed * 1000 + 21 + _rate_seed(rate))
    stride = rng.choice([1024, 2048, 4096])
    repeat = fault_count_from_rate(rate, n_rows, min_count=1, max_count=64)
    offsets = column_offsets(rng, n_rows, stride, repeat)
    rep = rng.randint(0, R - 1)
    return [{"plane": 0, "replicas": [rep], "offset": o, "mask": rng.randint(1, 255)} for o in offsets]


def gen_sid_like_hot_domain(seed: int, n_rows: int, rate: float) -> list[dict[str, Any]]:
    rng = random.Random(seed * 1000 + 22 + _rate_seed(rate))
    count = fault_count_from_rate(rate, n_rows, min_count=1, max_count=200)
    offsets = random_offsets(rng, n_rows, count)
    rep = rng.randint(0, R - 1)
    return [{"plane": 0, "replicas": [rep], "offset": o, "mask": rng.randint(1, 255)} for o in offsets]


# ── Suite C: Transient Temporal Regime ───────────────────────
# Rate controls window size or fault count.

def gen_temporal_window_single_stage(seed: int, n_rows: int, rate: float) -> list[dict[str, Any]]:
    rng = random.Random(seed * 1000 + 30 + _rate_seed(rate))
    window_size = fault_count_from_rate(rate, n_rows, min_count=16, max_count=2048)
    window_start = rng.randint(0, max(n_rows - window_size, 0))
    offsets = list(range(window_start, window_start + window_size))
    return [{"plane": 0, "replicas": same_all(), "offset": o,
             "mask": rng.randint(1, 255), "temporal_resolve_prob": 0.7} for o in offsets]


def gen_transient_burst_repeated_read(seed: int, n_rows: int, rate: float) -> list[dict[str, Any]]:
    rng = random.Random(seed * 1000 + 31 + _rate_seed(rate))
    count = fault_count_from_rate(rate, n_rows, min_count=1, max_count=100)
    offsets = random_offsets(rng, n_rows, count)
    return [{"plane": 0, "replicas": same_all(), "offset": o,
             "mask": rng.randint(1, 255), "temporal_resolve_prob": 0.85} for o in offsets]


def gen_staggered_stage_hotspot(seed: int, n_rows: int, rate: float) -> list[dict[str, Any]]:
    rng = random.Random(seed * 1000 + 32 + _rate_seed(rate))
    window_size = fault_count_from_rate(rate, n_rows, min_count=8, max_count=512)
    window_start = rng.randint(0, max(n_rows - window_size, 0))
    offsets = list(range(window_start, window_start + window_size))
    rep_map = [rng.randint(0, R - 1) for _ in range(3)]
    entries = []
    for i, o in enumerate(offsets):
        entries.append({
            "plane": 0,
            "replicas": [rep_map[i % len(rep_map)]],
            "offset": o,
            "mask": rng.randint(1, 255),
            "temporal_resolve_prob": 0.6,
        })
    return entries


# ── Suite D: Hard Negative / Defeat Regime ───────────────────
# Rate controls fault count / burst length while preserving hard-negative nature.

def gen_same_fault_all_replicas(seed: int, n_rows: int, rate: float) -> list[dict[str, Any]]:
    rng = random.Random(seed * 1000 + 40 + _rate_seed(rate))
    count = fault_count_from_rate(rate, n_rows, min_count=1, max_count=200)
    offsets = random_offsets(rng, n_rows, count)
    return [{"plane": 0, "replicas": same_all(), "offset": o, "mask": rng.randint(1, 255)}
            for o in offsets]


def gen_cross_domain_all_stage_burst(seed: int, n_rows: int, rate: float) -> list[dict[str, Any]]:
    rng = random.Random(seed * 1000 + 41 + _rate_seed(rate))
    burst_len = fault_count_from_rate(rate, n_rows, min_count=1, max_count=64)
    offsets = contiguous_offsets(rng, n_rows, burst_len)
    entries = []
    for plane in (0, 1, 7):
        for o in offsets:
            entries.append({"plane": plane, "replicas": same_all(), "offset": o, "mask": rng.randint(1, 255)})
    return entries


# ── Catalogue ────────────────────────────────────────────────

FAULT_SUITES: dict[str, dict[str, Any]] = {
    "suite_a": {
        "label": "null_sanity",
        "expected": "all_tied",
        "families": {
            "null_no_fault": {
                "generator": gen_null_no_fault,
                "description": "Null test: no faults — all policies tied at zero error",
            },
            "null_no_fault_alt": {
                "generator": gen_null_no_fault_alt,
                "description": "Null test: no faults (alternate seed family)",
            },
        },
    },
    "suite_b": {
        "label": "spatial_regime",
        "expected": "spatial_and_spatial_temporal_beat_naive",
        "families": {
            "regional_single_domain_burst": {
                "generator": gen_regional_single_domain_burst,
                "description": "Contiguous burst in one spatial domain",
            },
            "column_like_repeated_offset": {
                "generator": gen_column_like_repeated_offset,
                "description": "Strided repeated offsets (column pattern)",
            },
            "sid_like_hot_domain": {
                "generator": gen_sid_like_hot_domain,
                "description": "Many faults in single hot domain",
            },
        },
    },
    "suite_c": {
        "label": "transient_temporal_regime",
        "expected": "temporal_and_spatial_temporal_help",
        "families": {
            "temporal_window_single_stage": {
                "generator": gen_temporal_window_single_stage,
                "description": "Transient faults within a narrow window",
            },
            "transient_burst_repeated_read": {
                "generator": gen_transient_burst_repeated_read,
                "description": "Burst of transients that resolve on re-read",
            },
            "staggered_stage_hotspot": {
                "generator": gen_staggered_stage_hotspot,
                "description": "Faults migrate across read attempts",
            },
        },
    },
    "suite_d": {
        "label": "hard_negative_defeat",
        "expected": "no_false_credit",
        "families": {
            "same_fault_all_replicas": {
                "generator": gen_same_fault_all_replicas,
                "description": "Identical fault in every replica",
            },
            "cross_domain_all_stage_burst": {
                "generator": gen_cross_domain_all_stage_burst,
                "description": "Burst affecting all replicas across domains",
            },
        },
    },
}


def iter_family_generators() -> list[tuple[str, str, str, Callable]]:
    """Yield (suite_id, family_name, family_desc, generator_fn)."""
    result: list[tuple[str, str, str, Callable]] = []
    for suite_id, suite in FAULT_SUITES.items():
        for fam_name, fam in suite["families"].items():
            result.append((suite_id, fam_name, fam["description"], fam["generator"]))
    return result
