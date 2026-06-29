#!/usr/bin/env python3
"""NMR-D Claim-1 Closure: graded vs uniform repair accuracy evaluator.

CPU-based evaluation for the logical independent-fault model.
Dataset access required (H200 node typical).

Pipeline:
  load planes -> canonical fault stream -> replicate per r_vector ->
  detect/vote -> segment classification (severity-aware) -> metrics ->
  uniform family reduction (fixed-best + oracle-best) -> paired deltas + CI95

Usage:
  # smoke:  small rows, 1 dataset, 5 seeds
  python3 scripts/phase4_nmr_d_claim1_evaluator.py \\
    --mode full --dataset cesm_atm_cloud --n-rows 500000 --seeds 5

  # production: both primary fields, 30 seeds
  python3 scripts/phase4_nmr_d_claim1_evaluator.py \\
    --mode full --seeds 30 \\
    --dataset-dir ${WORK_DIR}/datasets/locality_sensitivity

  # custom dataset path
  python3 scripts/phase4_nmr_d_claim1_evaluator.py \\
    --mode stochastic --dataset cesm_atm_cloud --seeds 30 \\
    --dataset-dir /custom/path
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shlex
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

PLANE_COUNT = 8
PLANE_WEIGHTS = [1 << (8 * (7 - p)) for p in range(PLANE_COUNT)]
SEGMENT_SIZE = 1024

# Budget model: 8 baseline + 3 extra = 11 total replicas.
BASE_PLANE_COPIES = 1
EXTRA_REPLICA_BUDGET = 3
TOTAL_REPLICAS = (BASE_PLANE_COPIES * PLANE_COUNT) + EXTRA_REPLICA_BUDGET  # 11

FAULT_PATTERNS_DETERMINISTIC = [
    "single_replica_plane0",
    "single_replica_plane1",
    "single_replica_random_plane",
    "structured_single_domain_burst",
    "same_fault_all_replicas_plane0",
]
FAULT_RATES = ["1e-07", "1e-06", "1e-05"]

# Graded B=3: 3 extra replicas → 3 on MSB, 2 on plane 1, 1 each on planes 2-7.
GRADED_B3 = "graded"
GRADED_R_VECTOR = [3, 2, 1, 1, 1, 1, 1, 1]  # sum = 11

# Uniform repair family (significance-agnostic, total replicas = 11 each).
UNIFORM_REPAIR_FAMILY: dict[str, list[int]] = {
    "u_p0_concentrated": [4, 1, 1, 1, 1, 1, 1, 1],
    "u_p0p7_stacked":    [3, 1, 1, 1, 1, 1, 1, 2],
    "u_p0p1p2_stacked":  [2, 2, 2, 1, 1, 1, 1, 1],
    "u_spread3":         [2, 1, 2, 1, 2, 1, 1, 1],
    "u_spread4":         [2, 1, 2, 1, 1, 2, 1, 1],
    "u_p7_concentrated": [1, 1, 1, 1, 1, 1, 1, 4],
}


# ── Helpers ──

def load_planes(artifact_dir: Path, n_rows: int) -> list[bytes]:
    planes: list[bytes] = []
    for p in range(PLANE_COUNT):
        path = artifact_dir / f"plane_{p:03d}.bin"
        if not path.is_file():
            path = artifact_dir / f"plane_{p}.bin"
        if path.is_file():
            data = path.read_bytes()[:n_rows]
        else:
            data = bytes(n_rows)
        if len(data) < n_rows:
            data = data + bytes(n_rows - len(data))
        planes.append(data)
    return planes


def compute_clean_sum(planes: list[bytes]) -> int:
    return sum(sum(p) * PLANE_WEIGHTS[pi] for pi, p in enumerate(planes))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Voting ──

def vote_byte_majority(values: list[int]) -> tuple[int | None, bool]:
    """Return (voted_byte, had_majority).

    For r < 3 returns (None, False) — no majority possible.
    For r >= 3: majority exists if a value appears > r/2 times.
    Tie-break: when no strict majority, pick smallest byte.
    """
    r = len(values)
    if r < 3:
        return (None, False)
    cnt = Counter(values)
    max_count = max(cnt.values())
    if max_count > r / 2:
        return (next(v for v, c in cnt.items() if c == max_count), True)
    tied = sorted(v for v, c in cnt.items() if c == max_count)
    return (tied[0], False)


# ── Fault generation ──

def deterministic_fault_events(mode: str, rng: random.Random,
                                n_rows: int) -> list[list[tuple[int, int, int]]]:
    """Return list per plane of [(replica_idx, offset, mask), ...]."""
    mut: list[list[tuple[int, int, int]]] = [[] for _ in range(PLANE_COUNT)]

    if mode == "single_replica_plane0":
        offset = rng.randint(0, n_rows - 1)
        mut[0].append((0, offset, 0xFF))
    elif mode == "single_replica_plane1":
        offset = rng.randint(0, n_rows - 1)
        mut[1].append((0, offset, 0xFF))
    elif mode == "single_replica_random_plane":
        p = rng.randint(0, PLANE_COUNT - 1)
        offset = rng.randint(0, n_rows - 1)
        mut[p].append((0, offset, 0xFF))
    elif mode == "structured_single_domain_burst":
        start = rng.randint(0, n_rows - 16)
        for j in range(8):
            mut[0].append((0, start + j, 0xFF))
    elif mode == "same_fault_all_replicas_plane0":
        offset = rng.randint(0, n_rows - 1)
        mut[0].append((-1, offset, 0xFF))
    return mut


def generate_canonical_fault_stream(
    rng: random.Random, n_rows: int, fault_rate: float
) -> list[list[tuple[int, int]]]:
    """Canonical fault stream per plane: [(offset, mask), ...].

    Policy-independent — same stream for ALL policies at a given seed.
    Number of faults per plane = max(1, int(fault_rate * n_rows)).
    """
    stream: list[list[tuple[int, int]]] = [[] for _ in range(PLANE_COUNT)]
    count = max(1, int(fault_rate * n_rows))
    for p in range(PLANE_COUNT):
        offsets = rng.sample(range(n_rows), count)
        masks = [rng.randint(1, 255) for _ in range(count)]
        for o, m in zip(offsets, masks):
            stream[p].append((o, m))
    return stream


def project_canonical_stream(
    stream: list[list[tuple[int, int]]],
    r_vector: list[int],
) -> list[list[tuple[int, int, int]]]:
    """Project canonical fault stream onto a policy's r_vector.

    Each canonical event (offset, mask) is assigned to one replica
    via round-robin: event_idx % r_p.
    This preserves determinism: same stream → same per-replica faults
    for any policy with the same r_vector.
    """
    mutations: list[list[tuple[int, int, int]]] = [[] for _ in range(PLANE_COUNT)]
    for p in range(PLANE_COUNT):
        r_p = r_vector[p]
        if r_p == 0:
            continue
        for event_idx, (offset, mask) in enumerate(stream[p]):
            rep = event_idx % r_p
            mutations[p].append((rep, offset, mask))
    return mutations


# ── Apply faults and vote ──

def apply_and_vote_sparse(
    clean_planes: list[bytes],
    r_vector: list[int],
    mutations: list[list[tuple[int, int, int]]],
) -> dict[int, dict[int, int]]:
    """Sparse vote: only compute voted bytes for rows that have faults.

    Returns {plane: {row_index: voted_byte}} with only affected rows.
    For r_p < 3: returns empty dict (no majority voting, detect-only).
    """
    result: dict[int, dict[int, int]] = {}
    for p in range(PLANE_COUNT):
        r_p = r_vector[p]
        if r_p < 3:
            continue
        affected_rows: set[int] = set()
        for rep_idx, offset, mask in mutations[p]:
            affected_rows.add(offset)
        if not affected_rows:
            continue
        replicas = [bytearray(clean_planes[p]) for _ in range(r_p)]
        for rep_idx, offset, mask in mutations[p]:
            if rep_idx < r_p:
                replicas[rep_idx][offset] ^= mask
        voted: dict[int, int] = {}
        for i in affected_rows:
            vals = [replicas[rep][i] for rep in range(r_p)]
            vb, had_majority = vote_byte_majority(vals)
            if had_majority:
                voted[i] = vb
        result[p] = voted
    return result


SEVERITY_ORDER = {"degraded": 4, "unprotected": 3, "repaired": 2, "clean": 1}


def classify_segments_sparse(
    clean_planes: list[bytes],
    voted_sparse: dict[int, dict[int, int]],
    mutations: list[list[tuple[int, int, int]]],
    r_vector: list[int],
    segment_size: int = SEGMENT_SIZE,
) -> dict[tuple[int, int], str]:
    """Per (segment_idx, plane) outcome with severity aggregation.

    Rules:
    - r_p == 0: no replica → clean (shouldn't happen with faults).
    - r_p == 1: unprotected (one replica, no detection or repair).
    - r_p == 2: detect-only. A fault is detected but no majority
      repair is possible.  Outcome = unprotected (no repair).
    - r_p >= 3: majority voting.  If voted != clean → degraded.
      If voted == clean (or no majority) → degraded for that offset.
      But per-segment: if ANY byte in the segment is degraded,
      the whole segment is degraded (severity dominates).
      Otherwise repaired.

    When MULTIPLE faults hit the same (segment, plane), the
    most severe outcome wins (degraded > unprotected > repaired).
    """
    SEVERITY = {"degraded": 4, "unprotected": 3, "repaired": 2, "clean": 1}
    outcome: dict[tuple[int, int], str] = {}

    for p in range(PLANE_COUNT):
        r_p = r_vector[p]
        for rep_idx, offset, mask in mutations[p]:
            seg_idx = offset // segment_size
            key = (seg_idx, p)
            if r_p == 1:
                cand = "unprotected"
            elif r_p == 2:
                cand = "unprotected"
            else:
                voted_rows = voted_sparse.get(p, {})
                if offset in voted_rows:
                    if voted_rows[offset] == clean_planes[p][offset]:
                        cand = "repaired"
                    else:
                        cand = "degraded"
                else:
                    cand = "degraded"
            existing = outcome.get(key, "clean")
            if SEVERITY.get(cand, 0) > SEVERITY.get(existing, 0):
                outcome[key] = cand
    return outcome


# ── Metrics ──

@dataclass
class CellMetrics:
    dataset: str = ""
    n_rows: int = 0
    scale: int = 1
    policy: str = ""
    allocation_r: str = ""
    fault_pattern: str = ""
    fault_rate: str = ""
    seed: int = 0
    k: int = 2

    activated_fault: int = 0
    repair_invoked: int = 0
    repair_success: int = 0
    fallback_used: int = 0
    contains_truth: int = 0
    certified: int = 0
    silent_wrong: int = 0
    total_segments: int = 0
    segments_clean: int = 0
    segments_repair_invoked: int = 0
    segments_repair_success: int = 0
    segments_degraded: int = 0
    segments_unprotected: int = 0

    err_fault_repaired_only: float = 0.0
    err_fault_conditioned_user_observed: float = 0.0
    err_fault_user_observed: float = 0.0
    fallback_rate: float = 0.0
    silent_wrong_rate: float = 0.0
    certified_rate: float = 0.0
    contains_truth_rate: float = 0.0

    clean_answer: float = 0.0
    delivered_answer: float = 0.0
    bound_width: float = 0.0
    bound_width_inflation: float = 0.0


def compute_cell_metrics(
    clean_planes: list[bytes],
    voted_sparse: dict[int, dict[int, int]],
    outcomes: dict[tuple[int, int], str],
    mutations: list[list[tuple[int, int, int]]],
    r_vector: list[int],
    dataset: str, n_rows: int, scale: int,
    policy: str, fault_pattern: str, fault_rate: str, seed: int,
    kw_clean_sum: int = 0,
) -> CellMetrics:
    m = CellMetrics(
        dataset=dataset, n_rows=n_rows, scale=scale,
        policy=policy, allocation_r="|".join(str(r) for r in r_vector),
        fault_pattern=fault_pattern, fault_rate=fault_rate, seed=seed,
    )
    n_segments = (n_rows + SEGMENT_SIZE - 1) // SEGMENT_SIZE
    m.total_segments = n_segments

    clean_sum = kw_clean_sum if kw_clean_sum else compute_clean_sum(clean_planes)
    clean_answer = float(clean_sum) / scale
    m.clean_answer = clean_answer

    delivered_sum = clean_sum
    done_planes: set[int] = set()
    for p, voted_rows in voted_sparse.items():
        done_planes.add(p)
        for i, vb in voted_rows.items():
            delivered_sum += (int(vb) - int(clean_planes[p][i])) * PLANE_WEIGHTS[p]
    for p in range(PLANE_COUNT):
        if p in done_planes:
            continue
        for rep_idx, offset, mask in mutations[p]:
            fb = int(clean_planes[p][offset]) ^ mask
            delivered_sum += (fb - int(clean_planes[p][offset])) * PLANE_WEIGHTS[p]
    delivered_answer = float(delivered_sum) / scale
    m.delivered_answer = delivered_answer

    abs_err = abs(delivered_answer - clean_answer)
    m.err_fault_user_observed = abs_err / abs(clean_answer) if clean_answer != 0.0 else abs_err

    activated_set: set[int] = set()
    for p in range(PLANE_COUNT):
        for rep_idx, offset, mask in mutations[p]:
            activated_set.add(offset // SEGMENT_SIZE)

    for (seg_idx, p), status in outcomes.items():
        if status == "clean":
            m.segments_clean += 1
        elif status == "repaired":
            m.segments_repair_invoked += 1
            m.segments_repair_success += 1
        elif status == "degraded":
            m.segments_degraded += 1
            m.segments_repair_invoked += 1
        elif status == "unprotected":
            m.segments_unprotected += 1

    m.activated_fault = len(activated_set)
    m.repair_invoked = m.segments_repair_invoked
    m.repair_success = m.segments_repair_success
    m.fallback_used = 1 if m.segments_unprotected > 0 and m.activated_fault > 0 else 0
    m.fallback_rate = m.segments_unprotected / max(m.total_segments, 1)

    m.silent_wrong = m.segments_degraded
    m.silent_wrong_rate = m.segments_degraded / max(m.total_segments, 1)

    m.certified = m.total_segments - m.segments_degraded
    m.certified_rate = m.certified / max(m.total_segments, 1)

    m.contains_truth = m.certified
    m.contains_truth_rate = m.contains_truth / max(m.total_segments, 1)

    repaired_error = 0.0
    cond_error = 0.0
    cond_count = 0
    bound_widen = 0.0
    faulted_bytes: dict[tuple[int, int], int] = {}
    for p in range(PLANE_COUNT):
        for rep_idx, offset, mask in mutations[p]:
            faulted_bytes[(offset, p)] = int(clean_planes[p][offset]) ^ mask

    for (seg_idx, p), status in outcomes.items():
        if status == "clean":
            continue
        start = seg_idx * SEGMENT_SIZE
        end = min(start + SEGMENT_SIZE, n_rows)
        count = end - start
        if status in ("degraded", "unprotected"):
            bound_widen += count * 255.0 * PLANE_WEIGHTS[p] / scale
        if status in ("repaired", "degraded"):
            for i in range(start, end):
                vb = voted_sparse.get(p, {}).get(i, clean_planes[p][i])
                diff = abs(int(vb) - int(clean_planes[p][i]))
                repaired_error += diff * PLANE_WEIGHTS[p]
        if status in ("degraded", "unprotected", "repaired"):
            for i in range(start, end):
                if status == "unprotected":
                    fb = faulted_bytes.get((i, p), int(clean_planes[p][i]))
                    diff = abs(fb - int(clean_planes[p][i]))
                else:
                    vb = voted_sparse.get(p, {}).get(i, clean_planes[p][i])
                    diff = abs(int(vb) - int(clean_planes[p][i]))
                cond_error += diff * PLANE_WEIGHTS[p]
                cond_count += 1

    if m.segments_repair_invoked > 0:
        m.err_fault_repaired_only = (repaired_error / scale) / abs(clean_answer) if clean_answer != 0.0 else repaired_error / scale
    m.err_fault_conditioned_user_observed = (cond_error / scale) / abs(clean_answer) if clean_answer != 0.0 and cond_count > 0 else 0.0
    q_err = 0.5 / scale
    bf_free = 2.0 * q_err * n_rows
    m.bound_width = bf_free + bound_widen
    m.bound_width_inflation = m.bound_width / bf_free if bf_free > 0 else 1.0

    return m


# ── CI95 ──

def ci95(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    t_val = 1.96 if n >= 30 else {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
    }.get(n, 2.042)
    half = t_val * std / math.sqrt(n)
    return mean, mean - half, mean + half


# ── Main evaluator ──

class NMRDClaim1Evaluator:
    def __init__(self, n_rows: int = 500000, scale: int = 1,
                 dataset_base_dir: str | None = None):
        self.n_rows = n_rows
        self.scale = scale
        self.job_id = os.environ.get("SLURM_JOB_ID", "cpu_local")
        self.out_root = Path(
            f"results/reliability_layer1/phase4/nmr_d_claim1_closure/job_{self.job_id}"
        )
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.all_rows: list[dict] = []
        self.delta_rows: list[dict] = []
        self.coverage_rows: list[dict] = []
        self.uniform_family_rows: list[dict] = []
        self.storage_rows: list[dict] = []
        self.dataset_base_dir = Path(dataset_base_dir) if dataset_base_dir else None

    def _dataset_path(self, dataset: str) -> Path:
        if self.dataset_base_dir:
            return self.dataset_base_dir / dataset / "seg4096"
        # fallback to hardcoded paths
        paths = {
            "cesm_atm_cloud": "${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_cloud/seg4096",
            "hurricane_u": "${WORK_DIR}/datasets/locality_sensitivity/hurricane_u/seg4096",
            "cesm_atm_q": "${WORK_DIR}/datasets/locality_sensitivity/cesm_atm_q/seg4096",
            "hurricane_tc": "${WORK_DIR}/datasets/locality_sensitivity/hurricane_tc/seg4096",
        }
        return Path(paths[dataset])

    def load_planes(self, dataset: str) -> list[bytes]:
        ds_path = self._dataset_path(dataset)
        if not ds_path.is_dir():
            raise FileNotFoundError(f"Dataset not found: {ds_path}")
        return load_planes(ds_path, self.n_rows)

    def _get_n_rows(self, dataset: str) -> int:
        ds_path = self._dataset_path(dataset)
        for name in ["plane_000.bin", "plane_0.bin"]:
            p = ds_path / name
            if p.is_file():
                return p.stat().st_size
        return 500000

    # ── Run phases ──

    def run_deterministic(self, dataset: str) -> None:
        print(f"\n{'=' * 60}")
        print(f"Deterministic suite: {dataset}")
        print(f"{'=' * 60}")
        clean_planes = self.load_planes(dataset)
        clean_sum = compute_clean_sum(clean_planes)
        policies: list[tuple[str, list[int]]] = [
            (GRADED_B3, GRADED_R_VECTOR)
        ] + [(f"uniform_{name}", rv) for name, rv in UNIFORM_REPAIR_FAMILY.items()]

        det_seeds = 30
        total_det = len(FAULT_PATTERNS_DETERMINISTIC) * det_seeds
        det_done = 0
        for pattern in FAULT_PATTERNS_DETERMINISTIC:
            for seed in range(det_seeds):
                rng = random.Random(seed)
                events = deterministic_fault_events(pattern, rng, self.n_rows)
                self._evaluate_one(clean_planes, clean_sum, dataset,
                                   policies, events,
                                   fault_pattern=pattern, fault_rate="deterministic",
                                   seed=seed)
                det_done += 1
                if det_done % 25 == 0:
                    print(f"  deterministic: {det_done}/{total_det}", flush=True)

    def run_stochastic_sweep(self, dataset: str, fault_rate: str, seeds: int) -> None:
        print(f"\n{'=' * 60}")
        print(f"Stochastic sweep: {dataset} @ {fault_rate} ({seeds} seeds)")
        print(f"{'=' * 60}")
        clean_planes = self.load_planes(dataset)
        clean_sum = compute_clean_sum(clean_planes)
        policies: list[tuple[str, list[int]]] = [
            (GRADED_B3, GRADED_R_VECTOR)
        ] + [(f"uniform_{name}", rv) for name, rv in UNIFORM_REPAIR_FAMILY.items()]

        for seed in range(seeds):
            rng = random.Random(seed)
            rate_float = float(fault_rate)
            canonical = generate_canonical_fault_stream(rng, self.n_rows, rate_float)
            for policy_name, r_vector in policies:
                mutations = project_canonical_stream(canonical, r_vector)
                self._evaluate_one_stochastic(
                    clean_planes, clean_sum, dataset,
                    policy_name, r_vector, mutations,
                    fault_rate=fault_rate, seed=seed,
                )
            if (seed + 1) % 10 == 0:
                print(f"  stochastic: seed={seed + 1}/{seeds}", flush=True)

    # ── Per-cell evaluation ──

    def _evaluate_one(self, clean_planes: list[bytes], clean_sum: int,
                       dataset: str,
                       policies: list[tuple[str, list[int]]],
                       events: list[list[tuple[int, int, int]]],
                       fault_pattern: str, fault_rate: str, seed: int) -> None:
        for policy_name, r_vector in policies:
            adapted = self._adapt_mutations(events, r_vector)
            voted_sparse = apply_and_vote_sparse(clean_planes, r_vector, adapted)
            outcomes = classify_segments_sparse(clean_planes, voted_sparse, adapted, r_vector)
            m = compute_cell_metrics(
                clean_planes, voted_sparse, outcomes, adapted, r_vector,
                dataset, self.n_rows, self.scale,
                policy_name, fault_pattern, fault_rate, seed,
                kw_clean_sum=clean_sum,
            )
            self.all_rows.append(asdict(m))

    def _evaluate_one_stochastic(self, clean_planes: list[bytes], clean_sum: int,
                                  dataset: str,
                                  policy_name: str, r_vector: list[int],
                                  mutations: list[list[tuple[int, int, int]]],
                                  fault_rate: str, seed: int) -> None:
        voted_sparse = apply_and_vote_sparse(clean_planes, r_vector, mutations)
        outcomes = classify_segments_sparse(clean_planes, voted_sparse, mutations, r_vector)
        m = compute_cell_metrics(
            clean_planes, voted_sparse, outcomes, mutations, r_vector,
            dataset, self.n_rows, self.scale,
            policy_name, "stochastic", fault_rate, seed,
            kw_clean_sum=clean_sum,
        )
        self.all_rows.append(asdict(m))

    @staticmethod
    def _adapt_mutations(events: list[list[tuple[int, int, int]]],
                          r_vector: list[int]) -> list[list[tuple[int, int, int]]]:
        """Adapt deterministic events to match r_vector."""
        adapted: list[list[tuple[int, int, int]]] = [[] for _ in range(PLANE_COUNT)]
        for p in range(PLANE_COUNT):
            for rep_idx, offset, mask in events[p]:
                if rep_idx == -1:
                    for r in range(r_vector[p]):
                        adapted[p].append((r, offset, mask))
                elif rep_idx < r_vector[p]:
                    adapted[p].append((rep_idx, offset, mask))
                elif r_vector[p] > 0:
                    adapted[p].append((0, offset, mask))
        return adapted

    # ── Post-processing ──

    def compute_uniform_family_summary(self) -> None:
        if not self.all_rows:
            return
        by_key: dict[tuple[str, str, str], list[dict]] = {}
        for r in self.all_rows:
            if not r["policy"].startswith("uniform_"):
                continue
            key = (r["dataset"], r["fault_rate"], r["fault_pattern"])
            by_key.setdefault(key, []).append(r)

        # Fixed-best: pick one best member from all-seed means
        for key, rows in by_key.items():
            dataset, fault_rate, pattern = key
            by_member: dict[str, list[dict]] = {}
            for r in rows:
                by_member.setdefault(r["policy"], []).append(r)
            member_means: list[tuple[str, float, float]] = []
            for mname, mrows in by_member.items():
                rep_mean = sum(r["err_fault_repaired_only"] for r in mrows) / len(mrows)
                deliv_mean = sum(r["err_fault_conditioned_user_observed"] for r in mrows) / len(mrows)
                member_means.append((mname, rep_mean, deliv_mean))
            fixed_best_rep = min(member_means, key=lambda x: x[1])[0]
            fixed_best_del = min(member_means, key=lambda x: x[2])[0]

            for r in rows:
                self.uniform_family_rows.append({
                    "dataset": r["dataset"],
                    "fault_rate": r["fault_rate"],
                    "fault_pattern": r["fault_pattern"],
                    "uniform_member": r["policy"],
                    "allocation_r": r["allocation_r"],
                    "err_fault_repaired_only": r["err_fault_repaired_only"],
                    "err_fault_conditioned_user_observed": r["err_fault_conditioned_user_observed"],
                    "is_fixed_best_repair": "true" if r["policy"] == fixed_best_rep else "false",
                    "is_fixed_best_delivered": "true" if r["policy"] == fixed_best_del else "false",
                })

    def compute_paired_deltas(self) -> None:
        if not self.all_rows:
            return
        cells: set[tuple] = set()
        for r in self.all_rows:
            cells.add((r["dataset"], r["fault_rate"], r["fault_pattern"]))
        for dataset, fault_rate, pattern in sorted(cells):
            graded_by_seed: dict[int, dict] = {}
            uniform_by_seed: dict[int, dict[str, dict]] = {}
            for r in self.all_rows:
                if (r["dataset"], r["fault_rate"], r["fault_pattern"]) != (dataset, fault_rate, pattern):
                    continue
                s = r["seed"]
                if r["policy"] == GRADED_B3:
                    graded_by_seed[s] = r
                elif r["policy"].startswith("uniform_"):
                    uniform_by_seed.setdefault(s, {})[r["policy"]] = r

            seeds_common = sorted(set(graded_by_seed) & set(uniform_by_seed))
            if len(seeds_common) < 2:
                continue

            # Determine fixed-best uniform from all-seed means
            all_member_means: dict[str, list[float]] = {}
            all_member_deliv: dict[str, list[float]] = {}
            for s in seeds_common:
                for mname, mr in uniform_by_seed[s].items():
                    all_member_means.setdefault(mname, []).append(mr["err_fault_repaired_only"])
                    all_member_deliv.setdefault(mname, []).append(mr["err_fault_conditioned_user_observed"])
            fixed_best_repair = min(all_member_means, key=lambda k: sum(all_member_means[k]) / len(all_member_means[k]))
            fixed_best_deliver = min(all_member_deliv, key=lambda k: sum(all_member_deliv[k]) / len(all_member_deliv[k]))

            # --- Fixed-best delta ---
            deltas_repair_fixed: list[float] = []
            deltas_deliv_fixed: list[float] = []
            for s in seeds_common:
                g = graded_by_seed[s]
                u_rep = uniform_by_seed[s][fixed_best_repair]
                u_del = uniform_by_seed[s][fixed_best_deliver]
                deltas_repair_fixed.append(u_rep["err_fault_repaired_only"] - g["err_fault_repaired_only"])
                deltas_deliv_fixed.append(u_del["err_fault_conditioned_user_observed"] - g["err_fault_conditioned_user_observed"])

            dr_f_mean, dr_f_lo, dr_f_hi = ci95(deltas_repair_fixed)
            dd_f_mean, dd_f_lo, dd_f_hi = ci95(deltas_deliv_fixed)

            self.delta_rows.append({
                "dataset": dataset,
                "fault_rate": fault_rate,
                "fault_pattern": pattern,
                "method": "fixed_best",
                "metric": "delta_repair",
                "delta_mean": dr_f_mean,
                "ci95_low": dr_f_lo,
                "ci95_high": dr_f_hi,
                "n_pairs": len(deltas_repair_fixed),
                "delta_positive_frac": sum(1 for d in deltas_repair_fixed if d > 0) / max(len(deltas_repair_fixed), 1),
                "graded_mean": sum(graded_by_seed[s]["err_fault_repaired_only"] for s in seeds_common) / len(seeds_common),
                "uniform_mean": sum(uniform_by_seed[s][fixed_best_repair]["err_fault_repaired_only"] for s in seeds_common) / len(seeds_common),
                "best_uniform_member": fixed_best_repair,
            })
            self.delta_rows.append({
                "dataset": dataset,
                "fault_rate": fault_rate,
                "fault_pattern": pattern,
                "method": "fixed_best",
                "metric": "delta_delivered",
                "delta_mean": dd_f_mean,
                "ci95_low": dd_f_lo,
                "ci95_high": dd_f_hi,
                "n_pairs": len(deltas_deliv_fixed),
                "delta_positive_frac": sum(1 for d in deltas_deliv_fixed if d > 0) / max(len(deltas_deliv_fixed), 1),
                "graded_mean": sum(graded_by_seed[s]["err_fault_conditioned_user_observed"] for s in seeds_common) / len(seeds_common),
                "uniform_mean": sum(uniform_by_seed[s][fixed_best_deliver]["err_fault_conditioned_user_observed"] for s in seeds_common) / len(seeds_common),
                "best_uniform_member": fixed_best_deliver,
            })

            # --- Oracle-best delta (diagnostic, per-seed best) ---
            deltas_repair_oracle: list[float] = []
            deltas_deliv_oracle: list[float] = []
            for s in seeds_common:
                g = graded_by_seed[s]
                all_u = list(uniform_by_seed[s].values())
                best_rep = min(all_u, key=lambda u: u["err_fault_repaired_only"])
                best_del = min(all_u, key=lambda u: u["err_fault_conditioned_user_observed"])
                deltas_repair_oracle.append(best_rep["err_fault_repaired_only"] - g["err_fault_repaired_only"])
                deltas_deliv_oracle.append(best_del["err_fault_conditioned_user_observed"] - g["err_fault_conditioned_user_observed"])

            dr_o_mean, dr_o_lo, dr_o_hi = ci95(deltas_repair_oracle)
            dd_o_mean, dd_o_lo, dd_o_hi = ci95(deltas_deliv_oracle)

            self.delta_rows.append({
                "dataset": dataset,
                "fault_rate": fault_rate,
                "fault_pattern": pattern,
                "method": "oracle_best",
                "metric": "delta_repair",
                "delta_mean": dr_o_mean,
                "ci95_low": dr_o_lo,
                "ci95_high": dr_o_hi,
                "n_pairs": len(deltas_repair_oracle),
                "delta_positive_frac": sum(1 for d in deltas_repair_oracle if d > 0) / max(len(deltas_repair_oracle), 1),
                "graded_mean": sum(graded_by_seed[s]["err_fault_repaired_only"] for s in seeds_common) / len(seeds_common),
                "uniform_mean": sum(min(u["err_fault_repaired_only"] for u in uniform_by_seed[s].values()) for s in seeds_common) / len(seeds_common),
                "best_uniform_member": "per_seed_best",
            })
            self.delta_rows.append({
                "dataset": dataset,
                "fault_rate": fault_rate,
                "fault_pattern": pattern,
                "method": "oracle_best",
                "metric": "delta_delivered",
                "delta_mean": dd_o_mean,
                "ci95_low": dd_o_lo,
                "ci95_high": dd_o_hi,
                "n_pairs": len(deltas_deliv_oracle),
                "delta_positive_frac": sum(1 for d in deltas_deliv_oracle if d > 0) / max(len(deltas_deliv_oracle), 1),
                "graded_mean": sum(graded_by_seed[s]["err_fault_conditioned_user_observed"] for s in seeds_common) / len(seeds_common),
                "uniform_mean": sum(min(u["err_fault_conditioned_user_observed"] for u in uniform_by_seed[s].values()) for s in seeds_common) / len(seeds_common),
                "best_uniform_member": "per_seed_best",
            })

    def compute_coverage(self) -> None:
        for dataset in set(r["dataset"] for r in self.all_rows):
            det_total = sum(1 for r in self.all_rows
                            if r["dataset"] == dataset
                            and r["fault_rate"] == "deterministic"
                            and r["repair_invoked"] > 0)
            self.coverage_rows.append({
                "check_id": f"det_repair_invoked_{dataset}",
                "check_description": f"Deterministic repair_invoked count for {dataset}",
                "dataset": dataset,
                "fault_rate": "deterministic",
                "required": 30,
                "observed": det_total,
                "pass": "true" if det_total >= 30 else "false",
            })
            same_fault_recovered = sum(1 for r in self.all_rows
                                        if r["dataset"] == dataset
                                        and r["fault_pattern"] == "same_fault_all_replicas_plane0"
                                        and r["repair_success"] > 0)
            self.coverage_rows.append({
                "check_id": f"same_fault_negative_control_{dataset}",
                "check_description": f"Same-fault-all false recovery count for {dataset}",
                "dataset": dataset,
                "fault_rate": "deterministic",
                "required": 0,
                "observed": same_fault_recovered,
                "pass": "true" if same_fault_recovered == 0 else "false",
            })
            for rate in FAULT_RATES:
                count = sum(1 for r in self.all_rows
                            if r["dataset"] == dataset
                            and r["fault_rate"] == rate
                            and r["activated_fault"] > 0)
                self.coverage_rows.append({
                    "check_id": f"sto_activated_fault_{dataset}_{rate}",
                    "check_description": f"Stochastic activated_fault for {dataset} @ {rate}",
                    "dataset": dataset,
                    "fault_rate": rate,
                    "required": 30,
                    "observed": count,
                    "pass": "true" if count >= 30 else "false",
                })

    def compute_storage(self) -> None:
        policy_vectors: list[tuple[str, list[int]]] = [
            (GRADED_B3, GRADED_R_VECTOR)
        ] + [(f"uniform_{name}", rv) for name, rv in UNIFORM_REPAIR_FAMILY.items()]
        for pname, rv in policy_vectors:
            total = sum(rv)
            storage = total * self.n_rows
            extra = total - PLANE_COUNT
            self.storage_rows.append({
                "policy": pname,
                "allocation_r": "|".join(str(r) for r in rv),
                "total_replicas": total,
                "base_copies": PLANE_COUNT,
                "extra_replicas": extra,
                "storage_bytes": storage,
                "storage_overhead": f"{total / PLANE_COUNT:.3f}x",
            })

    def write_provenance(self) -> dict:
        prov = {
            "job_id": self.job_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python_version": sys.version,
            "command_line": shlex.join(sys.argv),
            "git_commit": os.popen("git rev-parse HEAD 2>/dev/null").read().strip() or "unknown",
            "branch": os.popen("git branch --show-current 2>/dev/null").read().strip() or "unknown",
            "n_rows": self.n_rows,
            "scale": self.scale,
        }
        path = self.out_root / "provenance_manifest.json"
        path.write_text(json.dumps(prov, indent=2))
        print(f"Provenance: {path}")
        return prov

    def write_handoff_json(self) -> None:
        marker = {
            "job_id": self.job_id,
            "experiment": "nmr_d_claim1_closure",
            "status": "complete",
            "n_rows": self.n_rows,
            "branch": os.popen("git branch --show-current 2>/dev/null").read().strip() or "unknown",
            "commit": os.popen("git rev-parse HEAD 2>/dev/null").read().strip() or "unknown",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results_dir": str(self.out_root),
            "datasets": list(set(r["dataset"] for r in self.all_rows)) if self.all_rows else [],
            "total_cells": len(self.all_rows),
        }
        path = self.out_root / "handoff.json"
        path.write_text(json.dumps(marker, indent=2))
        print(f"Handoff: {path}")

    def write_csvs(self) -> None:
        fieldnames_matrix = list(asdict(CellMetrics()).keys())
        csv_path = self.out_root / "nmr_d_claim1_matrix.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames_matrix, extrasaction="ignore")
            w.writeheader()
            for r in self.all_rows:
                w.writerow(r)
        print(f"Matrix: {csv_path} ({len(self.all_rows)} rows)")

        delta_fields = ["dataset", "fault_rate", "fault_pattern", "method", "metric",
                        "delta_mean", "ci95_low", "ci95_high", "n_pairs",
                        "delta_positive_frac", "graded_mean", "uniform_mean",
                        "best_uniform_member"]
        csv_delta = self.out_root / "nmr_d_paired_delta_summary.csv"
        with open(csv_delta, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=delta_fields, extrasaction="ignore")
            w.writeheader()
            for r in self.delta_rows:
                w.writerow(r)
        print(f"Delta: {csv_delta} ({len(self.delta_rows)} rows)")

        cov_fields = ["check_id", "check_description", "dataset", "fault_rate",
                       "required", "observed", "pass"]
        csv_cov = self.out_root / "nmr_d_coverage_manifest.csv"
        with open(csv_cov, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cov_fields, extrasaction="ignore")
            w.writeheader()
            for r in self.coverage_rows:
                w.writerow(r)
        print(f"Coverage: {csv_cov} ({len(self.coverage_rows)} rows)")

        uf_fields = ["dataset", "fault_rate", "fault_pattern", "uniform_member",
                      "allocation_r", "err_fault_repaired_only",
                      "err_fault_conditioned_user_observed",
                      "is_fixed_best_repair", "is_fixed_best_delivered"]
        csv_uf = self.out_root / "nmr_d_uniform_family_summary.csv"
        with open(csv_uf, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=uf_fields, extrasaction="ignore")
            w.writeheader()
            for r in self.uniform_family_rows:
                w.writerow(r)
        print(f"Uniform family: {csv_uf} ({len(self.uniform_family_rows)} rows)")

        st_fields = ["policy", "allocation_r", "total_replicas",
                      "base_copies", "extra_replicas",
                      "storage_bytes", "storage_overhead"]
        csv_st = self.out_root / "nmr_d_latency_storage_summary.csv"
        with open(csv_st, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=st_fields, extrasaction="ignore")
            w.writeheader()
            for r in self.storage_rows:
                w.writerow(r)
        print(f"Storage: {csv_st} ({len(self.storage_rows)} rows)")

        self.write_provenance()
        self.write_handoff_json()

    def run(self, mode: str, dataset: str | None = None,
            fault_rate: str | None = None, seeds: int = 30,
            fixed_rows: int = 0) -> None:
        datasets = [dataset] if dataset else ["cesm_atm_cloud", "hurricane_u"]
        for ds in datasets:
            if ds not in ["cesm_atm_cloud", "hurricane_u", "cesm_atm_q", "hurricane_tc"]:
                print(f"SKIP: unknown dataset {ds}")
                continue
            try:
                self._dataset_path(ds)
            except KeyError:
                print(f"SKIP: no path for dataset {ds}")
                continue
            self.n_rows = fixed_rows if fixed_rows > 0 else self._get_n_rows(ds)
            if mode in ("deterministic", "full"):
                self.run_deterministic(ds)
            if mode in ("stochastic", "full"):
                rates = [fault_rate] if fault_rate else FAULT_RATES
                for fr in rates:
                    self.run_stochastic_sweep(ds, fr, seeds)

        self.compute_uniform_family_summary()
        self.compute_paired_deltas()
        self.compute_coverage()
        self.compute_storage()
        self.write_csvs()
        self.print_verdict()

    def print_verdict(self) -> None:
        print("\n" + "=" * 60)
        print("NMR-D CLAIM-1 VERDICT")
        print("=" * 60)
        all_pass = True
        for cr in self.coverage_rows:
            if cr["pass"] == "false":
                print(f"  ❌ Coverage fail: {cr['check_id']} "
                      f"(observed={cr['observed']} < required={cr['required']})")
                all_pass = False
        if all_pass:
            print("  ✅ All coverage gates pass")

        sam_fail = sum(1 for r in self.all_rows
                       if r["fault_pattern"] == "same_fault_all_replicas_plane0"
                       and r["repair_success"] > 0)
        if sam_fail > 0:
            print(f"  ❌ Same-fault-all replicas falsely recovered: {sam_fail} cells")
            all_pass = False
        else:
            print("  ✅ Same-fault-all negative control preserved")

        if not all_pass:
            print("\n⚠️  Coverage or control failures — verdict may be NEEDS_MORE_EVENTS")

        same_fault_cells = [r for r in self.all_rows
                            if r["fault_pattern"] == "same_fault_all_replicas_plane0"
                            and r["fault_rate"] == "deterministic"]
        false_rec = sum(1 for r in same_fault_cells if r["repair_success"] > 0) if same_fault_cells else 0
        total_sf = len(same_fault_cells)
        if same_fault_cells:
            print(f"  Same-fault-all: {false_rec}/{total_sf} falsely recovered "
                  f"(must be 0) {'✅' if false_rec == 0 else '❌'}")

        all_coverage_pass = all(cr["pass"] == "true" for cr in self.coverage_rows)
        if not all_coverage_pass:
            print(f"\n  ⚠️  Coverage failures exist — verdict: NEEDS_MORE_EVENTS")
        elif false_rec > 0:
            print(f"\n  ❌ Neg control violated — verdict: NEEDS_FIXES")
        else:
            primary_fields = ["cesm_atm_cloud", "hurricane_u"]
            for method in ["fixed_best", "oracle_best"]:
                repair_deltas = [r for r in self.delta_rows
                                 if r["method"] == method
                                 and r["metric"] == "delta_repair"
                                 and r["dataset"] in primary_fields]
                deliv_deltas = [r for r in self.delta_rows
                                if r["method"] == method
                                and r["metric"] == "delta_delivered"
                                and r["dataset"] in primary_fields]
                r_pos = sum(1 for r in repair_deltas if r["ci95_low"] > 0)
                d_pos = sum(1 for r in deliv_deltas if r["ci95_low"] > 0)
                print(f"\n  [{method}]")
                print(f"    CI95_low(delta_repair) > 0: {r_pos}/{len(repair_deltas)} cells")
                print(f"    CI95_low(delta_delivered) > 0: {d_pos}/{len(deliv_deltas)} cells")

        print(f"\nResults: {self.out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="full",
                        choices=["deterministic", "stochastic", "full"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--fault-rate", default=None)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--n-rows", type=int, default=0)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--dataset-dir", default=None,
                        help="Base directory for dataset locality_sensitivity folders")
    parser.add_argument("--require-h200", action="store_true",
                        help="Exit with error if not running on H200")
    args = parser.parse_args()

    if args.require_h200:
        import subprocess
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if "H200" not in gpu:
            print(f"FATAL: --require-h200 but GPU is '{gpu}'", file=sys.stderr)
            sys.exit(2)

    fixed_n_rows = args.n_rows if args.n_rows > 0 else 500000
    evaluator = NMRDClaim1Evaluator(
        n_rows=fixed_n_rows, scale=args.scale,
        dataset_base_dir=args.dataset_dir,
    )
    evaluator.run(
        mode=args.mode,
        dataset=args.dataset,
        fault_rate=args.fault_rate,
        seeds=args.seeds,
        fixed_rows=args.n_rows,
    )


if __name__ == "__main__":
    main()
