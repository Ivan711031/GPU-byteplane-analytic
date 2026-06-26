"""Claim 2 Evaluator: answer-quality metrics under diversity policies.

Compares 4 diversity policies (r=3) across fault suites A-D using:
- delivered answer error
- decision flip rate
- contains truth rate
- escape rate
- expected bound width
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from claim2_fault_suites import FAULT_SUITES, R, PLANE_WEIGHTS

SEGMENT_SIZE = 1024


@dataclass
class CellResult:
    dataset: str = ""
    n_rows: int = 0
    scale: int = 0
    suite: str = ""
    fault_family: str = ""
    rate: str = ""
    policy: str = ""
    seed: int = 0

    # Primary
    delivered_answer: float = 0.0
    clean_answer: float = 0.0
    absolute_error: float = 0.0
    relative_error: float = 0.0
    decision_flip: int = 0
    decision_flip_rate: float = 0.0
    contains_truth: bool = True
    expected_bound_width: float = 0.0

    # Secondary
    escape_rate: float = 0.0
    fallback_rate: float = 0.0
    uncertified_rate: float = 0.0
    vote_recovered: bool = False
    detected: bool = False

    def row_dict(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "n_rows": str(self.n_rows),
            "suite": self.suite,
            "fault_family": self.fault_family,
            "rate": self.rate,
            "policy": self.policy,
            "seed": str(self.seed),
            "delivered_answer": f"{self.delivered_answer:.10f}",
            "clean_answer": f"{self.clean_answer:.10f}",
            "absolute_error": f"{self.absolute_error:.10e}",
            "relative_error": f"{self.relative_error:.10e}",
            "decision_flip": str(self.decision_flip),
            "decision_flip_rate": f"{self.decision_flip_rate:.10f}",
            "contains_truth": str(self.contains_truth).lower(),
            "expected_bound_width": f"{self.expected_bound_width:.10e}",
            "escape_rate": f"{self.escape_rate:.10f}",
            "fallback_rate": f"{self.fallback_rate:.10f}",
            "uncertified_rate": f"{self.uncertified_rate:.10f}",
            "vote_recovered": str(self.vote_recovered).lower(),
            "detected": str(self.detected).lower(),
        }


CELL_FIELDS = list(CellResult().row_dict().keys())


# ── Planes I/O ───────────────────────────────────────────────

def load_clean_planes(artifact_dir: Path, n_rows: int) -> list[bytes]:
    planes: list[bytes] = []
    for p in range(8):
        path = artifact_dir / f"plane_{p:03d}.bin"
        if path.is_file():
            data = path.read_bytes()[:n_rows]
        else:
            path = artifact_dir / f"plane_{p}.bin"
            if path.is_file():
                data = path.read_bytes()[:n_rows]
            else:
                data = bytes(n_rows)
        if len(data) < n_rows:
            data = data + bytes(n_rows - len(data))
        planes.append(data)
    return planes


def load_artifact_metadata(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "artifact.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def compute_clean_sum(planes: list[bytes]) -> int:
    return sum(sum(planes[p]) * PLANE_WEIGHTS[p] for p in range(8))


# ── Fault application with diversity ─────────────────────────

def expand_fault_entries(
    entries: list[dict[str, Any]],
    policy: str,
    seed: int,
    n_rows: int,
) -> dict[int, list[list[tuple[int, int]]]]:
    """Expand fault entries into per-replica per-plane fault lists.

    Returns {plane: [[(offset, mask), ...] for replica 0..R-1]}.

    Policy behavior:
    - no_diversity_naive: same entries for ALL replicas
    - spatial_only: each replica gets independent fault rolls
    - temporal_only: same base entries, but each "read" resolves transient faults
    - spatial_temporal: both spatial separation AND temporal resolution
    """
    rng = random.Random(seed * 10000 + 42)

    # Build per-plane sets of unique (offset, mask) — dedup by offset
    base: dict[int, dict[tuple[int, int], list[int]]] = defaultdict(dict)
    # base[plane][offset] = (mask, [target_replicas])
    for entry in entries:
        plane = entry.get("plane", 0)
        offset = entry["offset"]
        mask = entry["mask"]
        reps = entry.get("replicas", [0])
        if offset not in base[plane]:
            base[plane][offset] = (mask, reps[:])
        else:
            existing_mask, existing_reps = base[plane][offset]
            for r in reps:
                if r not in existing_reps:
                    existing_reps.append(r)

    if policy == "no_diversity_naive_r3":
        result: dict[int, list[list[tuple[int, int]]]] = {}
        for plane, offsets in base.items():
            all_faults = [(o, m) for o, (m, _) in offsets.items()]
            result[plane] = [all_faults[:] for _ in range(R)]
        return result

    if policy == "spatial_only_diverse_r3":
        result = {}
        for plane, offsets in base.items():
            per_rep: list[list[tuple[int, int]]] = [[] for _ in range(R)]
            for offset, (mask, target_reps) in offsets.items():
                for rep_idx in target_reps:
                    if rep_idx < R:
                        rep_rng = random.Random(seed * 10000 + rep_idx * 1000 + offset + 42)
                        per_rep[rep_idx].append((offset, rep_rng.randint(1, 255)))
            result[plane] = per_rep
        return result

    if policy == "temporal_only_diverse_r3":
        result = {}
        for plane, offsets in base.items():
            per_rep: list[list[tuple[int, int]]] = [[] for _ in range(R)]
            for offset, (mask, target_reps) in offsets.items():
                resolve_prob = 0.0
                for entry in entries:
                    if entry.get("offset") == offset:
                        resolve_prob = entry.get("temporal_resolve_prob", 0.0)
                        break
                for rep_idx in target_reps:
                    if rep_idx < R:
                        rep_rng = random.Random(seed * 10000 + rep_idx * 100 + offset + 42)
                        if rep_rng.random() >= resolve_prob:
                            per_rep[rep_idx].append((offset, mask))
            result[plane] = per_rep
        return result

    if policy == "spatial_temporal_diverse_r3":
        result = {}
        for plane, offsets in base.items():
            per_rep: list[list[tuple[int, int]]] = [[] for _ in range(R)]
            for offset, (mask, target_reps) in offsets.items():
                resolve_prob = 0.0
                for entry in entries:
                    if entry.get("offset") == offset:
                        resolve_prob = entry.get("temporal_resolve_prob", 0.0)
                        break
                for rep_idx in target_reps:
                    if rep_idx < R:
                        rep_rng = random.Random(seed * 10000 + rep_idx * 1000 + offset + 42)
                        if rep_rng.random() >= resolve_prob:
                            per_rep[rep_idx].append((offset, rep_rng.randint(1, 255)))
            result[plane] = per_rep
        return result

    raise ValueError(f"unknown policy: {policy}")


def apply_faults(
    clean_planes: list[bytes],
    per_plane_faults: dict[int, list[list[tuple[int, int]]]],
) -> list[list[bytearray]]:
    replicas: list[list[bytearray]] = []
    for p in range(8):
        plane_replicas: list[bytearray] = []
        plane_faults = per_plane_faults.get(p, [])
        src = array("B", clean_planes[p])
        for rep_idx in range(R):
            data = bytearray(src)
            if rep_idx < len(plane_faults):
                for offset, mask in plane_faults[rep_idx]:
                    data[offset] ^= mask
            plane_replicas.append(data)
        replicas.append(plane_replicas)
    return replicas


# ── Voting ───────────────────────────────────────────────────

def majority_vote_planes(
    replicas: list[list[bytearray]],
) -> list[bytes]:
    """Majority-vote each plane across R replicas (numpy vectorized)."""
    voted: list[bytes] = []
    for p in range(8):
        a = np.frombuffer(replicas[p][0], dtype=np.uint8)
        b = np.frombuffer(replicas[p][1], dtype=np.uint8)
        c = np.frombuffer(replicas[p][2], dtype=np.uint8)
        result = np.where(a == b, a, np.where(a == c, a, b))
        voted.append(result.tobytes())
    return voted


# ── Answer-quality metrics ───────────────────────────────────

def planes_to_np(planes: list[bytes], n_rows: int) -> np.ndarray:
    """Convert 8 plane byte arrays to (8, n_rows) float64 matrix."""
    out = np.empty((8, n_rows), dtype=np.float64)
    for p in range(8):
        out[p, :] = list(planes[p][:n_rows])
    return out


def compute_answer_quality(
    clean_np: np.ndarray,
    voted_planes: list[bytes],
    scale: int,
    n_rows: int,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute answer-quality metrics (numpy-vectorized)."""
    pw = np.array(PLANE_WEIGHTS, dtype=np.float64)
    sf = float(scale)
    thresh_scaled = float(threshold * sf)

    # Voted to numpy (float64 to avoid uint64 overflow when summing across rows)
    voted_np = np.empty((8, n_rows), dtype=np.float64)
    for p in range(8):
        voted_np[p, :] = list(voted_planes[p][:n_rows])

    # Per-row sums via dot product: (8, n) · (8,)² = (n,)
    clean_rows = (pw[:, None] * clean_np).sum(axis=0)
    delivered_rows = (pw[:, None] * voted_np).sum(axis=0)

    clean_sum = int(clean_rows.sum())
    delivered_sum = int(delivered_rows.sum())
    clean_answer = clean_sum / sf
    delivered_answer = delivered_sum / sf
    abs_err = abs(delivered_answer - clean_answer)
    rel_err = abs_err / abs(clean_answer) if clean_answer != 0 else abs_err

    # Mismatches
    mismatches = int((clean_np != voted_np).sum())
    escape_rate = mismatches / (n_rows * 8) if n_rows > 0 else 0.0

    # Bound width: sum of |diff| * weight for all mismatched bytes
    diff = (voted_np.astype(np.int64) - clean_np.astype(np.int64))
    bound_width = float(np.abs(diff).astype(np.float64).sum(axis=1) @ pw.astype(np.float64)) / sf

    # Decision flip
    clean_bool = clean_rows > thresh_scaled
    delivered_bool = delivered_rows > thresh_scaled
    decision_flip = int((clean_bool != delivered_bool).sum())
    decision_flip_rate = decision_flip / n_rows

    lo = delivered_answer - bound_width / 2.0
    hi = delivered_answer + bound_width / 2.0
    contains_truth = not (lo > clean_answer or hi < clean_answer)

    return {
        "clean_answer": clean_answer,
        "delivered_answer": delivered_answer,
        "absolute_error": abs_err,
        "relative_error": rel_err,
        "decision_flip": decision_flip,
        "decision_flip_rate": decision_flip_rate,
        "contains_truth": contains_truth,
        "expected_bound_width": bound_width,
        "escape_rate": escape_rate,
        "vote_recovered": mismatches == 0,
        "detected": mismatches > 0,
    }


# ── Main evaluator ───────────────────────────────────────────

def evaluate_cell(
    clean_planes: list[bytes],
    clean_np: np.ndarray,
    entries: list[dict[str, Any]],
    policy: str,
    seed: int,
    n_rows: int,
    scale: int,
    dataset: str,
    suite: str,
    fault_family: str,
    rate: str,
) -> CellResult:
    per_plane = expand_fault_entries(entries, policy, seed, n_rows)
    replicas = apply_faults(clean_planes, per_plane)
    voted = majority_vote_planes(replicas)
    quality = compute_answer_quality(clean_np, voted, scale, n_rows)

    return CellResult(
        dataset=dataset,
        n_rows=n_rows,
        scale=scale,
        suite=suite,
        fault_family=fault_family,
        rate=rate,
        policy=policy,
        seed=seed,
        **{k: v for k, v in quality.items() if k in CellResult.__dataclass_fields__},
    )


# ── CLI ──────────────────────────────────────────────────────

POLICIES = [
    "no_diversity_naive_r3",
    "spatial_only_diverse_r3",
    "temporal_only_diverse_r3",
    "spatial_temporal_diverse_r3",
]

CANDIDATE_RATES = ["1e-07", "3e-07", "1e-06", "3e-06", "1e-05", "3e-05", "1e-04"]


def parse_fault_rate(rate_str: str) -> float:
    return float(rate_str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="hurricane_u")
    parser.add_argument("--n-rows", type=int, default=500000)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--suite", type=str, default=None,
                        help="Filter to specific suite(s). Comma-separated.")
    parser.add_argument("--fault-family", type=str, default=None,
                        help="Filter to specific fault family")
    parser.add_argument("--rate", type=str, default=None,
                        help="Fault rate (e.g. 1e-06). If omitted, runs all candidate rates.")
    parser.add_argument("--policy", type=str, default=None,
                        help="Policy name. If omitted, runs all policies.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Seeds. Default: 0-4 (5 seeds)")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    if args.seeds is None:
        seeds = list(range(5))
    else:
        seeds = args.seeds

    rates = [args.rate] if args.rate else CANDIDATE_RATES
    policies = [args.policy] if args.policy else POLICIES

    clean_planes = load_clean_planes(args.artifact_dir, args.n_rows)
    clean_np = planes_to_np(clean_planes, args.n_rows)
    meta = load_artifact_metadata(args.artifact_dir)
    scale = args.scale or meta.get("scale", 1)

    rows: list[CellResult] = []

    filter_suites = [s.strip() for s in args.suite.split(",")] if args.suite else None
    for suite_id, suite in FAULT_SUITES.items():
        if filter_suites and suite_id not in filter_suites:
            continue
        for fam_name, fam in suite["families"].items():
            if args.fault_family and fam_name != args.fault_family:
                continue
            generator = fam["generator"]
            for rate_str in rates:
                rate_val = parse_fault_rate(rate_str)
                for seed in seeds:
                    entries = generator(seed, args.n_rows, rate_val)
                    # Evaluate even with empty entries (Suite A null test)
                    for policy in policies:
                        result = evaluate_cell(
                            clean_planes=clean_planes,
                            clean_np=clean_np,
                            entries=entries,
                            policy=policy,
                            seed=seed,
                            n_rows=args.n_rows,
                            scale=scale,
                            dataset=args.dataset,
                            suite=suite_id,
                            fault_family=fam_name,
                            rate=rate_str,
                        )
                        rows.append(result)

    print(f"Evaluated {len(rows)} cells", file=sys.stderr)

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CELL_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r.row_dict())
        print(f"CSV: {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
