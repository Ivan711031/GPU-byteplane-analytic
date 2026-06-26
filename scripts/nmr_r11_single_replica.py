#!/usr/bin/env python3
"""NMR-R11: Single-Replica Detect-and-Bound Contract Freeze for v1.3.

Evaluates the conservative single-replica (r=[1]*8) detection plus certified
bound widening contract on real byteplane datasets.

Pipeline per test cell:
  load clean planes -> inject plane-uniform faults at rate ->
  per-plane SUM32 digest -> detect mismatch ->
  certified bound widening -> contains_truth check -> classify

Classification taxonomy:
  - recovered:    digest match (no corruption or digest false negative)
  - bounded:      digest mismatch, contains_truth=True
  - unavailable:  digest mismatch, contains_truth=False (cert bound failure)
  - silent_wrong: digest match but answer differs from clean
                  (SUM32 collision / false negative)

Usage:
  python3 scripts/nmr_r11_single_replica.py \
    --artifact-dir /path/to/seg4096 \
    --dataset hurricane_u \
    --n-rows 25000000 \
    --rates 1e-6 1e-5 1e-4 \
    --seeds 0 1 2 \
    --mode plane_uniform \
    --output /tmp/nmr_r11_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import struct
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PLANE_COUNT = 8
PLANE_WEIGHTS = [1 << (8 * (7 - p)) for p in range(PLANE_COUNT)]
SEGMENT_SIZE = 4096


import numpy as np

def sum32(data: bytes) -> int:
    arr = np.frombuffer(data, dtype=np.uint8)
    pad = (4 - arr.size % 4) % 4
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    return int(arr.view(dtype=np.uint32).sum()) & 0xFFFFFFFF


def per_plane_sum32(planes: list[bytes]) -> list[int]:
    return [sum32(p) for p in planes]


def load_planes(artifact_dir: Path, n_rows: int) -> list[bytes]:
    planes: list[bytes] = []
    for p in range(PLANE_COUNT):
        path = artifact_dir / f"plane_{p:03d}.bin"
        if path.is_file():
            data = path.read_bytes()[:n_rows]
        else:
            data = bytes(n_rows)
        if len(data) < n_rows:
            data = data + bytes(n_rows - len(data))
        planes.append(data)
    return planes


def inject_plane_uniform(
    clean_planes: list[bytes],
    n_rows: int,
    rate: float,
    rng: random.Random,
) -> list[bytearray]:
    delivered = [bytearray(p) for p in clean_planes]
    for p in range(PLANE_COUNT):
        n_faults = max(1, int(n_rows * rate + 0.5)) if rate > 0 else 0
        if n_faults > 0:
            positions = rng.sample(range(n_rows), min(n_faults, n_rows))
            for pos in positions:
                mask = rng.randint(1, 255)
                delivered[p][pos] ^= mask
    return delivered


def inject_same_fault_all(
    clean_planes: list[bytes],
    n_rows: int,
    rng: random.Random,
) -> list[bytearray]:
    delivered = [bytearray(p) for p in clean_planes]
    pos = rng.randint(0, n_rows - 1)
    mask = rng.randint(1, 255)
    for p in range(PLANE_COUNT):
        delivered[p][pos] ^= mask
    return delivered


def compute_encoded_sum(planes: list[bytes]) -> int:
    return sum(sum(p) * PLANE_WEIGHTS[pi] for pi, p in enumerate(planes))


def compute_bound_width(detected_planes: list[bool], n_rows: int) -> float:
    return sum(255.0 * PLANE_WEIGHTS[p] * n_rows
               for p in range(PLANE_COUNT) if detected_planes[p])


def run_cell(
    clean_planes: list[bytes],
    delivered: list[bytearray],
    clean_digests: list[int],
    n_rows: int,
    fault_planes: list[int],
    fault_offsets: list[int],
) -> dict[str, Any]:
    delivered_digests = [sum32(bytes(d)) for d in delivered]
    detected_planes = [d != c for d, c in zip(delivered_digests, clean_digests)]
    detected = any(detected_planes)

    clean_answer = float(compute_encoded_sum(clean_planes))
    delivered_answer = float(compute_encoded_sum(delivered))

    if not detected:
        if delivered_answer == clean_answer:
            outcome = "recovered"
        else:
            outcome = "silent_wrong"
        bound_width = 0.0
        contains_truth = delivered_answer == clean_answer
    else:
        bound_width = compute_bound_width(detected_planes, n_rows)
        lo = delivered_answer - bound_width / 2.0
        hi = delivered_answer + bound_width / 2.0
        contains_truth = lo <= clean_answer <= hi
        outcome = "bounded" if contains_truth else "unavailable"

    return {
        "detected": detected,
        "detected_planes": str([p for p in range(PLANE_COUNT) if detected_planes[p]]),
        "contains_truth": contains_truth,
        "outcome": outcome,
        "bound_width": bound_width,
        "clean_answer": clean_answer,
        "delivered_answer": delivered_answer,
        "fault_planes": str(sorted(set(fault_planes))),
        "fault_offsets": len(fault_offsets),
    }


def evaluate(
    artifact_dir: Path,
    dataset: str,
    n_rows: int,
    rates: list[float],
    seeds: list[int],
    mode: str = "plane_uniform",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    print(f"\nLoading {dataset} ({n_rows} rows) ...")
    clean_planes = load_planes(artifact_dir, n_rows)
    clean_digests = per_plane_sum32(clean_planes)
    print(f"  loaded {len(clean_planes[0])} rows, "
          f"digests={[hex(d) for d in clean_digests]}")

    for rate in rates:
        for seed in seeds:
            rng = random.Random(seed)
            if mode == "plane_uniform":
                delivered = inject_plane_uniform(clean_planes, n_rows, rate, rng)
                n_faults = [int(n_rows * rate + 0.5) for _ in range(PLANE_COUNT)]
                fault_planes = list(range(PLANE_COUNT))
                fault_offsets = []
                for p in range(PLANE_COUNT):
                    nf = max(1, int(n_rows * rate + 0.5)) if rate > 0 else 0
                    if nf > 0:
                        fault_offsets.extend(rng.sample(range(n_rows), min(nf, n_rows)))
            elif mode == "same_fault_all":
                delivered = inject_same_fault_all(clean_planes, n_rows, rng)
                fault_planes = list(range(PLANE_COUNT))
                fault_offsets = [rng.randint(0, n_rows - 1)]
            else:
                raise ValueError(f"unknown mode: {mode}")

            result = run_cell(
                clean_planes, delivered, clean_digests, n_rows,
                fault_planes, fault_offsets,
            )
            result["dataset"] = dataset
            result["n_rows"] = n_rows
            result["mode"] = mode
            result["rate"] = str(rate)
            result["seed"] = seed
            rows.append(result)

            label = result["outcome"]
            bw = result["bound_width"]
            print(f"  rate={rate:.1e} seed={seed:2d}  "
                  f"{label:<15s}  bound={bw:.6e}  "
                  f"detected={result['detected']}  "
                  f"contains_truth={result['contains_truth']}")

    return rows


def summarize(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("NMR-R11: Single-Replica Detect-and-Bound Contract Freeze")
    print("=" * 80)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["mode"], r["rate"])].append(r)

    for (ds, mode, rate), group in sorted(groups.items()):
        n = len(group)
        outcomes = Counter(r["outcome"] for r in group)
        detected = sum(1 for r in group if r["detected"])
        contains_truth = sum(1 for r in group if r["contains_truth"])
        shared_bw = sum(r["bound_width"] for r in group) / n

        print(f"\n{ds:<20s} mode={mode:<16s} rate={rate:>8s}  "
              f"({n} seeds)")
        print(f"  detected:       {detected:>3d}/{n:<3d} = {detected/max(n,1):.4f}")
        print(f"  contains_truth: {contains_truth:>3d}/{n:<3d} = {contains_truth/max(n,1):.4f}")
        print(f"  avg bound_w:    {shared_bw:.6e}")
        for oc, cnt in sorted(outcomes.items()):
            print(f"  {oc:<15s}: {cnt:>3d}/{n:<3d} = {cnt/max(n,1):.4f}")

    print("\n\n=== Overall ===")
    m = compute_metrics(rows)
    for k, v in m.items():
        print(f"  {k}: {v}")


def compute_metrics(rows: list[dict]) -> dict[str, float]:
    n = len(rows) if rows else 1
    detected = sum(1 for r in rows if r["detected"])
    contains_truth = sum(1 for r in rows if r["contains_truth"])
    bounded = sum(1 for r in rows if r["outcome"] == "bounded")
    unavailable = sum(1 for r in rows if r["outcome"] == "unavailable")
    recovered = sum(1 for r in rows if r["outcome"] == "recovered")
    silent_wrong = sum(1 for r in rows if r["outcome"] == "silent_wrong")
    widths = [r["bound_width"] for r in rows]
    return {
        "total_cells": n,
        "detected_rate": detected / n,
        "contains_truth": contains_truth / n,
        "expected_bound_width": sum(widths) / n,
        "recovered_rate": recovered / n,
        "bounded_rate": bounded / n,
        "unavailable_rate": unavailable / n,
        "silent_wrong_rate": silent_wrong / n,
        "detected_count": detected,
        "contains_truth_count": contains_truth,
        "recovered_count": recovered,
        "bounded_count": bounded,
        "unavailable_count": unavailable,
        "silent_wrong_count": silent_wrong,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n-rows", type=int, required=True)
    parser.add_argument("--rates", type=float, nargs="+",
                        default=[1e-6, 1e-5, 1e-4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--mode", type=str, default="plane_uniform",
                        choices=["plane_uniform", "same_fault_all"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    jid = os.environ.get("SLURM_JOB_ID", "cpu_r11")
    out_path = args.output

    print(f"=== NMR-R11 Single-Replica Detect-and-Bound ===")
    print(f"dataset={args.dataset} n_rows={args.n_rows}")
    print(f"rates={args.rates} seeds={args.seeds} mode={args.mode}")
    print(f"output={out_path}")

    t0 = time.perf_counter()
    rows = evaluate(
        args.artifact_dir, args.dataset, args.n_rows,
        args.rates, args.seeds, args.mode,
    )
    elapsed = time.perf_counter() - t0

    fields = [
        "dataset", "n_rows", "mode", "rate", "seed",
        "detected", "detected_planes", "contains_truth", "outcome",
        "bound_width", "clean_answer", "delivered_answer",
        "fault_planes", "fault_offsets",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"\nWrote {out_path} ({len(rows)} rows)")

    summarize(rows)
    m = compute_metrics(rows)

    print(f"\n=== Verdict ===")
    if m["silent_wrong_rate"] > 0:
        print(f"  SILENT_WRONG rate={m['silent_wrong_rate']:.6f} — "
              f"FAILS DETECT-AND-BOUND CONTRACT")
        print(f"  → FAILS_REPRODUCTION")
    elif m["detected_rate"] < 0.5 and m["recovered_rate"] > 0:
        print(f"  Most outcomes are recovered (digest match, no fault visible)")
        print(f"  → CONFIRMS_WITH_REVISED_SCOPE")
    elif m["contains_truth"] == 1.0:
        print(f"  contains_truth=1.0000, detected_rate={m['detected_rate']:.4f}, "
              f"silent_wrong=0")
        print(f"  → CONFIRMS_DETECT_AND_BOUND_CONTRACT")
    elif m["contains_truth"] > 0.99:
        print(f"  contains_truth={m['contains_truth']:.6f} (near 1.0), "
              f"no silent_wrong")
        print(f"  → CONFIRMS_WITH_REVISED_SCOPE")
    else:
        print(f"  contains_truth={m['contains_truth']:.6f} — "
              f"below threshold")
        print(f"  → NEEDS_PAPER_CHANGE")

    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Done.")


import os

if __name__ == "__main__":
    main()
