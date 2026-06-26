#!/usr/bin/env python3
"""Phase 3 Z2-B / Z2-C: Real-dataset SDC matrix + SUM32 escape rate.

Z2-B: Uniform faults on real datasets, 3 lanes, per-plane injection.
Z2-C: Cluster/burst/adversarial_cancel faults, escape rate with 95% CI.

CPU-only. Uses existing Z2 evaluator / Z2A kernel / Y0 evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase3_y0_evaluator import (
    compute_delivered_answer_with_degradation,
    compute_voted_planes,
)
from phase3_z2_sdc_evaluator import (
    sum32,
    per_plane_sum32,
    LANE_RAW_UNCHECKED,
    LANE_RAW_DIGEST,
    LANE_BYTEPLANE,
    EVENT_SDC,
    EVENT_CERT_BOUND_FAILURE,
    EVENT_HARD_FAIL,
    EVENT_BOUNDED_DEGRADED,
    EVENT_CLEAN,
    EVENT_UNDETECTED,
    aggregate_events,
)

PLANE_WEIGHTS = [1 << (8 * (7 - p)) for p in range(8)]
SEGMENT_SIZE = 4096


def sum32_raw(data: bytes) -> int:
    padded = data + b'\x00' * ((4 - len(data) % 4) % 4)
    total = 0
    for i in range(0, len(padded), 4):
        total += struct.unpack('<I', padded[i:i+4])[0]
    return total & 0xFFFFFFFF


def raw_f64_sum(data: bytes) -> float:
    n = len(data) // 8
    total = 0.0
    for i in range(n):
        total += struct.unpack("<d", data[i*8:(i+1)*8])[0]
    return total


# ── Dataset loading ──────────────────────────────────────────────────

def load_planes(artifact_dir: Path, offset: int, count: int) -> list[bytes]:
    plane_files = sorted(artifact_dir.glob("plane_*.bin"))
    plane_map: dict[int, Path] = {}
    for pf in plane_files:
        try:
            plane_map[int(pf.stem.split("_")[-1])] = pf
        except (IndexError, ValueError):
            continue
    planes: list[bytes] = []
    for p in range(8):
        if p in plane_map:
            data = plane_map[p].read_bytes()
            planes.append(data[offset:offset + count])
        else:
            planes.append(bytes(count))
    return planes


def load_raw_f64le(path: Path, offset: int, count: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset * 8)
        return f.read(count * 8)


def derive_plane_seed(base: int, plane: int, rate: str, mode: str) -> int:
    h = hashlib.sha256(f"{base}:p{plane}:{rate}:{mode}".encode()).hexdigest()
    return int(h[:16], 16)


# ── Fault generation ─────────────────────────────────────────────────

def gen_uniform(n: int, rate: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    n_faults = int(rate * n)
    if n_faults == 0:
        return []
    offs = sorted(rng.sample(range(n), n_faults))
    masks = [rng.randint(1, 255) for _ in range(n_faults)]
    return [{"offset": o, "mask": m} for o, m in zip(offs, masks)]


def gen_cluster(n: int, rate: float, seed: int,
                seg_size: int = SEGMENT_SIZE) -> list[dict]:
    rng = random.Random(seed)
    n_faults = int(rate * n)
    if n_faults == 0:
        return []
    n_segs = (n + seg_size - 1) // seg_size
    n_cl = max(1, n_faults // 50)
    fpc = n_faults // n_cl
    entries: list[dict] = []
    for _ in range(n_cl):
        si = rng.randint(0, n_segs - 1)
        start = si * seg_size
        end = min(start + seg_size, n)
        avail = end - start
        n_here = min(fpc, avail)
        if n_here <= 0:
            continue
        for o in sorted(rng.sample(range(start, end), n_here)):
            entries.append({"offset": o, "mask": rng.randint(1, 255)})
    return entries[:n_faults]


def gen_burst(n: int, rate: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    n_faults = int(rate * n)
    if n_faults == 0:
        return []
    entries: list[dict] = []
    while len(entries) < n_faults:
        bs = rng.randint(0, n - 1)
        bl = min(rng.randint(4, 64), n - bs)
        for i in range(bl):
            if len(entries) >= n_faults:
                break
            entries.append({"offset": bs + i, "mask": rng.randint(1, 255)})
    return entries


def gen_adversarial(clean_bytes: bytes, n: int, rate: float, seed: int,
                    seg_size: int = SEGMENT_SIZE) -> list[dict]:
    rng = random.Random(seed)
    raw_n = int(rate * n)
    n_faults = max(2, (raw_n // 2) * 2)  # ensure even, minimum 2
    if n_faults == 0:
        return []
    n_segs = (n + seg_size - 1) // seg_size
    entries: list[dict] = []
    for si in range(n_segs):
        if len(entries) >= n_faults:
            break
        start = si * seg_size
        end = min(start + seg_size, n)
        if end - start < 2:
            continue
        ev: dict[int, list[int]] = {}
        od: dict[int, list[int]] = {}
        for i in range(start, end):
            v = clean_bytes[i]
            m4 = i % 4
            if v % 2 == 0:
                ev.setdefault(m4, []).append(i)
            else:
                od.setdefault(m4, []).append(i)
        for m4 in range(4):
            e = ev.get(m4, [])
            o = od.get(m4, [])
            pairs = min(len(e), len(o), (n_faults - len(entries)) // 2)
            for j in range(pairs):
                if len(entries) + 2 > n_faults:
                    break
                entries.append({"offset": e[j], "mask": 1})
                entries.append({"offset": o[j], "mask": 1})
    return entries


FAULT_GENS = {
    "uniform": gen_uniform,
    "cluster": gen_cluster,
    "burst": gen_burst,
    "adversarial_cancel": gen_adversarial,
}


# ── Clopper-Pearson (no scipy dependency) ────────────────────────────

def _binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0 if (p <= 0.0 and k > 0) or (p >= 1.0 and k < n) else 1.0
    import math as _m
    log_c = _m.lgamma(n + 1) - _m.lgamma(k + 1) - _m.lgamma(n - k + 1)
    return _m.exp(log_c + k * _m.log(p) + (n - k) * _m.log(1.0 - p))


def _clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """95% Clopper-Pearson CI for binomial proportion k/n.
    Pure Python, no scipy dependency.
    """
    if n == 0:
        return 0.0, 1.0
    alpha = 1.0 - conf
    lo = 0.0
    hi = 1.0
    if k > 0:
        lo = _cp_lower(k, n, alpha / 2)
    if k < n:
        hi = _cp_upper(k, n, alpha / 2)
    return lo, hi


def _cp_lower(k: int, n: int, tail: float) -> float:
    """Solve P(X >= k | p=L) = tail for L.
    P(X >= k) increases with p. So if s > tail, mid is too high.
    """
    if k == n:
        return tail ** (1.0 / n)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        s = sum(_binom_pmf(i, n, mid) for i in range(k, n + 1))
        if abs(s - tail) < 1e-12:
            return mid
        if s > tail:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def _cp_upper(k: int, n: int, tail: float) -> float:
    """Solve P(X <= k | p=U) = tail for U.
    P(X <= k) decreases with p. So if s > tail, mid is too low.
    """
    if k == 0:
        return 1.0 - tail ** (1.0 / n)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        s = sum(_binom_pmf(i, n, mid) for i in range(k + 1))
        if abs(s - tail) < 1e-12:
            return mid
        if s > tail:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ── Apply faults ─────────────────────────────────────────────────────

def apply_entries_to_raw(raw: bytes, entries: list[dict], n_rows: int) -> bytes:
    faulted = bytearray(raw)
    for e in entries:
        row = e["offset"]
        if row >= n_rows:
            continue
        byte_pos = e.get("byte_pos", row % 8)
        idx = row * 8 + byte_pos
        if idx < len(faulted):
            faulted[idx] ^= e["mask"]
    return bytes(faulted)


def gen_raw_entries(n: int, rate: float, seed: int) -> list[dict]:
    """Generate raw byte-level fault entries.
    Each entry corrupts one byte of one FP64 value.
    n = number of FP64 values, so n*8 total bytes.
    """
    rng = random.Random(seed)
    n_faults = int(rate * n)
    if n_faults == 0:
        return []
    rows = rng.sample(range(n), n_faults)
    byte_positions = [rng.randint(0, 7) for _ in range(n_faults)]
    masks = [rng.randint(1, 255) for _ in range(n_faults)]
    return [{"offset": r, "byte_pos": bp, "mask": m}
            for r, bp, m in zip(rows, byte_positions, masks)]


def apply_entries_to_plane(plane_data: bytes, entries: list[dict],
                           n_rows: int) -> bytes:
    faulted = bytearray(plane_data)
    for e in entries:
        o = e["offset"]
        if o >= n_rows:
            continue
        faulted[o] ^= e["mask"]
    return bytes(faulted)


def write_fault_plan(plan_dir: Path, plane: int, rate_str: str,
                     seed: int, entries: list[dict]) -> None:
    d = plan_dir / f"plane{plane}" / f"rate{rate_str}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"seed_{seed}.json").write_text(
        json.dumps({"metadata": {}, "entries": entries}) + "\n")


# ── Per-lane classification (one injection) ──────────────────────────

def classify_raw_unchecked_inj(raw: bytes, clean_sum: float,
                                entries: list[dict],
                                n_rows: int) -> tuple[str, float | None]:
    if not entries:
        return EVENT_CLEAN, clean_sum
    faulted = apply_entries_to_raw(raw, entries, n_rows)
    fs = raw_f64_sum(faulted)
    return (EVENT_SDC, fs) if fs != clean_sum else (EVENT_CLEAN, fs)


def classify_raw_digest_inj(raw: bytes, clean_sum: float, ref32: int,
                             entries: list[dict],
                             n_rows: int) -> tuple[str, float | None]:
    if not entries:
        return EVENT_CLEAN, clean_sum
    faulted = apply_entries_to_raw(raw, entries, n_rows)
    fs32 = sum32_raw(faulted)
    if fs32 != ref32:
        return EVENT_HARD_FAIL, None
    fs = raw_f64_sum(faulted)
    return (EVENT_SDC, fs) if fs != clean_sum else (EVENT_CLEAN, fs)


def classify_one_injection_bp(
    planes: list[bytes], entries: list[dict],
    plane: int, n_rows: int, ref32_planes: list[int],
    ds_name: str, rv: list[int], seed: int, rate_str: str,
) -> tuple[str, float | None]:
    if not entries:
        return EVENT_CLEAN, None
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        write_fault_plan(td_p, plane, rate_str, seed, entries)
        fp = str(td_p / f"plane{plane}" / f"rate{rate_str}" / f"seed_{seed}.json")
        fpp = {plane: [fp]}

        result = compute_delivered_answer_with_degradation(
            clean_planes=planes, fault_plan_paths=fpp,
            r_vector=rv, scale=1, n_rows=n_rows, dataset=ds_name,
            policy="graded",
            allocation_r="|".join(str(r) for r in rv),
            segment_size=SEGMENT_SIZE, seed=seed, fault_rate=rate_str,
        )
        voted = compute_voted_planes(planes, fpp, rv)
        detected = any(f != r for f, r in zip(per_plane_sum32(voted), ref32_planes))
        clean_ans = float(sum(sum(p) * PLANE_WEIGHTS[pi] for pi, p in enumerate(planes)))

    if not detected:
        return EVENT_UNDETECTED, result.delivered_answer
    lo = result.delivered_answer - result.bound_width / 2.0
    hi = result.delivered_answer + result.bound_width / 2.0
    if lo <= clean_ans <= hi:
        return EVENT_BOUNDED_DEGRADED, result.delivered_answer
    return EVENT_CERT_BOUND_FAILURE, result.delivered_answer


# ── Z2-B runner ──────────────────────────────────────────────────────

def run_z2b(artifact_dir: Path, raw_path: Path, raw_offset: int,
            n_rows: int, fault_rates: list[str], seeds: list[int],
            rv: list[int], ds_name: str) -> list[dict[str, Any]]:
    planes = load_planes(artifact_dir, raw_offset, n_rows)
    raw = load_raw_f64le(raw_path, raw_offset, n_rows)
    raw_clean_sum = raw_f64_sum(raw)
    raw_ref32 = sum32_raw(raw)
    ref32_planes = per_plane_sum32(planes)

    rows: list[dict[str, Any]] = []
    for rate_str in fault_rates:
        rate_val = float(rate_str)
        for seed in seeds:
            bp_events: list[str] = []
            for plane in range(8):
                p_seed = derive_plane_seed(seed, plane, rate_str, "uniform")
                entries = gen_uniform(n_rows, rate_val, p_seed)
                ev, _ = classify_one_injection_bp(
                    planes, entries, plane, n_rows, ref32_planes,
                    ds_name, rv, p_seed, rate_str,
                )
                bp_events.append(ev)

            raw_entries = gen_raw_entries(n_rows, rate_val, seed)
            ev_u, _ = classify_raw_unchecked_inj(raw, raw_clean_sum, raw_entries, n_rows)
            ev_d, _ = classify_raw_digest_inj(raw, raw_clean_sum, raw_ref32, raw_entries, n_rows)

            row: dict[str, Any] = {
                "dataset": ds_name, "n_rows": n_rows,
                "fault_rate": rate_str, "fault_mode": "uniform",
                "seed": str(seed), "n_injected_bp": len(bp_events),
                "n_injected_raw": len(raw_entries),
            }
            for k, v in aggregate_events(bp_events).items():
                row[f"{LANE_BYTEPLANE}_{k}"] = v
            for k, v in aggregate_events([ev_u]).items():
                row[f"{LANE_RAW_UNCHECKED}_{k}"] = v
            for k, v in aggregate_events([ev_d]).items():
                row[f"{LANE_RAW_DIGEST}_{k}"] = v
            rows.append(row)
    return rows


# ── Z2-C runner ──────────────────────────────────────────────────────

def run_z2c_escape(artifact_dir: Path, n_rows: int,
                   fault_rates: list[str], seeds: list[int],
                   rv: list[int], ds_name: str) -> list[dict[str, Any]]:
    planes = load_planes(artifact_dir, 0, n_rows)
    ref32_planes = per_plane_sum32(planes)
    modes = ["cluster", "burst", "adversarial_cancel"]

    rows: list[dict[str, Any]] = []
    for rate_str, seed, mode in itertools.product(fault_rates, seeds, modes):
        rate_val = float(rate_str)
        n_escaped = 0
        n_total = 0
        for plane in range(8):
            p_seed = derive_plane_seed(seed, plane, rate_str, mode)
            gen = FAULT_GENS[mode]
            if mode == "adversarial_cancel":
                entries = gen(planes[plane], n_rows, rate_val, p_seed)
            else:
                entries = gen(n_rows, rate_val, p_seed)
            if not entries:
                continue
            n_total += 1
            ev, _ = classify_one_injection_bp(
                planes, entries, plane, n_rows, ref32_planes,
                ds_name, rv, p_seed, rate_str,
            )
            if ev == EVENT_UNDETECTED:
                n_escaped += 1

        denom = max(n_total, 1)
        und_rate = n_escaped / denom
        lo_ci, hi_ci = _clopper_pearson(n_escaped, denom, 0.95)

        rows.append({
            "dataset": ds_name, "n_rows": n_rows,
            "fault_rate": rate_str, "fault_mode": mode,
            "seed": str(seed),
            "n_injected": n_total, "n_escaped": n_escaped,
            "undetected_rate": f"{und_rate:.10e}",
            "ci_95_lower": f"{lo_ci:.10e}",
            "ci_95_upper": f"{hi_ci:.10e}",
        })
    return rows


# ── CSV / main ───────────────────────────────────────────────────────

def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"  {'Appended' if exists else 'Wrote'} {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--raw-f64le", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-rows", type=int, default=1000000)
    parser.add_argument("--raw-offset", type=int, default=0)
    parser.add_argument("--fault-rates", nargs="+", default=["1e-06"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--experiment", default="z2b",
                    choices=["z2b", "z2c", "both"])
    parser.add_argument("--r-vector", type=int, nargs=8,
                    default=[1, 1, 1, 1, 1, 1, 1, 1])
    args = parser.parse_args()

    jid = os.environ.get("SLURM_JOB_ID", "cpu_local")
    out_root = Path(f"results/reliability_layer1/phase3/phase3z_z2/job_{jid}")

    lane_prefix = {
        LANE_RAW_UNCHECKED: "raw_fp64_unchecked",
        LANE_RAW_DIGEST: "raw_fp64_digest_hard_fail",
        LANE_BYTEPLANE: "byteplane_sum32",
    }
    metric_keys = ["sdc_rate", "cert_bound_failure_rate", "detected_rate",
                   "bounded_degraded_rate", "hard_fail_rate",
                   "undetected_rate", "clean_rate"]

    if args.experiment in ("z2b", "both"):
        print(f"\n=== Z2-B: {args.dataset} n_rows={args.n_rows} "
              f"rates={args.fault_rates} seeds={args.seeds} ===")
        rows = run_z2b(
            artifact_dir=args.artifact_dir, raw_path=args.raw_f64le,
            raw_offset=args.raw_offset, n_rows=args.n_rows,
            fault_rates=args.fault_rates, seeds=args.seeds,
            rv=args.r_vector, ds_name=args.dataset,
        )
        z2b_fields = ["dataset", "n_rows", "fault_rate", "fault_mode",
                       "seed", "n_injected_bp", "n_injected_raw"]
        for ln in [LANE_BYTEPLANE, LANE_RAW_UNCHECKED, LANE_RAW_DIGEST]:
            for mk in metric_keys:
                z2b_fields.append(f"{lane_prefix[ln]}_{mk}")
        write_csv(out_root / "z2b_sdc_summary.csv", z2b_fields, rows)

    if args.experiment in ("z2c", "both"):
        print(f"\n=== Z2-C: {args.dataset} n_rows={args.n_rows} "
              f"rates={args.fault_rates} seeds={args.seeds} ===")
        rows = run_z2c_escape(
            artifact_dir=args.artifact_dir, n_rows=args.n_rows,
            fault_rates=args.fault_rates, seeds=args.seeds,
            rv=args.r_vector, ds_name=args.dataset,
        )
        z2c_fields = [
            "dataset", "n_rows", "fault_rate", "fault_mode", "seed",
            "n_injected", "n_escaped",
            "undetected_rate", "ci_95_lower", "ci_95_upper",
        ]
        write_csv(out_root / "z2c_escape_rate.csv", z2c_fields, rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
