#!/usr/bin/env python3
"""Phase 3-Z2: SDC Containment vs Raw Baselines.

Classifies each injected fault into event classes per PRD §4 taxonomy.

Detection primitive: artifact-time SUM32 reference + runtime parallel SUM32
recompute (NEVER called "CRC" or "cryptographic").  Uses parallel SUM32
warp/block reduction as validated in Z0f.

Three lanes compared:
  raw_fp64_unchecked       no integrity check, any fault → silent wrong answer
  raw_fp64_digest_hard_fail  SUM32 detect → hard fail / recompute (no bound)
  byteplane_sum32            SUM32 detect → certified bounded degradation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_oracle import apply_fault_plan
from phase3_y0_evaluator import (
    compute_delivered_answer_with_degradation,
    compute_clean_sum,
    compute_voted_planes,
    load_clean_planes,
    load_artifact_metadata,
)
from build_reliability_tiny_fixture import generate_fault_plans, SCALE, TINY_N

SEGMENT_SIZE = 1024

LANE_RAW_UNCHECKED = "raw_fp64_unchecked"
LANE_RAW_DIGEST = "raw_fp64_digest_hard_fail"
LANE_BYTEPLANE = "byteplane_sum32"

EVENT_SDC = "sdc_event"
EVENT_CERT_BOUND_FAILURE = "cert_bound_failure"
EVENT_HARD_FAIL = "hard_fail_event"
EVENT_BOUNDED_DEGRADED = "bounded_degraded_event"
EVENT_CLEAN = "clean"
EVENT_UNDETECTED = "undetected"

ALL_EVENTS = [EVENT_SDC, EVENT_CERT_BOUND_FAILURE, EVENT_HARD_FAIL,
              EVENT_BOUNDED_DEGRADED, EVENT_CLEAN, EVENT_UNDETECTED]


def sum32(data: bytes) -> int:
    """Parallel-safe SUM32 digest: sum of uint32 words, mod 2^32."""
    padded = data + b'\x00' * ((4 - len(data) % 4) % 4)
    total = 0
    for i in range(0, len(padded), 4):
        total += struct.unpack('<I', padded[i:i+4])[0]
    return total & 0xFFFFFFFF


def per_plane_sum32(planes: list[bytes]) -> list[int]:
    return [sum32(p) for p in planes]


def _event_label(e: str) -> str:
    return {
        EVENT_SDC: "SDC",
        EVENT_CERT_BOUND_FAILURE: "CertBoundFailure",
        EVENT_HARD_FAIL: "HardFail",
        EVENT_BOUNDED_DEGRADED: "BoundedDegraded",
        EVENT_CLEAN: "Clean",
        EVENT_UNDETECTED: "Undetected",
    }.get(e, e)


# ── Per-lane classifiers ──────────────────────────────────────────

def classify_raw_unchecked(
    clean_planes: list[bytes],
    fault_plan_paths: dict[int, list[str]],
    scale: int,
) -> tuple[str, float | None]:
    """raw_fp64_unchecked: no detection, any byte change → SDC."""
    faulted: list[bytes] = []
    for p in range(8):
        paths = fault_plan_paths.get(p, [])
        faulted.append(apply_fault_plan(clean_planes[p], paths[0]) if paths else clean_planes[p])

    clean_ans = compute_clean_sum(clean_planes) / scale
    fault_ans = compute_clean_sum(faulted) / scale
    return (EVENT_SDC, fault_ans) if fault_ans != clean_ans else (EVENT_CLEAN, fault_ans)


def classify_raw_digest(
    clean_planes: list[bytes],
    fault_plan_paths: dict[int, list[str]],
    scale: int,
    ref_sum32: list[int],
) -> tuple[str, float | None]:
    """raw_fp64_digest_hard_fail: SUM32 detection → hard fail; escape → SDC."""
    faulted: list[bytes] = []
    for p in range(8):
        paths = fault_plan_paths.get(p, [])
        faulted.append(apply_fault_plan(clean_planes[p], paths[0]) if paths else clean_planes[p])

    if any(f != r for f, r in zip(per_plane_sum32(faulted), ref_sum32)):
        return EVENT_HARD_FAIL, None

    clean_ans = compute_clean_sum(clean_planes) / scale
    fault_ans = compute_clean_sum(faulted) / scale
    return (EVENT_SDC, fault_ans) if fault_ans != clean_ans else (EVENT_CLEAN, fault_ans)


def classify_byteplane(
    clean_planes: list[bytes],
    fault_plan_paths: dict[int, list[str]],
    r_vector: list[int],
    scale: int,
    n_rows: int,
    ref_sum32: list[int],
    segment_size: int = SEGMENT_SIZE,
    **y0_kw: Any,
) -> tuple[str, float | None]:
    """byteplane_sum32: SUM32 detection + certified bounded degrade (Y0 pipeline)."""
    result = compute_delivered_answer_with_degradation(
        clean_planes=clean_planes,
        fault_plan_paths=fault_plan_paths,
        r_vector=r_vector,
        scale=scale,
        n_rows=n_rows,
        segment_size=segment_size,
        **y0_kw,
    )

    faulted_voted = compute_voted_planes(clean_planes, fault_plan_paths, r_vector)
    detected = any(f != r for f, r in zip(per_plane_sum32(faulted_voted), ref_sum32))

    clean_ans = compute_clean_sum(clean_planes) / scale
    lo = result.delivered_answer - result.bound_width / 2.0
    hi = result.delivered_answer + result.bound_width / 2.0
    contains = lo <= clean_ans <= hi

    if not detected:
        return EVENT_UNDETECTED, result.delivered_answer
    return (EVENT_BOUNDED_DEGRADED, result.delivered_answer) if contains else (EVENT_CERT_BOUND_FAILURE, result.delivered_answer)


# ── Metrics ────────────────────────────────────────────────────────

def aggregate_events(events: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {e: 0 for e in ALL_EVENTS}
    for e in events:
        counts[e] = counts.get(e, 0) + 1
    total = len(events) or 1

    sdc = counts[EVENT_SDC]
    cbf = counts[EVENT_CERT_BOUND_FAILURE]
    hf = counts[EVENT_HARD_FAIL]
    bd = counts[EVENT_BOUNDED_DEGRADED]
    und = counts[EVENT_UNDETECTED]

    return {
        "sdc_rate": sdc / total,
        "cert_bound_failure_rate": cbf / total,
        "detected_rate": (hf + bd + cbf) / total,
        "bounded_degraded_rate": bd / total,
        "hard_fail_rate": hf / total,
        "undetected_rate": und / total,
        "clean_rate": counts[EVENT_CLEAN] / total,
    }


# ── Smoke test ─────────────────────────────────────────────────────

def run_smoke(
    artifact_dir: Path,
    fault_plan_dir: Path,
    dataset: str,
    n_rows: int,
    scale: int,
    r_vector: list[int],
    policy: str,
    seeds: list[int],
    fault_rate: str,
    segment_size: int = SEGMENT_SIZE,
) -> dict[str, Any]:
    clean_planes = load_clean_planes(artifact_dir, n_rows)
    ref_sum32 = per_plane_sum32(clean_planes)
    allocation_r = "|".join(str(r) for r in r_vector)

    per_lane: dict[str, list[dict]] = {
        LANE_RAW_UNCHECKED: [],
        LANE_RAW_DIGEST: [],
        LANE_BYTEPLANE: [],
    }

    for plane in range(8):
        for seed in seeds:
            paths: dict[int, list[str]] = {}
            for rep in range(r_vector[plane]):
                fp = (
                    fault_plan_dir / f"plane{plane}" / f"rate{fault_rate}"
                    / f"seed_{seed}.json"
                )
                if fp.is_file():
                    paths.setdefault(plane, []).append(str(fp))
            if not paths:
                continue

            ev, ans = classify_raw_unchecked(clean_planes, paths, scale)
            per_lane[LANE_RAW_UNCHECKED].append({"plane": plane, "seed": seed, "event": ev, "answer": ans})

            ev, ans = classify_raw_digest(clean_planes, paths, scale, ref_sum32)
            per_lane[LANE_RAW_DIGEST].append({"plane": plane, "seed": seed, "event": ev, "answer": ans})

            ev, ans = classify_byteplane(
                clean_planes, paths, r_vector, scale, n_rows, ref_sum32,
                segment_size=segment_size, dataset=dataset, policy=policy,
                allocation_r=allocation_r, seed=seed, fault_rate=fault_rate,
            )
            per_lane[LANE_BYTEPLANE].append({"plane": plane, "seed": seed, "event": ev, "answer": ans})

    return {
        "dataset": dataset, "n_rows": n_rows, "scale": scale,
        "policy": policy, "allocation_r": allocation_r,
        "fault_rate": fault_rate, "r_vector": r_vector, "seeds": seeds,
        "per_lane_events": per_lane,
        "lane_metrics": {
            lane: aggregate_events([e["event"] for e in events])
            for lane, events in per_lane.items()
        },
    }


def print_results(result: dict[str, Any]) -> None:
    print(f"\ndataset={result['dataset']}  n_rows={result['n_rows']}  scale={result['scale']}")
    print(f"policy={result['policy']}  r_vector={result['r_vector']}  fault_rate={result['fault_rate']}")
    for lane in [LANE_RAW_UNCHECKED, LANE_RAW_DIGEST, LANE_BYTEPLANE]:
        m = result["lane_metrics"][lane]
        events = result["per_lane_events"][lane]
        print(f"\n  [{lane}]")
        print(f"    total_faults:           {len(events)}")
        for k in ("sdc_rate", "cert_bound_failure_rate", "detected_rate",
                   "bounded_degraded_rate", "hard_fail_rate", "undetected_rate", "clean_rate"):
            print(f"    {k:28s}  {m[k]:.10e}")
        for ev in events:
            lbl = _event_label(ev["event"])
            ans = f"  ans={ev['answer']:.10f}" if ev["answer"] is not None else ""
            print(f"      plane={ev['plane']} seed={ev['seed']} → {lbl}{ans}")


def write_csv(csv_path: Path, results: list[dict[str, Any]]) -> None:
    fields = ["dataset", "n_rows", "scale", "policy", "allocation_r",
              "fault_rate", "lane", "sdc_rate", "cert_bound_failure_rate",
              "detected_rate", "bounded_degraded_rate", "hard_fail_rate",
              "undetected_rate", "clean_rate", "total_faults"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            for lane in [LANE_RAW_UNCHECKED, LANE_RAW_DIGEST, LANE_BYTEPLANE]:
                m = r["lane_metrics"][lane]
                events = r["per_lane_events"][lane]
                row = {"dataset": r["dataset"], "n_rows": str(r["n_rows"]),
                       "scale": str(r["scale"]), "policy": r["policy"],
                       "allocation_r": r["allocation_r"],
                       "fault_rate": r["fault_rate"], "lane": lane,
                       "total_faults": str(len(events))}
                row.update({k: f"{m[k]:.10e}" for k in fields if k in m})
                w.writerow(row)
    print(f"CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("/tmp/z2_tiny/artifacts"))
    parser.add_argument("--fault-plan-dir", type=Path, default=Path("/tmp/z2_tiny/fault_plans"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--dataset", default="tiny_fixture")
    parser.add_argument("--n-rows", type=int, default=0)
    parser.add_argument("--scale", type=int, default=100)
    parser.add_argument("--policy", default="graded", choices=["graded", "uniform_repair_fraction"])
    parser.add_argument("--r-vector", type=int, nargs=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--fault-rate", default="1e-03")
    parser.add_argument("--segment-size", type=int, default=SEGMENT_SIZE)
    parser.add_argument("--build-fixture", action="store_true")
    args = parser.parse_args()

    if args.build_fixture:
        from build_reliability_tiny_fixture import (
            generate_tiny_fixture_raw, convert_to_planes,
            generate_fault_plans, ENCODED_FIXTURE_VALUES, FP64_ROUNDTRIPPABLE,
        )
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        raw_path = args.artifact_dir / "tiny_raw.f64le.bin"
        raw_vals = generate_tiny_fixture_raw(raw_path)
        meta = convert_to_planes(raw_vals, SCALE, args.artifact_dir, raw_path)
        print(f"Fixture built: n_rows={meta['n_rows']} sum={meta['clean_encoded_sum']}")
        target_planes = list(range(8))
        fault_rates_float = [float(args.fault_rate)]
        generate_fault_plans(args.artifact_dir, args.fault_plan_dir,
                             target_planes, fault_rates_float, args.seeds, meta)
        print(f"Fault plans: {args.fault_plan_dir}")
        if not args.n_rows:
            args.n_rows = meta["n_rows"]
        if args.scale == 100:
            args.scale = SCALE

    if args.r_vector is None:
        args.r_vector = [3, 2, 1, 1, 1, 1, 1, 1]

    meta = load_artifact_metadata(args.artifact_dir)
    nr = args.n_rows or meta["n_rows"]
    sc = args.scale or meta.get("scale", 100)

    result = run_smoke(
        artifact_dir=args.artifact_dir,
        fault_plan_dir=args.fault_plan_dir,
        dataset=args.dataset,
        n_rows=nr,
        scale=sc,
        r_vector=args.r_vector,
        policy=args.policy,
        seeds=args.seeds,
        fault_rate=args.fault_rate,
        segment_size=args.segment_size,
    )

    print("\n" + "=" * 72)
    print("Phase 3-Z2 SDC Containment — CPU Smoke")
    print("=" * 72)
    print_results(result)

    # ── Verdict ──
    bp_sdc = result["lane_metrics"][LANE_BYTEPLANE]["sdc_rate"]
    bp_cbf = result["lane_metrics"][LANE_BYTEPLANE]["cert_bound_failure_rate"]
    raw_sdc = result["lane_metrics"][LANE_RAW_UNCHECKED]["sdc_rate"]

    print(f"\n--- Verdict ---")
    print(f"byteplane sdc_rate:             {bp_sdc:.10e}")
    print(f"byteplane cert_bound_fail_rate:  {bp_cbf:.10e}")
    print(f"raw_fp64_unchecked sdc_rate:    {raw_sdc:.10e}")

    if bp_sdc > 0:
        verdict = "SDC_PARTIAL_RESIDUAL"
    elif bp_cbf > 0:
        verdict = "SDC_PARTIAL_RESIDUAL"
    else:
        verdict = "SDC_CONTAINED"
    print(f"Verdict: {verdict}")

    csv_path = args.csv
    if csv_path:
        write_csv(csv_path, [result])
    else:
        jid = os.environ.get("SLURM_JOB_ID", "cpu_smoke")
        write_csv(Path(f"results/reliability_layer1/phase3/phase3z_z2/job_{jid}/z2_sdc_summary.csv"), [result])

    if verdict != "SDC_CONTAINED":
        sys.exit(1)


if __name__ == "__main__":
    main()
