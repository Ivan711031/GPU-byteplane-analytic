#!/usr/bin/env python3
"""NMR-A2-GPU: structured fault injection through the H200 NMR pipeline.

This runner keeps the existing structured software fault families from NMR-A,
but executes them through the current GPU NMR path on H200 using explicit fault
plans shared by a CPU oracle and the GPU benchmark.

Scope boundary:
- HBM-inspired software injection only.
- Not physical HBM SID / channel / bank validation.
- Pre-fused GPU path only; kernel fusion belongs to issue #189.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PLANE_COUNT = 8
PLANE_WEIGHTS = [1 << (8 * (7 - plane)) for plane in range(PLANE_COUNT)]

DATASET_PATHS: dict[str, str] = {
    "hurricane_u": "/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg4096",
    "cesm_atm_cloud": "/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg4096",
}

POLICIES: dict[str, list[int]] = {
    "naive_same_region_replication_r3": [3, 1, 1, 1, 1, 1, 1, 1],
    "logical_striping_diversity_r3": [3, 1, 1, 1, 1, 1, 1, 1],
    "graded_nmr_B11": [3, 2, 1, 1, 1, 1, 1, 1],
    "uniformish_B11": [2, 1, 2, 1, 2, 1, 1, 1],
    "certified_degradation_fallback": [1, 1, 1, 1, 1, 1, 1, 1],
}

SEEDS = [0, 1, 2]
N_ROWS = 500_000

FAULT_SCENARIOS: list[dict[str, Any]] = [
    {
        "scenario_id": 0,
        "fault_family": "single_domain_burst",
        "plane_label": "msb",
        "plane": 0,
        "burst_length": 4,
        "replica_mode": "policy_domain_aware",
    },
    {
        "scenario_id": 1,
        "fault_family": "regional_multi_byte",
        "plane_label": "msb",
        "plane": 0,
        "regional_cluster_size": 64,
        "replica_mode": "policy_domain_aware",
    },
    {
        "scenario_id": 2,
        "fault_family": "column_like_repeated_offset",
        "plane_label": "msb",
        "plane": 0,
        "column_like_stride": 4096,
        "repeat_count": 8,
        "replica_mode": "policy_domain_aware",
    },
    {
        "scenario_id": 3,
        "fault_family": "plane_localized_msb",
        "plane_label": "msb",
        "plane": 0,
        "burst_length": 8,
        "replica_mode": "single_replica",
    },
    {
        "scenario_id": 4,
        "fault_family": "plane_localized_lsb",
        "plane_label": "lsb",
        "plane": 7,
        "burst_length": 8,
        "replica_mode": "single_replica",
    },
    {
        "scenario_id": 5,
        "fault_family": "same_fault_all_correlated_control",
        "plane_label": "msb",
        "plane": 0,
        "burst_length": 4,
        "replica_mode": "same_fault_all",
    },
    {
        "scenario_id": 6,
        "fault_family": "mixed_independent_plus_correlated",
        "plane_label": "mixed",
        "plane": 0,
        "burst_length": 4,
        "replica_mode": "mixed",
    },
]

GPU_RAW_FIELDS = [
    "dataset",
    "n_rows",
    "path",
    "replica_policy",
    "r_vector",
    "fault_mode",
    "fault_plane",
    "seed",
    "latency_ms",
    "contains_truth",
    "recovered_rate",
    "detected_rate",
    "uncertified_rate",
    "certified_availability",
    "bound_width",
    "bound_width_norm",
    "vote_recovered",
    "fallback_used",
    "same_fault_false_recovery",
    "silent_wrong",
    "cert_bound_failure",
    "hard_fail",
    "bounded_degrade",
    "cpu_gpu_classification_match",
    "verdict_cell",
]

CANONICAL_FIELDS = [
    "dataset",
    "n_rows",
    "policy",
    "r_vector",
    "fault_family",
    "plane_label",
    "seed",
    "fault_entry_count",
    "affected_planes",
    "affected_replicas",
    "replicas_hit_count",
    "planes_hit_count",
    "high_significance_hit",
    "replica_loss_correlation",
    "latency_ms",
    "contains_truth",
    "recovered_rate",
    "detected_rate",
    "uncertified_rate",
    "certified_availability",
    "bound_width",
    "bound_width_norm",
    "vote_recovered",
    "same_fault_false_recovery",
    "silent_wrong",
    "cert_bound_failure",
    "hard_fail",
    "bounded_degrade",
    "cpu_gpu_classification_match",
    "verdict_cell",
]

PLAN_FIELDS = [
    "dataset",
    "policy",
    "fault_family",
    "plane_label",
    "seed",
    "fault_entry_count",
    "affected_planes",
    "affected_replicas",
    "replicas_hit_count",
    "planes_hit_count",
    "high_significance_hit",
    "replica_loss_correlation",
]

POLICY_SUMMARY_FIELDS = [
    "dataset",
    "policy",
    "fault_family",
    "n_cells",
    "classification_match_rate",
    "contains_truth_rate",
    "recovered_rate",
    "certified_bounded_rate",
    "uncertified_or_unrecoverable_rate",
    "same_fault_false_recovery_count",
    "cert_bound_failure_count",
    "hard_fail_count",
]

FAMILY_SUMMARY_FIELDS = [
    "dataset",
    "fault_family",
    "n_cells",
    "classification_match_rate",
    "contains_truth_rate",
    "recovered_rate",
    "certified_bounded_rate",
    "uncertified_or_unrecoverable_rate",
    "same_fault_false_recovery_count",
    "cert_bound_failure_count",
    "hard_fail_count",
]

ORACLE_FIELDS = [
    "dataset",
    "n_rows",
    "policy",
    "r_vector",
    "fault_family",
    "seed",
    "cpu_outcome",
    "gpu_outcome",
    "classification_match",
    "cpu_contains_truth",
    "cpu_detected",
]


def load_planes(artifact_dir: Path, n_rows: int) -> list[bytes]:
    planes: list[bytes] = []
    for plane in range(PLANE_COUNT):
        path = artifact_dir / f"plane_{plane:03d}.bin"
        if path.is_file():
            data = path.read_bytes()[:n_rows]
        else:
            data = bytes(n_rows)
        if len(data) < n_rows:
            data = data + bytes(n_rows - len(data))
        planes.append(data)
    return planes


def available_plane_indices(artifact_dir: Path) -> list[int]:
    return [plane for plane in range(PLANE_COUNT) if (artifact_dir / f"plane_{plane:03d}.bin").is_file()]


def sum32(data: bytes) -> int:
    # Match the GPU SUM32 primitive used by bench_nmr_b_e2e_pipeline.
    return sum(data) & 0xFFFFFFFF


def per_plane_sum32(planes: list[bytes]) -> list[int]:
    return [sum32(plane) for plane in planes]


def byte_majority_vote(replicas: list[list[bytearray]], r_vector: list[int]) -> list[bytes]:
    voted: list[bytes] = []
    for plane in range(PLANE_COUNT):
        replica_count = r_vector[plane]
        if replica_count <= 1:
            voted.append(bytes(replicas[plane][0]) if replica_count == 1 else bytes())
            continue
        n = len(replicas[plane][0])
        result = bytearray(n)
        for index in range(n):
            vals = [replicas[plane][replica][index] for replica in range(replica_count)]
            result[index] = max(set(vals), key=lambda value: (vals.count(value), value))
        voted.append(bytes(result))
    return voted


def compute_encoded_sum(planes: list[bytes]) -> float:
    return float(sum(sum(plane) * PLANE_WEIGHTS[index] for index, plane in enumerate(planes)))


def compute_bound_width(detected_planes: list[bool], n_rows: int) -> float:
    return float(
        sum(255.0 * PLANE_WEIGHTS[plane] * n_rows for plane in range(PLANE_COUNT) if detected_planes[plane])
    )


def validate_fault_plan(
    fault_plan: list[dict[str, Any]],
    plane_lengths: list[int],
    r_vector: list[int],
) -> None:
    errors: list[str] = []
    for index, entry in enumerate(fault_plan):
        plane = int(entry["plane"])
        replica = int(entry["replica"])
        offset = int(entry["offset"])

        if plane < 0 or plane >= len(plane_lengths):
            errors.append(f"entry[{index}] plane={plane} out of range")
            continue

        replica_limit = r_vector[plane] if plane < len(r_vector) else 1
        if replica < 0 or replica >= max(replica_limit, 1):
            errors.append(f"entry[{index}] plane={plane} replica={replica} out of range")

        plane_length = plane_lengths[plane]
        if offset < 0 or offset >= plane_length:
            errors.append(
                f"entry[{index}] plane={plane} offset={offset} out of range for plane_length={plane_length}"
            )

    if errors:
        preview = "; ".join(errors[:5])
        raise ValueError(f"invalid fault plan: {preview}")


def cpu_oracle(
    clean_planes: list[bytes],
    r_vector: list[int],
    fault_plan: list[dict[str, Any]],
    n_rows: int,
) -> dict[str, Any]:
    validate_fault_plan(fault_plan, [len(plane) for plane in clean_planes], r_vector)
    clean_digests = per_plane_sum32(clean_planes)

    replicas: list[list[bytearray]] = []
    for plane in range(PLANE_COUNT):
        replica_count = r_vector[plane] if plane < len(r_vector) else 1
        replicas.append([bytearray(clean_planes[plane]) for _ in range(replica_count)])

    for entry in fault_plan:
        plane = entry["plane"]
        replica = entry["replica"]
        offset = entry["offset"]
        mask = entry["mask"]
        replicas[plane][replica][offset] ^= mask

    voted = byte_majority_vote(replicas, r_vector)
    voted_digests = per_plane_sum32(voted)

    detected_planes = [voted_digest != clean_digest for voted_digest, clean_digest in zip(voted_digests, clean_digests)]
    detected = any(detected_planes)
    vote_recovered = not detected

    clean_answer = compute_encoded_sum(clean_planes)
    delivered_answer = compute_encoded_sum(voted)

    if vote_recovered:
        bound_width = 0.0
        contains_truth = True
        outcome = "vote_recovered"
    else:
        bound_width = compute_bound_width(detected_planes, n_rows)
        lo = delivered_answer - bound_width / 2.0
        hi = delivered_answer + bound_width / 2.0
        contains_truth = lo <= clean_answer <= hi
        outcome = "bounded_degraded" if contains_truth else "cert_bound_failure"

    return {
        "vote_recovered": vote_recovered,
        "detected": detected,
        "contains_truth": contains_truth,
        "outcome": outcome,
        "bound_width": bound_width,
        "detected_planes": [plane for plane, hit in enumerate(detected_planes) if hit],
    }


def replica_count_for_plane(r_vector: list[int], plane: int) -> int:
    if plane >= len(r_vector):
        return 1
    return max(r_vector[plane], 1)


def contiguous_offsets(rng: random.Random, n_rows: int, length: int) -> list[int]:
    if length <= 0:
        return []
    base = rng.randint(0, max(n_rows - length, 0))
    return [base + delta for delta in range(length)]


def column_offsets(rng: random.Random, n_rows: int, stride: int, repeat_count: int) -> list[int]:
    stride = max(stride, 1)
    if repeat_count <= 0:
        return []
    base = rng.randint(0, min(stride - 1, max(n_rows - 1, 0)))
    offsets: list[int] = []
    for index in range(repeat_count):
        offset = base + index * stride
        if offset >= n_rows:
            break
        offsets.append(offset)
    return offsets


def policy_domain_targets(
    policy_name: str,
    r_vector: list[int],
    plane: int,
    rng: random.Random,
) -> list[int]:
    replica_count = replica_count_for_plane(r_vector, plane)
    if replica_count <= 1:
        return [0]
    if policy_name == "naive_same_region_replication_r3":
        return list(range(replica_count))
    return [rng.randrange(replica_count)]


def single_replica_target(r_vector: list[int], plane: int, rng: random.Random) -> list[int]:
    replica_count = replica_count_for_plane(r_vector, plane)
    if replica_count <= 1:
        return [0]
    return [rng.randrange(replica_count)]


def add_entries(
    plan: list[dict[str, Any]],
    *,
    plane: int,
    replicas: list[int],
    offsets: list[int],
    mask: int,
) -> None:
    for offset in offsets:
        for replica in replicas:
            plan.append({
                "plane": plane,
                "replica": replica,
                "offset": offset,
                "mask": mask,
            })


def generate_fault_plan(
    policy_name: str,
    scenario: dict[str, Any],
    seed: int,
    r_vector: list[int],
    n_rows: int,
    lsb_plane: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed * 1000 + scenario["scenario_id"])
    plane = lsb_plane if scenario["plane_label"] == "lsb" else int(scenario["plane"])
    fault_family = scenario["fault_family"]
    plan: list[dict[str, Any]] = []

    if fault_family == "single_domain_burst":
        offsets = contiguous_offsets(rng, n_rows, int(scenario["burst_length"]))
        add_entries(
            plan,
            plane=plane,
            replicas=policy_domain_targets(policy_name, r_vector, plane, rng),
            offsets=offsets,
            mask=0xFF,
        )
    elif fault_family == "regional_multi_byte":
        offsets = contiguous_offsets(rng, n_rows, int(scenario["regional_cluster_size"]))
        add_entries(
            plan,
            plane=plane,
            replicas=policy_domain_targets(policy_name, r_vector, plane, rng),
            offsets=offsets,
            mask=0xFF,
        )
    elif fault_family == "column_like_repeated_offset":
        offsets = column_offsets(
            rng,
            n_rows,
            int(scenario["column_like_stride"]),
            int(scenario["repeat_count"]),
        )
        add_entries(
            plan,
            plane=plane,
            replicas=policy_domain_targets(policy_name, r_vector, plane, rng),
            offsets=offsets,
            mask=0xFF,
        )
    elif fault_family in {"plane_localized_msb", "plane_localized_lsb"}:
        offsets = contiguous_offsets(rng, n_rows, int(scenario["burst_length"]))
        add_entries(
            plan,
            plane=plane,
            replicas=single_replica_target(r_vector, plane, rng),
            offsets=offsets,
            mask=0xFF,
        )
    elif fault_family == "same_fault_all_correlated_control":
        offsets = contiguous_offsets(rng, n_rows, int(scenario["burst_length"]))
        add_entries(
            plan,
            plane=plane,
            replicas=list(range(replica_count_for_plane(r_vector, plane))),
            offsets=offsets,
            mask=0xFF,
        )
    elif fault_family == "mixed_independent_plus_correlated":
        correlated_offsets = contiguous_offsets(rng, n_rows, int(scenario["burst_length"]))
        add_entries(
            plan,
            plane=0,
            replicas=list(range(replica_count_for_plane(r_vector, 0))),
            offsets=correlated_offsets,
            mask=0xFF,
        )
        lsb_rng = random.Random(seed * 1000 + scenario["scenario_id"] + 77)
        independent_offsets = contiguous_offsets(lsb_rng, n_rows, 4)
        add_entries(
            plan,
            plane=lsb_plane,
            replicas=single_replica_target(r_vector, lsb_plane, lsb_rng),
            offsets=independent_offsets,
            mask=0x55,
        )
    else:
        raise ValueError(f"unsupported fault family: {fault_family}")

    return plan


def summarize_fault_plan(
    fault_plan: list[dict[str, Any]],
    r_vector: list[int],
) -> dict[str, Any]:
    planes = sorted({entry["plane"] for entry in fault_plan})
    per_plane_replicas: dict[int, set[int]] = defaultdict(set)
    for entry in fault_plan:
        per_plane_replicas[entry["plane"]].add(entry["replica"])

    replica_labels = []
    for plane in sorted(per_plane_replicas):
        for replica in sorted(per_plane_replicas[plane]):
            replica_labels.append(f"p{plane}:r{replica}")

    replica_loss_correlation = 0.0
    for plane, replicas in per_plane_replicas.items():
        replica_loss_correlation = max(
            replica_loss_correlation,
            len(replicas) / replica_count_for_plane(r_vector, plane),
        )

    return {
        "fault_entry_count": len(fault_plan),
        "affected_planes": "|".join(str(plane) for plane in planes) if planes else "NA",
        "affected_replicas": "|".join(replica_labels) if replica_labels else "NA",
        "replicas_hit_count": sum(len(replicas) for replicas in per_plane_replicas.values()),
        "planes_hit_count": len(planes),
        "high_significance_hit": 0 in planes,
        "replica_loss_correlation": replica_loss_correlation,
    }


def default_gpu_row(verdict: str = "hard_fail") -> dict[str, str]:
    return {
        "dataset": "unknown",
        "n_rows": "0",
        "path": "gpu_structured_fault",
        "replica_policy": "gpu_structured_fault",
        "r_vector": "",
        "fault_mode": "unknown",
        "fault_plane": "NA",
        "seed": "0",
        "latency_ms": "NA",
        "contains_truth": "0.0",
        "recovered_rate": "0.0",
        "detected_rate": "0.0",
        "uncertified_rate": "1.0" if verdict == "cert_bound_failure" else "0.0",
        "certified_availability": "0.0",
        "bound_width": "NA",
        "bound_width_norm": "NA",
        "vote_recovered": "false",
        "fallback_used": "false",
        "same_fault_false_recovery": "false",
        "silent_wrong": "true" if verdict == "silent_wrong" else "false",
        "cert_bound_failure": "true" if verdict == "cert_bound_failure" else "false",
        "hard_fail": "true" if verdict == "hard_fail" else "false",
        "bounded_degrade": "true" if verdict == "bounded_degraded" else "false",
        "cpu_gpu_classification_match": "NA",
        "verdict_cell": verdict,
    }


def parse_gpu_csv_row(line: str) -> dict[str, str]:
    parts = next(csv.reader([line]))
    if len(parts) != len(GPU_RAW_FIELDS):
        raise ValueError(f"unexpected GPU CSV width: got {len(parts)} expected {len(GPU_RAW_FIELDS)}")
    return dict(zip(GPU_RAW_FIELDS, parts))


def primary_fault_plane(fault_plan: list[dict[str, Any]]) -> int:
    if not fault_plan:
        return 0
    return int(fault_plan[0]["plane"])


def gpu_runner_one(
    gpu_bench: Path,
    dataset_path: str,
    r_vector: list[int],
    fault_plan_path: str,
    raw_csv_path: Path,
    *,
    fault_family: str,
    seed: int,
    n_rows: int,
    timeout_s: int,
    fault_plane: int,
) -> dict[str, str]:
    r_vec_str = "|".join(str(replica_count) for replica_count in r_vector)
    cmd = [
        str(gpu_bench),
        "--dataset", dataset_path,
        "--r-vector", r_vec_str,
        "--fault-plan", fault_plan_path,
        "--fault-mode", fault_family,
        "--fault-plane", str(fault_plane),
        "--seed", str(seed),
        "--csv", str(raw_csv_path),
        "--iters", "10",
        "--n-rows", str(n_rows),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"  GPU FAILED: timeout after {timeout_s}s")
        return default_gpu_row("hard_fail")

    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip().replace("\n", " ")
        print(f"  GPU FAILED ({result.returncode}): {msg[:400]}")
        return default_gpu_row("hard_fail")

    lines = raw_csv_path.read_text().splitlines()
    for line in reversed(lines):
        if line.strip() and not line.startswith("dataset,"):
            return parse_gpu_csv_row(line.strip())
    return default_gpu_row("hard_fail")


def build_canonical_row(
    *,
    ds_name: str,
    n_rows: int,
    policy_name: str,
    r_vector: list[int],
    scenario: dict[str, Any],
    seed: int,
    plan_meta: dict[str, Any],
    gpu_row: dict[str, str],
    classification_match: str,
) -> dict[str, Any]:
    verdict = gpu_row.get("verdict_cell", "hard_fail")
    same_fault_false_recovery = (
        scenario["fault_family"] == "same_fault_all_correlated_control" and verdict == "vote_recovered"
    )

    return {
        "dataset": ds_name,
        "n_rows": str(n_rows),
        "policy": policy_name,
        "r_vector": "|".join(str(replica_count) for replica_count in r_vector),
        "fault_family": scenario["fault_family"],
        "plane_label": scenario["plane_label"],
        "seed": str(seed),
        "fault_entry_count": str(plan_meta["fault_entry_count"]),
        "affected_planes": plan_meta["affected_planes"],
        "affected_replicas": plan_meta["affected_replicas"],
        "replicas_hit_count": str(plan_meta["replicas_hit_count"]),
        "planes_hit_count": str(plan_meta["planes_hit_count"]),
        "high_significance_hit": "true" if plan_meta["high_significance_hit"] else "false",
        "replica_loss_correlation": f"{plan_meta['replica_loss_correlation']:.6f}",
        "latency_ms": gpu_row.get("latency_ms", "NA"),
        "contains_truth": gpu_row.get("contains_truth", "0.0"),
        "recovered_rate": gpu_row.get("recovered_rate", "0.0"),
        "detected_rate": gpu_row.get("detected_rate", "0.0"),
        "uncertified_rate": gpu_row.get("uncertified_rate", "0.0"),
        "certified_availability": gpu_row.get("certified_availability", "0.0"),
        "bound_width": gpu_row.get("bound_width", "NA"),
        "bound_width_norm": gpu_row.get("bound_width_norm", "NA"),
        "vote_recovered": gpu_row.get("vote_recovered", "false"),
        "same_fault_false_recovery": "true" if same_fault_false_recovery else gpu_row.get("same_fault_false_recovery", "false"),
        "silent_wrong": "true" if verdict == "silent_wrong" else "false",
        "cert_bound_failure": "true" if verdict == "cert_bound_failure" else "false",
        "hard_fail": "true" if verdict == "hard_fail" else "false",
        "bounded_degrade": "true" if verdict == "bounded_degraded" else "false",
        "cpu_gpu_classification_match": classification_match,
        "verdict_cell": verdict,
    }


def run_one_config(
    *,
    gpu_bench: Path,
    ds_name: str,
    ds_path: str,
    policy_name: str,
    r_vector: list[int],
    scenario: dict[str, Any],
    seed: int,
    n_rows: int,
    raw_csv_path: Path,
    tmp_dir: Path,
    timeout_s: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    available_planes = available_plane_indices(Path(ds_path))
    lsb_plane = max(available_planes) if available_planes else PLANE_COUNT - 1
    fault_plan = generate_fault_plan(policy_name, scenario, seed, r_vector, n_rows, lsb_plane)
    clean_planes = load_planes(Path(ds_path), n_rows)
    validate_fault_plan(fault_plan, [len(plane) for plane in clean_planes], r_vector)
    plan_meta = summarize_fault_plan(fault_plan, r_vector)

    fault_plan_path = tmp_dir / f"fp_{policy_name}_{scenario['fault_family']}_s{seed}.txt"
    with fault_plan_path.open("w") as handle:
        for entry in fault_plan:
            handle.write(f"{entry['plane']} {entry['replica']} {entry['offset']} {entry['mask']}\n")

    cpu_result = cpu_oracle(clean_planes, r_vector, fault_plan, n_rows)

    gpu_row = gpu_runner_one(
        gpu_bench,
        ds_path,
        r_vector,
        str(fault_plan_path),
        raw_csv_path,
        fault_family=scenario["fault_family"],
        seed=seed,
        n_rows=n_rows,
        timeout_s=timeout_s,
        fault_plane=primary_fault_plane(fault_plan),
    )

    gpu_outcome = gpu_row.get("verdict_cell", "hard_fail")
    classification_match = "true" if gpu_outcome == cpu_result["outcome"] else "false"

    canonical_row = build_canonical_row(
        ds_name=ds_name,
        n_rows=n_rows,
        policy_name=policy_name,
        r_vector=r_vector,
        scenario=scenario,
        seed=seed,
        plan_meta=plan_meta,
        gpu_row=gpu_row,
        classification_match=classification_match,
    )

    oracle_row = {
        "dataset": ds_name,
        "n_rows": n_rows,
        "policy": policy_name,
        "r_vector": "|".join(str(replica_count) for replica_count in r_vector),
        "fault_family": scenario["fault_family"],
        "seed": seed,
        "cpu_outcome": cpu_result["outcome"],
        "gpu_outcome": gpu_outcome,
        "classification_match": classification_match,
        "cpu_contains_truth": cpu_result["contains_truth"],
        "cpu_detected": cpu_result["detected"],
    }

    plan_row = {
        "dataset": ds_name,
        "policy": policy_name,
        "fault_family": scenario["fault_family"],
        "plane_label": scenario["plane_label"],
        "seed": seed,
        **plan_meta,
    }
    return canonical_row, oracle_row, plan_row


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def grouped_summary_rows(
    rows: list[dict[str, Any]],
    *,
    key_fields: list[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row[field]) for field in key_fields)
        groups[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        n = len(group)
        vote_recovered_count = sum(1 for row in group if row["verdict_cell"] == "vote_recovered")
        bounded_degraded_count = sum(1 for row in group if row["verdict_cell"] == "bounded_degraded")
        cert_bound_failure_count = sum(1 for row in group if row["cert_bound_failure"] == "true")
        hard_fail_count = sum(1 for row in group if row["hard_fail"] == "true")
        uncertified_or_unrecoverable_count = sum(
            1
            for row in group
            if row["verdict_cell"] not in {"vote_recovered", "bounded_degraded"}
        )
        out_row = {field: value for field, value in zip(key_fields, key)}
        out_row.update({
            "n_cells": n,
            "classification_match_rate": sum(1 for row in group if row["cpu_gpu_classification_match"] == "true") / n,
            "contains_truth_rate": sum(1 for row in group if row["contains_truth"] in {"1.0", "True", "true"}) / n,
            "recovered_rate": vote_recovered_count / n,
            "certified_bounded_rate": bounded_degraded_count / n,
            "uncertified_or_unrecoverable_rate": uncertified_or_unrecoverable_count / n,
            "same_fault_false_recovery_count": sum(1 for row in group if row["same_fault_false_recovery"] == "true"),
            "cert_bound_failure_count": cert_bound_failure_count,
            "hard_fail_count": hard_fail_count,
        })
        summary_rows.append(out_row)
    return summary_rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    compare_rows = [row for row in rows if row["cpu_gpu_classification_match"] in {"true", "false"}]
    return {
        "total_rows": len(rows),
        "fault_family_count": len({row["fault_family"] for row in rows}),
        "compare_rows": len(compare_rows),
        "compare_match_count": sum(1 for row in compare_rows if row["cpu_gpu_classification_match"] == "true"),
        "compare_mismatch_count": sum(1 for row in compare_rows if row["cpu_gpu_classification_match"] == "false"),
        "contains_truth_fail_count": sum(
            1 for row in rows if row["contains_truth"] not in {"1.0", "True", "true"}
        ),
        "same_fault_false_recovery_count": sum(1 for row in rows if row["same_fault_false_recovery"] == "true"),
        "silent_wrong_count": sum(1 for row in rows if row["silent_wrong"] == "true"),
        "cert_bound_failure_count": sum(1 for row in rows if row["cert_bound_failure"] == "true"),
        "hard_fail_count": sum(1 for row in rows if row["hard_fail"] == "true"),
        "datasets": sorted({row["dataset"] for row in rows}),
        "fault_families": sorted({row["fault_family"] for row in rows}),
        "policies": sorted({row["policy"] for row in rows}),
    }


def compute_verdict(summary: dict[str, Any]) -> str:
    if summary["fault_family_count"] < 3:
        return "NMR_A_GPU_STRUCTURED_MODEL_NEEDS_FIXES"
    if summary["compare_mismatch_count"] > 0:
        return "NMR_A_GPU_STRUCTURED_MODEL_NEEDS_FIXES"
    if summary["contains_truth_fail_count"] > 0:
        return "NMR_A_GPU_STRUCTURED_MODEL_NEEDS_FIXES"
    if summary["same_fault_false_recovery_count"] > 0:
        return "NMR_A_GPU_STRUCTURED_MODEL_NEEDS_FIXES"
    if summary["silent_wrong_count"] > 0:
        return "NMR_A_GPU_STRUCTURED_MODEL_NEEDS_FIXES"
    if summary["cert_bound_failure_count"] > 0:
        return "NMR_A_GPU_STRUCTURED_MODEL_NEEDS_FIXES"
    if summary["hard_fail_count"] > 0:
        return "NMR_A_GPU_STRUCTURED_MODEL_NEEDS_FIXES"
    return "NMR_A_GPU_STRUCTURED_MODEL_SCOPED_SUPPORTED"


def print_family_summary(family_rows: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 72}")
    print("Per-family summary")
    print(f"{'=' * 72}")
    for row in family_rows:
        print(
            f"  {row['dataset']:<16s} {row['fault_family']:<34s} "
            f"n={int(row['n_cells']):>3d} "
            f"match={float(row['classification_match_rate']):.3f} "
            f"recovered={float(row['recovered_rate']):.3f} "
            f"bounded={float(row['certified_bounded_rate']):.3f} "
            f"unrec={float(row['uncertified_or_unrecoverable_rate']):.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="NMR-A2 GPU structured fault runner")
    parser.add_argument("--gpu-bench", required=True)
    parser.add_argument("--n-rows", type=int, default=N_ROWS)
    parser.add_argument("--gpu-timeout-s", type=int, default=1800)
    args = parser.parse_args()

    gpu_bench = Path(args.gpu_bench)
    if not gpu_bench.exists():
        print(f"FATAL: {gpu_bench} not found")
        sys.exit(2)

    job_id = os.environ.get("SLURM_JOB_ID", "gpu_a2")
    out_root = Path(f"results/v1_3_freeze/structured_faults/job_{job_id}")
    out_root.mkdir(parents=True, exist_ok=True)

    raw_csv_path = out_root / "nmr_a_gpu_structured_raw.csv"
    raw_csv_path.write_text("")

    canonical_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []

    print("=== NMR-A2 GPU Structured Fault Injection ===")
    print(f"n_rows={args.n_rows} seeds={SEEDS}")
    print(f"datasets={list(DATASET_PATHS.keys())}")
    print(f"policies={list(POLICIES.keys())}")
    print(f"fault_families={[scenario['fault_family'] for scenario in FAULT_SCENARIOS]}")
    print("path=pre-fused bench_nmr_b_e2e_pipeline")

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nmr_a_structured_fp_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for ds_name, ds_path in DATASET_PATHS.items():
            if not Path(ds_path).is_dir():
                print(f"SKIP: dataset missing: {ds_name}")
                continue
            print(f"\n{'=' * 72}\n{ds_name}\n{'=' * 72}")
            for policy_name, r_vector in POLICIES.items():
                print(f"\n  Policy: {policy_name} r={r_vector}")
                for scenario in FAULT_SCENARIOS:
                    for seed in SEEDS:
                        canonical_row, oracle_row, plan_row = run_one_config(
                            gpu_bench=gpu_bench,
                            ds_name=ds_name,
                            ds_path=ds_path,
                            policy_name=policy_name,
                            r_vector=r_vector,
                            scenario=scenario,
                            seed=seed,
                            n_rows=args.n_rows,
                            raw_csv_path=raw_csv_path,
                            tmp_dir=tmp_path,
                            timeout_s=args.gpu_timeout_s,
                        )
                        canonical_rows.append(canonical_row)
                        oracle_rows.append(oracle_row)
                        plan_rows.append(plan_row)
                        mark = "✅" if canonical_row["cpu_gpu_classification_match"] == "true" else "❌"
                        print(
                            f"    {scenario['fault_family']:<34s} s={seed} "
                            f"GPU={canonical_row['verdict_cell']:<20s} {mark}"
                        )
    elapsed = time.perf_counter() - t0

    canonical_csv_path = out_root / "nmr_a_structured_fault_matrix.csv"
    write_csv(canonical_csv_path, CANONICAL_FIELDS, canonical_rows)

    plan_csv_path = out_root / "nmr_a_fault_plan_summary.csv"
    write_csv(plan_csv_path, PLAN_FIELDS, plan_rows)

    policy_summary_rows = grouped_summary_rows(
        canonical_rows,
        key_fields=["dataset", "policy", "fault_family"],
    )
    policy_summary_path = out_root / "nmr_a_policy_summary.csv"
    write_csv(policy_summary_path, POLICY_SUMMARY_FIELDS, policy_summary_rows)

    family_summary_rows = grouped_summary_rows(
        canonical_rows,
        key_fields=["dataset", "fault_family"],
    )
    family_summary_path = out_root / "nmr_a_fault_family_summary.csv"
    write_csv(family_summary_path, FAMILY_SUMMARY_FIELDS, family_summary_rows)

    oracle_csv_path = out_root / "nmr_a_oracle_comparison.csv"
    write_csv(oracle_csv_path, ORACLE_FIELDS, oracle_rows)

    summary = build_summary(canonical_rows)
    verdict = compute_verdict(summary)

    handoff_json = {
        "job_id": job_id,
        "experiment": "nmr_a_gpu_structured_fault",
        "path_scope": "pre-fused bench_nmr_b_e2e_pipeline",
        "verdict": verdict,
        "summary": summary,
        "artifacts": {
            "canonical_csv": str(canonical_csv_path),
            "raw_gpu_csv": str(raw_csv_path),
            "fault_plan_summary_csv": str(plan_csv_path),
            "policy_summary_csv": str(policy_summary_path),
            "fault_family_summary_csv": str(family_summary_path),
            "oracle_csv": str(oracle_csv_path),
        },
    }
    (out_root / "handoff.json").write_text(json.dumps(handoff_json, indent=2) + "\n")

    provenance_manifest = {
        "job_id": job_id,
        "cwd": str(Path.cwd()),
        "gpu_bench": str(gpu_bench),
        "datasets": DATASET_PATHS,
        "policies": POLICIES,
        "fault_families": [scenario["fault_family"] for scenario in FAULT_SCENARIOS],
        "n_rows": args.n_rows,
        "seeds": SEEDS,
        "scope_boundary": "HBM-inspired structured software injection only; no physical HBM diversity claim.",
    }
    (out_root / "provenance_manifest.json").write_text(json.dumps(provenance_manifest, indent=2) + "\n")
    (out_root / "verdict.txt").write_text(verdict + "\n")

    print(f"\n{'=' * 72}")
    print("NMR-A2 GPU Structured Fault Summary")
    print(f"{'=' * 72}")
    print(f"  Total rows:                 {summary['total_rows']}")
    print(f"  Fault families exercised:   {summary['fault_family_count']}")
    print(f"  CPU/GPU matches:            {summary['compare_match_count']}/{summary['compare_rows']}")
    print(f"  contains_truth failures:    {summary['contains_truth_fail_count']}")
    print(f"  same-fault false recovery:  {summary['same_fault_false_recovery_count']}")
    print(f"  silent wrong rows:          {summary['silent_wrong_count']}")
    print(f"  cert bound failures:        {summary['cert_bound_failure_count']}")
    print(f"  hard fails:                 {summary['hard_fail_count']}")
    print(f"  elapsed_s:                  {elapsed:.2f}")
    print(f"  verdict:                    {verdict}")

    print_family_summary(family_summary_rows)
    print(f"\nResults: {canonical_csv_path}")


if __name__ == "__main__":
    main()
