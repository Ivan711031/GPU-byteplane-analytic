#!/usr/bin/env python3
"""NMR-B: GPU End-to-End NMR Pipeline Runner.

This runner has two phases:
1. A CPU/GPU shared-fault-plan correctness pass.
2. A representative GPU-only full-data scale-up pass.
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
from pathlib import Path
from typing import Any

PLANE_COUNT = 8
PLANE_WEIGHTS = [1 << (8 * (7 - p)) for p in range(PLANE_COUNT)]

DATASET_PATHS: dict[str, str] = {
    "hurricane_u": "/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg4096",
    "cesm_atm_cloud": "/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg4096",
}

FAULT_CONTROLS = [
    "single_replica_independent_plane0",
    "single_replica_independent_plane1",
    "single_replica_independent_random_plane",
    "same_fault_all_replicas_plane0",
    "uniformish_r0_2_non_majority_fallback",
    "clean_no_fault",
]

SEEDS = [0, 1, 2]

R_VECTORS: dict[str, list[int]] = {
    "graded_nmr_r_3_2_1_1_1_1_1_1": [3, 2, 1, 1, 1, 1, 1, 1],
    "uniformish_r_2_1_2_1_2_1_1_1": [2, 1, 2, 1, 2, 1, 1, 1],
    "certified_degradation_r1_all_planes": [1, 1, 1, 1, 1, 1, 1, 1],
    "byteplane_normal_k2": [1, 1, 1, 1, 1, 1, 1, 1],
}

N_ROWS_SMOKE = 100_000
N_ROWS_LARGE = 500_000

# Representative full-data rows after smoke/oracle passes.
FULL_SCALE_CASES = [
    ("graded_nmr_r_3_2_1_1_1_1_1_1", "clean_no_fault", 0),
    ("graded_nmr_r_3_2_1_1_1_1_1_1", "single_replica_independent_plane0", 0),
    ("graded_nmr_r_3_2_1_1_1_1_1_1", "single_replica_independent_plane1", 0),
    ("graded_nmr_r_3_2_1_1_1_1_1_1", "same_fault_all_replicas_plane0", 0),
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
    "path",
    "replica_policy",
    "r_vector",
    "k",
    "fault_mode",
    "fault_plane",
    "seed",
    "latency_ms",
    "overhead_vs_raw_fused",
    "overhead_vs_byteplane",
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
    "correct_clean",
    "cpu_gpu_classification_match",
    "verdict_cell",
]


def generate_fault_plan(
    fault_mode: str, seed: int, r_vector: list[int], n_rows: int
) -> list[dict[str, Any]]:
    """Deterministic fault plan shared by CPU oracle and GPU benchmark."""
    rng = random.Random(seed)
    plan: list[dict[str, Any]] = []

    if fault_mode == "clean_no_fault":
        return plan

    if fault_mode == "single_replica_independent_plane0":
        offset = rng.randint(0, n_rows - 1)
        plan.append({"plane": 0, "replica": 0, "offset": offset, "mask": 0xFF})
    elif fault_mode == "single_replica_independent_plane1":
        offset = rng.randint(0, n_rows - 1)
        plan.append({"plane": 1, "replica": 0, "offset": offset, "mask": 0xFF})
    elif fault_mode == "single_replica_independent_random_plane":
        plane = rng.randint(0, PLANE_COUNT - 1)
        offset = rng.randint(0, n_rows - 1)
        plan.append({"plane": plane, "replica": 0, "offset": offset, "mask": 0xFF})
    elif fault_mode == "same_fault_all_replicas_plane0":
        offset = rng.randint(0, n_rows - 1)
        for rep in range(r_vector[0] if r_vector else 1):
            plan.append({"plane": 0, "replica": rep, "offset": offset, "mask": 0xFF})
    elif fault_mode == "uniformish_r0_2_non_majority_fallback":
        offset = rng.randint(0, n_rows - 1)
        for rep in range(min(r_vector[0] if r_vector else 1, 2)):
            plan.append({"plane": 0, "replica": rep, "offset": offset, "mask": 0xFF})

    return plan


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


def load_value_count(artifact_dir: Path) -> int:
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    return int(manifest["value_count"])


def sum32(data: bytes) -> int:
    # The project defines SUM32 here as a bytewise modular sum, matching the GPU kernel.
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
        for idx in range(n):
            vals = [replicas[plane][rep][idx] for rep in range(replica_count)]
            result[idx] = max(set(vals), key=lambda value: (vals.count(value), value))
        voted.append(bytes(result))
    return voted


def compute_encoded_sum(planes: list[bytes]) -> float:
    return float(sum(sum(plane) * PLANE_WEIGHTS[idx] for idx, plane in enumerate(planes)))


def compute_bound_width(detected_planes: list[bool], n_rows: int) -> float:
    return float(
        sum(255.0 * PLANE_WEIGHTS[plane] * n_rows for plane in range(PLANE_COUNT) if detected_planes[plane])
    )


def cpu_oracle(
    clean_planes: list[bytes],
    r_vector: list[int],
    fault_plan: list[dict[str, Any]],
    n_rows: int,
) -> dict[str, Any]:
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
        if plane < len(replicas) and replica < len(replicas[plane]) and offset < len(replicas[plane][replica]):
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


def default_gpu_row(verdict: str = "hard_fail") -> dict[str, str]:
    return {
        "dataset": "unknown",
        "n_rows": "0",
        "path": "gpu_e2e",
        "replica_policy": "gpu_e2e",
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


def gpu_runner_one(
    gpu_bench: Path,
    dataset_path: str,
    r_vector: list[int],
    fault_plan_path: str,
    raw_csv_path: Path,
    *,
    fault_mode: str,
    n_rows: int,
    timeout_s: int,
) -> dict[str, str]:
    r_vec_str = "|".join(str(replica_count) for replica_count in r_vector)
    cmd = [
        str(gpu_bench),
        "--dataset", dataset_path,
        "--r-vector", r_vec_str,
        "--fault-plan", fault_plan_path,
        "--fault-mode", fault_mode,
        "--csv", str(raw_csv_path),
        "--iters", "30",
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


def infer_fault_plane(fault_plan: list[dict[str, Any]]) -> str:
    if not fault_plan:
        return "NA"
    planes = sorted({entry["plane"] for entry in fault_plan})
    if len(planes) == 1:
        return str(planes[0])
    return "|".join(str(plane) for plane in planes)


def k_for_path(path_name: str) -> str:
    return "2" if path_name == "byteplane_normal_k2" else "NA"


def build_canonical_row(
    *,
    ds_name: str,
    n_rows: int,
    path_name: str,
    r_vector: list[int],
    fault_mode: str,
    fault_plan: list[dict[str, Any]],
    seed: int,
    gpu_row: dict[str, str],
    classification_match: str,
) -> dict[str, str]:
    verdict = gpu_row.get("verdict_cell", "hard_fail")
    same_fault_false_recovery = fault_mode.startswith("same_fault_all") and verdict == "vote_recovered"
    correct_clean = fault_mode == "clean_no_fault" and verdict == "vote_recovered"

    return {
        "dataset": ds_name,
        "n_rows": str(n_rows),
        "path": path_name,
        "replica_policy": path_name,
        "r_vector": "|".join(str(replica_count) for replica_count in r_vector),
        "k": k_for_path(path_name),
        "fault_mode": fault_mode,
        "fault_plane": infer_fault_plane(fault_plan),
        "seed": str(seed),
        "latency_ms": gpu_row.get("latency_ms", "NA"),
        "overhead_vs_raw_fused": "NA",
        "overhead_vs_byteplane": "NA",
        "contains_truth": gpu_row.get("contains_truth", "0.0"),
        "recovered_rate": gpu_row.get("recovered_rate", "0.0"),
        "detected_rate": gpu_row.get("detected_rate", "0.0"),
        "uncertified_rate": gpu_row.get("uncertified_rate", "0.0"),
        "certified_availability": gpu_row.get("certified_availability", "0.0"),
        "bound_width": gpu_row.get("bound_width", "NA"),
        "bound_width_norm": gpu_row.get("bound_width_norm", "NA"),
        "vote_recovered": gpu_row.get("vote_recovered", "false"),
        "fallback_used": gpu_row.get("fallback_used", "false"),
        "same_fault_false_recovery": "true" if same_fault_false_recovery else gpu_row.get("same_fault_false_recovery", "false"),
        "silent_wrong": "true" if verdict == "silent_wrong" else "false",
        "cert_bound_failure": "true" if verdict == "cert_bound_failure" else "false",
        "hard_fail": "true" if verdict == "hard_fail" else "false",
        "bounded_degrade": "true" if verdict == "bounded_degraded" else "false",
        "correct_clean": "true" if correct_clean else "false",
        "cpu_gpu_classification_match": classification_match,
        "verdict_cell": verdict,
    }


def run_one_config(
    *,
    gpu_bench: Path,
    ds_name: str,
    ds_path: str,
    r_vector: list[int],
    path_name: str,
    fault_mode: str,
    seed: int,
    n_rows: int,
    raw_csv_path: Path,
    tmp_dir: Path,
    compare_to_cpu: bool,
    timeout_s: int,
) -> tuple[dict[str, str], dict[str, Any] | None]:
    fault_plan = generate_fault_plan(fault_mode, seed, r_vector, n_rows)

    fault_plan_path = tmp_dir / f"fp_{path_name}_{fault_mode}_s{seed}_{n_rows}.txt"
    with fault_plan_path.open("w") as handle:
        for entry in fault_plan:
            handle.write(f"{entry['plane']} {entry['replica']} {entry['offset']} {entry['mask']}\n")

    cpu_result = None
    if compare_to_cpu:
        clean_planes = load_planes(Path(ds_path), n_rows)
        cpu_result = cpu_oracle(clean_planes, r_vector, fault_plan, n_rows)

    gpu_row = gpu_runner_one(
        gpu_bench,
        ds_path,
        r_vector,
        str(fault_plan_path),
        raw_csv_path,
        fault_mode=fault_mode,
        n_rows=n_rows,
        timeout_s=timeout_s,
    )

    gpu_outcome = gpu_row.get("verdict_cell", "hard_fail")
    classification_match = "not_run"
    oracle_row = None
    if cpu_result is not None:
        classification_match = "true" if gpu_outcome == cpu_result["outcome"] else "false"
        oracle_row = {
            "dataset": ds_name,
            "n_rows": n_rows,
            "path": path_name,
            "r_vector": "|".join(str(replica_count) for replica_count in r_vector),
            "fault_mode": fault_mode,
            "seed": seed,
            "cpu_outcome": cpu_result["outcome"],
            "gpu_outcome": gpu_outcome,
            "classification_match": classification_match,
            "cpu_contains_truth": cpu_result["contains_truth"],
            "cpu_detected": cpu_result["detected"],
        }

    canonical_row = build_canonical_row(
        ds_name=ds_name,
        n_rows=n_rows,
        path_name=path_name,
        r_vector=r_vector,
        fault_mode=fault_mode,
        fault_plan=fault_plan,
        seed=seed,
        gpu_row=gpu_row,
        classification_match=classification_match,
    )
    return canonical_row, oracle_row


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_latency_summary(rows: list[dict[str, str]], out_path: Path) -> None:
    groups: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        latency = row.get("latency_ms", "NA")
        if not latency or latency == "NA":
            continue
        key = (row["dataset"], row["n_rows"], row["path"], row["fault_mode"])
        groups.setdefault(key, []).append(float(latency))

    summary_rows: list[dict[str, Any]] = []
    for (dataset, n_rows, path_name, fault_mode), values in sorted(groups.items()):
        summary_rows.append({
            "dataset": dataset,
            "n_rows": n_rows,
            "path": path_name,
            "fault_mode": fault_mode,
            "count": len(values),
            "latency_ms_mean": f"{sum(values) / len(values):.6f}",
            "latency_ms_min": f"{min(values):.6f}",
            "latency_ms_max": f"{max(values):.6f}",
        })

    write_csv(
        out_path,
        ["dataset", "n_rows", "path", "fault_mode", "count", "latency_ms_mean", "latency_ms_min", "latency_ms_max"],
        summary_rows,
    )


def build_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    compare_rows = [row for row in rows if row["cpu_gpu_classification_match"] in {"true", "false"}]
    scale_rows = [row for row in rows if row["cpu_gpu_classification_match"] == "not_run"]
    match_count = sum(1 for row in compare_rows if row["cpu_gpu_classification_match"] == "true")

    contains_truth_fail_count = sum(1 for row in rows if row["contains_truth"] not in {"1.0", "True", "true"})
    clean_no_fault_fail_count = sum(
        1 for row in rows if row["fault_mode"] == "clean_no_fault" and row["correct_clean"] != "true"
    )
    same_fault_false_recovery_count = sum(1 for row in rows if row["same_fault_false_recovery"] == "true")
    hard_fail_count = sum(1 for row in rows if row["hard_fail"] == "true")
    cert_bound_failure_count = sum(1 for row in rows if row["cert_bound_failure"] == "true")

    return {
        "total_rows": len(rows),
        "compare_rows": len(compare_rows),
        "compare_match_count": match_count,
        "compare_mismatch_count": len(compare_rows) - match_count,
        "scale_rows": len(scale_rows),
        "contains_truth_fail_count": contains_truth_fail_count,
        "clean_no_fault_fail_count": clean_no_fault_fail_count,
        "same_fault_false_recovery_count": same_fault_false_recovery_count,
        "hard_fail_count": hard_fail_count,
        "cert_bound_failure_count": cert_bound_failure_count,
        "datasets": sorted({row["dataset"] for row in rows}),
    }


def compute_verdict(summary: dict[str, Any], scale_requested: bool) -> str:
    if summary["hard_fail_count"] > 0:
        return "NMR_B_NEEDS_FIXES"
    if summary["compare_mismatch_count"] > 0:
        return "NMR_B_NEEDS_FIXES"
    if summary["clean_no_fault_fail_count"] > 0:
        return "NMR_B_NEEDS_FIXES"
    if summary["same_fault_false_recovery_count"] > 0:
        return "NMR_B_NEEDS_FIXES"
    if summary["contains_truth_fail_count"] > 0:
        return "NMR_B_NEEDS_FIXES"
    if summary["cert_bound_failure_count"] > 0:
        return "NMR_B_NEEDS_FIXES"
    if scale_requested and summary["scale_rows"] > 0:
        return "NMR_B_GPU_E2E_SUPPORTED"
    return "NMR_B_GPU_E2E_CORRECT_BUT_OVERHEAD_SCOPED"


def main() -> None:
    parser = argparse.ArgumentParser(description="NMR-B GPU E2E Pipeline Runner")
    parser.add_argument("--gpu-bench", required=True)
    parser.add_argument("--n-rows", type=int, default=N_ROWS_LARGE)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run the minimal CPU/GPU oracle smoke on hurricane_u only",
    )
    parser.add_argument(
        "--scale-gpu-only",
        action="store_true",
        help="After the oracle phase, run a representative GPU-only full-data scale matrix",
    )
    parser.add_argument("--gpu-timeout-s", type=int, default=1800)
    args = parser.parse_args()

    gpu_bench = Path(args.gpu_bench)
    if not gpu_bench.exists():
        print(f"FATAL: {gpu_bench} not found")
        sys.exit(2)

    job_id = os.environ.get("SLURM_JOB_ID", "gpu_b")
    out_root = Path(f"results/reliability_layer1/phase4/nmr_b_gpu_e2e_pipeline/job_{job_id}")
    out_root.mkdir(parents=True, exist_ok=True)

    raw_csv_path = out_root / "nmr_b_gpu_e2e_raw.csv"
    if raw_csv_path.exists():
        raw_csv_path.unlink()

    canonical_rows: list[dict[str, str]] = []
    oracle_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="nmr_b_fp_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        dataset_items = list(DATASET_PATHS.items())
        if args.smoke_only:
            dataset_items = [("hurricane_u", DATASET_PATHS["hurricane_u"])]
            iter_controls = FAULT_CONTROLS[:4]
            iter_n_rows = N_ROWS_SMOKE
        else:
            iter_controls = FAULT_CONTROLS
            iter_n_rows = args.n_rows

        for ds_name, ds_path in dataset_items:
            if not Path(ds_path).is_dir():
                print(f"SKIP: dataset missing: {ds_name}")
                continue
            print(f"\n{'=' * 60}\n{ds_name}\n{'=' * 60}")
            for path_name, r_vector in R_VECTORS.items():
                print(f"\n  Path: {path_name} r={r_vector}")
                for fault_mode in iter_controls:
                    for seed in SEEDS:
                        row, oracle_row = run_one_config(
                            gpu_bench=gpu_bench,
                            ds_name=ds_name,
                            ds_path=ds_path,
                            r_vector=r_vector,
                            path_name=path_name,
                            fault_mode=fault_mode,
                            seed=seed,
                            n_rows=iter_n_rows,
                            raw_csv_path=raw_csv_path,
                            tmp_dir=tmp_path,
                            compare_to_cpu=True,
                            timeout_s=args.gpu_timeout_s,
                        )
                        canonical_rows.append(row)
                        if oracle_row is not None:
                            oracle_rows.append(oracle_row)
                        match = row["cpu_gpu_classification_match"]
                        mark = "✅" if match == "true" else "❌"
                        print(
                            f"    {fault_mode:<40s} s={seed}  "
                            f"GPU={row['verdict_cell']:<20s}  {mark}"
                        )

        if args.scale_gpu_only:
            print(f"\n{'=' * 60}\nfull-data gpu scale\n{'=' * 60}")
            for ds_name, ds_path in DATASET_PATHS.items():
                if not Path(ds_path).is_dir():
                    print(f"SKIP: dataset missing: {ds_name}")
                    continue
                full_n_rows = load_value_count(Path(ds_path))
                print(f"\n  Dataset: {ds_name} full_n_rows={full_n_rows}")
                for path_name, fault_mode, seed in FULL_SCALE_CASES:
                    row, _ = run_one_config(
                        gpu_bench=gpu_bench,
                        ds_name=ds_name,
                        ds_path=ds_path,
                        r_vector=R_VECTORS[path_name],
                        path_name=path_name,
                        fault_mode=fault_mode,
                        seed=seed,
                        n_rows=full_n_rows,
                        raw_csv_path=raw_csv_path,
                        tmp_dir=tmp_path,
                        compare_to_cpu=False,
                        timeout_s=args.gpu_timeout_s,
                    )
                    canonical_rows.append(row)
                    print(
                        f"    {path_name:<32s} {fault_mode:<34s} "
                        f"GPU={row['verdict_cell']:<20s} latency={row['latency_ms']} ms"
                    )

    canonical_csv_path = out_root / "nmr_b_gpu_e2e.csv"
    write_csv(canonical_csv_path, CANONICAL_FIELDS, canonical_rows)

    oracle_csv_path = out_root / "nmr_b_oracle_comparison.csv"
    write_csv(
        oracle_csv_path,
        [
            "dataset",
            "n_rows",
            "path",
            "r_vector",
            "fault_mode",
            "seed",
            "cpu_outcome",
            "gpu_outcome",
            "classification_match",
            "cpu_contains_truth",
            "cpu_detected",
        ],
        oracle_rows,
    )

    latency_summary_path = out_root / "nmr_b_latency_summary.csv"
    write_latency_summary(canonical_rows, latency_summary_path)

    summary = build_summary(canonical_rows)
    verdict = compute_verdict(summary, args.scale_gpu_only)

    handoff_json = {
        "job_id": job_id,
        "experiment": "nmr_b_gpu_e2e_pipeline",
        "verdict": verdict,
        "summary": summary,
        "artifacts": {
            "canonical_csv": str(canonical_csv_path),
            "raw_gpu_csv": str(raw_csv_path),
            "oracle_csv": str(oracle_csv_path),
            "latency_summary_csv": str(latency_summary_path),
        },
    }
    (out_root / "handoff.json").write_text(json.dumps(handoff_json, indent=2) + "\n")

    provenance_manifest = {
        "job_id": job_id,
        "cwd": str(Path.cwd()),
        "gpu_bench": str(gpu_bench),
        "datasets": {
            name: {
                "artifact_path": path,
                "value_count": load_value_count(Path(path)) if Path(path).is_dir() else None,
            }
            for name, path in DATASET_PATHS.items()
        },
    }
    (out_root / "provenance_manifest.json").write_text(json.dumps(provenance_manifest, indent=2) + "\n")
    (out_root / "verdict.txt").write_text(verdict + "\n")

    print(f"\n{'=' * 60}")
    print("NMR-B GPU E2E Summary")
    print(f"{'=' * 60}")
    print(f"  Total rows:              {summary['total_rows']}")
    print(f"  Oracle compare rows:     {summary['compare_rows']}")
    print(f"  Compare matches:         {summary['compare_match_count']}/{summary['compare_rows']}")
    print(f"  Full-data scale rows:    {summary['scale_rows']}")
    print(f"  clean-no-fault failures: {summary['clean_no_fault_fail_count']}")
    print(f"  same-fault false recover:{summary['same_fault_false_recovery_count']}")
    print(f"  contains_truth failures: {summary['contains_truth_fail_count']}")
    print(f"  hard fails:              {summary['hard_fail_count']}")
    print(f"  cert bound failures:     {summary['cert_bound_failure_count']}")
    print(f"\n  Verdict: {verdict}\n")


if __name__ == "__main__":
    main()
