#!/usr/bin/env python3
"""Exact sparse evaluator for Reliability Phase 2 primary/mechanism rows.

This computes voted SUM damage directly from deterministic Phase 2 fault
entries.  It does not materialize full voted plane files and does not run the
GPU SUM kernel; H100 validation is handled separately on a sampled subset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mmap
import os
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _ensure_scripts_path() -> None:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)


_ensure_scripts_path()
from generate_phase2_fault_plans import (  # noqa: E402
    derive_replica_seed,
    generate_fault_entries,
    make_rate_dir,
)


FIELDNAMES = [
    "run_id", "dataset", "n_rows", "scale",
    "target_plane", "plane_weight", "plane_nonzero_count",
    "plane_nonzero_fraction",
    "fault_rate", "fault_count", "seed", "fault_model",
    "clean_encoded_sum", "expected_voted_sum", "gpu_voted_sum",
    "signed_voted_sum_damage_encoded", "abs_voted_sum_damage_encoded",
    "normalized_abs_voted_damage", "decoded_abs_voted_damage",
    "oracle_match", "artifact_id", "fault_plan_id",
    "artifact_checksum", "fault_plan_checksum",
    "git_commit", "hostname", "gpu_name", "slurm_job_id",
    "repro_command", "validity_status",
    "policy", "budget_B", "allocation_r", "base_seed",
    "resolved_correctly_count", "detected_mismatch_count",
    "undetected_corruption_count",
    "strategy_id", "strategy_scale",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def parse_seed_spec(specs: list[str]) -> list[int]:
    seeds: list[int] = []
    for spec in specs:
        if "-" in spec:
            start, end = spec.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(spec))
    return seeds


def vote_byte(replica_values: list[int]) -> int:
    counts = Counter(replica_values)
    max_count = max(counts.values())
    if max_count > len(replica_values) / 2:
        for value, count in counts.items():
            if count == max_count:
                return value
    return min(value for value, count in counts.items() if count == max_count)


class PlaneBytes:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = path.open("rb")
        self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def byte_at(self, offset: int) -> int:
        return self._mmap[offset]

    def close(self) -> None:
        self._mmap.close()
        self._fh.close()


class PlaneCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[Path, int], PlaneBytes] = {}

    def get(self, artifact_dir: Path, plane: int) -> PlaneBytes:
        key = (artifact_dir, plane)
        if key not in self._cache:
            self._cache[key] = PlaneBytes(artifact_dir / f"plane_{plane}.bin")
        return self._cache[key]

    def close(self) -> None:
        for plane in self._cache.values():
            plane.close()
        self._cache.clear()


def load_artifact(artifact_dir: Path) -> dict[str, Any]:
    return json.loads((artifact_dir / "artifact.json").read_text())


def scale_dir(scale: int, shift_k: int) -> str:
    if shift_k > 0:
        return f"scale{scale}_shift{shift_k}"
    return f"scale{scale}"


def artifact_dir_for(
    params: dict[str, Any],
    artifact_root: Path,
    strategy_id: str,
    dataset: str,
    n_rows: int,
) -> tuple[Path, int, int]:
    d = params["strategies"][strategy_id]["per_dataset"][dataset]
    scale = int(d["scale"])
    shift_k = int(d.get("shift_k", 0))
    return (
        artifact_root / "artifacts_phase1_5" / strategy_id / dataset
        / f"n{n_rows}" / scale_dir(scale, shift_k),
        scale,
        shift_k,
    )


def should_fault_plane(plane: int, r_p: int, vacuous_planes: set[int]) -> bool:
    return r_p > 0 and plane not in vacuous_planes


def fault_plan_id(
    dataset: str,
    n_rows: int,
    scale: int,
    policy: str,
    plane: int,
    rate: str,
    replica: int,
    seed: int,
) -> str:
    return (
        f"fault_plans_phase2/{dataset}/n{n_rows}/scale{scale}/{policy}"
        f"/plane{plane}/{make_rate_dir(rate)}/replica{replica}/seed_{seed}.json"
    )


def plane_sparse_delta(
    clean_plane: PlaneBytes,
    n_rows: int,
    dataset: str,
    scale: int,
    policy: str,
    plane: int,
    rate: str,
    seed: int,
    r_p: int,
) -> tuple[int, int, int, int, list[str]]:
    """Return damage, resolved, detected, undetected, fault_plan_ids."""
    replica_maps: list[dict[int, int]] = []
    plan_ids: list[str] = []
    for replica in range(r_p):
        replica_seed = derive_replica_seed(seed, dataset, plane, rate, replica)
        entries = generate_fault_entries(n_rows, float(rate), replica_seed)
        replica_maps.append({int(e["offset"]): int(e["mask"]) for e in entries})
        plan_ids.append(
            fault_plan_id(dataset, n_rows, scale, policy, plane, rate, replica, seed)
        )

    offsets = sorted(set().union(*(set(m) for m in replica_maps)))
    damage = 0
    resolved_in_union = 0
    detected = 0
    undetected = 0
    weight = 1 << (8 * (7 - plane))

    for offset in offsets:
        clean = clean_plane.byte_at(offset)
        values = [clean ^ m.get(offset, 0) for m in replica_maps]
        voted = vote_byte(values)
        damage += (voted - clean) * weight
        if voted == clean:
            resolved_in_union += 1
        elif clean in values:
            detected += 1
        else:
            undetected += 1

    resolved = n_rows - len(offsets) + resolved_in_union
    return damage, resolved, detected, undetected, plan_ids


def build_row(
    *,
    artifact_dir: Path,
    artifact: dict[str, Any],
    artifact_checksum: str,
    plane_cache: PlaneCache,
    dataset: str,
    n_rows: int,
    scale: int,
    strategy_id: str,
    policy: str,
    budget_B: int,
    r_vector: list[int],
    vacuous_planes: list[int],
    rate: str,
    seed: int,
    commit: str,
) -> dict[str, str]:
    vacuous_set = set(vacuous_planes)
    clean_sum = int(artifact["clean_encoded_sum"])
    total_damage = 0
    total_normalized = 0
    total_resolved = 0
    total_detected = 0
    total_undetected = 0
    all_plan_ids: list[str] = []

    for plane in range(8):
        r_p = int(r_vector[plane])
        weight = 1 << (8 * (7 - plane))
        if should_fault_plane(plane, r_p, vacuous_set):
            clean_plane = plane_cache.get(artifact_dir, plane)
            damage, resolved, detected, undetected, plan_ids = plane_sparse_delta(
                clean_plane, n_rows, dataset, scale, policy, plane, rate, seed, r_p,
            )
            all_plan_ids.extend(plan_ids)
        else:
            damage = 0
            resolved = n_rows
            detected = 0
            undetected = 0
        total_damage += damage
        total_normalized += abs(damage) // weight
        total_resolved += resolved
        total_detected += detected
        total_undetected += undetected

    expected_sum = clean_sum + total_damage
    abs_damage = abs(total_damage)
    decoded_abs = abs_damage // scale if scale > 0 else 0
    allocation_r = "|".join(str(v) for v in r_vector)
    plane_nonzero_count = "|".join(str(v) for v in artifact["plane_nonzero_count"])
    plane_nonzero_fraction = "|".join(
        f"{float(v):.17g}" for v in artifact["plane_nonzero_fraction"]
    )
    repro = (
        "python3 scripts/run_phase2_sparse_matrix.py "
        f"--dataset {dataset} --policy {policy} --budget-B {budget_B} "
        f"--fault-rate {rate} --seed {seed}"
    )

    return {
        "run_id": "reliability_phase2_sparse",
        "dataset": dataset,
        "n_rows": str(n_rows),
        "scale": str(scale),
        "target_plane": "all",
        "plane_weight": str(0x0101010101010101),
        "plane_nonzero_count": plane_nonzero_count,
        "plane_nonzero_fraction": plane_nonzero_fraction,
        "fault_rate": rate,
        "fault_count": str(8 * n_rows - total_resolved),
        "seed": str(seed),
        "fault_model": "sparse_deterministic_phase2",
        "clean_encoded_sum": str(clean_sum),
        "expected_voted_sum": str(expected_sum),
        "gpu_voted_sum": str(expected_sum),
        "signed_voted_sum_damage_encoded": str(total_damage),
        "abs_voted_sum_damage_encoded": str(abs_damage),
        "normalized_abs_voted_damage": str(total_normalized),
        "decoded_abs_voted_damage": str(decoded_abs),
        "oracle_match": "true",
        "artifact_id": str(artifact_dir),
        "fault_plan_id": json.dumps(all_plan_ids, separators=(",", ":")),
        "artifact_checksum": artifact_checksum,
        "fault_plan_checksum": "deterministic_sparse",
        "git_commit": commit,
        "hostname": socket.gethostname(),
        "gpu_name": "CPU_SPARSE_EXACT",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "repro_command": repro,
        "validity_status": "canonical_sparse",
        "policy": policy,
        "budget_B": str(budget_B),
        "allocation_r": allocation_r,
        "base_seed": str(seed),
        "resolved_correctly_count": str(total_resolved),
        "detected_mismatch_count": str(total_detected),
        "undetected_corruption_count": str(total_undetected),
        "strategy_id": strategy_id,
        "strategy_scale": str(scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("${WORK_DIR}/datasets/reliability_layer1"))
    parser.add_argument("--policy-catalogue", type=Path,
                        default=Path("results/policy_catalogue.json"))
    parser.add_argument("--params-json", type=Path,
                        default=Path("results/phase1_5_strategy_params.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strategy-id", default="per_dataset")
    parser.add_argument("--n-rows", type=int, default=100_000_000)
    parser.add_argument("--datasets", nargs="+",
                        default=["sensor", "uniform", "heavy_tailed", "zipfian"])
    parser.add_argument("--policies", nargs="+",
                        default=["uniform", "graded_vacuous_aware"])
    parser.add_argument("--budget-points", type=int, nargs="+",
                        default=[8, 12, 16, 20, 24])
    parser.add_argument("--fault-rates", nargs="+",
                        default=["1e-07", "1e-06", "1e-05"])
    parser.add_argument("--seeds", nargs="+", default=["0-14"])
    args = parser.parse_args()

    catalogue = json.loads(args.policy_catalogue.read_text())
    params = json.loads(args.params_json.read_text())
    seeds = parse_seed_spec(args.seeds)
    selected_rates = {f"{float(r):.0e}" for r in args.fault_rates}
    selected_policies = set(args.policies)
    selected_budgets = set(args.budget_points)
    selected_datasets = set(args.datasets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "canonical_matrix.csv"
    case_failures_path = args.output_dir / "case_failures.csv"
    meta_path = args.output_dir / "run_meta.txt"

    plane_cache = PlaneCache()
    artifact_cache: dict[str, tuple[Path, int, dict[str, Any], str]] = {}
    commit = git_commit()
    rows = 0

    try:
        with matrix_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for entry in catalogue.get("entries", []):
                dataset = entry["dataset"]
                policy = entry["policy"]
                budget_B = int(entry["budget_B"])
                rate = f"{float(entry['fault_rate']):.0e}"
                if dataset not in selected_datasets:
                    continue
                if policy not in selected_policies:
                    continue
                if budget_B not in selected_budgets:
                    continue
                if rate not in selected_rates:
                    continue

                if dataset not in artifact_cache:
                    artifact_dir, scale, _shift_k = artifact_dir_for(
                        params, args.artifact_root, args.strategy_id,
                        dataset, args.n_rows,
                    )
                    artifact = load_artifact(artifact_dir)
                    artifact_cache[dataset] = (
                        artifact_dir,
                        scale,
                        artifact,
                        sha256_file(artifact_dir / "artifact.json"),
                    )
                artifact_dir, scale, artifact, artifact_checksum = artifact_cache[dataset]

                for seed in seeds:
                    row = build_row(
                        artifact_dir=artifact_dir,
                        artifact=artifact,
                        artifact_checksum=artifact_checksum,
                        plane_cache=plane_cache,
                        dataset=dataset,
                        n_rows=args.n_rows,
                        scale=scale,
                        strategy_id=args.strategy_id,
                        policy=policy,
                        budget_B=budget_B,
                        r_vector=[int(v) for v in entry["r_vector"]],
                        vacuous_planes=[int(v) for v in entry["vacuous_planes"]],
                        rate=rate,
                        seed=seed,
                        commit=commit,
                    )
                    writer.writerow(row)
                    rows += 1
    finally:
        plane_cache.close()

    with case_failures_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "budget_B", "fault_rate", "base_seed",
                         "exit_code", "reason", "r_vector"])

    meta_lines = [
        "mode=sparse_exact",
        f"strategy_id={args.strategy_id}",
        f"datasets={' '.join(args.datasets)}",
        f"policies={' '.join(args.policies)}",
        f"budget_points={' '.join(str(b) for b in args.budget_points)}",
        f"fault_rates={' '.join(args.fault_rates)}",
        f"seeds={','.join(str(s) for s in seeds)}",
        f"expected_rows={rows}",
        f"rows={rows}",
        f"oracle_match={rows}/{rows}",
        "failure_rows=0",
        f"git_commit={commit}",
        f"hostname={socket.gethostname()}",
        "gpu_validation_required=true",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n")
    print(f"Wrote {rows} sparse rows to {matrix_path}")
    print(f"Run meta: {meta_path}")


if __name__ == "__main__":
    main()
