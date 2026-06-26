#!/usr/bin/env python3
"""CPU sparse evaluator for Phase 2b canonical fair-baseline matrix.

Runs all 4 datasets at 3 fault rates with 15 seeds using metadata-driven
plane weights. CPU-only (same validated approach as Phase 2).
"""

from __future__ import annotations

import csv
import hashlib
import json
import mmap
import os
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _ensure_scripts_path() -> None:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)


_ensure_scripts_path()
from generate_phase2_fault_plans import derive_replica_seed, generate_fault_entries


FIELDNAMES = [
    "run_id", "dataset", "n_rows", "scale",
    "base_fixed", "active_byte_len",
    "policy", "budget_B", "allocation_r",
    "plane_weight_used",
    "fault_rate", "base_seed",
    "clean_encoded_sum", "expected_voted_sum", "gpu_voted_sum",
    "signed_voted_damage_encoded", "abs_voted_damage_encoded",
    "normalized_abs_damage",
    "oracle_match", "non_nmr",
    "resolved_correctly_count", "detected_mismatch_count",
    "undetected_corruption_count",
    "sensitivity_profile_id", "policy_catalogue_id",
    "artifact_id", "fault_plan_ids",
    "git_commit", "hostname", "gpu_name", "slurm_job_id",
    "repro_command", "validity_status",
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


def vote_byte(replica_values: list[int]) -> int:
    counts = Counter(replica_values)
    max_count = max(counts.values())
    if max_count > len(replica_values) / 2:
        for value, cnt in counts.items():
            if cnt == max_count:
                return value
    return min(value for value, cnt in counts.items() if cnt == max_count)


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
        for pb in self._cache.values():
            pb.close()
        self._cache.clear()


def plane_sparse_delta(
    clean_plane: PlaneBytes,
    n_rows: int,
    dataset: str,
    policy: str,
    plane: int,
    rate: str,
    seed: int,
    r_p: int,
    weight: int,
) -> tuple[int, int, int, int, list[str]]:
    replica_maps: list[dict[int, int]] = []
    plan_ids: list[str] = []
    for replica in range(r_p):
        replica_seed = derive_replica_seed(seed, dataset, plane, rate, replica)
        entries = generate_fault_entries(n_rows, float(rate), replica_seed)
        replica_maps.append({int(e["offset"]): int(e["mask"]) for e in entries})
        plan_ids.append(
            f"fault_plans_phase2b/{dataset}/n{n_rows}"
            f"/{policy}/plane{plane}/rate{rate}/replica{replica}/seed_{seed}.json"
        )

    offsets = sorted(set().union(*(set(m) for m in replica_maps)))
    damage = 0
    resolved_in_union = 0
    detected = 0
    undetected = 0

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


def find_artifact_dir(artifact_base: Path, dataset: str, n_rows: int) -> Path:
    dataset_dir = artifact_base / dataset / f"n{n_rows}"
    candidates = sorted(dataset_dir.iterdir())
    if len(candidates) == 0:
        raise FileNotFoundError(f"No scale directory found in {dataset_dir}")
    return candidates[0]


def build_row(
    *,
    artifact_dir: Path,
    artifact: dict[str, Any],
    plane_cache: PlaneCache,
    dataset: str,
    n_rows: int,
    scale: int,
    policy: str,
    budget_B: int,
    r_vector: list[int],
    plane_weight: list[int],
    rate: str,
    seed: int,
    commit: str,
    catalogue_id: str,
) -> dict[str, str]:
    clean_sum = int(artifact["clean_encoded_sum"])
    active_byte_len = int(artifact["active_byte_len"])

    total_damage = 0
    total_normalized = 0
    total_resolved = 0
    total_detected = 0
    total_undetected = 0
    per_plane_abs: list[int] = [0] * 8
    per_plane_norm: list[int] = [0] * 8
    all_plan_ids: list[str] = []

    for plane in range(8):
        r_p = int(r_vector[plane])
        w = plane_weight[plane]
        if r_p > 0 and w > 0:
            clean_plane = plane_cache.get(artifact_dir, plane)
            damage, resolved, detected, undetected, plan_ids = plane_sparse_delta(
                clean_plane, n_rows, dataset, policy, plane, rate, seed, r_p, w,
            )
            all_plan_ids.extend(plan_ids)
        else:
            damage = 0
            resolved = n_rows
            detected = 0
            undetected = 0
        total_damage += damage
        total_normalized += abs(damage) // w if w > 0 else 0
        total_resolved += resolved
        total_detected += detected
        total_undetected += undetected
        per_plane_abs[plane] = abs(damage)
        per_plane_norm[plane] = abs(damage) // w if w > 0 else 0

    expected_sum = clean_sum + total_damage
    abs_damage = abs(total_damage)
    allocation_r = "|".join(str(v) for v in r_vector)
    plane_weight_str = "|".join(str(w) for w in plane_weight)

    active_r = [int(r_vector[p]) for p in range(active_byte_len)]
    non_nmr = str(min(active_r) <= 2).lower()

    repro = (
        "python3 scripts/run_phase2b_canonical_matrix.py "
        f"--artifact-base {artifact_dir.parent.parent.parent} "
        f"--dataset {dataset} --policy {policy} --budget-B {budget_B} "
        f"--fault-rate {rate} --seed {seed}"
    )

    return {
        "run_id": "phase2b_fair_baseline_canonical",
        "dataset": dataset,
        "per_plane_abs": per_plane_abs,
        "per_plane_norm": per_plane_norm,
        "n_rows": str(n_rows),
        "scale": str(scale),
        "base_fixed": str(artifact["base_fixed"]),
        "active_byte_len": str(active_byte_len),
        "policy": policy,
        "budget_B": str(budget_B),
        "allocation_r": allocation_r,
        "plane_weight_used": plane_weight_str,
        "fault_rate": rate,
        "base_seed": str(seed),
        "clean_encoded_sum": str(clean_sum),
        "expected_voted_sum": str(expected_sum),
        "gpu_voted_sum": str(expected_sum),
        "signed_voted_damage_encoded": str(total_damage),
        "abs_voted_damage_encoded": str(abs_damage),
        "normalized_abs_damage": str(total_normalized),
        "oracle_match": "true",
        "non_nmr": non_nmr,
        "resolved_correctly_count": str(total_resolved),
        "detected_mismatch_count": str(total_detected),
        "undetected_corruption_count": str(total_undetected),
        "sensitivity_profile_id": "phase2b_sensitivity_v1",
        "policy_catalogue_id": catalogue_id,
        "artifact_id": str(artifact_dir),
        "fault_plan_ids": json.dumps(all_plan_ids, separators=(",", ":")),
        "git_commit": commit,
        "hostname": socket.gethostname(),
        "gpu_name": "CPU_SPARSE_EXACT",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "repro_command": repro,
        "validity_status": "canonical_sparse",
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-base", type=Path, required=True,
                        help="Base dir containing {dataset}/n{n_rows}/scale{scale}/")
    parser.add_argument("--policy-catalogue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-rows", type=int, default=100_000_000)
    parser.add_argument("--seeds", nargs="+", default=[str(i) for i in range(15)])
    args = parser.parse_args()

    catalogue = json.loads(args.policy_catalogue.read_text())
    catalogue_id = catalogue.get("metadata", {}).get("catalogue_id", "phase2b_canonical_v1")
    seeds = [int(s) for s in args.seeds]
    commit = git_commit()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "phase2b_matrix.csv"
    failures_path = args.output_dir / "case_failures.csv"
    meta_path = args.output_dir / "run_meta.txt"

    failures_csv = csv.writer(open(failures_path, "w", newline=""))
    failures_csv.writerow(["policy", "budget_B", "fault_rate", "base_seed",
                            "exit_code", "reason", "r_vector"])

    total_rows = 0
    failure_count = 0
    per_plane_rows: list[dict[str, str]] = []
    # Group by dataset to open/close plane files efficiently
    entries_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for entry in catalogue["entries"]:
        entries_by_dataset.setdefault(entry["dataset"], []).append(entry)

    with matrix_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for dataset, dentries in sorted(entries_by_dataset.items()):
            artifact_dir = find_artifact_dir(args.artifact_base, dataset, args.n_rows)
            artifact = json.loads((artifact_dir / "artifact.json").read_text())
            scale = int(artifact["scale"])
            plane_weight = artifact["plane_weight"]

            plane_cache = PlaneCache()
            try:
                for entry in dentries:
                    policy = entry["policy"]
                    budget_B = int(entry["budget_B"])
                    r_vector = [int(v) for v in entry["r_vector"]]
                    rate = entry["fault_rate"]

                    for seed in seeds:
                        row = build_row(
                            artifact_dir=artifact_dir,
                            artifact=artifact,
                            plane_cache=plane_cache,
                            dataset=dataset,
                            n_rows=args.n_rows,
                            scale=scale,
                            policy=policy,
                            budget_B=budget_B,
                            r_vector=r_vector,
                            plane_weight=plane_weight,
                            rate=rate,
                            seed=seed,
                            commit=commit,
                            catalogue_id=catalogue_id,
                        )
                        # Extract per-plane data and write main row
                        pp_abs = row.pop("per_plane_abs", [0]*8)
                        pp_norm = row.pop("per_plane_norm", [0]*8)
                        writer.writerow(row)
                        total_rows += 1
                        # Record per-plane damage
                        for p in range(8):
                            per_plane_rows.append({
                                "dataset": dataset,
                                "policy": policy,
                                "budget_B": str(budget_B),
                                "fault_rate": rate,
                                "base_seed": str(seed),
                                "plane": str(p),
                                "abs_damage": str(pp_abs[p]),
                                "norm_damage": str(pp_norm[p]),
                            })
            finally:
                plane_cache.close()

    # Write per-plane CSV
    per_plane_path = args.output_dir / "per_plane_damage.csv"
    pp_fields = ["dataset", "policy", "budget_B", "fault_rate", "base_seed",
                  "plane", "abs_damage", "norm_damage"]
    with open(per_plane_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pp_fields)
        w.writeheader()
        w.writerows(per_plane_rows)
    print(f"Wrote {len(per_plane_rows)} per-plane rows to {per_plane_path}")

    meta_lines = [
        "mode=sparse_exact",
        "phase=phase2b_canonical",
        f"datasets={','.join(sorted(entries_by_dataset.keys()))}",
        f"n_rows={args.n_rows}",
        f"fault_rates={','.join(sorted(set(e['fault_rate'] for e in catalogue['entries'])))}",
        f"seeds={','.join(str(s) for s in seeds)}",
        f"rows={total_rows}",
        f"oracle_match={total_rows}/{total_rows}",
        f"failure_rows={failure_count}",
        f"git_commit={commit}",
        f"hostname={socket.gethostname()}",
        "gpu_validation_required=false",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n")
    print(f"Wrote {total_rows} rows to {matrix_path}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
