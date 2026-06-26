"""Tests for run_phase2_sparse_matrix.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from generate_phase2_fault_plans import derive_replica_seed, generate_fault_entries
from phase2_vote import vote_plane
from run_phase2_sparse_matrix import PlaneCache, build_row


def clean_sum(planes: list[bytes]) -> int:
    total = 0
    for p, data in enumerate(planes):
        total += sum(data) * (1 << (8 * (7 - p)))
    return total


def materialized_expected(
    planes: list[bytes],
    dataset: str,
    n_rows: int,
    rate: str,
    seed: int,
    r_vector: list[int],
    vacuous_planes: list[int],
) -> tuple[int, int, int, int]:
    voted_planes: list[bytes] = []
    total_resolved = 0
    total_detected = 0
    total_undetected = 0
    vacuous = set(vacuous_planes)
    for p, clean in enumerate(planes):
        r_p = r_vector[p]
        if r_p <= 0 or p in vacuous:
            voted_planes.append(clean)
            total_resolved += n_rows
            continue

        replicas = []
        for j in range(r_p):
            data = bytearray(clean)
            replica_seed = derive_replica_seed(seed, dataset, p, rate, j)
            for entry in generate_fault_entries(n_rows, float(rate), replica_seed):
                data[entry["offset"]] ^= entry["mask"]
            replicas.append(bytes(data))

        voted, stats = vote_plane(replicas, clean)
        voted_planes.append(voted)
        total_resolved += stats["resolved_correctly"]
        total_detected += stats["detected_mismatch"]
        total_undetected += stats["undetected_corruption"]

    return (
        clean_sum(voted_planes),
        total_resolved,
        total_detected,
        total_undetected,
    )


def test_sparse_row_matches_materialized_vote() -> None:
    n_rows = 64
    dataset = "tiny"
    rate = "2e-01"
    seed = 3
    r_vector = [1, 1, 2, 1, 1, 1, 1, 1]
    vacuous_planes = [0, 1]
    planes = [
        bytes(((p * 17 + i * 3) % 256 for i in range(n_rows)))
        for p in range(8)
    ]
    expected_sum, resolved, detected, undetected = materialized_expected(
        planes, dataset, n_rows, rate, seed, r_vector, vacuous_planes,
    )

    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp)
        for p, data in enumerate(planes):
            (artifact_dir / f"plane_{p}.bin").write_bytes(data)
        artifact = {
            "dataset": dataset,
            "n_rows": n_rows,
            "scale": 10,
            "plane_nonzero_count": [sum(1 for b in data if b) for data in planes],
            "plane_nonzero_fraction": [
                sum(1 for b in data if b) / n_rows for data in planes
            ],
            "clean_encoded_sum": str(clean_sum(planes)),
        }
        (artifact_dir / "artifact.json").write_text(json.dumps(artifact))

        cache = PlaneCache()
        try:
            row = build_row(
                artifact_dir=artifact_dir,
                artifact=artifact,
                artifact_checksum="test",
                plane_cache=cache,
                dataset=dataset,
                n_rows=n_rows,
                scale=10,
                strategy_id="per_dataset",
                policy="graded_naive",
                budget_B=sum(r_vector),
                r_vector=r_vector,
                vacuous_planes=vacuous_planes,
                rate=rate,
                seed=seed,
                commit="test",
            )
        finally:
            cache.close()

    assert int(row["expected_voted_sum"]) == expected_sum
    assert int(row["gpu_voted_sum"]) == expected_sum
    assert int(row["resolved_correctly_count"]) == resolved
    assert int(row["detected_mismatch_count"]) == detected
    assert int(row["undetected_corruption_count"]) == undetected
    assert row["oracle_match"] == "true"
