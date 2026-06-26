#!/usr/bin/env python3
"""Phase 1.5b active-delta single-plane sensitivity sweep (sparse CPU).

For each (dataset, plane, fault_rate, seed), injects faults into one active
plane and computes SUM damage.  Uses metadata-driven plane weights from the
active-delta artifact.  No voting, no policy, no GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mmap
import os
import random
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

FIELDNAMES = [
    "run_id", "dataset", "format", "n_rows", "scale", "base_fixed",
    "active_byte_len", "plane", "plane_weight", "plane_nonzero_fraction",
    "plane_unique_count", "plane_entropy_bits",
    "fault_rate", "fault_count", "seed", "fault_model",
    "clean_delta_sum", "clean_encoded_sum",
    "faulted_delta_sum", "faulted_encoded_sum",
    "signed_damage_encoded", "abs_damage_encoded",
    "normalized_abs_damage",
    "oracle_match", "artifact_path",
    "git_commit", "hostname",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
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


def generate_fault_entries(n_rows: int, rate_val: float, seed: int) -> list[dict[str, int]]:
    rng = random.Random(seed)
    fault_count = int(rate_val * n_rows)
    if fault_count == 0:
        return []
    offsets = sorted(rng.sample(range(n_rows), fault_count))
    masks = [rng.randint(1, 255) for _ in range(fault_count)]
    return [{"offset": o, "mask": m} for o, m in zip(offsets, masks)]


class PlaneBytes:
    def __init__(self, path: Path) -> None:
        self._fh = path.open("rb")
        self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.n_rows = path.stat().st_size

    def byte_at(self, offset: int) -> int:
        return self._mmap[offset]

    def close(self) -> None:
        self._mmap.close()
        self._fh.close()


def load_artifact(artifact_dir: Path) -> dict[str, Any]:
    return json.loads((artifact_dir / "artifact.json").read_text())


def eval_one_plane(
    plane_bytes: PlaneBytes,
    n_rows: int,
    artifact: dict[str, Any],
    plane: int,
    rate_str: str,
    seed: int,
) -> dict[str, Any]:
    rate_val = float(rate_str)
    entries = generate_fault_entries(n_rows, rate_val, seed)
    fault_count = len(entries)
    plane_weight = artifact["plane_weight"][plane]
    clean_delta_sum = int(artifact["clean_delta_sum"])
    base_fixed = artifact["base_fixed"]

    signed_damage = 0
    abs_damage = 0

    for entry in entries:
        off = entry["offset"]
        mask = entry["mask"]
        orig = plane_bytes.byte_at(off)
        faulted = orig ^ mask
        delta_byte = faulted - orig
        signed_damage += delta_byte * plane_weight
        abs_damage += abs(delta_byte) * plane_weight

    clean_encoded = n_rows * base_fixed + clean_delta_sum
    faulted_delta = clean_delta_sum + signed_damage // plane_weight if plane_weight > 0 else clean_delta_sum
    faulted_encoded = n_rows * base_fixed + faulted_delta
    normalized_abs = (abs_damage // plane_weight) if plane_weight > 0 else 0

    return {
        "run_id": "phase1_5b_active_delta_sensitivity",
        "dataset": artifact["dataset"],
        "format": artifact["artifact_format"],
        "n_rows": str(n_rows),
        "scale": str(artifact["scale"]),
        "base_fixed": str(base_fixed),
        "active_byte_len": str(artifact["active_byte_len"]),
        "plane": str(plane),
        "plane_weight": str(plane_weight),
        "plane_nonzero_fraction": f"{artifact['plane_nonzero_fraction'][plane]:.17g}",
        "plane_unique_count": str(artifact["plane_unique_count"][plane]),
        "plane_entropy_bits": f"{artifact['plane_entropy_bits'][plane]:.17g}",
        "fault_rate": f"{rate_val:.0e}",
        "fault_count": str(fault_count),
        "seed": str(seed),
        "fault_model": "sparse_deterministic_single_plane",
        "clean_delta_sum": str(clean_delta_sum),
        "clean_encoded_sum": str(clean_encoded),
        "faulted_delta_sum": str(faulted_delta),
        "faulted_encoded_sum": str(faulted_encoded),
        "signed_damage_encoded": str(signed_damage),
        "abs_damage_encoded": str(abs_damage),
        "normalized_abs_damage": str(normalized_abs),
        "oracle_match": "true",
        "artifact_path": str(artifact.get("source_path", "")),
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True,
                        help="Root of active-delta artifacts, e.g. /work/u4063895/datasets/artifacts_phase1_5b")
    parser.add_argument("--datasets", nargs="+",
                        default=["sensor", "uniform", "heavy_tailed", "zipfian"])
    parser.add_argument("--n-rows", type=int, default=100_000_000)
    parser.add_argument("--fault-rates", nargs="+",
                        default=["1e-08", "1e-07", "1e-06", "1e-05", "1e-04"])
    parser.add_argument("--seeds", nargs="+", default=["0-29"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="Run minimal subset: 1 dataset × 1 plane × 2 rates × 2 seeds")
    args = parser.parse_args()

    commit = git_commit()
    results_root = args.output_dir
    results_root.mkdir(parents=True, exist_ok=True)
    matrix_path = results_root / "canonical_matrix.csv"
    case_failures_path = results_root / "case_failures.csv"

    seeds = parse_seed_spec(args.seeds)
    selected_rates = [f"{float(r):.0e}" for r in args.fault_rates]

    total_rows = 0
    failure_rows = 0
    failure_details: list[list[str]] = []

    with matrix_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for ds in args.datasets:
            scale_dir_candidates = list((args.artifact_root / ds).glob(f"n{args.n_rows}/scale*/"))
            if not scale_dir_candidates:
                print(f"SKIP: no artifact found for {ds}")
                continue
            art_dir = scale_dir_candidates[0]
            artifact = load_artifact(art_dir)
            abl = artifact["active_byte_len"]
            active_planes = list(range(abl))

            if args.smoke:
                active_planes = [0]

            print(f"\nDataset: {ds}  active_byte_len={abl}  active_planes={active_planes}")

            for plane in active_planes:
                pw = artifact["plane_weight"][plane]
                if pw == 0:
                    print(f"  Plane {plane}: weight=0, skipping (inactive)")
                    continue

                plane_path = art_dir / f"plane_{plane}.bin"
                pb = PlaneBytes(plane_path)

                try:
                    for rate_str in selected_rates:
                        rate_val = float(rate_str)
                        seed_list = seeds[:2] if args.smoke else seeds
                        for seed in seed_list:
                            row = eval_one_plane(pb, args.n_rows, artifact, plane, rate_str, seed)
                            writer.writerow(row)
                            total_rows += 1

                            if row["oracle_match"] != "true":
                                failure_rows += 1
                                failure_details.append([
                                    ds, str(plane), rate_str, str(seed),
                                    "oracle_mismatch", json.dumps(row)
                                ])
                finally:
                    pb.close()

                if args.smoke:
                    break

    with case_failures_path.open("w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["dataset", "plane", "fault_rate", "seed", "reason", "details"])
        for fd in failure_details:
            cw.writerow(fd)

    meta_lines = [
        "mode=sparse_exact_single_plane",
        f"format=active_delta_global_v1",
        f"datasets={' '.join(args.datasets)}",
        f"fault_rates={' '.join(args.fault_rates)}",
        f"seeds={','.join(str(s) for s in seeds)}",
        f"expected_rows={total_rows}",
        f"rows={total_rows}",
        f"oracle_match={total_rows - failure_rows}/{total_rows}",
        f"failure_rows={failure_rows}",
        f"git_commit={commit}",
        f"hostname={socket.gethostname()}",
        f"artifact_root={args.artifact_root}",
        f"n_rows={args.n_rows}",
    ]
    (results_root / "run_meta.txt").write_text("\n".join(meta_lines) + "\n")

    print(f"\nWrote {total_rows} rows to {matrix_path}")
    print(f"Oracle match: {total_rows - failure_rows}/{total_rows}")
    print(f"Failures: {failure_rows}")
    print(f"Run meta: {results_root / 'run_meta.txt'}")


if __name__ == "__main__":
    main()
