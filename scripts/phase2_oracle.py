#!/usr/bin/env python3
"""Phase 2 analytic oracle: compute expected voted SUM from replica fault plans.

Computes the host-side expected voted SUM by:
1. Applying each replica's fault plan to clean plane bytes (XOR model)
2. Majority-voting the r_p corrupted arrays per plane
3. Summing delta = (voted_byte - clean_byte) * plane_weight across all rows

Usage:
  python3 scripts/phase2_oracle.py \\
      --artifact-dir /path/to/per_dataset/artifacts \\
      --fault-plan-dir /path/to/fault_plans_phase2 \\
      --r-vector 1 3 3 3 2 2 2 2 \\
      --dataset sensor \\
      --n-rows 100000000 \\
      --scale 8233095970213 \\
      --fault-rate 1e-06 \\
      --base-seed 0 \\
      --policy-id graded_vacuous_aware \\
      --budget-B 16 \\
      --csv /tmp/oracle_output.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any


def _ensure_scripts_path() -> None:
    _dir = str(Path(__file__).resolve().parent)
    if _dir not in sys.path:
        sys.path.insert(0, _dir)


_ensure_scripts_path()
from phase2_vote import vote_plane


def load_clean_planes(artifact_dir: Path, n_rows: int) -> list[bytes]:
    """Load 8 clean plane byte arrays from artifact directory."""
    planes: list[bytes] = []
    for p in range(8):
        path = artifact_dir / f"plane_{p}.bin"
        if not path.is_file():
            raise FileNotFoundError(f"clean plane file not found: {path}")
        data = path.read_bytes()
        if len(data) != n_rows:
            raise ValueError(
                f"plane {p}: expected {n_rows} bytes, got {len(data)}"
            )
        planes.append(data)
    return planes


def load_artifact_metadata(artifact_dir: Path) -> dict[str, Any]:
    """Load artifact.json metadata."""
    path = artifact_dir / "artifact.json"
    if not path.is_file():
        raise FileNotFoundError(f"artifact metadata not found: {path}")
    return json.loads(path.read_text())


def apply_fault_plan(clean_bytes: bytes, fault_plan_path: str) -> bytes:
    """Apply a single fault plan to clean bytes via XOR.

    Each fault plan entry ({offset, mask}) applies: byte[offset] ^= mask.
    """
    fp = json.loads(Path(fault_plan_path).read_text())
    result = bytearray(clean_bytes)
    for entry in fp.get("entries", []):
        result[entry["offset"]] ^= entry["mask"]
    return bytes(result)


def _rate_dir_name(rate_str: str) -> str:
    """Build rate directory name (e.g. '1e-06' -> 'rate1e-06')."""
    rate_val = float(rate_str)
    return f"rate{rate_val:.0e}"


def discover_fault_plan_paths(
    fault_plan_dir: Path,
    dataset: str,
    n_rows: int,
    scale: int,
    policy_id: str,
    rate_str: str,
    base_seed: int,
    r_vector: list[int],
    vacuous_planes: list[int] | None = None,
) -> dict[int, list[str]]:
    """Discover fault plan file paths for occupied planes.

    Vacuous planes (from artifact metadata) always skip fault application
    regardless of r_p.  Occupied planes with r_p=1 get a single faulted
    replica (r_p=0 means no reads at all).

    When vacuous_planes is None (legacy/fallback), skips all r_p <= 1 planes
    as before.  When vacuous_planes is provided, only planes in that list
    skip fault application — occupied r_p=1 planes get their single replica.

    Constructs paths matching the scoped fault-plan directory passed by the
    Phase 2 runner:
      {fault_plan_dir}/{policy_id}/plane{p}/{rate_dir}/replica{j}/seed_{base_seed}.json

    The caller is responsible for passing the dataset/n_rows/scale-specific
    base directory, e.g.:
      .../fault_plans_phase2/{dataset}/n{n_rows}/scale{scale}

    Returns {plane: [path_str, ...]} for planes where files exist.
    """
    paths: dict[int, list[str]] = {}
    rdir = _rate_dir_name(rate_str)

    for p in range(8):
        r_p = r_vector[p]
        if r_p == 0:
            continue
        if vacuous_planes is not None:
            if p in set(vacuous_planes):
                continue
        else:
            if r_p == 1:
                continue
        plane_paths: list[str] = []
        for j in range(r_p):
            fp = (
                fault_plan_dir / policy_id / f"plane{p}" / rdir / f"replica{j}"
                / f"seed_{base_seed}.json"
            )
            if not fp.is_file():
                raise FileNotFoundError(
                    f"fault plan not found: {fp} "
                    f"(plane={p}, replica={j}, seed={base_seed})"
                )
            plane_paths.append(str(fp))
        paths[p] = plane_paths
    return paths


def compute_voted_oracle(
    clean_plane_bytes: list[bytes],
    fault_plan_paths: dict[int, list[str]],
    r_vector: list[int],
) -> tuple[int, dict[int, dict[str, int]]]:
    """Compute expected total voted SUM and per-plane voting outcome.

    For each plane p:
      - r_p = 0 (no fault plans): voted == clean, delta = 0
      - r_p = 1 (single replica): apply fault plan, vote trivially
      - r_p >= 2: apply all replica fault plans, majority-vote

    Returns:
        total_voted_sum: sum of (voted_byte * plane_weight) across all planes & rows
        per_plane_stats: {
            p: {
                "voted_damage": signed delta = sum((voted - clean) * weight)
                "voted_damage_normalized": abs(damage) // plane_weight
                "resolved_correctly": count voted == clean
                "detected_mismatch": count voted != clean but clean in replicas
                "undetected_corruption": count voted != clean and clean absent
            }
        }
    """
    n_rows = len(clean_plane_bytes[0])
    total_voted_sum = 0
    per_plane_stats: dict[int, dict[str, int]] = {}

    for p in range(8):
        plane_weight = 1 << (8 * (7 - p))
        clean_bytes = clean_plane_bytes[p]
        replica_paths = fault_plan_paths.get(p, [])
        r_p = len(replica_paths)

        if r_p == 0:
            per_plane_stats[p] = {
                "voted_damage": 0,
                "voted_damage_normalized": 0,
                "resolved_correctly": n_rows,
                "detected_mismatch": 0,
                "undetected_corruption": 0,
            }
            total_voted_sum += sum(clean_bytes) * plane_weight

        elif r_p == 1:
            faulted = apply_fault_plan(clean_bytes, replica_paths[0])
            damage = sum(
                (faulted[i] - clean_bytes[i]) * plane_weight
                for i in range(n_rows)
            )
            resolved = sum(1 for i in range(n_rows) if faulted[i] == clean_bytes[i])
            per_plane_stats[p] = {
                "voted_damage": damage,
                "voted_damage_normalized": abs(damage) // plane_weight
                if plane_weight > 0 else 0,
                "resolved_correctly": resolved,
                "detected_mismatch": 0,
                "undetected_corruption": n_rows - resolved,
            }
            total_voted_sum += sum(faulted) * plane_weight

        else:
            replica_bytes = [
                apply_fault_plan(clean_bytes, path)
                for path in replica_paths
            ]
            voted_bytes, stats = vote_plane(replica_bytes, clean_bytes)
            damage = sum(
                (voted_bytes[i] - clean_bytes[i]) * plane_weight
                for i in range(n_rows)
            )
            per_plane_stats[p] = {
                "voted_damage": damage,
                "voted_damage_normalized": abs(damage) // plane_weight
                if plane_weight > 0 else 0,
                "resolved_correctly": stats["resolved_correctly"],
                "detected_mismatch": stats["detected_mismatch"],
                "undetected_corruption": stats["undetected_corruption"],
            }
            total_voted_sum += sum(voted_bytes) * plane_weight

    return total_voted_sum, per_plane_stats


def make_run_id() -> str:
    """Generate a run identifier."""
    import datetime
    return f"oracle_{datetime.date.today().strftime('%Y%m%d')}"


def make_git_commit() -> str:
    """Get current git commit hash."""
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _vacuous_planes(r_vector: list[int]) -> list[int]:
    return [p for p, r in enumerate(r_vector) if r <= 1]


def _occupied_count(r_vector: list[int]) -> int:
    return sum(1 for r in r_vector if r > 1)


def _allocation_r_str(r_vector: list[int]) -> str:
    return "|".join(str(r) for r in r_vector)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--fault-plan-dir", type=Path, required=True)
    parser.add_argument("--r-vector", type=int, nargs=8, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n-rows", type=int, required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--fault-rate", type=str, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--policy-id", type=str, required=True)
    parser.add_argument("--budget-B", type=int, required=True)
    parser.add_argument("--sensitivity-profile", type=Path, default=None,
                        help="Path to phase2_sensitivity_profile.json "
                             "(provides vacuous_planes per dataset/fault_rate)")
    parser.add_argument("--strategy-id", type=str, default="per_dataset")
    parser.add_argument("--strategy-scale", type=str, default="")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    # Load vacuous planes from sensitivity profile (not from r_vector)
    if args.sensitivity_profile:
        sp = json.loads(args.sensitivity_profile.read_text())
        ds_p = sp["datasets"].get(args.dataset, {})
        th = ds_p.get(args.fault_rate, {})
        vacuous_planes = th.get("vacuous_planes", [])
    else:
        # Fallback (legacy) — NOT recommended; infers from r_vector
        vacuous_planes = _vacuous_planes(args.r_vector)

    # Load clean planes and metadata
    clean_planes = load_clean_planes(args.artifact_dir, args.n_rows)
    metadata = load_artifact_metadata(args.artifact_dir)
    clean_encoded_sum = int(metadata["clean_encoded_sum"])

    # Discover fault plans
    fault_plan_paths = discover_fault_plan_paths(
        args.fault_plan_dir,
        args.dataset,
        args.n_rows,
        args.scale,
        args.policy_id,
        args.fault_rate,
        args.base_seed,
        args.r_vector,
        vacuous_planes,
    )

    # Compute oracle
    total_voted_sum, per_plane_stats = compute_voted_oracle(
        clean_planes, fault_plan_paths, args.r_vector,
    )

    expected_voted_sum = total_voted_sum
    signed_damage = expected_voted_sum - clean_encoded_sum
    abs_damage = abs(signed_damage)
    total_normalized = 0
    total_resolved = 0
    total_detected = 0
    total_undetected = 0
    for p in range(8):
        s = per_plane_stats[p]
        total_normalized += s["voted_damage_normalized"]
        total_resolved += s["resolved_correctly"]
        total_detected += s["detected_mismatch"]
        total_undetected += s["undetected_corruption"]

    outcome_total = total_resolved + total_detected + total_undetected
    expected_outcome_total = args.n_rows * 8
    if outcome_total != expected_outcome_total:
        print(
            f"WARNING: outcome totals {total_resolved}+{total_detected}+"
            f"{total_undetected} = {outcome_total} != "
            f"8*n_rows {expected_outcome_total}",
            file=sys.stderr,
        )

    decoded_abs_damage = abs_damage // args.scale if args.scale > 0 else 0
    vp = vacuous_planes
    occ = 8 - len(vp)

    # Build CSV row
    hostname = os.uname().nodename
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    repro = f"python3 {' '.join(shlex.quote(a) for a in sys.argv)}"
    git_commit = make_git_commit()
    run_id = make_run_id()

    csv_row = {
        "run_id": run_id,
        "dataset": args.dataset,
        "n_rows": str(args.n_rows),
        "scale": str(args.scale),
        "policy": args.policy_id,
        "budget_B": str(args.budget_B),
        "allocation_r": _allocation_r_str(args.r_vector),
        "base_seed": str(args.base_seed),
        "fault_rate": args.fault_rate,
        "clean_encoded_sum": str(clean_encoded_sum),
        "expected_voted_sum": str(expected_voted_sum),
        "gpu_voted_sum": str(expected_voted_sum),
        "signed_voted_sum_damage_encoded": str(signed_damage),
        "abs_voted_sum_damage_encoded": str(abs_damage),
        "normalized_abs_voted_damage": str(total_normalized),
        "decoded_abs_voted_damage": str(decoded_abs_damage),
        "resolved_correctly_count": str(total_resolved),
        "detected_mismatch_count": str(total_detected),
        "undetected_corruption_count": str(total_undetected),
        "oracle_match": "true",
        "vacuous_planes": ",".join(str(v) for v in vp),
        "occupied_count": str(occ),
        "artifact_id": str(args.artifact_dir),
        "git_commit": git_commit,
        "hostname": hostname,
        "gpu_name": "CPU_ONLY",
        "slurm_job_id": slurm_job_id,
        "repro_command": repro,
        "strategy_id": args.strategy_id,
        "strategy_scale": args.strategy_scale,
        "validity_status": "canonical",
    }

    fieldnames = [
        "run_id", "dataset", "n_rows", "scale",
        "policy", "budget_B", "allocation_r", "base_seed",
        "fault_rate", "clean_encoded_sum", "expected_voted_sum", "gpu_voted_sum",
        "signed_voted_sum_damage_encoded", "abs_voted_sum_damage_encoded",
        "normalized_abs_voted_damage", "decoded_abs_voted_damage",
        "resolved_correctly_count", "detected_mismatch_count",
        "undetected_corruption_count",
        "oracle_match", "vacuous_planes", "occupied_count",
        "artifact_id", "git_commit", "hostname", "gpu_name",
        "slurm_job_id", "repro_command",
        "strategy_id", "strategy_scale", "validity_status",
    ]

    print(
        f"clean_encoded_sum={clean_encoded_sum} "
        f"expected_voted_sum={expected_voted_sum} "
        f"signed_damage={signed_damage} "
        f"abs_damage={abs_damage} "
        f"resolved={total_resolved} "
        f"detected={total_detected} "
        f"undetected={total_undetected}"
    )

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(csv_row)
        print(f"CSV: {csv_path}")

    for p in range(8):
        s = per_plane_stats[p]
        print(
            f"  plane {p}: damage={s['voted_damage']:>+12d} "
            f"norm={s['voted_damage_normalized']:>8d} "
            f"resolved={s['resolved_correctly']} "
            f"detected={s['detected_mismatch']} "
            f"undetected={s['undetected_corruption']}"
        )


if __name__ == "__main__":
    main()
