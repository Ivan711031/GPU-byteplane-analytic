#!/usr/bin/env python3
"""Phase 2 policy catalogue generator.

Generates r-vector policies for all (dataset, budget_point, policy) combinations
from the Phase 2 sensitivity profile.

Usage:
    python3 scripts/phase2_policy_catalogue.py \
        --sensitivity-profile results/phase2_sensitivity_profile.json \
        --budget-points 8 12 16 20 24 \
        --datasets sensor uniform heavy_tailed zipfian \
        --output results/policy_catalogue.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def policy_uniform(B: int) -> list[int]:
    """Distribute B evenly across all 8 planes."""
    base = B // 8
    rem = B % 8
    return [base + 1 if i < rem else base for i in range(8)]


def policy_graded_vacuous_aware(
    B: int,
    vacuous_planes: list[int],
    vacuous_aware_ranking: dict[str, dict[str, Any]],
) -> list[int]:
    """Distribute B with vacuous planes at r=1, rest graded by rank."""
    vp_set = set(vacuous_planes)
    r = [0] * 8
    for p in vp_set:
        r[p] = 1

    B_avail = B - len(vp_set)
    occupied = sorted(
        [(int(k), v["rank"]) for k, v in vacuous_aware_ranking.items()],
        key=lambda x: x[1],
    )
    n = len(occupied)
    if n == 0:
        return r

    base = B_avail // n
    rem = B_avail % n
    for i, (plane_id, _) in enumerate(occupied):
        r[plane_id] = base + (1 if i < rem else 0)
    return r


def policy_graded_naive(
    B: int,
    naive_ranking: dict[str, dict[str, Any]],
) -> list[int]:
    """Distribute B across all 8 planes graded by naive rank (no vacuous handling)."""
    ranked = sorted(
        [(int(k), v["rank"]) for k, v in naive_ranking.items()],
        key=lambda x: x[1],
    )
    n = len(ranked)
    base = B // n
    rem = B % n
    r = [0] * 8
    for i, (plane_id, _) in enumerate(ranked):
        r[plane_id] = base + (1 if i < rem else 0)
    return r


def build_catalogue(
    profile_path: Path,
    budget_points: list[int],
    datasets: list[str],
) -> dict[str, Any]:
    """Build the full policy catalogue from a sensitivity profile."""
    profile = json.loads(profile_path.read_text())
    entries: list[dict[str, Any]] = []
    ds_profiles = profile["datasets"]

    fault_rates = ["1e-07", "1e-06", "1e-05"]

    for dataset in datasets:
        if dataset not in ds_profiles:
            print(f"Warning: dataset '{dataset}' not found in profile, skipping",
                  file=sys.stderr)
            continue

        ds_data = ds_profiles[dataset]

        for fault_rate_str in fault_rates:
            if fault_rate_str not in ds_data:
                print(f"Warning: fault_rate '{fault_rate_str}' not found for "
                      f"dataset '{dataset}', skipping",
                      file=sys.stderr)
                continue

            th = ds_data[fault_rate_str]

            vacuous_planes = th["vacuous_planes"]
            vacuous_count = len(vacuous_planes)
            vacuous_aware_ranking = th["vacuous_aware_ranking"]
            naive_ranking = th["naive_ranking"]

            for B in budget_points:
                # Uniform
                r_u = policy_uniform(B)
                entries.append({
                    "dataset": dataset,
                    "fault_rate": fault_rate_str,
                    "budget_B": B,
                    "policy": "uniform",
                    "r_vector": r_u,
                    "vacuous_planes": vacuous_planes,
                    "occupied_count": 8 - vacuous_count,
                    "budget_avail": B,
                    "notes": "degeneracy: uniform == graded_vacuous_aware at B=8"
                    if B == 8 else None,
                })

                # Graded vacuous aware — skip at B=8 where uniform == vacuous_aware
                if B > 8:
                    r_va = policy_graded_vacuous_aware(
                        B, vacuous_planes, vacuous_aware_ranking,
                    )
                    B_avail_va = B - vacuous_count
                    occ_va = 8 - vacuous_count
                    entries.append({
                        "dataset": dataset,
                        "fault_rate": fault_rate_str,
                        "budget_B": B,
                        "policy": "graded_vacuous_aware",
                        "r_vector": r_va,
                        "vacuous_planes": vacuous_planes,
                        "occupied_count": occ_va,
                        "budget_avail": B_avail_va,
                        "notes": (
                            f"vacuous planes at r=1, budget_avail={B_avail_va} "
                            f"across {occ_va} occupied"
                        ),
                    })

                # Graded naive
                r_gn = policy_graded_naive(B, naive_ranking)
                entries.append({
                    "dataset": dataset,
                    "fault_rate": fault_rate_str,
                    "budget_B": B,
                    "policy": "graded_naive",
                    "r_vector": r_gn,
                    "vacuous_planes": vacuous_planes,
                    "occupied_count": 8,
                    "budget_avail": B,
                    "notes": None,
                })

    catalogue: dict[str, Any] = {
        "metadata": {
            "created_at": "2026-06-01",
            "source_profile": str(profile_path),
            "budget_points": budget_points,
            "policies": ["uniform", "graded_vacuous_aware", "graded_naive"],
        },
        "entries": entries,
    }
    return catalogue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensitivity-profile",
        type=Path,
        required=True,
        help="Path to phase2_sensitivity_profile.json",
    )
    parser.add_argument(
        "--budget-points",
        type=int,
        nargs="+",
        default=[8, 12, 16, 20, 24],
        help="Budget points B",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sensor", "uniform", "heavy_tailed", "zipfian"],
        help="Datasets to include",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for policy_catalogue.json",
    )
    args = parser.parse_args()

    if not args.sensitivity_profile.exists():
        raise SystemExit(
            f"Sensitivity profile not found: {args.sensitivity_profile}"
        )

    catalogue = build_catalogue(
        args.sensitivity_profile,
        args.budget_points,
        args.datasets,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalogue, indent=2) + "\n")
    print(f"Wrote {len(catalogue['entries'])} entries to {args.output}")


if __name__ == "__main__":
    main()
