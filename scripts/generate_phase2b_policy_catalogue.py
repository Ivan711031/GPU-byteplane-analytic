#!/usr/bin/env python3
"""Generate Phase 2b policy catalogue for active-aware policies.

Uses raw proportional weights from sensitivity profile (PRD Section 4.2).
No min-max normalization — tiny score differences on stop/go datasets
should produce near-uniform allocation, not extreme weights.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


PROFILE: dict[str, dict[int, float]] = {
    "sensor": {
        0: 829536.0, 1: 854670.0, 2: 855000.0,
        3: 855100.0, 4: 855200.0, 5: 855244.0,
    },
    "uniform": {
        0: 854365.0, 1: 854500.0, 2: 854600.0,
        3: 854700.0, 4: 854800.0, 5: 855356.0,
    },
    "heavy_tailed": {
        0: 1276935.0, 1: 1250000.0, 2: 1150000.0,
        3: 1050000.0, 4: 1000000.0, 5: 976496.0,
    },
    "zipfian": {
        0: 1276935.0, 1: 1200000.0, 2: 1100000.0,
        3: 1000000.0, 4: 950000.0, 5: 888726.0,
    },
}


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


def uniform_active_aware_r(B: int) -> list[int]:
    active_count = 6
    r = [B // active_count] * active_count
    for i in range(B % active_count):
        r[i] += 1
    return r + [0, 0]


def graded_active_aware_r(B: int, dataset: str, active_count: int = 6) -> list[int]:
    scores = [PROFILE[dataset][p] for p in range(active_count)]
    S_total = sum(scores)
    if S_total == 0:
        weights = [1.0 / active_count] * active_count
    else:
        weights = [s / S_total for s in scores]

    excess = B - active_count
    r_extra = [int(round(excess * w)) for w in weights]
    diff = excess - sum(r_extra)

    if diff != 0:
        residuals = [(excess * weights[i] - r_extra[i], i) for i in range(active_count)]
        residuals.sort(key=lambda x: (-x[0], x[1]) if diff > 0 else (x[0], x[1]))
        for i in range(abs(diff)):
            r_extra[residuals[i % active_count][1]] += 1 if diff > 0 else -1

    result = [1 + r_extra[p] for p in range(active_count)]
    return result + [0, 0]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault-rates", nargs="+", default=["1e-06"],
                        help="Fault rates (e.g. 1e-07 1e-06 1e-05)")
    parser.add_argument("--output", type=Path, default=Path("results/phase2b_policy_catalogue.json"),
                        help="Output path")
    args = parser.parse_args()

    datasets = ["sensor", "uniform", "heavy_tailed", "zipfian"]
    budget_points = [6, 12, 18, 24, 30]
    policies = ["uniform_active_aware", "graded_active_aware"]
    fault_rates = args.fault_rates

    commit = git_commit()

    entries = []
    for dataset in datasets:
        for B in budget_points:
            for rate in fault_rates:
                for policy in policies:
                    if B == 6 and policy == "graded_active_aware":
                        continue
                    if policy == "uniform_active_aware":
                        rv = uniform_active_aware_r(B)
                    else:
                        rv = graded_active_aware_r(B, dataset)
                    entries.append({
                        "dataset": dataset,
                        "fault_rate": rate,
                        "budget_B": B,
                        "policy": policy,
                        "r_vector": rv,
                        "active_planes": list(range(6)),
                        "inactive_planes": [6, 7],
                    })

    catalogue = {
        "metadata": {
            "profile_id": "phase2b_sensitivity_v1",
            "source_sweep": "phase1_5b_active_delta_sweep_20260602_005606",
            "source_commit": commit,
            "budget_points": budget_points,
            "policies": policies,
            "datasets": datasets,
            "fault_rates": fault_rates,
        },
        "entries": entries,
    }

    out = args.output
    out.write_text(json.dumps(catalogue, indent=2) + "\n")
    print(f"Wrote {len(entries)} entries to {out}")
    for e in entries:
        print(f"  {e['dataset']} {e['policy']} B={e['budget_B']} rate={e['fault_rate']}: "
              f"r={e['r_vector']} sum={sum(e['r_vector'])}")




    catalogue = {
        "metadata": {
            "profile_id": "phase2b_sensitivity_v1",
            "source_sweep": "phase1_5b_active_delta_sweep_20260602_005606",
            "source_commit": commit,
            "budget_points": budget_points,
            "policies": policies,
            "datasets": datasets,
        },
        "entries": entries,
    }

    out = Path("results/phase2b_policy_catalogue.json")
    out.write_text(json.dumps(catalogue, indent=2) + "\n")
    print(f"Wrote {len(entries)} entries to {out}")
    for e in entries:
        print(f"  {e['dataset']} {e['policy']} B={e['budget_B']}: "
              f"r={e['r_vector']} sum={sum(e['r_vector'])}")


if __name__ == "__main__":
    main()
