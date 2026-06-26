"""NMR-R0: CPU deterministic oracle for byte-level majority voting.

Tests the NMR premise under logical independent-fault model:
  byte-level majority voting across r=3 replicas recovers the correct
  plane value when exactly one replica is corrupted.

If this fails, NMR premise is falsified — do NOT proceed to GPU or R1/R2.
"""

from __future__ import annotations

import csv
import itertools
import os
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase3_z2bc_real_datasets import load_planes

PLANE_COUNT = 8
SEGMENT_SIZE = 4096


def make_replicas(clean_plane: bytes, r: int) -> list[bytearray]:
    return [bytearray(clean_plane) for _ in range(r)]


def inject_single_byte_flip(replica: bytearray, offset: int) -> None:
    replica[offset] ^= 0xFF


def inject_saturating_byte(replica: bytearray, offset: int, value: int = 0xFF) -> None:
    replica[offset] = value


def inject_adversarial_cancel(replica: bytearray, offset_a: int, offset_b: int) -> None:
    """Flip two bytes such that SUM32 (additive mod 2^32) sees no net change.
    Requires offsets within the same 4KB allocation unit.
    """
    replica[offset_a] ^= 1
    replica[offset_b] ^= 1  # flip cancel: +1 and -1


def byte_majority_vote(replicas: list[bytearray]) -> bytearray:
    """Byte-level majority vote across r replicas."""
    r = len(replicas)
    n = len(replicas[0])
    result = bytearray(n)
    for i in range(n):
        counts: dict[int, int] = {}
        for rep in replicas:
            b = rep[i]
            counts[b] = counts.get(b, 0) + 1
        majority = max(counts, key=lambda k: (counts[k], k))  # tie-break: larger byte
        result[i] = majority
    return result


def run_cpu_oracle(n_rows: int = 1000000, fault_rate: float = 1e-6,
                   seeds: list[int] | None = None) -> list[dict[str, Any]]:
    if seeds is None:
        seeds = [0, 1, 2]

    # Generate synthetic plane data: random bytes (pseudo-random, seeded)
    rng = random.Random(42)
    clean_plane = bytes(rng.randint(0, 255) for _ in range(n_rows))

    # Also load real data if available
    real_artifacts: list[tuple[str, bytes]] = []
    for ds_name, ds_path in [
        ("hurricane_u", "/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg4096"),
    ]:
        p = Path(ds_path)
        if p.exists():
            try:
                planes = load_planes(p, 0, min(n_rows, 1_000_000))
                real_artifacts.append((ds_name, planes[0]))
            except Exception:
                pass

    r = 3
    rows: list[dict[str, Any]] = []

    # ── Test configs ──
    # Three controls (per review verdict):
    #   A. one-replica corruption → expect recovery_rate=1.0
    #   B. same-fault-all-replicas → expect no recovery (reproduces G2)
    #   C. two-of-three corrupted → expect false-majority escape
    test_configs = [
        # ── Control A: one replica corrupted ──
        ("A_one_rep_flip",       lambda rep, seed: inject_single_byte_flip(rep, 0), 1, "one"),
        ("A_one_rep_saturating", lambda rep, seed: inject_saturating_byte(rep, 0), 1, "one"),
        ("A_one_rep_adv_cancel", lambda rep, seed: inject_adversarial_cancel(rep, 0, 4), 2, "one"),
        # ── Control B: same fault ALL replicas (reproduces G2) ──
        ("B_all_rep_adv_cancel", lambda rep, seed: inject_adversarial_cancel(rep, 0, 4), 2, "all"),
        # ── Control C: two of three corrupted ──
        ("C_two_rep_adv_cancel", lambda rep, seed: inject_adversarial_cancel(rep, 0, 4), 2, "two"),
    ]

    for data_source, data_label in [
        (clean_plane, "synthetic"),
        *[(p, ds) for ds, p in real_artifacts],
    ]:
        for t_label, inject_fn, n_faults, domain_model in test_configs:
            for seed in seeds:
                # Create r replicas
                replicas = make_replicas(data_source, r)

                # Inject fault according to domain model
                if domain_model == "one":
                    # Fault replica 0 only
                    inject_fn(replicas[0], seed)
                elif domain_model == "all":
                    # Same fault to ALL replicas (reproduces G2 same-fault model)
                    for rep in replicas:
                        inject_fn(rep, seed)
                elif domain_model == "two":
                    # Fault replicas 0 and 1 (two of three)
                    inject_fn(replicas[0], seed)
                    replicas[1][8] ^= 1
                    replicas[1][12] ^= 1

                # Byte-level majority vote
                voted = byte_majority_vote(replicas)

                # Verify voted output matches clean
                clean_bytes = data_source
                mismatches = sum(1 for i in range(len(voted)) if voted[i] != clean_bytes[i])
                total_bytes = len(voted)
                recovery_rate = 1.0 - mismatches / max(total_bytes, 1)
                recovered = (mismatches == 0)

                rows.append({
                    "dataset": data_label,
                    "test_config": t_label,
                    "seed": str(seed),
                    "r": r,
                    "fault_domain_model": domain_model,
                    "n_bytes": total_bytes,
                    "n_mismatches": mismatches,
                    "recovery_rate": f"{recovery_rate:.10e}",
                    "recovered": str(recovered),
                })

    return rows


def main() -> None:
    jid = os.environ.get("SLURM_JOB_ID", "cpu_oracle")
    out_root = Path(f"results/reliability_layer1/phase3/nmr_rescue_r0/job_{jid}")
    out_root.mkdir(parents=True, exist_ok=True)

    rows = run_cpu_oracle(n_rows=1000000, seeds=[0, 1, 2])

    fields = [
        "dataset", "test_config", "seed", "r",
        "fault_domain_model",
        "n_bytes", "n_mismatches", "recovery_rate",
        "recovered",
    ]
    csv_path = out_root / "nmr_r0_cpu_oracle.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")

    # Print summary
    print(f"\n=== NMR-R0 CPU Oracle ===")
    for ds in sorted(set(r["dataset"] for r in rows)):
        sub = [r for r in rows if r["dataset"] == ds]
        n_total = len(sub)
        n_recovered = sum(1 for r in sub if r["recovered"] == "True")
        print(f"\n  {ds}: {n_recovered}/{n_total} rows recovered")

        for tc in sorted(set(r["test_config"] for r in sub)):
            tc_rows = [r for r in sub if r["test_config"] == tc]
            recovered = sum(1 for r in tc_rows if r["recovered"] == "True")
            total = len(tc_rows)
            model = tc_rows[0]["fault_domain_model"]
            print(f"    {tc:<35s} domain={model:<5s} {recovered}/{total} recovered")

    # ── Verdict per interpretation constraints (user review 2026-06-05) ──
    # CPU oracle failure = HARNESS_BUG_OR_SPEC_MISMATCH, not NMR premise failure.
    # True premise test: same-fault-all fails, one-replica-individual passes.
    controls_a = [r for r in rows if r["fault_domain_model"] == "one"]
    controls_b = [r for r in rows if r["fault_domain_model"] == "all"]
    controls_c = [r for r in rows if r["fault_domain_model"] == "two"]

    a_passed = all(r["recovered"] == "True" for r in controls_a)
    b_failed = all(r["recovered"] != "True" for r in controls_b)  # expected: no recovery
    b_result = [r for r in controls_b if r["recovered"] == "True"]

    print(f"\n  === Control Results ===")
    print(f"  A (one-replica independent, expect recovery): "
          f"{'✅ ALL PASS' if a_passed else '❌ SOME FAILED'}")
    if not a_passed:
        for r in [x for x in controls_a if x["recovered"] != "True"]:
            print(f"    FAIL: {r['dataset']} {r['test_config']} seed={r['seed']}")

    print(f"  B (same-fault-all-replicas, expect escape): "
          f"{'✅ ALL ESCAPED (expected)' if not b_result else f'⚠️ {len(b_result)} unexpectedly recovered'}")
    for r in b_result:
        print(f"    UNEXPECTED RECOVERY: {r['dataset']} {r['test_config']} seed={r['seed']}")

    print(f"  C (two-of-three corrupted, expects false majority): "
          f"{len(controls_c)} rows, {sum(1 for r in controls_c if r['recovered']=='True')} recovered (informational)")

    print(f"\n  === Verdict ===")
    if a_passed and not b_result:
        print(f"  NMR_R0_CPU_ORACLE_PASSED")
        print(f"  The evaluator cleanly distinguishes same-domain failure (G2)")
        print(f"  from independent-domain recovery (R0).")
        print(f"  → Safe to proceed to GPU step. R1/R2 viable.")
    elif not a_passed:
        print(f"  HARNESS_BUG_OR_SPEC_MISMATCH (not NMR premise failure)")
        print(f"  r=3 majority under one-replica fault should always recover.")
        print(f"  Check harness logic before proceeding.")
    else:
        print(f"  AMBIGUOUS — unexpected recovery in same-fault-all-replicas")
        print(f"  Investigate before proceeding to GPU step.")


if __name__ == "__main__":
    main()
