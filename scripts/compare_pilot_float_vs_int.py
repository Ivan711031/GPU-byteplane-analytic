#!/usr/bin/env python3
"""Compare old (float-bug) pilot vs new (int-fix) pilot frontier CSVs.

Produces a per-cell comparison table for every (dataset, family,
severity, rate, policy_B, policy_type).

Usage:
  python3 scripts/compare_pilot_float_vs_int.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# ── Old pilot: float bug (jobs 108937-108940 combined) ──────────
OLD_JOB_DIRS = {
    ("hurricane_u", 1e-9):  "claim1_nosum32_pilot/job_108937/hurricane_u",
    ("hurricane_u", 1e-7):  "claim1_nosum32_pilot/job_108937/hurricane_u",
    ("hurricane_u", 1e-5):  "claim1_nosum32_pilot/job_108938/hurricane_u",
    ("hurricane_u", 1e-3):  "claim1_nosum32_pilot/job_108938/hurricane_u",
    ("cesm_atm_cloud", 1e-9):  "claim1_nosum32_pilot/job_108939/cesm_atm_cloud",
    ("cesm_atm_cloud", 1e-7):  "claim1_nosum32_pilot/job_108939/cesm_atm_cloud",
    ("cesm_atm_cloud", 1e-5):  "claim1_nosum32_pilot/job_108940/cesm_atm_cloud",
    ("cesm_atm_cloud", 1e-3):  "claim1_nosum32_pilot/job_108940/cesm_atm_cloud",
}

# ── New pilot: int fix (job 109311, one job for all rates) ──────
NEW_JOB_DIR = "claim1_nosum32_pilot_intfix/job_109311"

RESULTS_DIR = BASE / "results" / "reliability_layer1"


def load_frontier(path: Path) -> dict[tuple, dict]:
    """Load frontier CSV into dict keyed by (dataset, family, severity, rate, B, type)."""
    rows = {}
    if not path.exists():
        return rows
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (
                r["dataset"],
                r["fault_family"],
                r["severity_label"],
                float(r["rate"]),
                int(r["policy_B"]),
                r["policy_type"],
            )
            rows[key] = {
                "correct_answer_rate": float(r.get("correct_answer_rate", 0)),
                "safe_answer_rate": float(r.get("safe_answer_rate", 0)),
                "silent_wrong_rate": float(r.get("silent_wrong_rate", 0)),
            }
    return rows


def load_outcome_summary(path: Path) -> dict[tuple, dict]:
    """Load outcome_summary CSV keyed by (dataset, family, rate)."""
    rows = {}
    if not path.exists():
        return rows
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["dataset"], r["fault_family"], float(r["rate"]))
            rows[key] = {
                "total_cells": int(r["total_cells"]),
                "exact_correct_rate": float(r.get("exact_correct_rate", 0)),
                "majority_recovered_rate": float(r.get("majority_recovered_rate", 0)),
                "detected_unavailable_rate": float(r.get("detected_unavailable_rate", 0)),
                "silent_wrong_rate": float(r.get("silent_wrong_rate", 0)),
            }
    return rows


# ── Load old data from multiple jobs ─────────────────────────────
old_outcome: dict[tuple, dict] = {}
old_frontier: dict[tuple, dict] = {}
for (ds, rate), rel_dir in OLD_JOB_DIRS.items():
    job_path = RESULTS_DIR / rel_dir
    old_outcome.update(load_outcome_summary(job_path / "claim1_realistic_outcome_summary.csv"))
    old_frontier.update(load_frontier(job_path / "claim1_realistic_policy_frontier_nosum32.csv"))

# ── Load new data ────────────────────────────────────────────────
new_path = RESULTS_DIR / NEW_JOB_DIR
new_outcome: dict[tuple, dict] = {}
new_frontier: dict[tuple, dict] = {}
for dataset in ("hurricane_u", "cesm_atm_cloud"):
    new_outcome.update(load_outcome_summary(
        new_path / dataset / "claim1_realistic_outcome_summary.csv"))
    new_frontier.update(load_frontier(
        new_path / dataset / "claim1_realistic_policy_frontier_nosum32.csv"))


# ── 1. Outcome summary comparison ────────────────────────────────
print("=" * 120)
print("COMPARISON 1: Outcome Summary (across ALL cells per family/rate)")
print("=" * 120)
print(f"{'dataset':<18} {'family':>2} {'rate':>8} | "
      f"{'n_old':>5} {'exact_old':>9} {'mr_old':>9} {'du_old':>9} {'sw_old':>9} | "
      f"{'n_new':>5} {'exact_new':>9} {'mr_new':>9} {'du_new':>9} {'sw_new':>9} | "
      f"{'delta_sw':>7}")
print("-" * 120)

all_keys = sorted(set(list(old_outcome.keys()) + list(new_outcome.keys())))
for key in all_keys:
    ds, fam, rate = key
    o = old_outcome.get(key, {})
    n = new_outcome.get(key, {})
    o_ec = o.get("exact_correct_rate", -1)
    o_mr = o.get("majority_recovered_rate", -1)
    o_du = o.get("detected_unavailable_rate", -1)
    o_sw = o.get("silent_wrong_rate", -1)
    n_ec = n.get("exact_correct_rate", -1)
    n_mr = n.get("majority_recovered_rate", -1)
    n_du = n.get("detected_unavailable_rate", -1)
    n_sw = n.get("silent_wrong_rate", -1)
    o_n = o.get("total_cells", -1)
    n_n = n.get("total_cells", -1)

    if o_sw != -1 and n_sw != -1 and abs(o_sw - n_sw) > 0.001:
        delta_sw = n_sw - o_sw
        flag = " <<<" if abs(delta_sw) > 0.02 else ""
    else:
        delta_sw = n_sw - o_sw if (o_sw != -1 and n_sw != -1) else 0
        flag = ""

    print(f"{ds:<18} {fam:>2} {rate:>8.0e} | "
          f"{o_n:>5} {o_ec:>9.4f} {o_mr:>9.4f} {o_du:>9.4f} {o_sw:>9.4f} | "
          f"{n_n:>5} {n_ec:>9.4f} {n_mr:>9.4f} {n_du:>9.4f} {n_sw:>9.4f} | "
          f"{delta_sw:>+7.4f}{flag}")


# ── 2. Frontier comparison per policy ────────────────────────────
print("\n" + "=" * 120)
print("COMPARISON 2: Per-Policy Frontier (correct, safe, sw)  —  only showing differences")
print("=" * 120)

show_diff_only = True
count_diffs = 0

for key in sorted(set(list(old_frontier.keys()) + list(new_frontier.keys()))):
    ds, fam, sev, rate, B, ptype = key
    o = old_frontier.get(key, {})
    n = new_frontier.get(key, {})

    o_corr = o.get("correct_answer_rate", -1)
    o_safe = o.get("safe_answer_rate", -1)
    o_sw = o.get("silent_wrong_rate", -1)
    n_corr = n.get("correct_answer_rate", -1)
    n_safe = n.get("safe_answer_rate", -1)
    n_sw = n.get("silent_wrong_rate", -1)

    if o_corr == -1 and n_corr == -1:
        continue

    has_diff = (abs(o_corr - n_corr) > 0.001 or
                abs(o_safe - n_safe) > 0.001 or
                abs(o_sw - n_sw) > 0.001)

    if show_diff_only and not has_diff:
        continue

    count_diffs += 1
    o_line = f"old={o_corr:.4f}/{o_safe:.4f}/{o_sw:.4f}" if o_corr != -1 else "old=N/A"
    n_line = f"new={n_corr:.4f}/{n_safe:.4f}/{n_sw:.4f}" if n_corr != -1 else "new=N/A"

    print(f"{ds:<12} {fam:>2} {sev:<30} {rate:>8.0e} B={B:>2} {ptype:<8} "
          f"{o_line}  |  {n_line}")


# ── 3. Targeted summary: key policies per family ─────────────────
print("\n" + "=" * 120)
print("COMPARISON 3: Key Policies — F7, F8, F1 at B=1/2/8/16")
print("=" * 120)

TARGET_FAMILIES = ["F1", "F4", "F5", "F7", "F8"]
TARGET_RATES = [1e-5]
TARGET_BS = [(0, "graded"), (1, "graded"), (2, "graded"),
             (8, "graded"), (8, "uniform"), (16, "graded"), (16, "uniform")]

print(f"{'ds':<12} {'fam':>2} {'Btype':<12} {'rate':>8} | "
      f"{'old_corr':>8} {'old_safe':>8} {'old_sw':>8} | "
      f"{'new_corr':>8} {'new_safe':>8} {'new_sw':>8} | "
      f"{'Δcorr':>6} {'Δsafe':>6} {'Δsw':>6}")
print("-" * 120)

for ds in ("hurricane_u", "cesm_atm_cloud"):
    for fam in TARGET_FAMILIES:
        for rate in TARGET_RATES:
            for B, ptype in TARGET_BS:
                sev_keys = set()
                for k in list(old_frontier.keys()) + list(new_frontier.keys()):
                    if k[0] == ds and k[1] == fam:
                        sev_keys.add(k[2])
                for sev in sorted(sev_keys):
                    key = (ds, fam, sev, rate, B, ptype)
                    o = old_frontier.get(key, {})
                    n = new_frontier.get(key, {})
                    if not o and not n:
                        continue
                    o_c = o.get("correct_answer_rate", -1)
                    o_s = o.get("safe_answer_rate", -1)
                    o_w = o.get("silent_wrong_rate", -1)
                    n_c = n.get("correct_answer_rate", -1)
                    n_s = n.get("safe_answer_rate", -1)
                    n_w = n.get("silent_wrong_rate", -1)

                    o_line = (f"{o_c:>8.4f} {o_s:>8.4f} {o_w:>8.4f}"
                              if o_c != -1 else f"{'N/A':>8} {'N/A':>8} {'N/A':>8}")
                    n_line = (f"{n_c:>8.4f} {n_s:>8.4f} {n_w:>8.4f}"
                              if n_c != -1 else f"{'N/A':>8} {'N/A':>8} {'N/A':>8}")

                    has_all = (o_c != -1 and n_c != -1)
                    d_c = f"{n_c - o_c:>+6.4f}" if has_all else "  N/A "
                    d_s = f"{n_s - o_s:>+6.4f}" if has_all else "  N/A "
                    d_w = f"{n_w - o_w:>+6.4f}" if has_all else "  N/A "

                    flag = ""
                    if has_all and abs(n_w - o_w) > 0.1:
                        flag = " <<<"

                    print(f"{ds:<12} {fam:>2} {f'{B}_{ptype}':<12} {rate:>8.0e} | "
                          f"{o_line} | {n_line} | {d_c} {d_s} {d_w}{flag}")

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)

total_old = len(old_outcome)
total_new = len(new_outcome)
common = set(old_outcome.keys()) & set(new_outcome.keys())

cells_flipped = 0
for k in common:
    o = old_outcome[k].get("silent_wrong_rate", 0)
    n = new_outcome[k].get("silent_wrong_rate", 0)
    if abs(n - o) > 0.01:
        cells_flipped += 1

print(f"Old total outcome rows: {total_old}")
print(f"New total outcome rows: {total_new}")
print(f"Common (dataset, family, rate) pairs: {len(common)}")
print(f"Pairs with sw diff > 0.01: {cells_flipped}")
print(f"Frontier cells with any diff: {count_diffs}")
print()

# List specific flipped cells
print("Largest silent_wrong rate changes:")
changes = []
for k in common:
    o = old_outcome[k].get("silent_wrong_rate", 0)
    n = new_outcome[k].get("silent_wrong_rate", 0)
    delta = n - o
    if abs(delta) > 0.01:
        changes.append((abs(delta), k, o, n, delta))
changes.sort(reverse=True)
for _, k, o, n, d in changes[:15]:
    print(f"  {k[0]:<12} {k[1]:>2} rate={k[2]:.0e}: "
          f"old_sw={o:.4f} → new_sw={n:.4f} (Δ={d:+.4f})")
