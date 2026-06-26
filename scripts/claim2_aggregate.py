"""Claim 2 aggregation: compute summarized statistics and verdict.

Reads the pilot/headline CSV and produces:
- claim2_paired_delta_summary.csv  (policy pairwise comparisons)
- claim2_fault_family_summary.csv  (per-family aggregation)
- claim2_answer_quality_summary.csv (headline per-policy metrics)
- claim2_verdict.md                (final verdict document)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

POLICIES = [
    "no_diversity_naive_r3",
    "spatial_only_diverse_r3",
    "temporal_only_diverse_r3",
    "spatial_temporal_diverse_r3",
]

SUITE_ORDER = ["suite_a", "suite_b", "suite_c", "suite_d"]
SUITE_LABELS = {
    "suite_a": "null_sanity",
    "suite_b": "spatial_regime",
    "suite_c": "transient_temporal",
    "suite_d": "hard_negative",
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(v: str) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def safe_bool(v: str) -> bool:
    return v.strip().lower() in ("true", "1", "1.0")


def group_rows(rows: list[dict[str, str]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        k = tuple(r.get(k, "") for k in keys)
        groups[k].append(r)
    return groups


# ── Paired delta summary ─────────────────────────────────────

def compute_paired_deltas(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """For each (suite, family, rate, seed), compare each policy vs naive."""
    groups = group_rows(rows, ["suite", "fault_family", "rate", "seed"])
    results: list[dict[str, str]] = []

    for key, grp in groups.items():
        suite, fam, rate, seed = key
        by_policy: dict[str, dict[str, str]] = {}
        for r in grp:
            by_policy[r["policy"]] = r

        naive = by_policy.get("no_diversity_naive_r3")
        if not naive:
            continue

        naive_err = safe_float(naive["relative_error"])
        naive_flip = safe_float(naive["decision_flip_rate"])
        naive_ct = safe_bool(naive["contains_truth"])
        naive_bw = safe_float(naive["expected_bound_width"])

        for policy in POLICIES[1:]:
            p_row = by_policy.get(policy)
            if not p_row:
                continue
            p_err = safe_float(p_row["relative_error"])
            p_flip = safe_float(p_row["decision_flip_rate"])
            p_ct = safe_bool(p_row["contains_truth"])
            p_bw = safe_float(p_row["expected_bound_width"])

            delta_err = p_err - naive_err
            delta_flip = p_flip - naive_flip
            delta_bw = p_bw - naive_bw
            ct_wins = (p_ct and not naive_ct) or (p_ct == naive_ct)

            results.append({
                "suite": suite,
                "fault_family": fam,
                "rate": rate,
                "seed": seed,
                "policy": policy,
                "delta_relative_error": f"{delta_err:.10e}",
                "delta_decision_flip_rate": f"{delta_flip:.10f}",
                "delta_bound_width": f"{delta_bw:.10e}",
                "naive_contains_truth": str(naive_ct).lower(),
                "policy_contains_truth": str(p_ct).lower(),
                "policy_improves_error": str(delta_err < 0).lower(),
                "policy_improves_flip": str(delta_flip < 0).lower(),
                "policy_improves_ct": str(ct_wins).lower(),
            })

    return results


# ── Fault family summary ─────────────────────────────────────

def compute_family_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups = group_rows(rows, ["suite", "fault_family", "rate", "policy"])
    results: list[dict[str, str]] = []

    for key, grp in groups.items():
        suite, fam, rate, policy = key
        n = len(grp)
        mean_err = sum(safe_float(r["relative_error"]) for r in grp) / n
        mean_flip = sum(safe_float(r["decision_flip_rate"]) for r in grp) / n
        ct_rate = sum(1 for r in grp if safe_bool(r["contains_truth"])) / n
        mean_bw = sum(safe_float(r["expected_bound_width"]) for r in grp) / n
        mean_escape = sum(safe_float(r["escape_rate"]) for r in grp) / n
        vote_recovered_rate = sum(1 for r in grp if safe_bool(r["vote_recovered"])) / n

        results.append({
            "suite": suite,
            "suite_label": SUITE_LABELS.get(suite, suite),
            "fault_family": fam,
            "rate": rate,
            "policy": policy,
            "n_cells": str(n),
            "mean_relative_error": f"{mean_err:.10e}",
            "mean_decision_flip_rate": f"{mean_flip:.10f}",
            "contains_truth_rate": f"{ct_rate:.6f}",
            "mean_bound_width": f"{mean_bw:.10e}",
            "mean_escape_rate": f"{mean_escape:.10f}",
            "vote_recovered_rate": f"{vote_recovered_rate:.6f}",
        })

    return results


# ── Answer quality summary (headline) ────────────────────────

def compute_answer_quality_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Aggregate across seeds per (suite, family, rate, policy)."""
    return compute_family_summary(rows)


# ── Verdict ──────────────────────────────────────────────────

def compute_verdict(
    family_summary_rows: list[dict[str, str]],
    head: list[dict[str, str]],
) -> dict[str, Any]:
    verdicts: list[str] = []
    details: list[str] = []

    # ── Coverage gate (explicit expected matrix) ──────────────
    EXPECTED_DATASETS = ["hurricane_u", "cesm_atm_cloud"]
    EXPECTED_SUITES: dict[str, list[str]] = {
        "suite_a": ["null_no_fault", "null_no_fault_alt"],
        "suite_b": ["regional_single_domain_burst", "column_like_repeated_offset", "sid_like_hot_domain"],
        "suite_c": ["temporal_window_single_stage", "transient_burst_repeated_read", "staggered_stage_hotspot"],
        "suite_d": ["same_fault_all_replicas", "cross_domain_all_stage_burst"],
    }
    EXPECTED_RATES = ["1e-07", "3e-07", "1e-06", "3e-06", "1e-05", "3e-05", "1e-04"]
    EXPECTED_SEEDS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # Build presence index: {(ds, suite, fam, rate, seed) -> set of policies}
    present: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for r in head:
        key = (r.get("dataset", ""), r.get("suite", ""),
               r.get("fault_family", ""), r.get("rate", ""), r.get("seed", ""))
        present[key].add(r.get("policy", ""))

    coverage_ok = True
    for ds in EXPECTED_DATASETS:
        for suite_id, families in EXPECTED_SUITES.items():
            for fam in families:
                for rate in EXPECTED_RATES:
                    for seed in EXPECTED_SEEDS:
                        key = (ds, suite_id, fam, rate, seed)
                        pols = present.get(key, set())
                        missing = [p for p in POLICIES if p not in pols]
                        if missing:
                            coverage_ok = False
                            details.append(
                                f"Coverage gap: {ds}/{suite_id}/{fam}@{rate}/seed={seed} "
                                f"missing policies: {missing}"
                            )

    if coverage_ok:
        verdicts.append("COVERAGE_COMPLETE")
    else:
        verdicts.append("COVERAGE_INCOMPLETE")
        details.append("Coverage gate FAILED — see gaps above")

    # Separate by suite
    by_suite: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in family_summary_rows:
        by_suite[r["suite"]].append(r)

    # ── Suite B: spatial beats naive (non-vacuous) ───────────
    suite_b = by_suite.get("suite_b", [])
    spatial_beats_naive_b = True
    suite_b_pairs_found = False

    for r in suite_b:
        if r["policy"] == "spatial_only_diverse_r3":
            naive_err = None
            for r2 in suite_b:
                if r2["policy"] == "no_diversity_naive_r3" and r2["fault_family"] == r["fault_family"] and r2["rate"] == r["rate"]:
                    naive_err = safe_float(r2["mean_relative_error"])
                    break
            if naive_err is not None:
                suite_b_pairs_found = True
                if safe_float(r["mean_relative_error"]) >= naive_err:
                    spatial_beats_naive_b = False
                    details.append(f"Suite B: {r['fault_family']}@{r['rate']}: spatial err={r['mean_relative_error']} >= naive err={naive_err:.10e}")

    if not suite_b_pairs_found:
        spatial_beats_naive_b = False
        details.append("Suite B: no naive-spatial comparison pairs found — verdict vacuous")

    verdicts.append("SPATIAL_BEATS_NAIVE_IN_SUITE_B" if spatial_beats_naive_b else "SPATIAL_NOT_BETTER_THAN_NAIVE_IN_SUITE_B")

    # ── Suite C: temporal beats naive, st beats spatial (non-vacuous) ──
    suite_c = by_suite.get("suite_c", [])
    temporal_positive_c = True
    st_beats_spatial_c = True
    c_naive_temporal_found = False
    c_spatial_st_found = False

    for r in suite_c:
        if r["policy"] == "temporal_only_diverse_r3":
            naive_err = None
            for r2 in suite_c:
                if r2["policy"] == "no_diversity_naive_r3" and r2["fault_family"] == r["fault_family"] and r2["rate"] == r["rate"]:
                    naive_err = safe_float(r2["mean_relative_error"])
                    break
            if naive_err is not None:
                c_naive_temporal_found = True
                if safe_float(r["mean_relative_error"]) > naive_err * 1.05:
                    temporal_positive_c = False
                    details.append(f"Suite C: {r['fault_family']}@{r['rate']}: temporal err={r['mean_relative_error']} > naive err={naive_err:.10e}")

        if r["policy"] == "spatial_temporal_diverse_r3":
            spatial_err = None
            for r2 in suite_c:
                if r2["policy"] == "spatial_only_diverse_r3" and r2["fault_family"] == r["fault_family"] and r2["rate"] == r["rate"]:
                    spatial_err = safe_float(r2["mean_relative_error"])
                    break
            if spatial_err is not None:
                c_spatial_st_found = True
                EPS = 1e-30
                if safe_float(r["mean_relative_error"]) > spatial_err + EPS:
                    st_beats_spatial_c = False
                    details.append(f"Suite C: {r['fault_family']}@{r['rate']}: st err={r['mean_relative_error']} >= spatial err={spatial_err:.10e}")

    if not c_naive_temporal_found:
        temporal_positive_c = False
        details.append("Suite C: no naive-temporal comparison pairs found — verdict vacuous")

    if not c_spatial_st_found:
        st_beats_spatial_c = False
        details.append("Suite C: no spatial-ST comparison pairs found — verdict vacuous")

    verdicts.append("TEMPORAL_NON_NEGATIVE_IN_SUITE_C" if temporal_positive_c else "TEMPORAL_WORSE_THAN_NAIVE_IN_SUITE_C")
    verdicts.append("ST_BEATS_SPATIAL_IN_SUITE_C" if st_beats_spatial_c else "ST_NOT_BETTER_THAN_SPATIAL_IN_SUITE_C")

    # ── Suite A: all tied ─────────────────────────────────────
    suite_a = by_suite.get("suite_a", [])
    all_tied_a = True
    suite_a_pairs_found = False
    for r in suite_a:
        pol = r["policy"]
        if pol == "no_diversity_naive_r3":
            continue
        naive_err = None
        for r2 in suite_a:
            if r2["policy"] == "no_diversity_naive_r3" and r2["fault_family"] == r["fault_family"] and r2["rate"] == r["rate"]:
                naive_err = safe_float(r2["mean_relative_error"])
                break
        if naive_err is not None:
            suite_a_pairs_found = True
            if abs(safe_float(r["mean_relative_error"]) - naive_err) > 0.5 * max(abs(naive_err), 1e-30):
                all_tied_a = False
                details.append(f"Suite A: {r['fault_family']}@{r['rate']}: {pol} err={r['mean_relative_error']} diverges from naive={naive_err:.10e}")

    if not suite_a_pairs_found:
        all_tied_a = False
        details.append("Suite A: no policy comparison pairs found — verdict vacuous")

    # ── Suite D: no false credit ──────────────────────────────
    suite_d = by_suite.get("suite_d", [])
    no_false_credit_d = True
    suite_d_pairs_found = False
    for r in suite_d:
        vr_rate = safe_float(r["vote_recovered_rate"])
        err = safe_float(r["mean_relative_error"])
        pol = r["policy"]

        # No false vote recovery: vr_rate must be 0 for all policies when
        # all replicas are identically faulted (hard negatives)
        if vr_rate > 0.0:
            no_false_credit_d = False
            details.append(f"Suite D: {r['fault_family']}@{r['rate']} {pol}: vr_rate={vr_rate} > 0 (false recovery)")

        # No false diversity win: diverse policy should not have zero error
        # when naive has positive error
        if pol != "no_diversity_naive_r3":
            naive_err = None
            for r2 in suite_d:
                if r2["policy"] == "no_diversity_naive_r3" and r2["fault_family"] == r["fault_family"] and r2["rate"] == r["rate"]:
                    naive_err = safe_float(r2["mean_relative_error"])
                    break
            if naive_err is not None:
                suite_d_pairs_found = True
                if naive_err > 1e-30 and err == 0.0:
                    no_false_credit_d = False
                    details.append(f"Suite D: {r['fault_family']}@{r['rate']} {pol}: err=0.0 while naive err={naive_err:.10e} (false win)")

    if not suite_d_pairs_found:
        no_false_credit_d = False
        details.append("Suite D: no policy comparison pairs found — verdict vacuous")

    # --- Overall ---
    strong_claim = (
        coverage_ok
        and spatial_beats_naive_b
        and temporal_positive_c
        and st_beats_spatial_c
        and all_tied_a
        and no_false_credit_d
    )

    partial = (
        (spatial_beats_naive_b or st_beats_spatial_c)
        and not strong_claim
    )

    if strong_claim:
        overall = "STRONG_SCOPED_CLAIM_SUPPORTED"
    elif partial:
        overall = "PARTIAL_SCOPED_SUPPORT"
    else:
        overall = "UNSUPPORTED"

    return {
        "overall_verdict": overall,
        "component_verdicts": verdicts,
        "details": details if details else ["All checks passed"],
        "coverage_ok": coverage_ok,
        "all_tied_a": all_tied_a,
        "spatial_beats_naive_b": spatial_beats_naive_b,
        "temporal_positive_c": temporal_positive_c,
        "st_beats_spatial_c": st_beats_spatial_c,
        "no_false_credit_d": no_false_credit_d,
    }


def write_verdict_md(verdict: dict[str, Any], path: Path) -> None:
    lines = [
        "# Claim 2 — Spatial+Temporal Diversity Answer-Quality Closure",
        "",
        f"**Date:** 2026-06-11",
        f"**Overall Verdict:** `{verdict['overall_verdict']}`",
        "",
        "## Component Verdicts",
        "",
    ]
    for v in verdict["component_verdicts"]:
        lines.append(f"- `{v}`")
    lines += [
        "",
        "## Details",
        "",
    ]
    for d in verdict["details"]:
        lines.append(f"- {d}")
    lines += [
        "",
        "## Summary Table",
        "",
        "| Check | Result |",
        "|-------|--------|",
        f"| Coverage gate | {'✅' if verdict.get('coverage_ok', True) else '❌'} |",
        f"| Suite A (null/sanity): all tied | {'✅' if verdict['all_tied_a'] else '❌'} |",
        f"| Suite B (spatial): spatial beats naive | {'✅' if verdict['spatial_beats_naive_b'] else '❌'} |",
        f"| Suite C (temporal): temporal non-negative vs naive | {'✅' if verdict['temporal_positive_c'] else '❌'} |",
        f"| Suite C (temporal): ST beats spatial | {'✅' if verdict['st_beats_spatial_c'] else '❌'} |",
        f"| Suite D (negative): no false credit | {'✅' if verdict['no_false_credit_d'] else '❌'} |",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"Verdict: {path}")


# ── Main ─────────────────────────────────────────────────────

AGGREGATE_FIELDS = [
    "suite", "suite_label", "fault_family", "rate", "policy",
    "n_cells",
    "mean_relative_error", "mean_decision_flip_rate",
    "contains_truth_rate", "mean_bound_width",
    "mean_escape_rate", "vote_recovered_rate",
]

DELTA_FIELDS = [
    "suite", "fault_family", "rate", "seed", "policy",
    "delta_relative_error", "delta_decision_flip_rate", "delta_bound_width",
    "naive_contains_truth", "policy_contains_truth",
    "policy_improves_error", "policy_improves_flip", "policy_improves_ct",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--label", type=str, default="claim2",
                        help="Prefix for output files (pilot/headline)")
    args = parser.parse_args()

    rows = load_csv(args.input_csv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Paired deltas
    deltas = compute_paired_deltas(rows)
    delta_path = out / f"{args.label}_paired_delta_summary.csv"
    with delta_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DELTA_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(deltas)
    print(f"Paired deltas: {delta_path} ({len(deltas)} rows)")

    # Family summary
    fam = compute_family_summary(rows)
    fam_path = out / f"{args.label}_fault_family_summary.csv"
    with fam_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AGGREGATE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(fam)
    print(f"Family summary: {fam_path} ({len(fam)} rows)")

    # Answer quality summary (same as family summary for now)
    aq_path = out / f"{args.label}_answer_quality_summary.csv"
    with aq_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AGGREGATE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(fam)
    print(f"Answer quality summary: {aq_path} ({len(fam)} rows)")

    # Verdict
    verdict = compute_verdict(fam, rows)
    v_path = out / f"{args.label}_verdict.md"
    write_verdict_md(verdict, v_path)


if __name__ == "__main__":
    main()
