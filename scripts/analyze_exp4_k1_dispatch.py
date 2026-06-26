#!/usr/bin/env python3
"""Exp4 k1 dispatch integration analysis.

Reads the b1 sweep precision-throughput table and the k1 formal probe
summary_pairs.csv, computes per-dataset mean speedup, and produces a
dispatch-optimized throughput table where k=1 rows reflect the
byte_mask_specialized_k1 speedup.

Usage:
    python scripts/analyze_exp4_k1_dispatch.py

Output:
    results/exp4/b1_20260510_174435_job44743_NVIDIAH200/
        count_precision_throughput_with_k1_dispatch.csv
        count_epsilon_to_kstar_with_k1_dispatch.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────
B1_DIR = Path("results/exp4/b1_20260510_174435_job44743_NVIDIAH200")
K1_DIR = Path("results/exp4/byte_mask_k1_formal_probe_20260516_231329_job52947_NVIDIAH200")

B1_PT_CSV = B1_DIR / "count_precision_throughput.csv"
B1_EK_CSV = B1_DIR / "count_epsilon_to_kstar.csv"
K1_PAIRS_CSV = K1_DIR / "summary_pairs.csv"


def main():
    # ── 1. Read data ──────────────────────────────────────────────────────
    print("Reading b1 sweep precision-throughput table...")
    pt = pd.read_csv(B1_PT_CSV)
    print(f"  {len(pt)} rows")

    print("Reading b1 sweep epsilon-to-kstar table...")
    ek = pd.read_csv(B1_EK_CSV)
    print(f"  {len(ek)} rows")

    print("Reading k1 formal probe summary_pairs...")
    pairs = pd.read_csv(K1_PAIRS_CSV)
    print(f"  {len(pairs)} rows")

    # ── 2. Compute per-dataset mean speedup ──────────────────────────────
    # The k1 probe uses integer selectivity labels (1, 50, 99).
    # We compute per-dataset mean speedup since the speedup is very stable
    # across selectivities and block sizes (stddev ~0.0023).
    speedup_by_ds = pairs.groupby("dataset")["speedup"].mean().to_dict()
    global_mean = pairs["speedup"].mean()

    print("\nPer-dataset mean k1 speedup:")
    for ds, sp in sorted(speedup_by_ds.items()):
        print(f"  {ds}: {sp:.4f}x")
    print(f"  Global mean: {global_mean:.4f}x")

    # ── 3. Dispatch-optimized precision-throughput table ──────────────────
    # For k=1 rows, multiply rows_per_sec by the per-dataset k1 speedup.
    # For k>1 rows, dispatch throughput = generic throughput (no change).
    pt_dispatch = pt.copy()
    pt_dispatch["k1_dispatch_rows_per_sec"] = pt_dispatch.apply(
        lambda r: r["rows_per_sec"] * speedup_by_ds.get(r["dataset"], global_mean)
        if r["max_filter_planes"] == 1
        else r["rows_per_sec"],
        axis=1,
    )

    # Also add a column showing the speedup factor applied
    pt_dispatch["k1_speedup_factor"] = pt_dispatch.apply(
        lambda r: speedup_by_ds.get(r["dataset"], global_mean)
        if r["max_filter_planes"] == 1
        else 1.0,
        axis=1,
    )

    out_pt = B1_DIR / "count_precision_throughput_with_k1_dispatch.csv"
    pt_dispatch.to_csv(out_pt, index=False)
    print(f"\nWrote: {out_pt} ({len(pt_dispatch)} rows)")

    # Summary stats
    k1_rows = pt_dispatch[pt_dispatch["max_filter_planes"] == 1]
    print(f"  k=1 rows: {len(k1_rows)}")
    print(f"  k>1 rows: {len(pt_dispatch) - len(k1_rows)}")
    if len(k1_rows) > 0:
        print(f"  k=1 generic throughput range: "
              f"{k1_rows['rows_per_sec'].min():.2e} – {k1_rows['rows_per_sec'].max():.2e}")
        print(f"  k=1 dispatch throughput range: "
              f"{k1_rows['k1_dispatch_rows_per_sec'].min():.2e} – "
              f"{k1_rows['k1_dispatch_rows_per_sec'].max():.2e}")

    # ── 4. Dispatch-optimized epsilon-to-kstar table ─────────────────────
    # For k*=1 rows, look up the dispatch throughput from the precision table.
    # For k*>1 rows, use generic throughput.
    # We need to join ek (which has kstar) with pt (which has throughput).
    # Join key: (dataset, artifact, selectivity, k=max_filter_planes).

    # Build a lookup: (dataset, artifact_label, selectivity, k) -> rows_per_sec
    # and (dataset, artifact_label, selectivity, k) -> k1_dispatch_rows_per_sec
    pt_lookup_generic = {}
    pt_lookup_dispatch = {}
    for _, row in pt_dispatch.iterrows():
        key = (row["dataset"], row["artifact_label"], row["selectivity"],
                int(row["max_filter_planes"]))
        pt_lookup_generic[key] = row["rows_per_sec"]
        pt_lookup_dispatch[key] = row["k1_dispatch_rows_per_sec"]

    # Add dispatch throughput to epsilon-to-kstar table
    ek_dispatch = ek.copy()
    ek_dispatch["generic_throughput_at_kstar"] = np.nan
    ek_dispatch["k1_dispatch_throughput_at_kstar"] = np.nan

    for idx, row in ek_dispatch.iterrows():
        kstar = int(row["kstar"])
        if kstar < 1:
            # kstar == -1 means unmet; no throughput to report
            continue
        key = (row["dataset"], row["artifact"], row["selectivity"], kstar)
        if key in pt_lookup_generic:
            ek_dispatch.at[idx, "generic_throughput_at_kstar"] = pt_lookup_generic[key]
            ek_dispatch.at[idx, "k1_dispatch_throughput_at_kstar"] = pt_lookup_dispatch[key]

    out_ek = B1_DIR / "count_epsilon_to_kstar_with_k1_dispatch.csv"
    ek_dispatch.to_csv(out_ek, index=False)
    print(f"\nWrote: {out_ek} ({len(ek_dispatch)} rows)")

    # Summary: how many k*=1 rows got dispatch throughput
    kstar1 = ek_dispatch[ek_dispatch["kstar"] == 1]
    kstar1_with_dispatch = kstar1[kstar1["k1_dispatch_throughput_at_kstar"].notna()]
    print(f"  k*=1 rows: {len(kstar1)}")
    print(f"  k*=1 rows with dispatch throughput: {len(kstar1_with_dispatch)}")
    if len(kstar1_with_dispatch) > 0:
        mean_speedup = (
            kstar1_with_dispatch["k1_dispatch_throughput_at_kstar"]
            / kstar1_with_dispatch["generic_throughput_at_kstar"]
        ).mean()
        print(f"  Mean dispatch speedup at k*=1: {mean_speedup:.4f}x")

    # ── 5. k*=1 prevalence summary ────────────────────────────────────────
    # How many (dataset, artifact, selectivity, epsilon) combinations map to k*=1?
    total = len(ek_dispatch)
    k1_count = len(ek_dispatch[ek_dispatch["kstar"] == 1])
    k1_pct = 100.0 * k1_count / total if total > 0 else 0
    print(f"\nk*=1 prevalence: {k1_count}/{total} ({k1_pct:.1f}%) of ε→k* mappings resolve to k*=1")

    # Per-dataset breakdown
    for ds in sorted(ek_dispatch["dataset"].unique()):
        ds_rows = ek_dispatch[ek_dispatch["dataset"] == ds]
        ds_k1 = len(ds_rows[ds_rows["kstar"] == 1])
        ds_pct = 100.0 * ds_k1 / len(ds_rows) if len(ds_rows) > 0 else 0
        print(f"  {ds}: {ds_k1}/{len(ds_rows)} ({ds_pct:.1f}%)")


if __name__ == "__main__":
    main()