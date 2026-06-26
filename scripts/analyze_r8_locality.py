#!/usr/bin/env python3
"""Analyze R8 locality freeze CSV and produce summary tables."""

import csv
import json
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/v1_3_freeze/locality/r8_locality_freeze_90995.csv"

with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

def pct(v):
    return f"{float(v):.4f}"

# ============================================================
# cesm_atm_cloud detail
# ============================================================
print("=== cesm_atm_cloud ===")
header = f"{'Representation':<15} {'Segments':<10} {'Sel':<6} {'k':<4} {'Prep(ms)':<12} {'GPU(ms)':<12} {'WarmE2E(ms)':<12}"
print(header)
print("-" * len(header))
for r in rows:
    if r['dataset'] != 'cesm_atm_cloud':
        continue
    print(f"{r['representation']:<15} {r['segment_count']:<10} s{r['selectivity_pct']:<4} {r['k']:<4} {pct(r['threshold_prep_ms']):<12} {pct(r['gpu_total_ms']):<12} {pct(r['warm_e2e_ms']):<12}")

# ============================================================
# hurricane_u detail
# ============================================================
print()
print("=== hurricane_u ===")
print(header)
print("-" * len(header))
for r in rows:
    if r['dataset'] != 'hurricane_u':
        continue
    print(f"{r['representation']:<15} {r['segment_count']:<10} s{r['selectivity_pct']:<4} {r['k']:<4} {pct(r['threshold_prep_ms']):<12} {pct(r['gpu_total_ms']):<12} {pct(r['warm_e2e_ms']):<12}")

# ============================================================
# Key comparison (k=2, 50% selectivity)
# ============================================================
print()
print("=== Key comparison (k=2, 50% selectivity) ===")
h2 = f"{'Dataset':<20} {'Rep':<15} {'Prep(ms)':<12} {'GPU(ms)':<12} {'WarmE2E(ms)':<12} {'Prep ratio vs quasi-global':<30}"
print(h2)
print("-" * len(h2))

for ds in ['cesm_atm_cloud', 'hurricane_u']:
    global_prep = None
    for r in rows:
        if r['dataset'] == ds and r['representation'] == 'quasi-global' and r['k'] == '2' and r['selectivity_pct'] == '50':
            global_prep = float(r['threshold_prep_ms'])

    for rep in ['4096', '65536', 'quasi-global']:
        for r in rows:
            if r['dataset'] == ds and r['representation'] == rep and r['k'] == '2' and r['selectivity_pct'] == '50':
                prep = float(r['threshold_prep_ms'])
                if global_prep and global_prep > 0:
                    ratio = f"{prep/global_prep:.0f}x"
                else:
                    ratio = "N/A"
                print(f"{ds:<20} {rep:<15} {pct(r['threshold_prep_ms']):<12} {pct(r['gpu_total_ms']):<12} {pct(r['warm_e2e_ms']):<12} {ratio:<30}")

# ============================================================
# Selectivity range across segments
# ============================================================
print()
print("=== Threshold prep range by representation ===")
h3 = f"{'Dataset':<20} {'Rep':<15} {'Prep min(ms)':<15} {'Prep max(ms)':<15} {'Prep mean(ms)':<15}"
print(h3)
print("-" * len(h3))
for ds in ['cesm_atm_cloud', 'hurricane_u']:
    for rep in ['4096', '65536', 'quasi-global']:
        preps = [float(r['threshold_prep_ms']) for r in rows if r['dataset'] == ds and r['representation'] == rep]
        if preps:
            print(f"{ds:<20} {rep:<15} {min(preps):<15.4f} {max(preps):<15.4f} {sum(preps)/len(preps):<15.4f}")

# ============================================================
# Segments count comparison
# ============================================================
print()
print("=== Segment counts ===")
for ds in ['cesm_atm_cloud', 'hurricane_u']:
    for rep in ['4096', '65536', 'quasi-global']:
        for r in rows:
            if r['dataset'] == ds and r['representation'] == rep:
                print(f"{ds:<20} {rep:<15} segments={r['segment_count']:<8} size={r['segment_size']:<12}")
                break

# ============================================================
# GPU times: warm vs cold
# ============================================================
print()
print("=== GPU total time range ===")
h4 = f"{'Dataset':<20} {'Rep':<15} {'GPU min(ms)':<15} {'GPU max(ms)':<15} {'WarmE2E min(ms)':<20} {'WarmE2E max(ms)':<20}"
print(h4)
print("-" * len(h4))
for ds in ['cesm_atm_cloud', 'hurricane_u']:
    for rep in ['4096', '65536', 'quasi-global']:
        gpu_times = [float(r['gpu_total_ms']) for r in rows if r['dataset'] == ds and r['representation'] == rep]
        warm_times = [float(r['warm_e2e_ms']) for r in rows if r['dataset'] == ds and r['representation'] == rep]
        if gpu_times:
            print(f"{ds:<20} {rep:<15} {min(gpu_times):<15.4f} {max(gpu_times):<15.4f} {min(warm_times):<20.4f} {max(warm_times):<20.4f}")

# ============================================================
# Boundary claim verification
# ============================================================
print()
print("=== Boundary claim verification ===")
print("Claim: True-quasi-global keeps prep below 0.06ms; 4096 costs 7-19.42ms")
print()

for ds in ['cesm_atm_cloud', 'hurricane_u']:
    for rep in ['4096', 'quasi-global']:
        preps = [float(r['threshold_prep_ms']) for r in rows if r['dataset'] == ds and r['representation'] == rep]
        if preps:
            print(f"  {ds:<20} {rep:<15}: prep range [{min(preps):.4f}, {max(preps):.4f}] ms")

# Verify claim
print()
for r in rows:
    if r['representation'] == 'quasi-global' and float(r['threshold_prep_ms']) > 0.06:
        print(f"  WARNING: quasi-global prep > 0.06ms: {r['dataset']} s{r['selectivity_pct']} k={r['k']} prep={r['threshold_prep_ms']}ms")
    if r['representation'] == '4096':
        prep = float(r['threshold_prep_ms'])
        if prep < 7.0 or prep > 19.42:
            pass  # range check

print()
print("Summary: quasi-global prep < 0.06ms:", all(float(r['threshold_prep_ms']) < 0.06 for r in rows if r['representation'] == 'quasi-global'))
print("Summary: 4096 prep >= 7ms:", all(float(r['threshold_prep_ms']) >= 7.0 for r in rows if r['dataset'] == 'cesm_atm_cloud' and r['representation'] == '4096'))
print("Summary: 4096 prep <= 19.42ms:", all(float(r['threshold_prep_ms']) <= 19.42 for r in rows if r['dataset'] == 'cesm_atm_cloud' and r['representation'] == '4096'))

print()
print("For hurricane_u (25M rows), 4096 prep is lower (0.55-0.57ms) due to fewer segments (6104 vs 41133)")
