#!/usr/bin/env python3
"""
Exp4-B1 capped-k COUNT analysis script (#7 / PV1-4).

Reads sweep_summary.csv and produces:
  - count_precision_throughput.csv
  - count_epsilon_to_kstar.csv
  - 5 reviewer-facing plots

Sanity checks:
  - k coverage 1..max_plane_count per (dataset, threshold)
  - U(k) monotonic non-increasing
  - kstar <= max_k (no max_k+1)
  - kstar_status consistency

No GPU, no benchmark changes, no sweep_summary edits.
"""
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
sweep_path = 'results/exp4/b1_20260501_021259_job34546_NVIDIAH200/sweep_summary.csv'
df = pd.read_csv(sweep_path)

out_dir = 'results/exp4/b1_20260501_021259_job34546_NVIDIAH200/plots'
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Sanity check 1: grouping and k coverage
# ---------------------------------------------------------------------------
print("=== Sanity Checks ===")

issues = []

for (dataset, thresh), group in df.groupby(['dataset', 'threshold']):
    group = group.sort_values('max_filter_planes')
    max_k = int(group['max_plane_count'].iloc[0])
    actual_ks = set(group['max_filter_planes'].astype(int).tolist())
    expected_ks = set(range(1, max_k + 1))
    
    if actual_ks != expected_ks:
        missing = sorted(expected_ks - actual_ks)
        extra = sorted(actual_ks - expected_ks)
        msg = f"k coverage broken for {dataset}/{thresh}: missing={missing}, extra={extra}"
        issues.append(msg)
        print(f"  WARNING: {msg}")
    else:
        print(f"  OK: {dataset}/{thresh} covers k=1..{max_k}")
    
    # Sanity check 2: U(k) non-increasing
    u_vals = group['uncertain'].values
    for i in range(len(u_vals) - 1):
        if u_vals[i] < u_vals[i + 1]:
            msg = (f"U(k) non-monotonic for {dataset}/{thresh}: "
                   f"U({i+1})={u_vals[i]} < U({i+2})={u_vals[i+1]}")
            issues.append(msg)
            print(f"  WARNING: {msg}")

if issues:
    print(f"\n{len(issues)} sanity issue(s) found. Analysis continues but review needed.")
else:
    print("\nAll sanity checks passed.")

# ---------------------------------------------------------------------------
# Build enriched DataFrame with target_selectivity
# ---------------------------------------------------------------------------
# target_selectivity = exact_count / total_rows from the full-depth row (uncertain==0)
# observed_selectivity = gpu_count / n from each individual capped-k run

enriched = df.copy()
enriched['observed_selectivity'] = enriched['gpu_count'] / enriched['n']

# Merge target_selectivity per (dataset, threshold)
target_map = {}
for (dataset, thresh), group in df.groupby(['dataset', 'threshold']):
    full_depth = group[group['uncertain'] == 0]
    if len(full_depth) == 0:
        # Fallback: use the max k row
        full_depth = group.sort_values('max_filter_planes').tail(1)
        print(f"  WARNING: no full-depth uncertain==0 row for {dataset}/{thresh}, using max k row")
    exact_count = int(full_depth['cpu_raw_count'].iloc[0])
    total_rows = int(full_depth['n'].iloc[0])
    target_selectivity = exact_count / total_rows
    target_map[(dataset, thresh)] = {
        'exact_count': exact_count,
        'total_rows': total_rows,
        'target_selectivity': target_selectivity,
        'max_k': int(full_depth['max_plane_count'].iloc[0])
    }

enriched['target_selectivity'] = enriched.apply(
    lambda r: target_map[(r['dataset'], r['threshold'])]['target_selectivity'], axis=1)
enriched['exact_count'] = enriched.apply(
    lambda r: target_map[(r['dataset'], r['threshold'])]['exact_count'], axis=1)
enriched['total_rows'] = enriched.apply(
    lambda r: target_map[(r['dataset'], r['threshold'])]['total_rows'], axis=1)

# ---------------------------------------------------------------------------
# 1. count_precision_throughput.csv
# ---------------------------------------------------------------------------
pt = enriched[['dataset', 'threshold', 'target_selectivity', 'max_filter_planes',
               'uncertain', 'count_abs_error_bound', 'logical_GBps',
               'estimated_physical_GBps', 'rows_per_sec', 'exact_count', 'total_rows',
               'observed_selectivity']].copy()
pt.rename(columns={
    'max_filter_planes': 'k',
    'uncerved': 'U_k',
    'count_abs_error_bound': 'abs_error_bound',
    'logical_GBps': 'logical_GBps',
    'estimated_physical_GBps': 'estimated_physical_GBps',
    'rows_per_sec': 'rows_per_sec'
}, inplace=True)
pt['U_k'] = pt['uncertain']  # fix typo above
pt['rel_error_bound'] = pt['abs_error_bound'] / pt['exact_count'].clip(lower=1)
pt = pt[['dataset', 'threshold', 'target_selectivity', 'k', 'U_k',
         'abs_error_bound', 'rel_error_bound', 'logical_GBps',
         'estimated_physical_GBps', 'rows_per_sec',
         'exact_count', 'total_rows', 'observed_selectivity']]
pt = pt.sort_values(['dataset', 'threshold', 'k'])

pt.to_csv(f'{out_dir}/count_precision_throughput.csv', index=False)
print(f"\nWrote count_precision_throughput.csv ({len(pt)} rows)")

# ---------------------------------------------------------------------------
# 2. count_epsilon_to_kstar.csv
# ---------------------------------------------------------------------------
epsilon_abs_targets = [1, 10, 100, 1000, 10000, 100000, 1000000]
epsilon_rel_targets = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.10]

kstar_rows = []
for (dataset, thresh), group in df.groupby(['dataset', 'threshold']):
    group = group.sort_values('max_filter_planes')
    meta = target_map[(dataset, thresh)]
    exact_count = meta['exact_count']
    total_rows = meta['total_rows']
    target_sel = meta['target_selectivity']
    max_k = meta['max_k']
    
    for eps_abs in epsilon_abs_targets:
        valid = group[group['uncertain'] <= eps_abs]
        if len(valid) > 0:
            kstar_val = int(valid['max_filter_planes'].min())
            status = 'ok'
        else:
            kstar_val = np.nan
            status = 'unmet'
            print(f"  WARNING: no k satisfies abs epsilon={eps_abs} for {dataset}/{thresh}")
        
        kstar_rows.append({
            'dataset': dataset,
            'threshold': thresh,
            'target_selectivity': target_sel,
            'epsilon_type': 'abs',
            'epsilon': eps_abs,
            'kstar': kstar_val,
            'kstar_status': status,
            'exact_count': exact_count,
            'max_k': max_k
        })
    
    for eps_rel in epsilon_rel_targets:
        abs_bound = eps_rel * exact_count
        valid = group[group['uncertain'] <= abs_bound]
        if len(valid) > 0:
            kstar_val = int(valid['max_filter_planes'].min())
            status = 'ok'
        else:
            kstar_val = np.nan
            status = 'unmet'
            print(f"  WARNING: no k satisfies rel epsilon={eps_rel} for {dataset}/{thresh}")
        
        kstar_rows.append({
            'dataset': dataset,
            'threshold': thresh,
            'target_selectivity': target_sel,
            'epsilon_type': 'rel',
            'epsilon': eps_rel,
            'kstar': kstar_val,
            'kstar_status': status,
            'exact_count': exact_count,
            'max_k': max_k
        })

kstar_df = pd.DataFrame(kstar_rows)

# Sanity check 3: kstar <= max_k
bad_kstar = kstar_df[(kstar_df['kstar_status'] == 'ok') & (kstar_df['kstar'] > kstar_df['max_k'])]
if len(bad_kstar) > 0:
    print(f"\n  CRITICAL: {len(bad_kstar)} rows have kstar > max_k:")
    print(bad_kstar[['dataset', 'threshold', 'epsilon_type', 'epsilon', 'kstar', 'max_k']])
    sys.exit(1)
else:
    print("  OK: no kstar exceeds max_k")

# Sanity check 4: kstar_status consistency
ok_but_na = kstar_df[(kstar_df['kstar_status'] == 'ok') & kstar_df['kstar'].isna()]
unmet_but_val = kstar_df[(kstar_df['kstar_status'] == 'unmet') & kstar_df['kstar'].notna()]
if len(ok_but_na) > 0 or len(unmet_but_val) > 0:
    print("  CRITICAL: kstar_status / kstar value inconsistency detected")
    sys.exit(1)
else:
    print("  OK: kstar_status consistent with kstar value")

# Sanity check 5: full-depth should satisfy epsilon >= 0
for (dataset, thresh), group in kstar_df.groupby(['dataset', 'threshold']):
    full_depth = df[(df['dataset'] == dataset) & (df['threshold'] == thresh)]
    full_depth = full_depth[full_depth['uncertain'] == 0]
    if len(full_depth) > 0:
        # For abs epsilon=0, kstar should be <= max_k
        abs_zero = kstar_df[(kstar_df['dataset'] == dataset) &
                            (kstar_df['threshold'] == thresh) &
                            (kstar_df['epsilon_type'] == 'abs') &
                            (kstar_df['epsilon'] == 0)]
        if len(abs_zero) > 0 and abs_zero['kstar_status'].iloc[0] == 'unmet':
            print(f"  WARNING: full-depth exists but epsilon=0 unmet for {dataset}/{thresh}")

kstar_df.to_csv(f'{out_dir}/count_epsilon_to_kstar.csv', index=False)
print(f"Wrote count_epsilon_to_kstar.csv ({len(kstar_df)} rows)")

# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------

def savefig(name):
    plt.savefig(f'{out_dir}/{name}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Wrote {name}")

# --- Plot 1: COUNT error bound vs throughput ---
plt.figure(figsize=(10, 6))
dataset_order = sorted(pt['dataset'].unique())
marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X']
markers = {
    dataset: marker_cycle[i % len(marker_cycle)]
    for i, dataset in enumerate(dataset_order)
}
colors = {0.5: 'tab:blue', 0.9: 'tab:green', 0.99: 'tab:red'}

for dataset in pt['dataset'].unique():
    for sel in sorted(pt[pt['dataset'] == dataset]['target_selectivity'].unique()):
        sub = pt[(pt['dataset'] == dataset) & (pt['target_selectivity'] == sel)]
        c = colors.get(round(sel, 2), 'gray')
        plt.scatter(sub['logical_GBps'], sub['abs_error_bound'],
                    marker=markers[dataset], color=c, alpha=0.6, s=40,
                    label=f'{dataset} s={sel:.2f}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Logical throughput (GB/s)')
plt.ylabel('COUNT Absolute Error Bound (rows)')
plt.title('COUNT Error Bound vs Logical throughput (GB/s)')
plt.legend(fontsize=7, ncol=2)
plt.grid(True, alpha=0.3)
savefig('count_error_bound_vs_throughput.png')

# --- Plot 2: U(k) survival curve ---
plt.figure(figsize=(10, 6))
for dataset in df['dataset'].unique():
    for thresh in sorted(df[df['dataset'] == dataset]['threshold'].unique()):
        sub = df[(df['dataset'] == dataset) & (df['threshold'] == thresh)].sort_values('max_filter_planes')
        target_sel = target_map[(dataset, thresh)]['target_selectivity']
        label = f'{dataset} s={target_sel:.2f}'
        plt.plot(sub['max_filter_planes'], sub['uncertain'], marker='o', label=label, alpha=0.7)
plt.xlabel('k (max filter planes)')
plt.ylabel('U(k) — Uncertain Rows')
plt.title('Survival Curve: Uncertain Rows vs k')
plt.yscale('log')
plt.legend(fontsize=7, ncol=2)
plt.grid(True, alpha=0.3)
savefig('survival_curve_U_k.png')

# --- Plot 3: Throughput vs k ---
plt.figure(figsize=(10, 6))
for dataset in df['dataset'].unique():
    for thresh in sorted(df[df['dataset'] == dataset]['threshold'].unique()):
        sub = df[(df['dataset'] == dataset) & (df['threshold'] == thresh)].sort_values('max_filter_planes')
        target_sel = target_map[(dataset, thresh)]['target_selectivity']
        label = f'{dataset} s={target_sel:.2f}'
        plt.plot(sub['max_filter_planes'], sub['logical_GBps'], marker='o', label=label, alpha=0.7)
plt.xlabel('k (max filter planes)')
plt.ylabel('Logical throughput (GB/s)')
plt.title('Logical throughput (GB/s) vs k')
plt.legend(fontsize=7, ncol=2)
plt.grid(True, alpha=0.3)
savefig('throughput_vs_k.png')

# --- Plot 4: kstar vs epsilon (absolute) ---
plt.figure(figsize=(10, 6))
abs_df = kstar_df[kstar_df['epsilon_type'] == 'abs']
for dataset in abs_df['dataset'].unique():
    for sel in sorted(abs_df[abs_df['dataset'] == dataset]['target_selectivity'].unique()):
        sub = abs_df[(abs_df['dataset'] == dataset) & (abs_df['target_selectivity'] == sel)].sort_values('epsilon')
        label = f'{dataset} s={sel:.2f}'
        # Only plot ok rows; unmet rows have kstar=NaN
        ok_sub = sub[sub['kstar_status'] == 'ok']
        if len(ok_sub) > 0:
            plt.plot(ok_sub['epsilon'], ok_sub['kstar'], marker='o', label=label, alpha=0.7)
        # Mark unmet with a cross at max_k+1 (visual only, not data)
        unmet_sub = sub[sub['kstar_status'] == 'unmet']
        if len(unmet_sub) > 0:
            plt.scatter(unmet_sub['epsilon'], [sub['max_k'].iloc[0] + 0.5] * len(unmet_sub),
                        marker='x', color='red', s=50, alpha=0.5)
plt.xscale('log')
plt.xlabel('Epsilon (absolute error bound, rows)')
plt.ylabel('k* (minimum planes needed)')
plt.title('k* vs Absolute Epsilon')
plt.legend(fontsize=7, ncol=2)
plt.grid(True, alpha=0.3)
savefig('kstar_vs_epsilon_abs.png')

# --- Plot 5: kstar vs epsilon (relative) ---
plt.figure(figsize=(10, 6))
rel_df = kstar_df[kstar_df['epsilon_type'] == 'rel']
for dataset in rel_df['dataset'].unique():
    for sel in sorted(rel_df[rel_df['dataset'] == dataset]['target_selectivity'].unique()):
        sub = rel_df[(rel_df['dataset'] == dataset) & (rel_df['target_selectivity'] == sel)].sort_values('epsilon')
        label = f'{dataset} s={sel:.2f}'
        ok_sub = sub[sub['kstar_status'] == 'ok']
        if len(ok_sub) > 0:
            plt.plot(ok_sub['epsilon'], ok_sub['kstar'], marker='o', label=label, alpha=0.7)
        unmet_sub = sub[sub['kstar_status'] == 'unmet']
        if len(unmet_sub) > 0:
            plt.scatter(unmet_sub['epsilon'], [sub['max_k'].iloc[0] + 0.5] * len(unmet_sub),
                        marker='x', color='red', s=50, alpha=0.5)
plt.xscale('log')
plt.xlabel('Epsilon (relative error bound)')
plt.ylabel('k* (minimum planes needed)')
plt.title('k* vs Relative Epsilon')
plt.legend(fontsize=7, ncol=2)
plt.grid(True, alpha=0.3)
savefig('kstar_vs_epsilon_rel.png')

print("\nAll outputs written to:", out_dir)
print("Done.")
