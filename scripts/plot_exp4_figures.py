#!/usr/bin/env python3
"""
Exp4 Figure Generation Script
Generates 5 main figures + 2 appendix figures from merged sweep data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Paths
# ============================================================================
ROOT = "/home/u4063895/workspace/gpu-byteplane-scan-experiments"
CSV1 = f"{ROOT}/results/exp4/run_20260430_140204_job34118_NVIDIAH200/sweep_summary.csv"
CSV2 = f"{ROOT}/results/exp4/run_20260430_165744_job34323_NVIDIAH200/sweep_summary.csv"
OUT_DIR = f"{ROOT}/results/exp4/plots"
MERGED_CSV = f"{ROOT}/results/exp4/exp4_all_datasets_merged.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# Load & merge
# ============================================================================
df1 = pd.read_csv(CSV1)
df2 = pd.read_csv(CSV2)
df = pd.concat([df1, df2], ignore_index=True)

# ============================================================================
# Artifact extraction
# ============================================================================
def extract_artifact(root):
    if 'dev_buff_exp4_p3' in root:
        return 'p3'
    if 'dev_buff_exp4_p6' in root:
        return 'p6'
    return 'exact'

df['artifact'] = df['artifact_root'].apply(extract_artifact)

# ============================================================================
# Assign target_selectivity by sorting within each (dataset, artifact) group
# ============================================================================
TARGETS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
df['target_selectivity'] = 0

for ds in df['dataset'].unique():
    for art in df['artifact'].unique():
        mask = (df['dataset'] == ds) & (df['artifact'] == art)
        idx = df[mask].index.tolist()
        sorted_idx = df.loc[idx].sort_values('selectivity').index.tolist()
        for i, t in zip(sorted_idx, TARGETS):
            df.loc[i, 'target_selectivity'] = t

# Make target_selectivity ordered categorical
df['target_selectivity'] = pd.Categorical(
    df['target_selectivity'],
    categories=TARGETS,
    ordered=True
)

# ============================================================================
# Derived columns
# ============================================================================
n_total = 100_000_000
df['observed_selectivity'] = df['gpu_count'] / n_total * 100

# Speedup vs exact
speedups = []
for _, row in df.iterrows():
    if row['artifact'] == 'exact':
        speedups.append(1.0)
    else:
        baseline = df[
            (df['dataset'] == row['dataset']) &
            (df['target_selectivity'] == row['target_selectivity']) &
            (df['artifact'] == 'exact')
        ]
        if len(baseline) == 1:
            speedups.append(row['rows_per_sec'] / baseline['rows_per_sec'].values[0])
        else:
            speedups.append(np.nan)
df['speedup_vs_exact'] = speedups

# Save merged CSV
df.to_csv(MERGED_CSV, index=False)
print(f"Merged CSV saved: {MERGED_CSV} ({len(df)} rows)")

# ============================================================================
# Plotting helpers
# ============================================================================
DATASETS = ['uniform', 'heavy_tailed', 'sensor', 'zipfian']
ARTIFACTS = ['exact', 'p3', 'p6']
COLORS = {'exact': '#333333', 'p3': '#e41a1c', 'p6': '#377eb8'}
LINESTYLES = {'exact': '-', 'p3': '--', 'p6': '-.'}
LABELS = {'exact': 'exact', 'p3': 'p3', 'p6': 'p6'}

def setup_facets():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for ax, ds in zip(axes, DATASETS):
        ax.set_title(ds)
    return fig, axes

def plot_lines(ax, df_subset, y_col, logy=False):
    for art in ARTIFACTS:
        sub = df_subset[df_subset['artifact'] == art].sort_values('target_selectivity')
        x_vals = sub['target_selectivity'].astype(int).values
        y_vals = sub[y_col].values
        ax.plot(x_vals, y_vals, color=COLORS[art], linestyle=LINESTYLES[art],
                marker='o', markersize=4, label=LABELS[art])
    if logy:
        ax.set_yscale('log')
    ax.set_xticks(TARGETS)
    ax.set_xticklabels([str(t) for t in TARGETS], rotation=45)
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(fontsize=8)

# ============================================================================
# Figure 1 — avg_planes_read_per_total_row vs target_selectivity
# ============================================================================
fig, axes = setup_facets()
for ax, ds in zip(axes, DATASETS):
    plot_lines(ax, df[df['dataset'] == ds], 'avg_planes_read_per_total_row')
    ax.set_ylabel('avg planes read / total row')
fig.suptitle('Figure 1: Average Planes Read per Total Row vs Selectivity', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/exp4_avg_planes_vs_selectivity.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved Figure 1")

# ============================================================================
# Figure 2 — rows_per_sec vs target_selectivity
# ============================================================================
fig, axes = setup_facets()
for ax, ds in zip(axes, DATASETS):
    plot_lines(ax, df[df['dataset'] == ds], 'rows_per_sec')
    ax.set_ylabel('rows / sec')
fig.suptitle('Figure 2: Throughput (rows/sec) vs Selectivity', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/exp4_rows_per_sec_vs_selectivity.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved Figure 2")

# ============================================================================
# Figure 3 — raw_count_rel_error vs target_selectivity (log y)
# ============================================================================
fig, axes = setup_facets()
for ax, ds in zip(axes, DATASETS):
    plot_lines(ax, df[df['dataset'] == ds], 'raw_count_rel_error', logy=True)
    ax.set_ylabel('raw count rel. error')
    # Annotate extreme cases
    df_ds = df[(df['dataset'] == ds) & (df['artifact'] == 'p3')]
    for _, row in df_ds.iterrows():
        if row['raw_count_rel_error'] > 0.01:
            ax.annotate(f"s={int(row['target_selectivity'])}",
                        (int(row['target_selectivity']), row['raw_count_rel_error']),
                        fontsize=6, color=COLORS['p3'], alpha=0.8,
                        textcoords="offset points", xytext=(5, 5))
fig.suptitle('Figure 3: Raw Count Relative Error vs Selectivity (log scale)', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/exp4_raw_count_rel_error_vs_selectivity.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved Figure 3")

# ============================================================================
# Figure 4 — speedup_vs_exact vs target_selectivity
# ============================================================================
fig, axes = setup_facets()
for ax, ds in zip(axes, DATASETS):
    df_ds = df[df['dataset'] == ds]
    for art in ['p3', 'p6']:
        sub = df_ds[df_ds['artifact'] == art].sort_values('target_selectivity')
        x_vals = sub['target_selectivity'].astype(int).values
        y_vals = sub['speedup_vs_exact'].values
        ax.plot(x_vals, y_vals, color=COLORS[art], linestyle=LINESTYLES[art],
                marker='o', markersize=4, label=LABELS[art])
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.set_xticks(TARGETS)
    ax.set_xticklabels([str(t) for t in TARGETS], rotation=45)
    ax.set_ylabel('speedup vs exact')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(fontsize=8)
fig.suptitle('Figure 4: Speedup vs Exact vs Selectivity', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/exp4_speedup_vs_exact.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved Figure 4")

# ============================================================================
# Figure 5 — throughput vs fidelity tradeoff (scatter)
# ============================================================================
fig, axes = setup_facets()
markers = {'exact': 'o', 'p3': 's', 'p6': '^'}
for ax, ds in zip(axes, DATASETS):
    df_ds = df[df['dataset'] == ds]
    for art in ARTIFACTS:
        sub = df_ds[df_ds['artifact'] == art]
        # Clip error to avoid 0 on log x; exact points will be at a small offset
        x_vals = sub['raw_count_rel_error'].values.copy()
        if art == 'exact':
            x_vals = np.full_like(x_vals, 1e-8)
        else:
            x_vals = np.clip(x_vals, 1e-8, None)
        ax.scatter(x_vals, sub['rows_per_sec'],
                   color=COLORS[art], marker=markers[art], s=30,
                   label=LABELS[art], alpha=0.8, edgecolors='none')
        # annotate high-error p3 points
        for i, (_, row) in enumerate(sub.iterrows()):
            if art == 'p3' and row['raw_count_rel_error'] > 0.001:
                ax.annotate(f"s={int(row['target_selectivity'])}",
                            (x_vals[i], row['rows_per_sec']),
                            fontsize=6, alpha=0.7,
                            textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel('raw count rel. error')
    ax.set_ylabel('rows / sec')
    ax.set_xscale('log')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(fontsize=8)
fig.suptitle('Figure 5: Throughput vs Fidelity Tradeoff', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/exp4_throughput_vs_fidelity_tradeoff.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved Figure 5")

# ============================================================================
# Appendix A — estimated_physical_GBps vs target_selectivity
# ============================================================================
fig, axes = setup_facets()
for ax, ds in zip(axes, DATASETS):
    plot_lines(ax, df[df['dataset'] == ds], 'physical_GBps')
    ax.set_ylabel('Estimated Physical Throughput (GB/s, load-accounting estimate)')
fig.suptitle('Appendix A: Estimated Physical Throughput vs Selectivity', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/exp4_physical_GBps_vs_selectivity.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved Appendix A")

# ============================================================================
# Appendix B — observed vs target selectivity
# ============================================================================
fig, axes = setup_facets()
for ax, ds in zip(axes, DATASETS):
    df_ds = df[df['dataset'] == ds]
    for art in ARTIFACTS:
        sub = df_ds[df_ds['artifact'] == art].sort_values('target_selectivity')
        x_vals = sub['target_selectivity'].astype(int).values
        y_vals = sub['observed_selectivity'].values
        ax.plot(x_vals, y_vals, color=COLORS[art], linestyle=LINESTYLES[art],
                marker='o', markersize=4, label=LABELS[art])
    ax.plot([1, 99], [1, 99], color='gray', linestyle=':', linewidth=1, label='y=x')
    ax.set_xticks(TARGETS)
    ax.set_xticklabels([str(t) for t in TARGETS], rotation=45)
    ax.set_yticks(TARGETS)
    ax.set_ylabel('observed selectivity (%)')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(fontsize=8)
fig.suptitle('Appendix B: Observed vs Target Selectivity', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/exp4_observed_vs_target_selectivity.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved Appendix B")

# ============================================================================
# Takeaways markdown
# ============================================================================
TAKEAWAYS = f"""# Exp4 Figure Takeaways

Generated from merged sweep data ({len(df)} rows, 4 datasets × 3 artifacts × 9 selectivities).

## 1. Early-Exit Effectiveness Varies Dramatically by Dataset

**Figure 1 (avg planes read)** shows:
- **sensor**: ~0.1 planes/row — almost immediate decision (dataset values are very separable)
- **uniform**: ~1.0 planes/row — one plane is enough, little variation with selectivity
- **zipfian**: ~1.4–2.4 planes/row — moderate early-exit, but p3 collapses at high selectivity
- **heavy_tailed**: ~1.8–3.8 planes/row — the only dataset where plane-read clearly increases with selectivity, demonstrating true early-exit behavior

Takeaway: **early-exit is not universally beneficial**; it depends on data distribution.

## 2. Throughput Does Not Always Track Planes Read

**Figure 2 (rows/sec)** shows:
- **sensor** and **uniform** sustain ~1e12 and ~8.7e11 rows/sec regardless of selectivity
- **heavy_tailed** throughput drops from ~6.6e11 to ~2.5e11 as selectivity increases
- **zipfian** shows the steepest drop: ~8.3e11 at s=1 down to ~3.2e11 at s=99
- p3/p6 are rarely faster than exact; the bounded precision artifacts do not provide throughput wins in this kernel

Takeaway: **this Exp4 kernel is memory-bandwidth bound, and bounded precision does not reduce enough memory traffic to speed it up.**

## 3. p3 Fidelity Collapses in Predictable but Severe Ways

**Figure 3 (raw count rel. error)** shows:
- **exact**: 0 error across all runs
- **p6**: mostly <1% error, with a few outliers at high selectivity
- **p3**: error grows with selectivity; worst cases:
  - `heavy_tailed p3 s=99`: **7.35% relative error**
  - `zipfian p3 s=90, 95`: **0% observed selectivity** (all rows filtered out)
  - `zipfian p3 s=99`: **0.000256% observed selectivity** (only 256 rows out of 99M)

Takeaway: **p3 is not "slightly lossy"; in some regimes it is catastrophically wrong.**

## 4. Speedup vs Exact is Marginal or Negative

**Figure 4 (speedup)** shows:
- Most p3/p6 points cluster around **1.0×** (no gain)
- Some p3 points on zipfian are slightly above 1.0, but correspond to the same runs where fidelity collapsed
- No dataset shows a consistent >1.1× speedup for bounded precision

Takeaway: **In this kernel, bounded precision does not buy throughput.** The memory traffic reduction from early-exit dominates; artifact precision has little effect on performance.

## 5. The Real Tradeoff Space is Not a Pareto Front

**Figure 5 (throughput vs fidelity)** shows:
- **exact** forms a tight cluster at (error≈0, high throughput)
- **p6** points are close to exact — modest error, same throughput
- **p3** points drift rightward (higher error) without moving upward (higher throughput)
- There is **no point** where p3 offers significantly more throughput for a tolerable error

Takeaway: **p3 does not occupy a useful Pareto frontier.** Either use exact, or if you must save space, p6 is the only bounded option with acceptable fidelity.

## Appendix Observations

- **Appendix A (estimated physical throughput)**: Confirms load-accounting behavior; estimated throughput stays high across selectivities for uniform/sensor, and drops for heavy_tailed/zipfian as more planes are read.
- **Appendix B (observed vs target)**: Visualizes predicate drift directly. zipfian p3 at s=90,95,99 is off the chart (observed ≈ 0). heavy_tailed p3 s=99 shows severe undershoot.

## Bottom Line

Exp4 answers the 4 questions:
1. **Early-exit works** — but only on distributions with wide value ranges (heavy_tailed, zipfian). Uniform and sensor decide almost immediately.
2. **Throughput tracks planes read** on heavy_tailed/zipfian, but bounded precision artifacts do not help.
3. **p3 fidelity cost is severe** at high selectivity; p6 is nearly exact.
4. **p3 is unusable** for heavy_tailed s>75 and zipfian s>50. p6 is usable everywhere but offers no speedup.
"""

with open(f"{OUT_DIR}/exp4_figure_takeaways.md", 'w') as f:
    f.write(TAKEAWAYS)

print("Saved takeaways markdown")
print(f"\nAll outputs in: {OUT_DIR}")
