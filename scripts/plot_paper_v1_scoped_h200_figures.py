import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create the output directory
OUTPUT_DIR = "results/paper_v1/figures_scoped_h200"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use a clean, modern aesthetic for paper figures
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# Color palette: HSL-inspired professional palettes
COLOR_K2 = "#2b5c8f"    # Deep elegant blue for k=2
COLOR_KMAX = "#d95f02"  # Contrasting premium orange/amber for k=max
COLOR_RAW = "#666666"   # Muted grey for baselines

def make_figure_1():
    print("Generating Figure 1: Scientific Headline Speedups...")
    
    # 1. Load scientific_locality_attribution.csv (for cesm_atm_cloud and hurricane_u)
    # Filter for segment_size in ('168480000', '25000000') (quasi-global single-segment)
    df_main = pd.read_csv("results/exp4_filter_aggregate/scientific_locality_attribution_57768/scientific_locality_attribution.csv")
    df_main_filtered = df_main[df_main['segment_size'].isin([168480000, 25000000])].copy()
    
    # Keep only relevant columns and map names
    # Column in df_main: bp_vs_raw_speedup
    df_main_subset = df_main_filtered[['dataset', 'selectivity', 'k', 'k_type', 'bp_vs_raw_speedup', 'warm_e2e_ms', 'raw_fused_ms']].copy()
    df_main_subset.rename(columns={'bp_vs_raw_speedup': 'speedup'}, inplace=True)
    
    # 2. Load scientific_extra_fields_quasi_global.csv (for cesm_atm_q and hurricane_tc)
    df_extra = pd.read_csv("results/exp4_filter_aggregate/scientific_extra_fields_quasi_global_58440/scientific_extra_fields_quasi_global.csv")
    
    # Map selectivity to standard 10, 50, 90 labels for cesm_atm_q and hurricane_tc
    # Let's inspect the selectivity float values and categorize them
    def map_selectivity(row):
        sel = row['selectivity']
        dataset = row['dataset']
        if sel < 0.15:
            return 10
        elif sel < 0.55:
            return 50
        else:
            return 90

    df_extra['selectivity_pct'] = df_extra.apply(map_selectivity, axis=1)
    
    # In df_extra, k=2 is k_star, and k=6 is k_max (max_filter_planes)
    df_extra['k_type'] = df_extra['max_filter_planes'].apply(lambda x: 'k_star' if x == 2 else 'k_max')
    
    # Map columns to match df_main
    df_extra_subset = df_extra[['dataset', 'selectivity_pct', 'max_filter_planes', 'k_type', 'speedup_vs_raw', 'ms_per_iter', 'raw_baseline_ms_per_iter']].copy()
    df_extra_subset.rename(columns={
        'selectivity_pct': 'selectivity',
        'max_filter_planes': 'k',
        'speedup_vs_raw': 'speedup',
        'ms_per_iter': 'warm_e2e_ms',
        'raw_baseline_ms_per_iter': 'raw_fused_ms'
    }, inplace=True)
    
    # 3. Combine both dataframes
    df_headline = pd.concat([df_main_subset, df_extra_subset], ignore_index=True)
    
    # Filter only for k_type in ('k_star', 'k_max') to follow the k=2 vs k=max layout
    # Since k_star is k=2 and k_max is k=max (k=7, 8 or 6)
    # Ensure cesm_atm_cloud and hurricane_u k_type is mapped correctly if there are minor naming variations
    df_headline['k_label'] = df_headline['k_type'].map({'k_star': 'Shallow-k (k=2)', 'k_max': 'Fallback (k=max)'})
    
    # Clean dataset names for presentation
    dataset_map = {
        'cesm_atm_cloud': 'CESM Cloud Fraction',
        'hurricane_u': 'Hurricane U-Wind',
        'cesm_atm_q': 'CESM Specific Humidity',
        'hurricane_tc': 'Hurricane Temp (bfp-dec11)'
    }
    df_headline['dataset_clean'] = df_headline['dataset'].map(dataset_map)
    
    # Filter to only the 4 target datasets and standard selectivities
    df_headline = df_headline[df_headline['dataset'].isin(dataset_map.keys())].copy()
    
    # Precise filter to exclude non-plotted rows (e.g., k=1 and other references)
    df_headline = df_headline[df_headline['k_label'].notna()].copy()
    
    # Save the source CSV snapshot for claim-hygiene
    df_headline.to_csv(f"{OUTPUT_DIR}/src_fig1_scientific_headline.csv", index=False)
    
    # Now, let's create a beautiful plot
    # Grouped bar chart: x-axis = Dataset, y-axis = Speedup, hue = k_label, facets = selectivity
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    
    selectivities = [10, 50, 90]
    for i, sel in enumerate(selectivities):
        ax = axes[i]
        df_sel = df_headline[df_headline['selectivity'] == sel].copy()
        
        # We want to plot k=2 vs k=max grouped by dataset
        sns.barplot(
            data=df_sel,
            x='dataset_clean',
            y='speedup',
            hue='k_label',
            ax=ax,
            palette=[COLOR_K2, COLOR_KMAX],
            edgecolor="black",
            linewidth=0.8
        )
        
        if sel == 90:
            ax.set_title("Target Selectivity: 90%\n(High-Sel. Bucket*)", fontweight='bold', fontsize=11)
        else:
            ax.set_title(f"Target Selectivity: {sel}%", fontweight='bold', fontsize=12)
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel("Warm E2E Speedup vs. Raw Fused FP64", fontweight='bold')
        else:
            ax.set_ylabel("")
            
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        ax.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7) # baseline threshold
        
        # Legend configuration
        if i == 0:
            ax.legend(title="Execution Depth")
        else:
            ax.get_legend().remove()
            
    plt.suptitle("Scientific Headline Speedup on H200 (True Quasi-Global Single-Segment)", y=0.98, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f"{OUTPUT_DIR}/fig1_scientific_headline.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/fig1_scientific_headline.pdf", bbox_inches='tight')
    plt.close()
    print("Figure 1 generated successfully.")

def make_figure_2():
    print("Generating Figure 2: SUM error / speedup tradeoff...")
    
    # Load canonical v2 SUM precision-throughput data
    df = pd.read_csv("results/precision_throughput/exp3_v2_sum_precision_throughput.csv")
    
    # Clean dataset names (only include plotted datasets)
    dataset_map = {
        'sensor': 'Sensor',
        'heavy_tailed': 'Synthetic Heavy-tailed'
    }
    df['dataset_clean'] = df['dataset'].map(dataset_map)
    df = df[df['dataset_clean'].notna()].copy()
    
    # Save the source CSV snapshot
    df.to_csv(f"{OUTPUT_DIR}/src_fig2_sum_tradeoff.csv", index=False)
    
    # Plot: y-axis = relative error, x-axis = throughput (logical GB/s or GRows/sec)
    # We will use two representative datasets to show the clear trend: Sensor and Heavy-tailed
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    datasets_plot = ['sensor', 'heavy_tailed']
    for i, ds in enumerate(datasets_plot):
        ax = axes[i]
        df_ds = df[df['dataset'] == ds].copy()
        
        # Sort by k to draw a nice curve
        df_ds = df_ds.sort_values(by=['artifact_label', 'k'])
        
        # Group by artifact_label (representation precision) and plot curves
        artifacts = sorted(df_ds['artifact_label'].unique(), key=lambda x: int(x[1:]) if x.startswith('p') else 0)
        
        # We will use lineplots where x is throughput (logical_GBps) and y is rel_error_vs_encoded_full_depth
        for art in artifacts:
            df_art = df_ds[df_ds['artifact_label'] == art]
            
            # Scatter and Line
            ax.plot(
                df_art['logical_GBps'],
                df_art['rel_error_vs_encoded_full_depth'],
                marker='o',
                label=f"Artifact {art}",
                linewidth=1.5,
                markersize=6
            )
            
            # Annotate k values on the curve for representative artifacts
            # Annotate k=1, k=2, k=3, k=max
            for _, row in df_art.iterrows():
                k_val = int(row['k'])
                if k_val in (1, 2, 3, int(row['max_plane_count'])):
                    ax.annotate(
                        f"k={k_val}",
                        (row['logical_GBps'], row['rel_error_vs_encoded_full_depth']),
                        textcoords="offset points",
                        xytext=(0, 7),
                        ha='center',
                        fontsize=8,
                        alpha=0.8
                    )
                    
        ax.set_title(f"Dataset: {dataset_map[ds]}", fontweight='bold', fontsize=12)
        ax.set_xlabel("Throughput (Logical Workload-Normalized GB/s)", fontweight='bold')
        if i == 0:
            ax.set_ylabel("Relative SUM Error vs. Encoded Full-Depth SUM", fontweight='bold')
        else:
            ax.set_ylabel("")
            
        # Log scale for relative error to show extremely tiny error at deeper k
        ax.set_yscale('symlog', linthresh=1e-8)
        # Format y axis
        ax.yaxis.grid(True, which='both')
        ax.legend(title="Precision Level")
        
    plt.suptitle("Throughput vs. Relative SUM Error Tradeoff on H200 (Progressive Capped-k)", y=0.98, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f"{OUTPUT_DIR}/fig2_sum_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/fig2_sum_tradeoff.pdf", bbox_inches='tight')
    plt.close()
    print("Figure 2 generated successfully.")

def make_figure_3():
    print("Generating Figure 3: Threshold-Preparation Locality...")
    
    # Load manual source locality table compiled from committed reports
    df_loc = pd.read_csv(f"{OUTPUT_DIR}/src_fig3_threshold_locality_manual_source.csv")
    
    # Save the snapshot representing the exact input for this figure
    df_loc.to_csv(f"{OUTPUT_DIR}/src_fig3_threshold_locality.csv", index=False)
    
    # We will generate a bar chart showing the comparison of prep time vs. scan time to illustrate the bottleneck
    # Let's filter for selectivity = 50% as the most representative middle selectivity
    df_plot = df_loc[df_loc['selectivity'] == "50%"].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    
    datasets_plot = ["CESM Cloud Fraction", "Hurricane U-Wind"]
    for i, ds in enumerate(datasets_plot):
        ax = axes[i]
        df_ds = df_plot[df_plot['dataset'] == ds].copy()
        
        # Set up stacked bar chart data
        x = np.arange(len(df_ds))
        width = 0.5
        
        # Plot Scan only time and Prep time stacked
        rects1 = ax.bar(x, df_ds['scan_only_ms'], width, label='GPU scan+agg execution', color=COLOR_K2, edgecolor='black')
        rects2 = ax.bar(x, df_ds['prep_time_ms'], width, bottom=df_ds['scan_only_ms'], label='Host threshold prep', color="#e34a33", edgecolor='black')
        
        ax.set_title(f"{ds} (s=50%)", fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(df_ds['segment_size'])
        ax.set_xlabel("Segment Size (Locality Model)")
        if i == 0:
            ax.set_ylabel("Execution Time Breakdown (ms)", fontweight='bold')
            ax.legend()
            
        # Add labels on top of the bars to make prep time explicitly visible
        for j, rect in enumerate(rects2):
            total_height = df_ds.iloc[j]['warm_e2e_ms']
            prep_val = df_ds.iloc[j]['prep_time_ms']
            ax.annotate(
                f"{prep_val:.2f} ms prep\n({total_height:.2f} ms total)",
                xy=(rect.get_x() + rect.get_width() / 2, total_height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9
            )
            
    plt.suptitle("Threshold-Preparation Locality Breakdown on H200 (50% Selectivity)", y=0.98, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f"{OUTPUT_DIR}/fig3_threshold_locality.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/fig3_threshold_locality.pdf", bbox_inches='tight')
    plt.close()
    print("Figure 3 generated successfully.")

def make_figure_4():
    print("Generating Figure 4: Q8 NCU Physical-Traffic Attribution...")
    
    # Load physical traffic data for cesm_atm_q s10 from Job 59096
    df = pd.read_csv("results/exp4_filter_aggregate/q8_hbm_transaction_20260523_003039_job59096_NVIDIAH200/q8_hbm_transaction_summary.csv")
    
    # Convert dram bytes string/float to consistent float MB
    def parse_dram(val):
        val_str = str(val).strip()
        if 'Gbyte' in val_str:
            return float(val_str.replace('Gbyte', '').strip()) * 1024.0
        elif 'Mbyte' in val_str:
            return float(val_str.replace('Mbyte', '').strip())
        else:
            return float(val_str)
            
    df['dram_mb'] = df['dram_bytes_or_sector_proxy'].apply(parse_dram)
    
    # Save the source CSV snapshot
    df.to_csv(f"{OUTPUT_DIR}/src_fig4_ncu_traffic.csv", index=False)
    
    # We will create a grouped bar chart with 3 columns representing:
    # 1. DRAM Physical Bytes (MB)
    # 2. L2 Sector Reads (Millions)
    # 3. L1-TEX Sector Reads (Millions)
    # And we compare: Raw FP64 baseline, Byte-plane k=2, and Byte-plane k=max(6)
    
    # Get values
    row_raw = df[df['k'] == 'raw'].iloc[0]
    row_k2 = df[df['k'] == '2'].iloc[0]
    row_kmax = df[df['k'] == 'max'].iloc[0]
    
    categories = ['DRAM Bytes (MB)', 'L2 Sectors (Millions)', 'L1-TEX Sectors (Millions)']
    
    # Values
    raw_vals = [row_raw['dram_mb'], row_raw['l2_sector_proxy'] / 1e6, row_raw['l1tex_sector_proxy'] / 1e6]
    k2_vals = [row_k2['dram_mb'], row_k2['l2_sector_proxy'] / 1e6, row_k2['l1tex_sector_proxy'] / 1e6]
    kmax_vals = [row_kmax['dram_mb'], row_kmax['l2_sector_proxy'] / 1e6, row_kmax['l1tex_sector_proxy'] / 1e6]
    
    # We will normalize each metric to the Raw FP64 baseline to show the physical traffic reduction ratio
    # OR we can plot raw values on log scale/multiple subplots
    # Multiple subplots is cleaner due to different units
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    
    titles = [
        "A. DRAM Physical Reads (MB)",
        "B. L2 Cache Read Sectors (M)",
        "C. L1-TEX Read Sectors (M)"
    ]
    
    units = ["Physical Read (MB)", "Sectors (Millions)", "Sectors (Millions)"]
    
    for i in range(3):
        ax = axes[i]
        
        bars = ax.bar(
            ['Raw FP64', 'Byte-plane (k=2)', 'Byte-plane (k=max)'],
            [raw_vals[i], k2_vals[i], kmax_vals[i]],
            color=[COLOR_RAW, COLOR_K2, COLOR_KMAX],
            edgecolor='black',
            width=0.5
        )
        
        ax.set_title(titles[i], fontweight='bold', fontsize=12)
        ax.set_ylabel(units[i], fontweight='bold')
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9
            )
            
        # Add reduction label relative to Raw FP64
        k2_reduction = raw_vals[i] / k2_vals[i]
        kmax_reduction = raw_vals[i] / kmax_vals[i]
        
        ax.annotate(
            f"{k2_reduction:.2f}x red.",
            xy=(1, k2_vals[i] / 2),
            color='white', fontweight='bold', ha='center', va='center', fontsize=9
        )
        ax.annotate(
            f"{kmax_reduction:.2f}x red.",
            xy=(2, kmax_vals[i] / 2),
            color='white', fontweight='bold', ha='center', va='center', fontsize=9
        )
        
    plt.suptitle("Q8 NCU Physical-Traffic Attribution on H200 (CESM Specific Humidity, s10)", y=0.98, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f"{OUTPUT_DIR}/fig4_ncu_traffic.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/fig4_ncu_traffic.pdf", bbox_inches='tight')
    plt.close()
    print("Figure 4 generated successfully.")

def make_figure_5():
    print("Generating Figure 5: Prompt5 k-Depth Break-Even / Dispatch Evidence...")
    
    # Load Prompt5 CSV data
    # 1. heavy_tailed_p6 from Job 59418
    df_ht = pd.read_csv("results/prompt5_h200_viability/ncu_kdepth_20260523_165051_job59418_NVIDIAH200/k_depth_ncu_matrix.csv")
    df_ht_s50 = df_ht[(df_ht['selectivity_label'] == 's50') & (df_ht['kernel'] == 'fused')].copy()
    
    # Add raw FP64 row to df_ht_s50
    df_ht_raw = df_ht[(df_ht['selectivity_label'] == 's50') & (df_ht['kernel'] == 'raw_fp64')].iloc[0]
    
    # 2. uniform_p10 from Job 59368
    df_uni = pd.read_csv("results/prompt5_h200_viability/ncu_kdepth_20260523_135553_job59368_NVIDIAH200/k_depth_ncu_matrix.csv")
    df_uni_s50 = df_uni[(df_uni['selectivity_label'] == 's50') & (df_uni['kernel'] == 'fused')].copy()
    df_uni_raw = df_uni[(df_uni['selectivity_label'] == 's50') & (df_uni['kernel'] == 'raw_fp64')].iloc[0]
    
    # Combined snapshot (keep only s50 selectivity rows actually plotted)
    df_ht_plotted = df_ht[df_ht['selectivity_label'] == 's50'].copy()
    df_uni_plotted = df_uni[df_uni['selectivity_label'] == 's50'].copy()
    df_combined = pd.concat([df_ht_plotted, df_uni_plotted], ignore_index=True)
    df_combined.to_csv(f"{OUTPUT_DIR}/src_fig5_prompt5_break_even.csv", index=False)
    
    # We will plot k vs speedup_vs_raw (Left axis) and k vs dram_ratio_vs_raw (Right axis)
    # for both uniform_p10 s50 and heavy_tailed_p6 s50
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: uniform_p10 s50
    # Sort uniform by k
    df_uni_s50 = df_uni_s50.sort_values(by='k')
    df_uni_s50['k_int'] = df_uni_s50['k'].astype(int)
    
    # Left axis: Latency Speedup
    color = COLOR_K2
    ax1.set_xlabel('Execution Depth (k)', fontweight='bold')
    ax1.set_ylabel('Benchmark Latency Speedup vs. Raw FP64', color=color, fontweight='bold')
    line1 = ax1.plot(df_uni_s50['k_int'], df_uni_s50['speedup_vs_raw'], marker='s', color=color, linewidth=2, label='Speedup vs. Raw (Left)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7) # break-even boundary
    ax1.axhline(1.10, color='green', linestyle=':', linewidth=1, alpha=0.7) # strong dispatch boundary
    
    # Add text label for strong vs marginal vs fallback
    ax1.text(2, 1.15, "Strong Path (>=1.10x)", color="green", fontsize=9, ha='center')
    ax1.text(3, 1.05, "Marginal Path", color="orange", fontsize=9, ha='center')
    ax1.text(4, 0.94, "Fallback (Reject)", color="red", fontsize=9, ha='center')
    
    # Right axis: DRAM read ratio
    color = COLOR_KMAX
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('NCU Physical DRAM Read Ratio vs. Raw', color=color, fontweight='bold')
    line2 = ax1_twin.plot(df_uni_s50['k_int'], df_uni_s50['dram_ratio_vs_raw'], marker='o', color=color, linestyle='--', linewidth=2, label='DRAM Traffic Ratio (Right)')
    ax1_twin.tick_params(axis='y', labelcolor=color)
    ax1_twin.set_ylim(0, 1.1)
    
    # Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.set_title("A. Synthetic Uniform (uniform_p10, s50)", fontweight='bold', fontsize=12)
    ax1.set_xticks([2, 3, 4])
    
    # Panel B: heavy_tailed_p6 s50
    # Sort heavy_tailed by k
    df_ht_s50 = df_ht_s50.sort_values(by='k')
    df_ht_s50['k_int'] = df_ht_s50['k'].astype(int)
    
    # Left axis: Latency Speedup
    color = COLOR_K2
    ax2.set_xlabel('Execution Depth (k)', fontweight='bold')
    line3 = ax2.plot(df_ht_s50['k_int'], df_ht_s50['speedup_vs_raw'], marker='s', color=color, linewidth=2, label='Speedup vs. Raw (Left)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(1.10, color='green', linestyle=':', linewidth=1, alpha=0.7)
    
    # Add text label for strong vs marginal vs fallback
    ax2.text(1, 1.15, "Strong", color="green", fontsize=9, ha='center')
    ax2.text(2, 1.05, "Marginal", color="orange", fontsize=9, ha='center')
    ax2.text(3, 0.93, "Fallback", color="red", fontsize=9, ha='center')
    
    # Right axis: DRAM read ratio
    color = COLOR_KMAX
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('NCU Physical DRAM Read Ratio vs. Raw', color=color, fontweight='bold')
    line4 = ax2_twin.plot(df_ht_s50['k_int'], df_ht_s50['dram_ratio_vs_raw'], marker='o', color=color, linestyle='--', linewidth=2, label='DRAM Traffic Ratio (Right)')
    ax2_twin.tick_params(axis='y', labelcolor=color)
    ax2_twin.set_ylim(0, 1.1)
    
    # Legend
    lines2 = line3 + line4
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc='upper right')
    ax2.set_title("B. Synthetic Heavy-tailed (heavy_tailed_p6, s50)", fontweight='bold', fontsize=12)
    ax2.set_xticks([1, 2, 3, 4, 6])
    
    plt.suptitle("Prompt5 k-Depth Latency vs. Physical Traffic Tradeoff on H200", y=0.98, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f"{OUTPUT_DIR}/fig5_prompt5_break_even.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/fig5_prompt5_break_even.pdf", bbox_inches='tight')
    plt.close()
    print("Figure 5 generated successfully.")

if __name__ == "__main__":
    make_figure_1()
    make_figure_2()
    make_figure_3()
    make_figure_4()
    make_figure_5()
    print("All paper figures generated successfully.")
