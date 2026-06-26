
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

PLOT_DIR = "results/paper_v1/plots_gemini"
os.makedirs(PLOT_DIR, exist_ok=True)

def fig1_transfer_cost():
    print("Generating Figure 1: Transfer Cost...")
    df_artifact = pd.read_csv("results/paper_v1/artifact_size_fidelity_transfer.csv")
    df_raw = pd.read_csv("results/paper_v1/raw_fp64_transfer_baseline.csv")
    
    # Filter for p6 artifact as representative mainline
    df_artifact = df_artifact[df_artifact['artifact_label'] == 'p6'].drop_duplicates(subset=['dataset'])
    
    # Prepare comparison data
    plot_data = []
    for dataset in df_raw['dataset'].unique():
        raw_row = df_raw[df_raw['dataset'] == dataset].iloc[0]
        art_rows = df_artifact[df_artifact['dataset'] == dataset]
        
        # Raw FP64
        plot_data.append({
            'dataset': dataset,
            'type': 'Raw FP64',
            'bytes_per_row': 8.0,
            'cudaMemcpy_ms': raw_row['cudaMemcpy_ms']
        })
        
        # Encoded (p6)
        if not art_rows.empty:
            art_row = art_rows.iloc[0]
            plot_data.append({
                'dataset': dataset,
                'type': 'Encoded (p6)',
                'bytes_per_row': art_row['bytes_per_row'],
                'cudaMemcpy_ms': art_row['cudaMemcpy_ms']
            })
            
    df_plot = pd.DataFrame(plot_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.barplot(data=df_plot, x='dataset', y='bytes_per_row', hue='type', ax=ax1)
    ax1.set_ylabel("Bytes per Row")
    ax1.set_title("A. Payload Size Reduction")
    ax1.legend(title=None)
    
    sns.barplot(data=df_plot, x='dataset', y='cudaMemcpy_ms', hue='type', ax=ax2)
    ax2.set_ylabel("HtoD Memcpy (ms)")
    ax2.set_title("B. Transfer Time Reduction")
    ax2.legend(title=None)
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/transfer_cost_vs_raw_fp64.png", dpi=300)
    plt.close()

def fig2_suitability_matrix():
    print("Generating Figure 2: Suitability Matrix...")
    df = pd.read_csv("results/paper_v1/new_encoder_predicate_drift.csv")
    
    # Map verdict to numeric for heatmap
    verdict_map = {'acceptable': 2, 'caution': 1, 'reject': 0}
    df['verdict_score'] = df['count_verdict'].map(verdict_map)
    
    # Get worst verdict and most frequent suitability per (dataset, artifact_label)
    pivot_verdict = df.groupby(['dataset', 'artifact_label'])['verdict_score'].min().unstack()
    pivot_suit = df.groupby(['dataset', 'artifact_label'])['count_suitability'].first().unstack()
    
    # Reorder columns for logical progression p2 -> p10
    cols = sorted(pivot_verdict.columns, key=lambda x: int(x[1:]) if x.startswith('p') else 0)
    pivot_verdict = pivot_verdict[cols]
    pivot_suit = pivot_suit[cols]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_verdict, annot=pivot_suit, fmt="", cmap="RdYlGn", cbar=False)
    plt.title("Encoder Fidelity & Suitability Matrix")
    plt.xlabel("Artifact Label (Plane Count)")
    plt.ylabel("Dataset")
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/encoder_fidelity_suitability_matrix.png", dpi=300)
    plt.close()

def fig3_predicate_drift():
    print("Generating Figure 3: Predicate Drift...")
    df = pd.read_csv("results/paper_v1/new_encoder_predicate_drift.csv")
    
    # Filter for target datasets
    df = df[df['dataset'].isin(['heavy_tailed', 'zipfian'])]
    
    # Sort artifact labels
    df['art_num'] = df['artifact_label'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values(['dataset', 'art_num', 'target_selectivity'])
    
    g = sns.FacetGrid(df, col="dataset", hue="artifact_label", height=5, aspect=1.2, sharey=True)
    g.map(sns.lineplot, "target_selectivity", "selectivity_drift_pp", marker="o")
    
    g.add_legend(title="Artifact")
    g.set_axis_labels("Target Selectivity (%)", "Selectivity Drift (pp)")
    g.set_titles("{col_name}")
    
    for ax in g.axes.flat:
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/predicate_drift_failure_boundary.png", dpi=300)
    plt.close()

def fig4_k_tradeoff():
    print("Generating Figure 4: K-Tradeoff...")
    df = pd.read_csv("results/exp4/b1_20260510_174435_job44743_NVIDIAH200/count_precision_throughput.csv")
    
    # Canonical mainline: p6
    df_p6 = df[df['artifact_label'] == 'p6'].copy()
    df_p6['uncertainty_fraction'] = df_p6['uncertain'] / 1e8
    df_p6['throughput_gbps'] = df_p6['rows_per_sec'] / 1e9
    
    datasets = df_p6['dataset'].unique()
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(2, n_datasets, figsize=(4*n_datasets, 8), sharex=True)
    
    if n_datasets == 1:
        axes = axes.reshape(2, 1)

    for i, dataset in enumerate(datasets):
        subset = df_p6[df_p6['dataset'] == dataset].sort_values('max_filter_planes')
        
        # Panel A: Uncertainty
        sns.lineplot(data=subset, x='max_filter_planes', y='uncertainty_fraction', marker='s', ax=axes[0, i])
        axes[0, i].set_title(f"{dataset}")
        axes[0, i].set_ylabel("Uncertainty Fraction")
        
        # Panel B: Throughput
        sns.lineplot(data=subset, x='max_filter_planes', y='throughput_gbps', marker='o', ax=axes[1, i], color='orange')
        axes[1, i].set_ylabel("Throughput (Grows/sec)")
        axes[1, i].set_xlabel("Max Filter Planes (k)")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/count_k_tradeoff_mainline.png", dpi=300)
    plt.close()

def fig5_epsilon_heatmap():
    print("Generating Figure 5: Epsilon Heatmap...")
    df = pd.read_csv("results/exp4/b1_20260510_174435_job44743_NVIDIAH200/count_epsilon_to_kstar.csv")
    
    # Filter for representative mainline artifact: p6
    # Also handle 'artifact' column name vs 'artifact_label'
    art_col = 'artifact' if 'artifact' in df.columns else 'artifact_label'
    df = df[df[art_col] == 'p6'].copy()

    # Representative epsilons
    rep_eps = [0.0, 1e-4, 1e-2, 0.1]
    df_rep = df[df['epsilon'].isin(rep_eps)].copy()
    
    # Calculate kstar ratio
    df_rep['kstar_ratio'] = df_rep['kstar'] / df_rep['max_plane_count']
    
    # Pivot based on heavy_tailed as representative
    df_ht = df_rep[df_rep['dataset'] == 'heavy_tailed']
    
    if df_ht.empty:
        if not df_rep.empty:
            df_ht = df_rep[df_rep['dataset'] == df_rep['dataset'].unique()[0]]
        else:
            print("No data for Figure 5.")
            return
            
    pivot = df_ht.groupby(['selectivity', 'epsilon'])['kstar_ratio'].mean().unstack()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    dataset_name = df_ht['dataset'].iloc[0]
    plt.title(f"Search Depth (k*) Ratio vs. Epsilon and Selectivity\n(Dataset: {dataset_name}, Artifact: p6)")
    plt.xlabel("Epsilon (Error Tolerance)")
    plt.ylabel("Selectivity")
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/epsilon_to_kstar_heatmap.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    fig1_transfer_cost()
    fig2_suitability_matrix()
    fig3_predicate_drift()
    fig4_k_tradeoff()
    fig5_epsilon_heatmap()
    print("Done.")
