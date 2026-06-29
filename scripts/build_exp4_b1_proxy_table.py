import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("${PROJ_DIR}/workspace/gpu-byteplane-scan-experiments")

# ── 1. Read inputs ──────────────────────────────────────────────
sweep_path = BASE / "results/exp4/b1_20260510_174435_job44743_NVIDIAH200/sweep_summary.csv"
eps_path   = BASE / "results/exp4/b1_20260510_174435_job44743_NVIDIAH200/count_epsilon_to_kstar.csv"
transfer_path = BASE / "results/paper_v1/artifact_size_fidelity_transfer.csv"
raw_transfer_path = BASE / "results/paper_v1/raw_fp64_transfer_baseline.csv"
ncu_blocker_path = BASE / "research/2026-05-11_Exp4B1_NCU_Physical_Traffic_Audit_Status.md"

for p in [sweep_path, eps_path, transfer_path, raw_transfer_path, ncu_blocker_path]:
    if not p.exists():
        raise FileNotFoundError(f"Required input missing: {p}")

sweep = pd.read_csv(sweep_path)
eps_df = pd.read_csv(eps_path)
transfer = pd.read_csv(transfer_path)
raw_transfer = pd.read_csv(raw_transfer_path)

print(f"sweep_summary: {len(sweep)} rows")
print(f"epsilon_to_kstar: {len(eps_df)} rows")
print(f"transfer: {len(transfer)} rows")
print(f"raw_transfer: {len(raw_transfer)} rows")

# ── 2. Normalize artifact_label ────────────────────────────────
sweep["artifact_label"] = "p" + sweep["precision_decimals"].astype(str)
eps_df["artifact_label"] = eps_df["artifact"]

# ── 3. Transfer data join ──────────────────────────────────────
# sweep has (dataset, artifact_label, threshold) per row.
# Transfer data has (dataset, artifact_label, target_selectivity, threshold, ...).
# Use threshold-based merge with tolerance.
sweep_sorted = sweep.sort_values("threshold").reset_index(drop=True)
transfer_sorted = transfer.sort_values("threshold").reset_index(drop=True)

# Use merge_asof with by keys for exact dataset/artifact match
# but direction='nearest' with tolerance on threshold
# First, need to ensure we match within threshold tolerance
joined = pd.merge_asof(
    sweep_sorted,
    transfer_sorted,
    on="threshold",
    by=["dataset", "artifact_label"],
    direction="nearest",
    tolerance=1e-6
)

missing_join = joined[joined["target_selectivity"].isna()]
if len(missing_join) > 0:
    print(f"WARNING: {len(missing_join)} sweep rows could not join to transfer. "
          f"Datasets: {missing_join[['dataset', 'artifact_label', 'threshold']].drop_duplicates().to_dict('records')}")
    # Keep rows but transfer fields will be NA

# ── 4. Raw transfer baseline join ──────────────────────────────
joined = joined.merge(raw_transfer, on="dataset", how="left", suffixes=("", "_raw"))

# ── 5. Artifact class policy ──────────────────────────────────
def classify_artifact(dataset, label):
    if dataset == "heavy_tailed":
        if label in ("p2", "p3"):
            return "reject"
        if label in ("p4", "p5"):
            return "side_study"
    if dataset == "zipfian":
        if label == "p2":
            return "reject"
        if label == "p4":
            return "side_study"
    return "mainline"

joined["artifact_class"] = joined.apply(
    lambda r: classify_artifact(r["dataset"], r["artifact_label"]), axis=1
)

# ── 6. Epsilon-kstar marking ──────────────────────────────────
# Filter epsilon_to_kstar to epsilon=0 and epsilon=1e-3
eps_focus = eps_df[eps_df["epsilon"].isin([0.0, 0.001])].copy()

# Find full-depth sweep rows (k = max_plane_count)
full_depth = sweep[sweep["max_filter_planes"] == sweep["max_plane_count"]].copy()

# Match eps_focus to full_depth sweep rows by (dataset, artifact_label, approximate selectivity)
kstar_records = []
for _, ek in eps_focus.iterrows():
    mask = (
        (full_depth["dataset"] == ek["dataset"]) &
        (full_depth["artifact_label"] == ek["artifact_label"])
    )
    candidates = full_depth[mask].copy()
    candidates["sel_diff"] = np.abs(candidates["selectivity"] - ek["selectivity"])
    candidates = candidates[candidates["sel_diff"] <= 1e-6]
    if len(candidates) == 0:
        print(f"WARNING: No full-depth sweep row matching eps_kstar: "
              f"dataset={ek['dataset']} artifact={ek['artifact_label']} "
              f"eps_sel={ek['selectivity']}")
        continue
    candidates = candidates.sort_values("sel_diff")
    best = candidates.iloc[0]
    kstar_records.append({
        "dataset": ek["dataset"],
        "artifact_label": ek["artifact_label"],
        "threshold": best["threshold"],
        "epsilon": ek["epsilon"],
        "kstar": ek["kstar"]
    })

kstar_df = pd.DataFrame(kstar_records)

# Pivot to get epsilon_0_kstar and epsilon_1e3_kstar per (dataset, artifact_label, threshold)
if len(kstar_df) > 0:
    kstar_pivot = kstar_df.pivot_table(
        index=["dataset", "artifact_label", "threshold"],
        columns="epsilon",
        values="kstar",
        aggfunc="first"
    ).reset_index()
    kstar_pivot.columns = ["dataset", "artifact_label", "threshold", "kstar_0", "kstar_1e3"]
else:
    kstar_pivot = pd.DataFrame(columns=["dataset", "artifact_label", "threshold", "kstar_0", "kstar_1e3"])

joined = joined.merge(kstar_pivot, on=["dataset", "artifact_label", "threshold"], how="left")

# Determine epsilon_marker
def get_epsilon_marker(row):
    k = row["max_filter_planes"]
    k0 = row.get("kstar_0")
    k1 = row.get("kstar_1e3")
    if pd.notna(k0) and k == int(k0):
        return "epsilon_0_kstar"
    if pd.notna(k1) and k == int(k1):
        return "epsilon_1e-3_kstar"
    return "none"

joined["epsilon_marker"] = joined.apply(get_epsilon_marker, axis=1)

# ── 7. Full-depth count for execution_rel_error_bound ──────────
# Self-join: for each row, get the gpu_count from the full-depth (max k) row
full_depth_counts = sweep[sweep["max_filter_planes"] == sweep["max_plane_count"]][
    ["dataset", "artifact_label", "threshold", "gpu_count"]
].rename(columns={"gpu_count": "full_depth_count"})

joined = joined.merge(full_depth_counts, on=["dataset", "artifact_label", "threshold"], how="left")

# ── 8. Derived fields ─────────────────────────────────────────
joined["estimated_pack_load_bytes_per_row"] = (
    joined["estimated_pack_load_bytes"] / joined["n"].replace(0, np.nan)
)
joined["estimated_pack_load_bytes_vs_raw_ratio"] = (
    joined["estimated_pack_load_bytes"] / (joined["n"] * 8).replace(0, np.nan)
)
joined["estimated_physical_GBps_proxy"] = joined["estimated_physical_GBps"]
joined["U_k"] = joined["uncertain"]
joined["execution_rel_error_bound"] = (
    joined["U_k"] / joined["full_depth_count"].fillna(1).clip(lower=1)
)
joined["transfer_speedup_vs_raw"] = (
    joined["cudaMemcpy_ms_raw"] / joined["cudaMemcpy_ms"].replace(0, np.nan)
)

# ── 9. Traffic claim level ────────────────────────────────────
def get_traffic_claim_level(row):
    has_h2d = pd.notna(row.get("transfer_bytes")) and pd.notna(row.get("cudaMemcpy_ms"))
    has_kernel_proxy = pd.notna(row.get("estimated_pack_load_bytes"))
    if has_h2d:
        return "h2d_measured"
    if has_kernel_proxy:
        return "kernel_proxy_only"
    return "ncu_required"

def get_traffic_claim_reason(row):
    level = row["traffic_claim_level"]
    if level == "h2d_measured":
        return "H2D transfer bytes and cudaMemcpy timing are measured via CUDA events; kernel traffic fields are proxy-only"
    if level == "kernel_proxy_only":
        return "Kernel-level estimated pack-load bytes; no NCU profiler data available"
    return "No H2D or kernel proxy data available; NCU profiling required"

joined["traffic_claim_level"] = joined.apply(get_traffic_claim_level, axis=1)
joined["traffic_claim_reason"] = joined.apply(get_traffic_claim_reason, axis=1)

# ── 10. Select output columns ─────────────────────────────────
out_cols = [
    "dataset",
    "artifact_label",
    "artifact_class",
    "target_selectivity",
    "threshold",
    "max_filter_planes",
    "epsilon_marker",
    "n",
    "ms_per_iter",
    "rows_per_sec",
    "max_plane_count",
    "avg_planes_read_per_total_row",
    "logical_bytes",
    "estimated_pack_load_bytes",
    "estimated_pack_load_bytes_per_row",
    "estimated_pack_load_bytes_vs_raw_ratio",
    "logical_GBps",
    "estimated_physical_GBps_proxy",
    "U_k",
    "execution_rel_error_bound",
    "bytes_per_row",
    "transfer_bytes",
    "cudaMemcpy_ms",
    "transfer_bytes_raw",
    "cudaMemcpy_ms_raw",
    "transfer_speedup_vs_raw",
    "gpu_eq_cpu_encoded",
    "load_strategy",
    "kernel_path",
    "traffic_claim_level",
    "traffic_claim_reason",
]

# Rename columns to match spec
col_map = {
    "max_filter_planes": "k",
    "target_selectivity": "target_selectivity_pct",
    "bytes_per_row": "bytes_per_row_artifact",
    "transfer_bytes": "artifact_transfer_bytes",
    "cudaMemcpy_ms": "artifact_cudaMemcpy_ms",
    "transfer_bytes_raw": "raw_transfer_bytes",
    "cudaMemcpy_ms_raw": "raw_cudaMemcpy_ms",
}

result = joined[out_cols].rename(columns=col_map)

# Ensure all required columns present
required = [
    "dataset", "artifact_label", "artifact_class", "target_selectivity_pct",
    "threshold", "k", "epsilon_marker", "n", "ms_per_iter", "rows_per_sec",
    "max_plane_count", "avg_planes_read_per_total_row", "logical_bytes",
    "estimated_pack_load_bytes", "estimated_pack_load_bytes_per_row",
    "estimated_pack_load_bytes_vs_raw_ratio", "logical_GBps",
    "estimated_physical_GBps_proxy", "U_k", "execution_rel_error_bound",
    "bytes_per_row_artifact", "artifact_transfer_bytes", "artifact_cudaMemcpy_ms",
    "raw_transfer_bytes", "raw_cudaMemcpy_ms", "transfer_speedup_vs_raw",
    "gpu_eq_cpu_encoded", "load_strategy", "kernel_path",
    "traffic_claim_level", "traffic_claim_reason"
]

missing_cols = [c for c in required if c not in result.columns]
if missing_cols:
    raise ValueError(f"Missing required output columns: {missing_cols}")

print(f"\nOutput rows: {len(result)}")
print(f"Unique artifacts: {result[['dataset', 'artifact_label']].drop_duplicates().shape[0]}")
print(f"Kstar rows (epsilon=0): {(result['epsilon_marker'] == 'epsilon_0_kstar').sum()}")
print(f"Kstar rows (epsilon=1e-3): {(result['epsilon_marker'] == 'epsilon_1e-3_kstar').sum()}")

# ── 11. Write CSV ──────────────────────────────────────────────
out_path = BASE / "results/paper_v1/exp4_b1_physical_traffic_proxy.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(out_path, index=False)
print(f"\nWrote: {out_path}")
