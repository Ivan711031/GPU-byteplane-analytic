#!/usr/bin/env python3
import os
import subprocess
import numpy as np
import pandas as pd
import math
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_main_workspace_dir(root_dir):
    if ".worktrees" in root_dir:
        parts = root_dir.split(".worktrees")
        return parts[0].rstrip("/")
    return root_dir

MAIN_DIR = get_main_workspace_dir(ROOT_DIR)

# Datasets definition
DATASETS = {
    "cesm_atm_q": {
        "dataset_path": os.path.join(ROOT_DIR, "datasets/scientific/dev_buff_v2_scientific_quasi_global/cesm_atm_q/bfp-dec12"),
        "raw_path": os.path.join(ROOT_DIR, "results/buff_encoder_v2/raw_scientific/cesm_atm_q.f64le.bin"),
        "max_k": 6,
        "ref_csv": os.path.join(MAIN_DIR, "results/exp4_filter_aggregate/scientific_extra_fields_quasi_global_58440/scientific_extra_fields_quasi_global.csv"),
        "paper_headline_compatible": "yes",
        "artifact_locality": "true_quasi_global",
        "segment_count": 1
    },
    "cesm_atm_cloud": {
        "dataset_path": os.path.join(ROOT_DIR, "datasets/locality_sensitivity/cesm_atm_cloud/seg_global"),
        "raw_path": os.path.join(ROOT_DIR, "results/buff_encoder_v2/raw_scientific/cesm_atm_cloud.f64le.bin"),
        "max_k": 7,
        "ref_csv": os.path.join(MAIN_DIR, "results/locality_sensitivity/sum_sweep.csv"),
        "paper_headline_compatible": "yes",
        "artifact_locality": "true_quasi_global",
        "segment_count": 1
    },
    "hurricane_u": {
        "dataset_path": os.path.join(ROOT_DIR, "datasets/locality_sensitivity/hurricane_u/seg_global"),
        "raw_path": os.path.join(ROOT_DIR, "results/buff_encoder_v2/raw_scientific/hurricane_u.f64le.bin"),
        "max_k": 8,
        "ref_csv": os.path.join(MAIN_DIR, "results/locality_sensitivity/sum_sweep.csv"),
        "paper_headline_compatible": "yes",
        "artifact_locality": "true_quasi_global",
        "segment_count": 1
    },
    "hurricane_tc": {
        "dataset_path": os.path.join(ROOT_DIR, "datasets/scientific/dev_buff_v2_scientific_quasi_global/hurricane_tc/bfp-dec11"),
        "raw_path": os.path.join(ROOT_DIR, "results/buff_encoder_v2/raw_scientific/hurricane_tc.f64le.bin"),
        "max_k": 6,
        "ref_csv": os.path.join(MAIN_DIR, "results/exp4_filter_aggregate/scientific_extra_fields_quasi_global_58440/scientific_extra_fields_quasi_global.csv"),
        "paper_headline_compatible": "yes",
        "artifact_locality": "true_quasi_global",
        "segment_count": 1
    },
    "cesm_atm_cloud_local": {
        "dataset_path": os.path.join(ROOT_DIR, "datasets/scientific/dev_buff_v2_scientific/cesm_atm_cloud/bfp-dec12"),
        "raw_path": os.path.join(ROOT_DIR, "results/buff_encoder_v2/raw_scientific/cesm_atm_cloud.f64le.bin"),
        "max_k": 7,
        "ref_csv": os.path.join(MAIN_DIR, "results/exp4_filter_aggregate/scientific_sweep_58825/sweep_results.csv"),
        "paper_headline_compatible": "no",
        "artifact_locality": "local",
        "segment_count": 41134
    },
    "hurricane_u_local": {
        "dataset_path": os.path.join(ROOT_DIR, "datasets/scientific/dev_buff_v2_scientific/hurricane_u/bfp-dec12"),
        "raw_path": os.path.join(ROOT_DIR, "results/buff_encoder_v2/raw_scientific/hurricane_u.f64le.bin"),
        "max_k": 6,
        "ref_csv": os.path.join(MAIN_DIR, "results/exp4_filter_aggregate/scientific_sweep_58825/sweep_results.csv"),
        "paper_headline_compatible": "no",
        "artifact_locality": "local",
        "segment_count": 6105
    }
}

SELECTIVITIES = [10, 50, 90]

MODES = {
    "generic": {
        "bin": os.path.join(ROOT_DIR, "build_filter_aggregate/buff_cpu_proxy"),
        "csv": os.path.join(ROOT_DIR, "results/buff_cpu_proxy/buff_cpu_proxy_sweep.csv")
    },
    "native": {
        "bin": os.path.join(ROOT_DIR, "build_filter_aggregate/buff_cpu_proxy_native"),
        "csv": os.path.join(ROOT_DIR, "results/buff_cpu_proxy/buff_cpu_proxy_sweep_native.csv")
    }
}

HEADLINE_DATASETS = ["cesm_atm_q", "cesm_atm_cloud", "hurricane_u", "hurricane_tc"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated list of datasets to run")
    parser.add_argument("--replace-output", action="store_true", help="Overwrite existing CSV outputs instead of preserving older rows")
    args = parser.parse_args()

    if args.datasets:
        target_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        for d in target_datasets:
            if d not in DATASETS:
                print(f"Error: Unknown dataset '{d}'")
                exit(1)
    else:
        target_datasets = HEADLINE_DATASETS

    global_validation_failed = False
    all_results = {}
    cpu_model = "unknown"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    partition = os.environ.get("SLURM_JOB_PARTITION", "unknown")
    job_tag = os.environ.get("SLURM_JOB_ID", str(os.getpid()))

    for mode_name, mode_config in MODES.items():
        bin_path = mode_config["bin"]
        output_csv = mode_config["csv"]
        raw_csv = output_csv.replace(".csv", f"_{job_tag}_raw.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        # Load existing results to merge unless the caller asked for a clean overwrite.
        if not args.replace_output and os.path.exists(output_csv):
            try:
                existing_df = pd.read_csv(output_csv)
                # Keep rows for datasets that are NOT in target_datasets
                keep_df = existing_df[~existing_df["dataset"].isin(target_datasets)].copy()
                
                # Make sure the kept rows have the metadata columns added by this sweep.
                for col in ["paper_headline_compatible", "artifact_locality", "segment_count", "build_type"]:
                    if col not in keep_df.columns:
                        if col == "build_type":
                            keep_df[col] = f"{mode_name}_scalar"
                        else:
                            keep_df[col] = keep_df["dataset"].map(lambda d: DATASETS.get(d, {}).get(col, "unknown"))
                
                existing_runs = keep_df.to_dict(orient="records")
                print(f"Loaded {len(existing_runs)} existing rows from {output_csv} to preserve.")
            except Exception as e:
                print(f"Warning: Could not read existing CSV {output_csv}: {e}")
                existing_runs = []
        else:
            existing_runs = []
            
        print("=" * 80)
        print(f"Starting BUFF-style CPU Progressive Proxy Sweep in MODE: {mode_name}")
        print(f"Binary: {bin_path}")
        print(f"Output will be saved/updated to: {output_csv}")
        print("=" * 80)
        
        runs = []
        
        for ds_name in target_datasets:
            config = DATASETS[ds_name]
            print("-" * 80)
            print(f"Dataset: {ds_name}")
            
            # Load raw data to compute quantiles
            if not os.path.exists(config["raw_path"]):
                print(f"Error: Raw file not found: {config['raw_path']}")
                global_validation_failed = True
                continue
            
            print(f"Loading raw file {config['raw_path']}...")
            raw_data = np.fromfile(config["raw_path"], dtype=np.float64)
            
            # Load reference CSV
            ref_df = None
            if os.path.exists(config["ref_csv"]):
                ref_df = pd.read_csv(config["ref_csv"], float_precision="round_trip")
                print(f"Loaded reference CSV: {config['ref_csv']} with {len(ref_df)} rows.")
            else:
                print(f"Warning: Reference CSV not found: {config['ref_csv']}")
                
            for sel in SELECTIVITIES:
                quantile = 1.0 - (sel / 100.0)
                # Compute default fallback threshold
                threshold = np.quantile(raw_data, quantile)
                
                # Attempt to extract threshold from ref_df
                if ref_df is not None:
                    ref_ds_name = ds_name[:-6] if ds_name.endswith("_local") else ds_name
                    df_ds = ref_df[ref_df["dataset"] == ref_ds_name]
                    if len(df_ds) > 0:
                        if "representation_label" in ref_df.columns:
                            rep_label = "quasi-global" if config["artifact_locality"] == "true_quasi_global" else "4096"
                            df_ds = df_ds[df_ds["representation_label"] == rep_label]
                            if len(df_ds) > 0:
                                if float(df_ds["selectivity"].max()) <= 1.0:
                                    target_sel = sel / 100.0
                                    df_sel = df_ds[np.isclose(df_ds["selectivity"], target_sel, atol=0.01)]
                                else:
                                    target_sel = 100 - sel
                                    df_sel = df_ds[np.isclose(df_ds["selectivity"], target_sel, atol=1.0)]
                                if len(df_sel) > 0:
                                    threshold = float(df_sel.iloc[0]["threshold"])
                        else:
                            target_frac = sel / 100.0
                            df_sel = df_ds[np.isclose(df_ds["selectivity"], target_frac, atol=0.01)]
                            if len(df_sel) > 0:
                                threshold = float(df_sel.iloc[0]["threshold"])
                            else:
                                df_sel = df_ds[np.isclose(df_ds["selectivity"], float(sel), atol=1.0)]
                                if len(df_sel) > 0:
                                    threshold = float(df_sel.iloc[0]["threshold"])
                                    
                threshold_arg = repr(float(threshold))
                print(f"  Selectivity: s{sel} (Quantile: {quantile:.2f}) -> Threshold: {threshold_arg}")
                
                for k in [2, config["max_k"]]:
                    print(f"    Running k={k}...")
                    
                    # Execute CPU proxy binary (writing to raw_csv)
                    cmd = [
                        bin_path,
                        "--dataset", config["dataset_path"],
                        "--raw", config["raw_path"],
                        "--threshold", threshold_arg,
                        "--selectivity", f"s{sel}",
                        "--max-filter-planes", str(k),
                        "--iters", "100",
                        "--warmup", "20",
                        "--csv", raw_csv
                    ]
                    
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.returncode != 0:
                        print(f"      ERROR: binary failed with exit code {res.returncode}")
                        print(res.stderr)
                        global_validation_failed = True
                        continue
                    
                    # Parse output for count and sum
                    count = None
                    val_sum = None
                    median_ms = None
                    
                    for line in res.stdout.splitlines():
                        if "Result:" in line:
                            parts = line.split()
                            for p in parts:
                                if p.startswith("count="):
                                    count = int(p.split("=")[1])
                                elif p.startswith("sum="):
                                    val_sum = float(p.split("=")[1])
                        elif "Timing (ms):" in line:
                            parts = line.split()
                            for p in parts:
                                if p.startswith("median="):
                                    median_ms = float(p.split("=")[1].replace(",", ""))
                                    
                    print(f"      Proxy Results: count={count}, sum={val_sum:.6f}, median_ms={median_ms:.3f}")
                    
                    # Match against reference for live feedback
                    val_status = "SKIP_NO_REF"
                    gpu_latency = 0.0
                    gpu_sum = 0.0
                    gpu_count = 0
                    precision = "UNKNOWN"
                    n_rows = 0
                    planes = 0
                     
                    if ref_df is not None:
                        # 1. Match dataset and k
                        ref_ds_name = ds_name[:-6] if ds_name.endswith("_local") else ds_name
                        plane_col = "max_filter_planes" if "max_filter_planes" in ref_df.columns else "k"
                        
                        ref_rows = ref_df[
                            (ref_df["dataset"] == ref_ds_name) & 
                            (ref_df[plane_col] == k)
                        ]
                        
                        # Apply representation label filter if present (locality_sensitivity sum_sweep.csv)
                        if "representation_label" in ref_df.columns:
                            rep_label = "quasi-global" if config["artifact_locality"] == "true_quasi_global" else "4096"
                            if rep_label in ref_rows["representation_label"].values:
                                ref_rows = ref_rows[ref_rows["representation_label"] == rep_label]
                        
                        if len(ref_rows) == 0:
                            print(f"      ERROR: No reference rows found for dataset={ref_ds_name}, k={k}")
                            global_validation_failed = True
                            continue
                        
                        target_sel_frac = sel / 100.0
                        threshold_tolerance = max(1e-12, abs(threshold) * 1e-6)
                        
                        matched_row = None
                        best_thresh_diff = float("inf")
                        inferred_by_threshold_only = False
                        
                        # 2. Match selectivity and threshold
                        if "selectivity" in ref_df.columns:
                            matching_rows_with_sel = []
                            for idx, row in ref_rows.iterrows():
                                ref_sel_frac = row["selectivity"]
                                if ref_sel_frac > 1.0:
                                    ref_sel_frac = ref_sel_frac / 100.0
                                
                                thresh_diff = abs(row["threshold"] - threshold)
                                
                                # Check normal selectivity or complement
                                sel_diff = abs(ref_sel_frac - target_sel_frac)
                                is_sel_match = (sel_diff <= 0.05)
                                
                                sel_diff_complement = abs((1.0 - ref_sel_frac) - target_sel_frac)
                                if sel_diff_complement <= 0.05:
                                    is_sel_match = True
                                    sel_diff = sel_diff_complement
                                
                                is_thresh_match = (thresh_diff <= threshold_tolerance) or math.isclose(row["threshold"], threshold, rel_tol=1e-6, abs_tol=1e-12)
                                
                                if is_thresh_match:
                                    matching_rows_with_sel.append((row, thresh_diff, sel_diff, is_sel_match))
                            
                            if len(matching_rows_with_sel) > 0:
                                strict_matches = [m for m in matching_rows_with_sel if m[3]]
                                if len(strict_matches) > 0:
                                    strict_matches.sort(key=lambda x: x[2])
                                    matched_row = strict_matches[0][0]
                                    best_thresh_diff = strict_matches[0][1]
                                else:
                                    matching_rows_with_sel.sort(key=lambda x: x[1])
                                    matched_row = matching_rows_with_sel[0][0]
                                    best_thresh_diff = matching_rows_with_sel[0][1]
                                    inferred_by_threshold_only = True
                            else:
                                best_diff = float("inf")
                                for idx, row in ref_rows.iterrows():
                                    d = abs(row["threshold"] - threshold)
                                    if d < best_diff:
                                        best_diff = d
                                print(f"      ERROR: Nearest threshold difference ({best_diff:.18f}) exceeds tolerance ({threshold_tolerance})")
                                global_validation_failed = True
                                continue
                        else:
                            # Match threshold only
                            matching_rows = []
                            for idx, row in ref_rows.iterrows():
                                thresh_diff = abs(row["threshold"] - threshold)
                                is_thresh_match = (thresh_diff <= threshold_tolerance) or math.isclose(row["threshold"], threshold, rel_tol=1e-6, abs_tol=1e-12)
                                if is_thresh_match:
                                    matching_rows.append((row, thresh_diff))
                            
                            if len(matching_rows) > 0:
                                matching_rows.sort(key=lambda x: x[1])
                                matched_row = matching_rows[0][0]
                                best_thresh_diff = matching_rows[0][1]
                                inferred_by_threshold_only = True
                            else:
                                best_diff = float("inf")
                                for idx, row in ref_rows.iterrows():
                                    d = abs(row["threshold"] - threshold)
                                    if d < best_diff:
                                        best_diff = d
                                print(f"      ERROR: Nearest threshold difference ({best_diff:.18f}) exceeds tolerance ({threshold_tolerance})")
                                global_validation_failed = True
                                continue
                                
                        if matched_row is not None:
                            ref_sel_str = f"{matched_row['selectivity']:.6f}" if "selectivity" in matched_row else "N/A"
                            print(f"      Matched reference row identity: dataset={matched_row['dataset']}, threshold={matched_row['threshold']:.18f}, selectivity={ref_sel_str}, gpu_count={matched_row['gpu_count']}, gpu_sum={matched_row['gpu_sum']:.6f}")
                            
                            if inferred_by_threshold_only:
                                print(f"      [INFO] Selectivity inferred only by nearest threshold (target: {target_sel_frac:.2f}, ref selectivity: {ref_sel_str})")
                                
                            gpu_count = int(matched_row["gpu_count"])
                            gpu_sum = float(matched_row["gpu_sum"])
                            gpu_latency = float(matched_row["ms_per_iter"])
                            
                            prec_mode = matched_row.get("precision_mode", "bounded")
                            prec_dec = matched_row.get("precision_decimals", 0)
                            if "representation_label" in ref_df.columns:
                                if "cloud" in ds_name:
                                    precision = "bounded (dec15)"
                                elif "hurricane_u" in ds_name:
                                    precision = "bounded (dec11)"
                                else:
                                    precision = "bounded (dec15)"
                            else:
                                precision = f"{prec_mode} (dec{prec_dec})"
                                
                            n_rows = int(matched_row.get("n", 168480000 if "cloud" in ds_name or "cesm" in ds_name else 25000000))
                            planes = int(ref_df[ref_df["dataset"] == ref_ds_name][plane_col].max())
                            
                            count_match = (count == gpu_count)
                            sum_match = math.isclose(val_sum, gpu_sum, rel_tol=1e-5, abs_tol=1e-5)
                            
                            if count_match and sum_match:
                                print(f"      [PASS] Matches GPU reference (count={gpu_count}, sum={gpu_sum:.6f})")
                                val_status = "PASS"
                            else:
                                print(f"      [FAIL] MISMATCH against GPU reference!")
                                print(f"        Expected: count={gpu_count}, sum={gpu_sum:.6f}")
                                print(f"        Got:      count={count}, sum={val_sum:.6f}")
                                val_status = "FAIL"
                                global_validation_failed = True
                    
                    speedup = median_ms / gpu_latency if gpu_latency > 0 else 0.0
                    
                    runs.append({
                        "dataset": ds_name,
                        "precision": precision,
                        "threshold": threshold,
                        "selectivity": f"s{sel}",
                        "k": k,
                        "rows": n_rows,
                        "planes": planes,
                        "cpu_warm_ms": median_ms,
                        "validation_status": val_status,
                        "reference_sum": gpu_sum,
                        "measured_sum": val_sum,
                        "gpu_latency_ms": gpu_latency,
                        "gpu_vs_cpu_speedup": speedup,
                        "build_type": f"{mode_name}_scalar",
                        "paper_headline_compatible": config["paper_headline_compatible"],
                        "artifact_locality": config["artifact_locality"],
                        "segment_count": config["segment_count"]
                    })
                    
        # Save finalized merged results
        final_runs = existing_runs + runs
        if final_runs:
            joined_df = pd.DataFrame(final_runs)
            cols_order = [
                "dataset", "precision", "threshold", "selectivity", "k", "rows", "planes",
                "cpu_warm_ms", "validation_status", "build_type", "reference_sum", "measured_sum",
                "gpu_latency_ms", "gpu_vs_cpu_speedup", "paper_headline_compatible",
                "artifact_locality", "segment_count"
            ]
            for col in cols_order:
                if col not in joined_df.columns:
                    joined_df[col] = None
            joined_df = joined_df[cols_order]
            joined_df.to_csv(output_csv, index=False)
            all_results[mode_name] = joined_df
            print(f"Saved comprehensive joined CSV for {mode_name} to {output_csv}")
            
    # Print comparative results
    if "generic" in all_results and "native" in all_results:
        df_gen = all_results["generic"]
        df_nat = all_results["native"]
        
        print("\n### BUFF CPU Baseline Sweep Comparison Table")
        print("| Dataset | Locality | Selectivity | k | GPU Latency (ms) | CPU Generic (ms) | CPU Native (ms) | Speedup (Gen vs GPU) | Speedup (Nat vs GPU) | Validation |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for i in range(len(df_gen)):
            r_gen = df_gen.iloc[i]
            r_nat_matches = df_nat[
                (df_nat["dataset"] == r_gen["dataset"]) & 
                (df_nat["selectivity"] == r_gen["selectivity"]) & 
                (df_nat["k"] == r_gen["k"])
            ]
            if len(r_nat_matches) > 0:
                r_nat = r_nat_matches.iloc[0]
                gpu_lat = r_gen["gpu_latency_ms"]
                ms_gen = r_gen["cpu_warm_ms"]
                ms_nat = r_nat["cpu_warm_ms"]
                
                sp_gen = ms_gen / gpu_lat if gpu_lat > 0 else 0.0
                sp_nat = ms_nat / gpu_lat if gpu_lat > 0 else 0.0
                
                print(f"| {r_gen['dataset']} | {r_gen['artifact_locality']} | {r_gen['selectivity']} | {r_gen['k']} | {gpu_lat:.3f} | {ms_gen:.3f} | {ms_nat:.3f} | {sp_gen:.2f}x | {sp_nat:.2f}x | {r_gen['validation_status']} |")

    if all_results:
        q7_rows = []
        for _, df in all_results.items():
            for _, row in df.iterrows():
                ds = row["dataset"]
                config = DATASETS.get(ds, {})
                if config.get("artifact_locality") != "true_quasi_global":
                    continue
                q7_rows.append({
                    "field": ds,
                    "artifact_path": config.get("dataset_path", ""),
                    "artifact_locality": row.get("artifact_locality", config.get("artifact_locality", "unknown")),
                    "segment_count": row.get("segment_count", config.get("segment_count", "")),
                    "paper_headline_compatible": row.get("paper_headline_compatible", config.get("paper_headline_compatible", "unknown")),
                    "selectivity": row["selectivity"],
                    "k": row["k"],
                    "build_type": row["build_type"],
                    "cpu_model": cpu_model,
                    "partition": partition,
                    "cpu_time_ms_median": row["cpu_warm_ms"],
                    "gpu_time_ms_median": row["gpu_latency_ms"],
                    "gpu_over_cpu_speedup": row["gpu_vs_cpu_speedup"],
                    "validation_pass": row["validation_status"] == "PASS",
                })
        if q7_rows:
            q7_csv = os.path.join(ROOT_DIR, "results/buff_cpu_proxy/buff_cpu_proxy_sweep_true_quasi_global.csv")
            pd.DataFrame(q7_rows).to_csv(q7_csv, index=False)
            print(f"Saved Q7 true quasi-global crossover CSV to {q7_csv}")

    if global_validation_failed:
        print("\nSweep validation FAILED!")
        exit(1)
    else:
        print("\nSweep validation PASSED successfully for all modes!")
        exit(0)

if __name__ == "__main__":
    main()
