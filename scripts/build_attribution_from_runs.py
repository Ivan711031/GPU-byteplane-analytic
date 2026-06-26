#!/usr/bin/env python3
"""
Post-process individual benchmark CSVs from the attribution run into:
1. all_runs.csv (correctly parsed)
2. attribution_raw_fused_vs_byteplane_fused.csv
3. Print summary table
"""
import csv
import glob
import os
import sys

def main():
    result_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not result_dir:
        # Find latest attribution dir
        base = "results/exp4_filter_aggregate"
        dirs = glob.glob(f"{base}/attribution_*_job57433*/")
        if not dirs:
            dirs = sorted(glob.glob(f"{base}/attribution_*/"))
        if not dirs:
            print("No attribution dirs found")
            sys.exit(1)
        result_dir = dirs[-1]
    
    print(f"Processing: {result_dir}")
    
    # Find all run_*.csv files
    run_files = sorted(glob.glob(os.path.join(result_dir, "run_*.csv")))
    print(f"Found {len(run_files)} run files")
    
    if not run_files:
        print("No run files found")
        sys.exit(1)
    
    # Parse each run CSV
    rows = []
    for fpath in run_files:
        with open(fpath) as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    
    print(f"Parsed {len(rows)} data rows")
    
    # Build all_runs.csv
    all_runs_path = os.path.join(result_dir, "all_runs.csv")
    all_fields = [
        'dataset', 'selectivity_pct', 'threshold', 'k', 'k_star_k', 'max_planes',
        'ms_per_iter_fused', 'raw_baseline_ms_per_iter',
        'gpu_count', 'enc_count', 'count_ok', 'validated', 'iters', 'warmup'
    ]
    
    with open(all_runs_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        
        for r in rows:
            dataset = r.get('dataset', '')
            selectivity = float(r.get('selectivity', '0'))
            selectivity_pct = int(round(selectivity * 100))
            threshold = float(r.get('threshold', '0'))
            n = int(r.get('n', '0'))
            iters = int(r.get('iters', '200'))
            warmup = int(r.get('warmup', '20'))
            fused_ms = float(r.get('ms_per_iter', '0'))
            raw_ms = float(r.get('raw_baseline_ms_per_iter', '0'))
            gpu_count = int(r.get('gpu_count', '0'))
            enc_count = int(r.get('cpu_enc_count', '0'))
            validated = r.get('validated', 'false')
            count_ok = (gpu_count == enc_count)
            
            # Extract k from max_filter_planes
            k_str = r.get('max_filter_planes', '')
            try:
                k = int(k_str)
            except ValueError:
                k = 0
            
            max_planes = r.get('max_planes_read', '')  # This field exists
            
            # Determine k* status and max_planes from dataset metadata
            k_star_map = {'sensor': 1, 'uniform': 1, 'heavy_tailed': 4, 'zipfian': 3}
            max_p_map = {'sensor': 5, 'uniform': 6, 'heavy_tailed': 8, 'zipfian': 8}
            k_star = k_star_map.get(dataset, 1)
            max_p = max_p_map.get(dataset, 8)
            
            is_k_star = (k == k_star)
            
            writer.writerow({
                'dataset': dataset,
                'selectivity_pct': selectivity_pct,
                'threshold': threshold,
                'k': k,
                'k_star_k': 'true' if is_k_star else 'false',
                'max_planes': max_p,
                'ms_per_iter_fused': fused_ms,
                'raw_baseline_ms_per_iter': raw_ms,
                'gpu_count': gpu_count,
                'enc_count': enc_count,
                'count_ok': str(count_ok),
                'validated': validated,
                'iters': iters,
                'warmup': warmup
            })
    
    print(f"Written: {all_runs_path}")
    
    # Build attribution CSV
    # cuDF baselines (masked_reduce, gpu_hot_path at s≈50%)
    cudf_baseline = {
        ('sensor', 50): 4.643,
        ('uniform', 50): 4.768,
        ('heavy_tailed', 50): 4.767,
        ('zipfian', 50): 4.764,
    }
    
    attr_path = os.path.join(result_dir, "attribution_raw_fused_vs_byteplane_fused.csv")
    attr_fields = [
        'dataset', 'selectivity_pct', 'threshold', 'k', 'is_k_star', 'max_planes',
        'fused_ms', 'raw_fused_ms', 'cudf_ms',
        'speedup_fused_vs_cudf', 'speedup_raw_vs_cudf',
        'speedup_bp_full_vs_raw', 'speedup_bp_kstar_vs_raw',
        'count_ok'
    ]
    
    attr_rows = []
    for r in rows:
        dataset = r.get('dataset', '')
        selectivity = float(r.get('selectivity', '0'))
        selectivity_pct = int(round(selectivity * 100))
        threshold = float(r.get('threshold', '0'))
        fused_ms = float(r.get('ms_per_iter', '0'))
        raw_ms = float(r.get('raw_baseline_ms_per_iter', '0'))
        gpu_count = int(r.get('gpu_count', '0'))
        enc_count = int(r.get('cpu_enc_count', '0'))
        count_ok = (gpu_count == enc_count)
        
        k_str = r.get('max_filter_planes', '')
        try:
            k = int(k_str)
        except ValueError:
            k = 0
        
        k_star_map = {'sensor': 1, 'uniform': 1, 'heavy_tailed': 4, 'zipfian': 3}
        max_p_map = {'sensor': 5, 'uniform': 6, 'heavy_tailed': 8, 'zipfian': 8}
        k_star = k_star_map.get(dataset, 1)
        max_p = max_p_map.get(dataset, 8)
        is_k_star = (k == k_star)
        
        # cuDF speedup
        cudf_key = (dataset, selectivity_pct)
        if cudf_key in cudf_baseline:
            cudf_ms = cudf_baseline[cudf_key]
            speedup_fused_vs_cudf = cudf_ms / fused_ms if fused_ms > 0 else float('nan')
            speedup_raw_vs_cudf = cudf_ms / raw_ms if raw_ms > 0 else float('nan')
        else:
            cudf_ms = float('nan')
            speedup_fused_vs_cudf = float('nan')
            speedup_raw_vs_cudf = float('nan')
        
        # Byteplane vs raw
        speedup_bp_full_vs_raw = float('nan')
        if k == max_p:
            speedup_bp_full_vs_raw = raw_ms / fused_ms if fused_ms > 0 else float('nan')
        
        speedup_bp_kstar_vs_raw = float('nan')
        if is_k_star:
            speedup_bp_kstar_vs_raw = raw_ms / fused_ms if fused_ms > 0 else float('nan')
        
        row = {
            'dataset': dataset,
            'selectivity_pct': selectivity_pct,
            'threshold': threshold,
            'k': k,
            'is_k_star': str(is_k_star),
            'max_planes': max_p,
            'fused_ms': fused_ms,
            'raw_fused_ms': raw_ms,
            'cudf_ms': cudf_ms,
            'speedup_fused_vs_cudf': speedup_fused_vs_cudf,
            'speedup_raw_vs_cudf': speedup_raw_vs_cudf,
            'speedup_bp_full_vs_raw': speedup_bp_full_vs_raw,
            'speedup_bp_kstar_vs_raw': speedup_bp_kstar_vs_raw,
            'count_ok': str(count_ok),
        }
        attr_rows.append(row)
    
    with open(attr_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=attr_fields)
        writer.writeheader()
        writer.writerows(attr_rows)
    
    print(f"Written: {attr_path}")
    
    # Print summary table
    print()
    print("=" * 130)
    print("ATTRIBUTION TABLE: Raw-Fused vs Byteplane-Fused Filter+Aggregate")
    print("=" * 130)
    header = f"{'Dataset':<14} {'Sel':>3} {'k':>2} {'k*':>3} {'max':>4} {'Fused(ms)':>10} {'Raw(ms)':>10} {'cuDF(ms)':>10} {'Fus/cuDF':>9} {'Raw/cuDF':>9} {'BP_full/raw':>11} {'BP_k*/raw':>10} {'CountOK':>8}"
    print(header)
    print("-" * 130)
    
    for a in sorted(attr_rows, key=lambda x: (x['dataset'], x['selectivity_pct'], x['k'])):
        ds = a['dataset']
        sel = a['selectivity_pct']
        k = a['k']
        is_ks = '✓' if a['is_k_star'] == 'True' else ' '
        mp = a['max_planes']
        fms = a['fused_ms']
        rms = a['raw_fused_ms']
        
        def fmt(val, fmt_str=".4f"):
            if isinstance(val, float) and (val != val):  # nan check
                return "N/A"
            return f"{val:{fmt_str}}"
        
        cudf_s = fmt(a['cudf_ms'], '.3f')
        fc_s = fmt(a['speedup_fused_vs_cudf'], '.2f') + 'x' if not (isinstance(a['speedup_fused_vs_cudf'], float) and a['speedup_fused_vs_cudf'] != a['speedup_fused_vs_cudf']) else "N/A"
        rc_s = fmt(a['speedup_raw_vs_cudf'], '.2f') + 'x' if not (isinstance(a['speedup_raw_vs_cudf'], float) and a['speedup_raw_vs_cudf'] != a['speedup_raw_vs_cudf']) else "N/A"
        bpf_s = fmt(a['speedup_bp_full_vs_raw'], '.3f') + 'x' if not (isinstance(a['speedup_bp_full_vs_raw'], float) and a['speedup_bp_full_vs_raw'] != a['speedup_bp_full_vs_raw']) else "N/A"
        bpk_s = fmt(a['speedup_bp_kstar_vs_raw'], '.3f') + 'x' if not (isinstance(a['speedup_bp_kstar_vs_raw'], float) and a['speedup_bp_kstar_vs_raw'] != a['speedup_bp_kstar_vs_raw']) else "N/A"
        cok = '✓' if a['count_ok'] == 'True' else '✗'
        
        print(f"{ds:<14} {sel:>3} {k:>2} {is_ks:>3} {mp:>4} {fms:>10.6f} {rms:>10.6f} {cudf_s:>10} {fc_s:>9} {rc_s:>9} {bpf_s:>11} {bpk_s:>10} {cok:>8}")
    
    print("=" * 130)
    
    # Key findings
    print()
    print("KEY FINDINGS (at s≈50%)")
    print("-" * 80)
    
    s50_rows = [a for a in attr_rows if a['selectivity_pct'] == 50]
    for a in sorted(s50_rows, key=lambda x: x['dataset']):
        ds = a['dataset']
        k = a['k']
        fms = a['fused_ms']
        rms = a['raw_fused_ms']
        cudf_ms = a['cudf_ms']
        is_full = (k == a['max_planes'])
        is_ks = a['is_k_star'] == 'True'
        
        fusion_gain = cudf_ms / rms if cudf_ms == cudf_ms else float('nan')
        bp_gain = rms / fms
        
        if is_full:
            print(f"  {ds} s=50% k=max={k}: fused={fms:.6f}ms raw={rms:.6f}ms cuDF={cudf_ms if cudf_ms==cudf_ms else 'N/A'}ms")
            bg_str = f"{bp_gain:.3f}x" if bp_gain == bp_gain else "N/A"
            fg_str = f"{fusion_gain:.2f}x" if fusion_gain == fusion_gain else "N/A"
            print(f"    BP_full/raw = {bg_str}  |  Fusion-only (cuDF→raw) = {fg_str}")
        if is_ks:
            bp_ks_gain = rms / fms
            bg_ks_str = f"{bp_ks_gain:.3f}x" if bp_ks_gain == bp_ks_gain else "N/A"
            print(f"    BP_k*={k}/raw = {bg_ks_str}")
        if k == 1 and not is_ks:
            bp_k1_gain = rms / fms
            print(f"    BP_k=1/raw = {bp_k1_gain:.3f}x")
    
    print()
    print(f"Attribution CSV: {attr_path}")
    print(f"All runs CSV: {all_runs_path}")

if __name__ == '__main__':
    main()
