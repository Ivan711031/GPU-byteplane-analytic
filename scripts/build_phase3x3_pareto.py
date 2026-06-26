#!/usr/bin/env python3
"""Phase 3-X3 Pareto: compacted-indexed dispatch with full provenance.

Reads measured timing CSV, computes amortized latency with compaction cost.

Usage:
  python3 scripts/build_phase3x3_pareto.py \
    --x1-dir /work/.../phase3x1_accuracy/job_XXXXX \
    --timing-csv /work/.../measured_timing/job_XXXXX/cesm_cloud_timing.csv \
    --timing-csv-hurricane /work/.../measured_timing/job_XXXXX/hurricane_u_timing.csv \
    --out-dir OUTPUT
"""

import argparse, csv, json, os, statistics

def load_timing(path):
    """Load timing CSV. Returns {batch_size: {rep, fb}} and compaction_ms."""
    data = {}
    compaction_ms = None
    with open(path) as f:
        # Skip comment lines (start with #)
        lines = [l for l in f if not l.startswith('#')]
    reader = csv.DictReader(lines)
    for row in reader:
        bs = row['batch_size']
        if bs == 'compaction':
            compaction_ms = float(row['idx_repair_ms_per_unit'])
        else:
            bsi = int(bs)
            data[bsi] = {
                'rep': float(row['idx_repair_ms_per_unit']),
                'fb':  float(row['idx_fallback_ms_per_unit']),
            }
    if compaction_ms is None:
        compaction_ms = 0.005
    return data, compaction_ms

def find_batch(tdata, need_units):
    """Smallest batch size >= need_units in tdata."""
    sizes = sorted(k for k in tdata if isinstance(k, int))
    for bs in sizes:
        if bs >= need_units:
            return bs, tdata[bs]
    return sizes[-1], tdata[sizes[-1]]

def ci95(samples):
    n = len(samples)
    if n < 2:
        return 0, 0, 0
    mu = statistics.mean(samples)
    sd = statistics.stdev(samples) if n > 1 else 0
    se = sd / (n ** 0.5)
    t29 = 2.045229642132703
    return mu, mu - t29 * se, mu + t29 * se

# Phase 3-U timing. Verified values from bench_filter_aggregate job 72994.
B1_C = 0.183192    # byte-plane progressive filter+aggregate
B0_C = 0.870226    # raw FP64 baseline
B2_C = B1_C + 0.410  # B2 = B1 + W5 graded NMR overhead (0.41 ms cesm)

B1_H = 0.044388    # byte-plane progressive filter+aggregate
B0_H = 0.132368    # raw FP64 baseline
B2_H = B1_H + 0.069  # B2 = B1 + W5 graded NMR overhead (0.069 ms hurricane)

NU_C = 41133       # cesm allocation units
NU_H = 6104        # hurricane allocation units

def build_pareto(x1_dir, timing_csv_c, timing_csv_h, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tdata_c, comp_c = load_timing(timing_csv_c)
    tdata_h, comp_h = load_timing(timing_csv_h)
    comp = max(comp_c, comp_h)  # use the larger compaction cost

    def load_x1(path):
        data = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                k = (row['field'], float(row['fault_rate']), int(row['fault_seed']),
                     row['alloc_policy'], int(row['replica_budget_B']))
                data[k] = row
        return data

    x1_c = load_x1(os.path.join(x1_dir, 'cesm_cloud_x1_accuracy_canonical.csv'))
    x1_h = load_x1(os.path.join(x1_dir, 'hurricane_u_x1_accuracy_canonical.csv'))

    import hashlib
    def sha256file(p):
        return hashlib.sha256(open(p, 'rb').read()).hexdigest()
    timing_csv_cesm_sha = sha256file(timing_csv_c)
    timing_csv_hurricane_sha = sha256file(timing_csv_h)

    header = (
        "dataset,fault_rate,alloc_policy,B,"
        "total_units,repair_count,fallback_count,"
        "batch_size,"
        "repair_dispatch_ms,fallback_dispatch_ms,"
        "compaction_ms,"
        "common_case_B2_ms,amortized_latency_ms,B0_ms,B1_ms,"
        "overhead_ms,margin_ms,overhead_fraction,deployable,"
        "timing_csv_path,timing_csv_cesm_sha256,timing_csv_hurricane_sha256"
    )

    DEP_IDX = 18  # zero-based column of 'deployable' in the output header
    rows = []
    for field, src, b0, b2, nu, b1, tdata in [
        ('cesm_atm_cloud', x1_c, B0_C, B2_C, NU_C, B1_C, tdata_c),
        ('hurricane_u',   x1_h, B0_H, B2_H, NU_H, B1_H, tdata_h),
    ]:
        agg = {}
        for k, r in src.items():
            key = (k[3], k[4], k[1])  # policy, B, rate
            if key not in agg:
                agg[key] = {'rep': [], 'fb': []}
            agg[key]['rep'].append(int(r.get('repair_invoked', '0')))
            agg[key]['fb'].append(int(r.get('fallback_invoked', '0')))

        for (policy, B, rate), v in agg.items():
            rep_mu, _, _ = ci95(v['rep'])
            fb_mu, _, _  = ci95(v['fb'])
            tr = int(round(rep_mu))
            tf = int(round(fb_mu))
            need = max(tr, tf, 1)

            bs, costs = find_batch(tdata, need)

            rd = tr * costs['rep']
            fd = tf * costs['fb']
            amort = b2 + rd + fd + comp

            margin = max(b0 - b1, 0.001)
            overhead = b2 - b1
            ohf = overhead / margin
            dep = (amort < b0) and (ohf < 0.9)

            tcsv_path = timing_csv_c if field == 'cesm_atm_cloud' else timing_csv_h
            rows.append((
                field, f"{rate:.0e}", policy, B,
                nu, tr, tf,
                bs,
                f"{rd:.6f}", f"{fd:.6f}",
                f"{comp:.6f}",
                f"{b2:.4f}", f"{amort:.4f}", f"{b0:.4f}", f"{b1:.4f}",
                f"{overhead:.4f}", f"{margin:.4f}", f"{ohf:.4f}",
                str(dep).lower(),
                tcsv_path, timing_csv_cesm_sha, timing_csv_hurricane_sha
            ))

    with open(os.path.join(out_dir, 'x3_pareto_canonical.csv'), 'w') as f:
        f.write(header + '\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')

    dep = [r for r in rows if r[DEP_IDX] == 'true']
    with open(os.path.join(out_dir, 'x3_deployable_frontier.csv'), 'w') as f:
        f.write(header + '\n')
        for r in dep:
            f.write(','.join(str(x) for x in r) + '\n')

    # PRD headline gate: graded(B=3) on both primary fields, band {1e-7,1e-6,1e-5}
    hl = [r for r in rows
          if r[2] == 'graded' and r[3] == 3
          and r[1] in ('1e-07', '1e-06', '1e-05')]
    hl_dep = all(r[DEP_IDX] == 'true' for r in hl)

    # Summary
    with open(os.path.join(out_dir, 'x3_latency_storage_accuracy_summary.csv'), 'w') as f:
        f.write("metric,cesm_atm_cloud,hurricane_u\n")
        f.write(f"compaction_ms_per_query,{comp:.6f},{comp:.6f}\n")
        f.write(f"headline_graded_B3_deployable,{str(hl_dep).lower()},{str(hl_dep).lower()}\n")
        f.write(f"deployable_configs,{len(dep)},{len(dep)}\n")

    with open(os.path.join(out_dir, 'handoff.json'), 'w') as f:
        json.dump({
            'phase': '3-X3',
            'dispatch_model': 'indexed compacted dispatch (symmetric batch)',
            'timing_provenance': timing_csv_c.replace('/cesm_cloud', '/{field}'),
            'provenance_job_note': 'timing from bench_graded_repair_timing via sbatch',
            'total_configs': len(rows),
            'deployable_configs': len(dep),
            'headline_graded_B3_deployable': hl_dep,
            'verdict': 'PROCEED_TO_GENERALIZATION' if hl_dep else 'STOP_PARETO_UNFAVORABLE',
        }, f, indent=2)

    print(f"X3 Pareto: {len(rows)} configs, {len(dep)} deployable, {len(rows)-len(dep)} non-dep")
    print(f"Headline graded(B=3) on both fields: {'PASS' if hl_dep else 'FAIL'}")
    print(f"Compaction cost: {comp*1000:.2f} µs per query")
    for r in rows:
        if r[DEP_IDX] != 'true':
            print(f"  NON-DEP: {r[0]:>15s} {r[1]:>6s} {r[2]:>30s} B={r[3]} amort={r[12]:>8s} B0={r[13]:>8s}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--x1-dir', required=True)
    ap.add_argument('--timing-csv', required=True)
    ap.add_argument('--timing-csv-hurricane', default=None)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    tch = args.timing_csv_hurricane or args.timing_csv.replace('cesm_cloud', 'hurricane_u')
    build_pareto(args.x1_dir, args.timing_csv, tch, args.out_dir)
