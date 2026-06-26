#!/usr/bin/env python3
"""Re-aggregate individual attribution CSVs into master CSV.

Fixes the K_TYPE bash-in-f-string bug from the original sbatch script.
"""

import csv
import json
import os
import sys
from pathlib import Path

# Configuration
if len(sys.argv) > 1:
    RESULT_DIR = Path(sys.argv[1])
else:
    # Find the latest scientific_locality_attribution directory
    base = Path("results/exp4_filter_aggregate")
    dirs = sorted(base.glob("scientific_locality_attribution_*"))
    if not dirs:
        print("No attribution directories found")
        sys.exit(1)
    RESULT_DIR = dirs[-1]

LOCALITY_ROOT = Path("datasets/locality_sensitivity")

# Threshold prep times from Task 3 locality sweep (ms)
PREP_MS = {
    ("cesm_atm_cloud", "seg4096", 10): 10.41,
    ("cesm_atm_cloud", "seg4096", 50): 17.47,
    ("cesm_atm_cloud", "seg4096", 90): 19.42,
    ("cesm_atm_cloud", "seg65536", 10): 0.68,
    ("cesm_atm_cloud", "seg65536", 50): 1.10,
    ("cesm_atm_cloud", "seg65536", 90): 1.21,
    ("cesm_atm_cloud", "seg_global", 10): 0.00,
    ("cesm_atm_cloud", "seg_global", 50): 0.00,
    ("cesm_atm_cloud", "seg_global", 90): 0.00,
    ("hurricane_u", "seg4096", 10): 2.50,
    ("hurricane_u", "seg4096", 50): 3.35,
    ("hurricane_u", "seg4096", 90): 2.55,
    ("hurricane_u", "seg65536", 10): 0.20,
    ("hurricane_u", "seg65536", 50): 0.20,
    ("hurricane_u", "seg65536", 90): 0.19,
    ("hurricane_u", "seg_global", 10): 0.00,
    ("hurricane_u", "seg_global", 50): 0.00,
    ("hurricane_u", "seg_global", 90): 0.00,
}

# k* for SUM at epsilon=1e-3 (all scientific datasets)
K_STAR_SUM = 2

# Representation labels
REP_LABELS = {
    "seg4096": "4096",
    "seg16384": "16384",
    "seg65536": "65536",
    "seg_global": "quasi-global",
}

# Sweep configuration
DATASETS = [
    ("cesm_atm_cloud", "0.1714855656027794", "0.02812151238322258", "0.0"),
    ("hurricane_u", "65.99803924560547", "53.02256774902344", "41.232883071899415"),
]
SEGMENTS = ["seg4096", "seg65536", "seg_global"]
SELECTIVITIES = [10, 50, 90]


def get_k_type(k, k_star, max_k):
    if k == k_star:
        return "k_star"
    elif k == max_k:
        return "k_max"
    else:
        return "reference"


def main():
    output_csv = RESULT_DIR / "scientific_locality_attribution.csv"

    header = (
        "dataset,representation,segment_size,segment_count,selectivity,threshold,k,max_planes,"
        "byteplane_fused_ms,raw_fused_ms,bp_vs_raw_speedup,"
        "threshold_prep_ms,warm_e2e_ms,"
        "gpu_count,cpu_enc_count,enc_count_rel_err,count_encoded_match,cpu_raw_count,raw_count_rel_err,count_ok,"
        "gpu_sum,cpu_enc_sum,enc_sum_rel_err,cpu_raw_sum,sum_abs_err,sum_rel_err,raw_sum_abs_err,raw_sum_rel_err,"
        "raw_baseline_count,raw_baseline_sum,"
        "k_type"
    )

    rows = []
    missing = []
    parse_errors = []

    for dataset, thr10, thr50, thr90 in DATASETS:
        for seg in SEGMENTS:
            artifact_dir = LOCALITY_ROOT / dataset / seg
            summary_path = artifact_dir / "summary.json"

            if not summary_path.exists():
                missing.append(f"{dataset}/{seg}: summary.json not found")
                continue

            with open(summary_path) as f:
                summary = json.load(f)

            seg_size = summary["segment_size"]
            seg_count = summary["segment_count"]
            max_k = summary["max_plane_count"]
            rep_label = REP_LABELS.get(seg, seg)

            thresholds = {10: thr10, 50: thr50, 90: thr90}

            for sel, thr in thresholds.items():
                prep_ms = PREP_MS.get((dataset, seg, sel), 0.0)

                for k in range(1, max_k + 1):
                    csv_name = f"{dataset}_{seg}_s{sel}_k{k}.csv"
                    csv_path = RESULT_DIR / csv_name

                    if not csv_path.exists():
                        missing.append(csv_name)
                        continue

                    try:
                        with open(csv_path) as f:
                            reader = csv.DictReader(f)
                            row = next(reader)
                    except Exception as e:
                        parse_errors.append(f"{csv_name}: {e}")
                        continue

                    bp_ms = row.get("ms_per_iter", "NA")
                    raw_ms = row.get("raw_baseline_ms_per_iter", "NA")
                    gpu_count = row.get("gpu_count", "NA")
                    cpu_enc_count = row.get("cpu_enc_count", "NA")
                    enc_count_rel_err = row.get("enc_count_rel_err", "NA")
                    cpu_raw_count = row.get("cpu_raw_count", "NA")
                    raw_count_rel_err = row.get("raw_count_rel_err", "NA")
                    count_ok = row.get("validated", "NA")
                    gpu_sum = row.get("gpu_sum", "NA")
                    cpu_enc_sum = row.get("cpu_enc_sum", "NA")
                    enc_sum_rel_err = row.get("enc_sum_rel_err", "NA")
                    cpu_raw_sum = row.get("cpu_raw_sum", "NA")
                    sum_abs_err = row.get("raw_sum_abs_err", "NA")
                    sum_rel_err = row.get("raw_sum_rel_err", "NA")
                    raw_baseline_count = row.get("raw_baseline_count", "NA")
                    raw_baseline_sum = row.get("raw_baseline_sum", "NA")

                    # Compute speedup
                    try:
                        bp_val = float(bp_ms)
                        raw_val = float(raw_ms)
                        speedup = f"{raw_val / bp_val:.4f}" if bp_val > 0 else "NA"
                    except (ValueError, ZeroDivisionError):
                        speedup = "NA"

                    # Compute warm E2E
                    try:
                        warm_e2e = f"{prep_ms + float(bp_ms):.4f}"
                    except ValueError:
                        warm_e2e = "NA"

                    if gpu_count != "NA" and cpu_enc_count != "NA":
                        count_encoded_match = str(gpu_count == cpu_enc_count).lower()
                    else:
                        count_encoded_match = "NA"

                    k_type = get_k_type(k, K_STAR_SUM, max_k)

                    row_str = (
                        f"{dataset},{rep_label},{seg_size},{seg_count},{sel},{thr},{k},{max_k},"
                        f"{bp_ms},{raw_ms},{speedup},"
                        f"{prep_ms:.2f},{warm_e2e},"
                        f"{gpu_count},{cpu_enc_count},{enc_count_rel_err},{count_encoded_match},{cpu_raw_count},{raw_count_rel_err},{count_ok},"
                        f"{gpu_sum},{cpu_enc_sum},{enc_sum_rel_err},{cpu_raw_sum},{sum_abs_err},{sum_rel_err},{sum_abs_err},{sum_rel_err},"
                        f"{raw_baseline_count},{raw_baseline_sum},"
                        f"{k_type}"
                    )
                    rows.append(row_str)

    # Write output
    with open(output_csv, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(row + "\n")

    print(f"Wrote {len(rows)} rows to {output_csv}")
    if missing:
        print(f"Missing {len(missing)} CSV files:")
        for m in missing:
            print(f"  {m}")
    if parse_errors:
        print(f"Parse errors: {len(parse_errors)}")
        for e in parse_errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
