#!/usr/bin/env python3
"""Aggregate map-seed sweep results into a summary CSV."""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

SWEEP_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
OUT = SWEEP_DIR / "nmr_d2_mapseed_aggregate.csv"

rows = []
for seed_dir in sorted(SWEEP_DIR.glob("seed_*/")):
    ms = seed_dir.name.replace("seed_", "")
    p = seed_dir / "nmr_d2_paired_delta_summary.csv"
    if not p.is_file():
        continue
    with p.open() as f:
        for r in csv.DictReader(f):
            r["map_seed"] = ms
            rows.append(r)

if not rows:
    print("No data found")
    sys.exit(1)

dataset = rows[0]["dataset"]
by_seed = defaultdict(list)
for r in rows:
    by_seed[r["map_seed"]].append(r)

print(f"=== Map-Seed Sweep Aggregate: {dataset} ===")
print(f"Total cells: {len(rows)}, Map seeds: {len(by_seed)}\n")

with OUT.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["map_seed", "n_cells", "graded_wins", "uniform_wins", "tie",
                "delta_relbw_mean", "delta_relbw_min", "delta_relbw_max", "delta_msb_mean"])
    for ms in sorted(by_seed, key=int):
        subset = by_seed[ms]
        v = Counter(r["claim_verdict"] for r in subset)
        rb = [float(r["delta_relative_bound_width"]) for r in subset]
        mb = [float(r["delta_msb_coverage_rate"]) for r in subset]
        w.writerow([ms, len(subset), v.get("GRADED_WINS", 0), v.get("UNIFORM_WINS", 0),
                    v.get("TIE", 0), sum(rb) / len(rb), min(rb), max(rb), sum(mb) / len(mb)])
        print(f"  seed {ms}: GRADED_WINS={v.get('GRADED_WINS', 0)}/{len(subset)}  "
              f"delta_relbw=({min(rb):.4e}, {sum(rb)/len(rb):.4e}, {max(rb):.4e})  "
              f"delta_msb={sum(mb)/len(mb):.4e}")

    vb = Counter(r["claim_verdict"] for r in rows)
    rb_all = [float(r["delta_relative_bound_width"]) for r in rows]
    mb_all = [float(r["delta_msb_coverage_rate"]) for r in rows]
    w.writerow(["OVERALL", len(rows), vb.get("GRADED_WINS", 0), vb.get("UNIFORM_WINS", 0),
                vb.get("TIE", 0), sum(rb_all) / len(rb_all), min(rb_all), max(rb_all),
                sum(mb_all) / len(mb_all)])
    print(f"\n  OVERALL: GRADED_WINS={vb.get('GRADED_WINS', 0)}/{len(rows)}  "
          f"delta_relbw=({min(rb_all):.4e}, {sum(rb_all)/len(rb_all):.4e}, {max(rb_all):.4e})")
    print(f"  Verdict distribution: {dict(vb)}")

print(f"\nAggregate: {OUT}")
