#!/usr/bin/env python3
"""NMR-D2 normalizer sanity checker.

Verifies that the D2 graded-vs-uniform conclusion is robust to the choice of
denominator for relative bound width.  Reads canonical stochastic CSVs and
recomputes claim matrices under multiple denominator definitions.

Denominators tested:
  1. clean_encoded_sum      — current canonical denominator (sum of byte×weight)
  2. max_possible_encoded_sum — 255 × weight × n_rows per plane (upper bound)
  3. encoded_range           — max_possible - clean_encoded_sum (dynamic range)
  4. raw_domain_sum          — decoded FP64 sum from manifest.json (if available)
  5. thresholded_*           — NOT AVAILABLE (D2 does not run thresholded queries)

If any denominator is unstable (near zero, sign-changing, or causes division
by zero), emit UNSTABLE_DENOMINATOR and do not report a misleading ratio.

Usage:
  python3 scripts/validate_nmr_d2_normalizer.py \\
      --stochastic-csv results/.../nmr_d2_stochastic_matrix.csv \\
      --artifact-dir /path/to/planes \\
      --manifest /path/to/manifest.json \\
      --n-rows N \\
      --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PLANES = 8
PLANE_WEIGHTS = [1 << (8 * (7 - p)) for p in range(PLANES)]
UNSTABLE = "UNSTABLE_DENOMINATOR"


def compute_clean_encoded_sum(plane_dir: Path, n_rows: int) -> float:
    total = 0
    for p in range(PLANES):
        for fmt in (f"plane_{p}.bin", f"plane_{p:03d}.bin"):
            path = plane_dir / fmt
            if path.is_file():
                data = path.read_bytes()[:n_rows]
                break
        else:
            continue
        total += sum(data) * PLANE_WEIGHTS[p]
    return float(total)


def compute_max_possible_encoded_sum(n_rows: int) -> float:
    return float(sum(255 * PLANE_WEIGHTS[p] for p in range(PLANES)) * n_rows)


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text())


def is_stable(value: float) -> tuple[bool, str]:
    if math.isnan(value) or math.isinf(value):
        return False, UNSTABLE
    if abs(value) < 1e-30:
        return False, UNSTABLE
    return True, "STABLE"


def compute_relative(
    bw: float, denominator: float, label: str
) -> tuple[float, str]:
    stable, status = is_stable(denominator)
    if not stable:
        return 0.0, status
    return bw / abs(denominator), "STABLE"


HEADLINE_METRICS = [
    "repair_coverage_rate",
    "msb_coverage_rate",
    "repair_failure_degrade_rate",
    "expected_bound_width",
    "relative_bound_width",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stochastic-csv", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--n-rows", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.stochastic_csv.open(newline="") as f:
        stoch_rows = list(csv.DictReader(f))

    dataset = stoch_rows[0]["dataset"]

    clean_encoded = compute_clean_encoded_sum(args.artifact_dir, args.n_rows)
    max_possible = compute_max_possible_encoded_sum(args.n_rows)
    encoded_range = max_possible - clean_encoded

    denom_info: dict[str, dict[str, Any]] = {
        "clean_encoded_sum": {
            "value": clean_encoded,
            "source": f"sum(byte[p] × weight[p]) from plane files",
            "available": True,
        },
        "max_possible_encoded_sum": {
            "value": max_possible,
            "source": f"255 × weight[p] × n_rows per plane (theoretical max)",
            "available": True,
        },
        "encoded_range": {
            "value": encoded_range,
            "source": f"max_possible − clean_encoded_sum",
            "available": True,
        },
    }

    raw_sum = None
    if args.manifest and args.manifest.is_file():
        manifest = load_manifest(args.manifest)
        raw_sum = float(manifest.get("exact_sum", 0))
        denom_info["raw_domain_sum"] = {
            "value": raw_sum,
            "source": f"manifest.json exact_sum={raw_sum}",
            "available": True,
        }
    else:
        denom_info["raw_domain_sum"] = {
            "value": 0.0,
            "source": "not available (no manifest.json or manifest not provided)",
            "available": False,
        }

    for d in ["thresholded_encoded_sum", "thresholded_raw_sum"]:
        denom_info[d] = {
            "value": 0.0,
            "source": "not available (D2 does not run thresholded queries)",
            "available": False,
        }

    for name, info in denom_info.items():
        if info["available"]:
            stable, status = is_stable(info["value"])
            info["status"] = status

    policies = ["graded_seg_B3", "uniform_spread_seg_B3"]
    g_pol, u_pol = policies

    all_results: list[dict] = []
    normalizer_rows: list[dict] = []

    for denom_name, denom_info_d in denom_info.items():
        if not denom_info_d["available"]:
            continue
        if denom_info_d.get("status") == UNSTABLE:
            continue
        denom_val = denom_info_d["value"]

        for row in stoch_rows:
            bw = float(row["expected_bound_width"])
            rel, status = compute_relative(bw, denom_val, denom_name)

            all_results.append({
                "dataset": dataset,
                "denominator": denom_name,
                "policy": row["policy"],
                "fault_rate": row["fault_rate"],
                "seed": row["seed"],
                "n_segments": row["n_segments"],
                "n_rows": row["n_rows"],
                "expected_bound_width": bw,
                "denominator_value": denom_val,
                "denominator_status": status,
                "relative_bound_width": rel,
                "msb_coverage_rate": float(row["msb_coverage_rate"]),
                "repair_coverage_rate": float(row["repair_coverage_rate"]),
                "repair_failure_degrade_rate": float(row.get("repair_failure_degrade_rate", 0)),
            })

    delta_rows: list[dict] = []
    grouped: dict = defaultdict(dict)
    for r in all_results:
        key = (r["fault_rate"], r["seed"], r["denominator"])
        grouped[key][r["policy"]] = r

    for key, pr in grouped.items():
        rate, seed, denom_name = key
        graded = pr.get(g_pol)
        uniform = pr.get(u_pol)
        if graded is None or uniform is None:
            continue

        d_msb = graded["msb_coverage_rate"] - uniform["msb_coverage_rate"]
        d_relbw = graded["relative_bound_width"] - uniform["relative_bound_width"]
        d_bw = graded["expected_bound_width"] - uniform["expected_bound_width"]
        d_repair = graded["repair_coverage_rate"] - uniform["repair_coverage_rate"]

        graded_wins = 0
        uniform_wins = 0
        if d_msb > 0.001:
            graded_wins += 1
        elif d_msb < -0.001:
            uniform_wins += 1
        if d_relbw < -1e-6:
            graded_wins += 1
        elif d_relbw > 1e-6:
            uniform_wins += 1

        if graded_wins == uniform_wins:
            verdict = "TIE"
        else:
            verdict = "GRADED_WINS" if graded_wins > uniform_wins else "UNIFORM_WINS"

        delta_rows.append({
            "dataset": dataset,
            "denominator": denom_name,
            "fault_rate": rate,
            "seed": seed,
            "delta_msb_coverage_rate": d_msb,
            "delta_relative_bound_width": d_relbw,
            "delta_expected_bound_width": d_bw,
            "delta_repair_coverage_rate": d_repair,
            "claim_verdict": verdict,
            "graded_rel_bw": graded["relative_bound_width"],
            "uniform_rel_bw": uniform["relative_bound_width"],
        })

    val_fieldnames = [
        "dataset", "denominator", "denominator_value", "denominator_status",
        "source", "available",
    ]
    val_rows = []
    for name, info in denom_info.items():
        val_rows.append({
            "dataset": dataset,
            "denominator": name,
            "denominator_value": info["value"],
            "denominator_status": info.get("status", "N/A"),
            "source": info["source"],
            "available": str(info["available"]),
        })
    val_path = args.output_dir / "normalizer_validation.csv"
    with val_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=val_fieldnames)
        w.writeheader()
        w.writerows(val_rows)
    print(f"normalizer_validation: {val_path} ({len(val_rows)} denominators)")

    cm_fieldnames = [
        "dataset", "denominator", "fault_rate", "seed",
        "delta_msb_coverage_rate", "delta_relative_bound_width",
        "delta_expected_bound_width", "delta_repair_coverage_rate",
        "claim_verdict", "graded_rel_bw", "uniform_rel_bw",
    ]
    cm_path = args.output_dir / "nmr_d2_normalizer_claim_matrix.csv"
    with cm_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cm_fieldnames)
        w.writeheader()
        w.writerows(delta_rows)
    print(f"normalizer_claim_matrix: {cm_path} ({len(delta_rows)} rows)")

    by_denom: dict[str, list[dict]] = defaultdict(list)
    for dr in delta_rows:
        by_denom[dr["denominator"]].append(dr)

    agg_fieldnames = [
        "dataset", "denominator", "n_cells", "n_graded_wins", "n_uniform_wins", "n_tie",
        "delta_relbw_mean", "delta_relbw_min", "delta_relbw_max",
        "delta_msb_mean",
    ]
    agg_rows = []
    for denom_name, rows in sorted(by_denom.items()):
        verdicts = Counter(r["claim_verdict"] for r in rows)
        relbw_vals = [r["delta_relative_bound_width"] for r in rows]
        msb_vals = [r["delta_msb_coverage_rate"] for r in rows]
        agg_rows.append({
            "dataset": dataset,
            "denominator": denom_name,
            "n_cells": len(rows),
            "n_graded_wins": verdicts.get("GRADED_WINS", 0),
            "n_uniform_wins": verdicts.get("UNIFORM_WINS", 0),
            "n_tie": verdicts.get("TIE", 0),
            "delta_relbw_mean": sum(relbw_vals) / len(relbw_vals) if relbw_vals else 0,
            "delta_relbw_min": min(relbw_vals) if relbw_vals else 0,
            "delta_relbw_max": max(relbw_vals) if relbw_vals else 0,
            "delta_msb_mean": sum(msb_vals) / len(msb_vals) if msb_vals else 0,
        })
    agg_path = args.output_dir / "nmr_d2_normalizer_aggregate_summary.csv"
    with agg_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fieldnames)
        w.writeheader()
        w.writerows(agg_rows)
    print(f"normalizer_aggregate: {agg_path} ({len(agg_rows)} denominators)")

    print(f"\n=== Normalizer Sanity: {dataset} ===")
    print(f"\nDenominators:")
    for name, info in denom_info.items():
        mark = "STABLE" if info.get("status") == "STABLE" else "UNAVAILABLE" if not info["available"] else "UNSTABLE"
        print(f"  [{mark}] {name}: {info['source']}")
        if info["available"]:
            print(f"       value={info['value']:.4e}  status={info.get('status', 'N/A')}")

    print(f"\nVerdicts per denominator:")
    for ar in agg_rows:
        d = ar["denominator"]
        print(f"  {d}: GRADED_WINS={ar['n_graded_wins']}/{ar['n_cells']}  "
              f"delta_relbw=({ar['delta_relbw_min']:.4e}, {ar['delta_relbw_mean']:.4e}, {ar['delta_relbw_max']:.4e})")

    stable_consistent = all(a["n_uniform_wins"] == 0 for a in agg_rows if a["denominator"] != "raw_domain_sum")
    print(f"\n  {'PASS: graded wins under all stable denominators' if stable_consistent else 'FAIL: some denominator flips the verdict'}")


if __name__ == "__main__":
    main()
