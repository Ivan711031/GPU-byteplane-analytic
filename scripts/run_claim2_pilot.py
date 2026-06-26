"""Claim 2 Pilot Matrix Runner.

Runs the full pilot matrix on CPU:
- 2 primary datasets (hurricane_u, cesm_atm_cloud)
- 4 diversity policies
- 7 candidate fault rates
- All fault suites (A, B, C, D)
- 10 paired seeds per cell

Outputs:
  claim2_pilot_matrix.csv
  claim2_paired_delta_summary.csv
  claim2_fault_family_summary.csv
  claim2_answer_quality_summary.csv
  claim2_verdict.md
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from claim2_evaluator import (
    load_clean_planes,
    load_artifact_metadata,
    planes_to_np,
    evaluate_cell,
    CANDIDATE_RATES,
    POLICIES,
    CELL_FIELDS,
)
from claim2_fault_suites import FAULT_SUITES, iter_family_generators
from claim2_aggregate import (
    compute_paired_deltas,
    compute_family_summary,
    compute_verdict,
    write_verdict_md,
    DELTA_FIELDS,
    AGGREGATE_FIELDS,
)

DATASET_PATHS: dict[str, str] = {
    "hurricane_u": "/work/u4063895/datasets/locality_sensitivity/hurricane_u/seg4096",
    "cesm_atm_cloud": "/work/u4063895/datasets/locality_sensitivity/cesm_atm_cloud/seg4096",
}
N_ROWS = 500_000
PILOT_SEEDS = list(range(10))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=str, nargs="*",
                        default=["hurricane_u"],
                        help="Datasets to run. Default: hurricane_u for smoke")
    parser.add_argument("--n-rows", type=int, default=N_ROWS)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--policies", type=str, nargs="*", default=None)
    parser.add_argument("--rates", type=str, nargs="*", default=None)
    parser.add_argument("--suite", type=str, default=None,
                        help="Filter to specific suite(s). Comma-separated or repeat.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/reliability_layer1/phase3/claim2_pilot"))
    parser.add_argument("--label", type=str, default="claim2_pilot")
    args = parser.parse_args()

    datasets = args.datasets or list(DATASET_PATHS.keys())
    seeds = args.seeds or PILOT_SEEDS
    policies = args.policies or POLICIES
    rates = args.rates or CANDIDATE_RATES

    out_dir = Path(args.output_dir)
    jid = os.environ.get("SLURM_JOB_ID", "local")
    out_dir = out_dir / f"job_{jid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cells: list[dict[str, str]] = []

    for ds_name in datasets:
        ds_path = DATASET_PATHS.get(ds_name)
        if ds_path is None:
            print(f"Unknown dataset: {ds_name}, skipping", file=sys.stderr)
            continue
        artifact_dir = Path(ds_path)
        print(f"Loading {ds_name} from {artifact_dir} ...", file=sys.stderr)
        clean_planes = load_clean_planes(artifact_dir, args.n_rows)
        clean_np = planes_to_np(clean_planes, args.n_rows)
        meta = load_artifact_metadata(artifact_dir)
        scale = meta.get("scale", 1)

        filter_suites = [s.strip() for s in args.suite.split(",")] if args.suite else None
        for suite_id, suite in FAULT_SUITES.items():
            if filter_suites and suite_id not in filter_suites:
                continue
            for fam_name, fam in suite["families"].items():
                generator = fam["generator"]
                for rate_str in rates:
                    rate = float(rate_str)
                    for seed in seeds:
                        entries = generator(seed, args.n_rows, rate)
                        # Evaluate even with empty entries (Suite A null test)
                        for policy in policies:
                            result = evaluate_cell(
                                clean_planes=clean_planes,
                                clean_np=clean_np,
                                entries=entries,
                                policy=policy,
                                seed=seed,
                                n_rows=args.n_rows,
                                scale=scale,
                                dataset=ds_name,
                                suite=suite_id,
                                fault_family=fam_name,
                                rate=rate_str,
                            )
                            all_cells.append(result.row_dict())

                    # Progress
                    n_done = len(all_cells)
                    if n_done % 100 == 0:
                        print(f"  {n_done} cells computed ...", file=sys.stderr)

    # Write raw CSV
    csv_path = out_dir / f"{args.label}_matrix.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CELL_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_cells)
    print(f"Raw matrix: {csv_path} ({len(all_cells)} rows)", file=sys.stderr)

    # Aggregate
    deltas = compute_paired_deltas(all_cells)
    delta_path = out_dir / f"{args.label}_paired_delta_summary.csv"
    with delta_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DELTA_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(deltas)
    print(f"Paired deltas: {delta_path} ({len(deltas)} rows)", file=sys.stderr)

    fam = compute_family_summary(all_cells)
    fam_path = out_dir / f"{args.label}_fault_family_summary.csv"
    with fam_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AGGREGATE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(fam)
    print(f"Family summary: {fam_path} ({len(fam)} rows)", file=sys.stderr)

    aq_path = out_dir / f"{args.label}_answer_quality_summary.csv"
    with aq_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AGGREGATE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(fam)
    print(f"Answer quality summary: {aq_path} ({len(fam)} rows)", file=sys.stderr)

    verdict = compute_verdict(fam, all_cells)
    v_path = out_dir / f"{args.label}_verdict.md"
    write_verdict_md(verdict, v_path)


if __name__ == "__main__":
    main()
