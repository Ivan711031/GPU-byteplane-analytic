#!/usr/bin/env python3
"""Mechanism check pass and gate for Phase 2 v1 (Issue #131 P2-9).

Re-evaluates primary-win cells using graded_naive (does NOT exclude vacuous
planes) against uniform.  Answers: does the graded advantage survive when
vacuous-plane exclusion is forbidden?

Yes -> the primary advantage is due to genuine sensitivity-based allocation.
No  -> the primary advantage is an artifact of excluding vacuous planes.

Usage (dev-machine / mock):
  python3 scripts/run_phase2_mechanism_check.py \\
      --primary-summary results/phase2/policy_ratio_summary.csv \\
      --policy-catalogue results/policy_catalogue.json \\
      --sensitivity-profile results/phase2_sensitivity_profile.json \\
      --phase2-matrix results/phase2_matrix.csv \\
      --output-dir results/mechanism_check --mock

Usage (H100 orchestrator – generate manifest):
  python3 scripts/run_phase2_mechanism_check.py \\
      --primary-summary ... --policy-catalogue ... \\
      --sensitivity-profile ... --phase2-matrix ... \\
      --output-dir ... --generate-manifest \\
      --fault-plan-dir /path/to/fault_plans --artifact-root /path/to/artifacts

After H100 runs complete (re-run with existing results):
  python3 scripts/run_phase2_mechanism_check.py \\
      --primary-summary ... --policy-catalogue ... \\
      --sensitivity-profile ... --phase2-matrix ... \\
      --output-dir ...
  # expects phase2_mechanism_check_matrix.csv in output-dir
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


# ── Constants ──────────────────────────────────────────────────────────

PRIMARY_RATIO_THRESHOLD = 1.10
CELL_GRADED_MIN = 10
ROBUST_FRACTION_REQUIRED = 0.80
PRIMARY_POLICY = "graded_vacuous_aware"
MECHANISM_POLICY = "graded_naive"
UNIFORM_POLICY = "uniform"


# ── Data helpers ────────────────────────────────────────────────────────


def _parse_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def _parse_int(s: str) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


def _r_vector_str(r: list[int]) -> str:
    return "|".join(str(v) for v in r)


# ── Step 1: Read primary-win cells ─────────────────────────────────────


def read_policy_ratio_summary(path: Path) -> list[dict[str, Any]]:
    """Read policy_ratio_summary.csv and return all rows."""
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def identify_primary_win_cells(
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter rows where policy_ratio > threshold AND cell_graded >= min."""
    wins: list[dict[str, Any]] = []
    for row in summary_rows:
        pr = _parse_float(row.get("policy_ratio", "0"))
        cg = _parse_float(row.get("cell_graded", "0"))
        if pr > PRIMARY_RATIO_THRESHOLD and cg >= CELL_GRADED_MIN:
            wins.append(row)
    return wins


# ── Step 2: Look up graded_naive r_vector from catalogue ───────────────


def load_policy_catalogue(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def find_catalogue_entry(
    catalogue: dict[str, Any],
    dataset: str,
    budget_B: int,
    policy: str,
    fault_rate: str | None = None,
) -> dict[str, Any] | None:
    for entry in catalogue.get("entries", []):
        if (
            entry.get("dataset") == dataset
            and entry.get("budget_B") == budget_B
            and entry.get("policy") == policy
            and (fault_rate is None or entry.get("fault_rate") == fault_rate)
        ):
            return entry
    return None


def lookup_graded_naive_r_vector(
    catalogue: dict[str, Any],
    dataset: str,
    budget_B: int,
    fault_rate: str = "1e-06",
) -> list[int] | None:
    entry = find_catalogue_entry(
        catalogue, dataset, budget_B, MECHANISM_POLICY, fault_rate,
    )
    if entry is None:
        return None
    return entry.get("r_vector")


# ── Step 3: Mock graded_naive results ──────────────────────────────────


def generate_mock_mechanism_results(
    primary_wins: list[dict[str, Any]],
    uniform_results: dict[tuple[str, int, str], float],
    base_seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate synthetic graded_naive results for testing.

    Uses the uniform result as a baseline and the cell's primary_ratio
    to produce naive_ratio values that exercise both robust and
    vacuous_artifact categories.
    """
    import random as _random
    rng = _random.Random(base_seed)
    rows: list[dict[str, Any]] = []
    for cell in primary_wins:
        ds = cell["dataset"]
        bb = _parse_int(cell.get("budget_B", "0"))
        fr = cell.get("fault_rate", "1e-6")
        key = (ds, bb, fr)
        uniform_med = uniform_results.get(key, 1e12)
        primary_ratio = _parse_float(cell.get("policy_ratio", "1.15"))
        # boundary: graded_naive_med where primary_ratio == 1.25 * naive_ratio
        boundary = uniform_med * 1.25 / max(primary_ratio, 1.01)
        roll = rng.random()
        if roll < 0.80:
            # robust: graded_naive_med <= boundary
            graded_naive_med = rng.uniform(uniform_med * 0.70, boundary)
        else:
            # vacuous_artifact: graded_naive_med > boundary
            graded_naive_med = rng.uniform(boundary, uniform_med * 1.30)
        row = {
            "dataset": ds,
            "budget_B": str(bb),
            "fault_rate": fr,
            "policy_id": MECHANISM_POLICY,
            "gpu_voted_sum": str(int(graded_naive_med)),
            "abs_voted_sum_damage_encoded": str(int(graded_naive_med)),
        }
        rows.append(row)
    return rows


# ── Step 4: Read uniform results from phase2_matrix ────────────────────


def load_uniform_results(
    matrix_path: Path,
) -> dict[tuple[str, int, str], float]:
    """Read phase2_matrix.csv and build uniform median per cell.

    Returns {(dataset, budget_B, fault_rate): median_abs_damage}.
    """
    rows: list[dict[str, Any]] = []
    with matrix_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    cells: dict[tuple[str, int, str], list[float]] = {}
    for row in rows:
        if row.get("policy") != UNIFORM_POLICY:
            continue
        ds = row["dataset"]
        bb = _parse_int(row.get("budget_B", "0"))
        fr = row.get("fault_rate", "")
        ad = _parse_float(row.get("abs_voted_sum_damage_encoded", "0"))
        key = (ds, bb, fr)
        cells.setdefault(key, []).append(ad)

    result: dict[tuple[str, int, str], float] = {}
    for key, vals in cells.items():
        svals = sorted(vals)
        n = len(svals)
        if n == 0:
            continue
        if n % 2 == 1:
            median = svals[n // 2]
        else:
            median = (svals[n // 2 - 1] + svals[n // 2]) / 2.0
        result[key] = median
    return result


def load_mechanism_results(
    matrix_path: Path,
) -> dict[tuple[str, int, str], list[float]]:
    """Read phase2_mechanism_check_matrix.csv and return abs_damage per cell.

    Returns {(dataset, budget_B, fault_rate): [abs_damage, ...]}.
    """
    cells: dict[tuple[str, int, str], list[float]] = {}
    if not matrix_path.is_file():
        return cells
    with matrix_path.open(newline="") as f:
        for row in csv.DictReader(f):
            ds = row.get("dataset", "")
            bb = _parse_int(row.get("budget_B", "0"))
            fr = row.get("fault_rate", "")
            ad = _parse_float(row.get("abs_voted_sum_damage_encoded", "0"))
            key = (ds, bb, fr)
            cells.setdefault(key, []).append(ad)
    return cells


def compute_cell_median(vals: list[float]) -> float:
    svals = sorted(vals)
    n = len(svals)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return svals[n // 2]
    return (svals[n // 2 - 1] + svals[n // 2]) / 2.0


# ── Step 5: Mechanism check status & gate verdict ──────────────────────


def compute_mechanism_check_status(
    primary_ratio: float,
    naive_ratio: float,
) -> str:
    if primary_ratio <= 1.25 * naive_ratio:
        return "robust"
    return "vacuous_artifact"


def build_summary_rows(
    primary_wins: list[dict[str, Any]],
    uniform_results: dict[tuple[str, int, str], float],
    graded_naive_results: dict[tuple[str, int, str], list[float]],
) -> list[dict[str, Any]]:
    """Build per-cell summary with primary_ratio, graded_naive_ratio, status."""
    summary: list[dict[str, Any]] = []
    for cell in primary_wins:
        ds = cell["dataset"]
        bb = _parse_int(cell.get("budget_B", "0"))
        fr = cell.get("fault_rate", "")
        primary_ratio = _parse_float(cell.get("policy_ratio", "0"))
        primary_cell_graded = _parse_float(cell.get("cell_graded", "0"))
        primary_cell_uniform = _parse_float(cell.get("cell_uniform", "0"))

        key = (ds, bb, fr)
        gn_damages = graded_naive_results.get(key, [])
        if not gn_damages:
            # No graded_naive results available yet
            summary.append({
                "dataset": ds,
                "budget_B": str(bb),
                "fault_rate": fr,
                "primary_ratio": f"{primary_ratio:.4f}",
                "primary_cell_graded": f"{primary_cell_graded}",
                "primary_cell_uniform": f"{primary_cell_uniform}",
                "graded_naive_median_damage": "",
                "graded_naive_ratio": "",
                "mechanism_check_status": "pending",
            })
            continue

        gn_median = compute_cell_median(gn_damages)
        uniform_median = uniform_results.get(key, float("nan"))
        if uniform_median and uniform_median > 0 and gn_median > 0:
            gn_ratio = uniform_median / gn_median
        else:
            gn_ratio = float("nan")

        status = compute_mechanism_check_status(primary_ratio, gn_ratio) if not (
            gn_ratio != gn_ratio  # nan check
        ) else "unknown"

        summary.append({
            "dataset": ds,
            "budget_B": str(bb),
            "fault_rate": fr,
            "primary_ratio": f"{primary_ratio:.4f}",
            "primary_cell_graded": f"{primary_cell_graded}",
            "primary_cell_uniform": f"{primary_cell_uniform}",
            "graded_naive_median_damage": f"{gn_median:.0f}",
            "graded_naive_ratio": f"{gn_ratio:.4f}" if not (
                gn_ratio != gn_ratio
            ) else "nan",
            "mechanism_check_status": status,
        })
    return summary


def compute_gate_verdict(
    summary_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, int]]:
    """Compute PASS/FAIL verdict.

    PASS: >=80% of primary-win cells are robust.
    FAIL: <80% robust.

    Only counts cells with a concrete status (not 'pending' or 'unknown').
    """
    counts: dict[str, int] = {
        "robust": 0,
        "vacuous_artifact": 0,
        "pending": 0,
        "unknown": 0,
    }
    for row in summary_rows:
        s = row.get("mechanism_check_status", "pending")
        if s in counts:
            counts[s] += 1
        else:
            counts["unknown"] += 1

    resolved = counts["robust"] + counts["vacuous_artifact"]
    if resolved == 0:
        return "INCONCLUSIVE (no resolved cells)", counts

    robust_frac = counts["robust"] / resolved
    if robust_frac < ROBUST_FRACTION_REQUIRED:
        return "FAIL", counts
    return "PASS", counts


# ── Manifest / sbatch generation ───────────────────────────────────────


def generate_mechanism_check_manifest(
    primary_wins: list[dict[str, Any]],
    catalogue: dict[str, Any],
    output_dir: Path,
    fault_plan_dir: Path,
    artifact_root: Path,
    base_seed: int = 0,
) -> list[dict[str, Any]]:
    """Generate a JSON manifest describing the GPU runs needed.

    Each manifest entry contains all parameters needed to invoke
    phase2_oracle.py with the graded_naive policy.
    """
    manifest: list[dict[str, Any]] = []
    for cell in primary_wins:
        ds = cell["dataset"]
        bb = _parse_int(cell.get("budget_B", "0"))
        fr = cell.get("fault_rate", "1e-6")

        rv = lookup_graded_naive_r_vector(catalogue, ds, bb, fr)
        if rv is None:
            print(
                f"Warning: no graded_naive catalogue entry for "
                f"{ds} B={bb}, skipping",
                file=sys.stderr,
            )
            continue

        manifest.append({
            "dataset": ds,
            "budget_B": bb,
            "fault_rate": fr,
            "policy": MECHANISM_POLICY,
            "r_vector": rv,
            "fault_plan_dir": str(
                fault_plan_dir / ds / f"n100000000/scale{cell.get('scale', '0')}"
                / MECHANISM_POLICY
            ),
            "artifact_root": str(artifact_root),
            "base_seed": base_seed,
        })

    manifest_path = output_dir / "mechanism_check_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {manifest_path} ({len(manifest)} entries)")
    return manifest


def write_sbatch_script(
    output_dir: Path,
    manifest_path: Path,
    partition: str = "dev",
    time: str = "0-01:00",
) -> Path:
    """Write an sbatch submission script that processes the manifest.

    This is a template; the user customizes before submitting.
    """
    sbatch_path = output_dir / "run_mechanism_check.sbatch"
    content = f"""#!/bin/bash
#SBATCH -J mechanism_check
#SBATCH -p {partition}
#SBATCH --gres=gpu:1
#SBATCH -t {time}
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# --- Environment (user must fill in) ---
# ml load ...
# conda activate ...

MANIFEST="{manifest_path}"
OUTDIR="{output_dir}"
MATRIX_CSV="${{OUTDIR}}/phase2_mechanism_check_matrix.csv"

echo "=== Mechanism Check GPU Run ==="
echo "Manifest: ${{MANIFEST}}"
echo "Output: ${{MATRIX_CSV}}"

# Validate GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: ${{GPU_NAME}}"
if [[ "${{GPU_NAME}}" != *"H200"* && "${{GPU_NAME}}" != *"H100"* ]]; then
    echo "ERROR: expected H200 or H100, got ${{GPU_NAME}}" >&2
    exit 2
fi

# Process each manifest entry
echo "" > "${{MATRIX_CSV}}"
HEADER_WRITTEN=0
for row in $(python3 -c "
import json
m = json.loads(open('${{MANIFEST}}').read())
for i, e in enumerate(m):
    print(f'{{i}}')
"); do
    ENTRY=$(python3 -c "
import json
m = json.loads(open('${{MANIFEST}}').read())
e = m[int('${{row}}')]
print(json.dumps(e))
")
    DATASET=$(echo "${{ENTRY}}" | python3 -c "import sys,json; print(json.load(sys.stdin)['dataset'])")
    BUDGET_B=$(echo "${{ENTRY}}" | python3 -c "import sys,json; print(json.load(sys.stdin)['budget_B'])")
    FAULT_RATE=$(echo "${{ENTRY}}" | python3 -c "import sys,json; print(json.load(sys.stdin)['fault_rate'])")
    R_VECTOR=$(echo "${{ENTRY}}" | python3 -c "
import sys,json
e = json.load(sys.stdin)
print(' '.join(str(x) for x in e['r_vector']))
")
    FAULT_PLAN_DIR=$(echo "${{ENTRY}}" | python3 -c "import sys,json; print(json.load(sys.stdin)['fault_plan_dir'])")
    ARTIFACT_ROOT=$(echo "${{ENTRY}}" | python3 -c "import sys,json; print(json.load(sys.stdin)['artifact_root'])")
    BASE_SEED=$(echo "${{ENTRY}}" | python3 -c "import sys,json; print(json.load(sys.stdin)['base_seed'])")

    # Derive artifact path
    ARTIFACT_DIR="${{ARTIFACT_ROOT}}/${{DATASET}}"

    python3 scripts/phase2_oracle.py \\
        --artifact-dir "${{ARTIFACT_DIR}}/artifacts" \\
        --fault-plan-dir "${{FAULT_PLAN_DIR}}" \\
        --r-vector ${{R_VECTOR}} \\
        --dataset "${{DATASET}}" \\
        --n-rows 100000000 \\
        --scale $(python3 -c "
import json
m = json.loads(open('${{MANIFEST}}').read())
e = m[int('${{row}}')]
print(e.get('scale', 0))
") \\
        --fault-rate "${{FAULT_RATE}}" \\
        --base-seed ${{BASE_SEED}} \\
        --policy-id graded_naive \\
        --budget-B ${{BUDGET_B}} \\
        --sensitivity-profile results/phase2_sensitivity_profile.json \\
        --csv "${{OUTDIR}}/_tmp_${{row}}.csv"

    # Append to matrix (skip header after first)
    if [ "${{HEADER_WRITTEN}}" = "0" ]; then
        cat "${{OUTDIR}}/_tmp_${{row}}.csv" > "${{MATRIX_CSV}}"
        HEADER_WRITTEN=1
    else
        tail -n +2 "${{OUTDIR}}/_tmp_${{row}}.csv" >> "${{MATRIX_CSV}}"
    fi
    rm "${{OUTDIR}}/_tmp_${{row}}.csv"
done

echo "=== DONE ==="
python3 scripts/run_phase2_mechanism_check.py \\
    --primary-summary .../policy_ratio_summary.csv \\
    --policy-catalogue results/policy_catalogue.json \\
    --sensitivity-profile results/phase2_sensitivity_profile.json \\
    --phase2-matrix .../phase2_matrix.csv \\
    --output-dir "${{OUTDIR}}"
"""
    sbatch_path.write_text(content)
    sbatch_path.chmod(0o755)
    print(f"sbatch script: {sbatch_path}")
    return sbatch_path


# ── CSV writers ────────────────────────────────────────────────────────


SUMMARY_FIELDNAMES = [
    "dataset", "budget_B", "fault_rate",
    "primary_ratio", "primary_cell_graded", "primary_cell_uniform",
    "graded_naive_median_damage", "graded_naive_ratio",
    "mechanism_check_status",
]


def write_mechanism_check_matrix(
    output_dir: Path,
    graded_naive_rows: list[dict[str, Any]],
) -> Path:
    """Write phase2_mechanism_check_matrix.csv.

    Uses the same schema as phase2_matrix.csv but only for graded_naive runs.
    """
    path = output_dir / "phase2_mechanism_check_matrix.csv"
    if not graded_naive_rows:
        # Create empty with header
        path.write_text("dataset,budget_B,fault_rate,policy_id,gpu_voted_sum,abs_voted_sum_damage_encoded\n")
        return path

    fieldnames = list(graded_naive_rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(graded_naive_rows)
    print(f"Mechanism check matrix: {path} ({len(graded_naive_rows)} rows)")
    return path


def write_summary_csv(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
) -> Path:
    path = output_dir / "mechanism_check_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary: {path} ({len(summary_rows)} rows)")
    return path


def write_report(
    output_dir: Path,
    verdict: str,
    counts: dict[str, int],
    summary_rows: list[dict[str, Any]],
) -> Path:
    path = output_dir / "mechanism_check_report.txt"
    resolved = (
        counts.get("robust", 0)
        + counts.get("vacuous_artifact", 0)
    )
    robust_frac = (
        counts.get("robust", 0) / resolved
        if resolved > 0 else 0.0
    )

    lines = [
        "=== Mechanism Check Gate Report ===",
        "",
        f"Primary-win cells identified: {len(summary_rows)}",
        f"Resolved cells: {resolved}",
        "",
        f"  robust:            {counts.get('robust', 0)}",
        f"  vacuous_artifact:  {counts.get('vacuous_artifact', 0)}",
        f"  pending/unknown:   {counts.get('pending', 0) + counts.get('unknown', 0)}",
        "",
        f"Robust fraction: {robust_frac:.1%}  (threshold: {ROBUST_FRACTION_REQUIRED:.0%})",
        "",
        f"Verdict: {verdict}",
        "",
    ]
    if verdict == "PASS":
        lines.append(
            "PASS: The graded advantage is not mostly due to vacuous-plane "
            "exclusion. >=80% of primary-win cells are robust."
        )
    elif verdict == "FAIL":
        lines.append(
            f"FAIL: Only {robust_frac:.1%} of primary-win cells are robust "
            f"(below {ROBUST_FRACTION_REQUIRED:.0%} threshold). "
            "Most of the graded advantage may come from "
            "vacuous-plane exclusion."
        )
    else:
        lines.append("INCONCLUSIVE: No resolved cells to evaluate.")

    path.write_text("\n".join(lines) + "\n")
    print(f"Report: {path}")
    return path


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 v1 mechanism check pass and gate",
    )
    parser.add_argument(
        "--primary-summary", type=Path, required=True,
        help="Path to policy_ratio_summary.csv",
    )
    parser.add_argument(
        "--policy-catalogue", type=Path, required=True,
        help="Path to policy_catalogue.json",
    )
    parser.add_argument(
        "--sensitivity-profile", type=Path, required=True,
        help="Path to phase2_sensitivity_profile.json",
    )
    parser.add_argument(
        "--phase2-matrix", type=Path, required=True,
        help="Path to phase2_matrix.csv (for uniform results)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for mechanism check artifacts",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use synthetic data for testing (no GPU required)",
    )
    parser.add_argument(
        "--generate-manifest", action="store_true",
        help="Generate fault plan manifest and sbatch script",
    )
    parser.add_argument(
        "--fault-plan-dir", type=Path, default=None,
        help="Base directory for fault plans (required for --generate-manifest)",
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=None,
        help="Artifact root directory (required for --generate-manifest)",
    )
    parser.add_argument(
        "--base-seed", type=int, default=0,
        help="Base seed for fault plan generation",
    )
    args = parser.parse_args()

    # Validate inputs
    for p in [args.primary_summary, args.policy_catalogue,
              args.sensitivity_profile, args.phase2_matrix]:
        if not p.is_file():
            raise SystemExit(f"Input not found: {p}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ──────────────────────────────────────────────────
    summary_rows = read_policy_ratio_summary(args.primary_summary)
    primary_wins = identify_primary_win_cells(summary_rows)
    print(
        f"Primary summary: {len(summary_rows)} cells, "
        f"{len(primary_wins)} primary wins"
    )

    if not primary_wins:
        msg = "No primary-win cells found. Nothing to check."
        print(msg)
        write_report(output_dir, "NO_OP", {}, [])
        sys.exit(0)

    catalogue = load_policy_catalogue(args.policy_catalogue)

    # ── Mock mode ────────────────────────────────────────────────────
    if args.mock:
        print("MOCK mode: generating synthetic graded_naive results")
        uniform_results = load_uniform_results(args.phase2_matrix)
        mock_gn_results = generate_mock_mechanism_results(
            primary_wins, uniform_results, args.base_seed,
        )

        # Build per-cell lookup for graded_naive
        gn_cell_map: dict[tuple[str, int, str], list[float]] = {}
        for row in mock_gn_results:
            key = (
                row["dataset"],
                _parse_int(row.get("budget_B", "0")),
                row["fault_rate"],
            )
            gn_cell_map.setdefault(key, []).append(
                _parse_float(row.get("abs_voted_sum_damage_encoded", "0"))
            )

        write_mechanism_check_matrix(output_dir, mock_gn_results)

    else:
        # ── Real / manifest mode ─────────────────────────────────────
        gn_cell_map = load_mechanism_results(
            output_dir / "phase2_mechanism_check_matrix.csv"
        )

        if not gn_cell_map and args.generate_manifest:
            # Generate manifest and sbatch
            if args.fault_plan_dir is None or args.artifact_root is None:
                raise SystemExit(
                    "--fault-plan-dir and --artifact-root required "
                    "with --generate-manifest"
                )
            generate_mechanism_check_manifest(
                primary_wins, catalogue, output_dir,
                args.fault_plan_dir, args.artifact_root,
                args.base_seed,
            )
            write_sbatch_script(output_dir, output_dir / "mechanism_check_manifest.json")
            print("Manifest and sbatch script generated. Submit to H200 partition.")
            sys.exit(0)

        if not gn_cell_map:
            raise SystemExit(
                "No mechanism check results found. "
                "Run in --mock mode or provide --generate-manifest to "
                "create H200 submission artifacts, or place "
                "phase2_mechanism_check_matrix.csv in the output directory."
            )

    # ── Compute summary and verdict ──────────────────────────────────
    uniform_results = load_uniform_results(args.phase2_matrix)
    summary_rows = build_summary_rows(
        primary_wins, uniform_results, gn_cell_map,
    )

    verdict, counts = compute_gate_verdict(summary_rows)

    # ── Write outputs ────────────────────────────────────────────────
    write_summary_csv(output_dir, summary_rows)
    write_report(output_dir, verdict, counts, summary_rows)

    print(f"\nVerdict: {verdict}")
    print(f"  robust:            {counts.get('robust', 0)}")
    print(f"  vacuous_artifact:  {counts.get('vacuous_artifact', 0)}")

    if verdict == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
