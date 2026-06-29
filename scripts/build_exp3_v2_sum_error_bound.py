#!/usr/bin/env python3
"""Build the full 19-artifact v2 SUM error/bound matrix.

For each of the 19 exported v2 runtime artifacts (p2..p10 across 4 datasets),
compute progressive SUM at k=1..max_plane_count and the execution-depth
approximation error vs. encoded_full_depth (full-depth artifact decode).

The analytic bound is the worst-case omitted-plane tail contribution only
(no quantization term), since both progressive and full-depth values
come from the same bounded artifact.

Output: results/exp3_v2_sum_precision/
  - exp3_v2_sum_error_bound_all.csv  (103 rows)
  - coverage_summary.csv
  - bound_validation_summary.csv
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# Matrix definition (19 artifacts, 103 k-rows)
# ------------------------------------------------------------------
EXPECTED: list[tuple[str, str, int]] = [
    ("heavy_tailed", "p2", 7),
    ("heavy_tailed", "p3", 7),
    ("heavy_tailed", "p4", 7),
    ("heavy_tailed", "p5", 8),
    ("heavy_tailed", "p6", 8),
    ("sensor", "p2", 2),
    ("sensor", "p4", 3),
    ("sensor", "p6", 3),
    ("sensor", "p8", 4),
    ("sensor", "p10", 5),
    ("uniform", "p2", 3),
    ("uniform", "p4", 4),
    ("uniform", "p6", 4),
    ("uniform", "p8", 5),
    ("uniform", "p10", 6),
    ("zipfian", "p2", 6),
    ("zipfian", "p4", 6),
    ("zipfian", "p6", 7),
    ("zipfian", "p8", 8),
]

ARTIFACT_ROOT = Path("${WORK_DIR}/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p")
OUT_DIR = Path("results/exp3_v2_sum_precision")
ARTIFACT_VERSION = "v2_2026-05-10"

FIELD_NAMES = [
    "dataset",
    "artifact_version",
    "artifact_label",
    "precision_power",
    "k",
    "effective_k",
    "max_plane_count",
    "progressive_sum",
    "encoded_full_depth_sum",
    "abs_error_vs_encoded_full_depth",
    "rel_error_vs_encoded_full_depth",
    "analytic_abs_bound_vs_encoded_full_depth",
    "analytic_rel_bound_vs_encoded_full_depth",
    "bound_gap_vs_encoded_full_depth",
]


# ------------------------------------------------------------------
# Segment data
# ------------------------------------------------------------------
@dataclass
class SegmentInfo:
    row_offset: int
    row_count: int
    active_plane_count: int
    segment_base: float
    bases: list[float]
    omitted_tail_bounds: list[float]


def _compute_max_values(total_bits: int) -> list[int]:
    if total_bits <= 0:
        return []
    plane_count = (total_bits + 7) // 8
    max_vals: list[int] = []
    for p in range(plane_count):
        if p + 1 == plane_count:
            trailing = total_bits - 8 * p
            width = 8 if trailing <= 0 else trailing
        else:
            width = 8
        max_vals.append((1 << width) - 1)
    return max_vals


def _load_segments(artifact_dir: Path, max_k: int = 8) -> list[SegmentInfo]:
    """Load segment metadata and precompute per-segment omitted-tail bounds."""
    segments: list[SegmentInfo] = []
    with (artifact_dir / "segment_meta.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_offset = int(row["row_offset"])
            row_count = int(row["row_count"])
            active_plane_count = int(row["active_plane_count"])
            fractional_bits = int(row["effective_fractional_bits"])
            integer_offset_bits = int(row["integer_offset_bits"])
            total_bits = fractional_bits + integer_offset_bits

            max_vals = _compute_max_values(total_bits)
            bases = [float(row.get(f"plane_basis_{p}", "0")) for p in range(active_plane_count)]

            omitted = [0.0] * (max_k + 1)
            for k in range(1, max_k + 1):
                keep = min(k, active_plane_count)
                tail = 0.0
                for p in range(keep, active_plane_count):
                    if p < len(max_vals):
                        tail += float(max_vals[p]) * bases[p]
                omitted[k] = tail

            seg_base = float(row.get("segment_base", "0"))
            segments.append(SegmentInfo(
                row_offset=row_offset,
                row_count=row_count,
                active_plane_count=active_plane_count,
                segment_base=seg_base,
                bases=bases,
                omitted_tail_bounds=omitted,
            ))
    return segments


# ------------------------------------------------------------------
# Artifact analysis (CPU-only, no raw FP64 needed)
# ------------------------------------------------------------------
def _validate_artifact_version(manifest: dict, summary: dict, dataset: str, label: str) -> None:
    """Validate artifact version against source metadata if available."""
    src_version = (
        manifest.get("artifact_version") or summary.get("artifact_version")
    )
    if src_version is not None:
        if src_version != ARTIFACT_VERSION:
            raise ValueError(
                f"{dataset}_{label}: artifact_version {src_version!r} "
                f"!= expected {ARTIFACT_VERSION!r}"
            )
    else:
        # artifact_version field is absent from current manifest/summary schema;
        # version is asserted as the runtime artifact root version.
        # See research/2026-05-10_New_Encoder_Artifact_Export_Report.md
        print(
            f"  [{dataset}_{label}] artifact_version field absent from metadata; "
            f"version asserted as runtime root version {ARTIFACT_VERSION}",
            flush=True,
        )


def _analyze_artifact(artifact_dir: Path) -> list[dict]:
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    summary = json.loads((artifact_dir / "summary.json").read_text())
    dataset: str = manifest["dataset"]
    # artifact_label is NOT in manifest.json; derive as suffix after dataset prefix
    dir_name = artifact_dir.name  # e.g. "heavy_tailed_p2"
    prefix = dataset + "_"
    label = dir_name[len(prefix):] if dir_name.startswith(prefix) else dir_name
    max_plane_count: int = manifest["max_plane_count"]
    precision_power: int = summary["precision_power"]
    value_count: int = manifest["value_count"]

    # Hard metadata-validation gates
    expected_label = f"p{precision_power}"
    if label != expected_label:
        raise ValueError(
            f"{dataset}: derived label {label!r} != expected {expected_label!r} "
            f"(precision_power={precision_power})"
        )
    _validate_artifact_version(manifest, summary, dataset, label)

    segments = _load_segments(artifact_dir, max_k=max_plane_count)

    # Memory-map plane files
    planes = [
        np.memmap(artifact_dir / f"plane_{p:03d}.bin", dtype=np.uint8, mode="r")
        for p in range(max_plane_count)
    ]
    for p in planes:
        if p.shape[0] != value_count:
            raise ValueError(f"{dataset}_{label}: plane length mismatch")

    # Progressive sums: 1-indexed, [k] = sum using first k planes
    prog_sums = [0.0] * (max_plane_count + 1)
    bound_sums = [0.0] * (max_plane_count + 1)

    n_seg = len(segments)
    for si, seg in enumerate(segments):
        sl = slice(seg.row_offset, seg.row_offset + seg.row_count)
        current = np.full(seg.row_count, seg.segment_base, dtype=np.float64)

        for k in range(1, max_plane_count + 1):
            if k <= seg.active_plane_count:
                plane_idx = k - 1
                vals = np.asarray(planes[plane_idx][sl], dtype=np.float64)
                current += vals * seg.bases[plane_idx]
                prog_sums[k] += float(np.sum(current, dtype=np.float64))
                bound_sums[k] += float(seg.row_count) * seg.omitted_tail_bounds[k]
            else:
                prog_sums[k] += float(np.sum(current, dtype=np.float64))
                bound_sums[k] += float(seg.row_count) * seg.omitted_tail_bounds[seg.active_plane_count]

        # Handle k where plane_count is 0 (shouldn't happen for valid artifacts)
        if seg.active_plane_count == 0:
            for k in range(1, max_plane_count + 1):
                prog_sums[k] += float(np.sum(current, dtype=np.float64))

        if (si + 1) % 5000 == 0 or si + 1 == n_seg:
            print(f"  [{dataset}_{label}] segment {si+1}/{n_seg}", flush=True)

    encoded_full = prog_sums[max_plane_count]

    rows: list[dict] = []
    for k in range(1, max_plane_count + 1):
        effective_k = min(k, max_plane_count)
        prog = prog_sums[k]
        aerr = abs(prog - encoded_full)
        rerr = aerr / max(abs(encoded_full), 1e-300)
        abnd = bound_sums[k]
        rbnd = abnd / max(abs(encoded_full), 1e-300)
        gap = abnd / max(aerr, 1e-300) if aerr > 0 else (1.0 if abnd == 0.0 else math.inf)

        rows.append({
            "dataset": dataset,
            "artifact_version": ARTIFACT_VERSION,
            "artifact_label": label,
            "precision_power": precision_power,
            "k": k,
            "effective_k": effective_k,
            "max_plane_count": max_plane_count,
            "progressive_sum": prog,
            "encoded_full_depth_sum": encoded_full,
            "abs_error_vs_encoded_full_depth": aerr,
            "rel_error_vs_encoded_full_depth": rerr,
            "analytic_abs_bound_vs_encoded_full_depth": abnd,
            "analytic_rel_bound_vs_encoded_full_depth": rbnd,
            "bound_gap_vs_encoded_full_depth": gap,
        })

    return rows


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> int:
    t0 = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for ds, label, mpc in EXPECTED:
        artifact_dir = ARTIFACT_ROOT / f"{ds}_{label}"
        if not artifact_dir.is_dir():
            print(f"ERROR: artifact directory not found: {artifact_dir}", file=sys.stderr)
            return 1

        t1 = time.time()
        print(f"Processing {ds}_{label} (max_plane_count={mpc})...", flush=True)
        rows = _analyze_artifact(artifact_dir)
        all_rows.extend(rows)
        elapsed = time.time() - t1
        print(f"  Done ({len(rows)} rows, {elapsed:.1f}s)", flush=True)

    # Sort rows
    all_rows.sort(key=lambda r: (r["dataset"], r["artifact_label"], r["k"]))

    # Write error/bound CSV
    out_path = OUT_DIR / "exp3_v2_sum_error_bound_all.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {out_path} ({len(all_rows)} rows)")

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------
    gates_ok = True
    combos_seen = set()
    for r in all_rows:
        combos_seen.add((r["dataset"], r["artifact_label"]))
    if len(combos_seen) != len(EXPECTED):
        print(f"GATE FAIL: expected {len(EXPECTED)} combos, got {len(combos_seen)}")
        gates_ok = False
    else:
        print(f"GATE OK: {len(combos_seen)} unique combos")

    if len(all_rows) != sum(m for _, _, m in EXPECTED):
        print(f"GATE FAIL: expected {sum(m for _, _, m in EXPECTED)} rows, got {len(all_rows)}")
        gates_ok = False
    else:
        print(f"GATE OK: {len(all_rows)} total rows")

    # k-range coverage
    obs_ks: dict[tuple[str, str], set[int]] = defaultdict(set)
    dup = 0
    for r in all_rows:
        key = (r["dataset"], r["artifact_label"])
        kv = r["k"]
        if kv in obs_ks[key]:
            dup += 1
        obs_ks[key].add(kv)
    if dup:
        print(f"GATE FAIL: {dup} duplicate keys")
        gates_ok = False

    for ds, al, mpc in EXPECTED:
        exp_ks = set(range(1, mpc + 1))
        act_ks = obs_ks.get((ds, al), set())
        if act_ks != exp_ks:
            print(f"GATE FAIL: {ds}_{al} k-range expected={sorted(exp_ks)} got={sorted(act_ks)}")
            gates_ok = False

    if gates_ok:
        print("GATE OK: all k-ranges present, no duplicates")

    # Bound validation
    bound_fails = 0
    for r in all_rows:
        abnd = r["analytic_abs_bound_vs_encoded_full_depth"]
        aerr = r["abs_error_vs_encoded_full_depth"]
        if abnd + 1e-12 < aerr:
            bound_fails += 1
            if bound_fails <= 3:
                print(f"BOUND FAIL: {r['dataset']}_{r['artifact_label']} k={r['k']}: "
                      f"bound={abnd} < error={aerr}")

    bv_rows = [{
        "dataset": ds,
        "artifact_label": al,
        "total_rows": len([r for r in all_rows if r["dataset"] == ds and r["artifact_label"] == al]),
        "bound_fails": sum(
            1 for r in all_rows
            if r["dataset"] == ds and r["artifact_label"] == al
            and r["analytic_abs_bound_vs_encoded_full_depth"] + 1e-12 < r["abs_error_vs_encoded_full_depth"]
        ),
        "bound_pass": str(all(
            r["analytic_abs_bound_vs_encoded_full_depth"] + 1e-12 >= r["abs_error_vs_encoded_full_depth"]
            for r in all_rows if r["dataset"] == ds and r["artifact_label"] == al
        )).lower(),
    } for ds, al, _ in EXPECTED]

    bv_path = OUT_DIR / "bound_validation_summary.csv"
    with bv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "artifact_label", "total_rows",
                                           "bound_fails", "bound_pass"])
        w.writeheader()
        w.writerows(bv_rows)
    print(f"Wrote {bv_path}")

    if bound_fails:
        print(f"BOUND FAIL: {bound_fails} total bound violations", file=sys.stderr)
        gates_ok = False
    else:
        print(f"BOUND OK: all {len(all_rows)} rows satisfy bound >= error")

    # Coverage summary
    cov_rows = []
    for ds, al, mpc in EXPECTED:
        act = obs_ks.get((ds, al), set())
        exp = set(range(1, mpc + 1))
        missing = sorted(exp - act)
        extra = sorted(act - exp)
        cov_rows.append({
            "dataset": ds,
            "artifact_label": al,
            "expected_max_k": mpc,
            "expected_rows": mpc,
            "observed_rows": len(act),
            "missing_k": ",".join(str(k) for k in missing) if missing else "",
            "extra_k": ",".join(str(k) for k in extra) if extra else "",
            "complete": str(act == exp).lower(),
        })

    cov_path = OUT_DIR / "coverage_summary.csv"
    with cov_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "artifact_label", "expected_max_k", "expected_rows",
            "observed_rows", "missing_k", "extra_k", "complete",
        ])
        w.writeheader()
        w.writerows(cov_rows)
    print(f"Wrote {cov_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Total rows: {len(all_rows)}")
    print(f"All gates {'PASSED' if gates_ok else 'FAILED'}")
    return 0 if gates_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
