#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import struct
from pathlib import Path
from array import array


ROOT = Path("${PROJ_DIR}/gpu-byteplane-scan-experiments")
RAW_ROOT = Path("/c/Users/Nick/Downloads/dev_full/dev")
ART_ROOT = Path("/c/Users/Nick/Downloads/dataset_extracted/BUFF_DEV/exp_runtime_by_p")
OUT_CSV = ROOT / "results/buff_encoder_v2/fidelity_audit.csv"
OUT_MD = ROOT / "research/2026-05-10_New_Encoder_Fidelity_Audit_Report.md"
VERSION = "v2_2026-05-10"
SELECTIVITIES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def load_raw(dataset: str) -> array:
    arr = array("d")
    with (RAW_ROOT / f"{dataset}.f64le.bin").open("rb") as f:
        arr.fromfile(f, 100_000_000)
    return arr


def decode_artifact(artifact_dir: Path, value_count: int, segment_size: int) -> array:
    meta_path = artifact_dir / "segment_meta.csv"
    with meta_path.open(newline="") as f:
        reader = csv.DictReader(f)
        metas = list(reader)

    out = array("d", [0.0]) * value_count
    plane_cache = {}

    for row in metas:
        seg_idx = int(row["segment_index"])
        start = int(row["row_offset"])
        n = int(row["row_count"])
        active = int(row["active_plane_count"])
        dlen = int(row["fractional_bits"])
        base_int = int(row["integer_base_hex"], 16)
        scale = float(1 << dlen)

        planes = []
        for i in range(active):
            p = plane_cache.get(i)
            if p is None:
                p = (artifact_dir / f"plane_{i:03d}.bin").read_bytes()
                plane_cache[i] = p
            planes.append(p[start : start + n])

        for j in range(n):
            packed = 0
            for bits in planes:
                packed = (packed << 1) | (bits[j] & 1)
            out[start + j] = (packed + base_int) / scale

    return out


def verdict_count(pp_drift: float) -> str:
    ad = abs(pp_drift)
    if ad > 1.0:
        return "catastrophic"
    if ad > 0.1:
        return "caution"
    return "acceptable"


def verdict_sum(rel_err: float) -> str:
    if rel_err > 0.01:
        return "catastrophic"
    if rel_err > 0.001:
        return "caution"
    return "acceptable"


def percentile(values: array, p: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(vals[int(k)])
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return float(d0 + d1)


def main() -> None:
    rows = []
    report = []
    datasets = {
        "uniform": ["p2", "p4", "p6", "p8", "p10"],
        "sensor": ["p2", "p4", "p6", "p8", "p10"],
        "heavy_tailed": ["p2", "p3", "p4", "p5", "p6"],
        "zipfian": ["p2", "p4", "p6", "p8"],
    }

    for ds, arts in datasets.items():
        raw = load_raw(ds)
        total = len(raw)
        raw_sum = math.fsum(raw)
        thresholds = {s: percentile(raw, s) for s in SELECTIVITIES}
        raw_counts = {s: sum(1 for v in raw if v > t) for s, t in thresholds.items()}

        report.append(f"## {ds}")
        for art in arts:
            art_dir = ART_ROOT / f"{ds}_{art}"
            with (art_dir / "summary.json").open() as f:
                summary = json.load(f)
            decoded = decode_artifact(art_dir, int(summary["value_count"]), int(summary["segment_size"]))
            enc_sum = math.fsum(decoded)
            sum_abs = abs(raw_sum - enc_sum)
            sum_rel = sum_abs / abs(raw_sum) if raw_sum != 0 else 0.0
            max_count_drift = 0.0
            max_count_sel = None
            for s, t in thresholds.items():
                enc_count = sum(1 for v in decoded if v > t)
                raw_count = raw_counts[s]
                drift = (enc_count - raw_count) / total * 100.0
                max_count_drift = max(max_count_drift, abs(drift))
                max_count_sel = s if abs(drift) >= max_count_drift else max_count_sel
                rows.append({
                    "dataset": ds,
                    "artifact_version": VERSION,
                    "artifact_label": art,
                    "target_selectivity": float(s),
                    "threshold": t,
                    "raw_count": raw_count,
                    "encoded_count": enc_count,
                    "count_abs_error": abs(raw_count - enc_count),
                    "selectivity_drift_pp": drift,
                    "count_verdict": verdict_count(drift),
                    "raw_sum": raw_sum,
                    "encoded_sum": enc_sum,
                    "sum_abs_error": sum_abs,
                    "sum_rel_error": sum_rel,
                    "sum_verdict": verdict_sum(sum_rel),
                })
            report.append(f"- `{art}`: max COUNT drift {max_count_drift:.6f} pp; SUM rel error {sum_rel:.6e}")
            if max_count_drift > 1.0 or sum_rel > 0.01:
                report.append(f"  - Catastrophic: yes")
            verdict = "reject" if max_count_drift > 1.0 or sum_rel > 0.01 else ("side study" if max_count_drift > 0.1 or sum_rel > 0.001 else "COUNT mainline / SUM mainline")
            report.append(f"  - Verdict: {verdict}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(["# v2 Encoder Fidelity Audit Report", "", *report, ""]))


if __name__ == "__main__":
    main()
