#!/usr/bin/env python3
"""Diagnose sensor p10 count mismatch: exact byte-level vs CSV-decoded comparison.

Also handles all_qualified / all_disqualified segments (matching benchmark logic).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ARTIFACT_DIR = Path("/work/u4063895/datasets/synthetic/dev_buff_v2_20260510/exp_runtime_by_p/sensor_p10")
RAW_PATH = Path("/work/u4063895/datasets/synthetic/dev/sensor.f64le.bin")


def le_hex_to_u64(text: str) -> int:
    if text.startswith("0x") or text.startswith("0X"):
        text = text[2:]
    return int(text, 16)


def main():
    raw = np.memmap(RAW_PATH, dtype="<f8", mode="r")
    total = len(raw)
    print(f"Total rows: {total}")

    threshold = float(np.quantile(raw, 0.50))
    print(f"Threshold (np.quantile 0.50): {threshold:.17g}")
    raw_count = int(np.count_nonzero(raw > threshold))
    print(f"Raw count (x > threshold): {raw_count:,}")

    with open(ARTIFACT_DIR / "summary.json") as f:
        summary = json.load(f)
    max_planes = int(summary["max_plane_count"])
    print(f"Max planes: {max_planes}")

    with open(ARTIFACT_DIR / "segment_meta.csv", newline="") as f:
        meta = list(csv.DictReader(f))
    print(f"Segments: {len(meta)}")

    # Open plane memmaps
    planes = {}
    for i in range(max_planes):
        p = ARTIFACT_DIR / f"plane_{i:03d}.bin"
        if p.exists():
            planes[i] = np.memmap(p, dtype=np.uint8, mode="r", shape=(total,))

    exact_count = 0
    d3like_count = 0
    d3like_exact_count = 0
    seg_contrib = []

    for seg_idx, row in enumerate(meta):
        start = int(row["row_offset"])
        cnt = int(row["row_count"])
        stop = start + cnt
        active = int(row["active_plane_count"])
        frac = int(row["fractional_bits"])
        int_base_u64 = le_hex_to_u64(row["integer_base_hex"])
        seg_base = float(row["segment_base"])
        seg_max_base = float(row["segment_base"]) + np.float64(2.0 ** float(row["integer_offset_bits"])) - np.float64(2.0 ** (-float(frac)))
        total_bits = frac + int(row["integer_offset_bits"])

        scale_f = np.float64(2.0 ** frac)
        t_code = np.uint64(np.floor(np.float64(threshold) * scale_f))

        # ---- Determine segment classification (matching benchmark) ----
        if float(threshold) < seg_base:
            # all_qualified
            exact_count += cnt
            d3like_count += cnt
            d3like_exact_count += cnt
            continue
        elif float(threshold) >= float(seg_max_base):
            # all_disqualified
            continue
        else:
            # Mixed segment: use byte-level comparison
            pass

        combined_T = (int(t_code) - int_base_u64) & 0xFFFFFFFFFFFFFFFF

        # ---- Reconstruct combined_code from plane bytes ----
        combined_code = np.zeros(cnt, dtype=np.uint64)
        bits_remaining = total_bits
        for plane_idx in range(max_planes):
            if bits_remaining <= 0:
                break
            nbits = min(bits_remaining, 8)
            shift = bits_remaining - nbits
            combined_code += planes[plane_idx][start:stop].astype(np.uint64) * np.uint64(1 << shift)
            bits_remaining -= nbits

        # ---- Exact byte-level comparison ----
        exact_qual = combined_code > np.uint64(combined_T)
        n_exact = int(np.count_nonzero(exact_qual))
        exact_count += n_exact

        # ---- D3-style (CSV segment_base + CSV plane_basis) ----
        plane_bases = [float(row.get(f"plane_basis_{i}", "0")) for i in range(active)]
        decoded = np.full(cnt, seg_base, dtype=np.float64)
        for plane_idx in range(active):
            decoded += planes[plane_idx][start:stop].astype(np.float64) * plane_bases[plane_idx]
        d3like_count += int(np.count_nonzero(decoded > threshold))

        # ---- D3-style with exact int_base/2^f ----
        exact_base = np.float64(int_base_u64) / scale_f
        decoded_exact = np.full(cnt, exact_base, dtype=np.float64)
        bits_remaining = total_bits
        for plane_idx in range(max_planes):
            if bits_remaining <= 0:
                break
            nbits = min(bits_remaining, 8)
            shift = bits_remaining - nbits
            weight = np.float64(1 << shift) / scale_f
            decoded_exact += planes[plane_idx][start:stop].astype(np.float64) * weight
            bits_remaining -= nbits
        d3like_exact_count += int(np.count_nonzero(decoded_exact > threshold))

        # Track mismatch per segment
        d3_exact_qual = decoded_exact > threshold
        mismatch = exact_qual & ~d3_exact_qual
        n_mm = int(np.count_nonzero(mismatch))
        if n_mm != 0:
            seg_contrib.append((seg_idx, n_exact, cnt, n_mm,
                                seg_base, int_base_u64, frac,
                                int(combined_T)))

        if seg_idx % 5000 == 0:
            print(f"  seg {seg_idx}/{len(meta)}...")

    print(f"\n=== Results ===")
    print(f"  Exact byte-level (benchmark):         {exact_count:,}")
    print(f"  D3-style (CSV segment_base):          {d3like_count:,}")
    print(f"  D3-style (exact int_base/2^f):        {d3like_exact_count:,}")
    print(f"  Raw (> threshold):                     {raw_count:,}")

    print(f"\n  Delta exact vs D3_csv:                {exact_count - d3like_count:+d}")
    print(f"  Delta exact vs D3_exact_base:         {exact_count - d3like_exact_count:+d}")
    print(f"  Delta D3_csv vs D3_exact_base:        {d3like_count - d3like_exact_count:+d}")
    print(f"  Delta exact vs raw:                   {exact_count - raw_count:+d}")

    seg_contrib.sort(key=lambda x: -x[3])
    print(f"\n  Top segments with exact!=D3_exact_base mismatch:")
    for seg_idx, n_ex, cnt, n_mm, sb, ib, f, ct in seg_contrib[:10]:
        print(f"    seg={seg_idx:5d} exact_qual={n_ex:6d}/{cnt}  mismatches={n_mm:6d}  "
              f"seg_base={sb:.6f}  combined_T={ct:20d}")

    # ---- Dump per-segment mismatch CSV ----
    csv_path = Path("results/diagnostics/sensor_p10_seg_mismatch.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment_index", "row_count", "exact_qualified", "mismatches",
                     "mismatch_rate", "seg_base", "int_base_u64", "frac", "combined_T"])
        for seg_idx, n_ex, cnt, n_mm, sb, ib, fv, ct in seg_contrib:
            w.writerow([seg_idx, cnt, n_ex, n_mm,
                        f"{n_mm / cnt:.6f}" if cnt else "N/A",
                        sb, ib, fv, ct])
    print(f"\n  Wrote per-segment mismatch CSV: {csv_path}")


if __name__ == "__main__":
    main()
