#!/usr/bin/env python3
"""CPU-side regression guard for CESM Q threshold decoding."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CESM_META = ROOT / "results/buff_encoder_v2/containers_scientific/probe_cesm_atm_q_p12/segment_meta.csv"
CESM_THRESH = ROOT / "results/buff_encoder_v2/containers_scientific/cesm_atm_q_threshold_behavior.csv"
CESM_QUASI_GLOBAL = ROOT / "results/exp4_filter_aggregate/scientific_extra_fields_quasi_global_58440/scientific_extra_fields_quasi_global.csv"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"missing required file: {path}")


def read_rows(path: Path) -> list[dict[str, str]]:
    require_file(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no data rows in {path}")
    return rows


def get_fixed_len_bits(
    fractional_bits: int,
    integer_offset_bits: int,
    active_plane_count: int,
    plane_basis: list[float],
) -> int:
    if active_plane_count == 0:
        return 0
    if active_plane_count > 1 and plane_basis and plane_basis[0] > 0.0:
        _, exp_val = math.frexp(plane_basis[0])
        log2_b0 = exp_val - 1
        return log2_b0 + 8 + fractional_bits
    raw_total = fractional_bits + integer_offset_bits
    return 8 if raw_total > 8 else raw_total


def find_row(rows: list[dict[str, str]], label: str, predicate) -> dict[str, str]:
    for row in rows:
        if predicate(row):
            return row
    raise SystemExit(f"missing expected row: {label}")


def main() -> int:
    meta = find_row(
        read_rows(CESM_META),
        "cesm_atm_q segment 0",
        lambda row: row["segment_index"] == "0",
    )
    plane_basis = [
        float(meta[key])
        for key in sorted((key for key in meta if key.startswith("plane_basis_")), key=lambda key: int(key.rsplit("_", 1)[1]))
    ]
    cesm_width = get_fixed_len_bits(
        int(meta["fractional_bits"]),
        int(meta["integer_offset_bits"]),
        int(meta["active_plane_count"]),
        plane_basis,
    )
    if cesm_width != 45:
        raise SystemExit(f"CESM Q width mismatch: expected 45 bits, got {cesm_width}")

    threshold_row = find_row(
        read_rows(CESM_THRESH),
        "cesm_atm_q s10 threshold row",
        lambda row: row["target_selectivity"] == "10",
    )
    if threshold_row["threshold"] != "0.004105758853256702":
        raise SystemExit(f"CESM Q s10 threshold mismatch: {threshold_row['threshold']}")
    if not (
        threshold_row["raw_count"] == threshold_row["decoded_count"] == threshold_row["runtime_count"]
    ):
        raise SystemExit(
            "CESM Q s10 threshold counts disagree: "
            f"raw={threshold_row['raw_count']} decoded={threshold_row['decoded_count']} runtime={threshold_row['runtime_count']}"
        )
    if threshold_row["runtime_decoded_match"] != "True":
        raise SystemExit("CESM Q s10 runtime decoded match is false")

    quasi_row = find_row(
        read_rows(CESM_QUASI_GLOBAL),
        "cesm_atm_q k=max row",
        lambda row: row["dataset"] == "cesm_atm_q"
        and row["threshold"] == "0.0041057588532567024"
        and row["max_filter_planes"] == "6",
    )
    if quasi_row["gpu_count"] != quasi_row["cpu_raw_count"]:
        raise SystemExit(
            "CESM Q k=max count mismatch: "
            f"gpu={quasi_row['gpu_count']} raw={quasi_row['cpu_raw_count']}"
        )
    if quasi_row["gpu_sum"] != quasi_row["cpu_raw_sum"]:
        raise SystemExit(
            "CESM Q k=max sum mismatch: "
            f"gpu={quasi_row['gpu_sum']} raw={quasi_row['cpu_raw_sum']}"
        )

    control_width = get_fixed_len_bits(48, 0, 6, [0.00390625, 0.0, 0.0, 0.0, 0.0, 0.0])
    if control_width != 48:
        raise SystemExit(f"8-divisible control width mismatch: expected 48 bits, got {control_width}")

    print(f"OK: CESM Q get_fixed_len_bits() -> {cesm_width} bits")
    print("OK: CESM Q s10 threshold row matches raw threshold and counts")
    print("OK: CESM Q k=max row matches raw count and sum")
    print("OK: 48-bit control case passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
