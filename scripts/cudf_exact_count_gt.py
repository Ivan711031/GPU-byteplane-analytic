#!/usr/bin/env python3
"""cuDF exact COUNT(value > threshold) and optional U-row refinement."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def load_gpu_stack() -> tuple[Any, Any]:
    try:
        import cupy as cp  # type: ignore
        import cudf  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on RAPIDS env
        raise RuntimeError(
            "cuDF/CuPy are not available in this Python environment. Use a RAPIDS conda environment."
        ) from exc
    return cp, cudf


def count_chunk_gpu(values: np.ndarray, threshold: float, cp: Any, cudf: Any) -> int:
    gpu_values = cp.asarray(values)
    series = cudf.Series(gpu_values)
    return int((series > threshold).sum())


def resolve_exact_count_field(row: dict[str, str]) -> str:
    candidates = [
        key
        for key in ("refined_exact_count", "exp2_exact", "cuDF_exact_count", "exact_count", "cudf_refined_exact_count")
        if row.get(key, "") != ""
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "primitive CSV row lacks an exact count field; expected refined_exact_count, exp2_exact, cuDF_exact_count, or exact_count"
        )
    raise ValueError(
        f"primitive CSV row has multiple populated exact count fields: {', '.join(candidates)}"
    )


def resolve_raw_path(row: dict[str, str]) -> Path:
    artifact_root = Path(row["artifact_root"])
    candidate = artifact_root.parents[1] / "dev" / f"{row['dataset']}.f64le.bin"
    if candidate.exists():
        return candidate
    fallback = artifact_root.with_name(f"{row['dataset']}.f64le.bin")
    if fallback.exists():
        return fallback
    return candidate


def sweep_join(
    sweep_csv: Path,
    output: Path,
    *,
    repeat: int,
    raw_root: Path | None = None,
) -> None:
    epsilon = 0.001
    cp, cudf = load_gpu_stack()

    rows = list(csv.DictReader(sweep_csv.open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"no rows found in sweep csv: {sweep_csv}")

    output.parent.mkdir(parents=True, exist_ok=True)

    first = rows[0]
    byteplane_setup_ms = (
        float(first.get("artifact_load_ms", 0) or 0)
        + float(first.get("segment_minmax_ms", 0) or 0)
        + float(first.get("gpu_stage_ms", 0) or 0)
        + float(first.get("raw_stage_ms", 0) or 0)
    )

    fieldnames = list(rows[0].keys()) + [
        # cuDF baseline timings (backward-compatible names)
        "cudf_setup_ms",
        "cudf_per_query_ms_p10",
        "cudf_per_query_ms_median",
        "cudf_per_query_ms_p90",
        # cuDF count & match (backward-compatible; also emitted as raw_cudf_*)
        "cudf_count",
        "cudf_count_matches",
        # Clarified raw-FP64 columns
        "raw_cudf_count",
        "raw_cudf_count_matches",
        "raw_cudf_delta",
        "raw_cudf_abs_delta",
        "baseline_semantics",
        # Performance
        "speedup_vs_cudf_full",
        "breakeven_queries",
    ]

    series_cache: dict[Path, tuple[Any, float]] = {}

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            threshold = float(row["threshold"])
            raw_path = (
                raw_root / f"{row['dataset']}.f64le.bin"
                if raw_root is not None
                else resolve_raw_path(row)
            )

            if raw_path not in series_cache:
                if raw_path.stat().st_size % 8 != 0:
                    raise ValueError(f"raw file is not FP64 aligned: {raw_path}")
                values = np.memmap(raw_path, dtype="<f8", mode="r")
                t0 = cp.cuda.Event()
                t1 = cp.cuda.Event()
                t0.record()
                gpu_values = cp.asarray(np.asarray(values))
                series = cudf.Series(gpu_values)
                t1.record()
                t1.synchronize()
                setup_ms = float(cp.cuda.get_elapsed_time(t0, t1))
                series_cache[raw_path] = (series, setup_ms)
            else:
                series, setup_ms = series_cache[raw_path]

            _ = int((series > threshold).sum())

            timings: list[float] = []
            count: int | None = None
            for _ in range(repeat):
                start = cp.cuda.Event()
                end = cp.cuda.Event()
                start.record()
                current_count = int((series > threshold).sum())
                end.record()
                end.synchronize()
                timings.append(float(cp.cuda.get_elapsed_time(start, end)))
                if count is None:
                    count = current_count

            timings_sorted = sorted(timings)
            n = len(timings_sorted)
            median = timings_sorted[n // 2]
            p10 = timings_sorted[max(0, int(n * 0.1))]
            p90 = timings_sorted[min(n - 1, int(n * 0.9))]

            exact_field = resolve_exact_count_field(row)
            exact_count = int(row[exact_field])

            per_query_ms_median = float(row.get("per_query_ms_median", 0) or 0)
            speedup = median / per_query_ms_median if per_query_ms_median > 0 else ""
            diff = median - per_query_ms_median
            if diff <= 0.0:
                breakeven = float("inf")
            else:
                breakeven = max(0.0, (byteplane_setup_ms - setup_ms) / diff)

            raw_count = int(count) if count is not None else 0
            raw_matches = raw_count == exact_count
            raw_delta = exact_count - raw_count
            raw_abs_delta = abs(raw_delta)

            writer.writerow(
                {
                    **row,
                    # backward-compatible cuDF columns
                    "cudf_setup_ms": setup_ms,
                    "cudf_per_query_ms_p10": p10,
                    "cudf_per_query_ms_median": median,
                    "cudf_per_query_ms_p90": p90,
                    "cudf_count": raw_count,
                    "cudf_count_matches": raw_matches,
                    # clarified raw-FP64 columns
                    "raw_cudf_count": raw_count,
                    "raw_cudf_count_matches": raw_matches,
                    "raw_cudf_delta": raw_delta,
                    "raw_cudf_abs_delta": raw_abs_delta,
                    "baseline_semantics": "raw_fp64",
                    # performance
                    "speedup_vs_cudf_full": speedup,
                    "breakeven_queries": breakeven,
                }
            )


def count_gt(
    raw_path: Path,
    threshold: float,
    *,
    state_path: Path | None,
    q: int | None,
    batch_rows: int,
) -> dict[str, Any]:
    if raw_path.stat().st_size % 8 != 0:
        raise ValueError(f"raw file is not FP64 aligned: {raw_path}")
    rows = raw_path.stat().st_size // 8
    values = np.memmap(raw_path, dtype="<f8", mode="r")
    states = None
    if state_path is not None:
        if state_path.stat().st_size != rows:
            raise ValueError("row-state file size does not match raw row count")
        states = np.memmap(state_path, dtype=np.uint8, mode="r")

    cp, cudf = load_gpu_stack()

    count = 0
    candidate_rows = 0
    start_time = time.perf_counter()
    for start in range(0, rows, batch_rows):
        stop = min(rows, start + batch_rows)
        chunk = values[start:stop]
        if states is not None:
            mask = states[start:stop] == 2
            candidate_rows += int(mask.sum())
            if not mask.any():
                continue
            chunk = chunk[mask]
            if chunk.size == 0:
                continue

        count += count_chunk_gpu(np.asarray(chunk), threshold, cp, cudf)
    cp.cuda.Stream.null.synchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    timing_key = "cudf_refine_ms" if states is not None else "cudf_full_ms"

    result: dict[str, Any] = {
        "raw_path": str(raw_path),
        "rows": int(rows),
        "threshold": float(threshold),
        "mode": "cudf",
        "count_gt": int(count),
        "state_path": str(state_path) if state_path else "",
        "candidate_rows": int(candidate_rows) if states is not None else int(rows),
        "elapsed_ms": elapsed_ms,
        timing_key: elapsed_ms,
    }
    if q is not None:
        result["Q"] = int(q)
        result["refined_exact_count"] = int(q + count)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cuDF exact count or U-row refinement.")
    parser.add_argument("--sweep-csv")
    parser.add_argument("--output")
    parser.add_argument("--repeat", default=11, type=int)
    parser.add_argument("--raw-root")
    parser.add_argument("--raw-path")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--state-path")
    parser.add_argument("--q", type=int)
    parser.add_argument("--batch-rows", default=8_000_000, type=int)
    args = parser.parse_args()

    if args.sweep_csv or args.output:
        if not args.sweep_csv or not args.output:
            parser.error("--sweep-csv and --output must be used together")
        sweep_join(
            Path(args.sweep_csv),
            Path(args.output),
            repeat=args.repeat,
            raw_root=Path(args.raw_root) if args.raw_root else None,
        )
        return 0

    if not args.raw_path or args.threshold is None:
        parser.error("--raw-path and --threshold are required unless --sweep-csv/--output are used")

    result = count_gt(
        Path(args.raw_path),
        args.threshold,
        state_path=Path(args.state_path) if args.state_path else None,
        q=args.q,
        batch_rows=args.batch_rows,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
