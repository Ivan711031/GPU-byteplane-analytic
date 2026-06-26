#!/usr/bin/env python3
"""cuDF raw-FP64 filter+aggregate baseline: COUNT(*), SUM(x), AVG(x) WHERE x > threshold.

CuPy CUDA Event timing (p10/median/p90 per iteration).  Validates against
raw_baseline_count / raw_baseline_sum from the formal sweep CSV and outputs one
row per representative (dataset, k) point so results directly join the fused /
raw-CUDA columns.

Strategies:
  compact_then_reduce:  mask = series > t;  count = |mask|;  sum_ = sum(series[mask])
  masked_reduce:        mask = series > t;  count = |mask|;  sum_ = sum(where(mask, val, 0))

Timing policies:
  gpu_hot_path:         Uses pylibcudf for device-side reductions (cudf.Series.sum()
                        returns host scalars and forces readback).  CUDA event
                        wraps mask creation, compaction/materialization, and all
                        reductions through pylibcudf.reduce.reduce(),
                        apply_boolean_mask / copy_if_else.  Pylibcudf scalars
                        remain device-side until after event end+synchronize;
                        Python int/float conversions happen outside timed region.
                        Requires pylibcudf (RAPIDS 24.02+).
  query_result_latency: Uses high-level cuDF API (cudf.Series.sum() etc.).
                        CUDA event end recorded *after* int()/float() conversions
                        — includes host scalar readback in timed region.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Lazy RAPIDS imports
# ---------------------------------------------------------------------------

def _load_gpu_stack() -> tuple[Any, Any]:
    """Import cudf + cupy; raise with a human-readable message if missing."""
    try:
        import cupy as cp  # type: ignore
        import cudf       # type: ignore  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "cuDF/CuPy are not available.  Use a RAPIDS conda environment "
            "(e.g. `conda activate gpu-byteplane-rapids` on the GPU node)."
        ) from exc
    return cp, cudf


# ---------------------------------------------------------------------------
# pylibcudf integration (for true device-side gpu_hot_path)
# ---------------------------------------------------------------------------

def _load_pylibcudf() -> Any:
    """Import pylibcudf; raise RuntimeError if not available.

    pylibcudf is required for the ``gpu_hot_path`` timing policy so we can
    reduce device-side and keep scalars on the GPU until after the CUDA event.
    """
    try:
        import pylibcudf as plc  # type: ignore  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "pylibcudf is not available.  The gpu_hot_path timing policy "
            "requires pylibcudf for true device-side reductions (cudf.Series.sum() "
            "returns host scalars and forces readback).  Use "
            "--timing-policy query_result_latency instead, or upgrade to a "
            "RAPIDS version (24.02+) that ships pylibcudf."
        ) from None
    return plc


def _cudf_col_to_plc(col: Any) -> Any:
    """Convert a cudf Series or Column to a pylibcudf Column.

    Handles version differences: cudf Columns expose ``to_pylibcudf()``
    since RAPIDS 24.02; if missing we raise with a clear message.
    """
    # Unwrap Series -> Column if needed
    if hasattr(col, "_column"):
        col = col._column
    if hasattr(col, "to_pylibcudf"):
        try:
            return col.to_pylibcudf(mode="read")
        except TypeError:
            return col.to_pylibcudf()
    raise RuntimeError(
        "Cannot convert cudf column to pylibcudf Column.  "
        "Your RAPIDS version may predate the pylibcudf Python bindings.  "
        "Use --timing-policy query_result_latency instead."
    )


def _plc_scalar_value(scalar: Any, pytype: type) -> Any:
    """Extract a Python ``int`` or ``float`` from a pylibcudf Scalar.

    Prefer ``.to_py()`` (current API); fall back to ``.value`` (legacy).
    """
    if hasattr(scalar, "to_py"):
        return pytype(scalar.to_py())
    if hasattr(scalar, "value"):
        v = scalar.value
        if callable(v):
            return pytype(v())
        return pytype(v)
    # Fallback: try direct conversion
    return pytype(scalar)


# ---------------------------------------------------------------------------
# Raw file resolution (keep the common-case logic simple)
# ---------------------------------------------------------------------------

def _resolve_raw_path(artifact_root: str, dataset: str,
                      raw_root: str | None) -> Path:
    """Return the path to the raw FP64 LE file for *dataset*.

    Resolution order:
    1. --raw-root / <dataset>.f64le.bin (if provided)
    2. ``artifact_root/../../dev/<dataset>.f64le.bin``
    3. ``<repo_root>/../../dev/<dataset>.f64le.bin`` (hopeful heuristic)
    """
    if raw_root is not None:
        candidate = Path(raw_root) / f"{dataset}.f64le.bin"
        if candidate.is_file():
            return candidate

    art = Path(artifact_root)
    # artifact_root is  …/exp_runtime_by_p/<artifact_name>
    # raw files live in       …/dev/<dataset>.f64le.bin
    candidate = art.parents[2] / "dev" / f"{dataset}.f64le.bin"
    if candidate.is_file():
        return candidate

    # Fallback: try relative to this repo's own root
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root.parents[1] / "datasets" / "synthetic" / "dev" / f"{dataset}.f64le.bin"
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Cannot locate raw FP64 file for dataset={dataset}.  "
        f"Tried --raw-root, artifact_root ../../dev/, and repo-relative "
        f"path.  Pass --raw-root explicitly."
    )


# ---------------------------------------------------------------------------
# Strategy / timing-policy constants
# ---------------------------------------------------------------------------

STRATEGY_COMPACT_THEN_REDUCE = "compact_then_reduce"
STRATEGY_MASKED_REDUCE = "masked_reduce"
STRATEGIES = [STRATEGY_COMPACT_THEN_REDUCE, STRATEGY_MASKED_REDUCE]

TIMING_GPU_HOT_PATH = "gpu_hot_path"
TIMING_QUERY_RESULT_LATENCY = "query_result_latency"
TIMING_POLICIES = [TIMING_GPU_HOT_PATH, TIMING_QUERY_RESULT_LATENCY]

# ---------------------------------------------------------------------------
# Combinatoric expansion (--strategy all / --timing-policy all)
# ---------------------------------------------------------------------------

def _expand_variants(strategy_arg: str, timing_policy_arg: str
                     ) -> list[tuple[str, str]]:
    """Return the list of (strategy, timing_policy) pairs to run."""
    strategies = STRATEGIES if strategy_arg == "all" else [strategy_arg]
    policies = TIMING_POLICIES if timing_policy_arg == "all" else [timing_policy_arg]
    return list(itertools.product(strategies, policies))


# ---------------------------------------------------------------------------
# Per-iteration timed expression
# ---------------------------------------------------------------------------

def _warmup_iter(series: Any, cp: Any, threshold: float,
                 strategy: str) -> None:
    """Run one warmup iteration (no event timing overhead) to prime caches."""
    mask = series > threshold
    cp.cuda.Stream.null.synchronize()
    if strategy == STRATEGY_COMPACT_THEN_REDUCE:
        _ = int(mask.sum())
        _ = float(series[mask].sum())
    else:
        _ = int(mask.sum())
        _ = float(series.where(mask, 0).sum())
    cp.cuda.Stream.null.synchronize()


def _run_gpu_hot_path_pylibcudf(
    series: Any, cp: Any, plc: Any, threshold: float, strategy: str
) -> tuple[float, int, float]:
    """True device-side hot path using pylibcudf reductions.

    CUDA event wraps only GPU-side operations:
      - Boolean mask creation (``series > threshold``)
      - Mask compaction or materialization via ``apply_boolean_mask`` /
        ``copy_if_else``
      - Device-side reductions via ``pylibcudf.reduce.reduce()``

    Pylibcudf scalars (count, sum) remain device-side until **after**
    ``end_ev.synchronize()``.  Python ``int``/``float`` conversions happen
    outside the timed region.

    Returns ``(elapsed_ms, count, sum_)``.
    """
    series_col = _cudf_col_to_plc(series)

    start_ev = cp.cuda.Event()
    end_ev = cp.cuda.Event()

    start_ev.record()

    # 1. Create boolean mask on device
    mask_series = series > threshold
    mask_col = _cudf_col_to_plc(mask_series)

    # 2. Device-side count = reduce_sum(cast(mask, int64))
    int64_dtype = plc.DataType(plc.TypeId.INT64)
    float64_dtype = plc.DataType(plc.TypeId.FLOAT64)
    count_scalar = plc.reduce.reduce(
        plc.unary.cast(mask_col, int64_dtype),
        plc.aggregation.sum(),
        int64_dtype,
    )

    # 3. Device-side sum
    if strategy == STRATEGY_COMPACT_THEN_REDUCE:
        # A compact: keep only rows where mask is True, then sum them
        compacted_table = plc.stream_compaction.apply_boolean_mask(plc.Table([series_col]), mask_col)
        compacted = compacted_table.columns()[0]
        sum_scalar = plc.reduce.reduce(compacted, plc.aggregation.sum(),
                                       float64_dtype)
    else:
        # B masked: where(mask, value, 0) then sum
        zero = plc.Scalar.from_py(0.0, dtype=float64_dtype)
        masked = plc.copying.copy_if_else(series_col, zero, mask_col)
        sum_scalar = plc.reduce.reduce(masked, plc.aggregation.sum(),
                                       float64_dtype)

    end_ev.record()
    end_ev.synchronize()
    elapsed_ms = float(cp.cuda.get_elapsed_time(start_ev, end_ev))

    # Convert device scalars to Python *after* timing
    count = _plc_scalar_value(count_scalar, int)
    sum_ = _plc_scalar_value(sum_scalar, float)

    return elapsed_ms, count, sum_


def _run_timed_iter(
    series: Any,
    cp: Any,
    threshold: float,
    strategy: str,
    timing_policy: str,
    plc: Any = None,
) -> tuple[float, int, float]:
    """Run one CUDA-event-timed iteration.

    Returns ``(elapsed_ms, count, sum_)``.

    ``gpu_hot_path``:
        Uses pylibcudf for true device-side reductions (see
        ``_run_gpu_hot_path_pylibcudf``).  Requires ``plc`` (pylibcudf module).

    ``query_result_latency``:
        Uses cuDF high-level API (``cudf.Series.sum()`` etc.) and wall-clock
        timing around the Python calls so host scalar readback/synchronization is
        included in the timed region.
    """
    if timing_policy == TIMING_GPU_HOT_PATH:
        if plc is None:
            plc = _load_pylibcudf()
        return _run_gpu_hot_path_pylibcudf(series, cp, plc, threshold, strategy)

    # --- TIMING_QUERY_RESULT_LATENCY ---
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    mask = series > threshold
    count = int(mask.sum())
    if strategy == STRATEGY_COMPACT_THEN_REDUCE:
        if count == 0:
            sum_ = 0.0
        else:
            sum_ = float(series[mask].sum())
    else:
        sum_ = float(series.where(mask, 0).sum())
    cp.cuda.Stream.null.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return elapsed_ms, count, sum_


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    cp, cudf = _load_gpu_stack()

    # -- read formal sweep CSV ------------------------------------------------
    formal_csv = Path(args.sweep_csv)
    with formal_csv.open(newline="", encoding="utf-8") as fh:
        all_rows = list(csv.DictReader(fh))

    if not all_rows:
        raise SystemExit(f"No rows in sweep CSV: {formal_csv}")

    # -- filter to representative (artifact, k) combos ------------------------
    TARGETS: dict[str, set[int]] = {
        "sensor_p10":       {1, 5},
        "uniform_p10":      {1, 6},
        "heavy_tailed_p6":  {4, 8},
        "zipfian_p8":       {3, 8},
        "bfp-dec12":        {1, 3, 6},
        "bfp-dec11":        {1, 3, 6},
    }

    def _artifact_name(row: dict[str, str]) -> str:
        return Path(row["artifact_root"]).name

    rows_by_target: dict[tuple[str, int], dict[str, str]] = {}
    for r in all_rows:
        art = _artifact_name(r)
        k_set = TARGETS.get(art)
        if k_set is None:
            continue
        k = int(r["max_filter_planes"])
        if k not in k_set:
            continue
        key = (art, k)
        prev = rows_by_target.get(key)
        if prev is None or abs(float(r["selectivity"]) - 0.5) < abs(float(prev["selectivity"]) - 0.5):
            rows_by_target[key] = r

    rows = [
        rows_by_target[(artifact, k)]
        for artifact, k_set in TARGETS.items()
        for k in sorted(k_set)
        if (artifact, k) in rows_by_target
    ]

    if not rows:
        raise SystemExit(
            "No rows matched the representative s≈50% (artifact,k) combos.  "
            "Check --sweep-csv and TARGETS."
        )

    print(f"Selected {len(rows)} representative s≈50% rows from {len(all_rows)} sweep rows.")

    # -- expand strategy × timing_policy variants -------------------------
    variants = _expand_variants(args.strategy, args.timing_policy)
    print(f"Strategy × timing_policy variants: {variants}")

    # -- load pylibcudf eagerly if gpu_hot_path is among the variants ----
    _plc = None
    if any(policy == TIMING_GPU_HOT_PATH for _, policy in variants):
        _plc = _load_pylibcudf()
        print("pylibcudf loaded for device-side gpu_hot_path timing.")

    device_props = cp.cuda.runtime.getDeviceProperties(0)
    device_name = device_props["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8")
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    # -- output header --------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_fields = [
        "dataset",
        "strategy",
        "timing_policy",
        "target_selectivity",
        "threshold",
        "max_filter_planes",
        "cudf_setup_ms",
        "cudf_filter_aggregate_ms_p10",
        "cudf_filter_aggregate_ms_median",
        "cudf_filter_aggregate_ms_p90",
        "cudf_count",
        "cudf_sum",
        "cudf_avg",
        "raw_cpu_or_raw_cuda_count_match",
        "raw_cpu_or_raw_cuda_sum_rel_err",
        "fused_ms_per_iter",
        "raw_cuda_ms_per_iter",
        "speedup_fused_vs_cudf",
        "speedup_raw_cuda_vs_cudf",
        "n",
        "iters",
        "warmup",
        "device",
        "job_id",
        "formal_csv",
        "raw_path",
    ]

    # -- per-dataset cache ----------------------------------------------------
    series_cache: dict[str, tuple[Any, float]] = {}  # raw_path -> (series, setup_ms)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_fields)
        writer.writeheader()

        for row in rows:
            dataset = row["dataset"]
            threshold = float(row["threshold"])
            raw_path = _resolve_raw_path(row["artifact_root"], dataset,
                                         args.raw_root)
            raw_path_str = str(raw_path)

            # --- cudf setup (upload + Series construction, CUDA-event timed) ---
            if raw_path_str not in series_cache:
                if raw_path.stat().st_size % 8 != 0:
                    raise ValueError(f"raw file is not FP64 aligned: {raw_path}")
                values = np.memmap(raw_path, dtype="<f8", mode="r")
                t0_ev = cp.cuda.Event()
                t1_ev = cp.cuda.Event()
                t0_ev.record()
                gpu_values = cp.asarray(np.asarray(values))
                series = cudf.Series(gpu_values)
                t1_ev.record()
                t1_ev.synchronize()
                setup_ms = float(cp.cuda.get_elapsed_time(t0_ev, t1_ev))
                series_cache[raw_path_str] = (series, setup_ms)
            else:
                series, setup_ms = series_cache[raw_path_str]

            # --- per strategy/timing_policy warmup (optionally) ------------
            # Pre-warmup once per strategy (GPU work is same per strategy,
            # timing policy does not change GPU kernels).
            warmed_strategies: set[str] = set()

            # --- timed iterations for each variant ---------------------------
            for strategy, timing_policy in variants:
                # --- warmup this strategy (once per dataset × strategy) ------
                if strategy not in warmed_strategies:
                    for _ in range(max(1, args.warmup)):
                        _warmup_iter(series, cp, threshold, strategy)
                    warmed_strategies.add(strategy)

                # --- timed iterations ---
                timings_ms: list[float] = []
                count_last: int | None = None
                sum_last: float | None = None

                for _ in range(args.iters):
                    elapsed_ms, c, s = _run_timed_iter(
                        series, cp, threshold, strategy, timing_policy,
                        plc=_plc)
                    timings_ms.append(elapsed_ms)
                    count_last = c
                    sum_last = s

                timings_sorted = sorted(timings_ms)
                n_timings = len(timings_sorted)
                median = timings_sorted[n_timings // 2]
                p10 = timings_sorted[max(0, int(n_timings * 0.10))]
                p90 = timings_sorted[min(n_timings - 1, int(n_timings * 0.90))]

                count_cudf = int(count_last) if count_last is not None else 0
                sum_cudf = float(sum_last) if sum_last is not None else 0.0
                avg_cudf = (sum_cudf / count_cudf) if count_cudf > 0 else float("nan")

                # --- validation against formal sweep raw baseline ---
                raw_baseline_count = int(row["raw_baseline_count"])
                raw_baseline_sum = float(row["raw_baseline_sum"])

                count_match = "true" if count_cudf == raw_baseline_count else "false"

                if raw_baseline_sum != 0.0:
                    sum_rel_err = abs(sum_cudf - raw_baseline_sum) / abs(raw_baseline_sum)
                else:
                    sum_rel_err = abs(sum_cudf - raw_baseline_sum) if sum_cudf != 0.0 else 0.0

                # --- speedups ---
                fused_ms = float(row.get("ms_per_iter", 0) or 0)
                raw_cuda_ms = float(row.get("raw_baseline_ms_per_iter", 0) or 0)

                if timing_policy == TIMING_GPU_HOT_PATH:
                    speedup_fused = (median / fused_ms) if fused_ms > 0 else ""
                    speedup_raw = (median / raw_cuda_ms) if raw_cuda_ms > 0 else ""
                else:
                    # The formal fused/raw CUDA columns are GPU hot-path timings
                    # and do not include equivalent scalar D2H result latency.
                    # Leave speedups blank for query_result_latency rows rather
                    # than mixing timing policies.
                    speedup_fused = ""
                    speedup_raw = ""

                writer.writerow({
                    "dataset": dataset,
                    "strategy": strategy,
                    "timing_policy": timing_policy,
                    "target_selectivity": row.get("selectivity", ""),
                    "threshold": f"{threshold:.16g}",
                    "max_filter_planes": row.get("max_filter_planes", ""),
                    "cudf_setup_ms": f"{setup_ms:.6f}",
                    "cudf_filter_aggregate_ms_p10": f"{p10:.6f}",
                    "cudf_filter_aggregate_ms_median": f"{median:.6f}",
                    "cudf_filter_aggregate_ms_p90": f"{p90:.6f}",
                    "cudf_count": str(count_cudf),
                    "cudf_sum": f"{sum_cudf:.16g}",
                    "cudf_avg": f"{avg_cudf:.16g}",
                    "raw_cpu_or_raw_cuda_count_match": count_match,
                    "raw_cpu_or_raw_cuda_sum_rel_err": f"{sum_rel_err:.16g}",
                    "fused_ms_per_iter": f"{fused_ms:.6f}",
                    "raw_cuda_ms_per_iter": f"{raw_cuda_ms:.6f}",
                    "speedup_fused_vs_cudf": f"{speedup_fused:.6f}" if isinstance(speedup_fused, float) else "",
                    "speedup_raw_cuda_vs_cudf": f"{speedup_raw:.6f}" if isinstance(speedup_raw, float) else "",
                    "n": row.get("n", ""),
                    "iters": str(args.iters),
                    "warmup": str(args.warmup),
                    "device": str(device_name),
                    "job_id": job_id,
                    "formal_csv": str(formal_csv.resolve()),
                    "raw_path": raw_path_str,
                })

    print(f"Wrote {len(rows) * len(variants)} rows to {out_path.resolve()}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="cuDF raw-FP64 filter+aggregate baseline (COUNT/SUM/AVG)")
    parser.add_argument("--sweep-csv", required=True,
                        help="Path to formal sweep sweep_results.csv")
    parser.add_argument("--output", required=True,
                        help="Output CSV path for cuDF baseline rows")
    parser.add_argument("--raw-root", default=None,
                        help="Directory containing <dataset>.f64le.bin raw files")
    parser.add_argument("--iters", type=int, default=200,
                        help="Timed iterations per row (default: 200)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations per dataset/threshold (default: 10)")
    parser.add_argument(
        "--strategy", default=STRATEGY_COMPACT_THEN_REDUCE,
        choices=[STRATEGY_COMPACT_THEN_REDUCE, STRATEGY_MASKED_REDUCE, "all"],
        help="Filter+aggregate strategy (default: compact_then_reduce). "
             "Use 'all' to emit rows for both strategies.")
    parser.add_argument(
        "--timing-policy", default=TIMING_QUERY_RESULT_LATENCY,
        choices=[TIMING_GPU_HOT_PATH, TIMING_QUERY_RESULT_LATENCY, "all"],
        help="CUDA event timing policy (default: query_result_latency). "
             "Use 'all' to emit rows for both policies.")
    args = parser.parse_args()

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
