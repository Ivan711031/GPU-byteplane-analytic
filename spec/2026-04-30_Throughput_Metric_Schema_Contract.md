# Throughput Metric Schema Contract for Exp3/Exp4 Outputs

**Date**: 2026-04-30  
**Scope**: Exp3 (SUM aggregation) and Exp4 (COUNT filter) benchmarks  
**Status**: v1 contract

---

## Executive Summary

This document defines the **semantic contract** for throughput metrics in Exp3 and Exp4 benchmark CSV outputs. The contract ensures:

1. **Logical throughput** (logical_GBps, logical_bytes_per_iter) is measured against the logical scan footprint, not physical HBM traffic.
2. **Estimated physical throughput** (estimated_physical_GBps, estimated_pack_load_bytes) is clearly marked as an estimate derived from kernel instrumentation, not profiler evidence.
3. **Comparability** across raw FP64, fixed-depth encoded, and progressive encoded execution modes.
4. **No confusion** between different throughput interpretations in plots, reports, and claims.

---

## Core Definitions

### 1. Logical Throughput

**Logical bytes per iteration** = total bytes logically read by the kernel to process the dataset, where "logical" means:
- Each row is counted once per read attempt, regardless of whether the data comes from a higher-level cache.
- Early-exit or progressive skipping does not reduce the logical byte count for rows that were processed.

#### Calculation

```
logical_bytes = n_rows * (active_plane_count_at_k_or_full_depth)
logical_GBps = (logical_bytes / ms_per_iter) / 1e9 GB/ms * 1000 ms/sec
             = (logical_bytes / (ms_per_iter / 1000.0)) / 1e9
```

Where:
- `n_rows` is the total number of rows processed.
- `active_plane_count_at_k_or_full_depth` is the number of byte-planes logically accessed per row:
  - For **raw FP64** runs: 8 bytes per row (double-precision IEEE754).
  - For **fixed-depth encoded** runs: `active_plane_count` (full depth without early-exit).
  - For **progressive k-limited** runs (Exp3): `k` subcolumns × 1 byte/subcolumn = `k` bytes.
  - For **progressive capped-k** runs (Exp4 COUNT): varies per row; reported as average or bound.
- `ms_per_iter` is the wall-clock time per iteration in milliseconds, measured by CUDA events or similar.

#### Interpretation

- **Logical throughput is the "best-case" ceiling** if all logical bytes could be served from L1 cache with zero memory latency.
- Logical throughput **can exceed nominal HBM bandwidth** (e.g., 141 GB/s on H200) if most data is served from cache and the logical byte count is high.
- Logical throughput is **not a physical bandwidth claim** without profiler data (NCU, Nsight Compute).

---

### 2. Estimated Physical Throughput

**Estimated physical bytes per iteration** = bytes inferred to actually pass through HBM or deeper cache levels, based on kernel instrumentation or structural assumptions about memory access patterns.

#### Calculation

For **rowpack16 access patterns** (Exp3, Exp4):

```
estimated_pack_load_bytes = number_of_rowpack_16_plane_loads * 16 bytes
```

Where `number_of_rowpack_16_plane_loads` is the kernel's count of rowpack16-granule memory operations.

```
estimated_physical_GBps = (estimated_pack_load_bytes / ms_per_iter) / 1e9 GB/ms * 1000 ms/sec
                        = (estimated_pack_load_bytes / (ms_per_iter / 1000.0)) / 1e9
```

#### Interpretation

- **Estimated physical throughput is an approximation**, not a ground-truth HBM traffic measurement.
- The estimate is valid only if:
  1. The kernel's memory access pattern is documented (e.g., rowpack16 = aligned 16-byte loads per 16 rows).
  2. The kernel instrumentation (counters, atomics, or grid-stride loops) correctly tallies the number of logical memory operations.
  3. The cache hierarchy behavior is similar to the test environment (GPU model, memory bandwidth, other workloads).
- **If profiler data (NCU) is available**, use the profiler-measured HBM bytes instead. Report both estimated and measured alongside each other.
- **If estimated_physical_GBps exceeds logical_GBps**, the arithmetic is suspect; report and investigate.

---

### 3. Rows Per Second

```
rows_per_sec = n_rows / (ms_per_iter / 1000.0)
             = n_rows * 1000.0 / ms_per_iter

billion_rows_per_sec = rows_per_sec / 1e9
```

#### Interpretation

- **Rows per second is hardware- and query-independent**, independent of logical or physical byte counts.
- Useful for comparing filter predicates or aggregation complexity across the same dataset and encoding.
- Not a "throughput" in the GB/s sense; purely a row-processing rate.

---

### 4. Timing Denominator

All throughput calculations use **wall-clock iteration time (`ms_per_iter`)** as the denominator:

```
ms_per_iter = total_milliseconds / number_of_iterations
```

Where:
- `total_milliseconds` is the sum of CUDA event times across all iterations (typically 10–20 iterations per run).
- **Do not include warmup iterations** in the denominator.
- **Do not include host-side data preparation** (malloc, memcpy setup, CSV writing).

#### Rationale

- Wall-clock time is reproducible and auditable in CI/benchmark runs.
- Does not require profiler access (works on any NVIDIA GPU, any CUDA environment).
- Allows fair comparison across different GPU models and architectures.

---

## Comparison Across Execution Modes

### Raw FP64 Baseline

For a run that scans raw little-endian FP64 data:

- `logical_bytes = n_rows * 8`
- `logical_GBps = (n_rows * 8 / (ms_per_iter / 1000.0)) / 1e9`
- `estimated_physical_GBps`: If the run uses a simple pointer-chase with no special prefetch or cache strategy, estimated ≈ logical (assuming all 8 bytes actually pass through HBM).
- **Baseline for speedup attribution**: Always include a raw FP64 run in the same sweep so that `logical_GBps(encoded) / logical_GBps(raw)` isolates encoding overhead.

### Fixed-Depth Encoded

For a run that reads the full encoded depth without progressive early-exit:

- `logical_bytes = n_rows * active_plane_count` (full depth, no progressive skipping).
- `logical_GBps` is the reported throughput for this fixed-depth layout.
- **Speedup vs. raw**: `speedup_from_encoding = logical_GBps(fixed_encoded) / logical_GBps(raw)`.
  - If `speedup_from_encoding > 1`, the encoding layout (fewer planes, better cache behavior) is faster.
  - If `speedup_from_encoding < 1`, the encoding has overhead that outweighs plane reduction (rare, investigate).
- **Speedup vs. progressive**: `speedup_from_early_exit = logical_GBps(progressive) / logical_GBps(fixed_encoded)`.
  - If `speedup_from_early_exit > 1`, progressive k-limiting or early-exit reduces the average logical byte count.

### Progressive K-Limited or Capped-k

For Exp3 (k-limited SUM) or Exp4 (capped-k COUNT):

- `logical_bytes = n_rows * k_avg`, where `k_avg` is the average number of subcolumns/planes read per row.
- For **Exp3 progressive SUM**: `k_avg = refinement_depth + 1` (constant across rows).
- For **Exp4 capped-k COUNT**: `k_avg` varies per row and per threshold; report as Q(k), D(k), U(k) if applicable, and compute average.
- `logical_GBps = (n_rows * k_avg / (ms_per_iter / 1000.0)) / 1e9`.
- **Error/approximation metrics** (e.g., SUM relative error, COUNT interval bounds) are reported separately, not folded into throughput.

---

## CSV Column Specification

### Required Columns (All Benchmarks)

| Column Name | Semantics | Example Values |
|---|---|---|
| `experiment` | Benchmark suite identifier | `exp3_progressive_sum`, `exp4_progressive_filter` |
| `dataset` | Synthetic or real dataset name | `uniform`, `sensor`, `synthetic_nyc_taxi` |
| `n` | Number of rows scanned | 100000000 |
| `ms_per_iter` | Median wall-clock time per iteration in milliseconds | 50.123 |
| `rows_per_sec` | Rows per second (`n * 1000.0 / ms_per_iter`) | 1.99e9 |
| `logical_bytes` | Logical bytes scanned (derived; for reference) | 200000000 |
| `logical_GBps` | Logical throughput in GB/s | 3.98 |
| `estimated_physical_bytes` | Estimated HBM/deep-cache bytes (or `NA` if unknown) | 100000000 or `NA` |
| `estimated_physical_GBps` | Estimated physical throughput in GB/s (or `NA`) | 1.99 or `NA` |
| `device` | GPU model | `NVIDIA H200`, `NVIDIA H100 NVL` |
| `validated` | Boolean or enum; indicates correctness gate pass | `true`, `false`, `encoded_mismatch_ok` |

### Exp3-Specific Columns

| Column Name | Semantics | Example Values |
|---|---|---|
| `aggregation` | Aggregation operation | `sum`, `count`, `avg`, `variance` |
| `mode` | Data layout / encoding mode | `raw_fp64`, `encoded_exact`, `encoded_progressive` |
| `k` or `refinement_depth` | Number of subcolumns or planes read | 1, 2, 3, ..., max_active_plane_count |
| `gpu_approx_sum` / `cpu_approx_sum` / `exact_sum` | Result values for validation | (numeric) |
| `abs_exact_cpu_diff` | Absolute difference from exact reference | (numeric or `NA`) |

### Exp4-Specific Columns

| Column Name | Semantics | Example Values |
|---|---|---|
| `threshold` | Predicate threshold value | 0, 500, 1000 |
| `selectivity` | Estimated fraction of rows passing predicate | 0.5, 0.1, 0.99 |
| `gpu_count` | GPU kernel result | (integer count) |
| `cpu_encoded_count` | CPU encoded-artifact reference | (integer count) |
| `cpu_raw_count` | CPU raw FP64 reference | (integer count) |
| `gpu_eq_cpu_encoded` | Correctness gate (`gpu_count == cpu_encoded_count`) | `true`, `false` |
| `count_lower` / `count_upper` / `count_abs_error_bound` | Interval bounds (for capped-k) | (integers) |
| `avg_planes_read` | Average planes read per row (estimated or actual) | (float or `NA`) |
| `max_planes_read` | Maximum planes read in this run | (integer) |

---

## Legacy Column Aliasing and Renaming

### `physical_GBps` → `estimated_physical_GBps`

**Status**: In Exp4 outputs, the column `physical_GBps` currently exists and has been used to mean estimated physical throughput.

**Action**:
- **Recommended**: Rename to `estimated_physical_GBps` to avoid confusion with actual profiler-measured physical bandwidth.
- **Interim**: If renaming across all historical results is impractical, document in the CSV header comment or a README that `physical_GBps` **is not a measured quantity** and should be interpreted as an estimate based on rowpack16 counters.
- **Report requirement**: Every paper plot or claim referencing `physical_GBps` must include a disclaimer or caption stating: *"Estimated physical throughput based on kernel instrumentation, not HBM profiler data."*

### `logical_bytes_per_iter` vs. `logical_bytes`

Both names appear in the codebase; standardize to `logical_bytes` (shorter, consistent with `estimated_physical_bytes`).

### `rows_per_sec` vs. `billion_rows_per_sec`

If both are reported, they must be consistent: `billion_rows_per_sec = rows_per_sec / 1e9`. Avoid duplication; prefer `rows_per_sec` and compute `billion_rows_per_sec` in post-processing if needed for plots.

---

## Acceptance Criteria Validation

### For Exp3 Outputs

- [ ] Every row has consistent `logical_bytes = n * subcolumns_at_k` and `logical_GBps = logical_bytes / (ms_per_iter / 1000) / 1e9`.
- [ ] If `estimated_physical_bytes` is present, it is clearly labeled and accompanied by a note on how it was derived.
- [ ] `rows_per_sec = n * 1000 / ms_per_iter` for every row.
- [ ] For raw FP64 runs, `logical_bytes = n * 8`.
- [ ] For encoded runs, `logical_bytes = n * active_plane_count_at_k`.
- [ ] At least one raw FP64 baseline is included in the same sweep for speedup comparison.

### For Exp4 Outputs

- [ ] Every row has `logical_bytes = n * avg_planes_read` and `logical_GBps` calculated consistently.
- [ ] `gpu_eq_cpu_encoded` gates correctness; rows with `false` are marked `validated=false`.
- [ ] `estimated_physical_bytes` and `estimated_physical_GBps` (formerly `physical_GBps`) are documented as estimates.
- [ ] `rows_per_sec` matches the expected rate for the dataset and filter selectivity.
- [ ] If `avg_planes_read` is estimated rather than instrumented, this is noted in the benchmark output or README.

### For All Outputs

- [ ] No plot axis or report text confuses logical throughput with physical HBM bandwidth without explanation.
- [ ] If two runs have different `logical_GBps` but should be compared (e.g., different k values), the caption explains whether the difference is due to encoding benefits, early-exit, or other factors.
- [ ] The CSV header includes optional comments (e.g., `# logical_GBps: bytes per row * rows per second / 1e9`) so that downstream consumers can understand the semantics.

---

## Implementation Guidance

### Benchmark Code Changes

1. **Calculate and validate** `logical_bytes` before writing CSV:
   ```cpp
   double logical_bytes = n_rows * logical_subcolumns_or_planes;
   double logical_GBps = (logical_bytes / (ms_per_iter / 1000.0)) / 1e9;
   ```

2. **Instrument or estimate** `estimated_pack_load_bytes`:
   - If the kernel has counters for memory operations, sum them and scale by load width.
   - If only heuristic estimation is available, add a comment in the CSV or report.

3. **Validate consistency**:
   - Check that `logical_GBps >= 0`.
   - Check that `estimated_physical_GBps <= logical_GBps` (unless there is a specific reason for cache-line reading overhead).
   - If `estimated_physical_GBps > logical_GBps`, log a warning and investigate.

4. **Document assumptions** in the benchmark runner or handoff README:
   - How is `ms_per_iter` computed (sum, median, mean)?
   - Are warmup iterations included?
   - Is host-side overhead excluded?
   - How is `estimated_pack_load_bytes` derived (kernel counters, heuristic, profiler)?

### Plotting and Reporting

1. **Axis labels** must distinguish logical from physical:
   - ✅ `Logical Throughput (GB/s)`, `Estimated Physical Throughput (GB/s)`
   - ❌ `Throughput (GB/s)` (ambiguous)
   - ❌ `Physical Throughput (GB/s)` (implies measured, not estimated)

2. **Figure captions** must explain why logical throughput can exceed nominal HBM bandwidth:
   - Example: *"Logical throughput (3.2 GB/s) exceeds H200 HBM read bandwidth (141 GB/s) when scaled to effective plane count, but estimated physical throughput (0.8 GB/s) reflects actual memory operations. The gap illustrates cache effectiveness for this access pattern."*

3. **Speedup claims** must cite the baseline and metric used:
   - ✅ *"Encoding achieves 2.5× speedup in logical throughput vs. raw FP64."*
   - ✅ *"Capped-k COUNT provides 1.8× speedup in logical throughput at k=4 vs. full-depth k=8."*
   - ❌ *"Encoding achieves 2.5× speedup."* (speedup in what metric?)

---

## Future Extensions

### Profiler-Measured Physical Throughput

If NCU or Nsight Compute profiler data becomes available:

1. Add columns: `measured_hbm_bytes`, `measured_physical_GBps` with prefix `profiler_`.
2. Report both estimated and measured in the same CSV or as a separate profiler CSV with join keys.
3. Update plots and captions to compare estimated vs. measured.
4. Document the profiler command and environment used.

### Multi-GPU / Distributed Scaling

If Exp3 or Exp4 scales to multiple GPUs:

1. Clarify whether `logical_GBps` and `estimated_physical_GBps` are:
   - Per-GPU aggregates (divide total by GPU count).
   - Total system throughput (sum across GPUs).
2. Add a `gpu_count` column (or clarify it if already present).
3. Document scaling efficiency and any per-GPU variance.

---

## References

- Exp3 benchmark code: `benchmarks/experiment3/bench_progressive_aggregation.cu`
- Exp4 benchmark code: `benchmarks/experiment4/bench_progressive_filter.cu`
- Previous metrics discussions: `handoff/2026-04-30_Exp4_Metrics_Patch_Before_Early_Exit_Sweep_Handoff.md`
- Data format contract: `research/2026-04-30_Exp3_Exp4_Required_Data_Format_Contract.md`

---

## Sign-Off

This contract is the authoritative source for throughput metric semantics in Exp3 and Exp4 outputs. All future sweeps, plots, and papers must conform to this contract or explicitly document deviations and their rationales.

**Version**: v1  
**Status**: Approved for use in Exp3/Exp4 pipelines (PV1-8A)  
**Date**: 2026-04-30
