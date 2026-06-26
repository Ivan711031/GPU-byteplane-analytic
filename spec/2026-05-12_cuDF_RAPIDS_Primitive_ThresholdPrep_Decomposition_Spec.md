# cuDF/RAPIDS Primitive Threshold-Prep Decomposition Spec

Date: 2026-05-12
Scope: Measure and decompose host-side threshold preparation. No encoder format change yet.
Status: Diagnostic implementation spec. Decides whether threshold-prep can become an artifact/static cost or must remain a per-query cost.

## References

- [Malloc-Hoist Orchestration Fix Spec](./2026-05-12_cuDF_RAPIDS_Primitive_MallocHoist_Orchestration_Fix_Spec.md)
- Malloc-hoist report: `research/2026-05-12_cuDF_RAPIDS_Primitive_MallocHoist_Orchestration_Fix_Report.md`
- [Warm-Resident Per-Query Spec](./2026-05-12_cuDF_RAPIDS_Primitive_WarmResident_PerQuery_Spec.md)
- [Per-Query Gap Profile Spec](./2026-05-12_cuDF_RAPIDS_Primitive_PerQuery_Gap_Profile_Spec.md)

## 1. Why This Spec Exists

The malloc-hoist and residual-overhead work proved that the compact-U prepared-query GPU hot loop is fast:

- sensor exact `dev_buff_exp3`, four-row warm gate: `per_query_ms_median ~= 0.56-0.60 ms`
- cuDF per-query median: `~1.35-1.41 ms`
- prepared hot-loop speedup: `> 2.3x` on all four rows

But the current `threshold_prep_ms` column is large:

| selectivity | k | threshold_prep_ms | prepared per_query_ms | threshold + prepared | cuDF per_query_ms |
|---:|---:|---:|---:|---:|---:|
| 90 | 1 | 2.040715 | 0.595849 | 2.636564 | 1.409280 |
| 90 | 2 | 1.444493 | 0.595115 | 2.039608 | 1.370176 |
| 99 | 1 | 0.914363 | 0.558601 | 1.472964 | 1.367424 |
| 99 | 2 | 0.905632 | 0.561804 | 1.467436 | 1.345632 |

So the current result supports a **prepared-predicate hot-loop** claim, but not yet a **threshold-inclusive ad-hoc query** claim.

The open question:

> Is `threshold_prep_ms` mostly threshold-static / artifact-static work that can be moved to artifact load or encoder metadata, or is it mostly truly threshold-dependent work that must be paid for every new predicate?

This spec answers that with instrumentation and a cost model. It does not change the artifact format yet.

## 2. Current Code Under Test

Target worktree/branch:

```text
.worktrees/compact-u-verdict-gate
feature/compact-u-verdict-gate
```

Relevant file:

```text
benchmarks/experiment4/byteplane_count_gt.cu
```

Relevant functions:

- `prepare_resident_query(...)`
- `compute_threshold_bytes(...)`
- helpers:
  - `le_bytes_to_u64(...)`
  - `subtract_le_bytes(...)`
  - `extract_plane_bytes_from_combined_le(...)`

Current important detail:

`threshold_prep_ms` is assigned immediately after `compute_threshold_bytes(...)`, before the later loop that builds:

- `segment_mode`
- `active_plane_count`
- `total_bits_by_segment`
- `h_threshold_flat`
- `h_threshold_combined`

Therefore the existing `threshold_prep_ms` is not the full `prepare_resident_query(...)` cost. It mostly measures `compute_threshold_bytes(...)`.

## 3. Goal

Produce a measurement-backed decomposition of threshold preparation into:

| Bucket | Meaning |
|---|---|
| `threshold_static_ms` | work independent of threshold `T`, movable to artifact load or encoder metadata |
| `threshold_dependent_ms` | work that truly depends on `T` and must run for a new predicate |
| `threshold_classify_ms` | `T` vs per-segment min/max classification (`all_qualified`, `all_disqualified`, mixed) |
| `threshold_encode_ms` | converting `T` into segment code space |
| `threshold_base_ms` | computing or materializing segment base-shifted code |
| `threshold_subtract_extract_ms` | `combined_T = T_code - base_code`, plane-byte extraction, `threshold_combined` |
| `threshold_pack_ms` | building `segment_mode`, `active_plane_count`, `total_bits_by_segment`, flat arrays |
| `threshold_alloc_count` | number of CPU heap allocations in threshold prep, if practical to count |
| `threshold_total_prepare_ms` | full `prepare_resident_query(...)` wall time |

Then decide whether the next step should be:

1. artifact/encoder static-context export,
2. CPU allocation/preallocation cleanup,
3. GPU-side threshold-prep experiment,
4. or prepared-predicate framing only.

## 4. Non-Goals

- No artifact format change.
- No encoder output change.
- No new CUDA kernel.
- No GPU-side threshold-prep implementation.
- No matrix expansion beyond sensor exact `dev_buff_exp3`.
- No attempt to optimize cuDF.
- No change to compact-U correctness semantics.
- No push.

## 5. Required Instrumentation

### 5.1 Add threshold-prep timing fields

Add diagnostic fields to `ByteplaneCountGtResult` and JSON/CSV output.

Required new columns:

```text
threshold_total_prepare_ms
threshold_compute_bytes_ms
threshold_classify_ms
threshold_encode_ms
threshold_base_ms
threshold_subtract_extract_ms
threshold_pack_ms
threshold_static_candidate_ms
threshold_dependent_candidate_ms
```

Optional if easy:

```text
threshold_alloc_count
threshold_mixed_segments
threshold_allq_segments
threshold_alld_segments
```

Naming can be adjusted if there is an existing local convention, but the report must include an explicit mapping.

### 5.2 Preserve old `threshold_prep_ms`

Keep `threshold_prep_ms` emitted for continuity.

For this spec, also document exactly what it measures after the patch. Preferred:

```text
threshold_prep_ms = threshold_total_prepare_ms
```

If changing that field risks downstream breakage, leave it unchanged and make the new `threshold_total_prepare_ms` authoritative.

### 5.3 Instrument inside `compute_threshold_bytes`

Break down this loop:

```cpp
for each segment:
    read segment min/max
    classify all-qualified/all-disqualified/mixed
    compute scale/base metadata
    encode threshold into segment code space
    compute or materialize base-shifted code
    subtract
    extract plane bytes
    compute threshold_combined
```

The implementation does not need perfect nanosecond attribution. It needs stable enough bucket timings to decide whether static extraction is worth doing.

Minimum acceptable buckets:

- classify
- encode T
- base/materialization
- subtract/extract
- pack/flatten
- total

### 5.4 Identify static candidates without changing behavior

For each segment, classify the following as static or threshold-dependent in comments/report:

Static candidates:

- `scale_exponent = -fractional_bits`
- `uses_bounded_precision`
- `integer_base_u64`
- `base_shifted_le` when it does not depend on `T`
- `total_bits = fractional_bits + integer_offset_bits`
- `active_plane_count` for fixed `k`
- fixed output offsets and flat-array layout

Threshold-dependent:

- `threshold_fp64 < seg_min`
- `threshold_fp64 >= seg_max`
- `t_code_le`
- `combined_T_le`
- `threshold_bytes`
- `threshold_combined`
- `segment_mode`, because it depends on all-qualified/all-disqualified/mixed classification

### 5.5 CPU allocation audit

Without adding heavy machinery, identify obvious CPU allocation sources:

- `std::vector<SegmentThresholdInfo> result`
- per-segment `threshold_bytes.resize(...)`
- `t_code_le`
- `base_shifted_le`
- `combined_T_le`
- output flat arrays in `PreparedResidentQuery`

If practical, add a lightweight counter for known vector creation/resizes inside `compute_threshold_bytes` and `prepare_resident_query`. Do not override global `operator new`.

## 6. Benchmark Plan

Use the existing H200-only Slurm discipline.

Run only sensor exact `dev_buff_exp3`.

### 6.1 Single-cell diagnostic

First run:

| Axis | Value |
|---|---|
| dataset | `sensor` |
| selectivity | `99` |
| k | `1` |
| mode | `--compact-u-refine-raw` |
| repeat | `5` |

Purpose:

- confirm instrumentation compiles
- confirm correctness still passes
- capture threshold-prep decomposition for the same row used in prior reports

### 6.2 Four-row diagnostic matrix

If the single-cell diagnostic succeeds, run the existing warm-resident matrix:

| Axis | Values |
|---|---|
| dataset | `sensor` |
| selectivity | `90,99` |
| k | `1,2` |
| artifact | exact `dev_buff_exp3` |
| mode | `--compact-u-refine-raw` |

Do not expand to other datasets.

## 7. Output Requirements

Write:

```text
research/2026-05-12_cuDF_RAPIDS_Primitive_ThresholdPrep_Decomposition_Report.md
```

The report must include:

### 7.1 Source-level decomposition

A table:

| operation | depends on T | depends on k | depends on segment metadata | depends on raw min/max | static candidate |
|---|---|---|---|---|---|

Include rows for:

- classify min/max
- scale exponent
- threshold encode
- base encode/materialization
- subtract
- plane-byte extraction
- `threshold_combined`
- `segment_mode`
- `active_plane_count`
- `total_bits_by_segment`
- flatten `threshold_bytes`

### 7.2 Timing table

For each of the four rows:

| sel | k | threshold_total_prepare_ms | classify | encode | base | subtract_extract | pack | prepared_hot_loop_ms | cudf_ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Also include:

```text
threshold_plus_hot_loop_ms = threshold_total_prepare_ms + per_query_ms_median
threshold_inclusive_speedup = cudf_per_query_ms_median / threshold_plus_hot_loop_ms
prepared_speedup = cudf_per_query_ms_median / per_query_ms_median
```

### 7.3 Static-vs-dependent estimate

For each row:

```text
static_candidate_fraction = threshold_static_candidate_ms / threshold_total_prepare_ms
dependent_fraction = threshold_dependent_candidate_ms / threshold_total_prepare_ms
```

If exact static timing is not yet measurable without refactor, provide a conservative estimate and clearly label it as estimated.

### 7.4 Decision

Choose exactly one:

- `recommend: write artifact static-context spec`
- `recommend: write CPU threshold-prep allocation/preallocation spec`
- `recommend: write GPU-side threshold-prep experiment spec`
- `recommend: keep prepared-predicate framing; do not chase threshold-inclusive claim now`

Decision thresholds:

- If `static_candidate_fraction >= 0.50`, prefer artifact static-context spec.
- If allocation/preallocation overhead is visibly dominant, prefer CPU allocation/preallocation spec.
- If `threshold_dependent_candidate_ms` remains too high after static accounting and is mostly arithmetic per segment, consider GPU-side threshold-prep experiment.
- If static fraction is `< 0.20` and dependent work dominates, keep prepared-predicate framing.

## 8. Definition of Done

This spec is done when:

1. The code emits threshold-prep decomposition fields without breaking existing CSV consumers.
2. Single-cell diagnostic completes on H200 with `ExitCode=0:0`.
3. Four-row diagnostic CSV exists for sensor `sel={90,99}`, `k={1,2}`.
4. All rows have `cudf_count_matches = true`.
5. The report includes source-level dependency classification.
6. The report includes timing breakdown and threshold-inclusive speedup.
7. The report makes exactly one next-step recommendation from §7.4.
8. No artifact/encoder format change was made.
9. No push was performed.

## 9. One-Sentence Summary

Instrument and decompose the host threshold-preparation path to decide whether the remaining `0.9-2.0 ms` per-query cost can be moved into artifact/encoder static metadata, reduced by CPU allocation cleanup, or must be treated as the cost that separates prepared-predicate wins from ad-hoc threshold-inclusive queries.
