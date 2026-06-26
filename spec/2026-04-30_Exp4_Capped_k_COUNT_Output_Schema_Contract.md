# Exp4 Capped-k COUNT Output Schema and Metric Contract

**Date**: 2026-04-30  
**Scope**: Exp4 progressive COUNT WHERE x > threshold predicate branch  
**Status**: v1 contract  
**Depends on**: `spec/2026-04-30_Throughput_Metric_Schema_Contract.md` (PV1-8A)  
**Implementation Handoff**: GitHub issues #4 (PV1-1), #5 (PV1-2)

---

## Executive Summary

This document defines the **semantic contract** for capped-k COUNT output in Exp4. The contract ensures:

1. **Deterministic classification** of rows into three categories (Q, D, U) after reading at most k planes.
2. **Error bound semantics** where `count_abs_error_bound = U(k)` is the maximum error between exact COUNT and any COUNT consistent with k planes.
3. **Full-depth alignment** where capped-k with k ≥ active_plane_count matches exact encoded-domain COUNT.
4. **Correctness gatekeeping** that distinguishes GPU kernel correctness (`gpu_count == cpu_encoded_count`) from raw fidelity (`cpu_encoded_count ≈ cpu_raw_count`).
5. **Throughput accounting** using the contract from PV1-8A (logical vs. estimated physical).
6. **Per-round diagnostics** supporting survival and pack-utilization analysis.

---

## 1. Core Definitions: Q, D, U Classification

### 1.1 Setup

For a given row with value `v_raw` (in raw FP64 semantics):

- **Threshold**: `t` (in encoded threshold semantics).
- **After k planes read**: The row has partial information as a pair of bounds:
  - `lower_bound(k)` = minimum possible value with k MSB planes read.
  - `upper_bound(k)` = maximum possible value with k MSB planes read.

Both bounds are in the same threshold-encoding space as `t`.

### 1.2 Three Categories After Capped-k

**Q(k) – Certainly Qualified:**

```
count += 1  IF lower_bound(k) > t
```

A row is in Q if its lower bound exceeds the threshold, meaning **all possible values consistent with k planes are > t**.

**D(k) – Certainly Disqualified:**

```
count += 1  IF upper_bound(k) <= t
```

A row is in D if its upper bound does not exceed the threshold, meaning **all possible values consistent with k planes are <= t**.

**U(k) – Uncertain:**

```
count += 1  IF lower_bound(k) <= t < upper_bound(k)
```

A row is in U if the threshold falls between the bounds, meaning **k planes do not determine the predicate outcome**.

### 1.3 Full-Depth Equivalence

When `k >= active_plane_count` (per segment), all rows are fully resolved:

```
U(k) = 0
Q(k) + D(k) = n
Q(k) + D(k) + U(k) == gpu_count == cpu_encoded_count
```

This is the **full-depth correctness gate**.

### 1.4 Error Bound Semantics

The error bound is defined in **count-space**, not value-space:

```
count_abs_error_bound = U(k)
```

Interpretation:
- The exact COUNT (full-depth) lies in the interval `[Q(k), Q(k) + U(k)]`.
- The maximum error is `U(k)` rows.
- There is **no tighter bound** on exact COUNT without reading beyond k planes.

Example:
```
k = 3 planes on uniform dataset (8 planes total):
Q(3) = 40000000, U(3) = 5000000, D(3) = 55000000
exact COUNT = 42000000

Error interval: [40000000, 45000000] ✓ (exact COUNT is in this interval)
count_abs_error_bound = 5000000 ✓
```

---

## 2. CSV Output Schema

### 2.1 Main Output CSV

#### Mandatory Columns (All Capped-k COUNT Runs)

| Column | Semantics | Type | Example |
|---|---|---|---|
| `experiment` | Benchmark identifier | string | `exp4_progressive_filter` |
| `dataset` | Dataset name | string | `uniform`, `sensor`, `heavy_tailed`, `zipfian` |
| `artifact_root` | Path to encoded artifact | string | `/work/.../dev_buff_exp3/uniform` or `dev_buff_exp4_p3/...` |
| `precision_mode` | Encoding mode | enum | `exact`, `bounded` |
| `precision_decimals` | For bounded mode, fractional bits | int | `3`, `6`, `NA` |
| `threshold` | Predicate threshold value (encoded space) | float64 | `0.5`, `1000.0` |
| `selectivity` | Expected fraction passing predicate | float | `0.01`, `0.5`, `0.99` |
| `n` | Total rows scanned | int64 | `100000000` |
| `iters` | Number of kernel iterations | int | `20` |
| `warmup` | Number of warmup iterations (not timed) | int | `3` |
| `ms_per_iter` | Wall-clock time per iteration (ms) | float | `50.123` |
| `rows_per_sec` | `n * 1000 / ms_per_iter` | float64 | `1.996e9` |
| `max_plane_count` | Total planes in artifact | int | `8`, `10` |
| `segment_rows` | Rows per segment | int | `4096` |

#### Capped-k Classification Columns

| Column | Semantics | Type | Formula |
|---|---|---|---|
| `max_filter_planes` | Cap imposed on this run; -1 if disabled | int | `3`, `5`, `-1` |
| `certainly_qualified` | Q(k) | int64 | See §1.2 |
| `certainly_disqualified` | D(k) | int64 | See §1.2 |
| `uncertain` | U(k) | int64 | See §1.2 |
| `count_lower` | Lower bound on exact COUNT | int64 | `Q(k)` |
| `count_upper` | Upper bound on exact COUNT | int64 | `Q(k) + U(k)` |
| `count_abs_error_bound` | Maximum error in COUNT | int64 | `U(k)` |

#### GPU/CPU Validation Columns

| Column | Semantics | Type | Notes |
|---|---|---|---|
| `gpu_count` | COUNT result from GPU kernel | int64 | Should equal `certainly_qualified` or `certainly_qualified + uncertain` depending on tie-breaking |
| `cpu_encoded_count` | CPU reference using same encoded artifact + threshold | int64 | **Correctness gate**: `gpu_count == cpu_encoded_count` must be true |
| `cpu_raw_count` | CPU reference scanning raw FP64 | int64 | For raw fidelity; does not gate kernel correctness |
| `gpu_eq_cpu_encoded` | `gpu_count == cpu_encoded_count` | bool | **Blocking gate**: must be `true` for run to be `validated=true` |
| `cpu_encoded_eq_cpu_raw` | `cpu_encoded_count == cpu_raw_count` | bool | **Non-blocking**: can be `false` for bounded-precision artifacts |
| `raw_count_abs_error` | `abs(cpu_encoded_count - cpu_raw_count)` | int64 | Fidelity metric; not a correctness failure |
| `raw_count_rel_error` | `raw_count_abs_error / max(cpu_raw_count, 1)` | float | Fidelity metric |

#### Pack-Utilization Columns

| Column | Semantics | Type | Notes |
|---|---|---|---|
| `fully_resolved_packs` | Number of 16-row packs with U=0 after capped-k | int64 | Indicator of rowpack16 efficiency |
| `partially_active_packs` | Number of 16-row packs with U>0 | int64 | Packs still processing unresolved rows |
| `avg_active_rows_per_active_pack` | `active_rows / partially_active_packs` | float | Average unresolved rows per partially-active pack |
| `useful_row_fraction` | Fraction of total rows in active packs | float | `[0, 1]`; indicates work distribution |

#### Throughput Columns (Inherit from PV1-8A)

| Column | Semantics | Type | Notes |
|---|---|---|---|
| `logical_bytes` | `n * k_avg` (bytes per row × avg planes read) | int64 | See PV1-8A §1.1 |
| `logical_GBps` | `logical_bytes / (ms_per_iter / 1000) / 1e9` | float | Logical throughput; not physical bandwidth |
| `avg_planes_read_per_total_row` | Average planes read across all rows | float | `[0, max_filter_planes]` |
| `max_planes_read` | Maximum planes read in this run | int | Should equal `max_filter_planes` or less if hit full-depth early |
| `estimated_pack_load_bytes` | Estimated HBM bytes from rowpack16 counters | int64 | See PV1-8A §2 |
| `estimated_physical_GBps` | `estimated_pack_load_bytes / (ms_per_iter / 1000) / 1e9` | float | Estimated, not measured |

#### Metadata Columns

| Column | Semantics | Type | Notes |
|---|---|---|---|
| `device` | GPU model | string | `NVIDIA H200`, `NVIDIA H100 NVL` |
| `gpu_tag` | GPU identifier or index | string | For multi-GPU runs |
| `job_id` | Slurm job ID or batch run ID | string | Traceability |
| `validated` | Run passed correctness gate | bool | `true` iff `gpu_eq_cpu_encoded == true` |
| `load_strategy` | Kernel memory strategy | string | `rowpack16`, `rowpack32` |
| `benchmark` | Benchmark runner identifier | string | `progressive_filter` |
| `kernel_path` | GPU kernel code path | string | `progressive_filter_rowpack16_passive` |

---

### 2.2 Per-Round Survival Sidecar CSV

#### Purpose

Optional, written to separate file (e.g., `rounds_survival.csv` in same output directory).

Records survival and pack-utilization metrics for each k value from 1 to `max_filter_planes`.

#### Schema

| Column | Semantics | Type | Notes |
|---|---|---|---|
| `dataset` | Dataset name (join key) | string | Same as main CSV |
| `threshold` | Threshold value (join key) | float64 | Same as main CSV |
| `k` | Round number (1-indexed) | int | 1, 2, 3, ..., max_filter_planes |
| `qualified_at_k` | Rows newly resolved as qualified at round k | int64 | Cumulative count in main CSV: sum of `qualified_at_k` for all rounds |
| `disqualified_at_k` | Rows newly resolved as disqualified at round k | int64 | Cumulative count in main CSV: sum of `disqualified_at_k` |
| `uncertain_after_k` | Rows still uncertain after round k | int64 | This is U(k) in main CSV |
| `planes_read_per_round` | Total byte-plane read bytes in round k (across all rows) | int64 | Diagnostic; for bandwidth accounting |
| `pack_load_bytes_per_round` | Estimated rowpack16 loads in round k | int64 | Supports estimated physical throughput per round |

#### Interpretation

- Sum across rounds: `sum(qualified_at_k) + sum(disqualified_at_k) + uncertain_after_k_final = n`.
- `uncertain_after_k` should decrease monotonically as k increases.
- At full depth: `uncertain_after_k_final = 0`.

---

## 3. Full-Depth Equivalence and Validation

### 3.1 Full-Depth Capped-k Behavior

When `max_filter_planes >= max(segment.active_plane_count)`:

```
Capped-k kernel ≈ Exact encoded-domain early-exit kernel

Expected properties:
1. gpu_count == cpu_encoded_count == Q(k) + D(k)
2. U(k) == 0
3. All rows resolved; no ambiguity
4. Throughput may differ due to control flow or early-exit overhead
```

### 3.2 Validation Rule

Every capped-k run **must validate** at full depth:

1. **Add a full-depth run** to each sweep (set `max_filter_planes = -1` or a value exceeding max segment depth).
2. **Check**: `gpu_count_full == cpu_encoded_count`.
3. **If mismatch**: Investigate and resolve before proceeding to k-limited runs.

This ensures the CPU reference is trustworthy.

---

## 4. Correctness Gates and Fidelity Reporting

### 4.1 Blocking Correctness Gate

```
PASS: gpu_count == cpu_encoded_count
FAIL: gpu_count != cpu_encoded_count
```

**Action**: If FAIL, the run is marked `validated=false`. Do not include in precision-throughput curves or claims.

**Meaning**: The GPU kernel computed the encoded-domain COUNT incorrectly.

### 4.2 Non-Blocking Fidelity Checks

```
CHECK: cpu_encoded_count == cpu_raw_count
PASS: True (no raw data precision loss)
FAIL: False (bounded precision loses some information)
```

**Action**: If FAIL, report the error metrics (`raw_count_abs_error`, `raw_count_rel_error`) but do not fail the run.

**Meaning**: Bounded-precision encoding (p3, p6) intentionally trades raw FP64 fidelity for fewer planes. The COUNT is still correct in encoded-domain semantics.

### 4.3 CSV Flag Column

```
validated = (gpu_eq_cpu_encoded) ? "true" : "false"
```

Optional extended fields:
```
gpu_eq_cpu_encoded = gpu_count == cpu_encoded_count        # Blocking
cpu_encoded_eq_cpu_raw = cpu_encoded_count == cpu_raw_count  # Non-blocking
```

---

## 5. Semantics of k and Threshold Encoding

### 5.1 Threshold Encoding Space

All thresholds, bounds, and values are in the **threshold-encoding space**:

1. **Floating-point threshold** `t_float` (raw input).
2. **Byte-plane representation**: `t_encoded = to_byte_planes(t_float)`.
3. **After k planes**: Lower and upper bounds are formed by reading k MSB planes and filling omitted planes with 0 (lower) or all-1s (upper).

Example:
```
t_float = 500.0
t_encoded (8 planes) = [0x02, 0x0C, 0x12, ..., 0x00]  (big-endian)

After k=2 planes:
lower_bound = [0x02, 0x0C, 0x00, 0x00, ..., 0x00]  (omitted planes = 0)
upper_bound = [0x02, 0x0C, 0xFF, 0xFF, ..., 0xFF]  (omitted planes = 1)
```

### 5.2 Selectivity

`selectivity = cpu_raw_count / n`

This is the expected fraction of rows passing the predicate in raw FP64 semantics. It helps downstream analysis understand the workload (selective vs. permissive predicate).

---

## 6. Per-Round Instrumentation Requirements for #4 and #5

### 6.1 For #4 (PV1-1: Tracer Bullet)

**Minimal requirements:**
- [ ] Output `max_filter_planes`, `certainly_qualified`, `certainly_disqualified`, `uncertain` columns.
- [ ] At full depth: `uncertain == 0` and matches exact encoded COUNT.
- [ ] Smoketest passes: one exact artifact, one selectivity regime, one threshold.

### 6.2 For #5 (PV1-2: Q/D/U + Survival + Pack Metrics)

**Extended requirements:**
- [ ] All columns from §2.1.
- [ ] Per-round sidecar CSV (§2.2) if `--output-rounds-csv` is used.
- [ ] Pack-utilization metrics: `fully_resolved_packs`, `partially_active_packs`, `avg_active_rows_per_active_pack`, `useful_row_fraction`.
- [ ] Survival metrics: `qualified_per_round`, `disqualified_per_round` (tracked internally; exported to sidecar CSV).
- [ ] H200 smoke runs on 2 selectivities with different survival behavior (e.g., one selective, one permissive).

---

## 7. Legacy Column Aliasing

### 7.1 Column Name Consistency

For Exp4 capped-k outputs, use the following names (no legacy aliases):

| New (Standard) | Old (Not Used) | Notes |
|---|---|---|
| `certainly_qualified` | `qualified` | Emphasizes "certain" classification |
| `certainly_disqualified` | `disqualified` | Emphasizes "certain" classification |
| `uncertain` | `ambiguous`, `unresolved` | Standard in literature |
| `count_lower` | N/A | Clarity: lower bound on exact COUNT |
| `count_upper` | N/A | Clarity: upper bound on exact COUNT |
| `count_abs_error_bound` | `error_bound`, `abs_error` | Precision: absolute value, count-space |
| `gpu_eq_cpu_encoded` | N/A | Clarity: gate condition |
| `cpu_encoded_eq_cpu_raw` | N/A | Clarity: fidelity check |

### 7.2 Inheritance from Exp4-A (Exact Baseline)

Exp4-A outputs existing columns (§2.1, rows 1–14). For capped-k runs:
- Keep existing columns for backward compatibility.
- When `max_filter_planes < 0` (disabled), treat as full-depth exact; expect `uncertain = 0`.

---

## 8. Throughput Interpretation for Capped-k

### 8.1 Logical Throughput

```
logical_bytes = n * avg_planes_read_per_total_row
logical_GBps = (logical_bytes / (ms_per_iter / 1000)) / 1e9
```

**Interpretation**: Logical throughput reflects the planes actually read. As k decreases, fewer planes are read (on average), so logical throughput increases.

**Example**:
```
Exact (k=-1): avg_planes=6, logical_GBps=1500
Capped k=5: avg_planes=5, logical_GBps=1250 (lower because fewer planes, but work done is less)
Capped k=3: avg_planes=3, logical_GBps=750
```

### 8.2 Estimated Physical Throughput

```
estimated_physical_GBps = (estimated_pack_load_bytes / (ms_per_iter / 1000)) / 1e9
```

**Interpretation**: Estimated physical throughput accounts for rowpack16 load patterns. Early-exit and survival dynamics can create false-sharing effects; rowpack16 may load more packs than strictly necessary.

**Example**:
```
Capped k=5, survival high (U small):
  logical_GBps = 1250, estimated_physical_GBps = 900
  
Capped k=3, survival low (U large, many rows in partially-active packs):
  logical_GBps = 750, estimated_physical_GBps = 1100
  
The apparent "inversion" (physical > logical) suggests false-sharing or pack-level inefficiency.
```

---

## 9. Report and Documentation Requirements

### 9.1 Benchmark Runner README

Every capped-k sweep must include a README documenting:

1. **Artifacts used**: dataset name, artifact path, mode (exact/bounded).
2. **Threshold strategy**: list of thresholds and their selectivities.
3. **k range**: min and max `max_filter_planes` tested.
4. **Full-depth validation**: confirmation that full-depth runs passed `gpu_eq_cpu_encoded`.
5. **Survival patterns**: qualitative description of how U(k) changes across datasets (e.g., "uniform: fast convergence, heavy_tailed: slow").
6. **Pack-utilization trends**: note any false-sharing patterns observed.

### 9.2 CSV Header Comments

Optional: Include comments in CSV header lines explaining column semantics:

```csv
# Experiment: exp4_progressive_filter
# Dataset: uniform (synthetic, exact artifact)
# Threshold strategy: selectivity 1%, 50%, 99%
# Full-depth validation: PASS (gpu_count == cpu_encoded_count at k=-1)
#
# Q/D/U semantics: after reading k planes, rows are classified as:
#   Q(k) = lower_bound > threshold
#   D(k) = upper_bound <= threshold
#   U(k) = lower_bound <= threshold < upper_bound
# count_abs_error_bound = U(k)
#
dataset,artifact_root,...
```

### 9.3 Issue Handoff for #4 and #5

When implementation is complete, attach CSV samples and README to the issue so that downstream agents (#6, #7, #8) can understand the contract without re-reading this document.

---

## 10. Acceptance Criteria

### For Issue #3 (PV1-0 – This Document)

- [✓] A schema document defines the CSV columns for capped-k COUNT runs, including k, threshold, selectivity, gpu_count, cpu_encoded_count, Q, D, U, count_lower, count_upper, count_abs_error_bound, and throughput fields from PV1-8A.
  - **Location**: §2.1, §2.2, §8.

- [✓] The schema defines full-depth behavior: full-depth capped-k must match exact encoded-domain COUNT.
  - **Location**: §3.1, §3.2.

- [✓] The schema defines correctness gates: kernel correctness is gpu_count == cpu_encoded_count; raw fidelity is reported separately and does not fail the kernel.
  - **Location**: §4.1, §4.2, §4.3.

- [✓] The schema defines per-round or per-k survival fields needed by PV1-2.
  - **Location**: §2.2 (per-round sidecar CSV), §6.2 (requirements for #5).

- [✓] No large H200 sweep is required for this issue.
  - **Action**: This is a schema/documentation task only.

---

## 11. Implementation Roadmap

### Issue #4 (PV1-1): Tracer Bullet

Minimal implementation using this schema:
- [ ] Modify `bench_progressive_filter.cu` to cap active_plane_count on host.
- [ ] CPU mirror computes Q, D, U after the cap.
- [ ] Output Q, D, U columns (and counts).
- [ ] Smoketest: exact artifact, full-depth validation passes.
- [ ] No per-round sidecar CSV yet.

### Issue #5 (PV1-2): Survival + Pack Metrics

Extended implementation:
- [ ] Add pack-utilization tracking to CPU mirror.
- [ ] Compute `fully_resolved_packs`, `partially_active_packs`, etc.
- [ ] Optional per-round sidecar CSV.
- [ ] H200 smoke: 2 selectivity regimes with visible survival differences.

### Issue #6 (PV1-3): Full Synthetic Sweep

Production sweep using finalized schema:
- [ ] All datasets (sensor, uniform, heavy_tailed, zipfian).
- [ ] Exact and bounded artifacts (p3, p6).
- [ ] Multiple thresholds/selectivity points.
- [ ] GPU packing policy: batch into single H200 allocation.

### Issue #7 (PV1-4): Epsilon-to-kstar Join

Downstream analysis:
- [ ] Join capped-k results by dataset/threshold/k.
- [ ] Compute epsilon (error bound normalized).
- [ ] Map k* = argmin(k : U(k) <= epsilon).
- [ ] Plot precision-throughput curves.

---

## References

- **PV1-8A**: `spec/2026-04-30_Throughput_Metric_Schema_Contract.md` (throughput semantics)
- **Exp4-B1 Implementation Spec**: `spec/2026-04-30_Exp4-B1_Capped-k_COUNT_Survival_Instrumentation_Spec.md`
- **Paper Guide v3**: `research/2026-05-01_Paper_Guide_Progressive_Byteplane.md` (research narrative)
- **Benchmark Code**: `benchmarks/experiment4/bench_progressive_filter.cu`

---

## Sign-Off

This schema contract is the authoritative source for capped-k COUNT output semantics in Exp4. All implementations (#4, #5, #6, #7) and all paper claims must conform to this contract or explicitly document deviations.

**Version**: v1  
**Status**: Approved for implementation (PV1-0)  
**Date**: 2026-04-30
