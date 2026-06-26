# Exp6 FOR Segment Granularity Sensitivity Spec

Date: 2026-05-12
Status: Spec only. Do not implement in this task.
Recommended agent: `gpt-5.4 high` for first implementation pass; `gpt-5.4-mini`
is acceptable only for read-only inventory/reporting.

## 1. Research Question

The current paper-facing artifacts use `segment_size = 4096` as an engineering
default inherited from the Exp2/Exp3/Exp4 artifact contract.  This is reasonable
and already validated, but it is not an experimentally justified optimum.

Exp6 asks:

> How does FOR segment granularity affect byte-plane artifact size, active plane
> count, Q/D/U convergence, epsilon-to-k*, throughput, and per-segment overhead?

The goal is not to find one universal segment size.  The goal is to derive a
design rule:

> Smaller segments reduce local value range and may improve precision
> convergence, but too-small segments increase metadata, threshold preparation,
> and kernel bookkeeping overhead.  The useful point is distribution-dependent.

## 2. Motivation

The byte-plane representation is segment-local.  Each segment has its own base,
integer offset width, active plane count, and predicate classification behavior.
Therefore `segment_size` is a real physical design parameter, not just a file
format constant.

Expected tradeoff:

```text
smaller segment_size
  -> smaller local value range
  -> lower integer_offset_bits may be possible
  -> fewer active planes / faster U(k) convergence may be possible
  -> more segments
  -> more metadata
  -> more threshold classification work
  -> more host prep / kernel bookkeeping

larger segment_size
  -> fewer segments
  -> lower metadata/prep overhead
  -> larger local value range
  -> higher integer_offset_bits / active planes may be needed
  -> slower U(k) convergence
```

This directly addresses the paper subquestion:

> How do FOR parameters, especially segment size and integer bit-width, affect
> convergence speed?

## 3. Current Baseline State

Known current default:

```text
segment_size = 4096
```

Why it exists:

- aligns Exp2, Exp3, and Exp4 artifact metadata;
- compatible with current runtime loaders;
- used by v2 synthetic artifacts in current paper evidence;
- acceptable throughput in Exp3 compared with the earlier `segment_rows=1048576`
  synthetic throughput setting.

Existing evidence:

- Exp3 SUM compared `segment_rows=1048576` vs `4096`.
- `4096` preserved the trend and stayed roughly within 95% of the larger
  segment throughput for most depths.
- Exp4 COUNT has not had a complete segment-size sensitivity sweep.

Therefore:

- `4096` is a valid engineering default.
- It should not be described as the optimal segment size.
- Exp6 is needed if the paper wants a real design-space claim.

## 4. Scope

### In scope

1. Generate or locate equivalent artifacts at multiple `segment_size` values.
2. Measure artifact metadata and payload size.
3. Run Exp4 COUNT capped-k sweep for selected segment sizes.
4. Compute Q/D/U, `epsilon -> k*`, throughput, and representation fidelity.
5. Compare metadata/prep overhead if existing runners expose it.
6. Produce a report that recommends whether `4096` remains a good default.

### Out of scope

- No cuDF work.
- No exp2 branch inspection unless the assigned implementation agent is
  explicitly told to work on encoder provenance separately.
- No new encoder semantics.
- No side-table design.
- No G-ALP/cuSZ/ZFP baselines.
- No real-data validation in this spec.
- No claim that the best segment size is universal.

## 5. Input Assumptions

Use the current mainline bounded fixed-point artifact semantics documented in:

```text
research/2026-05-10_New_Encoder_Design_Note.md
research/2026-05-10_New_Encoder_Semantics_Contract.md
```

Do not inspect or modify `buff_encoder` in this spec.  If the implementation
agent cannot generate alternate segment-size artifacts from mainline tooling,
it should stop and write a blocker report rather than silently switching to a
different branch or encoder.

## 6. Experiment Matrix

### 6.1 Segment sizes

Initial smoke:

```text
segment_size = 1024, 4096, 16384
```

Full synthetic matrix after smoke:

```text
segment_size = 512, 1024, 2048, 4096, 8192, 16384
```

Optional extension if runtime remains cheap:

```text
segment_size = 32768, 65536
```

Do not include extremely small segment sizes unless there is a clear reason.
Very small segments can be dominated by metadata and threshold-prep overhead.

### 6.2 Datasets

Smoke:

```text
sensor
heavy_tailed
```

Reason:

- `sensor` is a favorable compact-range case.
- `heavy_tailed` is the hard distribution and likely exposes the segment-size
  tradeoff.

Full matrix:

```text
uniform
sensor
heavy_tailed
zipfian
```

### 6.3 Artifact precision

Smoke:

```text
p6
```

Full matrix:

```text
p4
p6
p8 where artifact generation remains within the active-plane ceiling
```

Do not force unavailable precision levels.  If a dataset/precision/segment-size
combination fails artifact generation because `active_plane_count` exceeds the
allowed ceiling, record it as a valid failure point.

### 6.4 Query

COUNT first:

```text
COUNT(*) WHERE x > threshold
```

Thresholds:

```text
target_selectivity = 50, 90, 99
```

Optional full selectivity grid, only after smoke:

```text
target_selectivity = 1, 5, 10, 25, 50, 75, 90, 95, 99
```

### 6.5 k sweep

For every generated artifact:

```text
k = 1 .. max_plane_count
```

Do not hard-code max plane count.  Read it from the artifact manifest.

## 7. Required Measurements

### 7.1 Artifact-level fields

For every `(dataset, precision, segment_size)`:

```text
dataset
artifact_label
segment_size
value_count
segment_count
bytes_per_row
main_artifact_bytes
side_artifact_bytes
total_artifact_bytes
active_plane_count_min
active_plane_count_mean
active_plane_count_p50
active_plane_count_p95
active_plane_count_max
integer_offset_bits_min
integer_offset_bits_mean
integer_offset_bits_p50
integer_offset_bits_p95
integer_offset_bits_max
fractional_bits
generation_status
generation_failure_reason
```

### 7.2 Fidelity fields

For every `(dataset, precision, segment_size, selectivity)`:

```text
threshold
raw_count
encoded_full_depth_count
count_abs_error
selectivity_drift_pp
count_verdict
raw_sum
encoded_sum
sum_abs_error
sum_rel_error
sum_verdict
```

COUNT fidelity must remain separate from GPU kernel correctness.

### 7.3 COUNT execution fields

For every `(dataset, precision, segment_size, selectivity, k)`:

```text
k
max_plane_count
ms_per_iter
rows_per_sec
avg_planes_read_per_total_row
estimated_pack_load_bytes
Q_k
D_k
U_k
count_lower
count_upper
execution_rel_error_bound
gpu_count
cpu_encoded_count
gpu_eq_cpu_encoded
fully_resolved_packs
partially_active_packs
avg_active_rows_per_active_pack
useful_row_fraction
```

### 7.4 Epsilon-to-kstar fields

For every `(dataset, precision, segment_size, selectivity, epsilon)`:

```text
epsilon
kstar
kstar_status
rows_per_sec_at_kstar
speedup_vs_full_depth_encoded
avg_planes_read_at_kstar
estimated_pack_load_bytes_at_kstar
U_kstar
execution_rel_error_bound_at_kstar
```

Required epsilon values:

```text
epsilon = 0
epsilon = 1e-5
epsilon = 1e-4
epsilon = 1e-3
epsilon = 1e-2
```

State clearly whether epsilon is relative to encoded full-depth count or raw
count.  Preferred for Exp6:

```text
execution epsilon is relative to encoded full-depth COUNT
```

Representation fidelity is a separate column.

### 7.5 Optional overhead fields

If existing tools expose them without new CUDA work:

```text
threshold_prep_ms
segment_classification_ms
metadata_h2d_ms
kernel_only_ms
```

These are useful because smaller segments increase segment count.  If not
available, do not block the first Exp6 report; record the missing overhead as a
limitation.

## 8. Deliverables

### 8.1 Smoke report

After the smoke matrix:

```text
research/2026-05-12_Exp6_FOR_Segment_Granularity_Smoke_Report.md
```

It must answer:

1. Can alternate segment-size artifacts be generated from mainline tooling?
2. Do the generated artifacts load in Exp4 COUNT?
3. Does `4096` remain plausible?
4. Is the full matrix worth running?

### 8.2 Full result tables

```text
results/exp6/segment_granularity_artifact_summary.csv
results/exp6/segment_granularity_count_sweep.csv
results/exp6/segment_granularity_epsilon_kstar.csv
results/exp6/segment_granularity_fidelity.csv
```

### 8.3 Final report

```text
research/2026-05-12_Exp6_FOR_Segment_Granularity_Sensitivity_Report.md
```

Required report sections:

1. Experiment matrix and generation status.
2. Segment size vs artifact size.
3. Segment size vs active plane count / integer offset bits.
4. Segment size vs Q/D/U convergence.
5. Segment size vs epsilon-to-k*.
6. Segment size vs throughput.
7. Segment size vs fidelity drift.
8. Recommended default and decision rule.
9. Caveats and follow-up.

### 8.4 Optional figures

```text
results/exp6/plots/segment_size_vs_bytes_per_row.png
results/exp6/plots/segment_size_vs_active_planes.png
results/exp6/plots/segment_size_vs_kstar.png
results/exp6/plots/segment_size_vs_rows_per_sec_at_kstar.png
results/exp6/plots/segment_size_tradeoff_table.png
```

Keep figures simple.  Do not mix all datasets, precisions, selectivities, and
segment sizes in one unreadable scatter plot.

## 9. Acceptance Criteria

Smoke passes if:

1. At least one non-4096 segment-size artifact is generated or located.
2. Exp4 COUNT can load it.
3. A small k sweep produces valid Q/D/U rows.
4. `gpu_eq_cpu_encoded == true` for all smoke rows.
5. The smoke report states whether full Exp6 is unblocked.

Full Exp6 passes if:

1. All generated artifact rows have artifact summary metrics.
2. Every successful artifact has full k coverage.
3. U(k) is monotonic non-increasing per query.
4. `kstar <= max_plane_count` for all `kstar_status == ok`.
5. Kernel correctness passes:
   `gpu_count == cpu_encoded_count`.
6. The report separates:
   - representation fidelity,
   - execution Q/D/U error,
   - throughput,
   - metadata/prep overhead.
7. The report gives a concrete recommendation:
   - keep 4096,
   - use a different fixed default,
   - or use a distribution-dependent policy.

## 10. Decision Rules

Use these conservative rules in the final report.

### 10.1 Keep 4096

Recommend keeping `4096` if:

- smaller segments reduce k* only marginally, and
- throughput at k* does not improve by at least 10%, or
- fidelity does not improve, or
- metadata/prep overhead grows enough to erase the gain.

### 10.2 Smaller default

Recommend a smaller default if:

- k* decreases meaningfully on hard distributions,
- rows/sec at k* improves by at least 10%,
- bytes_per_row does not increase materially,
- representation fidelity is not worse,
- segment overhead remains acceptable.

### 10.3 Distribution-dependent policy

Recommend adaptive segment sizing if:

- uniform/sensor prefer larger or current segments,
- heavy_tailed/zipfian prefer smaller segments,
- no single fixed value is near-optimal across datasets.

The policy should be stated as a heuristic, not a full optimizer:

```text
choose smaller segments when p95 integer_offset_bits or active_plane_count is
high and metadata overhead remains below a fixed budget
```

## 11. Implementation Notes For The Next Agent

1. Start with an inventory step.  Identify whether mainline tooling can generate
   artifacts with configurable `segment_size`.
2. Do not inspect `exp2` branch unless the user explicitly redirects the task.
3. Do not edit benchmark kernels for the smoke unless absolutely required.
4. Do not run a full H200 matrix before a CPU/local artifact inventory and a
   minimal smoke plan exist.
5. If running H200 jobs, follow project Slurm rules:
   - H200 only,
   - hardware gate before benchmark,
   - no account embedded in scripts,
   - batch full matrix only after smoke passes.
6. If NCU remains blocked, do not make physical traffic claims.  Exp6 can still
   use proxy bytes and throughput.

## 12. Paper Claim Boundary

Supported after successful Exp6:

> FOR segment size changes the precision-throughput curve by changing local
> range, active plane count, Q/D/U convergence, and metadata overhead.

Potential paper wording:

> Segment granularity exposes a classic compression/execution tradeoff: smaller
> segments improve local precision convergence but increase metadata and
> per-segment predicate work.  Our results show that the default 4096-row segment
> is a robust engineering point, while skewed distributions may benefit from
> smaller segments.

Do not claim:

> 4096 is optimal.

unless the full Exp6 matrix supports that statement.
