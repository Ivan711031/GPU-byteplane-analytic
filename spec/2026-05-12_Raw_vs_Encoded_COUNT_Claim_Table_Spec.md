# Raw-vs-Encoded COUNT Claim Table Spec

Date: 2026-05-12
Status: Spec only. No GPU job expected.
Recommended agent: `gpt-5.4-mini`

## 1. Goal

Create the final joined COUNT evidence table for the paper.

The table must answer:

> For each paper-facing bounded-p artifact and selectivity point, how does
> progressive encoded COUNT at epsilon-selected `k*` compare with raw FP64 COUNT,
> full-depth encoded COUNT, and artifact transfer cost?

This is a join/analysis task over existing results.  Do not modify CUDA kernels.
Do not run a new H200 benchmark unless a required CSV is missing and the report
explains why existing data cannot answer the question.

## 2. Inputs

Use these files from the `main` worktree:

### 2.1 Progressive encoded k-sweep

```text
results/exp4/b1_20260510_174435_job44743_NVIDIAH200/sweep_summary.csv
results/exp4/b1_20260510_174435_job44743_NVIDIAH200/count_epsilon_to_kstar.csv
```

Known status:

- 792 rows.
- `gpu_count == cpu_encoded_count` passed for all rows.
- This sweep is encoded-domain characterization.
- It does not carry raw validation for every capped-k row.

### 2.2 Full-budget raw-validated anchor

```text
results/exp4/b1_20260510_190958_job45163_NVIDIAH200/sweep_summary.csv
```

Known status:

- 171 rows.
- 19 artifacts x 9 selectivities.
- `gpu_count == cpu_encoded_count` passed for all rows.
- `cpu_raw_count` is non-NA for all rows.
- `max_filter_planes == max_plane_count` for all rows.

### 2.3 Raw FP64 COUNT direct baseline

```text
results/raw_count_baseline/run_20260506_153627_job39205_H200/raw_fp64_count_baseline.csv
```

Known status:

- 12 rows.
- datasets: uniform, heavy_tailed, sensor, zipfian.
- selectivities: 50, 90, 99.
- `gpu_raw_count == cpu_raw_count` passed for all rows.

This baseline is direct raw FP64 scan runtime.  It only covers selectivities
50/90/99.  Do not invent raw runtime for selectivities 1/5/10/25/75/95 unless
you explicitly mark it unavailable.

### 2.4 Size/fidelity/transfer evidence

```text
results/paper_v1/artifact_size_fidelity_transfer.csv
results/paper_v1/raw_fp64_transfer_baseline.csv
```

Known status:

- 19 artifacts x 9 selectivities = 171 rows.
- Contains raw-vs-encoded COUNT fidelity drift.
- Contains payload-only H2D transfer timing for each artifact.
- Contains raw FP64 H2D baseline.

## 3. Mainline Artifact Policy

Classify artifacts using the existing paper figure reports:

Mainline:

- uniform: p2, p4, p6, p8, p10
- sensor: p2, p4, p6, p8, p10
- heavy_tailed: p6
- zipfian: p6, p8

Side-study:

- heavy_tailed: p4, p5
- zipfian: p4

Reject:

- heavy_tailed: p2, p3
- zipfian: p2

The output table may include all artifacts, but it must include an
`artifact_class` column with values:

```text
mainline
side_study
reject
```

Paper-facing summary tables should filter to `artifact_class == mainline` unless
the report is explaining failure modes.

## 4. Epsilon Policy

Produce rows for at least these epsilon settings:

```text
epsilon = 0
epsilon = 1e-3
```

Interpretation:

- `epsilon = 0`: exact encoded-domain full-depth or first k where U(k)=0.
- `epsilon = 1e-3`: representative relaxed COUNT error budget.

If `count_epsilon_to_kstar.csv` has only relative epsilon semantics, document
that explicitly in the report.  If the file's epsilon semantics are ambiguous,
read the generating script/report and state the conclusion.  Do not silently
rename it to absolute epsilon.

## 5. Join Keys

### 5.1 Normalize artifact labels

The progressive sweep has `artifact_root` and `precision_decimals`; derive:

```text
artifact_label = p{precision_decimals}
```

or parse it from `artifact_root` if that is more reliable.

The kstar and transfer files already use `artifact` or `artifact_label`.

### 5.2 Normalize selectivity

Use percentage labels for final output:

```text
target_selectivity_pct in {1,5,10,25,50,75,90,95,99}
```

The progressive sweep often stores `selectivity` as an observed fraction, not
the original target label.  Prefer joining using the full-budget anchor and/or
transfer table thresholds:

```text
dataset
artifact_label
threshold
target_selectivity_pct
```

When matching floats:

- use threshold tolerance `abs(a-b) <= 1e-9 * max(1, abs(a), abs(b))`;
- if multiple candidates match, report a failure instead of picking one silently.

### 5.3 Raw runtime join

Raw FP64 COUNT runtime only covers selectivity 50/90/99.

For selectivities outside 50/90/99:

- keep the row if useful,
- set raw runtime fields to empty/NA,
- set `raw_runtime_available = false`.

For 50/90/99:

- join by `dataset` and `target_selectivity_pct`;
- verify raw counts match the full-budget anchor's `cpu_raw_count`.

## 6. Output CSV

Write:

```text
results/paper_v1/raw_vs_encoded_count_comparison.csv
```

Required columns:

```text
dataset
artifact_label
artifact_class
target_selectivity_pct
threshold
epsilon
kstar
max_plane_count
raw_runtime_available
raw_ms
raw_rows_per_sec
raw_count
encoded_full_depth_ms
encoded_full_depth_rows_per_sec
encoded_full_depth_count
encoded_full_depth_eq_raw
representation_count_abs_error
representation_selectivity_drift_pp
count_verdict
progressive_ms_at_kstar
progressive_rows_per_sec_at_kstar
progressive_count_lower
progressive_count_upper
U_kstar
execution_rel_error_bound
gpu_eq_cpu_encoded_at_kstar
speedup_vs_raw
speedup_vs_full_depth_encoded
bytes_per_row
artifact_cudaMemcpy_ms
raw_cudaMemcpy_ms
transfer_speedup_vs_raw
claim_verdict
claim_reason
```

### 6.1 Column definitions

`encoded_full_depth_*` comes from job 45163 when possible.  It is the
`k=max_plane_count` raw-validated anchor.

`progressive_*_at_kstar` comes from job 44743 by selecting the row where:

```text
dataset, artifact_label, target_selectivity_pct, k == kstar
```

`progressive_count_lower = count_lower`

`progressive_count_upper = count_upper`

`U_kstar = uncertain`

`execution_rel_error_bound = U_kstar / max(encoded_full_depth_count, 1)`

`speedup_vs_raw = raw_ms / progressive_ms_at_kstar` when raw runtime exists.

`speedup_vs_full_depth_encoded = encoded_full_depth_ms / progressive_ms_at_kstar`

`transfer_speedup_vs_raw = raw_cudaMemcpy_ms / artifact_cudaMemcpy_ms`

### 6.2 Claim verdict

Set `claim_verdict` using this conservative policy:

```text
supported_faster_than_raw
conditional_prepared_or_encoded_win
encoded_only_win
unsupported
unavailable_raw_runtime
reject_artifact
```

Rules:

1. If `artifact_class == reject`, verdict is `reject_artifact`.
2. If raw runtime is unavailable, verdict is `unavailable_raw_runtime`.
3. If `count_verdict != acceptable`, verdict cannot be `supported_faster_than_raw`.
4. If `encoded_full_depth_eq_raw == false`, verdict cannot be an unconditional
   raw-semantics win.  It can only be conditional on encoded-domain semantics.
5. If `speedup_vs_raw > 1.0`, `count_verdict == acceptable`, and
   `encoded_full_depth_eq_raw == true`, verdict may be
   `supported_faster_than_raw`.
6. If `speedup_vs_full_depth_encoded > 1.0` but raw comparison is unavailable or
   not semantically clean, verdict is `encoded_only_win` or
   `conditional_prepared_or_encoded_win`.
7. Otherwise use `unsupported`.

`claim_reason` must be a short human-readable sentence.

## 7. Report

Write:

```text
research/2026-05-12_Raw_vs_Encoded_COUNT_Claim_Table_Report.md
```

Required sections:

1. Inputs and row counts.
2. Join quality:
   - rows matched,
   - rows missing,
   - duplicate-key failures,
   - selectivity/threshold tolerance policy.
3. Mainline result table for selectivities 50/90/99 and epsilon 1e-3.
4. Exact encoded-domain result table for epsilon 0.
5. Verdict:
   - faster-than-raw COUNT supported / unsupported / conditional.
6. Important caveats:
   - representation error vs execution error,
   - raw runtime only available for 50/90/99,
   - transfer timing is H2D payload-only,
   - no NCU physical traffic claim.
7. Recommended paper wording.

## 8. Acceptance Criteria

Pass only if:

1. Output CSV exists and has the required columns.
2. Every output row has a deterministic `claim_verdict`.
3. No many-to-one or many-to-many join is silently accepted.
4. The report explicitly says whether faster-than-raw COUNT is supported.
5. The report separates:
   - kernel correctness,
   - representation fidelity,
   - execution Q/D/U error,
   - raw FP64 runtime comparison.

Fail and report blocker if:

1. required input CSV is missing,
2. threshold joins are ambiguous,
3. kstar rows cannot be mapped back to sweep rows,
4. raw baseline schema differs from the assumptions and cannot be normalized.

## 9. Out of Scope

- No CUDA changes.
- No new Slurm job.
- No cuDF comparison.
- No exp2 branch inspection.
- No G-ALP/cuSZ/ZFP work.
- No NCU profiling.
