# Raw FP64 COUNT Comparator Spec

**Date:** 2026-05-04
**Scope:** Spec only, no implementation
**Related evidence:** `research/2026-05-02_Exp4_Status_Summary.md`, `research/2026-05-01_Paper_v1_Evidence_Manifest_Draft.md`, `research/2026-05-01_PV1-6_Raw_FP64_SUM_Comparator_Report.md`, `research/2026-05-01_PV1-5_Fixed_Depth_COUNT_Baseline_Report.md`, `research/2026-05-04_Exp4_COUNT_Error_Decomposition_Claim_Audit.md`, `spec/2026-04-30_Exp4_Capped_k_COUNT_Output_Schema_Contract.md`, `results/exp4/b1_20260501_021259_job34546_NVIDIAH200/`, `results/exp4/fixed_depth_count_20260501_233811_job35142_NVIDIAH200/`

## 1. Decision

The raw FP64 COUNT comparator is **not a paper-v1 blocker**.

It is an **optional reference baseline** that becomes necessary only if the paper wants to claim or quantify:

- `faster than raw FP64 COUNT`
- raw-vs-encoded COUNT overhead on the same dataset / threshold pair
- a COUNT baseline analogous to the existing raw FP64 SUM comparator

Paper v1 can already support the synthetic exact-artifact COUNT story from Exp4 without this comparator. The comparator is still useful because it closes the same kind of raw-fidelity gap that the SUM branch already closed.

## 2. Semantics

### 2.1 Query definition

The comparator must implement the raw query:

`SELECT COUNT(*) FROM T WHERE x > threshold`

where:

- `T` is a raw FP64 column stored in `.f64le.bin`
- `x` is each decoded `double` value from that file
- `threshold` is the same FP64 threshold used to label the paired Exp4 encoded result row
- the comparison is strictly `>`; it is not `>=`

### 2.2 Raw CPU reference

The CPU reference is the exact raw FP64 count:

`cpu_raw_count = sum_i [x_i > threshold]`

Implementation notes:

- read the input file as raw little-endian FP64 values
- count qualifying rows with integer accumulation
- do not route through encoded-domain logic
- do not use the progressive kernel for the timed path

### 2.3 GPU raw reference

The GPU benchmark must compute the same raw predicate directly on the raw doubles and produce:

`gpu_raw_count`

Validation gate:

`gpu_raw_count == cpu_raw_count`

If the gate fails, the run is invalid.

### 2.4 Relationship to Exp4 encoded COUNT

The raw comparator is a separate baseline from Exp4 encoded COUNT:

- Exp4 kernel correctness gate: `gpu_count == cpu_encoded_count`
- Exp4 raw-fidelity gate: `cpu_encoded_count == cpu_raw_count`
- raw comparator gate: `gpu_raw_count == cpu_raw_count`

These are different checks and must not be collapsed into one.

## 3. Comparison Matrix

The comparator should be paired with the existing Exp4 encoded results using the same dataset and threshold row keys.

### 3.1 Required matrix for paper-v1 reference use

Use the exact-artifact COUNT evidence chain already present in Exp4:

- `uniform`
- `heavy_tailed`
- selectivities: `50`, `90`, `99`

Use the thresholds already emitted by the encoded-result rows in:

- `results/exp4/b1_20260501_021259_job34546_NVIDIAH200/sweep_summary.csv`
- `results/exp4/fixed_depth_count_20260501_233811_job35142_NVIDIAH200/fixed_depth_count_summary.csv`

The raw comparator should be keyed by:

- `dataset`
- `target_selectivity`
- `threshold`

That makes the raw run joinable to both Exp4 encoded views without inventing a second threshold grid.

### 3.2 Optional extension matrix

If the project later wants a broader dev-set reference baseline, the same contract can be extended to additional Exp4-compatible exact artifact rows where a matching encoded result exists.

That extension is optional and not required for this memo.

## 4. Result Schema

The raw comparator output should be minimal and self-contained.

### 4.1 Main CSV fields

Recommended columns:

- `benchmark`
- `dataset`
- `raw_path`
- `threshold`
- `target_selectivity`
- `n`
- `iters`
- `warmup`
- `ms_per_iter`
- `gpu_raw_count`
- `cpu_raw_count`
- `gpu_eq_cpu_raw`
- `raw_abs_error`
- `raw_rel_error`
- `rows_per_sec`
- `logical_bytes`
- `logical_GBps`
- `estimated_physical_GBps`
- `validated`
- `device`
- `job_id`
- `kernel_path`

### 4.2 Output semantics

- `logical_bytes = n * 8`
- `logical_GBps = logical_bytes / seconds / 1e9`
- `estimated_physical_GBps` is a load-accounting estimate only
- `estimated_physical_GBps` must **not** be described as profiler-measured HBM traffic
- for a raw FP64 scan, it is acceptable to set `estimated_physical_GBps = logical_GBps` by convention only, as the SUM comparator did

### 4.3 Optional join-view fields

For analysis against Exp4 encoded results, a derived join table may also carry:

- `encoded_experiment`
- `encoded_gpu_count`
- `encoded_cpu_encoded_count`
- `encoded_cpu_raw_count`
- `gpu_eq_cpu_encoded`
- `cpu_encoded_eq_cpu_raw`
- `raw_vs_encoded_abs_error`
- `raw_vs_encoded_rel_error`

These are join-time analysis fields, not required in the raw comparator kernel output itself.

## 5. Validation Gates

### 5.1 Blocking gate

The only blocking gate for the raw comparator is:

`gpu_raw_count == cpu_raw_count`

If this fails, the benchmark run is invalid.

### 5.2 Report-only checks

The following are report-only when joined against Exp4 encoded results:

- `cpu_encoded_count == cpu_raw_count`
- `gpu_count == cpu_encoded_count`
- `raw_abs_error`
- `raw_rel_error`

These help place the raw comparator relative to the existing encoded evidence, but they are not the raw comparator’s own correctness gate.

### 5.3 Do not mix validation domains

Do not use the raw comparator to validate the progressive kernel timed path.

Do not add raw FP64 comparison into the progressive kernel’s timed path.

Do not use raw COUNT evidence to claim anything about `U(k)` except when the comparison is explicitly against a full-depth encoded baseline on the same dataset / threshold row.

## 6. Paper Positioning

### 6.1 Why it is optional

Paper v1 already has enough evidence for:

- Exp4 capped-k COUNT semantics
- Q/D/U survival
- fixed-depth encoded COUNT baseline
- logical vs estimated physical throughput terminology

The missing raw COUNT comparator only matters if the paper wants an explicit raw baseline for COUNT, not if it only wants to describe encoded COUNT behavior.

### 6.2 Why it is still useful

The raw comparator would let the paper state, with direct evidence:

- raw COUNT result
- encoded COUNT result
- fixed-depth encoded COUNT result
- progressive capped-k COUNT result

That is the cleanest way to discuss raw-vs-encoded overhead without conflating correctness with representation fidelity.

### 6.3 What it must not claim

Do not claim:

- faster-than-raw COUNT before the comparator exists
- profiler-measured HBM bandwidth
- that the raw comparator is the main paper baseline
- that a raw comparison changes the meaning of `U(k)`

## 7. Recommended Follow-up Scope

If implementation is approved later, open a separate issue with this scope:

### Title

`Implement raw FP64 COUNT comparator for Exp4 reference baseline`

### Scope

- add `benchmarks/experiment4/bench_raw_fp64_count.cu`
- add a runner that points at the same exact-artifact Exp4 datasets and thresholds
- emit a CSV compatible with the schema above
- validate with `gpu_raw_count == cpu_raw_count`
- keep the comparator separate from the progressive kernel path

### Acceptance criteria

- `uniform` and `heavy_tailed`
- selectivities `50`, `90`, `99`
- raw CPU and GPU counts agree
- logical vs estimated physical throughput fields are present
- no change to the progressive encoded COUNT kernel
- no profiler traffic claim

### Out of scope

- kernel edits to `bench_progressive_filter.cu`
- CSV schema changes in the existing Exp4 encoded benchmark
- benchmark-script changes unrelated to the new raw comparator
- H200 execution in this issue

## 8. Bottom Line

The raw FP64 COUNT comparator should be treated as an **optional reference baseline for paper v1**, not a blocker.

It becomes a blocker only if the paper wants a direct `raw COUNT` comparison on the COUNT branch. Otherwise, the existing Exp4 exact-artifact evidence chain is sufficient for the synthetic-method narrative.
