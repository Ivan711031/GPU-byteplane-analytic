# Exp3 Real BuFF Variable-Plane Input Spec

Date: 2026-04-27  
Status: Implementation-ready draft  
Owner: Nick / Codex  
Target audience: developer extending Exp3 from synthetic fixed 8-plane input to real BuFF-encoded DEV input

## 1. Purpose

This spec defines the next Exp3 step after the synthetic `rowpack16` benchmark:

> Feed real DEV data through the BuFF encoder, allow the GPU kernel to read real encoded subcolumns whose plane count may exceed 8, and verify that CPU approximate and GPU approximate agree on the same encoded input.

This spec exists because the current Exp3 implementation is still a synthetic microbenchmark:

- synthetic `uint8` subcolumns are generated inside `bench_progressive_aggregation.cu`
- `EXP3_MAX_SUBCOLUMNS = 8`
- refinement depth is limited to `0..7`
- no real encoded DEV input is consumed

That was acceptable for the original throughput-only v1 goal, but it is not sufficient for the next claim:

> On the same encoded DEV input, CPU approximate and GPU approximate should be nearly identical.

## 2. Problem Statement

The current assumption

```text
real BuFF input will fit in 8 byte planes
```

is unsafe.

The reason is fundamental:

- raw `double` storage is fixed at 64 bits
- BuFF does **not** split the original IEEE FP64 bit pattern into 8 bytes
- BuFF first converts each segment into a common binary fixed-point code
- then it stores `[integer_offset | fractional_bits]` as byte-oriented subcolumns

Therefore:

```text
plane_count = ceil((fractional_bits + integer_offset_bits) / 8)
```

and this quantity can exceed 8 even though each original value is an FP64.

## 3. Evidence

Using the local scan tool

`buff_encoder/buff_plane_scan.cpp`

with:

```bash
/tmp/buff_plane_scan \
  --input-dir /work/u4063895/datasets/synthetic/dev \
  --segment-size 4096 \
  --max-values 8192
```

the first small sample already shows:

- `heavy_tailed`: `max_plane_count = 14`
- `uniform`: `max_plane_count = 9`
- `zipfian`: `max_plane_count = 10`
- `sensor`: `max_plane_count = 7`

Therefore:

- `EXP3_MAX_SUBCOLUMNS = 8` is a synthetic benchmark assumption, not a valid real-data assumption
- real DEV input support requires a kernel/input path that can handle variable plane counts

## 4. Goal

Implement a real-data Exp3 mode with these properties:

1. DEV FP64 input is encoded using the existing BuFF codec logic.
2. Encoded artifacts are materialized in an Exp3-ready layout under `/work`, not inside the repo.
3. Exp3 can read encoded DEV subcolumns instead of synthetic generated subcolumns.
4. The GPU path supports segment plane counts greater than 8.
5. CPU approximate and GPU approximate are both computed from the same encoded artifact.
6. Differences are reported clearly rather than hidden behind a hardcoded tolerance choice.

## 5. Non-Goals

Do not do these in this phase:

- do not remove the existing synthetic mode
- do not rewrite the BuFF encoder from scratch
- do not implement MIN / MAX / VAR / COUNT in the GPU kernel
- do not change the paper claim into “all BuFF workloads are solved”
- do not attempt multi-GPU
- do not do full CPU data scans on the login node

## 6. Existing Components To Reuse

### 6.1 Reuse as-is

Use the current BuFF codec as the single source of truth for encoding semantics:

- `buff_encoder/buff_codec.hpp`
- `buff_encoder/buff_codec.cpp`

This code already defines:

- segment-level `fractional_bits`
- segment-level integer base
- integer offset width
- high-to-low byte planes
- top-k zero-fill truncation semantics

### 6.2 Do not treat BUFF64 as the final GPU input layout

The current BUFF64 file format is a correct encoded container, but it is not yet the simplest GPU runtime layout because:

- segment records are variable-length
- `integer_base_le` is stored as LE bytes
- per-segment plane count is variable
- the current Exp3 host code expects flat plane arrays and flat metadata arrays

Therefore the implementation should **reuse the codec**, but may add a thin exporter/loader layer that converts BUFF64 semantics into an Exp3-ready runtime layout.

This is not a second encoder. It is a loader/export adapter.

## 7. Required Output Artifacts

Store large encoded artifacts under `/work`, not in the repo.

Suggested structure:

```text
/work/u4063895/datasets/synthetic/dev_buff_exp3/
  manifest.json
  sensor/
    plane_000.bin
    plane_001.bin
    ...
    segment_meta.csv
    summary.json
  uniform/
  heavy_tailed/
  zipfian/
```

Repo-side outputs should remain lightweight:

```text
results/exp3_real/
  plane_scan_report.txt
  cpu_gpu_validation.csv
  run_<timestamp>_job<id>_<gpu_tag>/
```

## 7.1 Encoded DEV Artifact Schema

The encoded DEV export is a loader contract, not a benchmark concern. The benchmark only consumes this layout after it is fixed.

Suggested root layout:

```text
/work/u4063895/datasets/synthetic/dev_buff_exp3/<dataset>/
   manifest.json
   summary.json
   segment_meta.csv
   plane_000.bin
   plane_001.bin
   ...
```

Contract details:

- `manifest.json` is the dataset-level summary and must include at least `dataset`, `segment_size`, `value_count`, `segment_count`, `max_plane_count`, `segment_plane_count_min`, `segment_plane_count_max`, `encoded_layout`, and `exact_sum`.
- `segment_meta.csv` is the segment-level loader table and must include at least `segment_index`, `row_offset`, `row_count`, `active_plane_count`, `segment_base`, and `plane_basis[]`.
- `plane_###.bin` files are plane-major byte arrays. Each file stores `value_count` bytes in original row order.
- Missing planes for shorter segments are zero-padded at export time. Zero-padding is a runtime convenience and must not change BuFF semantics.
- `segment_base` is the per-row FP64 contribution of the integer base term.
- `plane_basis[p]` is the per-row FP64 multiplier for plane `p` in that segment. It is explicit metadata, not derived inside the benchmark.

Validation contract:

- `exact_sum` comes from raw FP64 reference data or from the encode-time reference pass. It must not be recomputed from the exported artifact.
- CPU approximate validation must read the exported artifact back through the loader and reconstruct the same top-k semantics as `buff_codec`.
- GPU approximate validation must run on the same exported artifact and report absolute differences, not just a pass/fail flag.

## 8. Data Model For Real Exp3 Mode

### 8.1 Segment assumptions

- segment size stays `4096` to align with Exp2
- values remain finite, non-negative FP64
- one dataset file is processed independently from the others

### 8.2 Plane layout

Each dataset should be exported as:

- one contiguous byte array per plane
- plane-major layout
- rows preserved in original order
- missing planes for shorter segments must be handled explicitly

Because plane count varies by segment, the runtime needs metadata to answer:

- how many planes exist globally for this dataset
- how many planes are valid for each segment
- what basis weight applies to each segment and plane
- what segment base contribution applies

### 8.3 Padding rule

To keep the GPU loader simple, the recommended runtime layout is:

- choose a dataset-level `max_plane_count`
- materialize plane files `0..max_plane_count-1`
- for segments whose actual `plane_count < max_plane_count`, fill missing higher plane rows with zero

This keeps plane arrays rectangular and allows one kernel launch shape per dataset.

Important:

- this padding is a runtime/export convenience
- it must not change the BuFF semantics
- padded planes represent absent low-significance bytes and therefore contribute zero

## 9. Kernel Direction

## 9.1 Keep the synthetic fast path

Do not delete the current fixed-8 synthetic path.

Reason:

- it is already validated
- it is still useful as the clean throughput microbenchmark
- it avoids mixing “best-case synthetic performance” with “real-data correctness mode”

## 9.2 Add a real-data variable-plane path

Add a second kernel path for real encoded DEV input.

Minimum requirement:

- runtime `max_plane_count` greater than or equal to the dataset maximum plane count
- refinement depth sweep from `0` to `max_plane_count - 1`
- correct handling when some segments have fewer active planes than the current refinement depth

## 9.3 Recommended implementation shape

For the first real-data implementation, prefer correctness and clarity over heroic specialization.

Recommended structure:

1. Keep the current synthetic kernel family unchanged.
2. Add a separate real-data kernel family.
3. In the real-data kernel, use:
   - a runtime `plane_count`
   - arrays of plane pointers
   - per-segment active plane count
   - per-segment basis values
4. Keep `uint64_t` integer accumulation per plane.
5. Combine integer plane sums with per-plane basis values after block reduction.

## 9.4 Accumulator strategy

The current logic

```text
sum bytes as integers first
then multiply by basis
then add segment base
```

should be preserved.

This is good for the CPU/GPU agreement goal because:

- byte accumulation is exact in `uint64_t`
- only the final weighted combination uses FP64
- numerical differences are then mostly due to final reduction order, not per-row floating point accumulation

## 10. CPU/GPU Agreement Target

The validation target is not:

```text
GPU approximate must equal exact FP64 result
```

The validation target is:

```text
CPU approximate and GPU approximate are nearly identical
when run on the same encoded DEV input and the same refinement depth
```

This isolates the question that matters right now:

> Does the GPU implementation preserve the same BuFF approximation semantics as the CPU path?

The report should include at least:

- dataset
- refinement depth
- exact sum
- CPU approximate sum
- GPU approximate sum
- `abs(cpu - gpu)`
- `abs(exact - cpu)`
- `abs(exact - gpu)`

Do not hide the differences behind a single pass/fail threshold in the main artifact. Show the actual gaps.

## 11. CLI Changes

The benchmark should gain an input mode distinction.

Suggested shape:

```bash
./build/exp3/bench_progressive_aggregation \
  --mode synthetic_fixed_point_subcolumns
```

and

```bash
./build/exp3/bench_progressive_aggregation \
  --mode encoded_dev_subcolumns \
  --encoded-root /work/u4063895/datasets/synthetic/dev_buff_exp3/uniform
```

Suggested additional metadata fields in CSV:

- `dataset`
- `mode`
- `max_plane_count`
- `segment_plane_count_min`
- `segment_plane_count_max`
- `encoded_layout`
- `approximate_result`

## 12. Offline Preparation Plan

The offline preparation phase should run on CPU allocation, not login node.

Recommended order:

1. Run full plane-count scan on all 4 DEV files.
   verify: produce exact `plane_count` histogram and true dataset maxima
2. Export Exp3-ready encoded artifacts under `/work`.
   verify: metadata row counts and plane lengths match expectations
3. Compute CPU approximate sums from the same encoded artifacts.
   verify: CPU approximate matches the existing CPU decode semantics for each depth
4. Run the GPU benchmark in encoded mode.
   verify: GPU approximate is nearly identical to CPU approximate

If `ngstest` cannot finish the offline preparation, move to `ngs8g`. Do not run full scans on login node.

## 13. Success Criteria

Success for this phase means all of the following are true:

1. The full DEV plane-count scan is complete for all 4 datasets.
2. Real encoded DEV artifacts exist under `/work`.
3. Exp3 can run in a real encoded input mode without using synthetic generated subcolumns.
4. Real mode supports dataset maxima above 8 planes.
5. A validation artifact clearly reports CPU approximate vs GPU approximate for every dataset and refinement depth.
6. The synthetic path still builds and runs unchanged.

## 14. Risks

### 14.1 Performance regression

A runtime variable-plane kernel may be slower than the fixed-8 synthetic path.

That is acceptable in this phase.

The immediate goal is:

- semantic correctness on real input
- real precision-throughput data path

Optimization can come after correctness.

### 14.2 Metadata complexity

Per-segment plane count and basis handling add metadata complexity.

Mitigation:

- keep the runtime layout rectangular with padded zero planes
- keep segment metadata flat and explicit

### 14.3 Confusing two different throughput claims

There will now be two valid but different Exp3 stories:

- synthetic fixed-8 throughput microbenchmark
- real DEV encoded throughput + CPU/GPU agreement path

These must remain clearly labeled in CSVs and reports.

## 15. Immediate Next Steps

1. Run the full `buff_plane_scan` on CPU allocation for all 4 DEV datasets.
   verify: exact max plane count and histogram per dataset
2. Define the Exp3-ready encoded artifact schema under `/work`.
   verify: enough metadata exists to reconstruct CPU and GPU approximate sums
3. Add real encoded input mode to `bench_progressive_aggregation.cu`.
   verify: one small end-to-end dataset succeeds before scaling to full run
4. Extend the kernel path to handle variable plane counts above 8.
   verify: at least one dataset with `plane_count > 8` passes CPU/GPU agreement validation
