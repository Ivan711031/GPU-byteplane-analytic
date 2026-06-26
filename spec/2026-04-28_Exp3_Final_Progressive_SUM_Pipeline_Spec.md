# Exp3 Final Progressive SUM Pipeline Spec

Date: 2026-04-28
Status: Implementation-ready design checkpoint
Owner: Nick / Codex
Target audience: developers implementing the real-data Exp3 v1 pipeline

## 1. Decision Summary

Exp3 v1 is now scoped to the core performance result:

```text
SUM progressive precision-throughput curve on real BuFF-encoded DEV artifacts
```

The immediate deliverable is:

```text
dataset x k -> throughput
dataset x k -> approximate SUM
dataset x k -> abs error
dataset x k -> bound
```

Other aggregations are not part of Exp3 v1:

- defer `COUNT`
- defer `VAR`
- defer `MIN`
- defer `MAX`

The implementation order after SUM should be:

```text
SUM -> COUNT -> VAR -> MIN/MAX if needed
```

`COUNT(*)` is exact under the progressive model because it does not depend on numeric reconstruction. It is mainly useful as a throughput and control-path baseline. This is different from Exp2's `COUNT(x > t)` study, where threshold classification can change under truncation and therefore has an error model. `VAR` is important later because Exp2 already contains its error study, but it should not block the Exp3 v1 SUM curve.

## 2. Canonical Definition of k

For real Exp3 v1:

```text
k = number of leading byte-planes read from the artifact
```

This must map directly to the exported files:

```text
k = 1 reads plane_000.bin
k = 2 reads plane_000.bin, plane_001.bin
k = 3 reads plane_000.bin, plane_001.bin, plane_002.bin
...
k = max_plane_count reads every exported plane
```

Do not define:

```text
k = 0 means integer-only
```

That abstraction is invalid for the real BuFF artifact because the encoded codeword `[o_i | f_i]` is sliced across byte boundaries. A byte-plane is an I/O and storage unit, not an integer/fractional semantic unit. Integer offset bits and fractional bits may share a byte boundary, so an "integer-only plane" is not a well-defined representation in the exported byte-plane layout.

The old synthetic draft language around "integer subcolumn + fractional subcolumns" is only a synthetic mental model. It must not be used as the real-data kernel contract.

## 3. Exp2 / Exp3 Responsibility Split

Exp3 does not redefine error.

The final paper pipeline is a join:

```text
Exp2: error(k), bound(k)
Exp3: throughput(k), gpu_approx_sum(k)
join: precision-throughput curve
```

Exp2 owns the BuFF truncation model. For the current finite non-negative FP64 DEV datasets, Exp2 defines top-k zero-fill truncation:

```text
read the first k highest-order subcolumns
zero-fill omitted low-order bits
materialize the approximate value
compare against the exact value
```

The per-segment omitted-bit bound from Exp2 is:

```text
E_seg(k) = 2^{-F_seg} * (2^{tau_seg(k)} - 1)
```

For SUM:

```text
abs(SUM_exact - SUM_truncated(k)) <= sum_seg n_seg * E_seg(k)
```

Exp3 should verify that GPU and CPU produce the same approximate SUM for the same artifact and k, but the final precision-throughput plot should use the Exp2 error columns:

```text
approx-vs-exact error = paper precision result
gpu-vs-cpu difference = engineering correctness check
```

Do not mix those two meanings.

## 4. Artifact to Kernel Mapping

The real-data artifact is the kernel contract:

```text
/work/u4063895/datasets/synthetic/dev_buff_exp3/<dataset>/
  manifest.json
  summary.json
  segment_meta.csv
  plane_000.bin
  plane_001.bin
  ...
```

The kernel must consume the leading plane files in artifact order:

```text
for k = 1..max_plane_count:
  read plane_000.bin .. plane_{k-1}.bin
  reconstruct the same top-k zero-fill approximation as Exp2
  output SUM_approx(k)
```

`k` is therefore a physical I/O decision:

```text
how many leading byte-plane arrays are loaded from global memory
```

It is not a request to read an integer part, a fractional part, or an inferred semantic field.

Segments with fewer active planes than the dataset `max_plane_count` keep using zero-padded missing planes from the export layout. That preserves BuFF semantics because absent low-significance planes contribute zero under the top-k zero-fill model.

## 5. Kernel Design

The production real SUM kernel should specialize both the number of planes read and the dataset plane capacity:

```cpp
template <int K_READ, int PLANE_COUNT>
__global__ void progressive_sum_real_rowpack16_specialized(...);
```

`K_READ` is compile-time `k`.

`PLANE_COUNT` is the compile-time maximum plane count for the artifact or dispatch bucket.

The important property is:

```text
no runtime per-plane loop in the hot path
```

The 2026-04-28 A/B already shows why this is required:

```text
same data, same depth9, same rowpack16
runtime kernel:     1201 GB/s
specialized kernel: 3443 GB/s
speedup:            2.866x
GPU sum:            identical
```

The root cause is no longer speculative. The runtime generalized kernel introduces control overhead, constant-load pressure, and local-stack traffic. If `k` stays runtime via:

```cpp
for (int j = 0; j < k; ++j) { ... }
```

the implementation risks reintroducing the same bottleneck that caused the real-vs-synthetic 3x gap.

Therefore each plotted point should use a compile-time specialization:

```text
K_READ = 1
K_READ = 2
...
K_READ = PLANE_COUNT
```

This is part of the research claim:

```text
progressive precision + compile-time specialization recovers memory-bound throughput
```

## 6. Dispatch Strategy

Use a small explicit dispatch table instead of a generic runtime loop.

Known real artifact plane counts from the current DEV export checks:

```text
sensor:       7
uniform:      10
zipfian:      11
heavy_tailed: 16
```

The v1 dispatch can support exactly these production cases:

```cpp
dispatch_sum_specialized(dataset_max_plane_count, k) {
  case 7:  dispatch_K<7>(k);
  case 10: dispatch_K<10>(k);
  case 11: dispatch_K<11>(k);
  case 16: dispatch_K<16>(k);
  default: dispatch_runtime_fallback(k);
}
```

Within each `dispatch_K<PLANE_COUNT>`, choose the corresponding `K_READ` specialization.

Keep a runtime fallback for safety and debugging, but do not use it for the official Exp3 v1 throughput baseline. The official baseline must record that the specialized path was used.

## 7. SUM Reconstruction Contract

The kernel should continue using the existing integer-then-FP combination structure:

```text
sum byte plane values as integers
combine plane sums with per-segment basis values
add segment base contribution
```

This keeps most accumulation exact in integer registers and limits FP64 work to the final combination. That is also the right validation shape because CPU and GPU approximate results then differ mainly by final reduction order, not by per-row floating-point accumulation.

For each segment and k:

```text
segment_sum_approx(k)
  = row_count * segment_base
  + sum_{p=0..k-1} plane_sum[p] * plane_basis[p]
```

Then the dataset result is:

```text
SUM_approx(k) = sum over segments segment_sum_approx(k)
```

The exact formula should reuse the existing real encoded artifact metadata. Do not derive an integer/fractional split inside the kernel.

## 8. Correctness Outputs

Exp3 should emit both engineering correctness and paper-join columns.

Required correctness columns:

```text
dataset
k
max_plane_count
kernel_path
cpu_approx_sum
gpu_approx_sum
abs_cpu_gpu_diff
exact_sum
abs_exact_cpu_diff
abs_exact_gpu_diff
validated
```

Interpretation:

- `abs_cpu_gpu_diff` checks GPU implementation correctness.
- `abs_exact_cpu_diff` and `abs_exact_gpu_diff` are useful diagnostics.
- Final precision curve should still join against Exp2's `abs_error` and `bound`.

The validation target is:

```text
GPU approximate SUM matches CPU approximate SUM for the same artifact and k
```

It is not:

```text
GPU approximate SUM equals exact FP64 SUM at every k
```

Only the full-depth point is expected to approach the full exported reconstruction semantics.

## 9. Throughput CSV Schema

The official specialized real Exp3 v1 run should write one row per dataset and k.

Suggested file:

```text
results/exp3_real_specialized_sum/run_<timestamp>_job<id>_<gpu_tag>/throughput.csv
```

Required columns:

```text
experiment
dataset
aggregation
mode
kernel_path
k
logical_subcolumns_read
max_plane_count
segment_rows
n
iters
warmup
ms_per_iter
rows_per_sec
billion_rows_per_sec
logical_bytes_per_iter
logical_GBps
gpu_approx_sum
cpu_approx_sum
abs_cpu_gpu_diff
exact_sum
abs_exact_gpu_diff
device
gpu_tag
validated
```

Use these canonical values:

```text
experiment=exp3_real_progressive_sum
aggregation=sum
mode=encoded_dev_subcolumns
kernel_path=specialized_rowpack16
logical_subcolumns_read=k
logical_bytes_per_iter=n * k
```

`logical_GBps` is still useful for kernel diagnosis, but the final paper plot should prefer `billion_rows_per_sec` because the x-axis is throughput over rows at a chosen precision.

## 10. Joined Precision-Throughput CSV Schema

The join step combines Exp2 error rows with Exp3 specialized throughput rows.

Suggested file:

```text
results/precision_throughput/exp3_real_sum_specialized_precision_throughput.csv
```

Join keys:

```text
dataset
aggregation=sum
k
```

Required output columns:

```text
dataset
aggregation
k
throughput_billion_rows_per_sec
throughput_logical_GBps
exp3_kernel_path
exp3_gpu_approx_sum
exp3_cpu_approx_sum
exp3_abs_cpu_gpu_diff
exp2_abs_error
exp2_bound
exp2_gap
exp2_relative_error
segment_rows
max_plane_count
gpu_tag
```

The joined file is the single source for the paper precision-throughput curve.

## 11. Plot Contract

The main paper plot should show:

```text
x-axis: throughput, preferably billion rows/sec
y-axis: error, from Exp2 abs_error or relative_error
point: one k
line: one dataset
aggregation: SUM
```

Recommended plot variants:

```text
SUM absolute error vs throughput
SUM relative error vs throughput
SUM bound vs throughput
SUM gap(bound/error) vs k for sanity
```

Plot labels should make the key semantics explicit:

```text
k = leading byte-planes read
error = Exp2 zero-fill truncation error
throughput = Exp3 specialized real-data rowpack16
```

Do not label k as "fractional planes" or "refinement rounds" unless the caption explicitly defines it as leading byte-planes.

## 12. Implementation Work Packages

### Worker A: Specialized Real SUM Kernel

Owns:

```text
benchmarks/experiment3/exp3_kernels_progressive.cuh
benchmarks/experiment3/bench_progressive_aggregation.cu
```

Tasks:

- Add `template<int K_READ, int PLANE_COUNT>` real SUM kernel.
- Preserve rowpack16 global load shape.
- Remove runtime per-plane hot-path arrays for specialized kernels.
- Keep runtime fallback for non-production/debug cases.
- Emit `kernel_path=specialized_rowpack16` for official runs.

Validation:

```text
same artifact and k: specialized GPU sum == runtime GPU sum within tolerance
specialized path is selected for plane counts 7, 10, 11, 16
```

### Worker B: Runner and CSV

Owns:

```text
scripts/legacy/root_runners/run_exp3_real_throughput_array.sh
scripts/legacy/root_runners/run_exp3.sh
```

Tasks:

- Add a specialized SUM run mode for all four real DEV datasets.
- Sweep `k=1..max_plane_count` for each dataset.
- Write `throughput.csv` with the schema in this spec.
- Do not require NCU for this baseline.

Validation:

```text
four datasets complete
all k values present
validated=true for every official row
```

### Worker C: Exp2/Exp3 Join and Plots

Owns:

```text
results/precision_throughput/
scripts or plotting utilities used for precision-throughput output
```

Tasks:

- Join Exp2 `results/exp2/results/metrics.csv` with the new Exp3 specialized throughput CSV.
- Preserve Exp2 error semantics and bounds.
- Generate SUM precision-throughput plots.
- Label `k` as leading byte-planes read.

Validation:

```text
joined rows = sum over datasets of max_plane_count
no missing dataset/k pairs
plot points match joined CSV rows
```

### Worker D: Spec/Report Integration

Owns:

```text
research/
handoff/
README or paper notes if applicable
```

Tasks:

- Summarize the 2026-04-28 runtime-vs-specialized A/B.
- State that the 3x gap is attributable to runtime generalized control/local-stack overhead.
- Link the final SUM curve to the Exp2 error model.
- Keep the `ncu` limitation explicit: fresh real NCU metrics are not required for the v1 baseline.

Validation:

```text
report explains why specialized k is required
report does not claim MIN/MAX/VAR/COUNT are implemented in Exp3 v1
```

## 13. Acceptance Criteria

Exp3 v1 is complete when:

```text
1. Specialized real SUM kernel supports plane counts 7, 10, 11, and 16.
2. Every official throughput point uses compile-time K_READ specialization.
3. Runtime fallback remains available but is not used for the official baseline.
4. Four datasets have complete k sweeps.
5. GPU approximate SUM matches CPU approximate SUM for every dataset/k row.
6. Exp2 metrics and Exp3 throughput are joined by dataset, aggregation=sum, and k.
7. The final plot shows SUM precision-throughput curves for all four datasets.
```

## 14. Non-Goals for This Phase

Do not spend this phase on:

```text
fresh NCU collection on compute nodes
MIN/MAX/VAR/COUNT GPU kernels
multi-word accumulation
new encoder semantics
integer-only k
changing Exp2 error definitions
```

The most valuable next result is the specialized real SUM curve, not more profiler archaeology.

## 15. One-Sentence Project Direction

The goal is not to add more aggregations yet; it is to finish the SUM progressive precision-throughput curve by making `k` correspond exactly to the artifact byte-plane order and making each `k` a compile-time specialized real kernel so runtime control overhead does not destroy throughput again.
