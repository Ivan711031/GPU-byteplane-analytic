# Exp1 Contiguous64 Vectorized Baseline Spec

Date: 2026-04-17  
Status: Draft for implementation  
Owner: Nick / Codex  
Target audience: junior developer implementing the next Exp1 contiguous baseline update

## 1. Goal

Add a stronger contiguous FP64 baseline for Experiment 1 by extending the existing contiguous baseline binary with:

- `--variant scalar`
- `--variant vec2`

The implementation must stay in the existing contiguous baseline benchmark:

- `benchmarks/experiment1/bench_contiguous_baseline.cu`

Do not implement this spec yet in this document update. This file is the implementation specification only.

## 2. Why This Is Needed

The current Exp1 result compares:

```text
rowpack16 k=8   uses 128-bit row-wise packed loads
contiguous64    uses scalar 64-bit loads
```

That comparison is useful, but it mixes two effects:

1. byte-plane layout and progressive scan behavior
2. load instruction width / vectorized load efficiency

For a more rigorous Exp1 argument, contiguous64 also needs a vectorized 128-bit load baseline. Then `rowpack16 k=8` can be compared against both:

```text
contiguous64 scalar
contiguous64 vec2
```

This separates:

- full precision layout penalty or parity at `k=8`
- progressive benefit at `k<8`, where rowpack16 reads fewer byte-planes

## 3. References To Read First

Before editing code, read these files in this order:

1. `benchmarks/experiment1/bench_contiguous_baseline.cu`  
   This is the file to modify. It currently owns the contiguous baseline CLI, kernel, timing, occupancy grid, allocation, and CSV output.

2. `benchmarks/experiment1/bench_byteplane_scan.cu`  
   Use this as the reference for adding a small variant parser and dispatch path. In particular, look at how `--byte_variant baseline|ilp4` is parsed and reported.

3. `benchmarks/experiment1/exp1_scan_common.cuh`  
   Use this as the reference for the shared reduction helpers if you choose to deduplicate later. Do not refactor into this file unless the implementation naturally needs it.

4. `benchmarks/experiment1/exp1_kernels_rowpack.cuh`  
   Use this only to understand the rowpack16 comparison target. Do not move rowpack code into the contiguous baseline.

5. `benchmarks/experiment1/CMakeLists.txt`  
   Confirm the existing `bench_contiguous_baseline` target. This spec should not require adding a new binary.

6. `scripts/run_exp1_baseline.sh` and `scripts/legacy/root_runners/run_exp1_baseline.sh`
   Update these only after the benchmark supports `--variant`, so scalar and vec2 can both be run and recorded.

Related design context:

- `spec/2026-04-12_新增exp1 baseline.md`
- `spec/2026-04-15_Exp1_Rowpack_Strategy_and_Kernel_Organization_Spec.md`

## 4. Scope

This task must implement only the first-stage contiguous baseline improvement:

- keep the existing scalar contiguous baseline
- add a vectorized contiguous baseline that reads two consecutive FP64 rows per 128-bit load
- expose the selection through `--variant scalar|vec2`
- add a `variant` CSV column while keeping `benchmark=contiguous64`
- add verification steps that check whether vec2 actually compiles to a 128-bit global load path

Do not implement ILP variants in this stage.

Explicitly out of scope:

- `contiguous64_ilp4`
- `contiguous64_vec2_ilp4`
- changing byte-plane kernels
- changing rowpack kernels
- adding a new binary
- adding a new byte-plane strategy
- changing the meaning of `k`
- changing Exp1 validation logic
- changing the timing model
- changing the reduction semantics
- introducing shared-memory staging

## 5. Required Naming

Use these names consistently in code comments, CSV output, logs, and scripts:

```text
benchmark = contiguous64
variant   = scalar | vec2
```

Meaning:

- `scalar`: current one-thread, one-`uint64_t` load path
- `vec2`: one-thread, one-128-bit load path, reading two consecutive `uint64_t` rows

Do not rename the benchmark to `contiguous64_vec2`. Keep `benchmark=contiguous64` and put the distinction in `variant`.

## 6. CLI Requirements

Extend the existing binary:

```bash
./build/exp1/bench_contiguous_baseline \
  --variant scalar \
  --device 0 \
  --n 100000000 \
  --block 256 \
  --grid_mul 1 \
  --warmup 10 \
  --iters 1000 \
  --csv results/exp1/contiguous_scalar.csv
```

```bash
./build/exp1/bench_contiguous_baseline \
  --variant vec2 \
  --device 0 \
  --n 100000000 \
  --block 256 \
  --grid_mul 1 \
  --warmup 10 \
  --iters 1000 \
  --csv results/exp1/contiguous_vec2.csv
```

Requirements:

- Add `--variant scalar|vec2`.
- Default must be `scalar`, so existing scripts without `--variant` keep working.
- Reject unknown variants with a clear error.
- Do not add `--k`, `--plane_bytes`, or byte-plane options to this binary.
- Do not create a separate `bench_contiguous_vec2` binary.

Recommended implementation shape:

```cpp
enum class ContiguousVariant
{
  Scalar,
  Vec2,
};
```

Add a parser function similar to the byte benchmark's `parse_byte_variant`.

## 7. Kernel Requirements

### 7.1 Preserve Scalar Kernel

Keep the existing scalar semantics:

```cpp
__global__ void scan_contiguous_u64(
    const uint64_t *__restrict__ data,
    uint64_t n,
    unsigned long long *__restrict__ per_block_out);
```

The scalar kernel must continue to:

- use grid-stride traversal over `n` logical rows
- read one `uint64_t` per visited row
- accumulate in registers
- use the same block reduction pattern

You may rename it to `scan_contiguous_u64_scalar` if that makes dispatch clearer, but do not change its behavior.

### 7.2 Add Vec2 Kernel

Add a new kernel whose intent is 128-bit vectorized loading:

```cpp
__global__ void scan_contiguous_u64_vec2(
    const uint64_t *__restrict__ data,
    uint64_t n,
    unsigned long long *__restrict__ per_block_out);
```

The vec2 kernel must:

- treat the input as pairs of consecutive `uint64_t`
- use one vector load per pair where possible
- accumulate both lanes into the per-thread sum
- handle odd `n` correctly
- use the same block reduction pattern as scalar
- preserve the same timing and output semantics

Recommended kernel shape:

```cpp
__global__ void scan_contiguous_u64_vec2(
    const uint64_t *__restrict__ data,
    uint64_t n,
    unsigned long long *__restrict__ per_block_out)
{
  unsigned long long sum = 0;
  uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

  uint64_t pair_count = n / 2;
  const ulonglong2 *__restrict__ data2 =
      reinterpret_cast<const ulonglong2 *>(data);

  for (uint64_t i = tid; i < pair_count; i += stride)
  {
    ulonglong2 v = data2[i];
    sum += static_cast<unsigned long long>(v.x);
    sum += static_cast<unsigned long long>(v.y);
  }

  if ((n & 1ull) && tid == 0)
    sum += static_cast<unsigned long long>(data[n - 1]);

  block_reduce_store(sum, per_block_out);
}
```

The exact code may differ, but the behavior must match this structure.

Alignment note:

- `cudaMalloc` returns sufficiently aligned memory for a `ulonglong2` reinterpretation.
- Because `data` is a `uint64_t *`, pair index `i` maps to logical rows `2*i` and `2*i+1`.
- Do not use byte pointer arithmetic for this path unless needed for a verified alignment fix.

Odd `n` note:

- The tail row for odd `n` must be counted exactly once.
- A simple `if ((n & 1ull) && tid == 0)` tail is acceptable because the tail is one element and outside the timed bottleneck.

## 8. Host Dispatch Requirements

The host-side timing path must dispatch based on `opt.variant`.

Required behavior:

- warmup launches use the selected kernel
- timed launches use the selected kernel
- occupancy grid sizing uses the selected kernel pointer
- log output prints the selected variant
- CSV output writes the selected variant

Keep the dispatch simple. A `switch` or small `if` is enough. Do not introduce a registry, class hierarchy, or template abstraction just for two variants.

Important detail:

```cpp
int grid = occupancy_grid(
    opt.device,
    opt.block_threads,
    opt.grid_mul,
    selected_kernel_pointer,
    0);
```

The selected kernel pointer matters because scalar and vec2 may have different register usage and occupancy.

## 9. CSV Requirements

Change contiguous baseline CSV output to include `benchmark` and `variant`.

Required header:

```text
benchmark,variant,n,logical_bytes,physical_bytes_per_row,block,grid,warmup,iters,ms_per_iter,logical_GBps,device,sm,cc_major,cc_minor
```

Required values:

```text
benchmark = contiguous64
variant = scalar or vec2
logical_bytes = n * 8
physical_bytes_per_row = 8
logical_GBps = logical_bytes / seconds / 1e9
```

Rationale:

- For contiguous full-FP64 scan, both scalar and vec2 logically read 8 bytes per row.
- Vec2 changes load organization, not the logical bytes per row.
- The CSV should make this explicit.

Do not report `k=1` or `plane_bytes=8` for this benchmark. Those fields are byte-plane-specific and make the contiguous output harder to interpret.

If existing plotting scripts still expect the old header, update those scripts in the implementation task or clearly document the migration.

## 10. Script Requirements

Update `scripts/legacy/root_runners/run_exp1_baseline.sh` after the binary supports variants.

Minimum expected behavior:

- build the same `bench_contiguous_baseline` binary
- run `--variant scalar`
- run `--variant vec2`
- write separate CSV files, for example:
  - `contiguous64_scalar.csv`
  - `contiguous64_vec2.csv`
- record both commands in `run_meta.txt`
- record both commands in `repro_command.txt`
- update the Nsight Compute command template so a developer can profile both variants

Do not remove scalar from the runner. The point is to compare scalar and vec2 side by side.

## 11. Comparison Matrix For Exp1 Report

The implementation enables two separate comparisons.

### 11.1 Full Precision Access Efficiency

Compare:

```text
rowpack16 k=8
contiguous64 scalar
contiguous64 vec2
```

This answers:

```text
At full precision, does byte-plane rowpack16 have a layout penalty compared with an optimized contiguous scan?
```

Interpretation:

- If `rowpack16 k=8` is close to `contiguous64 vec2`, then byte-plane layout is not causing a large full-precision penalty.
- If `contiguous64 vec2` is clearly faster, say that rowpack16 has some full-precision layout or instruction-mix overhead.

Do not claim that rowpack16 beats the best possible contiguous scan unless the vec2 result supports it.

### 11.2 Progressive Scan Benefit

Compare:

```text
rowpack16 k=1..7
contiguous64 vec2
```

This answers:

```text
When a query needs only the first k byte-planes, does rowpack16 gain throughput by reading fewer bytes?
```

For `k<8`, rowpack16 reading fewer bytes is not an unfair advantage. It is the actual progressive scan property being tested.

## 12. Verification Requirements

The implementation is not complete until both functional and code-generation checks are done.

### 12.1 Functional / Runtime Verification

Build:

```bash
cmake -S benchmarks/experiment1 -B build/exp1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp1 -j
```

Run small smoke tests:

```bash
./build/exp1/bench_contiguous_baseline --variant scalar --n 1000000 --warmup 2 --iters 5 --csv /tmp/contiguous64_scalar.csv
./build/exp1/bench_contiguous_baseline --variant vec2 --n 1000000 --warmup 2 --iters 5 --csv /tmp/contiguous64_vec2.csv
./build/exp1/bench_contiguous_baseline --variant vec2 --n 1000001 --warmup 2 --iters 5 --csv /tmp/contiguous64_vec2_odd.csv
```

Check:

- all commands exit successfully
- CSV files contain `benchmark=contiguous64`
- CSV files contain the correct `variant`
- `logical_bytes` equals `n * 8`
- odd `n` does not crash or drop the last row

### 12.2 128-bit Load Verification

The vec2 implementation must be checked to ensure it actually uses a vectorized 128-bit global load path. Writing `ulonglong2` is not sufficient by itself.

Use at least one of the following:

1. SASS inspection with `cuobjdump` or `nvdisasm`
2. Nsight Compute source/SASS view
3. Nsight Compute memory instruction metrics

Example commands to adapt:

```bash
cuobjdump --dump-sass build/exp1/bench_contiguous_baseline > /tmp/bench_contiguous_baseline.sass
```

Then search for the vec2 kernel:

```bash
grep -n "scan_contiguous_u64_vec2" /tmp/bench_contiguous_baseline.sass
```

Expected evidence:

- the vec2 kernel should show a 128-bit global load instruction, for example a load with `128` in the instruction form
- scalar should show narrower load behavior

The exact SASS mnemonic may vary by CUDA version and architecture. Do not hard-code the spec to one mnemonic. The implementation report must state what was observed.

If vec2 does not compile into a 128-bit load:

- first try preserving `ulonglong2` and checking alignment assumptions
- then consider an explicitly aligned two-`uint64_t` struct
- do not claim the vec2 baseline is vectorized until the generated code supports that claim

### 12.3 Performance Sanity Check

Run a normal-size benchmark after smoke tests:

```bash
./build/exp1/bench_contiguous_baseline --variant scalar --n 100000000 --block 256 --grid_mul 1 --warmup 10 --iters 1000 --csv results/exp1/contiguous64_scalar.csv
./build/exp1/bench_contiguous_baseline --variant vec2 --n 100000000 --block 256 --grid_mul 1 --warmup 10 --iters 1000 --csv results/exp1/contiguous64_vec2.csv
```

Expected result:

- vec2 may be faster, similar, or slower than scalar
- do not force the result
- report the result as measured

The success criterion is not "vec2 must beat scalar." The success criterion is "we have a correctly implemented and verified vectorized contiguous baseline."

## 13. Implementation Checklist

1. Add `ContiguousVariant` enum and parser to `bench_contiguous_baseline.cu`.
2. Add `--variant scalar|vec2` to usage text.
3. Preserve scalar kernel behavior.
4. Add `scan_contiguous_u64_vec2`.
5. Dispatch warmup, timed run, and occupancy grid sizing using the selected variant.
6. Add `variant` to `Options` and `RunResult` if useful.
7. Replace contiguous baseline CSV header with the required `benchmark,variant,...` schema.
8. Print variant in stderr progress output.
9. Update `scripts/legacy/root_runners/run_exp1_baseline.sh` to run both variants.
10. Update run metadata and repro command output for both variants.
11. Build and run scalar smoke test.
12. Build and run vec2 smoke test with even `n`.
13. Build and run vec2 smoke test with odd `n`.
14. Inspect generated SASS or Nsight Compute output for vec2 load width.
15. Record the verification result in the implementation summary.

## 14. Deliverables For The Implementation Task

When implementing this spec, return:

1. files changed
2. summary of scalar/vec2 behavior
3. exact build command used
4. exact smoke test commands used
5. CSV schema after the change
6. SASS or Nsight Compute evidence for whether vec2 produced a 128-bit load
7. caveats, especially if the compiler did not emit the expected load width

## 15. Wording For Report

Use this framing in the Exp1 report:

```text
To ensure that rowpack16 is not only outperforming a scalar contiguous baseline,
we add a vectorized contiguous64 baseline that uses 128-bit loads over two
consecutive FP64 rows. This separates two effects: (1) whether byte-plane
layout can match optimized contiguous scan at k=8, and (2) whether progressive
subcolumn reading provides speedup at k<8 by reducing bytes read.
```

Chinese version:

```text
為了避免 rowpack16 只是打贏一個 scalar contiguous baseline，
我們加入使用 128-bit load 的 contiguous64_vec2 baseline。
這可以把兩件事分開：第一，k=8 時 byte-plane layout 是否能接近
最佳化 contiguous scan；第二，k<8 時 progressive subcolumn scan 是否
因少讀 bytes 而獲得加速。
```
