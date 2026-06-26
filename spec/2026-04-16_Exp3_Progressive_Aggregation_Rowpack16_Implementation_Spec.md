# Exp3 Progressive Aggregation Rowpack16 Implementation Spec

Date: 2026-04-16  
Status: Implementation-ready draft  
Owner: Nick / Codex  
Target audience: junior developer implementing Experiment 3 v1

## 1. Purpose

Experiment 3 is the next step after Exp1.

Exp1 answered the hardware-access question:

> Can byte-plane/subcolumn layout be scanned efficiently on H200?

The current Exp1 answer is yes, if the implementation uses row-wise packed loads:

- `rowpack4`: one `uint32_t` load = 4 consecutive rows from the same byte-plane.
- `rowpack16`: one `uint4` / 128-bit load = 16 consecutive rows from the same byte-plane.

The latest Exp1 rowpack report shows:

```text
rowpack16 k=8: 4394.268 logical GB/s
contiguous64:  3929.343 logical GB/s
byte ilp4 k=8: 2399.070 logical GB/s
```

NCU confirms the mechanism:

```text
byte_ilp4:  LDG.E.U8:8b
rowpack4:   LDG.E:32b
rowpack16:  LDG.E.128:128b
```

Therefore, Exp3 should not restart from scalar `uint8_t` loads as the main path.

Exp3 v1 should use the Exp1 conclusion directly:

> Use row-wise packed subcolumn loads, with `rowpack16` as the primary performance path.

The goal of this spec is to define a concrete, minimal, verifiable Exp3 v1 implementation.

## 2. Exp3 Research Goal

The high-level Experiment 3 goal from the initial guide is:

> Implement a Buff-style progressive aggregation kernel and show that each additional subcolumn reduces throughput while reducing error.

The final paper deliverable is:

```text
precision-throughput curve
```

where each point corresponds to:

```text
(refinement depth, throughput, error)
```

However, Experiment 2 is not ready yet. Real dataset encoding and final error analysis are not available.

Therefore, Exp3 must start with a performance-only synthetic benchmark.

Exp3 v1 deliverable:

```text
refinement_depth -> throughput
```

Later, Exp2 will provide:

```text
refinement_depth -> error
```

The final curve is a join:

```text
Exp3 throughput CSV + Exp2 error CSV
```

## 3. Scope Summary

### 3.1 Implement now

Implement a standalone synthetic progressive aggregation benchmark:

```text
benchmarks/experiment3/bench_progressive_aggregation.cu
```

It must:

- allocate synthetic Buff-style subcolumns on GPU
- run progressive SUM aggregation for `refinement_depth = 0..K_MAX`
- use `rowpack16` for `uint8_t` encoded byte subcolumns
- output throughput CSV
- optionally validate numerical output outside the timed loop
- follow Exp1 run directory / metadata conventions

### 3.2 Defer

Do not implement these in v1:

- real dataset loader
- complete Buff encoder
- Exp2 empirical error measurement
- final plotting pipeline
- MIN / MAX / VAR
- progressive filter
- multi-GPU
- multi-word accumulation

## 4. Key Definitions

### 4.1 `refinement_depth`

Use `refinement_depth`, not only `k`, in Exp3.

Reason:

- Exp1 uses `k` to mean number of byte-planes read.
- Buff aggregation sums encoded byte subcolumns independently and combines the intermediate sums with per-subcolumn basis weights.

Define:

```text
refinement_depth = 0
  read encoded subcolumn 0 only

refinement_depth = 1
  read encoded subcolumns 0..1

refinement_depth = 2
  read encoded subcolumns 0..2

...

refinement_depth = F
  read encoded subcolumns 0..F
```

CSV may also include a short `k` alias if helpful, but `refinement_depth` is the canonical name.

### 4.2 Logical subcolumns read

For Exp3 v1:

```text
logical_subcolumns_read = 1 + refinement_depth
```

where `refinement_depth=0` still reads one leading encoded byte subcolumn.

### 4.3 Logical bytes

For the Buff-aligned v1 default layout, every encoded subcolumn is one byte:

```text
logical_bytes = n * (1 + refinement_depth)
```

### 4.4 Row-wise packed load

`rowpack16` means:

```text
one 128-bit load from one subcolumn
that load contains 16 consecutive rows from the same subcolumn
```

It does not mean:

```text
read 16 precision bytes from the same row
```

This distinction must be documented in code comments and README.

## 5. Buff-Aligned Synthetic Model

Exp3 v1 needs a simple synthetic model that is close enough to Buff-style progressive aggregation to measure kernel behavior, but not blocked by the final encoder.

After reviewing Buff Section 3.4.2, the important aggregation rule is:

```text
sum each encoded subcolumn as integers,
then combine subcolumn sums with their corresponding basis weights,
then add the base contribution.
```

Therefore Exp3 v1 should model byte subcolumns and basis weights directly.

Use this v1 model:

```text
value_i(depth) =
  base_segment(i)
  + sum_{j=0..depth} subcolumn_j[i] * basis_j
```

For SUM:

```text
SUM(depth) =
  N * base
  + sum_{j=0..depth} SUM(subcolumn_j) * basis_j
```

For v1 synthetic default:

```text
base_segment = 0.0
basis_j = 2^(-8 * j)
```

This makes validation simple while still preserving the kernel structure:

```text
integer accumulation per byte subcolumn + weighted final combination
```

The implementation should still allocate and use segment metadata arrays for base and basis weights, because real encoded segments may have different range and split metadata.

### 5.1 Error bound in v1

Exp3 v1 is not responsible for final Exp2 error analysis.

However, because the v1 synthetic model has a clean fixed-point interpretation, it may output a synthetic worst-case bound:

```text
per_value_abs_error_bound(depth) = 255 * sum_{j=depth+1..max_depth} basis_j
sum_abs_error_bound(depth) = n * per_value_abs_error_bound(depth)
avg_abs_error_bound(depth) = per_value_abs_error_bound(depth)
```

Important:

```text
This synthetic bound is not the final Exp2 result.
```

If included in CSV, name it clearly:

```text
synthetic_sum_abs_error_bound
synthetic_avg_abs_error_bound
error_bound_source=synthetic_fixed_point
```

Do not call it the final paper error unless Exp2 later confirms it.

## 6. Data Layout

### 6.1 Required arrays

Allocate these GPU arrays:

```cpp
uint8_t  *d_subcolumns[MAX_SUBCOLUMNS];
double   *d_segment_base;
double   *d_subcolumn_basis;
double   *d_partial_out;
```

Required v1 defaults:

```text
MAX_SUBCOLUMNS = 8
subcolumn type = uint8_t
base type = double
subcolumn basis type = double
partial output type = double
```

Required allocation sizes:

```text
d_subcolumns[p]:       n bytes for each p in [0, MAX_SUBCOLUMNS)
d_segment_base:        num_segments doubles
d_subcolumn_basis:     num_segments * MAX_SUBCOLUMNS doubles
d_partial_out:         grid doubles
```

Index basis weights as:

```cpp
double basis = d_subcolumn_basis[segment_id * MAX_SUBCOLUMNS + p];
```

Do not allocate `d_subcolumn_basis` as a single global 8-element array in v1.

Reason:

- the synthetic default uses the same basis values for every segment
- the real Buff/FOR path may have segment-specific basis metadata
- using the full segment-major layout now prevents a later kernel ABI change

### 6.2 Encoded subcolumns

Use SOA layout:

```text
subcolumn[0][row]
subcolumn[1][row]
...
subcolumn[7][row]
```

Each encoded byte subcolumn is a separate `cudaMalloc` allocation in v1.

Reason:

- This matches Exp1 byte-plane allocation.
- Separate `cudaMalloc` gives sufficient base alignment for `uint4` rowpack16 loads.

### 6.3 Integer/fraction split note

The initial project guide describes Buff-style data as:

```text
integer subcolumn + fractional subcolumns
```

Buff's paper text is slightly more general: after float splitting and byte-oriented padding, each byte unit is a subcolumn with a corresponding basis. The first byte subcolumn may contain integer bits plus padded fractional bits.

Therefore, the v1 primary implementation should use:

```text
uint8_t byte subcolumns + basis_j
```

not a mandatory:

```text
uint32_t integer column + uint8_t fractional planes
```

A separate `uint32_t integer + uint8_t frac` mode may be useful later as a fixed-point/FOR sensitivity experiment, but it should not be the primary Buff-aligned v1.

### 6.4 Future slab allocation warning

Do not implement slab allocation in v1.

If a future version uses:

```cpp
uint8_t *d_slab;
subcolumn[p] = d_slab + p * pitch;
```

then `pitch` must be aligned to at least 16 bytes for `uint4` loads:

```text
pitch = round_up(n, 16)
```

Otherwise `rowpack16` may become misaligned for planes after plane 0.

## 7. Segment and Tile Model

### 7.1 Why segment/tile is needed

The initial Exp3 guide says:

```text
Each thread block processes one segment's subcolumn data.
FOR base and subcolumn basis weights are applied once per block or segment, not per row.
```

But one large segment can contain far more rows than one block should process directly.

Therefore, v1 should use a segment-tile model:

```text
segment = logical encoded segment with one base and one basis vector
tile    = block-sized chunk inside a segment
block   = processes one tile
```

Multiple blocks may process the same segment.

Each block:

1. accumulates subcolumn sums for its tile
2. loads that segment's base/basis metadata
3. applies base/basis weights once to produce one partial SUM
4. writes one `double` partial result to `d_partial_out[blockIdx.x]`

### 7.2 Required mapping

Use:

```text
segment_rows
items_per_thread
pack_width
tile_rows = block_threads * items_per_thread * pack_width
tiles_per_segment = ceil_div(segment_rows, tile_rows)
num_segments = ceil_div(n, segment_rows)
grid = num_segments * tiles_per_segment
```

This rectangular grid is acceptable for v1 even though the last segment may contain fewer than `tiles_per_segment` real tiles.

Every kernel must guard empty tiles:

```cpp
if (tile_start >= n || tile_start >= segment_start + segment_rows) {
  d_partial_out[blockIdx.x] = 0.0;
  return;
}
```

Then compute `tile_end`.

Do not compute:

```cpp
tile_rows_actual = tile_end - tile_start;
```

until after the empty-tile guard. Otherwise an unsigned underflow is possible when `tile_start >= n`.

Future optimization:

```text
compact grid for the final partial segment
```

This can reduce empty blocks, but it requires a more complex block-to-segment mapping. Do not implement it in v1 unless the empty-block overhead is measurable.

For rowpack16:

```text
pack_width = 16
```

For rowpack4 if implemented:

```text
pack_width = 4
```

For scalar fallback:

```text
pack_width = 1
```

### 7.3 Tile row range

For a given block:

```cpp
uint64_t segment_id = blockIdx.x / tiles_per_segment;
uint64_t tile_in_segment = blockIdx.x % tiles_per_segment;

uint64_t segment_start = segment_id * segment_rows;
uint64_t tile_start = segment_start + tile_in_segment * tile_rows;
uint64_t tile_end = min(tile_start + tile_rows,
                        min(segment_start + segment_rows, n));
```

This ensures a tile never crosses a segment boundary.

That matters because each segment has its own:

```text
base
basis weights
```

### 7.4 Required rowpack16 alignment and tail handling

`tile_end` boundary clamping is necessary but not sufficient for `rowpack16`.

The implementation must never issue a `uint4` load unless all 16 bytes are inside the current tile and the starting row is 16-byte aligned relative to the subcolumn base pointer.

Use this rule:

```text
scalar prologue:
  rows from tile_start until the next 16-row aligned row, if tile_start is not aligned

rowpack16 body:
  full 16-row packs where row + 15 < tile_end

scalar epilogue:
  remaining rows after the last full 16-row pack
```

Reference pseudocode:

```cpp
uint64_t aligned_start = tile_start;

while (aligned_start < tile_end && (aligned_start & 15ULL) != 0)
  ++aligned_start;

uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16) * 16;

for (uint64_t row = tile_start + threadIdx.x; row < aligned_start; row += blockDim.x) {
  // scalar uint8_t load for prologue rows
}

uint64_t first_pack = aligned_start / 16;
uint64_t pack_count = (full_pack_end - aligned_start) / 16;

for (uint64_t pack = threadIdx.x; pack < pack_count; pack += blockDim.x) {
  uint64_t row = aligned_start + pack * 16;
  // safe uint4 load from subcolumn[p] + row
}

for (uint64_t row = full_pack_end + threadIdx.x; row < tile_end; row += blockDim.x) {
  // scalar uint8_t load for epilogue rows
}
```

Important:

```text
The scalar prologue and scalar epilogue are still parallel block-stride loops.
Do not let every thread execute the same scalar row range.
```

The `first_pack` value is useful if the implementation indexes a `uint4 *` view:

```cpp
const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[p]);
uint4 pack_value = plane128[first_pack + pack];
```

For the v1 default configuration:

```text
segment_rows = 1048576
tile_rows = 4096
```

both are divisible by 16, so only the global `n` tail should normally hit the scalar epilogue.

Still implement the scalar prologue/epilogue in the kernel. Do not rely on all future CLI inputs being multiples of 16.

Host-side padding is optional future work, not the v1 default. If padding is introduced later:

- padded bytes must be initialized to zero
- `n` must remain the logical row count used for validation and throughput
- kernels must not include padded rows in the logical SUM

### 7.5 Default values

Recommended v1 defaults:

```text
n = 100000000
segment_rows = 1048576
block_threads = 256
items_per_thread = 1
pack_width = 16 for rowpack16
tile_rows = 4096 for rowpack16
```

This gives many blocks:

```text
100000000 / 4096 ~= 24415 tiles
```

which is enough to saturate the GPU.

Do not use one block per 1M-row segment; that would create too few blocks.

## 8. Kernel Variants

### 8.1 Required v1 kernel

Implement:

```cpp
template <int DEPTH, int ITEMS_PER_THREAD>
__global__ void progressive_sum_rowpack16_u8subcols(...);
```

Where:

```text
DEPTH = refinement_depth in [0, 7] for 8 encoded subcolumns
ITEMS_PER_THREAD = fixed template value, default 1
```

The subcolumn loop over `DEPTH` must be compile-time specialized:

```cpp
#pragma unroll
for (int p = 0; p <= DEPTH; ++p)
```

Do not use a runtime `for (p < refinement_depth)` in the hot path.

### 8.2 Optional v1 comparison kernels

Optional, if time permits:

```cpp
progressive_sum_scalar_u8subcols<DEPTH>
progressive_sum_rowpack4_u8subcols<DEPTH>
```

These are useful ablations, but not required for the first Exp3 v1.

The main path is `rowpack16`.

### 8.3 Do not implement shared128 in Exp3 v1

Do not port Exp1 `shared128` into Exp3 v1.

Reason for v1:

- Exp1 already shows `shared128` is a diagnostic path.
- It has `4x` overfetch.
- It performed much worse than rowpack.
- Exp3 should not start from a diagnostic overfetch strategy.

Do not describe `shared128` as a permanent anti-pattern.

It may be worth re-evaluating later if Exp3/Exp4 introduces cross-subcolumn filtering, reuse, broadcast, or bitwise logic that benefits from shared memory staging. That is deferred work, not the v1 main path.

## 9. Kernel Computation Details

### 9.1 Subcolumn accumulation

Each row has one byte in each encoded subcolumn:

```cpp
uint8_t subcolumn_j_i;
```

Each thread accumulates one integer sum per active subcolumn:

```cpp
uint64_t sub0_sum = 0;
uint64_t sub1_sum = 0;
...
```

For `uint8_t` subcolumns, one subcolumn's maximum sum is:

```text
255 * n
```

For `n = 10^9`, this is `2.55e11`, which fits in `uint64_t`.

For default synthetic values, validation should use small values like:

```text
subcolumn_j[i] = j + 1
```

so overflow cannot occur.

For performance runs, values can still be small; memory behavior is what matters.

### 9.2 Explicit accumulators

For `DEPTH <= 7` with 8 total subcolumns, the simplest implementation uses separate accumulators:

```cpp
uint64_t sub0 = 0;
uint64_t sub1 = 0;
...
uint64_t sub7 = 0;
```

Then only use subcolumns `0..DEPTH`.

Alternative:

```cpp
uint64_t sub_sums[8];
```

But be careful: local arrays can become local memory if the compiler cannot keep them in registers.

Recommended v1:

```text
Use explicit scalar accumulators or a templated helper that compiler can fully unroll.
Check NCU local spill requests.
```

### 9.3 Rowpack16 subcolumn load

Use the same mechanism proven in Exp1:

```cpp
const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[p]);
uint4 pack = plane128[pack_index];
```

Then use:

```cpp
byte_sum_u32(pack.x)
byte_sum_u32(pack.y)
byte_sum_u32(pack.z)
byte_sum_u32(pack.w)
```

Reference:

- `benchmarks/experiment1/exp1_kernels_rowpack.cuh`

Required safety rule:

```text
Only use the uint4 path for full aligned 16-row chunks.
Use scalar uint8_t loads for scalar prologue and tail rows.
```

This is required for correctness when `n`, `segment_rows`, or a segment-local tile boundary is not a multiple of 16.

### 9.4 Applying base and basis

After block reduction of subcolumn sums:

```text
partial_sum =
  tile_rows_actual * base
  + sub0_sum * basis0
  + sub1_sum * basis1
  + ...
  + subDEPTH_sum * basisDEPTH
```

Use `double` for this final computation.

This final FP work occurs once per block, not once per row.

That preserves the initial guide's intent:

```text
integer accumulation of byte subcolumns in the hot loop
one segment/tile-level weighted floating-point combination
```

### 9.5 Block reduction

Exp1 currently reduces one scalar `unsigned long long`.

Exp3 needs to reduce multiple quantities:

```text
sub0_sum
sub1_sum
...
sub7_sum
```

Implement a small reduction helper in Exp3 common code.

Options:

1. Reduce each accumulator independently using a warp/block reduction helper.
2. Store per-thread partials in shared memory and reduce all fields.

Recommended v1:

```text
Use a simple templated block reduction helper per uint64 accumulator.
Call it once per active subcolumn accumulator.
```

Critical correctness rule:

```text
Do not directly reuse Exp1 `block_reduce_store` as a return-value helper and
then call it repeatedly for sub0, sub1, ..., sub7.
```

Exp1's helper is safe for its original use because each kernel calls it once
and it writes one output value. It uses a shared `warp_sums` buffer and does
not need a trailing `__syncthreads()` for that one-shot pattern.

For Exp3, if the same shared buffer is reused across multiple reductions inside
one block, the helper must either:

1. end with a `__syncthreads()` before returning, or
2. use a different shared-memory offset for each accumulator, or
3. use one fused/vectorized multi-accumulator reduction.

Otherwise a later accumulator reduction can overwrite shared memory while warp
0 is still reading the previous accumulator's warp partials.

This is not the most instruction-minimal approach, but it is easy to review and safe.

Performance impact should be small because the kernel is memory dominated for large `n`.

If reduction overhead appears in NCU, optimize later.

Do not hide this risk. At `DEPTH=7`, a naive implementation may perform 8 separate block reductions, which can add repeated shared-memory traffic and repeated `__syncthreads()`.

Deferred v1.1 optimization:

```text
fused/vectorized reduction
```

Possible design:

- each thread keeps one accumulator per active subcolumn
- write shared memory in SOA form: `shared[subcolumn][thread]`
- reduce multiple subcolumns in one staged reduction pass where possible
- assign different warps to finalize different subcolumns
- keep one final barrier before block output

Do not implement this before the simple version is profiled. The first implementation should be correct, readable, and measurable.

## 10. Output Semantics

Each block writes:

```cpp
d_partial_out[blockIdx.x] = partial_sum;
```

The timed benchmark does not need to copy the full output back to host.

For validation:

1. Run one non-timed launch.
2. Copy `d_partial_out` to host.
3. Sum all partials on CPU.
4. Compare to expected synthetic SUM.

Do not copy `d_partial_out` during the timed loop.

## 11. File Layout

Create:

```text
benchmarks/experiment3/
  CMakeLists.txt
  README.md
  bench_progressive_aggregation.cu
  exp3_common.cuh
  exp3_kernels_progressive.cuh
```

Add scripts:

```text
scripts/legacy/root_runners/run_exp3.sh
scripts/legacy/root_runners/run_exp3.sh
```

### 11.1 `bench_progressive_aggregation.cu`

Responsibilities:

- parse CLI
- allocate synthetic arrays
- initialize synthetic data
- allocate `d_partial_out`
- dispatch templated kernels
- warmup and timed loops
- CSV output
- optional validation
- cleanup

Do not put large kernel bodies here.

### 11.2 `exp3_common.cuh`

Responsibilities:

- CUDA reduction helpers
- `ceil_div`
- fixed constants
- plane pointer structs
- `byte_sum_u32`
- small device helpers

Reuse ideas from:

- `benchmarks/experiment1/exp1_scan_common.cuh`
- `benchmarks/experiment1/exp1_kernels_rowpack.cuh`

Do not include host CLI parsing here.

### 11.3 `exp3_kernels_progressive.cuh`

Responsibilities:

- `progressive_sum_rowpack16_u8subcols<DEPTH, ITEMS_PER_THREAD>`
- kernel pointer helper
- launch helper

Expected helpers:

```cpp
const void *progressive_sum_rowpack16_kernel_ptr(int refinement_depth);

void launch_progressive_sum_rowpack16(
    int refinement_depth,
    int grid,
    int block_threads,
    ...);
```

Use switch dispatch for `DEPTH=0..7` when `--subcolumns 8`.

Do not use runtime-depth hot loops.

## 12. CLI Design

Required CLI:

```bash
./build/exp3/bench_progressive_aggregation \
  --device 0 \
  --n 100000000 \
  --segment_rows 1048576 \
  --subcolumns 8 \
  --refine_min 0 \
  --refine_max 7 \
  --load_strategy rowpack16 \
  --block 256 \
  --items_per_thread 1 \
  --warmup 10 \
  --iters 200 \
  --csv results/exp3/progressive_aggregation.csv
```

### 12.1 Required options

Implement:

```text
--device
--n
--segment_rows
--subcolumns
--refine_min
--refine_max
--load_strategy
--block
--items_per_thread
--warmup
--iters
--csv
--validate
```

### 12.2 `load_strategy`

For v1:

```text
rowpack16
```

Optional:

```text
scalar
rowpack4
```

If only `rowpack16` is implemented, reject other values with a clear error.

### 12.3 Argument constraints

Validate:

```text
n > 0
segment_rows > 0
subcolumns in [1, 8]
refine_min >= 0
refine_max < subcolumns
refine_min <= refine_max
block is multiple of 32
items_per_thread == 1 in v1 unless additional template dispatch is implemented
```

If `segment_rows` is smaller than `tile_rows`, allow it; `tiles_per_segment=1`.

If `segment_rows` is not a multiple of `tile_rows`, allow it; the last tile in each segment handles the segment tail.

## 13. CSV Schema

Use this schema:

```text
benchmark,dataset,mode,aggregation,load_strategy,refinement_depth,
n,segment_rows,tile_rows,subcolumns,subcolumn_bits,
logical_subcolumns_read,logical_bytes,block,grid,warmup,iters,
ms_per_iter,rows_per_sec,billion_rows_per_sec,logical_GBps,
accumulator_bits,base_value,basis_mode,
synthetic_sum_abs_error_bound,synthetic_avg_abs_error_bound,error_bound_source,
validated,device,sm,cc_major,cc_minor
```

### 13.1 Fixed v1 values

```text
benchmark=progressive_aggregation
dataset=synthetic
mode=synthetic_fixed_point_subcolumns
aggregation=sum
load_strategy=rowpack16
subcolumn_bits=8
accumulator_bits=64
base_value=0
basis_mode=synthetic_pow2
error_bound_source=synthetic_fixed_point
```

### 13.2 Throughput formulas

```text
seconds = ms_per_iter / 1000
rows_per_sec = n / seconds
billion_rows_per_sec = rows_per_sec / 1e9
logical_GBps = logical_bytes / seconds / 1e9
```

### 13.3 Logical bytes formula

```text
logical_bytes = n * (1 + refinement_depth)
```

because v1 reads byte subcolumns `0..refinement_depth`.

## 14. Synthetic Data Initialization

Implement a non-timed initialization kernel.

Default values:

```text
subcolumn[p][i] = p + 1
base_segment = 0.0
basis[p] = 2^(-8 * p)
```

Initialize metadata for every segment:

```cpp
for (uint64_t s = 0; s < num_segments; ++s) {
  h_segment_base[s] = 0.0;
  for (int p = 0; p < MAX_SUBCOLUMNS; ++p) {
    h_subcolumn_basis[s * MAX_SUBCOLUMNS + p] = pow(2.0, -8.0 * p);
  }
}
```

Do not initialize only one global basis vector unless the kernel signature is explicitly changed to use global basis weights.

This makes validation easy:

```text
expected_sum(depth) =
  n * (1 * 2^0)
  + n * (2 * 2^-8)
  + n * (3 * 2^-16)
  + ...
  + n * ((depth + 1) * 2^(-8 * depth))
```

For `depth = 0`:

```text
expected_sum = n
```

Use `double` expected value and a small tolerance.

Suggested tolerance:

```text
absolute tolerance = max(1e-6 * abs(expected), 1e-3)
```

Because the synthetic basis decays exponentially, deep subcolumns can contribute
values near the precision limit of a large `double` SUM. For example, at
`n=1e8` and `depth=7`, the leading term is around `1e8`, while the `p=7`
term is around `1e-8`.

Small differences between CPU sequential summation and GPU block-parallel
summation are expected because floating-point addition is not associative.

If floating-point summation order causes larger differences, report the observed error and justify the tolerance.

## 15. Validation

If `--validate` is passed:

1. Launch the selected kernel once outside timed loop.
2. Copy `d_partial_out` to host.
3. Sum `grid` partial values.
4. Compare with synthetic expected sum.
5. Record `validated=true` in CSV.

If validation is not run:

```text
validated=false
```

Validation must not be included in `ms_per_iter`.

## 16. Benchmark Runner

Add:

```text
scripts/legacy/root_runners/run_exp3.sh
```

Follow the style of:

```text
scripts/legacy/root_runners/run_exp1.sh
```

The runner should:

- create `results/exp3/run_<timestamp>_job<id>_<gpu_tag>/`
- write `setup_estimate.txt`
- build `benchmarks/experiment3`
- run the benchmark
- write `run_meta.txt`
- write `repro_command.txt`
- write `ncu_command_template.txt`

Add root wrapper:

```text
scripts/legacy/root_runners/run_exp3.sh
```

following the existing root `scripts/legacy/root_runners/run_exp1.sh` convention.

## 17. CMake

Add:

```text
benchmarks/experiment3/CMakeLists.txt
```

Use the same compile style as Exp1:

```cmake
cmake_minimum_required(VERSION 3.24)
project(gpu_byteplane_scan_experiment3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(bench_progressive_aggregation bench_progressive_aggregation.cu)

target_compile_options(bench_progressive_aggregation PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -lineinfo --ptxas-options=-v>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_progressive_aggregation PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
```

Build command:

```bash
cmake -S benchmarks/experiment3 -B build/exp3 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp3 -j
```

During implementation, keep the build log and inspect `ptxas` output for:

```text
registers per thread
stack frame size
spill stores
spill loads
local memory usage
```

Acceptance target for v1:

```text
spill stores = 0
spill loads = 0
```

## 18. Benchmark Plan

### 18.1 Smoke tests

Run on GPU:

```bash
./build/exp3/bench_progressive_aggregation \
  --device 0 \
  --n 1024 \
  --segment_rows 1024 \
  --subcolumns 3 \
  --refine_min 0 \
  --refine_max 2 \
  --load_strategy rowpack16 \
  --block 256 \
  --items_per_thread 1 \
  --warmup 0 \
  --iters 1 \
  --validate \
  --csv /tmp/exp3_smoke.csv
```

Acceptance:

- exits successfully
- outputs 3 CSV rows for depths 0, 1, 2
- validation passes

Also run one boundary smoke test:

```bash
./build/exp3/bench_progressive_aggregation \
  --device 0 \
  --n 1031 \
  --segment_rows 257 \
  --subcolumns 3 \
  --refine_min 0 \
  --refine_max 2 \
  --load_strategy rowpack16 \
  --block 256 \
  --items_per_thread 1 \
  --warmup 0 \
  --iters 1 \
  --validate \
  --csv /tmp/exp3_boundary_smoke.csv
```

This test intentionally uses non-16-multiple `n` and `segment_rows` to exercise scalar prologue/epilogue handling.

### 18.2 Full benchmark

Run:

```bash
./build/exp3/bench_progressive_aggregation \
  --device 0 \
  --n 100000000 \
  --segment_rows 1048576 \
  --subcolumns 8 \
  --refine_min 0 \
  --refine_max 7 \
  --load_strategy rowpack16 \
  --block 256 \
  --items_per_thread 1 \
  --warmup 10 \
  --iters 1000 \
  --csv results/exp3/progressive_aggregation_rowpack16.csv
```

Expected qualitative result:

```text
refinement_depth increases -> logical bytes increases
rows/sec should generally decrease
logical_GBps should approach Exp1 rowpack16-style bandwidth if aggregation overhead is small
```

Do not require perfect monotonicity for every point until real profiling confirms overheads.

### 18.3 Comparison targets

Compare against:

- Exp1 `rowpack16` throughput
- Exp1 `contiguous64` throughput
- Exp1 `byte ilp4` as scalar-load ablation

Do not compare Exp3 row-for-row against Exp1 as identical work.

Reason:

Exp3 does more work:

- weighted subcolumn aggregation
- segment base/basis correction
- multiple accumulator reductions

The relevant question is whether Exp3 remains bandwidth-oriented and scales with refinement depth.

## 19. Nsight Compute Plan

Profile:

```text
refinement_depth = 0
refinement_depth = 4
refinement_depth = 7
```

For:

```text
load_strategy=rowpack16
```

Use environment noted in Exp1 report:

```bash
ml load cuda/12.6
source activate gpu-byteplane-scan
```

Use:

```bash
ncu --set full \
  --target-processes all \
  --launch-count 1 \
  --launch-skip 0 \
  --import-source yes \
  --source-folders <repo-root> \
  --export <output> \
  ./build/exp3/bench_progressive_aggregation ...
```

Inspect:

- SASS load opcode
- DRAM throughput
- L2 throughput
- achieved occupancy
- registers/thread
- local spill requests
- executed instructions
- Long Scoreboard
- eligible warps / scheduler

Expected:

```text
encoded subcolumn loads should use LDG.E.128 or equivalent 128-bit load
local spill requests should be 0
DRAM/L2 utilization should rise with refinement depth
```

Do not claim success unless NCU confirms the wide load.

## 20. Relationship to Exp2

Exp3 v1 does not wait for Exp2.

Exp3 v1 produces:

```text
dataset=synthetic
refinement_depth -> throughput
```

Exp2 will later produce:

```text
dataset
aggregation
refinement_depth
error
error_bound
```

The join key should be:

```text
dataset, aggregation, refinement_depth
```

For v1:

```text
dataset=synthetic
aggregation=sum
```

When Exp2 is ready, add an analysis script that joins:

```text
results/exp3/...csv
results/exp2/...csv
```

and emits:

```text
precision_throughput_curve.csv
```

Do not block Exp3 kernel development on this script.

## 21. Acceptance Criteria

### 21.1 Code acceptance

Required files exist:

```text
benchmarks/experiment3/CMakeLists.txt
benchmarks/experiment3/README.md
benchmarks/experiment3/bench_progressive_aggregation.cu
benchmarks/experiment3/exp3_common.cuh
benchmarks/experiment3/exp3_kernels_progressive.cuh
scripts/legacy/root_runners/run_exp3.sh
scripts/legacy/root_runners/run_exp3.sh
```

### 21.2 Build acceptance

This must pass:

```bash
cmake -S benchmarks/experiment3 -B build/exp3 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp3 -j
```

### 21.3 Runtime acceptance

Smoke test with `--validate` must pass.

Full run must produce one CSV row per refinement depth.

### 21.4 Semantic acceptance

- `refinement_depth=0` reads encoded subcolumn 0 only.
- `refinement_depth=d` reads exactly encoded subcolumns `0..d`.
- rowpack16 reads consecutive rows from the same encoded subcolumn.
- rowpack16 never reads additional precision bytes from the same row.
- rowpack16 never issues an out-of-bounds `uint4` load.
- non-16-row tails are handled by scalar `uint8_t` loads.
- scalar prologue and epilogue rows are processed with block-stride loops, not duplicated by every thread.
- empty tiles in the rectangular grid write `0.0` and return before computing tile row counts.
- per-segment basis is read as `d_subcolumn_basis[segment_id * MAX_SUBCOLUMNS + p]`.
- base/basis correction is applied once per tile/block, not per row.

### 21.5 Performance acceptance

Do not require a fixed GB/s target before first run.

But the first H200 full run should report:

- rows/sec
- billion rows/sec
- logical GB/s
- `refinement_depth=0,3,7` comparison

If `rowpack16` Exp3 throughput is far below Exp1 rowpack16, inspect:

- subcolumn load shape
- reduction overhead
- tile size
- register spills
- SASS load opcode

### 21.6 NCU acceptance

For `refinement_depth=7`, NCU should confirm:

- 128-bit encoded subcolumn loads
- scalar `uint8_t` loads only for boundary prologue/epilogue rows
- zero local spills
- reasonable occupancy

If NCU shows scalar `LDG.E.U8` for the main full-pack body, the implementation does not satisfy this spec.

## 22. Non-Goals

Do not implement:

- real Buff encoding
- real dataset loading
- Exp2 final error analysis
- multi-GPU
- filter kernels
- shared128 strategy in v1
- 2-byte subcolumn variant
- CUB/Thrust reductions
- a generic query engine

Do not change:

- Exp1 code
- Exp1 results
- contiguous baseline

## 23. Risks and Mitigations

### 23.1 Reduction overhead

Exp3 reduces multiple accumulators, unlike Exp1.

Mitigation:

- keep v1 simple
- profile first
- optimize reduction only if NCU shows it matters

If NCU shows repeated barrier cost or shared-memory serialization at high depth, implement the deferred fused/vectorized reduction described in Section 9.5.

### 23.2 Rowpack16 boundary and alignment bugs

`uint4` loads are unsafe for partial 16-row chunks.

Mitigation:

- implement scalar prologue for unaligned tile starts
- implement scalar epilogue for tails
- only issue `uint4` loads when `row` is 16-row aligned and `row + 15 < tile_end`
- add smoke tests where `n` is not divisible by 16
- add smoke tests where `segment_rows` is not divisible by 16

### 23.3 First subcolumn may not match real Buff bit split

The v1 model treats every encoded subcolumn as a full byte with a simple basis
sequence. Real Buff byte-oriented splitting may pack integer bits plus some
fractional bits into the first byte.

Mitigation:

- state that v1 is a synthetic Buff-aligned performance benchmark
- keep `basis_j` explicit in metadata
- replace synthetic basis generation with real encoder metadata in v2

### 23.4 Register pressure

Multiple subcolumn accumulators may increase registers/thread.

Mitigation:

- use compile-time `DEPTH`
- avoid local arrays that spill
- compile with `--ptxas-options=-v`
- inspect registers/thread and spill counts in the build log
- verify NCU local spill requests

### 23.5 Segment-tile mismatch

Tiles must not cross segment boundaries.

Mitigation:

- use `segment_id` and `tile_in_segment` mapping
- clamp `tile_end` to both segment end and `n`

### 23.6 FP64 final combination overhead

The final per-block combination converts integer accumulator values to `double`
and applies base/basis weights.

Mitigation:

- keep `double` as v1 default for correctness and Buff-aligned semantics
- remember this work happens once per block, not once per row
- only add an FP32 ablation if NCU shows final conversion or FP64 issue stalls are measurable

### 23.7 Shared128 may become useful in a later query shape

Exp1 shows `shared128` is not a good main path for pure byte-plane scan.

Mitigation:

- do not implement `shared128` in Exp3 v1
- keep it as deferred work for future cross-subcolumn filtering, broadcast, or reuse-heavy kernels

### 23.8 Confusing v1 synthetic error with final Exp2 error

Mitigation:

- name synthetic bounds explicitly
- keep final precision-throughput curve join as later work

## 24. Developer Implementation Order

Follow this order:

1. Create `benchmarks/experiment3/CMakeLists.txt`.
2. Create `bench_progressive_aggregation.cu` with CLI parsing and empty skeleton.
3. Create `exp3_common.cuh` with reduction helpers and `byte_sum_u32`.
4. Implement synthetic allocation and initialization.
5. Implement `progressive_sum_rowpack16_u8subcols<DEPTH, ITEMS_PER_THREAD>`.
6. Implement switch dispatch for `DEPTH=0..7`.
7. Implement timing loop and CSV output.
8. Implement `--validate`.
9. Add `benchmarks/experiment3/README.md`.
10. Add `scripts/legacy/root_runners/run_exp3.sh`.
11. Add root `scripts/legacy/root_runners/run_exp3.sh`.
12. Run smoke test.
13. Run H200 full benchmark.
14. Run NCU at depths `0,3,7`.
15. Write a handoff report with results.

## 25. Developer Report Format

When done, report with:

1. Summary
2. Files Added
3. CLI
4. Data Layout
5. Kernel Mapping
6. Validation Result
7. Smoke Test Result
8. Full Benchmark Result
9. NCU Result
10. Caveats
11. Deferred Work

Avoid unsupported claims.

Do not say:

```text
Exp3 is complete
```

unless throughput, validation, and NCU evidence are all available.

Say instead:

```text
Exp3 v1 synthetic progressive SUM throughput benchmark is implemented.
```

## 26. References Inside This Repo

Read these before implementing:

- `spec/M03-initial-guide.md`
- `spec/2026-04-14_Exp3_Progressive_Aggregation.md`
- `research/2026-04-16_exp1_rowpack_benchmark_ncu_report.md`
- `benchmarks/experiment1/exp1_kernels_rowpack.cuh`
- `benchmarks/experiment1/exp1_scan_common.cuh`
- `scripts/legacy/root_runners/run_exp1.sh`
- `benchmarks/experiment1/CMakeLists.txt`

Use Exp1 rowpack as the model for:

- row-wise packed load semantics
- `uint4` load
- byte summation helper
- tail handling
- compile-time depth dispatch pattern

Use Exp1 runner as the model for:

- run directory
- metadata
- repro command
- NCU command template
