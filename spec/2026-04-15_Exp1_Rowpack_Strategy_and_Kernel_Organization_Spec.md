# Exp1 Rowpack Strategy and Kernel Organization Spec

Date: 2026-04-15  
Status: Draft for implementation planning  
Owner: Nick / Codex  
Target audience: junior developer implementing the next Exp1 kernel iteration

## 1. Background

Experiment 1 is the first gate of the project.

The core question is:

> Can GPU byte-plane scan save bandwidth roughly proportional to the number of byte-planes read, while approaching the throughput of a contiguous full-precision scan when `k = 8`?

The current benchmark target is:

- `benchmarks/experiment1/bench_byteplane_scan.cu`

The current baseline target is:

- `benchmarks/experiment1/bench_contiguous_baseline.cu`

The current byte-plane benchmark already contains several experimental paths:

- scalar byte path: `strategy=byte`
- old packed path: `strategy=packed32`
- shared-memory staging path: `strategy=shared128`
- 2-byte-plane path: `--plane_bytes 2 --strategy byte`

However, these paths are all implemented in one `.cu` file, their names are not precise enough for research communication, and not all paths have the same level of compile-time specialization.

This spec defines the next implementation direction:

1. Keep the current scalar byte baseline and ILP4 variant.
2. Formalize row-wise packed loads as first-class strategies.
3. Add a true 128-bit row-wise packed strategy.
4. Keep shared-memory staging as a diagnostic strategy.
5. Keep 2-byte planes as a fallback / sensitivity strategy.
6. Split kernel implementations into small focused files without introducing unnecessary abstraction.

## 2. Key Conceptual Rule

The project goal is progressive byte-plane precision.

That means:

```text
For each row, only the first k byte-planes are read.
```

The project goal does not require:

```text
Every GPU memory instruction must load exactly one byte.
```

Therefore, wider loads are valid if and only if they load consecutive rows from the same byte-plane.

Valid row-wise packing:

```text
plane 7:
row0_b7 row1_b7 row2_b7 row3_b7 row4_b7 ...

rowpack4 reads:
row0_b7 row1_b7 row2_b7 row3_b7
```

Invalid precision-wise packing:

```text
row0:
b7 b6 b5 b4 b3 b2 b1 b0

invalid packed read:
row0_b7 row0_b6 row0_b5 row0_b4
```

The first preserves progressive precision.

The second violates progressive precision because `k = 1` would secretly load more than one precision byte for the same row.

## 3. Current State

### 3.1 Scalar byte path

Current location:

- `benchmarks/experiment1/bench_byteplane_scan.cu`

Current kernels:

- `scan_planes_u8_byte_unrolled<K>`
- `scan_planes_u8_byte_ilp4<K>`

Current CLI:

```bash
--strategy byte
--byte_variant baseline|ilp4
```

This path is the current cleanest A/B comparison:

- `baseline`: scalar byte load, one accumulator
- `ilp4`: scalar byte load, 4 independent accumulators

This path must be preserved.

### 3.2 Old packed32 path

Current kernel:

- `scan_planes_u8_packed32`

Current CLI:

```bash
--strategy packed32
```

Current issue:

- It is conceptually row-wise packed loading.
- It reads four consecutive rows from one byte-plane.
- But the name `packed32` is ambiguous.
- It uses runtime `k`, not `template<int K>`.
- It is not yet a clean peer to the scalar byte `baseline` / `ilp4` variants.

This path should be replaced conceptually by `rowpack4`.

Compatibility note:

- Do not delete `packed32` immediately.
- Either keep it as a backward-compatible alias to `rowpack4`, or keep the old path temporarily while adding `rowpack4`.
- The implementation report must state which choice was made.

### 3.3 shared128 path

Current kernel:

- `scan_planes_u8_shared128`

Current CLI:

```bash
--strategy shared128
```

This is not the same as `rowpack16`.

`shared128` means:

```text
one warp stages a 128-byte chunk into shared memory,
then lanes consume individual bytes from shared memory
```

`rowpack16` means:

```text
one thread performs a 128-bit load from one byte-plane,
representing 16 consecutive rows
```

`shared128` may overfetch and should be treated as a diagnostic path, not the primary semantic implementation.

### 3.4 2-byte planes

Current path:

```bash
--plane_bytes 2 --strategy byte
```

Current kernel:

- `scan_planes_u16`

This is a valid fallback / sensitivity path, but it changes precision granularity:

```text
8 one-byte planes -> 4 two-byte planes
```

This should not be used as the first main result for byte-level progressive precision.

## 4. Naming Decisions

Use these names in code comments, docs, CSV metadata where possible, and implementation reports.

### 4.1 `byte`

Scalar byte-plane load.

Meaning:

```text
one thread reads one uint8_t from one byte-plane per logical row
```

Allowed variants:

```text
baseline
ilp4
```

### 4.2 `rowpack4`

Row-wise 32-bit packed load.

Meaning:

```text
one thread reads one uint32_t from one byte-plane
that uint32_t contains 4 consecutive rows from that plane
```

This is the cleaned-up successor to the old `packed32` strategy.

### 4.3 `rowpack16`

Row-wise 128-bit packed load.

Meaning:

```text
one thread reads one 128-bit vector from one byte-plane
that vector contains 16 consecutive rows from that plane
```

Recommended CUDA type:

```cpp
uint4
```

This is not shared-memory staging.

### 4.4 `shared128`

Warp-level shared-memory staging.

Meaning:

```text
one warp stages 128 bytes from one byte-plane into shared memory,
then lanes consume individual bytes
```

This is diagnostic because it may overfetch logical bytes.

### 4.5 `plane16`

Research-facing name for the 2-byte-plane path.

Current CLI can remain:

```bash
--plane_bytes 2 --strategy byte
```

Do not force a CLI rename in this task unless the implementation is already touching strategy parsing in a controlled way.

## 5. Goals

### 5.1 Performance goals

The new implementation should allow the following comparisons:

```text
contiguous64
byte baseline
byte ilp4
rowpack4
rowpack16
shared128
plane16
```

The most important comparison is:

```text
contiguous64 vs rowpack16 k=8
```

because both read:

```text
n * 8 logical bytes
```

The key question is whether row-wise packed loads reduce the gap between byte-plane scan and contiguous scan.

### 5.2 Semantic goals

All rowpack strategies must preserve this rule:

```text
k controls how many byte-planes are read.
```

For `k = 1`, rowpack strategies must read only one byte-plane.

For `k = 8`, rowpack strategies must read all eight byte-planes.

No strategy may read extra precision bytes from the same row.

### 5.3 Code organization goals

The current benchmark file is becoming too large.

This task should split kernel implementations into focused files while keeping the benchmark simple.

The goal is not to create a class hierarchy, registry system, or generic framework.

The goal is only to make the kernel paths understandable and maintainable.

## 6. Non-Goals

This task must not:

- implement Exp2 error analysis
- implement Exp3 Buff-style aggregation
- implement Exp4 progressive filter
- modify `bench_contiguous_baseline.cu`
- change the timing model
- change the reduction sink semantics
- remove the scalar byte baseline
- remove the byte ILP4 variant
- change the meaning of `k`
- add query correctness logic into the timed benchmark loop
- introduce CUB, Thrust, or third-party dependencies
- introduce a large framework or class hierarchy
- claim performance improvements without benchmark and Nsight Compute evidence

## 7. Proposed File Layout

Create small CUDA header files under:

```text
benchmarks/experiment1/
```

Recommended layout:

```text
benchmarks/experiment1/
  bench_byteplane_scan.cu
  bench_contiguous_baseline.cu
  exp1_scan_common.cuh
  exp1_kernels_byte.cuh
  exp1_kernels_rowpack.cuh
  exp1_kernels_shared.cuh
  exp1_kernels_plane16.cuh
```

### 7.1 `bench_byteplane_scan.cu`

Responsibilities:

- CLI parsing
- options struct
- device allocation
- pointer setup
- strategy dispatch
- occupancy grid selection
- warmup loop
- timed loop
- CSV writing
- cleanup

It should include the kernel headers.

It should not contain large kernel bodies after this refactor.

### 7.2 `exp1_scan_common.cuh`

Responsibilities:

- `warp_reduce_sum_ull`
- `block_reduce_store`
- plane pointer structs
- small device helpers shared by kernels

Candidate contents:

```cpp
template <typename T, int N>
struct PlanePointers
{
  const T *ptrs[N];
};

using U8Planes = PlanePointers<uint8_t, 8>;
```

Do not move CLI parsing or host benchmark logic here.

### 7.3 `exp1_kernels_byte.cuh`

Responsibilities:

- scalar byte baseline kernel
- scalar byte ILP4 kernel
- byte kernel pointer helpers
- byte launch helpers

Expected contents:

```cpp
template <int K>
__global__ void scan_planes_u8_byte_unrolled(...);

template <int K>
__global__ void scan_planes_u8_byte_ilp4(...);

const void *scan_planes_u8_byte_kernel_ptr(ByteVariant variant, int k);

void launch_scan_planes_u8_byte(ByteVariant variant, int k, ...);
```

No rowpack code should be added here.

### 7.4 `exp1_kernels_rowpack.cuh`

Responsibilities:

- `rowpack4` kernel
- `rowpack16` kernel
- rowpack pointer helpers
- rowpack launch helpers

Expected contents:

```cpp
template <int K>
__global__ void scan_planes_u8_rowpack4(...);

template <int K>
__global__ void scan_planes_u8_rowpack16(...);

const void *scan_planes_u8_rowpack4_kernel_ptr(int k);
const void *scan_planes_u8_rowpack16_kernel_ptr(int k);

void launch_scan_planes_u8_rowpack4(int k, ...);
void launch_scan_planes_u8_rowpack16(int k, ...);
```

All `K` loops must be compile-time specialized.

### 7.5 `exp1_kernels_shared.cuh`

Responsibilities:

- `shared128` staging kernel
- shared strategy launch helper if needed

This path can remain runtime `k` initially if the implementation scope must stay small, but the developer should document this caveat.

If time permits, add templated `K` specialization later.

### 7.6 `exp1_kernels_plane16.cuh`

Responsibilities:

- `scan_planes_u16`
- plane16 launch helper if needed

This path can remain close to existing code.

Do not mix it with rowpack kernels.

## 8. Strategy and CLI Design

### 8.1 Required strategies

After implementation, the benchmark should support at least:

```bash
--strategy byte
--strategy rowpack4
--strategy rowpack16
--strategy shared128
```

For compatibility, either:

```bash
--strategy packed32
```

should remain accepted as an alias of `rowpack4`, or it should remain accepted as the old implementation until explicitly removed in a later cleanup.

Preferred behavior:

```text
packed32 is accepted as a backward-compatible alias for rowpack4.
CSV writes strategy=rowpack4.
run_meta or stderr may mention that packed32 was normalized to rowpack4.
```

If that is too invasive, keep `packed32` as-is and add `rowpack4` separately.

The implementation report must clearly state which behavior was chosen.

### 8.2 `byte_variant`

`--byte_variant` only applies to:

```bash
--strategy byte --plane_bytes 1
```

Valid values remain:

```bash
baseline
ilp4
```

Do not make `rowpack4` or `rowpack16` byte variants unless there is a strong reason.

Reason:

- `byte_variant` currently means scalar byte kernel structure.
- `rowpack4` and `rowpack16` are different memory access strategies.
- Keeping them as strategies makes benchmark output easier to interpret.

### 8.3 `plane_bytes=2`

Keep existing behavior:

```bash
--plane_bytes 2 --strategy byte
```

This runs the 2-byte-plane path.

Do not force users to pass `--strategy plane16` in this task.

Optional future cleanup:

```bash
--strategy plane16
```

could be added later.

## 9. Kernel Semantics

### 9.1 Common rules for all byte-plane strategies

All byte-plane strategies must satisfy:

- `K` is in `[1, 8]`
- only planes `0..K-1` are read
- no out-of-bounds loads
- final value is accumulated into thread-local sum first
- final thread-local sum is passed once to `block_reduce_store`
- `block_reduce_store` behavior is unchanged
- `d_out` contains one partial sum per block

### 9.2 `rowpack4` kernel semantics

For each plane `p < K`, each thread reads four consecutive rows from that plane.

Conceptual mapping:

```text
pack_index = logical pack position
base_row = pack_index * 4

uint32_t pack = load plane[p][base_row : base_row + 3]
```

Then unpack:

```text
byte0 -> row base_row + 0
byte1 -> row base_row + 1
byte2 -> row base_row + 2
byte3 -> row base_row + 3
```

For Exp1 dummy reduction, all four bytes are accumulated into the same thread-local sum.

Important:

- This benchmark does not need to preserve per-row output order.
- It only needs to read the correct bytes and reduce them.
- Therefore, processing four rows per thread is acceptable.

### 9.3 `rowpack16` kernel semantics

For each plane `p < K`, each thread reads sixteen consecutive rows from that plane.

Conceptual mapping:

```text
pack_index = logical pack position
base_row = pack_index * 16

uint4 pack = load plane[p][base_row : base_row + 15]
```

Then unpack 16 bytes and accumulate them.

Use a helper to sum bytes from each 32-bit lane:

```cpp
__device__ __forceinline__
unsigned int byte_sum_u32(uint32_t x)
{
  return (x & 0xffu) +
         ((x >> 8) & 0xffu) +
         ((x >> 16) & 0xffu) +
         ((x >> 24) & 0xffu);
}
```

Then:

```cpp
sum += byte_sum_u32(pack.x);
sum += byte_sum_u32(pack.y);
sum += byte_sum_u32(pack.z);
sum += byte_sum_u32(pack.w);
```

The exact helper may be optimized later, but correctness and clarity come first.

## 10. Alignment and Tail Handling

### 10.1 Separate plane allocation

Current allocation uses one `cudaMalloc` per plane.

This is generally safe for `uint32_t` and `uint4` alignment at the start of each plane.

### 10.2 Slab allocation warning

If a future implementation changes to one large slab:

```cpp
plane[p] = slab + p * n;
```

then each plane start must be aligned for the widest load.

For `rowpack16`, each plane start must be at least 16-byte aligned.

If `n` is not a multiple of 16, the implementation must add pitch padding:

```text
plane_pitch = round_up(n, 16)
plane[p] = slab + p * plane_pitch
```

This task should not introduce slab allocation unless explicitly scoped.

### 10.3 Tail handling for `rowpack4`

Define:

```cpp
uint64_t n4 = n / 4;
uint64_t tail_start = n4 * 4;
```

Main loop:

```text
process n4 full packs
```

Tail loop:

```text
process rows tail_start..n-1 using scalar byte loads
```

Tail must read only planes `0..K-1`.

Tail must not read out of bounds.

### 10.4 Tail handling for `rowpack16`

Define:

```cpp
uint64_t n16 = n / 16;
uint64_t tail_start = n16 * 16;
```

Main loop:

```text
process n16 full packs
```

Tail loop:

```text
process rows tail_start..n-1 using scalar byte loads
```

Do not use partial `uint4` loads for tail.

Keep tail simple and conservative.

## 11. Occupancy and Grid Sizing

Each strategy must query occupancy using the exact kernel that will be launched.

Required:

```text
byte baseline -> occupancy query byte baseline<K>
byte ilp4     -> occupancy query byte ilp4<K>
rowpack4      -> occupancy query rowpack4<K>
rowpack16     -> occupancy query rowpack16<K>
shared128     -> occupancy query shared128
plane16       -> occupancy query u16 kernel
```

For templated `K` strategies, compute `grid_by_k[1..8]`.

`d_out` allocation must use the maximum grid required by the selected strategy and selected `K` sweep.

Do not allocate or free `d_out` inside the timed loop.

Do not allocate or free `d_out` inside the `k` sweep loop.

## 12. CSV and Metadata

### 12.1 Keep existing CSV schema if possible

Avoid changing the CSV schema unless necessary.

Current schema:

```text
strategy,plane_bytes,k,n,logical_bytes,overfetch_factor,block,grid,warmup,iters,ms_per_iter,logical_GBps,device,sm,cc_major,cc_minor
```

This is sufficient if `strategy` clearly reports:

```text
byte
rowpack4
rowpack16
shared128
```

### 12.2 `byte_variant` visibility

Current CSV does not include `byte_variant`.

This is a known limitation.

Options:

1. Keep CSV unchanged and rely on run metadata / command line for `byte_variant`.
2. Add a new `variant` column.

Preferred for this task:

```text
Keep CSV unchanged unless needed.
Do not break existing plotting scripts without explicit reason.
```

If the developer chooses to add `variant`, they must update plotting scripts and document the schema change.

### 12.3 Logical bytes

For byte-plane paths:

```text
logical_bytes = n * k * plane_bytes
```

For `rowpack4` and `rowpack16`, logical bytes are still:

```text
n * k
```

because each row contributes one byte per plane.

### 12.4 Overfetch factor

For rowpack strategies:

```text
overfetch_factor = 1.0
```

assuming tails are handled conservatively and full packs do not read past `n`.

For `shared128`, keep:

```text
overfetch_factor = 4.0
```

or compute it more precisely in a future task.

Do not claim shared128 saves physical HBM bytes unless physical traffic is validated with Nsight Compute.

## 13. Validation

This task should add an optional non-timed validation mode if implementation time allows.

Recommended CLI:

```bash
--validate
```

Validation must not run inside the timed loop.

Validation flow:

1. Run selected kernel once.
2. Copy `d_out` from device to host.
3. Sum all block partials on host.
4. Compare against expected value.

### 13.1 Expected value for byte-plane paths

Current initialization:

```cpp
cudaMemset(plane, 0xAB, n * sizeof(uint8_t));
```

Expected byte-plane sum:

```text
expected = n * k * 0xAB
```

This applies to:

- scalar byte baseline
- scalar byte ILP4
- rowpack4
- rowpack16
- shared128 if no logical overcount occurs

### 13.2 Expected value for plane16

For `cudaMemset(..., 0xAB, n * sizeof(uint16_t))`, each `uint16_t` value is:

```text
0xABAB
```

Expected plane16 sum:

```text
expected = n * k * 0xABAB
```

where `k` is in `[1, 4]` for 2-byte planes.

### 13.3 Expected value for contiguous baseline

Do not modify `bench_contiguous_baseline.cu` in this task.

If validating contiguous in a future task:

```text
expected = n * 0xABABABABABABABAB
```

with appropriate unsigned 64-bit / wider host accumulation consideration.

## 14. Benchmark Plan

After implementation, run the following benchmark matrix on H200.

### 14.1 Core full sweep

Common parameters:

```bash
--device 0
--n 100000000
--plane_bytes 1
--k_min 1
--k_max 8
--block 256
--grid_mul 1
--warmup 10
--iters 1000
```

Run:

```bash
--strategy byte --byte_variant baseline
--strategy byte --byte_variant ilp4
--strategy rowpack4
--strategy rowpack16
--strategy shared128
```

Also run:

```bash
bench_contiguous_baseline
```

with:

```bash
--n 100000000
--block 256
--grid_mul 1
--warmup 10
--iters 1000
```

### 14.2 Key comparisons

Primary:

```text
contiguous64 vs rowpack16 k=8
contiguous64 vs rowpack4 k=8
byte ilp4 k=8 vs rowpack4 k=8
byte ilp4 k=8 vs rowpack16 k=8
```

Progressive scaling:

```text
rowpack4 k=1..8 runtime curve
rowpack16 k=1..8 runtime curve
```

Efficiency:

```text
logical_GBps(rowpack*) / logical_GBps(contiguous64)
```

### 14.3 Nsight Compute profiling

Profile at least:

```text
k = 1
k = 4
k = 8
```

For:

```text
byte ilp4
rowpack4
rowpack16
contiguous64
```

Metrics / sections to inspect:

- achieved occupancy
- register count
- local memory spills
- global load instruction type
- Long Scoreboard stalls
- Eligible Warps / Scheduler
- Issued Warps / Scheduler
- DRAM throughput
- L1/TEX sectors
- L2 sectors
- global load efficiency / transaction shape

Do not claim improvement unless both runtime and profiler evidence support it.

## 15. Implementation Steps

### Step 1: Create kernel header files

Create:

```text
exp1_scan_common.cuh
exp1_kernels_byte.cuh
exp1_kernels_rowpack.cuh
exp1_kernels_shared.cuh
exp1_kernels_plane16.cuh
```

Move existing kernel code with minimal changes.

Verify build still passes before adding new rowpack kernels.

Success criteria:

```text
Existing byte baseline and ilp4 produce same benchmark results within normal run variance.
```

### Step 2: Move scalar byte kernels

Move:

```text
scan_planes_u8_byte_unrolled<K>
scan_planes_u8_byte_ilp4<K>
byte kernel pointer helpers
byte launch helpers
```

to:

```text
exp1_kernels_byte.cuh
```

Do not change kernel semantics.

Success criteria:

```text
--strategy byte --byte_variant baseline works
--strategy byte --byte_variant ilp4 works
```

### Step 3: Move shared and plane16 kernels

Move:

```text
scan_planes_u8_shared128
scan_planes_u16
```

to their own files.

Do not optimize them in this step.

Success criteria:

```text
--strategy shared128 works
--plane_bytes 2 --strategy byte works
```

### Step 4: Implement `rowpack4`

Add:

```cpp
template <int K>
__global__ void scan_planes_u8_rowpack4(...);
```

Requirements:

- `K` compile-time specialized
- `#pragma unroll` on plane loop
- read `uint32_t` packs from each plane
- unpack four bytes in registers
- accumulate into thread-local sum
- conservative scalar tail
- `block_reduce_store` unchanged

Add:

```cpp
const void *scan_planes_u8_rowpack4_kernel_ptr(int k);
void launch_scan_planes_u8_rowpack4(int k, ...);
```

Success criteria:

```text
--strategy rowpack4 --k_min 1 --k_max 8 works
validation expected sum passes if --validate is implemented
```

### Step 5: Implement `rowpack16`

Add:

```cpp
template <int K>
__global__ void scan_planes_u8_rowpack16(...);
```

Requirements:

- `K` compile-time specialized
- `#pragma unroll` on plane loop
- read `uint4` packs from each plane
- each `uint4` represents 16 consecutive rows
- unpack/sum 16 bytes in registers
- conservative scalar tail
- `block_reduce_store` unchanged

Add:

```cpp
const void *scan_planes_u8_rowpack16_kernel_ptr(int k);
void launch_scan_planes_u8_rowpack16(int k, ...);
```

Success criteria:

```text
--strategy rowpack16 --k_min 1 --k_max 8 works
validation expected sum passes if --validate is implemented
```

### Step 6: Wire strategy dispatch

Update parsing:

```text
byte
rowpack4
rowpack16
shared128
packed32 compatibility behavior
```

Update warmup and timed launch paths.

Update occupancy query path.

Update `d_out` max grid allocation.

Do not allocate memory inside timed loop.

Success criteria:

```text
Every supported strategy launches correct kernel.
Every strategy uses occupancy query for actual kernel.
```

### Step 7: Optional validation

Add:

```bash
--validate
```

Implementation should be simple and non-timed.

If validation is too large for this task, explicitly defer it and state so in the implementation report.

### Step 8: Documentation update

Update:

```text
benchmarks/experiment1/README.md
```

Document:

- `byte`
- `rowpack4`
- `rowpack16`
- `shared128`
- `plane_bytes=2`
- compatibility status of `packed32`
- warning that rowpack packs rows, not precision bytes

Do not over-explain paper theory in README.

Link or mention:

```text
research/2026-04-15_exp1_byteplane_baseline_and_packed_reads_notes.md
```

if useful.

## 16. Acceptance Criteria

### 16.1 Semantic acceptance

- `k` still means number of byte-planes read.
- `rowpack4` does not read extra byte-planes.
- `rowpack16` does not read extra byte-planes.
- `shared128` does not read extra byte-planes, but overfetch behavior is documented.
- `plane_bytes=2` behavior is unchanged unless explicitly scoped.

### 16.2 Code acceptance

- Existing scalar byte baseline remains available.
- Existing scalar byte ILP4 remains available.
- `rowpack4` exists as a first-class benchmark path.
- `rowpack16` exists as a first-class benchmark path.
- Kernel code is split into focused files.
- No large abstraction framework is introduced.
- No benchmark timing logic is changed without explicit explanation.

### 16.3 Build acceptance

The following must build:

```bash
cmake -S benchmarks/experiment1 -B build/exp1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp1 -j
```

### 16.4 Smoke-test acceptance

At minimum, run small smoke tests:

```bash
./build/exp1/bench_byteplane_scan --device 0 --n 1024 --plane_bytes 1 --strategy byte --byte_variant baseline --k_min 1 --k_max 2 --warmup 0 --iters 1 --csv /tmp/exp1_byte_baseline.csv

./build/exp1/bench_byteplane_scan --device 0 --n 1024 --plane_bytes 1 --strategy byte --byte_variant ilp4 --k_min 1 --k_max 2 --warmup 0 --iters 1 --csv /tmp/exp1_byte_ilp4.csv

./build/exp1/bench_byteplane_scan --device 0 --n 1024 --plane_bytes 1 --strategy rowpack4 --k_min 1 --k_max 2 --warmup 0 --iters 1 --csv /tmp/exp1_rowpack4.csv

./build/exp1/bench_byteplane_scan --device 0 --n 1024 --plane_bytes 1 --strategy rowpack16 --k_min 1 --k_max 2 --warmup 0 --iters 1 --csv /tmp/exp1_rowpack16.csv

./build/exp1/bench_byteplane_scan --device 0 --n 1024 --plane_bytes 1 --strategy shared128 --k_min 1 --k_max 2 --warmup 0 --iters 1 --csv /tmp/exp1_shared128.csv

./build/exp1/bench_byteplane_scan --device 0 --n 1024 --plane_bytes 2 --strategy byte --k_min 1 --k_max 2 --warmup 0 --iters 1 --csv /tmp/exp1_plane16.csv
```

### 16.5 Benchmark acceptance

On H200, collect full runs for:

```text
byte baseline
byte ilp4
rowpack4
rowpack16
shared128
contiguous64
```

At minimum, report:

- `ms_per_iter`
- `logical_GBps`
- `logical_GBps / contiguous64_GBps`
- `k=8` comparison against contiguous64

### 16.6 Profiler acceptance

For `rowpack4` and `rowpack16`, Nsight Compute must confirm that the compiler generated wider global load instructions than scalar byte load.

If the compiler does not generate expected wide loads, document that fact and inspect source/SASS before claiming results.

## 17. Expected Risks

### 17.1 Register pressure

`rowpack16` may increase register usage due to unpacking and temporary values.

This may reduce occupancy.

The implementation must not assume `rowpack16` is faster.

### 17.2 Integer ALU overhead

Unpacking bytes adds integer operations.

This may reduce speedup if the kernel becomes instruction-heavy.

### 17.3 Alignment issues

`rowpack16` requires correct alignment.

Separate `cudaMalloc` plane allocation should be safe, but future slab allocation must use padded pitch.

### 17.4 CSV ambiguity

If CSV only has `strategy` and not `variant`, byte baseline vs ILP4 must be tracked through run metadata.

This is acceptable for now but should be documented.

### 17.5 shared128 interpretation

`shared128` may look fast under logical GB/s while physically reading more bytes.

Do not use it as the main evidence for bandwidth saving unless physical traffic is reported.

## 18. Developer Report Format

When implementation is complete, report in this structure:

1. Summary
2. Files Changed
3. New Strategies Added
4. Compatibility Behavior for `packed32`
5. Kernel Semantics
6. Tail Handling
7. Occupancy / Dispatch Changes
8. Validation Status
9. Build / Smoke Test Results
10. Benchmark Results if Run
11. Nsight Compute Results if Run
12. Caveats / Deferred Work

Avoid claims like:

- "rowpack16 is definitely faster"
- "this reaches memory bandwidth"
- "shared128 proves the idea works"

unless benchmark and profiler data support the claim.

## 19. Recommended Next Decision

Before implementation, decide one compatibility detail:

```text
Should --strategy packed32 become an alias for rowpack4,
or should packed32 remain as the old runtime-k implementation while rowpack4 is added separately?
```

Recommended:

```text
Keep packed32 as backward-compatible alias to rowpack4,
and write strategy=rowpack4 in new result files.
```

Reason:

- reduces duplicate strategy meanings
- makes future plots clearer
- keeps old commands mostly working

If preserving exact old `packed32` behavior is important for historical comparison, then keep it separate temporarily and document it as legacy.
