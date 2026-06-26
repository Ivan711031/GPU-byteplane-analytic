# Spec: Exp4-v1 Progressive Filter COUNT

## Objective

Exp4-v1 驗證 progressive byte-plane predicate evaluation 的核心價值：對 `WHERE x > threshold` 查詢，從 MSB byte-plane 開始逐層讀取，利用 early decision 減少記憶體讀取量，並量化 throughput 與 selectivity 的關係。

**Query:**
```sql
SELECT COUNT(*) FROM T WHERE x > threshold;
```

**Why:**
- Exp3 已證明 aggregation 的 throughput 隨讀取 planes 數量線性變化。
- Exp4 要回答：filter predicate 是否也能因 early decision 而節省頻寬？在哪些 selectivity / distribution 下有效？
- 這是論文 Figure「progressive filter throughput vs. selectivity」的基礎實驗。

**Success Criteria:**
1. v1a smoke：對單一 threshold，GPU `COUNT(*)` 結果與 CPU encoded reference 完全一致（`gpu_count == cpu_encoded_count`）。
2. v1b sweep：對 9 個 target selectivity points，產出 throughput vs. selectivity curve，並標示 average planes read per row。
3. 所有 dataset（Sensor, Uniform, Heavy-tailed, Zipfian）皆可執行，max_plane_count 不硬寫 8。
4. Validation 分兩層報告：kernel correctness（gpu vs encoded）與 encoding fidelity（encoded vs raw）。

---

## Tech Stack

- **Language:** CUDA C++ (device code), C++17 (host code)
- **Build:** CMake ≥ 3.18
- **Target:** H100/H200 (CUDA_ARCH=90)
- **Data format:** Exp3 encoded-dev artifact v1 (plane-major zero-padded)
- **Reuse:** Exp3 的 rowpack16 load pattern、block reduce、real-data layout loader

---

## Commands

### Build
```bash
cmake -S benchmarks/experiment4 \
  -B build/exp4 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90

cmake --build build/exp4 -j
```

### Smoke Test (v1a)
```bash
./build/exp4/bench_progressive_filter \
  --device 0 \
  --encoded-root /work/u4063895/datasets/synthetic/dev_buff_exp3/uniform \
  --threshold 500.0 \
  --load_strategy rowpack16 \
  --block 256 \
  --warmup 10 \
  --iters 200 \
  --validate \
  --csv /tmp/exp4_smoke.csv
```

### Official Sweep (v1b)
```bash
SELECTIVITIES="1 5 10 25 50 75 90 95 99" \
./scripts/run_exp4.sh \
  --encoded-root /work/u4063895/datasets/synthetic/dev_buff_exp3/uniform \
  --output-dir results/exp4/uniform
```

### Validation-Only (no timing)
```bash
./build/exp4/bench_progressive_filter \
  --encoded-root ... \
  --threshold ... \
  --validate-only \
  --csv /tmp/exp4_validate.csv
```

---

## Project Structure

```
gpu-byteplane-scan-experiments/
├── benchmarks/
│   ├── experiment3/              # Exp3 reference (reuse files)
│   │   ├── exp3_common.cuh
│   │   ├── exp3_real_data_layout.hpp
│   │   └── exp3_real_data_layout.cpp
│   └── experiment4/              # Exp4 v1
│       ├── bench_progressive_filter.cu    # main benchmark harness
│       ├── exp4_kernels_filter.cuh        # filter kernels
│       ├── exp4_common.cuh                # Exp4-specific constants (if any)
│       └── CMakeLists.txt                 # build target
├── scripts/
│   └── run_exp4.sh             # official sweep runner
├── results/
│   └── exp4/                   # run artifacts
│       └── run_<timestamp>_.../
│           ├── output.csv
│           ├── run_meta.txt
│           ├── repro_command.txt
│           └── ncu_command_template.txt
└── spec/
    └── 2026-04-29_Exp4_v1_Progressive_Filter_COUNT_Spec.md   # this file
```

---

## Code Style

Follow existing Exp3 conventions:

```cpp
// Kernel: template-specialized depth, rowpack16 load
__global__ void progressive_filter_rowpack16(
    Exp3RuntimeU8Subcolumns subcolumns,
    uint64_t n,
    const uint8_t* d_threshold_bytes,   // per-segment threshold bytes
    const uint32_t* d_active_plane_count, // per-segment round limit
    uint64_t* d_block_counts)           // per-block qualified count
{
  // One block = one tile, tile does NOT cross segment boundary
  uint64_t segment_id = blockIdx.x / tiles_per_segment;
  uint32_t max_rounds = d_active_plane_count[segment_id];
  
  uint16_t active_mask = 0xFFFF;     // still ambiguous
  uint16_t qualified_mask = 0x0000;  // decided x > T
  
  for (uint32_t round = 0; round < max_rounds && active_mask != 0; ++round) {
    uint4 pack = load_plane_128bit(subcolumns.ptrs[round], pack_index);
    uint8_t thresh_byte = d_threshold_bytes[segment_id * max_plane_count + round];
    // ... compare 16 rows ...
  }
  // strict > : remaining active rows are disqualified
  uint32_t local_count = __popc(qualified_mask);
  uint32_t block_count = exp3_block_reduce_sum_ull(local_count, reduce_smem);
  if (threadIdx.x == 0) {
    d_block_counts[blockIdx.x] = block_count;
  }
}
```

**Naming:**
- Kernel names: `progressive_filter_<strategy>`
- Device functions: `exp4_` prefix if new, reuse `exp3_` if identical
- Constants: `EXP4_` prefix
- Files: `snake_case.cu/.cuh`

---

## Testing Strategy

### Validation Layers

**Layer 1 — Kernel Correctness (hard gate):**
```
gpu_count == cpu_encoded_count
```
- CPU 用 encoded artifact 逐 segment、逐 row 做 exact predicate evaluation。
- GPU 結果必須完全匹配。
- 若失敗，kernel 有 bug，禁止進入 sweep。

**Layer 2 — Encoding Fidelity (report only):**
```
cpu_encoded_count == cpu_raw_count
```
- CPU 同時對 raw FP64 資料做 `count(raw_x > threshold_fp64)`。
- 若 encoded != raw，表示 threshold 附近因 quantization 產生 predicate drift。
- Spec 要求記錄，但不阻塞 kernel correctness。

### Test Matrix

| Stage | Dataset | Threshold | Purpose |
|-------|---------|-----------|---------|
| v1a smoke | Uniform | 500.0 | Basic correctness |
| v1a smoke | Uniform | 0.0 | All qualified edge case |
| v1a smoke | Uniform | 1000.0 | All disqualified edge case |
| v1a boundary | any | quantile-based | Boundary row alignment (non-4096-multiple segment) |
| v1b sweep | all 4 datasets | 9 selectivities | Throughput curve |

---

## Boundaries

### Always
- Reuse `exp3_real_data_layout` for artifact loading.
- Use `rowpack16` (128-bit coalesced loads) for all plane reads.
- Tile must not cross segment boundary.
- Kernel round limit = per-segment `active_plane_count`.
- Validation must report both `gpu_matches_cpu_encoded` and `encoded_matches_raw`.

### Ask First
- Adding new kernel strategies (compaction, deferred gather).
- Changing threshold encoding policy (current: reuse `encode_value_to_code`).
- Modifying Exp3 shared files (`exp3_common.cuh`, `exp3_real_data_layout`).
- Supporting predicates other than `x > T` (`>=`, `<`, `<=`, `BETWEEN`).

### Never
- Hard-code max_plane_count = 8.
- Output full bitmask as main benchmark result (count only for v1).
- Skip validation for any official sweep run.
- Read beyond `active_plane_count` (zero padding must not be counted as logical bytes).
- Mix aggregate approximation error into filter validation (filter v1 is exact predicate).

---

## Blocking Assertions

以下前提必須在實作 kernel 前確認。若任何一條無法確認，禁止進入 implementation phase：

1. **plane_000 is the most-significant byte plane.**
   - Verified by `plane_lsb_start(total_bits, 0)` returning highest start bit in `buff_codec.cpp`.

2. **Plane order is big-endian by significance.**
   - Round 0 reads plane_000 (highest byte), round 1 reads plane_001, etc.

3. **Threshold encoding must reuse the same `encode_value_to_code()` path as data export.**
   - `export_encoded_dev_layout.cpp` uses `buff::encode_segment()` → `encode_value_to_code()`.
   - Threshold must follow identical path with segment's `fractional_bits`.

4. **Threshold exactness must be asserted or reported.**
   - For each segment, compute `threshold_code = encode_value_to_code(threshold_fp64, -fractional_bits)`.
   - Assert that `threshold_code` is exact integer (no rounding/truncation needed).
   - If inexact, mark threshold as unsupported for raw predicate validation.

5. **Kernel round limit = per-segment `active_plane_count`, not `max_plane_count`.**
   - `segment_meta.csv` has `active_plane_count` per segment.
   - Plane files are zero-padded to `max_plane_count`; reading padding invalidates measurements.

6. **Validation must split `gpu_count == cpu_encoded_count` and `cpu_encoded_count == cpu_raw_count`.**
   - Separate booleans in CSV: `gpu_matches_cpu_encoded`, `encoded_matches_raw`.
   - Final `validated` is true only if both are true.

7. **Fast-path metadata must be reported in v1b sweep.**
   - Count of all-qualified segments, all-disqualified segments, mixed segments.
   - Count of fast-path rows vs. GPU-processed rows.

8. **max_filter_planes must come from artifact metadata.**
   - Read from `manifest.json` or `segment_meta.csv`.
   - Heavy-tailed has 16 planes; kernel must support up to runtime limit (32).

---

## Implementation Plan

### Phase 1: v1a Smoke (No Fast-Path)

**Goal:** Verify kernel state machine correctness.

1. **Harness setup** (`bench_progressive_filter.cu`):
   - Parse `--encoded-root`, `--threshold`, `--validate`.
   - Load dataset via `exp3_real::load_dataset()`.
   - Compute per-segment threshold bytes on CPU.

2. **Kernel** (`exp4_kernels_filter.cuh`):
   - `progressive_filter_rowpack16_passive()`.
   - One block = one tile (tile rows = blockDim.x * 16).
   - Tile does not cross segment boundary.
   - Dual-mask state machine: `active_mask`, `qualified_mask`.
   - Early exit when `active_mask == 0`.

3. **Validation**:
   - CPU encoded reference: iterate all segments, all rows, decode approximate value, compare with threshold.
   - CPU raw reference: read `.f64le.bin`, count `raw_x > threshold_fp64`.
   - Assert `gpu_count == cpu_encoded_count`.
   - Report `cpu_encoded_count == cpu_raw_count`.

### Phase 2: v1b Official Sweep (With Fast-Path)

**Goal:** Produce throughput vs. selectivity curve.

1. **Selectivity sweep script** (`scripts/run_exp4.sh`):
   - For each target selectivity (1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%):
     a. Compute `threshold_fp64 = quantile(raw_data, 1 - selectivity)`.
     b. Classify segments: all-qualified / all-disqualified / mixed.
     c. For mixed segments, compute threshold bytes and launch GPU kernel.
     d. Sum fast-path counts + GPU counts.
     e. Record all metrics to CSV.

2. **Metrics**:
   - `avg_planes_read_per_row`: sum(actual rounds) / total rows.
   - `logical_bytes_read`: n * avg_planes_read_per_row.
   - `fast_path_all_qualified_segments`, `fast_path_all_disqualified_segments`, `mixed_segments`.

### Phase 3: v2 (Future, Not in v1)

- `SELECT AVG(x) FROM T WHERE x > threshold` (filter + aggregate precision-throughput curve).
- Deferred gather / stream compaction ablation.

---

## CSV Schema

```csv
benchmark,dataset,query,predicate,threshold_fp64,threshold_source,target_selectivity,observed_selectivity_raw,observed_selectivity_encoded,value_count,segment_size,segment_count,max_plane_count,active_plane_count_min,active_plane_count_max,max_filter_planes,kernel_path,load_strategy,output_mode,avg_planes_read_per_total_row,avg_planes_read_per_gpu_processed_row,pack_plane_loads,logical_bytes_read,estimated_pack_load_bytes,ms_per_iter,rows_per_sec,billion_rows_per_sec,logical_GBps,qualified_count_gpu,qualified_count_cpu_encoded,qualified_count_cpu_raw,gpu_matches_cpu_encoded,encoded_matches_raw,validated,fast_path_all_qualified_segments,fast_path_all_disqualified_segments,mixed_segments,fast_path_rows,gpu_processed_rows,device,sm,cc_major,cc_minor
```

**Field Descriptions:**
- `avg_planes_read_per_total_row`: actual rounds executed, averaged over all rows including fast-path rows (fast-path rows count as 0 rounds).
- `avg_planes_read_per_gpu_processed_row`: actual rounds executed, averaged over only GPU-processed rows (mixed segments). Separates segment-level pruning from byte-plane-level early exit.
- `logical_bytes_read`: `value_count * avg_planes_read_per_total_row`.
- `estimated_pack_load_bytes`: physical 128-bit loads * 16 bytes (for comparison with logical bytes).
- `fast_path_*`: segment counts and row counts handled by CPU fast-path.
- `gpu_matches_cpu_encoded`: boolean, Layer 1 validation.
- `encoded_matches_raw`: boolean, Layer 2 validation.
- `validated`: true only if both layers pass.

---

## Open Questions

1. **Threshold quantile computation:** Should we compute quantiles from raw `.f64le.bin` or from decoded artifact? Raw is ground truth, but requires reading large file. Decoded is faster but may have quantization noise near threshold.
   - **Proposed:** Use raw file for v1b sweep. Compute once and cache threshold values.

2. **Segment min/max for fast-path (v1b only, does not block v1a):**
   `segment_meta.csv` currently has `segment_base` but not `segment_min` or `segment_max`.
   Fast-path requires both:
   - `T < segment_min` → all qualified
   - `T >= segment_max` → all disqualified
   **Resolution:** Before v1b official sweep, modify `export_encoded_dev_layout.cpp` to add `segment_min` and `segment_max` columns to `segment_meta.csv`. v1a smoke does not use fast-path and is not blocked by this change.

3. **Heavy-tailed 16 planes:** Is this a bug in Exp2 encoding or legitimate artifact? If bug, should Exp4 still support it?
   - **Decision:** Exp4 supports whatever artifact exports. Document `max_plane_count` in CSV. Do not hard-code 8.

---

## References

- `benchmarks/experiment3/exp3_kernels_progressive.cuh`: rowpack16 load pattern reference.
- `benchmarks/experiment3/exp3_real_data_layout.hpp/.cpp`: artifact loader and `plane_basis_for_segment()`.
- `buff_encoder/buff_codec.cpp`: `encode_value_to_code()`, `plane_lsb_start()`, encoding semantics.
- `buff_encoder/buff_error_study.cpp`: COUNT(x > t) validation reference.
- `spec/M03-initial-guide.md`: Original Exp4 design notes.

---

*Spec version: 2026-04-29 v1*
*Status: Ready for implementation*
*Next step: Implement v1a smoke kernel + harness*
