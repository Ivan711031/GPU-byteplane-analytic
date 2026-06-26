# Experiment 3: Progressive Aggregation Kernel

Date: 2026-04-14  
Status: Draft for discussion  
Owner: Nick / Codex

## 1. Goal

Experiment 3 aims to measure the core performance tradeoff of Buff-style progressive aggregation:

- Throughput should decrease as each additional subcolumn is read.
- Error should decrease as more fractional subcolumns are included.
- The final output should support a precision-throughput curve that combines Exp3 throughput with Experiment 2 error results.

Primary question:

> For progressive aggregation, how many billion rows/sec can the GPU process at each refinement depth `k`, and how does that pair with the error bound from Experiment 2?

## 2. Current Readiness

We can start with the kernel performance part now.

Ready:

- CUDA benchmark structure exists in Exp0/Exp1.
- Run directory and metadata conventions exist.
- H100/H200 tracking convention exists.
- Exp1 already measures byte-plane/subcolumn-style scan throughput.
- Contiguous baseline has been added as a separate Exp1 baseline binary.

Not ready / needs definition:

- Experiment 2 error CSV/schema is not visible in the repo yet.
- Real encoded dataset format is not defined in this repo yet.
- Segment metadata format is not defined yet.
- Exact formula for `error_bound` must be agreed before final curve generation.

Therefore, Exp3 should start as a standalone performance microbenchmark with synthetic internally initialized subcolumn data, then later connect to Experiment 2 outputs.

## 3. Proposed Scope

### Phase 1: Synthetic progressive aggregation benchmark

Implement a standalone benchmark:

`benchmarks/experiment3/bench_progressive_aggregation.cu`

This benchmark should:

- Allocate synthetic subcolumn arrays on device.
- Run a progressive aggregation kernel for `k = 0..K_MAX`.
- Measure rows/sec and GB/s using CUDA event timing.
- Output one CSV row per `k`.
- Keep dataset/error integration out of phase 1.

### Phase 2: Connect to Experiment 2

After Experiment 2 schema is available, add a plotting or join step that combines:

- Exp3 throughput by `dataset, k`.
- Exp2 error or error bound by `dataset, k`.

Output:

- precision-throughput curve.

## 4. Kernel Model

### Logical data model

Each value is represented by:

- integer subcolumn
- zero or more fractional subcolumns
- segment-level FOR base
- segment-level scale

Draft interpretation:

```text
value_approx(k) = (FOR_base + integer_part + fractional_refinement(k)) * scale
```

The exact fixed-point reconstruction needs confirmation.

### Kernel work assignment

Initial design:

- Each thread block processes one segment or one segment tile.
- Threads use grid-stride or block-local striding within the segment.
- Each thread accumulates partial integer/fractional sums in registers.
- The block reduces partial sums into a per-block result.
- Segment-level `FOR_base` and `scale` are applied once per block or once per segment output, not per row.

The v1 benchmark may use one uniform segment size and one block per segment tile if a segment is larger than one block can cover efficiently.

### Rounds

Round definition:

- `k = 0`: read integer subcolumn only.
- `k = 1`: read integer subcolumn + fractional subcolumn 1.
- `k = 2`: read integer subcolumn + fractional subcolumns 1..2.
- ...
- `k = K_MAX`: read integer subcolumn + fractional subcolumns 1..K_MAX.

This needs confirmation if "Round 0" should count as `k=1` in plots.

## 5. Accumulation and Overflow

Known bound from discussion:

- For `uint8` subcolumn values, max per row is 255.
- For `10^9` rows, partial sum is about `2.55e11`, which fits in `uint64_t`.

Initial v1 rule:

- Use `uint64_t` accumulation for `uint8` subcolumns.
- Record accumulator type in CSV/metadata.
- Do not implement multi-word accumulation in v1.

Overflow gate for later:

```text
max_subcolumn_value * rows_per_output_sum * number_of_terms
```

If this may exceed `uint64_t`, v2 must add either:

- wider staged accumulation
- multi-word accumulation
- segmented host-side accumulation
- a narrower per-output aggregation contract

Open issue:

- Need to decide whether integer subcolumn can be 8-bit, 16-bit, 32-bit, or a template/runtime option.

## 6. Draft CLI

Suggested options:

```bash
./build/exp3/bench_progressive_aggregation \
  --device 0 \
  --n 100000000 \
  --segment_rows 1048576 \
  --int_bits 32 \
  --frac_bits 8 \
  --k_min 0 \
  --k_max 8 \
  --block 256 \
  --grid_mul 1 \
  --warmup 10 \
  --iters 200 \
  --csv results/exp3/progressive_aggregation.csv
```

Notes:

- `--n` should be at least `100000000` for performance testing.
- No small smoke test should be used as the main validation result.
- `--segment_rows` may be changed after deciding the segment model.

## 7. Draft CSV Schema

Suggested output columns:

```text
benchmark,dataset,mode,k,n,segment_rows,int_bits,frac_bits,logical_subcolumns_read,
logical_bytes,block,grid,warmup,iters,ms_per_iter,rows_per_sec,billion_rows_per_sec,
logical_GBps,accumulator_bits,device,sm,cc_major,cc_minor
```

Suggested fixed values for phase 1:

- `benchmark=progressive_aggregation`
- `dataset=synthetic`
- `mode=synthetic_subcolumns`
- `accumulator_bits=64`

Potential later columns after Experiment 2 integration:

```text
approximate_result,error_bound,relative_error_bound
```

Open question:

- Should Exp3 raw benchmark CSV include `error_bound`, or should error stay in a separate joined analysis CSV from Exp2?

## 8. Draft Output Files

Proposed run structure:

```text
results/exp3/run_<timestamp>_job<id>_<gpu_tag>/
  exp3_progressive_aggregation.csv
  setup_estimate.txt
  run_meta.txt
  repro_command.txt
  ncu_command_template.txt
```

This follows the Exp0/Exp1 runner convention.

## 9. Implementation Plan

Step 1:

- Add `benchmarks/experiment3/CMakeLists.txt`.
- Add `benchmarks/experiment3/bench_progressive_aggregation.cu`.
- Add a minimal `scripts/legacy/root_runners/run_exp3.sh`.
- Add root Slurm wrapper `scripts/legacy/root_runners/run_exp3.sh`.

Step 2:

- Implement synthetic allocation and initialization.
- Use `uint8_t` fractional subcolumns.
- Choose one integer subcolumn type after discussion.
- Implement `k=0..K_MAX` sweep.

Step 3:

- Run on `dev` partition with `--n 100000000` or larger.
- After `sbatch`, immediately check `squeue -j <job_id>`.
- Exclude known bad node `25a-hgpn080` unless the cluster issue is resolved.

Step 4:

- Add plotting only after Exp2 error schema is available.

## 10. Open Questions for Nick

1. What is the exact fixed-point formula?
   - Is it `FOR_base + integer_part + fractional_refinement`, or does each subcolumn have a positional weight?

2. What should `k` mean in plots?
   - Option A: `k=0` means integer-only.
   - Option B: `k=1` means integer-only, matching "one subcolumn read".

3. What integer subcolumn width should v1 support?
   - `uint8`, `uint16`, `uint32`, or `uint64`?

4. What should a segment mean in v1?
   - Fixed `segment_rows` only?
   - One block per segment?
   - Multiple blocks per large segment tile?

5. Should Exp3 compute and output `approximate_result` in phase 1?
   - If yes, should host reduce per-block outputs into one final result?
   - If no, phase 1 remains throughput-only and result correctness is handled later.

6. Should `error_bound` be generated inside Exp3 or joined from Experiment 2?
   - Recommendation: join from Experiment 2, unless the error formula is already fixed.

7. Which datasets should phase 2 support?
   - Synthetic only first?
   - Or named datasets from Experiment 2 once available?

## 11. Non-Goals for v1

- No real dataset reader.
- No multi-word accumulation.
- No automatic precision-throughput plotting until Experiment 2 schema is ready.
- No changes to Exp1 byte-plane benchmark behavior.
- No packed/shared strategy variants in Exp3 v1.
