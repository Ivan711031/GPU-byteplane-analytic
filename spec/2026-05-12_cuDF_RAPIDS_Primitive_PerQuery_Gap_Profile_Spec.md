# cuDF/RAPIDS Primitive Per-Query 2× Gap Profile Spec

Date: 2026-05-12
Scope: One binary question. No matrix expansion. No new feature.
Status: Narrow profiling spec. Decides whether to keep chasing the warm-resident cuDF win or declare an honest negative result.

## References

- [Warm-Resident Per-Query Spec (gate failed honestly)](./2026-05-12_cuDF_RAPIDS_Primitive_WarmResident_PerQuery_Spec.md)
- [Verdict Gate Spec (superseded)](./2026-05-12_cuDF_RAPIDS_Primitive_CompactU_Verdict_Gate_Spec.md)
- [CompactU Implementation Spec](../research/2026-05-12_cuDF_RAPIDS_Primitive_CompactU_Implementation_Spec.md)
- [Branch Report](../research/2026-05-12_cuDF_RAPIDS_Primitive_Branch_Report.md)

## 1. The Single Question

> On sensor exact `dev_buff_exp3` at `(sel = 99, k = 1)`, byteplane per-query is 2.8–3.3 ms versus cuDF 1.4–1.5 ms. Is this gap dominated by obvious orchestration overhead (launch latency, redundant syncs, per-query malloc/memset, missing kernel fusion, unnecessary D2H), or is each kernel already at or near its bandwidth/SOL floor with little room left?

The answer must be one of:

- **A. Closeable.** A specific orchestration or launch-shape problem accounts for ≥ 0.7 ms of the gap and has a clear fix that can be implemented without changing kernel algorithms.
- **B. At floor.** Every per-query kernel is bandwidth-bound at ≥ 70% SOL of HBM3 on H200, launch + sync overhead is ≤ 0.1 ms total, and there is no obvious fusion/hoist candidate. The 2× gap is structural for this kernel decomposition.

This spec does **not** ask the agent to implement any fix. The deliverable is the answer, the evidence behind it, and a one-line decision recommendation.

## 2. Why So Narrow

Setup is structurally unwinnable for compact-U exact refinement: byteplane setup is approximately `cuDF setup + artifact load + plane stage`, because refinement requires raw FP64 on device. Setup-side optimization on this gate cannot beat cuDF. Therefore this spec restricts the question to the per-query 2× gap only.

If the per-query gap is also at floor, the cuDF route as currently framed (warm-resident exact COUNT vs cuDF reduction) cannot win on this artifact. That is an answer, not a failure.

## 3. Single Test Configuration

| Axis | Value | Why |
|---|---|---|
| dataset | `sensor` | matches every prior gate; widely characterized |
| artifact | exact `dev_buff_exp3` | safety baseline |
| selectivity | 99 | exercises real refinement: `U = 50,357` |
| k | 1 | smallest depth that still produces non-zero U |
| mode | `--compact-u-refine-raw` | the path under measurement |
| block-threads | 256 | matches prior gates |

One row. Five warm repetitions. No other axis.

## 4. Required Profiling Commands

Run on the same H200 node as job 47890. Use `sbatch --wait` so output returns in-session.

### 4.1 nsys timeline (orchestration view)

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=results/exp4/perquery_gap_nsys_<jobid> \
  --force-overwrite=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  <byteplane_count_gt_batch_cli> \
    --artifact-root <path> \
    --raw-root <path> \
    --segment-minmax <path> \
    --thresholds <T_for_sel99> \
    --ks 1 \
    --compact-u-refine-raw \
    --repeat 5
```

If the batch CLI does not yet bracket the per-query loop with `cudaProfilerStart/Stop`, the agent should add those calls around the per-query repeat loop before running (this is a small instrumentation change, not an algorithm change; document it in the result packet).

Extract from the nsys report:

- per-kernel duration (mean, min, max, p50) for the 5 repetitions, separated by kernel name
- gaps between kernel completions and the next launch within one query iteration (launch latency proxy)
- explicit synchronization points (cudaEventSynchronize, cudaDeviceSynchronize, cudaStreamSynchronize) inside the per-query loop, with their durations
- any cudaMalloc/cudaFree/cudaMemset/cudaMemcpy calls inside the per-query loop (these are the most likely hoist candidates)

### 4.2 ncu kernel SOL (kernel-floor view)

Run ncu only on the per-query kernels, not on setup. Use one repetition for kernel-level metrics:

```bash
ncu \
  --set full \
  --target-processes all \
  --kernel-name regex:"byteplane_count_gt_kernel|compact_u_indices_kernel|compact_u_refine_raw_kernel" \
  --launch-skip 0 --launch-count 3 \
  --output results/exp4/perquery_gap_ncu_<jobid> \
  <same binary, same args, but --repeat 1>
```

From each kernel's ncu report, extract:

- DRAM throughput (GB/s) and % of peak HBM3 on H200 (theoretical ~3.35 TB/s)
- Compute throughput (% of SOL)
- Achieved occupancy
- Memory pipe bottleneck flag (if ncu marks one)
- Bytes read / written
- Duration

For context, cuDF's reduction kernel on the same column should also be profiled in the same job, using the cuDF baseline runner with `nsys` and `ncu` wrappers. Compare cuDF's reduction kernel SOL to byteplane's three kernels.

### 4.3 Code-side audit (no profiler)

Read `benchmarks/experiment4/byteplane_count_gt.cu` between the start and end of the per-query loop in `byteplane_count_gt_batch` (around lines 1468–1525 plus the inner `gpu_count_gt_resident` call). List every:

- `cudaMalloc` / `cudaMallocAsync`
- `cudaFree` / `cudaFreeAsync`
- `cudaMemset` / `cudaMemsetAsync`
- `cudaMemcpy*` (H2D and D2H)
- `cudaEventRecord` / `cudaEventSynchronize`
- `cudaStreamSynchronize` / `cudaDeviceSynchronize`

For each, mark whether it could be hoisted out of the loop (reused buffer, reused stream, fewer syncs) without changing correctness. Per-query result `refined_hits` D2H is acceptable; per-query scratch allocations are not.

## 5. Breakeven Formula Audit (Side Task)

In job 47890 the verdict packet reported `breakeven_queries = 521K` on every row. The math:

```
(byteplane_setup - cudf_setup) / (cudf_per_query - byteplane_per_query)
= (1065 - 544) / (1.48 - 3.33)
= 521 / (-1.85)
= -281.6
```

A negative denominator means byteplane per-query is slower than cuDF per-query, so **no amount of additional queries amortizes the setup gap**. The honest reported value is `inf` or `unattainable`. The prior spec already said:

> "If the denominator is `<= 0`, emit `breakeven_queries = inf` to make the row trivially identifiable."

The `521K` figure suggests the runner clamped the denominator to a positive `epsilon` (probably `1e-3`) instead of branching on its sign. That gives `521 / 1e-3 = 521,000`, the exact value reported.

Required:

1. Locate the breakeven computation in `scripts/cudf_exact_count_gt.py` (or wherever the merged CSV is produced).
2. Read the literal source lines that produce `breakeven_queries`.
3. Confirm whether the implementation uses an `epsilon` clamp on the denominator or a sign check.
4. If clamp: change to a sign check that emits `float('inf')` (and `inf` literal in the CSV) when `cudf_per_query_median - byteplane_per_query_median <= 0`.
5. Re-emit the merged CSV from the existing 47890 outputs (no re-run of the H200 job required for this single column fix).

Report the source line numbers in the result packet so the change is reviewable from Mac without sync.

## 6. Deliverable Packet

Return one combined packet:

### A. Profiling raw outputs

- nsys report path and a 20-line summary of the per-query timeline (kernel order, kernel duration, inter-kernel gap, sync calls)
- ncu reports paths and a per-kernel table:
  | kernel | duration (μs) | DRAM GB/s | DRAM % peak | compute % SOL | occupancy | bytes_read | bytes_written |
- cuDF reduction kernel row in the same table for direct comparison

### B. Code-side audit

The list from §4.3 with each item annotated as `HOIST_CANDIDATE` or `NEEDED_PER_QUERY`.

### C. Verdict

One paragraph picking exactly one of:

- **Closeable.** Name the top 1–2 specific orchestration fixes that, if implemented, plausibly cut at least 0.7 ms from per-query. Estimate the cut (rough is fine).
- **At floor.** Show the kernel SOL numbers that justify "near floor" and confirm no obvious hoist candidate remains.

### D. Decision recommendation

One line, exactly one of:

- `recommend: implement orchestration fix, then re-run warm-resident gate on sensor sel=99 k=1`
- `recommend: declare honest negative result for warm-resident cuDF gate; pivot research framing to transfer-bound / cold-path / multi-predicate`

### E. Breakeven fix

The result of §5: source line numbers, the change, and the re-emitted merged CSV's new `breakeven_queries` value for the four 47890 rows (should be `inf` for all four if the audit confirms the suspicion).

## 7. Definition of Done

This spec is done when the packet contains:

1. nsys per-query timeline summary with per-kernel durations and inter-kernel gaps.
2. ncu kernel SOL table for `byteplane_count_gt_kernel`, `compact_u_indices_kernel`, `compact_u_refine_raw_kernel`, and cuDF's reduction kernel.
3. The §4.3 code-side audit list with hoist-candidate annotations.
4. A single verdict paragraph: Closeable or At floor.
5. A single decision recommendation line.
6. The §5 breakeven audit result and the re-emitted CSV.
7. Slurm job COMPLETED with ExitCode=0:0.

## 8. Non-Goals

- No matrix expansion. One row only.
- No `uniform` / `heavy_tailed` / `zipfian`. Sensor only.
- No bounded `p3/p6`.
- No new kernels. No kernel rewrites.
- No Sirius / DuckDB / RAPIDS upstream work.
- No cold-path or transfer-bound measurement. That belongs to a different spec.
- No implementation of any orchestration fix in this spec. The next spec (if any) handles implementation.

## 9. Decision Tree After This Spec

```
If verdict = Closeable:
    next spec = "compact-U warm-resident orchestration fix"
        scope: implement the named top 1–2 fixes
        gate: re-run §3 of the Warm-Resident Per-Query Spec, same matrix
              (sensor, sel ∈ {90, 99}, k ∈ {1, 2})
        DoD: speedup_vs_cudf_full ≥ 1.1 per row (formula already correct)

If verdict = At floor:
    next spec = "cuDF route honest negative result write-up + research pivot"
        scope: paper-facing summary of what compact-U warm-resident proved and didn't
        pivot direction: transfer-bound / cold-path / progressive multi-predicate
        the byteplane research thesis (Direction Report §Evaluation Questions
        #5 "Transfer payoff" and the broader bytes_read framing) remains valid;
        only the specific "warm exact COUNT vs cuDF reduction" framing is retired
```

## 10. One-Sentence Summary

Take the 3 ms byteplane per-query apart with nsys and ncu on a single sensor sel=99 k=1 cell, decide whether the 2× gap to cuDF is orchestration noise or a structural kernel floor, and let that decide whether the cuDF route gets one more fix-and-retry or an honest negative result.
