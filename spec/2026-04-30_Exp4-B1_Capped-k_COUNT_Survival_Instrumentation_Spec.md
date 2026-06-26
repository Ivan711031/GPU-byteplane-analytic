# Exp4-B1 Spec: Capped-k COUNT + Survival Instrumentation

**Research Branch:** Exp4 predicate branch  
**Semantic Change:** YES — adds capped-k execution semantics, Q/D/U output, and per-round survival metrics. Requires explicit review against guide v3 Rule 2.  
**Created:** 2026-04-30  
**Depends on:** `research/2026-05-01_Paper_Guide_Progressive_Byteplane.md` v3 (scope = SUM + COUNT), `PV1-8A` / GitHub issue `#2` (throughput metric schema), `PV1-0` / GitHub issue `#3` (Exp4 capped-k output schema)

---

## 0. Issue Mapping and Agent Ownership

This spec is the implementation handoff for two local backlog issues:

```text
PV1-1 / GitHub #4:
  Exp4-B1 capped-k COUNT tracer bullet on exact artifact

PV1-2 / GitHub #5:
  Exp4-B1 Q/D/U survival and pack-utilization metrics
```

If an agent is already implementing:

```text
--max-filter-planes k
Q(k), D(k), U(k)
count_lower / count_upper / count_abs_error_bound
survival metrics
pack-utilization metrics
```

then that agent is working on `#4` and `#5` together. Do not start another independent agent on the same code path unless the first one is stopped or explicitly reassigned.

This spec is not:

```text
GitHub #2 / PV1-8A:
  shared throughput metric schema contract

GitHub #3 / PV1-0:
  Exp4 COUNT capped-k output schema contract

GitHub #6 / PV1-3:
  full synthetic capped-k sweep

GitHub #7 / PV1-4:
  epsilon-to-kstar join and plots
```

Recommended ordering:

```text
#2 -> #3 -> #4/#5 -> #6 -> #7
```

---

## 1. Purpose

This spec defines the next implementation task for the Exp4 predicate branch.

**Current state:** Exp4-A exact progressive COUNT baseline is complete. It produces exact encoded-domain counts with early-exit, but cannot answer "what is the COUNT error bound if we stop after k planes?"

**Target state:** Exp4-B1 adds `--max-filter-planes k` mode. For each k, it outputs:
- `Q(k)` = confirmed qualified rows
- `D(k)` = confirmed disqualified rows
- `U(k)` = unresolved ambiguous rows after k planes
- `count_lower = Q(k)`, `count_upper = Q(k) + U(k)`
- `count_abs_error_bound = U(k)`
- Throughput at depth k
- Per-round survival and pack-utilization diagnostics

This is the **minimum evidence needed** to claim a COUNT precision-throughput curve (Paper v1 Gate G3).

---

## 2. Scope

### In Scope
1. Add `--max-filter-planes k` to `bench_progressive_filter`.
2. Implement capped-k semantics in CPU mirror (`cpu_encoded_stats`).
3. Output Q/D/U and count bounds in main CSV.
4. Output per-round survival sidecar CSV.
5. Smoke validation: full-depth k must match exact COUNT.
6. H200 sweep for exact artifact on 2 datasets × 3 selectivities.

### Out of Scope
- Kernel-side per-round Q/D/U atomics (too expensive; use CPU mirror).
- Strategy ablation (Exp4-C; deferred until survival patterns are known).
- Real dataset (Step 4 of roadmap; after synthetic capped-k curve is established).
- p3/p6 bounded artifact sweeps for this step (focus on exact artifact first to establish execution-depth semantics).
- Filter+aggregate (Exp4-D).

---

## 3. Background: Why Capped-k COUNT is Different from Exact Early-Exit

**Exact early-exit (Exp4-A):**
```text
Read planes until row is resolved or active_plane_count reached.
Output: exact encoded COUNT.
Question answered: "How many planes do rows typically need?"
```

**Capped-k COUNT (Exp4-B1):**
```text
Read at most k planes, even if some rows are still ambiguous.
Output: Q, D, U + throughput(k).
Question answered: "If I stop at k, what is my COUNT error bound?"
```

The key difference is that **capped-k deliberately leaves some rows unresolved** to save HBM bandwidth. The system trades exactness for speed, but provides a deterministic error bound.

---

## 4. Technical Design

### 4.1 Kernel Changes: Minimal

The GPU kernel `progressive_filter_rowpack16_passive` already reads `d_active_plane_count[segment_id]` to bound rounds.

**Approach:** Cap the host-side `h_active_plane_count` array before uploading to device.

```cpp
for (size_t seg = 0; seg < num_segments; ++seg) {
  h_active_plane_count[seg] = dataset.segments[seg].active_plane_count;
  if (opt.max_filter_planes >= 0) {
    h_active_plane_count[seg] = std::min(
      h_active_plane_count[seg],
      static_cast<uint32_t>(opt.max_filter_planes)
    );
  }
}
```

**Why this works:**
- The kernel uses `d_active_plane_count[seg]` as loop bound per segment.
- Capping on host means zero kernel code changes.
- Throughput and estimated bytes automatically reflect the cap.
- GPU/CPU validation remains valid if CPU mirror uses the same cap.

### 4.2 CPU Mirror Changes: Core Work

Modify `cpu_encoded_stats()` and `evaluate_scalar_row()` in `bench_progressive_filter.cu`.

#### 4.2.1 Pass Cap into CPU Mirror

Change signature:
```cpp
// Old:
EncodedReferenceStats cpu_encoded_stats(const Dataset& dataset, ...);

// New:
EncodedReferenceStats cpu_encoded_stats(
  const Dataset& dataset,
  const std::vector<uint32_t>& effective_max_rounds,  // per-segment cap
  ...
);
```

Inside `cpu_encoded_stats`, use `effective_max_rounds[seg]` instead of `segment.active_plane_count` as the round loop bound.

#### 4.2.2 Extend `EncodedReferenceStats`

```cpp
struct EncodedReferenceStats {
  // Existing fields
  uint64_t qualified_count = 0;
  uint64_t total_planes_read = 0;
  uint64_t gpu_processed_rows = 0;
  uint32_t max_planes_read = 0;
  uint64_t estimated_pack_load_bytes = 0;

  // New: count classification after cap
  uint64_t certainly_qualified = 0;      // Q: lower bound > threshold
  uint64_t certainly_disqualified = 0;   // D: upper bound <= threshold
  uint64_t uncertain = 0;                // U: bounds straddle threshold

  // New: per-round survival (size = max cap across segments)
  std::vector<uint64_t> qualified_per_round;
  std::vector<uint64_t> disqualified_per_round;
  std::vector<uint64_t> unresolved_per_round;
  std::vector<uint64_t> planes_read_per_round;
  std::vector<uint64_t> pack_load_bytes_per_round;

  // New: pack utilization at cap
  uint64_t fully_resolved_packs = 0;
  uint64_t partially_active_packs = 0;
  double avg_active_rows_per_active_pack = 0.0;
  double useful_row_fraction = 0.0;
};
```

#### 4.2.3 Per-Row Classification Logic

For each row (scalar or pack lane), after the round loop ends (either resolved or hit cap):

```text
IF resolved as qualified at round r:
  ++stats.qualified_per_round[r]
  ++stats.certainly_qualified

ELSE IF resolved as disqualified at round r:
  ++stats.disqualified_per_round[r]
  ++stats.certainly_disqualified

ELSE (hit cap without resolution):
  ++stats.unresolved_per_round[cap-1]
  
  // Compute integer combined-code bounds after k planes
  lower_combined = partial_code_with_omitted_bits_zero
  upper_combined = partial_code_with_omitted_bits_ones
  
  IF lower_combined > threshold_combined:
    ++stats.certainly_qualified   // Actually certain despite not resolved
  ELSE IF upper_combined <= threshold_combined:
    ++stats.certainly_disqualified // Actually certain despite not resolved
  ELSE:
    ++stats.uncertain              // True ambiguity
```

**Note on "certain despite not resolved":** Some rows may hit the cap but have bounds that still allow classification. This is a refinement over naive "unresolved = U".

#### 4.2.4 Pack Utilization Metrics

During the pack loop, track per-pack state:

```text
For each 16-row pack:
  active_mask after cap = remaining unresolved lanes
  
  IF active_mask == 0:
    ++fully_resolved_packs
  ELSE IF pack was ever active (i.e., not all_qualified / all_disqualified at start):
    ++partially_active_packs
    active_row_count += popcount(active_mask)
    total_rows_in_active_packs += 16
```

After the loop:
```text
avg_active_rows_per_active_pack = active_row_count / partially_active_packs
useful_row_fraction = active_row_count / total_rows_in_active_packs
```

### 4.3 Threshold and Error Bound Semantics

The error bound is **count-space**, not value-space:

```text
count_abs_error_bound = uncertain
                        = max possible difference between
                          exact COUNT and any COUNT consistent with k planes

exact COUNT ∈ [certainly_qualified, certainly_qualified + uncertain]
            = [Q, Q + U]
```

This is tighter than worst-case value-space bound because it leverages the predicate threshold directly.

### 4.4 Command-Line Interface

Add to `Options` struct and parser:

```text
--max-filter-planes N
  Default: -1 (disabled, use full active_plane_count per segment)
  Range: 1 <= N <= manifest.max_plane_count
  
--output-rounds-csv PATH
  Default: "" (disabled)
  If set, write per-round survival metrics to sidecar CSV.
```

### 4.5 CSV Output Formats

#### 4.5.1 Main CSV (Expanded Schema)

Append new columns to existing schema:

```text
# Existing columns 1-23 remain unchanged
# New columns:
max_filter_planes,
certainly_qualified,
certainly_disqualified,
uncertain,
count_lower,
count_upper,
count_abs_error_bound,
fully_resolved_packs,
partially_active_packs,
avg_active_rows_per_active_pack,
useful_row_fraction
```

Full column list becomes 34 fields.

#### 4.5.2 Rounds Sidecar CSV

Only emitted if `--output-rounds-csv` is set. One row per `(dataset, threshold, k, round)`.

```text
dataset,artifact_root,threshold,selectivity,max_filter_planes,round,
qualified,disqualified,unresolved,
planes_read,pack_load_bytes
```

This allows reconstructing survival curves:
```text
U(k) = cumulative unresolved after round k
Q(k) = cumulative qualified by round k
```

---

## 5. Validation and Correctness

### 5.1 Gate 1: Full-Depth Match

When `--max-filter-planes` equals or exceeds a segment's `active_plane_count`, the result must be identical to exact mode:

```text
certainly_qualified + uncertain == exact qualified_count
uncertain == 0
gpu_count == cpu_encoded_count
```

### 5.2 Gate 2: Monotonicity

For increasing k:
```text
U(k) is non-increasing
Q(k) is non-decreasing
Q(k) + U(k) is non-increasing
rows_per_sec(k) is generally non-increasing as k grows, subject to timing noise
```

Minor violations due to timing noise are acceptable; systematic violations indicate bugs.

### 5.3 Gate 3: Count Bound Correctness

For any k, the true exact COUNT must fall within `[count_lower, count_upper]`:

```text
count_lower <= exact_COUNT <= count_upper
```

This is verified by comparing against `cpu_encoded_count` from an uncapped run.

### 5.4 Gate 4: CPU/GPU Agreement Under Cap

When `--validate` is used with `--max-filter-planes`:
```text
gpu_count == cpu_encoded_stats.qualified_count
```

The CPU mirror and GPU kernel must agree even under the cap.

---

## 6. Smoke Test Plan

### 6.1 Minimal Smoke: Build on Login, Run on H200 Only

Login node may be used for editing, building, and job submission only. Do not run the benchmark binary on the login node.

```bash
# Build on login node
cmake -S benchmarks/experiment4 -B build/exp4 -DCMAKE_BUILD_TYPE=Release
cmake --build build/exp4 -j
```

Run the smoke test only inside an H200 allocation. The job must fail fast if the device is not H200.

```bash
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q H200 || exit 2

# Smoke: uniform, exact artifact, selectivity 50, one capped-k case
./build/exp4/bench_progressive_filter \
  --device 0 \
  --encoded-root /work/$USER/datasets/synthetic/dev_buff_exp3/uniform \
  --threshold $(python3 -c "import numpy as np; print(np.quantile(np.fromfile('/work/$USER/datasets/synthetic/dev/uniform.f64le.bin', dtype=np.float64), 0.50))") \
  --block 256 \
  --warmup 0 \
  --iters 1 \
  --validate \
  --max-filter-planes 3 \
  --csv /tmp/exp4_b1_smoke_k3.csv \
  --output-rounds-csv /tmp/exp4_b1_smoke_k3_rounds.csv
```

Acceptance:
- CSV produced with new columns populated
- Rounds sidecar produced
- `uncertain > 0` (expected for k=3 on uniform)
- `gpu_count == cpu_encoded_count`

### 6.2 Full-Depth Gate Smoke

Run same benchmark with `--max-filter-planes 10` (or > active_plane_count). Verify:
- `uncertain == 0`
- `count_lower == count_upper == exact_count`
- Matches uncapped run exactly

### 6.3 Monotonicity Smoke

Run k=1,2,3,4 sequentially on same dataset/threshold. Verify:
- `U(1) >= U(2) >= U(3) >= U(4)`
- Throughput decreases (or stays within noise)

---

## 7. H200 Sweep Plan

### 7.1 First Batch: Synthetic Exact Artifact

Focus: establish COUNT execution-depth semantics.

```text
Datasets:   uniform, heavy_tailed
Artifacts:  exact only (no p3/p6 for this batch)
Selectivities: 50, 90, 99
k values:   1 .. active_plane_count for each dataset
Iterations: 200
Warmup:     10
```

Estimated runs:
- uniform:     10 planes × 3 selectivities = 30 runs
- heavy_tailed: 16 planes × 3 selectivities = 48 runs
- Total: 78 runs

### 7.2 Sweep Script Extension

Extend `run_exp4.sh` to support `--max-planes-sweep` mode:

```bash
# New usage:
sbatch --account=<PROJECT_ID> \
  scripts/run_exp4_b1.sh \
  --dataset uniform \
  --artifacts exact \
  --selectivities "50 90 99" \
  --max-planes-sweep \
  --output-dir results/exp4/b1_uniform_$(date +%Y%m%d_%H%M%S)
```

When `--max-planes-sweep` is set, the inner loop iterates k from 1 to max_active_plane_count instead of running a single exact pass.

### 7.3 Output Structure

```
results/exp4/b1_uniform_TIMESTAMP/
├── run_uniform_exact_s50_k1_JOBID.csv
├── run_uniform_exact_s50_k2_JOBID.csv
├── ...
├── run_uniform_exact_s99_k10_JOBID.csv
├── sweep_summary_uniform_TIMESTAMP_jobJOBID.csv
├── rounds_sidecar_uniform_TIMESTAMP_jobJOBID.csv
└── run_meta.txt
```

---

## 8. Backward Compatibility

### 8.1 Default Behavior Unchanged

Without `--max-filter-planes`, benchmark behaves exactly as before:
- Uses full `active_plane_count` per segment
- Produces exact encoded COUNT
- Existing 23-field CSV schema unchanged (new columns appended, parsers that read by index need update; parsers that read by header are fine)

### 8.2 Existing Sweeps

Existing Exp4 sweep scripts without capped-k options continue to run exact-COUNT sweeps exactly as before.

### 8.3 p3/p6 Artifacts

Capped-k mode works with any artifact including p3/p6, but the first batch focuses on exact to isolate execution-depth semantics from representation fidelity questions.

---

## 9. Acceptance Criteria

The task is complete when ALL of the following are true:

1. `bench_progressive_filter` builds successfully with `--max-filter-planes` support.
2. Smoke test (Section 6.1) produces correct Q/D/U output.
3. Full-depth gate (Section 6.2) shows `uncertain == 0` and matches exact baseline.
4. Monotonicity smoke (Section 6.3) passes.
5. H200 sweep (Section 7) completes for uniform + heavy_tailed exact artifact.
6. Sweep outputs include both main CSV (with new columns) and rounds sidecar.
7. `sweep_summary.csv` merges correctly across all k values.
8. `run_meta.txt` records the new parameters and any gate failures.
9. A Python script can read the merged CSV and produce:
   - `U(k)` vs k plot
   - Throughput(k) vs k plot
   - `epsilon -> k*` table (for a given selectivity)

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CPU mirror Q/D/U logic has off-by-one in bound calculation | Medium | High | Exhaustive unit test with small synthetic segments; compare against brute-force per-row decode |
| Rounds sidecar file becomes very large (one row per k × round × run) | Medium | Low | Compress or aggregate after validation; sidecar is diagnostic, not publication artifact |
| H200 queue delays sweep completion | High | Low | Submit as single batch job; wall-time estimate 2-3 hours for 78 runs |
| Per-round instrumentation slows CPU mirror too much for validation | Low | Medium | Make instrumentation conditional on `--validate`; non-validate runs skip per-round histograms |

---

## 11. References

- `research/2026-05-01_Paper_Guide_Progressive_Byteplane.md` v3, Sections 6.4 (Exp4), 10.1 (Paper v1 Gates), 13 (Exp4 Roadmap)
- Local-only runbook: RUN_EXP4_GUIDE.md (existing sweep infrastructure, see local docs/runbooks/)
- Source files:
  - `benchmarks/experiment4/bench_progressive_filter.cu`
  - `benchmarks/experiment4/exp4_kernels_filter.cuh`
  - `scripts/run_exp4_b1.sh`

---

## 12. Implementation Notes

### Status: Working-tree Implementation Reported, Verification Required (2026-04-30)

The current working tree contains changes in `bench_progressive_filter.cu` and a new `scripts/run_exp4_b1.sh`. Treat the following as implementation notes to verify, not as paper-ready evidence:

- `Options` struct should be extended with `max_filter_planes` and `output_rounds_csv`.
- `parse_args()` should parse `--max-filter-planes` and `--output-rounds-csv`.
- `SegmentThresholdInfo` should store `threshold_combined` for bound checking.
- `RowEval` should include qualified/disqualified/unresolved classification.
- `EncodedReferenceStats` should include Q/D/U, per-round histograms, and pack utilization.
- `evaluate_scalar_row()` should accept a `max_rounds` cap and return classification.
- `combined_bounds_after_k_planes()` should compute integer combined-code bounds.
- `cpu_encoded_stats()` should:
  - Accepts per-segment `effective_max_rounds`
  - Tracks per-round Q/D/U histograms
  - Tracks pack utilization (fully_resolved_packs, partially_active_packs, etc.)
  - Classifies unresolved rows using combined-code bounds vs threshold
- Main function should apply the cap to `h_active_plane_count` before GPU upload.
- Main CSV schema should include capped-k metrics.
- Rounds sidecar CSV should be emitted when `--output-rounds-csv` is set.
- `scripts/run_exp4_b1.sh` should support H200 k-sweep submission.

### Zero Kernel Changes Required

The GPU kernel `progressive_filter_rowpack16_passive` requires **no modifications**. The cap is applied on the host side by limiting `h_active_plane_count[seg]` before uploading to `d_active_plane_count`. This is intentional: the kernel already reads `d_active_plane_count[segment_id]` as its round bound.

### Next Steps (Requires H200)

1. **Build on login node:**
   ```bash
   cd /home/$USER/workspace/gpu-byteplane-scan-experiments
   cmake -S benchmarks/experiment4 -B build/exp4 \
     -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
   cmake --build build/exp4 -j
   ```

2. **Smoke test on H200:**
   ```bash
   srun -p dev --gres=gpu:1 --time=00:10:00 bash scripts/test_exp4_b1_smoke.sh
   ```
   (Script not yet created; can use manual srun with `--max-filter-planes 3 --validate`)

3. **Submit full sweep:**
   ```bash
   sbatch --account=<PROJECT_ID> scripts/run_exp4_b1.sh \
     --datasets uniform,heavy_tailed \
     --artifacts exact \
     --selectivities "50 90 99"
   ```

### Known Limitations

- Per-round instrumentation is **always enabled** when `--validate` is used. There is no conditional skip for non-validate runs yet.
- The `count_abs_error_bound` equals `uncertain` (true ambiguity rows). This is a count-space bound, not a value-space bound. It assumes each ambiguous row can flip the COUNT by at most 1.
- `useful_row_fraction` and `avg_active_rows_per_active_pack` are only meaningful when there are partially active packs. For high-selectivity or very easy thresholds, these may be 0/NA.
