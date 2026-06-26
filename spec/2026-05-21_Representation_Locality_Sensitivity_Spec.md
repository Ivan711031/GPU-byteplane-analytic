# Representation Locality Sensitivity Spec: What Does Segment Locality Buy?

**Date:** 2026-05-21
**Status:** Spec only. Do not implement in this task.
**Replaces research question framing of:** `spec/2026-05-12_Exp6_FOR_Segment_Granularity_Sensitivity_Spec.md` (Exp6)
**Relationship to Exp6:** This spec inherits the Exp6 dimension (segment size) but sharpens the research question, adds mandatory cost accounting, and requires a definitive payoff table. It does not replace Exp6 — Exp6 remains the general design-space exploration. This spec is the **paper-facing decisive experiment** derived from that exploration.

---

## 1. Research Question

The existing artifacts use `segment_size = 4096` as the default representation locality. This was a reasonable engineering choice inherited from early experiments (Exp2/Exp3/Exp4), but the question was never decisively answered:

> Does segment-local bounded-float representation provide enough value in k\*, ambiguity decay, SUM error bounds, and scan latency to justify its higher threshold-prep cost and metadata overhead?

**Framed as a payoff question:**

| Benefit of fine locality | Cost of fine locality |
|---|---|
| Lower integer offset bits per segment | More segments → more metadata |
| Fewer active planes | More threshold-prep work (classify, encode, pack) |
| Faster U(k) decay (lower k\*) | More bytes in side tables |
| Tighter per-segment SUM error bounds | Higher per-segment bookkeeping in kernel |
| Greater early-termination potential | Diminishing returns vs quasi-global |

The experiment is a **controlled comparison of four representation locality levels** across three datasets and two operators, with mandatory cost instrumentation at every point.

### 1.1 What This Spec Is NOT

- This is **not** a search for a single "optimal" segment size. The goal is a tradeoff characterization.
- This is **not** an Exp6 replacement. Exp6 sweeps more granularities (512–16384) and more datasets (uniform, zipfian). This spec picks a decisive subset for the paper-facing claim.
- This is **not** an encoder semantics experiment. The encoder is unchanged; only the segment-size parameter varies.

---

## 2. Required Representation Variants

Four levels of representation locality:

| Label | segment_size | Rationale |
|---|---|---|
| **4096** | 4,096 | Current engineering default; established baseline from Exp2/Exp3/Exp4 |
| **16384** | 16,384 | Medium locality; 4× coarser than baseline, 4× fewer segments |
| **65536** | 65,536 | Coarse locality; 16× coarser than baseline, 16× fewer segments |
| **quasi-global** | `value_count` (entire column = 1 segment) | One code-space for the whole column; eliminates segment prep overhead entirely |

### 2.1 Quasi-Global Definition

**Quasi-global is defined as true globalized encoding:** one segment covering the entire column (`segment_size = value_count`). This is supported by the mainline `buff_tool encode --segment-size N` command, which accepts any positive integer. There is no hard-coded upper limit in the codec (`segment_size` is stored as `uint64_t` in the file header).

**What this means for each dataset:**

| Dataset | value_count | quasi-global segment_size | segments |
|---|---|---|---|
| cesm_atm_cloud | 168,480,000 | 168,480,000 | 1 |
| hurricane_u | 25,000,000 | 25,000,000 | 1 |
| heavy_tailed | 1,000,000 | 1,000,000 | 1 |

**Limitation stated explicitly:** Quasi-global encoding with a single segment produces a single set of base/scale values and integer offset bits for the entire column. This is the **coarsest possible locality** and serves as the "zero-prep-cost" upper bound. It is NOT a BUFF clone — BUFF uses block-wise encoding with smaller blocks. Quasi-global is purely a locality baseline: it represents what the representation looks like when locality is completely disabled.

**Practical note for implementation:** `export-runtime` with a single segment produces plane files where each plane is `value_count` bytes (one byte per row). For cesm_atm_cloud (168M rows), each plane is ~161 MB. This is valid and loadable by the current Exp4 runner, but the agent should verify there are no implicit array-size limits in the benchmark kernel before running the full sweep.

---

## 3. Datasets

Exactly three datasets, chosen to span the relevant dimensions:

| Dataset | Rows | Type | Segment count @4096 | Why included |
|---|---|---|---|---|
| **cesm_atm_cloud** | 168,480,000 | Real scientific (CLUBB cloud field) | 41,133 | Large column; threshold prep is most expensive here (6.8–14ms in existing data); shows whether locality benefits survive at scale |
| **hurricane_u** | 25,000,000 | Real scientific (wind field) | 6,104 | Smaller scientific field; allows cross-check of scale sensitivity |
| **heavy_tailed** | 1,000,000 | Synthetic (power-law) | 245 | Stress case with extreme skew; should show strongest local-range benefit if it exists |

### 3.1 Precision

Use a single precision level for all artifacts: the best available precision (p12 / dec12, i.e. 12-bit decimal / 50 fractional bits). This avoids confounding the locality comparison with precision effects. Rationale:
- The research question is about locality, not precision.
- Existing Exp4 formal sweeps at p10/p12 show the pattern is consistent across precision levels.
- If the implementation agent cannot generate p12 artifacts for all segment sizes, p10 (10 decimal bits) is an acceptable fallback, stated explicitly.

---

## 4. Operators

### 4.1 COUNT

```
COUNT(*) WHERE x > threshold
```

Threshold selectivities: **10%, 50%, 90%**

These three points capture:
- **10%** — low selectivity, early termination is most beneficial
- **50%** — moderate selectivity, the crossover regime
- **90%** — high selectivity, worst case for progressive filtering

### 4.2 SUM

```
SUM(x) WHERE x > threshold  →  deterministic absolute and relative error bounds
```

Same thresholds as COUNT (10%, 50%, 90%).

SUM is mandatory because:
- SUM error bounds are segment-local (per-segment max_abs_error accumulates)
- Larger segments → wider local range → potentially looser SUM bounds
- The SUM operator reveals whether finer locality buys tighter aggregate error guarantees

### 4.3 k Sweep

For every (dataset, representation, operator, threshold):

```
k = 1 .. max_plane_count
```

Read `max_plane_count` from the artifact manifest. Do not hard-code.

---

## 5. Epsilon Ladder

### 5.1 COUNT epsilon (execution epsilon, relative to encoded full-depth COUNT)

```
epsilon = 0, 1e-4, 1e-3, 1e-2
```

### 5.2 SUM epsilon (deterministic bound)

For SUM, epsilon is defined differently: it is the **deterministic absolute error bound**, not a relative tolerance. The reported metric is:

```
sum_abs_bound(k) = sum over segments of segment_max_abs_error_bound
sum_rel_bound(k) = sum_abs_bound(k) / |raw_sum|
```

Report both `sum_abs_bound(k)` and `sum_rel_bound(k)` for each k. Then compute `k*` as the smallest k where `sum_rel_bound(k) <= epsilon_target`.

```
epsilon_target = 1e-2, 1e-3, 1e-4
```

---

## 6. Required Outputs / Metrics

### 6.1 Representation-level metrics (per dataset, per representation)

| Field | Description |
|---|---|
| dataset | cesm_atm_cloud / hurricane_u / heavy_tailed |
| representation_label | 4096 / 16384 / 65536 / quasi-global |
| segment_size | 4096 / 16384 / 65536 / value_count |
| value_count | Total rows in the dataset |
| segment_count | ceil(value_count / segment_size) |
| total_artifact_bytes | Sum of all plane files + segment_meta.csv + side tables |
| bytes_per_row | total_artifact_bytes / value_count |
| metadata_bytes | Estimated: segment_meta.csv size + side-table overhead |
| main_bytes_total | Sum of plane file sizes |
| active_plane_count_mean | Mean across segments |
| active_plane_count_p95 | 95th percentile across segments |
| active_plane_count_max | Maximum across segments |
| integer_offset_bits_mean | Mean across segments |
| integer_offset_bits_p95 | 95th percentile across segments |
| integer_offset_bits_max | Maximum across segments |
| max_plane_count | Maximum across segments (for k sweep ceiling) |
| generation_status | ok / failed (with reason) |

### 6.2 COUNT metrics (per dataset, per representation, per selectivity, per k)

| Field | Description |
|---|---|
| selectivity | 10% / 50% / 90% |
| k | 1 .. max_plane_count |
| Q_k | Qualified count (rows that pass filter at k planes) |
| D_k | Disqualified count (rows that fail filter at k planes) |
| U_k | Uncertain count (rows that need more planes) |
| U(k) decay | U_k / total_rows (fraction remaining uncertain) |
| count_lower | Q_k |
| count_upper | Q_k + U_k |
| execution_rel_error_bound | (U_k) / max(1, Q_k) |

### 6.3 SUM metrics (per dataset, per representation, per selectivity, per k)

| Field | Description |
|---|---|
| k | 1 .. max_plane_count |
| raw_sum | SUM of qualifying rows in raw FP64 |
| encoded_sum | SUM of qualifying rows from encoded (at k planes) |
| sum_abs_error | |raw_sum - encoded_sum| |
| sum_rel_error | sum_abs_error / |raw_sum| |
| sum_abs_bound(k) | Σ segment_max_abs_error_bound for qualifying segments |
| sum_rel_bound(k) | sum_abs_bound(k) / |raw_sum| |
| segment_max_abs_error_bound | Per-segment error bound function |

### 6.4 Epsilon-to-k\* metrics (per dataset, per representation, per selectivity, per epsilon)

| Field | COUNT | SUM |
|---|---|---|
| epsilon | 0, 1e-4, 1e-3, 1e-2 | 1e-4, 1e-3, 1e-2 (as rel bound targets) |
| k\* | Smallest k where ε bound satisfied | Smallest k where sum_rel_bound(k) ≤ ε |
| U(k\*) | Fraction uncertain at k\* | N/A |
| rows_per_sec_at_k\* | Rows/sec throughput at k\* planes | Same |

### 6.5 Cost metrics — MANDATORY (NOT optional)

These are the central measurements. Without them the payoff question cannot be answered.

| Field | Description | Source |
|---|---|---|
| **threshold_prep_ms** | End-to-end host-side time to translate a threshold into segment-classified metadata | Existing benchmark timer (already captured in Exp4 scientific sweeps) |
| threshold_classify_ms | Sub-step: segment range classification (ALLD/ALLQ/MIXED) | Existing breakdown |
| threshold_encode_ms | Sub-step: floating-point-to-fixed-point encoding per segment | Existing breakdown |
| threshold_pack_ms | Sub-step: packing metadata arrays for GPU transfer | Existing breakdown |
| **segment_classification_ms** | Time to determine segment disposition (all-qualify, all-disqualify, mixed) | Usually included in classify_ms |
| **scan_latency_ms** | GPU kernel time only (device-side progressive scan) | Existing: gpu_total_ms or per_query_ms_median |
| **artifact_load_ms** | Time to load artifact bytes from disk to host memory | Existing: artifact_load_ms |
| metadata_h2d_ms | Time to transfer segment metadata from host to device | If separately measurable |
| **end_to_end_latency_ms** | Sum: artifact_load_ms + threshold_prep_ms + scan_latency_ms (for a single cold query) | Computed from above components |

**Existing baseline (for reference):** From the Exp4 cesm_atm_cloud sweep at segment_size=4096 (41,133 segments):

| Component | Typical value (50% selectivity) |
|---|---|
| artifact_load_ms | ~764 ms (one-time, amortizable) |
| threshold_prep_ms | ~13 ms |
| threshold_classify_ms | ~1.3 ms |
| threshold_encode_ms | ~2.5 ms |
| threshold_pack_ms | ~0.27 ms |
| scan_latency_ms (gpu_total_ms) | ~1.5 ms |
| **E2E (cold, excluding load)** | **~14.5 ms** |

For hurricane_u at segment_size=4096 (6,104 segments):

| Component | Typical value (50% selectivity) |
|---|---|
| threshold_prep_ms | ~3.2 ms |
| scan_latency_ms | ~0.4 ms |
| **E2E (cold, excl load)** | **~3.6 ms** |

---

## 7. Experiment Matrix

### 7.1 Full Matrix

```
Representations: 4 (4096, 16384, 65536, quasi-global)
Datasets:        3 (cesm_atm_cloud, hurricane_u, heavy_tailed)
Operators:       2 (COUNT, SUM)
Selectivities:   3 (10%, 50%, 90%)
k:               1 .. max_plane_count (per artifact)
Epsilons:        4 for COUNT, 3 for SUM
```

Total artifact-generation points: 4 × 3 = **12 artifacts**

Total query points (COUNT + SUM): for each artifact, up to `max_plane_count` × 3 selectivities, plus epsilon-to-k\* derived values. Worst-case estimate (max_plane_count ≈ 8):

- COUNT: ~12 × 8 × 3 = 288 rows (plus epsilon-to-k\* derived)
- SUM: ~12 × 8 × 3 = 288 rows (plus epsilon-to-k\* derived)

### 7.2 Smoke Gate

Before the full sweep, run a minimal smoke:

- 1 dataset (heavy_tailed — fastest to encode, most informative)
- 4 representations (4096, 16384, 65536, quasi-global)
- 1 operator (COUNT)
- 1 selectivity (50%)
- k = 1, 2, max_plane_count

Gate criteria:
1. All 4 artifacts generate successfully.
2. All 4 artifacts load in the COUNT benchmark runner.
3. `gpu_count == cpu_encoded_count` for all 12 k-points.
4. Threshold prep latency is measurable and non-zero.

---

## 8. Locality Payoff Table

The final deliverable must include a table of this form (one per operator or consolidated):

### 8.1 Consolidated Payoff Table (per dataset, 50% selectivity)

| Representation | Size (B/row) | Metadata (bytes) | Prep ms | COUNT k\* (ε=1e-3) | SUM k\* (ε=1e-3) | Scan ms @ k\* | E2E ms (cold) | Verdict |
|---|---|---|---|---|---|---|---|---|
| 4096 | | | | | | | | |
| 16384 | | | | | | | | |
| 65536 | | | | | | | | |
| quasi-global | | | | | | | | |

**E2E ms = artifact_load_ms + threshold_prep_ms + scan_latency_ms** (cold query; if load is one-time amortized, also report "E2E ms (warm)" excluding load).

### 8.2 Questions This Table Must Answer

1. **Does finer locality buy lower k\*?** Compare COUNT k\* and SUM k\* across the four representations.
2. **Does finer locality buy faster U(k) decay?** Compare U_k at each k across representations.
3. **Does finer locality buy tighter SUM error bounds?** Compare sum_rel_bound(k) across representations.
4. **Does finer locality buy smaller artifacts?** Compare bytes_per_row.
5. **Are these gains enough to offset higher prep latency?** Compare E2E ms (cold) — the key payoff metric.
6. **Does quasi-global win on E2E despite worse k\*?** If prep dominates, quasi-global may have lower total latency even at higher k\*.

---

## 9. Decision Rules

These rules are stated explicitly so the report conclusion is unambiguous.

### 9.1 When 4096 stays justified

Maintain `segment_size = 4096` as the recommended default if ALL of the following hold:

- **k\* benefit is real but small**: COUNT k\* at 4096 is ≤ 2 planes lower than at 65536 or quasi-global for most (dataset, selectivity) combinations.
- **E2E latency is not dominated by prep**: The prep time at 4096 adds ≤ 2 ms compared to quasi-global, and total E2E is still ≤ 2× quasi-global E2E.
- **Bytes-per-row is better**: 4096 bytes-per-row ≤ 65536 bytes-per-row (i.e., finer locality compresses better).

If prep at 4096 is 5–10× worse than quasi-global (as the existing cesm_atm_cloud data hints: ~13ms vs ~0ms), then 4096 is **NOT justified on latency grounds alone** and the recommendation must acknowledge this.

### 9.2 When coarser locality (16384, 65536, quasi-global) wins

Recommend a coarser locality if:

- **Prep latency dominates**: threshold_prep_ms at 4096 is ≥ 50% of total E2E and is ≥ 3× the prep time at the coarser level.
- **k\* does not degrade catastrophically**: COUNT k\* at 16384 or 65536 is within 1 plane of 4096 k\*.
- **Bytes-per-row at coarser locality is competitive**: ≤ 1.5× the bytes-per-row at 4096.

### 9.3 When segment locality is NOT worth its query-time cost

This is the decisive negative case:

- **Prep latency is the bottleneck**: threshold_prep_ms at any segment size is ≥ 5× scan_latency_ms (the GPU kernel is waiting on the CPU).
- **k\* is already low for quasi-global**: quasi-global k\* ≤ 3 for COUNT at 50% selectivity (meaning the global encoding is already precise enough).
- **SUM error bounds are not significantly tighter for finer locality**: sum_rel_bound at quasi-global is within 2× of 4096 at the same k.

If all three conditions hold, the paper should state:

> Segment-local representation does not provide enough query-time value to justify its prep cost for these datasets. Quasi-global encoding achieves comparable k\* and error bounds with latency dominated by the GPU scan rather than host-side planning.

If only some conditions hold, the recommendation is qualified (e.g., "fine locality matters for heavy-tailed but not for cesm_atm_cloud").

### 9.4 Distribution-dependent fallback

If no single representation wins across all datasets, recommend a dataset-adaptive heuristic:

```
For datasets where p95 integer_offset_bits > 4 at 65536:
  prefer 4096 or 16384
Else:
  prefer 65536 or quasi-global
```

---

## 10. Deliverables

### 10.1 Smoke results

```
results/locality_sensitivity/smoke/2026-05-21_Locality_Smoke_Gate.csv
```

### 10.2 Full result tables

```
results/locality_sensitivity/artifact_summary.csv
results/locality_sensitivity/count_sweep.csv
results/locality_sensitivity/sum_sweep.csv
results/locality_sensitivity/epsilon_kstar_count.csv
results/locality_sensitivity/epsilon_kstar_sum.csv
results/locality_sensitivity/locality_payoff_table.csv
```

### 10.3 Final report

```
research/2026-05-21_Representation_Locality_Sensitivity_Report.md
```

Required sections:

1. **Experiment matrix and generation status** — which artifacts succeeded/failed
2. **Representation metrics** — size, active planes, offset bits per locality level
3. **COUNT convergence** — Q/D/U, U(k) decay, k\* for each locality level
4. **SUM error bounds** — deterministic absolute/relative bounds, k\* for each locality level
5. **Cost breakdown** — threshold_prep_ms, scan_latency_ms, E2E ms comparison
6. **Locality Payoff Table** — the consolidated comparison (Section 8.1)
7. **Decision verdict** — which representation(s) win, with rationale
8. **Caveats and follow-up** — known limitations; whether GPU-offloaded prep would change the conclusion

### 10.4 Required figures

```
results/locality_sensitivity/plots/kstar_vs_locality.png
results/locality_sensitivity/plots/prep_vs_locality.png
results/locality_sensitivity/plots/e2e_vs_locality.png
results/locality_sensitivity/plots/uk_decay_by_locality.png
results/locality_sensitivity/plots/bytes_per_row_vs_locality.png
results/locality_sensitivity/plots/sum_bound_vs_k_by_locality.png
```

---

## 11. Acceptance Criteria

### 11.1 Smoke Gate

1. All 4 representations for heavy_tailed generate and load successfully.
2. `gpu_count == cpu_encoded_count` for all smoke points.
3. Threshold prep latency is measurable across all 4 representations.
4. The payoff table can be populated with at least heavy_tailed data.

### 11.2 Full Experiment

1. All 12 artifacts have complete representation metrics.
2. Every successful artifact has full k coverage (k=1..max_plane_count) for COUNT and SUM.
3. U(k) is monotonic non-increasing for every (dataset, representation, selectivity) combination.
4. k\* ≤ max_plane_count for all epsilon values.
5. Kernel correctness: `gpu_count == cpu_encoded_count` for every query point.
6. The payoff table is complete for all datasets (with failed artifacts noted explicitly).
7. The report contains an explicit verdict answering the research question, referencing the payoff table.

---

## 12. Implementation Notes for the Next Agent

1. **Quasi-global pragmatics**: Encode with `--segment-size VALUE_COUNT`. The `buff_tool` supports this. Verify that `export-runtime` and the benchmark kernel work with a single-segment artifact (no implicit array-size limit). If the kernel crashes with a single segment, fall back to the largest segment size that works and label it "quasi-global (proxy: segment_size=N)".

2. **Scientific datasets**: The existing `build_scientific_v2_artifact.py` hardcodes `SEGMENT_SIZE = 4096`. To generate alternate segment-size artifacts, either:
   - Modify the script to accept `--segment-size` (preferred, cleanest), or
   - Use `buff_tool encode --segment-size N` directly followed by `buff_tool export-runtime ...`.

3. **Synthetic dataset**: `build_v2_smoke_artifacts.py` already accepts `--segment-size`. Use it for heavy_tailed.

4. **Precision note**: The heavy_tailed synthetic artifact in Exp4 used p8 (8 decimal bits). If the scientific artifacts use p12 (dec12), maintain that precision for the synthetic too. If p12 encoding fails for heavy_tailed (e.g., integer offset bits exceed 64), fall back to p10 then p8, and note the precision used.

5. **Cost measurement**: The existing Exp4 scientific sweep already captures all required cost fields (`threshold_prep_ms`, `threshold_classify_ms`, `threshold_encode_ms`, `threshold_pack_ms`, `gpu_total_ms`). The agent should verify these fields are present in the benchmark output before adding new instrumentation.

6. **Running the experiment**: Follow project Slurm rules:
   - H200 only (hardware gate before benchmark).
   - No account embedded in scripts.
   - Smoke first, then full sweep.
   - Batch full matrix (12 artifacts × up to 8 k-points × 3 selectivities × 2 operators) in as few jobs as possible.

7. **Avoiding Exp6 duplication**: This spec uses a narrower dataset/representation matrix than Exp6. If the agent has already run Exp6 and has relevant data points (e.g., 16384 on heavy_tailed), those can be reused if the cost metrics are present. The agent must verify that the existing Exp6 data includes threshold_prep_ms for all segment sizes — if it was marked optional in Exp6 and not collected, re-run is required.

---

## 13. Paper Claim Boundary

### Supported (if data confirms)

> Finer segment locality reduces local value range, lowering integer offset bits and active plane counts, which speeds up U(k) convergence and tightens per-segment SUM error bounds. However, the threshold preparation cost scales linearly with segment count; for large columns (168M rows, 41K segments at 4096), host-side prep latency (6–14 ms) dominates the GPU scan latency (1–2 ms), making coarser locality or quasi-global encoding more attractive for ad-hoc single-query workloads.

### Not supported (would require additional experiments)

- A claim that any single segment size is universally optimal.
- A claim about GPU-offloaded threshold preparation (acknowledged as future work in the threshold-prep analysis).
- A claim about multi-query amortization (if the prepared predicate framing is used, this should be stated as an assumption).

### Potential paper wording

> We compared four levels of representation locality — 4096-row segments (fine), 16K-row segments (medium), 65K-row segments (coarse), and column-level global encoding — across three datasets (two scientific, one synthetic) using COUNT and SUM queries at multiple selectivity points. [Results: summary of payoff table]. We find that [X] locality represents the best tradeoff for [Y] workloads, and that segment-local encoding is [justified / not justified] for ad-hoc queries because [Z].

---

## Appendix A: Relationship to Existing Exp6 Spec

| Dimension | Exp6 (2026-05-12) | This spec (2026-05-21) |
|---|---|---|
| Research question | "How does FOR segment granularity affect..." | "Does segment locality provide enough value to justify its cost?" |
| Segment sizes | 512, 1024, 2048, 4096, 8192, 16384 (+optional 32768, 65536) | 4096, 16384, 65536, quasi-global |
| Datasets | uniform, sensor, heavy_tailed, zipfian | cesm_atm_cloud, hurricane_u, heavy_tailed |
| Operators | COUNT only | COUNT + SUM |
| Precisions | p4, p6, p8 | Single precision (p12 or best available) |
| Cost metrics | Optional (Section 7.5) | **Mandatory** (Section 6.5) |
| Payoff table | Not required | Central deliverable (Section 8) |
| Decision rules | Keep 4096 vs smaller vs adaptive | Keep 4096 vs coarser wins vs locality not worth cost vs adaptive |
| Paper-facing | General design-space exploration | Decisive experiment for paper claim |
