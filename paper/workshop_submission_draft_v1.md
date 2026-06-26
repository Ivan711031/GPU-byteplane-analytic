# A Scoped H200 Operator Study of BUFF-Style Progressive Byte-Plane Filter-Aggregate Execution

**Status:** submission-facing Markdown draft v1  
**Scope:** workshop paper draft; no experiments run for this file  
**Primary evidence:** committed H200 scientific, Q8, Prompt5, Exp3, and figure-packet artifacts  

## Abstract

Scientific post-analysis often evaluates threshold-conditioned reductions over large floating-point fields. BUFF-style byte-oriented bounded-float storage, DAQ-style deterministic approximation, and Column-Sketch-style predicate ambiguity already establish the key representational and semantic ingredients for progressive compressed-domain query execution. This paper asks a narrower systems question: how does this inherited progressive byte-plane execution model behave when implemented as a GPU-resident fused Filter+Aggregate operator on NVIDIA H200?

We evaluate BUFF-style byte-plane artifacts under a warm, device-resident workload model for repeated scientific threshold analysis. Warm E2E latency includes per-query threshold preparation when applicable and resident GPU execution; it excludes cold artifact construction, initial host-to-device upload, file I/O, and offline threshold percentile selection unless explicitly stated. The primary runtime baseline is a same-H200 raw-fused FP64 CUDA primitive that applies the threshold and accumulates COUNT and SUM in one kernel without materializing predicate masks or survivor buffers.

On four tested true-quasi-global CESM/Hurricane scalar fields, the H200 executor reports k=2 shallow-depth speedups of about 2.71x-5.96x over raw-fused FP64 CUDA, while k=max rows remain diagnostic/fallback evidence rather than shallow-k wins. The 5.96x upper endpoint reflects a `cesm_atm_q` high-selectivity bucket row whose k=2 and k=max selectivity labels are not identical (65.02% vs. 90.00%); this is reported as bucket evidence rather than a paired-threshold comparison. The scientific headline is SUM-oriented fused Filter+Aggregate execution. COUNT is reported because the fused primitive naturally produces it and because predicate ambiguity is part of the inherited semantics, but COUNT drift and unresolved rows are treated as mechanism and limitation evidence, not as a headline success.

We attribute the result through deployment-locality accounting, a scoped scientific NCU physical-traffic case, synthetic k-depth dispatch evidence, and explicit fallback behavior. True-quasi-global locality keeps per-query threshold preparation below 0.06 ms in the tested path, whereas fine-local 4096-segment artifacts can incur 7-19.42 ms preparation costs. Q8 provides one scoped scientific-field physical-traffic audit for `cesm_atm_q` s10; Prompt5 provides synthetic k-depth break-even evidence over `uniform_p10` and `heavy_tailed_p6`. The resulting claim is intentionally narrow: progressive byte-plane execution is useful on H200 when the query remains in a measured latency-winning shallow-k regime, when locality avoids threshold-preparation domination, and when raw FP64 fallback is used past empirical break-even.

## 1. Introduction

Large scientific simulations and observational pipelines produce floating-point arrays that are scanned repeatedly during post-analysis. A scientist may evaluate how much cloud signal lies above a threshold, how a storm-field summary changes across cutoff values, or how a thresholded average shifts under a changing mask. These workloads are not full SQL pipelines. They are scan-heavy threshold-conditioned reductions over arrays that may already be prepared and resident during repeated analysis.

Conventional GPU analytical execution typically treats values as fixed-precision FP32 or FP64 buffers and accelerates the operator over that representation. A different path is possible when data are stored in significance order. If the query can tolerate bounded loss or ambiguity, an executor may read only leading byte planes rather than the full representation. Prior work already provides the conceptual ingredients: BUFF-style byte-oriented bounded-float storage [BUFF], deterministic approximate query processing [DAQ], and tri-state predicate ambiguity [ColumnSketch]. This paper does not claim these ideas as new.

The question here is hardware and execution oriented. Given BUFF-style byte-plane artifacts and inherited deterministic progressive semantics, what happens when the operator is implemented as a fused Filter+Aggregate primitive on H200? H200 provides high memory bandwidth, but fewer logical bytes do not automatically imply lower latency. Reconstruction instructions, cache transactions, threshold preparation, layout, and dispatch depth all decide whether reading fewer byte planes actually wins.

The evaluated query shape is a single-column thresholded reduction:

```sql
SELECT COUNT(*), SUM(x), SUM(x) / COUNT(*)
FROM field
WHERE x > threshold;
```

The implementation fuses predicate evaluation and aggregation. The paper is SUM-primary. COUNT is secondary: it is reported because the fused primitive produces it and because Q/D/U ambiguity explains the predicate mechanism, but the strongest scientific result is for SUM-derived reductions. AVG is derived from SUM and COUNT. MIN, MAX, and VAR are outside the evaluated submission scope.

Building on BUFF-style bounded-float byte-oriented representation and DAQ / Column-Sketch-style deterministic progressive semantics, our contribution is a scoped GPU-resident execution and locality-attribution study of progressive byte-plane fused Filter+Aggregate on NVIDIA H200. On four tested true-quasi-global CESM atmosphere and Hurricane Isabel scalar fields under a warm device-resident repeated-threshold workload, the H200 implementation achieves k=2 shallow-depth speedups of about 2.71x-5.96x over a hand-tuned raw-fused FP64 CUDA primitive. We additionally report deployment-locality accounting, one scoped scientific-field NCU physical-traffic case, synthetic H200 k-depth dispatch evidence, and an initial scalar/native BUFF-style CPU progressive proxy as a deployment sanity baseline. The paper explicitly does not claim a new bounded-float representation, a new Q/D/U predicate logic, a new deterministic SUM bound, an optimized BUFF baseline win, universal HBM-traffic reduction, universal scientific-analytics speedup, or DBMS-scale evidence.

## 2. Background and Related Work Positioning

BUFF is the closest representation ancestor. It establishes byte-oriented bounded-float columnar storage and progressive query behavior over floating-point data [BUFF]. This paper uses BUFF-style terminology deliberately. The byte-oriented bounded-float layout, progressive predicate decomposition, and deterministic omitted-tail SUM behavior are inherited lineage, not new contributions.

DAQ frames deterministic approximate query processing around bounded answers rather than random sampling error [DAQ]. Column Sketch exposes a related tri-state predicate structure in which rows can be definitely qualified, definitely disqualified, or unresolved under the current approximation [ColumnSketch]. This paper relies on that semantic intuition. Partial precision does not produce hidden exact answers; it produces bounded sums, definite classifications, and ambiguous rows.

ALP, FastLanes, G-ALP, and BtrBlocks operate at a different layer [ALP, FastLanes, GALP, BtrBlocks]. They improve floating-point storage, vectorized decoding, file/block organization, or compression/decompression efficiency. They are representation and storage neighbors, not controlled runtime baselines in this submission. We do not claim better compression ratio than these systems, and we do not claim runtime wins over them.

cuDF is a mature GPU dataframe stack [cuDF]. It is useful as an external anchor for raw FP64 dataframe-style execution, but it is not the primary primitive baseline. The primary baseline is raw-fused CUDA: a single H200 kernel over raw FP64 values that applies the threshold and accumulates COUNT and SUM without materializing masks or filtered survivor columns. This comparison isolates the value and cost of byte-plane execution from generic dataframe materialization effects.

### 2.1 Protection Allocation Sensitivity

When protection storage budget B (in replica-equivalent units) is constrained, allocation across the byte planes can be optimized per query functional. Define the sensitivity profile s_i(F) = U_i(F) / max_j U_j(F), where U_i(F) is the per-plane contribution ceiling for functional F, normalized so the most sensitive plane has s = 1.

For SUM and MAX, the MSB dominates: s_0 = 1.0, s_1 ≈ 0.0039, s_2 ≈ 1.5e-5, with lower planes negligible. For COUNT (filtered), sensitivity is threshold-local: only the plane that contains the threshold's byte representation has non-negligible s_i; planes far from the cutoff have near-zero sensitivity.

Uniform allocation is the flat-profile degenerate case: when s_i = 1 for all i, the allocation is uniform and independent of the functional. Byte-plane graded allocation admits a constrained integer programming formulation: minimize Σ_i |r_i − B · s_i / Σ_j s_j| subject to Σ_i r_i ≤ B, r_i ≥ 1, r_i ∈ ℤ. An optimal solver for this allocation is deferred to future work.

## 3. Problem Scope and Execution Model

The target workload is warm, device-resident, repeated threshold analysis over already prepared scientific fields. Warm E2E timing includes per-query threshold preparation when applicable and GPU execution over resident artifacts. It excludes cold artifact construction, initial H2D transfer, file I/O, and offline percentile selection unless a table explicitly states otherwise. A workflow that touches a field only once should not assume the reported speedups.

Execution depth k is the runtime knob. At shallow k, the operator reads leading byte planes and performs less reconstruction. At larger k, it reads more of the encoded value and performs more reconstruction. At k=max, the operator reaches the artifact's maximum filter depth. In the current progressive executor, k=max can still benefit from early exit and should not be described as a synthetic forced full-plane scan.

Representation precision, execution depth, query error, and physical traffic are separate axes. Representation precision describes the stored artifact. Execution depth k describes how many leading byte planes the operator reads. Query error describes approximation relative to the selected reference. Physical traffic describes what the GPU actually moves through the memory hierarchy, which must be measured using profiler counters rather than logical GB/s.

The paper uses the following baseline taxonomy.

| Evidence lane | Role in this paper | Forbidden interpretation |
| --- | --- | --- |
| Raw-fused FP64 CUDA | Primary same-H200 primitive runtime baseline | Full DBMS baseline. |
| cuDF | External mature-stack anchor | Primary baseline or proof that byte-plane alone causes all speedup. |
| BUFF | Representation and progressive query lineage | Runtime opponent or system beaten by this work. |
| DAQ / Column Sketch | Deterministic / ambiguous semantics lineage | New semantics introduced by this paper. |
| ALP / FastLanes / G-ALP / BtrBlocks | Related representation/storage work | Controlled runtime baselines or defeated systems. |
| CPU proxy | Scalar/native BUFF-style deployment sanity baseline | Optimized BUFF, AVX2, AVX-512, multithreaded, or official BUFF baseline. |

An optimized AVX2 / AVX-512 / multithreaded BUFF CPU baseline would be the most informative additional CPU comparison, but it is outside this workshop-scope submission. The current paper therefore does not claim to beat BUFF.

## 4. Inherited Semantics and Accuracy Reporting

The artifacts are BUFF-style bounded-float byte-plane data with segment metadata and active plane counts. In the current H200 scientific path, the key deployment distinction is locality. Fine-local artifacts divide a field into many segments and require many per-segment threshold decisions. True quasi-global artifacts use `segment_count=1`, making one segment cover the whole column. True quasi-global is a degenerate deployment choice within the inherited artifact family, not a new representation.

We do not introduce a new deterministic SUM bound, a new Q/D/U logic, or a new tri-state predicate decomposition. The bound formula and predicate decomposition we report are taken from BUFF and DAQ/Column-Sketch lineage, and our use of them is correctness checking and execution measurement on the inherited model.

For SUM-derived reductions, omitted low-order byte planes induce bounded execution-depth error relative to an encoded full-depth reference. Exp3 validates this inherited omitted-tail behavior across the current v2 precision-throughput matrix, with 103/103 bound-validation rows, but that is mechanism support rather than a new theory claim.

For COUNT, partial predicates can leave rows unresolved. At depth k, some rows are definitely qualified Q(k), some are definitely disqualified D(k), and some remain unresolved U(k), yielding an encoded-domain interval `COUNT in [Q(k), Q(k)+U(k)]`. This explains why shallow-k filtering can reduce work while preserving a deterministic answer contract, but it also explains why COUNT remains secondary. Encoded-domain correctness and raw-domain drift are separate claims.

## 5. Experimental Setup and Artifacts

### Validation Scope

The certified-bound and SDC-containment properties described in this work are deterministic properties of the byte-plane representation, validated by a CPU reference implementation on real datasets (cesm_atm_cloud, hurricane_u). The performance and economic claims are measured on real NVIDIA H200 hardware. This separation is intentional: correctness is representation-level and hardware-independent, while performance depends on the GPU memory hierarchy and execution model. The GPU detection and containment behavior was confirmed to match the CPU reference model on H200: an end-to-end GPU injection smoke test injected byte faults into device-resident artifacts, detected all faults via parallel SUM32 (detected_rate=1.0), and produced certified intervals containing the correct answer (contains_truth=1.0) across both tested datasets with representative planes spanning the significance range. The integer-encoded accumulator is design-level bit-for-bit; clean-run end-to-end verification was not separately performed.

Containment in this work relies on SUM32 detection combined with certified bound widening, not on majority voting from replicated storage. All results use a single replica (r=[1]*8).

**Scope notes.** Filtered-envelope certified-bound validation is scoped to hurricane_u (56/56 tested configurations). Cesm filtered-SUM results are reported separately. For cesm_atm_cloud, the cloud thresholds were computed on the non-zero subset of the distribution (clear-sky zero region excluded); zero rows are covered by the valid-ceiling invariant.

All official GPU measurements in this submission scope are H200 measurements unless explicitly labeled otherwise. The headline scientific evidence uses four tested true-quasi-global scalar fields:

- `cesm_atm_cloud`
- `hurricane_u`
- `cesm_atm_q`
- `hurricane_tc`

The first two fields come from the scientific locality attribution run. The latter two come from the corrected extra-field true-quasi-global run. `hurricane_tc` uses `bfp-dec11`, while the other headline fields use `bfp-dec12`; this precision difference is reported in the table and captions rather than hidden.

The paper-facing figures are stored under `results/paper_v1/figures_scoped_h200/`. Each figure has PNG/PDF outputs, source CSV snapshots, caption draft, source-data note, and claim boundary in the packet README.

## 6. Evaluation

Before reporting results, we restate the submission scope. All headline speedups below are warm device-resident operator timings. They exclude cold artifact construction, initial transfer, file I/O, and offline percentile selection. The scientific headline is limited to four tested fields from CESM/Hurricane and should not be read as universal scientific workload evidence.

### 6.1 Scientific headline: four-field true-quasi-global results

Figure 1 / Table 1 reports warm device-resident speedup over raw-fused FP64 CUDA for the four tested true-quasi-global fields. The table separates k=2 dispatch-selected shallow-depth evidence from k=max diagnostic/fallback rows. k=1 support-sweep rows are not included in the headline range.

| Field | Precision | k=2 shallow speedup range | k=max diagnostic/fallback speedup range | SUM behavior | COUNT caveat |
| --- | --- | ---: | ---: | --- | --- |
| `cesm_atm_cloud` | `bfp-dec12` | 3.40-4.70x | 1.49-2.98x | Below 0.01% relative error across selectivities at k=2. | Near-zero at s10/s50; about -1.2% at s90. |
| `hurricane_u` | `bfp-dec12` | 2.71-4.44x | 1.24-2.59x | Below 0.01% relative error at k=2; exact at k=max. | Near-zero at k=2; exact at k=max. |
| `cesm_atm_q` | `bfp-dec12` | 3.47-5.96x | 1.62-4.83x | Below 0.073% relative error at k=2; exact at k=max. | Below 0.008% at s10 and below 0.34% at s50 for k=2; exact at k=max. |
| `hurricane_tc` | `bfp-dec11` | 2.89-5.55x | 1.56-4.46x | Below 3e-5 relative error at k=2; exact at k=max=6. | Below 0.008% at k=2; exact at k=max=6. |

**Caption / source note.** Warm device-resident H200 speedup over raw-fused FP64 CUDA for four tested true-quasi-global CESM/Hurricane fields. k=2 rows are the primary shallow-k evidence; k=max rows are diagnostic/fallback evidence. `hurricane_tc` uses `bfp-dec11`. For `cesm_atm_q`, the high-selectivity row uses non-identical reported selectivity labels: 65.02% at k=2 versus 90.00% at k=max due to threshold-encoding drift. These should be compared as reported high-selectivity bucket evidence, not paired threshold evidence. COUNT error varies by field and selectivity and is secondary to the SUM-oriented headline.

The result supports a scoped claim: the operator can win over a strong same-GPU raw-fused CUDA primitive when locality and execution depth are favorable. It does not show universal scientific speedup, cold end-to-end acceleration, or general COUNT accuracy.

### 6.2 SUM error / speedup tradeoff

Figure 2 reports SUM precision-throughput mechanism evidence from the Exp3 v2 matrix. Relative SUM error is measured against encoded full-depth SUM, isolating execution-depth error from representation quantization. The figure separates representation precision p from execution depth k.

Progressive SUM-derived aggregates show rapid error decay with k. For heavy-tailed p2/p3/p4, k=2 remains around 1e-3 while k=3 reaches below 1e-4. The analytic execution-depth bound contains the measured error across the validated matrix. This supports the mechanism story behind SUM-derived shallow execution, but it is not a new SUM-bound theory and not the four-field scientific headline by itself.

COUNT is intentionally absent from this figure's headline. COUNT follows predicate ambiguity and raw-domain drift behavior rather than the same omitted-tail SUM error curve.

### 6.3 Threshold-preparation locality

Figure 3 reports threshold-preparation locality. Fine-local 4096-segment artifacts incur per-query CPU threshold-preparation costs that can dominate warm E2E time, reaching 7-19.42 ms on the CESM path. True-quasi-global artifacts reduce preparation to below 0.06 ms in the tested path.

This result is deployment accounting, not representation novelty. It explains why the same inherited artifact family can be useful or useless depending on segment locality. A byte-plane executor that saves GPU memory traffic can still lose end-to-end latency if host-side threshold preparation dominates the query.

### 6.4 Q8 scoped NCU physical-traffic attribution

Figure 4 reports one scoped scientific NCU case: `cesm_atm_q` at the s10 threshold on H200. In that case, the k=2 byte-plane fused path reads 186.49 MB through the DRAM proxy, compared with 1,382.40 MB for raw-fused FP64, a 7.41x reduction. L2 and L1/TEX sector counters move in the same direction, with about 7.29x and 5.75x reductions respectively.

The consistency across DRAM, L2, L1/TEX, and runtime direction makes Q8 useful as attribution evidence. It is still only one scientific-field NCU case. It does not prove universal physical HBM traffic reduction across fields, selectivities, or k values.

Logical GB/s is not physical bandwidth. Physical-traffic claims in this paper rely on NCU counters. Logical throughput is used only as workload-normalized rate.

### 6.5 Prompt5 synthetic k-depth break-even and fallback

Figure 5 reports synthetic H200 k-depth behavior over `uniform_p10` and `heavy_tailed_p6`. Prompt5 shows that physical DRAM traffic savings persist across k, but latency wins do not.

The dispatch rule is empirical:

- k=1 is the strongest measured region.
- k=2 is usable but marginal.
- k=3 is regime-dependent and unsafe for heavy-tailed cases.
- k=4 and deeper should be treated as fallback/diagnostic unless separately calibrated.
- Raw FP64 fallback is part of the design.

This evidence prevents the paper from claiming that byte-plane execution always wins. The supported conclusion is narrower: use byte-plane execution only for measured latency-winning shallow-k regimes and fall back to raw-fused FP64 past break-even.

### 6.6 CPU proxy and cuDF anchor

The CPU proxy is an initial scalar/native BUFF-style progressive implementation over the same artifacts. It is not optimized BUFF, not AVX2, not AVX-512, not multithreaded, and not an official BUFF baseline. Its role is deployment sanity checking, not a headline victory. The paper should not report CPU-vs-GPU ratios in the abstract or contribution list.

cuDF remains an external mature-stack anchor. Because dataframe APIs can materialize masks and survivor buffers, cuDF is not the primary primitive baseline for the main runtime claim. The raw-fused CUDA baseline is the correct primitive comparison point.

## 7. Discussion and Limitations

### Limitation: Adversarial Cancellation

The SUM32 modular checksum used for fault detection is structurally vulnerable to adversarial byte-pair cancellation within one allocation unit (segment). Because SUM32 is additive modulo 2^32, equal-and-opposite byte deltas within a segment leave the sum invariant, defeating detection.

In measured evaluations on hurricane_u, adversarial cancellation achieved a 100% escape rate (180/180 injection events; 95% Clopper-Pearson confidence interval [0.980, 1.0]). On cesm_atm_cloud, no cancelling byte pairs could be formed because the upper-plane bytes at the tested offset are uniformly zero, making the attack inapplicable rather than detected.

This is an adversarial worst-case result, not a random-fault rate. Under realistic correlated faults (cluster/burst), the escape rate was 0% across 960 injection events, with a 95% CP upper bound of 0.77%.

Two mitigation paths were evaluated on H200. First, digest-variant upgrades: a position-weighted accumulator (Σ i·b_i) or a Fletcher-like two-accumulator digest reduces the adversarial cancellation escape to 0% (0/9 on the SUM32-specific cancel set) while adding only 0.086 ms latency (10% of the B0 = 0.870 ms raw-fused FP64 baseline). By contrast, purely wider additive variants (SUM64, dual-SUM32) remain fully vulnerable, confirming that the structural weakness requires a non-additive combiner. Second, replication with majority voting (r=[3,2,1,1,1,1,1,1]) was tested but the same fault entries are applied to all replicas under the current single-fault-domain model, so identical cancellation pairs would need to affect two out of three replicas independently. True diverse placement across independent fault domains is not evaluated here.

This paper is intentionally narrow. It does not claim a new representation, new semantics, new deterministic bound, full DBMS integration, or universal speedup. The value is a measured H200 execution story over an inherited representation and semantic model.

Four scientific fields are enough for a workshop-scoped systems result, not enough for universal scientific workload generalization. Broader scientific coverage remains future work. Q8 is one scientific-field NCU case; broader physical-traffic coverage would strengthen a larger paper but is not necessary for this scoped submission. Prompt5 is synthetic dispatch evidence and must remain visually and rhetorically separate from the four-field scientific headline.

The workload is warm and resident. Cold artifact construction is not amortized in the reported numbers. A workflow that touches a field only once should not assume the reported speedups. The paper is about repeated threshold analysis over prepared artifacts, not ingestion-to-answer acceleration.

The optimized BUFF CPU baseline remains out of scope. The current CPU proxy is scalar/native. It is useful for deployment sanity and order-of-magnitude context, but it does not establish that the H200 executor beats an optimized BUFF implementation.

COUNT remains secondary. Encoded-domain predicate intervals, raw-domain drift, and shallow-k ambiguity must be reported as limitations. SUM-derived reductions are the safer scientific headline.

Finally, the dispatch policy is empirical, not a formal cost model. The paper reports measured H200 behavior under rowpack16 / byte-mask execution. A future system could learn or model dispatch more formally, but this submission should not claim that result.

## 8. Conclusion

This paper studies a narrow point in the compressed analytics design space: GPU-resident fused Filter+Aggregate execution over BUFF-style progressive byte-plane artifacts on NVIDIA H200. The representation and deterministic semantics are inherited from prior work; the contribution is the measured execution, locality, physical-traffic, and fallback behavior.

On four tested true-quasi-global CESM/Hurricane scalar fields, dispatch-selected k=2 shallow execution reports about 2.71x-5.96x warm device-resident speedup over a hand-tuned raw-fused FP64 CUDA primitive, while k=max rows remain diagnostic/fallback evidence. The 5.96x upper endpoint should be read with the `cesm_atm_q` high-selectivity bucket caveat described in §6.1, not as a paired-threshold k=2/k=max comparison. True-quasi-global locality keeps threshold preparation below 0.06 ms in the tested path, whereas fine-local 4096-segment artifacts can make threshold preparation dominate. Q8 provides one scoped scientific physical-traffic attribution case, and Prompt5 shows that physical traffic savings alone do not guarantee latency wins beyond shallow k.

The resulting systems lesson is conditional rather than universal: significance-ordered byte-plane artifacts can be useful GPU query substrates when the workload is warm and resident, when execution depth remains before empirical break-even, when locality avoids threshold-preparation domination, and when raw FP64 fallback is preserved.

## 9. Future Work

**Graded replication allocation.** The byte-plane representation admits graded protection, where different planes receive different replica counts based on their query-functional sensitivity (§2.1). This paper evaluates only a single-replica configuration (r=[1]*8). A graded allocation minimizing the total storage budget for a target containment probability is formulated as a constrained integer program but not solved; an optimal solver and experimental validation are deferred to future work.

**Spatial and temporal diverse placement.** Correlated faults within a single allocation unit can bypass additive detection (adversarial cancellation, §7). Spatial diversity (replicas placed on independently failing memory channels or DIMMs) and temporal diversity (repeated reads or votes over time) can reduce the probability that correlated faults affect all replicas simultaneously. These mechanisms are designed but not experimentally evaluated in this work.

**NMR majority voting.** NMR (N-modular redundancy) with majority voting on replicated byte-plane reads can recover from detection escapes, including adversarial cancellation. With r > 1 replicas per plane, the voter selects the majority accumulator; the per-plane containment probability approaches 1.0 as r increases. This mechanism is a natural extension of byte-plane graded storage but is not evaluated with measured data in the current study.

All reported certified-bound and SDC-containment results in this paper use a single replica (r=[1]*8) and obtain containment from SUM32 detection combined with bound widening, not from majority voting or diversity.

**MAX and top-k certified bounds.** The Z1a contract defines closed-form U_i ceilings for MAX and top-k functionals, but preliminary validation revealed that the active_count=1 assumption (Z1a §4) is insufficient for datasets with negative values. Identity-shift — where corruption promotes a different row to the maximum — requires tracking the second-largest value rather than inflating to n_rows, which would destroy bound usability. A correct MAX bound derivation and validation are deferred to future work.

## References To Resolve Before Submission

The final submission must replace these labels with verified venue-ready bibliography entries:

- [BUFF] BUFF / byte-oriented bounded-float compressed query processing paper.
- [DAQ] Deterministic approximate query processing / DAQ paper.
- [ColumnSketch] Column Sketch paper.
- [ALP] ALP floating-point compression paper.
- [FastLanes] FastLanes paper/system.
- [GALP] G-ALP or GPU/ALP-related floating-point compression work.
- [BtrBlocks] BtrBlocks paper/system.
- [cuDF] RAPIDS cuDF / libcudf documentation or paper reference.

## Submission Guardrails

Before converting this Markdown draft to the final template, keep these guardrails intact:

- Do not remove `four tested fields` qualifiers.
- Do not remove `warm device-resident` qualifiers.
- Do not report 1.2-5.96x without explaining it spans reported k=2 and k=max rows.
- Do not merge Prompt5 synthetic rows into the scientific headline.
- Do not present Q8 as universal traffic proof.
- Do not present COUNT as the headline result.
- Do not call the CPU proxy optimized BUFF.
- Do not say the paper introduces representation or semantics.
- Do not convert this into a full DBMS claim.