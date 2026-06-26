# Fail-Soft Byte-Plane Analytics: Progressive GPU Execution with Certified Bounded Degradation

**Status:** ACM RACS 2026 submission-facing Markdown draft.  
**Detector resolution used in this revision:** fallback contract — every reliability guarantee is conditional on successful detection. SUM32 is the evaluated lightweight detector and has known cancellation limits.

## Abstract

Repeated scientific post-analysis scans floating-point fields with threshold-conditioned reductions. A conventional FP64 buffer exposes neither a natural runtime depth knob nor a representation-derived response when corruption is detected but cannot be repaired. We study significance-ordered byte planes as one GPU primitive with two contracts. The execution contract reads only the leading `k` planes and falls back to raw FP64 when calibrated measurements do not justify progressive execution. The reliability contract uses the same per-plane contribution ceilings to recover a detected corruption when valid replicas exist, or otherwise return a certified interval instead of an unchecked scalar.

On four CESM/Hurricane fields resident on an NVIDIA H200, warm `k=2` fused Filter+Aggregate is **2.69×–6.14× faster** than a raw-fused FP64 CUDA primitive; the 6.14× endpoint is a non-paired high-selectivity bucket and is treated as scoped bucket evidence. Two-field profiling attributes the mechanism to about **7.2× lower DRAM reads** at `k=2`. Under CPU software fault injection over two real artifacts, the single-replica detect-and-bound path converts all 24 tested injection cells into truth-containing intervals. A separate pre-fused H200 NMR path agrees with a CPU oracle on all **210/210** tested structured-fault classifications, with 25.7% recovered and 74.3% bounded-degraded. A later HBM-inspired structured campaign over **8,800** evaluations shows that graded and uniform have identical zero silent-wrong and zero uncertified outcomes under SUM32, while graded delivers dramatically tighter certified bounds at matched storage. At matched `E=3` extra storage, significance-aware allocation yields **345×–432× narrower certified interval width** than uniform spread. A fused NMR-C v2 top-`k` extension shows a scoped point-answer win for graded on `cesm_atm_cloud` at matched realized storage, but no analogous separation on `hurricane_u`. Full unfused NMR costs **2.47×–2.49× raw FP64**, establishing a negative cost boundary rather than a low-overhead claim. All reliability results use software/logical faults on normal hardware and are conditional on successful detection.

## 1. Introduction

Scientific simulations and observational pipelines produce floating-point fields that are scanned repeatedly during post-analysis. Typical questions apply a threshold and then compute a SUM, COUNT, or derived average. When an artifact is already prepared and resident on a GPU, this is fundamentally a memory-traffic problem: the operator repeatedly reads the same values, often with less precision than the stored representation provides.

A fixed FP64 buffer hides two useful structures. First, it offers no native execution depth: the operator reads complete values even when leading significance components are sufficient for the required answer quality. Second, after a corruption is detected, the raw representation does not expose a fixed relationship between the affected byte position and its maximum contribution to the aggregate. Standard protection mechanisms—ECC, checksums, replication, recomputation, and checkpoint/restart—remain valid, but without a clean copy they normally abort or retry rather than return a representation-derived degraded answer.

A significance-ordered byte-plane artifact exposes both structures. Plane `0` contains the most significant byte of every encoded value, plane `1` the next byte, and so on. The same per-plane contribution ceiling answers two questions:

1. How much uncertainty remains if execution stops after the leading `k` planes?
2. How far can a detected corruption in a known set of planes move the aggregate?

We use this shared invariant to build a **dual-contract** system. In the normal case, the runtime selects depth `k` and executes a fused threshold Filter+Aggregate over leading planes. In the fault path, a detector localizes affected planes, the system attempts replica recovery where valid copies exist, and otherwise adds the affected planes' contribution ceilings to a certified interval. If the detector, localization metadata, or bound preconditions are unavailable, the system returns UNAVAILABLE rather than an unchecked scalar.

The paper answers four research questions:

- **RQ1:** Does progressive byte-plane execution provide a useful H200 operating region?
- **RQ2:** Under detected software/logical faults, does the system recover or return a truth-containing certified interval?
- **RQ3:** At matched storage, does significance-aware protection allocation reduce certified uncertainty relative to uniform spread?
- **RQ4:** What latency and storage boundary prevents current full replication from being claimed as low-overhead?

The evaluated operator is:

```sql
SELECT COUNT(*), SUM(x), SUM(x) / COUNT(*)
FROM field
WHERE x > threshold;
```

SUM is the primary scientific result. COUNT is produced by the fused kernel but remains secondary because shallow partial precision can leave predicates qualified, disqualified, or unresolved.

**Contributions.** This paper makes four scoped contributions.

1. **A representation-derived dual contract.** One significance ordering supplies both an execution-depth bound and a detected-fault answer-impact radius.
2. **A fail-soft response path.** The system distinguishes EXACT, RECOVERED, BOUNDED, and UNAVAILABLE outcomes, with certified radius semantics audited and unit-tested across all evaluator sites used by the paper.
3. **A significance-aware protection policy.** Under matched `E=3` extra storage, protecting high-impact planes first reduces certified interval width by 345×–432× relative to uniform spread.
4. **A measured operating region and cost boundary.** Four-field H200 execution, two-field NCU attribution, locality and `k`-depth calibration establish when the normal path is useful, while a measured 2.47×–2.49× full-NMR ratio shows why full replication is not the default path.

## 2. Background and Related Work

**Progressive floating-point representations.** BUFF is the closest representation ancestor: byte-oriented bounded-float storage supports progressive predicate and aggregate processing [BUFF]. We inherit the significance-ordered layout and deterministic omitted-tail reasoning. DAQ motivates deterministic approximate answers expressed as intervals rather than sampling confidence [DAQ]. Column Sketch supplies the qualified/disqualified/unresolved vocabulary for partial predicate evaluation [ColumnSketch]. Our contribution is not a new byte layout; it is the systems composition of normal-path depth control and detected-fault degradation through one contribution map.

**Compression and GPU analytics.** ALP, FastLanes, G-ALP, and BtrBlocks target floating-point compression, decoding, block organization, or vectorized execution [ALP, FastLanes, GALP, BtrBlocks]. cuDF provides a mature GPU dataframe stack [cuDF]. We do not claim compression-ratio or general runtime superiority over these systems. The primary execution comparator is a raw-fused FP64 CUDA kernel that applies the threshold and accumulates COUNT and SUM without materializing masks or survivors.

**Resilient execution.** GPU reliability mechanisms include ECC, page retirement, XID/DCGM telemetry, replication, recomputation, and checkpoint/restart [NVIDIAPageRetirement, NVIDIAXid, NVIDIADCGM]. Field studies and fault-injection work motivate software-level response testing [AmpereMemErrors, HardDataSoftErrors, GPUFaultInjection]. These mechanisms can compose with byte planes. Our narrower distinction is the response after detection when repair is unavailable: significance ordering supplies a conservative answer-impact radius instead of forcing immediate output loss.

## 3. System Design

### 3.1 Representation and execution contract

Let encoded plane `i` have positional weight `w_i`, scale `s`, and at most `A` active rows for functional `F`. A conservative contribution ceiling is

```text
U_i(F, A) = A · 255 · w_i / s.
```

The exact active-count definition depends on the operator: all rows for an unfiltered SUM, or qualified plus unresolved rows for a thresholded partial execution. Because `w_i` decreases by approximately 256× per byte plane, the leading planes dominate both answer value and worst-case uncertainty.

At runtime the dispatcher selects depth `k`. The fused kernel reads the leading planes needed for predicate evaluation and SUM reconstruction. The unread tail contributes an execution uncertainty radius `R_exec(k)` derived from the corresponding `U_i` values. `k` is therefore a runtime policy knob, not a storage precision setting. Representation precision, execution depth, answer error, and physical memory traffic remain separate quantities.

The dispatcher is empirical. It uses calibrated field/selectivity/depth measurements and retains raw FP64 fallback. The design does not assume that every query or every `k` benefits from byte-plane execution.

### 3.2 Detector-parametric fault contract

The reliability contract begins only after successful detection. The evaluated paths use per-plane or per-allocation-unit SUM32 metadata, defined as a modular sum over raw bytes. SUM32 is lightweight but additive: equal-and-opposite byte changes can cancel. Consequently, this paper does not claim universal detection. Detector construction is an orthogonal interface, and every recovery-or-bound guarantee is conditioned on a detector mismatch that correctly identifies the affected plane set `P`.

For detected planes `P`, the fault-impact radius is

```text
R_fault(P) = Σ_{i ∈ P} U_i(F, A_i).
```

The returned interval uses **radius semantics**:

```text
R_total = R_exec(k) + R_fault(P)
certified interval = [delivered − R_total, delivered + R_total].
```

`R_total` is a one-sided displacement. It is not a full width to be divided by two. A dedicated math audit corrected every evaluator used by the paper that had mixed these conventions. For filtered SUM/COUNT, the tight implementation uses `quantization_full_width/2 + fault_widening`; other evaluated consumers conservatively use a full quantization width as a radius.

### 3.3 State machine

Figure 1 is generated from the accompanying architecture source.

1. **EXACT.** No mismatch is observed and the result satisfies the normal execution contract at selected depth `k`. This is exact with respect to that selected-`k` contract, not necessarily bit-identical to raw FP64 at shallow depth. A detector collision is an undetected, out-of-contract event.
2. **RECOVERED.** A mismatch is localized, replicas exist, majority voting constructs a candidate, and the candidate passes stored integrity metadata.
3. **BOUNDED.** Repair is unavailable or fails validation, but the affected planes and contribution metadata are sufficient to construct `delivered ± R_total`.
4. **UNAVAILABLE.** Required detector, localization, or bound metadata is absent or outside the supported contract.

The system does not compare against clean truth at runtime. Clean truth is used only by the evaluator to verify that the constructed interval contains it.

### 3.4 Protection policies

The system separates three decisions.

- **How deeply to execute:** choose `k` or raw FP64 fallback.
- **Where to spend replica storage:** allocate extra copies to planes/segments with the largest contribution ceilings.
- **Whether recovery is valid:** accept a replica vote only when the candidate satisfies integrity metadata; otherwise fall back to bounding.

Full NMR is therefore one policy point, not the paper identity. The default fail-soft path can operate with one copy per plane and return a certified interval after detection. Selective protection spends extra storage where it most reduces that interval. Full replication is measured separately as a cost boundary.

## 4. Methodology and Scope

### 4.1 Hardware, fields, and execution protocol

All official GPU latency and traffic measurements use one NVIDIA H200. The four execution fields are real quasi-global CESM/Hurricane scalar artifacts:

| Field | Rows | Encoding | Approximate artifact scale |
|---|---:|---|---:|
| `cesm_atm_cloud` | 168.48M | `bfp-dec12` | ~1.2–1.3 GB |
| `cesm_atm_q` | 168.48M | `bfp-dec12` | ~1.2 GB |
| `hurricane_u` | 25M | `bfp-dec12` | ~0.2 GB |
| `hurricane_tc` | 25M | `bfp-dec11` | ~0.2 GB |

Warm timing includes per-query threshold preparation when applicable and GPU execution over resident artifacts. It excludes offline encoding, file I/O, initial host-to-device transfer, and cold artifact construction. The largest tested artifact is far below the H200's full memory capacity; this is an operator study, not a capacity-scale or multi-tenant benchmark.

### 4.2 Evidence classes

The paper labels each result by execution engine rather than treating all jobs submitted to a GPU partition as GPU evidence.

| Evidence | Engine / method | Scope used in this paper |
|---|---|---|
| Four-field latency, NCU, `k` dispatch | H200 CUDA measurements | Normal-path operating region |
| Single-replica 24-cell containment | CPU software-injection evaluator over real byte-plane artifacts + mathematical ceiling | Headline detect-and-bound contract |
| Structured 210-case campaign | H200 pre-fused NMR path with CPU oracle, 500K-row slices | Recovery-or-bound implementation support |
| HBM-inspired realistic structured campaign | Frozen structured logical replay over real artifacts | Detector-conditioned safety parity + bound-tightness refinement |
| Matched-storage graded vs uniform | CPU/NumPy logical policy evaluation | Protection-allocation objective only |
| NMR-C v2 `k`-aware frontier | H200 fused-kernel measurement with CPU truth reference | Scoped point-answer extension |
| Full-NMR latency | H200 measurement | Negative cost boundary |
| Logical voting controls | CPU/NumPy logical model | Appendix mechanism controls only |

Faults are software-injected byte/plane corruptions. The experiments do not characterize real HBM fault frequencies, ECC behavior, physical channel/bank/SID placement, or aged-device behavior.

### 4.3 Metrics and provenance

Execution uses warm latency, speedup over raw-fused FP64, and profiler-reported DRAM reads. Reliability uses detection status, recovery classification, certified interval construction, truth containment in the evaluator, and UNAVAILABLE/silent-wrong counts. The allocation study reports normalized **total certified interval width**; it is not observed point-answer error. The scoped NMR-C v2 extension additionally reports `error_to_truth` and `error_vs_clean_k` for prefix queries that read only the leading `k` planes. Full-NMR cost is the measured ratio to raw FP64 for the current unfused vote/read-compare path.

Canonical evidence and reproduction metadata are maintained in the accompanying artifact package. Internal issue, branch, job, and commit identifiers are intentionally excluded from the anonymous manuscript. The paper uses no temporal/ST error-reduction percentage from the separate logical diversity matrix.

## 5. Evaluation

### 5.1 RQ1: H200 execution operating region

Table 1 summarizes warm `k=2` execution over the four real fields.

| Field | `k=2` speedup vs raw FP64 | `k=max` diagnostic range | SUM reading | COUNT reading |
|---|---:|---:|---|---|
| `cesm_atm_cloud` | 3.64×–4.92× | 1.53×–3.13× | low relative error; exact in tested rows | secondary; drift at high selectivity |
| `hurricane_u` | 2.69×–3.90× | 1.24×–2.05× | low relative error; exact in tested rows | near-zero drift in tested rows |
| `cesm_atm_q` | 3.58×–6.14× | 1.66×–4.96× | machine-epsilon scale in freeze | large high-selectivity drift |
| `hurricane_tc` | 2.94×–5.72× | 1.58×–4.57× | below `3e-5%` relative | exact at `k=max=6` |

The supported headline is **2.69×–6.14×**, conditional on warm residency, quasi-global locality, shallow depth, and SUM-primary interpretation. The 6.14× endpoint is a `cesm_atm_q` high-selectivity bucket whose `k=2` and `k=max` selectivity labels are not a paired-threshold comparison; it is retained as scoped bucket evidence, not a universal endpoint.

**Traffic mechanism.** At the `s10` threshold on `cesm_atm_q` and `hurricane_tc`, the `k=2` path reads about 0.135× and 0.139× the DRAM bytes of raw FP64, or roughly **7.2× less DRAM traffic**. L2 and L1/TEX counters move in the same direction. This is two-field attribution, not a universal traffic law.

**Locality boundary.** Fine 4096-segment layouts were rejected as the headline mode because host-side threshold preparation can dominate the GPU operator. The final execution claims use quasi-global single-segment artifacts. Fine locality may be useful only when preparation is amortized.

**Depth calibration.** A 200-iteration warm-cache synthetic microbenchmark shows the speedup declining as more planes are read:

| `k` | `uniform_p10` | `heavy_tailed_p6` |
|---:|---:|---:|
| 1 | 4.19× | 4.46× |
| 2 | 2.85× | 2.81× |
| 3 | 2.36× | 2.12× |
| 4 | 2.02× | 1.70× |
| max | 1.56× | 1.27× |

The policy is therefore empirical: shallow `k` is the primary operating region; raw FP64 remains the fallback when calibration does not justify byte-plane execution.

### 5.2 RQ2: Recovery or certified bounding after detection

**Single-replica headline.** The conservative spine uses `r=[1,1,1,1,1,1,1,1]`, so no voting is possible. A CPU evaluator loads real `hurricane_u` and `cesm_atm_cloud` byte-plane artifacts, injects random byte faults, compares per-plane SUM32 metadata, and widens by `255 · w_i · n_rows` for each detected plane. Across two datasets, three rates, three seeds, and same-fault-all controls, all **24/24** tested cells were detected and classified BOUNDED; all certified intervals contained the clean answer, with zero silent-wrong and zero UNAVAILABLE outcomes.

The important property is mathematical, not the number 24: once affected planes are correctly detected and localized, the contribution ceiling covers every possible byte value in those planes. The 24-cell matrix validates the evaluator and outcome wiring under the tested random injections. It does not prove SUM32 detects adversarial cancellation patterns.

**H200 structured path.** A separate pre-fused H200 pipeline executes `replicate → vote → digest/detect → certified bound → classify` on 500K-row slices from the two real fields. Seven structured software-fault families, five policies, and three seeds produce **210 configurations**. GPU classifications agree with the CPU oracle in **210/210** cases; truth containment is 210/210; there are no silent-wrong, false-recovery, certified-bound-failure, or hard-fail outcomes.

The outcome split is:

| Outcome | Share |
|---|---:|
| RECOVERED | 25.7% |
| BOUNDED | 74.3% |
| UNAVAILABLE / silent wrong | 0% |

This result supports **recovery-or-bound**, not a high repair-rate claim. Same-fault-all controls are never falsely recovered: identical corruption across replicas falls through to certified bounding.

**Structured realistic detector line.** A later HBM-inspired campaign replays frozen structured fault plans across 20 policies, eight fault families, two real datasets, two rate anchors, and 10 seeds, yielding **8,800** evaluations. All policies retain `silent_wrong_rate=0` and `uncertified_rate=0`; 29.5% of evaluations are `exact_correct` and 70.5% are `certified_degraded`. This strengthens the RQ2 interpretation: under the detector-conditioned contract, the main policy discriminator is not binary safety versus failure, but the quality of the certified answer returned after detection.

### 5.3 RQ3: Significance-aware allocation at matched storage

The allocation experiment compares two segment-level policies at matched `E=3` extra storage, or 11 total plane-replica units over eight baseline planes.

- The **graded** policy protects plane 0 at `r=3`, protects approximately half of plane 1 segments at `r=3`, and leaves planes 2–7 at `r=1`.
- The **uniform-spread** policy distributes `r=3` over approximately 18.75% of segments in every plane.

Both policies consume the same storage. They are similar for repair coverage and point-answer behavior; the discriminating metric is certified interval width after unrepaired corruption.

| Dataset | Fault rate | Uniform / graded width ratio |
|---|---:|---:|
| `hurricane_u` | `2e-5` | ~345× |
| `hurricane_u` | `2e-4` | ~432× |
| `cesm_atm_cloud` | `2e-5` | ~348× |
| `cesm_atm_cloud` | `2e-4` | ~419× |

Across the four dataset-rate aggregates, significance-aware allocation yields **345×–432× narrower certified interval width**. Per-seed results remain within the broader 256×–445× envelope. This is a policy result about where to spend protection storage. It does not establish point-accuracy superiority, higher total repair coverage, physical HBM robustness, or an optimal allocation solver.

**Structured realistic refinement.** The later 8,800-evaluation HBM-inspired campaign refines this policy interpretation under a broader structured-fault spectrum. Graded and uniform again tie on detector-facing safety outcomes (`silent_wrong_rate=0`, `uncertified_rate=0` for all 20 policies), but at matched budget `B=8` the median certified bound width is approximately zero for `graded_B8` and about `5.5e18` for `uniform_full_r2` on both `hurricane_u` and `cesm_atm_cloud`. The policy advantage is therefore fail-soft answer quality after detection, not better detector hit rate.

**Scoped fused top-`k` extension.** A later NMR-C v2 fused-kernel pilot evaluates point-answer quality for SUM queries that read only the leading `k` planes. On `cesm_atm_cloud`, the clean matched mid-budget comparison is `graded_B7` (`B=7`) versus `uniform_full_r2` (`B=7`), and graded is consistently closer to `clean_k_answer` across `k=1,2,4,7`. On `hurricane_u`, the clean matched pairs at `B=0` and `B=16` tie, while the apparent mid-budget separation is not a fair matched-storage comparison (`B=7` versus `B=8`) and is further masked by large encoding approximation. We therefore treat this as a **scoped point-answer extension** on one dataset, not as a universal replacement for the certified-width objective above.

### 5.4 RQ4: Full-replication cost boundary

The current unfused full-NMR vote/read-compare path measures **2.47× raw FP64** on `hurricane_u` and **2.49× raw FP64** on `cesm_atm_cloud`. This is approximately **2.5× raw FP64** on both datasets.

The result is intentionally negative. Voting and replica reads are not free, and the current implementation is not a low-overhead deployment path. The paper therefore keeps single-replica detection and selective protection as first-class policy points, retains raw FP64 fallback, and treats fused vote+digest optimization as future work rather than rewriting 2.5× as a success. The later NMR-C v2 top-`k` extension does not overturn this boundary: it studies selective mid-budget protection on a different prefix-query contract, not cheap full-`r=3` voting.

## 6. Discussion and Limitations

**Detector precondition.** The response contract is detector-parametric. Most evaluated evidence uses SUM32, which detects the tested random and structured injections but has known additive cancellation escapes. Unless a stronger detector result is accepted before submission, the conservative statement remains: recovery-or-bound is guaranteed only after successful detection and localization. Detector collision is an undetected, out-of-contract event.

**Detector-conditioned interpretation of graded gains.** Across the later realistic structured campaign, graded does not improve silent-wrong or uncertified rates relative to uniform under SUM32; its consistent benefit is tighter certified bounds after detection. Claims of graded superiority in this paper are therefore claims about fail-soft answer quality under the detector-conditioned contract, not universal detection superiority.

**Hardware scope.** All latency and traffic results use normal H200 hardware. Reliability faults are injected in software. The paper does not evaluate aged or faulty HBM, ECC telemetry-driven placement, physical channel/bank/SID separation, row hammer, retention faults, multi-GPU operation, concurrent workloads, or long-duration behavior.

**Evidence separation.** The single-replica contract is a CPU logical evaluator plus a mathematical ceiling over real artifacts. The structured recovery path is exercised on H200 and compared with a CPU oracle. The realistic structured campaign refines the detector-conditioned policy story. The allocation result is a CPU/NumPy policy evaluation, and the NMR-C v2 top-`k` pilot is a scoped fused-kernel point-answer extension. These evidence classes answer different questions and are not merged into a single “H200 reliability” claim.

**Contract boundary.** The paper's main reliability claims are detector-conditioned. We do not generalize them to detector-free replication-only semantics; removing the detector changes the contract being studied and can change which policy looks preferable.

**Operator and scale scope.** The implementation covers a single-column thresholded Filter+Aggregate with SUM primary. COUNT is secondary; AVG is derived. JOIN, GROUP BY, MIN, MAX, VAR, DBMS integration, full-capacity HBM scale, and cold pipeline costs are outside the evaluated scope.

**Generality.** Four scientific fields support a scoped systems result, not universal workload generality. The NCU mechanism is shown on two fields. The depth sweep is synthetic. The NMR-C v2 top-`k` point-answer extension is dataset-scoped: `cesm_atm_cloud` supports a graded mid-budget advantage, whereas `hurricane_u` does not provide a clean separation. An optimized full-NMR implementation could differ from the measured 2.5× boundary.

**Diversity controls.** A separate CPU/NumPy logical matrix shows that R=3 majority voting recovers tested single-replica faults and that same-fault-all cases receive no false recovery credit. Temporal/ST error-reduction results from that matrix are omitted: the temporal behavior is encoded into the generator, the naive baseline is not a fair independent-fault comparator, and the reported relative-error regime is not meaningful for a main-body answer-quality claim.

## 7. Conclusion

Significance-ordered byte planes expose a systems invariant that fixed FP64 buffers hide. The same per-plane contribution ceiling controls how deeply a GPU operator must execute and how far detected corruption can move an aggregate. This yields a dual-contract primitive: progressive execution in the normal case, and recovery or certified bounded degradation after successful detection.

On four real CESM/Hurricane fields, warm H200 `k=2` execution is 2.69×–6.14× faster than raw-fused FP64 under the measured locality and depth conditions, with two-field profiling showing about 7.2× lower DRAM reads. The single-replica path returns truth-containing intervals in all 24 tested logical-injection cells, while the pre-fused H200 structured path matches a CPU oracle in 210/210 configurations. At matched storage, significance-aware protection reduces certified interval width by 345×–432× relative to uniform spread; a later 8,800-evaluation realistic campaign shows that this policy advantage appears as dramatically tighter certified bounds under structured detected faults even when graded and uniform are safety-equivalent on `silent_wrong` and `uncertified` rates. A fused NMR-C v2 top-`k` extension further gives a scoped point-answer win on `cesm_atm_cloud` only. Full NMR remains approximately 2.5× raw FP64, so the result is a cost boundary rather than a low-overhead claim.

The paper does not claim universal detection, physical HBM validation, temporal hardware independence, or production-ready NMR. It establishes a fail-soft representation and response contract whose normal execution and degraded-answer semantics are governed by the same significance ordering.

## Appendix A. Scoped NMR Extensions and Logical Voting Controls

The retained NMR-side controls and extensions are:

- **Single-replica positive control:** majority voting recovers the tested faults that affect only one of three replicas.
- **Same-fault-all negative control:** identical corruption across all replicas receives zero false recovery credit.
- **Structured realistic refinement:** under the detector-conditioned SUM32 contract, a later 8,800-evaluation campaign shows `silent_wrong_rate=0` and `uncertified_rate=0` for all 20 policies, while graded's advantage appears in certified bound tightness rather than detector hit rate.
- **Scoped fused top-`k` extension:** on `cesm_atm_cloud`, a later NMR-C v2 pilot shows that matched-storage graded mid-budget protection can preserve the `clean_k` answer better than uniform when only a prefix of planes is read; this does not generalize to `hurricane_u`.

These controls and extensions refine the NMR story without changing the paper identity. Temporal/ST error-reduction percentages are excluded from the abstract, contributions, RQs, figures, and conclusion.

## References

- [BUFF] Liu et al. BUFF: byte-oriented bounded-float compressed query processing. PVLDB, 2021.
- [DAQ] Potti and Patel. DAQ: deterministic approximate query processing over bit-sliced indexes. SIGMOD, 2015.
- [ColumnSketch] Hentschel et al. Column Sketches: a scan accelerator for predicate evaluation. SIGMOD, 2018.
- [ALP] Afroozeh, Kuffo, and Boncz. ALP: Adaptive Lossless Floating-Point Compression. SIGMOD, 2024.
- [FastLanes] Afroozeh, Felius, and Boncz. Accelerating GPU Data Processing using FastLanes Compression. DaMoN, 2024.
- [GALP] Hepkema et al. G-ALP: GPU-oriented ALP and FastLanes floating-point compression. DaMoN, 2025.
- [BtrBlocks] BtrBlocks: Efficient Columnar Compression for Data Lakes. SIGMOD, 2023.
- [cuDF] RAPIDS cuDF / libcudf documentation.
- [NVIDIAPageRetirement] NVIDIA Dynamic Page Retirement documentation.
- [NVIDIAXid] NVIDIA Xid Errors documentation.
- [NVIDIADCGM] NVIDIA DCGM Field Identifiers documentation.
- [AmpereMemErrors] Zhu et al. Large-scale GPU memory-error characterization on Ampere-class accelerators.
- [HardDataSoftErrors] GPU memory reliability and hard-data/soft-error characterization literature.
- [GPUFaultInjection] GPU software fault-injection methodology literature.
