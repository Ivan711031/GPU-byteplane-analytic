# Mainline Paper Evidence Roadmap Spec

Date: 2026-05-12
Branch/worktree: `main` / `.worktrees/main`
Status: Spec only. No implementation in this file.

## 1. Decision

Stop the cuDF route for now. Treat it as mature-system baseline evidence and a
prepared-predicate side result, not as the next mainline task.

The next useful work is to make the existing mainline evidence paper-usable:

```text
bounded-p artifact
  -> execution depth k
  -> Q/D/U bound
  -> epsilon -> k*
  -> throughput
  -> bytes/transfer/traffic evidence
  -> raw and full-depth baselines
```

The most important gap is not another kernel optimization. The gap is that the
existing results are split across several CSVs and reports. The paper needs a
single claim table that joins raw COUNT, encoded full-depth COUNT, progressive
k*, throughput, representation drift, and transfer/bytes evidence.

## 2. Current Evidence State

Use only the `main` worktree state for this roadmap. Do not inspect or depend
on the `exp2` branch, even though the current `buff_encoder` lineage has changed.

### 2.1 Completed and usable

New bounded fixed-point artifact semantics:

- `research/2026-05-10_New_Encoder_Design_Note.md`
- `research/2026-05-10_New_Encoder_Semantics_Contract.md`

COUNT progressive sweep on v2 artifacts:

- `results/exp4/b1_20260510_174435_job44743_NVIDIAH200/sweep_summary.csv`
- `results/exp4/b1_20260510_174435_job44743_NVIDIAH200/count_epsilon_to_kstar.csv`
- report: `research/2026-05-10_Exp4B1_Issue38_Closure_Note.md`

Full-budget raw-validated COUNT anchor:

- `results/exp4/b1_20260510_190958_job45163_NVIDIAH200/sweep_summary.csv`
- report: `research/2026-05-10_Exp4B1_Issue36_Closure_Note.md`

Raw FP64 COUNT direct comparator:

- `results/raw_count_baseline/run_20260506_153627_job39205_H200/raw_fp64_count_baseline.csv`
- report: `research/2026-05-06_Raw_FP64_COUNT_Comparator_Initial_Report.md`

Artifact size, fidelity, and H2D transfer:

- `results/paper_v1/artifact_size_fidelity_transfer.csv`
- `results/paper_v1/raw_fp64_transfer_baseline.csv`
- report: `research/2026-05-10_Artifact_Size_Fidelity_Transfer_Report.md`

Existing figure packs:

- `research/2026-05-10_Exp4_v2_Paper_Figure_Pack_Report.md`
- `research/2026-05-10_Exp4_v2_Claim_Figure_Report.md`
- `results/paper_v1/plots_exp4_v2/`
- `results/paper_v1/plots_exp4_v2_claims/`

Raw FP64 SUM comparator exists, but this roadmap focuses on COUNT first:

- `research/2026-05-01_PV1-6_Raw_FP64_SUM_Comparator_Report.md`

### 2.2 Still missing or weak

1. No single raw-vs-encoded COUNT claim table joins:
   - raw FP64 COUNT runtime,
   - full-depth encoded COUNT,
   - progressive k* COUNT,
   - representation fidelity drift,
   - transfer bytes/time.

2. No final paper-facing decision table that says, per artifact/selectivity:
   - supported,
   - conditional,
   - unsupported,
   - reason.

3. Nsight Compute physical traffic is still blocked:
   - `research/2026-05-11_Exp4B1_NCU_Physical_Traffic_Audit_Status.md`
   - no paper claim should currently say measured Exp4-B1 DRAM/L2/global-load
     traffic.

4. Existing plots are useful, but several are still diagnostics. The paper needs
   a small number of claim tables/figures derived from the final joined evidence.

## 3. Follow-Up Work Packages

### WP1: Raw-vs-Encoded COUNT Claim Table

Spec:

- `spec/2026-05-12_Raw_vs_Encoded_COUNT_Claim_Table_Spec.md`

Purpose:

Join existing CSVs. Do not run a new H200 job. Produce:

- `results/paper_v1/raw_vs_encoded_count_comparison.csv`
- `research/2026-05-12_Raw_vs_Encoded_COUNT_Claim_Table_Report.md`

This is the highest-priority next task because it directly decides whether the
paper can say anything stronger than "progressive improves over full-depth
encoded execution."

### WP2: Physical-Traffic Proxy Evidence Pack

Spec:

- `spec/2026-05-12_Exp4B1_Physical_Traffic_Proxy_Evidence_Spec.md`

Purpose:

Since fresh NCU is blocked, build a conservative evidence pack from existing
payload bytes, estimated pack-load bytes, H2D transfer timing, and kernel timing.
It must explicitly label itself as a proxy, not measured DRAM/L2 evidence.

Produce:

- `results/paper_v1/exp4_b1_physical_traffic_proxy.csv`
- `research/2026-05-12_Exp4B1_Physical_Traffic_Proxy_Evidence_Report.md`

This helps the paper say "logical bytes and transfer payload scale as expected"
without overclaiming "measured HBM traffic."

### WP3: Final Figure/Table Cleanup

Do this only after WP1 and WP2.

Expected output:

- one table for raw-vs-encoded COUNT,
- one table for artifact suitability,
- one figure for `epsilon -> k*`,
- one figure for speedup at `k*`,
- one figure/table for bytes/transfer proxy.

This can be handled by a smaller model after the joined CSVs exist.

### WP4: External Baseline Feasibility

Defer until after WP1/WP2.

Candidates:

- G-ALP / FastLanes GPU: closest GPU float codec, but not progressive bounded
  query execution.
- cuSZ / ZFP CUDA: decompress-then-query baselines.

Do not let these block the main paper chain. They are optional related-work
baselines, not the current highest-value task.

## 4. Claim Boundary For The Paper

Supported by current evidence:

1. Bounded fixed-point artifacts have explicit monotonic truncation semantics.
2. COUNT has separate representation error and execution error.
3. Q/D/U gives a deterministic encoded-domain COUNT interval:
   `COUNT_full_encoded in [Q(k), Q(k)+U(k)]`.
4. `epsilon -> k*` can be computed from the Q/D/U curve.
5. Smaller encoded payloads reduce H2D transfer time.
6. Progressive execution can improve throughput at relaxed epsilon versus
   full-depth encoded execution.

Not yet supported until WP1:

1. Faster-than-raw COUNT on the final bounded-p matrix.
2. A clean table showing when progressive encoded COUNT beats raw FP64 COUNT.
3. A final per-artifact paper verdict that combines fidelity, compression,
   transfer, and progressive speed.

Not supported until NCU is fixed:

1. Measured Exp4-B1 DRAM traffic reduction.
2. Measured L2/global-load traffic reduction.
3. HBM bandwidth utilization claims for Exp4-B1.

## 5. Agent Assignment Guidance

WP1 can be assigned to `gpt-5.4-mini` if it is limited to CSV joins, checks, and
report writing. Escalate to `gpt-5.4 high` only if the join exposes semantic
ambiguities in thresholds or selectivity labels.

WP2 can also be assigned to `gpt-5.4-mini` because it is an analysis/report task,
not a CUDA implementation task.

Do not assign a GPU benchmark job for these two specs unless the report proves an
input CSV is missing or unusable.

## 6. Stop Conditions

Stop and report, rather than improvising, if:

1. a required input CSV is missing,
2. joins cannot be made one-to-one,
3. raw baseline selectivity labels cannot be matched to encoded thresholds,
4. output would require claiming measured physical traffic without NCU evidence,
5. the task would require inspecting the `exp2` branch or changing the encoder.

## 7. Expected Next Decision

After WP1:

- If progressive at `k*` beats raw FP64 COUNT on the mainline artifacts with
  acceptable fidelity, the paper can include a conditional faster-than-raw
  COUNT result.
- If it does not, the paper should frame COUNT primarily as a deterministic
  precision-throughput operator and compare speedup against full-depth encoded
  execution, not raw FP64.

After WP2:

- If proxy bytes and timing show monotonic scaling, use them as supporting
  systems evidence.
- Keep the NCU gap as a limitation until the platform/tooling blocker is solved.

