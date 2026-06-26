# Mainline Precision-Throughput Evidence Pack Spec

Date: 2026-05-12
Scope: Recenter the project on epsilon-to-k and precision-throughput curves.
Status: Analysis/report spec. No new CUDA kernels unless a concrete evidence gap is found.

## 1. Why This Spec Exists

The cuDF/RAPIDS primitive route is now closed as a side result:

- prepared compact-U hot loop beats cuDF on the sensor exact COUNT matrix,
- ad-hoc threshold-inclusive comparison does not beat cuDF,
- further cuDF work does not directly answer the project's core research question.

The main paper deliverable is the precision-throughput curve:

> Given a query, error budget epsilon, and data distribution, how many byte-plane subcolumns k are needed to place the result within epsilon of exact, and what throughput is achieved at that k on GPU HBM?

This spec asks the next agent to consolidate the existing Exp4 evidence and identify the smallest remaining work needed for the paper.

## 2. Core Questions

The report must answer these questions from existing results first:

1. Does reading fewer byte-plane subcolumns reduce GPU work/throughput cost in the expected direction?
2. For COUNT, does `U(k)` provide a valid error bound and epsilon-to-k* mapping?
3. For each data distribution and artifact class, where is the knee of the curve?
4. Which artifacts are mainline, side-study, or reject based on size/fidelity/transfer?
5. What is already paper-ready, and what is still missing for SUM/AVG/MIN/MAX/VAR or FOR-parameter analysis?

Do not open new experiments until the evidence inventory is complete.

## 3. Inputs

Read these existing reports/results:

```text
research/2026-05-02_Exp4_Status_Summary.md
research/2026-05-10_Exp4_v2_Paper_Figure_Pack_Report.md
research/2026-05-10_Exp4_v2_Claim_Figure_Report.md
research/2026-05-10_Artifact_Size_Fidelity_Transfer_Report.md
research/2026-05-12_cuDF_RAPIDS_Primitive_Route_Closeout_Report.md
```

Primary result directories:

```text
results/exp4/b1_20260510_174435_job44743_NVIDIAH200/
results/exp4/b1_20260510_190958_job45163_NVIDIAH200/
results/paper_v1/
```

Scripts to inspect, not rewrite unless needed:

```text
scripts/plot_exp4_v2_paper_figures.py
scripts/plot_exp4_v2_claim_figures.py
scripts/analyze_exp4_b1.py
```

## 4. Required Output

Create:

```text
research/2026-05-12_Mainline_Precision_Throughput_Evidence_Pack_Report.md
```

The report must contain:

### 4.1 Claim inventory

A table:

| claim | current evidence | status | blocking gap |
|---|---|---|---|

At minimum include:

- COUNT epsilon-to-k* curve
- COUNT U(k) error bound
- throughput vs k
- artifact size/fidelity/transfer tradeoff
- bounded precision drift caveat
- cuDF prepared-predicate side result
- SUM/AVG/MIN/MAX/VAR status
- FOR segment-size / integer-bit-width status

### 4.2 Paper-ready figures

List each figure currently usable for the paper:

| figure | path | claim it supports | caveat |
|---|---|---|---|

Use the existing v2 figure reports as the starting point.

### 4.3 Precision-throughput table

For the mainline artifacts, produce or point to a compact table:

| dataset | artifact | selectivity | epsilon | k* | throughput_at_k* | full_depth_throughput | speedup_at_k* | error_bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|

If the table already exists, cite the path.  If it must be regenerated, state the exact script command but do not run an expensive job.

### 4.4 Gap list

Rank missing work by paper value:

1. must-have for current paper,
2. useful but optional,
3. future work.

The gap list must explicitly decide whether each of these belongs in the current paper:

- SUM
- AVG
- MIN
- MAX
- VAR
- progressive filter warp-utilization strategy comparison
- FOR parameter sweep
- cuDF/RAPIDS primitive route

### 4.5 Next spec recommendation

Choose exactly one next spec:

- `COUNT precision-throughput paper closeout`
- `SUM/AVG extension spec`
- `FOR parameter sensitivity spec`
- `progressive filter strategy comparison spec`
- `paper writing / figure cleanup spec`

Do not recommend more than one primary next step.

## 5. Rules

- Do not run H200 jobs unless an existing result is missing and the report explains why the job is required.
- Do not submit Slurm work from this spec unless explicitly approved after the evidence inventory.
- Do not edit CUDA code.
- Do not change cuDF route code.
- Do not push.
- If only documents/scripts are touched, keep the patch small and scoped.

## 6. Definition of Done

This spec is done when:

1. The evidence-pack report exists.
2. Every core question in §2 has an explicit answer or named gap.
3. Paper-ready figures are listed with paths and caveats.
4. The precision-throughput table is either included or pointed to by path.
5. The report makes one primary next-step recommendation.
6. cuDF is correctly classified as side result/baseline, not mainline.

## 7. One-Sentence Summary

Stop optimizing the cuDF side route, consolidate the existing COUNT precision-throughput evidence into a paper-ready evidence pack, and pick the single highest-value next spec for the main thesis.
