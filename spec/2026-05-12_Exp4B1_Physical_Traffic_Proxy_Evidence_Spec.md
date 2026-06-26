# Exp4-B1 Physical-Traffic Proxy Evidence Spec

Date: 2026-05-12
Status: Spec only. No NCU job expected.
Recommended agent: `gpt-5.4-mini`

## 1. Goal

Build a conservative physical-traffic proxy evidence pack for Exp4-B1 while
fresh Nsight Compute capture remains blocked.

This spec does **not** claim measured DRAM, L2, or global-load traffic. It
produces a proxy table from existing:

- logical bytes,
- estimated pack-load bytes,
- average planes read,
- kernel time,
- H2D payload bytes,
- measured H2D transfer time.

The output should support careful paper wording such as:

> The measured H2D transfer time scales linearly with encoded payload size, and
> the kernel-level proxy bytes decrease with execution depth k. Full physical
> DRAM/L2 validation remains blocked by the current NCU profiling issue.

Do not write:

> Exp4-B1 measured DRAM traffic decreases by X.

That claim requires NCU and is currently unsupported.

## 2. Inputs

### 2.1 Encoded k-sweep

```text
results/exp4/b1_20260510_174435_job44743_NVIDIAH200/sweep_summary.csv
```

Useful columns:

```text
dataset
artifact_root
precision_mode
precision_decimals
threshold
selectivity
n
ms_per_iter
avg_planes_read_per_total_row
max_planes_read
logical_bytes
logical_GBps
estimated_physical_GBps
estimated_pack_load_bytes
rows_per_sec
max_plane_count
max_filter_planes
uncertain
count_abs_error_bound
gpu_eq_cpu_encoded
load_strategy
kernel_path
```

### 2.2 Epsilon-to-kstar

```text
results/exp4/b1_20260510_174435_job44743_NVIDIAH200/count_epsilon_to_kstar.csv
```

Use it to mark representative `kstar` points for:

```text
epsilon = 0
epsilon = 1e-3
```

### 2.3 Transfer evidence

```text
results/paper_v1/artifact_size_fidelity_transfer.csv
results/paper_v1/raw_fp64_transfer_baseline.csv
```

Useful columns:

```text
dataset
artifact_label
target_selectivity
bytes_per_row
transfer_bytes
cudaMemcpy_ms
effective_transfer_GBps
count_verdict
sum_verdict
```

### 2.4 NCU blocker note

```text
research/2026-05-11_Exp4B1_NCU_Physical_Traffic_Audit_Status.md
```

The report must cite this status and preserve the blocker.

## 3. Output CSV

Write:

```text
results/paper_v1/exp4_b1_physical_traffic_proxy.csv
```

Required columns:

```text
dataset
artifact_label
artifact_class
target_selectivity_pct
threshold
k
epsilon_marker
n
ms_per_iter
rows_per_sec
max_plane_count
avg_planes_read_per_total_row
logical_bytes
estimated_pack_load_bytes
estimated_pack_load_bytes_per_row
estimated_pack_load_bytes_vs_raw_ratio
logical_GBps
estimated_physical_GBps_proxy
U_k
execution_rel_error_bound
bytes_per_row_artifact
artifact_transfer_bytes
artifact_cudaMemcpy_ms
raw_transfer_bytes
raw_cudaMemcpy_ms
transfer_speedup_vs_raw
gpu_eq_cpu_encoded
load_strategy
kernel_path
traffic_claim_level
traffic_claim_reason
```

### 3.1 Derived fields

`artifact_label`:

```text
p{precision_decimals}
```

`target_selectivity_pct`:

Use the same normalization policy as
`spec/2026-05-12_Raw_vs_Encoded_COUNT_Claim_Table_Spec.md`.

`estimated_pack_load_bytes_per_row`:

```text
estimated_pack_load_bytes / n
```

`estimated_pack_load_bytes_vs_raw_ratio`:

```text
estimated_pack_load_bytes / (n * 8)
```

`estimated_physical_GBps_proxy`:

Use existing `estimated_physical_GBps` but rename in the output to make clear it
is a proxy, not a profiler metric.

`execution_rel_error_bound`:

```text
U_k / max(full_depth_encoded_count, 1)
```

If full-depth encoded count is not available in this table, use the exact count
from `count_epsilon_to_kstar.csv` after validating the key. If unavailable,
leave NA and explain in the report.

`epsilon_marker`:

```text
none
epsilon_0_kstar
epsilon_1e-3_kstar
```

Rows may include all k values, but mark the kstar rows.

`traffic_claim_level`:

```text
h2d_measured
kernel_proxy_only
ncu_required
```

Rules:

- H2D transfer fields are measured CUDA memcpy timing, so those can be
  `h2d_measured`.
- Kernel bytes/traffic fields are proxy-only.
- Any DRAM/L2/global-load statement is `ncu_required`.

## 4. Report

Write:

```text
research/2026-05-12_Exp4B1_Physical_Traffic_Proxy_Evidence_Report.md
```

Required sections:

1. Why this report exists:
   - NCU capture is currently blocked.
   - This report is a proxy pack, not a replacement for NCU.
2. Input inventory and row counts.
3. H2D measured result:
   - bytes per row vs cudaMemcpy time,
   - raw FP64 transfer baseline,
   - which artifacts transfer faster than raw.
4. Kernel proxy result:
   - `k` vs `estimated_pack_load_bytes_per_row`,
   - `k` vs `avg_planes_read_per_total_row`,
   - representative rows for mainline artifacts.
5. Epsilon-kstar proxy:
   - compare full-depth vs epsilon 1e-3 kstar where available.
6. Claim boundary:
   - allowed wording,
   - forbidden wording.
7. Next NCU unblock step:
   - once a single NCU smoke produces `.ncu-rep`, run representative matrix.

## 5. Recommended Figures

Generate figures only if straightforward:

```text
results/paper_v1/plots_exp4_b1_proxy/bytes_per_row_vs_h2d_ms.png
results/paper_v1/plots_exp4_b1_proxy/k_vs_estimated_pack_bytes_per_row.png
results/paper_v1/plots_exp4_b1_proxy/kstar_vs_full_depth_proxy_bytes.png
```

If plotting would take longer than the CSV/report, skip plots and state that the
CSV is ready for later figure cleanup.

## 6. Acceptance Criteria

Pass if:

1. Output CSV exists with required columns.
2. Report clearly says this is proxy evidence, not measured physical traffic.
3. H2D transfer claims are separated from kernel proxy claims.
4. No row claims measured DRAM/L2/global-load traffic.
5. The report identifies exactly what NCU result is still needed.

Fail and report blocker if:

1. required input CSVs are missing,
2. artifact labels cannot be normalized,
3. kstar rows cannot be marked,
4. transfer baseline cannot be joined to artifacts.

## 7. Out of Scope

- No Slurm job.
- No Nsight Compute run.
- No CUDA source changes.
- No cuDF route.
- No exp2 branch inspection.
- No external baselines.

