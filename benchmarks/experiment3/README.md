# Experiment 3 - Progressive Aggregation

目標：實作 Buff-style progressive SUM aggregation benchmark，支援：

- synthetic fixed-point subcolumns throughput path
- real DEV encoded subcolumns path

synthetic v1 不依賴外部 dataset，也不等待 Exp2 error analysis。它只產生：

```text
refinement_depth -> throughput
```

之後再與 Exp2 的：

```text
refinement_depth -> error
```

合併成 precision-throughput curve。

## Data Model

V1 使用 synthetic fixed-point subcolumns：

```text
subcolumn[p][row] = p + 1
base[segment] = 0.0
basis[segment][p] = 2^(-8p)
```

SUM 定義：

```text
SUM(depth) =
  n * base
  + sum_{p=0..depth} SUM(subcolumn[p]) * basis[p]
```

`refinement_depth=0` 仍然讀一個 leading encoded byte subcolumn。

## Load Strategy

V1 只支援：

```text
rowpack16
```

也就是每個 thread 對同一個 subcolumn 讀取連續 16 rows 的 `uint4`。這不是讀同一 row 的 16 個 precision bytes。

Boundary handling：

- 只有完整且 16-row aligned 的 chunk 使用 `uint4`
- scalar prologue / epilogue 使用 block-stride `uint8_t` loads
- `n` 或 `segment_rows` 不是 16 的倍數時仍應正確

## Build

```bash
cmake -S benchmarks/experiment3 -B build/exp3 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp3 -j
```

Build targets:

- `bench_progressive_aggregation`
- `export_encoded_dev_layout`

Build log 會包含 `--ptxas-options=-v`，用來檢查 registers/thread 與 spill load/store。

## Run

### Synthetic Mode

推薦使用 runner：

```bash
./scripts/run_exp3.sh
```

輸出目錄：

```text
results/exp3/run_<timestamp>_job<jobid_or_nojob>_<gpu_tag>/
```

直接執行：

```bash
./build/exp3/bench_progressive_aggregation \
  --device 0 \
  --n 100000000 \
  --segment_rows 1048576 \
  --subcolumns 8 \
  --refine_min 0 \
  --refine_max 7 \
  --load_strategy rowpack16 \
  --block 256 \
  --items_per_thread 1 \
  --warmup 10 \
  --iters 200 \
  --csv results/exp3/progressive_aggregation_rowpack16.csv
```

### Real Encoded DEV Mode

先在 CPU allocation 上把 raw FP64 dataset 匯出成 Exp3-ready artifact。不要在 login node 對完整資料做 full scan。

```bash
./build/exp3/export_encoded_dev_layout \
  --input /work/u4063895/datasets/synthetic/dev/uniform.f64le.bin \
  --output-root /work/u4063895/datasets/synthetic/dev_buff_exp3 \
  --segment-size 4096 \
  [--precision-decimals N]
```

如果要一次匯出 4 個 DEV datasets，使用 CPU job wrapper：

```bash
sbatch run_exp3_export_dev.sh
```

匯出完成後，real mode benchmark 直接讀 dataset root：

```bash
./build/exp3/bench_progressive_aggregation \
  --device 0 \
  --mode encoded_dev_subcolumns \
  --encoded-root /work/u4063895/datasets/synthetic/dev_buff_exp3/uniform \
  --refine_min 0 \
  --refine_max 8 \
  --load_strategy rowpack16 \
  --block 256 \
  --items_per_thread 1 \
  --warmup 10 \
  --iters 200 \
  --validate \
  --csv results/exp3/encoded_uniform.csv
```

runner 也支援 real mode，預設 synthetic 不變：

```bash
MODE=encoded_dev_subcolumns \
ENCODED_ROOT=/work/u4063895/datasets/synthetic/dev_buff_exp3/uniform \
VALIDATE=1 \
./scripts/run_exp3.sh
```

## Smoke Tests

Basic validation:

```bash
./build/exp3/bench_progressive_aggregation \
  --device 0 \
  --n 1024 \
  --segment_rows 1024 \
  --subcolumns 3 \
  --refine_min 0 \
  --refine_max 2 \
  --load_strategy rowpack16 \
  --block 256 \
  --items_per_thread 1 \
  --warmup 0 \
  --iters 1 \
  --validate \
  --csv /tmp/exp3_smoke.csv
```

Boundary validation:

```bash
./build/exp3/bench_progressive_aggregation \
  --device 0 \
  --n 1031 \
  --segment_rows 257 \
  --subcolumns 3 \
  --refine_min 0 \
  --refine_max 2 \
  --load_strategy rowpack16 \
  --block 256 \
  --items_per_thread 1 \
  --warmup 0 \
  --iters 1 \
  --validate \
  --csv /tmp/exp3_boundary_smoke.csv
```

## CSV

CSV contains:

- `refinement_depth`
- `logical_subcolumns_read = refinement_depth + 1`
- `logical_bytes = n * logical_subcolumns_read`
- `rows_per_sec`
- `billion_rows_per_sec`
- `logical_GBps`
- synthetic error bounds clearly marked as `synthetic_fixed_point`
- real mode 額外會填 `encoded_layout`, `max_plane_count`, `exact_sum`, `cpu_approximate_sum`, `gpu_approximate_sum`

Synthetic error bounds are not the final Exp2 paper error.
