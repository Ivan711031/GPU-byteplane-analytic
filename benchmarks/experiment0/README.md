# Experiment 0 - HBM bandwidth microbenchmark

這個實驗用 `cudaEvent` 量測 GPU global memory 的有效讀取頻寬，並用 dummy reduction 避免讀取被編譯器最佳化掉。

支援三種模式：

- `seq`: coalesced sequential read
- `masked`: 部分 lane inactive 的讀取
- `gather`: 透過 index array 的隨機讀取

## Build

```bash
cmake -S benchmarks/experiment0 -B build/exp0 -DCMAKE_BUILD_TYPE=Release
cmake --build build/exp0 -j
```

若在 Hopper（H100/H200）上，常用：

```bash
cmake -S benchmarks/experiment0 -B build/exp0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp0 -j
```

## Run

### 推薦：使用 runner

```bash
./scripts/run_exp0.sh
```

Runner 預設會跑 `seq/masked/gather`，並把結果寫到：

- `results/exp0/run_<timestamp>_job<jobid_or_nojob>_<gpu_tag>/`

同一個 run directory 內會有：

- `exp0_seq.csv`
- `exp0_masked.csv`
- `exp0_gather.csv`
- `run_meta.txt`
- `repro_command.txt`
- `ncu_command_template.txt`

### 直接執行單一 mode

```bash
./build/exp0/bench_hbm_bw --mode seq --csv exp0_hbm_bw.csv
```

常用參數：

```bash
./build/exp0/bench_hbm_bw \
  --mode masked \
  --mask_stride 2 --mask_active 1 \
  --bytes_min 1MB --bytes_max 8GB --bytes_mult 2 \
  --block 256 --grid_mul 1 \
  --warmup 10 --iters 200 \
  --csv exp0_masked.csv
```

## CSV 欄位

`mode,bytes,effective_bytes,block,grid,warmup,iters,ms_per_iter,GBps,active_frac,gather_span_bytes,device,sm,cc_major,cc_minor`

註：

- `GBps` 使用 1e9（GB/s）。
- `effective_bytes` 是 `masked` 模式實際有效讀取量（`bytes * active_frac`）。
- `gather_span_bytes` 是 `gather` 模式索引覆蓋範圍。
