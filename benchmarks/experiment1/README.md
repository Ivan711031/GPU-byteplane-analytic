# Experiment 1 - Byte-plane scaling (SOA)

目標：驗證讀更少 byte-planes 是否等比例節省頻寬。

資料 layout 採 SOA：每個 FP64 value 的 8 個 bytes 拆成獨立連續 planes。

## 重要行為

- 這個 benchmark 只量測掃描讀取吞吐。
- 程式在 runtime 直接配置並初始化 planes，不需要外部 dataset 檔案。
- CSV 會寫入 `device/sm/cc`，方便分辨實際跑在 H100 或 H200。

## Build

```bash
cmake -S benchmarks/experiment1 -B build/exp1 -DCMAKE_BUILD_TYPE=Release
cmake --build build/exp1 -j
```

Hopper（H100/H200）常用：

```bash
cmake -S benchmarks/experiment1 -B build/exp1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp1 -j
```

## Run

### 推薦：使用 runner

```bash
./scripts/run_exp1.sh
```

Runner 預設配置：

- `plane_bytes=1`
- `strategy=byte`
- `k=1..8`

輸出目錄：

- `results/exp1/run_<timestamp>_job<jobid_or_nojob>_<gpu_tag>/`

同一個 run directory 內會有：

- `exp1.csv`
- `setup_estimate.txt`（memory setup 摘要）
- `run_meta.txt`
- `repro_command.txt`
- `ncu_command_template.txt`

### 直接執行

```bash
./build/exp1/bench_byteplane_scan --csv exp1.csv
```

常用參數：

```bash
./build/exp1/bench_byteplane_scan \
  --device 0 \
  --n 100000000 \
  --plane_bytes 1 \
  --strategy byte \
  --k_min 1 --k_max 8 \
  --block 256 --grid_mul 1 \
  --warmup 10 --iters 200 \
  --csv exp1_byteplane_scan.csv
```

可加 `--validate` 執行每個 `k` 的非計時總和檢查；validation 不放進 timed loop。

## Strategies

- `--strategy byte`: scalar byte-plane load；`--byte_variant baseline|ilp4` 只適用於此策略。
- `--strategy rowpack4`: row-wise `uint32` packed load；每次從同一個 byte-plane 讀 4 個連續 rows。
- `--strategy rowpack16`: row-wise `uint4` 128-bit packed load；每次從同一個 byte-plane 讀 16 個連續 rows。
- `--strategy shared128`: warp 先 stage 128B 到 shared，再消耗 32 個 logical rows（diagnostic overfetch path）。
- `--strategy packed32`: backward-compatible alias；程式會正規化成 `rowpack4`，CSV 也寫 `rowpack4`。

另支援 `--plane_bytes 2`（4 個 `uint16` planes），此時策略固定為 `byte`（u16 loads）。

`rowpack4` / `rowpack16` pack 的是同一個 byte-plane 裡的連續 rows，不是同一 row 的多個 precision bytes；`k` 仍然只控制讀取前幾個 byte-planes。

## Output

CSV 欄位包含：

- `logical_bytes = n * k * plane_bytes`
- `logical_GBps`：用 logical bytes 計算的 GB/s（1e9）
- `overfetch_factor`：`shared128` 會是 `4.0`
