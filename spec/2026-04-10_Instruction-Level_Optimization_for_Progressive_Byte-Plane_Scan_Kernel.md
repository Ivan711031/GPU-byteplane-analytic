Design Doc: Instruction-Level Optimization for Progressive Byte-Plane Scan Kernel

Author: Nick (Smallfu666)
Date: 2026-04-10
Status: Implementation Ready

## 1. Context & Scope

目前的 `bench_byteplane_scan.cu` (Experiment 1) 旨在驗證 progressive byte-plane scan 演算法在 H200 上的 throughput scaling 表現。然而，透過 Nsight Compute 進行 profiling 後發現，kernel 的執行效率遠低於預期，且並未呈現 memory bound 應有的 scaling 特性。本檔案旨在定義問題根本原因，並提出 instruction-level 的優化方案。

## 2. Problem Statement

我們在觀察 `strategy=byte` 的實驗數據與 ncu profiling 報告時，發現了嚴重的不一致與效能瓶頸：

- Instruction-Level Bottleneck (非 Memory Bound)：
  根據 CSV 數據，當 `k=1` 時執行時間為 `0.165 ms`，而 `k=8` 時為 `0.404 ms`。若程式為 memory bound，處理 8 倍資料量的時間應接近 8 倍。實際上僅增加約 2.4 倍，證明 GPU 絕大部分時間消耗在 instruction issue 與 branch prediction，而非 memory transfer。

- Long Scoreboard Stalls：
  ncu 的 Source view 顯示，核心的 memory load 行 (`sum += ...`) 佔據了 63.47% 的 Stall Scoreboard 與 41.96% 的 Instructions Executed。

- Root Causes：
  - Failed Loop Unrolling: 變數 `k` 作為 runtime argument 傳入，導致 nvcc 無法在 compile time 預測迴圈次數，進而無法執行 loop unrolling。這產生了極大的 branch overhead。
  - Insufficient Instruction Payload: 每個 thread 發出的 `LDG.E.U8` instruction 僅請求 1 byte。要達到 H200 4.8 TB/s 的 memory bandwidth，這種極小的 payload 會讓 memory controller 被 instruction overhead 淹沒，無法有效 saturate 頻寬。

(註：原先 ncu 警告的 Global Store Access Pattern 僅發生在每個 block 結束時的單一 thread 寫入，對整體效能影響趨近於零，確認為 false alarm。)

## 3. Goals & Non-Goals

Goals:

- 消除 kernel 內的 branch overhead，強制 compiler 進行 loop unrolling。
- 提升單一 memory instruction 的 payload，減少發射 load instructions 的次數。
- 讓 `k=1..8` 的 throughput scaling 曲線符合 memory bound 的反比線性預期。

Non-Goals:

- 修改 experiment0 的 HBM bandwidth baseline。
- 在本階段實作完整的 Buff-style encoding 或 predicate evaluation (Progressive Filter)。

## 4. Proposed Design

為瞭解決上述問題，我們將採取兩階段的重構策略：

### 4.1. Action A: Template-Based Loop Unrolling

將原本由 runtime 傳入的參數 `k`，改為 template parameter，讓編譯器在 compile time 就能確定迴圈邊界，進而將迴圈完全展開。

Kernel Implementation:

```cpp
template <int K>
__global__ void scan_planes_u8_byte_unrolled(const uint8_t* const* __restrict__ planes,
                                             uint64_t n,
                                             unsigned long long* __restrict__ per_block_out) {
  unsigned long long sum = 0;
  uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                 static_cast<uint64_t>(threadIdx.x);
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);

  for (uint64_t i = tid; i < n; i += stride) {
    #pragma unroll
    for (int p = 0; p < K; p++) {
      sum += static_cast<unsigned long long>(planes[p][i]);
    }
  }
  block_reduce_store(sum, per_block_out);
}
```

Host Dispatch Logic:
在 host 端使用 `switch-case` 來 launch 對應的 template kernel：

```cpp
switch (k) {
  case 1: scan_planes_u8_byte_unrolled<1><<<grid, opt.block_threads>>>(...); break;
  case 2: scan_planes_u8_byte_unrolled<2><<<grid, opt.block_threads>>>(...); break;
  // ... implement cases up to 8
  case 8: scan_planes_u8_byte_unrolled<8><<<grid, opt.block_threads>>>(...); break;
}
```

### 4.2. Action B: Shift to Vectorized Access Strategy (packed32)

單純的 loop unrolling 仍無法解決 1-byte payload 的問題。必須將實驗重心轉移到已實作的 `packed32` strategy。此策略透過將 pointer 轉型為 `uint32_t*`，讓單一 thread 使用 4 bytes 的 payload 進行 memory load，單一 warp 即可發出 128 bytes 的 coalesced memory request，大幅降低 memory latency 與 scoreboard stalls。

## 5. Alternatives Considered

- 使用 `#pragma unroll` 搭配 Runtime Variable：曾考慮直接在現有程式碼加上 `#pragma unroll`。然而，由於迴圈上限 `k` 在 compile time 未知，nvcc 會忽略此 directive，無法產生效用。
- Manual Loop Expansion：手動寫死 8 個不同的 kernel 或在一個 kernel 內寫死 8 個 if 條件。此作法違反 DRY 原則，維護成本高，且無法有效利用 compiler optimization。因此選擇 template 方案。

## 6. Metrics & Evaluation

優化是否成功，將由以下 Metrics 進行驗證：

- Throughput Scaling (CSV Metric)：
  重新執行 `scripts/legacy/root_runners/run_exp1.sh` 後，`strategy=packed32` 在 `k=8` 與 `k=4` 的 `logical_GBps` 必須大幅超越 `strategy=byte` 的數據，並接近 exp0 所測得的 baseline (約 4000+ GB/s)。

- Stall Scoreboard Reduction (ncu Metric)：
  透過 Nsight Compute 重新 profiling 後，Source view 中的 load instruction 所在行的 Stall Scoreboard 佔比必須顯著下降（預期降至 20% 以下）。

- Achieved Occupancy (ncu Metric)：
  觀察 register 壓力是否因為 loop unrolling 而改變，確保 active warps 數量足以隱藏 memory latency。
