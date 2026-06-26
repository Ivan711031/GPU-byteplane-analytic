# Exp1 Byte Kernel ILP Optimization Spec

Date: 2026-04-15  
Status: Draft for discussion  
Owner: Nick / Codex

## 1. 任務定位

本任務只針對 `strategy=byte` 的 kernel path 做 **最小語意漂移** 的 micro-optimization 與 benchmark-oriented refactor。

本任務的唯一目的，是回答下面這個問題：

> 在不改變 byte baseline benchmark 核心語意的前提下，若只改善 kernel 內部的 instruction-level parallelism (ILP) 與 dependency shape，`byte` 路徑的效能是否能改善，且 Nsight Compute 的 Long Scoreboard / Eligible Warps 指標是否隨之改善？

這是一個 **kernel-structure optimization task**，不是 validation task，不是 reporting task，也不是 layout redesign task。

本輪不處理：

- `packed32`
- `shared128`
- `plane_bytes=2`
- validation
- CSV schema
- contiguous baseline
- plotting
- broader file split / class abstraction
- prefetch API / Unified Memory hint
- `cp.async`
- shared-memory staging redesign

## 2. 核心原則

### 2.1 隻改 byte hot path 的 dependency shape

本輪只允許修改 `strategy=byte` 的 hot path，使其產生更多 **independent accumulation chains**，藉此測試是否能改善 latency hiding。

### 2.2 不改 benchmark 核心語意

必須保留以下語意不變：

- SOA plane layout 不變
- `k` 仍代表「讀前 k 個 planes」
- grid-stride traversal 的本質不變
- reduction sink 不變
- timing model 不變
- block reduction 行為不變
- 每個 logical element 對應的 plane 讀取集合不變
- 不引入新的資料重排、預轉碼、shared staging、block tiling

### 2.3 本輪優化的重點是 ILP，不是換一個 benchmark

本輪允許：

- 將單一 accumulator 改成多個 accumulator
- 將 outer traversal 做固定倍率 unroll
- 將 load 與 accumulate 的排列改成更有利於 latency hiding 的形式

本輪不允許：

- 改變 global memory access 顆粒語意
- 把 byte path 偷偷改成 packed/shared path
- 引入 shared-memory cache 作為主要資料路徑
- 用改變 benchmark 性質的方式換取 throughput

### 2.4 不對 compiler / hardware 做未驗證宣稱

本任務可以要求：

- 增加 independent accumulators
- 調整 loop structure
- 讓 compiler 更容易暴露 ILP

但不得在 spec、commit message、summary 中宣稱：

- 一定提升 ILP
- 一定減少 scoreboard stall
- 一定增加 memory requests in flight
- 一定不會增加 register spill
- 一定會更快

這些都必須視最終 Nsight Compute / SASS / runtime 結果而定。

## 3. 當前程式的已知問題

目前 `byte` kernel 為：

- 單一 `sum`
- `for i in grid-stride`
- `for p in 0..K-1`
- 每次讀 `planes.ptrs[p][i]` 後立刻累加到同一個 accumulator

這種寫法的已知風險是：

1. load-to-use dependency 太緊  
	每次 global load 很快就被使用，latency hiding 空間有限。
2. 單一 accumulator 形成長依賴鏈  
	`sum += ...` 會讓 instruction stream 容易變成單條 serial dependency chain。
3. 雖然 `K` 已 compile-time specialization，但 `i` 維度仍只有單輪 traversal  
	可能導致每個 thread 在單位時間內缺少足夠 independent work。
4. Nsight Compute 已顯示：
	- Long Scoreboard stalls 高
		- Eligible Warps / Scheduler 低
		- Issued Warps / Scheduler 低
		- DRAM throughput 未達硬體上限附近

因此目前最合理的工作假說是：

> 問題不只是讀了多少 bytes，而是目前 byte kernel 的 dependency chain 與 ILP 結構不足，使 warp 在等待 global load operand ready 時沒有足夠 ready work 可發。

## 4. 本輪唯一要做的事

### 4.1 為 byte kernel 新增 ILP variant，使用多 accumulator + 固定倍率 outer unroll

**問題**  
目前 byte kernel 使用單一 accumulator，且每次只處理一個 `i`。這容易使 load 與 accumulate 緊耦合，限制 ILP。

**要求**  
必須新增一個新的 `byte` kernel variant，其核心做法為：

- 使用固定數量的獨立 accumulators
- 將 outer grid-stride traversal 做固定倍率 unroll
- 每個 accumulator 對應一個不同的 `i + t * stride`
- 最後再將所有 accumulators 合併後送入 `block_reduce_store`

**本輪固定要求**  
先只實作 **4-way ILP variant**。  
不得同時實作 2-way / 4-way / 8-way 多版本 sweep。

**強制實作方向**

概念上必須接近：

C++

unsigned long long sum0 = 0;  
unsigned long long sum1 = 0;  
unsigned long long sum2 = 0;  
unsigned long long sum3 = 0;  
  
for (uint64_t i = tid; i + 3 * stride < n; i += 4 * stride) {  
#pragma unroll
  for (int p = 0; p < K; ++p) {  
    sum0 += ... [i + 0 * stride]  
    sum1 += ... [i + 1 * stride]  
    sum2 += ... [i + 2 * stride]  
    sum3 += ... [i + 3 * stride]  
  }  
}

最後再處理 tail。

**目的**  
本輪要測的不是更少指令，而是更好的 dependency shape。  
不接受表面 unroll、實際仍只有單條 accumulation chain 的寫法。

### 4.2 必須保留現有 byte baseline kernel，新增 variant，而不是覆蓋舊 kernel

**問題**  
如果直接覆蓋舊 kernel，後續無法做乾淨 A/B comparison，也無法判斷改善是否真的來自 ILP 改動。

**要求**  
必須保留目前現有的 byte baseline kernel，不得刪除。

必須新增第二個 variant，例如命名方向可類似：

C++

scan_planes_u8_byte_baseline<K>  
scan_planes_u8_byte_ilp4<K>

命名可調整，但必須滿足：

- baseline 與 ilp4 可同時存在
- host 端可明確 dispatch 到其中之一
- kernel identity 不可模糊化

**不允許**

- 不允許直接把舊 kernel 改掉後假裝是同一版本
- 不允許只靠 comment 區分 baseline / ilp4
- 不允許用巨集把兩種 kernel 摻成 unreadable code

### 4.3 Host-side 必須新增 byte variant 選項，但 scope 僅限 byte path

**要求**  
必須在 `strategy=byte` 之內，再新增一個小範圍 variant 選項，例如概念上可為：

- `byte_baseline`
- `byte_ilp4`

或

- `strategy=byte`
- `--byte_kernel baseline|ilp4`

兩者擇一即可。

**要求原則**

- 只能影響 `strategy=byte`
- 不得擴散到 `packed32` / `shared128`
- dispatch 必須清楚對應到具體 kernel instantiation
- 不允許引入過度抽象的 class hierarchy / registry system

**推薦形式**

較推薦：

C++

--strategy byte  
--byte_variant baseline|ilp4

因為這樣語意最乾淨：strategy 還是 byte，variant 才是 baseline/ilp4。

### 4.4 compile-time specialization 必須完整保留

**要求**

- `K` 的有效範圍固定為 1 到 8
- baseline 與 ilp4 兩種 byte kernel 都必須維持 `template <int K>`
- plane loop 仍必須是 compile-time specialization
- inner plane loop 仍必須保留 `#pragma unroll`
- 不允許退回 runtime `for (p = 0; p < k; ++p)` hot path

**目的**

本輪要比較的是：

- 同一組 benchmark 語意下
- baseline dependency shape
- ilp4 dependency shape

不接受把 runtime loop overhead 混進比較。

### 4.5 tail handling 必須正確且保守

**問題**  
ILP4 版本會先處理 `i, i+stride, i+2*stride, i+3*stride`。若 `n` 不是 `4*stride` 的整數倍，必須處理尾端剩餘元素。

**要求**

- 必須保守處理 tail
- 不允許 out-of-bounds load
- 不允許為了簡化 code 假設 `n % (4*stride) == 0`
- 不允許依賴 undefined behavior

**實作方式**  
可接受：

- 先跑 main loop，條件為 `i + 3 * stride < n`
- 再用 second loop 處理剩餘的 `i`

或其他等價、可審查、可證明正確的保守寫法。

### 4.6 accumulation merge 必須在 thread-local 完成後再進 block reduction

**要求**

- `sum0..sum3` 必須先在 thread-local scope 合併成單一值
- 再呼叫既有 `block_reduce_store`

概念上可為：

C++

unsigned long long sum = sum0 + sum1 + sum2 + sum3;  
block_reduce_store(sum, per_block_out);

**不允許**

- 不允許修改 `block_reduce_store`
- 不允許把 4 個 sums 分別送進 reduction
- 不允許改變 reduction sink semantics

### 4.7 Occupancy query 必須將 baseline 與 ilp4 分開處理

**問題**  
ILP4 很可能增加 register footprint，進而改變 occupancy。若仍沿用 baseline 的 occupancy query，benchmark comparison 會不乾淨。

**要求**

若 byte path 提供 baseline 與 ilp4 兩個 kernel variant，則 occupancy query 必須對應到實際被 launch 的 kernel pointer。

也就是說：

- baseline query baseline kernel pointer
- ilp4 query ilp4 kernel pointer

**要求形式**

必須保留類似下列 helper 的概念：

C++

const void* scan_planes_u8_byte_kernel_ptr(ByteVariant v, int k);  
void launch_scan_planes_u8_byte(ByteVariant v, int k, ...);

命名可不同，但語意必須明確。

### 4.8 `d_out` 容量仍只能一次性組態，不得在 sweep 中重配

這條沿用上一輪原則，不能退步。

**要求**

- host 端先根據本輪實際會用到的 byte kernel variant + `K=1..8`
- 做 occupancy query
- 求出 `max_grid`
- 一次性 `cudaMalloc(d_out)`
- sweep 中只重用
- 不得在 timed loop / sweep loop 中 `cudaMalloc` / `cudaFree`

若本輪只跑單一 variant，則只需要針對該 variant 的 `<1>.. <8>` 求最大 grid。

## 5. 嚴格邊界

### 5.1 本輪允許改動的內容

只允許修改以下範圍：

- `strategy=byte` 的 kernel 實作
- `strategy=byte` 的 host-side variant parsing / dispatch
- `strategy=byte` 的 kernel pointer helper
- `strategy=byte` 的 occupancy query wiring
- 與 byte variant 對應的 `d_out` 安全容量管理
- 少量必要的 enum / helper type 定義

### 5.2 本輪禁止改動的內容

不得做以下事情：

- 不得修改 `packed32`
- 不得修改 `shared128`
- 不得修改 `scan_planes_u16`
- 不得加入 validation code
- 不得修改 CSV schema
- 不得加入新的 benchmark mode beyond byte variant selection
- 不得加入 contiguous baseline code
- 不得改 timing path
- 不得改 allocation layout
- 不得改 reduction semantics
- 不得加入 async stream
- 不得加入 pinned memory
- 不得加入 `cudaMemAdvise`
- 不得加入 `cudaMemPrefetchAsync`
- 不得加入 `__ldg`
- 不得加入 `__launch_bounds__`
- 不得加入 shared-memory staging
- 不得加入 `cp.async`
- 不得引入 CUB / Thrust / 第三方依賴
- 不得順手重構整個檔案結構
- 不得把 baseline kernel 刪除

## 6. 預期效果假說

本輪唯一性能假說是：

1. baseline byte kernel 目前可能受限於單一 accumulator 與緊密的 load-to-use dependency chain。
2. 若改為多 accumulator + fixed outer unroll，則 scheduler 可能看到較多 independent work。
3. 若目前 bottleneck 的確包含 ILP 不足，則 ilp4 版本可能在以下至少一部分指標上改善：
	- `ms_per_iter`
		- Long Scoreboard stalls
		- Eligible Warps / Scheduler
		- Issued Warps / Scheduler

**明確限制**

- 本規格不保證一定加速
- 本規格不保證 occupancy 一定更好
- 本規格不保證 register usage 不增加
- 本規格不保證 Long Scoreboard 一定下降

本規格只保證：  
這次改動是在**不改 benchmark 核心語意**前提下，對 dependency shape 做明確、可審查、可 A/B 比較的實驗性優化。

## 7. 驗收標準

任務完成必須同時滿足以下條件。

### 7.1 Kernel path

- 保留原本 byte baseline kernel
- 新增 byte ilp4 kernel
- 兩者皆使用 `template <int K>`
- 兩者皆維持 compile-time plane-loop specialization
- ilp4 kernel 使用多個獨立 accumulators
- ilp4 kernel 使用固定 4-way outer traversal unroll
- tail handling 保守且無越界
- `block_reduce_store` 未被修改

### 7.2 Launch path

- host 端可明確選 baseline 或 ilp4
- dispatch 明確對應到 `<1>.. <8>` specialization
- occupancy query 對應到實際 variant
- `d_out` 仍為一次性最大安全容量分配
- sweep loop 中不存在 `cudaMalloc` / `cudaFree` for `d_out`

### 7.3 Scope discipline

- `packed32` 未被修改
- `shared128` 未被修改
- `plane_bytes=2` path 未被修改
- timing path 未被修改
- CSV schema 未被修改
- baseline kernel 未被刪除

### 7.4 Semantics

- 仍然只讀前 `k` 個 planes
- SOA plane layout 不變
- grid-stride traversal 的本質不變
- reduction sink 不變
- benchmark 核心語意不漂移

## 8. 流程聲明

本 PR 是 **byte-kernel-structure PR**。  
它的責任邊界是：

- 保留 baseline
- 新增 ilp4 variant
- 讓 A/B profile 與 benchmark comparison 成為可能

因此：

- 本 PR 不要求 correctness validation framework
- 本 PR 不要求最終性能結論
- 本 PR 不要求 Nsight Compute 報告整合進程式碼

這不是放棄驗證，而是明確切開責任邊界：  
本 PR 負責產生一個乾淨、可比較的 kernel variant；後續流程再做 benchmark / profiler /分析。

## 9. 開發者回報格式

完成後，開發者必須用以下格式回報：

1. `Summary`
2. `New Byte Variant Added`
3. `Kernel Interface Changes`
4. `Dispatch / Occupancy Changes`
5. `Tail Handling`
6. `Behavior Preserved`
7. `What Was Explicitly Not Changed`
8. `Open Caveats`

**禁止回報方式**

不得寫：

- 「已優化很多」
- 「理論上會更快」
- 「ILP 一定提高」
- 「應該比較不會 stall」
- 「大概不會 spill」

所有描述都必須對應實際改動內容，而不是猜測。