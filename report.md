# m03 GPU Byte-Plane Analytics Report

## 問題背景

目前 GPU 上的 scan / filter / aggregate 幾乎都是 memory-bandwidth bound。如果處理浮點數時不是固定用 FP64 讀到底，而採用 [Decomposed Bounded Floats for Fast Compression and Queries](https://www.vldb.org/pvldb/vol14/p2586-liu.pdf) 的方法拆成依 significance 排序的 byte planes，系統是否能：

1. 只讀前幾層就得到夠好的答案，換到更高吞吐？
2. 在 detect 到資料出錯時，不直接 abort，而是回傳有界限的 fail-soft answer？
3. 利用依 significance 排序的特性設計 reliability 機制？

- **execution contract**：查詢只讀前 `k` 個高顯著性 planes。
- **reliability contract**：若 fault 被偵測到，系統依受影響 planes 的最大答案影響，回傳 certified bounded answer，而不是未檢查的錯誤標量。

## Research Questions

**Q1.** byte-plane execution 在 GPU 上是否真的可行？尤其 byte-granularity layout 會不會被 coalescing 與 memory transaction 成本吃掉？

**Q2.** 在真實 scientific workloads 上，progressive fused Filter+Aggregate 是否能在可以接受誤差下比 raw FP64 更快？

**Q3.** 同一個 significance ordering，是否也能變成 reliability 設計槓桿，讓 detected corruption 被轉成 bounded degradation？

**Q4.** 在加入 replication 的情況下，graded 的擺放跟 uniform 擺放的結果差異？

## 方法設計

### 1. 用 row-wise packing 處理 coalescing

一開始設計的 byte-plane layout 因為 1-byte load 太碎，讓 GPU 根本沒吃滿 HBM。後來採用 **rowpack16** 的方式，保留 progressive `k` 語意，但把同一 plane 上連續 rows 改成 128-bit global loads。

### 2. 用 fused Filter+Aggregate 做 execution contract

系統的主要 operator 不是一般 SQL engine，而是單欄位、thresholded 的 fused Filter+Aggregate。主角是 SUM；COUNT 雖然會一起算，但因為 shallow `k` 可能留下 unresolved rows，所以 COUNT 並非這次最主要的研究主題。

### 3. 用 detect-and-bound 做 reliability contract

fault path 不是把 NMR 當唯一答案，而是採取：

- 先 detect 或 vote
- 若有有效 replica 則 recover
- 否則依 plane contribution ceiling 回傳 certified bound
- 若前提不成立，明確回傳 unavailable

## 實驗設計

### Workload

| Field | Rows | 用途 |
|---|---|---:|
| `cesm_atm_cloud` | 168.48M | execution + reliability |
| `cesm_atm_q` | 168.48M | execution |
| `hurricane_u` | 25M | execution + reliability |
| `hurricane_tc` | 25M | execution |

- `cesm_atm_cloud`、`cesm_atm_q` 取自公開的 SDRBench CESM-ATM。
- `hurricane_u`、`hurricane_tc` 取自 Hurricane Isabel simulation dataset。

合成資料集（10⁸ rows）：

| Synthetic dataset | 用途 |
|---|---|
| `sensor` | 早期 precision-throughput 與 predicate drift 驗證 |
| `uniform` | k-depth、COUNT、dispatch 與 break-even 驗證 |
| `heavy_tailed` | 壓力測試偏斜分布下的 progressive 行為 |
| `zipfian` | 壓力測試極端偏斜與長尾分布下的 progressive 行為 |

### 環境設定

- GPU execution 測量：NVIDIA H200
- runtime baseline：raw-fused FP64 CUDA primitive
- cuDF 作外部成熟系統 anchor
- reliability fault model：software / logical fault injection

### 參數選擇

**Execution**

- rowpack feasibility：比較 `byte_ilp4_k8`、`rowpack16_k8`、`contiguous64`
- scientific execution：主看 warm `k=2`
- k-depth dispatch：看 `k=1` 到 `k=max`
- locality：比較 single-segment 與 4096-segment

**Reliability**

- single-replica detect-and-bound：3 seeds，24-cell matrix
- structured-fault H200 NMR path：7 fault families × 5 policies × 3 seeds，210 configurations
- realistic structured campaign：F1–F8 × 20 policies × 11 severity cells × 10 seeds × 3 replication levels，8,800 evaluations

## 實驗結果

### Execution

**Rowpack feasibility**

Exp1 顯示 `rowpack16_k8` 可達 4394 GB/s，高於 `byte_ilp4_k8` 的 2399 GB/s，也高於 `contiguous64` 的 3929 GB/s。

**Scientific headline**（warm, quasi-global, shallow k=2）

- `cesm_atm_cloud`：3.64×–4.92× vs raw FP64
- `hurricane_u`：2.69×–3.90×
- `cesm_atm_q`：3.58×–6.14×
- `hurricane_tc`：2.94×–5.72×

**Locality boundary**

當 segment 太細時，host-side threshold preparation 吃掉收益。single-segment total latency ~0.23 ms；4096-segment 上升到 ~17.62 ms。

**Traffic attribution**

`k=2` 的 DRAM read 降到 raw 的 ~0.135×（`cesm_atm_q` 為例），L2、L1/TEX counters 也同方向下降。Execution gain 與實際較少的資料搬移一致。

**Break-even / dispatch**

`k=2` 穩定有利；到 `k=4` 端到端 latency 優勢已經消失，需設計 raw FP64 fallback。

### Reliability

**Detect-and-bound**

- single-replica 24-cell matrix：24/24 回傳 truth-containing bounded answer
- structured-fault H200 path：210/210 CPU/GPU classification 一致，210/210 contains truth

**Graded vs uniform bound width**

| Dataset | Fault rate | uniform/graded ratio |
|---|---|---|
| hurricane_u | 2e-5 | 345× |
| hurricane_u | 2e-4 | 432× |
| cesm_atm_cloud | 2e-5 | 348× |
| cesm_atm_cloud | 2e-4 | 419× |

**NMR latency**

Shallow prefix (`k=1,2`) 開銷可接受（0.49×–0.87× of raw）；full prefix 成本快速上升（1.82×–2.63×）。

## 結論

1. **Execution**：byte-plane 是有效的 GPU primitive。在 shallow-k、warm residency、好 locality 的條件下，progressive fused Filter+Aggregate 對真實 scientific fields 可達 2.69×–6.14× speedup。Speedup 與 memory traffic reduction 方向一致。當 locality 太細或 k 太深時，threshold prep 與額外 plane 讀取會吃掉優勢，需要 raw FP64 fallback。

2. **Reliability**：同一個 significance ordering 可以把 detected corruption 轉成 recover-or-bound 的 fail-soft contract。Graded protection 在 matched storage 下提供比 uniform 緊 345×–432× 的 certified bound。Fused shallow-prefix NMR 有機會維持在 raw 以下，但 full-prefix / full-replica voting 仍然昂貴。
