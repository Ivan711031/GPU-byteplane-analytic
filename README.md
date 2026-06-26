# m03 GPU Byte-Plane Analytics report

## 問題背景

目前GPU 上的 scan / filter / aggregate 幾乎都是 memory-bandwidth bound，如果再處理浮點數時不是固定用 FP64 讀到底，而採用[Decomposed Bounded Floats for Fast Compression and Queries](https://www.vldb.org/pvldb/vol14/p2586-liu.pdf)的方法拆成依 significance 排序的 byte planes，系統是否能：

1. 只讀前幾層就得到夠好的答案，換到更高吞吐？
2. 在detect到資料出錯時，不直接 abort，而是回傳有界限的 fail-soft answer？
3. 利用依 significance 排序的特性設計 reliability 機制？
![image](https://hackmd.io/_uploads/Hka2B8jzzx.png)

- **execution contract**：查詢只讀前 `k` 個高顯著性 planes。
- **reliability contract**：若 fault 被偵測到，系統依受影響 planes 的最大答案影響，回傳 certified bounded answer，而不是未檢查的錯誤標量。

## Research Questions

**Q1.** byte-plane execution 在 GPU 上是否真的可行？尤其 byte-granularity layout 會不會被 coalescing 與 memory transaction 成本吃掉？

**Q2.** 在真實 scientific workloads 上，progressive fused Filter+Aggregate 是否能在可以接受誤差下比 raw FP64 更快？

**Q3.** 同一個 significance ordering，是否也能變成 reliability 設計槓桿，讓 detected corruption 被轉成 bounded degradation？

**Q4.** 在加入 replication 的情況下，graded 的擺放跟uniform 擺放的結果差異？

## 方法設計

### 1. 用row-wise packing 處理 coalescing

一開始設計的 byte-plane layout 因為 1-byte load 太碎，讓 GPU 根本沒吃滿 HBM。後來採用 **rowpack16** 的方式，保留 progressive `k` 語意，但把同一 plane 上連續 rows 改成 128-bit global loads

### 2. 用fused Filter+Aggregate 做 execution contract

系統的主要 operator 不是一般 SQL engine，而是單欄位、thresholded 的 fused Filter+Aggregate。主角是 SUM；COUNT 雖然會一起算，但因為 shallow `k` 可能留下 unresolved rows，所以 COUNT 並非這次最主要的研究主題

### 3. 用 detect-and-bound 做 reliability contract

fault path 不是把 NMR 當唯一答案，而是採取：

- 先 detect 或 vote
- 若有有效 replica 則 recover
- 否則依 plane contribution ceiling 回傳 certified bound
- 若前提不成立，明確回傳 unavailable


## 實驗設計

### Workload 來源與合理性


| Field | Rows | 用途 |
|---|---:|---|
| `cesm_atm_cloud` | 168.48M | execution + reliability |
| `cesm_atm_q` | 168.48M | execution |
| `hurricane_u` | 25M | execution + reliability |
| `hurricane_tc` | 25M | execution |

來源補充：

- `cesm_atm_cloud`、`cesm_atm_q` 取自公開的 `SDRBench CESM-ATM` 欄位資料；本 repo 使用的欄位分別是 `CLOUD`（cloud fraction）與 `Q`（specific humidity）。
- `hurricane_u`、`hurricane_tc` 取自公開的 `Hurricane Isabel` simulation dataset；本 repo 使用的欄位分別是 `U`（U-wind component）與 `TC`（temperature）。

以上四個資料集都是真實科學欄位，對應post-analysis scan workload。另外，早期機制驗證使用以下人工資料集，資料集 row 數皆為10^8：

| Synthetic dataset | 用途 |
|---|---|
| `sensor` | 早期 precision-throughput 與 predicate drift 機制驗證 |
| `uniform` | `k`-depth、COUNT、dispatch 與 break-even 機制驗證 |
| `heavy_tailed` | 壓力測試偏斜分布下的 progressive 行為 |
| `zipfian` | 壓力測試極端偏斜與長尾分布下的 progressive 行為 |


### Workload 特徵

- 四個 headline execution fields 都是 warm、device-resident、true-quasi-global single-segment artifact。
- `cesm_atm_cloud`、`cesm_atm_q` 代表大規模 CESM 場；`hurricane_u`、`hurricane_tc` 代表較小但仍真實的 Hurricane 場。
- 人工資料集方面，`sensor` 代表低動態範圍、較平滑的數值分布；`uniform` 代表較平均的分布；`heavy_tailed` 與 `zipfian` 則用來模擬偏斜與極端值較多的情況。
- 在 reliability 這邊的實驗上，`cesm_atm_cloud` 沒有encoding noise，可以透過跟clean k的比較來看出不同 policy 的差異；`hurricane_u` 因為存在encoding noise，故較難看出不同policy 之間的差異（因為會被encoding noise稀釋）。

### 環境設定

- GPU execution 測量：NVIDIA H200
- runtime baseline：raw-fused FP64 CUDA primitive
- cuDF 作外部成熟系統 anchor
- reliability fault model：software / logical fault injection，在正常硬體上的 byte-plane artifact 內，依 seed 對特定位元組、plane、segment 或 replica 注入邏輯性 corruption，本次共設計8種fault model


### 參數選擇

**Execution**

- rowpack feasibility：比較 `byte_ilp4_k8`、`rowpack16_k8`、`contiguous64`
- scientific execution：主看 warm `k=2`
- `k`-depth dispatch：看 `k=1` 到 `k=max`
- locality：比較 single-segment 與 4096-segment

**Reliability**

single-replica detect-and-bound：

- datasets：`hurricane_u`、`cesm_atm_cloud`
- fault rates：`1e-6`、`1e-5`、`1e-4`
- seeds：3
- 額外控制：same-fault-all negative control
- 總規模：24-cell matrix

structured-fault H200 NMR path：

- datasets：`hurricane_u`、`cesm_atm_cloud`
- fault families：7 類 structured software faults
- policies：5
- seeds：3
- 執行單位：500K-row slices
- 總規模：210 configurations

realistic structured campaign：

- datasets：`hurricane_u`、`cesm_atm_cloud`
- policies：20（3 個 uniform + 17 個 graded）
- fault families：F1–F8
- severity cells：11
- fault rate：`1e-7`、`1e-5`
- seeds：10
- replication 數量：`B=0`、`B=8`、`B=16`
- 總規模：每個 dataset 4,400 evaluations，合計 8,800 evaluations

### 統計顯著性


- execution 使用同一台 H200、同一個 raw baseline、warm device-resident 條件做對照。
- reliability 的研究中使用 multi-seed 設計，並在 realistic campaign 裡用 frozen fault-plan replay，確保不同 policy 面對的是完全相同的 fault 事件。
- matched-budget 比較只在 storage 真正相同時才採信。


## 實驗結果

### Execution 結果


Exp1 顯示 `rowpack16_k8` 可達 `4394 GB/s`，高於 `byte_ilp4_k8` 的 `2399 GB/s`，也高於 `contiguous64` 的 `3929 GB/s`。因為 rowpack16 把 byte-granularity load 的 coalescing 問題處理掉

### scientific headline

![圖 1](https://hackmd.io/_uploads/H1npSBjMGx.png)


**在 warm、device-resident、quasi-global 的條件下，shallow `k=2` 是主 operating point。**

- `cesm_atm_cloud`: `3.64×–4.92×`
- `hurricane_u`: `2.69×–3.90×`
- `cesm_atm_q`: `3.58×–6.14×`
- `hurricane_tc`: `2.94×–5.72×`

當`k=2` 這個淺層 operating point 穩定有利。圖中的 `k=max` 是 diagnostic/fallback 對照，當在讀更深的 planes 時，優勢會縮小

### locality boundary

![image](https://hackmd.io/_uploads/BJdxLrjzMg.png)


**execution win 依賴 locality，當 segment 太細時，host-side threshold preparation 會吃掉收益。**

在 `CESM Cloud Fraction` 上，quasi-global single-segment 的 total latency 約 `0.23 ms`，但 `4096` segment 會上升到約 `17.62 ms`；其中大部分不是 GPU scan，而是 threshold prep。在 `Hurricane U-Wind` 上也能看到相同情況。

### traffic attribution

![image](https://hackmd.io/_uploads/SkKW8rozzx.png)


**shallow `k` 的 latency 改善，至少在代表性 scientific case 上，和較低的 memory traffic 方向一致。**

以 `cesm_atm_q, s10` 為例，`k=2` 相對 raw FP64：

- runtime 約為 `0.174×`
- DRAM read 約為 `0.135×`
- L2、L1/TEX counters 也同方向下降

因此，這裡可以保守地說，`k=2` 的 execution gain 不是單純量測噪音，而是和實際較少的資料搬移一致；但這仍是 scoped attribution，不應被寫成普遍物理定律。

### break-even / dispatch

![image](https://hackmd.io/_uploads/BJsf8SoGfg.png)


**raw FP64 fallback**

在 synthetic calibration 上，隨著 `k` 加深，latency speedup 會一路下降。例如 `uniform_p10` 在 `k=2` 時仍約 `1.09×`，到 `k=4` 已降到 `0.97×`；`heavy_tailed_p6` 在 `k=2` 約 `1.04×`，到 `k=3` 已降到 `0.96×`。也就是說，DRAM savings 可能還在，但端到端 latency 優勢已經消失，所以可以把 shallow `k` 當主路徑，並保留 raw FP64 fallback。

### 小結

- **Claim 1.** byte-plane fused Filter+Aggregate 在 H200 上確實存在一個可用的 execution operating region，也就是 warm、device-resident、quasi-global、shallow `k` 的 scientific post-analysis scan。
- **Claim 2.** 在這個 operating region 內，progressive execution 對真實 scientific fields 相對 raw-fused FP64 可達 `2.69×–6.14×` speedup
- **Claim 3.** 這個 speedup 與 memory traffic reduction 方向一致；代表性案例中 `k=2` 可把 DRAM read 降到 raw 的約 `0.135×`，所以 execution gain 有可觀測的機制支撐。
- **Claim 4.** 當 locality 太細或 `k` 太深時，threshold prep 與額外 plane 讀取會迅速吃掉優勢，因此設計 raw FP64 fallback機制

### reliability

**detect-and-bound**

- single-replica 24-cell matrix：`24/24` 都能回傳 truth-containing bounded answer
- structured-fault H200 path：`210/210` CPU/GPU classification 一致，`210/210` contains truth

reliability contract 是成立的：一旦 corruption 被偵測到，系統可以轉成 recover 或 bounded，而非 silent wrong。

**graded protection 能提供更緊的certified bound。**

在 matched storage 下，graded protection 對 certified bound width 的優勢很大，uniform/graded 的 bound-width ratio 約為：

- `345×`（`hurricane_u`, `2e-5`）
- `432×`（`hurricane_u`, `2e-4`）
- `348×`（`cesm_atm_cloud`, `2e-5`）
- `419×`（`cesm_atm_cloud`, `2e-4`）


證明了使用 graded 策略能得到比較 tight 的 certified bound，


**NMR latency 成本依賴讀取 prefix 的深度**：

- 在 shallow prefix 上開銷可接受。以 `P3_fused_vote_digest_inline_all_read_planes` 為例，`k=1` 時約為 raw 的 `0.49×`（`cesm_atm_cloud`）與 `0.56×`（`hurricane_u`）；`k=2` 時約為 raw 的 `0.74×` 與 `0.87×`。
- 但當範圍擴到 full prefix，成本會快速上升。
full-`k` 時，`cesm_atm_cloud` 約為 raw 的 `1.82×–1.90×`，`hurricane_u` 約為 `2.46×–2.63×`。

當 prefix 變深時，受保護 planes 跟著變多，replica reads 幾乎按 plane 數線性增加，vote/digest 也會在更多 plane 上重複發生；此時 byte-plane 路徑原本少讀資料的優勢會逐漸消失，只剩下更多讀取、更多比較、更多融合邏輯的成本

**fused shallow-prefix NMR 有機會維持在 raw 以下，但 full-prefix / full-replica voting 仍然昂貴。** 所以可以考慮把 NMR 當成 prefix-aware、k-aware 的機制，而不是把整個 full-r3 path 當成預設查詢路徑。

**top-`k` point-answer**

在 `cesm_atm_cloud` 上，graded 在 matched realized storage 下比 uniform 更接近 `clean_k`；在 `hurricane_u` 雖然encoding error較大，但在不看encoding error只看clean k error時也是由graded勝出uniform


## 結論


1. **在 execution 面，byte-plane 是有效的 GPU primitive，且在 shallow-`k`、warm residency、好 locality 的條件下更有價值。**
2. **在 reliability 面，同一個 significance ordering 可以把 detected corruption 轉成 recover-or-bound 的 fail-soft contract。**

## repo
https://github.com/Ivan711031/GPU-byteplane-analytic

## reference
Decomposed Bounded Floats for Fast Compression and Queries
http://www.vldb.org/pvldb/vol14/p2586-liu.pdf

A Study of the Fundamental Performance Characteristics of GPUs and CPUs for Database Analytics
https://anilshanbhag.in/static/papers/crystal_sigmod20.pdf

ALP: Adaptive Lossless floating-Point Compression
https://ir.cwi.nl/pub/33334/33334.pdf

G-ALP: Rethinking Light-weight Encodings for GPUs
https://dl.acm.org/doi/epdf/10.1145/3736227.3736242

GPU Acceleration of SQL Analytics on Compressed Data
https://dl.acm.org/doi/pdf/10.14778/3778092.3778095
