# 將 Byte-Plane Analytics 視為具雙重契約的 GPU 基元：漸進式執行與 Fail-Soft 可靠性

**狀態：** submission-facing Markdown draft v1.4（整合 v1.303 證據線）  
**和前一版草稿的關係：** v1.4 是 `workshop_submission_draft_v1_303.md` 的 submission-safe 改寫版。它保留已凍結的執行證據（四個 field 的 H200 結果、locality、k-depth dispatch）與可靠性證據（single-replica detect-and-bound、structured-fault NMR path、logical placement diversity、D2 graded allocation，以及 raw-FP64 comparator），**不新增任何實驗**。這一版的重點是重新整理敘事方式，讓整篇文章更像 systems paper，而不是證據清單：內部 result code 改用機制名稱表述、scope 邊界集中放在 §4，詳細證據表與 provenance 則移到 Appendix A。

## 摘要

科學後處理分析常常需要反覆掃描大型浮點欄位，並做帶有 threshold 條件的 reduction。當這些欄位不是以固定 FP32/FP64 buffer 儲存，而是採用 BUFF-style、依 significance 排序的 byte plane 形式時，這種 layout 會額外暴露出兩件事：第一，運算子其實需要把每個值讀到多深；第二，一旦偵測到資料損毀，這個錯誤最多會把答案推偏多少。我們把這樣的 layout 看成一個帶有雙重契約的 GPU primitive。**執行契約（execution contract）** 讓融合式的 Filter+Aggregate operator 只讀取前幾個 byte plane，並在執行時停在某個深度 *k*。**可靠性契約（reliability contract）** 則讓系統在偵測到但尚未修復的錯誤出現時，不是默默產生錯誤標量，而是回傳一個經過認證、帶有界限的答案區間，也就是 fail-soft degradation。

在四個 true-quasi-global CESM/Hurricane scalar field 上，我們在 NVIDIA H200 上測得 warm device-resident 的 *k*=2 執行，相較於手工調校的 raw-fused FP64 CUDA primitive，可達 **2.69×–6.14×** 的加速，同時 SUM 誤差維持在 machine epsilon 等級。這個優勢不是無條件成立的：它只出現在實測的 shallow-*k* 區間、locality 條件良好的情況下，而且一旦超過經驗上的 break-even 點，就應回退到 raw FP64。舉例來說，single-segment 的 threshold preparation 可控制在 0.06 ms 以下，但在 168M-row field 上，4096-segment 的 preparation 會拉高到約 14 ms。可靠性方面，single-replica pipeline 搭配 per-plane digest detection 與 certified bound widening，可以把測試矩陣中每一個注入的 corruption 都轉成「包含真值的 bounded answer」，且沒有 silent-wrong 結果；經由 H200 NMR path 跑出的 structured software-fault campaign，在所有測試案例上都和 CPU oracle 給出一致分類；而 significance-aware protection allocation 在相同額外儲存成本下，得到的 certified bound 比 true uniform-spread map 窄上數百倍。所有可靠性結果都建立在正常 H200 硬體上的 software/logical fault 證據；physical HBM placement diversity、temporal hardware diversity，以及 aged-GPU deployment 都明確不在本文範圍內（§4）。本文的貢獻不是新的表示法，也不是新的 bound，而是指出：同一種 significance-ordered representation，可以同時支撐 progressive execution 與 fail-soft reliability 這兩種可組合的 systems contract。

## 1. 引言

大型科學模擬與觀測管線會產生大量浮點陣列，而這些陣列在後處理時往往要被一掃再掃。研究者可能想知道某個 threshold 以上的雲訊號總量、某個風暴場摘要值在不同 cutoff 下怎麼變、或者套上不同 mask 之後，thresholded average 會怎麼移動。這些工作不是完整的 SQL pipeline，而是對大型陣列做反覆掃描、並配合 threshold 條件進行 reduction；而且在互動式或重複分析的場景裡，這些欄位往往早就準備好並常駐在裝置端。

傳統 GPU 分析執行通常把資料當成 FP32 或 FP64 buffer，然後針對這個固定表示法去加速 operator。這樣做會把兩種結構遮掉。第一，它遮掉了**執行深度（execution depth）**：系統很難自然地只讀取高 significance 的前幾層，並在夠用時提前停下來。第二，它遮掉了**答案影響的局部性（answer-impact locality）**：當記憶體錯誤已被偵測到、但還沒被修復時，raw FP64 buffer 並不會告訴你某個受損位元組最多會把 aggregate 影響到什麼程度。相對地，byte-plane representation 會把這兩件事都攤在檯面上。用來界定 early-stop 誤差的 per-plane contribution ceiling，同時也能界定當錯誤被限制在某個 plane 內時，它對最終答案可能造成的最大影響。

因此，我們把這種 layout 看成一個具備兩種契約的 systems primitive：

- **執行契約（execution contract）**：query 可以只讀取前幾個 byte plane，在執行時停在深度 *k*，並回傳帶界限的 approximate SUM-derived 結果；COUNT 的 tri-state ambiguity 則沿用既有 deterministic semantics。
- **可靠性契約（reliability contract）**：當偵測到某個 plane 或 segment 發生 corruption 時，系統可以把它轉成 bounded degradation，也就是依該 plane 的 contribution ceiling 擴大 certified interval，或回退到較淺但仍有效的深度，而不是默默吞掉這個錯誤。

**動機，但這裡只把它當動機。** 這兩種契約之所以值得看，是因為它們對「較低優先級、但帶寬需求高」的 accelerator 再利用場景很有吸引力。GPU 機群通常是多年期資本資產；隨著新一代 accelerator 持續推出，系統方會希望把舊設備的剩餘價值，留給那些不是最前線訓練、也不是極低延遲線上服務的工作。同時，GPU 記憶體可靠性在營運上也是真實存在的問題：ECC、dynamic page retirement、XID log、NVML 和 `nvidia-smi` 都會暴露記憶體錯誤的可見性，而大規模研究也已經記錄過生產環境中、以數千萬 device-hours 為尺度的 GPU 記憶體錯誤現象。這些背景說明，對某些分析型工作負載來說，用「帶界限的答案」取代「單一精確標量」是有意義的。但這裡必須先講清楚：我們只是把它當成**研究動機**。本文**不主張**現有系統已經在 aged 或 faulty-HBM GPU 上部署 byte-plane analytics backend，也沒有在退化硬體上做評估。本文的實驗平台是正常 H200 硬體，fault 則是 software-level fault injection。文中提到的「reuse」和「reliability-constrained」，描述的是一種 workload 與經濟場景，不是本文真的測過的 aged-hardware 條件。

本文評估的 query 型態，是單欄位的 thresholded reduction：

```sql
SELECT COUNT(*), SUM(x), SUM(x) / COUNT(*)
FROM field
WHERE x > threshold;
```

實作上，我們把 predicate evaluation 和 aggregation 融合在一起，並以 SUM 為主角。COUNT 之所以一併報告，是因為這個 fused primitive 本來就會產生 COUNT，而且 qualified / disqualified / unresolved（Q/D/U）ambiguity 能幫助說明 predicate 的運作機制；但更核心、也更強的結果，仍然是 SUM-derived reduction。AVG 由 SUM 和 COUNT 推導而來。MIN、MAX、VAR、join，以及完整 DBMS 整合，都不在本文範圍內。

**本文的貢獻如下。**

1. **證明 H200 上的執行可行性，並說清楚它的邊界。** 我們在 H200 上實作並量測 BUFF-style progressive byte-plane fused Filter+Aggregate，報告在四個 field 上，*k*=2 相較於 raw-fused FP64 baseline 可得到條件式的 2.69×–6.14× 加速，並透過 locality accounting、單一具範圍限定的 physical-traffic 案例、k-depth dispatch 證據，以及明確 fallback 機制，說明它**在哪些情況下不再划算**。
2. **提出一個源自表示法本身的可靠性契約。** 我們利用同一套 significance ordering，在偵測到 corruption 後回傳經認證的 bounded degradation，並把它和 raw FP64 做公平比較，避免 baseline 變成稻草人：raw FP64 也能配上各種可靠性工具；它缺少的是在 detection 發生後，能由 representation 本身直接導出的 answer-impact decomposition。
3. **在明確範圍內評估 significance-aware redundancy。** 在 software/logical fault model 下，我們評估 majority voting、經 H200 NMR path 跑出的 structured fault family、logical placement diversity，以及 significance-aware protection allocation，並清楚區分每項結果到底證明了什麼、又沒有證明什麼。

## 2. 背景與相關工作

**表示法與語意血統。** BUFF 是最接近的 representation ancestor：它提供 byte-oriented bounded-float columnar storage，以及對浮點資料的 progressive query behavior [BUFF]。我們刻意沿用 BUFF-style 術語，因為 byte-oriented layout、progressive predicate decomposition，以及 deterministic omitted-tail SUM behavior 都是承襲而來，不是本文新創。DAQ 以 bounded answer 而不是 sampling error 來界定 deterministic approximate query processing [DAQ]；Column Sketch 則清楚揭示 tri-state predicate 結構，也就是在當前 approximation 下，一筆 row 可能是 definitely qualified、definitely disqualified，或 unresolved [ColumnSketch]。我們依賴的正是這種語意直覺：partial precision 帶來的是 bounded sum、確定分類與不確定列，而不是把精確答案偷偷藏起來。

**儲存與壓縮的鄰近工作。** ALP、FastLanes、G-ALP 與 BtrBlocks 改進的是浮點儲存、向量化解碼、block 組織或壓縮效率 [ALP, FastLanes, GALP, BtrBlocks]。它們是 representation 與 storage 上的鄰近工作，但不是本文的受控 runtime baseline；我們不主張在 compression ratio 或 runtime 上勝過這些系統。

**執行 baseline。** cuDF 是成熟的 GPU dataframe stack [cuDF]，可作為 raw FP64 dataframe-style execution 的外部參考；但它不是本文的主要 baseline，因為 dataframe API 可能會 materialize mask 或 survivor buffer。本文主要的 runtime baseline 是 **raw-fused CUDA**：在 raw FP64 value 上直接用單一 H200 kernel 套 threshold，並累積 COUNT 和 SUM，且**不 materialize** mask 或 survivor column。這樣的比較，才能把 byte-plane execution 本身的價值與成本，和一般 dataframe materialization 效果分開看。

**reuse 與 reliability 的動機來源。** 本文的 workload 動機主要來自三類外部資料：AI GPU 資本密集度與 accelerator 折舊／使用壽命的討論 [CoreWeaveFT, CoreWeaveBI, AIChipDepreciationWSJ]；NVIDIA 提供的 operational reliability surface，例如 page retirement、XID、NVML、DCGM [NVIDIAPageRetirement, NVIDIAXid, NVIDIADCGM]；以及大型 GPU 記憶體錯誤與 fault-injection 研究 [AmpereMemErrors, HardDataSoftErrors, GPUFaultInjection]。這些資料支撐的是 lifecycle 與 reliability **動機**，也說明使用 software fault injection 來測試 response contract 是合理的；但它們**不**代表已有部署實務，也**不**等同於 physical fault-rate model（§4）。

## 3. Byte-Plane Primitive 與它的雙重契約

### 3.1 執行契約

執行深度 *k* 是執行期的控制旋鈕。當 *k* 較淺時，operator 只讀取前幾個 byte plane，因此重建工作較少；當 *k* 增加時，它會讀取更多 encoded value。當 *k*=max 時，operator 到達 artifact 的最大 filter depth；而在目前的 progressive executor 中，*k*=max 仍然可能受益於 early exit，並不是人為強迫的 full scan。

這裡有四個軸線必須分開：**representation precision**（artifact 儲存時的精度）、**execution depth** *k*（operator 實際讀到多少前導 plane）、**query error**（相對於所選 reference 的 approximation），以及 **physical traffic**（GPU 在記憶體階層中實際搬了多少資料，這必須從 profiler counter 讀，而不能只用 logical GB/s 推估）。

### 3.2 可靠性契約

對任一 functional *F* 來說，每個 byte plane 都有可量化的 per-plane contribution ceiling U_i(F)；而這個 ceiling 正好就是用來界定 omitted-tail execution-depth error 的那個上界。因此，如果 corruption 被限制在 plane *i* 內，它最多能對 aggregate 造成的影響，就是 U_i(F)。這使 byte-plane artifact 擁有一種 fault-impact decomposition：representation 本身就能把「錯誤發生在哪裡」映射成「答案最多會被推偏多少」。Raw FP64 沒有這種固定映射；同樣是 mantissa bit flip，影響可能幾乎沒有，也可能非常嚴重，事前無法給出固定界限。

這裡的契約是操作性的，不是新的理論定理。Bound 公式和 predicate decomposition 都來自 BUFF / DAQ / Column-Sketch 血統；我們做的是把它們組織成一個實際 response：先 detect、再 classify，接著依情況擴大 certified interval、回退到較淺但仍有效的深度，或明確標記答案不可用，而**不是**靜悄悄地回傳錯誤標量。

### 3.3 raw-FP64 comparator（不是稻草人）

如果要談可靠性契約，就必須公平對待 baseline。Raw FP64 同樣可以搭配 ECC、checksum、replication、recomputation、checkpoint/restart，以及應用層 sanity check；這些都是合理手段，也完全可以和 byte-plane artifact 並用。真正的差別比較窄：**當錯誤已經被偵測到，但一時之間拿不到乾淨副本時，representation 本身能提供什麼樣的回應。**

| 契約 | 能偵測 corruption 嗎？ | 偵測後怎麼處理 | 能提供經認證的 degraded answer 嗎？ |
|---|---|---|---|
| Raw FP64，未檢查 | 否 | 可能默默汙染 scalar | 否 |
| Raw FP64 + digest | 是，在 digest 能力範圍內 | Abort / retry / recompute / output loss | 否，沒有 representation-derived interval |
| Raw FP64 + replication / recompute | 透過跨副本投票或重播 | 在 independent-fault / clean-copy 假設下可恢復 | 否，沒有 representation-derived interval（recover-or-fail） |
| Byte-plane + digest + certified bound | 是，在 digest 能力範圍內 | 擴大 bound 或回退到較淺有效深度 | **是**，利用 answer-impact ceiling |
| Byte-plane + logical NMR（範圍受限） | logical model 下投票與／或 digest | replica fault 獨立時可恢復；否則退回 bound | 是，作為範圍受限 fallback |

重點不是誰比較會做 detection，因為兩邊都可以加 integrity check。重點在於：significance-ordered representation 會暴露一個可操作的 answer-impact decomposition，因此 detected-but-unrepaired corruption 可以轉成 bounded interval，而不是只能變成 output loss。

### 3.4 保護配置與 logical placement

當額外保護儲存預算 *E* 有限時，就可以依 query functional 的敏感度，把 protection allocation 分配到不同 plane。對 SUM 來說，MSB 最重要（s_0 = 1.0、s_1 ≈ 0.0039、s_2 ≈ 1.5e-5，後面幾個 plane 幾乎可忽略）；對 filtered COUNT 來說，敏感度主要集中在 threshold 附近。Uniform allocation 則對應到平坦 sensitivity profile 的退化情況。本文不主張某個最佳化 solver；我們評估的是這個直覺是否成立：對 SUM-like functional 而言，significance-aware allocation 應優先保護高 significance plane（§6.3）。另一個相關問題是 *placement* diversity，也就是既有 replica 是否暴露在相同 fault domain。本文談到的 placement diversity 全都停留在 logical/software 層級，不主張任何 physical HBM independence（§6.2）。

## 4. 實驗設定、範圍與證據 provenance

**工作負載模型。** 本文聚焦在 warm、device-resident、針對已準備好的科學欄位反覆做 threshold analysis 的工作負載。Warm E2E timing 在適用時包含 per-query threshold preparation，以及 resident artifact 上的 GPU execution；它**不**包含 cold artifact construction、初始 host-to-device transfer、file I/O，以及 offline percentile selection，除非表格有另外註明。換句話說，如果某個 workflow 只會碰這個 field 一次，就不應直接套用本文的 speedup 敘述。

**硬體與資料集。** 除非特別標示，所有正式 GPU 測量都在 NVIDIA H200（SM 9.0、141 GB HBM3、CUDA 12.6）上進行。四個 headline execution field 都是 true-quasi-global single-segment artifact：`cesm_atm_cloud`（168.48M rows）、`cesm_atm_q`（168.48M rows）、`hurricane_u`（25M rows）與 `hurricane_tc`（25M rows）。`hurricane_tc` 使用 `bfp-dec11`；其餘三者使用 `bfp-dec12`。兩個 reliability field 為 `hurricane_u` 與 `cesm_atm_cloud`。

**哪些結果在哪裡量測。** Latency 和 traffic 是在 H200 上直接量到的。Certified-bound 與 SDC-containment 屬於 representation-level 性質：它們透過 CPU reference implementation 驗證，並在 H200 的 smoke 與 structured-fault path 中，以 software-level（logical）fault injection 進行演練。這種切分是刻意的：containment 是表示法在 injected fault 下的契約，而 latency 則取決於 GPU 記憶體階層。

> **範圍與非主張（這一段讀一次即可，全文都適用）。**  
> *執行面。* 四-field speedup 只在 warm residency、favorable locality 和 shallow *k* 下成立；我們不主張普遍的 scientific speedup、cold end-to-end acceleration，或一般性的 COUNT accuracy。k-depth dispatch 證據是 synthetic，並且刻意與四-field scientific headline 分開。NCU physical-traffic 只有單一 field case，因此本文不主張普遍的 physical-traffic reduction。  
> *可靠性。* 所有可靠性結果都建立在正常 H200 硬體上的 software-level / logical fault injection。我們**不主張**：physical HBM channel / bank / stack / SID placement diversity；temporal hardware diversity；aged- 或 faulty-HBM deployment；graded allocation 在 point-answer accuracy 或 total-repair-coverage 上優於 uniform；或已經有一條可部署、低 overhead 的 full-NMR path。  
> *動機。* 「Accelerator reuse / reliability-constrained / depreciated-GPU」描述的是 workload 與經濟場景，不是實際評估過的退化裝置條件。  
> *語意。* 本文沒有提出新的 bounded-float representation、deterministic SUM bound 或 predicate logic；這些都屬於承襲。

各項結果對應的 freeze report、job ID、verdict 與 CSV path，都集中整理在 **Appendix A（Evidence Index）**，讓正文可以用機制名稱敘述，而不用一直暴露內部 result code。

## 5. 執行契約

### 5.1 H200 上四個 field 的 shallow-*k* 加速

已凍結的 four-field rerun 報告的是：在 warm device-resident 條件下，相較於 raw-fused FP64 CUDA baseline 的 speedup。本文主要採用 dispatch-selected shallow row 的 *k*=2 結果；*k*=max row 則屬於 diagnostic / fallback 證據。headline 的 *k*=2 區間是 **2.69×–6.14×**。

| Field | Precision | *k*=2 shallow speedup | *k*=max diagnostic speedup | *k*=2 的 SUM error | COUNT |
|---|---|---:|---:|---|---|
| `cesm_atm_cloud` | `bfp-dec12` | 3.64–4.92× | 1.53–3.13× | <0.01%（測試 row 中 bit-exact） | 次要；在 s90 漂移到約 −11% |
| `hurricane_u` | `bfp-dec12` | 2.69–3.90× | 1.24–2.05× | <0.01%（bit-exact）；*k*=max 時 exact | 次要；測試 row 中接近零 |
| `cesm_atm_q` | `bfp-dec12` | 3.58–6.14× | 1.66–4.96× | ≤4.4e-8%（machine-eps）；*k*=max 時 exact | 次要；高 selectivity bucket 漂移較大 |
| `hurricane_tc` | `bfp-dec11` | 2.94–5.72× | 1.58–4.57× | <3e-5% relative；*k*=max=6 時 exact | 次要；*k*=max=6 時 exact |

6.14× 這個上界來自 `cesm_atm_q` 的某個高 selectivity bucket；該 row 在 *k*=2 與 *k*=max 下的 selectivity label 並不相同（65.02% vs 90.00%，原因是 threshold-encoding drift），所以這裡應被解讀成 bucket 證據，而不是 paired-threshold comparison。本文真正要強調的是 SUM：在 *k*=2 下，四個 field 的 SUM relative error 都低於 5e-8%，其中三個 field 甚至是 bit-exact。COUNT 則從一開始就是次要結果，因為在 shallow *k* 且高 selectivity 時，unresolved row 的影響可能讓 COUNT drift 直接到數十個百分點（例如 `cesm_atm_q` 的高 selectivity bucket 出現 −27.75%），這也正是為什麼本文的科學性主張要建立在 SUM-derived reduction，而不是 COUNT 上。

因此，這裡較準確的解讀是：byte-plane execution 的確能在特定條件下打敗同一張 GPU 上、夠強的 raw-fused CUDA primitive，但這是一個**有條件成立的執行契約**。

### 5.2 什麼時候不再划算：locality、traffic、dispatch 與 fallback

**Locality 是關鍵。** Threshold preparation 是 host-side 成本，而且會隨 segment 數量放大。Single-segment（true-quasi-global）artifact 可以把 preparation 壓到 0.002–0.023 ms，低於 0.06 ms 上限，因此相對於 GPU execution 幾乎可忽略；但對 4096-segment 的 fine-local artifact 而言，這個成本在 168M-row CESM field 上會拉高到 8.08–13.97 ms，在 25M-row Hurricane field 上也有 0.55–0.57 ms。換句話說，segment granularity 是 deployment parameter，不是 representation guarantee：如果 workload 會重複查詢同一份 artifact，一次性的 preparation 成本有機會被攤掉；如果是 ad-hoc 工作負載，就應該選擇 coarse 或 single-segment locality。也就是說，就算 byte-plane executor 省下了 GPU traffic，只要 host-side preparation 變成 E2E latency 的主因，它一樣可能輸掉。

**一個範圍限定的 physical-traffic 案例。** 在 `cesm_atm_q` 的 s10 threshold 上，*k*=2 的 byte-plane fused path 透過 DRAM proxy 讀取 186.49 MB，而 raw-fused FP64 則讀取 1,382.40 MB，差距是 7.41×；L2 與 L1/TEX sector counter 也朝同一方向變化。這是一個很有用的 attribution 訊號，但它終究只是一個 scientific-field 的 NCU case；我們沒有進一步擴展它（因為沒有新的凍結 NCU 資料），因此本文不主張普遍的 physical-traffic reduction。

**k-depth dispatch。** 在兩組 synthetic distribution（`uniform_p10` 和 `heavy_tailed_p6`）上的 200 次 warm-cache microbenchmark 顯示，byte-plane execution 一直到 *k*=max 仍然快於 raw-fused FP64，但優勢會隨 *k* 增大而穩定縮小：

| *k* | `uniform_p10` | `heavy_tailed_p6` | Dispatch 解讀 |
|---:|---:|---:|---|
| 1 | 4.19× | 4.46× | 強勢優勢 |
| 2 | 2.85× | 2.81× | 強勢優勢 |
| 3 | 2.36× | 2.12× | 中度優勢 |
| 4 | 2.02× | 1.70× | 中度優勢 |
| max | 1.56× | 1.27× | 小幅／邊際優勢 |

這裡要特別把 physical traffic 和 latency 分開看：在同一組矩陣中，DRAM read 相對 raw 的比例，會從 *k*=1 時的 0.13，上升到 *k*=max 時約 0.75–0.79。也就是說，traffic 節省還在，但光靠「省了多少流量」並不足以決定 dispatch。本文採用的是經驗式 dispatch rule：*k*=1–2 屬於強勢優勢區、*k*=3–4 屬於中度優勢區，而 *k*=max 不作為 headline 結果；一旦校正過的量測顯示 byte-plane execution 不再值得，raw FP64 fallback 就應該啟動。

## 6. 可靠性契約

### 6.1 Single-replica detect-and-bound（最保守也最核心的主幹）

最保守的路徑，是用 single-replica artifact（`r=[1]*8`）搭配 per-plane SUM32 digest detection 與保守的 certified bound widening，不做 voting。在兩個 reliability field 上，於三個 fault rate（1e-6、1e-5、1e-4）和三個 seed 下，再加上 same-fault-all negative control（總共 24 個 cell）的 software-level logical fault injection 中，每一個 cell 都是 **detected（24/24）、truth-containing（24/24）、bounded（24/24）**，而且 silent-wrong、unavailable 和 false-recovery 全部為零。這正是可靠性契約最保守、也最重要的主張：一個已經被偵測到的 corruption，不會變成未檢查的錯誤 scalar，而是被轉成一個包含真值的回傳區間。

這條路徑有兩個邊界要先說清楚。第一，SUM32 是模 2^32 的加法，因此若同一個 allocation unit 內出現大小相反的 byte delta，理論上可能彼此抵銷；改用非加法式的 lightweight digest，雖然能以低成本消除測試過的 SUM32-specific cancel replay set，但這不等於對一般 adversarial robustness 提供完整保證。本文的契約是在已測 random-fault model 下的 detect-or-bound，不是對抗性完備性。第二，保守 bound（對每個被偵測 plane 使用 `255 × weight × rows`）本來就刻意放得很寬；若用 per-segment refinement，仍可保持 containment，同時把這個 bound 收緊好幾個數量級（§6.3）。

### 6.2 透過 H200 NMR path 檢驗 structured fault 與 logical placement

除了 single-replica detection 之外，我們也進一步問：在更有結構的 fault 下，replication 與 voting 能不能帶來更好的回應？這裡的 fault 仍然是 software/logical 層級。Pre-fused H200 NMR pipeline（`replicate → vote → digest/detect → certified bound → classify`）被拿來測七種 HBM-inspired structured fault family：single-domain burst、regional multi-byte、column-like repeated offset、plane-localized MSB、plane-localized LSB、same-fault-all correlated control，以及 mixed independent-plus-correlated。總共 210 個 configuration 中，GPU path 和 CPU oracle **在 210/210 個 classification 上完全一致**；truth containment 也是 210/210，且 silent-wrong、certified-bound escape、hard failure，以及 correlated control 下的 false recovery 全部為零。整體來看，25.7% 的 configuration 可以靠 vote 恢復，74.3% 則會退化為 certified bounded-degraded；恢復主要發生在高 significance plane 且只打到單一 replica 的情況，至於 LSB-localized 和 mixed fault，則多半是被 bounded，而不是直接 recovered。same-fault-all control 從未被誤判為 recovered，這剛好說明了這條邊界：majority voting 不能被假設能修復所有 replica 都遭到相同 corruption 的情況。

另一個 logical-placement 比較則檢查：現有 replica 是否其實暴露在同一個 fault domain。在 `graded_seg_B3` map 上做 structured clustered software injection 時，可以看到 logical replica 如果 colocated，all-replicas-hit rate 最高；若在邏輯上把 replica 分開，這個比例會明顯下降：

| Field | no-diversity | logical-spatial | logical-spatial + logical-temporal |
|---|---:|---:|---:|
| `hurricane_u` @ 1e-4 | 0.338 | 0.039 | 0.006 |
| `cesm_atm_cloud` @ 2e-4 | 0.560 | 0.175 | 0.038 |

這支持了 diversity-aware redundancy 的**方向是對的**。但這裡要再次強調：這仍然只是 logical/software 證據。這裡的 spatial 和 temporal 標籤描述的是 software model（其中 temporal 條件是一個每 replica 50% 的 masking model），不是對 physical HBM channel / bank / SID independence 或硬體 temporal behavior 的實測。

### 6.3 Significance-aware protection allocation

Protection allocation 這個問題，和 voting correctness、placement diversity 並不是同一層。先前工作先從反面把 framing 釐清：在修正過、且儲存量匹配的 evaluator 下，graded allocation 和 uniform allocation 在 repair 與 delivered-answer accuracy 上基本等價，差異只落在雜訊範圍內。因此，真正有辨識度的差異，不是「修好了多少 fault」，而是「哪些 significance 區域被保護」以及「留下來的 certified error 有多大」。

已凍結的 D2 比較，使用 segment-level protection map，在 matched **E=3 extra storage（8 個 plane 之外再增加 11 個 plane-replica unit）** 的條件下，針對 single-replica logical injection 與固定 protection-map seed 進行評估。Graded map 對所有 plane-0 segment 給 r=3 保護，並對約 50% 的 plane-1 segment 給 r=3；uniform-spread map 則是在每個 plane 中，對約 18.75% 的 segment 給 r=3。凍結的 verdict 是 **`GRADED_MSB_BOUND_WINS`**：在相同儲存成本下，graded protection 覆蓋了更多 MSB segment，並在 rate-level aggregate 上得到 **345×–432× 更窄的 certified bound**（per-seed row 則落在較寬的 256×–445× 範圍內）。工具的 CSV verdict 欄位會輸出 `GRADED_WINS`；但本文只把它解讀成它實際證明的內容，也就是 **MSB coverage 與 bound-width 的勝出**，而不是 point-answer accuracy 或 repair coverage 的勝出。這個方向在三種穩定的 encoded-domain normalizer 下都成立。完整的四列結果、comparator taxonomy 與 normalizer 檢查，見 Appendix A。

更精確地說，這個結果的含意是：significance-ordered representation 讓 protection allocation 有了明確的 **answer-impact objective**，也就是在固定 storage 下，優先保護 high-contribution plane，從而把 certified bound width 壓到最小。它並不表示 byte-plane 是唯一能做 protection-aware 設計的 representation，也不等於 physical layout robustness 已經被驗證。

### 6.4 成本邊界

目前的 full-NMR path 還不能當成部署結果。未融合的 vote/read-compare path 在測試 field 上仍然約為 **2.5× raw FP64**，digest-plus-vote path 也仍高於 raw。Single-replica detection 的成本很低（平行 SUM32 只會寫出小型 digest，因此仍落在 raw-fused baseline 的讀取預算內），但 voting 並不便宜。雖然 fused tracer-bullet prototype 暗示 plane-scoped fusion 可能有路，但這還不足以說明最終優化 kernel 已經完成。因此，本文最強的可靠性主張仍然是 response contract（§6.1），而不是低 overhead 的 full NMR。

## 7. 討論與限制

前面的 consolidated scope box（§4）已經列出主要非主張；這裡補充幾個特別重要的原因。

**動機和評估要分開看。** reuse / reliability-constrained framing 是設計動機，而且有 lifecycle 與 memory-error 證據支撐；但它不是本文真的評估過的條件。我們使用的 H200 不是 aged、faulty 或 degraded，fault 也都是以 software 注入。若未來要走向真實部署，還需要 GPU error telemetry、retired-page / channel / SID locality mapping，以及在 aged hardware 上重新驗證這套 response contract。

**Software fault injection 的角色。** software-level 的 byte / plane injection 很適合測試**回應契約**：被注入的 corruption 應該被 recover、被 bounded，或被標記 unavailable，而不是被 silent absorb。但它不會重現真實場域的 HBM fault-rate distribution，也不能拿來證明 physical placement 或 temporal hardware diversity。same-fault-all correlated corruption 在本文裡只是 voting 的 negative control；在那種情況下，containment 依賴的是 digest detection 加 certified-bound fallback，而不是 vote 本身。

**一般性仍有限。** 四個 scientific field 足以支撐一篇 workshop 規模的 systems result，但還不夠支撐普遍性的科學工作負載結論。單一 NCU case 提供的是 attribution，而不是普遍 traffic reduction 的證明。k-depth 與 logical-placement 結果都是 synthetic / logical model，因此才會被刻意和 scientific headline、physical HBM claim 分開寫。本文展示的可靠性契約目前只在 byte-plane artifact 上成立；是否能推廣到其他同樣暴露 contribution ceiling 的 progressive representation，屬於未來工作。

**Baselines 的邊界也要說清楚。** raw-fused FP64 CUDA primitive 是一個強的 same-GPU baseline；raw-FP64 reliability comparator（§3.3）則是為了避免 baseline 變成稻草人而特別補上的。最佳化的 AVX2 / AVX-512 / multithreaded CPU BUFF baseline，以及完整 cuDF pipeline 比較，都不在本文範圍內；因此我們不主張打敗最佳化 BUFF，而先前用到的 scalar / native CPU proxy 也只是 deployment-sanity check，不是 headline。

## 8. 結論

過去大家常把 byte-plane analytics 看成一種 compressed execution format。本文認為，更貼切的理解方式，是把它看成一個具 fault-awareness 的 progressive **primitive**，因為同一個核心性質，也就是 significance ordering，會同時暴露出 execution depth 和 fault impact。對四個 CESM/Hurricane field 而言，這使得 H200 executor 能在 shallow *k*、且 locality 與 fallback 條件都有利的情況下，相較於強健的 raw-fused FP64 baseline 拿到 2.69×–6.14× 的優勢。同樣的 ordering 也讓 detected corruption 可以被轉成 certified bounded degradation：single-replica pipeline 在測試矩陣中，把每個 injected fault 都限制成沒有 silent-wrong 的 bounded answer；structured-fault campaign 在所有測試分類上都與 CPU oracle 一致；而 significance-aware allocation 則在相同 storage 下帶來數百倍更緊的 certified bound。這些可靠性結果都只是 software/logical-fault 證據；physical HBM diversity、temporal hardware diversity、aged-GPU deployment，以及低 overhead 的 full NMR，仍是未來工作。本文真正的貢獻不是新的 representation，也不是新的 bound，而是指出：單一 significance-ordered representation 可以同時支撐 progressive execution 與 fail-soft reliability 這兩種可組合的 systems contract，因此它很可能成為 reused accelerator fleet 上、面向較低優先級但高帶寬分析工作的有力基底。

## 9. 未來工作

**Physical 與 temporal diversity。** 後續需要把 replica buffer 映射到 HBM placement / fault domain（channel / bank / stack / SID / page-retirement），或至少明確指出 placement 仍未知；同時也要在真實硬體上量測 staggered-read / staged-write 的 temporal behavior。現有 logical-placement 結果只能說明這個方向值得做，還不能當成驗證。

**真實 aged-GPU 評估。** 未來需要在 aged device 上收集 `nvidia-smi` / DCGM ECC、page-retirement telemetry 和 Xid log，並根據實測 fault locality，而不是 software injection，重新驗證這套 response contract。

**可部署的 NMR。** 後續值得發展 plane-scoped 的 fused vote+detect+bound kernel，並補上完整資料集的最佳化 GPU timing，同時設計 allocation policy，讓低 significance plane 可以留在成本較低的 single-replica path。

**Allocation 與 digest 強度。** 之後可擴充 map-seed sweep、加入 MSB-only 與 greedy-sensitivity comparator、納入 encoded-domain 以外的 query / application normalizer，並在 adversarial 與 multi-byte fault model 下測試更強的非加法式 digest。

**MAX 與 top-k 的 bound。** MAX / top-k 的 closed-form U_i ceiling 仍需要修正後的推導：對帶有負值的資料集來說，active_count=1 的假設不成立，因為 identity-shift 需要追蹤 second-largest value，而不能只是把上界簡單放大到 n_rows。

## Appendix A. 可靠性與證據細節

### A.1 Evidence index

正文中的每個機制名稱，都對應到一份已凍結的 evidence report。正文用機制名稱敘述，本表則保留 traceability。

| Mechanism（正文名稱） | Evidence freeze | Job / verdict | Scope |
|---|---|---|---|
| Four-field shallow-*k* execution（§5.1） | R7 four-field execution freeze | job 91065, `CONFIRMS_WITH_REVISED_RANGE`（2.69×–6.14×） | H200、四個 field、warm resident |
| Locality / threshold-prep boundary（§5.2） | R8 locality freeze | job 90995, `CONFIRMS_LOCALITY_BOUNDARY` | 54 個點、兩個 field |
| Scoped physical-traffic case（§5.2） | pre-existing Q8 / Prompt5 NCU | 一個 `cesm_atm_q` s10 case（無新 NCU freeze；R9 缺席） | 單一 field，僅 attribution |
| k-depth dispatch（§5.2） | R10 k-depth freeze | `CONFIRMS_WITH_REVISED_BREAKPOINTS` | synthetic warm-cache、200 次迭代 |
| Single-replica detect-and-bound（§6.1） | R11 single-replica freeze | `CONFIRMS_DETECT_AND_BOUND_CONTRACT`（24/24 bounded） | logical fault injection |
| Structured-fault NMR path（§6.2） | R12 structured-fault freeze | job 90969, `CONFIRMS_STRUCTURED_FAULT_CONTRACT`（210/210） | 7 類 fault family、logical |
| Logical placement diversity（§6.2） | R4 logical-diversity handoff | jobs 90944/90986，僅 logical/software | clustered software fault |
| Significance-aware allocation（§6.3） | R13 D2 freeze | job 91072, `CONFIRMS_GRADED_MSB_BOUND_WINS`（345×–432×） | matched E=3、固定 map seed |
| Raw-FP64 reliability comparator（§3.3） | R5 comparator note | 契約比較，不是 result | — |
| Reuse / reliability motivation（§1） | R6 motivation audit | `USE_WITH_SCOPED_MOTIVATION_LANGUAGE` | 僅動機 |

### A.2 D2 graded-vs-uniform allocation（完整表格）

Matched E=3 extra storage；single-replica logical fault injection；固定 protection-map seed。Relative bound 以 clean encoded sum 的百分比表示。

| Dataset | Rate | Graded MSB cov. | Uniform MSB cov. | Graded rel. bound | Uniform rel. bound | 解讀 |
|---|---:|---:|---:|---:|---:|---|
| `hurricane_u` | 2e-5 | 0.078–0.079 | 0.0005–0.0149 | 0.04%–0.08% | 17%–21% | certified bound 較小 |
| `hurricane_u` | 2e-4 | 0.557–0.561 | 0.073–0.102 | ~0.3% | 123%–132% | certified bound 較小 |
| `cesm_atm_cloud` | 2e-5 | 0.0786–0.0787 | 0.0007–0.0144 | 0.10%–0.20% | 44%–53% | certified bound 較小 |
| `cesm_atm_cloud` | 2e-4 | 0.558–0.560 | 0.087–0.106 | ~0.75% | 307%–321% | certified bound 較小 |

Rate-level aggregate 的 bound-width ratio（uniform/graded）：`hurricane_u` 約 345×（2e-5）／約 432×（2e-4）；`cesm_atm_cloud` 約 348×（2e-5）／約 419×（2e-4）。在 matched E=3 下，repair coverage 相近，因此它不是 verdict metric。三種穩定 encoded-domain normalizer（clean_encoded_sum、max_possible_encoded_sum、encoded_range）下的 verdict 都是 `GRADED_MSB_BOUND_WINS`，共 12/12 個 cell。Raw-domain denominator 把 decoded FP64 單位與 encoded bound 單位混在一起，因此只保留作為 sign-stable 的 stress check，而不是 paper-facing ratio；thresholded normalizer 因 D2 不執行 thresholded query，所以不可用。

### A.3 Reliability comparator taxonomy

| Comparator | 定義 | 它證明什麼 | 它沒有證明什麼 |
|---|---|---|---|
| Single-replica spine | `r=[1]*8`；digest detect → certified bound widening | 偵測到的 corruption 會回傳 bounded interval | recovery、voting、physical diversity |
| Historical plane-level comparator | `r=[2,1,2,1,2,1,1,1]`，總 B=11 | 早期的 storage-matched reference | 真正的 uniform spread 或 repair superiority |
| Corrected uniform-family evaluator | 修正後的 repair / delivered-accuracy evaluator | graded≈uniform，在 repair / accuracy 上差異只在雜訊內 | graded 有 accuracy 優勢 |
| D2 true uniform spread | segment-level 約 18.75% r=3 in every plane，E=3 | 公平的 spread baseline | physical HBM robustness 或 timing |
| `graded_seg_B3` | 100% plane-0 + 約 50% plane-1 為 r=3，E=3 | 更多 MSB segment、更小 certified-bound width | total repair-coverage superiority |

## 參考文獻

- [BUFF] Liu et al. BUFF byte-oriented bounded-float compressed query processing. PVLDB, 2021.
- [DAQ] Potti and Patel. DAQ deterministic approximate query processing over bit-sliced indexes. SIGMOD, 2015.
- [ColumnSketch] Hentschel et al. Column Sketches: a scan accelerator for predicate evaluation. SIGMOD, 2018.
- [ALP] Afroozeh, Kuffo, and Boncz. ALP: Adaptive Lossless Floating-Point Compression. SIGMOD, 2024.
- [FastLanes] Afroozeh, Felius, and Boncz. Accelerating GPU Data Processing using FastLanes Compression. DaMoN, 2024.
- [GALP] Hepkema et al. G-ALP: GPU-oriented ALP and FastLanes floating-point compression. DaMoN, 2025.
- [BtrBlocks] BtrBlocks: Efficient Columnar Compression for Data Lakes. SIGMOD, 2023.
- [cuDF] RAPIDS. cuDF / libcudf documentation and software.
- [CoreWeaveFT] Financial Times. Eight odd things in CoreWeave's IPO prospectus. 2025.
- [CoreWeaveBI] Business Insider. CoreWeave IPO debut: a potentially expensive hardware problem. 2025.
- [AIChipDepreciationWSJ] Wall Street Journal. The Accounting Uproar Over How Fast an AI Chip Depreciates. 2025.
- [NVIDIAPageRetirement] NVIDIA. Dynamic Page Retirement documentation.
- [NVIDIAXid] NVIDIA. Xid Errors documentation.
- [NVIDIADCGM] NVIDIA. DCGM Field Identifiers documentation.
- [AmpereMemErrors] Zhu et al. Understanding the Landscape of Ampere GPU Memory Errors. arXiv:2508.03513, 2025.
- [HardDataSoftErrors] Haque and Pande. Hard Data on Soft Errors. arXiv:0910.0505, 2009.
- [GPUFaultInjection] Guerrero-Balaguera et al. GPU permanent-fault / software-injection reliability studies. arXiv:2306.10856, 2023; arXiv:2205.12177, 2022.

*v1.4 synthesis 的證據 provenance：R4–R13 freeze report 與 R5/R6 audit 位於 `research/2026-06-11_*`，凍結 CSV 位於 `results/v1_3_freeze/`（詳見 Appendix A.1）。*
