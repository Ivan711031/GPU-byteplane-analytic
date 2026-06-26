# Spec: Exp4 Bounded-Precision Codec — Local Phase 1 (Mac)

## Objective

在 **不依賴 H200 / Slurm / 100M /work 資料集** 的前提下，先於 Mac 完成 bounded-precision codec 語義落地與本地驗證，建立可審閱、可回滾的本地提交檢查點（local commit only, no push）。

---

## Scope

### In Scope (Mac local)

1. C++ codec bounded-precision mode 實作（保留 exact-dyadic 預設行為）。
2. tiny A/B dataset 的 encode/decode 與 threshold semantics 診斷。
3. tiny artifact 匯出與 schema/layout 檢查。
4. spec/research 文檔更新，明確分離 Local CPU gate 與 Nano4 GPU gate。
5. local commit（不 push）。

### Out of Scope (Phase 1 不做)

1. Slurm submission。
2. H200 GPU kernel correctness/throughput 驗證。
3. full 100M DEV artifact 正式 export。
4. Exp4 v1b selectivity sweep。

---

## Core Semantics

### 1) Default behavior must stay exact-dyadic

- 既有呼叫路徑（未指定 precision）維持原行為。
- 目標：對現有 exact mode，plane_count / decode 結果 / threshold encoded-domain 行為不可回歸。

### 2) Bounded mode bit semantics

bounded mode 不覆蓋語義，必須同時保留：

- `raw_fractional_bits`: 由資料 dyadic 分解得到的原始 fractional bits（原本算法）。
- `precision_cap_bits`: 由 precision 設定映射出的上限 bits（來自 `PRECISION_MAP`）。
- `effective_fractional_bits`: `min(raw_fractional_bits, precision_cap_bits)`。

#### `PRECISION_MAP` / `precision_cap_bits` 規則

Phase 1 的 canonical 規則：

```text
precision_cap_bits = ceil(decimals * log2(10)), decimals >= 0 and integer
```

`PRECISION_MAP` 表格只作為預期值文件，不是第二個 source of truth。

| CLI `--precision-decimals` | `precision_cap_bits` | 對應十進位解析度（約） |
|---------------------------|---------------------|----------------------|
| 0 | 0 | ~1 |
| 1 | 4 | ~0.1 |
| 2 | 7 | ~0.01 |
| 3 | 10 | ~0.001 |
| 4 | 14 | ~0.0001 |
| 5 | 17 | ~0.00001 |
| 6 | 20 | ~0.000001 |

#### `plane_count` 推導

```
total_bits = effective_fractional_bits + integer_offset_bits
plane_count = max(1, ceil(total_bits / 8))
```

`plane_count` 是 **byte-plane 數**，不是 bit 數。

`integer_offset_bits` 必須根據 `effective_fractional_bits` 下的 bounded fixed-point encoded offset range 重新計算；除非診斷證明完全相同，不可盲目沿用 exact-dyadic 的 `integer_offset_bits`。

### 3) What must use effective_fractional_bits

以下實際編碼語義一律使用 `effective_fractional_bits`：

1. value 編碼 scale exponent
2. integer/fractional split
3. `integer_offset_bits` 推導
4. `plane_count` 推導
5. `plane_basis` 計算
6. decode 重建語義

### 4) Decode error bound

實作必須明確記錄 bounded fixed-point conversion 採用：

- truncation/floor，或
- round-to-nearest。

```
quantum = 2^(-effective_fractional_bits)
```

若使用 truncation/floor：

```text
abs(decoded - original) < quantum
```

若使用 round-to-nearest：

```text
abs(decoded - original) <= quantum / 2
```

> 註：segment base 是平移，不改變量化解析度，不應納入誤差上界公式。

---

## API / CLI / Metadata Changes

## A. Codec API (buff_encoder)

- 在 `encode_segment` / `encode_file` 增加可選 encode config（bounded mode 參數）。
- 不破壞舊呼叫；無 config 時為 exact-dyadic。

## B. CLI behavior

- `buff_tool encode` 增加 bounded mode 參數 `--precision-decimals N`（Phase 1 固定此名稱）。
- 未提供該參數時，行為必須與目前一致。
- Phase 1 不引入 `--scale`（除非現有程式已存在可直接重用的 scale abstraction）。
- CLI diagnostics 需可顯示 mode 與有效 bit 語義（至少於本地診斷輸出可見）。

## C. Artifact metadata (tiny export path + exp3 layout)

`segment_meta.csv` / `summary.json`（必要時 `manifest.json`）需記錄：

- `raw_fractional_bits`
- `precision_cap_bits`
- `effective_fractional_bits`

相容性要求：

- 既有讀取器不可因新欄位崩潰。
- 既有 `fractional_bits`（legacy column）消費端若仍存在，語義應對應 `effective_fractional_bits`。

---

## Local Diagnostics Plan (Mac)

## 1) Tiny codec diagnostics

使用 tiny A/B datasets（含 near-threshold 值）比較 exact vs bounded：

1. plane_count 變化（bounded 應在預期案例下降）。
2. decode error 是否滿足 precision bound。
3. threshold 語義：`cpu_encoded_count` 必須與 manual encoded-domain reference 完全一致；`cpu_encoded_count` vs `cpu_raw_count` 僅報告差異並做 near-threshold 解釋。

## 2) Tiny artifact validation

輸出（範例）：

- `tmp/tiny_buff_exact/`
- `tmp/tiny_buff_p3/`
- `tmp/tiny_buff_p6/`

每個 artifact 至少包含：

- `manifest.json`
- `summary.json`
- `segment_meta.csv`
- `plane_000.bin ...`

檢查：

1. schema 欄位完整（含三個 fractional bit 欄位）。
2. `active_plane_count` / `plane_basis` 合理。
3. decode 能還原 bounded-domain 值。
4. （記錄項，非 gate）bounded vs exact 的 encode time / decode time 與 artifact size 比例，供 Phase 2 規劃參考。

## 3) Exp4 threshold semantics (CPU-side only)

在 Mac 上跑 CPU path（不碰 GPU）驗證：

1. threshold fp64 -> encoded code
2. encoded code -> per-plane threshold bytes
3. row code vs threshold code 的比較邏輯
4. `cpu_raw_count`、`cpu_encoded_count`、near-threshold rows 記錄

並明確標註：此結果僅代表 local CPU semantics，不代表 GPU smoke。

---

## Acceptance Gates

## Mac Local Gate（必須全通過）

1. exact-dyadic default behavior preserved
2. bounded mode plane_count decreases where expected
3. decode error <= precision bound
4. CPU encoded-domain threshold count exactly matches manual encoded-domain reference
5. `cpu_encoded_count` vs `cpu_raw_count` 僅作 report，不作 fail gate；若不一致，需以 bounded precision 與 near-threshold rows 解釋
6. tiny artifact layout valid
7. report 明確寫出 **Mac local pass != GPU smoke pass**

## Nano4 Gate（Phase 2）

1. full bounded artifact export succeeds
2. H200 Exp4 smoke passes
3. `gpu_count == cpu_encoded_count`

---

## Delivery Workflow

1. 完成 Phase 1 實作與本地驗證。
2. 更新 spec/research。
3. **local commit only（不 push）**。
4. 回報：commit hash、changed files、local 測試結果、報告路徑。
5. 待 review 同意後才 push，然後再進 nano4 Phase 2。

---

## Notes / Constraints

- Phase 1 不得將 local 驗證包裝成 GPU correctness 結論。
- 不需要 H200、不需要 Slurm 的工作，一律先在 Mac 收斂。
- 本 spec 對應當前實作策略；若 `precision_cap_bits` canonical 規則需調整，必須先更新本 spec 再實作。
- **Legacy reader 相容性：** Phase 1 內需確認 exp3 artifact reader、segment_meta.csv consumer 能 forward-compatible 忽略新欄位；若不能，需於 Phase 1 一併修復。`fractional_bits`（legacy column）的語義對應關係應在 consumer 層以 alias 或 mapping 處理，避免破壞既有 pipeline。
- **Artifact 安全邊界：** Phase 1 不可覆蓋既有 `dev_buff_exp3` 或任何既有 exact-dyadic artifact。bounded tiny artifacts 僅能寫入 `tmp/` 或 `results/local_phase1/`。
- **預期收益（Phase 2 規劃參考）：** `plane_count` 下降將直接減少 `plane_*.bin` 檔案數與總大小。以 `precision_decimals=3`（`precision_cap_bits=10`）且 `raw_fractional_bits=52` 為例，fractional precision 下降 42 bits；實際 byte-plane 下降量為：
  ```text
  ceil((integer_offset_bits + 52) / 8) - ceil((integer_offset_bits + 10) / 8)
  ```
  因此通常是下降數個 byte-planes，而不是下降 42 個 planes。
