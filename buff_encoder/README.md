# BUFF Encoder in `buff_encoder/`

## TLDR;

`buff_encoder/` 這個資料夾裡的 BUFF 相關程式，主要分成三個部分：

- [buff_codec.hpp](buff_codec.hpp)
- [buff_codec.cpp](buff_codec.cpp)
- [buff_tool.cpp](buff_tool.cpp)

最常用的 CLI 是：

```bash
./bin/buff_tool encode --input INPUT.f64le.bin --output OUTPUT.buff64 --segment-size 4096 --precision-power 6
./bin/buff_tool inspect --input OUTPUT.buff64
./bin/buff_tool decode --input OUTPUT.buff64 --output ROUNDTRIP.f64le.bin
./bin/buff_tool export-runtime --input OUTPUT.buff64 --raw-input INPUT.f64le.bin --out-dir RUNTIME_ROOT --dataset DATASET_NAME
```

如果你只想記一件事：

> `.buff64` 是目前這版 C++ Rust-like BUFF encoder 的單一 container。
> 如果要給 Exp3 / Exp4 benchmark 用，還要再跑一次 `export-runtime`。

---

## Why:

這份 README 是寫給之後要直接碰 `buff_encoder/` 裡 encoder 的人看的。

它回答兩個問題：

1. **這個 BUFF encoder 怎麼用？**
2. **`.buff64` 要怎麼轉成目前 Exp3 / Exp4 可直接讀的格式？**

---

## What:

目前 `buff_encoder/` 這版 encoder 的定位是：

- bounded fixed-point
- whole-code offset
- MSB-first byte planes
- segment-local encoding

它不是：

- original exact dyadic artifact path
- `buff-major`
- direct `paradedb/buff-rs` wire format

目前限制：

- 只支援 `finite`
- 只支援 `non-negative`
- 如果 fixed-point 之後超出 `i64` 安全範圍，會直接報錯

---

## Files

### 1. [buff_codec.hpp](buff_codec.hpp)

公開 API 定義在這裡。

重要型別：

- `EncodeConfig`
- `PrecisionChoice`
- `EncodedSegment`
- `EncodedFileHeader`

重要函式：

- `encode_segment(...)`
- `decode_segment(...)`
- `decode_segment_top_k(...)`
- `segment_max_abs_error_bound(...)`
- `encode_file(...)`
- `decode_file(...)`
- `read_file_header(...)`
- `export_runtime_layout(...)`

### 2. [buff_codec.cpp](buff_codec.cpp)

真正的編碼/解碼/匯出實作都在這裡。

目前 `.buff64` 的 binary segment payload 也由這裡定義。

### 3. [buff_tool.cpp](buff_tool.cpp)

命令列工具入口。

目前支援四個子命令：

- `encode`
- `inspect`
- `decode`
- `export-runtime`

### 4. [buff_error_study.cpp](buff_error_study.cpp)

Exp2 error study 工具。

不是 encoder 本體，但會直接呼叫 `buff_codec` 做：

- `SUM`
- `COUNT`
- `AVG`
- `MIN`
- `MAX`
- `VAR`

的 bounded / top-k 實驗。

---

## How

## 1. Build

先編譯工具：

```bash
make buff
```

成功後會得到：

- [bin/buff_tool](/Users/heycool/Documents/codex_workspace/bin/buff_tool)

---

## 2. Encode raw FP64 to `.buff64`

### 指定 precision

```bash
./bin/buff_tool encode \
  --input data/dev/sensor.f64le.bin \
  --output data/tmp/sensor_p6.buff64 \
  --segment-size 4096 \
  --precision-power 6
```

### 自動選 precision

```bash
./bin/buff_tool encode \
  --input data/dev/sensor.f64le.bin \
  --output data/tmp/sensor_auto.buff64 \
  --segment-size 4096
```

參數說明：

- `--input`
  - raw little-endian `f64` binary
- `--output`
  - `.buff64` 輸出檔
- `--segment-size`
  - 每個 segment 的 row 數
- `--precision-power`
  - decimal precision level `p`
- `--max-values`
  - 只編前 N 筆，常用於 smoke test

---

## 3. Inspect `.buff64`

```bash
./bin/buff_tool inspect --input data/tmp/sensor_p6.buff64
```

目前只會顯示 file-level header，例如：

```text
total_values=100000000 segment_size=4096 segment_count=24415
```

注意：

- 這個 `inspect` 不會列出 per-segment 的 `plane_count`
- 如果你要詳細 metadata，要讀 `.buff64` payload 或看 exported runtime layout

---

## 4. Decode `.buff64`

```bash
./bin/buff_tool decode \
  --input data/tmp/sensor_p6.buff64 \
  --output data/tmp/sensor_p6.decoded.f64le.bin
```

這是 **full-depth bounded roundtrip**，不是 raw-exact roundtrip。

也就是：

- decoded 不一定完全等於 raw
- 但應滿足 bounded precision 語意

---

## 5. 轉成 Exp3 / Exp4 可直接讀的 artifact root

這一步很重要。

`.buff64` 本身是單一 binary container，**目前的 Exp3 / Exp4 loader 不能直接吃它**。
benchmark 需要的是這種 shared runtime layout：

```text
RUNTIME_ROOT/
  manifest.json
  summary.json
  segment_meta.csv
  plane_000.bin
  plane_001.bin
  ...
```

轉法：

```bash
./bin/buff_tool export-runtime \
  --input data/tmp/rust_buff_like_by_p/sensor_p6.buff64 \
  --raw-input data/dev/sensor.f64le.bin \
  --out-dir data/tmp/exp_runtime/sensor_p6 \
  --dataset sensor
```

參數說明：

- `--input`
  - `.buff64` container
- `--raw-input`
  - 原始 raw `.f64le.bin`
  - 用來補 `exact_sum` 等 raw-domain reference
- `--out-dir`
  - 匯出的 runtime artifact root
- `--dataset`
  - dataset 名稱，會寫進 `manifest.json`
- `--source-path`
  - optional，自訂 manifest 裡的 `source_path`

---

## `.buff64` 格式大意

### File header

先寫：

- magic
- `total_values`
- `segment_size`
- `segment_count`

### Segment payload

每個 segment 依序寫：

- `value_count`
- `precision_power`
- `fractional_bits`
- `integer_offset_bits`
- `fixed_len_bits`
- `base_width`
- `plane_count`
- `integer_base_le`
- `byte_planes`

所以 `.buff64` 是：

> file header + segment payloads + embedded planes

這也是為什麼它不能直接拿去餵目前的 Exp3 / Exp4 runtime loader。

---

## Runtime artifact 會產生什麼

`export-runtime` 會輸出：

- `manifest.json`
- `summary.json`
- `segment_meta.csv`
- `plane_000.bin ... plane_{max_plane_count-1}.bin`

其中：

- `plane_000.bin` = highest-significance plane
- plane file 是 dataset-level rectangular zero-padded layout
- `segment_meta.csv` 會保留：
  - `active_plane_count`
  - `segment_base`
  - `plane_basis_*`
  - `effective_fractional_bits` 等欄位

這樣 Exp3 / Exp4 就能直接讀。

---

## Programmatic API

如果你不是走 CLI，而是想直接在 C++ 裡用：

### Encode one segment

```cpp
buff::EncodeConfig cfg;
cfg.precision_power = 6;
buff::EncodedSegment seg = buff::encode_segment(values, cfg);
```

### Full-depth decode

```cpp
std::vector<double> decoded = buff::decode_segment(seg);
```

### Top-k decode

```cpp
std::vector<double> approx = buff::decode_segment_top_k(seg, k);
```

### Top-k error bound

```cpp
double bound = buff::segment_max_abs_error_bound(seg, k);
```

### Export Exp3 / Exp4 runtime layout

```cpp
buff::export_runtime_layout(
    encoded_path,
    raw_input_path,
    output_dir,
    dataset_name,
    source_path);
```

---

## Common paths in this workspace

### Raw source data

- [data/dev](/Users/heycool/Documents/codex_workspace/data/dev)

### `.buff64` artifacts

- [data/tmp/rust_buff_like](/Users/heycool/Documents/codex_workspace/data/tmp/rust_buff_like)
- [data/tmp/rust_buff_like_by_p](/Users/heycool/Documents/codex_workspace/data/tmp/rust_buff_like_by_p)

### Exp3 / Exp4-ready exported runtime roots

- [data/tmp/exp_runtime_by_p](/Users/heycool/Documents/codex_workspace/data/tmp/exp_runtime_by_p)

---

## Gotchas

### 1. `.buff64` 不是 benchmark final input

它是中間 container。
真正給 Exp3 / Exp4 用的，是 `export-runtime` 之後那套 plane-major root。

### 2. `precision_power` 不等於 plane count

關係是：

```text
p -> dlen
dlen + integer_offset_bits -> fixed_len_bits
fixed_len_bits -> plane_count
```

### 3. 超大 finite outlier 會拉大整個 segment

目前這版 encoder 不會把超大 finite outlier 另外拆出去。
它會增加：

- `integer_offset_bits`
- `fixed_len_bits`
- `plane_count`

如果 fixed-point 超過 `i64` 範圍，則 encode 直接報錯。

### 4. 只支援 finite non-negative values

目前不支援：

- negative
- NaN
- Infinity

---

