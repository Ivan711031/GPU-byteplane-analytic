# m03 GPU Byte-Plane Analytics

Byte-plane fused Filter+Aggregate on NVIDIA H200: progressive execution and reliability contracts.

## Reproduction Guide

### Prerequisites

- NVIDIA H200 GPU (CUDA arch 90), driver ≥ 550
- CUDA Toolkit ≥ 12.8, CMake ≥ 3.24, C++17 host compiler
- Python ≥ 3.10 with `numpy`
- `wget` (for dataset download)

### 1. Dataset Preparation

**Scientific datasets** (cesm_atm_cloud, cesm_atm_q, hurricane_u, hurricane_tc):

```bash
# Download raw SDRBench data (Hurricane Isabel + CESM-ATM)
./scripts/download_sdrbench.sh

# Build the encoder tool
g++ -O3 -std=c++20 -o bin/buff_tool buff_encoder/buff_tool.cpp buff_encoder/buff_codec.cpp

# Clean, encode, and export runtime artifacts for each dataset
./scripts/build_scientific_v2_artifact.py --dataset cesm_atm_cloud --buff-tool bin/buff_tool
./scripts/build_scientific_v2_artifact.py --dataset cesm_atm_q      --buff-tool bin/buff_tool
./scripts/build_scientific_v2_artifact.py --dataset hurricane_u     --buff-tool bin/buff_tool
./scripts/build_scientific_v2_artifact.py --dataset hurricane_tc    --buff-tool bin/buff_tool
```

**Synthetic datasets** (sensor, uniform, heavy_tailed, zipfian, 10⁸ rows each):

```bash
# Build the C generator
gcc -O3 -lm -o buff_encoder/synth_datasets buff_encoder/synth_datasets.c

# Generate all four synthetic datasets
./scripts/export_v2_artifacts.py
```

### 2. Build Benchmarks

Each experiment has its own CMake project under `benchmarks/experiment*/` and builds independently:

```bash
# Exp0 — HBM bandwidth characterization
cmake -S benchmarks/experiment0 -B build/exp0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp0 -j

# Exp1 — Rowpack feasibility (byte_ilp4_k8, rowpack16_k8, contiguous64)
cmake -S benchmarks/experiment1 -B build/exp1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp1 -j

# Exp3 — Progressive SUM on scientific data
cmake -S benchmarks/experiment3 -B build/exp3 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp3 -j

# Exp4 — Progressive COUNT + threshold prep
cmake -S benchmarks/experiment4 -B build/exp4 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp4 -j

# Exp4 Filter+Aggregate — Locality, traffic attribution, graded repair, NMR
cmake -S benchmarks/experiment4_filter_aggregate -B build/exp4_fa -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp4_fa -j

# Reliability Layer 1 — Detect-and-bound structured campaign
cmake -S benchmarks/reliability_layer1 -B build/rl1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/rl1 -j
```

### 3. Run Experiments

All run scripts accept environment variables for device, row count, kernel parameters, etc. Defaults match the paper settings.

| Experiment | Script | What It Measures |
|---|---|---|
| HBM bandwidth baseline | `./scripts/run_exp0.sh` | Sequential / masked / gather HBM bandwidth on H200 |
| Rowpack feasibility | `./scripts/run_exp1.sh` | byte_ilp4 vs rowpack16 vs contiguous64 throughput |
| Exp1 baseline | `./scripts/run_exp1_baseline.sh` | Baseline sweep for rowpack variants |
| Progressive SUM (scientific) | `./scripts/run_exp3.sh` | `k=1..max` progressive SUM on real fields |
| Raw FP64 SUM baseline | `./scripts/run_exp3_raw_fp64_sum.sh` | Raw-fused FP64 SUM on same datasets |
| Progressive COUNT | `./scripts/run_exp4.sh` | `k`-depth dispatch and break-even analysis |
| Raw FP64 COUNT baseline | `./scripts/run_exp4_raw_fp64_count.sh` | Raw-fused FP64 COUNT |
| Locality / traffic attribution | `./scripts/run_z0_bandwidth_smoke.sh` | seg_global vs seg4096 latency breakdown |
| Controlled-regime sweep | `./scripts/run_controlled_regime_sweep.sh` | Break-even frontier on synthetic distributions |
| Reliability: detect-and-bound | `./scripts/reliability_layer1` benchmarks | Single-replica 24-cell bounded answer test |
| Reliability: structured fault | `./scripts/run_nmr_a_structured_fault.sh` | 7 fault families × 5 policies × 3 seeds |
| Reliability: realistic campaign | `./scripts/run_nmr_claim1_realistic_pilot.sh` | F1–F8 × 20 policies × 10 seeds |
| Reliability: NMR pipeline | `./scripts/run_nmr_b_e2e_pipeline.sh` | End-to-end fused NMR latency |

All scripts output CSVs and metadata to `results/exp*/run_*/`. Navigate to the repo root and run:

```bash
DEVICE=0 ./scripts/run_exp3.sh              # progressive SUM
DEVICE=0 ./scripts/run_exp3_raw_fp64_sum.sh # raw FP64 baseline
```

### Key Parameters

| Env var | Default | Meaning |
|---|---|---|
| `DEVICE` | `0` | GPU device index |
| `N` | `100000000` | Synthetic row count |
| `K_MIN` / `K_MAX` | `1` / `8` | Plane prefix range |
| `WARMUP` | `10` | Warmup iterations |
| `ITERS` | varies | Measurement iterations |
| `CUDA_ARCH` | `90` | Target CUDA architecture |

Results can be plotted with `scripts/plot_paper_v1_scoped_h200_figures_v2.py`.

---

## Datasets

| Field | Rows | Source | Use |
|---|---|---|---|
| `cesm_atm_cloud` | 168.48M | SDRBench CESM-ATM — CLOUD | execution + reliability |
| `cesm_atm_q` | 168.48M | SDRBench CESM-ATM — Q | execution |
| `hurricane_u` | 25M | Hurricane Isabel — U | execution + reliability |
| `hurricane_tc` | 25M | Hurricane Isabel — TC | execution |
| `sensor` | 10⁸ | `synth_datasets.c` | mechanism validation |
| `uniform` | 10⁸ | `synth_datasets.c` | k-depth / dispatch calibration |
| `heavy_tailed` | 10⁸ | `synth_datasets.c` | skewed-distribution stress |
| `zipfian` | 10⁸ | `synth_datasets.c` | extreme-tail stress |

## Results Summary

### Execution

- **Rowpack feasibility**: `rowpack16_k8` reaches **4394 GB/s** vs `byte_ilp4_k8`'s 2399 GB/s and `contiguous64`'s 3929 GB/s.
- **Scientific headline** (warm, quasi-global, shallow `k=2`):
  - `cesm_atm_cloud`: 3.64×–4.92× vs raw FP64
  - `hurricane_u`: 2.69×–3.90×
  - `cesm_atm_q`: 3.58×–6.14×
  - `hurricane_tc`: 2.94×–5.72×
- **Locality**: single-segment total latency ~0.23 ms; 4096-segment rises to ~17.62 ms (threshold prep dominated).
- **Traffic attribution**: `k=2` reduces DRAM reads to ~0.135× of raw FP64 for `cesm_atm_q`.
- **Break-even**: shallow `k` has stable advantage; raw FP64 fallback kicks in by `k=4` on synthetic.

### Reliability

- **Single-replica detect-and-bound**: 24/24 cells return truth-containing bounded answers.
- **Structured-fault H200 path**: 210/210 contain truth; CPU/GPU classification fully consistent.
- **Graded vs uniform bound width**: graded provides 345×–432× tighter certified bounds.
- **NMR latency**: shallow-prefix (`k=1,2`) stays below raw FP64; full-prefix costs 1.82×–2.63×.

## Reference

- [Decomposed Bounded Floats for Fast Compression and Queries](https://www.vldb.org/pvldb/vol14/p2586-liu.pdf)
- [A Study of the Fundamental Performance Characteristics of GPUs and CPUs for Database Analytics](https://anilshanbhag.in/static/papers/crystal_sigmod20.pdf)
- [ALP: Adaptive Lossless floating-Point Compression](https://ir.cwi.nl/pub/33334/33334.pdf)
- [G-ALP: Rethinking Light-weight Encodings for GPUs](https://dl.acm.org/doi/epdf/10.1145/3736227.3736242)
- [GPU Acceleration of SQL Analytics on Compressed Data](https://dl.acm.org/doi/pdf/10.14778/3778092.3778095)
