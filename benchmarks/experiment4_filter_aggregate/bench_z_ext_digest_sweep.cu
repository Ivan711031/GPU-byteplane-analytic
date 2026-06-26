// Phase 3-Z Extension: Multi-digest latency sweep (E1 + E2)
// Measures latency of 5 digest variants + eager vote r=2/3/5 on H200.
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
//        && make bench_z_ext_digest_sweep
// Run:   ./bench_z_ext_digest_sweep --dataset PATH [--dataset PATH2] [--csv PATH]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "../experiment3/exp3_real_data_layout.hpp"

using namespace exp3_real;

constexpr uint64_t ALLOC_UNIT = 4096;
constexpr int ITERS = 100;
constexpr int REPEATS = 3;

static void die(const char *m) { std::fprintf(stderr, "FATAL: %s\n", m); std::exit(2); }
static void cuda_check(cudaError_t e, const char *w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error at %s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}

struct Timing { double ms_total = 0, ms_per_iter = 0; };
static Timing measure(const char *label, std::function<void(cudaStream_t)> fn, int iters, cudaStream_t s) {
    for (int i = 0; i < 2; i++) fn(s);
    cuda_check(cudaStreamSynchronize(s), "warmup");
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "ce"); cuda_check(cudaEventCreate(&stop), "ce");
    cuda_check(cudaEventRecord(start, s), "rec");
    for (int i = 0; i < iters; i++) fn(s);
    cuda_check(cudaEventRecord(stop, s), "rec"); cuda_check(cudaEventSynchronize(stop), "sync");
    float ms = 0;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cuda_check(cudaEventDestroy(start), "d"); cuda_check(cudaEventDestroy(stop), "d");
    Timing r; r.ms_total = ms; r.ms_per_iter = ms / iters;
    std::fprintf(stderr, "  %-55s %9.4f ms total  %9.6f ms/iter\n", label, ms, r.ms_per_iter);
    return r;
}

// ── Path A: baseline read ────────────────────────────────────────────
__global__ void read_baseline(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ── Path B: serial SUM32 (1 thread/unit) ─────────────────────────────
__global__ void sum32_serial(const uint8_t *__restrict__ src, uint64_t n_units,
    uint64_t unit_b, uint32_t *__restrict__ digests) {
    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    uint64_t base = u * unit_b;
    uint32_t s = 0;
    for (uint64_t b = 0; b < unit_b; b++) s += static_cast<uint32_t>(src[base + b]);
    digests[u] = s;
}

// ── SUM32 parallel (256 threads/block) ───────────────────────────────
__global__ void sum32_parallel(const uint8_t *__restrict__ src, uint64_t n_units,
    uint64_t unit_b, uint32_t *__restrict__ digests) {
    __shared__ uint32_t s_red[32];
    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    uint64_t base = u * unit_b;
    uint64_t tid = threadIdx.x;
    uint64_t stride = blockDim.x;
    uint32_t local = 0;
    for (uint64_t b = tid; b < unit_b; b += stride)
        local += static_cast<uint32_t>(src[base + b]);
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(0xFFFFFFFFu, local, off);
    if ((tid & 31) == 0) s_red[tid >> 5] = local;
    __syncthreads();
    if (tid < 32) {
        uint32_t w = s_red[tid];
        for (int off = 16; off > 0; off >>= 1)
            w += __shfl_down_sync(0xFFFFFFFFu, w, off);
        if (tid == 0) digests[u] = w;
    }
}

// ── SUM64 parallel ───────────────────────────────────────────────────
__global__ void sum64_parallel(const uint8_t *__restrict__ src, uint64_t n_units,
    uint64_t unit_b, uint64_t *__restrict__ digests) {
    __shared__ uint64_t s_red[32];
    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    uint64_t base = u * unit_b;
    uint64_t tid = threadIdx.x;
    uint64_t stride = blockDim.x;
    uint64_t local = 0;
    for (uint64_t b = tid; b < unit_b; b += stride)
        local += static_cast<uint64_t>(src[base + b]);
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(0xFFFFFFFFu, local, off);
    if ((tid & 31) == 0) s_red[tid >> 5] = local;
    __syncthreads();
    if (tid < 32) {
        uint64_t w = s_red[tid];
        for (int off = 16; off > 0; off >>= 1)
            w += __shfl_down_sync(0xFFFFFFFFu, w, off);
        if (tid == 0) digests[u] = w;
    }
}

// ── dual SUM32 parallel ──────────────────────────────────────────────
__global__ void dual_sum32_parallel(const uint8_t *__restrict__ src, uint64_t n_units,
    uint64_t unit_b, uint64_t *__restrict__ digests) {
    __shared__ uint32_t s_red0[32], s_red1[32];
    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    uint64_t base = u * unit_b;
    uint64_t tid = threadIdx.x;
    uint64_t stride = blockDim.x;
    uint64_t n4 = unit_b / 4;
    uint32_t local0 = 0, local1 = 0;
    for (uint64_t w = tid; w < n4; w += stride) {
        uint32_t val = reinterpret_cast<const uint32_t*>(src + base)[w];
        if (w % 2 == 0) local0 += val; else local1 += val;
    }
    for (int off = 16; off > 0; off >>= 1) {
        local0 += __shfl_down_sync(0xFFFFFFFFu, local0, off);
        local1 += __shfl_down_sync(0xFFFFFFFFu, local1, off);
    }
    if ((tid & 31) == 0) { s_red0[tid >> 5] = local0; s_red1[tid >> 5] = local1; }
    __syncthreads();
    if (tid < 32) {
        uint32_t w0 = s_red0[tid], w1 = s_red1[tid];
        for (int off = 16; off > 0; off >>= 1) {
            w0 += __shfl_down_sync(0xFFFFFFFFu, w0, off);
            w1 += __shfl_down_sync(0xFFFFFFFFu, w1, off);
        }
        if (tid == 0) digests[u] = (static_cast<uint64_t>(w1) << 32) | w0;
    }
}

// ── Position-weighted SUM32 ──────────────────────────────────────────
__global__ void pos_weighted_parallel(const uint8_t *__restrict__ src, uint64_t n_units,
    uint64_t unit_b, uint32_t *__restrict__ digests) {
    __shared__ uint32_t s_red[32];
    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    uint64_t base = u * unit_b;
    uint64_t tid = threadIdx.x;
    uint64_t stride = blockDim.x;
    // Each thread accumulates uint32 words with positional weight
    uint64_t local = 0;
    for (uint64_t w = tid; w < unit_b / 4; w += stride) {
        uint32_t val = reinterpret_cast<const uint32_t*>(src + base)[w];
        local += (w + 1) * static_cast<uint64_t>(val);
    }
    uint32_t loc32 = static_cast<uint32_t>(local & 0xFFFFFFFF);
    for (int off = 16; off > 0; off >>= 1)
        loc32 += __shfl_down_sync(0xFFFFFFFFu, loc32, off);
    if ((tid & 31) == 0) s_red[tid >> 5] = loc32;
    __syncthreads();
    if (tid < 32) {
        uint32_t w = s_red[tid];
        for (int off = 16; off > 0; off >>= 1)
            w += __shfl_down_sync(0xFFFFFFFFu, w, off);
        if (tid == 0) digests[u] = w;
    }
}

// ── XOR-rotate ───────────────────────────────────────────────────────
__global__ void xor_rotate_parallel(const uint8_t *__restrict__ src, uint64_t n_units,
    uint64_t unit_b, uint32_t *__restrict__ digests) {
    __shared__ uint32_t s_red[32];
    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    uint64_t base = u * unit_b;
    uint64_t tid = threadIdx.x;
    uint64_t stride = blockDim.x;
    uint64_t n4 = unit_b / 4;
    uint32_t sum_loc = 0, xor_loc = 0;
    for (uint64_t w = tid; w < n4; w += stride) {
        uint32_t val = reinterpret_cast<const uint32_t*>(src + base)[w];
        sum_loc += val;
        xor_loc ^= val;
        xor_loc = (xor_loc << 1) | (xor_loc >> 31);
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum_loc += __shfl_down_sync(0xFFFFFFFFu, sum_loc, off);
        xor_loc += __shfl_down_sync(0xFFFFFFFFu, xor_loc, off);
    }
    if ((tid & 31) == 0) s_red[tid >> 5] = sum_loc ^ xor_loc;
    __syncthreads();
    if (tid < 32) {
        uint32_t w = s_red[tid];
        for (int off = 16; off > 0; off >>= 1)
            w += __shfl_down_sync(0xFFFFFFFFu, w, off);
        if (tid == 0) digests[u] = w;
    }
}

// ── Fletcher-like ────────────────────────────────────────────────────
__global__ void fletcher_parallel(const uint8_t *__restrict__ src, uint64_t n_units,
    uint64_t unit_b, uint64_t *__restrict__ digests) {
    __shared__ uint32_t s_red1[32], s_red2[32];
    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    uint64_t base = u * unit_b;
    uint64_t tid = threadIdx.x;
    uint64_t stride = blockDim.x;
    uint64_t n4 = unit_b / 4;
    uint32_t s1 = 0, s2 = 0;
    for (uint64_t w = tid; w < n4; w += stride) {
        uint32_t val = reinterpret_cast<const uint32_t*>(src + base)[w];
        s1 += val;
        s2 += s1;
    }
    for (int off = 16; off > 0; off >>= 1) {
        s1 += __shfl_down_sync(0xFFFFFFFFu, s1, off);
        s2 += __shfl_down_sync(0xFFFFFFFFu, s2, off);
    }
    if ((tid & 31) == 0) { s_red1[tid >> 5] = s1; s_red2[tid >> 5] = s2; }
    __syncthreads();
    if (tid < 32) {
        uint32_t w1 = s_red1[tid], w2 = s_red2[tid];
        for (int off = 16; off > 0; off >>= 1) {
            w1 += __shfl_down_sync(0xFFFFFFFFu, w1, off);
            w2 += __shfl_down_sync(0xFFFFFFFFu, w2, off);
        }
        if (tid == 0) digests[u] = (static_cast<uint64_t>(w2) << 32) | w1;
    }
}

// ── Eager vote r=2 ──────────────────────────────────────────────────
__global__ void eager_vote_r2(const uint8_t *__restrict__ rep0,
    const uint8_t *__restrict__ rep1, uint8_t *__restrict__ voted, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    voted[i] = rep0[i];
}

// ── Eager vote r=3 ──────────────────────────────────────────────────
__global__ void eager_vote_r3(const uint8_t *__restrict__ rep0,
    const uint8_t *__restrict__ rep1, const uint8_t *__restrict__ rep2,
    uint8_t *__restrict__ voted, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t a = rep0[i], b = rep1[i], c = rep2[i];
    voted[i] = (a == b || a == c) ? a : b;
}

// ── Eager vote r=5 ──────────────────────────────────────────────────
__global__ void eager_vote_r5(const uint8_t *__restrict__ rep0,
    const uint8_t *__restrict__ rep1, const uint8_t *__restrict__ rep2,
    const uint8_t *__restrict__ rep3, const uint8_t *__restrict__ rep4,
    uint8_t *__restrict__ voted, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t v[5] = {rep0[i], rep1[i], rep2[i], rep3[i], rep4[i]};
    for (int j = 0; j < 5; j++) {
        int cnt = 0;
        for (int k = 0; k < 5; k++)
            if (v[k] == v[j]) cnt++;
        if (cnt >= 3) { voted[i] = v[j]; return; }
    }
    voted[i] = rep0[i];
}

// ── Measure one config (helper to avoid repetition) ──────────────────
static double measure_one(const char *label, std::function<void(cudaStream_t)> fn,
                          int iters, cudaStream_t s) {
    return measure(label, fn, iters, s).ms_per_iter;
}

int main(int argc, char **argv) {
    std::string csv_path = "z_ext_digest_sweep.csv";
    std::vector<std::string> dataset_paths;

    for (int i = 1; i < argc; ++i) {
        std::string_view a(argv[i]);
        auto val=[&](){if(++i>=argc)die("missing arg");return std::string(argv[i]);};
        if (a=="--help"||a=="-h") {
            std::fprintf(stderr,"Usage: %s --dataset PATH [--dataset PATH2] [--csv PATH]\n",argv[0]);
            return 0;
        } else if (a=="--dataset") dataset_paths.push_back(val());
        else if (a=="--csv") csv_path=val();
    }
    if (dataset_paths.empty()) die("--dataset required");

    cuda_check(cudaSetDevice(0), "dev");
    cudaDeviceProp p;
    cuda_check(cudaGetDeviceProperties(&p, 0), "prop");
    std::fprintf(stderr, "Device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    int T = 256;
    double B0_MS = 0.8702;

    FILE *fcsv = std::fopen(csv_path.c_str(), "w");
    if (!fcsv) die("write CSV");
    const char *jid = std::getenv("SLURM_JOB_ID"); if (!jid) jid = "NOJOB";
    std::fprintf(fcsv, "# z_ext_digest_sweep  %s sm_%d%d  job=%s  iters=%d  repeats=%d\n",
        p.name, p.major, p.minor, jid, ITERS, REPEATS);
    std::fprintf(fcsv, "dataset,n_units,alloc_unit,path,label,repeat,ms_per_query\n");

    for (auto &ds_path : dataset_paths) {
        Dataset ds = exp3_real::load_dataset(ds_path);
        uint64_t pb = ds.planes.empty() ? 0 : ds.planes[0].size();
        std::string ds_name = ds.manifest.dataset;

        std::fprintf(stderr, "\n=== %s  plane0=%" PRIu64 " ===\n", ds_name.c_str(), pb);

        std::vector<uint8_t*> d_planes;
        for (auto &pl : ds.planes) {
            uint8_t *ptr = nullptr;
            cuda_check(cudaMalloc(&ptr, pl.size()), "M pl");
            cuda_check(cudaMemcpy(ptr, pl.data(), pl.size(), cudaMemcpyHostToDevice), "H2D pl");
            d_planes.push_back(ptr);
        }

        uint64_t nu = (pb + ALLOC_UNIT - 1) / ALLOC_UNIT;
        std::fprintf(stderr, "  alloc_unit=%" PRIu64 "  units=%" PRIu64 "\n", ALLOC_UNIT, nu);

        uint8_t *d_r0, *d_r1, *d_r2, *d_r3, *d_r4, *d_v;
        uint32_t *d_sum32;
        uint64_t *d_sum64;
        cuda_check(cudaMalloc(&d_r0, pb), "M r0");
        cuda_check(cudaMalloc(&d_r1, pb), "M r1");
        cuda_check(cudaMalloc(&d_r2, pb), "M r2");
        cuda_check(cudaMalloc(&d_r3, pb), "M r3");
        cuda_check(cudaMalloc(&d_r4, pb), "M r4");
        cuda_check(cudaMalloc(&d_v, pb * 2), "M v");  // extra room for vote output
        cuda_check(cudaMalloc(&d_sum32, nu * sizeof(uint32_t)), "M sum32");
        cuda_check(cudaMalloc(&d_sum64, nu * sizeof(uint64_t)), "M sum64");
        cuda_check(cudaMemcpy(d_r0, ds.planes[0].data(), pb, cudaMemcpyHostToDevice), "H2D r0");
        cuda_check(cudaMemcpy(d_r1, ds.planes[0].data(), pb, cudaMemcpyHostToDevice), "H2D r1");
        cuda_check(cudaMemcpy(d_r2, ds.planes[0].data(), pb, cudaMemcpyHostToDevice), "H2D r2");
        cuda_check(cudaMemcpy(d_r3, ds.planes[0].data(), pb, cudaMemcpyHostToDevice), "H2D r3");
        cuda_check(cudaMemcpy(d_r4, ds.planes[0].data(), pb, cudaMemcpyHostToDevice), "H2D r4");

        // Config descriptors for the sweep
        struct Cfg { const char *path; const char *label; };
        Cfg cfgs[] = {
            {"A",  "baseline"},
            {"B",  "sum32_serial"},
            {"B2", "sum32_parallel"},
            {"C",  "sum64_parallel"},
            {"D",  "dual_sum32_parallel"},
            {"E",  "pos_weighted_parallel"},
            {"F",  "xor_rotate_parallel"},
            {"G",  "fletcher_parallel"},
            {"H",  "eager_vote_r2"},
            {"I",  "eager_vote_r3"},
            {"J",  "eager_vote_r5"},
        };
        constexpr int N_CFG = sizeof(cfgs) / sizeof(cfgs[0]);

        double results[N_CFG][REPEATS];
        std::memset(results, 0, sizeof(results));

        int blk_full = (static_cast<int>(pb) + T - 1) / T;

        for (int rep = 0; rep < REPEATS; rep++) {
            cudaStream_t st;
            cuda_check(cudaStreamCreate(&st), "st");

            results[0][rep] = measure_one("A: baseline",
                [&](cudaStream_t s) { read_baseline<<<blk_full, T, 0, s>>>(d_r0, d_v, pb); },
                ITERS, st);

            results[1][rep] = measure_one("B: sum32 serial",
                [&](cudaStream_t s) { sum32_serial<<<static_cast<int>(nu), 1, 0, s>>>(d_r0, nu, ALLOC_UNIT, d_sum32); },
                ITERS, st);

            results[2][rep] = measure_one("B2: sum32 parallel",
                [&](cudaStream_t s) { sum32_parallel<<<static_cast<int>(nu), T, 0, s>>>(d_r0, nu, ALLOC_UNIT, d_sum32); },
                ITERS, st);

            results[3][rep] = measure_one("C: sum64 parallel",
                [&](cudaStream_t s) { sum64_parallel<<<static_cast<int>(nu), T, 0, s>>>(d_r0, nu, ALLOC_UNIT, d_sum64); },
                ITERS, st);

            results[4][rep] = measure_one("D: dual sum32 parallel",
                [&](cudaStream_t s) { dual_sum32_parallel<<<static_cast<int>(nu), T, 0, s>>>(d_r0, nu, ALLOC_UNIT, d_sum64); },
                ITERS, st);

            results[5][rep] = measure_one("E: pos_weighted parallel",
                [&](cudaStream_t s) { pos_weighted_parallel<<<static_cast<int>(nu), T, 0, s>>>(d_r0, nu, ALLOC_UNIT, d_sum32); },
                ITERS, st);

            results[6][rep] = measure_one("F: xor_rotate parallel",
                [&](cudaStream_t s) { xor_rotate_parallel<<<static_cast<int>(nu), T, 0, s>>>(d_r0, nu, ALLOC_UNIT, d_sum32); },
                ITERS, st);

            results[7][rep] = measure_one("G: fletcher parallel",
                [&](cudaStream_t s) { fletcher_parallel<<<static_cast<int>(nu), T, 0, s>>>(d_r0, nu, ALLOC_UNIT, d_sum64); },
                ITERS, st);

            results[8][rep] = measure_one("H: eager_vote_r2",
                [&](cudaStream_t s) { eager_vote_r2<<<blk_full, T, 0, s>>>(d_r0, d_r1, d_v, pb); },
                ITERS, st);

            results[9][rep] = measure_one("I: eager_vote_r3",
                [&](cudaStream_t s) { eager_vote_r3<<<blk_full, T, 0, s>>>(d_r0, d_r1, d_r2, d_v, pb); },
                ITERS, st);

            results[10][rep] = measure_one("J: eager_vote_r5",
                [&](cudaStream_t s) { eager_vote_r5<<<blk_full, T, 0, s>>>(d_r0, d_r1, d_r2, d_r3, d_r4, d_v, pb); },
                ITERS, st);

            cuda_check(cudaStreamDestroy(st), "D st");
        }

        // Write CSV
        for (int ci = 0; ci < N_CFG; ci++) {
            for (int rep = 0; rep < REPEATS; rep++) {
                std::fprintf(fcsv, "%s,%" PRIu64 ",%" PRIu64 ",%s,%s,%d,%.12f\n",
                    ds_name.c_str(), nu, ALLOC_UNIT,
                    cfgs[ci].path, cfgs[ci].label, rep, results[ci][rep]);
            }
        }

        // Summary
        auto stats = [&](double *v) {
            double mn = v[0], mx = v[0], sum = 0;
            for (int i = 0; i < REPEATS; i++) { mn = std::min(mn, v[i]); mx = std::max(mx, v[i]); sum += v[i]; }
            double mean = sum / REPEATS;
            double var = 0; for (int i = 0; i < REPEATS; i++) var += (v[i]-mean)*(v[i]-mean);
            return std::make_tuple(mean, std::sqrt(var/REPEATS), mn, mx);
        };

        std::fprintf(stderr, "\n  Summary (%d repeats):\n", REPEATS);
        for (int ci = 0; ci < N_CFG; ci++) {
            auto [mn, sd, lo, hi] = stats(results[ci]);
            std::fprintf(stderr, "    %s: %.6f ± %.6f ms [%.6f, %.6f]  vsA=%.4f  vsB0=%.4f\n",
                cfgs[ci].path, mn, sd, lo, hi,
                mn / results[0][0], mn / B0_MS);
        }

        cuda_check(cudaFree(d_sum64), "F sum64");
        cuda_check(cudaFree(d_sum32), "F sum32");
        cuda_check(cudaFree(d_v), "F v");
        cuda_check(cudaFree(d_r4), "F r4");
        cuda_check(cudaFree(d_r3), "F r3");
        cuda_check(cudaFree(d_r2), "F r2");
        cuda_check(cudaFree(d_r1), "F r1");
        cuda_check(cudaFree(d_r0), "F r0");
        for (auto pp : d_planes) cuda_check(cudaFree(pp), "F pl");
    }

    fclose(fcsv);
    std::fprintf(stderr, "\nCSV: %s\n", csv_path.c_str());
    return 0;
}
