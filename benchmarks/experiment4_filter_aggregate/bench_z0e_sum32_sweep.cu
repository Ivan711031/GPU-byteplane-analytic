// Phase 3-Z0e: SUM32 measured allocation-unit sweep
// Measures A (baseline), B (SUM32 common), D (eager vote) at 4KB/16KB/64KB/256KB.
// 3 repeats per config for mean/std/min/max statistics.
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 && make bench_z0e_sum32_sweep
// Run:   ./bench_z0e_sum32_sweep --dataset PATH [--csv PATH] [--datasets PATH2,...]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "../experiment3/exp3_real_data_layout.hpp"

using namespace exp3_real;
namespace fs = std::filesystem;

constexpr int ITERS = 100;
constexpr int REPEATS = 3;

static void die(const char *m) { std::fprintf(stderr, "FATAL: %s\n", m); std::exit(2); }
static void cuda_check(cudaError_t e, const char *w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error at %s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}

struct Sample {
    double vals[REPEATS] = {};
    double mean = 0, stddev = 0, min_val = 0, max_val = 0;
};

static Sample compute_stats(const double *vals, int n) {
    Sample s;
    s.min_val = vals[0]; s.max_val = vals[0];
    double sum = 0;
    for (int i = 0; i < n; i++) {
        s.vals[i] = vals[i];
        sum += vals[i];
        if (vals[i] < s.min_val) s.min_val = vals[i];
        if (vals[i] > s.max_val) s.max_val = vals[i];
    }
    s.mean = sum / n;
    double var = 0;
    for (int i = 0; i < n; i++) var += (vals[i] - s.mean) * (vals[i] - s.mean);
    s.stddev = std::sqrt(var / n);
    return s;
}

static void measure(const char *label, std::function<void(cudaStream_t)> fn,
    int iters, cudaStream_t s, double &out_ms) {
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
    out_ms = ms / iters;
}

// Path A: baseline read
__global__ void read_baseline(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// Path B: read + SUM32 digest (1 thread/unit)
__global__ void read_sum32(const uint8_t *__restrict__ src, uint64_t n_units,
    uint64_t unit_b, uint32_t *__restrict__ digests) {
    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    uint64_t base = u * unit_b;
    uint32_t s = 0;
    for (uint64_t b = 0; b < unit_b; b++) s += static_cast<uint32_t>(src[base + b]);
    digests[u] = s;
}

// Path D: eager read3 + vote
__global__ void eager_vote(const uint8_t *__restrict__ rep0,
    const uint8_t *__restrict__ rep1, const uint8_t *__restrict__ rep2,
    uint8_t *__restrict__ voted, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t a = rep0[i], b0 = rep1[i], c = rep2[i];
    voted[i] = (a == b0 || a == c) ? a : b0;
}

int main(int argc, char **argv) {
    std::string csv_path = "z0e_sum32_sweep.csv";
    std::vector<std::string> dataset_paths;

    for (int i = 1; i < argc; ++i) {
        std::string_view a(argv[i]);
        auto val=[&](){if(++i>=argc)die("missing arg");return std::string(argv[i]);};
        if (a=="--help"||a=="-h") {
            std::fprintf(stderr,"Usage: %s --dataset PATH [--dataset PATH2 ...] [--csv PATH]\n",argv[0]);
            return 0;
        } else if (a=="--dataset") dataset_paths.push_back(val());
        else if (a=="--csv") csv_path=val();
        else { std::string m="unknown: ";m+=a;die(m.c_str()); }
    }
    if (dataset_paths.empty()) die("--dataset required");

    cuda_check(cudaSetDevice(0), "dev");
    cudaDeviceProp p;
    cuda_check(cudaGetDeviceProperties(&p, 0), "prop");
    std::fprintf(stderr, "Device: %s (sm_%d%d)  Mem: %.1f GB\n", p.name, p.major, p.minor, p.totalGlobalMem / 1e9);

    std::vector<uint64_t> unit_sizes = {4096, 16384, 65536, 262144};
    int T = 256;
    double B0_MS = 0.8702;

    FILE *fcsv = std::fopen(csv_path.c_str(), "w");
    if (!fcsv) die("write CSV");
    const char *jid = std::getenv("SLURM_JOB_ID"); if (!jid) jid = "NOJOB";
    std::fprintf(fcsv, "# z0e_sum32_sweep  %s sm_%d%d  job=%s  iters=%d  repeats=%d\n",
        p.name, p.major, p.minor, jid, ITERS, REPEATS);
    std::fprintf(fcsv, "dataset,n_units,alloc_unit,path,label,repeat,ms_per_query\n");

    for (auto &ds_path : dataset_paths) {
        Dataset ds = exp3_real::load_dataset(ds_path);
        uint64_t pb = ds.planes.empty() ? 0 : ds.planes[0].size();
        uint64_t n_rows = ds.manifest.value_count;
        std::string ds_name = ds.manifest.dataset;

        std::fprintf(stderr, "\n=== Dataset: %s  rows=%" PRIu64 "  plane0=%" PRIu64 " ===\n",
            ds_name.c_str(), n_rows, pb);

        // Upload planes
        std::vector<uint8_t*> d_planes;
        for (auto &pl : ds.planes) {
            uint8_t *ptr = nullptr;
            cuda_check(cudaMalloc(&ptr, pl.size()), "M pl");
            cuda_check(cudaMemcpy(ptr, pl.data(), pl.size(), cudaMemcpyHostToDevice), "H2D pl");
            d_planes.push_back(ptr);
        }

        for (uint64_t alloc_unit : unit_sizes) {
            uint64_t nu = (pb + alloc_unit - 1) / alloc_unit;
            std::fprintf(stderr, "\n  alloc_unit=%" PRIu64 "  units=%" PRIu64 "\n", alloc_unit, nu);

            uint8_t *d_r0, *d_r1, *d_r2, *d_v;
            uint32_t *d_sum;
            cuda_check(cudaMalloc(&d_r0, pb), "M r0");
            cuda_check(cudaMalloc(&d_r1, pb), "M r1");
            cuda_check(cudaMalloc(&d_r2, pb), "M r2");
            cuda_check(cudaMalloc(&d_v, pb), "M v");
            cuda_check(cudaMalloc(&d_sum, nu * sizeof(uint32_t)), "M sum");
            cuda_check(cudaMemcpy(d_r0, ds.planes[0].data(), pb, cudaMemcpyHostToDevice), "H2D r0");
            cuda_check(cudaMemcpy(d_r1, ds.planes[0].data(), pb, cudaMemcpyHostToDevice), "H2D r1");
            cuda_check(cudaMemcpy(d_r2, ds.planes[0].data(), pb, cudaMemcpyHostToDevice), "H2D r2");

            cudaStream_t st;
            cuda_check(cudaStreamCreate(&st), "st");

            double a_vals[REPEATS], b_vals[REPEATS], d_vals[REPEATS];

            for (int rep = 0; rep < REPEATS; rep++) {
                // A: baseline
                {
                    int blk = (static_cast<int>(pb) + T - 1) / T;
                    auto fn = [&](cudaStream_t s) { read_baseline<<<blk, T, 0, s>>>(d_r0, d_v, pb); };
                    ::measure("  A", fn, ITERS, st, a_vals[rep]);
                }
                // B: SUM32
                {
                    auto fn = [&](cudaStream_t s) { read_sum32<<<static_cast<int>(nu), 1, 0, s>>>(d_r0, nu, alloc_unit, d_sum); };
                    ::measure("  B", fn, ITERS, st, b_vals[rep]);
                }
                // D: eager vote
                {
                    int blk = (static_cast<int>(pb) + T - 1) / T;
                    auto fn = [&](cudaStream_t s) { eager_vote<<<blk, T, 0, s>>>(d_r0, d_r1, d_r2, d_v, pb); };
                    ::measure("  D", fn, ITERS, st, d_vals[rep]);
                }
            }

            Sample a_s = compute_stats(a_vals, REPEATS);
            Sample b_s = compute_stats(b_vals, REPEATS);
            Sample d_s = compute_stats(d_vals, REPEATS);

            for (int rep = 0; rep < REPEATS; rep++) {
                std::fprintf(fcsv, "%s,%" PRIu64 ",%" PRIu64 ",A,baseline,%d,%.12f\n",
                    ds_name.c_str(), nu, alloc_unit, rep, a_vals[rep]);
                std::fprintf(fcsv, "%s,%" PRIu64 ",%" PRIu64 ",B,sum32,%d,%.12f\n",
                    ds_name.c_str(), nu, alloc_unit, rep, b_vals[rep]);
                std::fprintf(fcsv, "%s,%" PRIu64 ",%" PRIu64 ",D,eager_vote,%d,%.12f\n",
                    ds_name.c_str(), nu, alloc_unit, rep, d_vals[rep]);
            }

            std::fprintf(stderr, "    A:  %.6f +- %.6f ms  [%.6f, %.6f]\n", a_s.mean, a_s.stddev, a_s.min_val, a_s.max_val);
            std::fprintf(stderr, "    B:  %.6f +- %.6f ms  [%.6f, %.6f]  B/A=%.4f  B/B0=%.4f\n",
                b_s.mean, b_s.stddev, b_s.min_val, b_s.max_val,
                b_s.mean / a_s.mean, b_s.mean / B0_MS);
            std::fprintf(stderr, "    D:  %.6f +- %.6f ms  [%.6f, %.6f]  D/A=%.4f  <B0=%d\n",
                d_s.mean, d_s.stddev, d_s.min_val, d_s.max_val,
                d_s.mean / a_s.mean, d_s.mean < B0_MS);

            cuda_check(cudaFree(d_sum), "F sum");
            cuda_check(cudaFree(d_v), "F v");
            cuda_check(cudaFree(d_r2), "F r2");
            cuda_check(cudaFree(d_r1), "F r1");
            cuda_check(cudaFree(d_r0), "F r0");
            cuda_check(cudaStreamDestroy(st), "D st");
        }

        for (auto pp : d_planes) cuda_check(cudaFree(pp), "F pl");
    }

    fclose(fcsv);
    std::fprintf(stderr, "\nCSV: %s\n", csv_path.c_str());
    return 0;
}
