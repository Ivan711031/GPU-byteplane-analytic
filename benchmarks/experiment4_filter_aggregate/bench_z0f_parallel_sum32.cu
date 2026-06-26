// Phase 3-Z0f: Parallel SUM32 warp/block reduction smoke
// Tests whether block-cooperative SUM32 can meet BW gate margin.
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 && make bench_z0f_parallel_sum32
// Run:   ./bench_z0f_parallel_sum32 --dataset PATH [--dataset PATH2 ...] [--csv PATH]

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
    std::fprintf(stderr, "  %-55s %9.4f ms total  %9.6f ms/iter  %9.6f ms/unit\n",
        label, ms, r.ms_per_iter, r.ms_per_iter / std::max(1.0, 1.0));
    return r;
}

// ===================================================================
// Path A: baseline read (byte copy, fully parallel)
// ===================================================================
__global__ void read_baseline(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ===================================================================
// Path B: serial SUM32 (1 thread/unit, sequential loop)
// ===================================================================
__global__ void sum32_serial(const uint8_t *__restrict__ src, uint64_t n_units,
    uint64_t unit_b, uint32_t *__restrict__ digests) {
    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    uint64_t base = u * unit_b;
    uint32_t s = 0;
    for (uint64_t b = 0; b < unit_b; b++) s += static_cast<uint32_t>(src[base + b]);
    digests[u] = s;
}

// ===================================================================
// Path B2: parallel SUM32 (256 threads/block, warp + block reduction)
// Each thread processes 4096/256 = 16 bytes, then block-reduce.
// ===================================================================
__global__ void sum32_parallel(const uint8_t *__restrict__ src, uint64_t n_units,
    uint64_t unit_b, uint32_t *__restrict__ digests) {
    __shared__ uint32_t s_red[32]; // one per warp

    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    uint64_t base = u * unit_b;
    uint64_t tid = threadIdx.x;
    uint64_t stride = blockDim.x;
    uint64_t n = unit_b;

    // Each thread accumulates bytes at offset tid, tid+stride, ...
    uint32_t local = 0;
    for (uint64_t b = tid; b < n; b += stride)
        local += static_cast<uint32_t>(src[base + b]);

    // Warp shuffle reduce (32 threads/warp)
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(0xFFFFFFFFu, local, off);

    // First thread of each warp writes to shared
    if ((tid & 31) == 0)
        s_red[tid >> 5] = local;
    __syncthreads();

    // Reduce warp results (one warp does this)
    if (tid < 32) {
        uint32_t w = s_red[tid];
        for (int off = 16; off > 0; off >>= 1)
            w += __shfl_down_sync(0xFFFFFFFFu, w, off);
        if (tid == 0) digests[u] = w;
    }
}

// ===================================================================
// Path D: eager read3 + vote
// ===================================================================
__global__ void eager_vote(const uint8_t *__restrict__ rep0,
    const uint8_t *__restrict__ rep1, const uint8_t *__restrict__ rep2,
    uint8_t *__restrict__ voted, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t a = rep0[i], b0 = rep1[i], c = rep2[i];
    voted[i] = (a == b0 || a == c) ? a : b0;
}

int main(int argc, char **argv) {
    std::string csv_path = "z0f_parallel_sum32_profile.csv";
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
    std::fprintf(fcsv, "# z0f_parallel_sum32  %s sm_%d%d  job=%s  iters=%d  repeats=%d\n",
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

        double a_reps[REPEATS], b_reps[REPEATS], b2_reps[REPEATS], d_reps[REPEATS];

        for (int rep = 0; rep < REPEATS; rep++) {
            cudaStream_t st;
            cuda_check(cudaStreamCreate(&st), "st");

            // A: baseline
            {
                int blk = (static_cast<int>(pb) + T - 1) / T;
                auto fn = [&](cudaStream_t s) { read_baseline<<<blk, T, 0, s>>>(d_r0, d_v, pb); };
                auto tr = measure("A: baseline", fn, ITERS, st);
                a_reps[rep] = tr.ms_per_iter;
            }

            // B: serial SUM32 (1 thread/unit)
            {
                auto fn = [&](cudaStream_t s) { sum32_serial<<<static_cast<int>(nu), 1, 0, s>>>(d_r0, nu, ALLOC_UNIT, d_sum); };
                auto tr = measure("B: SUM32 serial 1tpu", fn, ITERS, st);
                b_reps[rep] = tr.ms_per_iter;
            }

            // B2: parallel SUM32 (256 threads/block, warp+block reduce)
            {
                auto fn = [&](cudaStream_t s) { sum32_parallel<<<static_cast<int>(nu), T, 0, s>>>(d_r0, nu, ALLOC_UNIT, d_sum); };
                auto tr = measure("B2: SUM32 parallel 256t/block", fn, ITERS, st);
                b2_reps[rep] = tr.ms_per_iter;
            }

            // D: eager read3 + vote
            {
                int blk = (static_cast<int>(pb) + T - 1) / T;
                auto fn = [&](cudaStream_t s) { eager_vote<<<blk, T, 0, s>>>(d_r0, d_r1, d_r2, d_v, pb); };
                auto tr = measure("D: eager read3 + vote", fn, ITERS, st);
                d_reps[rep] = tr.ms_per_iter;
            }

            cuda_check(cudaStreamDestroy(st), "D st");
        }

        // Write CSV data
        for (int rep = 0; rep < REPEATS; rep++) {
            std::fprintf(fcsv, "%s,%" PRIu64 ",%" PRIu64 ",A,baseline,%d,%.12f\n",
                ds_name.c_str(), nu, ALLOC_UNIT, rep, a_reps[rep]);
            std::fprintf(fcsv, "%s,%" PRIu64 ",%" PRIu64 ",B,sum32_serial,%d,%.12f\n",
                ds_name.c_str(), nu, ALLOC_UNIT, rep, b_reps[rep]);
            std::fprintf(fcsv, "%s,%" PRIu64 ",%" PRIu64 ",B2,sum32_parallel,%d,%.12f\n",
                ds_name.c_str(), nu, ALLOC_UNIT, rep, b2_reps[rep]);
            std::fprintf(fcsv, "%s,%" PRIu64 ",%" PRIu64 ",D,eager_vote,%d,%.12f\n",
                ds_name.c_str(), nu, ALLOC_UNIT, rep, d_reps[rep]);
        }

        // Summary
        auto stats = [&](double *v) {
            double mn = v[0]; for (int i=1;i<REPEATS;i++) mn = std::min(mn, v[i]);
            double mx = v[0]; for (int i=1;i<REPEATS;i++) mx = std::max(mx, v[i]);
            double sum = 0; for (int i=0;i<REPEATS;i++) sum += v[i];
            double mean = sum / REPEATS;
            double var = 0; for (int i=0;i<REPEATS;i++) var += (v[i]-mean)*(v[i]-mean);
            return std::make_tuple(mean, std::sqrt(var/REPEATS), mn, mx);
        };

        auto [a_mn, a_sd, a_lo, a_hi] = stats(a_reps);
        auto [b_mn, b_sd, b_lo, b_hi] = stats(b_reps);
        auto [b2_mn, b2_sd, b2_lo, b2_hi] = stats(b2_reps);
        auto [d_mn, d_sd, d_lo, d_hi] = stats(d_reps);

        std::fprintf(stderr, "\n  Summary (%d repeats):\n", REPEATS);
        std::fprintf(stderr, "    A:  %.6f +- %.6f ms  [%.6f, %.6f]\n", a_mn, a_sd, a_lo, a_hi);
        std::fprintf(stderr, "    B:  %.6f +- %.6f ms  [%.6f, %.6f]  B/A=%.4f  B/B0=%.4f  B<B0=%d\n",
            b_mn, b_sd, b_lo, b_hi, b_mn/a_mn, b_mn/B0_MS, b_mn<B0_MS);
        std::fprintf(stderr, "    B2: %.6f +- %.6f ms  [%.6f, %.6f]  B2/A=%.4f  B2/B0=%.4f  B2<B0=%d  B2<0.9B0=%d\n",
            b2_mn, b2_sd, b2_lo, b2_hi, b2_mn/a_mn, b2_mn/B0_MS,
            b2_mn<B0_MS, b2_mn<0.9*B0_MS);
        std::fprintf(stderr, "    D:  %.6f +- %.6f ms  [%.6f, %.6f]  D/A=%.4f  D/B0=%.4f\n",
            d_mn, d_sd, d_lo, d_hi, d_mn/a_mn, d_mn/B0_MS);

        cuda_check(cudaFree(d_sum), "F sum");
        cuda_check(cudaFree(d_v), "F v");
        cuda_check(cudaFree(d_r2), "F r2");
        cuda_check(cudaFree(d_r1), "F r1");
        cuda_check(cudaFree(d_r0), "F r0");
        for (auto pp : d_planes) cuda_check(cudaFree(pp), "F pl");
    }

    fclose(fcsv);
    std::fprintf(stderr, "\nCSV: %s\n", csv_path.c_str());
    return 0;
}
