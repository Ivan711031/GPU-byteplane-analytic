// Phase 3-Z0d: Cheap integrity digest bandwidth de-risk
// Replaces CRC32 with parallel-reducible digests (xor64, sum32, combined).
// Artifact-time stores reference digest; runtime recomputes over actual bytes.
//
// Paths with digest types:
//   A: baseline read (no digest)
//   B: read + xor64 common
//   B2: read + sum32 common
//   B3: read + xor64+sum32 combined common
//   C: read + digest + detect mismatch + repair
//   D: eager read3 + vote

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "../experiment3/exp3_real_data_layout.hpp"

using namespace exp3_real;
namespace fs = std::filesystem;

constexpr uint64_t ALLOC_UNIT = 4096;
constexpr int ITERS = 100;

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
    std::fprintf(stderr, "  %-52s %9.4f ms total  %9.6f ms/iter\n", label, ms, r.ms_per_iter);
    return r;
}

// ===================================================================
// Path A: baseline read (byte copy)
// ===================================================================
__global__ void read_baseline(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ===================================================================
// Path B: read + compute XOR64 digest (fused: 1 thread/unit serial XOR)
// ===================================================================
__global__ void read_xor64(const uint8_t *__restrict__ src, uint64_t n_units,
    uint64_t unit_b, uint64_t *__restrict__ digests) {
    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    uint64_t base = u * unit_b;
    uint64_t x = 0;
    for (uint64_t b = 0; b < unit_b; b++)
        x ^= static_cast<uint64_t>(src[base + b]) << ((b & 7) * 8);
    digests[u] = x;
}

// ===================================================================
// Path B2: read + SUM32 digest (fused)
// ===================================================================
__global__ void read_sum32(const uint8_t *__restrict__ src, uint64_t n_units,
    uint64_t unit_b, uint32_t *__restrict__ digests) {
    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    uint64_t base = u * unit_b;
    uint32_t s = 0;
    for (uint64_t b = 0; b < unit_b; b++)
        s += static_cast<uint32_t>(src[base + b]);
    digests[u] = s;
}

// ===================================================================
// Path B3: read + combined XOR64+SUM32 (fused, 2 outputs)
// ===================================================================
__global__ void read_combined(const uint8_t *__restrict__ src, uint64_t n_units,
    uint64_t unit_b, uint64_t *__restrict__ xor_out, uint32_t *__restrict__ sum_out) {
    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    uint64_t base = u * unit_b;
    uint64_t x = 0;
    uint32_t s = 0;
    for (uint64_t b = 0; b < unit_b; b++) {
        uint8_t v = src[base + b];
        x ^= static_cast<uint64_t>(v) << ((b & 7) * 8);
        s += static_cast<uint32_t>(v);
    }
    xor_out[u] = x;
    sum_out[u] = s;
}

// ===================================================================
// Path C: read + combined digest + detect mismatch + repair
// Digest computed by thread 0. If mismatch flagged, all threads vote.
// ===================================================================
__global__ void read_digest_repair(const uint8_t *__restrict__ rep0,
    const uint8_t *__restrict__ rep1, const uint8_t *__restrict__ rep2,
    uint8_t *__restrict__ voted, uint64_t n_units, uint64_t unit_b,
    const uint8_t *__restrict__ fault_flags,
    uint64_t *__restrict__ xor_out, uint32_t *__restrict__ sum_out) {
    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    uint64_t base = u * unit_b;

    if (threadIdx.x == 0) {
        uint64_t x = 0; uint32_t s = 0;
        for (uint64_t b = 0; b < unit_b; b++) {
            uint8_t v = rep0[base + b];
            x ^= static_cast<uint64_t>(v) << ((b & 7) * 8);
            s += static_cast<uint32_t>(v);
        }
        xor_out[u] = x;
        sum_out[u] = s;
    }
    __syncthreads();

    if (fault_flags[u]) {
        uint64_t tid = threadIdx.x;
        uint64_t step = blockDim.x;
        for (uint64_t b = tid; b < unit_b; b += step) {
            uint64_t off = base + b;
            uint8_t a = rep0[off], b0 = rep1[off], c = rep2[off];
            voted[off] = (a == b0 || a == c) ? a : b0;
        }
    }
}

// ===================================================================
// Path D: eager read3 + warp vote
// ===================================================================
__global__ void eager_vote(const uint8_t *__restrict__ rep0,
    const uint8_t *__restrict__ rep1, const uint8_t *__restrict__ rep2,
    uint8_t *__restrict__ voted, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t a = rep0[i], b0 = rep1[i], c = rep2[i];
    voted[i] = (a == b0 || a == c) ? a : b0;
}

// ===================================================================
// Main
// ===================================================================
int main(int argc, char **argv) {
    std::string dataset_path, csv_path = "z0d_integrity_profile.csv";
    uint64_t forced_k = 0;

    for (int i = 1; i < argc; ++i) {
        std::string_view a(argv[i]);
        auto val=[&](){if(++i>=argc)die("missing arg");return std::string(argv[i]);};
        if (a=="--help"||a=="-h") {
            std::fprintf(stderr,"Usage: %s --dataset PATH [--csv PATH] [--k N]\n",argv[0]);
            return 0;
        } else if (a=="--dataset") dataset_path=val();
        else if (a=="--csv") csv_path=val();
        else if (a=="--k") forced_k=std::stoul(val());
        else { std::string m="unknown: ";m+=a;die(m.c_str()); }
    }
    if (dataset_path.empty()) die("--dataset required");

    cuda_check(cudaSetDevice(0), "dev");
    cudaDeviceProp p;
    cuda_check(cudaGetDeviceProperties(&p, 0), "prop");
    std::fprintf(stderr, "Device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    std::fprintf(stderr, "Mem: %.1f GB  BW: %.0f GB/s\n",
        p.totalGlobalMem / 1e9, 2.0 * p.memoryClockRate / 1e3 * p.memoryBusWidth / 8.0);

    Dataset ds = exp3_real::load_dataset(dataset_path);
    uint64_t pb = ds.planes.empty() ? 0 : ds.planes[0].size();
    uint64_t n_rows = ds.manifest.value_count;
    uint64_t nu = (pb + ALLOC_UNIT - 1) / ALLOC_UNIT;

    std::fprintf(stderr, "Dataset: %s  rows=%" PRIu64 "  p0=%" PRIu64 "  units=%" PRIu64 "\n",
        ds.manifest.dataset.c_str(), n_rows, pb, nu);

    // Upload plane data
    std::vector<uint8_t*> d_planes;
    for (auto &pl : ds.planes) {
        uint8_t *ptr = nullptr;
        cuda_check(cudaMalloc(&ptr, pl.size()), "M pl");
        cuda_check(cudaMemcpy(ptr, pl.data(), pl.size(), cudaMemcpyHostToDevice), "H2D pl");
        d_planes.push_back(ptr);
    }

    uint8_t *d_r0, *d_r1, *d_r2, *d_v, *d_ff;
    uint64_t *d_xor;
    uint32_t *d_sum;
    cuda_check(cudaMalloc(&d_r0, pb), "M r0");
    cuda_check(cudaMalloc(&d_r1, pb), "M r1");
    cuda_check(cudaMalloc(&d_r2, pb), "M r2");
    cuda_check(cudaMalloc(&d_v, pb), "M v");
    cuda_check(cudaMalloc(&d_ff, nu), "M ff");
    cuda_check(cudaMalloc(&d_xor, nu * sizeof(uint64_t)), "M xor");
    cuda_check(cudaMalloc(&d_sum, nu * sizeof(uint32_t)), "M sum");
    cuda_check(cudaMemset(d_ff, 0, nu), "Mset ff");
    cuda_check(cudaMemcpy(d_r0, ds.planes[0].data(), pb, cudaMemcpyHostToDevice), "H2D r0");
    cuda_check(cudaMemcpy(d_r1, ds.planes[0].data(), pb, cudaMemcpyHostToDevice), "H2D r1");
    cuda_check(cudaMemcpy(d_r2, ds.planes[0].data(), pb, cudaMemcpyHostToDevice), "H2D r2");

    cudaStream_t st;
    cuda_check(cudaStreamCreate(&st), "st");

    FILE *fcsv = std::fopen(csv_path.c_str(), "w");
    if (!fcsv) die("write CSV");
    const char *jid = std::getenv("SLURM_JOB_ID"); if (!jid) jid = "NOJOB";
    std::fprintf(fcsv, "# z0d_integrity_derisk  %s sm_%d%d  job=%s\n", p.name, p.major, p.minor, jid);
    std::fprintf(fcsv, "# dataset=%s  rows=%" PRIu64 "  plane0=%" PRIu64 "  units=%" PRIu64 "\n",
        ds.manifest.dataset.c_str(), n_rows, pb, nu);
    std::fprintf(fcsv, "path,label,ms_per_iter,ms_per_unit,ms_per_byte\n");

    std::vector<double> frs = {0.0, 1e-7, 1e-6, 1e-5};
    int T = 256;
    double B0_MS = 0.8702;

    std::fprintf(stderr, "\n=== Z0d: Cheap Integrity Digest Microbenchmark ===\n");

    // A: baseline
    {
        int blk = (static_cast<int>(pb) + T - 1) / T;
        auto fn = [&](cudaStream_t s) { read_baseline<<<blk, T, 0, s>>>(d_r0, d_v, pb); };
        auto tr = measure("A: single-replica baseline", fn, ITERS, st);
        std::fprintf(fcsv, "A,baseline,%.12f,%.12f,%.12f\n",
            tr.ms_per_iter, tr.ms_per_iter / nu, tr.ms_per_iter / pb);
    }

    // B: read + XOR64 digest
    {
        auto fn = [&](cudaStream_t s) { read_xor64<<<static_cast<int>(nu), 1, 0, s>>>(d_r0, nu, ALLOC_UNIT, d_xor); };
        auto tr = measure("B: read + XOR64 digest (fused)", fn, ITERS, st);
        std::fprintf(fcsv, "B,read_xor64,%.12f,%.12f,%.12f\n",
            tr.ms_per_iter, tr.ms_per_iter / nu, tr.ms_per_iter / pb);
    }

    // B2: read + SUM32 digest
    {
        auto fn = [&](cudaStream_t s) { read_sum32<<<static_cast<int>(nu), 1, 0, s>>>(d_r0, nu, ALLOC_UNIT, d_sum); };
        auto tr = measure("B2: read + SUM32 digest (fused)", fn, ITERS, st);
        std::fprintf(fcsv, "B2,read_sum32,%.12f,%.12f,%.12f\n",
            tr.ms_per_iter, tr.ms_per_iter / nu, tr.ms_per_iter / pb);
    }

    // B3: read + combined XOR64+SUM32
    {
        auto fn = [&](cudaStream_t s) { read_combined<<<static_cast<int>(nu), 1, 0, s>>>(d_r0, nu, ALLOC_UNIT, d_xor, d_sum); };
        auto tr = measure("B3: read + XOR64+SUM32 (combined)", fn, ITERS, st);
        std::fprintf(fcsv, "B3,read_combined,%.12f,%.12f,%.12f\n",
            tr.ms_per_iter, tr.ms_per_iter / nu, tr.ms_per_iter / pb);
    }

    // C: combined digest + repair dispatch
    for (double fr : frs) {
        uint64_t nf = static_cast<uint64_t>(nu * fr);
        if (fr > 0 && nf == 0) nf = 1;
        std::vector<uint8_t> h_ff(nu, 0);
        for (uint64_t u = 0; u < nf && u < nu; u++)
            h_ff[(u * 2654435761ull) % nu] = 1;
        cuda_check(cudaMemcpy(d_ff, h_ff.data(), nu, cudaMemcpyHostToDevice), "H2D ff");
        auto fn = [&](cudaStream_t s) {
            read_digest_repair<<<static_cast<int>(nu), T, 0, s>>>(
                d_r0, d_r1, d_r2, d_v, nu, ALLOC_UNIT, d_ff, d_xor, d_sum);
        };
        char lb[80];
        std::snprintf(lb, sizeof(lb), "C: digest+repair (fr=%.0e, nf=%" PRIu64 ")", fr, nf);
        auto tr = measure(lb, fn, ITERS, st);
        std::fprintf(fcsv, "C,digest_repair_fr%.0e,%.12f,%.12f,%.12f\n",
            fr, tr.ms_per_iter, tr.ms_per_iter / nu, tr.ms_per_iter / pb);
    }

    // D: eager read3 + vote
    {
        int blk = (static_cast<int>(pb) + T - 1) / T;
        auto fn = [&](cudaStream_t s) { eager_vote<<<blk, T, 0, s>>>(d_r0, d_r1, d_r2, d_v, pb); };
        auto tr = measure("D: eager read3 + warp vote", fn, ITERS, st);
        std::fprintf(fcsv, "D,eager_read3_vote,%.12f,%.12f,%.12f\n",
            tr.ms_per_iter, tr.ms_per_iter / nu, tr.ms_per_iter / pb);
    }

    fclose(fcsv);
    std::fprintf(stderr, "\nCSV: %s\n", csv_path.c_str());

    // Cleanup
    cuda_check(cudaFree(d_xor), "F xor"); cuda_check(cudaFree(d_sum), "F sum");
    cuda_check(cudaFree(d_ff), "F ff"); cuda_check(cudaFree(d_v), "F v");
    cuda_check(cudaFree(d_r2), "F r2"); cuda_check(cudaFree(d_r1), "F r1"); cuda_check(cudaFree(d_r0), "F r0");
    for (auto pp : d_planes) cuda_check(cudaFree(pp), "F pl");
    cuda_check(cudaStreamDestroy(st), "D st");
    return 0;
}
