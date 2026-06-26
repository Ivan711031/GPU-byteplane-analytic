// NMR-R0 GPU smoke: byte-level majority vote across r=3 separate buffers.
// Inject fault into one buffer → byte majority → verify matches clean.
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
//        && make bench_nmr_r0_gpu_smoke
// Run:   ./bench_nmr_r0_gpu_smoke --dataset PATH [--dataset PATH2]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "../experiment3/exp3_real_data_layout.hpp"

using namespace exp3_real;

static void die(const char *m) { std::fprintf(stderr, "FATAL: %s\n", m); std::exit(2); }
static void cuda_check(cudaError_t e, const char *w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error at %s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}

struct Timing { double ms = 0; };
static Timing measure(std::function<void(cudaStream_t)> fn, int iters, cudaStream_t s) {
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
    Timing r; r.ms = ms / iters; return r;
}

// ── Byte-level majority vote across r=3 buffers ──────────────────────
// Each thread reads one byte from each replica, selects majority, writes to voted.
__global__ void byte_majority_vote_r3(const uint8_t *__restrict__ r0,
    const uint8_t *__restrict__ r1, const uint8_t *__restrict__ r2,
    uint8_t *__restrict__ voted, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t v[3] = {r0[i], r1[i], r2[i]};
    // Count occurrences of each value
    int cnt[3] = {1, 1, 1};
    for (int j = 0; j < 3; j++)
        for (int k = j + 1; k < 3; k++)
            if (v[j] == v[k]) { cnt[j]++; cnt[k]++; }
    // Majority: value with count >= 2. Tie-break: larger byte value.
    int best = 0;
    for (int j = 1; j < 3; j++)
        if (cnt[j] > cnt[best] || (cnt[j] == cnt[best] && v[j] > v[best]))
            best = j;
    voted[i] = v[best];
}

int main(int argc, char **argv) {
    std::vector<std::string> dataset_paths;
    for (int i = 1; i < argc; ++i) {
        std::string_view a(argv[i]);
        auto val=[&](){if(++i>=argc)die("missing arg");return std::string(argv[i]);};
        if (a=="--dataset") dataset_paths.push_back(val());
        else if (a=="--help"||a=="-h") { std::fprintf(stderr,"Usage: %s --dataset PATH [--dataset PATH2]\n",argv[0]); return 0; }
    }
    if (dataset_paths.empty()) die("--dataset required");

    cuda_check(cudaSetDevice(0), "dev");
    cudaDeviceProp p;
    cuda_check(cudaGetDeviceProperties(&p, 0), "prop");
    std::fprintf(stderr, "Device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    std::fprintf(stderr, "Job: %s\n\n", std::getenv("SLURM_JOB_ID") ? std::getenv("SLURM_JOB_ID") : "NOJOB");

    int T = 256;

    for (auto &ds_path : dataset_paths) {
        Dataset ds = exp3_real::load_dataset(ds_path);
        if (ds.planes.empty()) { std::fprintf(stderr, "skip (no planes)\n"); continue; }

        // Use plane 0 (MSB) for the test
        uint8_t *clean_h = ds.planes[0].data();
        uint64_t n = ds.planes[0].size();
        std::string ds_name = ds.manifest.dataset;
        std::fprintf(stderr, "=== %s: plane0=%" PRIu64 " bytes ===\n", ds_name.c_str(), n);

        // Allocate 3 separate GPU buffers for replicas
        uint8_t *d_r0, *d_r1, *d_r2, *d_voted;
        cuda_check(cudaMalloc(&d_r0, n), "M r0");
        cuda_check(cudaMalloc(&d_r1, n), "M r1");
        cuda_check(cudaMalloc(&d_r2, n), "M r2");
        cuda_check(cudaMalloc(&d_voted, n), "M voted");

        // Copy same clean data to all 3 replicas
        cuda_check(cudaMemcpy(d_r0, clean_h, n, cudaMemcpyHostToDevice), "H2D r0");
        cuda_check(cudaMemcpy(d_r1, clean_h, n, cudaMemcpyHostToDevice), "H2D r1");
        cuda_check(cudaMemcpy(d_r2, clean_h, n, cudaMemcpyHostToDevice), "H2D r2");

        int blk = (static_cast<int>(n) + T - 1) / T;

        // ── Test A: one-replica fault ──
        // Inject fault into byte 0 of replica 0 only
        uint8_t fault_byte = clean_h[0] ^ 0xFF;
        cuda_check(cudaMemcpy(d_r0, &fault_byte, 1, cudaMemcpyHostToDevice), "inject A");

        byte_majority_vote_r3<<<blk, T>>>(d_r0, d_r1, d_r2, d_voted, n);
        cuda_check(cudaDeviceSynchronize(), "vote A");

        // Compare voted with clean
        std::vector<uint8_t> voted_h(n);
        cuda_check(cudaMemcpy(voted_h.data(), d_voted, n, cudaMemcpyDeviceToHost), "D2H A");
        uint64_t mismatches_a = 0;
        for (uint64_t i = 0; i < n; i++)
            if (voted_h[i] != clean_h[i]) mismatches_a++;
        double rec_a = 1.0 - (double)mismatches_a / (double)n;

        std::fprintf(stderr, "  A (one-replica fault): mismatches=%" PRIu64 "/%" PRIu64
            "  recovery=%.6f  %s\n", mismatches_a, n, rec_a,
            mismatches_a == 0 ? "✅ RECOVERED" : "❌ FAILED");

        // ── Timing: fused baseline read (no vote) vs read+vote ──
        constexpr int ITERS = 100;
        cudaStream_t st;
        cuda_check(cudaStreamCreate(&st), "st");
        auto t_baseline = measure([&](cudaStream_t s) {
            byte_majority_vote_r3<<<blk, T, 0, s>>>(d_r0, d_r0, d_r0, d_voted, n);
        }, ITERS, st);
        auto t_vote = measure([&](cudaStream_t s) {
            byte_majority_vote_r3<<<blk, T, 0, s>>>(d_r0, d_r1, d_r2, d_voted, n);
        }, ITERS, st);
        cuda_check(cudaStreamDestroy(st), "D st");
        double vote_overhead = t_vote.ms - t_baseline.ms;
        std::fprintf(stderr, "  Timing: baseline=%.6f ms  read+vote=%.6f ms  overhead=%.6f ms\n",
            t_baseline.ms, t_vote.ms, vote_overhead);

        // ── Test B: same-fault-all-replicas (reproduce G2) ──
        // Re-inject: fault ALL replicas at byte 0
        cuda_check(cudaMemcpy(d_r0, &fault_byte, 1, cudaMemcpyHostToDevice), "inject B r0");
        cuda_check(cudaMemcpy(d_r1, &fault_byte, 1, cudaMemcpyHostToDevice), "inject B r1");
        cuda_check(cudaMemcpy(d_r2, &fault_byte, 1, cudaMemcpyHostToDevice), "inject B r2");

        byte_majority_vote_r3<<<blk, T>>>(d_r0, d_r1, d_r2, d_voted, n);
        cuda_check(cudaDeviceSynchronize(), "vote B");

        cuda_check(cudaMemcpy(voted_h.data(), d_voted, n, cudaMemcpyDeviceToHost), "D2H B");
        uint64_t mismatches_b = 0;
        for (uint64_t i = 0; i < n; i++)
            if (voted_h[i] != clean_h[i]) mismatches_b++;
        double rec_b = 1.0 - (double)mismatches_b / (double)n;

        std::fprintf(stderr, "  B (same-fault-all):     mismatches=%" PRIu64 "/%" PRIu64
            "  recovery=%.6f  %s\n", mismatches_b, n, rec_b,
            mismatches_b == 0 ? "⚠️ UNEXPECTED RECOVERY" : "✅ ESCAPED (expected)");

        // ── Test C: two-of-three corrupted at different offsets ──
        // Reset to clean, inject fault into replica 0 and replica 1 at different offsets
        cuda_check(cudaMemcpy(d_r0, clean_h, n, cudaMemcpyHostToDevice), "reset C r0");
        cuda_check(cudaMemcpy(d_r1, clean_h, n, cudaMemcpyHostToDevice), "reset C r1");
        cuda_check(cudaMemcpy(d_r2, clean_h, n, cudaMemcpyHostToDevice), "reset C r2");

        uint8_t fb = clean_h[0] ^ 0xFF;
        uint8_t fb2 = clean_h[8] ^ 0xFF;
        cuda_check(cudaMemcpy(d_r0, &fb, 1, cudaMemcpyHostToDevice), "inject C r0");
        cuda_check(cudaMemcpy(d_r1 + 8, &fb2, 1, cudaMemcpyHostToDevice), "inject C r1");

        byte_majority_vote_r3<<<blk, T>>>(d_r0, d_r1, d_r2, d_voted, n);
        cuda_check(cudaDeviceSynchronize(), "vote C");

        cuda_check(cudaMemcpy(voted_h.data(), d_voted, n, cudaMemcpyDeviceToHost), "D2H C");
        uint64_t mismatches_c = 0;
        for (uint64_t i = 0; i < n; i++)
            if (voted_h[i] != clean_h[i]) mismatches_c++;

        cuda_check(cudaFree(d_r0), "F r0");
        cuda_check(cudaFree(d_r1), "F r1");
        cuda_check(cudaFree(d_r2), "F r2");
        cuda_check(cudaFree(d_voted), "F voted");

        std::fprintf(stderr, "  C (two-of-three diff):  mismatches=%" PRIu64 "/%" PRIu64
            "  %s\n", mismatches_c, n, mismatches_c == 0 ? "✅ RECOVERED (expected)" : "⚠️ PARTIAL");

        // ── Verdict ──
        bool pass = (mismatches_a == 0) && (mismatches_b > 0);
        std::fprintf(stderr, "\n  Verdict: %s\n",
            pass ? "NMR_R0_GPU_RECOVERY_CONFIRMED ✅" : "NMR_R0_GPU_KERNEL_NEEDS_FIXES ❌");
    }
    return 0;
}
