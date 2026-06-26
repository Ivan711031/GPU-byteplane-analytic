// NMR-C: NCU Free Redundancy Profiling Benchmark
// Measures latency and hardware metrics for 5 paths:
//   P0: baseline byte-plane k=2 filter+aggregate
//   P1: digest-only (per-plane SUM32)
//   P2: vote/read+compare (byte majority + digest detect)
//   P3: digest+vote (combined)
//   P4: raw FP64 reference
//
// Designed to be profiled with `ncu --set full` per path.
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
//        && make bench_nmr_c_ncu_profile
// Run:   ./bench_nmr_c_ncu_profile --dataset PATH --raw PATH [--path P0] [--csv PATH]

#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <chrono>
#include <string>
#include <string_view>
#include <vector>

#include "../experiment3/exp3_real_data_layout.hpp"
#include "filter_aggregate_kernels.cuh"

using namespace exp3_real;

// ===========================================================================
// Error handling
// ===========================================================================
[[noreturn]] static void die(const char *m) { std::fprintf(stderr, "FATAL: %s\n", m); std::exit(2); }
static void cuda_check(cudaError_t e, const char *w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error at %s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}

// ===========================================================================
// Timing
// ===========================================================================
struct Timing { double ms = 0; };
static Timing measure(std::function<void(cudaStream_t)> fn, int iters, cudaStream_t s) {
    for (int i = 0; i < 3; i++) fn(s);
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

// ===========================================================================
// Kernels
// ===========================================================================

// Per-block partial SUM32 — grid-stride loop, each block writes its partial sum.
// Host side sums block_partials[0..grid-1] to get final digest.
__global__ void digest_sum32_partial(const uint8_t *__restrict__ src, uint64_t n,
    uint32_t *__restrict__ block_partials) {
    uint64_t tid = threadIdx.x;
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
    uint32_t local = 0;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) + tid;
         i < n; i += stride)
        local += static_cast<uint32_t>(src[i]);
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(0xFFFFFFFFu, local, off);
    __shared__ uint32_t s_red[32];
    if ((tid & 31) == 0) s_red[tid >> 5] = local;
    __syncthreads();
    if (tid < 32) {
        uint32_t w = s_red[tid];
        for (int off = 16; off > 0; off >>= 1)
            w += __shfl_down_sync(0xFFFFFFFFu, w, off);
        if (tid == 0) block_partials[blockIdx.x] = w;
    }
}

// Grid-stride byte-level majority vote across r=3 replicas.
// Uses capped grid so each block processes multiple elements.
__global__ void byte_majority_vote_r3_gs(const uint8_t *__restrict__ r0,
    const uint8_t *__restrict__ r1, const uint8_t *__restrict__ r2,
    uint8_t *__restrict__ voted, uint64_t n) {
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
         i < n; i += stride) {
        uint8_t v[3] = {r0[i], r1[i], r2[i]};
        int cnt[3] = {1, 1, 1};
        for (int j = 0; j < 3; j++)
            for (int k = j + 1; k < 3; k++)
                if (v[j] == v[k]) { cnt[j]++; cnt[k]++; }
        int best = 0;
        for (int j = 1; j < 3; j++)
            if (cnt[j] > cnt[best] || (cnt[j] == cnt[best] && v[j] > v[best]))
                best = j;
        voted[i] = v[best];
    }
}

// ===========================================================================
// Path implementations
// ===========================================================================

enum class Path {
    P0_BASELINE_BYTEPLANE,
    P1_DIGEST_ONLY,
    P2_VOTE_READ_COMPARE,
    P3_DIGEST_PLUS_VOTE,
    P4_RAW_FUSED_FP64,
};

static const char* path_label(Path p) {
    switch (p) {
        case Path::P0_BASELINE_BYTEPLANE: return "P0_baseline_byteplane_k2";
        case Path::P1_DIGEST_ONLY:        return "P1_digest_only";
        case Path::P2_VOTE_READ_COMPARE:  return "P2_vote_read_compare";
        case Path::P3_DIGEST_PLUS_VOTE:   return "P3_digest_plus_vote";
        case Path::P4_RAW_FUSED_FP64:     return "P4_raw_fused_fp64_reference";
    }
    return "UNKNOWN";
}

static Path parse_path(const char *s) {
    std::string_view sv(s);
    if (sv == "P0" || sv == "P0_baseline_byteplane_k2") return Path::P0_BASELINE_BYTEPLANE;
    if (sv == "P1" || sv == "P1_digest_only")           return Path::P1_DIGEST_ONLY;
    if (sv == "P2" || sv == "P2_vote_read_compare")     return Path::P2_VOTE_READ_COMPARE;
    if (sv == "P3" || sv == "P3_digest_plus_vote")      return Path::P3_DIGEST_PLUS_VOTE;
    if (sv == "P4" || sv == "P4_raw_fused_fp64")        return Path::P4_RAW_FUSED_FP64;
    die("unknown path (use P0,P1,P2,P3,P4)");
    return Path::P0_BASELINE_BYTEPLANE;
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char **argv) {
    std::string dataset_path, raw_path, csv_path = "nmr_c_ncu_profile.csv";
    Path selected_path = Path::P0_BASELINE_BYTEPLANE;
    double threshold = 0.0;
    int iters = 50, warmup = 5, block_threads = 256, k2_planes = 2;
    bool path_set = false;

    for (int i = 1; i < argc; ++i) {
        std::string_view a(argv[i]);
        auto val = [&]() { if (++i >= argc) die("missing value"); return std::string(argv[i]); };
        if (a == "--help" || a == "-h") {
            std::fprintf(stderr,
                "Usage: %s --dataset PATH --raw PATH [--path P0|P1|P2|P3|P4]\n"
                "  [--csv PATH] [--threshold FP64] [--iters N] [--k2 N]\n",
                argv[0]);
            return 0;
        } else if (a == "--dataset")     dataset_path = val();
        else if (a == "--raw")           raw_path = val();
        else if (a == "--path")          { selected_path = parse_path(val().c_str()); path_set = true; }
        else if (a == "--csv")           csv_path = val();
        else if (a == "--threshold")     threshold = std::stod(val());
        else if (a == "--iters")         iters = std::stoi(val());
        else if (a == "--k2")            k2_planes = std::stoi(val());
        else if (a == "--block-threads") block_threads = std::stoi(val());
        else { std::string msg = "unknown: "; msg += a; die(msg.c_str()); }
    }
    if (dataset_path.empty()) die("--dataset required");
    if (raw_path.empty()) die("--raw required");

    cuda_check(cudaSetDevice(0), "dev");
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, 0), "prop");
    std::fprintf(stderr, "Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::fprintf(stderr, "Path: %s\n", path_label(selected_path));

    // Load dataset
    Dataset ds = exp3_real::load_dataset(dataset_path);
    uint64_t n = ds.manifest.value_count;
    uint64_t max_planes = ds.manifest.max_plane_count;
    const int PLANES = static_cast<int>(max_planes);
    std::string ds_name = ds.manifest.dataset;

    std::fprintf(stderr, "Dataset: %s n=%" PRIu64 " planes=%d\n", ds_name.c_str(), n, PLANES);

    // Upload all planes to GPU
    std::vector<uint8_t*> d_planes(PLANES);
    for (int p = 0; p < PLANES; p++) {
        cuda_check(cudaMalloc(&d_planes[p], ds.planes[p].size()), "M plane");
        cuda_check(cudaMemcpy(d_planes[p], ds.planes[p].data(), ds.planes[p].size(),
            cudaMemcpyHostToDevice), "H2D plane");
    }

    // Replica buffers (for P2/P3)
    constexpr int R3 = 3;
    uint8_t *d_r[R3] = {};
    for (int j = 0; j < R3; j++) {
        cuda_check(cudaMalloc(&d_r[j], n), "M r");
        cuda_check(cudaMemcpy(d_r[j], ds.planes[0].data(), n, cudaMemcpyHostToDevice), "H2D r");
    }
    uint8_t *d_voted = nullptr;
    cuda_check(cudaMalloc(&d_voted, n), "M voted");

    // Digest single-value output (host-reduce from per-block partials allocated later)
    uint32_t *d_digest = nullptr;
    cuda_check(cudaMalloc(&d_digest, sizeof(uint32_t)), "M digest");

    // Load raw FP64 data to GPU (for P4)
    double *d_raw = nullptr;
    if (!raw_path.empty() && std::filesystem::exists(raw_path)) {
        uint64_t file_size = std::filesystem::file_size(raw_path);
        cuda_check(cudaMalloc(&d_raw, file_size), "M raw");
        std::ifstream raw_in(raw_path, std::ios::binary);
        if (!raw_in) die("cannot open raw file");
        constexpr size_t CHUNK = 256ull * 1024ull * 1024ull;
        std::vector<double> chunk(CHUNK / sizeof(double));
        uint64_t remaining = file_size;
        uint64_t offset = 0;
        while (remaining > 0) {
            size_t to_read = static_cast<size_t>(std::min<uint64_t>(remaining, CHUNK));
            raw_in.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(to_read));
            cuda_check(cudaMemcpy(reinterpret_cast<uint8_t*>(d_raw) + offset,
                chunk.data(), to_read, cudaMemcpyHostToDevice), "H2D raw chunk");
            offset += to_read;
            remaining -= to_read;
        }
    }

    int T = block_threads;
    int blk = static_cast<int>((n + static_cast<uint64_t>(T) - 1) / static_cast<uint64_t>(T));
    // Grid-stride kernels use a capped grid to avoid 650K+ block launches on large datasets.
    // Vote kernels still use full blk (one element per thread).
    int grid_digest = std::min(blk, 512);
    // Per-block partial SUM32 array (based on capped digest grid)
    uint32_t *d_partial = nullptr;
    cuda_check(cudaMalloc(&d_partial, static_cast<size_t>(grid_digest) * sizeof(uint32_t)), "M partial");
    cudaStream_t st;
    cuda_check(cudaStreamCreate(&st), "st");

    // Default r-vector for graded NMR: r=[3,2,1,1,1,1,1,1]
    // Plane 0 gets r=3 replicas for majority vote.
    // For P2/P3, we use plane 0 with r=3 replicas.

    double latency_ms = 0.0;
    double effective_bytes = 0.0;

    switch (selected_path) {
    case Path::P0_BASELINE_BYTEPLANE: {
        // P0: Baseline byte-plane k=2 filter+aggregate
        // Uses the progressive_filter_sum_rowpack16_byte_mask kernel.
        // For simplicity, we do a direct read + approximate sum with k planes.
        // Set up minimal segment geometry (1 segment over whole dataset).
        uint64_t segment_rows = n;
        uint64_t tiles_per_segment = (n + (static_cast<uint64_t>(T) * FA_ROWPACK16_WIDTH) - 1)
            / (static_cast<uint64_t>(T) * FA_ROWPACK16_WIDTH);
        uint64_t grid = tiles_per_segment;

        // Build threshold bytes (set to 0 threshold = include all rows with byte > 0)
        std::vector<uint8_t> h_threshold_flat(static_cast<size_t>(max_planes), 0);
        uint8_t *d_threshold = nullptr;
        cuda_check(cudaMalloc(&d_threshold, max_planes), "M thresh");
        cuda_check(cudaMemcpy(d_threshold, h_threshold_flat.data(), max_planes,
            cudaMemcpyHostToDevice), "H2D thresh");

        uint32_t *d_active_count = nullptr;
        uint32_t active_val = static_cast<uint32_t>(k2_planes);
        cuda_check(cudaMalloc(&d_active_count, sizeof(uint32_t)), "M active");
        cuda_check(cudaMemcpy(d_active_count, &active_val, sizeof(uint32_t),
            cudaMemcpyHostToDevice), "H2D active");

        double seg_base = 0.0;
        double *d_seg_base = nullptr;
        cuda_check(cudaMalloc(&d_seg_base, sizeof(double)), "M base");
        cuda_check(cudaMemcpy(d_seg_base, &seg_base, sizeof(double),
            cudaMemcpyHostToDevice), "H2D base");

        std::vector<double> h_basis(static_cast<size_t>(max_planes), 256.0);
        double *d_basis = nullptr;
        cuda_check(cudaMalloc(&d_basis, max_planes * sizeof(double)), "M basis");
        cuda_check(cudaMemcpy(d_basis, h_basis.data(), max_planes * sizeof(double),
            cudaMemcpyHostToDevice), "H2D basis");

        uint64_t *d_counts = nullptr;
        double *d_sums = nullptr;
        cuda_check(cudaMalloc(&d_counts, grid * sizeof(uint64_t)), "M counts");
        cuda_check(cudaMalloc(&d_sums, grid * sizeof(double)), "M sums");

        FaRuntimeU8Subcolumns h_sc{};
        for (int p = 0; p < std::min(PLANES, FA_MAX_RUNTIME_PLANES); p++)
            h_sc.ptrs[p] = d_planes[p];
        for (int p = PLANES; p < FA_MAX_RUNTIME_PLANES; p++)
            h_sc.ptrs[p] = nullptr;

        auto fn = [&](cudaStream_t s) {
            progressive_filter_sum_rowpack16_byte_mask<<<static_cast<int>(grid), T, 0, s>>>(
                h_sc, n, segment_rows, tiles_per_segment,
                d_threshold, static_cast<int>(max_planes),
                static_cast<int>(max_planes), d_active_count,
                d_seg_base, d_basis, d_counts, d_sums);
        };

        auto timing = measure(fn, iters, st);
        latency_ms = timing.ms;
        effective_bytes = static_cast<double>(n) * static_cast<double>(k2_planes);

        cuda_check(cudaFree(d_threshold), "F thresh");
        cuda_check(cudaFree(d_active_count), "F active");
        cuda_check(cudaFree(d_seg_base), "F base");
        cuda_check(cudaFree(d_basis), "F basis");
        cuda_check(cudaFree(d_counts), "F counts");
        cuda_check(cudaFree(d_sums), "F sums");
        break;
    }

    case Path::P1_DIGEST_ONLY: {
        // P1: Per-plane SUM32 digest. No vote, no filter.
        // Read each plane and compute SUM32 digest (grid-stride + host-reduce partials).
        auto fn = [&](cudaStream_t s) {
            for (int p = 0; p < PLANES; p++) {
                digest_sum32_partial<<<grid_digest, T, 0, s>>>(d_planes[p], n, d_partial);
            }
        };
        auto timing = measure(fn, iters, st);
        latency_ms = timing.ms;
        effective_bytes = static_cast<double>(n) * static_cast<double>(PLANES);
        break;
    }

    case Path::P2_VOTE_READ_COMPARE: {
        // P2: Byte-level majority vote + digest detect (grid-stride kernels)
        auto fn = [&](cudaStream_t s) {
            byte_majority_vote_r3_gs<<<grid_digest, T, 0, s>>>(d_r[0], d_r[1], d_r[2], d_voted, n);
            digest_sum32_partial<<<grid_digest, T, 0, s>>>(d_voted, n, d_partial);
        };
        auto timing = measure(fn, iters, st);
        latency_ms = timing.ms;
        effective_bytes = static_cast<double>(n) * 4.0;
        break;
    }

    case Path::P3_DIGEST_PLUS_VOTE: {
        // P3: Digest + Vote combined (grid-stride kernels).
        uint32_t *d_partial2 = nullptr;
        cuda_check(cudaMalloc(&d_partial2, static_cast<size_t>(grid_digest) * sizeof(uint32_t)), "M partial2");

        auto fn = [&](cudaStream_t s) {
            for (int j = 0; j < R3; j++)
                digest_sum32_partial<<<grid_digest, T, 0, s>>>(d_r[j], n, d_partial2);
            byte_majority_vote_r3_gs<<<grid_digest, T, 0, s>>>(d_r[0], d_r[1], d_r[2], d_voted, n);
            digest_sum32_partial<<<grid_digest, T, 0, s>>>(d_voted, n, d_partial);
        };

        auto timing = measure(fn, iters, st);
        latency_ms = timing.ms;
        effective_bytes = static_cast<double>(n) * 6.0;
        cuda_check(cudaFree(d_partial2), "F partial2");
        break;
    }

    case Path::P4_RAW_FUSED_FP64: {
        // P4: Raw FP64 filter+aggregate baseline
        uint64_t *d_raw_counts = nullptr;
        double *d_raw_sums = nullptr;
        cuda_check(cudaMalloc(&d_raw_counts, static_cast<size_t>(blk) * sizeof(uint64_t)), "M rc");
        cuda_check(cudaMalloc(&d_raw_sums, static_cast<size_t>(blk) * sizeof(double)), "M rs");

        auto fn = [&](cudaStream_t s) {
            raw_fp64_filter_sum<<<blk, T, 0, s>>>(d_raw, n, threshold, d_raw_counts, d_raw_sums);
        };

        auto timing = measure(fn, iters, st);
        latency_ms = timing.ms;
        effective_bytes = static_cast<double>(n) * sizeof(double);

        cuda_check(cudaFree(d_raw_counts), "F rc");
        cuda_check(cudaFree(d_raw_sums), "F rs");
        break;
    }
    }

    // Compute bandwidth
    double sec = latency_ms / 1000.0;
    double bw_gb_s = (sec > 0.0) ? (effective_bytes / sec / 1e9) : 0.0;

    // Output CSV row
    FILE *fcsv = std::fopen(csv_path.c_str(), "a");
    if (!fcsv) die("cannot open CSV for append");
    if (std::ftell(fcsv) == 0) {
        std::fprintf(fcsv, "dataset,n_rows,path,latency_ms,effective_bytes,effective_bandwidth_gb_s\n");
    }
    std::fprintf(fcsv, "%s,%" PRIu64 ",%s,%.6f,%.0f,%.6f\n",
        ds_name.c_str(), n, path_label(selected_path),
        latency_ms, effective_bytes, bw_gb_s);
    std::fclose(fcsv);

    std::fprintf(stderr, "\nResults:\n");
    std::fprintf(stderr, "  path=%s  latency=%.6f ms  bw=%.6f GB/s\n",
        path_label(selected_path), latency_ms, bw_gb_s);
    std::fprintf(stderr, "  CSV: %s\n", csv_path.c_str());

    // Cleanup
    cuda_check(cudaStreamDestroy(st), "D st");
    for (int j = 0; j < R3; j++) cuda_check(cudaFree(d_r[j]), "F r");
    cuda_check(cudaFree(d_voted), "F voted");
    cuda_check(cudaFree(d_partial), "F partial");
    cuda_check(cudaFree(d_digest), "F digest");
    if (d_raw) cuda_check(cudaFree(d_raw), "F raw");
    for (int p = 0; p < PLANES; p++) cuda_check(cudaFree(d_planes[p]), "F plane");

    return 0;
}
