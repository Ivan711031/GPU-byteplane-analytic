// Phase 3-X3: GPU-measured repair/fallback costs with compacted dispatch.
// Indexed kernels consume a compacted index list (from flag compaction).
// Symmetric batch-size sweep: same batch sizes for repair and fallback.
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release && make bench_graded_repair_timing
// Run:   ./bench_graded_repair_timing --dataset PATH [--raw PATH] --csv output.csv

#include <cuda_runtime.h>
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "../experiment3/exp3_real_data_layout.hpp"
#include "filter_aggregate_kernels.cuh"

using namespace exp3_real;
namespace fs = std::filesystem;

constexpr uint64_t ALLOC_UNIT = 4096;
constexpr int ITERS = 500;

// ===========================================================================
// Indexed repair kernel: processes units listed in the compacted index array
// ===========================================================================
__global__ void repair_vote_indexed_kernel(
    const uint8_t *__restrict__ rep0,
    const uint8_t *__restrict__ rep1,
    const uint8_t *__restrict__ rep2,
    uint8_t *__restrict__ voted_out,
    const unsigned long long *__restrict__ indices,
    uint64_t n_indices,
    uint64_t plane_bytes)
{
    uint64_t idx = blockIdx.x;
    if (idx >= n_indices) return;
    uint64_t uid = indices[idx];
    uint64_t base = uid * ALLOC_UNIT;
    uint64_t tid = threadIdx.x;
    uint64_t stride = blockDim.x;
    for (uint64_t b = tid; b < ALLOC_UNIT && (base + b) < plane_bytes; b += stride)
    {
        uint64_t off = base + b;
        uint8_t a = rep0[off], b0 = rep1[off], c = rep2[off];
        voted_out[off] = (a == b0 || a == c) ? a : b0;
    }
}

// ===========================================================================
// Indexed fallback kernel: raw FP64 filter+aggregate for listed unit offsets
// Each unit = ALLOC_UNIT / sizeof(double) FP64 rows starting at row (uid * unit_rows)
// ===========================================================================
__global__ void fallback_indexed_kernel(
    const double *__restrict__ d_raw,
    const unsigned long long *__restrict__ indices,
    uint64_t n_indices,
    double threshold,
    uint64_t unit_rows,
    unsigned long long *__restrict__ d_counts,
    double *__restrict__ d_sums)
{
    uint64_t gi = blockIdx.x;
    if (gi >= n_indices) return;
    uint64_t uid = indices[gi];
    uint64_t row_start = uid * unit_rows;
    uint64_t tid = threadIdx.x;
    uint64_t stride = blockDim.x;
    unsigned long long count = 0;
    double sum = 0.0;
    for (uint64_t r = tid; r < unit_rows; r += stride)
    {
        double val = d_raw[row_start + r];
        if (val > threshold) { ++count; sum += val; }
    }
    // Block-level reduction
    __shared__ unsigned long long s_cnt[32];
    __shared__ double s_sum[32];
    for (int off = 16; off > 0; off >>= 1) {
        count += __shfl_down_sync(0xFFFFFFFF, count, off);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) { s_cnt[warp] = count; s_sum[warp] = sum; }
    __syncthreads();
    if (warp == 0) {
        unsigned long long wc = 0; double ws = 0.0;
        int n_warps = (blockDim.x + 31) / 32;
        for (int w = lane; w < n_warps; w += 32) { wc += s_cnt[w]; ws += s_sum[w]; }
        for (int off = 16; off > 0; off >>= 1) {
            wc += __shfl_down_sync(0xFFFFFFFF, wc, off);
            ws += __shfl_down_sync(0xFFFFFFFF, ws, off);
        }
        if (lane == 0) { d_counts[gi] = wc; d_sums[gi] = ws; }
    }
}

// ===========================================================================
// Flag-compaction kernel (conservative atomic approach)
// ===========================================================================
__global__ void compact_flags_kernel(
    const uint8_t *__restrict__ flags,
    unsigned long long *__restrict__ indices_out,
    unsigned long long *__restrict__ global_count,
    uint64_t n_units)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_units) return;
    if (flags[tid]) {
        unsigned long long pos = atomicAdd(global_count, 1ull);
        indices_out[pos] = tid;
    }
}

// ===========================================================================
// Helpers
// ===========================================================================
[[noreturn]] static void die(const char *m) { std::fprintf(stderr, "FATAL: %s\n", m); std::exit(2); }
static void cuda_check(cudaError_t e, const char *w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error at %s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}

// Time a kernel launch, return ms per unit. `process_units` function takes
// a batch_size and a base_offset and processes units [base, base+batch_size).
// The measurement covers different units each launch (not re-processing the prefix).
static double measure_per_unit(const char *label, int batch_units,
    std::function<void(int batch, uint64_t base_off, cudaStream_t)> process_units,
    uint64_t total_units, int iters, cudaStream_t stream)
{
    int n_launches = (static_cast<int>(total_units) + batch_units - 1) / batch_units;
    std::fprintf(stderr, "  %s batch=%d launches=%d:", label, batch_units, n_launches);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        uint64_t remaining = total_units;
        uint64_t off = 0;
        for (int l = 0; l < n_launches; ++l) {
            int this_batch = std::min(batch_units, static_cast<int>(remaining));
            process_units(this_batch, off, stream);
            remaining -= this_batch;
            off += this_batch;
        }
    }
    cuda_check(cudaStreamSynchronize(stream), "warmup sync");

    // Timed
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "evt");
    cuda_check(cudaEventCreate(&stop), "evt");
    cuda_check(cudaEventRecord(start, stream), "rec");

    for (int i = 0; i < iters; ++i) {
        uint64_t remaining = total_units;
        uint64_t off = 0;
        for (int l = 0; l < n_launches; ++l) {
            int this_batch = std::min(batch_units, static_cast<int>(remaining));
            process_units(this_batch, off, stream);
            remaining -= this_batch;
            off += this_batch;
        }
    }
    cuda_check(cudaEventRecord(stop, stream), "rec");
    cuda_check(cudaEventSynchronize(stop), "sync");

    float ms_total = 0;
    cuda_check(cudaEventElapsedTime(&ms_total, start, stop), "elapsed");
    cuda_check(cudaEventDestroy(start), "destroy");
    cuda_check(cudaEventDestroy(stop), "destroy");

    double ms_per_iter = ms_total / iters;
    double per_unit_ms = ms_per_iter / static_cast<double>(total_units);
    double per_unit_ns = per_unit_ms * 1e6;
    std::fprintf(stderr, " %.4fms total %.6fms/iter %.3fns/unit\n", ms_total, ms_per_iter, per_unit_ns);
    return per_unit_ms;
}

// Build a compacted index list from sparse flags (deterministic ~4% flagged)
static std::vector<unsigned long long> make_compacted_indices(uint64_t total_units) {
    std::vector<unsigned long long> idx;
    for (uint64_t u = 0; u < total_units; ++u)
        if ((u * 2654435761ull) % 100 < 4)
            idx.push_back(u);
    return idx;
}

int main(int argc, char **argv) {
    std::string dataset_path, raw_path, csv_path = "x3_timing_measured.csv";
    for (int i = 1; i < argc; ++i) {
        std::string_view a(argv[i]);
        auto val=[&](){if(++i>=argc)die("missing");return std::string(argv[i]);};
        if (a=="--help"||a=="-h") { std::fprintf(stderr,"Usage: %s --dataset PATH [--raw PATH] [--csv PATH]\n",argv[0]); return 0; }
        else if (a=="--dataset") dataset_path=val();
        else if (a=="--raw") raw_path=val();
        else if (a=="--csv") csv_path=val();
        else { std::string m="unknown: ";m+=a;die(m.c_str()); }
    }
    if (dataset_path.empty()) die("--dataset required");

    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::fprintf(stderr, "Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    Dataset dataset = exp3_real::load_dataset(dataset_path);
    uint64_t plane_bytes = dataset.planes.empty() ? 0 : dataset.planes[0].size();
    uint64_t n = dataset.manifest.value_count;
    uint64_t total_units = (plane_bytes + ALLOC_UNIT - 1) / ALLOC_UNIT;
    uint64_t unit_rows = std::min(ALLOC_UNIT / sizeof(double), n);

    std::fprintf(stderr, "Dataset: %s n=%" PRIu64 " plane_bytes=%" PRIu64 " total_units=%" PRIu64 " unit_rows=%" PRIu64 "\n",
        dataset.manifest.dataset.c_str(), n, plane_bytes, total_units, unit_rows);

    // Upload planes to GPU
    std::vector<uint8_t*> d_planes;
    for (auto &p : dataset.planes) {
        uint8_t *ptr = nullptr;
        cuda_check(cudaMalloc(&ptr, p.size()), "cudaMalloc plane");
        cuda_check(cudaMemcpy(ptr, p.data(), p.size(), cudaMemcpyHostToDevice), "cpy plane");
        d_planes.push_back(ptr);
    }

    // 3 replicas of plane 0 with tiny corruption in rep0
    uint8_t *d_rep0, *d_rep1, *d_rep2, *d_voted;
    cuda_check(cudaMalloc(&d_rep0, plane_bytes), "cudaMalloc rep0");
    cuda_check(cudaMalloc(&d_rep1, plane_bytes), "cudaMalloc rep1");
    cuda_check(cudaMalloc(&d_rep2, plane_bytes), "cudaMalloc rep2");
    cuda_check(cudaMalloc(&d_voted, plane_bytes), "cudaMalloc voted");
    cuda_check(cudaMemcpy(d_rep0, dataset.planes[0].data(), plane_bytes, cudaMemcpyHostToDevice), "cpy rep0");
    cuda_check(cudaMemcpy(d_rep1, dataset.planes[0].data(), plane_bytes, cudaMemcpyHostToDevice), "cpy rep1");
    cuda_check(cudaMemcpy(d_rep2, dataset.planes[0].data(), plane_bytes, cudaMemcpyHostToDevice), "cpy rep2");
    uint8_t flip = 0xFF;
    cuda_check(cudaMemcpy(d_rep0, &flip, 1, cudaMemcpyHostToDevice), "corrupt");

    // Upload raw data for fallback timing
    double *d_raw = nullptr;
    if (raw_path.empty() || !fs::exists(raw_path)) {
        std::vector<double> h_raw(n);
        for (uint64_t i = 0; i < n && i < plane_bytes; ++i) h_raw[i] = static_cast<double>(dataset.planes[0][i]);
        cuda_check(cudaMalloc(&d_raw, n * sizeof(double)), "cudaMalloc raw");
        cuda_check(cudaMemcpy(d_raw, h_raw.data(), n * sizeof(double), cudaMemcpyHostToDevice), "cpy raw");
    } else {
        uint64_t file_size = fs::file_size(raw_path);
        cuda_check(cudaMalloc(&d_raw, file_size), "cudaMalloc raw");
        std::ifstream fin(raw_path, std::ios::binary);
        constexpr size_t CHUNK = 256 * 1024 * 1024;
        std::vector<double> chunk(CHUNK / sizeof(double));
        uint64_t remaining = file_size, offset = 0;
        while (remaining > 0) {
            size_t to_read = std::min(remaining, CHUNK);
            fin.read(reinterpret_cast<char*>(chunk.data()), to_read);
            cuda_check(cudaMemcpy(reinterpret_cast<uint8_t*>(d_raw) + offset, chunk.data(), to_read, cudaMemcpyHostToDevice), "cpy raw");
            offset += to_read; remaining -= to_read;
        }
    }

    // Build compacted index list on host, upload to device
    auto comp_idx = make_compacted_indices(total_units);
    uint64_t n_compacted = comp_idx.size();
    unsigned long long *d_indices = nullptr;
    cuda_check(cudaMalloc(&d_indices, total_units * sizeof(unsigned long long)), "cudaMalloc indices");
    cuda_check(cudaMemcpy(d_indices, comp_idx.data(), n_compacted * sizeof(unsigned long long), cudaMemcpyHostToDevice), "cpy indices");

    // Batch sizes
    std::vector<int> batch_sizes;
    for (int b = 1; b <= 4096 && b <= static_cast<int>(total_units); b *= 4)
        batch_sizes.push_back(b);
    if (batch_sizes.empty() || batch_sizes.back() < static_cast<int>(total_units))
        batch_sizes.push_back(static_cast<int>(total_units));

    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "stream");

    // ===================================================================
    // Indexed REPAIR at multiple batch sizes
    // ===================================================================
    std::fprintf(stderr, "\n=== Indexed repair cost: batch-size sweep ===\n");
    std::vector<double> repair_per_unit(batch_sizes.size());

    for (size_t bi = 0; bi < batch_sizes.size(); ++bi) {
        int bs = batch_sizes[bi];
        auto proc = [&](int batch, uint64_t base_off, cudaStream_t s) {
            repair_vote_indexed_kernel<<<batch, 256, 0, s>>>(
                d_rep0, d_rep1, d_rep2, d_voted,
                d_indices + base_off, batch, plane_bytes);
        };
        repair_per_unit[bi] = measure_per_unit("repair_idx", bs, proc, n_compacted, ITERS, stream);
    }

    // ===================================================================
    // Indexed FALLBACK at multiple batch sizes (symmetric)
    // ===================================================================
    std::fprintf(stderr, "\n=== Indexed fallback cost: batch-size sweep ===\n");
    std::vector<double> fallback_per_unit(batch_sizes.size());

    // Pre-allocate per-index output buffers
    unsigned long long *d_fb_i_counts = nullptr;
    double *d_fb_i_sums = nullptr;
    cuda_check(cudaMalloc(&d_fb_i_counts, n_compacted * sizeof(unsigned long long)), "cudaMalloc fb_ic");
    cuda_check(cudaMalloc(&d_fb_i_sums, n_compacted * sizeof(double)), "cudaMalloc fb_is");

    for (size_t bi = 0; bi < batch_sizes.size(); ++bi) {
        int bs = batch_sizes[bi];
        auto proc = [&](int batch, uint64_t base_off, cudaStream_t s) {
            // Clear output buffers for this batch
            cuda_check(cudaMemsetAsync(d_fb_i_counts, 0, batch * sizeof(unsigned long long), s), "memset fb_ic");
            cuda_check(cudaMemsetAsync(d_fb_i_sums, 0, batch * sizeof(double), s), "memset fb_is");
            fallback_indexed_kernel<<<batch, 256, 0, s>>>(
                d_raw, d_indices + base_off, batch, 0.0, unit_rows, d_fb_i_counts, d_fb_i_sums);
        };
        fallback_per_unit[bi] = measure_per_unit("fallback_idx", bs, proc, n_compacted, ITERS, stream);
    }
    cuda_check(cudaFree(d_fb_i_counts), "free"); cuda_check(cudaFree(d_fb_i_sums), "free");

    // ===================================================================
    // Compaction timing
    // ===================================================================
    std::fprintf(stderr, "\n=== Compaction cost ===\n");
    uint8_t *d_flags = nullptr; unsigned long long *d_comout = nullptr, *d_comp_cnt = nullptr;
    cuda_check(cudaMalloc(&d_flags, total_units), "cudaMalloc flags");
    cuda_check(cudaMalloc(&d_comout, total_units * sizeof(unsigned long long)), "cudaMalloc comout");
    cuda_check(cudaMalloc(&d_comp_cnt, sizeof(unsigned long long)), "cudaMalloc compcnt");

    std::vector<uint8_t> h_flags(total_units, 0);
    for (auto u : comp_idx) h_flags[u] = 1;
    cuda_check(cudaMemcpy(d_flags, h_flags.data(), total_units, cudaMemcpyHostToDevice), "cpy flags");

    int comp_threads = 256;
    int comp_blocks = (static_cast<int>(total_units) + comp_threads - 1) / comp_threads;

    for (int i = 0; i < 5; ++i) {
        cuda_check(cudaMemsetAsync(d_comp_cnt, 0, sizeof(unsigned long long), stream), "memset cc");
        compact_flags_kernel<<<comp_blocks, comp_threads, 0, stream>>>(d_flags, d_comout, d_comp_cnt, total_units);
    }
    cuda_check(cudaStreamSynchronize(stream), "warmup sync");

    cudaEvent_t cs, ce;
    cuda_check(cudaEventCreate(&cs), "evt"); cuda_check(cudaEventCreate(&ce), "evt");
    cuda_check(cudaEventRecord(cs, stream), "rec");
    for (int i = 0; i < ITERS; ++i) {
        cuda_check(cudaMemsetAsync(d_comp_cnt, 0, sizeof(unsigned long long), stream), "memset cc");
        compact_flags_kernel<<<comp_blocks, comp_threads, 0, stream>>>(d_flags, d_comout, d_comp_cnt, total_units);
    }
    cuda_check(cudaEventRecord(ce, stream), "rec");
    cuda_check(cudaEventSynchronize(ce), "sync");
    float comp_ms = 0;
    cuda_check(cudaEventElapsedTime(&comp_ms, cs, ce), "elapsed");
    cuda_check(cudaEventDestroy(cs), "destroy"); cuda_check(cudaEventDestroy(ce), "destroy");
    comp_ms /= ITERS;

    std::fprintf(stderr, "  compaction: %f ms per query (%" PRIu64 " units, %" PRIu64 " flagged)\n", comp_ms, total_units, n_compacted);

    // ===================================================================
    // Cleanup
    // ===================================================================
    cuda_check(cudaFree(d_flags), "free"); cuda_check(cudaFree(d_comout), "free");
    cuda_check(cudaFree(d_comp_cnt), "free"); cuda_check(cudaFree(d_indices), "free");
    cuda_check(cudaFree(d_raw), "free raw");
    for (auto p : d_planes) cuda_check(cudaFree(p), "free plane");
    cuda_check(cudaFree(d_rep0), "free"); cuda_check(cudaFree(d_rep1), "free");
    cuda_check(cudaFree(d_rep2), "free"); cuda_check(cudaFree(d_voted), "free");
    cuda_check(cudaStreamDestroy(stream), "destroy stream");

    // ===================================================================
    // Validate
    // ===================================================================
    if (repair_per_unit.empty() || fallback_per_unit.empty())
        die("no timing rows generated");
    if (repair_per_unit.size() != batch_sizes.size())
        die("repair timing rows mismatch batch sizes");
    if (comp_ms <= 0)
        die("compaction not measured (comp_ms <= 0)");
    for (auto v : repair_per_unit)
        if (v <= 0) die("repair timing contains zero or negative value");
    for (auto v : fallback_per_unit)
        if (v <= 0) die("fallback timing contains zero or negative value");
    if (n_compacted == 0)
        die("compacted index count is zero despite non-zero flags requested");

    // ===================================================================
    // Write machine-readable CSV with provenance metadata
    // ===================================================================
    const char *job_id = std::getenv("SLURM_JOB_ID");
    if (!job_id) job_id = "NO_JOB_ID";

    // Try to get git commit SHA from environment or a known header
    // (can't easily get it at runtime, so embed from build time or env)
    const char *git_commit = std::getenv("GIT_COMMIT");
    if (!git_commit) git_commit = "UNKNOWN";

    // GPU name from device properties
    char gpu_name[128] = {0};
    std::snprintf(gpu_name, sizeof(gpu_name), "%s sm_%d%d", prop.name, prop.major, prop.minor);

    std::FILE *fcsv = std::fopen(csv_path.c_str(), "w");
    if (!fcsv) die("cannot write CSV");

    // Metadata header rows (prefixed with #)
    std::fprintf(fcsv, "# timing_model=indexed_compacted_list_batched\n");
    std::fprintf(fcsv, "# gpu_name=%s\n", gpu_name);
    std::fprintf(fcsv, "# job_id=%s\n", job_id);
    std::fprintf(fcsv, "# iters=%d\n", ITERS);

    // Column header and data
    std::fprintf(fcsv, "dataset,total_units,n_compacted,batch_size,"
                       "idx_repair_ms_per_unit,idx_fallback_ms_per_unit\n");
    for (size_t bi = 0; bi < batch_sizes.size(); ++bi) {
        std::fprintf(fcsv, "%s,%" PRIu64 ",%" PRIu64 ",%d,%.12f,%.12f\n",
            dataset.manifest.dataset.c_str(), total_units, n_compacted, batch_sizes[bi],
            repair_per_unit[bi], fallback_per_unit[bi]);
    }
    // Compaction row
    std::fprintf(fcsv, "%s,%" PRIu64 ",%" PRIu64 ",compaction,%.12f,%.12f\n",
        dataset.manifest.dataset.c_str(), total_units, n_compacted, comp_ms, comp_ms);
    std::fclose(fcsv);
    std::fprintf(stderr, "\nCSV: %s\n", csv_path.c_str());
    return 0;
}
