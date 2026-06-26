// Phase 3-Z0: Bandwidth De-risk Microbenchmark
// Measures 4 read paths on plane 0 at shallow k to decide whether
// "trade idle GPU compute for reliability" is real.
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release && make bench_z0_bandwidth_derisk
// Run:   ./bench_z0_bandwidth_derisk --dataset PATH [--csv output.csv]
//
// Paths:
//   A: single-replica baseline          read 1, no CRC, no vote
//   B: read-1 + CRC, no-fault common    read 1 + CRC, no mismatch
//   C: read-1 + CRC, detected + repair  read 1 + CRC, fetch replicas + vote on mismatch
//   D: eager read-N + vote              always read N, warp vote

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
#include <chrono>

#include "../experiment3/exp3_real_data_layout.hpp"

using namespace exp3_real;
namespace fs = std::filesystem;

constexpr uint64_t ALLOC_UNIT = 4096;
constexpr int ITERS = 100;

static void die(const char *m) { std::fprintf(stderr, "FATAL: %s\n", m); std::exit(2); }
static void cuda_check(cudaError_t e, const char *w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error at %s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}

// GPU CRC32 — software implementation matching zlib CRC-32 (IEEE polynomial)
// Uses 8-bit table-driven approach (no special intrinsics required)
__constant__ uint32_t crc32_table[256] = {
    0x00000000,0x77073096,0xEE0E612C,0x990951BA,0x076DC419,0x706AF48F,
    0xE963A535,0x9E6495A3,0x0EDB8832,0x79DCB8A4,0xE0D5E91E,0x97D2D988,
    0x09B64C2B,0x7EB17CBD,0xE7B82D07,0x90BF1D91,0x1DB71064,0x6AB020F2,
    0xF3B97148,0x84BE41DE,0x1ADAD47D,0x6DDDE4EB,0xF4D4B551,0x83D385C7,
    0x136C9856,0x646BA8C0,0xFD62F97A,0x8A65C9EC,0x14015C4F,0x63066CD9,
    0xFA0F3D63,0x8D080DF5,0x3B6E20C8,0x4C69105E,0xD56041E4,0xA2677172,
    0x3C03E4D1,0x4B04D447,0xD20D85FD,0xA50AB56B,0x35B5A8FA,0x42B2986C,
    0xDBBBC9D6,0xACBCF940,0x32D86CE3,0x45DF5C75,0xDCD60DCF,0xABD13D59,
    0x26D930AC,0x51DE003A,0xC8D75180,0xBFD06116,0x21B4F4B5,0x56B3C423,
    0xCFBA9599,0xB8BDA50F,0x2802B89E,0x5F058808,0xC60CD9B2,0xB10BE924,
    0x2F6F7C87,0x58684C11,0xC1611DAB,0xB6662D3D,0x76DC4190,0x01DB7106,
    0x98D220BC,0xEFD5102A,0x71B18589,0x06B6B51F,0x9FBFE4A5,0xE8B8D433,
    0x7807C9A2,0x0F00F934,0x9609A88E,0xE10E9818,0x7F6A0DBB,0x086D3D2D,
    0x91646C97,0xE6635C01,0x6B6B51F4,0x1C6C6162,0x856530D8,0xF262004E,
    0x6C0695ED,0x1B01A57B,0x8208F4C1,0xF50FC457,0x65B0D9C6,0x12B7E950,
    0x8BBEB8EA,0xFCB9887C,0x62DD1DDF,0x15DA2D49,0x8CD37CF3,0xFBD44C65,
    0x4DB26158,0x3AB551CE,0xA3BC0074,0xD4BB30E2,0x4ADFA541,0x3DD895D7,
    0xA4D1C46D,0xD3D6F4FB,0x4369E96A,0x346ED9FC,0xAD678846,0xDA60B8D0,
    0x44042D73,0x33031DE5,0xAA0A4C5F,0xDD0D7CC9,0x5005713C,0x270241AA,
    0xBE0B1010,0xC90C2086,0x5768B525,0x206F85B3,0xB966D409,0xCE61E49F,
    0x5EDEF90E,0x29D9C998,0xB0D09822,0xC7D7A8B4,0x59B33D17,0x2EB40D81,
    0xB7BD5C3B,0xC0BA6CAD,0xEDB88320,0x9ABFB3B6,0x03B6E20C,0x74B1D29A,
    0xEAD54739,0x9DD277AF,0x04DB2615,0x73DC1683,0xE3630B12,0x94643B84,
    0x0D6D6A3E,0x7A6A5AA8,0xE40ECF0B,0x9309FF9D,0x0A00AE27,0x7D079EB1,
    0xF00F9344,0x8708A3D2,0x1E01F268,0x6906C2FE,0xF762575D,0x806567CB,
    0x196C3671,0x6E6B06E7,0xFED41B76,0x89D32BE0,0x10DA7A5A,0x67DD4ACC,
    0xF9B9DF6F,0x8EBEEFF9,0x17B7BE43,0x60B08ED5,0xD6D6A3E8,0xA1D1937E,
    0x38D8C2C4,0x4FDFF252,0xD1BB67F1,0xA6BC5767,0x3FB506DD,0x48B2364B,
    0xD80D2BDA,0xAF0A1B4C,0x36034AF6,0x41047A60,0xDF60EFC3,0xA867DF55,
    0x316E8EEF,0x4669BE79,0xCB61B38C,0xBC66831A,0x256FD2A0,0x5268E236,
    0xCC0C7795,0xBB0B4703,0x220216B9,0x5505262F,0xC5BA3BBE,0xB2BD0B28,
    0x2BB45A92,0x5CB36A04,0xC2D7FFA7,0xB5D0CF31,0x2CD99E8B,0x5BDEAE1D,
    0x9B64C2B0,0xEC63F226,0x756AA39C,0x026D930A,0x9C0906A9,0xEB0E363F,
    0x72076785,0x05005713,0x95BF4A82,0xE2B87A14,0x7BB12BAE,0x0CB61B38,
    0x92D28E9B,0xE5D5BE0D,0x7CDCEFB7,0x0BDBDF21,0x86D3D2D4,0xF1D4E242,
    0x68DDB3F8,0x1FDA836E,0x81BE16CD,0xF6B9265B,0x6FB077E1,0x18B74777,
    0x88085AE6,0xFF0F6A70,0x66063BCA,0x11010B5C,0x8F659EFF,0xF862AE69,
    0x616BFFD3,0x166CCF45,0xA00AE278,0xD70DD2EE,0x4E048354,0x3903B3C2,
    0xA7672661,0xD06016F7,0x4969474D,0x3E6E77DB,0xAED16A4A,0xD9D65ADC,
    0x40DF0B66,0x37D83BF0,0xA9BCAE53,0xDEBB9EC5,0x47B2CF7F,0x30B5FFE9,
    0xBDBDF21C,0xCABAC28A,0x53B39330,0x24B4A3A6,0xBAD03605,0xCDD70693,
    0x54DE5729,0x23D967BF,0xB3667A2E,0xC4614AB8,0x5D681B02,0x2A6F2B94,
    0xB40BBE37,0xC30C8EA1,0x5A05DF1B,0x2D02EF8D,
};

__device__ __forceinline__ uint32_t crc32_hw(const uint8_t *data, uint64_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (uint64_t i = 0; i < len; i++) {
        crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

// Path A: Single-replica baseline — just read plane 0 bytes
__global__ void path_a_read_baseline(
    const uint8_t *__restrict__ src,
    uint8_t *__restrict__ dst,
    uint64_t n_bytes)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_bytes) return;
    dst[tid] = src[tid];
}

// Path B: Read-1 + CRC, no-fault common path
// Uses PTX CRC32 instruction for hardware-accelerated CRC
__global__ void path_b_read_crc_common(
    const uint8_t *__restrict__ src,
    uint64_t n_units,
    uint64_t unit_bytes,
    uint32_t *__restrict__ crc_out)
{
    uint64_t uid = blockIdx.x;
    if (uid >= n_units) return;
    uint64_t base = uid * unit_bytes;
    crc_out[uid] = crc32_hw(src + base, unit_bytes);
}

// Path C: Read-1 + CRC, detected + fetch replicas + vote
__global__ void path_c_read_crc_repair(
    const uint8_t *__restrict__ rep0,
    const uint8_t *__restrict__ rep1,
    const uint8_t *__restrict__ rep2,
    uint8_t *__restrict__ voted_out,
    uint64_t n_units,
    uint64_t unit_bytes,
    const uint8_t *__restrict__ fault_flags,
    uint32_t *__restrict__ crc_out)
{
    uint64_t uid = blockIdx.x;
    if (uid >= n_units) return;
    uint64_t base = uid * unit_bytes;

    crc_out[uid] = crc32_hw(rep0 + base, unit_bytes);

    // If fault flag set or CRC mismatch (simulated), fetch replicas + vote
    if (fault_flags[uid]) {
        uint64_t tid = threadIdx.x;
        uint64_t stride = blockDim.x;
        for (uint64_t b = tid; b < unit_bytes && (base + b) < unit_bytes * n_units; b += stride) {
            uint64_t off = base + b;
            uint8_t a = rep0[off], b0 = rep1[off], c = rep2[off];
            voted_out[off] = (a == b0 || a == c) ? a : b0;
        }
    }
}

// Path D: Eager read-N + warp vote (always read all 3 replicas)
__global__ void path_d_eager_read_vote(
    const uint8_t *__restrict__ rep0,
    const uint8_t *__restrict__ rep1,
    const uint8_t *__restrict__ rep2,
    uint8_t *__restrict__ voted_out,
    uint64_t n_bytes)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_bytes) return;
    uint8_t a = rep0[tid], b0 = rep1[tid], c = rep2[tid];
    voted_out[tid] = (a == b0 || a == c) ? a : b0;
}

// ===========================================================================
// Timing helpers
// ===========================================================================

struct TimingResult {
    double ms_total = 0;
    double ms_per_iter = 0;
};

static TimingResult measure_time(
    const char *label,
    std::function<void(cudaStream_t)> launch_fn,
    int iters, cudaStream_t stream)
{
    // Warmup (2 iters for smoke)
    for (int i = 0; i < 2; i++) launch_fn(stream);
    cuda_check(cudaStreamSynchronize(stream), "warmup sync");

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "evt create");
    cuda_check(cudaEventCreate(&stop), "evt create");
    cuda_check(cudaEventRecord(start, stream), "evt rec");

    for (int i = 0; i < iters; i++) launch_fn(stream);

    cuda_check(cudaEventRecord(stop, stream), "evt rec");
    cuda_check(cudaEventSynchronize(stop), "evt sync");

    float ms = 0;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "evt elapsed");
    cuda_check(cudaEventDestroy(start), "evt destroy");
    cuda_check(cudaEventDestroy(stop), "evt destroy");

    TimingResult r;
    r.ms_total = ms;
    r.ms_per_iter = ms / iters;
    std::fprintf(stderr, "  %-40s %8.4f ms total  %8.6f ms/iter\n", label, ms, r.ms_per_iter);
    return r;
}

// ===========================================================================
// Main
// ===========================================================================

int main(int argc, char **argv) {
    std::string dataset_path, csv_path = "z0_bandwidth_profile.csv";
    uint64_t forced_k = 0;
    double forced_fault_rate = -1.0; // -1 means sweep all

    for (int i = 1; i < argc; ++i) {
        std::string_view a(argv[i]);
        auto val=[&](){if(++i>=argc)die("missing value");return std::string(argv[i]);};
        if (a=="--help"||a=="-h") {
            std::fprintf(stderr,"Usage: %s --dataset PATH [--csv PATH] [--k N] [--fault-rate R]\n",argv[0]);
            return 0;
        } else if (a=="--dataset") dataset_path=val();
        else if (a=="--csv") csv_path=val();
        else if (a=="--k") forced_k=std::stoul(val());
        else if (a=="--fault-rate") forced_fault_rate=std::stod(val());
        else { std::string m="unknown: ";m+=a;die(m.c_str()); }
    }
    if (dataset_path.empty()) die("--dataset required");

    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::fprintf(stderr, "Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::fprintf(stderr, "Global memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    std::fprintf(stderr, "Memory clock: %.0f MHz, Bus width: %d bits\n",
        prop.memoryClockRate / 1e3, prop.memoryBusWidth);

    Dataset dataset = exp3_real::load_dataset(dataset_path);
    uint64_t plane_bytes = dataset.planes.empty() ? 0 : dataset.planes[0].size();
    uint64_t n = dataset.manifest.value_count;
    uint64_t total_units = (plane_bytes + ALLOC_UNIT - 1) / ALLOC_UNIT;

    std::fprintf(stderr, "Dataset: %s n=%" PRIu64 " plane_bytes=%" PRIu64 " total_units=%" PRIu64 "\n",
        dataset.manifest.dataset.c_str(), n, plane_bytes, total_units);

    // K values to sweep
    std::vector<uint64_t> k_values;
    if (forced_k > 0) {
        k_values.push_back(forced_k);
    } else {
        k_values = {1, 2, dataset.manifest.max_plane_count};
    }

    // Upload planes to GPU
    std::vector<uint8_t*> d_planes;
    for (auto &p : dataset.planes) {
        uint8_t *ptr = nullptr;
        cuda_check(cudaMalloc(&ptr, p.size()), "cudaMalloc plane");
        cuda_check(cudaMemcpy(ptr, p.data(), p.size(), cudaMemcpyHostToDevice), "cpy plane");
        d_planes.push_back(ptr);
    }

    // 3 replicas for plane 0
    uint8_t *d_rep0, *d_rep1, *d_rep2, *d_voted;
    cuda_check(cudaMalloc(&d_rep0, plane_bytes), "cudaMalloc rep0");
    cuda_check(cudaMalloc(&d_rep1, plane_bytes), "cudaMalloc rep1");
    cuda_check(cudaMalloc(&d_rep2, plane_bytes), "cudaMalloc rep2");
    cuda_check(cudaMalloc(&d_voted, plane_bytes), "cudaMalloc voted");
    cuda_check(cudaMemcpy(d_rep0, dataset.planes[0].data(), plane_bytes, cudaMemcpyHostToDevice), "cpy rep0");
    cuda_check(cudaMemcpy(d_rep1, dataset.planes[0].data(), plane_bytes, cudaMemcpyHostToDevice), "cpy rep1");
    cuda_check(cudaMemcpy(d_rep2, dataset.planes[0].data(), plane_bytes, cudaMemcpyHostToDevice), "cpy rep2");

    // Fault flags for path C (simulate p_fault fraction of units)
    uint8_t *d_fault_flags = nullptr;
    cuda_check(cudaMalloc(&d_fault_flags, total_units), "cudaMalloc fault_flags");
    cuda_check(cudaMemset(d_fault_flags, 0, total_units), "memset fault_flags");

    // CRC output buffers
    uint32_t *d_crc_out = nullptr;
    cuda_check(cudaMalloc(&d_crc_out, total_units * sizeof(uint32_t)), "cudaMalloc crc_out");

    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "stream");

    // ===================================================================
    // Sweep k values
    // ===================================================================
    std::FILE *fcsv = std::fopen(csv_path.c_str(), "w");
    if (!fcsv) die("cannot write CSV");

    // Metadata
    const char *job_id = std::getenv("SLURM_JOB_ID");
    if (!job_id) job_id = "NO_JOB_ID";
    char gpu_name[128] = {0};
    std::snprintf(gpu_name, sizeof(gpu_name), "%s sm_%d%d", prop.name, prop.major, prop.minor);
    std::fprintf(fcsv, "# z0_bandwidth_derisk\n");
    std::fprintf(fcsv, "# gpu_name=%s\n", gpu_name);
    std::fprintf(fcsv, "# job_id=%s\n", job_id);
    std::fprintf(fcsv, "# iters=%d\n", ITERS);
    std::fprintf(fcsv, "# dataset=%s\n", dataset.manifest.dataset.c_str());
    std::fprintf(fcsv, "# n_rows=%" PRIu64 "\n", n);
    std::fprintf(fcsv, "# plane_bytes=%" PRIu64 "\n", plane_bytes);
    std::fprintf(fcsv, "# total_units=%" PRIu64 "\n", total_units);
    std::fprintf(fcsv, "# alloc_unit=%" PRIu64 "\n", ALLOC_UNIT);

    std::fprintf(fcsv, "k,path,label,ms_per_iter,ms_per_unit,ms_per_byte\n");

    int threads = 256;

    for (uint64_t ki = 0; ki < k_values.size(); ki++) {
        uint64_t k = k_values[ki];
        uint64_t bytes_to_process = (k <= dataset.planes.size()) ? dataset.planes[0].size() : plane_bytes;
        // For shallow k, we only read planes 0..k-1, but we benchmark plane 0 specifically
        uint64_t bytes = dataset.planes[0].size();
        (void)bytes_to_process;

        std::fprintf(stderr, "\n=== k=%" PRIu64 " ===\n", k);

        // Path A: single-replica baseline
        {
            int blocks = (static_cast<int>(bytes) + threads - 1) / threads;
            auto launch = [&](cudaStream_t s) {
                path_a_read_baseline<<<blocks, threads, 0, s>>>(d_rep0, d_voted, bytes);
            };
            auto tr = measure_time("A: single-replica baseline", launch, ITERS, stream);
            std::fprintf(fcsv, "%" PRIu64 ",A,single_replica_baseline,%.12f,%.12f,%.12f\n",
                k, tr.ms_per_iter, tr.ms_per_iter / total_units, tr.ms_per_iter / bytes);
        }

        // Path B: read-1 + CRC, no-fault common
        {
            auto launch = [&](cudaStream_t s) {
                int blocks_b = static_cast<int>(total_units);
                path_b_read_crc_common<<<blocks_b, 1, 0, s>>>(d_rep0, total_units, ALLOC_UNIT, d_crc_out);
            };
            auto tr = measure_time("B: read-1 + CRC (no-fault common)", launch, ITERS, stream);
            std::fprintf(fcsv, "%" PRIu64 ",B,read1_crc_common,%.12f,%.12f,%.12f\n",
                k, tr.ms_per_iter, tr.ms_per_iter / total_units, tr.ms_per_iter / bytes);
        }

        // Path C: read-1 + CRC, detected + repair (at various fault rates)
        std::vector<double> fault_rates;
        if (forced_fault_rate >= 0) {
            fault_rates.push_back(forced_fault_rate);
        } else {
            fault_rates = {0.0, 1e-7, 1e-6, 1e-5};
        }

        for (double fr : fault_rates) {
            uint64_t n_faulted = static_cast<uint64_t>(total_units * fr);
            if (fr > 0 && n_faulted == 0) n_faulted = 1;

            // Set fault flags
            std::vector<uint8_t> h_fault_flags(total_units, 0);
            for (uint64_t u = 0; u < n_faulted && u < total_units; u++) {
                h_fault_flags[(u * 2654435761ull) % total_units] = 1;
            }
            cuda_check(cudaMemcpy(d_fault_flags, h_fault_flags.data(), total_units, cudaMemcpyHostToDevice), "cpy fault_flags");

            auto launch_c = [&](cudaStream_t s) {
                int blocks_c = static_cast<int>(total_units);
                path_c_read_crc_repair<<<blocks_c, 256, 0, s>>>(
                    d_rep0, d_rep1, d_rep2, d_voted,
                    total_units, ALLOC_UNIT, d_fault_flags, d_crc_out);
            };
            char label[80];
            std::snprintf(label, sizeof(label), "C: read1+CRC+repair (fr=%.0e)", fr);
            auto tr = measure_time(label, launch_c, ITERS, stream);
            std::fprintf(fcsv, "%" PRIu64 ",C,read1_crc_repair_fr%.0e,%.12f,%.12f,%.12f\n",
                k, fr, tr.ms_per_iter, tr.ms_per_iter / total_units, tr.ms_per_iter / bytes);
        }

        // Path D: eager read-N + vote (always 3 replicas, warp vote)
        {
            int blocks_d = (static_cast<int>(bytes) + threads - 1) / threads;
            auto launch_d = [&](cudaStream_t s) {
                path_d_eager_read_vote<<<blocks_d, threads, 0, s>>>(
                    d_rep0, d_rep1, d_rep2, d_voted, bytes);
            };
            auto tr = measure_time("D: eager read-N + warp vote", launch_d, ITERS, stream);
            std::fprintf(fcsv, "%" PRIu64 ",D,eager_readN_vote,%.12f,%.12f,%.12f\n",
                k, tr.ms_per_iter, tr.ms_per_iter / total_units, tr.ms_per_iter / bytes);
        }
    }

    // ===================================================================
    // Derived metrics row
    // ===================================================================
    std::fprintf(fcsv, "\n# Derived metrics (computed post-hoc):\n");
    std::fprintf(fcsv, "# crc_only_overhead = (path_B - path_A) / path_A\n");
    std::fprintf(fcsv, "# nmr_latency_inflation_lazy = amortized_lazy / path_A\n");
    std::fprintf(fcsv, "#   where amortized_lazy = (1-p_fault)*path_B + p_fault*path_C\n");
    std::fprintf(fcsv, "# nmr_latency_inflation_eager = path_D / path_A\n");

    fclose(fcsv);
    std::fprintf(stderr, "\nCSV: %s\n", csv_path.c_str());

    // Cleanup
    cuda_check(cudaFree(d_fault_flags), "free");
    cuda_check(cudaFree(d_crc_out), "free");
    cuda_check(cudaFree(d_voted), "free");
    cuda_check(cudaFree(d_rep2), "free");
    cuda_check(cudaFree(d_rep1), "free");
    cuda_check(cudaFree(d_rep0), "free");
    for (auto p : d_planes) cuda_check(cudaFree(p), "free plane");
    cuda_check(cudaStreamDestroy(stream), "destroy stream");

    return 0;
}
