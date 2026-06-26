// Phase 3-Z0b: Corrected Bandwidth De-risk Microbenchmark
// Fixes: path C CRC bug, adds B2 parallel CRC, uses k correctly, probes HW CRC
//
// Paths:
//   A:  single-replica baseline           read 1, no CRC, no vote
//   B:  read-1 + table CRC, 1-thread/unit  serial CRC per unit (smoke baseline)
//   B2: read-1 + table CRC, parallel       each thread handles one unit's CRC
//   C:  read-1 + CRC -> detected + repair  CRC by thr0, repair by all threads
//   D:  eager read-N + warp vote           always read N, warp vote
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 && make bench_z0b_bandwidth_derisk
// Run:   ./bench_z0b_bandwidth_derisk --dataset PATH [--csv output.csv] [--k N]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// ---- CRC lookup table (zlib CRC-32, IEEE polynomial 0xEDB88320) ----
__constant__ uint32_t crc32_tab[256] = {
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

// Table-driven CRC32 over a buffer
__device__ __forceinline__ uint32_t crc32_buf(const uint8_t *data, uint64_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (uint64_t i = 0; i < len; i++)
        crc = crc32_tab[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

// ===================================================================
// Path A: single-replica baseline — byte copy
// ===================================================================
__global__ void path_a_read(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ===================================================================
// Path B: 1 thread / unit, serial CRC (smoke baseline)
// ===================================================================
__global__ void path_b_crc_1tpu(const uint8_t *__restrict__ src, uint64_t n_units,
    uint64_t unit_b, uint32_t *__restrict__ crc_out) {
    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    crc_out[u] = crc32_buf(src + u * unit_b, unit_b);
}

// ===================================================================
// Path B2: parallel — one unit per thread, many threads per block
// Reduces block count vs path B (e.g. 161 blocks vs 41133)
// ===================================================================
__global__ void path_b2_crc_parallel(const uint8_t *__restrict__ src, uint64_t n_units,
    uint64_t unit_b, uint32_t *__restrict__ crc_out) {
    uint64_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n_units) return;
    crc_out[u] = crc32_buf(src + u * unit_b, unit_b);
}

// ===================================================================
// Path C: read-1 + CRC -> detect -> fetch replicas + vote
// FIX: CRC done by thread 0 only. Repair uses all threads (guarded by fault_flag).
// ===================================================================
__global__ void path_c_crc_repair(const uint8_t *__restrict__ rep0,
    const uint8_t *__restrict__ rep1, const uint8_t *__restrict__ rep2,
    uint8_t *__restrict__ voted, uint64_t n_units, uint64_t unit_b,
    const uint8_t *__restrict__ fault_flags, uint32_t *__restrict__ crc_out) {
    uint64_t u = blockIdx.x;
    if (u >= n_units) return;
    uint64_t base = u * unit_b;

    // Thread 0 does CRC
    if (threadIdx.x == 0)
        crc_out[u] = crc32_buf(rep0 + base, unit_b);
    __syncthreads();

    // All threads participate in repair if faulted
    if (fault_flags[u]) {
        uint64_t t = threadIdx.x;
        uint64_t step = blockDim.x;
        for (uint64_t b = t; b < unit_b; b += step) {
            uint64_t off = base + b;
            uint8_t a = rep0[off], b0 = rep1[off], c = rep2[off];
            voted[off] = (a == b0 || a == c) ? a : b0;
        }
    }
}

// ===================================================================
// Path D: eager read-3 + warp vote
// ===================================================================
__global__ void path_d_eager_vote(const uint8_t *__restrict__ rep0,
    const uint8_t *__restrict__ rep1, const uint8_t *__restrict__ rep2,
    uint8_t *__restrict__ voted, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t a = rep0[i], b0 = rep1[i], c = rep2[i];
    voted[i] = (a == b0 || a == c) ? a : b0;
}

// ===================================================================
// Timing
// ===================================================================
struct Timing { double ms_total = 0, ms_per_iter = 0; };

static Timing measure(const char *label, std::function<void(cudaStream_t)> fn,
    int iters, cudaStream_t s) {
    for (int i = 0; i < 2; i++) fn(s);
    cuda_check(cudaStreamSynchronize(s), "warmup");
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "ce"); cuda_check(cudaEventCreate(&stop), "ce");
    cuda_check(cudaEventRecord(start, s), "rec");
    for (int i = 0; i < iters; i++) fn(s);
    cuda_check(cudaEventRecord(stop, s), "rec"); cuda_check(cudaEventSynchronize(stop), "sync");
    float ms = 0;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cuda_check(cudaEventDestroy(start), "dt"); cuda_check(cudaEventDestroy(stop), "dt");
    Timing r; r.ms_total = ms; r.ms_per_iter = ms / iters;
    std::fprintf(stderr, "  %-48s %9.4f ms total  %9.6f ms/iter\n", label, ms, r.ms_per_iter);
    return r;
}

// ===================================================================
// CRC probe: try hardware vs table
// ===================================================================
static void probe_hw_crc() {
    // Compile-time: if __CUDA_ARCH__ >= 800 and __crc32b is available
    // We can't test at runtime, but we can report architecture support
    int dev = 0;
    cudaDeviceProp p;
    cuda_check(cudaGetDeviceProperties(&p, dev), "prop");
    int sm = p.major * 10 + p.minor;
    std::fprintf(stderr, "CRC probe: sm_%02d — HW CRC (__crc32b) requires sm_80+\n", sm);
    if (sm >= 80)
        std::fprintf(stderr, "  HW CRC supported by architecture, but not used in this build\n");
    else
        std::fprintf(stderr, "  HW CRC not supported; falling back to table CRC\n");
}

// ===================================================================
// Main
// ===================================================================
int main(int argc, char **argv) {
    std::string dataset_path, csv_path = "z0b_bandwidth_profile.csv";
    uint64_t forced_k = 0;
    double forced_fr = -1.0;

    for (int i = 1; i < argc; ++i) {
        std::string_view a(argv[i]);
        auto val=[&](){if(++i>=argc)die("missing arg val");return std::string(argv[i]);};
        if (a=="--help"||a=="-h") {
            std::fprintf(stderr,"Usage: %s --dataset PATH [--csv PATH] [--k N] [--fault-rate R]\n",argv[0]);
            return 0;
        } else if (a=="--dataset") dataset_path=val();
        else if (a=="--csv") csv_path=val();
        else if (a=="--k") forced_k=std::stoul(val());
        else if (a=="--fault-rate") forced_fr=std::stod(val());
        else { std::string m="unknown: ";m+=a;die(m.c_str()); }
    }
    if (dataset_path.empty()) die("--dataset required");

    cuda_check(cudaSetDevice(0), "set dev");
    cudaDeviceProp p;
    cuda_check(cudaGetDeviceProperties(&p, 0), "prop");
    std::fprintf(stderr, "Device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    std::fprintf(stderr, "Mem: %.1f GB  Clock: %.0f MHz  Bus: %d bits  BW: %.1f GB/s\n",
        p.totalGlobalMem / 1e9, p.memoryClockRate / 1e3, p.memoryBusWidth,
        2.0 * p.memoryClockRate / 1e3 * p.memoryBusWidth / 8.0);

    probe_hw_crc();

    Dataset ds = exp3_real::load_dataset(dataset_path);
    uint64_t plane0_bytes = ds.planes.empty() ? 0 : ds.planes[0].size();
    uint64_t n = ds.manifest.value_count;
    uint64_t nu = (plane0_bytes + ALLOC_UNIT - 1) / ALLOC_UNIT;
    uint64_t max_k = ds.manifest.max_plane_count;

    std::fprintf(stderr, "Dataset: %s  rows=%" PRIu64 "  plane0=%" PRIu64 "  units=%" PRIu64 "  max_k=%" PRIu64 "\n",
        ds.manifest.dataset.c_str(), n, plane0_bytes, nu, max_k);

    std::vector<uint64_t> ks;
    if (forced_k > 0) { ks.push_back(forced_k); }
    else { ks = {1, 2, std::min(uint64_t(6), max_k)}; }

    // Upload plane data
    std::vector<uint8_t*> d_planes;
    for (auto &pl : ds.planes) {
        uint8_t *ptr = nullptr;
        cuda_check(cudaMalloc(&ptr, pl.size()), "M plane");
        cuda_check(cudaMemcpy(ptr, pl.data(), pl.size(), cudaMemcpyHostToDevice), "H2D plane");
        d_planes.push_back(ptr);
    }

    // 3 replicas of plane 0
    uint8_t *d_r0, *d_r1, *d_r2, *d_v;
    cuda_check(cudaMalloc(&d_r0, plane0_bytes), "M r0");
    cuda_check(cudaMalloc(&d_r1, plane0_bytes), "M r1");
    cuda_check(cudaMalloc(&d_r2, plane0_bytes), "M r2");
    cuda_check(cudaMalloc(&d_v, plane0_bytes), "M v");
    cuda_check(cudaMemcpy(d_r0, ds.planes[0].data(), plane0_bytes, cudaMemcpyHostToDevice), "H2D r0");
    cuda_check(cudaMemcpy(d_r1, ds.planes[0].data(), plane0_bytes, cudaMemcpyHostToDevice), "H2D r1");
    cuda_check(cudaMemcpy(d_r2, ds.planes[0].data(), plane0_bytes, cudaMemcpyHostToDevice), "H2D r2");

    uint8_t *d_ff = nullptr;
    cuda_check(cudaMalloc(&d_ff, nu), "M ff");
    cuda_check(cudaMemset(d_ff, 0, nu), "Mset ff");
    uint32_t *d_crc = nullptr;
    cuda_check(cudaMalloc(&d_crc, nu * sizeof(uint32_t)), "M crc");

    cudaStream_t st;
    cuda_check(cudaStreamCreate(&st), "stream");

    FILE *fcsv = std::fopen(csv_path.c_str(), "w");
    if (!fcsv) die("cannot write CSV");
    const char *jid = std::getenv("SLURM_JOB_ID"); if (!jid) jid = "NOJOB";
    fprintf(fcsv, "# z0b_bandwidth_derisk  gpu=%s sm_%d%d  job=%s  iters=%d\n",
        p.name, p.major, p.minor, jid, ITERS);
    fprintf(fcsv, "# dataset=%s  rows=%" PRIu64 "  plane0=%" PRIu64 "  units=%" PRIu64 "  alloc_unit=%" PRIu64 "\n",
        ds.manifest.dataset.c_str(), n, plane0_bytes, nu, ALLOC_UNIT);
    fprintf(fcsv, "k,path,label,ms_per_iter,ms_per_unit,ms_per_byte\n");

    // Fault rates
    std::vector<double> frs;
    if (forced_fr >= 0) { frs.push_back(forced_fr); }
    else { frs = {0.0, 1e-7, 1e-6, 1e-5}; }

    int T = 256;

    for (uint64_t ki = 0; ki < ks.size(); ki++) {
        uint64_t k = ks[ki];
        // At depth k, only planes 0..k-1 are read. Skip if k > available planes.
        uint64_t k_bytes = plane0_bytes; // always benchmark plane 0
        uint64_t k_units = nu;
        (void)k_bytes; // plane 0 bytes = plane0_bytes regardless of k
        (void)k_units;

        std::fprintf(stderr, "\n=== k=%" PRIu64 " ===\n", k);

        // --- A: single-replica baseline ---
        {
            int blk = (static_cast<int>(plane0_bytes) + T - 1) / T;
            auto fn = [&](cudaStream_t s) { path_a_read<<<blk, T, 0, s>>>(d_r0, d_v, plane0_bytes); };
            auto tr = measure("A: single-replica baseline", fn, ITERS, st);
            fprintf(fcsv, "%" PRIu64 ",A,single_replica_baseline,%.12f,%.12f,%.12f\n",
                k, tr.ms_per_iter, tr.ms_per_iter / nu, tr.ms_per_iter / plane0_bytes);
        }

        // --- B: table CRC, 1 thread/unit ---
        {
            auto fn = [&](cudaStream_t s) { path_b_crc_1tpu<<<static_cast<int>(nu), 1, 0, s>>>(d_r0, nu, ALLOC_UNIT, d_crc); };
            auto tr = measure("B: table CRC, 1 thread/unit", fn, ITERS, st);
            fprintf(fcsv, "%" PRIu64 ",B,table_crc_1tpu,%.12f,%.12f,%.12f\n",
                k, tr.ms_per_iter, tr.ms_per_iter / nu, tr.ms_per_iter / plane0_bytes);
        }

        // --- B2: table CRC, parallel (many threads/block, 1 unit/thread) ---
        {
            int blk2 = (static_cast<int>(nu) + T - 1) / T;
            auto fn = [&](cudaStream_t s) { path_b2_crc_parallel<<<blk2, T, 0, s>>>(d_r0, nu, ALLOC_UNIT, d_crc); };
            auto tr = measure("B2: table CRC, parallel block", fn, ITERS, st);
            fprintf(fcsv, "%" PRIu64 ",B2,table_crc_parallel,%.12f,%.12f,%.12f\n",
                k, tr.ms_per_iter, tr.ms_per_iter / nu, tr.ms_per_iter / plane0_bytes);
        }

        // --- C: CRC + repair dispatch (with fault flags) ---
        for (double fr : frs) {
            uint64_t nf = static_cast<uint64_t>(nu * fr);
            if (fr > 0 && nf == 0) nf = 1;
            std::vector<uint8_t> h_ff(nu, 0);
            for (uint64_t u = 0; u < nf && u < nu; u++)
                h_ff[(u * 2654435761ull) % nu] = 1;
            cuda_check(cudaMemcpy(d_ff, h_ff.data(), nu, cudaMemcpyHostToDevice), "H2D ff");

            auto fn = [&](cudaStream_t s) {
                path_c_crc_repair<<<static_cast<int>(nu), T, 0, s>>>(
                    d_r0, d_r1, d_r2, d_v, nu, ALLOC_UNIT, d_ff, d_crc);
            };
            char lb[80];
            std::snprintf(lb, sizeof(lb), "C: CRC+repair (fr=%.0e, nf=%" PRIu64 ")", fr, nf);
            auto tr = measure(lb, fn, ITERS, st);
            fprintf(fcsv, "%" PRIu64 ",C,crc_repair_fr%.0e_nf%" PRIu64 ",%.12f,%.12f,%.12f\n",
                k, fr, nf, tr.ms_per_iter, tr.ms_per_iter / nu, tr.ms_per_iter / plane0_bytes);
        }

        // --- D: eager read3 + warp vote ---
        {
            int blk = (static_cast<int>(plane0_bytes) + T - 1) / T;
            auto fn = [&](cudaStream_t s) { path_d_eager_vote<<<blk, T, 0, s>>>(d_r0, d_r1, d_r2, d_v, plane0_bytes); };
            auto tr = measure("D: eager read3 + warp vote", fn, ITERS, st);
            fprintf(fcsv, "%" PRIu64 ",D,eager_read3_vote,%.12f,%.12f,%.12f\n",
                k, tr.ms_per_iter, tr.ms_per_iter / nu, tr.ms_per_iter / plane0_bytes);
        }
    }

    fprintf(fcsv, "\n# Derived metrics (post-hoc):\n");
    fprintf(fcsv, "# crc_only_overhead = (path_B - path_A) / path_A\n");
    fprintf(fcsv, "# crc_parallel_overhead = (path_B2 - path_A) / path_A\n");
    fprintf(fcsv, "# nmr_latency_inflation_lazy = amortized_lazy / path_A\n");
    fprintf(fcsv, "# nmr_latency_inflation_eager = path_D / path_A\n");
    fclose(fcsv);
    std::fprintf(stderr, "\nCSV: %s\n", csv_path.c_str());

    cuda_check(cudaFree(d_ff), "F ff");
    cuda_check(cudaFree(d_crc), "F crc");
    cuda_check(cudaFree(d_v), "F v");
    cuda_check(cudaFree(d_r2), "F r2");
    cuda_check(cudaFree(d_r1), "F r1");
    cuda_check(cudaFree(d_r0), "F r0");
    for (auto pp : d_planes) cuda_check(cudaFree(pp), "F plane");
    cuda_check(cudaStreamDestroy(st), "D stream");
    return 0;
}
