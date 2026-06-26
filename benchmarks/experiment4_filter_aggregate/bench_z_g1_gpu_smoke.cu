// Phase 3-Z G1: H200 end-to-end GPU injection smoke
// Inject byte fault -> parallel SUM32 detect -> certified-interval fallback.
// Compares GPU integer-encoded accumulator result vs CPU reference.
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
//        && make bench_z_g1_gpu_smoke
// Run:   srun --gres=gpu:1 ./bench_z_g1_gpu_smoke --dataset PATH [--dataset PATH2]

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

// ── Parallel SUM32 (warp/block reduce) ────────────────────────────────
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

// ── Integer-encoded SUM: accumulate plane bytes as uint64 per row ─────
__global__ void integer_sum_all_planes(uint8_t **d_planes,
    int n_planes, uint64_t row_count,
    uint64_t *__restrict__ d_row_codes) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row_count) return;
    uint64_t code = 0;
    for (int p = 0; p < n_planes; p++) {
        code = (code << 8) | static_cast<uint64_t>(d_planes[p][i]);
    }
    d_row_codes[i] = code;
}

// ── Per-unit SUM32 reference (CPU, for comparison) ────────────────────
static uint32_t cpu_sum32(const uint8_t *data, uint64_t n_bytes) {
    uint32_t s = 0;
    for (uint64_t i = 0; i < n_bytes; i++) s += static_cast<uint32_t>(data[i]);
    return s;
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
    std::fprintf(stderr, "Device: %s (sm_%d%d)\n\n", p.name, p.major, p.minor);
    const char *jid = std::getenv("SLURM_JOB_ID"); if (!jid) jid = "NOJOB";
    std::fprintf(stderr, "Job: %s\n\n", jid);

    int T = 256;
    uint64_t ALLOC_UNIT = 4096;

    for (auto &ds_path : dataset_paths) {
        Dataset ds = exp3_real::load_dataset(ds_path);
        int n_planes = static_cast<int>(ds.planes.size());
        if (n_planes == 0) { std::fprintf(stderr, "skip (no planes)\n"); continue; }
        uint64_t pb = ds.planes[0].size(); // bytes per plane
        uint64_t n_rows = ds.manifest.value_count;
        std::string ds_name = ds.manifest.dataset;
        uint64_t nu = (pb + ALLOC_UNIT - 1) / ALLOC_UNIT;

        std::fprintf(stderr, "=== %s ===\n", ds_name.c_str());
        std::fprintf(stderr, "  planes=%d  rows=%" PRIu64 "  plane_bytes=%" PRIu64 "\n", n_planes, n_rows, pb);

        // Allocate device planes
        std::vector<uint8_t*> d_planes(n_planes);
        std::vector<uint8_t*> d_planes_faulted(n_planes);
        for (int p = 0; p < n_planes; p++) {
            cuda_check(cudaMalloc(&d_planes[p], pb), "M pl");
            cuda_check(cudaMemcpy(d_planes[p], ds.planes[p].data(), pb, cudaMemcpyHostToDevice), "H2D pl");
            cuda_check(cudaMalloc(&d_planes_faulted[p], pb), "M pl_f");
            cuda_check(cudaMemcpy(d_planes_faulted[p], ds.planes[p].data(), pb, cudaMemcpyHostToDevice), "H2D pl_f");
        }

        uint32_t *d_digest_ref, *d_digest_fault;
        cuda_check(cudaMalloc(&d_digest_ref, nu * sizeof(uint32_t)), "M d_ref");
        cuda_check(cudaMalloc(&d_digest_fault, nu * sizeof(uint32_t)), "M d_fault");

        // Compute reference digests on clean data
        uint64_t *d_row_codes = nullptr;

        // For each test plane, inject a fault and verify detect+contain
        int test_planes[] = {0, 1, n_planes - 1};
        int n_test = 3;

        int total_tests = 0, detected = 0, contained = 0, gpu_cpu_match = 0;

        // CPU reference: compute integer-encoded sum of all planes (clean)
        std::vector<uint64_t> cpu_row_codes(n_rows, 0);
        for (uint64_t i = 0; i < n_rows; i++) {
            uint64_t code = 0;
            for (int p = 0; p < n_planes; p++) {
                code = (code << 8) | static_cast<uint64_t>(ds.planes[p][i]);
            }
            cpu_row_codes[i] = code;
        }
        uint64_t cpu_total_code = 0;
        for (uint64_t i = 0; i < n_rows; i++) cpu_total_code += cpu_row_codes[i];

        for (int ti = 0; ti < n_test; ti++) {
            int plane = test_planes[ti];
            if (plane >= n_planes) continue;

            // ── Test 1: Digest detect ──
            // Compute clean digest
            cuda_check(cudaMemcpy(d_planes_faulted[plane], ds.planes[plane].data(), pb, cudaMemcpyHostToDevice), "reset");
            sum32_parallel<<<static_cast<int>(nu), T>>>(d_planes[0], nu, ALLOC_UNIT, d_digest_ref);
            cuda_check(cudaDeviceSynchronize(), "digest_ref");

            // Inject fault: flip one byte at offset 0 in the test plane
            uint8_t fault_byte = ds.planes[plane].data()[0] ^ 0xFF;
            cuda_check(cudaMemcpy(d_planes_faulted[plane], &fault_byte, 1, cudaMemcpyHostToDevice), "inject");

            // Compute faulted digest
            sum32_parallel<<<static_cast<int>(nu), T>>>(d_planes_faulted[0], nu, ALLOC_UNIT, d_digest_fault);
            cuda_check(cudaDeviceSynchronize(), "digest_fault");

            // Compare digests (CPU-side for simplicity)
            std::vector<uint32_t> ref_dig(nu), fault_dig(nu);
            cuda_check(cudaMemcpy(ref_dig.data(), d_digest_ref, nu * sizeof(uint32_t), cudaMemcpyDeviceToHost), "D2H ref");
            cuda_check(cudaMemcpy(fault_dig.data(), d_digest_fault, nu * sizeof(uint32_t), cudaMemcpyDeviceToHost), "D2H fault");

            bool dig_match = true;
            for (uint64_t u = 0; u < nu && dig_match; u++)
                if (ref_dig[u] != fault_dig[u]) dig_match = false;

            total_tests++;
            if (!dig_match) detected++;
            std::fprintf(stderr, "  plane=%d fault@byte0: digest_match=%d  detected=%s\n",
                plane, (int)dig_match, dig_match ? "NO (escape!)" : "YES");

            // ── Test 2: Certified interval containment ──
            // After detection, fall back to full read with integer accumulator.
            // GPU: read all planes, compute integer-encoded code per row, sum.
            cuda_check(cudaMalloc(&d_row_codes, n_rows * sizeof(uint64_t)), "M codes");

            // Make array of device pointers on device
            uint8_t **d_dptr = nullptr;
            cuda_check(cudaMalloc(&d_dptr, n_planes * sizeof(uint8_t*)), "M dptr");
            // Copy d_planes_faulted pointers to device. For the faulted test,
            // we use the faulted plane data.
            std::vector<uint8_t*> h_dptrs(n_planes);
            for (int p = 0; p < n_planes; p++) {
                h_dptrs[p] = (p == plane) ? d_planes_faulted[p] : d_planes[p];
            }
            cuda_check(cudaMemcpy(d_dptr, h_dptrs.data(), n_planes * sizeof(uint8_t*), cudaMemcpyHostToDevice), "H2D dptr");

            int blk = (static_cast<int>(n_rows) + T - 1) / T;
            integer_sum_all_planes<<<blk, T>>>(d_dptr, n_planes, n_rows, d_row_codes);
            cuda_check(cudaDeviceSynchronize(), "int_sum");

            std::vector<uint64_t> gpu_row_codes(n_rows);
            cuda_check(cudaMemcpy(gpu_row_codes.data(), d_row_codes, n_rows * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H codes");
            cuda_check(cudaFree(d_row_codes), "F d_row");
            cuda_check(cudaFree(d_dptr), "F dptr");

            uint64_t gpu_total_code = 0;
            for (uint64_t i = 0; i < n_rows; i++) gpu_total_code += gpu_row_codes[i];

            // The certified interval is [total_code - U_i, total_code + U_i].
            // U_i = 255 * 256^(7-plane) * n_rows  (worst case per Z1a).
            // For the faulted plane:
            uint64_t plane_weight = 1;
            for (int p = 0; p < n_planes - 1 - plane; p++) plane_weight *= 256;
            // Actually for plane 0 (MSB): weight = 256^7
            // For plane p: weight = 256^(7-p) in an 8-plane artifact
            // But in our n_plane system (which may have fewer than 8):
            // The byte at plane p contributes to byte position p in the code.
            // U_i = 255 * weight[i] where weight[i] = 256^(n_planes-1-i)
            uint64_t u_i_weight = 1;
            for (int p = 0; p < n_planes - 1 - plane; p++) u_i_weight *= 256;
            uint64_t u_i = 255ULL * u_i_weight * n_rows;

            // The certified interval is [min(gpu_total_code, cpu_total_code) - U_i,
            //                            max(gpu_total_code, cpu_total_code) + U_i]
            // But actually, we need to check: does the certified interval contain the truth?
            // The truth is the clean (CPU) total. The GPU read includes the faulted data.
            // If the fault causes a wrong value, does the certified interval cover it?
            int64_t delta = static_cast<int64_t>(gpu_total_code) - static_cast<int64_t>(cpu_total_code);
            int64_t signed_delta = std::abs(delta);
            bool contains_truth = static_cast<uint64_t>(signed_delta) <= u_i;
            if (contains_truth) contained++;

            bool code_match = (gpu_total_code == cpu_total_code);
            if (code_match) gpu_cpu_match++;

            std::fprintf(stderr, "    cpu_total=%-20" PRIu64 "  gpu_total=%-20" PRIu64
                "  delta=%" PRId64 "  U_i=%-20" PRIu64 "  contains=%s  exact_match=%s\n",
                cpu_total_code, gpu_total_code, delta, u_i,
                contains_truth ? "YES" : "NO",
                code_match ? "YES" : "NO");
        }

        std::fprintf(stderr, "\n  === Summary ===\n");
        std::fprintf(stderr, "  tests=%d detected=%d containment=%d gpu_cpu_exact=%d\n",
            total_tests, detected, contained, gpu_cpu_match);
        std::fprintf(stderr, "  detected_rate=%.4f  contains_truth=%.4f\n",
            total_tests ? (double)detected/total_tests : 0,
            total_tests ? (double)contained/total_tests : 0);

        // Verdict
        bool pass_detect = (detected == total_tests);
        bool pass_contain = (contained == total_tests);
        std::fprintf(stderr, "\n  Verdict: %s\n",
            (pass_detect && pass_contain) ? "GPU_SMOKE_CONFIRMED" :
            (!pass_detect) ? "GPU_SMOKE_DIVERGENCE (detection failed)" :
            "GPU_SMOKE_DIVERGENCE (containment failed)");

        for (int p = 0; p < n_planes; p++) {
            cuda_check(cudaFree(d_planes[p]), "F pl");
            cuda_check(cudaFree(d_planes_faulted[p]), "F pl_f");
        }
        cuda_check(cudaFree(d_digest_ref), "F ref");
        cuda_check(cudaFree(d_digest_fault), "F fault");
    }
    return 0;
}
