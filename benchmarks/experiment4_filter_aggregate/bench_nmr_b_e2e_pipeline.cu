// NMR-B: GPU End-to-End NMR Pipeline
// Full pipeline on GPU: replicate -> vote -> digest -> detect -> bound -> classify
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
//        && make bench_nmr_b_e2e_pipeline
// Run:   ./bench_nmr_b_e2e_pipeline --dataset PATH [--r-vector "3|2|1|1|1|1|1|1"]
//         [--fault-mode single_replica_plane0] [--seed 0] [--csv PATH] [--n-rows N]

#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <random>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "../experiment3/exp3_real_data_layout.hpp"

using namespace exp3_real;

constexpr int MAX_R = 4;

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
// Utility: parse r-vector from "3|2|1|1|1|1|1|1"
// ===========================================================================
static int parse_r_vector(const char *s, int *r_out, int max_count) {
    int count = 0;
    const char *p = s;
    while (*p && count < max_count) {
        int val = 0;
        while (*p >= '0' && *p <= '9') { val = val * 10 + (*p - '0'); p++; }
        r_out[count++] = val;
        if (*p == '|') p++;
        else if (*p) break;
    }
    return count;
}

// ===========================================================================
// Kernels
// ===========================================================================

// Grid-stride byte-level majority vote for arbitrary r (optimized for r=2,3).
// Uses capped grid so each block processes multiple elements.
__global__ void byte_majority_vote_gpu_gs(
    const uint8_t *__restrict__ r0,
    const uint8_t *__restrict__ r1,
    const uint8_t *__restrict__ r2,
    uint8_t *__restrict__ voted,
    uint64_t n, int r) {
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
         i < n; i += stride) {
        uint8_t v = r0[i];
        if (r == 1) {
            voted[i] = v;
        } else if (r == 2) {
            uint8_t v1 = r1[i];
            voted[i] = (v >= v1) ? v : v1;
        } else {
            uint8_t vals[MAX_R];
            vals[0] = v; vals[1] = r1[i]; vals[2] = r2[i];
            int best = 0, best_cnt = 0;
            for (int j = 0; j < r; j++) {
                int cnt = 0;
                for (int k = 0; k < r; k++)
                    if (vals[k] == vals[j]) cnt++;
                if (cnt > best_cnt || (cnt == best_cnt && vals[j] > vals[best])) {
                    best = j;
                    best_cnt = cnt;
                }
            }
            voted[i] = vals[best];
        }
    }
}

// Per-block partial SUM32 — grid-stride loop, each block writes its partial sum.
__global__ void sum32_digest_partial(
    const uint8_t *__restrict__ src,
    uint64_t n,
    uint32_t *__restrict__ block_partials) {
    uint64_t tid = threadIdx.x;
    const uint32_t warp_count = static_cast<uint32_t>(blockDim.x >> 5);
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
        uint32_t w = (tid < warp_count) ? s_red[tid] : 0;
        for (int off = 16; off > 0; off >>= 1)
            w += __shfl_down_sync(0xFFFFFFFFu, w, off);
        if (tid == 0) block_partials[blockIdx.x] = w;
    }
}

// Helper: launch sum32_digest_partial, copy block_partials back, host-reduce.
static uint32_t compute_digest_host_reduce(
    const uint8_t *__restrict__ d_src, uint64_t n,
    uint32_t *d_partial, int grid, int T, cudaStream_t st) {
    sum32_digest_partial<<<grid, T, 0, st>>>(d_src, n, d_partial);
    cuda_check(cudaStreamSynchronize(st), "digest sync");
    std::vector<uint32_t> h_partial(static_cast<size_t>(grid));
    cuda_check(cudaMemcpy(h_partial.data(), d_partial,
        static_cast<size_t>(grid) * sizeof(uint32_t), cudaMemcpyDeviceToHost), "D2H partial");
    uint32_t total = 0;
    for (auto v : h_partial) total += v;
    return total;
}

// ===========================================================================
// Main
// ===========================================================================
// ── Simple fault plan parser (text format: "plane replica offset mask") ──
struct FaultEntry { int plane, replica; uint64_t offset; uint8_t mask; };
static std::vector<FaultEntry> load_fault_plan(const std::string &path) {
    std::vector<FaultEntry> plan;
    std::ifstream f(path);
    if (!f.is_open()) return plan;  // empty = no fault
    int p, r; unsigned long long o; unsigned m;
    while (f >> p >> r >> o >> m)
        plan.push_back({p, r, o, static_cast<uint8_t>(m)});
    return plan;
}

int main(int argc, char **argv) {
    std::string dataset_path, csv_path = "nmr_b_e2e.csv", fault_plan_path;
    std::string fault_mode = "shared_plan";
    int r_vector[8] = {3, 2, 1, 1, 1, 1, 1, 1};
    int iters = 30, fault_plane = 0, seed = 0;
    uint64_t n_rows_req = 0;

    for (int i = 1; i < argc; ++i) {
        std::string_view a(argv[i]);
        auto val = [&]() { if (++i >= argc) die("missing value"); return std::string(argv[i]); };
        if (a == "--help" || a == "-h") {
            std::fprintf(stderr,
                "Usage: %s --dataset PATH [--r-vector \"3|2|1|1|1|1|1|1\"]\n"
                "  [--fault-plan PATH] [--csv PATH] [--iters N] [--n-rows N]\n"
                "  [--fault-mode LABEL] [--fault-plane N]\n", argv[0]);
            return 0;
        } else if (a == "--dataset")     dataset_path = val();
        else if (a == "--r-vector")      parse_r_vector(val().c_str(), r_vector, 8);
        else if (a == "--fault-plan")    fault_plan_path = val();
        else if (a == "--fault-mode")    fault_mode = val();
        else if (a == "--fault-plane")   fault_plane = std::stoi(val());
        else if (a == "--seed")          seed = std::stoi(val());
        else if (a == "--csv")           csv_path = val();
        else if (a == "--iters")         iters = std::stoi(val());
        else if (a == "--n-rows")        n_rows_req = strtoull(val().c_str(), nullptr, 10);
        else { std::string msg = "unknown: "; msg += a; die(msg.c_str()); }
    }
    if (dataset_path.empty()) die("--dataset required");

    cuda_check(cudaSetDevice(0), "dev");
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, 0), "prop");
    std::fprintf(stderr, "Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    // Load dataset
    Dataset ds = exp3_real::load_dataset(dataset_path);
    uint64_t n = ds.manifest.value_count;
    if (n_rows_req > 0 && n_rows_req < n) n = n_rows_req;
    int actual_planes = static_cast<int>(ds.planes.size());
    if (actual_planes > 8) actual_planes = 8;  // safety clamp
    std::string ds_name = ds.manifest.dataset;

    std::fprintf(stderr, "Dataset: %s n=%" PRIu64 " planes=%d\n", ds_name.c_str(), n, actual_planes);
    std::fprintf(stderr, "R-vector: ");
    for (int p = 0; p < actual_planes; p++) std::fprintf(stderr, "%d ", r_vector[p]);
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "Fault: %s seed=%d\n", fault_mode.c_str(), seed);

    int T = 256;
    int blk = static_cast<int>((n + static_cast<uint64_t>(T) - 1) / static_cast<uint64_t>(T));
    int grid_digest = std::min(blk, 512);
    cudaStream_t st;
    cuda_check(cudaStreamCreate(&st), "st");

    // ── Upload clean planes (only actual_planes) ──
    std::vector<uint8_t*> d_clean(actual_planes, nullptr);
    for (int p = 0; p < actual_planes; p++) {
        cuda_check(cudaMalloc(&d_clean[p], n), "M clean");
        cuda_check(cudaMemcpy(d_clean[p], ds.planes[p].data(), n, cudaMemcpyHostToDevice), "H2D clean");
    }

    // ── Allocate replica buffers ──
    struct ReplicaSet { uint8_t *buf[MAX_R] = {}; int r = 0; };
    std::vector<ReplicaSet> d_replicas(actual_planes);
    for (int p = 0; p < actual_planes; p++) {
        int r_p = r_vector[p];
        if (r_p > MAX_R) r_p = MAX_R;
        d_replicas[p].r = r_p;
        for (int j = 0; j < r_p; j++) {
            cuda_check(cudaMalloc(&d_replicas[p].buf[j], n), "M rep");
            cuda_check(cudaMemcpy(d_replicas[p].buf[j], d_clean[p], n, cudaMemcpyDeviceToDevice), "D2D rep");
        }
    }

    // ── Voted output buffers ──
    std::vector<uint8_t*> d_voted(actual_planes, nullptr);
    for (int p = 0; p < actual_planes; p++) {
        cuda_check(cudaMalloc(&d_voted[p], n), "M voted");
    }

    // ── Digest buffers ──
    uint32_t *d_partial = nullptr;
    cuda_check(cudaMalloc(&d_partial, static_cast<size_t>(grid_digest) * sizeof(uint32_t)), "M partial");

    // ── Apply shared fault plan (from Python runner-generated file) ──
    auto fault_plan = load_fault_plan(fault_plan_path);
    for (const auto &e : fault_plan) {
        if (e.plane >= actual_planes) continue;
        if (e.replica >= d_replicas[e.plane].r || e.replica >= MAX_R) continue;
        if (e.offset >= n) continue;
        std::vector<uint8_t> corrupted(n);
        cuda_check(cudaMemcpy(corrupted.data(), d_replicas[e.plane].buf[e.replica],
            n, cudaMemcpyDeviceToHost), "D2H read back");
        corrupted[e.offset] ^= e.mask;
        cuda_check(cudaMemcpy(d_replicas[e.plane].buf[e.replica], corrupted.data(),
            n, cudaMemcpyHostToDevice), "H2D inject");
        std::fprintf(stderr, "  Injected: plane=%d replica=%d offset=%" PRIu64 " mask=%02x\n",
            e.plane, e.replica, e.offset, e.mask);
    }
    if (fault_plan.empty())
        std::fprintf(stderr, "  No fault entries (clean_no_fault)\n");

    // ── Compute clean digests (GPU) ──
    std::vector<uint32_t> h_clean_digests(actual_planes, 0);
    for (int p = 0; p < actual_planes; p++) {
        h_clean_digests[p] = compute_digest_host_reduce(d_clean[p], n, d_partial, grid_digest, T, st);
    }

    // ── Measure GPU pipeline: vote + digest ──
    auto t_pipeline = measure([&](cudaStream_t s) {
        for (int p = 0; p < actual_planes; p++) {
            int r_p = d_replicas[p].r;
            if (r_p == 0) continue;
            byte_majority_vote_gpu_gs<<<grid_digest, T, 0, s>>>(
                d_replicas[p].buf[0],
                d_replicas[p].buf[1 % (r_p > 0 ? r_p : 1)],
                d_replicas[p].buf[2 % (r_p > 0 ? r_p : 1)],
                d_voted[p], n, r_p);
        }
        for (int p = 0; p < actual_planes; p++) {
            if (d_replicas[p].r == 0) continue;
            sum32_digest_partial<<<grid_digest, T, 0, s>>>(d_voted[p], n, d_partial);
        }
    }, iters, st);

    // ── Read back voted digests ──
    std::vector<uint32_t> h_voted_digests(actual_planes, 0);
    for (int p = 0; p < actual_planes; p++) {
        if (d_replicas[p].r == 0) { h_voted_digests[p] = h_clean_digests[p]; continue; }
        h_voted_digests[p] = compute_digest_host_reduce(d_voted[p], n, d_partial, grid_digest, T, st);
    }

    // ── Classification ──
    bool detected = false;
    std::vector<int> detected_planes;
    for (int p = 0; p < actual_planes; p++) {
        if (h_voted_digests[p] != h_clean_digests[p]) {
            detected = true;
            detected_planes.push_back(p);
        }
    }
    bool vote_recovered = !detected;

    // Plane weights as double (1ULL << shift to avoid 32-bit overflow on planes 0/1)
    auto plane_weight = [](int p) -> double {
        return static_cast<double>(1ULL << (8 * (7 - p)));
    };

    // Compute answer and bound (host-side, using plane weights)
    auto compute_sum = [&](const std::vector<uint8_t*> &plane_ptrs, int plane_idx) -> double {
        if (plane_idx >= actual_planes) return 0.0;
        std::vector<uint8_t> h_plane(n);
        cuda_check(cudaMemcpy(h_plane.data(), plane_ptrs[plane_idx], n, cudaMemcpyDeviceToHost), "D2H compute");
        uint64_t sum = 0;
        for (uint64_t i = 0; i < n; i++) sum += h_plane[i];
        return static_cast<double>(sum) * plane_weight(plane_idx);
    };

    double clean_answer = 0.0;
    for (int p = 0; p < actual_planes; p++)
        clean_answer += compute_sum(d_clean, p);

    double delivered_answer = 0.0;
    for (int p = 0; p < actual_planes; p++)
        delivered_answer += compute_sum(d_voted, p);

    double bound_width = 0.0;
    for (int p : detected_planes)
        bound_width += 255.0 * plane_weight(p) * static_cast<double>(n);

    double lo = delivered_answer - bound_width / 2.0;
    double hi = delivered_answer + bound_width / 2.0;
    bool contains_truth = (!detected) || (lo <= clean_answer && clean_answer <= hi);

    const char *outcome = "vote_recovered";
    if (!vote_recovered) {
        outcome = contains_truth ? "bounded_degraded" : "cert_bound_failure";
    }

    // ── Sanity check: read back plane 0 voted data ──
    uint64_t mismatches_byte = 0;
    if (actual_planes > 0) {
        std::vector<uint8_t> h_voted(n);
        cuda_check(cudaMemcpy(h_voted.data(), d_voted[0], n, cudaMemcpyDeviceToHost), "D2H verify");
        std::vector<uint8_t> h_clean(n);
        cuda_check(cudaMemcpy(h_clean.data(), d_clean[0], n, cudaMemcpyDeviceToHost), "D2H verify clean");
        for (uint64_t i = 0; i < n; i++)
            if (h_voted[i] != h_clean[i]) mismatches_byte++;
    }

    // ── Report ──
    std::fprintf(stderr, "\n=== GPU E2E NMR Pipeline ===\n");
    std::fprintf(stderr, "  Latency:           %.6f ms\n", t_pipeline.ms);
    std::fprintf(stderr, "  Vote recovered:    %s\n", vote_recovered ? "YES" : "NO");
    std::fprintf(stderr, "  Detected:          %s\n", detected ? "YES" : "NO");
    std::fprintf(stderr, "  Contains truth:    %s\n", contains_truth ? "YES" : "NO");
    std::fprintf(stderr, "  Outcome:           %s\n", outcome);
    std::fprintf(stderr, "  Bound width:       %.6e\n", bound_width);
    std::fprintf(stderr, "  Mismatches (p0):   %" PRIu64 "\n", mismatches_byte);

    // ── CSV output ──
    FILE *fcsv = std::fopen(csv_path.c_str(), "a");
    if (!fcsv) die("cannot open CSV");
    if (std::ftell(fcsv) == 0) {
        std::fprintf(fcsv, "dataset,n_rows,path,replica_policy,r_vector,fault_mode,fault_plane,seed,"
            "latency_ms,contains_truth,recovered_rate,detected_rate,uncertified_rate,"
            "certified_availability,bound_width,bound_width_norm,"
            "vote_recovered,fallback_used,same_fault_false_recovery,"
            "silent_wrong,cert_bound_failure,hard_fail,bounded_degrade,"
            "cpu_gpu_classification_match,verdict_cell\n");
    }

    std::string r_vec_str;
    for (int p = 0; p < 8; p++) {
        if (p > 0) r_vec_str += "|";
        r_vec_str += std::to_string(r_vector[p]);
    }

    double recovered_rate = vote_recovered ? 1.0 : 0.0;
    double detected_rate = detected ? 1.0 : 0.0;
    double uncertified_rate = (strcmp(outcome, "cert_bound_failure") == 0) ? 1.0 : 0.0;
    double certified_availability = (strcmp(outcome, "bounded_degraded") == 0) ? 1.0 : 0.0;
    double bw_norm = (clean_answer != 0.0) ? bound_width / fabs(clean_answer) : bound_width;

    std::fprintf(fcsv, "%s,%" PRIu64 ",gpu_e2e,gpu_e2e,%s,%s,%d,%d,"
        "%.6f,%s,%.6f,%.6f,%.6f,%.6f,%.6e,%.6e,"
        "%s,%s,%s,"
        "%s,%s,%s,%s,"
        "%s,%s\n",
        ds_name.c_str(), n,
        r_vec_str.c_str(),
        fault_mode.c_str(), fault_plane, seed,
        t_pipeline.ms,
        contains_truth ? "1.0" : "0.0",
        recovered_rate, detected_rate, uncertified_rate, certified_availability,
        bound_width, bw_norm,
        vote_recovered ? "true" : "false",
        "false",
        (fault_mode.find("same_fault") != std::string::npos && vote_recovered) ? "true" : "false",
        "false", "false", "false",
        (strcmp(outcome, "bounded_degraded") == 0) ? "true" : "false",
        "NA",
        outcome);
    std::fclose(fcsv);
    std::fprintf(stderr, "\nCSV: %s\n", csv_path.c_str());

    // ── Cleanup ──
    for (int p = 0; p < actual_planes; p++) {
        for (int j = 0; j < d_replicas[p].r; j++)
            if (d_replicas[p].buf[j]) cuda_check(cudaFree(d_replicas[p].buf[j]), "F rep");
        cuda_check(cudaFree(d_voted[p]), "F voted");
        cuda_check(cudaFree(d_clean[p]), "F clean");
    }
    cuda_check(cudaFree(d_partial), "F partial");
    cuda_check(cudaStreamDestroy(st), "D st");

    return 0;
}
