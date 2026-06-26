// Phase 3-S: Fused NMR duplicate-read/compare overhead microbenchmark
// Measures the actual GPU cost of re-reading the highest-significance plane
// and comparing with a reference copy (simulating fused 2-replica NMR vote).
//
// The "fused" aspect: the first read populates L2 cache; the second read
// (duplicate) may hit cached data, giving the true fused overhead.
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release && make bench_nmr_overhead
// Run:   ./bench_nmr_overhead --dataset PATH

#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <string_view>
#include <vector>

#include "../experiment3/exp3_real_data_layout.hpp"

using namespace exp3_real;

namespace {

[[nodiscard]] const char* cuda_err_str(cudaError_t e) { return cudaGetErrorString(e); }
[[noreturn]] void die(const char* m) { std::fprintf(stderr, "error: %s\n", m); std::exit(2); }
void cuda_check(cudaError_t e, const char* w) {
  if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error at %s: %s\n", w, cuda_err_str(e)); std::exit(2); }
}

// Read + simple sum kernel (populates L2 cache for the plane data)
__global__ void read_and_sum_kernel(
    const uint8_t* __restrict__ plane,
    uint64_t n,
    unsigned long long* __restrict__ out)
{
  uint64_t idx = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  atomicAdd(out, (unsigned long long)(plane[idx]));
}

// Duplicate-read + compare kernel
__global__ void duplicate_read_compare_kernel(
    const uint8_t* __restrict__ plane,
    const uint8_t* __restrict__ ref_plane,
    uint64_t n,
    unsigned long long* __restrict__ d_mismatch_count)
{
  uint64_t idx = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  if (plane[idx] != ref_plane[idx])
    atomicAdd(d_mismatch_count, 1ull);
}

void print_usage(const char* argv0) {
  std::fprintf(stderr,
    "Usage: %s [options]\n"
    "Phase 3-S: Fused NMR duplicate-read/compare overhead\n"
    "Options:\n"
    "  --dataset PATH      Encoded dataset directory\n"
    "  --iters N           Iterations per measurement (default: 50)\n"
    "  --warmup N          Warmup iterations (default: 10)\n",
    argv0);
}

} // anonymous namespace

int main(int argc, char** argv) {
  std::string dataset_path;
  int iters = 50, warmup = 10;

  for (int i = 1; i < argc; ++i) {
    std::string_view a(argv[i]);
    auto val = [&]() { if (++i >= argc) die("missing value"); return std::string(argv[i]); };
    if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
    else if (a == "--dataset")      dataset_path = val();
    else if (a == "--iters")        { int v = std::stoi(val()); if (v <= 0) die("invalid --iters"); iters = v; }
    else if (a == "--warmup")       { int v = std::stoi(val()); if (v < 0) die("invalid --warmup"); warmup = v; }
      else { std::string msg = "unknown flag: "; msg += a; die(msg.c_str()); }
  }
  if (dataset_path.empty()) die("--dataset is required");

  cuda_check(cudaSetDevice(0), "cudaSetDevice");
  cudaDeviceProp prop;
  cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  std::fprintf(stderr, "Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  // Load dataset
  Dataset dataset = exp3_real::load_dataset(dataset_path);

  uint64_t n = 0;
  for (auto& seg : dataset.segments) n += seg.row_count;
  int max_planes = static_cast<int>(dataset.planes.size());

  std::fprintf(stderr, "Dataset: %s n=%" PRIu64 " segments=%zu max_planes=%d\n",
               dataset.manifest.dataset.c_str(), n, dataset.segments.size(), max_planes);

  uint64_t plane0_bytes = dataset.planes[0].size();
  std::fprintf(stderr, "Highest plane (plane_000): %" PRIu64 " bytes\n", plane0_bytes);

  // Copy plane_000 to device (the highest-significance plane)
  uint8_t* d_plane = nullptr;
  cuda_check(cudaMalloc(&d_plane, plane0_bytes), "cudaMalloc(plane)");
  cuda_check(cudaMemcpy(d_plane, dataset.planes[0].data(), plane0_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(plane)");

  // Reference copy
  uint8_t* d_ref = nullptr;
  cuda_check(cudaMalloc(&d_ref, plane0_bytes), "cudaMalloc(ref)");
  cuda_check(cudaMemcpy(d_ref, dataset.planes[0].data(), plane0_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(ref)");

  unsigned long long* d_mismatch = nullptr;
  cuda_check(cudaMalloc(&d_mismatch, sizeof(unsigned long long)), "cudaMalloc(mismatch)");

  unsigned long long* d_sum = nullptr;
  cuda_check(cudaMalloc(&d_sum, sizeof(unsigned long long)), "cudaMalloc(sum)");

  uint64_t threads = 256;
  uint64_t blocks = (plane0_bytes + threads - 1) / threads;

  cudaStream_t stream;
  cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

  // -------------------------------------------------------------------
  // Step 1: Warm up L2 cache with the plane data (simulates B1's read)
  // -------------------------------------------------------------------
  std::fprintf(stderr, "\n=== Pre-populating L2 cache (simulating fused read path) ===\n");
  unsigned long long warm_sum = 0;
  for (int i = 0; i < 5; ++i) {
    cuda_check(cudaMemset(d_sum, 0, sizeof(unsigned long long)), "prepop memset");
    read_and_sum_kernel<<<blocks, threads, 0, stream>>>(d_plane, plane0_bytes, d_sum);
    cuda_check(cudaStreamSynchronize(stream), "prepop sync");
  }
  cuda_check(cudaMemcpy(&warm_sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "prepop readback");
  std::fprintf(stderr, "  Warm-up sum (verification): %llu\n", warm_sum);

  // -------------------------------------------------------------------
  // Step 2: Fused duplicate-read + compare (immediately after warm-up)
  // -------------------------------------------------------------------
  std::fprintf(stderr, "\n=== Fused duplicate-read/compare ===\n");

  auto measure_dup = [&]() -> double {
    // Timing helper
    struct GT {
      double min = 1e30, max = 0, sum = 0; int cnt = 0;
      void record(double ms) { if (ms < min) min = ms; if (ms > max) max = ms; sum += ms; ++cnt; }
      double avg() const { return cnt ? sum / cnt : 0; }
    } t;

    for (int i = 0; i < warmup; ++i) {
      cuda_check(cudaMemset(d_mismatch, 0, sizeof(unsigned long long)), "warm memset");
      duplicate_read_compare_kernel<<<blocks, threads, 0, stream>>>(d_plane, d_ref, plane0_bytes, d_mismatch);
    }
    cuda_check(cudaStreamSynchronize(stream), "warm sync");

    for (int i = 0; i < iters; ++i) {
      cuda_check(cudaMemset(d_mismatch, 0, sizeof(unsigned long long)), "iter memset");
      auto start = std::chrono::steady_clock::now();
      duplicate_read_compare_kernel<<<blocks, threads, 0, stream>>>(d_plane, d_ref, plane0_bytes, d_mismatch);
      cuda_check(cudaStreamSynchronize(stream), "iter sync");
      auto end = std::chrono::steady_clock::now();
      t.record(std::chrono::duration<double, std::milli>(end - start).count());
    }
    return t.avg();
  };

  double t_dup = measure_dup();

  unsigned long long h_mismatch = 0;
  cuda_check(cudaMemcpy(&h_mismatch, d_mismatch, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "mismatch readback");

  // -------------------------------------------------------------------
  // Report
  // -------------------------------------------------------------------
  std::fprintf(stderr, "\n========================================\n");
  std::fprintf(stderr, "  Fused Duplicate-Read/Compare Report\n");
  std::fprintf(stderr, "========================================\n");
  std::fprintf(stderr, "  Dataset:        %s\n", dataset.manifest.dataset.c_str());
  std::fprintf(stderr, "  Plane size:     %" PRIu64 " bytes\n", plane0_bytes);
  std::fprintf(stderr, "  Blocks:         %" PRIu64 "\n", blocks);
  std::fprintf(stderr, "  Threads/block:  %" PRIu64 "\n", threads);
  std::fprintf(stderr, "  Iters:          %d\n", iters);
  std::fprintf(stderr, "\n");
  std::fprintf(stderr, "  Duplicate-read + compare:\n");
  std::fprintf(stderr, "    Avg:          %.6f ms\n", t_dup);
  std::fprintf(stderr, "  Mismatch count: %llu\n", h_mismatch);
  std::fprintf(stderr, "\n");

  if (h_mismatch == 0) {
    std::fprintf(stderr, "  Data integrity: PASS (plane == ref copy) ✅\n");
  } else {
    std::fprintf(stderr, "  Data integrity: FAIL (%llu mismatches) ❌\n", h_mismatch);
  }

  // Cleanup
  cuda_check(cudaFree(d_plane), "cudaFree(plane)");
  cuda_check(cudaFree(d_ref), "cudaFree(ref)");
  cuda_check(cudaFree(d_mismatch), "cudaFree(mismatch)");
  cuda_check(cudaFree(d_sum), "cudaFree(sum)");
  cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");

  return 0;
}
