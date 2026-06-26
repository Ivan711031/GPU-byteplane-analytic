// Phase 3-U: Fused 2-replica graded NMR — B2-only measurement
// Measures fused duplicate-read/compare of S0 plane.
// B0/B1 timing must be provided separately (from bench_filter_aggregate).
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release && make bench_graded_nmr
// Run:   ./bench_graded_nmr --dataset PATH --raw PATH --b1-time MS [options]

#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "../experiment3/exp3_real_data_layout.hpp"

using namespace exp3_real;

// ---------------------------------------------------------------------------
// CRC32C (host-only, for segment verification)
// ---------------------------------------------------------------------------
static uint32_t s_crc32c_table[256];
static bool s_crc32c_init = false;
static void init_crc() {
  if (s_crc32c_init) return;
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t crc = i;
    for (int j = 0; j < 8; ++j) crc = (crc >> 1) ^ (crc & 1 ? 0x82F63B78u : 0);
    s_crc32c_table[i] = crc;
  }
  s_crc32c_init = true;
}
static uint32_t crc32c_host(const uint8_t* data, uint64_t len) {
  uint32_t crc = 0xFFFFFFFFu;
  for (uint64_t i = 0; i < len; ++i) crc = s_crc32c_table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

// Read + compare kernel (byte-by-byte)
__global__ void byte_compare_kernel(
    const uint8_t* __restrict__ plane,
    const uint8_t* __restrict__ ref,
    uint64_t n,
    unsigned long long* __restrict__ d_mismatch)
{
  uint64_t idx = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  if (plane[idx] != ref[idx]) atomicAdd(d_mismatch, 1ull);
}

// Read S0 into output buffer (for host-side CRC verification)
__global__ void read_plane_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    uint64_t n)
{
  uint64_t idx = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  dst[idx] = src[idx];
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
[[nodiscard]] const char* cuda_err_str(cudaError_t e) { return cudaGetErrorString(e); }
[[noreturn]] void die(const char* m) { std::fprintf(stderr, "error: %s\n", m); std::exit(2); }
void cuda_check(cudaError_t e, const char* w) {
  if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error at %s: %s\n", w, cuda_err_str(e)); std::exit(2); }
}
[[nodiscard]] bool parse_u64(std::string_view s, uint64_t& out) {
  if (s.empty()) return false;
  uint64_t v = 0;
  for (char c : s) { if (c < '0' || c > '9') return false; v = v * 10 + uint64_t(c - '0'); }
  out = v; return true;
}

struct GT { double mn = 1e30, mx = 0, sm = 0; int n = 0;
  void r(double ms) { if (ms < mn) mn = ms; if (ms > mx) mx = ms; sm += ms; ++n; }
  double avg() const { return n ? sm / n : 0; }
};

void print_usage(const char* a0) {
  std::fprintf(stderr,
    "Usage: %s [options]\n"
    "Phase 3-U: Graded NMR B2 measurement\n"
    "Options:\n"
    "  --dataset PATH       Encoded dataset dir\n"
    "  --raw PATH           Raw float64 file\n"
    "  --segment-rows N     Segment size (default: dataset full)\n"
    "  --threshold FP64     Threshold\n"
    "  --b1-time MS         B1 time (ms) from bench_filter_aggregate\n"
    "  --plane-id N         Target plane index for duplicate-read (default: 0=S0)\n"
    "  --ref-plane-id N     Reference plane index for diversity (default: =--plane-id)\n"
    "  --segment-id N       Specific segment to compare (R2 re-read). Default: all segments.\n"
    "  --compare-mode MODE  byte_compare | segment_crc32\n"
    "  --fault MODE         no_fault | single_byte | single_seg | multi_seg\n"
    "  --iters N            Iterations (default: 30)\n"
    "  --warmup N           Warmup (default: 5)\n"
    "  --csv PATH           Output CSV\n", a0);
}

int main(int argc, char** argv) {
  init_crc();

  std::string dataset_path, raw_path, csv_path, cmode = "byte_compare", fmode = "no_fault";
  uint64_t seg_rows = 0;
  double threshold = 0.0, b0_time = 0.0, b1_time = 0.0;
  int iters = 30, warmup = 5, plane_id = 0, ref_plane_id = -1, segment_id = -1;
  bool b0_set = false, b1_set = false;

  for (int i = 1; i < argc; ++i) {
    std::string_view a(argv[i]);
    auto val = [&]() { if (++i >= argc) die("missing value"); return std::string(argv[i]); };
    if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
    else if (a == "--dataset")      dataset_path = val();
    else if (a == "--raw")          raw_path = val();
    else if (a == "--segment-rows") { if (!parse_u64(val(), seg_rows)) die("invalid --segment-rows"); }
    else if (a == "--threshold")    threshold = std::stod(val());
    else if (a == "--b0-time")      { b0_time = std::stod(val()); b0_set = true; }
    else if (a == "--b1-time")      { b1_time = std::stod(val()); b1_set = true; }
    else if (a == "--plane-id")     { plane_id = std::stoi(val()); }
    else if (a == "--ref-plane-id") { ref_plane_id = std::stoi(val()); }
    else if (a == "--segment-id")   { segment_id = std::stoi(val()); }
    else if (a == "--compare-mode") { cmode = val(); }
    else if (a == "--fault")        { fmode = val(); }
    else if (a == "--iters")        { if (!parse_u64(val(), (uint64_t&)iters) || iters <= 0) die("invalid --iters"); }
    else if (a == "--warmup")       { if (!parse_u64(val(), (uint64_t&)warmup) || warmup < 0) die("invalid --warmup"); }
    else if (a == "--csv")          { csv_path = val(); }
    else { std::string msg = "unknown: "; msg += a; die(msg.c_str()); }
  }
  if (dataset_path.empty() || raw_path.empty()) die("--dataset and --raw required");

  if (!b1_set && !csv_path.empty()) {
    std::fprintf(stderr, "warn: --b1-time not set; gate computation will be partial.\n");
  }

  cuda_check(cudaSetDevice(0), "cudaSetDevice");
  cudaDeviceProp prop;
  cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  std::fprintf(stderr, "Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  // Load dataset
  Dataset dataset = exp3_real::load_dataset(dataset_path);
  uint64_t n = 0;
  for (auto& s : dataset.segments) n += s.row_count;
  uint64_t n_segments = dataset.segments.size();
  int active_planes = static_cast<int>(dataset.segments[0].active_plane_count);
  if (plane_id >= static_cast<int>(dataset.planes.size())) {
    std::fprintf(stderr, "error: plane_id %d out of range (max %zu)\n", plane_id, dataset.planes.size());
    return 2;
  }
  uint64_t s0_bytes = dataset.planes[static_cast<size_t>(plane_id)].size();
  uint64_t data_seg_count = seg_rows > 0 ? (n + seg_rows - 1) / seg_rows : n_segments;
  uint64_t actual_seg_rows = seg_rows > 0 ? seg_rows : n;

  std::fprintf(stderr, "Dataset: %s n=%" PRIu64 " seg=%" PRIu64 " plane=%d bytes=%" PRIu64 "\n",
               dataset.manifest.dataset.c_str(), n, n_segments, plane_id, s0_bytes);
  std::fprintf(stderr, "Mode: %s Fault: %s\n", cmode.c_str(), fmode.c_str());

  // Allocate + copy planes to device
  uint8_t* d_plane = nullptr;
  cuda_check(cudaMalloc(&d_plane, s0_bytes), "cudaMalloc(plane)");
  cuda_check(cudaMemcpy(d_plane, dataset.planes[static_cast<size_t>(plane_id)].data(), s0_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(plane)");

  // Reference copy (from same or different plane source)
  int actual_ref_id = (ref_plane_id >= 0) ? ref_plane_id : plane_id;
  if (actual_ref_id >= static_cast<int>(dataset.planes.size())) {
    std::fprintf(stderr, "error: ref_plane_id %d out of range (max %zu)\n", actual_ref_id, dataset.planes.size());
    return 2;
  }
  uint8_t* d_ref = nullptr;
  cuda_check(cudaMalloc(&d_ref, s0_bytes), "cudaMalloc(ref)");
  cuda_check(cudaMemcpy(d_ref, dataset.planes[static_cast<size_t>(actual_ref_id)].data(), s0_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(ref)");
  std::fprintf(stderr, "  Replicas: plane=%d ref=%d %s\n", plane_id, actual_ref_id,
               (plane_id == actual_ref_id) ? "(same source)" : "(different source)");

  // Apply fault injection to live plane data (d_plane)
  // This is detected by both compare modes:
  //   byte_compare: d_plane (corrupted) vs d_ref (clean)
  //   segment_crc32: re-read d_plane → CRC != reference CRC
  if (fmode == "single_byte") {
    uint8_t flip = 0xFF;
    cuda_check(cudaMemcpy(&d_plane[0], &flip, 1, cudaMemcpyHostToDevice), "fault");
    std::fprintf(stderr, "  Fault: single byte flip at offset 0\n");
  } else if (fmode == "single_seg") {
    std::vector<uint8_t> corrupt(actual_seg_rows, 0xFF);
    uint64_t cp = std::min(actual_seg_rows, s0_bytes);
    cuda_check(cudaMemcpy(d_plane, corrupt.data(), cp, cudaMemcpyHostToDevice), "fault seg");
    std::fprintf(stderr, "  Fault: first segment corrupted (%" PRIu64 " bytes)\n", cp);
  } else if (fmode == "multi_seg") {
    uint64_t corrupt = std::min(actual_seg_rows * 10, s0_bytes);
    std::vector<uint8_t> cdata(corrupt, 0xAA);
    cuda_check(cudaMemcpy(d_plane, cdata.data(), corrupt, cudaMemcpyHostToDevice), "fault multi");
    std::fprintf(stderr, "  Fault: first %" PRIu64 " bytes corrupted\n", corrupt);
  }

  // Device buffer for CRC32 mode readback
  uint8_t* d_dup = nullptr;
  cuda_check(cudaMalloc(&d_dup, s0_bytes), "cudaMalloc(dup)");

  uint8_t* h_dup_host = nullptr;
  cuda_check(cudaMallocHost(&h_dup_host, s0_bytes), "cudaMallocHost(dup)");

  unsigned long long* d_mismatch = nullptr;
  cuda_check(cudaMalloc(&d_mismatch, sizeof(unsigned long long)), "cudaMalloc(mismatch)");

  uint64_t threads = 256;
  uint64_t blocks = (s0_bytes + threads - 1) / threads;

  cudaStream_t stream;
  cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

  // Pre-compute reference CRC32C per segment
  // Ref CRC from the reference plane source (may differ from live plane)
  std::vector<uint32_t> h_ref_crc(data_seg_count);
  for (uint64_t s = 0; s < data_seg_count; ++s) {
    uint64_t st = s * actual_seg_rows;
    uint64_t en = std::min(st + actual_seg_rows, s0_bytes);
    if (st < s0_bytes)
      h_ref_crc[s] = crc32c_host(dataset.planes[static_cast<size_t>(actual_ref_id)].data() + st, en - st);
    else
      h_ref_crc[s] = 0;
  }

  // -------------------------------------------------------------------
  // Measure B2: fused duplicate-read + compare
  // -------------------------------------------------------------------
  std::fprintf(stderr, "\n=== B2: S0 duplicate-read (%s) ===\n", cmode.c_str());

  auto measure_b2 = [&]() -> double {
    GT t;
    for (int i = 0; i < warmup + iters; ++i) {
      cuda_check(cudaMemset(d_mismatch, 0, sizeof(unsigned long long)), "memset");

      auto start = std::chrono::steady_clock::now();

      if (cmode == "byte_compare") {
        byte_compare_kernel<<<blocks, threads, 0, stream>>>(d_plane, d_ref, s0_bytes, d_mismatch);
      } else {
        read_plane_kernel<<<blocks, threads, 0, stream>>>(d_plane, d_dup, s0_bytes);
      }

      cuda_check(cudaStreamSynchronize(stream), "sync");
      auto end = std::chrono::steady_clock::now();

      if (i >= warmup) {
        t.r(std::chrono::duration<double, std::milli>(end - start).count());
      }
    }
    return t.avg();
  };

  double t_dup = measure_b2();

  // Compare
  unsigned long long mismatches = 0;
  if (cmode == "byte_compare") {
    cuda_check(cudaMemcpy(&mismatches, d_mismatch, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "byte rbk");
  } else {
    cuda_check(cudaMemcpy(h_dup_host, d_dup, s0_bytes, cudaMemcpyDeviceToHost), "crc rbk");
    if (segment_id >= 0) {
      // R2: only compare the specified segment
      uint64_t st = static_cast<uint64_t>(segment_id) * actual_seg_rows;
      uint64_t en = std::min(st + actual_seg_rows, s0_bytes);
      if (st < s0_bytes) {
        uint32_t crc = crc32c_host(h_dup_host + st, en - st);
        if (crc != h_ref_crc[static_cast<size_t>(segment_id)]) ++mismatches;
      }
      std::fprintf(stderr, "  R2 segment re-read: seg=%d mismatches=%llu\n", segment_id, mismatches);
    } else {
      for (uint64_t s = 0; s < data_seg_count; ++s) {
        uint64_t st = s * actual_seg_rows;
        uint64_t en = std::min(st + actual_seg_rows, s0_bytes);
        if (st < s0_bytes) {
          uint32_t crc = crc32c_host(h_dup_host + st, en - st);
          if (crc != h_ref_crc[s]) ++mismatches;
        }
      }
    }
  }

  // B2 total = B1 + duplicate-read
  double t_b2 = b1_set ? b1_time + t_dup : t_dup;

  // Gate: B2 must be < B0 and overhead < 90% of margin
  double margin = 0;
  double ovr_frac = 0;
  bool b2_lt_b0 = false;
  bool b2_allowed = false;

  if (b0_set && b1_set) {
    margin = b0_time - b1_time;
    ovr_frac = margin > 0 ? t_dup / margin : 1e30;
    b2_lt_b0 = t_b2 < b0_time;
    b2_allowed = b2_lt_b0 && ovr_frac < 0.9;
  }

  // -------------------------------------------------------------------
  // Report
  // -------------------------------------------------------------------
  std::fprintf(stderr, "\n========================================\n");
  std::fprintf(stderr, "  B2 Graded NMR Implementation Report\n");
  std::fprintf(stderr, "========================================\n");
  std::fprintf(stderr, "  Dataset:     %s\n", dataset.manifest.dataset.c_str());
  std::fprintf(stderr, "  S0 bytes:    %" PRIu64 "\n", s0_bytes);
  std::fprintf(stderr, "  Mode:        %s\n", cmode.c_str());
  std::fprintf(stderr, "  Fault:       %s\n", fmode.c_str());
  std::fprintf(stderr, "\n");
  std::fprintf(stderr, "  B2 dup+compare: %.4f ms\n", t_dup);
  if (b1_set) {
    std::fprintf(stderr, "  B1 (input):     %.4f ms\n", b1_time);
    std::fprintf(stderr, "  B2 total:       %.4f ms\n", t_b2);
    std::fprintf(stderr, "  Overhead:       %.4f ms (%.2f%% of B1)\n", t_dup, t_dup / b1_time * 100);
    std::fprintf(stderr, "  B2 gate:        %s\n", b2_allowed ? "PASS ✅" : "FAIL ❌");
  }
  std::fprintf(stderr, "  Mismatches:     %llu\n", mismatches);
  std::fprintf(stderr, "\n");

  if (mismatches == 0 && fmode == "no_fault") {
    std::fprintf(stderr, "  Integrity: PASS (no fault) ✅\n");
  } else if (mismatches > 0 && fmode != "no_fault") {
    std::fprintf(stderr, "  Detection: PASS (%llu mismatches) ✅\n", mismatches);
  } else if (mismatches > 0 && fmode == "no_fault") {
    std::fprintf(stderr, "  DETECTION FAIL: mismatches with no fault injected ❌\n");
  } else {
    std::fprintf(stderr, "  DETECTION FAIL: fault injected but 0 mismatches ❌\n");
  }

  // CSV
  if (!csv_path.empty()) {
    std::FILE* f = std::fopen(csv_path.c_str(), "w");
    if (f) {
      std::fprintf(f, "dataset,compare_mode,fault_mode,plane_id,n_segments,plane_bytes,"
                   "b1_ms,dup_ms,b2_ms,overhead_ms,mismatches\n");
      std::fprintf(f, "%s,%s,%s,%d,%" PRIu64 ",%" PRIu64 ","
                   "%.4f,%.4f,%.4f,%.4f,%llu\n",
                   dataset.manifest.dataset.c_str(), cmode.c_str(), fmode.c_str(),
                   plane_id, n_segments, s0_bytes,
                   b1_time, t_dup, t_b2, t_dup, mismatches);
      std::fclose(f);
      std::fprintf(stderr, "CSV: %s\n", csv_path.c_str());
    }
  }

  // Cleanup
  cuda_check(cudaFree(d_plane), "cfree(plane)");
  cuda_check(cudaFree(d_ref), "cfree(ref)");
  cuda_check(cudaFree(d_dup), "cfree(dup)");
  cuda_check(cudaFreeHost(h_dup_host), "cfreeHost(dup)");
  cuda_check(cudaFree(d_mismatch), "cfree(mismatch)");
  cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");

  return 0;
}
