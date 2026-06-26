#include <cuda_runtime.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr size_t kMiB = 1024ull * 1024ull;
constexpr size_t kGiB = 1024ull * 1024ull * 1024ull;

[[nodiscard]] const char* cuda_err_str(cudaError_t err) { return cudaGetErrorString(err); }

[[noreturn]] void die(const char* msg) {
  std::fprintf(stderr, "error: %s\n", msg);
  std::exit(2);
}

void cuda_check(cudaError_t err, const char* where) {
  if (err == cudaSuccess) return;
  std::fprintf(stderr, "cuda error at %s: %s\n", where, cuda_err_str(err));
  std::exit(2);
}

[[nodiscard]] bool starts_with(std::string_view s, std::string_view prefix) {
  return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

[[nodiscard]] bool parse_u64(std::string_view s, uint64_t& out) {
  if (s.empty()) return false;
  uint64_t value = 0;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
    uint64_t digit = static_cast<uint64_t>(c - '0');
    if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10ull) return false;
    value = value * 10ull + digit;
  }
  out = value;
  return true;
}

[[nodiscard]] bool parse_bytes(std::string_view s, size_t& out_bytes) {
  // Accept: plain integer bytes, or suffix: K/KB, M/MB, G/GB (base-2: KiB/MiB/GiB).
  if (s.empty()) return false;

  uint64_t multiplier = 1;
  std::string_view number = s;
  if (s.size() >= 2) {
    std::string_view suffix2 = s.substr(s.size() - 2);
    if (suffix2 == "KB" || suffix2 == "kb") {
      multiplier = 1024ull;
      number = s.substr(0, s.size() - 2);
    } else if (suffix2 == "MB" || suffix2 == "mb") {
      multiplier = 1024ull * 1024ull;
      number = s.substr(0, s.size() - 2);
    } else if (suffix2 == "GB" || suffix2 == "gb") {
      multiplier = 1024ull * 1024ull * 1024ull;
      number = s.substr(0, s.size() - 2);
    }
  }
  if (number == s && s.size() >= 1) {
    char last = s.back();
    if (last == 'K' || last == 'k') {
      multiplier = 1024ull;
      number = s.substr(0, s.size() - 1);
    } else if (last == 'M' || last == 'm') {
      multiplier = 1024ull * 1024ull;
      number = s.substr(0, s.size() - 1);
    } else if (last == 'G' || last == 'g') {
      multiplier = 1024ull * 1024ull * 1024ull;
      number = s.substr(0, s.size() - 1);
    }
  }

  uint64_t base = 0;
  if (!parse_u64(number, base)) return false;
  if (base > std::numeric_limits<uint64_t>::max() / multiplier) return false;
  uint64_t bytes_u64 = base * multiplier;
  if (bytes_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return false;
  out_bytes = static_cast<size_t>(bytes_u64);
  return true;
}

struct Options {
  size_t bytes_min = 1 * kMiB;
  size_t bytes_max = 8 * kGiB;
  uint64_t bytes_mult = 2;
  bool single_size = false;
  size_t bytes_single = 0;

  int device = 0;
  int block_threads = 256;
  int grid_mul = 1;  // multiplier over max occupancy grid
  int warmup = 10;
  int iters = 200;
  std::string mode = "seq";
  std::string csv_path = "exp0_hbm_bw.csv";

  int mask_stride = 2;  // active lanes per warp pattern
  int mask_active = 1;
  size_t gather_span_bytes = 0;  // 0 means full range
  uint32_t gather_seed = 1;
};

void print_usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s [options]\n"
               "\n"
               "Experiment 0: HBM bandwidth microbenchmark (dummy reduction).\n"
               "Measures effective bandwidth for coalesced sequential reads using cudaEvent timing.\n"
               "\n"
               "Options:\n"
               "  --device N            CUDA device index (default: 0)\n"
               "  --mode NAME           Benchmark mode: seq, masked, gather (default: seq)\n"
               "  --bytes B             Single size (e.g., 1048576, 1MB, 8GB)\n"
               "  --bytes_min B         Sweep min (default: 1MB)\n"
               "  --bytes_max B         Sweep max (default: 8GB)\n"
               "  --bytes_mult M        Sweep multiplier (default: 2)\n"
               "  --block T             Threads per block (default: 256)\n"
               "  --grid_mul M          Grid = SMs * maxActiveBlocksPerSM * M (default: 1)\n"
               "  --warmup N            Warmup iterations (default: 10)\n"
               "  --iters N             Timed iterations (default: 200)\n"
               "  --mask_stride N       Masked mode: lane pattern stride (1-32, default: 2)\n"
               "  --mask_active N       Masked mode: active lanes per stride (default: 1)\n"
               "  --gather_span B       Gather mode: index span bytes (0=full, default: 0)\n"
               "  --gather_seed N       Gather mode: index seed (default: 1)\n"
               "  --csv PATH            Output CSV path (default: exp0_hbm_bw.csv)\n"
               "\n",
               argv0);
}

Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    std::string_view a(argv[i]);
    auto need_value = [&](std::string_view flag) -> std::string_view {
      if (i + 1 >= argc) {
        std::string msg = "missing value for ";
        msg += flag;
        die(msg.c_str());
      }
      return std::string_view(argv[++i]);
    };

    if (a == "--help" || a == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (a == "--device") {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        die("invalid --device");
      opt.device = static_cast<int>(v);
    } else if (a == "--mode") {
      opt.mode = std::string(need_value(a));
    } else if (a == "--csv") {
      opt.csv_path = std::string(need_value(a));
    } else if (a == "--bytes") {
      size_t b = 0;
      if (!parse_bytes(need_value(a), b)) die("invalid --bytes");
      opt.single_size = true;
      opt.bytes_single = b;
    } else if (a == "--bytes_min") {
      size_t b = 0;
      if (!parse_bytes(need_value(a), b)) die("invalid --bytes_min");
      opt.bytes_min = b;
    } else if (a == "--bytes_max") {
      size_t b = 0;
      if (!parse_bytes(need_value(a), b)) die("invalid --bytes_max");
      opt.bytes_max = b;
    } else if (a == "--bytes_mult") {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v < 2) die("invalid --bytes_mult (need >= 2)");
      opt.bytes_mult = v;
    } else if (a == "--block") {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > 2048) die("invalid --block");
      opt.block_threads = static_cast<int>(v);
    } else if (a == "--grid_mul") {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > 1024) die("invalid --grid_mul");
      opt.grid_mul = static_cast<int>(v);
    } else if (a == "--warmup") {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v > 1000000) die("invalid --warmup");
      opt.warmup = static_cast<int>(v);
    } else if (a == "--iters") {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > 100000000) die("invalid --iters");
      opt.iters = static_cast<int>(v);
    } else if (a == "--mask_stride") {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > 32) die("invalid --mask_stride");
      opt.mask_stride = static_cast<int>(v);
    } else if (a == "--mask_active") {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v > 32) die("invalid --mask_active");
      opt.mask_active = static_cast<int>(v);
    } else if (a == "--gather_span") {
      size_t b = 0;
      if (!parse_bytes(need_value(a), b)) die("invalid --gather_span");
      opt.gather_span_bytes = b;
    } else if (a == "--gather_seed") {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v > std::numeric_limits<uint32_t>::max())
        die("invalid --gather_seed");
      opt.gather_seed = static_cast<uint32_t>(v);
    } else {
      std::string msg = "unknown arg: ";
      msg += std::string(a);
      die(msg.c_str());
    }
  }

  if (!opt.single_size && opt.bytes_min > opt.bytes_max) die("--bytes_min > --bytes_max");
  if (opt.single_size && opt.bytes_single == 0) die("--bytes must be > 0");
  if (opt.block_threads % 32 != 0) die("--block must be a multiple of 32");
  if (opt.mask_active < 0 || opt.mask_active > opt.mask_stride) die("--mask_active must be <= --mask_stride");

  return opt;
}

__device__ __forceinline__ unsigned long long warp_reduce_sum_ull(unsigned long long v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__global__ void seq_read_dummy_reduce(const uint4* __restrict__ data4,
                                      size_t n4,
                                      unsigned long long* __restrict__ per_block_out) {
  unsigned long long sum = 0;
  size_t tid = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
               static_cast<size_t>(threadIdx.x);
  size_t stride = static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);

  for (size_t i = tid; i < n4; i += stride) {
    uint4 v = data4[i];
    sum += static_cast<unsigned long long>(v.x);
    sum += static_cast<unsigned long long>(v.y);
    sum += static_cast<unsigned long long>(v.z);
    sum += static_cast<unsigned long long>(v.w);
  }

  sum = warp_reduce_sum_ull(sum);
  __shared__ unsigned long long warp_sums[32];  // up to 1024 threads/block
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();

  if (warp == 0) {
    unsigned long long block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0ull;
    block_sum = warp_reduce_sum_ull(block_sum);
    if (lane == 0) per_block_out[blockIdx.x] = block_sum;
  }
}

__global__ void masked_read_dummy_reduce(const uint4* __restrict__ data4,
                                         size_t n4,
                                         unsigned long long* __restrict__ per_block_out,
                                         int mask_stride,
                                         int mask_active) {
  unsigned long long sum = 0;
  size_t tid = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
               static_cast<size_t>(threadIdx.x);
  size_t stride = static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);
  int lane = threadIdx.x & 31;
  bool active = (mask_stride <= 1) ? true : ((lane % mask_stride) < mask_active);

  if (active) {
    for (size_t i = tid; i < n4; i += stride) {
      uint4 v = data4[i];
      sum += static_cast<unsigned long long>(v.x);
      sum += static_cast<unsigned long long>(v.y);
      sum += static_cast<unsigned long long>(v.z);
      sum += static_cast<unsigned long long>(v.w);
    }
  }

  sum = warp_reduce_sum_ull(sum);
  __shared__ unsigned long long warp_sums[32];
  int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();

  if (warp == 0) {
    unsigned long long block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0ull;
    block_sum = warp_reduce_sum_ull(block_sum);
    if (lane == 0) per_block_out[blockIdx.x] = block_sum;
  }
}

__global__ void init_gather_indices(uint32_t* __restrict__ indices,
                                    size_t n,
                                    uint32_t span,
                                    uint32_t seed) {
  size_t tid = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
               static_cast<size_t>(threadIdx.x);
  size_t stride = static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);
  if (span == 0) return;
  for (size_t i = tid; i < n; i += stride) {
    uint32_t x = static_cast<uint32_t>(i) ^ seed;
    x = x * 1664525u + 1013904223u;
    indices[i] = x % span;
  }
}

__global__ void gather_read_dummy_reduce(const uint4* __restrict__ data4,
                                         const uint32_t* __restrict__ indices,
                                         size_t n4,
                                         unsigned long long* __restrict__ per_block_out) {
  unsigned long long sum = 0;
  size_t tid = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
               static_cast<size_t>(threadIdx.x);
  size_t stride = static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);

  for (size_t i = tid; i < n4; i += stride) {
    uint32_t idx = indices[i];
    uint4 v = data4[idx];
    sum += static_cast<unsigned long long>(v.x);
    sum += static_cast<unsigned long long>(v.y);
    sum += static_cast<unsigned long long>(v.z);
    sum += static_cast<unsigned long long>(v.w);
  }

  sum = warp_reduce_sum_ull(sum);
  __shared__ unsigned long long warp_sums[32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();

  if (warp == 0) {
    unsigned long long block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0ull;
    block_sum = warp_reduce_sum_ull(block_sum);
    if (lane == 0) per_block_out[blockIdx.x] = block_sum;
  }
}

struct RunResult {
  size_t bytes = 0;
  double effective_bytes = 0.0;
  int block = 0;
  int grid = 0;
  int warmup = 0;
  int iters = 0;
  float ms_per_iter = 0.0f;
  double gb_per_s = 0.0;
  double active_frac = 1.0;
  size_t gather_span_bytes = 0;
};

RunResult run_seq(const Options& opt, size_t bytes) {
  if (bytes < sizeof(uint4)) bytes = sizeof(uint4);
  size_t n4 = bytes / sizeof(uint4);
  size_t bytes_used = n4 * sizeof(uint4);

  cuda_check(cudaSetDevice(opt.device), "cudaSetDevice");

  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, opt.device), "cudaGetDeviceProperties");

  int max_active = 0;
  cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &max_active, seq_read_dummy_reduce, opt.block_threads, 0),
             "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  int grid = prop.multiProcessorCount * max_active * opt.grid_mul;
  if (grid < 1) grid = 1;

  uint4* d_in = nullptr;
  unsigned long long* d_out = nullptr;

  cuda_check(cudaMalloc(&d_in, bytes_used), "cudaMalloc(d_in)");
  cuda_check(cudaMalloc(&d_out, static_cast<size_t>(grid) * sizeof(unsigned long long)),
             "cudaMalloc(d_out)");
  cuda_check(cudaMemset(d_in, 0xAB, bytes_used), "cudaMemset(d_in)");
  cuda_check(cudaMemset(d_out, 0, static_cast<size_t>(grid) * sizeof(unsigned long long)),
             "cudaMemset(d_out)");

  // Warmup.
  for (int i = 0; i < opt.warmup; i++) {
    seq_read_dummy_reduce<<<grid, opt.block_threads>>>(d_in, n4, d_out);
  }
  cuda_check(cudaGetLastError(), "warmup launch");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
  for (int i = 0; i < opt.iters; i++) {
    seq_read_dummy_reduce<<<grid, opt.block_threads>>>(d_in, n4, d_out);
  }
  cuda_check(cudaGetLastError(), "timed launch");
  cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
  cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

  float ms_total = 0.0f;
  cuda_check(cudaEventElapsedTime(&ms_total, start, stop), "cudaEventElapsedTime");

  cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
  cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

  cuda_check(cudaFree(d_in), "cudaFree(d_in)");
  cuda_check(cudaFree(d_out), "cudaFree(d_out)");

  float ms_per_iter = ms_total / static_cast<float>(opt.iters);
  double seconds = (static_cast<double>(ms_per_iter) / 1000.0);
  double gb_s = (static_cast<double>(bytes_used) / seconds) / 1e9;

  RunResult r{};
  r.bytes = bytes_used;
  r.effective_bytes = static_cast<double>(bytes_used);
  r.block = opt.block_threads;
  r.grid = grid;
  r.warmup = opt.warmup;
  r.iters = opt.iters;
  r.ms_per_iter = ms_per_iter;
  r.gb_per_s = gb_s;
  r.active_frac = 1.0;
  r.gather_span_bytes = 0;
  return r;
}

double masked_active_fraction(int mask_stride, int mask_active) {
  if (mask_stride <= 1) return 1.0;
  if (mask_active <= 0) return 0.0;
  int full = 32 / mask_stride;
  int rem = 32 % mask_stride;
  int active = full * mask_active + ((rem < mask_active) ? rem : mask_active);
  if (active < 0) active = 0;
  if (active > 32) active = 32;
  return static_cast<double>(active) / 32.0;
}

RunResult run_masked(const Options& opt, size_t bytes) {
  if (bytes < sizeof(uint4)) bytes = sizeof(uint4);
  size_t n4 = bytes / sizeof(uint4);
  size_t bytes_used = n4 * sizeof(uint4);

  cuda_check(cudaSetDevice(opt.device), "cudaSetDevice");

  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, opt.device), "cudaGetDeviceProperties");

  int max_active = 0;
  cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &max_active, masked_read_dummy_reduce, opt.block_threads, 0),
             "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  int grid = prop.multiProcessorCount * max_active * opt.grid_mul;
  if (grid < 1) grid = 1;

  uint4* d_in = nullptr;
  unsigned long long* d_out = nullptr;

  cuda_check(cudaMalloc(&d_in, bytes_used), "cudaMalloc(d_in)");
  cuda_check(cudaMalloc(&d_out, static_cast<size_t>(grid) * sizeof(unsigned long long)),
             "cudaMalloc(d_out)");
  cuda_check(cudaMemset(d_in, 0xAB, bytes_used), "cudaMemset(d_in)");
  cuda_check(cudaMemset(d_out, 0, static_cast<size_t>(grid) * sizeof(unsigned long long)),
             "cudaMemset(d_out)");

  for (int i = 0; i < opt.warmup; i++) {
    masked_read_dummy_reduce<<<grid, opt.block_threads>>>(
        d_in, n4, d_out, opt.mask_stride, opt.mask_active);
  }
  cuda_check(cudaGetLastError(), "warmup launch");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
  for (int i = 0; i < opt.iters; i++) {
    masked_read_dummy_reduce<<<grid, opt.block_threads>>>(
        d_in, n4, d_out, opt.mask_stride, opt.mask_active);
  }
  cuda_check(cudaGetLastError(), "timed launch");
  cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
  cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

  float ms_total = 0.0f;
  cuda_check(cudaEventElapsedTime(&ms_total, start, stop), "cudaEventElapsedTime");

  cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
  cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

  cuda_check(cudaFree(d_in), "cudaFree(d_in)");
  cuda_check(cudaFree(d_out), "cudaFree(d_out)");

  double active_frac = masked_active_fraction(opt.mask_stride, opt.mask_active);
  double effective_bytes = static_cast<double>(bytes_used) * active_frac;
  float ms_per_iter = ms_total / static_cast<float>(opt.iters);
  double seconds = (static_cast<double>(ms_per_iter) / 1000.0);
  double gb_s = (effective_bytes / seconds) / 1e9;

  RunResult r{};
  r.bytes = bytes_used;
  r.effective_bytes = effective_bytes;
  r.block = opt.block_threads;
  r.grid = grid;
  r.warmup = opt.warmup;
  r.iters = opt.iters;
  r.ms_per_iter = ms_per_iter;
  r.gb_per_s = gb_s;
  r.active_frac = active_frac;
  r.gather_span_bytes = 0;
  return r;
}

RunResult run_gather(const Options& opt, size_t bytes) {
  if (bytes < sizeof(uint4)) bytes = sizeof(uint4);
  size_t n4 = bytes / sizeof(uint4);
  size_t bytes_used = n4 * sizeof(uint4);

  if (n4 > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    die("gather mode requires <= 4B elements (uint32 indices)");
  }

  size_t span_bytes = opt.gather_span_bytes == 0 ? bytes_used : opt.gather_span_bytes;
  if (span_bytes > bytes_used) span_bytes = bytes_used;
  size_t span_elems = span_bytes / sizeof(uint4);
  if (span_elems == 0) span_elems = 1;

  cuda_check(cudaSetDevice(opt.device), "cudaSetDevice");

  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, opt.device), "cudaGetDeviceProperties");

  int max_active = 0;
  cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &max_active, gather_read_dummy_reduce, opt.block_threads, 0),
             "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  int grid = prop.multiProcessorCount * max_active * opt.grid_mul;
  if (grid < 1) grid = 1;

  uint4* d_in = nullptr;
  uint32_t* d_idx = nullptr;
  unsigned long long* d_out = nullptr;

  cuda_check(cudaMalloc(&d_in, bytes_used), "cudaMalloc(d_in)");
  cuda_check(cudaMalloc(&d_idx, n4 * sizeof(uint32_t)), "cudaMalloc(d_idx)");
  cuda_check(cudaMalloc(&d_out, static_cast<size_t>(grid) * sizeof(unsigned long long)),
             "cudaMalloc(d_out)");
  cuda_check(cudaMemset(d_in, 0xAB, bytes_used), "cudaMemset(d_in)");
  cuda_check(cudaMemset(d_out, 0, static_cast<size_t>(grid) * sizeof(unsigned long long)),
             "cudaMemset(d_out)");

  int init_block = 256;
  int init_grid = static_cast<int>((n4 + init_block - 1) / init_block);
  if (init_grid < 1) init_grid = 1;
  init_gather_indices<<<init_grid, init_block>>>(
      d_idx, n4, static_cast<uint32_t>(span_elems), opt.gather_seed);
  cuda_check(cudaGetLastError(), "init_gather_indices launch");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init_gather_indices)");

  for (int i = 0; i < opt.warmup; i++) {
    gather_read_dummy_reduce<<<grid, opt.block_threads>>>(d_in, d_idx, n4, d_out);
  }
  cuda_check(cudaGetLastError(), "warmup launch");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
  for (int i = 0; i < opt.iters; i++) {
    gather_read_dummy_reduce<<<grid, opt.block_threads>>>(d_in, d_idx, n4, d_out);
  }
  cuda_check(cudaGetLastError(), "timed launch");
  cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
  cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

  float ms_total = 0.0f;
  cuda_check(cudaEventElapsedTime(&ms_total, start, stop), "cudaEventElapsedTime");

  cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
  cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

  cuda_check(cudaFree(d_in), "cudaFree(d_in)");
  cuda_check(cudaFree(d_idx), "cudaFree(d_idx)");
  cuda_check(cudaFree(d_out), "cudaFree(d_out)");

  float ms_per_iter = ms_total / static_cast<float>(opt.iters);
  double seconds = (static_cast<double>(ms_per_iter) / 1000.0);
  double gb_s = (static_cast<double>(bytes_used) / seconds) / 1e9;

  RunResult r{};
  r.bytes = bytes_used;
  r.effective_bytes = static_cast<double>(bytes_used);
  r.block = opt.block_threads;
  r.grid = grid;
  r.warmup = opt.warmup;
  r.iters = opt.iters;
  r.ms_per_iter = ms_per_iter;
  r.gb_per_s = gb_s;
  r.active_frac = 1.0;
  r.gather_span_bytes = span_elems * sizeof(uint4);
  return r;
}

std::vector<size_t> make_sweep(const Options& opt) {
  std::vector<size_t> sizes;
  if (opt.single_size) {
    sizes.push_back(opt.bytes_single);
    return sizes;
  }
  if (opt.bytes_min == 0) die("--bytes_min must be > 0");
  if (opt.bytes_mult < 2) die("--bytes_mult must be >= 2");

  for (size_t b = opt.bytes_min; b <= opt.bytes_max;) {
    sizes.push_back(b);
    uint64_t next = static_cast<uint64_t>(b) * opt.bytes_mult;
    if (next > static_cast<uint64_t>(opt.bytes_max)) break;
    b = static_cast<size_t>(next);
    if (b == 0) break;
  }
  if (sizes.empty()) die("empty sweep");
  return sizes;
}

std::string csv_escape(std::string_view s) {
  bool needs_quotes = false;
  for (char c : s) {
    if (c == ',' || c == '"' || c == '\n' || c == '\r') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes) return std::string(s);
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('"');
  for (char c : s) {
    if (c == '"') out.push_back('"');
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt = parse_args(argc, argv);
  if (opt.mode != "seq" && opt.mode != "masked" && opt.mode != "gather") {
    die("unknown --mode (expected: seq, masked, gather)");
  }

  cuda_check(cudaSetDevice(opt.device), "cudaSetDevice");
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, opt.device), "cudaGetDeviceProperties");

  std::FILE* f = std::fopen(opt.csv_path.c_str(), "wb");
  if (!f) die("failed to open --csv output path");

  std::fprintf(
      f,
      "mode,bytes,effective_bytes,block,grid,warmup,iters,ms_per_iter,GBps,active_frac,"
      "gather_span_bytes,device,sm,cc_major,cc_minor\n");

  auto sweep = make_sweep(opt);
  for (size_t bytes : sweep) {
    RunResult r{};
    if (opt.mode == "seq") {
      r = run_seq(opt, bytes);
    } else if (opt.mode == "masked") {
      r = run_masked(opt, bytes);
    } else {
      r = run_gather(opt, bytes);
    }
    std::fprintf(f,
                 "%s,%" PRIu64 ",%.0f,%d,%d,%d,%d,%.6f,%.3f,%.3f,%" PRIu64 ",%s,%d,%d,%d\n",
                 opt.mode.c_str(),
                 static_cast<uint64_t>(r.bytes),
                 r.effective_bytes,
                 r.block,
                 r.grid,
                 r.warmup,
                 r.iters,
                 r.ms_per_iter,
                 r.gb_per_s,
                 r.active_frac,
                 static_cast<uint64_t>(r.gather_span_bytes),
                 csv_escape(prop.name).c_str(),
                 prop.multiProcessorCount,
                 prop.major,
                 prop.minor);
    std::fflush(f);

    std::fprintf(stderr,
                 "[exp0] mode=%s bytes=%zu block=%d grid=%d ms=%.3f GB/s=%.1f\n",
                 opt.mode.c_str(),
                 r.bytes,
                 r.block,
                 r.grid,
                 r.ms_per_iter,
                 r.gb_per_s);
  }

  std::fclose(f);
  return 0;
}
