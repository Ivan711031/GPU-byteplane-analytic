#include <cuda_runtime.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

namespace
{

enum class ContiguousVariant
{
  Scalar,
  Vec2,
};

[[nodiscard]] const char *cuda_err_str(cudaError_t err) { return cudaGetErrorString(err); }

[[noreturn]] void die(const char *msg)
{
  std::fprintf(stderr, "error: %s\n", msg);
  std::exit(2);
}

void cuda_check(cudaError_t err, const char *where)
{
  if (err == cudaSuccess)
    return;
  std::fprintf(stderr, "cuda error at %s: %s\n", where, cuda_err_str(err));
  std::exit(2);
}

[[nodiscard]] bool parse_u64(std::string_view s, uint64_t &out)
{
  if (s.empty())
    return false;
  uint64_t value = 0;
  for (char c : s)
  {
    if (c < '0' || c > '9')
      return false;
    uint64_t digit = static_cast<uint64_t>(c - '0');
    if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10ull)
      return false;
    value = value * 10ull + digit;
  }
  out = value;
  return true;
}

[[nodiscard]] bool parse_contiguous_variant(std::string_view s, ContiguousVariant &out)
{
  if (s == "scalar")
  {
    out = ContiguousVariant::Scalar;
    return true;
  }
  if (s == "vec2")
  {
    out = ContiguousVariant::Vec2;
    return true;
  }
  return false;
}

[[nodiscard]] const char *contiguous_variant_name(ContiguousVariant variant)
{
  switch (variant)
  {
  case ContiguousVariant::Scalar:
    return "scalar";
  case ContiguousVariant::Vec2:
    return "vec2";
  }
  return "scalar";
}

struct Options
{
  int device = 0;
  ContiguousVariant variant = ContiguousVariant::Scalar;
  uint64_t n = 100000000ull;
  int block_threads = 256;
  int grid_mul = 1;
  int warmup = 10;
  int iters = 200;
  std::string csv_path = "exp1_contiguous_baseline.csv";
};

void print_usage(const char *argv0)
{
  std::fprintf(stderr,
               "Usage: %s [options]\n"
               "\n"
               "Experiment 1 contiguous baseline: scalar or vec2 contiguous FP64 load path.\n"
               "\n"
               "Options:\n"
               "  --device N     CUDA device index (default: 0)\n"
               "  --variant NAME contiguous64 variant: scalar | vec2 (default: scalar)\n"
               "  --n N          Number of 8-byte elements (default: 100000000)\n"
               "  --block T      Threads per block (default: 256)\n"
               "  --grid_mul M   Grid = SMs * maxActiveBlocksPerSM * M (default: 1)\n"
               "  --warmup N     Warmup iterations (default: 10)\n"
               "  --iters N      Timed iterations (default: 200)\n"
               "  --csv PATH     Output CSV path (default: exp1_contiguous_baseline.csv)\n"
               "\n",
               argv0);
}

Options parse_args(int argc, char **argv)
{
  Options opt;
  for (int i = 1; i < argc; i++)
  {
    std::string_view a(argv[i]);
    auto need_value = [&](std::string_view flag) -> std::string_view
    {
      if (i + 1 >= argc)
      {
        std::string msg = "missing value for ";
        msg += flag;
        die(msg.c_str());
      }
      return std::string_view(argv[++i]);
    };

    if (a == "--help" || a == "-h")
    {
      print_usage(argv[0]);
      std::exit(0);
    }
    else if (a == "--variant")
    {
      ContiguousVariant variant = ContiguousVariant::Scalar;
      if (!parse_contiguous_variant(need_value(a), variant))
        die("invalid --variant");
      opt.variant = variant;
    }
    else if (a == "--device")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        die("invalid --device");
      opt.device = static_cast<int>(v);
    }
    else if (a == "--n")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0)
        die("invalid --n");
      opt.n = v;
    }
    else if (a == "--block")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > 2048)
        die("invalid --block");
      opt.block_threads = static_cast<int>(v);
    }
    else if (a == "--grid_mul")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > 1024)
        die("invalid --grid_mul");
      opt.grid_mul = static_cast<int>(v);
    }
    else if (a == "--warmup")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v > 1000000)
        die("invalid --warmup");
      opt.warmup = static_cast<int>(v);
    }
    else if (a == "--iters")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > 100000000)
        die("invalid --iters");
      opt.iters = static_cast<int>(v);
    }
    else if (a == "--csv")
    {
      opt.csv_path = std::string(need_value(a));
    }
    else
    {
      std::string msg = "unknown arg: ";
      msg += std::string(a);
      die(msg.c_str());
    }
  }

  if (opt.block_threads % 32 != 0)
    die("--block must be a multiple of 32");
  return opt;
}

__device__ __forceinline__ unsigned long long warp_reduce_sum_ull(unsigned long long v)
{
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

__device__ __forceinline__ void block_reduce_store(unsigned long long v,
                                                   unsigned long long *__restrict__ out)
{
  v = warp_reduce_sum_ull(v);
  __shared__ unsigned long long warp_sums[32]; // up to 1024 threads
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0)
    warp_sums[warp] = v;
  __syncthreads();

  if (warp == 0)
  {
    unsigned long long block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0ull;
    block_sum = warp_reduce_sum_ull(block_sum);
    if (lane == 0)
      out[blockIdx.x] = block_sum;
  }
}

__global__ void scan_contiguous_u64(const uint64_t *__restrict__ data,
                                    uint64_t n,
                                    unsigned long long *__restrict__ per_block_out)
{
  unsigned long long sum = 0;
  uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                 static_cast<uint64_t>(threadIdx.x);
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
  for (uint64_t i = tid; i < n; i += stride)
    sum += static_cast<unsigned long long>(data[i]);
  block_reduce_store(sum, per_block_out);
}

__global__ void scan_contiguous_u64_vec2(const uint64_t *__restrict__ data,
                                         uint64_t n,
                                         unsigned long long *__restrict__ per_block_out)
{
  unsigned long long sum = 0;
  uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                 static_cast<uint64_t>(threadIdx.x);
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
  uint64_t pair_count = n / 2ull;
  const ulonglong2 *__restrict__ data2 = reinterpret_cast<const ulonglong2 *>(data);

  for (uint64_t i = tid; i < pair_count; i += stride)
  {
    ulonglong2 v = data2[i];
    sum += static_cast<unsigned long long>(v.x);
    sum += static_cast<unsigned long long>(v.y);
  }

  if ((n & 1ull) && tid == 0)
    sum += static_cast<unsigned long long>(data[n - 1]);

  block_reduce_store(sum, per_block_out);
}

[[nodiscard]] const void *contiguous_kernel_ptr(ContiguousVariant variant)
{
  switch (variant)
  {
  case ContiguousVariant::Scalar:
    return reinterpret_cast<const void *>(scan_contiguous_u64);
  case ContiguousVariant::Vec2:
    return reinterpret_cast<const void *>(scan_contiguous_u64_vec2);
  }
  return reinterpret_cast<const void *>(scan_contiguous_u64);
}

int occupancy_grid(int device, int block_threads, int grid_mul, const void *kernel, size_t shmem)
{
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

  int max_active = 0;
  cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active, kernel, block_threads, shmem),
             "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  long long grid = static_cast<long long>(prop.multiProcessorCount) * max_active * grid_mul;
  if (grid < 1)
    grid = 1;
  if (grid > std::numeric_limits<int>::max())
    grid = std::numeric_limits<int>::max();
  return static_cast<int>(grid);
}

struct RunResult
{
  ContiguousVariant variant = ContiguousVariant::Scalar;
  uint64_t n = 0;
  int block = 0;
  int grid = 0;
  int warmup = 0;
  int iters = 0;
  float ms_per_iter = 0.0f;
  double logical_GBps = 0.0;
};

void launch_contiguous_kernel(ContiguousVariant variant,
                              int grid,
                              int block_threads,
                              const uint64_t *d_data,
                              uint64_t n,
                              unsigned long long *d_out)
{
  switch (variant)
  {
  case ContiguousVariant::Scalar:
    scan_contiguous_u64<<<grid, block_threads>>>(d_data, n, d_out);
    return;
  case ContiguousVariant::Vec2:
    scan_contiguous_u64_vec2<<<grid, block_threads>>>(d_data, n, d_out);
    return;
  }
  std::abort();
}

RunResult run_one(const Options &opt, const uint64_t *d_data, unsigned long long *d_out, int grid)
{
  cuda_check(cudaSetDevice(opt.device), "cudaSetDevice");

  for (int i = 0; i < opt.warmup; i++)
    launch_contiguous_kernel(opt.variant, grid, opt.block_threads, d_data, opt.n, d_out);
  cuda_check(cudaGetLastError(), "warmup launch");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
  for (int i = 0; i < opt.iters; i++)
    launch_contiguous_kernel(opt.variant, grid, opt.block_threads, d_data, opt.n, d_out);
  cuda_check(cudaGetLastError(), "timed launch");
  cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
  cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

  float ms_total = 0.0f;
  cuda_check(cudaEventElapsedTime(&ms_total, start, stop), "cudaEventElapsedTime");
  cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
  cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

  float ms_per_iter = ms_total / static_cast<float>(opt.iters);
  double seconds = static_cast<double>(ms_per_iter) / 1000.0;

  double logical_bytes = static_cast<double>(opt.n) * 8.0;
  double logical_GBps = (logical_bytes / seconds) / 1e9;

  RunResult r{};
  r.variant = opt.variant;
  r.n = opt.n;
  r.block = opt.block_threads;
  r.grid = grid;
  r.warmup = opt.warmup;
  r.iters = opt.iters;
  r.ms_per_iter = ms_per_iter;
  r.logical_GBps = logical_GBps;
  return r;
}

std::string csv_escape(std::string_view s)
{
  bool needs_quotes = false;
  for (char c : s)
  {
    if (c == ',' || c == '"' || c == '\n' || c == '\r')
    {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes)
    return std::string(s);
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('"');
  for (char c : s)
  {
    if (c == '"')
      out.push_back('"');
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

} // namespace

int main(int argc, char **argv)
{
  Options opt = parse_args(argc, argv);

  cuda_check(cudaSetDevice(opt.device), "cudaSetDevice");
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, opt.device), "cudaGetDeviceProperties");

  uint64_t *d_data = nullptr;
  cuda_check(cudaMalloc(&d_data, static_cast<size_t>(opt.n) * sizeof(uint64_t)), "cudaMalloc(d_data)");
  cuda_check(cudaMemset(d_data, 0xAB, static_cast<size_t>(opt.n) * sizeof(uint64_t)), "cudaMemset(d_data)");

  int grid = occupancy_grid(opt.device, opt.block_threads, opt.grid_mul,
                            contiguous_kernel_ptr(opt.variant), 0);

  unsigned long long *d_out = nullptr;
  cuda_check(cudaMalloc(&d_out, static_cast<size_t>(grid) * sizeof(unsigned long long)), "cudaMalloc(d_out)");
  cuda_check(cudaMemset(d_out, 0, static_cast<size_t>(grid) * sizeof(unsigned long long)), "cudaMemset(d_out)");

  RunResult r = run_one(opt, d_data, d_out, grid);

  std::FILE *f = std::fopen(opt.csv_path.c_str(), "wb");
  if (!f)
    die("failed to open --csv output path");
  std::fprintf(f,
               "benchmark,variant,n,logical_bytes,physical_bytes_per_row,block,grid,warmup,iters,ms_per_iter,logical_GBps,device,sm,cc_major,cc_minor\n");

  uint64_t logical_bytes = static_cast<uint64_t>(r.n) * 8ull;
  std::string device_name = csv_escape(prop.name);
  std::fprintf(f,
               "%s,%s,%" PRIu64 ",%" PRIu64 ",%d,%d,%d,%d,%d,%.6f,%.3f,%s,%d,%d,%d\n",
               "contiguous64",
               contiguous_variant_name(r.variant),
               static_cast<uint64_t>(r.n),
               logical_bytes,
               8,
               r.block,
               r.grid,
               r.warmup,
               r.iters,
               r.ms_per_iter,
               r.logical_GBps,
               device_name.c_str(),
               prop.multiProcessorCount,
               prop.major,
               prop.minor);
  std::fflush(f);
  std::fclose(f);

  std::fprintf(stderr,
               "[exp1-baseline] benchmark=contiguous64 variant=%s n=%" PRIu64 " ms=%.3f logical_GB/s=%.1f\n",
               contiguous_variant_name(r.variant),
               static_cast<uint64_t>(r.n),
               r.ms_per_iter,
               r.logical_GBps);

  cuda_check(cudaFree(d_out), "cudaFree(d_out)");
  cuda_check(cudaFree(d_data), "cudaFree(d_data)");
  return 0;
}
