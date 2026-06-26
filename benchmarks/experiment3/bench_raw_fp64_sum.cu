#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace
{

[[nodiscard]] const char *cuda_err_str(cudaError_t err) { return cudaGetErrorString(err); }

[[noreturn]] void die(const char *msg)
{
  std::fprintf(stderr, "error: %s\n", msg);
  std::exit(2);
}

[[noreturn]] void die(const std::string &msg)
{
  die(msg.c_str());
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

[[nodiscard]] std::string csv_escape(std::string_view s)
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

[[nodiscard]] std::filesystem::path derive_dataset_name(const std::filesystem::path &path)
{
  return path.filename();
}

[[nodiscard]] std::vector<double> read_raw_fp64(const std::filesystem::path &path)
{
  std::uintmax_t file_size = std::filesystem::file_size(path);
  if (file_size == 0)
    die("input file is empty");
  if ((file_size % sizeof(double)) != 0)
    die("input file size is not a multiple of 8 bytes");

  std::vector<double> values(static_cast<std::size_t>(file_size / sizeof(double)));
  std::ifstream input(path, std::ios::binary);
  if (!input)
    die("failed to open raw input file");
  input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(file_size));
  if (!input)
    die("failed to read raw input file");
  return values;
}

__device__ __forceinline__ double warp_reduce_sum(double v)
{
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

__device__ __forceinline__ void block_reduce_store(double v, double *__restrict__ out)
{
  v = warp_reduce_sum(v);
  __shared__ double warp_sums[64]; // up to 2048 threads
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0)
    warp_sums[warp] = v;
  __syncthreads();

  if (warp == 0)
  {
    double block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0)
      out[blockIdx.x] = block_sum;
  }
}

__global__ void sum_raw_fp64_vec2(const double *__restrict__ data,
                                  std::uint64_t n,
                                  double *__restrict__ per_block_out)
{
  double sum = 0.0;
  std::uint64_t tid = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) +
                      static_cast<std::uint64_t>(threadIdx.x);
  std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * static_cast<std::uint64_t>(blockDim.x);
  std::uint64_t pair_count = n / 2ull;
  const double2 *__restrict__ data2 = reinterpret_cast<const double2 *>(data);

  for (std::uint64_t i = tid; i < pair_count; i += stride)
  {
    double2 v = data2[i];
    sum += v.x + v.y;
  }

  if ((n & 1ull) && tid == 0)
    sum += data[n - 1];

  block_reduce_store(sum, per_block_out);
}

[[nodiscard]] int occupancy_grid(int device, int block_threads, int grid_mul, const void *kernel)
{
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

  int max_active = 0;
  cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active, kernel, block_threads, 0),
             "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  long long grid = static_cast<long long>(prop.multiProcessorCount) * max_active * grid_mul;
  if (grid < 1)
    grid = 1;
  if (grid > std::numeric_limits<int>::max())
    grid = std::numeric_limits<int>::max();
  return static_cast<int>(grid);
}

struct Options
{
  int device = 0;
  std::filesystem::path input_path;
  std::string dataset;
  int block_threads = 256;
  int grid_mul = 1;
  int warmup = 10;
  int iters = 200;
  std::string csv_path = "raw_fp64_sum_baseline.csv";
  bool validate = false;
};

struct RunResult
{
  std::uint64_t n = 0;
  int block = 0;
  int grid = 0;
  int warmup = 0;
  int iters = 0;
  float ms_per_iter = 0.0f;
  double rows_per_sec = 0.0;
  double logical_GBps = 0.0;
  double estimated_physical_GBps = 0.0;
  double cpu_sum = 0.0;
  double gpu_sum = 0.0;
  double abs_error = 0.0;
  double rel_error = 0.0;
  bool validated = false;
};

void print_usage(const char *argv0)
{
  std::fprintf(stderr,
               "Usage: %s --input PATH [options]\n"
               "\n"
               "Raw FP64 SUM comparator for Exp3 synthetic dev datasets.\n"
               "\n"
               "Options:\n"
               "  --input PATH     Raw .f64le.bin input file\n"
               "  --dataset NAME   Dataset label for CSV output (default: derived from input name)\n"
               "  --device N       CUDA device index (default: 0)\n"
               "  --block T        Threads per block (default: 256)\n"
               "  --grid_mul M     Grid = SMs * maxActiveBlocksPerSM * M (default: 1)\n"
               "  --warmup N       Warmup iterations (default: 10)\n"
               "  --iters N        Timed iterations (default: 200)\n"
               "  --validate       Enable CPU-vs-GPU validation\n"
               "  --csv PATH       Output CSV path (default: raw_fp64_sum_baseline.csv)\n",
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
        die(msg);
      }
      return std::string_view(argv[++i]);
    };

    if (a == "--help" || a == "-h")
    {
      print_usage(argv[0]);
      std::exit(0);
    }
    else if (a == "--input")
    {
      opt.input_path = std::filesystem::path(need_value(a));
    }
    else if (a == "--dataset")
    {
      opt.dataset = std::string(need_value(a));
    }
    else if (a == "--device")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        die("invalid --device");
      opt.device = static_cast<int>(v);
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
    else if (a == "--validate")
    {
      opt.validate = true;
    }
    else
    {
      std::string msg = "unknown arg: ";
      msg += std::string(a);
      die(msg);
    }
  }

  if (opt.input_path.empty())
    die("--input is required");
  if (opt.block_threads % 32 != 0)
    die("--block must be a multiple of 32");
  if (opt.dataset.empty())
    opt.dataset = derive_dataset_name(opt.input_path).string();
  return opt;
}

[[nodiscard]] double collect_partial_sum(double *d_partial_out, int grid)
{
  std::vector<double> h_partial(static_cast<std::size_t>(grid));
  cuda_check(cudaMemcpy(h_partial.data(),
                        d_partial_out,
                        static_cast<std::size_t>(grid) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(d_partial_out)");
  double observed = 0.0;
  for (double v : h_partial)
    observed += v;
  return observed;
}

RunResult run_one(const Options &opt, const std::vector<double> &h_data, double *d_data, double *d_partial_out, int grid)
{
  cuda_check(cudaSetDevice(opt.device), "cudaSetDevice");

  for (int i = 0; i < opt.warmup; ++i)
    sum_raw_fp64_vec2<<<grid, opt.block_threads>>>(d_data, static_cast<std::uint64_t>(h_data.size()), d_partial_out);
  cuda_check(cudaGetLastError(), "warmup launch");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
  for (int i = 0; i < opt.iters; ++i)
    sum_raw_fp64_vec2<<<grid, opt.block_threads>>>(d_data, static_cast<std::uint64_t>(h_data.size()), d_partial_out);
  cuda_check(cudaGetLastError(), "timed launch");
  cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
  cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

  float ms_total = 0.0f;
  cuda_check(cudaEventElapsedTime(&ms_total, start, stop), "cudaEventElapsedTime");
  cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
  cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

  float ms_per_iter = ms_total / static_cast<float>(opt.iters);
  double seconds = static_cast<double>(ms_per_iter) / 1000.0;
  std::uint64_t n = static_cast<std::uint64_t>(h_data.size());
  double logical_bytes = static_cast<double>(n) * 8.0;
  double rows_per_sec = static_cast<double>(n) / seconds;
  double logical_GBps = (logical_bytes / seconds) / 1e9;

  long double cpu_sum_ld = 0.0L;
  for (double v : h_data)
    cpu_sum_ld += static_cast<long double>(v);
  double cpu_sum = static_cast<double>(cpu_sum_ld);

  cuda_check(cudaMemset(d_partial_out, 0, static_cast<std::size_t>(grid) * sizeof(double)),
             "cudaMemset(d_partial_out)");
  sum_raw_fp64_vec2<<<grid, opt.block_threads>>>(d_data, n, d_partial_out);
  cuda_check(cudaGetLastError(), "observe launch");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(observe)");
  double gpu_sum = collect_partial_sum(d_partial_out, grid);

  double abs_error = std::fabs(cpu_sum - gpu_sum);
  double rel_error = abs_error / std::fmax(std::fabs(cpu_sum), 1.0);
  double tolerance = std::fmax(1e-12 * std::fmax(std::fmax(std::fabs(cpu_sum), std::fabs(gpu_sum)), 1.0), 1e-9);
  bool validated = abs_error <= tolerance;
  if (opt.validate && !validated)
  {
    std::fprintf(stderr,
                 "validation failed: cpu=%.17g gpu=%.17g abs=%.17g rel=%.17g tol=%.17g\n",
                 cpu_sum,
                 gpu_sum,
                 abs_error,
                 rel_error,
                 tolerance);
    std::exit(2);
  }

  RunResult r{};
  r.n = n;
  r.block = opt.block_threads;
  r.grid = grid;
  r.warmup = opt.warmup;
  r.iters = opt.iters;
  r.ms_per_iter = ms_per_iter;
  r.rows_per_sec = rows_per_sec;
  r.logical_GBps = logical_GBps;
  r.estimated_physical_GBps = logical_GBps;
  r.cpu_sum = cpu_sum;
  r.gpu_sum = gpu_sum;
  r.abs_error = abs_error;
  r.rel_error = rel_error;
  r.validated = validated;
  return r;
}

} // namespace

int main(int argc, char **argv)
{
  Options opt = parse_args(argc, argv);
  if (!std::filesystem::exists(opt.input_path))
    die("input file does not exist");

  std::vector<double> h_data = read_raw_fp64(opt.input_path);

  cuda_check(cudaSetDevice(opt.device), "cudaSetDevice");
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, opt.device), "cudaGetDeviceProperties");

  double *d_data = nullptr;
  cuda_check(cudaMalloc(&d_data, h_data.size() * sizeof(double)), "cudaMalloc(d_data)");
  cuda_check(cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(double), cudaMemcpyHostToDevice),
             "cudaMemcpy(d_data)");

  int grid = occupancy_grid(opt.device, opt.block_threads, opt.grid_mul, reinterpret_cast<const void *>(sum_raw_fp64_vec2));
  double *d_partial_out = nullptr;
  cuda_check(cudaMalloc(&d_partial_out, static_cast<std::size_t>(grid) * sizeof(double)), "cudaMalloc(d_partial_out)");
  cuda_check(cudaMemset(d_partial_out, 0, static_cast<std::size_t>(grid) * sizeof(double)),
             "cudaMemset(d_partial_out)");

  RunResult r = run_one(opt, h_data, d_data, d_partial_out, grid);

  std::FILE *f = std::fopen(opt.csv_path.c_str(), "wb");
  if (!f)
    die("failed to open --csv output path");

  std::fprintf(f,
               "benchmark,dataset,raw_path,aggregation,baseline_type,n,logical_bytes,ms_per_iter,rows_per_sec,"
               "logical_GBps,estimated_physical_GBps,cpu_sum,gpu_sum,abs_error,rel_error,validated,device,job_id,kernel_path,block,grid,warmup,iters,sm,cc_major,cc_minor\n");

  const char *job_id = std::getenv("SLURM_JOB_ID");
  std::string job_id_str = job_id ? job_id : "nojob";
  std::string raw_path = csv_escape(opt.input_path.string());
  std::string dataset = csv_escape(opt.dataset);
  std::string device_name = csv_escape(prop.name);
  std::string kernel_path = "raw_fp64_sum_vec2";

  std::uint64_t n = r.n;
  std::uint64_t logical_bytes = n * 8ull;
  std::fprintf(f,
               "raw_fp64_sum,%s,%s,sum,raw_fp64_sum,%" PRIu64 ",%" PRIu64 ",%.6f,%.17g,%.6f,%.6f,%.17g,%.17g,%.17g,%.17g,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d\n",
               dataset.c_str(),
               raw_path.c_str(),
               n,
               logical_bytes,
               r.ms_per_iter,
               r.rows_per_sec,
               r.logical_GBps,
               r.estimated_physical_GBps,
               r.cpu_sum,
               r.gpu_sum,
               r.abs_error,
               r.rel_error,
               r.validated ? "true" : "false",
               device_name.c_str(),
               job_id_str.c_str(),
               kernel_path.c_str(),
               r.block,
               r.grid,
               r.warmup,
               r.iters,
               prop.multiProcessorCount,
               prop.major,
               prop.minor);
  std::fflush(f);
  std::fclose(f);

  std::fprintf(stderr,
               "[raw-fp64-sum] dataset=%s n=%" PRIu64 " ms=%.6f logical_GB/s=%.3f validated=%s\n",
               opt.dataset.c_str(),
               n,
               r.ms_per_iter,
               r.logical_GBps,
               r.validated ? "true" : "false");

  cuda_check(cudaFree(d_partial_out), "cudaFree(d_partial_out)");
  cuda_check(cudaFree(d_data), "cudaFree(d_data)");
  return 0;
}
