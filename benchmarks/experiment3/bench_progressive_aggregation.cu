#include <cuda_runtime.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "exp3_kernels_progressive.cuh"
#include "exp3_kernels_extrema.cuh"
#include "exp3_real_data_layout.hpp"

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

struct Options
{
  int device = 0;
  uint64_t n = 100000000ull;
  uint64_t segment_rows = 1048576ull;
  int subcolumns = EXP3_MAX_SUBCOLUMNS;
  bool n_set = false;
  bool segment_rows_set = false;
  bool subcolumns_set = false;
  int refine_min = 0;
  int refine_max = 7;
  bool refine_min_set = false;
  bool refine_max_set = false;
  std::string mode = "synthetic_fixed_point_subcolumns";
  std::string encoded_root;
  std::string load_strategy = "rowpack16";
  std::string real_kernel_variant = "runtime";
  int block_threads = 256;
  int items_per_thread = 1;
  int warmup = 10;
  int iters = 200;
  std::string csv_path = "exp3_progressive_aggregation.csv";
  bool validate = false;
  std::string aggregation = "sum"; // "sum" or "minmax"
};

struct Derived
{
  uint64_t tile_rows = 0;
  uint64_t tiles_per_segment = 0;
  uint64_t num_segments = 0;
  uint64_t grid_u64 = 0;
  int grid = 0;
};

struct RunResult
{
  int refinement_depth = 0;
  float ms_per_iter = 0.0f;
  double rows_per_sec = 0.0;
  double billion_rows_per_sec = 0.0;
  double logical_GBps = 0.0;
  std::string kernel_path = "synthetic_template";
  double gpu_approximate_sum = std::numeric_limits<double>::quiet_NaN();
  double cpu_approximate_sum = std::numeric_limits<double>::quiet_NaN();
  double exact_sum = std::numeric_limits<double>::quiet_NaN();
};

struct MinMaxRunResult
{
  int refinement_depth = 0;
  float ms_per_iter = 0.0f;
  double rows_per_sec = 0.0;
  double billion_rows_per_sec = 0.0;
  double logical_GBps = 0.0;
  std::string kernel_path = "runtime_rowpack16_minmax";
  double min_lower = std::numeric_limits<double>::quiet_NaN();
  double min_upper = std::numeric_limits<double>::quiet_NaN();
  double max_lower = std::numeric_limits<double>::quiet_NaN();
  double max_upper = std::numeric_limits<double>::quiet_NaN();
  double exact_min = std::numeric_limits<double>::quiet_NaN();
  double exact_max = std::numeric_limits<double>::quiet_NaN();
  double cpu_min_lower = std::numeric_limits<double>::quiet_NaN();
  double cpu_min_upper = std::numeric_limits<double>::quiet_NaN();
  double cpu_max_lower = std::numeric_limits<double>::quiet_NaN();
  double cpu_max_upper = std::numeric_limits<double>::quiet_NaN();
  uint64_t min_candidates = 0;
  uint64_t max_candidates = 0;
  uint64_t total_rows = 0;
};

struct CsvContext
{
  std::string dataset = "synthetic";
  std::string mode = "synthetic_fixed_point_subcolumns";
  std::string basis_mode = "synthetic_pow2";
  std::string error_bound_source = "synthetic_fixed_point";
  std::string encoded_layout;
  uint32_t fractional_bits = 0;
  uint64_t max_plane_count = 0;
  uint64_t segment_plane_count_min = 0;
  uint64_t segment_plane_count_max = 0;
};

struct SyntheticContext
{
  std::vector<uint8_t *> d_subcolumn_storage;
  Exp3U8Subcolumns h_subcolumns{};
  std::vector<double> h_segment_base;
  std::vector<double> h_subcolumn_basis;
  double *d_segment_base = nullptr;
  double *d_subcolumn_basis = nullptr;
  double *d_partial_out = nullptr;
};

struct RealContext
{
  exp3_real::Dataset dataset;
  std::vector<uint8_t *> d_subcolumn_storage;
  Exp3RuntimeU8Subcolumns h_subcolumns{};
  std::vector<double> h_segment_base;
  std::vector<double> h_subcolumn_basis;
  double *d_segment_base = nullptr;
  double *d_subcolumn_basis = nullptr;
  double *d_partial_out = nullptr;
  double *d_extrema_out = nullptr; // 2 * grid doubles for min/max prefix per block
};

void print_usage(const char *argv0)
{
  std::fprintf(stderr,
               "Usage: %s [options]\n"
               "\n"
               "Experiment 3: synthetic or encoded-dev progressive SUM aggregation.\n"
               "\n"
               "Options:\n"
               "  --device N              CUDA device index (default: 0)\n"
               "  --mode NAME             synthetic_fixed_point_subcolumns | encoded_dev_subcolumns\n"
               "  --encoded-root PATH     Encoded dataset root for --mode encoded_dev_subcolumns\n"
               "  --n N                   Logical row count (synthetic default: 100000000)\n"
               "  --segment_rows N        Rows per metadata segment (synthetic default: 1048576)\n"
               "  --subcolumns N          Encoded byte subcolumns (synthetic default: 8)\n"
               "  --refine_min N          Minimum refinement depth (default: 0)\n"
               "  --refine_max N          Maximum refinement depth (synthetic default: 7)\n"
               "  --load_strategy NAME    rowpack16 only in v1 (default: rowpack16)\n"
               "  --real_kernel_variant NAME  runtime | specialized for encoded-dev mode (default: runtime)\n"
               "  --block T               Threads per block, multiple of 32 (default: 256)\n"
               "  --items_per_thread N    Only 1 supported in v1 (default: 1)\n"
               "  --warmup N              Warmup iterations (default: 10)\n"
               "  --iters N               Timed iterations (default: 200)\n"
"  --validate              Run non-timed validation per depth\n"
                "  --csv PATH              Output CSV path\n"
                "  --aggregation NAME      sum | minmax (default: sum)\n",
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
    else if (a == "--device")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        die("invalid --device");
      opt.device = static_cast<int>(v);
    }
    else if (a == "--mode")
    {
      opt.mode = std::string(need_value(a));
    }
    else if (a == "--encoded-root")
    {
      opt.encoded_root = std::string(need_value(a));
    }
    else if (a == "--n")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0)
        die("invalid --n");
      opt.n = v;
      opt.n_set = true;
    }
    else if (a == "--segment_rows")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0)
        die("invalid --segment_rows");
      opt.segment_rows = v;
      opt.segment_rows_set = true;
    }
    else if (a == "--subcolumns")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > static_cast<uint64_t>(EXP3_RUNTIME_MAX_SUBCOLUMNS))
        die("invalid --subcolumns");
      opt.subcolumns = static_cast<int>(v);
      opt.subcolumns_set = true;
    }
    else if (a == "--refine_min")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v >= static_cast<uint64_t>(EXP3_RUNTIME_MAX_SUBCOLUMNS))
        die("invalid --refine_min");
      opt.refine_min = static_cast<int>(v);
      opt.refine_min_set = true;
    }
    else if (a == "--refine_max")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v >= static_cast<uint64_t>(EXP3_RUNTIME_MAX_SUBCOLUMNS))
        die("invalid --refine_max");
      opt.refine_max = static_cast<int>(v);
      opt.refine_max_set = true;
    }
    else if (a == "--load_strategy")
    {
      opt.load_strategy = std::string(need_value(a));
    }
    else if (a == "--real_kernel_variant")
    {
      opt.real_kernel_variant = std::string(need_value(a));
    }
    else if (a == "--block")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > 1024)
        die("invalid --block");
      opt.block_threads = static_cast<int>(v);
    }
    else if (a == "--items_per_thread")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > 64)
        die("invalid --items_per_thread");
      opt.items_per_thread = static_cast<int>(v);
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
    else if (a == "--aggregation")
    {
      opt.aggregation = std::string(need_value(a));
      if (opt.aggregation != "sum" && opt.aggregation != "minmax")
        die("invalid --aggregation: must be 'sum' or 'minmax'");
    }
    else
    {
      std::string msg = "unknown arg: ";
      msg += std::string(a);
      die(msg);
    }
  }

  if (opt.block_threads % 32 != 0)
    die("--block must be a multiple of 32");
  if (opt.items_per_thread != 1)
    die("--items_per_thread must be 1 in v1");
  if (opt.load_strategy != "rowpack16")
    die("only --load_strategy rowpack16 is supported in v1");
  if (opt.real_kernel_variant != "runtime" && opt.real_kernel_variant != "specialized")
    die("invalid --real_kernel_variant");
  if (opt.mode != "synthetic_fixed_point_subcolumns" && opt.mode != "encoded_dev_subcolumns")
    die("invalid --mode");
  if (opt.mode == "encoded_dev_subcolumns" && opt.encoded_root.empty())
    die("--encoded-root is required for --mode encoded_dev_subcolumns");
  return opt;
}

void finalize_refinement_bounds(Options &opt)
{
  if (opt.mode == "synthetic_fixed_point_subcolumns" && opt.subcolumns > EXP3_MAX_SUBCOLUMNS)
    die("synthetic mode supports at most 8 subcolumns");
  if (!opt.refine_max_set)
    opt.refine_max = opt.subcolumns - 1;
  if (!opt.refine_min_set)
    opt.refine_min = 0;
  if (opt.refine_min > opt.refine_max)
    die("--refine_min > --refine_max");
  if (opt.refine_max >= opt.subcolumns)
    die("--refine_max must be < --subcolumns");
  if (opt.real_kernel_variant == "specialized" && opt.refine_max >= 16)
    die("--real_kernel_variant specialized supports refinement depths 0..15");
}

[[nodiscard]] uint64_t checked_mul_u64(uint64_t a, uint64_t b, const char *what)
{
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
  {
    std::string msg = "overflow while computing ";
    msg += what;
    die(msg);
  }
  return a * b;
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

[[nodiscard]] Derived compute_derived(const Options &opt)
{
  Derived d{};
  d.tile_rows = checked_mul_u64(static_cast<uint64_t>(opt.block_threads),
                                checked_mul_u64(static_cast<uint64_t>(opt.items_per_thread),
                                                static_cast<uint64_t>(EXP3_ROWPACK16_WIDTH),
                                                "items_per_thread * pack_width"),
                                "tile_rows");
  d.tiles_per_segment = exp3_ceil_div_u64(opt.segment_rows, d.tile_rows);
  d.num_segments = exp3_ceil_div_u64(opt.n, opt.segment_rows);
  d.grid_u64 = checked_mul_u64(d.num_segments, d.tiles_per_segment, "grid");
  if (d.grid_u64 == 0 || d.grid_u64 > static_cast<uint64_t>(std::numeric_limits<int>::max()))
    die("grid does not fit in CUDA launch int range");
  d.grid = static_cast<int>(d.grid_u64);
  return d;
}

[[nodiscard]] double basis_value(int p)
{
  return std::ldexp(1.0, -8 * p);
}

[[nodiscard]] double expected_sum(uint64_t n, int depth)
{
  double expected = 0.0;
  for (int p = 0; p <= depth; ++p)
    expected += static_cast<double>(n) * static_cast<double>(p + 1) * basis_value(p);
  return expected;
}

[[nodiscard]] double synthetic_avg_abs_error_bound(int depth, int subcolumns)
{
  double bound = 0.0;
  for (int p = depth + 1; p < subcolumns; ++p)
    bound += 255.0 * basis_value(p);
  return bound;
}

[[nodiscard]] double real_validation_tolerance(double cpu_sum, double gpu_sum)
{
  // Real-mode validation only checks GPU-vs-CPU agreement for the same
  // approximate SUM. Use a tight absolute floor with a relative guard so large
  // FP64 sums do not false-fail on harmless reduction-order noise.
  double magnitude = std::fmax(std::fabs(cpu_sum), std::fabs(gpu_sum));
  double rel_tolerance = 1e-12 * std::fmax(magnitude, 1.0);
  double abs_tolerance = 1e-9;
  return std::fmax(rel_tolerance, abs_tolerance);
}

[[nodiscard]] double collect_partial_sum(double *d_partial_out, int grid)
{
  std::vector<double> h_partial(static_cast<size_t>(grid));
  cuda_check(cudaMemcpy(h_partial.data(),
                        d_partial_out,
                        static_cast<size_t>(grid) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(d_partial_out)");
  double observed = 0.0;
  for (double v : h_partial)
    observed += v;
  return observed;
}

[[nodiscard]] bool use_specialized_real_kernel(const Options &opt, int max_planes)
{
  if (opt.mode != "encoded_dev_subcolumns")
    return false;
  if (opt.real_kernel_variant != "specialized")
    return false;
  return max_planes >= 1 && max_planes <= 16;
}

[[nodiscard]] const char *select_real_kernel_path(const Options &opt, int max_planes)
{
  return use_specialized_real_kernel(opt, max_planes) ? "specialized_rowpack16" : "runtime_rowpack16";
}

void validate_synthetic(int refinement_depth,
                        const Options &opt,
                        const Derived &derived,
                        const SyntheticContext &ctx)
{
  cuda_check(cudaMemset(ctx.d_partial_out, 0, static_cast<size_t>(derived.grid) * sizeof(double)),
             "cudaMemset(d_partial_out validate synthetic)");
  launch_progressive_sum_rowpack16(refinement_depth,
                                   derived.grid,
                                   opt.block_threads,
                                   ctx.h_subcolumns,
                                   opt.n,
                                   opt.segment_rows,
                                   derived.tiles_per_segment,
                                   ctx.d_segment_base,
                                   ctx.d_subcolumn_basis,
                                   ctx.d_partial_out);
  cuda_check(cudaGetLastError(), "validation launch synthetic");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(validation synthetic)");

  double observed = collect_partial_sum(ctx.d_partial_out, derived.grid);
  double expected = expected_sum(opt.n, refinement_depth);
  double abs_err = std::fabs(observed - expected);
  double tolerance = std::fmax(1e-6 * std::fabs(expected), 1e-3);
  if (abs_err > tolerance)
  {
    char msg[512];
    std::snprintf(msg,
                  sizeof(msg),
                  "validation failed for depth=%d: observed=%.17g expected=%.17g abs_err=%.17g tolerance=%.17g",
                  refinement_depth,
                  observed,
                  expected,
                  abs_err,
                  tolerance);
    die(msg);
  }

  std::fprintf(stderr,
               "[exp3] synthetic validate ok depth=%d observed=%.17g expected=%.17g abs_err=%.3g\n",
               refinement_depth,
               observed,
               expected,
               abs_err);
}

void launch_real_progressive_sum(int refinement_depth,
                                 const Options &opt,
                                 const Derived &derived,
                                 const RealContext &ctx)
{
  if (use_specialized_real_kernel(opt, static_cast<int>(ctx.dataset.manifest.max_plane_count)))
  {
    launch_progressive_sum_real_rowpack16_specialized(refinement_depth,
                                                      derived.grid,
                                                      opt.block_threads,
                                                      ctx.h_subcolumns,
                                                      opt.subcolumns,
                                                      opt.n,
                                                      opt.segment_rows,
                                                      derived.tiles_per_segment,
                                                      ctx.d_segment_base,
                                                      ctx.d_subcolumn_basis,
                                                      ctx.d_partial_out);
    return;
  }

  launch_progressive_sum_rowpack16_runtime(refinement_depth,
                                           derived.grid,
                                           opt.block_threads,
                                           ctx.h_subcolumns,
                                           opt.subcolumns,
                                           opt.n,
                                           opt.segment_rows,
                                           derived.tiles_per_segment,
                                           ctx.d_segment_base,
                                           ctx.d_subcolumn_basis,
                                           ctx.d_partial_out);
}

void validate_real(int refinement_depth,
                   const Options &opt,
                   const Derived &derived,
                   const RealContext &ctx)
{
  cuda_check(cudaMemset(ctx.d_partial_out, 0, static_cast<size_t>(derived.grid) * sizeof(double)),
             "cudaMemset(d_partial_out validate real)");
  launch_real_progressive_sum(refinement_depth, opt, derived, ctx);
  cuda_check(cudaGetLastError(), "validation launch real");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(validation real)");

  double gpu_sum = collect_partial_sum(ctx.d_partial_out, derived.grid);
  double cpu_sum = exp3_real::cpu_approximate_sum(ctx.dataset, refinement_depth);
  double abs_diff = std::fabs(cpu_sum - gpu_sum);
  double tolerance = real_validation_tolerance(cpu_sum, gpu_sum);
  if (abs_diff > tolerance)
  {
    char msg[768];
    std::snprintf(msg,
                  sizeof(msg),
                  "validation failed for depth=%d: gpu=%.17g cpu=%.17g abs_cpu_gpu=%.17g tolerance=%.17g (rel=%.17g abs=%.17g)",
                  refinement_depth,
                  gpu_sum,
                  cpu_sum,
                  abs_diff,
                  tolerance,
                  1e-12 * std::fmax(std::fmax(std::fabs(cpu_sum), std::fabs(gpu_sum)), 1.0),
                  1e-9);
    die(msg);
  }
  std::fprintf(stderr,
               "[exp3] real validate ok depth=%d gpu=%.17g cpu=%.17g abs_cpu_gpu=%.17g tolerance=%.17g abs_exact_gpu=%.17g abs_exact_cpu=%.17g\n",
               refinement_depth,
               gpu_sum,
               cpu_sum,
               abs_diff,
               tolerance,
               std::fabs(ctx.dataset.manifest.exact_sum - gpu_sum),
               std::fabs(ctx.dataset.manifest.exact_sum - cpu_sum));
}

RunResult run_one_synthetic(int refinement_depth,
                            const Options &opt,
                            const Derived &derived,
                            const SyntheticContext &ctx)
{
  for (int i = 0; i < opt.warmup; ++i)
  {
    launch_progressive_sum_rowpack16(refinement_depth,
                                     derived.grid,
                                     opt.block_threads,
                                     ctx.h_subcolumns,
                                     opt.n,
                                     opt.segment_rows,
                                     derived.tiles_per_segment,
                                     ctx.d_segment_base,
                                     ctx.d_subcolumn_basis,
                                     ctx.d_partial_out);
  }
  cuda_check(cudaGetLastError(), "warmup launch synthetic");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup synthetic)");

  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
  for (int i = 0; i < opt.iters; ++i)
  {
    launch_progressive_sum_rowpack16(refinement_depth,
                                     derived.grid,
                                     opt.block_threads,
                                     ctx.h_subcolumns,
                                     opt.n,
                                     opt.segment_rows,
                                     derived.tiles_per_segment,
                                     ctx.d_segment_base,
                                     ctx.d_subcolumn_basis,
                                     ctx.d_partial_out);
  }
  cuda_check(cudaGetLastError(), "timed launch synthetic");
  cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
  cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

  float ms_total = 0.0f;
  cuda_check(cudaEventElapsedTime(&ms_total, start, stop), "cudaEventElapsedTime");
  cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
  cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

  float ms_per_iter = ms_total / static_cast<float>(opt.iters);
  double seconds = static_cast<double>(ms_per_iter) / 1000.0;
  double logical_bytes = static_cast<double>(opt.n) * static_cast<double>(refinement_depth + 1);

  RunResult r{};
  r.refinement_depth = refinement_depth;
  r.ms_per_iter = ms_per_iter;
  r.rows_per_sec = static_cast<double>(opt.n) / seconds;
  r.billion_rows_per_sec = r.rows_per_sec / 1e9;
  r.logical_GBps = (logical_bytes / seconds) / 1e9;
  r.kernel_path = "synthetic_template";
  return r;
}

RunResult run_one_real(int refinement_depth,
                       const Options &opt,
                       const Derived &derived,
                       const RealContext &ctx)
{
  const char *kernel_path = select_real_kernel_path(opt, static_cast<int>(ctx.dataset.manifest.max_plane_count));
  for (int i = 0; i < opt.warmup; ++i)
  {
    launch_real_progressive_sum(refinement_depth, opt, derived, ctx);
  }
  cuda_check(cudaGetLastError(), "warmup launch real");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup real)");

  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
  for (int i = 0; i < opt.iters; ++i)
  {
    launch_real_progressive_sum(refinement_depth, opt, derived, ctx);
  }
  cuda_check(cudaGetLastError(), "timed launch real");
  cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
  cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

  float ms_total = 0.0f;
  cuda_check(cudaEventElapsedTime(&ms_total, start, stop), "cudaEventElapsedTime");
  cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
  cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

  float ms_per_iter = ms_total / static_cast<float>(opt.iters);
  double seconds = static_cast<double>(ms_per_iter) / 1000.0;
  double logical_bytes = static_cast<double>(opt.n) * static_cast<double>(refinement_depth + 1);

  RunResult r{};
  r.refinement_depth = refinement_depth;
  r.ms_per_iter = ms_per_iter;
  r.rows_per_sec = static_cast<double>(opt.n) / seconds;
  r.billion_rows_per_sec = r.rows_per_sec / 1e9;
  r.logical_GBps = (logical_bytes / seconds) / 1e9;
  r.kernel_path = kernel_path;

  cuda_check(cudaMemset(ctx.d_partial_out, 0, static_cast<size_t>(derived.grid) * sizeof(double)),
             "cudaMemset(d_partial_out observe real)");
  launch_real_progressive_sum(refinement_depth, opt, derived, ctx);
  cuda_check(cudaGetLastError(), "observe launch real");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(observe real)");
  r.gpu_approximate_sum = collect_partial_sum(ctx.d_partial_out, derived.grid);
  r.exact_sum = ctx.dataset.manifest.exact_sum;
  if (opt.validate)
    r.cpu_approximate_sum = exp3_real::cpu_approximate_sum(ctx.dataset, refinement_depth);
  return r;
}

void write_csv_header(std::FILE *f)
{
  std::fprintf(f,
               "benchmark,dataset,mode,aggregation,load_strategy,refinement_depth,"
               "kernel_path,"
               "n,segment_rows,tile_rows,subcolumns,subcolumn_bits,"
               "logical_subcolumns_read,logical_bytes,block,grid,warmup,iters,"
               "ms_per_iter,rows_per_sec,billion_rows_per_sec,logical_GBps,"
               "accumulator_bits,base_value,basis_mode,"
               "synthetic_sum_abs_error_bound,synthetic_avg_abs_error_bound,error_bound_source,"
               "validated,device,sm,cc_major,cc_minor,"
               "encoded_layout,max_plane_count,segment_plane_count_min,segment_plane_count_max,"
               "exact_sum,cpu_approximate_sum,gpu_approximate_sum,abs_cpu_gpu_diff,abs_exact_cpu_diff,abs_exact_gpu_diff\n");
}

void write_csv_row(std::FILE *f,
                   const Options &opt,
                   const Derived &derived,
                   const RunResult &r,
                   const CsvContext &ctx,
                   const cudaDeviceProp &prop)
{
  int logical_subcolumns_read = r.refinement_depth + 1;
  double logical_bytes = static_cast<double>(opt.n) * static_cast<double>(logical_subcolumns_read);
  double avg_bound = std::numeric_limits<double>::quiet_NaN();
  double sum_bound = std::numeric_limits<double>::quiet_NaN();
  if (opt.mode == "synthetic_fixed_point_subcolumns")
  {
    avg_bound = synthetic_avg_abs_error_bound(r.refinement_depth, opt.subcolumns);
    sum_bound = static_cast<double>(opt.n) * avg_bound;
  }

  double abs_cpu_gpu = std::numeric_limits<double>::quiet_NaN();
  double abs_exact_cpu = std::numeric_limits<double>::quiet_NaN();
  double abs_exact_gpu = std::numeric_limits<double>::quiet_NaN();
  if (!std::isnan(r.cpu_approximate_sum) && !std::isnan(r.gpu_approximate_sum))
    abs_cpu_gpu = std::fabs(r.cpu_approximate_sum - r.gpu_approximate_sum);
  if (!std::isnan(r.cpu_approximate_sum) && !std::isnan(r.exact_sum))
    abs_exact_cpu = std::fabs(r.exact_sum - r.cpu_approximate_sum);
  if (!std::isnan(r.gpu_approximate_sum) && !std::isnan(r.exact_sum))
    abs_exact_gpu = std::fabs(r.exact_sum - r.gpu_approximate_sum);

  std::fprintf(f,
               "progressive_aggregation,%s,%s,sum,rowpack16,%d,"
               "%s,"
               "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d,8,"
               "%d,%.0f,%d,%d,%d,%d,"
               "%.6f,%.6f,%.6f,%.3f,"
               "64,%.17g,%s,"
               "%.17g,%.17g,%s,"
               "%s,%s,%d,%d,%d,"
               "%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
               csv_escape(ctx.dataset).c_str(),
               csv_escape(ctx.mode).c_str(),
               r.refinement_depth,
               csv_escape(r.kernel_path).c_str(),
               opt.n,
               opt.segment_rows,
               derived.tile_rows,
               opt.subcolumns,
               logical_subcolumns_read,
               logical_bytes,
               opt.block_threads,
               derived.grid,
               opt.warmup,
               opt.iters,
               r.ms_per_iter,
               r.rows_per_sec,
               r.billion_rows_per_sec,
               r.logical_GBps,
               opt.mode == "synthetic_fixed_point_subcolumns" ? 0.0 : std::numeric_limits<double>::quiet_NaN(),
               csv_escape(ctx.basis_mode).c_str(),
               sum_bound,
               avg_bound,
               csv_escape(ctx.error_bound_source).c_str(),
               opt.validate ? "true" : "false",
               csv_escape(prop.name).c_str(),
               prop.multiProcessorCount,
               prop.major,
               prop.minor,
               csv_escape(ctx.encoded_layout).c_str(),
               ctx.max_plane_count,
               ctx.segment_plane_count_min,
               ctx.segment_plane_count_max,
               r.exact_sum,
               r.cpu_approximate_sum,
               r.gpu_approximate_sum,
               abs_cpu_gpu,
               abs_exact_cpu,
               abs_exact_gpu);
}

SyntheticContext setup_synthetic_context(const Options &opt, const Derived &derived)
{
  SyntheticContext ctx;
  ctx.d_subcolumn_storage.resize(static_cast<size_t>(opt.subcolumns), nullptr);
  for (int p = 0; p < opt.subcolumns; ++p)
  {
    uint8_t *ptr = nullptr;
    cuda_check(cudaMalloc(&ptr, static_cast<size_t>(opt.n) * sizeof(uint8_t)), "cudaMalloc(subcolumn)");
    cuda_check(cudaMemset(ptr, p + 1, static_cast<size_t>(opt.n) * sizeof(uint8_t)), "cudaMemset(subcolumn)");
    ctx.d_subcolumn_storage[static_cast<size_t>(p)] = ptr;
    ctx.h_subcolumns.ptrs[p] = ptr;
  }

  ctx.h_segment_base.assign(static_cast<size_t>(derived.num_segments), 0.0);
  ctx.h_subcolumn_basis.resize(static_cast<size_t>(derived.num_segments) * static_cast<size_t>(EXP3_MAX_SUBCOLUMNS));
  for (uint64_t s = 0; s < derived.num_segments; ++s)
  {
    for (int p = 0; p < EXP3_MAX_SUBCOLUMNS; ++p)
      ctx.h_subcolumn_basis[static_cast<size_t>(s) * EXP3_MAX_SUBCOLUMNS + static_cast<size_t>(p)] = basis_value(p);
  }

  cuda_check(cudaMalloc(&ctx.d_segment_base, ctx.h_segment_base.size() * sizeof(double)), "cudaMalloc(segment_base)");
  cuda_check(cudaMalloc(&ctx.d_subcolumn_basis, ctx.h_subcolumn_basis.size() * sizeof(double)), "cudaMalloc(subcolumn_basis)");
  cuda_check(cudaMalloc(&ctx.d_partial_out, static_cast<size_t>(derived.grid) * sizeof(double)), "cudaMalloc(partial_out)");
  cuda_check(cudaMemcpy(ctx.d_segment_base,
                        ctx.h_segment_base.data(),
                        ctx.h_segment_base.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(segment_base)");
  cuda_check(cudaMemcpy(ctx.d_subcolumn_basis,
                        ctx.h_subcolumn_basis.data(),
                        ctx.h_subcolumn_basis.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(subcolumn_basis)");
  return ctx;
}

RealContext setup_real_context(Options &opt, const Derived &derived)
{
  RealContext ctx;
  ctx.dataset = exp3_real::load_dataset(opt.encoded_root);

  if (opt.n_set && opt.n != ctx.dataset.manifest.value_count)
    die("--n does not match encoded manifest value_count");
  if (opt.segment_rows_set && opt.segment_rows != ctx.dataset.manifest.segment_size)
    die("--segment_rows does not match encoded manifest segment_size");
  if (opt.subcolumns_set && opt.subcolumns != static_cast<int>(ctx.dataset.manifest.max_plane_count))
    die("--subcolumns does not match encoded manifest max_plane_count");

  opt.n = ctx.dataset.manifest.value_count;
  opt.segment_rows = ctx.dataset.manifest.segment_size;
  opt.subcolumns = static_cast<int>(ctx.dataset.manifest.max_plane_count);

  ctx.d_subcolumn_storage.resize(static_cast<size_t>(opt.subcolumns), nullptr);
  for (int p = 0; p < opt.subcolumns; ++p)
  {
    const std::vector<uint8_t> &plane = ctx.dataset.planes[static_cast<size_t>(p)];
    uint8_t *ptr = nullptr;
    cuda_check(cudaMalloc(&ptr, plane.size() * sizeof(uint8_t)), "cudaMalloc(real plane)");
    cuda_check(cudaMemcpy(ptr, plane.data(), plane.size() * sizeof(uint8_t), cudaMemcpyHostToDevice),
               "cudaMemcpy(real plane)");
    ctx.d_subcolumn_storage[static_cast<size_t>(p)] = ptr;
    ctx.h_subcolumns.ptrs[p] = ptr;
  }
  for (int p = opt.subcolumns; p < EXP3_RUNTIME_MAX_SUBCOLUMNS; ++p)
    ctx.h_subcolumns.ptrs[p] = nullptr;

  ctx.h_segment_base.resize(static_cast<size_t>(derived.num_segments), 0.0);
  ctx.h_subcolumn_basis.resize(static_cast<size_t>(derived.num_segments) * static_cast<size_t>(opt.subcolumns), 0.0);
  for (const exp3_real::SegmentMeta &segment : ctx.dataset.segments)
  {
    ctx.h_segment_base[static_cast<size_t>(segment.segment_index)] = segment.segment_base;
    for (int p = 0; p < opt.subcolumns; ++p)
    {
      ctx.h_subcolumn_basis[static_cast<size_t>(segment.segment_index) * static_cast<size_t>(opt.subcolumns) +
                            static_cast<size_t>(p)] = segment.plane_basis[static_cast<size_t>(p)];
    }
  }

  cuda_check(cudaMalloc(&ctx.d_segment_base, ctx.h_segment_base.size() * sizeof(double)), "cudaMalloc(real segment_base)");
  cuda_check(cudaMalloc(&ctx.d_subcolumn_basis, ctx.h_subcolumn_basis.size() * sizeof(double)), "cudaMalloc(real subcolumn_basis)");
  cuda_check(cudaMalloc(&ctx.d_partial_out, static_cast<size_t>(derived.grid) * sizeof(double)), "cudaMalloc(real partial_out)");
  cuda_check(cudaMemcpy(ctx.d_segment_base,
                        ctx.h_segment_base.data(),
                        ctx.h_segment_base.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(real segment_base)");
  cuda_check(cudaMemcpy(ctx.d_subcolumn_basis,
                        ctx.h_subcolumn_basis.data(),
                        ctx.h_subcolumn_basis.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(real subcolumn_basis)");

  if (opt.aggregation == "minmax")
  {
    cuda_check(cudaMalloc(&ctx.d_extrema_out, static_cast<size_t>(derived.grid) * 2 * sizeof(double)),
               "cudaMalloc(real extrema_out)");
  }

  return ctx;
}

void free_synthetic_context(SyntheticContext &ctx)
{
  cuda_check(cudaFree(ctx.d_partial_out), "cudaFree(partial_out)");
  cuda_check(cudaFree(ctx.d_subcolumn_basis), "cudaFree(subcolumn_basis)");
  cuda_check(cudaFree(ctx.d_segment_base), "cudaFree(segment_base)");
  for (uint8_t *ptr : ctx.d_subcolumn_storage)
    cuda_check(cudaFree(ptr), "cudaFree(subcolumn)");
}

void free_real_context(RealContext &ctx)
{
  cuda_check(cudaFree(ctx.d_partial_out), "cudaFree(real partial_out)");
  if (ctx.d_extrema_out)
    cuda_check(cudaFree(ctx.d_extrema_out), "cudaFree(real extrema_out)");
  cuda_check(cudaFree(ctx.d_subcolumn_basis), "cudaFree(real subcolumn_basis)");
  cuda_check(cudaFree(ctx.d_segment_base), "cudaFree(real segment_base)");
  for (uint8_t *ptr : ctx.d_subcolumn_storage)
    cuda_check(cudaFree(ptr), "cudaFree(real plane)");
}

[[nodiscard]] exp3_real::ProgressiveExtremaSummary reduce_extrema_per_segment(
    double *d_extrema_out,
    int grid,
    const exp3_real::Dataset &dataset,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    int refinement_depth)
{
  std::vector<double> h_buf(static_cast<std::size_t>(grid) * 2);
  cuda_check(cudaMemcpy(h_buf.data(),
                        d_extrema_out,
                        h_buf.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(extrema_out reduce)");

  std::size_t keep_planes = static_cast<std::size_t>(refinement_depth) + 1U;

  exp3_real::ProgressiveExtremaSummary summary;
  summary.min.lower = std::numeric_limits<double>::infinity();
  summary.min.upper = std::numeric_limits<double>::infinity();
  summary.max.lower = -std::numeric_limits<double>::infinity();
  summary.max.upper = -std::numeric_limits<double>::infinity();

  for (const exp3_real::SegmentMeta &segment : dataset.segments)
  {
    std::size_t seg_idx = static_cast<std::size_t>(segment.segment_index);
    std::size_t seg_tile_start = seg_idx * static_cast<std::size_t>(tiles_per_segment);
    std::size_t seg_tile_end = std::min(seg_tile_start + static_cast<std::size_t>(tiles_per_segment),
                                        static_cast<std::size_t>(grid));

    double seg_min_prefix = std::numeric_limits<double>::infinity();
    double seg_max_prefix = -std::numeric_limits<double>::infinity();

    for (std::size_t tile = seg_tile_start; tile < seg_tile_end; ++tile)
    {
      double mp = h_buf[2 * tile];
      double xp = h_buf[2 * tile + 1];
      if (mp < seg_min_prefix)
        seg_min_prefix = mp;
      if (xp > seg_max_prefix)
        seg_max_prefix = xp;
    }

    double tail_upper = exp3_real::segment_tail_upper_bound(segment, keep_planes);

    if (seg_min_prefix < summary.min.lower)
      summary.min.lower = seg_min_prefix;
    if (seg_min_prefix + tail_upper < summary.min.upper)
      summary.min.upper = seg_min_prefix + tail_upper;
    if (seg_max_prefix > summary.max.lower)
      summary.max.lower = seg_max_prefix;
    if (seg_max_prefix + tail_upper > summary.max.upper)
      summary.max.upper = seg_max_prefix + tail_upper;
  }

  if (!std::isfinite(summary.min.lower))
  {
    summary.min.lower = 0.0;
    summary.min.upper = 0.0;
    summary.max.lower = 0.0;
    summary.max.upper = 0.0;
  }

  return summary;
}

void validate_real_minmax(int refinement_depth,
                         const Options &opt,
                         const Derived &derived,
                         const RealContext &ctx)
{
  cuda_check(cudaMemset(ctx.d_extrema_out, 0, static_cast<std::size_t>(derived.grid) * 2 * sizeof(double)),
             "cudaMemset(d_extrema_out validate minmax)");
  launch_progressive_extrema_rowpack16_runtime(refinement_depth,
                                               derived.grid,
                                               opt.block_threads,
                                               ctx.h_subcolumns,
                                               opt.subcolumns,
                                               opt.n,
                                               opt.segment_rows,
                                               derived.tiles_per_segment,
                                               ctx.d_segment_base,
                                               ctx.d_subcolumn_basis,
                                               ctx.d_extrema_out);
  cuda_check(cudaGetLastError(), "validation launch minmax");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(validation minmax)");

  exp3_real::ProgressiveExtremaSummary gpu_summary = reduce_extrema_per_segment(
      ctx.d_extrema_out, derived.grid, ctx.dataset, opt.segment_rows, derived.tiles_per_segment, refinement_depth);

  exp3_real::ProgressiveExtremaSummary cpu_summary = exp3_real::cpu_progressive_extrema_summary(ctx.dataset, refinement_depth);

  double tol = real_validation_tolerance(cpu_summary.min.lower, gpu_summary.min.lower);
  double abs_min_lower = std::fabs(cpu_summary.min.lower - gpu_summary.min.lower);
  double abs_min_upper = std::fabs(cpu_summary.min.upper - gpu_summary.min.upper);
  double abs_max_lower = std::fabs(cpu_summary.max.lower - gpu_summary.max.lower);
  double abs_max_upper = std::fabs(cpu_summary.max.upper - gpu_summary.max.upper);

  if (abs_min_lower > tol || abs_min_upper > tol || abs_max_lower > tol || abs_max_upper > tol)
  {
    char msg[1024];
    std::snprintf(msg, sizeof(msg),
                  "minmax validation failed for depth=%d:\n"
                  "  gpu  min_lower=%.17g min_upper=%.17g max_lower=%.17g max_upper=%.17g\n"
                  "  cpu  min_lower=%.17g min_upper=%.17g max_lower=%.17g max_upper=%.17g\n"
                  "  diff min_lower=%.3g min_upper=%.3g max_lower=%.3g max_upper=%.3g tol=%.3g",
                  refinement_depth,
                  gpu_summary.min.lower, gpu_summary.min.upper, gpu_summary.max.lower, gpu_summary.max.upper,
                  cpu_summary.min.lower, cpu_summary.min.upper, cpu_summary.max.lower, cpu_summary.max.upper,
                  abs_min_lower, abs_min_upper, abs_max_lower, abs_max_upper, tol);
    die(msg);
  }

  std::fprintf(stderr,
               "[exp3] minmax validate ok depth=%d gpu_min_lo=%.17g gpu_min_hi=%.17g gpu_max_lo=%.17g gpu_max_hi=%.17g\n",
               refinement_depth,
               gpu_summary.min.lower, gpu_summary.min.upper, gpu_summary.max.lower, gpu_summary.max.upper);
}

MinMaxRunResult run_one_real_minmax(int refinement_depth,
                                    const Options &opt,
                                    const Derived &derived,
                                    const RealContext &ctx)
{
  const char *kernel_path = "runtime_rowpack16_minmax";

  for (int i = 0; i < opt.warmup; ++i)
  {
    launch_progressive_extrema_rowpack16_runtime(refinement_depth,
                                                 derived.grid,
                                                 opt.block_threads,
                                                 ctx.h_subcolumns,
                                                 opt.subcolumns,
                                                 opt.n,
                                                 opt.segment_rows,
                                                 derived.tiles_per_segment,
                                                 ctx.d_segment_base,
                                                 ctx.d_subcolumn_basis,
                                                 ctx.d_extrema_out);
  }
  cuda_check(cudaGetLastError(), "warmup launch minmax");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup minmax)");

  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "cudaEventCreate(start minmax)");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop minmax)");

  cuda_check(cudaEventRecord(start), "cudaEventRecord(start minmax)");
  for (int i = 0; i < opt.iters; ++i)
  {
    launch_progressive_extrema_rowpack16_runtime(refinement_depth,
                                                 derived.grid,
                                                 opt.block_threads,
                                                 ctx.h_subcolumns,
                                                 opt.subcolumns,
                                                 opt.n,
                                                 opt.segment_rows,
                                                 derived.tiles_per_segment,
                                                 ctx.d_segment_base,
                                                 ctx.d_subcolumn_basis,
                                                 ctx.d_extrema_out);
  }
  cuda_check(cudaGetLastError(), "timed launch minmax");
  cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop minmax)");
  cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop minmax)");

  float ms_total = 0.0f;
  cuda_check(cudaEventElapsedTime(&ms_total, start, stop), "cudaEventElapsedTime(minmax)");
  cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start minmax)");
  cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop minmax)");

  float ms_per_iter = ms_total / static_cast<float>(opt.iters);
  double seconds = static_cast<double>(ms_per_iter) / 1000.0;
  double logical_bytes = static_cast<double>(opt.n) * static_cast<double>(refinement_depth + 1);

  MinMaxRunResult r{};
  r.refinement_depth = refinement_depth;
  r.ms_per_iter = ms_per_iter;
  r.rows_per_sec = static_cast<double>(opt.n) / seconds;
  r.billion_rows_per_sec = r.rows_per_sec / 1e9;
  r.logical_GBps = (logical_bytes / seconds) / 1e9;
  r.kernel_path = kernel_path;

  // Observe one run for correctness data
  cuda_check(cudaMemset(ctx.d_extrema_out, 0, static_cast<std::size_t>(derived.grid) * 2 * sizeof(double)),
             "cudaMemset(d_extrema_out observe minmax)");
  launch_progressive_extrema_rowpack16_runtime(refinement_depth,
                                               derived.grid,
                                               opt.block_threads,
                                               ctx.h_subcolumns,
                                               opt.subcolumns,
                                               opt.n,
                                               opt.segment_rows,
                                               derived.tiles_per_segment,
                                               ctx.d_segment_base,
                                               ctx.d_subcolumn_basis,
                                               ctx.d_extrema_out);
  cuda_check(cudaGetLastError(), "observe launch minmax");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(observe minmax)");

  exp3_real::ProgressiveExtremaSummary gpu_summary = reduce_extrema_per_segment(
      ctx.d_extrema_out, derived.grid, ctx.dataset, opt.segment_rows, derived.tiles_per_segment, refinement_depth);

  r.min_lower = gpu_summary.min.lower;
  r.min_upper = gpu_summary.min.upper;
  r.max_lower = gpu_summary.max.lower;
  r.max_upper = gpu_summary.max.upper;

  exp3_real::ExactExtrema exact = exp3_real::cpu_exact_extrema(ctx.dataset);
  r.exact_min = exact.min;
  r.exact_max = exact.max;

  if (opt.validate)
  {
    exp3_real::ProgressiveExtremaSummary cpu_summary = exp3_real::cpu_progressive_extrema_summary(ctx.dataset, refinement_depth);
    r.cpu_min_lower = cpu_summary.min.lower;
    r.cpu_min_upper = cpu_summary.min.upper;
    r.cpu_max_lower = cpu_summary.max.lower;
    r.cpu_max_upper = cpu_summary.max.upper;

    exp3_real::CandidateCounts cc = exp3_real::cpu_candidate_counts(ctx.dataset, refinement_depth);
    r.min_candidates = cc.min_candidates;
    r.max_candidates = cc.max_candidates;
    r.total_rows = cc.total_rows;
  }

  return r;
}

void write_csv_header_minmax(std::FILE *f)
{
  std::fprintf(f,
               "benchmark,dataset,mode,aggregation,load_strategy,refinement_depth,"
               "kernel_path,"
               "n,segment_rows,tile_rows,subcolumns,subcolumn_bits,"
               "logical_subcolumns_read,logical_bytes,block,grid,warmup,iters,"
               "ms_per_iter,rows_per_sec,billion_rows_per_sec,logical_GBps,"
               "validated,device,sm,cc_major,cc_minor,"
               "encoded_layout,max_plane_count,segment_plane_count_min,segment_plane_count_max,"
               "fractional_bits,"
               "exact_min,exact_max,"
               "min_lower,min_upper,max_lower,max_upper,"
               "bound_valid_min,bound_valid_max,"
               "cpu_min_lower,cpu_min_upper,cpu_max_lower,cpu_max_upper,"
               "abs_cpu_gpu_min_lower,abs_cpu_gpu_min_upper,abs_cpu_gpu_max_lower,abs_cpu_gpu_max_upper,"
               "min_candidates,max_candidates,total_rows\n");
}

void write_csv_row_minmax(std::FILE *f,
                          const Options &opt,
                          const Derived &derived,
                          const MinMaxRunResult &r,
                          const CsvContext &ctx,
                          const cudaDeviceProp &prop)
{
  int logical_subcolumns_read = r.refinement_depth + 1;
  double logical_bytes = static_cast<double>(opt.n) * static_cast<double>(logical_subcolumns_read);
  uint32_t fractional_bits = ctx.fractional_bits;

  bool bound_valid_min = !std::isnan(r.exact_min) && !std::isnan(r.min_lower) && !std::isnan(r.min_upper) &&
                         r.exact_min >= r.min_lower && r.exact_min <= r.min_upper;
  bool bound_valid_max = !std::isnan(r.exact_max) && !std::isnan(r.max_lower) && !std::isnan(r.max_upper) &&
                         r.exact_max >= r.max_lower && r.exact_max <= r.max_upper;

  double abs_cpu_gpu_min_lower = std::numeric_limits<double>::quiet_NaN();
  double abs_cpu_gpu_min_upper = std::numeric_limits<double>::quiet_NaN();
  double abs_cpu_gpu_max_lower = std::numeric_limits<double>::quiet_NaN();
  double abs_cpu_gpu_max_upper = std::numeric_limits<double>::quiet_NaN();
  if (!std::isnan(r.cpu_min_lower) && !std::isnan(r.min_lower))
    abs_cpu_gpu_min_lower = std::fabs(r.cpu_min_lower - r.min_lower);
  if (!std::isnan(r.cpu_min_upper) && !std::isnan(r.min_upper))
    abs_cpu_gpu_min_upper = std::fabs(r.cpu_min_upper - r.min_upper);
  if (!std::isnan(r.cpu_max_lower) && !std::isnan(r.max_lower))
    abs_cpu_gpu_max_lower = std::fabs(r.cpu_max_lower - r.max_lower);
  if (!std::isnan(r.cpu_max_upper) && !std::isnan(r.max_upper))
    abs_cpu_gpu_max_upper = std::fabs(r.cpu_max_upper - r.max_upper);

  std::fprintf(f,
               "progressive_aggregation,%s,%s,minmax,rowpack16,%d,"
               "%s,"
               "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d,8,"
               "%d,%.0f,%d,%d,%d,%d,"
               "%.6f,%.6f,%.6f,%.3f,"
               "%s,%s,%d,%d,%d,"
               "%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
               "%u,"
               "%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,"
               "%s,%s,"
               "%.17g,%.17g,%.17g,%.17g,"
               "%.17g,%.17g,%.17g,%.17g,"
               "%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
               csv_escape(ctx.dataset).c_str(),
               csv_escape(ctx.mode).c_str(),
               r.refinement_depth,
               csv_escape(r.kernel_path).c_str(),
               opt.n,
               opt.segment_rows,
               derived.tile_rows,
               opt.subcolumns,
               logical_subcolumns_read,
               logical_bytes,
               opt.block_threads,
               derived.grid,
               opt.warmup,
               opt.iters,
               r.ms_per_iter,
               r.rows_per_sec,
               r.billion_rows_per_sec,
               r.logical_GBps,
               opt.validate ? "true" : "false",
               csv_escape(prop.name).c_str(),
               prop.multiProcessorCount,
               prop.major,
               prop.minor,
               csv_escape(ctx.encoded_layout).c_str(),
               ctx.max_plane_count,
               ctx.segment_plane_count_min,
               ctx.segment_plane_count_max,
               fractional_bits,
               r.exact_min,
               r.exact_max,
               r.min_lower,
               r.min_upper,
               r.max_lower,
               r.max_upper,
               bound_valid_min ? "true" : "false",
               bound_valid_max ? "true" : "false",
               r.cpu_min_lower,
               r.cpu_min_upper,
               r.cpu_max_lower,
               r.cpu_max_upper,
                abs_cpu_gpu_min_lower,
                abs_cpu_gpu_min_upper,
                abs_cpu_gpu_max_lower,
                abs_cpu_gpu_max_upper,
                r.min_candidates,
                r.max_candidates,
                r.total_rows);
}

} // namespace

int main(int argc, char **argv)
{
  Options opt = parse_args(argc, argv);

  cuda_check(cudaSetDevice(opt.device), "cudaSetDevice");
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, opt.device), "cudaGetDeviceProperties");

  CsvContext csv_ctx;
  csv_ctx.mode = opt.mode;

  if (opt.mode == "encoded_dev_subcolumns")
  {
    exp3_real::Dataset preview = exp3_real::load_dataset(opt.encoded_root);
    if (opt.n_set && opt.n != preview.manifest.value_count)
      die("--n does not match encoded manifest value_count");
    if (opt.segment_rows_set && opt.segment_rows != preview.manifest.segment_size)
      die("--segment_rows does not match encoded manifest segment_size");
    if (opt.subcolumns_set && opt.subcolumns != static_cast<int>(preview.manifest.max_plane_count))
      die("--subcolumns does not match encoded manifest max_plane_count");
    opt.n = preview.manifest.value_count;
    opt.segment_rows = preview.manifest.segment_size;
    opt.subcolumns = static_cast<int>(preview.manifest.max_plane_count);
    csv_ctx.dataset = preview.manifest.dataset;
    csv_ctx.basis_mode = "per_segment_explicit";
    csv_ctx.error_bound_source = "encoded_dev_artifact";
    csv_ctx.encoded_layout = preview.manifest.encoded_layout;
    csv_ctx.fractional_bits = preview.segments.empty() ? 0U : preview.segments.front().fractional_bits;
    csv_ctx.max_plane_count = preview.manifest.max_plane_count;
    csv_ctx.segment_plane_count_min = preview.manifest.segment_plane_count_min;
    csv_ctx.segment_plane_count_max = preview.manifest.segment_plane_count_max;
  }
  else
  {
    csv_ctx.max_plane_count = static_cast<uint64_t>(opt.subcolumns);
    csv_ctx.segment_plane_count_min = static_cast<uint64_t>(opt.subcolumns);
    csv_ctx.segment_plane_count_max = static_cast<uint64_t>(opt.subcolumns);
  }

  finalize_refinement_bounds(opt);
  Derived derived = compute_derived(opt);

  std::FILE *f = std::fopen(opt.csv_path.c_str(), "wb");
  if (!f)
    die("failed to open --csv output path");
  if (opt.aggregation == "minmax")
    write_csv_header_minmax(f);
  else
    write_csv_header(f);

  if (opt.mode == "synthetic_fixed_point_subcolumns")
  {
    SyntheticContext ctx = setup_synthetic_context(opt, derived);
    for (int depth = opt.refine_min; depth <= opt.refine_max; ++depth)
    {
      if (opt.validate)
        validate_synthetic(depth, opt, derived, ctx);

      RunResult r = run_one_synthetic(depth, opt, derived, ctx);
      write_csv_row(f, opt, derived, r, csv_ctx, prop);
      std::fflush(f);
      std::fprintf(stderr,
                   "[exp3] synthetic depth=%d n=%" PRIu64 " grid=%d ms=%.3f rows/s=%.3e logical_GB/s=%.1f\n",
                   depth,
                   opt.n,
                   derived.grid,
                   r.ms_per_iter,
                   r.rows_per_sec,
                   r.logical_GBps);
    }
    free_synthetic_context(ctx);
  }
  else
  {
    RealContext ctx = setup_real_context(opt, derived);
    csv_ctx.dataset = ctx.dataset.manifest.dataset;
    csv_ctx.encoded_layout = ctx.dataset.manifest.encoded_layout;
    csv_ctx.fractional_bits = ctx.dataset.segments.empty() ? 0U : ctx.dataset.segments.front().fractional_bits;
    csv_ctx.max_plane_count = ctx.dataset.manifest.max_plane_count;
    csv_ctx.segment_plane_count_min = ctx.dataset.manifest.segment_plane_count_min;
    csv_ctx.segment_plane_count_max = ctx.dataset.manifest.segment_plane_count_max;

    if (opt.aggregation == "minmax")
    {
      for (int depth = opt.refine_min; depth <= opt.refine_max; ++depth)
      {
        if (opt.validate)
          validate_real_minmax(depth, opt, derived, ctx);

        MinMaxRunResult r = run_one_real_minmax(depth, opt, derived, ctx);
        write_csv_row_minmax(f, opt, derived, r, csv_ctx, prop);
        std::fflush(f);
        std::fprintf(stderr,
                     "[exp3] minmax depth=%d dataset=%s n=%" PRIu64 " grid=%d ms=%.3f rows/s=%.3e logical_GB/s=%.1f "
                     "min_lo=%.17g min_hi=%.17g max_lo=%.17g max_hi=%.17g\n",
                     depth,
                     ctx.dataset.manifest.dataset.c_str(),
                     opt.n,
                     derived.grid,
                     r.ms_per_iter,
                     r.rows_per_sec,
                     r.logical_GBps,
                     r.min_lower,
                     r.min_upper,
                     r.max_lower,
                     r.max_upper);
      }
    }
    else
    {
      for (int depth = opt.refine_min; depth <= opt.refine_max; ++depth)
      {
        if (opt.validate)
          validate_real(depth, opt, derived, ctx);

        RunResult r = run_one_real(depth, opt, derived, ctx);
        write_csv_row(f, opt, derived, r, csv_ctx, prop);
        std::fflush(f);
        std::fprintf(stderr,
                     "[exp3] real depth=%d dataset=%s n=%" PRIu64 " grid=%d ms=%.3f rows/s=%.3e logical_GB/s=%.1f gpu_sum=%.17g\n",
                     depth,
                     ctx.dataset.manifest.dataset.c_str(),
                     opt.n,
                     derived.grid,
                     r.ms_per_iter,
                     r.rows_per_sec,
                     r.logical_GBps,
                     r.gpu_approximate_sum);
      }
    }
    free_real_context(ctx);
  }

  std::fclose(f);
  return 0;
}
