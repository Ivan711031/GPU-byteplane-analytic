#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "filter_aggregate_kernels.cuh"
#include "../experiment3/exp3_real_data_layout.hpp"
#include "../../buff_encoder/buff_codec.hpp"

// ============================================================================
// Utility helpers (self-contained copies from Exp4 benchmark)
// ============================================================================

namespace
{

[[nodiscard]] const char *cuda_err_str(cudaError_t err) { return cudaGetErrorString(err); }

[[noreturn]] void die(const char *msg)
{
  std::fprintf(stderr, "error: %s\n", msg);
  std::exit(2);
}

[[noreturn]] void die(const std::string &msg) { die(msg.c_str()); }

void cuda_check(cudaError_t err, const char *where)
{
  if (err == cudaSuccess) return;
  std::fprintf(stderr, "cuda error at %s: %s\n", where, cuda_err_str(err));
  std::exit(2);
}

[[nodiscard]] bool parse_u64(std::string_view s, uint64_t &out)
{
  if (s.empty()) return false;
  uint64_t value = 0;
  for (char c : s)
  {
    if (c < '0' || c > '9') return false;
    uint64_t digit = static_cast<uint64_t>(c - '0');
    if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10ull) return false;
    value = value * 10ull + digit;
  }
  out = value;
  return true;
}

// ---------------------------------------------------------------------------
// Threshold encoding helpers (copied from exp4 bench_progressive_filter.cu)
// ---------------------------------------------------------------------------

struct SegmentThresholdInfo
{
  std::vector<uint8_t> threshold_bytes;
  bool all_qualified = false;
  bool all_disqualified = false;
  bool threshold_exact = true;
  uint64_t threshold_combined = 0;
};

struct PrecisionInfo
{
  std::string mode = "exact";
  std::string decimals = "NA";
};

uint64_t le_bytes_to_u64(const std::vector<uint8_t> &bytes)
{
  uint64_t value = 0;
  for (size_t i = 0; i < bytes.size() && i < sizeof(uint64_t); ++i)
    value |= static_cast<uint64_t>(bytes[i]) << (i * 8);
  return value;
}

static std::vector<uint8_t> subtract_le_bytes(const std::vector<uint8_t> &a,
                                              const std::vector<uint8_t> &b)
{
  std::vector<uint8_t> result(a.size(), 0);
  int borrow = 0;
  for (size_t i = 0; i < a.size(); ++i)
  {
    int val = static_cast<int>(a[i]) - borrow;
    if (i < b.size()) val -= static_cast<int>(b[i]);
    if (val < 0)
    {
      val += 256;
      borrow = 1;
    }
    else
    {
      borrow = 0;
    }
    result[i] = static_cast<uint8_t>(val);
  }
  while (!result.empty() && result.back() == 0) result.pop_back();
  return result;
}

static void extract_plane_bytes_from_combined_le(
    const std::vector<uint8_t> &combined_le,
    size_t total_bits,
    std::vector<uint8_t> &out_plane_bytes,
    size_t max_planes)
{
  auto bit_is_set = [&](size_t bit_index) -> bool
  {
    size_t byte_index = bit_index / 8;
    if (byte_index >= combined_le.size()) return false;
    return ((combined_le[byte_index] >> (bit_index % 8)) & 1U) != 0;
  };

  size_t plane_count = (total_bits == 0) ? 0 : (total_bits + 7) / 8;
  out_plane_bytes.resize(max_planes, 0);
  for (size_t plane = 0; plane < plane_count && plane < max_planes; ++plane)
  {
    size_t width = 8;
    if (plane + 1 == plane_count)
    {
      size_t trailing = total_bits - 8 * (plane_count - 1);
      width = (trailing == 0) ? 8 : trailing;
    }
    size_t start_bit = total_bits - 8 * plane - width;
    uint8_t byte = 0;
    for (size_t offset = 0; offset < width; ++offset)
    {
      if (bit_is_set(start_bit + offset))
        byte |= static_cast<uint8_t>(1U << offset);
    }
    out_plane_bytes[plane] = byte;
  }
}

std::vector<std::pair<double, double>> compute_segment_min_max(
    const std::filesystem::path &raw_path, uint64_t segment_rows, uint64_t num_segments)
{
  uint64_t file_size = std::filesystem::file_size(raw_path);
  if (file_size % sizeof(double) != 0) die("raw file size not aligned to FP64");
  uint64_t value_count = file_size / sizeof(double);

  std::ifstream input(raw_path, std::ios::binary);
  if (!input) die("failed to open raw file for segment min/max");

  std::vector<std::pair<double, double>> result;
  result.reserve(num_segments);

  std::vector<double> buffer(static_cast<size_t>(segment_rows));
  uint64_t remaining = value_count;
  for (uint64_t seg = 0; seg < num_segments; ++seg)
  {
    uint64_t current = std::min(remaining, segment_rows);
    input.read(reinterpret_cast<char *>(buffer.data()),
               static_cast<std::streamsize>(current * sizeof(double)));
    if (!input) die("failed to read raw file for segment min/max");

    double seg_min = buffer[0];
    double seg_max = buffer[0];
    for (uint64_t i = 1; i < current; ++i)
    {
      seg_min = std::min(seg_min, buffer[i]);
      seg_max = std::max(seg_max, buffer[i]);
    }
    result.emplace_back(seg_min, seg_max);
    remaining -= current;
  }
  return result;
}

std::vector<SegmentThresholdInfo> compute_threshold_bytes(
    const exp3_real::Dataset &dataset,
    const std::vector<std::pair<double, double>> &segment_min_max,
    double threshold_fp64)
{
  std::vector<SegmentThresholdInfo> result;
  result.reserve(dataset.segments.size());

  for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
  {
    const auto &segment = dataset.segments[seg];
    double seg_min = segment_min_max[seg].first;
    double seg_max = segment_min_max[seg].second;

    if (seg_min == std::numeric_limits<double>::lowest() && segment.integer_base_is_v2)
    {
      seg_min = segment.segment_base;
    }
    if (seg_max == std::numeric_limits<double>::max() && segment.integer_base_is_v2)
    {
      seg_max = segment.segment_base
                + std::ldexp(1.0, static_cast<int>(segment.integer_offset_bits))
                - std::ldexp(1.0, -static_cast<int>(segment.fractional_bits));
    }

    SegmentThresholdInfo info;

    if (threshold_fp64 < seg_min)
    {
      info.all_qualified = true;
      info.threshold_bytes.resize(dataset.manifest.max_plane_count, 0);
    }
    else if (threshold_fp64 >= seg_max)
    {
      info.all_disqualified = true;
      info.threshold_bytes.resize(dataset.manifest.max_plane_count, 0);
    }
    else
    {
      uint32_t frac = segment.fractional_bits;
      std::vector<uint8_t> t_code_le;

      uint64_t integer_base_u64 = le_bytes_to_u64(segment.integer_base_le);

      std::vector<uint8_t> combined_T_le;
      if (integer_base_u64 == 0)
      {
        if (segment.integer_base_is_v2)
        {
          long double scale = std::ldexp(1.0L, static_cast<int>(frac));
          uint64_t t_code = static_cast<uint64_t>(
              std::floor(static_cast<long double>(threshold_fp64) * scale));
          t_code_le.clear();
          for (size_t i = 0; i < sizeof(uint64_t); ++i)
            t_code_le.push_back(static_cast<uint8_t>((t_code >> (i * 8)) & 0xFF));
        }
        else
        {
          long double scale = std::ldexp(1.0L, static_cast<int>(frac));
          uint64_t t_code = static_cast<uint64_t>(
              std::floor(static_cast<long double>(threshold_fp64) * scale));
          t_code_le.clear();
          for (size_t i = 0; i < sizeof(uint64_t); ++i)
            t_code_le.push_back(static_cast<uint8_t>((t_code >> (i * 8)) & 0xFF));
        }
        combined_T_le = t_code_le;
      }
      else
      {
        std::vector<uint8_t> base_shifted_le;
        if (segment.integer_base_is_v2)
        {
          base_shifted_le = segment.integer_base_le;
          long double scale = std::ldexp(1.0L, static_cast<int>(frac));
          uint64_t t_code = static_cast<uint64_t>(
              std::floor(static_cast<long double>(threshold_fp64) * scale));
          t_code_le.clear();
          for (size_t i = 0; i < sizeof(uint64_t); ++i)
            t_code_le.push_back(static_cast<uint8_t>((t_code >> (i * 8)) & 0xFF));
        }
        else
        {
          {
            long double scale = std::ldexp(1.0L, static_cast<int>(frac));
            uint64_t t_code = static_cast<uint64_t>(
                std::floor(static_cast<long double>(threshold_fp64) * scale));
            t_code_le.clear();
            for (size_t i = 0; i < sizeof(uint64_t); ++i)
              t_code_le.push_back(static_cast<uint8_t>((t_code >> (i * 8)) & 0xFF));
          }
          {
            long double scale = std::ldexp(1.0L, static_cast<int>(frac));
            uint64_t base_code = static_cast<uint64_t>(
                std::floor(static_cast<long double>(integer_base_u64) * scale));
            base_shifted_le.clear();
            for (size_t i = 0; i < sizeof(uint64_t); ++i)
              base_shifted_le.push_back(static_cast<uint8_t>((base_code >> (i * 8)) & 0xFF));
          }
        }

        size_t max_len = std::max(t_code_le.size(), base_shifted_le.size());
        if (t_code_le.size() < max_len)
          t_code_le.resize(max_len, 0);
        if (base_shifted_le.size() < max_len)
          base_shifted_le.resize(max_len, 0);
        combined_T_le = subtract_le_bytes(t_code_le, base_shifted_le);
      }

      size_t total_bits = segment.get_fixed_len_bits();
      extract_plane_bytes_from_combined_le(
          combined_T_le, total_bits, info.threshold_bytes, dataset.manifest.max_plane_count);
      info.threshold_combined = le_bytes_to_u64(combined_T_le);
    }

    result.push_back(std::move(info));
  }

  return result;
}

PrecisionInfo infer_precision_info(const std::filesystem::path &encoded_root,
                                   const exp3_real::Dataset &dataset)
{
  PrecisionInfo info;

  std::string artifact_name = encoded_root.parent_path().filename().string();
  std::size_t p = artifact_name.rfind("_p");
  if (p != std::string::npos && p + 2 < artifact_name.size())
  {
    std::size_t d = p + 2;
    while (d < artifact_name.size() &&
           std::isdigit(static_cast<unsigned char>(artifact_name[d])) != 0)
      ++d;
    if (d > p + 2)
    {
      info.mode = "bounded";
      info.decimals = artifact_name.substr(p + 2, d - (p + 2));
      return info;
    }
  }

  bool bounded = false;
  bool cap_consistent = true;
  bool cap_set = false;
  uint32_t cap_bits = 0;
  for (const auto &segment : dataset.segments)
  {
    if (segment.raw_fractional_bits != segment.fractional_bits)
      bounded = true;
    if (segment.precision_cap_bits != segment.raw_fractional_bits)
      bounded = true;
    if (!cap_set)
    {
      cap_bits = segment.precision_cap_bits;
      cap_set = true;
    }
    else if (cap_bits != segment.precision_cap_bits)
    {
      cap_consistent = false;
    }
  }

  if (bounded)
  {
    info.mode = "bounded";
    if (cap_set && cap_consistent)
    {
      int decimals = static_cast<int>(std::floor(static_cast<double>(cap_bits) * std::log10(2.0)));
      if (decimals >= 0) info.decimals = std::to_string(decimals);
    }
  }

  return info;
}

// ---------------------------------------------------------------------------
// CSV formatting helpers
// ---------------------------------------------------------------------------

std::string format_double_17g(double value)
{
  char buf[128];
  std::snprintf(buf, sizeof(buf), "%.17g", value);
  return std::string(buf);
}

std::string csv_bool_or_na(bool available, bool value)
{
  if (!available) return "NA";
  return value ? "true" : "false";
}

std::string csv_u64_or_na(bool available, uint64_t value)
{
  if (!available) return "NA";
  return std::to_string(value);
}

std::string csv_u32_or_na(bool available, uint32_t value)
{
  if (!available) return "NA";
  return std::to_string(value);
}

std::string csv_double_or_na(bool available, double value)
{
  if (!available || !std::isfinite(value)) return "NA";
  return format_double_17g(value);
}

std::string slurm_job_id_or_na()
{
  const char *job_id = std::getenv("SLURM_JOB_ID");
  if (job_id == nullptr || job_id[0] == '\0') return "NA";
  return std::string(job_id);
}

// ---------------------------------------------------------------------------
// Options & derived geometry
// ---------------------------------------------------------------------------

struct Options
{
  std::string dataset_path;
  std::string raw_path;
  double threshold = 0.0;
  uint64_t n = 0;
  bool n_set = false;
  uint64_t segment_rows = 100000000ull;
  int block_threads = 256;
  int iters = 200;
  int warmup = 10;
  int max_filter_planes = -1;
  bool validate = false;
  std::string warp_strategy = "byte_mask";
  std::string csv_path = "exp4_filter_aggregate.csv";
};

struct Derived
{
  uint64_t tile_rows = 0;
  uint64_t tiles_per_segment = 0;
  uint64_t num_segments = 0;
  uint64_t grid = 0;
};

void print_usage(const char *argv0)
{
  std::fprintf(stderr,
               "Usage: %s [options]\n"
               "\n"
               "Experiment 4 Extension: Combined Filter + Aggregate (SUM).\n"
               "\n"
               "Options:\n"
               "  --dataset PATH        Path to encoded dataset directory\n"
               "  --raw PATH            Path to raw float64 data file\n"
               "  --threshold FP64      Predicate threshold (default: 0.0)\n"
               "  --n N                 Number of rows (optional, auto-detect)\n"
               "  --segment-rows N      Rows per segment (default: 100000000)\n"
               "  --block-threads N     Threads per block (default: 256)\n"
               "  --iters N             Number of iterations (default: 200)\n"
               "  --warmup N            Warmup iterations (default: 10)\n"
               "  --max-filter-planes N Read planes 0..N-1 (default: max available)\n"
               "  --validate            Run CPU reference and check correctness\n"
               "  --warp-strategy STR   Filter strategy (default: byte_mask, only byte_mask)\n"
               "  --csv PATH            Output CSV path\n",
               argv0);
}

Options parse_args(int argc, char **argv)
{
  Options opt;
  for (int i = 1; i < argc; ++i)
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
    else if (a == "--dataset")
    {
      opt.dataset_path = std::string(need_value(a));
    }
    else if (a == "--raw")
    {
      opt.raw_path = std::string(need_value(a));
    }
    else if (a == "--threshold")
    {
      opt.threshold = std::stod(std::string(need_value(a)));
    }
    else if (a == "--n")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0) die("invalid --n");
      opt.n = v;
      opt.n_set = true;
    }
    else if (a == "--segment-rows")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0) die("invalid --segment-rows");
      opt.segment_rows = v;
    }
    else if (a == "--block-threads")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > 1024) die("invalid --block-threads");
      opt.block_threads = static_cast<int>(v);
    }
    else if (a == "--iters")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > 100000000) die("invalid --iters");
      opt.iters = static_cast<int>(v);
    }
    else if (a == "--warmup")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v > 1000000) die("invalid --warmup");
      opt.warmup = static_cast<int>(v);
    }
    else if (a == "--max-filter-planes")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > 256) die("invalid --max-filter-planes");
      opt.max_filter_planes = static_cast<int>(v);
    }
    else if (a == "--validate")
    {
      opt.validate = true;
    }
    else if (a == "--warp-strategy")
    {
      std::string ws(need_value(a));
      if (ws != "byte_mask")
        die("invalid --warp-strategy (expected 'byte_mask')");
      opt.warp_strategy = ws;
    }
    else if (a == "--csv")
    {
      opt.csv_path = std::string(need_value(a));
    }
    else
    {
      std::string msg = "unknown arg: ";
      msg += std::string(a);
      die(msg);
    }
  }

  if (opt.dataset_path.empty()) die("--dataset is required");
  if (opt.block_threads % 32 != 0) die("--block-threads must be multiple of 32");
  return opt;
}

Derived compute_derived(uint64_t n, uint64_t segment_rows, int block_threads)
{
  Derived d{};
  d.tile_rows = static_cast<uint64_t>(block_threads) *
                static_cast<uint64_t>(FA_ROWPACK16_WIDTH);
  d.tiles_per_segment = fa_ceil_div_u64(segment_rows, d.tile_rows);
  d.num_segments = fa_ceil_div_u64(n, segment_rows);
  d.grid = d.num_segments * d.tiles_per_segment;
  if (d.grid == 0 || d.grid > static_cast<uint64_t>(std::numeric_limits<int>::max()))
    die("grid does not fit in int range");
  return d;
}

// ---------------------------------------------------------------------------
// CPU reference: compute qualified count + sum from raw float64 data
// ---------------------------------------------------------------------------
struct CpuReference
{
  uint64_t qualified_count = 0;
  double qualified_sum = 0.0;
  double qualified_avg = 0.0;
};

CpuReference cpu_raw_reference(const std::filesystem::path &raw_path, double threshold)
{
  uint64_t file_size = std::filesystem::file_size(raw_path);
  if (file_size % sizeof(double) != 0) die("raw file size not aligned to FP64");
  uint64_t value_count = file_size / sizeof(double);

  std::ifstream input(raw_path, std::ios::binary);
  if (!input) die("failed to open raw file");

  constexpr size_t kBatch = 1 << 20;
  std::vector<double> buffer(kBatch);
  uint64_t count = 0;
  double sum = 0.0;
  uint64_t remaining = value_count;

  while (remaining > 0)
  {
    size_t current = static_cast<size_t>(std::min<uint64_t>(remaining, kBatch));
    input.read(reinterpret_cast<char *>(buffer.data()),
               static_cast<std::streamsize>(current * sizeof(double)));
    if (!input) die("failed to read raw file");
    for (size_t i = 0; i < current; ++i)
    {
      if (buffer[i] > threshold)
      {
        ++count;
        sum += buffer[i];
      }
    }
    remaining -= current;
  }

  CpuReference ref{};
  ref.qualified_count = count;
  ref.qualified_sum = sum;
  ref.qualified_avg = (count > 0) ? (sum / static_cast<double>(count)) : 0.0;
  return ref;
}

// ---------------------------------------------------------------------------
// CPU encoded reference: mirrors GPU progressive byte-plane comparison exactly.
// Unresolved rows (all k bytes equal threshold bytes) are NOT counted as
// qualified, matching the GPU kernel semantics.
// For all_qualified segments, all rows are qualified and all planes are used
// for value reconstruction (matching the host-side bypass in GPU results
// collection).
// ---------------------------------------------------------------------------
struct CpuEncodedReference
{
  uint64_t qualified_count = 0;
  double qualified_sum = 0.0;
  double qualified_avg = 0.0;
};

CpuEncodedReference cpu_encoded_reference(
    const exp3_real::Dataset &dataset,
    const std::vector<SegmentThresholdInfo> &threshold_info,
    const std::vector<uint32_t> &active_plane_count,
    const std::vector<double> &h_segment_base,
    const std::vector<double> &h_subcolumn_basis,
    uint64_t max_plane_count)
{
  CpuEncodedReference ref{};

  for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
  {
    const auto &segment = dataset.segments[seg];
    const auto &info = threshold_info[seg];
    uint32_t max_rounds = active_plane_count[seg];

    if (info.all_qualified)
    {
      // All rows qualified — use all planes for reconstruction (matches
      // host-side bypass in GPU results collection).
      uint64_t seg_row_count = segment.row_count;
      uint64_t seg_start = segment.row_offset;
      ref.qualified_count += seg_row_count;

      double seg_sum = h_segment_base[seg] * static_cast<double>(seg_row_count);
      for (uint64_t p = 0; p < max_plane_count; ++p)
      {
        uint64_t plane_sum = 0;
        for (uint64_t row = seg_start; row < seg_start + seg_row_count; ++row)
          plane_sum += dataset.planes[static_cast<size_t>(p)][static_cast<size_t>(row)];
        seg_sum += h_subcolumn_basis[seg * static_cast<size_t>(max_plane_count) +
                                      static_cast<size_t>(p)] *
                    static_cast<double>(plane_sum);
      }
      ref.qualified_sum += seg_sum;
    }
    else if (info.all_disqualified)
    {
      // No rows qualified.
    }
    else
    {
      // Mixed segment: progressive byte-plane comparison with capped k.
      uint64_t seg_start = segment.row_offset;
      uint64_t seg_end = seg_start + segment.row_count;
      const double *seg_basis = h_subcolumn_basis.data() + seg * max_plane_count;
      double seg_base = h_segment_base[seg];

      for (uint64_t row = seg_start; row < seg_end; ++row)
      {
        bool qualified = false;
        bool active = true;
        for (uint32_t round = 0; round < max_rounds && active; ++round)
        {
          uint8_t row_byte = dataset.planes[static_cast<size_t>(round)][static_cast<size_t>(row)];
          uint8_t thresh_byte = info.threshold_bytes[round];
          if (row_byte > thresh_byte)
          {
            qualified = true;
            active = false;
          }
          else if (row_byte < thresh_byte)
          {
            active = false;
          }
        }
        // Unresolved rows (active still true after all rounds) are NOT
        // qualified — matches GPU kernel semantics.

        if (qualified)
        {
          double val = seg_base;
          for (uint32_t round = 0; round < max_rounds; ++round)
            val += seg_basis[static_cast<size_t>(round)] *
                   static_cast<double>(dataset.planes[static_cast<size_t>(round)][static_cast<size_t>(row)]);
          ++ref.qualified_count;
          ref.qualified_sum += val;
        }
      }
    }
  }

  ref.qualified_avg = (ref.qualified_count > 0)
                           ? (ref.qualified_sum / static_cast<double>(ref.qualified_count))
                           : 0.0;
  return ref;
}

} // anonymous namespace

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
  Options opt = parse_args(argc, argv);

  cudaDeviceProp prop{};
  cuda_check(cudaSetDevice(0), "cudaSetDevice");
  cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

  // -----------------------------------------------------------------------
  // Load encoded dataset
  // -----------------------------------------------------------------------
  exp3_real::Dataset dataset = exp3_real::load_dataset(opt.dataset_path);
  uint64_t n = dataset.manifest.value_count;
  uint64_t segment_rows = dataset.manifest.segment_size;
  uint64_t max_plane_count = dataset.manifest.max_plane_count;

  if (opt.n_set && opt.n != n)
    die("--n does not match encoded manifest value_count");
  opt.n = n;
  opt.segment_rows = segment_rows;

  std::fprintf(stderr,
               "[fa] dataset=%s n=%" PRIu64 " segments=%" PRIu64
               " max_planes=%" PRIu64 "\n",
               dataset.manifest.dataset.c_str(),
               n,
               dataset.manifest.segment_count,
               max_plane_count);

  // -----------------------------------------------------------------------
  // Build segment_base and subcolumn_basis arrays (copied from Exp3 setup)
  // -----------------------------------------------------------------------
  std::vector<double> h_segment_base(static_cast<size_t>(dataset.segments.size()), 0.0);
  std::vector<double> h_subcolumn_basis(
      static_cast<size_t>(dataset.segments.size()) * static_cast<size_t>(max_plane_count), 0.0);
  for (const auto &segment : dataset.segments)
  {
    size_t idx = static_cast<size_t>(segment.segment_index);
    h_segment_base[idx] = segment.segment_base;
    for (uint64_t p = 0; p < max_plane_count; ++p)
    {
      h_subcolumn_basis[idx * static_cast<size_t>(max_plane_count) + static_cast<size_t>(p)] =
          segment.plane_basis[static_cast<size_t>(p)];
    }
  }

  // Precompute segment sums to avoid CPU row double-loop during query time correction
  std::vector<double> h_segment_sums(dataset.segments.size(), 0.0);
  for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
  {
    uint64_t seg_row_count = dataset.segments[seg].row_count;
    uint64_t seg_start = dataset.segments[seg].row_offset;
    double seg_sum = h_segment_base[seg] * static_cast<double>(seg_row_count);
    for (uint64_t p = 0; p < max_plane_count; ++p)
    {
      uint64_t plane_sum = 0;
      for (uint64_t row = seg_start; row < seg_start + seg_row_count; ++row)
        plane_sum += dataset.planes[static_cast<size_t>(p)][static_cast<size_t>(row)];
      seg_sum += h_subcolumn_basis[seg * static_cast<size_t>(max_plane_count) + static_cast<size_t>(p)] *
                 static_cast<double>(plane_sum);
    }
    h_segment_sums[seg] = seg_sum;
  }

  // -----------------------------------------------------------------------
  // Compute threshold bytes from raw threshold
  // -----------------------------------------------------------------------
  // For thresholds, we need a threshold value. The user must provide a raw
  // data file so we can compute segment min/max.  If not provided, attempt
  // auto-discovery from the encoded-root parent directory.
  std::filesystem::path raw_path;
  PrecisionInfo precision_info = infer_precision_info(opt.dataset_path, dataset);

  if (!opt.raw_path.empty())
  {
    raw_path = std::filesystem::path(opt.raw_path);
  }
  else
  {
    raw_path = std::filesystem::path(opt.dataset_path).parent_path().parent_path() /
               "dev" /
               (dataset.manifest.dataset + ".f64le.bin");
  }

  if (!std::filesystem::exists(raw_path))
    die("raw data file not found; use --raw to specify path");

  std::fprintf(stderr, "[fa] computing segment min/max from %s...\n", raw_path.c_str());
  std::vector<std::pair<double, double>> segment_min_max =
      compute_segment_min_max(raw_path, segment_rows, dataset.segments.size());

  // -----------------------------------------------------------------------
  // Compute CPU raw reference first (to determine threshold)
  // -----------------------------------------------------------------------
  // We need a threshold to compute selectivity and validation.
  // For now, read the threshold from the environment or use a default.
  // Since the user must provide a threshold as part of the experiment,
  // we use a placeholder. The real threshold will come from the --threshold
  // option. Wait — the spec says --threshold is not an option for this
  // benchmark. Instead, we use the threshold from the raw data.
  //
  // Actually, looking at the spec again, the benchmark CSV requires a
  // 'threshold' column but there's no --threshold option.  The idea is
  // that we sweep thresholds externally by varying the dataset.
  // To make the benchmark work, let's compute the CPU reference first.
  //
  // Actually, re-reading the spec: the CSV requires 'threshold' and
  // 'selectivity' columns.  These must be provided somehow.  Let's add
  // --threshold as an option.
  //
  // Wait, the spec does NOT list --threshold in the options.  The spec says:
  //   --dataset PATH, --raw PATH, --n N, --segment-rows N,
  //   --block-threads N, --iters N, --warmup N, --max-filter-planes N,
  //   --validate, --warp-strategy STR
  //
  // But the CSV output has a 'threshold' column.  Since there's no
  // --threshold, we need another way to get it.  Looking at the
  // exp4 benchmark, it uses --threshold.  For this benchmark, maybe
  // the threshold is derived from the raw data?  Or maybe it's just
  // stored in the CSV as NA?  Let me re-read the spec...
  //
  // The CSV columns include 'threshold' and 'selectivity'.  Since
  // there's no --threshold option, let me add one for practicality.
  // This makes the benchmark actually usable.

  double threshold = opt.threshold;

  // Compute threshold bytes
  std::vector<SegmentThresholdInfo> threshold_info =
      compute_threshold_bytes(dataset, segment_min_max, threshold);

  uint64_t all_qualified_segments = 0;
  uint64_t all_disqualified_segments = 0;
  uint64_t all_qualified_rows = 0;
  double host_fastpath_correction_ms = 0.0;
  double dg_host_fastpath_correction_ms = 0.0;

  for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
  {
    if (threshold_info[seg].all_qualified)
    {
      all_qualified_segments++;
      all_qualified_rows += dataset.segments[seg].row_count;
    }
    else if (threshold_info[seg].all_disqualified)
    {
      all_disqualified_segments++;
    }
  }

  // Flatten threshold bytes for device
  std::vector<uint8_t> h_threshold_flat(
      dataset.segments.size() * max_plane_count, 0);
  for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
  {
    std::memcpy(h_threshold_flat.data() + seg * max_plane_count,
                threshold_info[seg].threshold_bytes.data(),
                max_plane_count);
  }

  // Active plane count (with cap)
  std::vector<uint32_t> h_active_plane_count(dataset.segments.size());
  for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
  {
    uint32_t ac = 0;
    if (!threshold_info[seg].all_qualified && !threshold_info[seg].all_disqualified)
    {
      ac = dataset.segments[seg].active_plane_count;
      if (opt.max_filter_planes >= 0 &&
          static_cast<uint32_t>(opt.max_filter_planes) < ac)
        ac = static_cast<uint32_t>(opt.max_filter_planes);
    }
    h_active_plane_count[seg] = ac;
  }

  // -----------------------------------------------------------------------
  // Derived geometry
  // -----------------------------------------------------------------------
  Derived derived = compute_derived(n, segment_rows, opt.block_threads);

  std::fprintf(stderr,
               "[fa] grid=%" PRIu64 " tile_rows=%" PRIu64 " tiles_per_segment=%" PRIu64 "\n",
               derived.grid,
               derived.tile_rows,
               derived.tiles_per_segment);

  // -----------------------------------------------------------------------
  // GPU setup
  // -----------------------------------------------------------------------
  FaRuntimeU8Subcolumns h_subcolumns{};
  std::vector<uint8_t *> d_subcolumn_storage;
  for (uint64_t p = 0; p < max_plane_count; ++p)
  {
    const std::vector<uint8_t> &plane = dataset.planes[static_cast<size_t>(p)];
    uint8_t *ptr = nullptr;
    cuda_check(cudaMalloc(&ptr, plane.size()), "cudaMalloc(plane)");
    cuda_check(cudaMemcpy(ptr, plane.data(), plane.size(), cudaMemcpyHostToDevice),
               "cudaMemcpy(plane)");
    d_subcolumn_storage.push_back(ptr);
    h_subcolumns.ptrs[static_cast<size_t>(p)] = ptr;
  }
  for (int p = static_cast<int>(max_plane_count); p < FA_MAX_RUNTIME_PLANES; ++p)
    h_subcolumns.ptrs[p] = nullptr;

  // Device buffers
  uint8_t *d_threshold_bytes = nullptr;
  uint32_t *d_active_plane_count_dev = nullptr;
  double *d_segment_base_dev = nullptr;
  double *d_subcolumn_basis_dev = nullptr;
  uint64_t *d_block_counts = nullptr;
  double *d_block_sums = nullptr;

  cuda_check(cudaMalloc(&d_threshold_bytes, h_threshold_flat.size() * sizeof(uint8_t)),
             "cudaMalloc(threshold)");
  cuda_check(cudaMemcpy(d_threshold_bytes, h_threshold_flat.data(),
                        h_threshold_flat.size() * sizeof(uint8_t),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(threshold)");

  cuda_check(cudaMalloc(&d_active_plane_count_dev, h_active_plane_count.size() * sizeof(uint32_t)),
             "cudaMalloc(active_plane_count)");
  cuda_check(cudaMemcpy(d_active_plane_count_dev, h_active_plane_count.data(),
                        h_active_plane_count.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(active_plane_count)");

  cuda_check(cudaMalloc(&d_segment_base_dev, h_segment_base.size() * sizeof(double)),
             "cudaMalloc(segment_base)");
  cuda_check(cudaMemcpy(d_segment_base_dev, h_segment_base.data(),
                        h_segment_base.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(segment_base)");

  cuda_check(cudaMalloc(&d_subcolumn_basis_dev, h_subcolumn_basis.size() * sizeof(double)),
             "cudaMalloc(subcolumn_basis)");
  cuda_check(cudaMemcpy(d_subcolumn_basis_dev, h_subcolumn_basis.data(),
                        h_subcolumn_basis.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(subcolumn_basis)");

  cuda_check(cudaMalloc(&d_block_counts, derived.grid * sizeof(uint64_t)),
             "cudaMalloc(block_counts)");
  cuda_check(cudaMalloc(&d_block_sums, derived.grid * sizeof(double)),
             "cudaMalloc(block_sums)");

  // -----------------------------------------------------------------------
  // Warmup
  // -----------------------------------------------------------------------
  for (int i = 0; i < opt.warmup; ++i)
  {
    progressive_filter_sum_rowpack16_byte_mask<<<static_cast<int>(derived.grid),
                                                  opt.block_threads>>>(
        h_subcolumns,
        n,
        segment_rows,
        derived.tiles_per_segment,
        d_threshold_bytes,
        static_cast<int>(max_plane_count),
        static_cast<int>(max_plane_count),
        d_active_plane_count_dev,
        d_segment_base_dev,
        d_subcolumn_basis_dev,
        d_block_counts,
        d_block_sums);
  }
  cuda_check(cudaGetLastError(), "warmup launch");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

  // -----------------------------------------------------------------------
  // Timed runs
  // -----------------------------------------------------------------------
  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
  for (int i = 0; i < opt.iters; ++i)
  {
    progressive_filter_sum_rowpack16_byte_mask<<<static_cast<int>(derived.grid),
                                                  opt.block_threads>>>(
        h_subcolumns,
        n,
        segment_rows,
        derived.tiles_per_segment,
        d_threshold_bytes,
        static_cast<int>(max_plane_count),
        static_cast<int>(max_plane_count),
        d_active_plane_count_dev,
        d_segment_base_dev,
        d_subcolumn_basis_dev,
        d_block_counts,
        d_block_sums);
  }
  cuda_check(cudaGetLastError(), "timed launch");
  cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
  cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

  float ms_total = 0.0f;
  cuda_check(cudaEventElapsedTime(&ms_total, start, stop), "cudaEventElapsedTime");
  cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
  cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

  float ms_per_iter = ms_total / static_cast<float>(opt.iters);
  double seconds = static_cast<double>(ms_per_iter) / 1000.0;
  double rows_per_sec = static_cast<double>(n) / seconds;

  // -----------------------------------------------------------------------
  // Collect GPU results
  // -----------------------------------------------------------------------
  uint64_t gpu_count = 0;
  double gpu_sum = 0.0;

  {
    std::vector<uint64_t> h_block_counts(static_cast<size_t>(derived.grid));
    std::vector<double> h_block_sums(static_cast<size_t>(derived.grid));

    cuda_check(cudaMemcpy(h_block_counts.data(), d_block_counts,
                          derived.grid * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(block_counts)");
    cuda_check(cudaMemcpy(h_block_sums.data(), d_block_sums,
                          derived.grid * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(block_sums)");

    auto start_corr = std::chrono::high_resolution_clock::now();
    for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
    {
      uint64_t tile_start = seg * derived.tiles_per_segment;
      uint64_t tile_end = tile_start + derived.tiles_per_segment;

      if (threshold_info[seg].all_qualified)
      {
        gpu_count += dataset.segments[seg].row_count;
        gpu_sum += h_segment_sums[seg];
      }
      else if (threshold_info[seg].all_disqualified)
      {
        // No rows qualified; kernel produces 0 for these segments.
      }
      else
      {
        // Mixed segment: use kernel output
        for (uint64_t t = tile_start; t < tile_end; ++t)
        {
          gpu_count += h_block_counts[static_cast<size_t>(t)];
          gpu_sum += h_block_sums[static_cast<size_t>(t)];
        }
      }
    }
    auto end_corr = std::chrono::high_resolution_clock::now();
    host_fastpath_correction_ms = std::chrono::duration<double, std::milli>(end_corr - start_corr).count();
  }

  double gpu_avg = (gpu_count > 0) ? (gpu_sum / static_cast<double>(gpu_count)) : 0.0;

  // -----------------------------------------------------------------------
  // Raw FP64 GPU baseline: scan raw float64 data, count + sum WHERE x > threshold
  // -----------------------------------------------------------------------
  double raw_baseline_ms_per_iter = 0.0;
  double raw_baseline_rows_per_sec = 0.0;
  uint64_t raw_baseline_count = 0;
  double raw_baseline_sum = 0.0;

  double *d_raw_data = nullptr;

  {
    // Load raw FP64 data to GPU
    uint64_t raw_file_size = std::filesystem::file_size(raw_path);
    if (raw_file_size % sizeof(double) != 0) die("raw file size not aligned to FP64");
    uint64_t raw_value_count = raw_file_size / sizeof(double);
    if (raw_value_count != n) die("raw file value count does not match encoded manifest");

    cuda_check(cudaMalloc(&d_raw_data, raw_file_size), "cudaMalloc(raw_data)");

    // Read raw data in chunks and upload
    {
      std::ifstream raw_input(raw_path, std::ios::binary);
      if (!raw_input) die("failed to open raw file for GPU upload");
      constexpr size_t kChunkSize = 256ull * 1024ull * 1024ull; // 256 MB
      std::vector<double> chunk(kChunkSize / sizeof(double));
      uint64_t remaining = raw_file_size;
      uint64_t offset = 0;
      while (remaining > 0)
      {
        size_t to_read = static_cast<size_t>(std::min<uint64_t>(remaining, kChunkSize));
        raw_input.read(reinterpret_cast<char *>(chunk.data()), static_cast<std::streamsize>(to_read));
        if (!raw_input) die("failed to read raw file chunk");
        cuda_check(cudaMemcpy(reinterpret_cast<uint8_t *>(d_raw_data) + offset,
                               chunk.data(), to_read, cudaMemcpyHostToDevice),
                   "cudaMemcpy(raw_data chunk)");
        offset += to_read;
        remaining -= to_read;
      }
    }

    // Allocate raw baseline output buffers
    int raw_grid = static_cast<int>((n + static_cast<uint64_t>(opt.block_threads) - 1) /
                                     static_cast<uint64_t>(opt.block_threads));
    uint64_t *d_raw_block_counts = nullptr;
    double *d_raw_block_sums = nullptr;
    cuda_check(cudaMalloc(&d_raw_block_counts, static_cast<size_t>(raw_grid) * sizeof(uint64_t)),
               "cudaMalloc(raw_block_counts)");
    cuda_check(cudaMalloc(&d_raw_block_sums, static_cast<size_t>(raw_grid) * sizeof(double)),
               "cudaMalloc(raw_block_sums)");

    // Warmup
    for (int i = 0; i < std::min(opt.warmup, 3); ++i)
    {
      raw_fp64_filter_sum<<<raw_grid, opt.block_threads>>>(
          d_raw_data, n, threshold, d_raw_block_counts, d_raw_block_sums);
    }
    cuda_check(cudaGetLastError(), "raw baseline warmup launch");
    cuda_check(cudaDeviceSynchronize(), "raw baseline warmup sync");

    // Timed runs
    cudaEvent_t raw_start{}, raw_stop{};
    cuda_check(cudaEventCreate(&raw_start), "cudaEventCreate(raw_start)");
    cuda_check(cudaEventCreate(&raw_stop), "cudaEventCreate(raw_stop)");

    cuda_check(cudaEventRecord(raw_start), "cudaEventRecord(raw_start)");
    for (int i = 0; i < opt.iters; ++i)
    {
      raw_fp64_filter_sum<<<raw_grid, opt.block_threads>>>(
          d_raw_data, n, threshold, d_raw_block_counts, d_raw_block_sums);
    }
    cuda_check(cudaGetLastError(), "raw baseline timed launch");
    cuda_check(cudaEventRecord(raw_stop), "cudaEventRecord(raw_stop)");
    cuda_check(cudaEventSynchronize(raw_stop), "raw baseline sync");

    float raw_ms_total = 0.0f;
    cuda_check(cudaEventElapsedTime(&raw_ms_total, raw_start, raw_stop), "raw baseline elapsed");
    cuda_check(cudaEventDestroy(raw_start), "cudaEventDestroy(raw_start)");
    cuda_check(cudaEventDestroy(raw_stop), "cudaEventDestroy(raw_stop)");

    raw_baseline_ms_per_iter = static_cast<double>(raw_ms_total) / static_cast<double>(opt.iters);
    raw_baseline_rows_per_sec = static_cast<double>(n) / (raw_baseline_ms_per_iter / 1000.0);

    // Collect raw baseline results
    std::vector<uint64_t> h_raw_block_counts(static_cast<size_t>(raw_grid));
    std::vector<double> h_raw_block_sums(static_cast<size_t>(raw_grid));
    cuda_check(cudaMemcpy(h_raw_block_counts.data(), d_raw_block_counts,
                          static_cast<size_t>(raw_grid) * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(raw_block_counts)");
    cuda_check(cudaMemcpy(h_raw_block_sums.data(), d_raw_block_sums,
                          static_cast<size_t>(raw_grid) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(raw_block_sums)");

    for (int i = 0; i < raw_grid; ++i)
    {
      raw_baseline_count += h_raw_block_counts[static_cast<size_t>(i)];
      raw_baseline_sum += h_raw_block_sums[static_cast<size_t>(i)];
    }

    std::fprintf(stderr,
                 "[fa] raw baseline: count=%" PRIu64 " sum=%.17g ms=%.3f rows/s=%.3e\n",
                 raw_baseline_count, raw_baseline_sum,
                 raw_baseline_ms_per_iter, raw_baseline_rows_per_sec);

    cuda_check(cudaFree(d_raw_block_sums), "cudaFree(raw_block_sums)");
    cuda_check(cudaFree(d_raw_block_counts), "cudaFree(raw_block_counts)");
    // Note: d_raw_data is NOT freed here — it's needed for deferred gather baseline
  }

  // -----------------------------------------------------------------------
  // Deferred gather baseline: filter → compact → gather+aggregate
  // -----------------------------------------------------------------------
  double dg_ms_per_iter = 0.0;
  double dg_rows_per_sec = 0.0;
  uint64_t dg_count = 0;
  double dg_sum = 0.0;
  double dg_avg = 0.0;
  double speedup_fused_vs_dg = 0.0;

  {
    // Reuse d_raw_data allocated and uploaded in the raw baseline section
    // -----------------------------------------------------------------------

    // Allocate deferred gather buffers
    uint64_t *d_dg_filter_counts = nullptr;
    uint64_t *d_dg_block_offsets = nullptr;
    uint64_t *d_qualified_indices = nullptr;
    uint64_t *d_dg_gather_counts = nullptr;
    double *d_dg_gather_sums = nullptr;

    cuda_check(cudaMalloc(&d_dg_filter_counts, derived.grid * sizeof(uint64_t)),
               "cudaMalloc(dg_filter_counts)");
    cuda_check(cudaMalloc(&d_dg_block_offsets, derived.grid * sizeof(uint64_t)),
               "cudaMalloc(dg_block_offsets)");
    cuda_check(cudaMalloc(&d_qualified_indices, n * sizeof(uint64_t)),
               "cudaMalloc(qualified_indices)");
    // Gather grid: will be set after we know qualified_count
    // For now, allocate with max grid size
    int gather_grid = static_cast<int>((n + static_cast<uint64_t>(opt.block_threads) - 1) /
                                       static_cast<uint64_t>(opt.block_threads));
    cuda_check(cudaMalloc(&d_dg_gather_counts, static_cast<size_t>(gather_grid) * sizeof(uint64_t)),
               "cudaMalloc(dg_gather_counts)");
    cuda_check(cudaMalloc(&d_dg_gather_sums, static_cast<size_t>(gather_grid) * sizeof(double)),
               "cudaMalloc(dg_gather_sums)");

    // -------------------------------------------------------------------
    // Phase 1: Filter count pass (warmup)
    // -------------------------------------------------------------------
    for (int i = 0; i < std::min(opt.warmup, 3); ++i)
    {
      progressive_filter_count_rowpack16_byte_mask<<<static_cast<int>(derived.grid),
                                                       opt.block_threads>>>(
          h_subcolumns, n, segment_rows, derived.tiles_per_segment,
          d_threshold_bytes, static_cast<int>(max_plane_count),
          static_cast<int>(max_plane_count),
          d_active_plane_count_dev, d_dg_filter_counts);
    }
    cuda_check(cudaGetLastError(), "dg filter count warmup");
    cuda_check(cudaDeviceSynchronize(), "dg filter count warmup sync");

    // -------------------------------------------------------------------
    // Phase 1: Filter count pass (timed)
    // -------------------------------------------------------------------
    cudaEvent_t dg_fc_start{}, dg_fc_stop{};
    cuda_check(cudaEventCreate(&dg_fc_start), "cudaEventCreate(dg_fc_start)");
    cuda_check(cudaEventCreate(&dg_fc_stop), "cudaEventCreate(dg_fc_stop)");
    cuda_check(cudaEventRecord(dg_fc_start), "cudaEventRecord(dg_fc_start)");
    for (int i = 0; i < opt.iters; ++i)
    {
      progressive_filter_count_rowpack16_byte_mask<<<static_cast<int>(derived.grid),
                                                       opt.block_threads>>>(
          h_subcolumns, n, segment_rows, derived.tiles_per_segment,
          d_threshold_bytes, static_cast<int>(max_plane_count),
          static_cast<int>(max_plane_count),
          d_active_plane_count_dev, d_dg_filter_counts);
    }
    cuda_check(cudaGetLastError(), "dg filter count timed");
    cuda_check(cudaEventRecord(dg_fc_stop), "cudaEventRecord(dg_fc_stop)");
    cuda_check(cudaEventSynchronize(dg_fc_stop), "dg filter count sync");
    float dg_fc_ms_total = 0.0f;
    cuda_check(cudaEventElapsedTime(&dg_fc_ms_total, dg_fc_start, dg_fc_stop), "dg filter count elapsed");
    cuda_check(cudaEventDestroy(dg_fc_start), "cudaEventDestroy(dg_fc_start)");
    cuda_check(cudaEventDestroy(dg_fc_stop), "cudaEventDestroy(dg_fc_stop)");
    double dg_fc_ms_per_iter = static_cast<double>(dg_fc_ms_total) / static_cast<double>(opt.iters);

    auto dg_host_start = std::chrono::high_resolution_clock::now();

    // -------------------------------------------------------------------
    // Collect filter counts and compute prefix sum (host-side)
    // -------------------------------------------------------------------
    std::vector<uint64_t> h_dg_filter_counts(static_cast<size_t>(derived.grid));
    cuda_check(cudaMemcpy(h_dg_filter_counts.data(), d_dg_filter_counts,
                          derived.grid * sizeof(uint64_t), cudaMemcpyDeviceToHost),
               "cudaMemcpy(dg_filter_counts)");

    // Handle all_qualified/all_disqualified segments (same as fused kernel)
    uint64_t dg_qualified_count = 0;
    for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
    {
      if (threshold_info[seg].all_qualified)
      {
        dg_qualified_count += dataset.segments[seg].row_count;
      }
      else if (threshold_info[seg].all_disqualified)
      {
        // No rows qualified
      }
      else
      {
        uint64_t tile_start = seg * derived.tiles_per_segment;
        uint64_t tile_end = tile_start + derived.tiles_per_segment;
        for (uint64_t t = tile_start; t < tile_end; ++t)
          dg_qualified_count += h_dg_filter_counts[static_cast<size_t>(t)];
      }
    }

    // Compute exclusive prefix sum of block counts for scatter offsets
    std::vector<uint64_t> h_dg_block_offsets(static_cast<size_t>(derived.grid));
    h_dg_block_offsets[0] = 0;
    for (size_t i = 1; i < static_cast<size_t>(derived.grid); ++i)
      h_dg_block_offsets[i] = h_dg_block_offsets[i - 1] + h_dg_filter_counts[i - 1];

    // Handle all_qualified segments: add their row indices on host side
    // First, adjust offsets for all_qualified segments
    // We need to fill in the qualified indices for all_qualified segments
    // and adjust the scatter offsets accordingly.
    // For simplicity, we handle this by:
    // 1. Computing the total count from all_qualified segments
    // 2. Filling those indices directly into h_qualified_indices
    // 3. Adjusting block offsets for mixed segments to account for all_qualified indices

    // Count all_qualified rows to determine offset adjustment
    uint64_t all_qualified_offset = 0;
    for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
    {
      if (threshold_info[seg].all_qualified)
        all_qualified_offset += dataset.segments[seg].row_count;
    }

    // Shift block offsets for mixed segments by all_qualified_offset
    for (size_t i = 0; i < static_cast<size_t>(derived.grid); ++i)
      h_dg_block_offsets[i] += all_qualified_offset;

    cuda_check(cudaMemcpy(d_dg_block_offsets, h_dg_block_offsets.data(),
                          derived.grid * sizeof(uint64_t), cudaMemcpyHostToDevice),
               "cudaMemcpy(dg_block_offsets)");

    // Fill all_qualified indices on host side, then upload
    std::vector<uint64_t> h_qualified_indices(n, 0); // worst case size
    uint64_t idx_offset = 0;
    for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
    {
      if (threshold_info[seg].all_qualified)
      {
        uint64_t seg_start = dataset.segments[seg].row_offset;
        uint64_t seg_count = dataset.segments[seg].row_count;
        for (uint64_t r = 0; r < seg_count; ++r)
          h_qualified_indices[idx_offset++] = seg_start + r;
      }
    }

    // Upload all_qualified indices (done once, now timed inside host path)
    cuda_check(cudaMemcpy(d_qualified_indices, h_qualified_indices.data(),
                          idx_offset * sizeof(uint64_t), cudaMemcpyHostToDevice),
               "cudaMemcpy(all_qualified_indices)");

    auto dg_host_end = std::chrono::high_resolution_clock::now();
    dg_host_fastpath_correction_ms = std::chrono::duration<double, std::milli>(dg_host_end - dg_host_start).count();

    // -------------------------------------------------------------------
    // Phase 2: Filter scatter pass (warmup)
    // -------------------------------------------------------------------
    for (int i = 0; i < std::min(opt.warmup, 3); ++i)
    {
      progressive_filter_scatter_rowpack16_byte_mask<<<static_cast<int>(derived.grid),
                                                        opt.block_threads>>>(
          h_subcolumns, n, segment_rows, derived.tiles_per_segment,
          d_threshold_bytes, static_cast<int>(max_plane_count),
          static_cast<int>(max_plane_count),
          d_active_plane_count_dev, d_dg_block_offsets, d_qualified_indices);
    }
    cuda_check(cudaGetLastError(), "dg scatter warmup");
    cuda_check(cudaDeviceSynchronize(), "dg scatter warmup sync");

    // -------------------------------------------------------------------
    // Phase 2: Filter scatter pass (timed)
    // -------------------------------------------------------------------
    cudaEvent_t dg_sc_start{}, dg_sc_stop{};
    cuda_check(cudaEventCreate(&dg_sc_start), "cudaEventCreate(dg_sc_start)");
    cuda_check(cudaEventCreate(&dg_sc_stop), "cudaEventCreate(dg_sc_stop)");
    cuda_check(cudaEventRecord(dg_sc_start), "cudaEventRecord(dg_sc_start)");
    for (int i = 0; i < opt.iters; ++i)
    {
      progressive_filter_scatter_rowpack16_byte_mask<<<static_cast<int>(derived.grid),
                                                        opt.block_threads>>>(
          h_subcolumns, n, segment_rows, derived.tiles_per_segment,
          d_threshold_bytes, static_cast<int>(max_plane_count),
          static_cast<int>(max_plane_count),
          d_active_plane_count_dev, d_dg_block_offsets, d_qualified_indices);
    }
    cuda_check(cudaGetLastError(), "dg scatter timed");
    cuda_check(cudaEventRecord(dg_sc_stop), "cudaEventRecord(dg_sc_stop)");
    cuda_check(cudaEventSynchronize(dg_sc_stop), "dg scatter sync");
    float dg_sc_ms_total = 0.0f;
    cuda_check(cudaEventElapsedTime(&dg_sc_ms_total, dg_sc_start, dg_sc_stop), "dg scatter elapsed");
    cuda_check(cudaEventDestroy(dg_sc_start), "cudaEventDestroy(dg_sc_start)");
    cuda_check(cudaEventDestroy(dg_sc_stop), "cudaEventDestroy(dg_sc_stop)");
    double dg_sc_ms_per_iter = static_cast<double>(dg_sc_ms_total) / static_cast<double>(opt.iters);



    // -------------------------------------------------------------------
    // Phase 3: Gather + aggregate (warmup)
    // -------------------------------------------------------------------
    int dg_gather_grid = static_cast<int>((dg_qualified_count + static_cast<uint64_t>(opt.block_threads) - 1) /
                                           static_cast<uint64_t>(opt.block_threads));
    if (dg_gather_grid < 1) dg_gather_grid = 1;

    for (int i = 0; i < std::min(opt.warmup, 3); ++i)
    {
      raw_fp64_gather_sum<<<dg_gather_grid, opt.block_threads>>>(
          d_raw_data, d_qualified_indices, dg_qualified_count,
          d_dg_gather_counts, d_dg_gather_sums);
    }
    cuda_check(cudaGetLastError(), "dg gather warmup");
    cuda_check(cudaDeviceSynchronize(), "dg gather warmup sync");

    // -------------------------------------------------------------------
    // Phase 3: Gather + aggregate (timed)
    // -------------------------------------------------------------------
    cudaEvent_t dg_ga_start{}, dg_ga_stop{};
    cuda_check(cudaEventCreate(&dg_ga_start), "cudaEventCreate(dg_ga_start)");
    cuda_check(cudaEventCreate(&dg_ga_stop), "cudaEventCreate(dg_ga_stop)");
    cuda_check(cudaEventRecord(dg_ga_start), "cudaEventRecord(dg_ga_start)");
    for (int i = 0; i < opt.iters; ++i)
    {
      raw_fp64_gather_sum<<<dg_gather_grid, opt.block_threads>>>(
          d_raw_data, d_qualified_indices, dg_qualified_count,
          d_dg_gather_counts, d_dg_gather_sums);
    }
    cuda_check(cudaGetLastError(), "dg gather timed");
    cuda_check(cudaEventRecord(dg_ga_stop), "cudaEventRecord(dg_ga_stop)");
    cuda_check(cudaEventSynchronize(dg_ga_stop), "dg gather sync");
    float dg_ga_ms_total = 0.0f;
    cuda_check(cudaEventElapsedTime(&dg_ga_ms_total, dg_ga_start, dg_ga_stop), "dg gather elapsed");
    cuda_check(cudaEventDestroy(dg_ga_start), "cudaEventDestroy(dg_ga_start)");
    cuda_check(cudaEventDestroy(dg_ga_stop), "cudaEventDestroy(dg_ga_stop)");
    double dg_ga_ms_per_iter = static_cast<double>(dg_ga_ms_total) / static_cast<double>(opt.iters);

    // -------------------------------------------------------------------
    // Collect gather results
    // -------------------------------------------------------------------
    std::vector<uint64_t> h_dg_gather_counts(static_cast<size_t>(dg_gather_grid));
    std::vector<double> h_dg_gather_sums(static_cast<size_t>(dg_gather_grid));
    cuda_check(cudaMemcpy(h_dg_gather_counts.data(), d_dg_gather_counts,
                          static_cast<size_t>(dg_gather_grid) * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(dg_gather_counts)");
    cuda_check(cudaMemcpy(h_dg_gather_sums.data(), d_dg_gather_sums,
                          static_cast<size_t>(dg_gather_grid) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(dg_gather_sums)");

    uint64_t dg_gather_count = 0;
    double dg_gather_sum = 0.0;
    for (int i = 0; i < dg_gather_grid; ++i)
    {
      dg_gather_count += h_dg_gather_counts[static_cast<size_t>(i)];
      dg_gather_sum += h_dg_gather_sums[static_cast<size_t>(i)];
    }

    // Add all_qualified segment sums (from raw FP64 data)
    // The gather kernel only processes mixed segments' qualified indices.
    // For all_qualified segments, we need to add their raw FP64 sums.
    // We already have the raw data on host from the CPU reference.
    // Actually, the gather kernel processes ALL qualified indices (including
    // all_qualified ones that we uploaded). So dg_gather_count should equal
    // dg_qualified_count. Let's verify.
    dg_count = dg_gather_count;
    dg_sum = dg_gather_sum;
    dg_avg = (dg_count > 0) ? (dg_sum / static_cast<double>(dg_count)) : 0.0;

    dg_ms_per_iter = dg_fc_ms_per_iter + dg_sc_ms_per_iter + dg_ga_ms_per_iter;
    dg_rows_per_sec = (dg_ms_per_iter > 0) ? (static_cast<double>(n) / (dg_ms_per_iter / 1000.0)) : 0.0;
    speedup_fused_vs_dg = (dg_rows_per_sec > 0) ? (rows_per_sec / dg_rows_per_sec) : 0.0;

    std::fprintf(stderr,
                 "[fa] deferred gather: count=%" PRIu64 " sum=%.17g ms=%.3f rows/s=%.3e\n"
                 "[fa]   filter_count=%.3fms filter_scatter=%.3fms gather=%.3fms total=%.3fms\n"
                 "[fa]   speedup fused_vs_dg=%.2fx\n",
                 dg_count, dg_sum, dg_ms_per_iter, dg_rows_per_sec,
                 dg_fc_ms_per_iter, dg_sc_ms_per_iter, dg_ga_ms_per_iter, dg_ms_per_iter,
                 speedup_fused_vs_dg);

    // Validation: dg_count must match gpu_count (same encoded predicate)
    if (opt.validate && dg_count != gpu_count)
    {
      std::fprintf(stderr, "[fa] WARNING: dg_count=%" PRIu64 " != gpu_count=%" PRIu64 "\n",
                    dg_count, gpu_count);
    }

    cuda_check(cudaFree(d_dg_gather_sums), "cudaFree(dg_gather_sums)");
    cuda_check(cudaFree(d_dg_gather_counts), "cudaFree(dg_gather_counts)");
    cuda_check(cudaFree(d_qualified_indices), "cudaFree(qualified_indices)");
    cuda_check(cudaFree(d_dg_block_offsets), "cudaFree(dg_block_offsets)");
    cuda_check(cudaFree(d_dg_filter_counts), "cudaFree(dg_filter_counts)");
    cuda_check(cudaFree(d_raw_data), "cudaFree(raw_data for dg)");
  }

// -----------------------------------------------------------------------
  // CPU encoded reference (hard correctness gate)
  // -----------------------------------------------------------------------
  CpuEncodedReference enc_ref{};
  bool has_enc_ref = false;
  bool enc_count_ok = false;
  bool enc_sum_ok = false;

  if (opt.validate)
  {
    std::fprintf(stderr, "[fa] running CPU encoded reference...\n");
    enc_ref = cpu_encoded_reference(dataset, threshold_info, h_active_plane_count,
                                    h_segment_base, h_subcolumn_basis, max_plane_count);
    has_enc_ref = true;

    // Hard gate: GPU must match encoded progressive semantics exactly
    enc_count_ok = (gpu_count == enc_ref.qualified_count);

    double enc_sum_tolerance = 1e-9;
    double enc_sum_abs_err = std::fabs(gpu_sum - enc_ref.qualified_sum);
    double enc_sum_rel_err = (std::fabs(enc_ref.qualified_sum) > 1e-30)
                                 ? (enc_sum_abs_err / std::fabs(enc_ref.qualified_sum))
                                 : enc_sum_abs_err;
    enc_sum_ok = (enc_sum_rel_err < enc_sum_tolerance);

    double enc_avg_abs_err = std::fabs(gpu_avg - enc_ref.qualified_avg);
    double enc_avg_rel_err = (std::fabs(enc_ref.qualified_avg) > 1e-30)
                                 ? (enc_avg_abs_err / std::fabs(enc_ref.qualified_avg))
                                 : enc_avg_abs_err;

    std::fprintf(stderr,
                 "[fa] encoded gate: gpu_count=%" PRIu64 " enc_count=%" PRIu64 " count_ok=%s\n"
                 "[fa]               gpu_sum=%.17g enc_sum=%.17g sum_rel_err=%.3g sum_ok=%s\n"
                 "[fa]               gpu_avg=%.17g enc_avg=%.17g avg_rel_err=%.3g\n",
                 gpu_count, enc_ref.qualified_count, enc_count_ok ? "PASS" : "FAIL",
                 gpu_sum, enc_ref.qualified_sum, enc_sum_rel_err, enc_sum_ok ? "PASS" : "FAIL",
                 gpu_avg, enc_ref.qualified_avg, enc_avg_rel_err);

    if (!enc_count_ok)
      die("VALIDATION FAILED: gpu_count != cpu_encoded_count");
    if (!enc_sum_ok)
      die("VALIDATION FAILED: gpu_sum exceeds tolerance vs cpu_encoded_sum");
  }

  // -----------------------------------------------------------------------
  // CPU raw reference (informational: quantifies capped-k approximation drift)
  // -----------------------------------------------------------------------
  CpuReference cpu_ref{};
  bool has_cpu_ref = false;

  if (opt.validate)
  {
    std::fprintf(stderr, "[fa] running CPU raw reference from %s...\n", raw_path.c_str());
    cpu_ref = cpu_raw_reference(raw_path, threshold);
    has_cpu_ref = true;

    uint64_t raw_count_abs_err = (gpu_count >= cpu_ref.qualified_count)
                                      ? (gpu_count - cpu_ref.qualified_count)
                                      : (cpu_ref.qualified_count - gpu_count);
    double raw_count_rel_err = static_cast<double>(raw_count_abs_err) /
                               static_cast<double>(std::max<uint64_t>(cpu_ref.qualified_count, 1ull));
    double raw_sum_abs_err = std::fabs(gpu_sum - cpu_ref.qualified_sum);
    double raw_sum_rel_err = (std::fabs(cpu_ref.qualified_sum) > 1e-30)
                                 ? (raw_sum_abs_err / std::fabs(cpu_ref.qualified_sum))
                                 : raw_sum_abs_err;
    double raw_avg_abs_err = std::fabs(gpu_avg - cpu_ref.qualified_avg);
    double raw_avg_rel_err = (std::fabs(cpu_ref.qualified_avg) > 1e-30)
                                 ? (raw_avg_abs_err / std::fabs(cpu_ref.qualified_avg))
                                 : raw_avg_abs_err;

    std::fprintf(stderr,
                 "[fa] raw FP64 (informational): gpu_count=%" PRIu64 " raw_count=%" PRIu64
                 " count_rel_err=%.6g\n"
                 "[fa]                             gpu_sum=%.17g raw_sum=%.17g sum_rel_err=%.6g\n"
                 "[fa]                             gpu_avg=%.17g raw_avg=%.17g avg_rel_err=%.6g\n",
                 gpu_count, cpu_ref.qualified_count, raw_count_rel_err,
                 gpu_sum, cpu_ref.qualified_sum, raw_sum_rel_err,
                 gpu_avg, cpu_ref.qualified_avg, raw_avg_rel_err);
  }

  // -----------------------------------------------------------------------
  // Compute metrics for CSV
  // -----------------------------------------------------------------------
  double selectivity = (n == 0) ? 0.0 : (static_cast<double>(gpu_count) / static_cast<double>(n));
  double logical_bytes = static_cast<double>(gpu_count) * static_cast<double>(max_plane_count);

  // Encoded reference error metrics (hard gate)
  uint64_t enc_count_abs_err = has_enc_ref
      ? ((gpu_count >= enc_ref.qualified_count)
              ? (gpu_count - enc_ref.qualified_count)
              : (enc_ref.qualified_count - gpu_count))
      : 0;
  double enc_count_rel_err = has_enc_ref
      ? (static_cast<double>(enc_count_abs_err) /
         static_cast<double>(std::max<uint64_t>(enc_ref.qualified_count, 1ull)))
      : 0.0;
  double enc_sum_abs_err = has_enc_ref ? std::fabs(gpu_sum - enc_ref.qualified_sum) : 0.0;
  double enc_sum_rel_err = (has_enc_ref && std::fabs(enc_ref.qualified_sum) > 1e-30)
                               ? (enc_sum_abs_err / std::fabs(enc_ref.qualified_sum))
                               : enc_sum_abs_err;
  double enc_avg_abs_err = has_enc_ref ? std::fabs(gpu_avg - enc_ref.qualified_avg) : 0.0;
  double enc_avg_rel_err = (has_enc_ref && std::fabs(enc_ref.qualified_avg) > 1e-30)
                               ? (enc_avg_abs_err / std::fabs(enc_ref.qualified_avg))
                               : enc_avg_abs_err;

  // Raw FP64 error metrics (informational)
  uint64_t raw_count_abs_err = has_cpu_ref
      ? ((gpu_count >= cpu_ref.qualified_count)
              ? (gpu_count - cpu_ref.qualified_count)
              : (cpu_ref.qualified_count - gpu_count))
      : 0;
  double raw_count_rel_err = has_cpu_ref
      ? (static_cast<double>(raw_count_abs_err) /
         static_cast<double>(std::max<uint64_t>(cpu_ref.qualified_count, 1ull)))
      : 0.0;
  double raw_sum_abs_err = has_cpu_ref ? std::fabs(gpu_sum - cpu_ref.qualified_sum) : 0.0;
  double raw_sum_rel_err = (has_cpu_ref && std::fabs(cpu_ref.qualified_sum) > 1e-30)
                               ? (raw_sum_abs_err / std::fabs(cpu_ref.qualified_sum))
                               : raw_sum_abs_err;
  double raw_avg_abs_err = has_cpu_ref ? std::fabs(gpu_avg - cpu_ref.qualified_avg) : 0.0;
  double raw_avg_rel_err = (has_cpu_ref && std::fabs(cpu_ref.qualified_avg) > 1e-30)
                               ? (raw_avg_abs_err / std::fabs(cpu_ref.qualified_avg))
                               : raw_avg_abs_err;

  // -----------------------------------------------------------------------
  // Write CSV
  // -----------------------------------------------------------------------
  std::FILE *f = std::fopen(opt.csv_path.c_str(), "wb");
  if (!f) die("failed to open CSV output");

  std::fprintf(f,
               "experiment,dataset,artifact_root,precision_mode,precision_decimals,"
               "threshold,selectivity,"
               "n,iters,warmup,ms_per_iter,"
               "gpu_count,"
               "cpu_enc_count,cpu_raw_count,"
               "gpu_sum,"
               "cpu_enc_sum,cpu_raw_sum,"
               "gpu_avg,"
               "cpu_enc_avg,cpu_raw_avg,"
               "enc_count_abs_err,enc_count_rel_err,enc_sum_abs_err,enc_sum_rel_err,"
               "enc_avg_abs_err,enc_avg_rel_err,"
               "raw_count_abs_err,raw_count_rel_err,raw_sum_abs_err,raw_sum_rel_err,"
               "raw_avg_abs_err,raw_avg_rel_err,"
               "avg_planes_read_per_total_row,max_planes_read,"
               "logical_bytes,rows_per_sec,"
"raw_baseline_ms_per_iter,raw_baseline_rows_per_sec,"
                "raw_baseline_count,raw_baseline_sum,"
                "speedup_vs_raw,"
                "dg_count,dg_sum,dg_avg,"
                "dg_ms_per_iter,dg_rows_per_sec,"
                "speedup_fused_vs_dg,"
                "max_filter_planes,validated,device,job_id,"
                "filter_aggregate_strategy,"
                "all_qualified_segments,"
                "all_disqualified_segments,"
                "all_qualified_rows,"
                "host_fastpath_correction_ms,"
                "dg_host_fastpath_correction_ms\n");

  std::string validated_csv = csv_bool_or_na(opt.validate,
      opt.validate && has_enc_ref && enc_count_ok && enc_sum_ok);
  std::string max_filter_planes_csv = (opt.max_filter_planes >= 0)
                                          ? std::to_string(opt.max_filter_planes) : "NA";
  std::string job_id_csv = slurm_job_id_or_na();
  std::string device_name_csv = prop.name;

  // Average planes read per total row = (qualified_count * max_rounds) / n
  double avg_planes_per_row = (n > 0)
      ? (static_cast<double>(gpu_count) * static_cast<double>(max_plane_count) /
         static_cast<double>(n))
      : 0.0;

  // Max planes read (across all segments before optional cap)
  uint32_t global_max_planes = 0;
  for (auto &a : h_active_plane_count)
    global_max_planes = std::max(global_max_planes, a);
  std::string max_planes_read_csv = std::to_string(global_max_planes);

  double speedup_vs_raw = (raw_baseline_rows_per_sec > 0)
      ? (rows_per_sec / raw_baseline_rows_per_sec) : 0.0;

  std::fprintf(f,
               "exp4_filter_aggregate,%s,%s,%s,%s,"
               "%.17g,%.17g,"
               "%" PRIu64 ",%d,%d,%.6f,"
               "%" PRIu64 ","
               "%" PRIu64 ",%" PRIu64 ","
               "%.17g,"
               "%.17g,%.17g,"
               "%.17g,"
               "%.17g,%.17g,"
               "%" PRIu64 ",%.17g,%.17g,%.17g,"
               "%.17g,%.17g,"
               "%" PRIu64 ",%.17g,%.17g,%.17g,"
               "%.17g,%.17g,"
               "%.17g,%s,"
               "%.0f,%.17g,"
"%.6f,%.17g,"
                "%" PRIu64 ",%.17g,"
                "%.6f,"
                "%" PRIu64 ",%.17g,%.17g,"
                "%.6f,%.17g,"
                "%.6f,"
                "%s,%s,%s,%s,"
                "%s,"
                "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.6f,%.6f\n",
               dataset.manifest.dataset.c_str(),
               opt.dataset_path.c_str(),
               precision_info.mode.c_str(),
               precision_info.decimals.c_str(),
               threshold,
               selectivity,
               n,
               opt.iters,
               opt.warmup,
               static_cast<double>(ms_per_iter),
               gpu_count,
               enc_ref.qualified_count,
               cpu_ref.qualified_count,
               gpu_sum,
               enc_ref.qualified_sum,
               cpu_ref.qualified_sum,
               gpu_avg,
               enc_ref.qualified_avg,
               cpu_ref.qualified_avg,
               enc_count_abs_err,
               enc_count_rel_err,
               enc_sum_abs_err,
               enc_sum_rel_err,
               enc_avg_abs_err,
               enc_avg_rel_err,
               raw_count_abs_err,
               raw_count_rel_err,
               raw_sum_abs_err,
               raw_sum_rel_err,
               raw_avg_abs_err,
               raw_avg_rel_err,
               avg_planes_per_row,
               max_planes_read_csv.c_str(),
               logical_bytes,
               rows_per_sec,
               raw_baseline_ms_per_iter,
               raw_baseline_rows_per_sec,
raw_baseline_count,
                raw_baseline_sum,
                speedup_vs_raw,
                dg_count,
                dg_sum,
                dg_avg,
                dg_ms_per_iter,
                dg_rows_per_sec,
                speedup_fused_vs_dg,
                max_filter_planes_csv.c_str(),
               validated_csv.c_str(),
               device_name_csv.c_str(),
               job_id_csv.c_str(),
               opt.warp_strategy.c_str(),
               all_qualified_segments,
               all_disqualified_segments,
               all_qualified_rows,
               host_fastpath_correction_ms,
               dg_host_fastpath_correction_ms);

  std::fclose(f);

  std::fprintf(stderr,
               "[fa] result: gpu_count=%" PRIu64 " gpu_sum=%.17g ms=%.3f rows/s=%.3e\n"
               "[fa]          raw_baseline: ms=%.3f rows/s=%.3e speedup=%.2fx\n",
               gpu_count, gpu_sum, ms_per_iter, rows_per_sec,
               raw_baseline_ms_per_iter, raw_baseline_rows_per_sec, speedup_vs_raw);

  // -----------------------------------------------------------------------
  // Cleanup
  // -----------------------------------------------------------------------
  cuda_check(cudaFree(d_block_sums), "cudaFree(block_sums)");
  cuda_check(cudaFree(d_block_counts), "cudaFree(block_counts)");
  cuda_check(cudaFree(d_subcolumn_basis_dev), "cudaFree(subcolumn_basis)");
  cuda_check(cudaFree(d_segment_base_dev), "cudaFree(segment_base)");
  cuda_check(cudaFree(d_active_plane_count_dev), "cudaFree(active_plane_count)");
  cuda_check(cudaFree(d_threshold_bytes), "cudaFree(threshold_bytes)");
  for (uint8_t *ptr : d_subcolumn_storage)
    cuda_check(cudaFree(ptr), "cudaFree(plane)");

  return 0;
}
