#include "byteplane_count_gt.hpp"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "exp3_common.cuh"
#include "exp3_real_data_layout.hpp"
#include "exp4_kernels_filter.cuh"

namespace byteplane_scan
{
namespace
{

constexpr std::uint8_t kSegmentModeMixed = 0;
constexpr std::uint8_t kSegmentModeAllQualified = 1;
constexpr std::uint8_t kSegmentModeAllDisqualified = 2;

struct SegmentThresholdInfo
{
  std::vector<std::uint8_t> threshold_bytes;
  bool all_qualified = false;
  bool all_disqualified = false;
  std::uint64_t threshold_combined = 0;
};

struct GpuBlockStats
{
  unsigned long long q;
  unsigned long long d;
  unsigned long long u;
  unsigned long long bytes_read;
};

struct GpuResidentArtifact
{
  Exp3RuntimeU8Subcolumns subcolumns{};
  std::vector<std::uint8_t *> plane_storage;
  std::uint64_t n = 0;
  std::uint64_t tiles_per_segment = 0;
  std::uint64_t grid = 0;
  double stage_ms = 0.0;
};

struct GpuRawArtifact
{
  double *values = nullptr;
  std::uint64_t n = 0;
  double stage_ms = 0.0;
};

struct GpuResidentScratch
{
  std::uint8_t *d_threshold_bytes = nullptr;
  std::uint8_t *d_segment_mode = nullptr;
  std::uint32_t *d_active_plane_count = nullptr;
  std::uint32_t *d_total_bits_by_segment = nullptr;
  std::uint64_t *d_threshold_combined = nullptr;
  GpuBlockStats *d_block_stats = nullptr;
  std::uint8_t *d_row_states = nullptr;
  unsigned long long *d_refined_hits = nullptr;
  std::uint32_t *d_u_indices = nullptr;
  unsigned int *d_u_count = nullptr;

  cudaEvent_t kernel_start{};
  cudaEvent_t kernel_stop{};
  cudaEvent_t compact_start{};
  cudaEvent_t compact_stop{};
  cudaEvent_t refine_start{};
  cudaEvent_t refine_stop{};

  void allocate(std::size_t segment_count, std::uint32_t max_plane_count,
                std::uint64_t row_count, std::uint64_t grid,
                bool need_row_states, bool need_refine, bool compact_u_mode);

  void deallocate();
};

struct PreparedResidentQuery
{
  std::vector<SegmentThresholdInfo> threshold_info;
  std::vector<std::uint8_t> segment_mode;
  std::vector<std::uint32_t> active_plane_count;
  std::vector<std::uint32_t> total_bits_by_segment;
  std::vector<std::uint8_t> h_threshold_flat;
  std::vector<std::uint64_t> h_threshold_combined;
  std::uint32_t effective_k = 0;
  double threshold_prep_ms = 0.0;
  double threshold_classify_ms = 0.0;
  double threshold_encode_ms = 0.0;
  double threshold_base_ms = 0.0;
  double threshold_subtract_extract_ms = 0.0;
  double threshold_pack_ms = 0.0;
  double threshold_static_candidate_ms = 0.0;
  double threshold_dependent_candidate_ms = 0.0;
  int threshold_mixed_segments = 0;
  int threshold_allq_segments = 0;
  int threshold_alld_segments = 0;
};

struct ResidentRunOptions
{
  bool upload_query_metadata = true;
  bool collect_phase_timing = true;
};

constexpr std::uint8_t kRowStateD = 0;
constexpr std::uint8_t kRowStateQ = 1;
constexpr std::uint8_t kRowStateU = 2;

[[noreturn]] void fail(const std::string &message)
{
  throw std::runtime_error(message);
}

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point stop)
{
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

void cuda_check(cudaError_t err, const char *where)
{
  if (err == cudaSuccess) return;
  fail(std::string("cuda error at ") + where + ": " + cudaGetErrorString(err));
}

void GpuResidentScratch::allocate(std::size_t segment_count, std::uint32_t max_plane_count,
                                  std::uint64_t row_count, std::uint64_t grid,
                                  bool need_row_states, bool need_refine, bool compact_u_mode)
{
  cuda_check(cudaMalloc(&d_threshold_bytes, segment_count * max_plane_count),
             "cudaMalloc(threshold_bytes)");
  cuda_check(cudaMalloc(&d_segment_mode, segment_count),
             "cudaMalloc(segment_mode)");
  cuda_check(cudaMalloc(&d_active_plane_count, segment_count * sizeof(std::uint32_t)),
             "cudaMalloc(active_plane_count)");
  cuda_check(cudaMalloc(&d_total_bits_by_segment, segment_count * sizeof(std::uint32_t)),
             "cudaMalloc(total_bits_by_segment)");
  cuda_check(cudaMalloc(&d_threshold_combined, segment_count * sizeof(std::uint64_t)),
             "cudaMalloc(threshold_combined)");
  cuda_check(cudaMalloc(&d_block_stats, grid * sizeof(GpuBlockStats)),
             "cudaMalloc(block_stats)");
  if (need_row_states)
    cuda_check(cudaMalloc(&d_row_states, row_count * sizeof(std::uint8_t)),
               "cudaMalloc(row_states)");
  if (need_refine)
    cuda_check(cudaMalloc(&d_refined_hits, sizeof(unsigned long long)),
               "cudaMalloc(refined_hits)");
  if (compact_u_mode)
  {
    cuda_check(cudaMalloc(&d_u_indices, row_count * sizeof(std::uint32_t)),
               "cudaMalloc(u_indices)");
    cuda_check(cudaMalloc(&d_u_count, sizeof(unsigned int)),
               "cudaMalloc(u_count)");
  }
  cuda_check(cudaEventCreate(&kernel_start), "cudaEventCreate(kernel_start)");
  cuda_check(cudaEventCreate(&kernel_stop), "cudaEventCreate(kernel_stop)");
  cuda_check(cudaEventCreate(&compact_start), "cudaEventCreate(compact_start)");
  cuda_check(cudaEventCreate(&compact_stop), "cudaEventCreate(compact_stop)");
  cuda_check(cudaEventCreate(&refine_start), "cudaEventCreate(refine_start)");
  cuda_check(cudaEventCreate(&refine_stop), "cudaEventCreate(refine_stop)");
}

void GpuResidentScratch::deallocate()
{
  if (d_threshold_bytes != nullptr) { cuda_check(cudaFree(d_threshold_bytes), "cudaFree(threshold_bytes)"); d_threshold_bytes = nullptr; }
  if (d_segment_mode != nullptr) { cuda_check(cudaFree(d_segment_mode), "cudaFree(segment_mode)"); d_segment_mode = nullptr; }
  if (d_active_plane_count != nullptr) { cuda_check(cudaFree(d_active_plane_count), "cudaFree(active_plane_count)"); d_active_plane_count = nullptr; }
  if (d_total_bits_by_segment != nullptr) { cuda_check(cudaFree(d_total_bits_by_segment), "cudaFree(total_bits_by_segment)"); d_total_bits_by_segment = nullptr; }
  if (d_threshold_combined != nullptr) { cuda_check(cudaFree(d_threshold_combined), "cudaFree(threshold_combined)"); d_threshold_combined = nullptr; }
  if (d_block_stats != nullptr) { cuda_check(cudaFree(d_block_stats), "cudaFree(block_stats)"); d_block_stats = nullptr; }
  if (d_row_states != nullptr) { cuda_check(cudaFree(d_row_states), "cudaFree(row_states)"); d_row_states = nullptr; }
  if (d_refined_hits != nullptr) { cuda_check(cudaFree(d_refined_hits), "cudaFree(refined_hits)"); d_refined_hits = nullptr; }
  if (d_u_indices != nullptr) { cuda_check(cudaFree(d_u_indices), "cudaFree(u_indices)"); d_u_indices = nullptr; }
  if (d_u_count != nullptr) { cuda_check(cudaFree(d_u_count), "cudaFree(u_count)"); d_u_count = nullptr; }
  if (kernel_start != nullptr) { cuda_check(cudaEventDestroy(kernel_start), "cudaEventDestroy(kernel_start)"); kernel_start = nullptr; }
  if (kernel_stop != nullptr) { cuda_check(cudaEventDestroy(kernel_stop), "cudaEventDestroy(kernel_stop)"); kernel_stop = nullptr; }
  if (compact_start != nullptr) { cuda_check(cudaEventDestroy(compact_start), "cudaEventDestroy(compact_start)"); compact_start = nullptr; }
  if (compact_stop != nullptr) { cuda_check(cudaEventDestroy(compact_stop), "cudaEventDestroy(compact_stop)"); compact_stop = nullptr; }
  if (refine_start != nullptr) { cuda_check(cudaEventDestroy(refine_start), "cudaEventDestroy(refine_start)"); refine_start = nullptr; }
  if (refine_stop != nullptr) { cuda_check(cudaEventDestroy(refine_stop), "cudaEventDestroy(refine_stop)"); refine_stop = nullptr; }
}

std::uint64_t ceil_div_u64(std::uint64_t x, std::uint64_t y)
{
  return (x + y - 1ull) / y;
}

std::uint64_t le_bytes_to_u64(const std::vector<std::uint8_t> &bytes)
{
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < bytes.size() && i < sizeof(std::uint64_t); ++i)
    value |= static_cast<std::uint64_t>(bytes[i]) << (i * 8);
  return value;
}

std::vector<std::uint8_t> subtract_le_bytes(const std::vector<std::uint8_t> &a,
                                            const std::vector<std::uint8_t> &b)
{
  std::vector<std::uint8_t> result(a.size(), 0);
  int borrow = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    int value = static_cast<int>(a[i]) - borrow;
    if (i < b.size()) value -= static_cast<int>(b[i]);
    if (value < 0)
    {
      value += 256;
      borrow = 1;
    }
    else
    {
      borrow = 0;
    }
    result[i] = static_cast<std::uint8_t>(value);
  }
  while (!result.empty() && result.back() == 0) result.pop_back();
  return result;
}

void extract_plane_bytes_from_combined_le(const std::vector<std::uint8_t> &combined_le,
                                          std::size_t total_bits,
                                          std::vector<std::uint8_t> &out_plane_bytes,
                                          std::size_t max_planes)
{
  auto bit_is_set = [&](std::size_t bit_index) -> bool
  {
    std::size_t byte_index = bit_index / 8;
    if (byte_index >= combined_le.size()) return false;
    return ((combined_le[byte_index] >> (bit_index % 8)) & 1U) != 0;
  };

  std::size_t plane_count = (total_bits == 0) ? 0 : (total_bits + 7) / 8;
  out_plane_bytes.resize(max_planes, 0);
  for (std::size_t plane = 0; plane < plane_count && plane < max_planes; ++plane)
  {
    std::size_t width = 8;
    if (plane + 1 == plane_count)
    {
      std::size_t trailing = total_bits - 8 * (plane_count - 1);
      width = (trailing == 0) ? 8 : trailing;
    }
    std::size_t start_bit = total_bits - 8 * plane - width;
    std::uint8_t byte = 0;
    for (std::size_t offset = 0; offset < width; ++offset)
    {
      if (bit_is_set(start_bit + offset))
        byte |= static_cast<std::uint8_t>(1U << offset);
    }
    out_plane_bytes[plane] = byte;
  }
}

std::vector<std::pair<double, double>> compute_segment_min_max(
    const std::filesystem::path &raw_path,
    std::uint64_t segment_rows,
    std::uint64_t num_segments)
{
  std::uint64_t file_size = std::filesystem::file_size(raw_path);
  if (file_size % sizeof(double) != 0) fail("raw file size is not FP64-aligned");
  std::uint64_t value_count = file_size / sizeof(double);

  std::ifstream input(raw_path, std::ios::binary);
  if (!input) fail("failed to open raw file for segment min/max: " + raw_path.string());

  std::vector<std::pair<double, double>> result;
  result.reserve(static_cast<std::size_t>(num_segments));
  std::vector<double> buffer(static_cast<std::size_t>(segment_rows));
  std::uint64_t remaining = value_count;
  for (std::uint64_t seg = 0; seg < num_segments; ++seg)
  {
    std::uint64_t current = std::min(remaining, segment_rows);
    if (current == 0) fail("raw file ended before all segments were read");
    input.read(reinterpret_cast<char *>(buffer.data()),
               static_cast<std::streamsize>(current * sizeof(double)));
    if (!input) fail("failed to read raw file for segment min/max");

    double seg_min = buffer[0];
    double seg_max = buffer[0];
    for (std::uint64_t i = 1; i < current; ++i)
    {
      seg_min = std::min(seg_min, buffer[i]);
      seg_max = std::max(seg_max, buffer[i]);
    }
    result.emplace_back(seg_min, seg_max);
    remaining -= current;
  }
  return result;
}

std::vector<std::pair<double, double>> read_segment_min_max_csv(
    const std::filesystem::path &path,
    std::uint64_t num_segments)
{
  std::ifstream input(path);
  if (!input) fail("failed to open segment min/max CSV: " + path.string());

  std::vector<std::pair<double, double>> result;
  result.reserve(static_cast<std::size_t>(num_segments));
  std::string line;
  if (!std::getline(input, line)) fail("empty segment min/max CSV: " + path.string());
  while (std::getline(input, line))
  {
    if (line.empty()) continue;
    std::size_t first = line.find(',');
    std::size_t second = first == std::string::npos ? std::string::npos : line.find(',', first + 1);
    if (first == std::string::npos || second == std::string::npos)
      fail("invalid segment min/max CSV row: " + line);
    double seg_min = std::stod(line.substr(first + 1, second - first - 1));
    double seg_max = std::stod(line.substr(second + 1));
    result.emplace_back(seg_min, seg_max);
  }
  if (result.size() != static_cast<std::size_t>(num_segments))
    fail("segment min/max CSV row count does not match artifact segments: " + path.string());
  return result;
}

std::filesystem::path default_raw_path(const std::filesystem::path &artifact_root,
                                       const std::string &dataset)
{
  return artifact_root.parent_path().parent_path() / "dev" / (dataset + ".f64le.bin");
}

void write_row_states(const std::filesystem::path &path,
                      const std::vector<std::uint8_t> &row_states)
{
  if (!path.parent_path().empty())
    std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) fail("failed to open row-state output path: " + path.string());
  out.write(reinterpret_cast<const char *>(row_states.data()),
            static_cast<std::streamsize>(row_states.size()));
  if (!out) fail("failed to write row-state output path: " + path.string());
}

std::vector<double> load_raw_values(const std::filesystem::path &path, std::uint64_t expected_count)
{
  std::uint64_t file_size = std::filesystem::file_size(path);
  if (file_size % sizeof(double) != 0) fail("raw file size is not FP64-aligned");
  std::uint64_t value_count = file_size / sizeof(double);
  if (expected_count != 0 && value_count != expected_count)
    fail("raw file row count does not match artifact value count: " + path.string());

  std::ifstream input(path, std::ios::binary);
  if (!input) fail("failed to open raw FP64 file: " + path.string());

  std::vector<double> values(static_cast<std::size_t>(value_count));
  input.read(reinterpret_cast<char *>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(double)));
  if (!input) fail("failed to read raw FP64 file: " + path.string());
  return values;
}

std::vector<std::pair<double, double>> compute_segment_min_max_from_values(
    const std::vector<double> &values,
    std::uint64_t segment_rows,
    std::uint64_t num_segments)
{
  std::vector<std::pair<double, double>> result;
  result.reserve(static_cast<std::size_t>(num_segments));
  std::uint64_t remaining = static_cast<std::uint64_t>(values.size());
  for (std::uint64_t seg = 0; seg < num_segments; ++seg)
  {
    std::uint64_t current = std::min(remaining, segment_rows);
    if (current == 0) fail("raw file ended before all segments were read");
    std::size_t offset = static_cast<std::size_t>(values.size() - remaining);
    double seg_min = values[offset];
    double seg_max = values[offset];
    for (std::uint64_t i = 1; i < current; ++i)
    {
      double value = values[offset + static_cast<std::size_t>(i)];
      seg_min = std::min(seg_min, value);
      seg_max = std::max(seg_max, value);
    }
    result.emplace_back(seg_min, seg_max);
    remaining -= current;
  }
  return result;
}

GpuRawArtifact stage_gpu_raw_values(const std::vector<double> &values)
{
  auto stage_start = std::chrono::steady_clock::now();
  GpuRawArtifact result;
  result.n = static_cast<std::uint64_t>(values.size());
  cuda_check(cudaMalloc(&result.values, values.size() * sizeof(double)), "cudaMalloc(raw_values)");
  cuda_check(cudaMemcpy(result.values, values.data(), values.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(raw_values)");
  result.stage_ms = elapsed_ms(stage_start, std::chrono::steady_clock::now());
  return result;
}

void free_gpu_raw_values(GpuRawArtifact &raw)
{
  if (raw.values != nullptr)
    cuda_check(cudaFree(raw.values), "cudaFree(raw_values)");
  raw.values = nullptr;
  raw.n = 0;
}

struct ThresholdPrepTiming {
  double classify_ms = 0.0;
  double encode_ms = 0.0;
  double base_ms = 0.0;
  double subtract_extract_ms = 0.0;
};

std::vector<std::uint8_t> local_encode_to_le_bytes(double value, std::int32_t scale_exponent)
{
  long double scaled = std::floor(std::ldexp(static_cast<long double>(value), -scale_exponent));
  if (scaled == 0.0L) return {};
  std::uint64_t code = static_cast<std::uint64_t>(scaled);
  std::vector<std::uint8_t> result;
  result.reserve(sizeof(std::uint64_t));
  for (std::size_t i = 0; i < sizeof(std::uint64_t); ++i)
    result.push_back(static_cast<std::uint8_t>((code >> (i * 8)) & 0xFF));
  return result;
}

std::vector<SegmentThresholdInfo> compute_threshold_bytes(
    const exp3_real::Dataset &dataset,
    const std::vector<std::pair<double, double>> &segment_min_max,
    double threshold_fp64,
    ThresholdPrepTiming *timing = nullptr)
{
  std::vector<SegmentThresholdInfo> result;
  result.reserve(dataset.segments.size());

  for (std::size_t seg = 0; seg < dataset.segments.size(); ++seg)
  {
    const auto &segment = dataset.segments[seg];
    double seg_min = segment_min_max[seg].first;
    double seg_max = segment_min_max[seg].second;

    if (seg_min == std::numeric_limits<double>::lowest() && segment.integer_base_is_v2)
      seg_min = segment.segment_base;
    if (seg_max == std::numeric_limits<double>::max() && segment.integer_base_is_v2)
      seg_max = segment.segment_base
                + std::ldexp(1.0, static_cast<int>(segment.integer_offset_bits))
                - std::ldexp(1.0, -static_cast<int>(segment.fractional_bits));

    auto classify_start = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    SegmentThresholdInfo info;
    if (threshold_fp64 < seg_min)
    {
      info.all_qualified = true;
      info.threshold_bytes.resize(dataset.manifest.max_plane_count, 0);
      result.push_back(std::move(info));
      if (timing) timing->classify_ms += elapsed_ms(classify_start, std::chrono::steady_clock::now());
      continue;
    }
    if (threshold_fp64 >= seg_max)
    {
      info.all_disqualified = true;
      info.threshold_bytes.resize(dataset.manifest.max_plane_count, 0);
      result.push_back(std::move(info));
      if (timing) timing->classify_ms += elapsed_ms(classify_start, std::chrono::steady_clock::now());
      continue;
    }
    if (timing) timing->classify_ms += elapsed_ms(classify_start, std::chrono::steady_clock::now());

    std::uint32_t frac = segment.fractional_bits;
    int32_t scale_exponent = -static_cast<int32_t>(frac);
    std::uint64_t integer_base_u64 = le_bytes_to_u64(segment.integer_base_le);
    bool uses_bounded_precision = segment.integer_base_is_v2 ||
                                  (segment.raw_fractional_bits != segment.fractional_bits) ||
                                  (segment.precision_cap_bits != segment.fractional_bits);
    std::vector<std::uint8_t> t_code_le;
    std::vector<std::uint8_t> base_shifted_le;

    if (uses_bounded_precision)
    {
      auto te_start = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      long double scale = std::ldexp(1.0L, static_cast<int>(frac));
      std::uint64_t t_code = static_cast<std::uint64_t>(
          std::floor(static_cast<long double>(threshold_fp64) * scale));
      for (std::size_t i = 0; i < sizeof(std::uint64_t); ++i)
        t_code_le.push_back(static_cast<std::uint8_t>((t_code >> (i * 8)) & 0xFF));
      if (timing) timing->encode_ms += elapsed_ms(te_start, std::chrono::steady_clock::now());
      auto bs_start = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      base_shifted_le = segment.integer_base_is_v2
                            ? segment.integer_base_le
                            : local_encode_to_le_bytes(
                                  static_cast<double>(integer_base_u64), scale_exponent);
      if (timing) timing->base_ms += elapsed_ms(bs_start, std::chrono::steady_clock::now());
    }
    else
    {
      auto te_start = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      t_code_le = local_encode_to_le_bytes(threshold_fp64, scale_exponent);
      if (timing) timing->encode_ms += elapsed_ms(te_start, std::chrono::steady_clock::now());
      auto bs_start = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      base_shifted_le = local_encode_to_le_bytes(
          static_cast<double>(integer_base_u64), scale_exponent);
      if (timing) timing->base_ms += elapsed_ms(bs_start, std::chrono::steady_clock::now());
    }

    auto subext_start = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::size_t max_len = std::max(t_code_le.size(), base_shifted_le.size());
    t_code_le.resize(max_len, 0);
    base_shifted_le.resize(max_len, 0);
    std::vector<std::uint8_t> combined_T_le = subtract_le_bytes(t_code_le, base_shifted_le);

    std::size_t total_bits = segment.get_fixed_len_bits();
    extract_plane_bytes_from_combined_le(
        combined_T_le, total_bits, info.threshold_bytes, dataset.manifest.max_plane_count);
    info.threshold_combined = le_bytes_to_u64(combined_T_le);
    if (timing) timing->subtract_extract_ms += elapsed_ms(subext_start, std::chrono::steady_clock::now());
    result.push_back(std::move(info));
  }

  return result;
}

void combined_bounds_after_k_planes(const exp3_real::Dataset &dataset,
                                    const exp3_real::SegmentMeta &segment,
                                    std::uint64_t row_in_segment,
                                    std::uint32_t kept_planes,
                                    std::uint64_t &out_lower,
                                    std::uint64_t &out_upper)
{
  std::size_t total_bits = segment.get_fixed_len_bits();
  std::size_t plane_count = (total_bits == 0) ? 0 : (total_bits + 7) / 8;
  out_lower = 0;
  out_upper = 0;

  std::uint32_t kept_bits = 0;
  std::uint32_t rounds = std::min<std::uint32_t>(kept_planes, static_cast<std::uint32_t>(plane_count));
  for (std::uint32_t plane = 0; plane < rounds; ++plane)
  {
    std::size_t width = 8;
    if (plane + 1 == plane_count)
    {
      std::size_t trailing = total_bits - 8 * (plane_count - 1);
      width = (trailing == 0) ? 8 : trailing;
    }
    std::size_t start_bit = total_bits - 8 * plane - width;
    std::uint8_t mask = (width == 8) ? 0xFFu : static_cast<std::uint8_t>((1u << width) - 1u);
    std::uint8_t plane_byte = dataset.planes[plane][segment.row_offset + row_in_segment] & mask;
    out_lower |= static_cast<std::uint64_t>(plane_byte) << start_bit;
    out_upper |= static_cast<std::uint64_t>(plane_byte) << start_bit;
    kept_bits += static_cast<std::uint32_t>(width);
  }

  if (kept_bits < total_bits)
  {
    std::uint32_t remaining_bits = static_cast<std::uint32_t>(total_bits) - kept_bits;
    std::uint64_t remaining_mask = (remaining_bits >= 64) ? ~0ull : ((1ull << remaining_bits) - 1ull);
    out_upper |= remaining_mask;
  }
}

ByteplaneCountGtResult cpu_count_gt(const exp3_real::Dataset &dataset,
                                    const std::vector<SegmentThresholdInfo> &threshold_info,
                                    const std::vector<std::uint32_t> &active_plane_count,
                                    std::uint32_t k,
                                    int block_threads,
                                    const std::filesystem::path &row_state_output_path)
{
  ByteplaneCountGtResult result;
  result.dataset = dataset.manifest.dataset;
  result.row_count = dataset.manifest.value_count;
  result.k = k;
  result.max_planes = static_cast<std::uint32_t>(dataset.manifest.max_plane_count);

  std::uint64_t tile_rows = static_cast<std::uint64_t>(block_threads) *
                            static_cast<std::uint64_t>(EXP3_ROWPACK16_WIDTH);
  std::vector<std::uint8_t> row_states;
  if (!row_state_output_path.empty())
    row_states.assign(static_cast<std::size_t>(dataset.manifest.value_count), kRowStateD);

  auto classify_unresolved = [&](const exp3_real::SegmentMeta &segment,
                                 const SegmentThresholdInfo &info,
                                 std::uint64_t row_in_segment,
                                 std::uint32_t rounds)
      -> std::uint8_t
  {
    std::uint64_t lower = 0;
    std::uint64_t upper = 0;
    combined_bounds_after_k_planes(dataset, segment, row_in_segment, rounds, lower, upper);
    if (lower > info.threshold_combined)
    {
      ++result.q;
      return kRowStateQ;
    }
    else if (upper <= info.threshold_combined)
    {
      ++result.d;
      return kRowStateD;
    }
    else
    {
      ++result.u;
      return kRowStateU;
    }
  };

  for (std::size_t seg = 0; seg < dataset.segments.size(); ++seg)
  {
    const auto &segment = dataset.segments[seg];
    const auto &info = threshold_info[seg];
    if (info.all_qualified)
    {
      result.q += segment.row_count;
      if (!row_states.empty())
      {
        std::fill(row_states.begin() + static_cast<std::ptrdiff_t>(segment.row_offset),
                  row_states.begin() + static_cast<std::ptrdiff_t>(segment.row_offset + segment.row_count),
                  kRowStateQ);
      }
      continue;
    }
    if (info.all_disqualified)
    {
      result.d += segment.row_count;
      continue;
    }

    std::uint32_t rounds = active_plane_count[seg];
    std::uint64_t seg_start = segment.row_offset;
    std::uint64_t seg_end = segment.row_offset + segment.row_count;
    for (std::uint64_t tile_start = seg_start; tile_start < seg_end; tile_start += tile_rows)
    {
      std::uint64_t tile_end = std::min(seg_end, tile_start + tile_rows);
      std::uint64_t aligned_start = tile_start;
      while (aligned_start < tile_end && (aligned_start & 15ull) != 0ull)
        ++aligned_start;
      std::uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;

      for (std::uint64_t row = tile_start; row < aligned_start; ++row)
      {
        bool active = true;
        for (std::uint32_t plane = 0; plane < rounds && active; ++plane)
        {
          ++result.bytes_read;
          std::uint8_t row_byte = dataset.planes[plane][row];
          std::uint8_t threshold_byte = info.threshold_bytes[plane];
          if (row_byte > threshold_byte)
          {
            ++result.q;
            if (!row_states.empty()) row_states[static_cast<std::size_t>(row)] = kRowStateQ;
            active = false;
          }
          else if (row_byte < threshold_byte)
          {
            ++result.d;
            if (!row_states.empty()) row_states[static_cast<std::size_t>(row)] = kRowStateD;
            active = false;
          }
        }
        if (active)
        {
          std::uint8_t state = classify_unresolved(segment, info, row - segment.row_offset, rounds);
          if (!row_states.empty()) row_states[static_cast<std::size_t>(row)] = state;
        }
      }

      for (std::uint64_t pack_start = aligned_start; pack_start < full_pack_end; pack_start += 16ull)
      {
        std::uint16_t active_mask = 0xFFFFu;
        std::array<std::uint64_t, 16> lower{};
        std::array<std::uint32_t, 16> kept_bits{};

        for (std::uint32_t plane = 0; plane < rounds && active_mask != 0u; ++plane)
        {
          result.bytes_read += 16ull;
          const auto &plane_bytes = dataset.planes[plane];
          for (int lane = 0; lane < 16; ++lane)
          {
            std::uint16_t bit = static_cast<std::uint16_t>(1u << lane);
            if ((active_mask & bit) == 0u) continue;

            const auto &seg_meta = segment;
            std::size_t total_bits = seg_meta.get_fixed_len_bits();
            std::size_t plane_count = (total_bits == 0) ? 0 : (total_bits + 7) / 8;
            std::size_t width = 8;
            if (plane + 1 == plane_count)
            {
              std::size_t trailing = total_bits - 8 * (plane_count - 1);
              width = (trailing == 0) ? 8 : trailing;
            }
            std::size_t start_bit = total_bits - 8 * plane - width;
            std::uint8_t mask = (width == 8) ? 0xFFu : static_cast<std::uint8_t>((1u << width) - 1u);
            std::uint8_t row_byte = plane_bytes[pack_start + static_cast<std::uint64_t>(lane)] & mask;
            std::uint8_t threshold_byte = info.threshold_bytes[plane];
            lower[lane] |= static_cast<std::uint64_t>(row_byte) << start_bit;
            kept_bits[lane] += static_cast<std::uint32_t>(width);

            if (row_byte > threshold_byte)
            {
              ++result.q;
              if (!row_states.empty())
                row_states[static_cast<std::size_t>(pack_start + static_cast<std::uint64_t>(lane))] = kRowStateQ;
              active_mask = static_cast<std::uint16_t>(active_mask & ~bit);
            }
            else if (row_byte < threshold_byte)
            {
              ++result.d;
              if (!row_states.empty())
                row_states[static_cast<std::size_t>(pack_start + static_cast<std::uint64_t>(lane))] = kRowStateD;
              active_mask = static_cast<std::uint16_t>(active_mask & ~bit);
            }
          }
        }

        for (int lane = 0; lane < 16; ++lane)
        {
          std::uint16_t bit = static_cast<std::uint16_t>(1u << lane);
          if ((active_mask & bit) == 0u) continue;

          std::size_t total_bits = segment.get_fixed_len_bits();
          std::uint64_t upper = lower[lane];
          if (kept_bits[lane] < total_bits)
          {
            std::uint32_t remaining_bits = static_cast<std::uint32_t>(total_bits) - kept_bits[lane];
            std::uint64_t remaining_mask =
                (remaining_bits >= 64) ? ~0ull : ((1ull << remaining_bits) - 1ull);
            upper |= remaining_mask;
          }
          if (lower[lane] > info.threshold_combined)
          {
            ++result.q;
            if (!row_states.empty())
              row_states[static_cast<std::size_t>(pack_start + static_cast<std::uint64_t>(lane))] = kRowStateQ;
          }
          else if (upper <= info.threshold_combined)
          {
            ++result.d;
            if (!row_states.empty())
              row_states[static_cast<std::size_t>(pack_start + static_cast<std::uint64_t>(lane))] = kRowStateD;
          }
          else
          {
            ++result.u;
            if (!row_states.empty())
              row_states[static_cast<std::size_t>(pack_start + static_cast<std::uint64_t>(lane))] = kRowStateU;
          }
        }
      }

      for (std::uint64_t row = full_pack_end; row < tile_end; ++row)
      {
        bool active = true;
        for (std::uint32_t plane = 0; plane < rounds && active; ++plane)
        {
          ++result.bytes_read;
          std::uint8_t row_byte = dataset.planes[plane][row];
          std::uint8_t threshold_byte = info.threshold_bytes[plane];
          if (row_byte > threshold_byte)
          {
            ++result.q;
            if (!row_states.empty()) row_states[static_cast<std::size_t>(row)] = kRowStateQ;
            active = false;
          }
          else if (row_byte < threshold_byte)
          {
            ++result.d;
            if (!row_states.empty()) row_states[static_cast<std::size_t>(row)] = kRowStateD;
            active = false;
          }
        }
        if (active)
        {
          std::uint8_t state = classify_unresolved(segment, info, row - segment.row_offset, rounds);
          if (!row_states.empty()) row_states[static_cast<std::size_t>(row)] = state;
        }
      }
    }
  }

  result.count = result.q;
  if (!row_state_output_path.empty())
    write_row_states(row_state_output_path, row_states);
  return result;
}

__device__ __forceinline__ std::uint32_t plane_width_bits(std::uint32_t total_bits,
                                                          std::uint32_t plane,
                                                          std::uint32_t plane_count)
{
  if (plane + 1u != plane_count) return 8u;
  std::uint32_t trailing = total_bits - 8u * (plane_count - 1u);
  return trailing == 0u ? 8u : trailing;
}

__device__ __forceinline__ std::uint64_t low_bits_mask(std::uint32_t bit_count)
{
  return bit_count >= 64u ? ~0ull : ((1ull << bit_count) - 1ull);
}

__global__ void direct_refine_raw_kernel(const double *__restrict__ raw_values,
                                         const std::uint8_t *__restrict__ row_states,
                                         std::uint64_t n,
                                         double threshold,
                                         unsigned long long *__restrict__ refined_hits)
{
  unsigned long long local_hits = 0;
  for (std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < n;
       row += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
  {
    if (row_states[row] == kRowStateU && raw_values[row] > threshold)
      ++local_hits;
  }

  __shared__ unsigned long long reduce_smem[32];
  unsigned long long block_hits = exp3_block_reduce_sum_ull(local_hits, reduce_smem);
  if (threadIdx.x == 0) atomicAdd(refined_hits, block_hits);
}

__global__ void compact_u_indices_kernel(const std::uint8_t *__restrict__ row_states,
                                         std::uint64_t n,
                                         std::uint32_t *__restrict__ u_indices,
                                         unsigned int *__restrict__ u_count)
{
  for (std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < n;
       row += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
  {
    if (row_states[row] != kRowStateU) continue;
    unsigned int slot = atomicAdd(u_count, 1u);
    u_indices[slot] = static_cast<std::uint32_t>(row);
  }
}

__global__ void compact_u_refine_raw_kernel(const double *__restrict__ raw_values,
                                            const std::uint32_t *__restrict__ u_indices,
                                            unsigned int u_count,
                                            double threshold,
                                            unsigned long long *__restrict__ refined_hits)
{
  unsigned long long local_hits = 0;
  for (unsigned int idx = static_cast<unsigned int>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < u_count;
       idx += static_cast<unsigned int>(gridDim.x) * blockDim.x)
  {
    std::uint32_t row = u_indices[idx];
    if (raw_values[row] > threshold) ++local_hits;
  }

  __shared__ unsigned long long reduce_smem[32];
  unsigned long long block_hits = exp3_block_reduce_sum_ull(local_hits, reduce_smem);
  if (threadIdx.x == 0) atomicAdd(refined_hits, block_hits);
}

__global__ void byteplane_count_gt_kernel(Exp3RuntimeU8Subcolumns subcolumns,
                                          std::uint64_t n,
                                          std::uint64_t segment_rows,
                                          std::uint64_t tiles_per_segment,
                                          const std::uint8_t *__restrict__ threshold_bytes,
                                          int threshold_stride,
                                          const std::uint64_t *__restrict__ threshold_combined,
                                          const std::uint32_t *__restrict__ active_plane_count,
                                          const std::uint32_t *__restrict__ total_bits_by_segment,
                                          const std::uint8_t *__restrict__ segment_mode,
                                          std::uint8_t *__restrict__ row_states,
                                          GpuBlockStats *__restrict__ block_stats)
{
  std::uint64_t tile_rows = static_cast<std::uint64_t>(blockDim.x) *
                            static_cast<std::uint64_t>(EXP3_ROWPACK16_WIDTH);
  std::uint64_t segment_id = static_cast<std::uint64_t>(blockIdx.x) / tiles_per_segment;
  std::uint64_t tile_in_segment = static_cast<std::uint64_t>(blockIdx.x) % tiles_per_segment;
  std::uint64_t segment_start = segment_id * segment_rows;
  std::uint64_t segment_end = segment_start + segment_rows;
  std::uint64_t tile_start = segment_start + tile_in_segment * tile_rows;

  unsigned long long local_q = 0;
  unsigned long long local_d = 0;
  unsigned long long local_u = 0;
  unsigned long long local_bytes = 0;

  if (tile_start < n && tile_start < segment_end)
  {
    std::uint64_t tile_end = tile_start + tile_rows;
    if (tile_end > segment_end) tile_end = segment_end;
    if (tile_end > n) tile_end = n;
    std::uint8_t mode = segment_mode[segment_id];
    if (mode == kSegmentModeAllQualified)
    {
      for (std::uint64_t row = tile_start + threadIdx.x; row < tile_end; row += blockDim.x)
      {
        ++local_q;
        if (row_states != nullptr) row_states[row] = kRowStateQ;
      }
    }
    else if (mode == kSegmentModeAllDisqualified)
    {
      for (std::uint64_t row = tile_start + threadIdx.x; row < tile_end; row += blockDim.x)
      {
        ++local_d;
        if (row_states != nullptr) row_states[row] = kRowStateD;
      }
    }
    else
    {
      std::uint64_t aligned_start = tile_start;
      while (aligned_start < tile_end && (aligned_start & 15ull) != 0ull)
        ++aligned_start;
      std::uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
      std::uint64_t first_pack = aligned_start / 16ull;
      std::uint64_t pack_count = (full_pack_end - aligned_start) / 16ull;

      std::uint32_t rounds = active_plane_count[segment_id];
      std::uint32_t total_bits = total_bits_by_segment[segment_id];
      std::uint32_t plane_count = (total_bits == 0u) ? 0u : ((total_bits + 7u) / 8u);
      std::uint64_t thresh_offset = segment_id * static_cast<std::uint64_t>(threshold_stride);
      std::uint64_t threshold = threshold_combined[segment_id];

      for (std::uint64_t row = tile_start + threadIdx.x; row < aligned_start; row += blockDim.x)
      {
        bool active = true;
        std::uint64_t lower = 0;
        std::uint32_t kept_bits = 0;
        for (std::uint32_t plane = 0; plane < rounds && active; ++plane)
        {
          std::uint32_t width = plane_width_bits(total_bits, plane, plane_count);
          std::uint32_t start_bit = total_bits - 8u * plane - width;
          std::uint8_t mask = width == 8u ? 0xFFu : static_cast<std::uint8_t>((1u << width) - 1u);
          std::uint8_t row_byte = subcolumns.ptrs[plane][row] & mask;
          std::uint8_t threshold_byte = threshold_bytes[thresh_offset + plane];
          ++local_bytes;
          lower |= static_cast<std::uint64_t>(row_byte) << start_bit;
          kept_bits += width;
          if (row_byte > threshold_byte)
          {
            ++local_q;
            if (row_states != nullptr) row_states[row] = kRowStateQ;
            active = false;
          }
          else if (row_byte < threshold_byte)
          {
            ++local_d;
            if (row_states != nullptr) row_states[row] = kRowStateD;
            active = false;
          }
        }
        if (active)
        {
          std::uint64_t upper = lower;
          if (kept_bits < total_bits) upper |= low_bits_mask(total_bits - kept_bits);
          if (lower > threshold)
          {
            ++local_q;
            if (row_states != nullptr) row_states[row] = kRowStateQ;
          }
          else if (upper <= threshold)
          {
            ++local_d;
            if (row_states != nullptr) row_states[row] = kRowStateD;
          }
          else
          {
            ++local_u;
            if (row_states != nullptr) row_states[row] = kRowStateU;
          }
        }
      }

      for (std::uint64_t pack = threadIdx.x; pack < pack_count; pack += blockDim.x)
      {
        std::uint64_t pack_idx = first_pack + pack;
        std::uint16_t active_mask = 0xFFFFu;
        std::uint64_t lower[16] = {};
        std::uint32_t kept_bits[16] = {};

        for (std::uint32_t plane = 0; plane < rounds && active_mask != 0u; ++plane)
        {
          const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[plane]);
          uint4 pack_data = plane128[pack_idx];
          std::uint8_t vals[16];
          exp4_detail::extract_uint4_bytes(pack_data, vals);

          std::uint32_t width = plane_width_bits(total_bits, plane, plane_count);
          std::uint32_t start_bit = total_bits - 8u * plane - width;
          std::uint8_t mask = width == 8u ? 0xFFu : static_cast<std::uint8_t>((1u << width) - 1u);
          std::uint8_t threshold_byte = threshold_bytes[thresh_offset + plane];
          local_bytes += 16ull;

#pragma unroll
          for (int i = 0; i < 16; ++i)
          {
            std::uint16_t bit = static_cast<std::uint16_t>(1u << i);
            if ((active_mask & bit) == 0u) continue;
            std::uint8_t row_byte = vals[i] & mask;
            lower[i] |= static_cast<std::uint64_t>(row_byte) << start_bit;
            kept_bits[i] += width;
            if (row_byte > threshold_byte)
            {
              ++local_q;
              if (row_states != nullptr)
                row_states[pack_idx * 16ull + static_cast<std::uint64_t>(i)] = kRowStateQ;
              active_mask = static_cast<std::uint16_t>(active_mask & ~bit);
            }
            else if (row_byte < threshold_byte)
            {
              ++local_d;
              if (row_states != nullptr)
                row_states[pack_idx * 16ull + static_cast<std::uint64_t>(i)] = kRowStateD;
              active_mask = static_cast<std::uint16_t>(active_mask & ~bit);
            }
          }
        }

#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
          std::uint16_t bit = static_cast<std::uint16_t>(1u << i);
          if ((active_mask & bit) == 0u) continue;
          std::uint64_t upper = lower[i];
          if (kept_bits[i] < total_bits) upper |= low_bits_mask(total_bits - kept_bits[i]);
          if (lower[i] > threshold)
          {
            ++local_q;
            if (row_states != nullptr)
              row_states[pack_idx * 16ull + static_cast<std::uint64_t>(i)] = kRowStateQ;
          }
          else if (upper <= threshold)
          {
            ++local_d;
            if (row_states != nullptr)
              row_states[pack_idx * 16ull + static_cast<std::uint64_t>(i)] = kRowStateD;
          }
          else
          {
            ++local_u;
            if (row_states != nullptr)
              row_states[pack_idx * 16ull + static_cast<std::uint64_t>(i)] = kRowStateU;
          }
        }
      }

      for (std::uint64_t row = full_pack_end + threadIdx.x; row < tile_end; row += blockDim.x)
      {
        bool active = true;
        std::uint64_t lower = 0;
        std::uint32_t kept_bits = 0;
        for (std::uint32_t plane = 0; plane < rounds && active; ++plane)
        {
          std::uint32_t width = plane_width_bits(total_bits, plane, plane_count);
          std::uint32_t start_bit = total_bits - 8u * plane - width;
          std::uint8_t mask = width == 8u ? 0xFFu : static_cast<std::uint8_t>((1u << width) - 1u);
          std::uint8_t row_byte = subcolumns.ptrs[plane][row] & mask;
          std::uint8_t threshold_byte = threshold_bytes[thresh_offset + plane];
          ++local_bytes;
          lower |= static_cast<std::uint64_t>(row_byte) << start_bit;
          kept_bits += width;
          if (row_byte > threshold_byte)
          {
            ++local_q;
            if (row_states != nullptr) row_states[row] = kRowStateQ;
            active = false;
          }
          else if (row_byte < threshold_byte)
          {
            ++local_d;
            if (row_states != nullptr) row_states[row] = kRowStateD;
            active = false;
          }
        }
        if (active)
        {
          std::uint64_t upper = lower;
          if (kept_bits < total_bits) upper |= low_bits_mask(total_bits - kept_bits);
          if (lower > threshold)
          {
            ++local_q;
            if (row_states != nullptr) row_states[row] = kRowStateQ;
          }
          else if (upper <= threshold)
          {
            ++local_d;
            if (row_states != nullptr) row_states[row] = kRowStateD;
          }
          else
          {
            ++local_u;
            if (row_states != nullptr) row_states[row] = kRowStateU;
          }
        }
      }
    }
  }

  __shared__ unsigned long long reduce_smem[32];
  unsigned long long block_q = exp3_block_reduce_sum_ull(local_q, reduce_smem);
  unsigned long long block_d = exp3_block_reduce_sum_ull(local_d, reduce_smem);
  unsigned long long block_u = exp3_block_reduce_sum_ull(local_u, reduce_smem);
  unsigned long long block_bytes = exp3_block_reduce_sum_ull(local_bytes, reduce_smem);

  if (threadIdx.x == 0)
    block_stats[blockIdx.x] = {block_q, block_d, block_u, block_bytes};
}

GpuResidentArtifact stage_gpu_artifact(const exp3_real::Dataset &dataset,
                                       const ByteplaneCountGtOptions &options)
{
  auto stage_start = std::chrono::steady_clock::now();
  cuda_check(cudaSetDevice(options.device), "cudaSetDevice");

  GpuResidentArtifact resident;
  resident.n = dataset.manifest.value_count;
  std::uint64_t tile_rows = static_cast<std::uint64_t>(options.block_threads) *
                            static_cast<std::uint64_t>(EXP3_ROWPACK16_WIDTH);
  resident.tiles_per_segment = ceil_div_u64(dataset.manifest.segment_size, tile_rows);
  std::uint64_t num_segments = ceil_div_u64(resident.n, dataset.manifest.segment_size);
  resident.grid = num_segments * resident.tiles_per_segment;
  if (resident.grid == 0 || resident.grid > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    fail("grid does not fit in int range");

  std::vector<std::uint8_t *> d_plane_storage;
  int plane_count = static_cast<int>(dataset.manifest.max_plane_count);
  for (int plane = 0; plane < plane_count; ++plane)
  {
    const auto &host_plane = dataset.planes[plane];
    std::uint8_t *ptr = nullptr;
    cuda_check(cudaMalloc(&ptr, host_plane.size()), "cudaMalloc(plane)");
    cuda_check(cudaMemcpy(ptr, host_plane.data(), host_plane.size(), cudaMemcpyHostToDevice),
               "cudaMemcpy(plane)");
    resident.plane_storage.push_back(ptr);
    resident.subcolumns.ptrs[plane] = ptr;
  }
  for (int plane = plane_count; plane < EXP3_RUNTIME_MAX_SUBCOLUMNS; ++plane)
    resident.subcolumns.ptrs[plane] = nullptr;

  resident.stage_ms = elapsed_ms(stage_start, std::chrono::steady_clock::now());
  return resident;
}

void free_gpu_artifact(GpuResidentArtifact &resident)
{
  for (std::uint8_t *ptr : resident.plane_storage)
    cuda_check(cudaFree(ptr), "cudaFree(plane)");
  resident.plane_storage.clear();
}

PreparedResidentQuery prepare_resident_query(const exp3_real::Dataset &dataset,
                                             const std::vector<std::pair<double, double>> &segment_min_max,
                                             double threshold,
                                             int requested_k)
{
  auto threshold_start = std::chrono::steady_clock::now();

  ThresholdPrepTiming timing;
  PreparedResidentQuery prepared;
  prepared.threshold_info = compute_threshold_bytes(dataset, segment_min_max, threshold, &timing);
  prepared.threshold_prep_ms = elapsed_ms(threshold_start, std::chrono::steady_clock::now());

  prepared.effective_k =
      requested_k < 0 ? static_cast<std::uint32_t>(dataset.manifest.max_plane_count)
                      : static_cast<std::uint32_t>(requested_k);
  if (prepared.effective_k == 0) fail("k must be positive");

  std::size_t segment_count = dataset.segments.size();
  prepared.segment_mode.assign(segment_count, kSegmentModeMixed);
  prepared.active_plane_count.assign(segment_count, 0);
  prepared.total_bits_by_segment.assign(segment_count, 0);
  prepared.h_threshold_flat.assign(segment_count * dataset.manifest.max_plane_count, 0);
  prepared.h_threshold_combined.assign(segment_count, 0);

  int allq = 0, alld = 0;
  auto pack_start = std::chrono::steady_clock::now();
  for (std::size_t seg = 0; seg < segment_count; ++seg)
  {
    const auto &segment = dataset.segments[seg];
    const auto &threshold_info = prepared.threshold_info[seg];
    if (threshold_info.all_qualified)
    {
      ++allq;
      prepared.segment_mode[seg] = kSegmentModeAllQualified;
    }
    else if (threshold_info.all_disqualified)
    {
      ++alld;
      prepared.segment_mode[seg] = kSegmentModeAllDisqualified;
    }

    prepared.active_plane_count[seg] =
        std::min<std::uint32_t>(prepared.effective_k, segment.active_plane_count);
    prepared.total_bits_by_segment[seg] =
        segment.fractional_bits + segment.integer_offset_bits;
    std::memcpy(prepared.h_threshold_flat.data() + seg * dataset.manifest.max_plane_count,
                threshold_info.threshold_bytes.data(),
                dataset.manifest.max_plane_count);
    prepared.h_threshold_combined[seg] = threshold_info.threshold_combined;
  }

  prepared.threshold_classify_ms = timing.classify_ms;
  prepared.threshold_encode_ms = timing.encode_ms;
  prepared.threshold_base_ms = timing.base_ms;
  prepared.threshold_subtract_extract_ms = timing.subtract_extract_ms;
  prepared.threshold_pack_ms = elapsed_ms(pack_start, std::chrono::steady_clock::now());
  prepared.threshold_static_candidate_ms = timing.base_ms;
  prepared.threshold_dependent_candidate_ms = timing.encode_ms + timing.subtract_extract_ms;
  prepared.threshold_allq_segments = allq;
  prepared.threshold_alld_segments = alld;
  prepared.threshold_mixed_segments = static_cast<int>(segment_count) - allq - alld;

  return prepared;
}

ByteplaneCountGtResult gpu_count_gt_resident(const exp3_real::Dataset &dataset,
                                              const GpuResidentArtifact &resident,
                                              const GpuRawArtifact *raw_artifact,
                                              const PreparedResidentQuery &prepared,
                                              const ByteplaneCountGtOptions &options,
                                              double gpu_stage_ms,
                                              GpuResidentScratch *scratch = nullptr,
                                              ResidentRunOptions run_options = {})
{
  auto gpu_total_start = std::chrono::steady_clock::now();

  bool need_row_states = !options.row_state_output_path.empty()
                      || options.direct_refine_raw
                      || options.compact_u_refine_raw;

  std::uint8_t *d_threshold_bytes = nullptr;
  std::uint8_t *d_segment_mode = nullptr;
  std::uint32_t *d_active_plane_count = nullptr;
  std::uint32_t *d_total_bits_by_segment = nullptr;
  std::uint64_t *d_threshold_combined = nullptr;
  GpuBlockStats *d_block_stats = nullptr;
  std::uint8_t *d_row_states = nullptr;
  if (scratch)
  {
    d_threshold_bytes = scratch->d_threshold_bytes;
    d_segment_mode = scratch->d_segment_mode;
    d_active_plane_count = scratch->d_active_plane_count;
    d_total_bits_by_segment = scratch->d_total_bits_by_segment;
    d_threshold_combined = scratch->d_threshold_combined;
    d_block_stats = scratch->d_block_stats;
    d_row_states = scratch->d_row_states;
  }
  else
  {
    cuda_check(cudaMalloc(&d_threshold_bytes, prepared.h_threshold_flat.size()), "cudaMalloc(threshold_bytes)");
    cuda_check(cudaMalloc(&d_segment_mode, prepared.segment_mode.size()), "cudaMalloc(segment_mode)");
    cuda_check(cudaMalloc(&d_active_plane_count, prepared.active_plane_count.size() * sizeof(std::uint32_t)),
               "cudaMalloc(active_plane_count)");
    cuda_check(cudaMalloc(&d_total_bits_by_segment, prepared.total_bits_by_segment.size() * sizeof(std::uint32_t)),
               "cudaMalloc(total_bits_by_segment)");
    cuda_check(cudaMalloc(&d_threshold_combined, prepared.h_threshold_combined.size() * sizeof(std::uint64_t)),
               "cudaMalloc(threshold_combined)");
    cuda_check(cudaMalloc(&d_block_stats, resident.grid * sizeof(GpuBlockStats)), "cudaMalloc(block_stats)");
    if (need_row_states)
      cuda_check(cudaMalloc(&d_row_states, resident.n * sizeof(std::uint8_t)), "cudaMalloc(row_states)");
  }
  if (run_options.upload_query_metadata)
  {
    cuda_check(cudaMemcpy(d_threshold_bytes, prepared.h_threshold_flat.data(),
                          prepared.h_threshold_flat.size(), cudaMemcpyHostToDevice),
               "cudaMemcpy(threshold_bytes)");
    cuda_check(cudaMemcpy(d_segment_mode, prepared.segment_mode.data(),
                          prepared.segment_mode.size(), cudaMemcpyHostToDevice),
               "cudaMemcpy(segment_mode)");
    cuda_check(cudaMemcpy(d_active_plane_count, prepared.active_plane_count.data(),
                          prepared.active_plane_count.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(active_plane_count)");
    cuda_check(cudaMemcpy(d_total_bits_by_segment, prepared.total_bits_by_segment.data(),
                          prepared.total_bits_by_segment.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(total_bits_by_segment)");
    cuda_check(cudaMemcpy(d_threshold_combined, prepared.h_threshold_combined.data(),
                          prepared.h_threshold_combined.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(threshold_combined)");
  }

  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  if (run_options.collect_phase_timing)
  {
    if (scratch)
    {
      start_event = scratch->kernel_start;
      stop_event = scratch->kernel_stop;
    }
    else
    {
      cuda_check(cudaEventCreate(&start_event), "cudaEventCreate(start)");
      cuda_check(cudaEventCreate(&stop_event), "cudaEventCreate(stop)");
    }
    cuda_check(cudaEventRecord(start_event), "cudaEventRecord(start)");
  }
  byteplane_count_gt_kernel<<<static_cast<int>(resident.grid), options.block_threads>>>(
      resident.subcolumns,
      resident.n,
      dataset.manifest.segment_size,
      resident.tiles_per_segment,
      d_threshold_bytes,
      static_cast<int>(dataset.manifest.max_plane_count),
      d_threshold_combined,
      d_active_plane_count,
      d_total_bits_by_segment,
      d_segment_mode,
      d_row_states,
      d_block_stats);
  cuda_check(cudaGetLastError(), "byteplane_count_gt_kernel launch");
  float primitive_ms = 0.0f;
  if (run_options.collect_phase_timing)
  {
    cuda_check(cudaEventRecord(stop_event), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");
    cuda_check(cudaEventElapsedTime(&primitive_ms, start_event, stop_event),
               "cudaEventElapsedTime(primitive)");
    if (!scratch)
    {
      cuda_check(cudaEventDestroy(stop_event), "cudaEventDestroy(stop)");
      cuda_check(cudaEventDestroy(start_event), "cudaEventDestroy(start)");
    }
  }

  std::vector<GpuBlockStats> h_block_stats(static_cast<std::size_t>(resident.grid));
  cuda_check(cudaMemcpy(h_block_stats.data(), d_block_stats, resident.grid * sizeof(GpuBlockStats),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(block_stats)");

  ByteplaneCountGtResult result;
  result.dataset = dataset.manifest.dataset;
  result.output_mode = options.direct_refine_raw
                           ? "direct_dense"
                           : (options.compact_u_refine_raw ? "compact_u" : "encoded_interval");
  result.row_state_output_path =
      (options.direct_refine_raw || options.compact_u_refine_raw)
          ? std::filesystem::path{}
          : options.row_state_output_path;
  result.row_count = resident.n;
  result.k = prepared.effective_k;
  result.max_planes = static_cast<std::uint32_t>(dataset.manifest.max_plane_count);
  result.gpu_stage_ms = gpu_stage_ms;
  result.primitive_ms = static_cast<double>(primitive_ms);
  result.raw_stage_ms = raw_artifact != nullptr ? raw_artifact->stage_ms : 0.0;
  result.used_gpu = true;
  for (const auto &stats : h_block_stats)
  {
    result.q += stats.q;
    result.d += stats.d;
    result.u += stats.u;
    result.bytes_read += stats.bytes_read;
  }
  result.count = result.q;

  if (!options.row_state_output_path.empty() && !options.direct_refine_raw && !options.compact_u_refine_raw)
  {
    std::vector<std::uint8_t> h_row_states(static_cast<std::size_t>(resident.n));
    cuda_check(cudaMemcpy(h_row_states.data(), d_row_states, resident.n * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(row_states)");
    write_row_states(options.row_state_output_path, h_row_states);
  }

  if (options.direct_refine_raw || options.compact_u_refine_raw)
  {
    if (raw_artifact == nullptr || raw_artifact->values == nullptr)
      fail("direct refine requested without staged raw FP64 values");

    unsigned long long *d_refined_hits = nullptr;
    if (scratch)
    {
      d_refined_hits = scratch->d_refined_hits;
    }
    else
    {
      cuda_check(cudaMalloc(&d_refined_hits, sizeof(unsigned long long)),
                 "cudaMalloc(refined_hits)");
    }
    cuda_check(cudaMemset(d_refined_hits, 0, sizeof(unsigned long long)),
               "cudaMemset(refined_hits)");

    if (options.direct_refine_raw)
    {
      cudaEvent_t refine_start = nullptr;
      cudaEvent_t refine_stop = nullptr;
      if (run_options.collect_phase_timing)
      {
        if (scratch)
        {
          refine_start = scratch->refine_start;
          refine_stop = scratch->refine_stop;
        }
        else
        {
          cuda_check(cudaEventCreate(&refine_start), "cudaEventCreate(refine_start)");
          cuda_check(cudaEventCreate(&refine_stop), "cudaEventCreate(refine_stop)");
        }
        cuda_check(cudaEventRecord(refine_start), "cudaEventRecord(refine_start)");
      }
      direct_refine_raw_kernel<<<static_cast<int>(resident.grid), options.block_threads>>>(
          raw_artifact->values,
          d_row_states,
          resident.n,
          options.threshold,
          d_refined_hits);
      cuda_check(cudaGetLastError(), "direct_refine_raw_kernel launch");
      float refine_ms = 0.0f;
      if (run_options.collect_phase_timing)
      {
        cuda_check(cudaEventRecord(refine_stop), "cudaEventRecord(refine_stop)");
        cuda_check(cudaEventSynchronize(refine_stop), "cudaEventSynchronize(refine_stop)");
        cuda_check(cudaEventElapsedTime(&refine_ms, refine_start, refine_stop),
                   "cudaEventElapsedTime(refine)");
        if (!scratch)
        {
          cuda_check(cudaEventDestroy(refine_stop), "cudaEventDestroy(refine_stop)");
          cuda_check(cudaEventDestroy(refine_start), "cudaEventDestroy(refine_start)");
        }
      }

      unsigned long long refined_hits = 0;
      cuda_check(cudaMemcpy(&refined_hits, d_refined_hits, sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(refined_hits)");
      if (!scratch)
        cuda_check(cudaFree(d_refined_hits), "cudaFree(refined_hits)");
      result.direct_refine_ms = static_cast<double>(refine_ms);
      result.refined_exact_count = result.q + static_cast<std::uint64_t>(refined_hits);
      result.count = result.refined_exact_count;
    }
    else
    {
      std::uint32_t *d_u_indices = nullptr;
      unsigned int *d_u_count = nullptr;
      if (scratch)
      {
        d_u_indices = scratch->d_u_indices;
        d_u_count = scratch->d_u_count;
      }
      else
      {
        cuda_check(cudaMalloc(&d_u_indices, resident.n * sizeof(std::uint32_t)),
                   "cudaMalloc(u_indices)");
        cuda_check(cudaMalloc(&d_u_count, sizeof(unsigned int)), "cudaMalloc(u_count)");
      }
      cuda_check(cudaMemset(d_u_count, 0, sizeof(unsigned int)), "cudaMemset(u_count)");

      cudaEvent_t compact_start = nullptr;
      cudaEvent_t compact_stop = nullptr;
      if (run_options.collect_phase_timing)
      {
        if (scratch)
        {
          compact_start = scratch->compact_start;
          compact_stop = scratch->compact_stop;
        }
        else
        {
          cuda_check(cudaEventCreate(&compact_start), "cudaEventCreate(compact_start)");
          cuda_check(cudaEventCreate(&compact_stop), "cudaEventCreate(compact_stop)");
        }
        cuda_check(cudaEventRecord(compact_start), "cudaEventRecord(compact_start)");
      }
      // compact_u_ms includes the dense row-state compaction pass over N.
      compact_u_indices_kernel<<<static_cast<int>(resident.grid), options.block_threads>>>(
          d_row_states,
          resident.n,
          d_u_indices,
          d_u_count);
      cuda_check(cudaGetLastError(), "compact_u_indices_kernel launch");
      float compact_ms = 0.0f;
      if (run_options.collect_phase_timing)
      {
        cuda_check(cudaEventRecord(compact_stop), "cudaEventRecord(compact_stop)");
        cuda_check(cudaEventSynchronize(compact_stop), "cudaEventSynchronize(compact_stop)");
        cuda_check(cudaEventElapsedTime(&compact_ms, compact_start, compact_stop),
                   "cudaEventElapsedTime(compact)");
        if (!scratch)
        {
          cuda_check(cudaEventDestroy(compact_stop), "cudaEventDestroy(compact_stop)");
          cuda_check(cudaEventDestroy(compact_start), "cudaEventDestroy(compact_start)");
        }
      }

      unsigned int u_count = 0;
      cuda_check(cudaMemcpy(&u_count, d_u_count, sizeof(unsigned int), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(u_count)");
      if (static_cast<std::uint64_t>(u_count) != result.u)
        fail("compact U count does not match row-state U count");

      std::uint64_t compact_u_bytes = static_cast<std::uint64_t>(u_count) * sizeof(std::uint32_t);
      result.compact_U_bytes = compact_u_bytes;
      result.compact_u_ms = static_cast<double>(compact_ms);

      if (u_count != 0u)
      {
        cudaEvent_t refine_start = nullptr;
        cudaEvent_t refine_stop = nullptr;
        if (run_options.collect_phase_timing)
        {
          if (scratch)
          {
            refine_start = scratch->refine_start;
            refine_stop = scratch->refine_stop;
          }
          else
          {
            cuda_check(cudaEventCreate(&refine_start), "cudaEventCreate(refine_start)");
            cuda_check(cudaEventCreate(&refine_stop), "cudaEventCreate(refine_stop)");
          }
          cuda_check(cudaEventRecord(refine_start), "cudaEventRecord(refine_start)");
        }
        compact_u_refine_raw_kernel<<<static_cast<int>(resident.grid), options.block_threads>>>(
            raw_artifact->values,
            d_u_indices,
            u_count,
            options.threshold,
            d_refined_hits);
        cuda_check(cudaGetLastError(), "compact_u_refine_raw_kernel launch");
        float refine_ms = 0.0f;
        if (run_options.collect_phase_timing)
        {
          cuda_check(cudaEventRecord(refine_stop), "cudaEventRecord(refine_stop)");
          cuda_check(cudaEventSynchronize(refine_stop), "cudaEventSynchronize(refine_stop)");
          cuda_check(cudaEventElapsedTime(&refine_ms, refine_start, refine_stop),
                     "cudaEventElapsedTime(refine)");
          if (!scratch)
          {
            cuda_check(cudaEventDestroy(refine_stop), "cudaEventDestroy(refine_stop)");
            cuda_check(cudaEventDestroy(refine_start), "cudaEventDestroy(refine_start)");
          }
        }
        result.compact_u_refine_ms = static_cast<double>(refine_ms);
      }

      unsigned long long refined_hits = 0;
      cuda_check(cudaMemcpy(&refined_hits, d_refined_hits, sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(refined_hits)");
      if (!scratch)
      {
        cuda_check(cudaFree(d_refined_hits), "cudaFree(refined_hits)");
        cuda_check(cudaFree(d_u_count), "cudaFree(u_count)");
        cuda_check(cudaFree(d_u_indices), "cudaFree(u_indices)");
      }
      result.refined_exact_count = result.q + static_cast<std::uint64_t>(refined_hits);
      result.count = result.refined_exact_count;
      result.total_refined_ms = result.gpu_total_ms;
    }
  }

  if (!scratch)
  {
    if (d_row_states != nullptr) cuda_check(cudaFree(d_row_states), "cudaFree(row_states)");
    cuda_check(cudaFree(d_block_stats), "cudaFree(block_stats)");
    cuda_check(cudaFree(d_threshold_combined), "cudaFree(threshold_combined)");
    cuda_check(cudaFree(d_total_bits_by_segment), "cudaFree(total_bits_by_segment)");
    cuda_check(cudaFree(d_active_plane_count), "cudaFree(active_plane_count)");
    cuda_check(cudaFree(d_segment_mode), "cudaFree(segment_mode)");
    cuda_check(cudaFree(d_threshold_bytes), "cudaFree(threshold_bytes)");
  }

  result.gpu_total_ms = elapsed_ms(gpu_total_start, std::chrono::steady_clock::now());
  return result;
}

ByteplaneCountGtResult gpu_count_gt(const exp3_real::Dataset &dataset,
                                    const GpuRawArtifact *raw_artifact,
                                    const PreparedResidentQuery &prepared,
                                    const ByteplaneCountGtOptions &options)
{
  GpuResidentArtifact resident = stage_gpu_artifact(dataset, options);
  ByteplaneCountGtResult result = gpu_count_gt_resident(dataset,
                                                        resident,
                                                        raw_artifact,
                                                        prepared,
                                                        options,
                                                        resident.stage_ms);
  free_gpu_artifact(resident);
  result.gpu_total_ms += result.gpu_stage_ms;
  if (options.direct_refine_raw && raw_artifact != nullptr)
  {
    // direct_total_ms excludes raw staging; raw_stage_ms is emitted separately.
    result.direct_total_ms = result.gpu_total_ms;
  }
  else if (options.compact_u_refine_raw && raw_artifact != nullptr)
  {
    result.total_refined_ms = result.gpu_total_ms;
  }
  return result;
}

} // namespace

ByteplaneCountGtResult byteplane_count_gt(const ByteplaneCountGtOptions &options)
{
  if (options.artifact_root.empty()) fail("artifact_root is required");
  if (options.block_threads <= 0 || options.block_threads > 1024 || options.block_threads % 32 != 0)
    fail("block_threads must be a positive multiple of 32 up to 1024");
  if (options.direct_refine_raw && options.compact_u_refine_raw)
    fail("direct and compact U refinement modes are mutually exclusive");
  if (options.no_gpu && options.compact_u_refine_raw)
    fail("compact U refinement requires GPU mode");

  auto load_start = std::chrono::steady_clock::now();
  exp3_real::Dataset dataset = exp3_real::load_dataset(options.artifact_root);
  double artifact_load_ms = elapsed_ms(load_start, std::chrono::steady_clock::now());
  if (dataset.manifest.max_plane_count > EXP3_RUNTIME_MAX_SUBCOLUMNS)
    fail("artifact max_plane_count exceeds runtime pointer capacity");

  std::vector<std::pair<double, double>> segment_min_max;
  std::filesystem::path raw_path;
  if (!options.raw_root.empty())
    raw_path = options.raw_root / (dataset.manifest.dataset + ".f64le.bin");
  else
    raw_path = default_raw_path(options.artifact_root, dataset.manifest.dataset);

  double segment_minmax_ms = 0.0;
  if (!options.segment_minmax_path.empty())
  {
    auto minmax_start = std::chrono::steady_clock::now();
    segment_min_max = read_segment_min_max_csv(options.segment_minmax_path, dataset.segments.size());
    segment_minmax_ms = elapsed_ms(minmax_start, std::chrono::steady_clock::now());
  }
  else if (std::filesystem::exists(raw_path))
  {
    auto minmax_start = std::chrono::steady_clock::now();
    segment_min_max = compute_segment_min_max(
        raw_path, dataset.manifest.segment_size, dataset.segments.size());
    segment_minmax_ms = elapsed_ms(minmax_start, std::chrono::steady_clock::now());
  }
  else
  {
    segment_min_max.resize(dataset.segments.size(),
                           {std::numeric_limits<double>::lowest(),
                            std::numeric_limits<double>::max()});
  }

  PreparedResidentQuery prepared =
      prepare_resident_query(dataset, segment_min_max, options.threshold, options.k);

  if (options.no_gpu)
  {
    auto start = std::chrono::steady_clock::now();
    ByteplaneCountGtResult result = cpu_count_gt(dataset,
                                                 prepared.threshold_info,
                                                 prepared.active_plane_count,
                                                 prepared.effective_k,
                                                 options.block_threads,
                                                 options.row_state_output_path);
    auto stop = std::chrono::steady_clock::now();
    result.artifact_load_ms = artifact_load_ms;
    result.segment_minmax_ms = segment_minmax_ms;
    result.threshold_prep_ms = prepared.threshold_prep_ms;
    result.threshold_classify_ms = prepared.threshold_classify_ms;
    result.threshold_encode_ms = prepared.threshold_encode_ms;
    result.threshold_base_ms = prepared.threshold_base_ms;
    result.threshold_subtract_extract_ms = prepared.threshold_subtract_extract_ms;
    result.threshold_pack_ms = prepared.threshold_pack_ms;
    result.threshold_static_candidate_ms = prepared.threshold_static_candidate_ms;
    result.threshold_dependent_candidate_ms = prepared.threshold_dependent_candidate_ms;
    result.threshold_allq_segments = prepared.threshold_allq_segments;
    result.threshold_alld_segments = prepared.threshold_alld_segments;
    result.threshold_mixed_segments = prepared.threshold_mixed_segments;
    result.primitive_ms = elapsed_ms(start, stop);
    result.output_mode = "encoded_interval";
    return result;
  }

  ByteplaneCountGtResult result = gpu_count_gt(dataset,
                                               nullptr,
                                               prepared,
                                               options);
  result.artifact_load_ms = artifact_load_ms;
  result.segment_minmax_ms = segment_minmax_ms;
  result.threshold_prep_ms = prepared.threshold_prep_ms;
  result.threshold_classify_ms = prepared.threshold_classify_ms;
  result.threshold_encode_ms = prepared.threshold_encode_ms;
  result.threshold_base_ms = prepared.threshold_base_ms;
  result.threshold_subtract_extract_ms = prepared.threshold_subtract_extract_ms;
  result.threshold_pack_ms = prepared.threshold_pack_ms;
  result.threshold_static_candidate_ms = prepared.threshold_static_candidate_ms;
  result.threshold_dependent_candidate_ms = prepared.threshold_dependent_candidate_ms;
  result.threshold_allq_segments = prepared.threshold_allq_segments;
  result.threshold_alld_segments = prepared.threshold_alld_segments;
  result.threshold_mixed_segments = prepared.threshold_mixed_segments;
  if (options.compact_u_refine_raw)
    result.total_refined_ms = result.gpu_total_ms;
  return result;
}

std::vector<ByteplaneCountGtResult> byteplane_count_gt_batch(
    const ByteplaneCountGtOptions &options,
    const std::vector<ByteplaneCountGtRequest> &requests)
{
  if (options.artifact_root.empty()) fail("artifact_root is required");
  if (requests.empty()) fail("at least one batch request is required");
  if (options.no_gpu) fail("byteplane_count_gt_batch currently requires GPU mode");
  if (options.block_threads <= 0 || options.block_threads > 1024 || options.block_threads % 32 != 0)
    fail("block_threads must be a positive multiple of 32 up to 1024");
  if (options.direct_refine_raw && options.compact_u_refine_raw)
    fail("direct and compact U refinement modes are mutually exclusive");

  auto load_start = std::chrono::steady_clock::now();
  exp3_real::Dataset dataset = exp3_real::load_dataset(options.artifact_root);
  double artifact_load_ms = elapsed_ms(load_start, std::chrono::steady_clock::now());
  if (dataset.manifest.max_plane_count > EXP3_RUNTIME_MAX_SUBCOLUMNS)
    fail("artifact max_plane_count exceeds runtime pointer capacity");

  std::vector<std::pair<double, double>> segment_min_max;
  std::filesystem::path raw_path;
  if (!options.raw_root.empty())
    raw_path = options.raw_root / (dataset.manifest.dataset + ".f64le.bin");
  else
    raw_path = default_raw_path(options.artifact_root, dataset.manifest.dataset);

  std::vector<double> raw_values;
  if (options.direct_refine_raw || options.compact_u_refine_raw)
    raw_values = load_raw_values(raw_path, dataset.manifest.value_count);

  double segment_minmax_ms = 0.0;
  if (!options.segment_minmax_path.empty())
  {
    auto minmax_start = std::chrono::steady_clock::now();
    segment_min_max = read_segment_min_max_csv(options.segment_minmax_path, dataset.segments.size());
    segment_minmax_ms = elapsed_ms(minmax_start, std::chrono::steady_clock::now());
  }
  else if (!raw_values.empty())
  {
    auto minmax_start = std::chrono::steady_clock::now();
    segment_min_max = compute_segment_min_max_from_values(
        raw_values, dataset.manifest.segment_size, dataset.segments.size());
    segment_minmax_ms = elapsed_ms(minmax_start, std::chrono::steady_clock::now());
  }
  else if (std::filesystem::exists(raw_path))
  {
    auto minmax_start = std::chrono::steady_clock::now();
    segment_min_max = compute_segment_min_max(
        raw_path, dataset.manifest.segment_size, dataset.segments.size());
    segment_minmax_ms = elapsed_ms(minmax_start, std::chrono::steady_clock::now());
  }
  else
  {
    segment_min_max.resize(dataset.segments.size(),
                           {std::numeric_limits<double>::lowest(),
                            std::numeric_limits<double>::max()});
  }

  GpuResidentArtifact resident = stage_gpu_artifact(dataset, options);
  GpuRawArtifact raw_artifact;
  if (options.direct_refine_raw || options.compact_u_refine_raw)
    raw_artifact = stage_gpu_raw_values(raw_values);

  bool scratch_needs_row_states = options.direct_refine_raw || options.compact_u_refine_raw;
  if (!scratch_needs_row_states)
    for (const auto &req : requests)
      if (!req.row_state_output_path.empty()) { scratch_needs_row_states = true; break; }

  GpuResidentScratch scratch;
  scratch.allocate(dataset.segments.size(),
                   dataset.manifest.max_plane_count,
                   resident.n,
                   resident.grid,
                   scratch_needs_row_states,
                   options.direct_refine_raw || options.compact_u_refine_raw,
                   options.compact_u_refine_raw);

  std::vector<ByteplaneCountGtResult> results;
  results.reserve(requests.size());
  for (const ByteplaneCountGtRequest &request : requests)
  {
    std::vector<double> per_query_times;
    ByteplaneCountGtResult result;
    PreparedResidentQuery prepared =
        prepare_resident_query(dataset, segment_min_max, request.threshold, request.k);

    ByteplaneCountGtOptions request_options = options;
    request_options.threshold = request.threshold;
    request_options.k = request.k;
    request_options.row_state_output_path = request.row_state_output_path;

    if (options.repeat > 0)
    {
      gpu_count_gt_resident(dataset,
                            resident,
                            (options.direct_refine_raw || options.compact_u_refine_raw)
                                ? &raw_artifact
                                : nullptr,
                            prepared,
                            request_options,
                            resident.stage_ms,
                            &scratch,
                            ResidentRunOptions{true, false});

      cudaProfilerStart();
      for (int iter = 0; iter < options.repeat; ++iter)
      {
        auto iter_start = std::chrono::steady_clock::now();
        ByteplaneCountGtResult iter_result = gpu_count_gt_resident(
            dataset,
            resident,
            (options.direct_refine_raw || options.compact_u_refine_raw)
                ? &raw_artifact
                : nullptr,
            prepared,
            request_options,
            resident.stage_ms,
            &scratch,
            ResidentRunOptions{false, false});
        double iter_ms = elapsed_ms(iter_start, std::chrono::steady_clock::now());

        per_query_times.push_back(iter_ms);
        if (iter == options.repeat - 1)
        {
          result = std::move(iter_result);
          result.threshold_prep_ms = prepared.threshold_prep_ms;
          result.threshold_classify_ms = prepared.threshold_classify_ms;
          result.threshold_encode_ms = prepared.threshold_encode_ms;
          result.threshold_base_ms = prepared.threshold_base_ms;
          result.threshold_subtract_extract_ms = prepared.threshold_subtract_extract_ms;
          result.threshold_pack_ms = prepared.threshold_pack_ms;
          result.threshold_static_candidate_ms = prepared.threshold_static_candidate_ms;
          result.threshold_dependent_candidate_ms = prepared.threshold_dependent_candidate_ms;
          result.threshold_allq_segments = prepared.threshold_allq_segments;
          result.threshold_alld_segments = prepared.threshold_alld_segments;
          result.threshold_mixed_segments = prepared.threshold_mixed_segments;
        }
      }
      cudaProfilerStop();

      std::vector<double> sorted = per_query_times;
      std::sort(sorted.begin(), sorted.end());
      result.per_query_ms_median = sorted[sorted.size() / 2];
      result.per_query_ms_p10 = sorted[static_cast<std::size_t>(sorted.size() * 0.1)];
      result.per_query_ms_p90 = sorted[static_cast<std::size_t>(sorted.size() * 0.9)];
    }
    else
    {
      auto iter_start = std::chrono::steady_clock::now();
      result = gpu_count_gt_resident(dataset,
                                     resident,
                                     (options.direct_refine_raw || options.compact_u_refine_raw)
                                         ? &raw_artifact
                                         : nullptr,
                                     prepared,
                                     request_options,
                                     resident.stage_ms,
                                     &scratch);
      double iter_ms = elapsed_ms(iter_start, std::chrono::steady_clock::now());
      result.threshold_prep_ms = prepared.threshold_prep_ms;
      result.threshold_classify_ms = prepared.threshold_classify_ms;
      result.threshold_encode_ms = prepared.threshold_encode_ms;
      result.threshold_base_ms = prepared.threshold_base_ms;
      result.threshold_subtract_extract_ms = prepared.threshold_subtract_extract_ms;
      result.threshold_pack_ms = prepared.threshold_pack_ms;
      result.threshold_static_candidate_ms = prepared.threshold_static_candidate_ms;
      result.threshold_dependent_candidate_ms = prepared.threshold_dependent_candidate_ms;
      result.threshold_allq_segments = prepared.threshold_allq_segments;
      result.threshold_alld_segments = prepared.threshold_alld_segments;
      result.threshold_mixed_segments = prepared.threshold_mixed_segments;
      result.per_query_ms_median = iter_ms;
      result.per_query_ms_p10 = 0.0;
      result.per_query_ms_p90 = 0.0;
    }

    result.artifact_load_ms = artifact_load_ms;
    result.segment_minmax_ms = segment_minmax_ms;
    if (options.compact_u_refine_raw)
      result.total_refined_ms = result.gpu_total_ms;

    results.push_back(result);
  }

  scratch.deallocate();
  if (options.direct_refine_raw || options.compact_u_refine_raw)
    free_gpu_raw_values(raw_artifact);
  free_gpu_artifact(resident);
  return results;
}

} // namespace byteplane_scan
