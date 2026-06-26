// ============================================================================
// bench_threshold_prep_decomp.cu
//
// Micro-benchmark: threshold preparation decomposition for byte-plane scan.
// Measures each sub-stage (classify, encode, subtract/extract, metadata pack,
// H2D copy) individually across target selectivities.
//
// Also includes a fused single-pass prototype that combines compute_threshold_bytes
// + metadata packing into one segment loop.
//
// Build: linked against byteplane_count_gt library (provides exp3_real_data_layout
//        and buff_codec linkage).
// ============================================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "exp3_real_data_layout.hpp"
#include "byteplane_count_gt.hpp"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr uint8_t kSegmentModeMixed = 0;
static constexpr uint8_t kSegmentModeAllQualified = 1;
static constexpr uint8_t kSegmentModeAllDisqualified = 2;

static constexpr int kWarmup = 20;
static constexpr int kIters = 200;

// Target selectivities (as fractions)
static constexpr double kSelectivities[] = {0.01, 0.10, 0.50, 0.90, 0.99};
static constexpr int kNumSelectivities = 5;

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------
[[noreturn]] void fail(const std::string &msg)
{
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  std::exit(2);
}

void cuda_check(cudaError_t err, const char *where)
{
  if (err == cudaSuccess) return;
  std::fprintf(stderr, "cuda error at %s: %s\n", where, cudaGetErrorString(err));
  std::exit(2);
}

// ---------------------------------------------------------------------------
// Local copy of load_raw_values (mirrors byteplane_count_gt.cu internals)
// ---------------------------------------------------------------------------
std::vector<double> load_raw_values(const std::filesystem::path &path,
                                    std::uint64_t expected_count)
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

// ---------------------------------------------------------------------------
// Timing utility
// ---------------------------------------------------------------------------
double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point stop)
{
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

// ---------------------------------------------------------------------------
// Little-endian byte utilities (mirror byteplane_count_gt.cu internals)
// ---------------------------------------------------------------------------
uint64_t le_bytes_to_u64(const std::vector<uint8_t> &bytes)
{
  uint64_t value = 0;
  for (size_t i = 0; i < bytes.size() && i < sizeof(uint64_t); ++i)
    value |= static_cast<uint64_t>(bytes[i]) << (i * 8);
  return value;
}

std::vector<uint8_t> subtract_le_bytes(const std::vector<uint8_t> &a,
                                       const std::vector<uint8_t> &b)
{
  std::vector<uint8_t> result(a.size(), 0);
  int borrow = 0;
  for (size_t i = 0; i < a.size(); ++i)
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
    result[i] = static_cast<uint8_t>(value);
  }
  while (!result.empty() && result.back() == 0) result.pop_back();
  return result;
}

void extract_plane_bytes_from_combined_le(const std::vector<uint8_t> &combined_le,
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

std::vector<uint8_t> local_encode_to_le_bytes(double value, int32_t scale_exponent)
{
  long double scaled = std::floor(std::ldexp(static_cast<long double>(value), -scale_exponent));
  if (scaled == 0.0L) return {};
  uint64_t code = static_cast<uint64_t>(scaled);
  std::vector<uint8_t> result;
  result.reserve(sizeof(uint64_t));
  for (size_t i = 0; i < sizeof(uint64_t); ++i)
    result.push_back(static_cast<uint8_t>((code >> (i * 8)) & 0xFF));
  return result;
}

// ---------------------------------------------------------------------------
// Mini SegmentThresholdInfo + ThresholdPrepTiming (matching library internals)
// ---------------------------------------------------------------------------
struct SegmentThresholdInfo
{
  std::vector<uint8_t> threshold_bytes;
  bool all_qualified = false;
  bool all_disqualified = false;
  uint64_t threshold_combined = 0;
};

struct ThresholdPrepTiming
{
  double classify_ms = 0.0;
  double encode_ms = 0.0;
  double base_ms = 0.0;
  double subtract_extract_ms = 0.0;
};

// ---------------------------------------------------------------------------
// compute_segment_min_max_from_values
// Mirrors byteplane_count_gt.cu's anonymous-namespace function.
// ---------------------------------------------------------------------------
std::vector<std::pair<double, double>> compute_segment_min_max_from_values(
    const std::vector<double> &values,
    uint64_t segment_rows,
    uint64_t num_segments)
{
  std::vector<std::pair<double, double>> result;
  result.reserve(static_cast<size_t>(num_segments));
  uint64_t remaining = static_cast<uint64_t>(values.size());
  for (uint64_t seg = 0; seg < num_segments; ++seg)
  {
    uint64_t current = std::min(remaining, segment_rows);
    if (current == 0) fail("data ended before all segments were read");
    size_t offset = static_cast<size_t>(values.size() - remaining);
    double seg_min = values[offset];
    double seg_max = values[offset];
    for (uint64_t i = 1; i < current; ++i)
    {
      double v = values[offset + static_cast<size_t>(i)];
      seg_min = std::min(seg_min, v);
      seg_max = std::max(seg_max, v);
    }
    result.emplace_back(seg_min, seg_max);
    remaining -= current;
  }
  return result;
}

// ---------------------------------------------------------------------------
// compute_threshold_bytes
// Mirrors byteplane_count_gt.cu's compute_threshold_bytes with timing param.
// ---------------------------------------------------------------------------
std::vector<SegmentThresholdInfo> compute_threshold_bytes(
    const exp3_real::Dataset &dataset,
    const std::vector<std::pair<double, double>> &segment_min_max,
    double threshold_fp64,
    ThresholdPrepTiming *timing = nullptr)
{
  std::vector<SegmentThresholdInfo> result;
  result.reserve(dataset.segments.size());

  for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
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

    auto classify_start = timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
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

    uint32_t frac = segment.fractional_bits;
    int32_t scale_exponent = -static_cast<int32_t>(frac);
    uint64_t integer_base_u64 = le_bytes_to_u64(segment.integer_base_le);
    bool uses_bounded_precision = segment.integer_base_is_v2 ||
                                  (segment.raw_fractional_bits != segment.fractional_bits) ||
                                  (segment.precision_cap_bits != segment.fractional_bits);
    std::vector<uint8_t> t_code_le;
    std::vector<uint8_t> base_shifted_le;

    if (uses_bounded_precision)
    {
      auto te_start = timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
      {
        long double scale = std::ldexp(1.0L, static_cast<int>(frac));
        uint64_t t_code = static_cast<uint64_t>(
            std::floor(static_cast<long double>(threshold_fp64) * scale));
        for (size_t i = 0; i < sizeof(uint64_t); ++i)
          t_code_le.push_back(static_cast<uint8_t>((t_code >> (i * 8)) & 0xFF));
      }
      if (timing) timing->encode_ms += elapsed_ms(te_start, std::chrono::steady_clock::now());
      auto bs_start = timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
      base_shifted_le = segment.integer_base_is_v2
                            ? segment.integer_base_le
                            : local_encode_to_le_bytes(
                                  static_cast<double>(integer_base_u64), scale_exponent);
      if (timing) timing->base_ms += elapsed_ms(bs_start, std::chrono::steady_clock::now());
    }
    else
    {
      auto te_start = timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
      t_code_le = local_encode_to_le_bytes(threshold_fp64, scale_exponent);
      if (timing) timing->encode_ms += elapsed_ms(te_start, std::chrono::steady_clock::now());
      auto bs_start = timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
      base_shifted_le = local_encode_to_le_bytes(
          static_cast<double>(integer_base_u64), scale_exponent);
      if (timing) timing->base_ms += elapsed_ms(bs_start, std::chrono::steady_clock::now());
    }

    auto subext_start = timing ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
    size_t max_len = std::max(t_code_le.size(), base_shifted_le.size());
    t_code_le.resize(max_len, 0);
    base_shifted_le.resize(max_len, 0);
    std::vector<uint8_t> combined_T_le = subtract_le_bytes(t_code_le, base_shifted_le);

    size_t total_bits = segment.get_fixed_len_bits();
    extract_plane_bytes_from_combined_le(
        combined_T_le, total_bits, info.threshold_bytes, dataset.manifest.max_plane_count);
    info.threshold_combined = le_bytes_to_u64(combined_T_le);
    if (timing) timing->subtract_extract_ms += elapsed_ms(subext_start, std::chrono::steady_clock::now());
    result.push_back(std::move(info));
  }

  return result;
}

// ---------------------------------------------------------------------------
// prepare_resident_query (multi-pass, with all sub-stage timings)
// Mirrors byteplane_count_gt.cu's prepare_resident_query.
// ---------------------------------------------------------------------------
struct PreparedQuery {
  std::vector<SegmentThresholdInfo> threshold_info;
  std::vector<uint8_t> segment_mode;
  std::vector<uint32_t> active_plane_count;
  std::vector<uint32_t> total_bits_by_segment;
  std::vector<uint8_t> h_threshold_flat;
  std::vector<uint64_t> h_threshold_combined;
  uint32_t effective_k = 0;

  double total_prep_ms = 0.0;
  double classify_ms = 0.0;
  double encode_ms = 0.0;
  double base_ms = 0.0;
  double subtract_extract_ms = 0.0;
  double pack_ms = 0.0;
};

PreparedQuery prepare_resident_query(const exp3_real::Dataset &dataset,
                                     const std::vector<std::pair<double, double>> &segment_min_max,
                                     double threshold,
                                     int requested_k)
{
  auto threshold_start = std::chrono::steady_clock::now();

  ThresholdPrepTiming timing;
  PreparedQuery prepared;
  prepared.threshold_info = compute_threshold_bytes(dataset, segment_min_max, threshold, &timing);
  prepared.total_prep_ms = elapsed_ms(threshold_start, std::chrono::steady_clock::now());
  // We'll recompute pack timing more specifically; the above total includes
  // the pack work below, so keep it as the overall wall clock.

  prepared.effective_k =
      requested_k < 0 ? static_cast<uint32_t>(dataset.manifest.max_plane_count)
                      : static_cast<uint32_t>(requested_k);
  if (prepared.effective_k == 0) fail("k must be positive");

  size_t segment_count = dataset.segments.size();
  prepared.segment_mode.assign(segment_count, kSegmentModeMixed);
  prepared.active_plane_count.assign(segment_count, 0);
  prepared.total_bits_by_segment.assign(segment_count, 0);
  prepared.h_threshold_flat.assign(segment_count * dataset.manifest.max_plane_count, 0);
  prepared.h_threshold_combined.assign(segment_count, 0);

  auto pack_start = std::chrono::steady_clock::now();
  for (size_t seg = 0; seg < segment_count; ++seg)
  {
    const auto &segment = dataset.segments[seg];
    const auto &ti = prepared.threshold_info[seg];
    if (ti.all_qualified)
      prepared.segment_mode[seg] = kSegmentModeAllQualified;
    else if (ti.all_disqualified)
      prepared.segment_mode[seg] = kSegmentModeAllDisqualified;

    prepared.active_plane_count[seg] =
        std::min<uint32_t>(prepared.effective_k, segment.active_plane_count);
    prepared.total_bits_by_segment[seg] =
        segment.fractional_bits + segment.integer_offset_bits;
    std::memcpy(prepared.h_threshold_flat.data() + seg * dataset.manifest.max_plane_count,
                ti.threshold_bytes.data(),
                dataset.manifest.max_plane_count);
    prepared.h_threshold_combined[seg] = ti.threshold_combined;
  }

  prepared.classify_ms = timing.classify_ms;
  prepared.encode_ms = timing.encode_ms;
  prepared.base_ms = timing.base_ms;
  prepared.subtract_extract_ms = timing.subtract_extract_ms;
  prepared.pack_ms = elapsed_ms(pack_start, std::chrono::steady_clock::now());

  return prepared;
}

// ---------------------------------------------------------------------------
// FUSED single-pass prototype
//
// Combines compute_threshold_bytes + metadata packing into a single segment
// loop. No separate vector<SegmentThresholdInfo> allocation per segment;
// threshold_bytes and metadata are written directly into flat arrays.
// ---------------------------------------------------------------------------
PreparedQuery prepare_resident_query_fused(const exp3_real::Dataset &dataset,
                                           const std::vector<std::pair<double, double>> &segment_min_max,
                                           double threshold,
                                           int requested_k)
{
  auto start = std::chrono::steady_clock::now();

  PreparedQuery prepared;
  size_t segment_count = dataset.segments.size();
  uint32_t max_planes = static_cast<uint32_t>(dataset.manifest.max_plane_count);

  prepared.effective_k =
      requested_k < 0 ? max_planes : static_cast<uint32_t>(requested_k);
  if (prepared.effective_k == 0) fail("k must be positive");

  prepared.segment_mode.assign(segment_count, kSegmentModeMixed);
  prepared.active_plane_count.assign(segment_count, 0);
  prepared.total_bits_by_segment.assign(segment_count, 0);
  prepared.h_threshold_flat.assign(segment_count * max_planes, 0);
  prepared.h_threshold_combined.assign(segment_count, 0);

  // No separate threshold_info vector — write directly into flat arrays.
  for (size_t seg = 0; seg < segment_count; ++seg)
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

    uint32_t active_planes = std::min<uint32_t>(prepared.effective_k, segment.active_plane_count);
    prepared.active_plane_count[seg] = active_planes;
    prepared.total_bits_by_segment[seg] = segment.fractional_bits + segment.integer_offset_bits;

    if (threshold < seg_min)
    {
      prepared.segment_mode[seg] = kSegmentModeAllQualified;
      // threshold_bytes remain zero-initialized (already done by assign)
      prepared.h_threshold_combined[seg] = 0;
      continue;
    }
    if (threshold >= seg_max)
    {
      prepared.segment_mode[seg] = kSegmentModeAllDisqualified;
      prepared.h_threshold_combined[seg] = 0;
      continue;
    }

    // Mixed segment: compute threshold bytes directly into flat array.
    uint32_t frac = segment.fractional_bits;
    int32_t scale_exponent = -static_cast<int32_t>(frac);
    uint64_t integer_base_u64 = le_bytes_to_u64(segment.integer_base_le);
    bool uses_bounded_precision = segment.integer_base_is_v2 ||
                                  (segment.raw_fractional_bits != segment.fractional_bits) ||
                                  (segment.precision_cap_bits != segment.fractional_bits);
    std::vector<uint8_t> t_code_le;
    std::vector<uint8_t> base_shifted_le;

    if (uses_bounded_precision)
    {
      {
        long double scale = std::ldexp(1.0L, static_cast<int>(frac));
        uint64_t t_code = static_cast<uint64_t>(
            std::floor(static_cast<long double>(threshold) * scale));
        for (size_t i = 0; i < sizeof(uint64_t); ++i)
          t_code_le.push_back(static_cast<uint8_t>((t_code >> (i * 8)) & 0xFF));
      }
      base_shifted_le = segment.integer_base_is_v2
                            ? segment.integer_base_le
                            : local_encode_to_le_bytes(
                                  static_cast<double>(integer_base_u64), scale_exponent);
    }
    else
    {
      t_code_le = local_encode_to_le_bytes(threshold, scale_exponent);
      base_shifted_le = local_encode_to_le_bytes(
          static_cast<double>(integer_base_u64), scale_exponent);
    }

    size_t max_len = std::max(t_code_le.size(), base_shifted_le.size());
    t_code_le.resize(max_len, 0);
    base_shifted_le.resize(max_len, 0);
    std::vector<uint8_t> combined_T_le = subtract_le_bytes(t_code_le, base_shifted_le);

    size_t total_bits = segment.get_fixed_len_bits();
    std::vector<uint8_t> plane_bytes(max_planes, 0);
    extract_plane_bytes_from_combined_le(
        combined_T_le, total_bits, plane_bytes, max_planes);

    // Write directly to flat array — no intermediate SegmentThresholdInfo.
    std::memcpy(prepared.h_threshold_flat.data() + seg * max_planes,
                plane_bytes.data(), max_planes);
    prepared.h_threshold_combined[seg] = le_bytes_to_u64(combined_T_le);
  }

  prepared.total_prep_ms = elapsed_ms(start, std::chrono::steady_clock::now());
  return prepared;
}

// ---------------------------------------------------------------------------
// Compute percentile thresholds from raw FP64 data
// ---------------------------------------------------------------------------
std::vector<double> compute_percentile_thresholds(const std::vector<double> &raw_values)
{
  // Fully sort a copy, then pick values at percentile indices.
  std::vector<double> sorted = raw_values;
  std::sort(sorted.begin(), sorted.end());

  std::vector<double> thresholds;
  thresholds.reserve(kNumSelectivities);
  for (int i = 0; i < kNumSelectivities; ++i)
  {
    // We want a threshold such that about selectivity fraction of values are <= it.
    // For `> threshold` queries, this means ~(1-selectivity) of values pass.
    // Actually, we need to think about this:
    // - "target selectivity 1%" means we want the count of values > threshold to be 1% of total
    // - So threshold should be at the 99th percentile (99% of values are <= threshold)
    // - Wait: count_gt = number of values > threshold.
    //   If count_gt / total = 0.01, then count_le / total = 0.99
    //   So threshold = value at index floor(0.99 * N)
    // Actually, let's think again:
    // selectivity = count_gt / N
    // count_gt = number of values strictly greater than threshold
    // So if we want selectivity = 0.01, we need threshold such that 99% of values are <= threshold
    // and 1% are > threshold.
    // threshold = sorted[N - 1 - floor(selectivity * N)]
    // For selectivity=0.01, N=168M: threshold = sorted[168M - 1 - 1.68M] = sorted[166.3M]
    // That doesn't feel right.
    //
    // Let's rephrase: sorted[0] is min, sorted[N-1] is max.
    // If threshold = sorted[k], then count_gt = N - 1 - k (assuming no duplicates at boundary).
    // We want count_gt / N = selectivity
    // N - 1 - k = selectivity * N
    // k = N - 1 - selectivity * N
    //
    // For selectivity=0.01: k = N - 1 - 0.01*N = 0.99*N - 1
    // For selectivity=0.50: k = 0.50*N - 1
    // For selectivity=0.99: k = 0.01*N - 1
    //
    // Hmm but our thresholds need to be double values, not indices. Let me just
    // compute the index and return the value at that index.

    double sel = kSelectivities[i];
    // index such that ~sel fraction of values are *greater* than threshold
    int64_t idx = static_cast<int64_t>(static_cast<double>(sorted.size()) * (1.0 - sel));
    if (idx < 0) idx = 0;
    if (idx >= static_cast<int64_t>(sorted.size()))
      idx = static_cast<int64_t>(sorted.size()) - 1;

    // Use a small epsilon to avoid boundary ambiguity
    double val = sorted[static_cast<size_t>(idx)];
    // Ensure we have valid finite threshold
    if (!std::isfinite(val)) val = 0.0;
    thresholds.push_back(val);
  }

  return thresholds;
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------
struct Stats {
  double p10 = 0.0;
  double median = 0.0;
  double p90 = 0.0;
  double mean = 0.0;
  double stddev = 0.0;
};

Stats compute_stats(std::vector<double> &samples)
{
  if (samples.empty()) return {};
  std::sort(samples.begin(), samples.end());
  size_t n = samples.size();
  Stats s;
  s.p10 = samples[static_cast<size_t>(0.10 * static_cast<double>(n - 1))];
  s.median = samples[n / 2];
  s.p90 = samples[static_cast<size_t>(0.90 * static_cast<double>(n - 1))];
  double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  s.mean = sum / static_cast<double>(n);
  double sq_sum = 0.0;
  for (double v : samples)
  {
    double d = v - s.mean;
    sq_sum += d * d;
  }
  s.stddev = std::sqrt(sq_sum / static_cast<double>(n));
  return s;
}

// ---------------------------------------------------------------------------
// CSV writing
// ---------------------------------------------------------------------------
void write_breakdown_csv(const std::string &path,
                         const std::string &dataset_name,
                         double threshold,
                         double target_sel,
                         uint64_t segment_count,
                         const std::vector<double> &prep_total,
                         const std::vector<double> &classify,
                         const std::vector<double> &encode,
                         const std::vector<double> &subext,
                         const std::vector<double> &pack,
                         const std::vector<double> &h2d,
                         const std::vector<double> &fused_total)
{
  bool write_header = !std::filesystem::exists(path);
  std::FILE *f = std::fopen(path.c_str(), "ab");
  if (!f) fail("cannot open " + path);

  if (write_header)
  {
    std::fprintf(f,
        "dataset,threshold,target_selectivity,segment_count,iteration,"
        "threshold_prep_total_ms,segment_classify_ms,threshold_encode_ms,"
        "subtract_extract_ms,metadata_pack_ms,metadata_h2d_ms,"
        "ns_per_segment_total,ns_per_segment_classify,"
        "fused_single_pass_total_ms\n");
  }

  size_t n = prep_total.size();
  for (size_t i = 0; i < n; ++i)
  {
    double ns_per_seg_total = (segment_count > 0)
        ? (prep_total[i] * 1e6 / static_cast<double>(segment_count)) : 0.0;
    double ns_per_seg_classify = (segment_count > 0)
        ? (classify[i] * 1e6 / static_cast<double>(segment_count)) : 0.0;
    double fused = (i < fused_total.size()) ? fused_total[i] : 0.0;

    std::fprintf(f, "%s,%.17g,%.17g,%llu,%zu,"
                    "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                    "%.3f,%.3f,%.6f\n",
                 dataset_name.c_str(), threshold, target_sel,
                 (unsigned long long)segment_count, i,
                 prep_total[i], classify[i], encode[i], subext[i],
                 pack[i], h2d[i],
                 ns_per_seg_total, ns_per_seg_classify,
                 fused);
  }

  std::fclose(f);
}

void write_summary_csv(const std::string &path,
                       const std::string &dataset_name,
                       double threshold,
                       double target_sel,
                       uint64_t segment_count,
                       const Stats &prep_total_s,
                       const Stats &classify_s,
                       const Stats &encode_s,
                       const Stats &subext_s,
                       const Stats &pack_s,
                       const Stats &h2d_s,
                       const Stats &fused_s)
{
  bool write_header = !std::filesystem::exists(path);
  std::FILE *f = std::fopen(path.c_str(), "ab");
  if (!f) fail("cannot open " + path);

  if (write_header)
  {
    std::fprintf(f,
        "dataset,threshold,target_selectivity,segment_count,"
        "threshold_prep_total_ms_p10,threshold_prep_total_ms_median,threshold_prep_total_ms_p90,threshold_prep_total_ms_mean,threshold_prep_total_ms_stddev,"
        "segment_classify_ms_p10,segment_classify_ms_median,segment_classify_ms_p90,segment_classify_ms_mean,segment_classify_ms_stddev,"
        "threshold_encode_ms_p10,threshold_encode_ms_median,threshold_encode_ms_p90,threshold_encode_ms_mean,threshold_encode_ms_stddev,"
        "subtract_extract_ms_p10,subtract_extract_ms_median,subtract_extract_ms_p90,subtract_extract_ms_mean,subtract_extract_ms_stddev,"
        "metadata_pack_ms_p10,metadata_pack_ms_median,metadata_pack_ms_p90,metadata_pack_ms_mean,metadata_pack_ms_stddev,"
        "metadata_h2d_ms_p10,metadata_h2d_ms_median,metadata_h2d_ms_p90,metadata_h2d_ms_mean,metadata_h2d_ms_stddev,"
        "classify_fraction,encode_fraction,pack_fraction,"
        "ns_per_segment_total,ns_per_segment_classify,"
        "fused_single_pass_total_ms_median,speedup_vs_current\n");
  }

  double classify_fraction = (prep_total_s.mean > 0.0) ? classify_s.mean / prep_total_s.mean : 0.0;
  double encode_fraction = (prep_total_s.mean > 0.0) ? encode_s.mean / prep_total_s.mean : 0.0;
  double pack_fraction = (prep_total_s.mean > 0.0) ? pack_s.mean / prep_total_s.mean : 0.0;
  double ns_per_seg_total = (segment_count > 0)
      ? (prep_total_s.mean * 1e6 / static_cast<double>(segment_count)) : 0.0;
  double ns_per_seg_classify = (segment_count > 0)
      ? (classify_s.mean * 1e6 / static_cast<double>(segment_count)) : 0.0;
  double speedup = (fused_s.median > 0.0 && prep_total_s.median > 0.0)
      ? prep_total_s.median / fused_s.median : 1.0;

  std::fprintf(f, "%s,%.17g,%.17g,%llu,"
                  "%.6f,%.6f,%.6f,%.6f,%.6f,"
                  "%.6f,%.6f,%.6f,%.6f,%.6f,"
                  "%.6f,%.6f,%.6f,%.6f,%.6f,"
                  "%.6f,%.6f,%.6f,%.6f,%.6f,"
                  "%.6f,%.6f,%.6f,%.6f,%.6f,"
                  "%.6f,%.6f,%.6f,%.6f,%.6f,"
                  "%.6f,%.6f,%.6f,"
                  "%.3f,%.3f,"
                  "%.6f,%.6f\n",
               dataset_name.c_str(), threshold, target_sel,
               (unsigned long long)segment_count,
               prep_total_s.p10, prep_total_s.median, prep_total_s.p90, prep_total_s.mean, prep_total_s.stddev,
               classify_s.p10, classify_s.median, classify_s.p90, classify_s.mean, classify_s.stddev,
               encode_s.p10, encode_s.median, encode_s.p90, encode_s.mean, encode_s.stddev,
               subext_s.p10, subext_s.median, subext_s.p90, subext_s.mean, subext_s.stddev,
               pack_s.p10, pack_s.median, pack_s.p90, pack_s.mean, pack_s.stddev,
               h2d_s.p10, h2d_s.median, h2d_s.p90, h2d_s.mean, h2d_s.stddev,
               classify_fraction, encode_fraction, pack_fraction,
               ns_per_seg_total, ns_per_seg_classify,
               fused_s.median, speedup);
  std::fclose(f);
}

// ---------------------------------------------------------------------------
// Print helper for stderr progress
// ---------------------------------------------------------------------------
void print_separator()
{
  std::fprintf(stderr, "----------------------------------------------------------------------\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
  // --- Parse minimal args ---
  std::string raw_root = "results/buff_encoder_v2/raw_scientific";
  int device = 0;
  bool skip_gpu = false;   // if true, skip H2D measurement (for login-node compilation test)

  for (int i = 1; i < argc; ++i)
  {
    std::string_view a(argv[i]);
    auto next = [&]() -> std::string_view {
      if (i + 1 >= argc) fail("missing value for " + std::string(a));
      return std::string_view(argv[++i]);
    };
    if (a == "--device")
      device = std::stoi(std::string(next()));
    else if (a == "--raw-root")
      raw_root = std::string(next());
    else if (a == "--skip-gpu")
      skip_gpu = true;
    else if (a == "--help")
    {
      std::fprintf(stderr,
          "Usage: %s [options]\n"
          "  --device N      CUDA device index (default: 0)\n"
          "  --raw-root PATH Raw FP64 data root dir (default: results/buff_encoder_v2/raw_scientific)\n"
          "  --skip-gpu      Skip GPU-dependent measurements (H2D copy)\n"
          "  --help          This message\n",
          argv[0]);
      return 0;
    }
    else
      fail("unknown arg: " + std::string(a));
  }

  // --- Datasets to benchmark ---
  struct DatasetInfo {
    const char *name;
    const char *artifact_root;
    const char *raw_filename;
  };
  DatasetInfo datasets[] = {
    {"cesm_atm_cloud",
     "datasets/scientific/dev_buff_v2_scientific/cesm_atm_cloud/bfp-dec12",
     "cesm_atm_cloud.f64le.bin"},
    {"hurricane_u",
     "datasets/scientific/dev_buff_v2_scientific/hurricane_u/bfp-dec12",
     "hurricane_u.f64le.bin"},
  };
  constexpr int kNumDatasets = 2;

  // --- Output paths ---
  std::filesystem::path out_dir = "results/threshold_prep_decomp";
  std::filesystem::create_directories(out_dir);
  std::string breakdown_csv = (out_dir / "threshold_prep_breakdown.csv").string();
  std::string summary_csv = (out_dir / "threshold_prep_stage_summary.csv").string();

  // --- GPU setup (for H2D timing) ---
  int device_id = device;
  if (!skip_gpu)
  {
    cuda_check(cudaSetDevice(device_id), "cudaSetDevice");
    // Verify H200
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");
    std::string dev_name(prop.name);
    if (dev_name.find("H200") == std::string::npos)
    {
      std::fprintf(stderr, "FATAL: Expected H200 GPU, got: %s\n", prop.name);
      return 2;
    }
    std::fprintf(stderr, "[device] %s  (SM %d.%d, %zu MB)\n",
                 prop.name, prop.major, prop.minor,
                 static_cast<size_t>(prop.totalGlobalMem >> 20));
  }
  else
  {
    std::fprintf(stderr, "[device] SKIPPED (--skip-gpu)\n");
  }

  // =========================================================================
  // Main benchmark loop over datasets
  // =========================================================================
  for (int di = 0; di < kNumDatasets; ++di)
  {
    const auto &dinfo = datasets[di];
    std::fprintf(stderr, "\n");
    print_separator();
    std::fprintf(stderr, "[dataset] %s\n", dinfo.name);
    print_separator();

    // --- Load encoded artifact ---
    std::fprintf(stderr, "[load] loading artifact from %s ...\n", dinfo.artifact_root);
    exp3_real::Dataset dataset = exp3_real::load_dataset(dinfo.artifact_root);
    uint64_t n = dataset.manifest.value_count;
    uint64_t segment_rows = dataset.manifest.segment_size;
    uint64_t num_segments = dataset.manifest.segment_count;
    uint32_t max_planes = static_cast<uint32_t>(dataset.manifest.max_plane_count);
    std::fprintf(stderr, "[load] n=%llu segments=%llu max_planes=%u\n",
                 (unsigned long long)n, (unsigned long long)num_segments,
                 (unsigned int)max_planes);

    // --- Load raw FP64 values ---
    std::filesystem::path raw_file = std::filesystem::path(raw_root) / dinfo.raw_filename;
    std::fprintf(stderr, "[load] loading raw values from %s ...\n", raw_file.c_str());
    std::vector<double> raw_values = load_raw_values(raw_file, n);
    std::fprintf(stderr, "[load] loaded %zu FP64 values\n", raw_values.size());

    // --- Compute segment min/max (once) ---
    std::fprintf(stderr, "[prep] computing segment min/max ...\n");
    auto smm_start = std::chrono::steady_clock::now();
    std::vector<std::pair<double, double>> segment_min_max =
        compute_segment_min_max_from_values(raw_values, segment_rows, num_segments);
    double smm_ms = elapsed_ms(smm_start, std::chrono::steady_clock::now());
    std::fprintf(stderr, "[prep] segment min/max: %.3f ms\n", smm_ms);

    // --- Compute percentile thresholds ---
    std::fprintf(stderr, "[prep] computing percentile thresholds from raw data ...\n");
    std::vector<double> thresholds = compute_percentile_thresholds(raw_values);
    for (int si = 0; si < kNumSelectivities; ++si)
    {
      std::fprintf(stderr, "  selectivity=%.0f%%  threshold=%.17g\n",
                   kSelectivities[si] * 100.0, thresholds[si]);
    }

    // --- Allocate GPU metadata buffers (once) ---
    // We'll use these for H2D timing. Pre-allocate to exclude alloc from timing.
    size_t threshold_flat_bytes = static_cast<size_t>(num_segments * max_planes);
    size_t segment_mode_bytes = static_cast<size_t>(num_segments);
    size_t active_plane_count_bytes = static_cast<size_t>(num_segments * sizeof(uint32_t));
    size_t total_bits_bytes = static_cast<size_t>(num_segments * sizeof(uint32_t));
    size_t threshold_combined_bytes = static_cast<size_t>(num_segments * sizeof(uint64_t));

    uint8_t  *d_threshold_flat = nullptr;
    uint8_t  *d_segment_mode = nullptr;
    uint32_t *d_active_plane_count = nullptr;
    uint32_t *d_total_bits = nullptr;
    uint64_t *d_threshold_combined = nullptr;

    if (!skip_gpu)
    {
      cuda_check(cudaMalloc(&d_threshold_flat, threshold_flat_bytes), "cudaMalloc(threshold_flat)");
      cuda_check(cudaMalloc(&d_segment_mode, segment_mode_bytes), "cudaMalloc(segment_mode)");
      cuda_check(cudaMalloc(&d_active_plane_count, active_plane_count_bytes), "cudaMalloc(active_plane_count)");
      cuda_check(cudaMalloc(&d_total_bits, total_bits_bytes), "cudaMalloc(total_bits)");
      cuda_check(cudaMalloc(&d_threshold_combined, threshold_combined_bytes), "cudaMalloc(threshold_combined)");
      // Warmup H2D to establish CUDA context
      cuda_check(cudaMemcpy(d_threshold_flat, dataset.planes[0].data(), 64, cudaMemcpyHostToDevice), "warmup h2d");
      cuda_check(cudaDeviceSynchronize(), "warmup sync");
    }

    // =====================================================================
    // Per-selectivity benchmark
    // =====================================================================
    for (int si = 0; si < kNumSelectivities; ++si)
    {
      double target_sel = kSelectivities[si];
      double threshold = thresholds[si];
      std::fprintf(stderr, "\n");
      print_separator();
      std::fprintf(stderr, "[bench] selectivity=%.0f%% threshold=%.17g\n",
                   target_sel * 100.0, threshold);
      print_separator();

      // --- Warmup: multi-pass path ---
      std::fprintf(stderr, "[warmup] multi-pass (%d iterations) ...\n", kWarmup);
      for (int w = 0; w < kWarmup; ++w)
      {
        PreparedQuery p = prepare_resident_query(dataset, segment_min_max, threshold, -1);
        (void)p;
      }

      // --- Timed iterations: multi-pass path ---
      std::fprintf(stderr, "[timed] multi-pass (%d iterations) ...\n", kIters);
      std::vector<double> prep_total_ms(kIters);
      std::vector<double> classify_ms(kIters);
      std::vector<double> encode_ms(kIters);
      std::vector<double> subext_ms(kIters);
      std::vector<double> pack_ms(kIters);
      std::vector<double> fused_ms(kIters);

      for (int ti = 0; ti < kIters; ++ti)
      {
        auto prep_start = std::chrono::steady_clock::now();
        ThresholdPrepTiming tpt;
        std::vector<SegmentThresholdInfo> ti_vec =
            compute_threshold_bytes(dataset, segment_min_max, threshold, &tpt);
        (void)ti_vec;

        // Metadata pack
        auto pack_start = std::chrono::steady_clock::now();
        // Simulate the pack step: create temporary metadata arrays
        std::vector<uint8_t> segmode(num_segments, kSegmentModeMixed);
        std::vector<uint32_t> act_plane(num_segments, 0);
        std::vector<uint8_t> thresh_flat(threshold_flat_bytes, 0);
        std::vector<uint64_t> thresh_combined(num_segments, 0);
        for (size_t s = 0; s < num_segments; ++s)
        {
          const auto &ti = ti_vec[s];
          if (ti.all_qualified) segmode[s] = kSegmentModeAllQualified;
          else if (ti.all_disqualified) segmode[s] = kSegmentModeAllDisqualified;
          act_plane[s] = std::min<uint32_t>(
              static_cast<uint32_t>(dataset.manifest.max_plane_count),
              dataset.segments[s].active_plane_count);
          std::memcpy(thresh_flat.data() + s * max_planes,
                      ti.threshold_bytes.data(), max_planes);
          thresh_combined[s] = ti.threshold_combined;
        }
        double pack_end_ms = elapsed_ms(pack_start, std::chrono::steady_clock::now());
        double total = elapsed_ms(prep_start, std::chrono::steady_clock::now());

        prep_total_ms[ti] = total;
        classify_ms[ti] = tpt.classify_ms;
        encode_ms[ti] = tpt.encode_ms;
        subext_ms[ti] = tpt.subtract_extract_ms;
        pack_ms[ti] = pack_end_ms;
      }

      // --- Timed iterations: fused single-pass ---
      std::fprintf(stderr, "[timed] fused single-pass (%d iterations) ...\n", kIters);
      for (int ti = 0; ti < kIters; ++ti)
      {
        PreparedQuery p = prepare_resident_query_fused(dataset, segment_min_max, threshold, -1);
        fused_ms[ti] = p.total_prep_ms;
      }

      // --- Timed iterations: H2D copy ---
      std::vector<double> h2d_ms(kIters, 0.0);
      if (!skip_gpu)
      {
        // Prepare one query to get metadata for H2D
        PreparedQuery q = prepare_resident_query(dataset, segment_min_max, threshold, -1);

        std::fprintf(stderr, "[timed] H2D copy (%d iterations) ...\n", kIters);
        for (int ti = 0; ti < kIters; ++ti)
        {
          auto h2d_start = std::chrono::steady_clock::now();
          cuda_check(cudaMemcpy(d_threshold_flat, q.h_threshold_flat.data(),
                                threshold_flat_bytes, cudaMemcpyHostToDevice),
                     "cudaMemcpy(threshold_flat)");
          cuda_check(cudaMemcpy(d_segment_mode, q.segment_mode.data(),
                                segment_mode_bytes, cudaMemcpyHostToDevice),
                     "cudaMemcpy(segment_mode)");
          cuda_check(cudaMemcpy(d_active_plane_count, q.active_plane_count.data(),
                                active_plane_count_bytes, cudaMemcpyHostToDevice),
                     "cudaMemcpy(active_plane_count)");
          cuda_check(cudaMemcpy(d_total_bits, q.total_bits_by_segment.data(),
                                total_bits_bytes, cudaMemcpyHostToDevice),
                     "cudaMemcpy(total_bits)");
          cuda_check(cudaMemcpy(d_threshold_combined, q.h_threshold_combined.data(),
                                threshold_combined_bytes, cudaMemcpyHostToDevice),
                     "cudaMemcpy(threshold_combined)");
          cuda_check(cudaDeviceSynchronize(), "h2d sync");
          h2d_ms[ti] = elapsed_ms(h2d_start, std::chrono::steady_clock::now());
        }
      }

      // --- Compute stats ---
      Stats prep_s = compute_stats(prep_total_ms);
      Stats classify_s = compute_stats(classify_ms);
      Stats encode_s = compute_stats(encode_ms);
      Stats subext_s = compute_stats(subext_ms);
      Stats pack_s = compute_stats(pack_ms);
      Stats h2d_s = compute_stats(h2d_ms);
      Stats fused_s = compute_stats(fused_ms);

      std::fprintf(stderr, "\n[results] selectivity=%.0f%%\n", target_sel * 100.0);
      std::fprintf(stderr, "  prep_total:       p10=%.4f median=%.4f p90=%.4f mean=%.4f std=%.4f ms\n",
                   prep_s.p10, prep_s.median, prep_s.p90, prep_s.mean, prep_s.stddev);
      std::fprintf(stderr, "  classify:         p10=%.4f median=%.4f p90=%.4f mean=%.4f std=%.4f ms\n",
                   classify_s.p10, classify_s.median, classify_s.p90, classify_s.mean, classify_s.stddev);
      std::fprintf(stderr, "  encode:           p10=%.4f median=%.4f p90=%.4f mean=%.4f std=%.4f ms\n",
                   encode_s.p10, encode_s.median, encode_s.p90, encode_s.mean, encode_s.stddev);
      std::fprintf(stderr, "  subtract_extract: p10=%.4f median=%.4f p90=%.4f mean=%.4f std=%.4f ms\n",
                   subext_s.p10, subext_s.median, subext_s.p90, subext_s.mean, subext_s.stddev);
      std::fprintf(stderr, "  pack:             p10=%.4f median=%.4f p90=%.4f mean=%.4f std=%.4f ms\n",
                   pack_s.p10, pack_s.median, pack_s.p90, pack_s.mean, pack_s.stddev);
      std::fprintf(stderr, "  h2d:              p10=%.4f median=%.4f p90=%.4f mean=%.4f std=%.4f ms\n",
                   h2d_s.p10, h2d_s.median, h2d_s.p90, h2d_s.mean, h2d_s.stddev);
      std::fprintf(stderr, "  fused_single:     p10=%.4f median=%.4f p90=%.4f mean=%.4f std=%.4f ms\n",
                   fused_s.p10, fused_s.median, fused_s.p90, fused_s.mean, fused_s.stddev);

      double frac_classify = classify_s.mean / prep_s.mean;
      double frac_encode = encode_s.mean / prep_s.mean;
      double frac_pack = pack_s.mean / prep_s.mean;
      std::fprintf(stderr, "[fractions] classify=%.1f%%  encode=%.1f%%  subext=%.1f%%  pack=%.1f%%  h2d=%.1f%%\n",
                   frac_classify * 100.0, frac_encode * 100.0,
                   subext_s.mean / prep_s.mean * 100.0,
                   frac_pack * 100.0,
                   h2d_s.mean / prep_s.mean * 100.0);
      std::fprintf(stderr, "  ns/segment total=%.0f  classify=%.0f\n",
                   prep_s.mean * 1e6 / static_cast<double>(num_segments),
                   classify_s.mean * 1e6 / static_cast<double>(num_segments));
      double speedup = prep_s.median / fused_s.median;
      std::fprintf(stderr, "[fused] speedup vs multi-pass: %.2fx\n", speedup);

      // --- Write CSVs ---
      write_breakdown_csv(breakdown_csv, dinfo.name, threshold, target_sel,
                          num_segments, prep_total_ms, classify_ms, encode_ms,
                          subext_ms, pack_ms, h2d_ms, fused_ms);
      write_summary_csv(summary_csv, dinfo.name, threshold, target_sel,
                        num_segments, prep_s, classify_s, encode_s, subext_s,
                        pack_s, h2d_s, fused_s);
      std::fprintf(stderr, "[csv] wrote row to %s and %s\n",
                   breakdown_csv.c_str(), summary_csv.c_str());
    } // per-selectivity

    // --- Free GPU buffers ---
    if (!skip_gpu)
    {
      cuda_check(cudaFree(d_threshold_flat), "cudaFree(threshold_flat)");
      cuda_check(cudaFree(d_segment_mode), "cudaFree(segment_mode)");
      cuda_check(cudaFree(d_active_plane_count), "cudaFree(active_plane_count)");
      cuda_check(cudaFree(d_total_bits), "cudaFree(total_bits)");
      cuda_check(cudaFree(d_threshold_combined), "cudaFree(threshold_combined)");
    }
  } // per-dataset

  std::fprintf(stderr, "\n[DONE] Results written to:\n");
  std::fprintf(stderr, "  %s\n", breakdown_csv.c_str());
  std::fprintf(stderr, "  %s\n", summary_csv.c_str());
  return 0;
}
