#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <cinttypes>

#include "exp3_real_data_layout.hpp"

// ---------------------------------------------------------------------------
// Structs & Helpers from bench_filter_aggregate.cu
// ---------------------------------------------------------------------------

struct SegmentThresholdInfo
{
  std::vector<uint8_t> threshold_bytes;
  bool all_qualified = false;
  bool all_disqualified = false;
  bool threshold_exact = true;
  uint64_t threshold_combined = 0;
};

struct Options
{
  std::string dataset_path;
  std::string raw_path;
  double threshold = 0.0;
  std::string selectivity = "unknown";
  int iters = 200;
  int warmup = 10;
  int max_filter_planes = -1;
  std::string csv_path = "";
};

void die(const std::string &msg)
{
  std::cerr << "Error: " << msg << std::endl;
  std::exit(1);
}

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
        long double scale = std::ldexp(1.0L, static_cast<int>(frac));
        uint64_t t_code = static_cast<uint64_t>(
            std::floor(static_cast<long double>(threshold_fp64) * scale));
        t_code_le.clear();
        for (size_t i = 0; i < sizeof(uint64_t); ++i)
          t_code_le.push_back(static_cast<uint8_t>((t_code >> (i * 8)) & 0xFF));
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

// ---------------------------------------------------------------------------
// CPU Progressive Scan Query Logic
// ---------------------------------------------------------------------------

struct QueryResult
{
  uint64_t qualified_count = 0;
  double qualified_sum = 0.0;
};

QueryResult run_query(
    const exp3_real::Dataset &dataset,
    const std::vector<SegmentThresholdInfo> &threshold_info,
    const std::vector<uint32_t> &active_plane_count,
    const std::vector<double> &h_segment_base,
    const std::vector<double> &h_subcolumn_basis,
    const std::vector<double> &h_segment_sums,
    uint64_t max_plane_count)
{
  QueryResult res{};

  for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
  {
    const auto &segment = dataset.segments[seg];
    const auto &info = threshold_info[seg];
    uint32_t max_rounds = active_plane_count[seg];

    if (info.all_qualified)
    {
      res.qualified_count += segment.row_count;
      res.qualified_sum += h_segment_sums[seg];
    }
    else if (info.all_disqualified)
    {
      // None qualified
    }
    else
    {
      uint64_t seg_start = segment.row_offset;
      uint64_t seg_end = seg_start + segment.row_count;
      const double *seg_basis = h_subcolumn_basis.data() + seg * max_plane_count;
      double seg_base = h_segment_base[seg];

      // Extract pointers to byte planes to avoid std::vector index overhead
      std::vector<const uint8_t *> plane_ptrs(max_rounds);
      for (uint32_t round = 0; round < max_rounds; ++round)
      {
        plane_ptrs[round] = dataset.planes[static_cast<size_t>(round)].data();
      }

      for (uint64_t row = seg_start; row < seg_end; ++row)
      {
        bool qualified = false;
        bool active = true;
        for (uint32_t round = 0; round < max_rounds && active; ++round)
        {
          uint8_t row_byte = plane_ptrs[round][row];
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

        if (qualified)
        {
          double val = seg_base;
          for (uint32_t round = 0; round < max_rounds; ++round)
          {
            val += seg_basis[round] * static_cast<double>(plane_ptrs[round][row]);
          }
          ++res.qualified_count;
          res.qualified_sum += val;
        }
      }
    }
  }

  return res;
}

// ---------------------------------------------------------------------------
// Arg parsing & Main
// ---------------------------------------------------------------------------

void print_usage(const char *argv0)
{
  std::fprintf(stderr,
               "Usage: %s [options]\n"
               "\n"
               "BUFF-style CPU Progressive Scan Proxy.\n"
               "\n"
               "Options:\n"
               "  --dataset PATH        Path to encoded dataset directory\n"
               "  --raw PATH            Path to raw float64 data file\n"
               "  --threshold FP64      Predicate threshold (default: 0.0)\n"
               "  --selectivity STR     Selectivity label: s10, s50, s90\n"
               "  --iters N             Number of iterations (default: 20)\n"
               "  --warmup N            Warmup iterations (default: 5)\n"
               "  --max-filter-planes N Read planes 0..N-1 (default: max available)\n"
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
    else if (a == "--selectivity")
    {
      opt.selectivity = std::string(need_value(a));
    }
    else if (a == "--iters")
    {
      opt.iters = std::stoi(std::string(need_value(a)));
    }
    else if (a == "--warmup")
    {
      opt.warmup = std::stoi(std::string(need_value(a)));
    }
    else if (a == "--max-filter-planes")
    {
      opt.max_filter_planes = std::stoi(std::string(need_value(a)));
    }
    else if (a == "--csv")
    {
      opt.csv_path = std::string(need_value(a));
    }
    else
    {
      die("unknown option: " + std::string(a));
    }
  }

  if (opt.dataset_path.empty())
    die("--dataset is required");

  return opt;
}

int main(int argc, char **argv)
{
  Options opt = parse_args(argc, argv);

  // Load encoded dataset
  std::cout << "[cpu_proxy] Loading dataset: " << opt.dataset_path << std::endl;
  exp3_real::Dataset dataset = exp3_real::load_dataset(opt.dataset_path);
  uint64_t n = dataset.manifest.value_count;
  uint64_t segment_rows = dataset.manifest.segment_size;
  uint64_t max_plane_count = dataset.manifest.max_plane_count;

  std::cout << "[cpu_proxy] Dataset loaded. n=" << n
            << ", segments=" << dataset.manifest.segment_count
            << ", max_planes=" << max_plane_count << std::endl;

  // Set up basis & bases
  std::vector<double> h_segment_base(dataset.segments.size());
  std::vector<double> h_subcolumn_basis(dataset.segments.size() * max_plane_count);
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

  // Precompute segment sums to avoid CPU row loop in all_qualified
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

  // Segment min/max
  std::vector<std::pair<double, double>> segment_min_max;
  if (!opt.raw_path.empty())
  {
    std::cout << "[cpu_proxy] Loading raw min/max from: " << opt.raw_path << std::endl;
    segment_min_max = compute_segment_min_max(opt.raw_path, segment_rows, dataset.segments.size());
  }
  else
  {
    std::cout << "[cpu_proxy] Using fallback min/max from bases" << std::endl;
    segment_min_max.resize(dataset.segments.size(),
                           {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max()});
  }

  // Compute threshold bytes
  std::vector<SegmentThresholdInfo> threshold_info =
      compute_threshold_bytes(dataset, segment_min_max, opt.threshold);

  // Active planes cap (k)
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
  // Warmup
  // -----------------------------------------------------------------------
  std::cout << "[cpu_proxy] Warming up (" << opt.warmup << " iterations)..." << std::endl;
  QueryResult last_res{};
  for (int i = 0; i < opt.warmup; ++i)
  {
    last_res = run_query(dataset, threshold_info, h_active_plane_count,
                         h_segment_base, h_subcolumn_basis, h_segment_sums,
                         max_plane_count);
  }

  // -----------------------------------------------------------------------
  // Timed Loop
  // -----------------------------------------------------------------------
  std::cout << "[cpu_proxy] Benchmarking (" << opt.iters << " iterations)..." << std::endl;
  std::vector<double> run_times_ms;
  run_times_ms.reserve(opt.iters);

  for (int i = 0; i < opt.iters; ++i)
  {
    auto start = std::chrono::high_resolution_clock::now();
    QueryResult res = run_query(dataset, threshold_info, h_active_plane_count,
                                h_segment_base, h_subcolumn_basis, h_segment_sums,
                                max_plane_count);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    run_times_ms.push_back(ms);

    if (res.qualified_count != last_res.qualified_count ||
        std::abs(res.qualified_sum - last_res.qualified_sum) > 1e-5)
    {
      die("non-deterministic query results across runs");
    }
  }

  // Compute stats
  std::sort(run_times_ms.begin(), run_times_ms.end());
  double median_ms = run_times_ms[run_times_ms.size() / 2];
  double min_ms = run_times_ms.front();
  double max_ms = run_times_ms.back();
  double sum_time_ms = std::accumulate(run_times_ms.begin(), run_times_ms.end(), 0.0);
  double mean_ms = sum_time_ms / opt.iters;

  std::printf("[cpu_proxy] Result: count=%" PRIu64 " sum=%.6f\n",
              last_res.qualified_count, last_res.qualified_sum);
  std::printf("[cpu_proxy] Timing (ms): min=%.3f, median=%.3f, mean=%.3f, max=%.3f\n",
              min_ms, median_ms, mean_ms, max_ms);

  // Write CSV
  if (!opt.csv_path.empty())
  {
    bool write_header = !std::filesystem::exists(opt.csv_path);
    std::ofstream out(opt.csv_path, std::ios::app);
    if (!out)
    {
      std::cerr << "Warning: failed to open output CSV: " << opt.csv_path << std::endl;
    }
    else
    {
      if (write_header)
      {
        out << "dataset,selectivity,threshold,k,cpu_warm_ms,count,sum\n";
      }
      out << dataset.manifest.dataset << ","
          << opt.selectivity << ","
          << opt.threshold << ","
          << (opt.max_filter_planes >= 0 ? opt.max_filter_planes : (int)max_plane_count) << ","
          << median_ms << ","
          << last_res.qualified_count << ","
          << last_res.qualified_sum << "\n";
    }
  }

  return 0;
}
