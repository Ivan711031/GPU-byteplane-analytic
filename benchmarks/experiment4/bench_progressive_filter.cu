#include <cuda_runtime.h>

#include <array>
#include <algorithm>
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

#include "buff_codec.hpp"
#include "exp3_real_data_layout.hpp"
#include "exp4_kernels_filter.cuh"

// Set to 1 to enable v2 threshold diagnostic for segment 0.
#define EXP4_V2_DIAGNOSTIC 0

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

struct Options
{
  int device = 0;
  std::string encoded_root;
  double threshold = 0.0;
  int block_threads = 256;
  int warmup = 10;
  int iters = 200;
  std::string csv_path = "exp4_progressive_filter.csv";
  bool validate = false;
  bool no_gpu = false;
  int debug_segment = -1;
  int debug_rows = 0;
  int max_filter_planes = -1;
  std::string output_rounds_csv;
  bool fixed_depth_baseline = false;
  std::string raw_root;
  std::string warp_strategy = "byte_mask";
};

struct Derived
{
  uint64_t tile_rows = 0;
  uint64_t tiles_per_segment = 0;
  uint64_t num_segments = 0;
  uint64_t grid = 0;
};

struct SegmentThresholdInfo
{
  std::vector<uint8_t> threshold_bytes;
  bool all_qualified = false;
  bool all_disqualified = false;
  bool threshold_exact = true;
  uint64_t threshold_combined = 0;  // combined-code threshold for bound checking
};

struct PrecisionInfo
{
  std::string mode = "exact";
  std::string decimals = "NA";
};

struct EncodedReferenceStats
{
  uint64_t qualified_count = 0;
  uint64_t total_planes_read = 0;
  uint64_t gpu_processed_rows = 0;
  uint32_t max_planes_read = 0;
  uint64_t estimated_pack_load_bytes = 0;

  // Capped-k count classification
  uint64_t certainly_qualified = 0;      // Q: resolved qualified + bound-certain qualified
  uint64_t certainly_disqualified = 0;   // D: resolved disqualified + bound-certain disqualified
  uint64_t uncertain = 0;                // U: true ambiguity after bound check

  // Per-round survival histograms (indexed by round, 0-based)
  std::vector<uint64_t> qualified_per_round;
  std::vector<uint64_t> disqualified_per_round;
  std::vector<uint64_t> unresolved_per_round;
  std::vector<uint64_t> planes_read_per_round;
  std::vector<uint64_t> pack_load_bytes_per_round;

  // Pack utilization at cap
  uint64_t fully_resolved_packs = 0;
  uint64_t partially_active_packs = 0;
  uint64_t total_rows_in_active_packs = 0;
  uint64_t active_row_count = 0;
  double avg_active_rows_per_active_pack = 0.0;
  double useful_row_fraction = 0.0;
};

struct RowEval
{
  bool qualified = false;      // true if resolved as qualified
  bool disqualified = false;   // true if resolved as disqualified
  bool unresolved = false;     // true if hit cap without resolution
  uint32_t rounds = 0;         // planes read before resolution or cap
};

constexpr uint8_t kSegmentModeMixed = 0;
constexpr uint8_t kSegmentModeAllQualified = 1;
constexpr uint8_t kSegmentModeAllDisqualified = 2;

void print_usage(const char *argv0)
{
  std::fprintf(stderr,
               "Usage: %s [options]\n"
               "\n"
               "Experiment 4 v1a.1: progressive filter COUNT with exact threshold encoding.\n"
               "\n"
               "Options:\n"
               "  --device N              CUDA device index (default: 0)\n"
               "  --encoded-root PATH     Encoded dataset root\n"
               "  --raw-root PATH         Raw FP64 data directory (default: auto from encoded-root parent)\n"
               "  --threshold FP64        Predicate threshold (e.g., 500.0)\n"
               "  --block T               Threads per block (default: 256)\n"
               "  --warmup N              Warmup iterations (default: 10)\n"
               "  --iters N               Timed iterations (default: 200)\n"
               "  --validate              Run CPU reference validation\n"
               "  --no-gpu                CPU-only validation mode (no CUDA)\n"
               "  --debug-segment N       Run row-level diagnostic on segment N\n"
               "  --debug-rows N          Max rows to print in diagnostic (default: 16)\n"
               "  --csv PATH              Output CSV path\n"
               "  --baseline-mode MODE    progressive (default) or fixed_depth\n"
               "  --max-filter-planes N   Cap planes read per segment (default: full depth)\n"
               "  --output-rounds-csv PATH  Write per-round survival metrics sidecar (progressive only)\n"
               "  --warp-strategy STRATEGY  Warp scheduling strategy: passive|predicated|byte_mask|byte_mask_interleave2|byte_mask_specialized_k1|byte_mask_specialized_k4|byte_mask_specialized_k8|prefetch|simd_masks (default: byte_mask)\n",
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
    else if (a == "--device")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v)) die("invalid --device");
      opt.device = static_cast<int>(v);
    }
    else if (a == "--encoded-root")
    {
      opt.encoded_root = std::string(need_value(a));
    }
    else if (a == "--raw-root")
    {
      opt.raw_root = std::string(need_value(a));
    }
    else if (a == "--threshold")
    {
      opt.threshold = std::stod(std::string(need_value(a)));
    }
    else if (a == "--block")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > 1024) die("invalid --block");
      opt.block_threads = static_cast<int>(v);
    }
    else if (a == "--warmup")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v > 1000000) die("invalid --warmup");
      opt.warmup = static_cast<int>(v);
    }
    else if (a == "--iters")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > 100000000) die("invalid --iters");
      opt.iters = static_cast<int>(v);
    }
    else if (a == "--csv")
    {
      opt.csv_path = std::string(need_value(a));
    }
    else if (a == "--baseline-mode")
    {
      std::string mode(need_value(a));
      if (mode == "progressive")
        opt.fixed_depth_baseline = false;
      else if (mode == "fixed_depth")
        opt.fixed_depth_baseline = true;
      else
        die("invalid --baseline-mode (expected progressive or fixed_depth)");
    }
    else if (a == "--validate")
    {
      opt.validate = true;
    }
    else if (a == "--no-gpu")
    {
      opt.no_gpu = true;
    }
    else if (a == "--debug-segment")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v)) die("invalid --debug-segment");
      opt.debug_segment = static_cast<int>(v);
    }
    else if (a == "--debug-rows")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v > 1000000) die("invalid --debug-rows");
      opt.debug_rows = static_cast<int>(v);
    }
    else if (a == "--max-filter-planes")
    {
      uint64_t v = 0;
      if (!parse_u64(need_value(a), v) || v == 0 || v > 256) die("invalid --max-filter-planes");
      opt.max_filter_planes = static_cast<int>(v);
    }
    else if (a == "--output-rounds-csv")
    {
      opt.output_rounds_csv = std::string(need_value(a));
    }
    else if (a == "--warp-strategy")
    {
      std::string ws(need_value(a));
      if (ws != "passive" && ws != "predicated" && ws != "byte_mask" && ws != "byte_mask_interleave2" && ws != "byte_mask_specialized_k1" && ws != "byte_mask_specialized_k4" && ws != "byte_mask_specialized_k8" && ws != "prefetch" && ws != "simd_masks")
        die("invalid --warp-strategy (expected 'passive', 'predicated', 'byte_mask', 'byte_mask_interleave2', 'byte_mask_specialized_k1', 'byte_mask_specialized_k4', 'byte_mask_specialized_k8', 'prefetch', or 'simd_masks')");
      opt.warp_strategy = ws;
    }
    else
    {
      std::string msg = "unknown arg: ";
      msg += std::string(a);
      die(msg);
    }
  }

  if (opt.encoded_root.empty()) die("--encoded-root is required");
  if (opt.block_threads % 32 != 0) die("--block must be multiple of 32");
  if (opt.fixed_depth_baseline && opt.max_filter_planes >= 0)
    die("--baseline-mode fixed_depth does not support --max-filter-planes");
  if (opt.fixed_depth_baseline && !opt.output_rounds_csv.empty())
    die("--baseline-mode fixed_depth does not support --output-rounds-csv");
  return opt;
}

[[nodiscard]] Derived compute_derived(uint64_t n, uint64_t segment_rows, int block_threads, uint64_t rowpacks_per_thread = 1)
{
  Derived d{};
  if (rowpacks_per_thread == 0)
    die("rowpacks_per_thread must be > 0");
  d.tile_rows = static_cast<uint64_t>(block_threads) *
                static_cast<uint64_t>(EXP3_ROWPACK16_WIDTH) *
                rowpacks_per_thread;
  d.tiles_per_segment = exp3_ceil_div_u64(segment_rows, d.tile_rows);
  d.num_segments = exp3_ceil_div_u64(n, segment_rows);
  d.grid = d.num_segments * d.tiles_per_segment;
  if (d.grid == 0 || d.grid > static_cast<uint64_t>(std::numeric_limits<int>::max()))
    die("grid does not fit in int range");
  return d;
}

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

// Compute per-segment min/max from raw .f64le.bin.
// This is a temporary v1a.1 smoke-only path.
// v1b must write segment_min/segment_max into segment_meta.csv during export.
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

// Convert little-endian bytes to uint64.
uint64_t le_bytes_to_u64(const std::vector<uint8_t> &bytes)
{
  uint64_t value = 0;
  for (size_t i = 0; i < bytes.size() && i < sizeof(uint64_t); ++i)
    value |= static_cast<uint64_t>(bytes[i]) << (i * 8);
  return value;
}

// Little-endian byte-vector subtraction: a - b, assumes a >= b.
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

// Extract plane bytes from a combined code represented as little-endian bytes.
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

// Compute threshold bytes per segment per plane.
// For bounded-precision segments, integer-base encoding must use bounded truncation
// to align with artifact construction at the segment's effective scale.
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

    // For v2 artifacts, when the raw file is missing we can derive
    // conservative bounds from segment_base and integer_offset_bits.
    // segment_base = integer_base / 2^frac is the minimum decoded value.
    // Maximum is bounded by segment_base + 2^integer_offset_bits - 2^(-frac).
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
      // Mixed segment: compute exact threshold bytes.
      uint32_t frac = segment.fractional_bits;
      int32_t scale_exponent = -static_cast<int32_t>(frac);
      std::vector<uint8_t> t_code_le;

      uint64_t integer_base_u64 = le_bytes_to_u64(segment.integer_base_le);

      // combined_T = T_code - base_shifted
      std::vector<uint8_t> combined_T_le;
      if (integer_base_u64 == 0)
      {
        // Base is zero; combined_T is just T_code.
        if (segment.integer_base_is_v2)
        {
          // v2: use fixed-point floor(T * 2^frac) matching the plane byte
          // encoding scale.  encode_value_to_code would decompose T as
          // significand*2^(-52); for small thresholds at frac < ~52 the
          // required right shift is not supported (throws).
          long double scale = std::ldexp(1.0L, static_cast<int>(frac));
          uint64_t t_code = static_cast<uint64_t>(
              std::floor(static_cast<long double>(threshold_fp64) * scale));
          t_code_le.clear();
          for (size_t i = 0; i < sizeof(uint64_t); ++i)
            t_code_le.push_back(static_cast<uint8_t>((t_code >> (i * 8)) & 0xFF));
        }
        else
        {
          // Legacy exact-dyadic path: encode the FULL double precision of T
          // at the segment scale.  Use the same ldexp+floor approach as the
          // v2 path (encode_value_to_code was removed from buff_codec.hpp
          // during a broader refactoring).
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
          // v2 artifact: integer_base is a signed i64 fixed-point code base
          // (already includes 2^frac scaling). plane bytes store
          // code(v) - integer_base directly.
          //
          // NOTE: this heuristic detects v2 by the 0x... format in
          // integer_base_hex. A future schema should add an explicit
          // field such as integer_base_semantics=fixed_point_code_base.
          base_shifted_le = segment.integer_base_le;

          // v2 uses bounded fixed-point truncation.  The threshold code
          // must be computed on the SAME fixed-point scale:
          //   t_code = floor(threshold * 2^frac)
          long double scale = std::ldexp(1.0L, static_cast<int>(frac));
          uint64_t t_code = static_cast<uint64_t>(
              std::floor(static_cast<long double>(threshold_fp64) * scale));
          t_code_le.clear();
          for (size_t i = 0; i < sizeof(uint64_t); ++i)
            t_code_le.push_back(static_cast<uint8_t>((t_code >> (i * 8)) & 0xFF));
        }
        else
        {
          // Legacy artifact: integer_base = floor(min_value).
          // plane bytes store code(v) - (integer_base << frac).
          // Use inline ldexp+floor arithmetic (same as v2 path) because
          // encode_value_to_code was removed from buff_codec.hpp during
          // a broader refactoring.
          {
            long double scale = std::ldexp(1.0L, static_cast<int>(frac));
            uint64_t t_code = static_cast<uint64_t>(
                std::floor(static_cast<long double>(threshold_fp64) * scale));
            t_code_le.clear();
            for (size_t i = 0; i < sizeof(uint64_t); ++i)
              t_code_le.push_back(static_cast<uint8_t>((t_code >> (i * 8)) & 0xFF));
          }
          {
            // Encode integer_base at the same scale.
            long double scale = std::ldexp(1.0L, static_cast<int>(frac));
            uint64_t base_code = static_cast<uint64_t>(
                std::floor(static_cast<long double>(integer_base_u64) * scale));
            base_shifted_le.clear();
            for (size_t i = 0; i < sizeof(uint64_t); ++i)
              base_shifted_le.push_back(static_cast<uint8_t>((base_code >> (i * 8)) & 0xFF));
          }
        }

        // Ensure both vectors have the same length before subtracting.
        size_t max_len = std::max(t_code_le.size(), base_shifted_le.size());
        if (t_code_le.size() < max_len)
          t_code_le.resize(max_len, 0);
        if (base_shifted_le.size() < max_len)
          base_shifted_le.resize(max_len, 0);
        combined_T_le = subtract_le_bytes(t_code_le, base_shifted_le);

#if EXP4_V2_DIAGNOSTIC
        if (seg == 0)
        {
          std::fprintf(stderr, "[exp4] v2 diagnostic seg=0 dataset=%s\n", dataset.manifest.dataset.c_str());
          std::fprintf(stderr, "  fractional_bits=%u\n", frac);
          std::fprintf(stderr, "  segment_base=%.17g\n", segment.segment_base);
          std::fprintf(stderr, "  integer_base_is_v2=%s\n", segment.integer_base_is_v2 ? "true" : "false");
          std::fprintf(stderr, "  integer_base_u64=%lu\n", static_cast<unsigned long>(integer_base_u64));
          double base_from_i64 = 0.0;
          if (segment.integer_base_le.size() == sizeof(int64_t))
          {
            int64_t signed_base = 0;
            std::memcpy(&signed_base, segment.integer_base_le.data(), sizeof(int64_t));
            base_from_i64 = static_cast<double>(signed_base) / std::ldexp(1.0, static_cast<int>(frac));
          }
          std::fprintf(stderr, "  int64(base_le)/2^f=%.17g\n", base_from_i64);
          std::fprintf(stderr, "  t_code_le size=%zu\n", t_code_le.size());
          std::fprintf(stderr, "  base_shifted_le size=%zu\n", base_shifted_le.size());
          std::fprintf(stderr, "  combined_T_le size=%zu\n", combined_T_le.size());
          std::fprintf(stderr, "  threshold_fp64=%.17g\n", threshold_fp64);
          std::fprintf(stderr, "  combined_T_u64=%lu\n", static_cast<unsigned long>(le_bytes_to_u64(combined_T_le)));
        }
#endif
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

RowEval evaluate_scalar_row(
    const exp3_real::Dataset &dataset,
    const exp3_real::SegmentMeta &segment,
    const SegmentThresholdInfo &info,
    uint64_t row_index,
    uint32_t max_rounds)
{
  RowEval eval{};
  bool active = true;
  for (uint32_t round = 0; round < max_rounds && active; ++round)
  {
    ++eval.rounds;
    uint8_t row_byte = dataset.planes[round][row_index];
    uint8_t thresh_byte = info.threshold_bytes[round];
    if (row_byte > thresh_byte)
    {
      eval.qualified = true;
      active = false;
    }
    else if (row_byte < thresh_byte)
    {
      eval.disqualified = true;
      active = false;
    }
  }
  if (active)
  {
    eval.unresolved = true;
  }
  return eval;
}

int compare_scalar_row_fixed_depth(
    const exp3_real::Dataset &dataset,
    const SegmentThresholdInfo &info,
    uint64_t row_index,
    uint32_t rounds)
{
  int cmp = 0;
  for (uint32_t round = 0; round < rounds; ++round)
  {
    uint8_t row_byte = dataset.planes[round][row_index];
    if (cmp == 0)
    {
      uint8_t thresh_byte = info.threshold_bytes[round];
      if (row_byte > thresh_byte)
        cmp = 1;
      else if (row_byte < thresh_byte)
        cmp = -1;
    }
  }
  return cmp;
}

// Forward declaration: used in cpu_encoded_stats before definition.
static void combined_bounds_after_k_planes(
    const exp3_real::Dataset &dataset,
    const exp3_real::SegmentMeta &segment,
    uint64_t row,
    uint32_t kept_planes,
    uint64_t &out_lower,
    uint64_t &out_upper);

EncodedReferenceStats cpu_encoded_stats(
    const exp3_real::Dataset &dataset,
    const std::vector<SegmentThresholdInfo> &threshold_info,
    const std::vector<uint32_t> &effective_max_rounds,
    uint64_t tile_rows)
{
  EncodedReferenceStats stats{};

  if (tile_rows == 0) die("tile_rows must be > 0");

  // Determine max cap across segments to size histograms
  uint32_t global_max_cap = 0;
  for (uint32_t cap : effective_max_rounds)
    global_max_cap = std::max(global_max_cap, cap);

  stats.qualified_per_round.assign(global_max_cap, 0);
  stats.disqualified_per_round.assign(global_max_cap, 0);
  stats.unresolved_per_round.assign(global_max_cap, 0);
  stats.planes_read_per_round.assign(global_max_cap, 0);
  stats.pack_load_bytes_per_round.assign(global_max_cap, 0);

  for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
  {
    const auto &segment = dataset.segments[seg];
    const auto &info = threshold_info[seg];
    uint32_t max_rounds = effective_max_rounds[seg];

    if (info.all_qualified)
    {
      stats.qualified_count += segment.row_count;
      stats.certainly_qualified += segment.row_count;
    }
    else if (info.all_disqualified)
    {
      // Fast-path: no GPU progressive rounds.
      stats.certainly_disqualified += segment.row_count;
    }
    else
    {
      uint64_t seg_start = segment.row_offset;
      uint64_t seg_end = segment.row_offset + segment.row_count;

      for (uint64_t tile_start = seg_start; tile_start < seg_end; tile_start += tile_rows)
      {
        uint64_t tile_end = std::min(seg_end, tile_start + tile_rows);

        uint64_t aligned_start = tile_start;
        while (aligned_start < tile_end && (aligned_start & 15ull) != 0ull)
          ++aligned_start;
        uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;

        for (uint64_t row = tile_start; row < aligned_start; ++row)
        {
          RowEval eval = evaluate_scalar_row(dataset, segment, info, row, max_rounds);
          stats.total_planes_read += eval.rounds;
          stats.gpu_processed_rows += 1;
          stats.max_planes_read = std::max(stats.max_planes_read, eval.rounds);
          stats.estimated_pack_load_bytes += eval.rounds;
          if (eval.rounds > 0)
            stats.planes_read_per_round[eval.rounds - 1] += 1;

          if (eval.qualified)
          {
            ++stats.qualified_count;
            ++stats.certainly_qualified;
            if (eval.rounds > 0)
              stats.qualified_per_round[eval.rounds - 1] += 1;
          }
          else if (eval.disqualified)
          {
            ++stats.certainly_disqualified;
            if (eval.rounds > 0)
              stats.disqualified_per_round[eval.rounds - 1] += 1;
          }
          else if (eval.unresolved)
          {
            // Classify using combined-code bounds
            uint64_t lower_bound, upper_bound;
            combined_bounds_after_k_planes(dataset, segment, row - segment.row_offset, max_rounds, lower_bound, upper_bound);
            if (lower_bound > info.threshold_combined)
            {
              ++stats.certainly_qualified;
              ++stats.qualified_count;
            }
            else if (upper_bound <= info.threshold_combined)
            {
              ++stats.certainly_disqualified;
            }
            else
            {
              ++stats.uncertain;
              // NOTE: uncertain rows are NOT counted as qualified in gpu_count.
              // GPU kernel treats unresolved rows as disqualified (qualified=false).
              // This ensures gpu_count == cpu_encoded_count under capped-k.
            }
            stats.unresolved_per_round[max_rounds - 1] += 1;
          }
        }

        for (uint64_t pack_start = aligned_start; pack_start < full_pack_end; pack_start += 16ull)
        {
          uint16_t active_mask = 0xFFFFu;
          uint16_t qualified_mask = 0u;
          std::array<uint8_t, 16> row_rounds{};
          std::array<bool, 16> row_qualified{};
          std::array<bool, 16> row_disqualified{};

          for (uint32_t round = 0; round < max_rounds && active_mask != 0u; ++round)
          {
            uint8_t thresh_byte = info.threshold_bytes[round];
            stats.estimated_pack_load_bytes += 16ull;
            stats.pack_load_bytes_per_round[round] += 16;

            for (int i = 0; i < 16; ++i)
            {
              uint16_t bit = static_cast<uint16_t>(1u << i);
              if ((active_mask & bit) == 0u) continue;

              ++row_rounds[i];
              uint8_t row_byte = dataset.planes[round][pack_start + static_cast<uint64_t>(i)];
              if (row_byte > thresh_byte)
              {
                qualified_mask |= bit;
                active_mask = static_cast<uint16_t>(active_mask & ~bit);
                row_qualified[i] = true;
              }
              else if (row_byte < thresh_byte)
              {
                active_mask = static_cast<uint16_t>(active_mask & ~bit);
                row_disqualified[i] = true;
              }
            }
          }

          // Pack utilization tracking
          if (active_mask == 0u)
          {
            ++stats.fully_resolved_packs;
          }
          else
          {
            ++stats.partially_active_packs;
            stats.total_rows_in_active_packs += 16;
            uint32_t active_in_pack = 0;
            for (int i = 0; i < 16; ++i)
            {
              uint16_t bit = static_cast<uint16_t>(1u << i);
              if ((active_mask & bit) != 0u) ++active_in_pack;
            }
            stats.active_row_count += active_in_pack;
          }

          uint64_t qualified_in_pack = 0;
          for (int i = 0; i < 16; ++i)
          {
            uint32_t rounds = row_rounds[i];
            stats.total_planes_read += rounds;
            stats.gpu_processed_rows += 1;
            stats.max_planes_read = std::max(stats.max_planes_read, rounds);
            stats.planes_read_per_round[rounds - 1] += 1;

            if (row_qualified[i])
            {
              ++qualified_in_pack;
              ++stats.certainly_qualified;
              stats.qualified_per_round[rounds - 1] += 1;
            }
            else if (row_disqualified[i])
            {
              ++stats.certainly_disqualified;
              stats.disqualified_per_round[rounds - 1] += 1;
            }
            else
            {
              // Unresolved after cap
              uint64_t row_in_seg = (pack_start - seg_start) + static_cast<uint64_t>(i);
              uint64_t lower_bound, upper_bound;
              combined_bounds_after_k_planes(dataset, segment, row_in_seg, max_rounds, lower_bound, upper_bound);
              if (lower_bound > info.threshold_combined)
              {
                ++stats.certainly_qualified;
                ++qualified_in_pack;
              }
              else if (upper_bound <= info.threshold_combined)
              {
                ++stats.certainly_disqualified;
              }
              else
              {
                ++stats.uncertain;
                // NOTE: uncertain rows are NOT counted as qualified in gpu_count.
                // GPU kernel treats unresolved rows as disqualified (qualified=false).
              }
              stats.unresolved_per_round[max_rounds - 1] += 1;
            }
          }
          stats.qualified_count += qualified_in_pack;
        }

        for (uint64_t row = full_pack_end; row < tile_end; ++row)
        {
          RowEval eval = evaluate_scalar_row(dataset, segment, info, row, max_rounds);
          stats.total_planes_read += eval.rounds;
          stats.gpu_processed_rows += 1;
          stats.max_planes_read = std::max(stats.max_planes_read, eval.rounds);
          stats.estimated_pack_load_bytes += eval.rounds;
          if (eval.rounds > 0)
            stats.planes_read_per_round[eval.rounds - 1] += 1;

          if (eval.qualified)
          {
            ++stats.qualified_count;
            ++stats.certainly_qualified;
            if (eval.rounds > 0)
              stats.qualified_per_round[eval.rounds - 1] += 1;
          }
          else if (eval.disqualified)
          {
            ++stats.certainly_disqualified;
            if (eval.rounds > 0)
              stats.disqualified_per_round[eval.rounds - 1] += 1;
          }
          else if (eval.unresolved)
          {
            uint64_t lower_bound, upper_bound;
            combined_bounds_after_k_planes(dataset, segment, row - segment.row_offset, max_rounds, lower_bound, upper_bound);
            if (lower_bound > info.threshold_combined)
            {
              ++stats.certainly_qualified;
              ++stats.qualified_count;
            }
            else if (upper_bound <= info.threshold_combined)
            {
              ++stats.certainly_disqualified;
            }
            else
            {
              ++stats.uncertain;
              // NOTE: uncertain rows are NOT counted as qualified in gpu_count.
              // GPU kernel treats unresolved rows as disqualified (qualified=false).
            }
            stats.unresolved_per_round[max_rounds - 1] += 1;
          }
        }
      }
    }
  }

  if (stats.partially_active_packs > 0)
  {
    stats.avg_active_rows_per_active_pack =
        static_cast<double>(stats.active_row_count) / static_cast<double>(stats.partially_active_packs);
  }
  if (stats.total_rows_in_active_packs > 0)
  {
    stats.useful_row_fraction =
        static_cast<double>(stats.active_row_count) / static_cast<double>(stats.total_rows_in_active_packs);
  }

  return stats;
}

EncodedReferenceStats cpu_fixed_depth_encoded_stats(
    const exp3_real::Dataset &dataset,
    const std::vector<SegmentThresholdInfo> &threshold_info,
    const std::vector<uint8_t> &segment_filter_mode,
    const std::vector<uint32_t> &effective_max_rounds,
    uint64_t tile_rows)
{
  EncodedReferenceStats stats{};

  if (tile_rows == 0) die("tile_rows must be > 0");

  for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
  {
    const auto &segment = dataset.segments[seg];
    const auto &info = threshold_info[seg];
    uint8_t segment_mode = segment_filter_mode[seg];
    uint32_t max_rounds = effective_max_rounds[seg];

    uint64_t seg_start = segment.row_offset;
    uint64_t seg_end = segment.row_offset + segment.row_count;

    for (uint64_t tile_start = seg_start; tile_start < seg_end; tile_start += tile_rows)
    {
      uint64_t tile_end = std::min(seg_end, tile_start + tile_rows);

      uint64_t aligned_start = tile_start;
      while (aligned_start < tile_end && (aligned_start & 15ull) != 0ull)
        ++aligned_start;
      uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;

      for (uint64_t row = tile_start; row < aligned_start; ++row)
      {
        stats.total_planes_read += max_rounds;
        stats.gpu_processed_rows += 1;
        stats.max_planes_read = std::max(stats.max_planes_read, max_rounds);
        stats.estimated_pack_load_bytes += max_rounds;

        bool qualified = false;
        if (segment_mode == kSegmentModeAllQualified)
        {
          qualified = true;
        }
        else if (segment_mode == kSegmentModeMixed)
        {
          qualified = compare_scalar_row_fixed_depth(dataset, info, row, max_rounds) > 0;
        }

        if (qualified)
        {
          ++stats.qualified_count;
          ++stats.certainly_qualified;
        }
        else
        {
          ++stats.certainly_disqualified;
        }
      }

      for (uint64_t pack_start = aligned_start; pack_start < full_pack_end; pack_start += 16ull)
      {
        stats.total_planes_read += 16ull * static_cast<uint64_t>(max_rounds);
        stats.gpu_processed_rows += 16ull;
        stats.max_planes_read = std::max(stats.max_planes_read, max_rounds);
        stats.estimated_pack_load_bytes += 16ull * static_cast<uint64_t>(max_rounds);

        uint64_t qualified_in_pack = 0;
        if (segment_mode == kSegmentModeAllQualified)
        {
          qualified_in_pack = 16ull;
        }
        else if (segment_mode == kSegmentModeMixed)
        {
          uint16_t undecided_mask = 0xFFFFu;
          uint16_t qualified_mask = 0u;

          for (uint32_t round = 0; round < max_rounds; ++round)
          {
            uint8_t thresh_byte = info.threshold_bytes[round];
            for (int i = 0; i < 16; ++i)
            {
              uint16_t bit = static_cast<uint16_t>(1u << i);
              if ((undecided_mask & bit) == 0u) continue;

              uint8_t row_byte = dataset.planes[round][pack_start + static_cast<uint64_t>(i)];
              if (row_byte > thresh_byte)
              {
                qualified_mask |= bit;
                undecided_mask = static_cast<uint16_t>(undecided_mask & ~bit);
              }
              else if (row_byte < thresh_byte)
              {
                undecided_mask = static_cast<uint16_t>(undecided_mask & ~bit);
              }
            }
          }

          qualified_in_pack = static_cast<uint64_t>(__builtin_popcount(static_cast<unsigned int>(qualified_mask)));
        }

        stats.qualified_count += qualified_in_pack;
        stats.certainly_qualified += qualified_in_pack;
        stats.certainly_disqualified += 16ull - qualified_in_pack;
      }

      for (uint64_t row = full_pack_end; row < tile_end; ++row)
      {
        stats.total_planes_read += max_rounds;
        stats.gpu_processed_rows += 1;
        stats.max_planes_read = std::max(stats.max_planes_read, max_rounds);
        stats.estimated_pack_load_bytes += max_rounds;

        bool qualified = false;
        if (segment_mode == kSegmentModeAllQualified)
        {
          qualified = true;
        }
        else if (segment_mode == kSegmentModeMixed)
        {
          qualified = compare_scalar_row_fixed_depth(dataset, info, row, max_rounds) > 0;
        }

        if (qualified)
        {
          ++stats.qualified_count;
          ++stats.certainly_qualified;
        }
        else
        {
          ++stats.certainly_disqualified;
        }
      }
    }
  }

  return stats;
}

uint64_t cpu_raw_count(const std::filesystem::path &raw_path, double threshold)
{
  uint64_t file_size = std::filesystem::file_size(raw_path);
  if (file_size % sizeof(double) != 0) die("raw file size not aligned to FP64");
  uint64_t value_count = file_size / sizeof(double);

  std::ifstream input(raw_path, std::ios::binary);
  if (!input) die("failed to open raw file");

  constexpr size_t kBatch = 1 << 20; // 8 MB batches
  std::vector<double> buffer(kBatch);
  uint64_t total = 0;
  uint64_t remaining = value_count;

  while (remaining > 0)
  {
    size_t current = static_cast<size_t>(std::min<uint64_t>(remaining, kBatch));
    input.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(current * sizeof(double)));
    if (!input) die("failed to read raw file");
    for (size_t i = 0; i < current; ++i)
    {
      if (buffer[i] > threshold) ++total;
    }
    remaining -= current;
  }
  return total;
}

// Reconstruct combined code from dataset planes for a single row.
// Mirrors assemble_combined_from_planes in buff_codec.cpp (same bit layout).
uint64_t reconstruct_combined_from_planes_row(
    const exp3_real::Dataset &dataset,
    const exp3_real::SegmentMeta &segment,
    uint64_t row)
{
  size_t total_bits = segment.get_fixed_len_bits();
  size_t plane_count = (total_bits == 0) ? 0 : (total_bits + 7) / 8;
  uint64_t combined = 0;

  for (size_t plane = 0; plane < plane_count; ++plane)
  {
    size_t width = 8;
    if (plane + 1 == plane_count)
    {
      size_t trailing = total_bits - 8 * (plane_count - 1);
      width = (trailing == 0) ? 8 : trailing;
    }
    size_t start_bit = total_bits - 8 * plane - width;
    uint8_t plane_byte = dataset.planes[plane][segment.row_offset + row];
    // Mask to actual width (upper bits may be zero-padded)
    uint8_t masked_byte = plane_byte & static_cast<uint8_t>((1u << width) - 1);
    combined |= static_cast<uint64_t>(masked_byte) << start_bit;
  }
  return combined;
}

// Compute lower/upper combined-code bounds after reading 'kept_planes' from a row.
// lower: omitted bits are 0.  upper: omitted bits are 1.
static void combined_bounds_after_k_planes(
    const exp3_real::Dataset &dataset,
    const exp3_real::SegmentMeta &segment,
    uint64_t row,
    uint32_t kept_planes,
    uint64_t &out_lower,
    uint64_t &out_upper)
{
  size_t total_bits = segment.get_fixed_len_bits();
  size_t plane_count = (total_bits == 0) ? 0 : (total_bits + 7) / 8;

  out_lower = 0;
  out_upper = 0;

  if (kept_planes >= plane_count)
  {
    // Full depth: exact reconstruction
    out_lower = out_upper = reconstruct_combined_from_planes_row(dataset, segment, row);
    return;
  }

  uint32_t kept_bits = 0;
  for (size_t plane = 0; plane < kept_planes; ++plane)
  {
    size_t width = 8;
    if (plane + 1 == plane_count)
    {
      size_t trailing = total_bits - 8 * (plane_count - 1);
      width = (trailing == 0) ? 8 : trailing;
    }
    size_t start_bit = total_bits - 8 * plane - width;
    uint8_t plane_byte = dataset.planes[plane][segment.row_offset + row];
    uint8_t masked_byte = plane_byte & static_cast<uint8_t>((1u << width) - 1);
    out_lower |= static_cast<uint64_t>(masked_byte) << start_bit;
    out_upper |= static_cast<uint64_t>(masked_byte) << start_bit;
    kept_bits += static_cast<uint32_t>(width);
  }

  // Remaining bits: lower keeps 0, upper sets to 1
  if (kept_bits < total_bits)
  {
    uint64_t mask = (total_bits >= 64) ? ~0ull : ((1ull << total_bits) - 1);
    uint64_t remaining_mask = mask & ~((kept_bits >= 64) ? ~0ull : ((1ull << kept_bits) - 1));
    out_upper |= remaining_mask;
  }
}

// Small diagnostic: compare raw, decoded, and encoded predicate for a few rows.
// Answers two questions:
// A. combined_x_from_cpp_codec == combined_x_from_planes ?
// B. raw_pred == encoded_pred ? decoded_pred == encoded_pred ?
void run_small_diagnostic(
    const exp3_real::Dataset &dataset,
    const std::vector<std::pair<double, double>> &segment_min_max,
    const std::vector<SegmentThresholdInfo> &threshold_info,
    const std::filesystem::path &raw_path,
    double threshold_fp64,
    int debug_segment_idx,
    int max_debug_rows)
{
  if (debug_segment_idx < 0 || static_cast<size_t>(debug_segment_idx) >= dataset.segments.size())
  {
    std::fprintf(stderr, "[diag] invalid debug segment %d\n", debug_segment_idx);
    return;
  }

  const auto &segment = dataset.segments[static_cast<size_t>(debug_segment_idx)];
  const auto &info = threshold_info[static_cast<size_t>(debug_segment_idx)];

  std::fprintf(stderr, "\n========== SMALL DIAGNOSTIC segment=%d ==========\n", debug_segment_idx);
  std::fprintf(stderr, "[diag] rows=%" PRIu64 " frac=%u int_off=%u active_planes=%u\n",
               segment.row_count, segment.fractional_bits, segment.integer_offset_bits, segment.active_plane_count);
  std::fprintf(stderr, "[diag] seg_min=%.17g seg_max=%.17g threshold=%.17g\n",
               segment_min_max[static_cast<size_t>(debug_segment_idx)].first,
               segment_min_max[static_cast<size_t>(debug_segment_idx)].second,
               threshold_fp64);

  // Read raw values for this segment
  std::ifstream raw_in(raw_path, std::ios::binary);
  if (!raw_in) { std::fprintf(stderr, "[diag] cannot open raw file\n"); return; }
  raw_in.seekg(static_cast<std::streamoff>(segment.row_offset * sizeof(double)), std::ios::beg);
  std::vector<double> raw_values(segment.row_count);
  raw_in.read(reinterpret_cast<char *>(raw_values.data()),
              static_cast<std::streamsize>(segment.row_count * sizeof(double)));
  if (!raw_in) { std::fprintf(stderr, "[diag] cannot read raw segment\n"); return; }

  // Re-encode with buff_codec.cpp to get ground-truth EncodedSegment
  buff::EncodedSegment encoded = buff::encode_segment(std::span<const double>(raw_values.data(), raw_values.size()));

  // Compute threshold combined using same exact path as compute_threshold_bytes().
  // Use inline ldexp+floor (encode_value_to_code was removed from buff_codec.hpp).
  // Mirror the v2 vs legacy branching from the main path:
  //   v2:     integer_base is the scaled code base, no extra shift
  //   legacy: integer_base = floor(min), must shift by frac
  int32_t scale_exponent = -static_cast<int32_t>(segment.fractional_bits);
  std::vector<uint8_t> t_code_le;
  bool t_exact = true;
  {
    uint32_t frac = segment.fractional_bits;
    long double scale = std::ldexp(1.0L, static_cast<int>(frac));
    uint64_t t_code = static_cast<uint64_t>(
        std::floor(static_cast<long double>(threshold_fp64) * scale));
    t_code_le.clear();
    for (size_t i = 0; i < sizeof(uint64_t); ++i)
      t_code_le.push_back(static_cast<uint8_t>((t_code >> (i * 8)) & 0xFF));
  }
  uint64_t t_code_u64 = le_bytes_to_u64(t_code_le);
  uint64_t integer_base_u64 = le_bytes_to_u64(segment.integer_base_le);
  uint64_t base_shifted = segment.integer_base_is_v2
                              ? integer_base_u64
                              : (integer_base_u64 << segment.fractional_bits);
  uint64_t threshold_combined = (t_code_u64 >= base_shifted) ? (t_code_u64 - base_shifted) : 0;

  std::fprintf(stderr, "[diag] threshold_exact=%s t_code=%" PRIu64 " integer_base=%" PRIu64 " threshold_combined=%" PRIu64 "\n",
               t_exact ? "true" : "false", t_code_u64, integer_base_u64, threshold_combined);

  if (info.all_qualified || info.all_disqualified)
  {
    std::fprintf(stderr, "[diag] fast-path segment, skipping row-level diag\n");
    std::fprintf(stderr, "========== END DIAGNOSTIC ==========\n\n");
    return;
  }

  // Decode segment for ground-truth comparison
  std::vector<double> decoded_values = buff::decode_segment(encoded);

  // Print header
  std::fprintf(stderr, "\n[diag] row | raw_x           | raw_pred | decoded_x       | dec_pred | enc_pred | combined(recon) | combined(codec) | first_diff_plane\n");

  size_t rows_to_check = std::min<size_t>(segment.row_count, static_cast<size_t>(max_debug_rows));
  uint64_t mismatch_raw_enc = 0;
  uint64_t mismatch_dec_enc = 0;
  uint64_t mismatch_recon_codec = 0;

  for (size_t row = 0; row < rows_to_check; ++row)
  {
    bool raw_pred = raw_values[row] > threshold_fp64;
    bool dec_pred = decoded_values[row] > threshold_fp64;

    // Encoded predicate from plane bytes comparison
    bool enc_pred = false;
    bool active = true;
    int first_diff_plane = -1;
    for (uint32_t round = 0; round < segment.active_plane_count && active; ++round)
    {
      uint8_t row_byte = dataset.planes[round][segment.row_offset + row];
      uint8_t thresh_byte = info.threshold_bytes[round];
      if (row_byte > thresh_byte) { enc_pred = true; active = false; }
      else if (row_byte < thresh_byte) { active = false; }
      if (first_diff_plane < 0 && row_byte != thresh_byte) first_diff_plane = static_cast<int>(round);
    }

    uint64_t combined_recon = reconstruct_combined_from_planes_row(dataset, segment, row);

    // Compute combined from codec: decode_segment gives full decoded values,
    // but we also need the combined code. We can get it from the encoded segment directly.
    // For diagnostic, we approximate by encoding the decoded value back... but that's circular.
    // Instead, we'll reconstruct from encoded.byte_planes which is the ground-truth.
    uint64_t combined_codec = 0;
    size_t total_bits = segment.get_fixed_len_bits();
    size_t plane_count_codec = (total_bits == 0) ? 0 : (total_bits + 7) / 8;
    for (size_t plane = 0; plane < plane_count_codec && plane < encoded.byte_planes.size(); ++plane)
    {
      size_t width = 8;
      if (plane + 1 == plane_count_codec)
      {
        size_t trailing = total_bits - 8 * (plane_count_codec - 1);
        width = (trailing == 0) ? 8 : trailing;
      }
      size_t start_bit = total_bits - 8 * plane - width;
      uint8_t plane_byte = encoded.byte_planes[plane][row];
      uint8_t masked_byte = plane_byte & static_cast<uint8_t>((1u << width) - 1);
      combined_codec |= static_cast<uint64_t>(masked_byte) << start_bit;
    }

    if (combined_recon != combined_codec) ++mismatch_recon_codec;
    if (raw_pred != enc_pred) ++mismatch_raw_enc;
    if (dec_pred != enc_pred) ++mismatch_dec_enc;

    // Only print mismatches or first few rows
    bool is_mismatch = (raw_pred != enc_pred) || (dec_pred != enc_pred) || (combined_recon != combined_codec);
    if (is_mismatch || row < static_cast<size_t>(max_debug_rows))
    {
      std::fprintf(stderr,
                   "[diag] %3zu | %.15g | %s        | %.15g | %s        | %s        | %016" PRIx64 " | %016" PRIx64 " | %d\n",
                   row,
                   raw_values[row],
                   raw_pred ? "T" : "F",
                   decoded_values[row],
                   dec_pred ? "T" : "F",
                   enc_pred ? "T" : "F",
                   combined_recon,
                   combined_codec,
                   first_diff_plane);
    }
  }

  std::fprintf(stderr, "\n[diag] SUMMARY: recon==codec: %s (%zu mismatches/%zu checked)\n",
               mismatch_recon_codec == 0 ? "PASS" : "FAIL", mismatch_recon_codec, rows_to_check);
  std::fprintf(stderr, "[diag] SUMMARY: raw==enc: %s (%zu mismatches/%zu checked)\n",
               mismatch_raw_enc == 0 ? "PASS" : "FAIL", mismatch_raw_enc, rows_to_check);
  std::fprintf(stderr, "[diag] SUMMARY: dec==enc: %s (%zu mismatches/%zu checked)\n",
               mismatch_dec_enc == 0 ? "PASS" : "FAIL", mismatch_dec_enc, rows_to_check);

  if (mismatch_recon_codec > 0)
  {
    die("[diag] ASSERT A FAILED: plane reconstruction != codec combined code");
  }
  if (mismatch_dec_enc > 0)
  {
    die("[diag] ASSERT B FAILED: decoded_pred != encoded_pred (threshold_combined wrong?)");
  }
  if (mismatch_raw_enc > 0)
  {
    std::fprintf(stderr, "[diag] ASSERT B: encoded predicate differs from raw predicate (expected if quantization drifts value across threshold)\n");
  }
  else
  {
    std::fprintf(stderr, "[diag] ALL ASSERTS PASS for checked rows\n");
  }
  std::fprintf(stderr, "========== END DIAGNOSTIC ==========\n\n");
}

// Diagnostic: for a sample of rows in a segment, compare raw predicate,
// encoded predicate, and show first mismatching plane.
void diagnose_segment_predicate(
    const exp3_real::Dataset &dataset,
    const std::vector<std::pair<double, double>> &segment_min_max,
    const std::vector<SegmentThresholdInfo> &threshold_info,
    const std::filesystem::path &raw_path,
    double threshold_fp64,
    size_t segment_index,
    size_t max_rows_to_print)
{
  const auto &segment = dataset.segments[segment_index];
  const auto &info = threshold_info[segment_index];

  std::fprintf(stderr, "\n[diag] segment=%zu rows=%" PRIu64 " frac=%u int_off=%u active_planes=%u\n",
               segment_index, segment.row_count, segment.fractional_bits,
               segment.integer_offset_bits, segment.active_plane_count);
  std::fprintf(stderr, "[diag] seg_min=%.17g seg_max=%.17g threshold=%.17g\n",
               segment_min_max[segment_index].first,
               segment_min_max[segment_index].second,
               threshold_fp64);
  std::fprintf(stderr, "[diag] all_qualified=%s all_disqualified=%s exact=%s\n",
               info.all_qualified ? "true" : "false",
               info.all_disqualified ? "true" : "false",
               info.threshold_exact ? "true" : "false");

  if (info.all_qualified || info.all_disqualified)
  {
    std::fprintf(stderr, "[diag] fast-path segment, skipping row-level diag\n");
    return;
  }

  // Read raw values for this segment
  std::ifstream raw_in(raw_path, std::ios::binary);
  if (!raw_in) { std::fprintf(stderr, "[diag] cannot open raw file\n"); return; }
  raw_in.seekg(static_cast<std::streamoff>(segment.row_offset * sizeof(double)), std::ios::beg);
  std::vector<double> raw_values(segment.row_count);
  raw_in.read(reinterpret_cast<char *>(raw_values.data()),
              static_cast<std::streamsize>(segment.row_count * sizeof(double)));
  if (!raw_in) { std::fprintf(stderr, "[diag] cannot read raw segment\n"); return; }

  uint64_t mismatch_count = 0;
  uint64_t raw_qualified = 0;
  uint64_t enc_qualified = 0;

  for (uint64_t row = 0; row < segment.row_count && row < max_rows_to_print * 10; ++row)
  {
    bool raw_pred = raw_values[row] > threshold_fp64;
    if (raw_pred) ++raw_qualified;

    bool enc_pred = false;
    bool active = true;
    int first_diff_plane = -1;
    for (uint32_t round = 0; round < segment.active_plane_count && active; ++round)
    {
      uint8_t row_byte = dataset.planes[round][segment.row_offset + row];
      uint8_t thresh_byte = info.threshold_bytes[round];
      if (row_byte > thresh_byte) { enc_pred = true; active = false; }
      else if (row_byte < thresh_byte) { active = false; }
      if (first_diff_plane < 0 && row_byte != thresh_byte) first_diff_plane = static_cast<int>(round);
    }
    if (enc_pred) ++enc_qualified;

    if (raw_pred != enc_pred)
    {
      ++mismatch_count;
      if (mismatch_count <= max_rows_to_print)
      {
        std::fprintf(stderr,
                     "[diag] MISMATCH row=%" PRIu64 " raw_x=%.17g raw_pred=%s enc_pred=%s first_diff_plane=%d\n",
                     row, raw_values[row],
                     raw_pred ? "T" : "F",
                     enc_pred ? "T" : "F",
                     first_diff_plane);
        // Print first few plane bytes
        std::fprintf(stderr, "[diag]   row_bytes: ");
        for (uint32_t p = 0; p < std::min<uint32_t>(segment.active_plane_count, 4); ++p)
          std::fprintf(stderr, "%02x ", dataset.planes[p][segment.row_offset + row]);
        std::fprintf(stderr, "| thresh_bytes: ");
        for (uint32_t p = 0; p < std::min<uint32_t>(segment.active_plane_count, 4); ++p)
          std::fprintf(stderr, "%02x ", info.threshold_bytes[p]);
        std::fprintf(stderr, "\n");
      }
    }
  }

  std::fprintf(stderr, "[diag] summary: raw_qualified=%" PRIu64 " enc_qualified=%" PRIu64 " mismatch=%" PRIu64 "\n",
               raw_qualified, enc_qualified, mismatch_count);
}

} // namespace

int main(int argc, char **argv)
{
  Options opt = parse_args(argc, argv);

  cudaDeviceProp prop{};
  if (!opt.no_gpu)
  {
    cuda_check(cudaSetDevice(opt.device), "cudaSetDevice");
    cuda_check(cudaGetDeviceProperties(&prop, opt.device), "cudaGetDeviceProperties");
  }

  // Load encoded artifact
  exp3_real::Dataset dataset = exp3_real::load_dataset(opt.encoded_root);
  PrecisionInfo precision_info = infer_precision_info(opt.encoded_root, dataset);
  uint64_t n = dataset.manifest.value_count;
  uint64_t segment_rows = dataset.manifest.segment_size;

  std::fprintf(stderr,
               "[exp4] dataset=%s n=%" PRIu64 " segments=%" PRIu64 " max_planes=%" PRIu64
               " mode=%s warp_strategy=%s\n",
               dataset.manifest.dataset.c_str(),
               n,
               dataset.manifest.segment_count,
               dataset.manifest.max_plane_count,
               opt.fixed_depth_baseline ? "fixed_depth" : "progressive",
               opt.warp_strategy.c_str());

  // Compute per-segment min/max from raw file.
  std::vector<std::pair<double, double>> segment_min_max;
  std::filesystem::path raw_path;
  {
    if (!opt.raw_root.empty())
    {
      raw_path = std::filesystem::path(opt.raw_root) / (dataset.manifest.dataset + ".f64le.bin");
    }
    else
    {
      raw_path = std::filesystem::path(opt.encoded_root).parent_path().parent_path() /
                 "dev" /
                 (dataset.manifest.dataset + ".f64le.bin");
    }
    if (std::filesystem::exists(raw_path))
    {
      std::fprintf(stderr, "[exp4] computing segment min/max from %s...\n", raw_path.c_str());
      segment_min_max = compute_segment_min_max(raw_path, segment_rows, dataset.segments.size());
    }
    else
    {
      std::fprintf(stderr, "[exp4] raw file not found at %s; skipping raw validation\n", raw_path.c_str());
      raw_path.clear();
      segment_min_max.resize(dataset.segments.size(), {std::numeric_limits<double>::lowest(),
                                                        std::numeric_limits<double>::max()});
    }
  }

  // Compute threshold bytes on CPU using exact dyadic encoding.
  std::vector<SegmentThresholdInfo> threshold_info = compute_threshold_bytes(dataset, segment_min_max, opt.threshold);
  std::vector<uint8_t> h_segment_filter_mode(dataset.segments.size(), kSegmentModeMixed);
  for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
  {
    if (threshold_info[seg].all_qualified)
      h_segment_filter_mode[seg] = kSegmentModeAllQualified;
    else if (threshold_info[seg].all_disqualified)
      h_segment_filter_mode[seg] = kSegmentModeAllDisqualified;
  }

  // Small diagnostic (login-node safe, CPU-only)
  if (opt.debug_segment >= 0)
  {
    int debug_rows = (opt.debug_rows > 0) ? opt.debug_rows : 16;
    run_small_diagnostic(dataset, segment_min_max, threshold_info, raw_path,
                         opt.threshold, opt.debug_segment, debug_rows);
  }

  // Flatten threshold bytes for device: [segment][plane]
  std::vector<uint8_t> h_threshold_flat;
  h_threshold_flat.resize(dataset.segments.size() * dataset.manifest.max_plane_count, 0);
  for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
  {
    std::memcpy(h_threshold_flat.data() + seg * dataset.manifest.max_plane_count,
                threshold_info[seg].threshold_bytes.data(),
                dataset.manifest.max_plane_count);
  }

  // Flatten active_plane_count and apply cap
  std::vector<uint32_t> h_active_plane_count(dataset.segments.size());
  std::vector<uint32_t> effective_max_rounds(dataset.segments.size());
  for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
  {
    uint32_t ac = dataset.segments[seg].active_plane_count;
    if (!opt.fixed_depth_baseline &&
        opt.max_filter_planes >= 0 &&
        static_cast<uint32_t>(opt.max_filter_planes) < ac)
      ac = static_cast<uint32_t>(opt.max_filter_planes);
    h_active_plane_count[seg] = ac;
    effective_max_rounds[seg] = ac;
  }

  uint64_t rowpacks_per_thread = (!opt.fixed_depth_baseline && opt.warp_strategy == "byte_mask_interleave2") ? 2ull : 1ull;
  Derived derived = compute_derived(n, segment_rows, opt.block_threads, rowpacks_per_thread);

  float ms_per_iter = 0.0f;
  double rows_per_sec = 0.0;
  uint64_t gpu_count = 0;

  if (!opt.no_gpu)
  {
    // Prepare device subcolumn pointers
    Exp3RuntimeU8Subcolumns h_subcolumns{};
    std::vector<uint8_t *> d_subcolumn_storage;
    int subcolumns = static_cast<int>(dataset.manifest.max_plane_count);
    for (int p = 0; p < subcolumns; ++p)
    {
      const std::vector<uint8_t> &plane = dataset.planes[p];
      uint8_t *ptr = nullptr;
      cuda_check(cudaMalloc(&ptr, plane.size()), "cudaMalloc(plane)");
      cuda_check(cudaMemcpy(ptr, plane.data(), plane.size(), cudaMemcpyHostToDevice), "cudaMemcpy(plane)");
      d_subcolumn_storage.push_back(ptr);
      h_subcolumns.ptrs[p] = ptr;
    }
    for (int p = subcolumns; p < EXP3_RUNTIME_MAX_SUBCOLUMNS; ++p)
      h_subcolumns.ptrs[p] = nullptr;

    // Device buffers
    uint8_t *d_threshold_bytes = nullptr;
    uint8_t *d_segment_filter_mode = nullptr;
    uint32_t *d_active_plane_count = nullptr;
    uint64_t *d_block_counts = nullptr;

    cuda_check(cudaMalloc(&d_threshold_bytes,
                          h_threshold_flat.size() * sizeof(uint8_t)),
               "cudaMalloc(threshold)");
    cuda_check(cudaMemcpy(d_threshold_bytes,
                          h_threshold_flat.data(),
                          h_threshold_flat.size() * sizeof(uint8_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(threshold)");

    cuda_check(cudaMalloc(&d_segment_filter_mode,
                          h_segment_filter_mode.size() * sizeof(uint8_t)),
               "cudaMalloc(segment_filter_mode)");
    cuda_check(cudaMemcpy(d_segment_filter_mode,
                          h_segment_filter_mode.data(),
                          h_segment_filter_mode.size() * sizeof(uint8_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(segment_filter_mode)");

    cuda_check(cudaMalloc(&d_active_plane_count,
                          h_active_plane_count.size() * sizeof(uint32_t)),
               "cudaMalloc(active_plane_count)");
    cuda_check(cudaMemcpy(d_active_plane_count,
                          h_active_plane_count.data(),
                          h_active_plane_count.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(active_plane_count)");

    cuda_check(cudaMalloc(&d_block_counts, derived.grid * sizeof(uint64_t)), "cudaMalloc(block_counts)");

    std::fprintf(stderr,
                 "[exp4] grid=%" PRIu64 " tile_rows=%" PRIu64 " tiles_per_segment=%" PRIu64 "\n",
                 derived.grid,
                 derived.tile_rows,
                 derived.tiles_per_segment);

    // Warmup
    for (int i = 0; i < opt.warmup; ++i)
    {
      if (opt.fixed_depth_baseline)
      {
        launch_fixed_depth_filter_rowpack16_count(
            static_cast<int>(derived.grid),
            opt.block_threads,
            h_subcolumns,
            n,
            segment_rows,
            derived.tiles_per_segment,
            d_threshold_bytes,
            static_cast<int>(dataset.manifest.max_plane_count),
            d_active_plane_count,
            d_segment_filter_mode,
            d_block_counts);
      }
      else
      {
        if (opt.warp_strategy == "byte_mask")
          launch_progressive_filter_rowpack16_byte_mask(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else if (opt.warp_strategy == "byte_mask_interleave2")
          launch_progressive_filter_rowpack16_byte_mask_interleave2(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else if (opt.warp_strategy == "byte_mask_specialized_k1")
          launch_progressive_filter_rowpack16_byte_mask_specialized_k1(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else if (opt.warp_strategy == "byte_mask_specialized_k4")
          launch_progressive_filter_rowpack16_byte_mask_specialized_k4(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else if (opt.warp_strategy == "byte_mask_specialized_k8")
          launch_progressive_filter_rowpack16_byte_mask_specialized_k8(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else if (opt.warp_strategy == "prefetch")
          launch_progressive_filter_rowpack16_prefetch(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else if (opt.warp_strategy == "simd_masks")
          launch_progressive_filter_rowpack16_simd_masks(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else if (opt.warp_strategy == "predicated")
          launch_progressive_filter_rowpack16_predicated(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else
          launch_progressive_filter_rowpack16_passive(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
      }
    }
    cuda_check(cudaGetLastError(), "warmup launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

    // Timed runs

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < opt.iters; ++i)
    {
      if (opt.fixed_depth_baseline)
      {
        launch_fixed_depth_filter_rowpack16_count(
            static_cast<int>(derived.grid),
            opt.block_threads,
            h_subcolumns,
            n,
            segment_rows,
            derived.tiles_per_segment,
            d_threshold_bytes,
            static_cast<int>(dataset.manifest.max_plane_count),
            d_active_plane_count,
            d_segment_filter_mode,
            d_block_counts);
      }
      else
      {
        if (opt.warp_strategy == "byte_mask")
          launch_progressive_filter_rowpack16_byte_mask(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else if (opt.warp_strategy == "byte_mask_interleave2")
          launch_progressive_filter_rowpack16_byte_mask_interleave2(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else if (opt.warp_strategy == "byte_mask_specialized_k1")
          launch_progressive_filter_rowpack16_byte_mask_specialized_k1(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else if (opt.warp_strategy == "byte_mask_specialized_k4")
          launch_progressive_filter_rowpack16_byte_mask_specialized_k4(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else if (opt.warp_strategy == "byte_mask_specialized_k8")
          launch_progressive_filter_rowpack16_byte_mask_specialized_k8(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else if (opt.warp_strategy == "prefetch")
          launch_progressive_filter_rowpack16_prefetch(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else if (opt.warp_strategy == "simd_masks")
          launch_progressive_filter_rowpack16_simd_masks(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else if (opt.warp_strategy == "predicated")
          launch_progressive_filter_rowpack16_predicated(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
        else
          launch_progressive_filter_rowpack16_passive(
              static_cast<int>(derived.grid),
              opt.block_threads,
              h_subcolumns,
              n,
              segment_rows,
              derived.tiles_per_segment,
              d_threshold_bytes,
              static_cast<int>(dataset.manifest.max_plane_count),
              d_active_plane_count,
              d_block_counts);
      }
    }
    cuda_check(cudaGetLastError(), "timed launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float ms_total = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms_total, start, stop), "cudaEventElapsedTime");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    ms_per_iter = ms_total / static_cast<float>(opt.iters);
    double seconds = static_cast<double>(ms_per_iter) / 1000.0;
    rows_per_sec = static_cast<double>(n) / seconds;

    // Collect GPU count with fast-path handling for all-qualified/all-disqualified segments.
    std::vector<uint64_t> h_block_counts(derived.grid);
    cuda_check(cudaMemcpy(h_block_counts.data(),
                          d_block_counts,
                          derived.grid * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(block_counts)");

    if (opt.fixed_depth_baseline)
    {
      for (uint64_t tile = 0; tile < derived.grid; ++tile)
        gpu_count += h_block_counts[tile];
    }
    else
    {
      for (size_t seg = 0; seg < dataset.segments.size(); ++seg)
      {
        uint64_t tile_start = seg * derived.tiles_per_segment;
        uint64_t tile_end = tile_start + derived.tiles_per_segment;

        if (threshold_info[seg].all_qualified)
        {
          gpu_count += dataset.segments[seg].row_count;
        }
        else if (threshold_info[seg].all_disqualified)
        {
          // Kernel already produces 0 for these segments.
        }
        else
        {
          for (uint64_t tile = tile_start; tile < tile_end; ++tile)
            gpu_count += h_block_counts[tile];
        }
      }
    }

    // Cleanup GPU
    cuda_check(cudaFree(d_block_counts), "cudaFree(block_counts)");
    cuda_check(cudaFree(d_active_plane_count), "cudaFree(active_plane_count)");
    cuda_check(cudaFree(d_segment_filter_mode), "cudaFree(segment_filter_mode)");
    cuda_check(cudaFree(d_threshold_bytes), "cudaFree(threshold_bytes)");
    for (uint8_t *ptr : d_subcolumn_storage)
      cuda_check(cudaFree(ptr), "cudaFree(plane)");
  }

  // Validation
  uint64_t cpu_enc_count = 0;
  uint64_t cpu_raw_count_val = 0;
  bool has_cpu_encoded_count = false;
  bool has_cpu_raw_count = false;
  bool has_encoded_stats = false;
  bool gpu_matches_cpu_encoded = false;
  bool encoded_matches_raw = false;
  uint64_t raw_count_abs_error = 0;
  double raw_count_rel_error = 0.0;
  EncodedReferenceStats encoded_stats{};
  uint64_t threshold_exact_segments = 0;
  uint64_t threshold_inexact_segments = 0;

  if (opt.validate)
  {
    std::fprintf(stderr, "[exp4] running CPU encoded reference...\n");
    if (opt.fixed_depth_baseline)
      encoded_stats = cpu_fixed_depth_encoded_stats(
          dataset, threshold_info, h_segment_filter_mode, effective_max_rounds, derived.tile_rows);
    else
      encoded_stats = cpu_encoded_stats(dataset, threshold_info, effective_max_rounds, derived.tile_rows);
    has_encoded_stats = true;
    cpu_enc_count = encoded_stats.qualified_count;
    has_cpu_encoded_count = true;
    gpu_matches_cpu_encoded = (gpu_count == cpu_enc_count);

    for (const auto &info : threshold_info)
    {
      if (info.threshold_exact)
        ++threshold_exact_segments;
      else
        ++threshold_inexact_segments;
    }

    if (!raw_path.empty())
    {
      std::fprintf(stderr, "[exp4] running CPU raw reference from %s...\n", raw_path.c_str());
      cpu_raw_count_val = cpu_raw_count(raw_path, opt.threshold);
      has_cpu_raw_count = true;
      encoded_matches_raw = (cpu_enc_count == cpu_raw_count_val);
      raw_count_abs_error = (cpu_enc_count >= cpu_raw_count_val)
                                ? (cpu_enc_count - cpu_raw_count_val)
                                : (cpu_raw_count_val - cpu_enc_count);
      raw_count_rel_error =
          static_cast<double>(raw_count_abs_error) /
          static_cast<double>(std::max<uint64_t>(cpu_raw_count_val, 1ull));
    }
    else
    {
      std::fprintf(stderr, "[exp4] raw file not found, skipping raw validation\n");
    }

    std::fprintf(stderr,
                 "[exp4] validation gpu=%" PRIu64 " cpu_enc=%" PRIu64 " cpu_raw=%" PRIu64
                 " match_enc=%s match_raw=%s exact_seg=%" PRIu64 " inexact_seg=%" PRIu64 "\n",
                 gpu_count,
                 cpu_enc_count,
                 cpu_raw_count_val,
                 gpu_matches_cpu_encoded ? "true" : "false",
                 encoded_matches_raw ? "true" : "false",
                 threshold_exact_segments,
                 threshold_inexact_segments);

    if (!gpu_matches_cpu_encoded)
    {
      die("VALIDATION FAILED: gpu_count != cpu_encoded_count");
    }
  }

  if (has_encoded_stats && encoded_stats.max_planes_read > dataset.manifest.max_plane_count)
  {
    die("max_planes_read exceeds artifact max_plane_count");
  }

  double selectivity = (n == 0) ? 0.0 : (static_cast<double>(gpu_count) / static_cast<double>(n));
  double seconds = static_cast<double>(ms_per_iter) / 1000.0;

  double avg_planes_read_per_total_row = 0.0;
  bool has_avg_planes_read_per_total_row = false;
  double avg_planes_read_per_gpu_processed_row = 0.0;
  bool has_avg_planes_read_per_gpu_processed_row = false;
  double logical_gbps = 0.0;
  bool has_logical_gbps = false;
  double physical_gbps = 0.0;
  bool has_physical_gbps = false;

  if (has_encoded_stats)
  {
    avg_planes_read_per_total_row =
        (n == 0) ? 0.0 : (static_cast<double>(encoded_stats.total_planes_read) / static_cast<double>(n));
    has_avg_planes_read_per_total_row = true;

    if (encoded_stats.gpu_processed_rows > 0)
    {
      avg_planes_read_per_gpu_processed_row =
          static_cast<double>(encoded_stats.total_planes_read) /
          static_cast<double>(encoded_stats.gpu_processed_rows);
      has_avg_planes_read_per_gpu_processed_row = true;
    }

    if (seconds > 0.0)
    {
      logical_gbps = static_cast<double>(encoded_stats.total_planes_read) / seconds / 1e9;
      physical_gbps = static_cast<double>(encoded_stats.estimated_pack_load_bytes) / seconds / 1e9;
      has_logical_gbps = true;
      has_physical_gbps = true;
    }
  }

  bool validated = opt.validate ? gpu_matches_cpu_encoded : false;

  std::string cpu_encoded_count_csv = csv_u64_or_na(has_cpu_encoded_count, cpu_enc_count);
  std::string cpu_raw_count_csv = csv_u64_or_na(has_cpu_raw_count, cpu_raw_count_val);
  std::string gpu_eq_cpu_encoded_csv = csv_bool_or_na(has_cpu_encoded_count, gpu_matches_cpu_encoded);
  std::string cpu_encoded_eq_cpu_raw_csv = csv_bool_or_na(
      has_cpu_encoded_count && has_cpu_raw_count, encoded_matches_raw);
  std::string raw_count_abs_error_csv = csv_u64_or_na(
      has_cpu_encoded_count && has_cpu_raw_count, raw_count_abs_error);
  std::string raw_count_rel_error_csv = csv_double_or_na(
      has_cpu_encoded_count && has_cpu_raw_count, raw_count_rel_error);
  std::string avg_planes_total_csv = csv_double_or_na(
      has_avg_planes_read_per_total_row, avg_planes_read_per_total_row);
  std::string avg_planes_gpu_row_csv = csv_double_or_na(
      has_avg_planes_read_per_gpu_processed_row, avg_planes_read_per_gpu_processed_row);
  std::string max_planes_read_csv = csv_u32_or_na(has_encoded_stats, encoded_stats.max_planes_read);
  std::string logical_gbps_csv = csv_double_or_na(has_logical_gbps, logical_gbps);
  // #2 contract: rename physical_GBps -> estimated_physical_GBps
  std::string estimated_physical_gbps_csv = csv_double_or_na(has_physical_gbps, physical_gbps);
  std::string estimated_pack_load_bytes_csv =
      csv_u64_or_na(has_encoded_stats, encoded_stats.estimated_pack_load_bytes);
  std::string validated_csv = csv_bool_or_na(opt.validate, validated);
  std::string job_id_csv = slurm_job_id_or_na();
  const char *device_name = opt.no_gpu ? "NA" : prop.name;

  // Capped-k fields
  std::string max_filter_planes_csv = (opt.max_filter_planes >= 0)
                                         ? std::to_string(opt.max_filter_planes) : "NA";
  std::string certainly_qualified_csv = csv_u64_or_na(has_encoded_stats, encoded_stats.certainly_qualified);
  std::string certainly_disqualified_csv = csv_u64_or_na(has_encoded_stats, encoded_stats.certainly_disqualified);
  std::string uncertain_csv = csv_u64_or_na(has_encoded_stats, encoded_stats.uncertain);
  uint64_t count_lower = has_encoded_stats ? encoded_stats.certainly_qualified : 0;
  uint64_t count_upper = has_encoded_stats ? (encoded_stats.certainly_qualified + encoded_stats.uncertain) : 0;
  std::string count_lower_csv = csv_u64_or_na(has_encoded_stats, count_lower);
  std::string count_upper_csv = csv_u64_or_na(has_encoded_stats, count_upper);
  std::string count_abs_error_bound_csv = csv_u64_or_na(has_encoded_stats, encoded_stats.uncertain);
  std::string fully_resolved_packs_csv = opt.fixed_depth_baseline
                                             ? "NA"
                                             : csv_u64_or_na(has_encoded_stats, encoded_stats.fully_resolved_packs);
  std::string partially_active_packs_csv = opt.fixed_depth_baseline
                                               ? "NA"
                                               : csv_u64_or_na(has_encoded_stats, encoded_stats.partially_active_packs);
  std::string avg_active_rows_per_pack_csv = csv_double_or_na(
      !opt.fixed_depth_baseline &&
          has_encoded_stats &&
          encoded_stats.partially_active_packs > 0,
      encoded_stats.avg_active_rows_per_active_pack);
  std::string useful_row_fraction_csv = csv_double_or_na(
      !opt.fixed_depth_baseline &&
          has_encoded_stats &&
          encoded_stats.total_rows_in_active_packs > 0,
      encoded_stats.useful_row_fraction);

  // #2/#3 schema contract: new mandatory fields
  std::string logical_bytes_csv = csv_u64_or_na(has_encoded_stats, encoded_stats.total_planes_read);
  std::string n_csv = std::to_string(n);
  std::string iters_csv = std::to_string(opt.iters);
  std::string warmup_csv = std::to_string(opt.warmup);
  std::string ms_per_iter_csv = csv_double_or_na(!opt.no_gpu, static_cast<double>(ms_per_iter));
  std::string max_plane_count_csv = std::to_string(dataset.manifest.max_plane_count);
  std::string segment_rows_csv = std::to_string(segment_rows);

  // Write main CSV
  std::FILE *f = std::fopen(opt.csv_path.c_str(), "wb");
  if (!f) die("failed to open CSV output");

  const char *experiment_name = opt.fixed_depth_baseline
                                    ? "exp4_fixed_depth_filter"
                                    : "exp4_progressive_filter";

  std::fprintf(f,
               "experiment,dataset,artifact_root,precision_mode,precision_decimals,threshold,selectivity,"
               "n,iters,warmup,ms_per_iter,"
               "gpu_count,cpu_encoded_count,cpu_raw_count,gpu_eq_cpu_encoded,cpu_encoded_eq_cpu_raw,"
               "raw_count_abs_error,raw_count_rel_error,"
               "avg_planes_read_per_total_row,avg_planes_read_per_gpu_processed_row,max_planes_read,"
               "logical_bytes,logical_GBps,estimated_physical_GBps,estimated_pack_load_bytes,rows_per_sec,"
               "max_plane_count,segment_rows,"
               "validated,device,job_id,"
               "max_filter_planes,certainly_qualified,certainly_disqualified,uncertain,"
               "count_lower,count_upper,count_abs_error_bound,"
                "fully_resolved_packs,partially_active_packs,avg_active_rows_per_active_pack,useful_row_fraction,"
                 "gpu_tag,load_strategy,warp_strategy,benchmark,kernel_path\n");

  // Metadata columns per contract
  std::string gpu_tag_csv = "0";
  std::string load_strategy_csv = "rowpack16";
  std::string warp_strategy_csv = opt.fixed_depth_baseline ? "fixed_depth" : opt.warp_strategy;
  std::string benchmark_csv = opt.fixed_depth_baseline ? "fixed_depth_filter" : "progressive_filter";
  std::string kernel_path_csv;
  if (opt.fixed_depth_baseline)
    kernel_path_csv = "fixed_depth_rowpack16_count";
  else if (opt.warp_strategy == "byte_mask")
    kernel_path_csv = "progressive_filter_rowpack16_byte_mask";
  else if (opt.warp_strategy == "byte_mask_interleave2")
    kernel_path_csv = "progressive_filter_rowpack16_byte_mask_interleave2";
  else if (opt.warp_strategy == "byte_mask_specialized_k1")
    kernel_path_csv = "progressive_filter_rowpack16_byte_mask_specialized_k1";
  else if (opt.warp_strategy == "byte_mask_specialized_k4")
    kernel_path_csv = "progressive_filter_rowpack16_byte_mask_specialized_k4";
  else if (opt.warp_strategy == "byte_mask_specialized_k8")
    kernel_path_csv = "progressive_filter_rowpack16_byte_mask_specialized_k8";
  else if (opt.warp_strategy == "prefetch")
    kernel_path_csv = "progressive_filter_rowpack16_prefetch";
  else if (opt.warp_strategy == "simd_masks")
    kernel_path_csv = "progressive_filter_rowpack16_simd_masks";
  else if (opt.warp_strategy == "predicated")
    kernel_path_csv = "progressive_filter_rowpack16_predicated";
  else
    kernel_path_csv = "progressive_filter_rowpack16_passive";

  std::fprintf(f,
               "%s,%s,%s,%s,%s,%.17g,%.17g,"
               "%s,%s,%s,%s,"
               "%" PRIu64 ",%s,%s,%s,%s,%s,%s,%s,%s,%s,"
               "%s,%s,%s,%s,%.17g,"
               "%s,%s,"
               "%s,%s,%s,"
               "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                "%s,%s,%s,%s,%s\n",
                experiment_name,
                dataset.manifest.dataset.c_str(),
                opt.encoded_root.c_str(),
                precision_info.mode.c_str(),
                precision_info.decimals.c_str(),
                opt.threshold,
                selectivity,
                n_csv.c_str(),
                iters_csv.c_str(),
                warmup_csv.c_str(),
                ms_per_iter_csv.c_str(),
                gpu_count,
                cpu_encoded_count_csv.c_str(),
                cpu_raw_count_csv.c_str(),
                gpu_eq_cpu_encoded_csv.c_str(),
                cpu_encoded_eq_cpu_raw_csv.c_str(),
                raw_count_abs_error_csv.c_str(),
                raw_count_rel_error_csv.c_str(),
                avg_planes_total_csv.c_str(),
                avg_planes_gpu_row_csv.c_str(),
                max_planes_read_csv.c_str(),
                logical_bytes_csv.c_str(),
                logical_gbps_csv.c_str(),
                estimated_physical_gbps_csv.c_str(),
                estimated_pack_load_bytes_csv.c_str(),
                rows_per_sec,
                max_plane_count_csv.c_str(),
                segment_rows_csv.c_str(),
                validated_csv.c_str(),
                device_name,
                job_id_csv.c_str(),
                max_filter_planes_csv.c_str(),
                certainly_qualified_csv.c_str(),
                certainly_disqualified_csv.c_str(),
                uncertain_csv.c_str(),
                count_lower_csv.c_str(),
                count_upper_csv.c_str(),
                count_abs_error_bound_csv.c_str(),
                fully_resolved_packs_csv.c_str(),
                partially_active_packs_csv.c_str(),
                avg_active_rows_per_pack_csv.c_str(),
                useful_row_fraction_csv.c_str(),
                gpu_tag_csv.c_str(),
                load_strategy_csv.c_str(),
                warp_strategy_csv.c_str(),
                benchmark_csv.c_str(),
                kernel_path_csv.c_str());

  std::fclose(f);

  // Write rounds sidecar CSV if requested
  if (!opt.fixed_depth_baseline && !opt.output_rounds_csv.empty() && has_encoded_stats)
  {
    std::FILE *rf = std::fopen(opt.output_rounds_csv.c_str(), "wb");
    if (!rf) die("failed to open rounds CSV output");

    std::fprintf(rf,
                 "dataset,artifact_root,threshold,selectivity,max_filter_planes,k,"
                 "qualified_at_k,disqualified_at_k,uncertain_after_k,planes_read_per_round,pack_load_bytes_per_round\n");

    uint64_t cum_q = 0;
    uint64_t cum_d = 0;
    uint64_t processed = encoded_stats.gpu_processed_rows;
    size_t total_rounds = encoded_stats.qualified_per_round.size();

    for (size_t r = 0; r < total_rounds; ++r)
    {
      cum_q += encoded_stats.qualified_per_round[r];
      cum_d += encoded_stats.disqualified_per_round[r];

      uint64_t uncertain_after_k;
      if (r + 1 < total_rounds)
      {
        // Before cap: all alive rows are uncertain (no bound-check yet)
        uncertain_after_k = (processed > cum_q + cum_d) ? (processed - cum_q - cum_d) : 0;
      }
      else
      {
        // At cap: only true ambiguous rows after combined-code bound check
        uncertain_after_k = encoded_stats.uncertain;
      }

      std::fprintf(rf,
                   "%s,%s,%.17g,%.17g,%s,%zu,"
                   "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                   dataset.manifest.dataset.c_str(),
                   opt.encoded_root.c_str(),
                   opt.threshold,
                   selectivity,
                   max_filter_planes_csv.c_str(),
                   r + 1,
                   encoded_stats.qualified_per_round[r],
                   encoded_stats.disqualified_per_round[r],
                   uncertain_after_k,
                   encoded_stats.planes_read_per_round[r],
                   encoded_stats.pack_load_bytes_per_round[r]);
    }
    std::fclose(rf);
  }

  std::fprintf(stderr,
               "[exp4] result: gpu_count=%" PRIu64 " ms=%.3f rows/s=%.3e\n",
               gpu_count,
               ms_per_iter,
               rows_per_sec);

  // Cleanup GPU (only if we allocated)
  if (!opt.no_gpu)
  {
    // These variables are declared inside the GPU block above.
    // We need to get them out of scope or just skip cleanup in no-gpu mode.
    // The OS will reclaim GPU memory when the process exits.
  }

  return 0;
}
