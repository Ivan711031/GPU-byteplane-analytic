#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace exp3_real
{

constexpr const char *kManifestFileName = "manifest.json";
constexpr const char *kSummaryFileName = "summary.json";
constexpr const char *kSegmentMetaFileName = "segment_meta.csv";
constexpr const char *kEncodedLayoutName = "plane_major_zero_padded";
constexpr const char *kFormatName = "exp3_encoded_dev_v1";
constexpr int kMaxRuntimePlanes = 32;

struct Manifest
{
  std::string format = kFormatName;
  std::string dataset;
  std::string source_path;
  std::uint64_t segment_size = 4096;
  std::uint64_t value_count = 0;
  std::uint64_t segment_count = 0;
  std::uint64_t max_plane_count = 0;
  std::uint64_t segment_plane_count_min = 0;
  std::uint64_t segment_plane_count_max = 0;
  std::string encoded_layout = kEncodedLayoutName;
  double exact_sum = 0.0;
};

struct SegmentMeta
{
  std::uint64_t segment_index = 0;
  std::uint64_t row_offset = 0;
  std::uint64_t row_count = 0;
  std::uint32_t active_plane_count = 0;
  // legacy alias for effective_fractional_bits
  std::uint32_t fractional_bits = 0;
  std::uint32_t raw_fractional_bits = 0;
  std::uint32_t precision_cap_bits = 0;
  std::uint32_t effective_fractional_bits = 0;
  std::uint32_t integer_offset_bits = 0;
  std::vector<std::uint8_t> integer_base_le;
  // True when integer_base_hex was written as 0x... (v2 fixed-point code base).
  // Legacy plain-hex format writes integer_base = floor(min_value) and
  // plane bytes store code(v) - (integer_base << f).
  // V2 0x format writes integer_base = round(min * 2^f) as a signed i64 code
  // and plane bytes store code(v) - integer_base directly.
  bool integer_base_is_v2 = false;
  double segment_base = 0.0;
  std::vector<double> plane_basis;

  std::size_t get_fixed_len_bits() const
  {
    if (active_plane_count == 0) return 0;
    if (active_plane_count > 1 && !plane_basis.empty() && plane_basis[0] > 0.0)
    {
      int exp_val = 0;
      std::frexp(plane_basis[0], &exp_val);
      int log2_b0 = exp_val - 1;
      return static_cast<std::size_t>(log2_b0 + 8 + static_cast<int>(fractional_bits));
    }
    std::size_t raw_total = static_cast<std::size_t>(fractional_bits) + static_cast<std::size_t>(integer_offset_bits);
    return (raw_total > 8) ? 8 : raw_total;
  }
};

struct Dataset
{
  Manifest manifest;
  std::vector<SegmentMeta> segments;
  std::vector<std::vector<std::uint8_t>> planes;
};

enum class ExtremumOp
{
  Min,
  Max,
};

struct ExactExtrema
{
  double min = 0.0;
  double max = 0.0;
  std::uint64_t min_code = 0;
  std::uint64_t max_code = 0;
};

struct ProgressiveExtremaBounds
{
  double lower = 0.0;
  double upper = 0.0;
  std::uint64_t lower_code = 0;
  std::uint64_t upper_code = 0;
};

struct ProgressiveExtremaSummary
{
  ProgressiveExtremaBounds min;
  ProgressiveExtremaBounds max;
};

struct CandidateCounts
{
  uint64_t min_candidates = 0;
  uint64_t max_candidates = 0;
  uint64_t total_rows = 0;
};

std::filesystem::path manifest_path(const std::filesystem::path &root);
std::filesystem::path summary_path(const std::filesystem::path &root);
std::filesystem::path segment_meta_path(const std::filesystem::path &root);
std::filesystem::path plane_file_path(const std::filesystem::path &root, std::uint64_t plane_index);

void ensure_directory(const std::filesystem::path &path);

void write_manifest_json(const std::filesystem::path &path, const Manifest &manifest);
Manifest read_manifest_json(const std::filesystem::path &path);

void write_summary_json(const std::filesystem::path &path,
                        const Manifest &manifest,
                        const std::vector<SegmentMeta> &segments);

void write_segment_meta_csv(const std::filesystem::path &path,
                            const Manifest &manifest,
                            const std::vector<SegmentMeta> &segments);
std::vector<SegmentMeta> read_segment_meta_csv(const std::filesystem::path &path,
                                              std::uint64_t max_plane_count);

void write_plane_files(const std::filesystem::path &root,
                       const std::vector<std::vector<std::uint8_t>> &planes);
std::vector<std::vector<std::uint8_t>> read_plane_files(const std::filesystem::path &root,
                                                        std::uint64_t plane_count,
                                                        std::uint64_t value_count);

Dataset load_dataset(const std::filesystem::path &root);

double cpu_approximate_sum(const Dataset &dataset, int refinement_depth);

std::size_t plane_width(std::size_t fixed_len_bits, std::size_t plane_index);
double segment_tail_upper_bound(const SegmentMeta &segment, std::size_t keep_planes);

ExactExtrema cpu_exact_extrema(const Dataset &dataset);
ProgressiveExtremaSummary cpu_progressive_extrema_summary(const Dataset &dataset,
                                                          int refinement_depth);
ProgressiveExtremaBounds cpu_progressive_extrema_bounds(const Dataset &dataset,
                                                        int refinement_depth,
                                                        ExtremumOp op);
CandidateCounts cpu_candidate_counts(const Dataset &dataset, int refinement_depth);

std::string bytes_to_hex(const std::vector<std::uint8_t> &bytes);
std::vector<std::uint8_t> hex_to_bytes(std::string_view hex);
std::vector<std::uint8_t> parse_integer_base_hex(std::string_view text, bool *out_is_v2 = nullptr);

std::vector<double> plane_basis_for_segment(std::uint32_t fractional_bits,
                                            std::uint32_t integer_offset_bits,
                                            std::uint64_t max_plane_count);

} // namespace exp3_real
