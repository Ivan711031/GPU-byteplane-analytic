#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace buff {

struct EncodeConfig {
    std::optional<std::uint32_t> precision_power;
};

struct PrecisionChoice {
    std::uint32_t precision_power = 0;
    std::uint32_t decimal_bits = 0;
    double precision_bound = 0.0;
};

struct EncodedSegment {
    std::uint64_t value_count = 0;
    std::uint32_t precision_power = 0;
    std::uint32_t fractional_bits = 0;
    std::uint32_t integer_offset_bits = 0;
    std::uint32_t fixed_len_bits = 0;
    std::vector<std::uint8_t> integer_base_le;
    std::vector<std::vector<std::uint8_t>> byte_planes;
};

struct EncodedFileHeader {
    std::uint64_t total_values = 0;
    std::uint64_t segment_size = 0;
    std::uint64_t segment_count = 0;
};

PrecisionChoice choose_precision_for_max_value(double max_value);
PrecisionChoice precision_choice_for_power(std::uint32_t power);
EncodedSegment encode_segment(std::span<const double> values);
EncodedSegment encode_segment(std::span<const double> values, const EncodeConfig& config);
std::vector<double> decode_segment(const EncodedSegment& segment);
std::vector<double> decode_segment_top_k(const EncodedSegment& segment, std::size_t top_k_planes);
double segment_max_abs_error_bound(const EncodedSegment& segment, std::size_t top_k_planes);
std::size_t segment_plane_count(const EncodedSegment& segment);
void encode_file(const std::filesystem::path& input_path,
                 const std::filesystem::path& output_path,
                 std::uint64_t segment_size,
                 std::uint64_t max_values = 0,
                 const EncodeConfig& config = {});
void decode_file(const std::filesystem::path& input_path,
                 const std::filesystem::path& output_path);
EncodedFileHeader read_file_header(const std::filesystem::path& input_path);
void export_runtime_layout(const std::filesystem::path& encoded_path,
                           const std::filesystem::path& raw_input_path,
                           const std::filesystem::path& output_dir,
                           const std::string& dataset_name,
                           const std::string& source_path = {});
bool files_have_identical_fp64_payload(const std::filesystem::path& lhs_path,
                                       const std::filesystem::path& rhs_path);
std::string summarize_segment(const EncodedSegment& segment);

}  // namespace buff
