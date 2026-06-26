#include "buff_codec.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace buff {
namespace {

constexpr std::array<char, 8> kMagic = {'B', 'U', 'F', 'F', '6', '4', '\0', '\3'};
constexpr std::uint64_t kFirstOne = UINT64_C(0x8000000000000000);
constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 12> kPrecisionOptions = {{
    {12, 50}, {11, 38}, {10, 35}, {9, 31}, {8, 28}, {7, 25},
    {6, 21},  {5, 18},  {4, 15},  {3, 11}, {2, 8},  {1, 5},
}};

template <typename UInt>
void write_pod(std::ostream& out, UInt value) {
    static_assert(std::is_integral_v<UInt>);
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) {
        throw std::runtime_error("failed to write output stream");
    }
}

template <typename UInt>
UInt read_pod(std::istream& in) {
    static_assert(std::is_integral_v<UInt>);
    UInt value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("failed to read input stream");
    }
    return value;
}

double get_precision_bound(std::uint32_t power) {
    return 0.49 * std::pow(10.0, -static_cast<int>(power));
}

std::int32_t precision_exponent(double precision) {
    std::uint64_t bits = std::bit_cast<std::uint64_t>(precision);
    return static_cast<std::int32_t>((bits >> 52U) & 0x7ffU) - 1023;
}

std::uint64_t count_values_in_fp64_file(const std::filesystem::path& input_path) {
    std::uint64_t file_size = std::filesystem::file_size(input_path);
    if (file_size % sizeof(double) != 0) {
        throw std::runtime_error("input file size is not aligned to FP64 rows");
    }
    return file_size / sizeof(double);
}

double scan_file_max_value(const std::filesystem::path& input_path,
                           std::uint64_t segment_size,
                           std::uint64_t max_values) {
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open input file for max-value scan: " + input_path.string());
    }

    std::vector<double> buffer(static_cast<std::size_t>(segment_size));
    std::uint64_t total_values = count_values_in_fp64_file(input_path);
    if (max_values != 0) {
        total_values = std::min(total_values, max_values);
    }

    double max_value = 0.0;
    std::uint64_t remaining = total_values;
    while (remaining > 0) {
        std::uint64_t current = std::min(remaining, segment_size);
        input.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(current * sizeof(double)));
        if (!input) {
            throw std::runtime_error("failed to read input FP64 payload while scanning max");
        }
        for (std::size_t index = 0; index < current; ++index) {
            double value = buffer[index];
            if (!std::isfinite(value) || value < 0.0) {
                throw std::runtime_error("codec only supports finite non-negative FP64 values");
            }
            if (value > max_value) {
                max_value = value;
            }
        }
        remaining -= current;
    }

    return max_value;
}

std::int64_t fetch_fixed_aligned(double value, std::int32_t precision_exp, std::uint32_t decimal_bits) {
    if (value == 0.0) {
        return 0;
    }
    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error("codec only supports finite non-negative FP64 values");
    }

    std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    std::int32_t exponent = static_cast<std::int32_t>((bits >> 52U) & 0x7ffU) - 1023;
    if (exponent < precision_exp) {
        return 0;
    }

    __uint128_t significand = static_cast<__uint128_t>((bits << 11U) | kFirstOne);
    std::int32_t shift = 63 - exponent - static_cast<std::int32_t>(decimal_bits);
    __uint128_t fixed = shift >= 0 ? (significand >> shift) : (significand << (-shift));
    if (fixed > static_cast<__uint128_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error("fixed-point value exceeded i64 range");
    }
    return static_cast<std::int64_t>(fixed);
}

std::size_t calc_fixed_len(std::uint64_t delta) {
    if (delta == 0) {
        return 0;
    }
    return static_cast<std::size_t>(std::bit_width(delta));
}

std::vector<std::uint8_t> i64_to_le_bytes(std::int64_t value) {
    std::vector<std::uint8_t> bytes(sizeof(value), 0);
    std::memcpy(bytes.data(), &value, sizeof(value));
    return bytes;
}

std::int64_t le_bytes_to_i64(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() != sizeof(std::int64_t)) {
        throw std::runtime_error("expected 8-byte base payload");
    }
    std::int64_t value = 0;
    std::memcpy(&value, bytes.data(), sizeof(value));
    return value;
}

std::size_t plane_count_for_fixed_len(std::size_t fixed_len) {
    return fixed_len == 0 ? 0 : ((fixed_len + 7U) / 8U);
}

std::size_t plane_width(std::size_t fixed_len, std::size_t plane_index) {
    std::size_t plane_count = plane_count_for_fixed_len(fixed_len);
    if (plane_index >= plane_count) {
        throw std::runtime_error("plane index is out of range");
    }
    if (plane_index + 1U == plane_count) {
        std::size_t trailing = fixed_len - 8U * (plane_count - 1U);
        return trailing == 0 ? 8U : trailing;
    }
    return 8U;
}

PrecisionChoice resolve_precision(const EncodeConfig& config, double max_value) {
    if (config.precision_power.has_value()) {
        PrecisionChoice choice = precision_choice_for_power(*config.precision_power);
        long double scaled = static_cast<long double>(max_value) *
                             std::ldexp(1.0L, static_cast<int>(choice.decimal_bits));
        if (scaled > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
            throw std::runtime_error("requested precision power overflows i64 fixed-point range");
        }
        return choice;
    }
    return choose_precision_for_max_value(max_value);
}

std::vector<double> materialize_segment(const EncodedSegment& segment, std::size_t top_k_planes) {
    std::vector<double> values(segment.value_count, 0.0);
    if (segment.value_count == 0) {
        return values;
    }

    std::int64_t base_fixed = le_bytes_to_i64(segment.integer_base_le);
    std::size_t plane_count = segment.byte_planes.size();
    std::size_t keep = std::min(top_k_planes, plane_count);
    std::size_t fixed_len = segment.fixed_len_bits;

    for (std::uint64_t row = 0; row < segment.value_count; ++row) {
        std::uint64_t prefix = 0;
        std::size_t kept_bits = 0;
        for (std::size_t plane = 0; plane < keep; ++plane) {
            if (segment.byte_planes[plane].size() != segment.value_count) {
                throw std::runtime_error("corrupt segment: plane length mismatch");
            }
            std::size_t width = plane_width(fixed_len, plane);
            prefix = (prefix << width) | static_cast<std::uint64_t>(segment.byte_planes[plane][row]);
            kept_bits += width;
        }
        std::size_t omitted_bits = fixed_len >= kept_bits ? (fixed_len - kept_bits) : 0;
        std::uint64_t approx_delta = prefix << omitted_bits;
        values[static_cast<std::size_t>(row)] =
            static_cast<double>(base_fixed + static_cast<std::int64_t>(approx_delta)) /
            std::ldexp(1.0, static_cast<int>(segment.fractional_bits));
    }

    return values;
}

void write_bytes(std::ostream& out, const std::vector<std::uint8_t>& bytes) {
    if (!bytes.empty()) {
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!out) {
            throw std::runtime_error("failed to write byte payload");
        }
    }
}

std::vector<std::uint8_t> read_bytes(std::istream& in, std::size_t size) {
    std::vector<std::uint8_t> bytes(size, 0);
    if (size == 0) {
        return bytes;
    }

    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (!in) {
        throw std::runtime_error("failed to read byte payload");
    }
    return bytes;
}

std::string json_escape(std::string_view text) {
    std::string escaped;
    escaped.reserve(text.size() + 8U);
    for (char ch : text) {
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped.push_back(ch);
                break;
        }
    }
    return escaped;
}

std::uint32_t exact_fractional_bits_for_value(double value) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error("runtime export only supports finite non-negative FP64 values");
    }
    if (value == 0.0) {
        return 0;
    }
    std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    std::uint64_t exponent_bits = (bits >> 52U) & 0x7ffU;
    std::uint64_t mantissa_bits = bits & ((UINT64_C(1) << 52U) - 1U);

    std::uint64_t significand = 0;
    std::int32_t exponent = 0;
    if (exponent_bits == 0) {
        if (mantissa_bits == 0) {
            return 0;
        }
        significand = mantissa_bits;
        exponent = -1074;
    } else {
        significand = (UINT64_C(1) << 52U) | mantissa_bits;
        exponent = static_cast<std::int32_t>(exponent_bits) - 1023 - 52;
    }

    std::uint32_t trailing = std::countr_zero(significand);
    exponent += static_cast<std::int32_t>(trailing);
    return exponent < 0 ? static_cast<std::uint32_t>(-exponent) : 0U;
}

std::string integer_base_hex_string(const std::vector<std::uint8_t>& bytes) {
    std::int64_t signed_value = le_bytes_to_i64(bytes);
    std::uint64_t unsigned_bits = 0;
    std::memcpy(&unsigned_bits, &signed_value, sizeof(unsigned_bits));
    std::ostringstream out;
    out << "0x" << std::hex << std::setfill('0') << std::setw(16) << unsigned_bits;
    return out.str();
}

EncodedFileHeader read_stream_header(std::istream& input) {
    std::array<char, kMagic.size()> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input) {
        throw std::runtime_error("failed to read BUFF header");
    }
    if (magic != kMagic) {
        throw std::runtime_error("input is not a supported BUFF64 file");
    }

    return EncodedFileHeader{
        .total_values = read_pod<std::uint64_t>(input),
        .segment_size = read_pod<std::uint64_t>(input),
        .segment_count = read_pod<std::uint64_t>(input),
    };
}

void write_stream_header(std::ostream& output, const EncodedFileHeader& header) {
    output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    if (!output) {
        throw std::runtime_error("failed to write BUFF header");
    }
    write_pod(output, header.total_values);
    write_pod(output, header.segment_size);
    write_pod(output, header.segment_count);
}

void write_segment(std::ostream& out, const EncodedSegment& segment) {
    write_pod(out, segment.value_count);
    write_pod(out, segment.precision_power);
    write_pod(out, segment.fractional_bits);
    write_pod(out, segment.integer_offset_bits);
    write_pod(out, segment.fixed_len_bits);
    write_pod(out, static_cast<std::uint32_t>(segment.integer_base_le.size()));
    write_pod(out, static_cast<std::uint32_t>(segment.byte_planes.size()));
    write_bytes(out, segment.integer_base_le);

    for (const auto& plane : segment.byte_planes) {
        if (plane.size() != segment.value_count) {
            throw std::runtime_error("byte plane length does not match segment length");
        }
        write_bytes(out, plane);
    }
}

EncodedSegment read_segment(std::istream& in) {
    EncodedSegment segment;
    segment.value_count = read_pod<std::uint64_t>(in);
    segment.precision_power = read_pod<std::uint32_t>(in);
    segment.fractional_bits = read_pod<std::uint32_t>(in);
    segment.integer_offset_bits = read_pod<std::uint32_t>(in);
    segment.fixed_len_bits = read_pod<std::uint32_t>(in);
    std::uint32_t base_width = read_pod<std::uint32_t>(in);
    std::uint32_t plane_count = read_pod<std::uint32_t>(in);
    segment.integer_base_le = read_bytes(in, base_width);
    segment.byte_planes.reserve(plane_count);
    for (std::uint32_t plane = 0; plane < plane_count; ++plane) {
        segment.byte_planes.push_back(read_bytes(in, static_cast<std::size_t>(segment.value_count)));
    }
    return segment;
}

}  // namespace

PrecisionChoice choose_precision_for_max_value(double max_value) {
    if (!std::isfinite(max_value) || max_value < 0.0) {
        throw std::runtime_error("max value must be finite and non-negative");
    }
    for (const auto& [power, decimal_bits] : kPrecisionOptions) {
        long double scaled = static_cast<long double>(max_value) *
                             std::ldexp(1.0L, static_cast<int>(decimal_bits));
        if (scaled <= static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
            return PrecisionChoice{
                .precision_power = power,
                .decimal_bits = decimal_bits,
                .precision_bound = get_precision_bound(power),
            };
        }
    }
    throw std::runtime_error("no safe precision setting for dataset max value");
}

PrecisionChoice precision_choice_for_power(std::uint32_t power) {
    for (const auto& [candidate_power, decimal_bits] : kPrecisionOptions) {
        if (candidate_power == power) {
            return PrecisionChoice{
                .precision_power = power,
                .decimal_bits = decimal_bits,
                .precision_bound = get_precision_bound(power),
            };
        }
    }
    throw std::runtime_error("unsupported precision power");
}

EncodedSegment encode_segment(std::span<const double> values) {
    double max_value = 0.0;
    for (double value : values) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error("codec only supports finite non-negative FP64 values");
        }
        if (value > max_value) {
            max_value = value;
        }
    }
    return encode_segment(values, EncodeConfig{.precision_power = choose_precision_for_max_value(max_value).precision_power});
}

EncodedSegment encode_segment(std::span<const double> values, const EncodeConfig& config) {
    EncodedSegment segment;
    segment.value_count = values.size();
    if (values.empty()) {
        return segment;
    }

    double max_value = 0.0;
    for (double value : values) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error("codec only supports finite non-negative FP64 values");
        }
        if (value > max_value) {
            max_value = value;
        }
    }

    PrecisionChoice choice = resolve_precision(config, max_value);
    segment.precision_power = choice.precision_power;
    segment.fractional_bits = choice.decimal_bits;
    std::int32_t precision_exp = precision_exponent(choice.precision_bound);

    std::vector<std::int64_t> fixed_values;
    fixed_values.reserve(values.size());
    std::int64_t min_fixed = std::numeric_limits<std::int64_t>::max();
    std::int64_t max_fixed = std::numeric_limits<std::int64_t>::min();
    for (double value : values) {
        std::int64_t fixed = fetch_fixed_aligned(value, precision_exp, choice.decimal_bits);
        if (fixed < min_fixed) {
            min_fixed = fixed;
        }
        if (fixed > max_fixed) {
            max_fixed = fixed;
        }
        fixed_values.push_back(fixed);
    }

    std::uint64_t delta = static_cast<std::uint64_t>(max_fixed - min_fixed);
    std::size_t fixed_len = calc_fixed_len(delta);
    segment.fixed_len_bits = static_cast<std::uint32_t>(fixed_len);
    if (fixed_len < segment.fractional_bits) {
        segment.integer_offset_bits = 0;
    } else {
        segment.integer_offset_bits = static_cast<std::uint32_t>(fixed_len - segment.fractional_bits);
    }
    segment.integer_base_le = i64_to_le_bytes(min_fixed);

    std::size_t plane_count = plane_count_for_fixed_len(fixed_len);
    if (plane_count > 8U) {
        throw std::runtime_error("Rust-like fixed-point codec exceeded 8 planes; precision selection is inconsistent");
    }
    segment.byte_planes.assign(plane_count, std::vector<std::uint8_t>(values.size(), 0));

    std::vector<std::uint64_t> delta_values;
    delta_values.reserve(values.size());
    for (std::int64_t fixed : fixed_values) {
        delta_values.push_back(static_cast<std::uint64_t>(fixed - min_fixed));
    }

    if (fixed_len == 0) {
        return segment;
    }

    for (std::size_t row = 0; row < values.size(); ++row) {
        std::uint64_t delta_value = delta_values[row];
        std::size_t remain = fixed_len;
        std::size_t plane = 0;

        if (remain < 8U) {
            std::uint64_t mask = (UINT64_C(1) << remain) - 1U;
            segment.byte_planes[0][row] = static_cast<std::uint8_t>(delta_value & mask);
            continue;
        }

        remain -= 8U;
        segment.byte_planes[plane++][row] = static_cast<std::uint8_t>(remain > 0 ? (delta_value >> remain) : delta_value);

        while (remain >= 8U) {
            remain -= 8U;
            segment.byte_planes[plane++][row] = static_cast<std::uint8_t>(remain > 0 ? (delta_value >> remain) : delta_value);
        }

        if (remain > 0) {
            std::uint64_t mask = (UINT64_C(1) << remain) - 1U;
            segment.byte_planes[plane][row] = static_cast<std::uint8_t>(delta_value & mask);
        }
    }

    return segment;
}

std::vector<double> decode_segment(const EncodedSegment& segment) {
    return materialize_segment(segment, segment.byte_planes.size());
}

std::vector<double> decode_segment_top_k(const EncodedSegment& segment, std::size_t top_k_planes) {
    return materialize_segment(segment, top_k_planes);
}

double segment_max_abs_error_bound(const EncodedSegment& segment, std::size_t top_k_planes) {
    double precision_floor = get_precision_bound(segment.precision_power);
    std::size_t fixed_len = segment.fixed_len_bits;
    std::size_t keep = std::min(top_k_planes, segment.byte_planes.size());
    if (keep >= segment.byte_planes.size()) {
        return precision_floor;
    }
    std::size_t kept_bits = 0;
    for (std::size_t plane = 0; plane < keep; ++plane) {
        kept_bits += plane_width(fixed_len, plane);
    }
    std::size_t omitted_bits = fixed_len >= kept_bits ? (fixed_len - kept_bits) : 0;
    return precision_floor +
           std::ldexp(1.0, static_cast<int>(omitted_bits) - static_cast<int>(segment.fractional_bits));
}

std::size_t segment_plane_count(const EncodedSegment& segment) {
    return segment.byte_planes.size();
}

void encode_file(const std::filesystem::path& input_path,
                 const std::filesystem::path& output_path,
                 std::uint64_t segment_size,
                 std::uint64_t max_values,
                 const EncodeConfig& config) {
    if (segment_size == 0) {
        throw std::runtime_error("segment size must be greater than zero");
    }

    EncodeConfig effective_config = config;
    if (!effective_config.precision_power.has_value()) {
        PrecisionChoice choice = choose_precision_for_max_value(scan_file_max_value(input_path, segment_size, max_values));
        effective_config.precision_power = choice.precision_power;
    }

    std::uint64_t total_values = count_values_in_fp64_file(input_path);
    if (max_values != 0) {
        total_values = std::min(total_values, max_values);
    }
    std::uint64_t segment_count = (total_values + segment_size - 1U) / segment_size;

    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open input file for encoding: " + input_path.string());
    }

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open output file for encoding: " + output_path.string());
    }

    write_stream_header(output, EncodedFileHeader{
                                    .total_values = total_values,
                                    .segment_size = segment_size,
                                    .segment_count = segment_count,
                                });

    std::vector<double> buffer(static_cast<std::size_t>(segment_size));
    std::uint64_t remaining = total_values;
    while (remaining > 0) {
        std::uint64_t current = std::min(remaining, segment_size);
        input.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(current * sizeof(double)));
        if (!input) {
            throw std::runtime_error("failed to read input FP64 payload");
        }

        EncodedSegment segment = encode_segment(
            std::span<const double>(buffer.data(), static_cast<std::size_t>(current)),
            effective_config);
        write_segment(output, segment);
        remaining -= current;
    }
}

void decode_file(const std::filesystem::path& input_path,
                 const std::filesystem::path& output_path) {
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open encoded file for decoding: " + input_path.string());
    }

    EncodedFileHeader header = read_stream_header(input);

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open decoded file path: " + output_path.string());
    }

    std::uint64_t written = 0;
    for (std::uint64_t segment_index = 0; segment_index < header.segment_count; ++segment_index) {
        EncodedSegment segment = read_segment(input);
        std::vector<double> values = decode_segment(segment);
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(double)));
        if (!output) {
            throw std::runtime_error("failed to write decoded FP64 payload");
        }
        written += segment.value_count;
    }

    if (written != header.total_values) {
        throw std::runtime_error("decoded row count did not match BUFF header");
    }
}

EncodedFileHeader read_file_header(const std::filesystem::path& input_path) {
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open encoded file header: " + input_path.string());
    }
    return read_stream_header(input);
}

void export_runtime_layout(const std::filesystem::path& encoded_path,
                           const std::filesystem::path& raw_input_path,
                           const std::filesystem::path& output_dir,
                           const std::string& dataset_name,
                           const std::string& source_path) {
    std::ifstream encoded(encoded_path, std::ios::binary);
    if (!encoded) {
        throw std::runtime_error("failed to open encoded file for runtime export: " + encoded_path.string());
    }
    std::ifstream raw(raw_input_path, std::ios::binary);
    if (!raw) {
        throw std::runtime_error("failed to open raw FP64 file for runtime export: " + raw_input_path.string());
    }

    EncodedFileHeader header = read_stream_header(encoded);
    std::uint64_t raw_value_count = count_values_in_fp64_file(raw_input_path);
    if (raw_value_count != header.total_values) {
        throw std::runtime_error("raw row count does not match encoded header");
    }

    std::filesystem::create_directories(output_dir);

    struct SegmentRow {
        std::uint64_t segment_index = 0;
        std::uint64_t row_offset = 0;
        std::uint64_t row_count = 0;
        std::uint32_t active_plane_count = 0;
        std::uint32_t fractional_bits = 0;
        std::uint32_t integer_offset_bits = 0;
        std::uint32_t raw_fractional_bits = 0;
        std::uint32_t precision_cap_bits = 0;
        std::uint32_t effective_fractional_bits = 0;
        std::string integer_base_hex;
        double segment_base = 0.0;
        std::vector<double> plane_basis;
    };

    std::vector<EncodedSegment> segments;
    segments.reserve(static_cast<std::size_t>(header.segment_count));
    std::vector<SegmentRow> segment_rows;
    segment_rows.reserve(static_cast<std::size_t>(header.segment_count));
    std::vector<double> raw_buffer(static_cast<std::size_t>(header.segment_size));

    std::uint64_t row_offset = 0;
    std::size_t dataset_max_plane_count = 0;
    std::size_t dataset_min_plane_count = std::numeric_limits<std::size_t>::max();
    long double exact_sum = 0.0L;
    double exact_min = std::numeric_limits<double>::infinity();
    double exact_max = 0.0;
    long double plane_count_total = 0.0L;

    for (std::uint64_t segment_index = 0; segment_index < header.segment_count; ++segment_index) {
        EncodedSegment segment = read_segment(encoded);
        segments.push_back(segment);

        std::uint64_t row_count = segment.value_count;
        raw.read(reinterpret_cast<char*>(raw_buffer.data()),
                 static_cast<std::streamsize>(row_count * sizeof(double)));
        if (!raw) {
            throw std::runtime_error("failed to read raw FP64 payload during runtime export");
        }

        std::uint32_t raw_fractional_bits = 0;
        for (std::size_t index = 0; index < row_count; ++index) {
            double value = raw_buffer[index];
            exact_sum += static_cast<long double>(value);
            exact_min = std::min(exact_min, value);
            exact_max = std::max(exact_max, value);
            raw_fractional_bits = std::max(raw_fractional_bits, exact_fractional_bits_for_value(value));
        }

        const EncodedSegment& stored = segments.back();
        std::size_t plane_count = stored.byte_planes.size();
        dataset_max_plane_count = std::max(dataset_max_plane_count, plane_count);
        dataset_min_plane_count = std::min(dataset_min_plane_count, plane_count);
        plane_count_total += static_cast<long double>(plane_count);

        SegmentRow row;
        row.segment_index = segment_index;
        row.row_offset = row_offset;
        row.row_count = row_count;
        row.active_plane_count = static_cast<std::uint32_t>(plane_count);
        row.fractional_bits = stored.fractional_bits;
        row.integer_offset_bits = stored.integer_offset_bits;
        row.raw_fractional_bits = raw_fractional_bits;
        row.precision_cap_bits = stored.fractional_bits;
        row.effective_fractional_bits = stored.fractional_bits;
        row.integer_base_hex = integer_base_hex_string(stored.integer_base_le);
        std::int64_t base_fixed = le_bytes_to_i64(stored.integer_base_le);
        row.segment_base = static_cast<double>(base_fixed) / std::ldexp(1.0, static_cast<int>(stored.fractional_bits));
        row.plane_basis.assign(plane_count, 0.0);
        for (std::size_t plane = 0; plane < plane_count; ++plane) {
            std::size_t start_bit = plane_width(stored.fixed_len_bits, plane);
            (void)start_bit;
            std::size_t lsb_start = stored.fixed_len_bits == 0 ? 0 : (stored.fixed_len_bits - plane_width(stored.fixed_len_bits, plane) - 8U * plane);
            if (stored.fixed_len_bits != 0) {
                lsb_start = stored.fixed_len_bits - 8U * plane - plane_width(stored.fixed_len_bits, plane);
            }
            row.plane_basis[plane] =
                std::ldexp(1.0, static_cast<int>(lsb_start) - static_cast<int>(stored.fractional_bits));
        }
        segment_rows.push_back(std::move(row));
        row_offset += row_count;
    }

    if (dataset_min_plane_count == std::numeric_limits<std::size_t>::max()) {
        dataset_min_plane_count = 0;
    }

    std::vector<std::ofstream> plane_outputs;
    plane_outputs.reserve(dataset_max_plane_count);
    for (std::size_t plane = 0; plane < dataset_max_plane_count; ++plane) {
        std::ostringstream name;
        name << "plane_" << std::setfill('0') << std::setw(3) << plane << ".bin";
        plane_outputs.emplace_back(output_dir / name.str(), std::ios::binary | std::ios::trunc);
        if (!plane_outputs.back()) {
            throw std::runtime_error("failed to open plane output file");
        }
    }

    std::vector<std::uint8_t> zero_pad;
    for (const EncodedSegment& segment : segments) {
        for (std::size_t plane = 0; plane < dataset_max_plane_count; ++plane) {
            if (plane < segment.byte_planes.size()) {
                const auto& bytes = segment.byte_planes[plane];
                plane_outputs[plane].write(reinterpret_cast<const char*>(bytes.data()),
                                          static_cast<std::streamsize>(bytes.size()));
            } else {
                zero_pad.assign(static_cast<std::size_t>(segment.value_count), 0);
                plane_outputs[plane].write(reinterpret_cast<const char*>(zero_pad.data()),
                                          static_cast<std::streamsize>(zero_pad.size()));
            }
            if (!plane_outputs[plane]) {
                throw std::runtime_error("failed to write plane output");
            }
        }
    }

    std::ofstream meta(output_dir / "segment_meta.csv", std::ios::trunc);
    if (!meta) {
        throw std::runtime_error("failed to open segment_meta.csv");
    }
    meta << "segment_index,row_offset,row_count,active_plane_count,fractional_bits,integer_offset_bits"
         << ",raw_fractional_bits,precision_cap_bits,effective_fractional_bits,integer_base_hex,segment_base";
    for (std::size_t plane = 0; plane < dataset_max_plane_count; ++plane) {
        meta << ",plane_basis_" << plane;
    }
    meta << '\n';
    meta << std::setprecision(17);
    for (const SegmentRow& row : segment_rows) {
        meta << row.segment_index << ','
             << row.row_offset << ','
             << row.row_count << ','
             << row.active_plane_count << ','
             << row.fractional_bits << ','
             << row.integer_offset_bits << ','
             << row.raw_fractional_bits << ','
             << row.precision_cap_bits << ','
             << row.effective_fractional_bits << ','
             << row.integer_base_hex << ','
             << row.segment_base;
        for (std::size_t plane = 0; plane < dataset_max_plane_count; ++plane) {
            double basis = plane < row.plane_basis.size() ? row.plane_basis[plane] : 0.0;
            meta << ',' << basis;
        }
        meta << '\n';
    }

    std::uint32_t precision_power = segments.empty() ? 0U : segments.front().precision_power;
    std::uint32_t decimal_bits = segments.empty() ? 0U : segments.front().fractional_bits;
    double quantization_bound = precision_power == 0 ? 0.0 : get_precision_bound(precision_power);
    std::string source = source_path.empty() ? raw_input_path.string() : source_path;

    {
        std::ofstream manifest(output_dir / "manifest.json", std::ios::trunc);
        if (!manifest) {
            throw std::runtime_error("failed to open manifest.json");
        }
        manifest << "{\n"
                 << "  \"format\": \"exp3_encoded_dev_v1\",\n"
                 << "  \"dataset\": \"" << json_escape(dataset_name) << "\",\n"
                 << "  \"source_path\": \"" << json_escape(source) << "\",\n"
                 << "  \"segment_size\": " << header.segment_size << ",\n"
                 << "  \"value_count\": " << header.total_values << ",\n"
                 << "  \"segment_count\": " << header.segment_count << ",\n"
                 << "  \"max_plane_count\": " << dataset_max_plane_count << ",\n"
                 << "  \"segment_plane_count_min\": " << dataset_min_plane_count << ",\n"
                 << "  \"segment_plane_count_max\": " << dataset_max_plane_count << ",\n"
                 << "  \"encoded_layout\": \"plane_major_zero_padded\",\n"
                 << "  \"exact_sum\": " << std::setprecision(17) << static_cast<double>(exact_sum) << "\n"
                 << "}\n";
    }

    {
        std::ofstream summary(output_dir / "summary.json", std::ios::trunc);
        if (!summary) {
            throw std::runtime_error("failed to open summary.json");
        }
        summary << "{\n"
                << "  \"format\": \"exp3_encoded_dev_v1\",\n"
                << "  \"encoded_layout\": \"plane_major_zero_padded\",\n"
                << "  \"dataset\": \"" << json_escape(dataset_name) << "\",\n"
                << "  \"source_path\": \"" << json_escape(source) << "\",\n"
                << "  \"encoded_path\": \"" << json_escape(encoded_path.string()) << "\",\n"
                << "  \"segment_size\": " << header.segment_size << ",\n"
                << "  \"value_count\": " << header.total_values << ",\n"
                << "  \"segment_count\": " << header.segment_count << ",\n"
                << "  \"precision_power\": " << precision_power << ",\n"
                << "  \"decimal_bits\": " << decimal_bits << ",\n"
                << "  \"quantization_bound\": " << std::setprecision(17) << quantization_bound << ",\n"
                << "  \"raw_fractional_bits_max\": " << ([&]() {
                       std::uint32_t v = 0;
                       for (const SegmentRow& row : segment_rows) {
                           v = std::max(v, row.raw_fractional_bits);
                       }
                       return v;
                   })() << ",\n"
                << "  \"precision_cap_bits\": " << decimal_bits << ",\n"
                << "  \"effective_fractional_bits\": " << decimal_bits << ",\n"
                << "  \"max_plane_count\": " << dataset_max_plane_count << ",\n"
                << "  \"segment_plane_count_min\": " << dataset_min_plane_count << ",\n"
                << "  \"segment_plane_count_max\": " << dataset_max_plane_count << ",\n"
                << "  \"mean_plane_count\": " << static_cast<double>(plane_count_total / static_cast<long double>(header.segment_count)) << ",\n"
                << "  \"exact_sum\": " << static_cast<double>(exact_sum) << ",\n"
                << "  \"exact_min\": " << exact_min << ",\n"
                << "  \"exact_max\": " << exact_max << "\n"
                << "}\n";
    }
}

bool files_have_identical_fp64_payload(const std::filesystem::path& lhs_path,
                                       const std::filesystem::path& rhs_path) {
    if (std::filesystem::file_size(lhs_path) != std::filesystem::file_size(rhs_path)) {
        return false;
    }

    std::ifstream lhs(lhs_path, std::ios::binary);
    std::ifstream rhs(rhs_path, std::ios::binary);
    if (!lhs || !rhs) {
        throw std::runtime_error("failed to open files for bitwise comparison");
    }

    constexpr std::size_t kChunkBytes = 1U << 20U;
    std::vector<char> lhs_buffer(kChunkBytes);
    std::vector<char> rhs_buffer(kChunkBytes);

    while (lhs && rhs) {
        lhs.read(lhs_buffer.data(), static_cast<std::streamsize>(lhs_buffer.size()));
        rhs.read(rhs_buffer.data(), static_cast<std::streamsize>(rhs_buffer.size()));
        std::streamsize lhs_read = lhs.gcount();
        std::streamsize rhs_read = rhs.gcount();
        if (lhs_read != rhs_read) {
            return false;
        }
        if (lhs_read == 0) {
            break;
        }
        if (std::memcmp(lhs_buffer.data(), rhs_buffer.data(), static_cast<std::size_t>(lhs_read)) != 0) {
            return false;
        }
    }

    return true;
}

std::string summarize_segment(const EncodedSegment& segment) {
    std::ostringstream builder;
    builder << "rows=" << segment.value_count
            << " base_bytes=" << segment.integer_base_le.size()
            << " precision_power=" << segment.precision_power
            << " decimal_bits=" << segment.fractional_bits
            << " integer_bits=" << segment.integer_offset_bits
            << " fixed_len_bits=" << segment.fixed_len_bits
            << " planes=" << segment.byte_planes.size();
    return builder.str();
}

}  // namespace buff
