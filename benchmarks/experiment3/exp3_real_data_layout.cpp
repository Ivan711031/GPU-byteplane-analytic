#include "exp3_real_data_layout.hpp"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace exp3_real
{
namespace
{

[[noreturn]] void fail(const std::string &message)
{
  throw std::runtime_error(message);
}

[[nodiscard]] std::string escape_json_string(std::string_view value)
{
  std::string out;
  out.reserve(value.size() + 8);
  for (char c : value)
  {
    switch (c)
    {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out.push_back(c);
      break;
    }
  }
  return out;
}

void write_json_string_field(std::ostream &out,
                             const char *key,
                             std::string_view value,
                             bool trailing_comma)
{
  out << "  \"" << key << "\": \"" << escape_json_string(value) << "\"";
  out << (trailing_comma ? ",\n" : "\n");
}

void write_json_u64_field(std::ostream &out, const char *key, std::uint64_t value, bool trailing_comma)
{
  out << "  \"" << key << "\": " << value;
  out << (trailing_comma ? ",\n" : "\n");
}

void write_json_double_field(std::ostream &out, const char *key, double value, bool trailing_comma)
{
  out << "  \"" << key << "\": " << std::setprecision(17) << value;
  out << (trailing_comma ? ",\n" : "\n");
}

[[nodiscard]] std::string read_file_text(const std::filesystem::path &path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input)
    fail("failed to open file: " + path.string());
  std::ostringstream buffer;
  buffer << input.rdbuf();
  if (!input && !input.eof())
    fail("failed to read file: " + path.string());
  return buffer.str();
}

[[nodiscard]] std::size_t find_json_key(const std::string &text, std::string_view key)
{
  std::string needle = "\"" + std::string(key) + "\"";
  std::size_t pos = text.find(needle);
  if (pos == std::string::npos)
    fail("missing JSON key: " + std::string(key));
  pos = text.find(':', pos + needle.size());
  if (pos == std::string::npos)
    fail("missing JSON key separator: " + std::string(key));
  return pos + 1;
}

[[nodiscard]] std::string read_json_string(const std::string &text, std::string_view key)
{
  std::size_t pos = find_json_key(text, key);
  pos = text.find('"', pos);
  if (pos == std::string::npos)
    fail("missing JSON string value for key: " + std::string(key));
  ++pos;
  std::string out;
  bool escape = false;
  for (; pos < text.size(); ++pos)
  {
    char c = text[pos];
    if (escape)
    {
      switch (c)
      {
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      case '\\':
      case '"':
        out.push_back(c);
        break;
      default:
        out.push_back(c);
        break;
      }
      escape = false;
      continue;
    }
    if (c == '\\')
    {
      escape = true;
      continue;
    }
    if (c == '"')
      return out;
    out.push_back(c);
  }
  fail("unterminated JSON string for key: " + std::string(key));
  return out;
}

[[nodiscard]] std::string read_json_number_token(const std::string &text, std::string_view key)
{
  std::size_t pos = find_json_key(text, key);
  while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r'))
    ++pos;
  std::size_t end = pos;
  while (end < text.size() && text[end] != ',' && text[end] != '}' && text[end] != '\n' && text[end] != '\r')
    ++end;
  if (end == pos)
    fail("missing JSON numeric value for key: " + std::string(key));
  return text.substr(pos, end - pos);
}

[[nodiscard]] std::uint64_t parse_u64(std::string_view text)
{
  std::size_t consumed = 0;
  unsigned long long value = std::stoull(std::string(text), &consumed, 10);
  if (consumed != text.size())
    fail("invalid integer: " + std::string(text));
  return static_cast<std::uint64_t>(value);
}

[[nodiscard]] double parse_double(std::string_view text)
{
  std::size_t consumed = 0;
  double value = std::stod(std::string(text), &consumed);
  if (consumed != text.size())
    fail("invalid floating-point value: " + std::string(text));
  return value;
}

[[nodiscard]] std::vector<std::string> split_csv_line(const std::string &line)
{
  std::vector<std::string> out;
  std::string current;
  std::istringstream input(line);
  while (std::getline(input, current, ','))
    out.push_back(current);
  if (!line.empty() && line.back() == ',')
    out.emplace_back();
  return out;
}

[[nodiscard]] std::string trim(std::string_view text)
{
  std::size_t start = 0;
  while (start < text.size() && (text[start] == ' ' || text[start] == '\t' || text[start] == '\n' || text[start] == '\r'))
    ++start;
  std::size_t end = text.size();
  while (end > start && (text[end - 1] == ' ' || text[end - 1] == '\t' || text[end - 1] == '\n' || text[end - 1] == '\r'))
    --end;
  return std::string(text.substr(start, end - start));
}

[[nodiscard]] std::vector<std::uint8_t> read_bytes_file(const std::filesystem::path &path)
{
  std::uint64_t file_size = std::filesystem::file_size(path);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
  if (file_size == 0)
    return bytes;

  std::ifstream input(path, std::ios::binary);
  if (!input)
    fail("failed to open plane file: " + path.string());
  input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!input)
    fail("failed to read plane file: " + path.string());
  return bytes;
}

std::vector<double> parse_plane_basis_row(const std::vector<std::string> &fields, std::size_t start_index)
{
  std::vector<double> basis;
  if (start_index >= fields.size())
    return basis;
  basis.reserve(fields.size() - start_index);
  for (std::size_t index = start_index; index < fields.size(); ++index)
    basis.push_back(parse_double(trim(fields[index])));
  return basis;
}

} // namespace

std::size_t plane_width(std::size_t fixed_len_bits, std::size_t plane_index)
{
  if (fixed_len_bits == 0)
    return 0;

  std::size_t plane_count = (fixed_len_bits + 7U) / 8U;
  if (plane_index >= plane_count)
    fail("plane index out of range while deriving plane width");

  if (plane_index + 1U == plane_count)
  {
    std::size_t trailing = fixed_len_bits - 8U * (plane_count - 1U);
    return trailing == 0 ? 8U : trailing;
  }
  return 8U;
}

double segment_tail_upper_bound(const SegmentMeta &segment, std::size_t keep_planes)
{
  std::size_t fixed_len_bits = segment.get_fixed_len_bits();
  std::size_t active_planes = std::min<std::size_t>(segment.active_plane_count, segment.plane_basis.size());
  std::size_t keep = std::min<std::size_t>(keep_planes, active_planes);

  double tail = 0.0;
  for (std::size_t plane = keep; plane < active_planes; ++plane)
  {
    std::size_t width = plane_width(fixed_len_bits, plane);
    std::uint64_t max_byte = width >= 8U ? 255U : ((std::uint64_t{1} << width) - 1U);
    tail += static_cast<double>(max_byte) * segment.plane_basis[plane];
  }
  return tail;
}

std::filesystem::path manifest_path(const std::filesystem::path &root)
{
  return root / kManifestFileName;
}

std::filesystem::path summary_path(const std::filesystem::path &root)
{
  return root / kSummaryFileName;
}

std::filesystem::path segment_meta_path(const std::filesystem::path &root)
{
  return root / kSegmentMetaFileName;
}

std::filesystem::path plane_file_path(const std::filesystem::path &root, std::uint64_t plane_index)
{
  char file_name[32];
  std::snprintf(file_name, sizeof(file_name), "plane_%03llu.bin", static_cast<unsigned long long>(plane_index));
  return root / file_name;
}

void ensure_directory(const std::filesystem::path &path)
{
  std::filesystem::create_directories(path);
}

void write_manifest_json(const std::filesystem::path &path, const Manifest &manifest)
{
  std::ofstream out(path, std::ios::trunc);
  if (!out)
    fail("failed to open manifest path: " + path.string());

  out << "{\n";
  write_json_string_field(out, "format", manifest.format, true);
  write_json_string_field(out, "dataset", manifest.dataset, true);
  write_json_string_field(out, "source_path", manifest.source_path, true);
  write_json_u64_field(out, "segment_size", manifest.segment_size, true);
  write_json_u64_field(out, "value_count", manifest.value_count, true);
  write_json_u64_field(out, "segment_count", manifest.segment_count, true);
  write_json_u64_field(out, "max_plane_count", manifest.max_plane_count, true);
  write_json_u64_field(out, "segment_plane_count_min", manifest.segment_plane_count_min, true);
  write_json_u64_field(out, "segment_plane_count_max", manifest.segment_plane_count_max, true);
  write_json_string_field(out, "encoded_layout", manifest.encoded_layout, true);
  write_json_double_field(out, "exact_sum", manifest.exact_sum, false);
  out << "}\n";
}

Manifest read_manifest_json(const std::filesystem::path &path)
{
  std::string text = read_file_text(path);
  Manifest manifest;
  manifest.format = read_json_string(text, "format");
  manifest.dataset = read_json_string(text, "dataset");
  manifest.source_path = read_json_string(text, "source_path");
  manifest.segment_size = parse_u64(read_json_number_token(text, "segment_size"));
  manifest.value_count = parse_u64(read_json_number_token(text, "value_count"));
  manifest.segment_count = parse_u64(read_json_number_token(text, "segment_count"));
  manifest.max_plane_count = parse_u64(read_json_number_token(text, "max_plane_count"));
  manifest.segment_plane_count_min = parse_u64(read_json_number_token(text, "segment_plane_count_min"));
  manifest.segment_plane_count_max = parse_u64(read_json_number_token(text, "segment_plane_count_max"));
  manifest.encoded_layout = read_json_string(text, "encoded_layout");
  manifest.exact_sum = parse_double(read_json_number_token(text, "exact_sum"));

  if (manifest.format != kFormatName)
    fail("unexpected manifest format: " + manifest.format);
  return manifest;
}

void write_summary_json(const std::filesystem::path &path,
                        const Manifest &manifest,
                        const std::vector<SegmentMeta> &segments)
{
  std::map<std::uint64_t, std::uint64_t> histogram;
  std::uint64_t raw_fractional_bits_min = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t raw_fractional_bits_max = 0;
  std::uint64_t precision_cap_bits_min = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t precision_cap_bits_max = 0;
  std::uint64_t effective_fractional_bits_min = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t effective_fractional_bits_max = 0;
  for (const SegmentMeta &segment : segments)
  {
    histogram[segment.active_plane_count] += 1;
    raw_fractional_bits_min = std::min<std::uint64_t>(raw_fractional_bits_min, segment.raw_fractional_bits);
    raw_fractional_bits_max = std::max<std::uint64_t>(raw_fractional_bits_max, segment.raw_fractional_bits);
    precision_cap_bits_min = std::min<std::uint64_t>(precision_cap_bits_min, segment.precision_cap_bits);
    precision_cap_bits_max = std::max<std::uint64_t>(precision_cap_bits_max, segment.precision_cap_bits);
    effective_fractional_bits_min =
        std::min<std::uint64_t>(effective_fractional_bits_min, segment.effective_fractional_bits);
    effective_fractional_bits_max =
        std::max<std::uint64_t>(effective_fractional_bits_max, segment.effective_fractional_bits);
  }

  if (segments.empty())
  {
    raw_fractional_bits_min = 0;
    precision_cap_bits_min = 0;
    effective_fractional_bits_min = 0;
  }

  std::ostringstream histogram_text;
  bool first = true;
  for (const auto &[plane_count, count] : histogram)
  {
    if (!first)
      histogram_text << ", ";
    first = false;
    histogram_text << plane_count << ':' << count;
  }

  std::ofstream out(path, std::ios::trunc);
  if (!out)
    fail("failed to open summary path: " + path.string());

  out << "{\n";
  write_json_string_field(out, "format", manifest.format, true);
  write_json_string_field(out, "dataset", manifest.dataset, true);
  write_json_u64_field(out, "segment_count", manifest.segment_count, true);
  write_json_u64_field(out, "max_plane_count", manifest.max_plane_count, true);
  write_json_u64_field(out, "segment_plane_count_min", manifest.segment_plane_count_min, true);
  write_json_u64_field(out, "segment_plane_count_max", manifest.segment_plane_count_max, true);
  write_json_u64_field(out, "raw_fractional_bits_min", raw_fractional_bits_min, true);
  write_json_u64_field(out, "raw_fractional_bits_max", raw_fractional_bits_max, true);
  write_json_u64_field(out, "precision_cap_bits_min", precision_cap_bits_min, true);
  write_json_u64_field(out, "precision_cap_bits_max", precision_cap_bits_max, true);
  write_json_u64_field(out, "effective_fractional_bits_min", effective_fractional_bits_min, true);
  write_json_u64_field(out, "effective_fractional_bits_max", effective_fractional_bits_max, true);
  write_json_string_field(out, "plane_count_histogram", histogram_text.str(), true);
  write_json_double_field(out, "exact_sum", manifest.exact_sum, false);
  out << "}\n";
}

void write_segment_meta_csv(const std::filesystem::path &path,
                            const Manifest &manifest,
                            const std::vector<SegmentMeta> &segments)
{
  std::ofstream out(path, std::ios::trunc);
  if (!out)
    fail("failed to open segment meta path: " + path.string());

  out << "segment_index,row_offset,row_count,active_plane_count,fractional_bits,integer_offset_bits,integer_base_hex,segment_base"
      << ",raw_fractional_bits,precision_cap_bits,effective_fractional_bits";
  for (std::uint64_t plane = 0; plane < manifest.max_plane_count; ++plane)
    out << ",plane_basis_" << plane;
  out << '\n';

  out << std::setprecision(17);
  for (const SegmentMeta &segment : segments)
  {
    out << segment.segment_index << ','
        << segment.row_offset << ','
        << segment.row_count << ','
        << segment.active_plane_count << ','
        << segment.fractional_bits << ','
        << segment.integer_offset_bits << ','
        << bytes_to_hex(segment.integer_base_le) << ','
        << segment.segment_base << ','
        << segment.raw_fractional_bits << ','
        << segment.precision_cap_bits << ','
        << segment.effective_fractional_bits;
    for (std::uint64_t plane = 0; plane < manifest.max_plane_count; ++plane)
    {
      double basis = plane < segment.plane_basis.size() ? segment.plane_basis[static_cast<std::size_t>(plane)] : 0.0;
      out << ',' << basis;
    }
    out << '\n';
  }
}

std::vector<SegmentMeta> read_segment_meta_csv(const std::filesystem::path &path,
                                               std::uint64_t max_plane_count)
{
  std::ifstream input(path);
  if (!input)
    fail("failed to open segment meta csv: " + path.string());

  std::string header_line;
  if (!std::getline(input, header_line))
    fail("empty segment meta csv: " + path.string());

  std::vector<std::string> header = split_csv_line(header_line);
  if (header.size() < 8)
    fail("segment meta csv header is too short: " + path.string());
  std::unordered_map<std::string, std::size_t> header_index;
  for (std::size_t index = 0; index < header.size(); ++index)
    header_index[trim(header[index])] = index;

  auto require_column = [&](const char *name) -> std::size_t
  {
    auto it = header_index.find(name);
    if (it == header_index.end())
      fail("segment meta csv missing required column: " + std::string(name));
    return it->second;
  };
  auto optional_column = [&](const char *name) -> std::size_t
  {
    auto it = header_index.find(name);
    if (it == header_index.end())
      return std::numeric_limits<std::size_t>::max();
    return it->second;
  };

  std::size_t idx_segment_index = require_column("segment_index");
  std::size_t idx_row_offset = require_column("row_offset");
  std::size_t idx_row_count = require_column("row_count");
  std::size_t idx_active_plane_count = require_column("active_plane_count");
  std::size_t idx_fractional_bits = require_column("fractional_bits");
  std::size_t idx_integer_offset_bits = require_column("integer_offset_bits");
  std::size_t idx_integer_base_hex = require_column("integer_base_hex");
  std::size_t idx_segment_base = require_column("segment_base");
  std::size_t idx_raw_fractional_bits = optional_column("raw_fractional_bits");
  std::size_t idx_precision_cap_bits = optional_column("precision_cap_bits");
  std::size_t idx_effective_fractional_bits = optional_column("effective_fractional_bits");

  std::size_t plane_basis_start = std::numeric_limits<std::size_t>::max();
  for (std::size_t index = 0; index < header.size(); ++index)
  {
    std::string name = trim(header[index]);
    if (name.size() >= 12 && name.compare(0, 12, "plane_basis_") == 0)
    {
      plane_basis_start = std::min(plane_basis_start, index);
    }
  }
  if (plane_basis_start == std::numeric_limits<std::size_t>::max())
    fail("segment meta csv missing plane_basis_* columns: " + path.string());

  std::vector<SegmentMeta> segments;
  std::string line;
  while (std::getline(input, line))
  {
    if (trim(line).empty())
      continue;
    std::vector<std::string> fields = split_csv_line(line);
    if (fields.size() < 8)
      fail("segment meta csv row is too short: " + line);

    auto field_at = [&](std::size_t index, const char *name) -> std::string
    {
      if (index >= fields.size())
        fail("segment meta csv row missing column '" + std::string(name) + "': " + line);
      return trim(fields[index]);
    };

    SegmentMeta segment;
    segment.segment_index = parse_u64(field_at(idx_segment_index, "segment_index"));
    segment.row_offset = parse_u64(field_at(idx_row_offset, "row_offset"));
    segment.row_count = parse_u64(field_at(idx_row_count, "row_count"));
    segment.active_plane_count = static_cast<std::uint32_t>(parse_u64(field_at(idx_active_plane_count, "active_plane_count")));
    segment.fractional_bits = static_cast<std::uint32_t>(parse_u64(field_at(idx_fractional_bits, "fractional_bits")));
    segment.integer_offset_bits = static_cast<std::uint32_t>(parse_u64(field_at(idx_integer_offset_bits, "integer_offset_bits")));
    segment.integer_base_le = parse_integer_base_hex(field_at(idx_integer_base_hex, "integer_base_hex"), &segment.integer_base_is_v2);
    segment.segment_base = parse_double(field_at(idx_segment_base, "segment_base"));
    segment.raw_fractional_bits = idx_raw_fractional_bits == std::numeric_limits<std::size_t>::max()
                                      ? segment.fractional_bits
                                      : static_cast<std::uint32_t>(parse_u64(field_at(idx_raw_fractional_bits, "raw_fractional_bits")));
    segment.precision_cap_bits = idx_precision_cap_bits == std::numeric_limits<std::size_t>::max()
                                     ? segment.fractional_bits
                                     : static_cast<std::uint32_t>(parse_u64(field_at(idx_precision_cap_bits, "precision_cap_bits")));
    segment.effective_fractional_bits = idx_effective_fractional_bits == std::numeric_limits<std::size_t>::max()
                                            ? segment.fractional_bits
                                            : static_cast<std::uint32_t>(parse_u64(field_at(idx_effective_fractional_bits, "effective_fractional_bits")));
    segment.fractional_bits = segment.effective_fractional_bits;
    segment.plane_basis = parse_plane_basis_row(fields, plane_basis_start);
    if (segment.plane_basis.size() < max_plane_count)
      segment.plane_basis.resize(static_cast<std::size_t>(max_plane_count), 0.0);
    if (segment.plane_basis.size() != max_plane_count)
      fail("segment meta plane basis width mismatch: " + path.string());
    if (segment.active_plane_count > max_plane_count)
      fail("segment active_plane_count exceeds manifest max_plane_count: " + path.string());
    segments.push_back(std::move(segment));
  }

  return segments;
}

void write_plane_files(const std::filesystem::path &root,
                       const std::vector<std::vector<std::uint8_t>> &planes)
{
  ensure_directory(root);
  for (std::size_t plane = 0; plane < planes.size(); ++plane)
  {
    std::filesystem::path path = plane_file_path(root, static_cast<std::uint64_t>(plane));
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
      fail("failed to open plane file for writing: " + path.string());
    if (!planes[plane].empty())
    {
      out.write(reinterpret_cast<const char *>(planes[plane].data()), static_cast<std::streamsize>(planes[plane].size()));
      if (!out)
        fail("failed to write plane file: " + path.string());
    }
  }
}

std::vector<std::vector<std::uint8_t>> read_plane_files(const std::filesystem::path &root,
                                                        std::uint64_t plane_count,
                                                        std::uint64_t value_count)
{
  std::vector<std::vector<std::uint8_t>> planes(static_cast<std::size_t>(plane_count));
  for (std::uint64_t plane = 0; plane < plane_count; ++plane)
  {
    std::filesystem::path path = plane_file_path(root, plane);
    std::uint64_t file_size = std::filesystem::file_size(path);
    if (file_size != value_count)
      fail("plane file size mismatch: " + path.string());
    planes[static_cast<std::size_t>(plane)] = read_bytes_file(path);
  }
  return planes;
}

Dataset load_dataset(const std::filesystem::path &root)
{
  Dataset dataset;
  dataset.manifest = read_manifest_json(manifest_path(root));
  dataset.segments = read_segment_meta_csv(segment_meta_path(root), dataset.manifest.max_plane_count);
  dataset.planes = read_plane_files(root, dataset.manifest.max_plane_count, dataset.manifest.value_count);

  if (dataset.segments.size() != dataset.manifest.segment_count)
    fail("segment meta row count does not match manifest.segment_count");

  std::uint64_t row_total = 0;
  for (std::size_t index = 0; index < dataset.segments.size(); ++index)
  {
    const SegmentMeta &segment = dataset.segments[index];
    if (segment.segment_index != index)
      fail("segment meta segment_index is not contiguous");
    if (segment.row_offset != row_total)
      fail("segment meta row_offset is not contiguous");
    if (segment.row_count == 0)
      fail("segment meta row_count must be positive");
    if (segment.active_plane_count > dataset.manifest.max_plane_count)
      fail("segment meta active_plane_count exceeds manifest.max_plane_count");
    row_total += segment.row_count;
  }
  if (row_total != dataset.manifest.value_count)
    fail("segment meta row count does not match manifest.value_count");

  if (dataset.manifest.encoded_layout != kEncodedLayoutName)
    fail("unexpected encoded layout: " + dataset.manifest.encoded_layout);

  return dataset;
}

double cpu_approximate_sum(const Dataset &dataset, int refinement_depth)
{
  if (refinement_depth < 0)
    fail("refinement_depth must be non-negative");

  std::size_t depth = static_cast<std::size_t>(refinement_depth);
  double total = 0.0;
  for (const SegmentMeta &segment : dataset.segments)
  {
    std::size_t plane_limit = std::min<std::size_t>({depth + 1, segment.plane_basis.size(), segment.active_plane_count});
    total += static_cast<double>(segment.row_count) * segment.segment_base;
    for (std::size_t plane = 0; plane < plane_limit; ++plane)
    {
      const std::vector<std::uint8_t> &plane_bytes = dataset.planes[plane];
      std::uint64_t plane_sum = 0;
      for (std::uint64_t row = 0; row < segment.row_count; ++row)
        plane_sum += static_cast<std::uint64_t>(plane_bytes[static_cast<std::size_t>(segment.row_offset + row)]);
      total += static_cast<double>(plane_sum) * segment.plane_basis[plane];
    }
  }
  return total;
}

ExactExtrema cpu_exact_extrema(const Dataset &dataset)
{
  return {cpu_progressive_extrema_summary(dataset, static_cast<int>(dataset.manifest.max_plane_count) - 1).min.lower,
          cpu_progressive_extrema_summary(dataset, static_cast<int>(dataset.manifest.max_plane_count) - 1).max.upper,
          0,
          0};
}

ProgressiveExtremaSummary cpu_progressive_extrema_summary(const Dataset &dataset, int refinement_depth)
{
  if (refinement_depth < 0)
    fail("refinement_depth must be non-negative");

  std::size_t keep_planes = static_cast<std::size_t>(refinement_depth) + 1U;
  ProgressiveExtremaSummary summary;
  summary.min.lower = std::numeric_limits<double>::infinity();
  summary.min.upper = std::numeric_limits<double>::infinity();
  summary.max.lower = -std::numeric_limits<double>::infinity();
  summary.max.upper = -std::numeric_limits<double>::infinity();

  for (const SegmentMeta &segment : dataset.segments)
  {
    std::size_t active_planes = std::min<std::size_t>(segment.active_plane_count, segment.plane_basis.size());
    std::size_t plane_limit = std::min<std::size_t>(keep_planes, active_planes);
    double tail_upper = segment_tail_upper_bound(segment, keep_planes);

    double segment_min_prefix = std::numeric_limits<double>::infinity();
    double segment_max_prefix = -std::numeric_limits<double>::infinity();
    for (std::uint64_t row = 0; row < segment.row_count; ++row)
    {
      double prefix = segment.segment_base;
      std::size_t row_index = static_cast<std::size_t>(segment.row_offset + row);
      for (std::size_t plane = 0; plane < plane_limit; ++plane)
        prefix += static_cast<double>(dataset.planes[plane][row_index]) * segment.plane_basis[plane];

      if (prefix < segment_min_prefix)
        segment_min_prefix = prefix;
      if (prefix > segment_max_prefix)
        segment_max_prefix = prefix;
    }

    if (segment_min_prefix < summary.min.lower)
      summary.min.lower = segment_min_prefix;
    if (segment_min_prefix + tail_upper < summary.min.upper)
      summary.min.upper = segment_min_prefix + tail_upper;
    if (segment_max_prefix > summary.max.lower)
      summary.max.lower = segment_max_prefix;
    if (segment_max_prefix + tail_upper > summary.max.upper)
      summary.max.upper = segment_max_prefix + tail_upper;
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

ProgressiveExtremaBounds cpu_progressive_extrema_bounds(const Dataset &dataset,
                                                        int refinement_depth,
                                                        ExtremumOp op)
{
  ProgressiveExtremaSummary summary = cpu_progressive_extrema_summary(dataset, refinement_depth);
  return op == ExtremumOp::Min ? summary.min : summary.max;
}

CandidateCounts cpu_candidate_counts(const Dataset &dataset, int refinement_depth)
{
  if (refinement_depth < 0)
    fail("refinement_depth must be non-negative");

  std::size_t keep_planes = static_cast<std::size_t>(refinement_depth) + 1U;
  CandidateCounts counts;

  std::vector<double> prefixes; // reused per segment

  for (const SegmentMeta &segment : dataset.segments)
  {
    std::size_t active_planes = std::min<std::size_t>(segment.active_plane_count, segment.plane_basis.size());
    std::size_t plane_limit = std::min<std::size_t>(keep_planes, active_planes);
    double tail_upper = segment_tail_upper_bound(segment, keep_planes);

    prefixes.clear();
    prefixes.reserve(static_cast<std::size_t>(segment.row_count));

    double segment_min_prefix = std::numeric_limits<double>::infinity();
    double segment_max_prefix = -std::numeric_limits<double>::infinity();

    for (std::uint64_t row = 0; row < segment.row_count; ++row)
    {
      double prefix = segment.segment_base;
      std::size_t row_index = static_cast<std::size_t>(segment.row_offset + row);
      for (std::size_t plane = 0; plane < plane_limit; ++plane)
        prefix += static_cast<double>(dataset.planes[plane][row_index]) * segment.plane_basis[plane];

      if (prefix < segment_min_prefix)
        segment_min_prefix = prefix;
      if (prefix > segment_max_prefix)
        segment_max_prefix = prefix;
      prefixes.push_back(prefix);
    }

    double min_ceiling = segment_min_prefix + tail_upper;
    double max_floor = segment_max_prefix - tail_upper;
    uint64_t seg_min_cand = 0;
    uint64_t seg_max_cand = 0;
    for (double prefix : prefixes)
    {
      if (prefix <= min_ceiling)
        seg_min_cand++;
      if (prefix >= max_floor)
        seg_max_cand++;
    }

    counts.min_candidates += seg_min_cand;
    counts.max_candidates += seg_max_cand;
    counts.total_rows += segment.row_count;
  }

  return counts;
}

std::string bytes_to_hex(const std::vector<std::uint8_t> &bytes)
{
  static const char kHexDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (std::uint8_t byte : bytes)
  {
    out.push_back(kHexDigits[(byte >> 4U) & 0x0fU]);
    out.push_back(kHexDigits[byte & 0x0fU]);
  }
  return out;
}

std::vector<std::uint8_t> hex_to_bytes(std::string_view hex)
{
  if (hex.size() % 2 != 0)
    fail("hex string must have even length");
  auto decode_digit = [](char c) -> int
  {
    if (c >= '0' && c <= '9')
      return c - '0';
    if (c >= 'a' && c <= 'f')
      return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F')
      return 10 + (c - 'A');
    return -1;
  };

  std::vector<std::uint8_t> bytes;
  bytes.reserve(hex.size() / 2);
  for (std::size_t index = 0; index < hex.size(); index += 2)
  {
    int hi = decode_digit(hex[index]);
    int lo = decode_digit(hex[index + 1]);
    if (hi < 0 || lo < 0)
      fail("invalid hex string");
    bytes.push_back(static_cast<std::uint8_t>((hi << 4) | lo));
  }
  return bytes;
}

[[nodiscard]] std::vector<std::uint8_t> parse_integer_base_hex(std::string_view text, bool *out_is_v2)
{
  if (text.size() >= 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X'))
  {
    if (out_is_v2)
      *out_is_v2 = true;

    text = text.substr(2);
    if (text.empty())
      return {};

    std::string padded;
    if (text.size() % 2 != 0)
    {
      padded = "0";
      padded += text;
      text = padded;
    }

    auto decode_digit = [](char c) -> int
    {
      if (c >= '0' && c <= '9')
        return c - '0';
      if (c >= 'a' && c <= 'f')
        return 10 + (c - 'a');
      if (c >= 'A' && c <= 'F')
        return 10 + (c - 'A');
      return -1;
    };

    std::vector<std::uint8_t> bytes;
    bytes.reserve(text.size() / 2);
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(text.size()) - 2; i >= 0; i -= 2)
    {
      int hi = decode_digit(text[i]);
      int lo = decode_digit(text[i + 1]);
      if (hi < 0 || lo < 0)
        fail("invalid hex digit in integer_base_hex");
      bytes.push_back(static_cast<std::uint8_t>((hi << 4) | lo));
    }

    // v2 integer_base is a signed i64 fixed-point code base.
    // It must occupy exactly 8 bytes (uint64 width) for
    // le_bytes_to_u64() to produce the correct value.
    if (bytes.size() > 8)
      fail("integer_base_hex v2: more than 8 bytes (exceeds int64_t width)");
    if (bytes.size() < 8)
      bytes.resize(8, 0);   // zero-pad to full uint64 width.
    //
    // NOTE: current v2 synthetic artifacts are non-negative (bit 63 = 0),
    // so zero-padding is correct.  Future negative base values will need
    // a sign-extension resize policy (0xff padding for negative) signalled
    // by an explicit schema field.
    return bytes;
  }

  if (out_is_v2)
    *out_is_v2 = false;
  return hex_to_bytes(text);
}

std::vector<double> plane_basis_for_segment(std::uint32_t fractional_bits,
                                            std::uint32_t integer_offset_bits,
                                            std::uint64_t max_plane_count)
{
  std::size_t total_bits = static_cast<std::size_t>(fractional_bits) + static_cast<std::size_t>(integer_offset_bits);
  std::size_t plane_count = total_bits == 0 ? 1U : (total_bits + 7U) / 8U;
  std::vector<double> basis(static_cast<std::size_t>(max_plane_count), 0.0);
  for (std::size_t plane = 0; plane < std::min<std::size_t>(plane_count, basis.size()); ++plane)
  {
    if (total_bits == 0)
    {
      basis[plane] = 0.0;
      continue;
    }
    std::size_t width = 8U;
    if (plane + 1U == plane_count)
    {
      std::size_t trailing = total_bits - 8U * (plane_count - 1U);
      width = trailing == 0 ? 8U : trailing;
    }
    std::size_t start_bit = total_bits - 8U * plane - width;
    basis[plane] = std::ldexp(1.0, static_cast<int>(start_bit) - static_cast<int>(fractional_bits));
  }
  return basis;
}

} // namespace exp3_real
