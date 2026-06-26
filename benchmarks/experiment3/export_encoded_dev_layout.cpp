#include "exp3_real_data_layout.hpp"

#include "../../buff_encoder/buff_codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

struct Options
{
  std::filesystem::path input_path;
  std::filesystem::path output_root;
  std::string dataset_name;
  std::uint64_t segment_size = 4096;
  std::uint64_t max_values = 0;
  std::optional<std::uint32_t> precision_power;
};

[[noreturn]] void fail(const std::string &message)
{
  throw std::runtime_error(message);
}

void print_usage(const char *argv0)
{
  std::cerr
      << "Usage:\n"
      << "  " << argv0
      << " --input FILE --output-root DIR [--dataset NAME] [--segment-size N] [--max-values N] [--precision-power N]\n";
}

std::uint64_t parse_u64(const std::string &text)
{
  std::size_t consumed = 0;
  unsigned long long value = std::stoull(text, &consumed, 10);
  if (consumed != text.size())
    fail("invalid integer: " + text);
  return static_cast<std::uint64_t>(value);
}

Options parse_args(int argc, char **argv)
{
  Options opt;
  for (int index = 1; index < argc; ++index)
  {
    std::string_view arg = argv[index];
    if (arg == "--input" && index + 1 < argc)
    {
      opt.input_path = argv[++index];
    }
    else if (arg == "--output-root" && index + 1 < argc)
    {
      opt.output_root = argv[++index];
    }
    else if (arg == "--dataset" && index + 1 < argc)
    {
      opt.dataset_name = argv[++index];
    }
    else if (arg == "--segment-size" && index + 1 < argc)
    {
      opt.segment_size = parse_u64(argv[++index]);
    }
    else if (arg == "--max-values" && index + 1 < argc)
    {
      opt.max_values = parse_u64(argv[++index]);
    }
    else if (arg == "--precision-power" && index + 1 < argc)
    {
      std::uint64_t parsed = parse_u64(argv[++index]);
      if (parsed > std::numeric_limits<std::uint32_t>::max())
        fail("--precision-power is out of range for uint32");
      opt.precision_power = static_cast<std::uint32_t>(parsed);
    }
    else if (arg == "--help" || arg == "-h")
    {
      print_usage(argv[0]);
      std::exit(0);
    }
    else
    {
      fail("unknown option: " + std::string(arg));
    }
  }

  if (opt.input_path.empty())
    fail("missing --input");
  if (opt.output_root.empty())
    fail("missing --output-root");
  if (opt.segment_size == 0)
    fail("--segment-size must be greater than zero");
  if (opt.dataset_name.empty())
  {
    opt.dataset_name = opt.input_path.filename().string();
    std::string suffix = ".f64le.bin";
    if (opt.dataset_name.size() > suffix.size() &&
        opt.dataset_name.compare(opt.dataset_name.size() - suffix.size(), suffix.size(), suffix) == 0)
    {
      opt.dataset_name.resize(opt.dataset_name.size() - suffix.size());
    }
  }
  return opt;
}

std::uint64_t file_value_count(const std::filesystem::path &input_path)
{
  std::uint64_t file_size = std::filesystem::file_size(input_path);
  if (file_size % sizeof(double) != 0)
    fail("input file size is not aligned to FP64 rows: " + input_path.string());
  return file_size / sizeof(double);
}

double scaled_le_bytes_to_double(const std::vector<std::uint8_t> &bytes, int scale_exponent)
{
  double value = 0.0;
  for (std::size_t index = 0; index < bytes.size(); ++index)
  {
    if (bytes[index] == 0)
      continue;
    value += std::ldexp(static_cast<double>(bytes[index]),
                        static_cast<int>(index * 8U) + scale_exponent);
  }
  return value;
}

void analyze_segments(const Options &opt,
                      exp3_real::Manifest &manifest,
                      std::vector<exp3_real::SegmentMeta> &segments)
{
  manifest.dataset = opt.dataset_name;
  manifest.source_path = opt.input_path.string();
  manifest.segment_size = opt.segment_size;
  manifest.value_count = file_value_count(opt.input_path);
  if (opt.max_values != 0)
    manifest.value_count = std::min(manifest.value_count, opt.max_values);

  std::ifstream input(opt.input_path, std::ios::binary);
  if (!input)
    fail("failed to open input file: " + opt.input_path.string());
  buff::EncodeConfig encode_config{
      .precision_power = opt.precision_power,
  };

  std::vector<double> buffer(static_cast<std::size_t>(opt.segment_size));
  std::uint64_t remaining = manifest.value_count;
  std::uint64_t row_offset = 0;
  std::uint64_t segment_index = 0;
  long double exact_sum = 0.0L;

  manifest.segment_plane_count_min = std::numeric_limits<std::uint64_t>::max();
  while (remaining > 0)
  {
    std::uint64_t current = std::min(remaining, opt.segment_size);
    input.read(reinterpret_cast<char *>(buffer.data()),
               static_cast<std::streamsize>(current * sizeof(double)));
    if (!input)
      fail("failed to read input file: " + opt.input_path.string());

    for (std::uint64_t row = 0; row < current; ++row)
      exact_sum += static_cast<long double>(buffer[static_cast<std::size_t>(row)]);

    buff::EncodedSegment encoded =
        buff::encode_segment(std::span<const double>(buffer.data(), static_cast<std::size_t>(current)), encode_config);

    exp3_real::SegmentMeta meta;
    meta.segment_index = segment_index;
    meta.row_offset = row_offset;
    meta.row_count = current;
    meta.active_plane_count = static_cast<std::uint32_t>(buff::segment_plane_count(encoded));
    meta.fractional_bits = encoded.fractional_bits;
    meta.raw_fractional_bits = encoded.fractional_bits;
    meta.precision_cap_bits = encoded.fractional_bits;
    meta.effective_fractional_bits = encoded.fractional_bits;
    meta.integer_offset_bits = encoded.integer_offset_bits;
    meta.integer_base_le = encoded.integer_base_le;
    meta.segment_base = scaled_le_bytes_to_double(meta.integer_base_le,
                                                  -static_cast<int>(meta.fractional_bits));

    manifest.segment_plane_count_min = std::min(manifest.segment_plane_count_min,
                                                static_cast<std::uint64_t>(meta.active_plane_count));
    manifest.segment_plane_count_max = std::max(manifest.segment_plane_count_max,
                                                static_cast<std::uint64_t>(meta.active_plane_count));
    segments.push_back(std::move(meta));

    remaining -= current;
    row_offset += current;
    ++segment_index;
  }

  manifest.segment_count = segment_index;
  manifest.max_plane_count = manifest.segment_plane_count_max;
  if (manifest.segment_count == 0)
    manifest.segment_plane_count_min = 0;
  manifest.exact_sum = static_cast<double>(exact_sum);

  if (manifest.max_plane_count > static_cast<std::uint64_t>(exp3_real::kMaxRuntimePlanes))
  {
    fail("dataset requires more planes than current runtime limit: max_plane_count=" +
         std::to_string(manifest.max_plane_count));
  }

  for (exp3_real::SegmentMeta &meta : segments)
  {
    meta.plane_basis = exp3_real::plane_basis_for_segment(meta.fractional_bits,
                                                          meta.integer_offset_bits,
                                                          manifest.max_plane_count);
  }
}

void export_plane_files(const Options &opt,
                        const exp3_real::Manifest &manifest,
                        const std::filesystem::path &dataset_root)
{
  std::vector<std::uint8_t> zeros(static_cast<std::size_t>(opt.segment_size), 0);
  std::vector<double> buffer(static_cast<std::size_t>(opt.segment_size));
  buff::EncodeConfig encode_config{
      .precision_power = opt.precision_power,
  };
  for (std::uint64_t plane = 0; plane < manifest.max_plane_count; ++plane)
  {
    std::filesystem::path plane_path = exp3_real::plane_file_path(dataset_root, plane);
    std::ofstream out(plane_path, std::ios::binary | std::ios::trunc);
    if (!out)
      fail("failed to open plane file for writing: " + plane_path.string());

    std::ifstream input(opt.input_path, std::ios::binary);
    if (!input)
      fail("failed to reopen input file: " + opt.input_path.string());

    // Serialize one plane per pass so each output file is dense and easy to verify.
    std::uint64_t remaining = manifest.value_count;
    while (remaining > 0)
    {
      std::uint64_t current = std::min(remaining, opt.segment_size);
      input.read(reinterpret_cast<char *>(buffer.data()),
                 static_cast<std::streamsize>(current * sizeof(double)));
      if (!input)
        fail("failed to read input file during plane export: " + opt.input_path.string());

      buff::EncodedSegment encoded =
          buff::encode_segment(std::span<const double>(buffer.data(), static_cast<std::size_t>(current)), encode_config);
      std::size_t plane_count = buff::segment_plane_count(encoded);
      if (plane < plane_count)
      {
        const std::vector<std::uint8_t> &plane_bytes = encoded.byte_planes[static_cast<std::size_t>(plane)];
        out.write(reinterpret_cast<const char *>(plane_bytes.data()),
                  static_cast<std::streamsize>(plane_bytes.size()));
      }
      else
      {
        out.write(reinterpret_cast<const char *>(zeros.data()),
                  static_cast<std::streamsize>(current));
      }
      if (!out)
        fail("failed to write plane file: " + plane_path.string());

      remaining -= current;
    }

    out.close();
    if (!out)
      fail("failed to close plane file: " + plane_path.string());

    if (std::filesystem::file_size(plane_path) != manifest.value_count)
    {
      fail("plane file size mismatch after export: " + plane_path.string());
    }
  }
}

} // namespace

int main(int argc, char **argv)
{
  try
  {
    Options opt = parse_args(argc, argv);
    std::filesystem::path dataset_root = opt.output_root / opt.dataset_name;
    exp3_real::ensure_directory(dataset_root);

    exp3_real::Manifest manifest;
    std::vector<exp3_real::SegmentMeta> segments;
    analyze_segments(opt, manifest, segments);

    export_plane_files(opt, manifest, dataset_root);
    exp3_real::write_manifest_json(exp3_real::manifest_path(dataset_root), manifest);
    exp3_real::write_summary_json(exp3_real::summary_path(dataset_root), manifest, segments);
    exp3_real::write_segment_meta_csv(exp3_real::segment_meta_path(dataset_root), manifest, segments);

    std::cout << "exported dataset=" << manifest.dataset
              << " values=" << manifest.value_count
              << " segments=" << manifest.segment_count
              << " max_plane_count=" << manifest.max_plane_count
              << " output_root=" << dataset_root
               << " mode=" << (opt.precision_power.has_value() ? "bounded_precision" : "exact_dyadic");
    if (opt.precision_power.has_value())
      std::cout << " precision_power=" << *opt.precision_power;
    std::cout << '\n';
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "export_encoded_dev_layout: " << error.what() << '\n';
    return 1;
  }
}
