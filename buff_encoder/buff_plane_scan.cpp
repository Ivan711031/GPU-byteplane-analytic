#include "buff_codec.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::filesystem::path input_dir;
    std::uint64_t segment_size = 4096;
    std::uint64_t max_values = 0;
};

struct FileStats {
    std::filesystem::path path;
    std::uint64_t total_values = 0;
    std::uint64_t scanned_values = 0;
    std::uint64_t segment_count = 0;
    std::uint64_t segments_gt8 = 0;
    std::uint64_t first_gt8_segment = 0;
    std::size_t first_gt8_plane_count = 0;
    std::size_t max_plane_count = 0;
    std::map<std::size_t, std::uint64_t> histogram;
};

void print_usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " --input-dir DIR [--segment-size N] [--max-values N]\n";
}

std::uint64_t parse_u64(const std::string& text) {
    std::size_t consumed = 0;
    unsigned long long value = std::stoull(text, &consumed, 10);
    if (consumed != text.size()) {
        throw std::runtime_error("invalid integer: " + text);
    }
    return static_cast<std::uint64_t>(value);
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int index = 1; index < argc; ++index) {
        std::string_view arg = argv[index];
        if (arg == "--input-dir" && index + 1 < argc) {
            opt.input_dir = argv[++index];
        } else if (arg == "--segment-size" && index + 1 < argc) {
            opt.segment_size = parse_u64(argv[++index]);
        } else if (arg == "--max-values" && index + 1 < argc) {
            opt.max_values = parse_u64(argv[++index]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + std::string(arg));
        }
    }

    if (opt.input_dir.empty()) {
        throw std::runtime_error("missing --input-dir");
    }
    if (opt.segment_size == 0) {
        throw std::runtime_error("--segment-size must be greater than zero");
    }
    return opt;
}

std::uint64_t file_value_count(const std::filesystem::path& input_path) {
    std::uint64_t file_size = std::filesystem::file_size(input_path);
    if (file_size % sizeof(double) != 0) {
        throw std::runtime_error("input file size is not aligned to FP64 rows: " + input_path.string());
    }
    return file_size / sizeof(double);
}

std::vector<std::filesystem::path> collect_input_files(const std::filesystem::path& input_dir) {
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() == ".bin" && entry.path().string().find(".f64le.bin") != std::string::npos) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::runtime_error("no .f64le.bin files found under: " + input_dir.string());
    }
    return files;
}

FileStats scan_file(const std::filesystem::path& input_path, const Options& opt) {
    FileStats stats;
    stats.path = input_path;
    stats.total_values = file_value_count(input_path);
    stats.scanned_values = stats.total_values;
    if (opt.max_values != 0) {
        stats.scanned_values = std::min(stats.scanned_values, opt.max_values);
    }
    stats.segment_count = (stats.scanned_values + opt.segment_size - 1) / opt.segment_size;

    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open input file: " + input_path.string());
    }

    std::vector<double> buffer(static_cast<std::size_t>(opt.segment_size));
    std::uint64_t remaining = stats.scanned_values;
    std::uint64_t segment_index = 0;
    while (remaining > 0) {
        std::uint64_t current = std::min(remaining, opt.segment_size);
        input.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(current * sizeof(double)));
        if (!input) {
            throw std::runtime_error("failed to read input FP64 payload: " + input_path.string());
        }

        buff::EncodedSegment segment =
            buff::encode_segment(std::span<const double>(buffer.data(), static_cast<std::size_t>(current)));
        std::size_t plane_count = buff::segment_plane_count(segment);
        stats.histogram[plane_count] += 1;
        stats.max_plane_count = std::max(stats.max_plane_count, plane_count);
        if (plane_count > 8) {
            stats.segments_gt8 += 1;
            if (stats.segments_gt8 == 1) {
                stats.first_gt8_segment = segment_index;
                stats.first_gt8_plane_count = plane_count;
            }
        }

        remaining -= current;
        ++segment_index;
    }

    return stats;
}

void print_histogram(const std::map<std::size_t, std::uint64_t>& histogram) {
    bool first = true;
    for (const auto& [plane_count, count] : histogram) {
        if (!first) {
            std::cout << ", ";
        }
        first = false;
        std::cout << plane_count << ":" << count;
    }
    if (first) {
        std::cout << "(empty)";
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);
        std::vector<std::filesystem::path> files = collect_input_files(opt.input_dir);

        std::cout << "input_dir=" << opt.input_dir << '\n';
        std::cout << "segment_size=" << opt.segment_size << '\n';
        if (opt.max_values != 0) {
            std::cout << "max_values=" << opt.max_values << '\n';
        }
        std::cout << "files=" << files.size() << '\n';

        for (const auto& file : files) {
            FileStats stats = scan_file(file, opt);
            std::cout << '\n';
            std::cout << "file=" << file.filename().string() << '\n';
            std::cout << "  total_values=" << stats.total_values << '\n';
            std::cout << "  scanned_values=" << stats.scanned_values << '\n';
            std::cout << "  segment_count=" << stats.segment_count << '\n';
            std::cout << "  max_plane_count=" << stats.max_plane_count << '\n';
            std::cout << "  segments_gt8=" << stats.segments_gt8 << '\n';
            if (stats.segments_gt8 != 0) {
                std::cout << "  first_gt8_segment=" << stats.first_gt8_segment << '\n';
                std::cout << "  first_gt8_plane_count=" << stats.first_gt8_plane_count << '\n';
            }
            std::cout << "  histogram=";
            print_histogram(stats.histogram);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "buff_plane_scan: " << error.what() << '\n';
        return 1;
    }
}
