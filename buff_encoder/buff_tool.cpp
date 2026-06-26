#include "buff_codec.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void print_usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " encode --input FILE --output FILE --segment-size N [--max-values N] [--precision-power N]\n"
        << "  " << argv0 << " decode --input FILE --output FILE\n"
        << "  " << argv0 << " inspect --input FILE\n"
        << "  " << argv0 << " export-runtime --input FILE.buff64 --raw-input FILE.f64le.bin --out-dir DIR --dataset NAME [--source-path PATH]\n";
}

std::uint64_t parse_u64(const std::string& text) {
    std::size_t consumed = 0;
    unsigned long long value = std::stoull(text, &consumed, 10);
    if (consumed != text.size()) {
        throw std::runtime_error("invalid integer: " + text);
    }
    return static_cast<std::uint64_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage(argv[0]);
            return 1;
        }

        std::string command = argv[1];
        std::filesystem::path input_path;
        std::filesystem::path output_path;
        std::filesystem::path out_dir;
        std::filesystem::path raw_input_path;
        std::uint64_t segment_size = 0;
        std::uint64_t max_values = 0;
        buff::EncodeConfig encode_config;
        std::string dataset_name;
        std::string source_path;

        for (int index = 2; index < argc; ++index) {
            std::string_view arg = argv[index];
            if (arg == "--input" && index + 1 < argc) {
                input_path = argv[++index];
            } else if (arg == "--raw-input" && index + 1 < argc) {
                raw_input_path = argv[++index];
            } else if (arg == "--output" && index + 1 < argc) {
                output_path = argv[++index];
            } else if (arg == "--out-dir" && index + 1 < argc) {
                out_dir = argv[++index];
            } else if (arg == "--segment-size" && index + 1 < argc) {
                segment_size = parse_u64(argv[++index]);
            } else if (arg == "--max-values" && index + 1 < argc) {
                max_values = parse_u64(argv[++index]);
            } else if (arg == "--precision-power" && index + 1 < argc) {
                encode_config.precision_power = static_cast<std::uint32_t>(parse_u64(argv[++index]));
            } else if (arg == "--dataset" && index + 1 < argc) {
                dataset_name = argv[++index];
            } else if (arg == "--source-path" && index + 1 < argc) {
                source_path = argv[++index];
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("unknown option: " + std::string(arg));
            }
        }

        if (command == "encode") {
            if (input_path.empty() || output_path.empty() || segment_size == 0) {
                throw std::runtime_error("encode requires --input, --output, and --segment-size");
            }
            buff::encode_file(input_path, output_path, segment_size, max_values, encode_config);
            buff::EncodedFileHeader header = buff::read_file_header(output_path);
            std::cout << "encoded " << header.total_values << " rows into " << output_path
                      << " with segment_size=" << header.segment_size
                      << " segment_count=" << header.segment_count << '\n';
            return 0;
        }

        if (command == "decode") {
            if (input_path.empty() || output_path.empty()) {
                throw std::runtime_error("decode requires --input and --output");
            }
            buff::decode_file(input_path, output_path);
            std::cout << "decoded " << input_path << " into " << output_path << '\n';
            return 0;
        }

        if (command == "inspect") {
            if (input_path.empty()) {
                throw std::runtime_error("inspect requires --input");
            }
            buff::EncodedFileHeader header = buff::read_file_header(input_path);
            std::cout << "total_values=" << header.total_values
                      << " segment_size=" << header.segment_size
                      << " segment_count=" << header.segment_count << '\n';
            return 0;
        }

        if (command == "export-runtime") {
            if (input_path.empty() || raw_input_path.empty() || out_dir.empty() || dataset_name.empty()) {
                throw std::runtime_error("export-runtime requires --input, --raw-input, --out-dir, and --dataset");
            }
            buff::export_runtime_layout(input_path, raw_input_path, out_dir, dataset_name, source_path);
            std::cout << "exported runtime artifact to " << out_dir
                      << " for dataset=" << dataset_name << '\n';
            return 0;
        }

        throw std::runtime_error("unknown command: " + command);
    } catch (const std::exception& error) {
        std::cerr << "buff_tool: " << error.what() << '\n';
        return 1;
    }
}
