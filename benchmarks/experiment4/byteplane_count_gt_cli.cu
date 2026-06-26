#include "byteplane_count_gt.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <string_view>

namespace
{

[[noreturn]] void die(const char *message)
{
  std::fprintf(stderr, "error: %s\n", message);
  std::exit(2);
}

bool parse_u64(std::string_view text, std::uint64_t &out)
{
  if (text.empty()) return false;
  std::uint64_t value = 0;
  for (char c : text)
  {
    if (c < '0' || c > '9') return false;
    value = value * 10ull + static_cast<std::uint64_t>(c - '0');
  }
  out = value;
  return true;
}

void print_usage(const char *argv0)
{
  std::fprintf(stderr,
               "Usage: %s --artifact-root PATH --threshold FP64 --k N [options]\n"
               "\n"
               "Options:\n"
               "  --artifact-root PATH  Encoded byteplane artifact directory\n"
               "  --raw-root PATH       Optional raw FP64 directory for segment min/max\n"
               "  --segment-minmax P    Optional cached segment,min,max CSV\n"
               "  --threshold FP64      Predicate threshold for value > threshold\n"
               "  --k N                 Number of byteplanes to read\n"
               "  --device N            CUDA device index (default: 0)\n"
               "  --block N             CUDA block threads (default: 256)\n"
               "  --row-state-output P  Optional uint8 row-state output: 0=D, 1=Q, 2=U\n"
               "  --no-gpu              Run CPU fallback only\n",
               argv0);
}

} // namespace

int main(int argc, char **argv)
{
  byteplane_scan::ByteplaneCountGtOptions opt;
  bool has_threshold = false;
  bool has_k = false;

  for (int i = 1; i < argc; ++i)
  {
    std::string_view arg(argv[i]);
    auto need_value = [&](std::string_view flag) -> std::string_view
    {
      if (i + 1 >= argc)
      {
        std::string message = "missing value for ";
        message += flag;
        die(message.c_str());
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h")
    {
      print_usage(argv[0]);
      return 0;
    }
    if (arg == "--artifact-root")
    {
      opt.artifact_root = std::string(need_value(arg));
    }
    else if (arg == "--raw-root")
    {
      opt.raw_root = std::string(need_value(arg));
    }
    else if (arg == "--segment-minmax")
    {
      opt.segment_minmax_path = std::string(need_value(arg));
    }
    else if (arg == "--threshold")
    {
      opt.threshold = std::stod(std::string(need_value(arg)));
      has_threshold = true;
    }
    else if (arg == "--k")
    {
      std::uint64_t value = 0;
      if (!parse_u64(need_value(arg), value) || value == 0 || value > 256)
        die("invalid --k");
      opt.k = static_cast<int>(value);
      has_k = true;
    }
    else if (arg == "--device")
    {
      std::uint64_t value = 0;
      if (!parse_u64(need_value(arg), value)) die("invalid --device");
      opt.device = static_cast<int>(value);
    }
    else if (arg == "--block")
    {
      std::uint64_t value = 0;
      if (!parse_u64(need_value(arg), value) || value == 0 || value > 1024)
        die("invalid --block");
      opt.block_threads = static_cast<int>(value);
    }
    else if (arg == "--row-state-output")
    {
      opt.row_state_output_path = std::string(need_value(arg));
    }
    else if (arg == "--no-gpu")
    {
      opt.no_gpu = true;
    }
    else
    {
      std::string message = "unknown argument: ";
      message += arg;
      die(message.c_str());
    }
  }

  if (opt.artifact_root.empty()) die("--artifact-root is required");
  if (!has_threshold) die("--threshold is required");
  if (!has_k) die("--k is required");

  try
  {
    byteplane_scan::ByteplaneCountGtResult result = byteplane_scan::byteplane_count_gt(opt);
    std::printf("{"
                "\"dataset\":\"%s\","
                "\"rows\":%llu,"
                "\"threshold\":%.17g,"
                "\"k\":%u,"
                "\"max_planes\":%u,"
                "\"count\":%llu,"
                "\"Q\":%llu,"
                "\"D\":%llu,"
                "\"U\":%llu,"
                "\"bytes_read\":%llu,"
                "\"artifact_load_ms\":%.6f,"
                "\"segment_minmax_ms\":%.6f,"
                "\"threshold_prep_ms\":%.6f,"
                "\"gpu_stage_ms\":%.6f,"
                "\"gpu_total_ms\":%.6f,"
                "\"primitive_ms\":%.6f,"
                "\"used_gpu\":%s"
                "}\n",
                result.dataset.c_str(),
                static_cast<unsigned long long>(result.row_count),
                opt.threshold,
                result.k,
                result.max_planes,
                static_cast<unsigned long long>(result.count),
                static_cast<unsigned long long>(result.q),
                static_cast<unsigned long long>(result.d),
                static_cast<unsigned long long>(result.u),
                static_cast<unsigned long long>(result.bytes_read),
                result.artifact_load_ms,
                result.segment_minmax_ms,
                result.threshold_prep_ms,
                result.gpu_stage_ms,
                result.gpu_total_ms,
                result.primitive_ms,
                result.used_gpu ? "true" : "false");
  }
  catch (const std::exception &ex)
  {
    std::fprintf(stderr, "error: %s\n", ex.what());
    return 2;
  }

  return 0;
}
