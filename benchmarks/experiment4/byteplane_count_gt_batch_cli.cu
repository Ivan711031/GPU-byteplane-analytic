#include "byteplane_count_gt.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

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

std::vector<std::string_view> split_csv(std::string_view text)
{
  std::vector<std::string_view> result;
  std::size_t start = 0;
  while (start <= text.size())
  {
    std::size_t comma = text.find(',', start);
    std::size_t stop = comma == std::string_view::npos ? text.size() : comma;
    if (stop > start) result.push_back(text.substr(start, stop - start));
    if (comma == std::string_view::npos) break;
    start = comma + 1;
  }
  return result;
}

std::vector<double> parse_f64_list(std::string_view text)
{
  std::vector<double> values;
  for (std::string_view item : split_csv(text))
    values.push_back(std::stod(std::string(item)));
  return values;
}

std::vector<int> parse_i32_list(std::string_view text)
{
  std::vector<int> values;
  for (std::string_view item : split_csv(text))
  {
    std::uint64_t parsed = 0;
    if (!parse_u64(item, parsed) || parsed == 0 || parsed > 256) die("invalid --ks");
    values.push_back(static_cast<int>(parsed));
  }
  return values;
}

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point stop)
{
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

void print_usage(const char *argv0)
{
  std::fprintf(stderr,
               "Usage: %s --artifact-root PATH --thresholds A,B --ks K1,K2 [options]\n"
               "\n"
               "Options:\n"
               "  --artifact-root PATH  Encoded byteplane artifact directory\n"
               "  --raw-root PATH       Optional raw FP64 directory for segment min/max\n"
               "  --segment-minmax P    Optional cached segment,min,max CSV\n"
               "  --row-state-dir DIR   Optional output dir for uint8 row states: 0=D, 1=Q, 2=U\n"
               "  --direct-refine-raw   Refine U rows against raw FP64 on device\n"
               "  --compact-u-refine-raw Compact U rows into device indices before refinement\n"
               "  --thresholds LIST     Comma-separated FP64 thresholds\n"
               "  --ks LIST             Comma-separated k values\n"
               "  --device N            CUDA device index (default: 0)\n"
               "  --block N             CUDA block threads (default: 256)\n"
               "  --repeat N            How many measured iterations per request (+1 warm-up). 0 = single shot.\n",
               argv0);
}

void print_result_json(const byteplane_scan::ByteplaneCountGtResult &result,
                       double threshold)
{
  std::printf("{"
              "\"dataset\":\"%s\","
              "\"rows\":%llu,"
              "\"threshold\":%.17g,"
              "\"k\":%u,"
              "\"max_planes\":%u,"
              "\"count\":%llu,"
              "\"refined_exact_count\":%llu,"
              "\"output_mode\":\"%s\","
              "\"Q\":%llu,"
              "\"D\":%llu,"
              "\"U\":%llu,"
              "\"U_count\":%llu,"
              "\"bytes_read\":%llu,"
              "\"compact_U_bytes\":%llu,"
              "\"artifact_load_ms\":%.6f,"
              "\"segment_minmax_ms\":%.6f,"
               "\"threshold_prep_ms\":%.6f,"
               "\"threshold_classify_ms\":%.6f,"
               "\"threshold_encode_ms\":%.6f,"
               "\"threshold_base_ms\":%.6f,"
               "\"threshold_subtract_extract_ms\":%.6f,"
               "\"threshold_pack_ms\":%.6f,"
               "\"threshold_static_candidate_ms\":%.6f,"
               "\"threshold_dependent_candidate_ms\":%.6f,"
               "\"threshold_mixed_segments\":%d,"
               "\"threshold_allq_segments\":%d,"
               "\"threshold_alld_segments\":%d,"
               "\"gpu_stage_ms\":%.6f,"
               "\"gpu_total_ms\":%.6f,"
               "\"primitive_ms\":%.6f,"
               "\"raw_stage_ms\":%.6f,"
               "\"direct_refine_ms\":%.6f,"
               "\"direct_total_ms\":%.6f,"
               "\"compact_u_refine_ms\":%.6f,"
               "\"total_refined_ms\":%.6f,"
                "\"compact_u_ms\":%.6f,"
                "\"per_query_ms_median\":%.6f,"
                "\"per_query_ms_p10\":%.6f,"
                "\"per_query_ms_p90\":%.6f,"
                "\"row_state_path\":\"%s\","
                "\"used_gpu\":%s"
                "}",
              result.dataset.c_str(),
              static_cast<unsigned long long>(result.row_count),
              threshold,
              result.k,
              result.max_planes,
              static_cast<unsigned long long>(result.count),
              static_cast<unsigned long long>(result.refined_exact_count),
              result.output_mode.c_str(),
              static_cast<unsigned long long>(result.q),
              static_cast<unsigned long long>(result.d),
              static_cast<unsigned long long>(result.u),
              static_cast<unsigned long long>(result.u),
              static_cast<unsigned long long>(result.bytes_read),
              static_cast<unsigned long long>(result.compact_U_bytes),
              result.artifact_load_ms,
              result.segment_minmax_ms,
               result.threshold_prep_ms,
               result.threshold_classify_ms,
               result.threshold_encode_ms,
               result.threshold_base_ms,
               result.threshold_subtract_extract_ms,
               result.threshold_pack_ms,
               result.threshold_static_candidate_ms,
               result.threshold_dependent_candidate_ms,
               result.threshold_mixed_segments,
               result.threshold_allq_segments,
               result.threshold_alld_segments,
               result.gpu_stage_ms,
              result.gpu_total_ms,
              result.primitive_ms,
              result.raw_stage_ms,
              result.direct_refine_ms,
              result.direct_total_ms,
              result.compact_u_refine_ms,
              result.total_refined_ms,
               result.compact_u_ms,
               result.per_query_ms_median,
               result.per_query_ms_p10,
               result.per_query_ms_p90,
               result.row_state_output_path.string().c_str(),
              result.used_gpu ? "true" : "false");
}

} // namespace

int main(int argc, char **argv)
{
  byteplane_scan::ByteplaneCountGtOptions opt;
  std::vector<double> thresholds;
  std::vector<int> ks;
  std::filesystem::path row_state_dir;
  bool direct_refine_raw = false;
  bool compact_u_refine_raw = false;

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
    else if (arg == "--thresholds")
    {
      thresholds = parse_f64_list(need_value(arg));
    }
    else if (arg == "--row-state-dir")
    {
      row_state_dir = std::string(need_value(arg));
    }
    else if (arg == "--direct-refine-raw")
    {
      direct_refine_raw = true;
    }
    else if (arg == "--compact-u-refine-raw")
    {
      compact_u_refine_raw = true;
    }
    else if (arg == "--ks")
    {
      ks = parse_i32_list(need_value(arg));
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
    else if (arg == "--repeat")
    {
      std::uint64_t value = 0;
      if (!parse_u64(need_value(arg), value)) die("invalid --repeat");
      opt.repeat = static_cast<int>(value);
    }
    else
    {
      std::string message = "unknown argument: ";
      message += arg;
      die(message.c_str());
    }
  }

  if (opt.artifact_root.empty()) die("--artifact-root is required");
  if (thresholds.empty()) die("--thresholds is required");
  if (ks.empty()) die("--ks is required");
  if (direct_refine_raw && compact_u_refine_raw)
    die("--direct-refine-raw and --compact-u-refine-raw are mutually exclusive");

  std::vector<byteplane_scan::ByteplaneCountGtRequest> requests;
  std::vector<double> request_thresholds;
  requests.reserve(thresholds.size() * ks.size());
  request_thresholds.reserve(thresholds.size() * ks.size());
  if (!direct_refine_raw && !compact_u_refine_raw && !row_state_dir.empty())
    std::filesystem::create_directories(row_state_dir);
  for (std::size_t threshold_index = 0; threshold_index < thresholds.size(); ++threshold_index)
  {
    double threshold = thresholds[threshold_index];
    for (int k : ks)
    {
      std::filesystem::path row_state_path;
      if (!direct_refine_raw && !compact_u_refine_raw && !row_state_dir.empty())
      {
        row_state_path = row_state_dir /
                         ("threshold_" + std::to_string(threshold_index) +
                          "_k_" + std::to_string(k) + ".u8");
      }
      requests.push_back({threshold, k, row_state_path});
      request_thresholds.push_back(threshold);
    }
  }

  opt.direct_refine_raw = direct_refine_raw;
  opt.compact_u_refine_raw = compact_u_refine_raw;

  try
  {
    auto start = std::chrono::steady_clock::now();
    std::vector<byteplane_scan::ByteplaneCountGtResult> results =
        byteplane_scan::byteplane_count_gt_batch(opt, requests);
    double batch_wall_ms = elapsed_ms(start, std::chrono::steady_clock::now());

    std::printf("{\"batch_wall_ms\":%.6f,\"results\":[", batch_wall_ms);
    for (std::size_t i = 0; i < results.size(); ++i)
    {
      if (i != 0) std::printf(",");
      print_result_json(results[i], request_thresholds[i]);
    }
    std::printf("]}\n");
  }
  catch (const std::exception &ex)
  {
    std::fprintf(stderr, "error: %s\n", ex.what());
    return 2;
  }

  return 0;
}
