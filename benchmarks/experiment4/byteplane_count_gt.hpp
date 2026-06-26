#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace byteplane_scan
{

struct ByteplaneCountGtOptions
{
  std::filesystem::path artifact_root;
  std::filesystem::path raw_root;
  std::filesystem::path segment_minmax_path;
  std::filesystem::path row_state_output_path;
  double threshold = 0.0;
  int k = -1;
  int device = 0;
  int block_threads = 256;
  bool no_gpu = false;
  bool direct_refine_raw = false;
  bool compact_u_refine_raw = false;
  int repeat = 0;
};

struct ByteplaneCountGtResult
{
  std::string dataset;
  std::filesystem::path row_state_output_path;
  std::uint64_t row_count = 0;
  std::uint64_t count = 0;
  std::uint64_t q = 0;
  std::uint64_t d = 0;
  std::uint64_t u = 0;
  std::uint64_t bytes_read = 0;
  std::uint32_t max_planes = 0;
  std::uint32_t k = 0;
  double artifact_load_ms = 0.0;
  double segment_minmax_ms = 0.0;
  double threshold_prep_ms = 0.0;
  double threshold_classify_ms = 0.0;
  double threshold_encode_ms = 0.0;
  double threshold_base_ms = 0.0;
  double threshold_subtract_extract_ms = 0.0;
  double threshold_pack_ms = 0.0;
  double threshold_static_candidate_ms = 0.0;
  double threshold_dependent_candidate_ms = 0.0;
  int threshold_mixed_segments = 0;
  int threshold_allq_segments = 0;
  int threshold_alld_segments = 0;
  double gpu_stage_ms = 0.0;
  double gpu_total_ms = 0.0;
  double primitive_ms = 0.0;
  std::uint64_t refined_exact_count = 0;
  double raw_stage_ms = 0.0;
  double direct_refine_ms = 0.0;
  double direct_total_ms = 0.0;
  double compact_u_refine_ms = 0.0;
  double total_refined_ms = 0.0;
  double compact_u_ms = 0.0;
  double per_query_ms_median = 0.0;
  double per_query_ms_p10 = 0.0;
  double per_query_ms_p90 = 0.0;
  std::uint64_t compact_U_bytes = 0;
  std::string output_mode;
  bool used_gpu = false;
};

struct ByteplaneCountGtRequest
{
  double threshold = 0.0;
  int k = -1;
  std::filesystem::path row_state_output_path;
};

ByteplaneCountGtResult byteplane_count_gt(const ByteplaneCountGtOptions &options);

std::vector<ByteplaneCountGtResult> byteplane_count_gt_batch(
    const ByteplaneCountGtOptions &options,
    const std::vector<ByteplaneCountGtRequest> &requests);

} // namespace byteplane_scan
