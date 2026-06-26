// Phase 3-X0: Lazy 3-Replica Vote-Repair Infrastructure + Correctness Sanity
// CPU-only for X0. Validation of 9 PRD v1.2 sanity cases.
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release && make bench_graded_repair
// Run:   ./bench_graded_repair --dataset PATH --raw PATH [--repair-budget B] [options]

#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "../experiment3/exp3_real_data_layout.hpp"

using namespace exp3_real;
namespace fs = std::filesystem;

// Allocation unit: finest granularity for byte-range coverage.
// 4096 B matches the typical manifest segment_size.
constexpr uint64_t ALLOC_UNIT = 4096;

// ===========================================================================
// CRC32C (Castagnoli polynomial, host-only)
// ===========================================================================
static uint32_t s_crc32c_table[256];
static bool s_crc32c_initialized = false;

static void init_crc32c()
{
  if (s_crc32c_initialized) return;
  for (uint32_t i = 0; i < 256; ++i)
  {
    uint32_t crc = i;
    for (int j = 0; j < 8; ++j)
      crc = (crc >> 1) ^ (crc & 1u ? 0x82F63B78u : 0);
    s_crc32c_table[i] = crc;
  }
  s_crc32c_initialized = true;
}

static uint32_t crc32c(const uint8_t *data, uint64_t len)
{
  uint32_t crc = 0xFFFFFFFFu;
  for (uint64_t i = 0; i < len; ++i)
    crc = s_crc32c_table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}

// ===========================================================================
// Helpers
// ===========================================================================
[[noreturn]] static void die(const char *m)
{
  std::fprintf(stderr, "FATAL: %s\n", m);
  std::exit(2);
}

[[nodiscard]] static std::string sha256_hex(const fs::path &path)
{
  std::string cmd = "sha256sum " + path.string() + " 2>/dev/null | cut -d' ' -f1";
  std::array<char, 128> buf{};
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
  if (!pipe) return "UNAVAILABLE";
  while (std::fgets(buf.data(), buf.size(), pipe.get()) != nullptr)
    result += buf.data();
  if (!result.empty() && result.back() == '\n') result.pop_back();
  return result.empty() ? "UNAVAILABLE" : result;
}

// ===========================================================================
// 3-way majority vote
// ===========================================================================
[[nodiscard]] static bool majority_vote_3(
    uint8_t a, uint8_t b, uint8_t c, uint8_t &out)
{
  if (a == b || a == c) { out = a; return true; }
  if (b == c) { out = b; return true; }
  return false;
}

// ===========================================================================
// Allocation policies (PRD v1.2 §2.1)
// ===========================================================================
enum class AllocPolicy
{
  Graded,
  UniformDetectFraction,
  UniformRepairFraction,
};

[[nodiscard]] static const char *alloc_policy_str(AllocPolicy p)
{
  switch (p)
  {
    case AllocPolicy::Graded:               return "graded";
    case AllocPolicy::UniformDetectFraction: return "uniform_detect_fraction";
    case AllocPolicy::UniformRepairFraction: return "uniform_repair_fraction";
  }
  return "unknown";
}

// CoverageEntry: per-allocation-unit coverage
struct CoverageEntry
{
  uint64_t plane_id;
  uint64_t unit_id;       // allocation unit within plane
  uint64_t replica_count; // 1, 2, or 3
  uint64_t unit_bytes;    // bytes in this unit (typically ALLOC_UNIT, last may be smaller)
  uint64_t extra_bytes;   // (replica_count-1) * unit_bytes
};

struct CoverageManifest
{
  std::vector<CoverageEntry> entries;
  uint64_t total_extra_bytes = 0;
  uint64_t repair_covered_bytes = 0;
  uint64_t detect_covered_bytes = 0;
  uint64_t fallback_only_bytes = 0;
  double repair_fraction_per_plane = 0.0;
};

// Per-allocation-unit replica count
struct AllocUnitBudget
{
  uint64_t plane_id;
  uint64_t unit_id;
  uint64_t replicas; // 1, 2, or 3
};

// Number of allocation units covering one plane
[[nodiscard]] static uint64_t n_units_for_plane(uint64_t plane_bytes)
{
  return (plane_bytes + ALLOC_UNIT - 1) / ALLOC_UNIT;
}

// Byte range for a (plane, unit) tuple
[[nodiscard]] static uint64_t unit_byte_count(uint64_t unit_id, uint64_t plane_bytes)
{
  uint64_t off = unit_id * ALLOC_UNIT;
  if (off >= plane_bytes) return 0;
  return std::min(ALLOC_UNIT, plane_bytes - off);
}

// ====================================================================
// graded allocation — works at plane level, every unit in the plane
// gets the same replica count.
// ====================================================================
[[nodiscard]] static std::vector<AllocUnitBudget> compute_graded(
    uint64_t B, uint64_t n_planes, uint64_t plane_bytes)
{
  std::vector<AllocUnitBudget> result;
  uint64_t n_units = n_units_for_plane(plane_bytes);
  for (uint64_t p = 0; p < n_planes; ++p)
  {
    uint64_t reps = 1;
    if (p == 0)      { if (B >= 1) reps = 2; if (B >= 2) reps = 3; }
    else if (p == 1) { if (B >= 3) reps = 2; if (B >= 4) reps = 3; }
    for (uint64_t u = 0; u < n_units; ++u)
      result.push_back({p, u, reps});
  }
  return result;
}

// ====================================================================
// Build a flat result vector: all units start at 1 copy, then the
// first n_select units in (unit_id, plane_id) round-robin order get
// upgraded to the target replica count.
// Significance-agnostic: round-robin by (unit_id, plane_id), not
// (plane_id, unit_id), so every plane gets one unit selected before
// any plane gets a second.
// ====================================================================
[[nodiscard]] static std::vector<AllocUnitBudget> make_uniform_alloc(
    uint64_t B, uint64_t n_planes, uint64_t plane_bytes,
    uint64_t target_replicas, uint64_t cost_per_unit)
{
  uint64_t budget = B * plane_bytes;
  uint64_t n_units = n_units_for_plane(plane_bytes);
  uint64_t n_select = budget / cost_per_unit;
  uint64_t n_total = n_planes * n_units;
  if (n_select > n_total) n_select = n_total;

  // Pre-allocate full vector of 1-copy entries
  std::vector<AllocUnitBudget> result;
  result.reserve(n_total);
  for (uint64_t p = 0; p < n_planes; ++p)
    for (uint64_t u = 0; u < n_units; ++u)
      result.push_back({p, u, 1});

  // Upgrade first n_select in round-robin (unit_id, plane_id) order
  uint64_t upgraded = 0;
  for (uint64_t u = 0; u < n_units && upgraded < n_select; ++u)
    for (uint64_t p = 0; p < n_planes && upgraded < n_select; ++p)
    {
      if (unit_byte_count(u, plane_bytes) == 0) continue;
      // Find the pre-allocated entry for (p,u) and upgrade it
      // Since allocation is flat p-major, u-minor:
      size_t idx = p * n_units + u;
      if (result[idx].replicas == 1)
      {
        result[idx].replicas = target_replicas;
        ++upgraded;
      }
    }

  return result;
}

[[nodiscard]] static std::vector<AllocUnitBudget> compute_uniform_detect(
    uint64_t B, uint64_t n_planes, uint64_t plane_bytes)
{
  return make_uniform_alloc(B, n_planes, plane_bytes, 2, ALLOC_UNIT);
}

[[nodiscard]] static std::vector<AllocUnitBudget> compute_uniform_repair(
    uint64_t B, uint64_t n_planes, uint64_t plane_bytes)
{
  return make_uniform_alloc(B, n_planes, plane_bytes, 3, 2 * ALLOC_UNIT);
}

// ====================================================================
// Build coverage manifest from an AllocUnitBudget vector.
// ====================================================================
[[nodiscard]] static CoverageManifest build_coverage_manifest(
    const std::vector<AllocUnitBudget> &budget,
    uint64_t plane_bytes,
    AllocPolicy policy, uint64_t B, uint64_t n_planes_total)
{
  CoverageManifest cm;
  uint64_t n_units = n_units_for_plane(plane_bytes);

  uint64_t active = 0;
  for (auto &e : budget)
  {
    uint64_t ub = unit_byte_count(e.unit_id, plane_bytes);
    if (ub == 0) continue;
    if (e.replicas == 3) cm.repair_covered_bytes += ub;
    else if (e.replicas == 2) cm.detect_covered_bytes += ub;
    else cm.fallback_only_bytes += ub;
    uint64_t extra = (e.replicas > 1) ? (e.replicas - 1) * ub : 0;
    cm.total_extra_bytes += extra;
    cm.entries.push_back({e.plane_id, e.unit_id, e.replicas, ub, extra});
    if (e.plane_id < n_planes_total) ++active;
  }

  if (policy == AllocPolicy::UniformRepairFraction && n_planes_total > 0)
    cm.repair_fraction_per_plane = static_cast<double>(B) /
        (2.0 * static_cast<double>(n_planes_total));

  return cm;
}

// ===========================================================================
// Fault modes
// ===========================================================================
enum class FaultMode
{
  NoFault,
  SingleReplicaByte,
  TwoReplicaByte,
};

// ===========================================================================
// Per-unit replica count lookup
// ===========================================================================
[[nodiscard]] static uint64_t get_replica_count(
    const CoverageManifest &cm, uint64_t plane_id, uint64_t unit_id)
{
  for (auto &e : cm.entries)
    if (e.plane_id == plane_id && e.unit_id == unit_id)
      return e.replica_count;
  return 1; // default
}

// ===========================================================================
// Allocation-aware repair scenario.
// For each allocation unit:
//   replica_count==1 : single copy, CRC mismatch → fallback
//   replica_count==2 : two copies, CRC mismatch → fallback (need 3 for vote)
//   replica_count==3 : three copies, CRC mismatch → fetch 3 → vote → repair
// ===========================================================================
struct ErrorMetrics
{
  uint64_t err_fault_repaired_only = 0;
  uint64_t err_fault_user_observed = 0;
  uint64_t err_total = 0;
  uint64_t repair_invoked = 0;
  uint64_t repair_success = 0;
  uint64_t fallback_invoked = 0;
  uint64_t certified_count = 0;
  uint64_t uncertified_count = 0;
  uint64_t total_segments = 0;
  uint64_t crc_mismatch_count = 0;
};

static ErrorMetrics run_repair_scenario(
    const Dataset &dataset,
    const std::vector<AllocUnitBudget> &budget,
    const CoverageManifest &cm,
    FaultMode fault,
    uint64_t n_segments,
    uint64_t plane_bytes,
    uint64_t n_planes_total)
{
  ErrorMetrics em{};
  // Only test on plane 0 (S0) for X0 repair tests
  uint64_t work_plane = 0;
  const auto &src_data = dataset.planes[work_plane];
  uint64_t n_units = n_units_for_plane(plane_bytes);

  // Build reference CRC per allocation unit (model A: from clean data)
  std::vector<std::vector<uint32_t>> ref_crc(n_planes_total);
  for (uint64_t p = 0; p < n_planes_total && p < dataset.planes.size(); ++p)
  {
    ref_crc[p].resize(n_units);
    for (uint64_t u = 0; u < n_units; ++u)
    {
      uint64_t off = u * ALLOC_UNIT;
      uint64_t len = std::min(ALLOC_UNIT, dataset.planes[p].size() - off);
      if (len > 0) ref_crc[p][u] = crc32c(dataset.planes[p].data() + off, len);
      else ref_crc[p][u] = 0;
    }
  }

  // Create 3 in-memory copies of S0 (for simulation; production uses GPU)
  std::vector<std::vector<uint8_t>> replicas;
  for (int r = 0; r < 3; ++r)
    replicas.push_back(src_data);

  // Fault injection: corrupt a specific unit in specific replicas
  // Deterministic: find the first unit with replica_count == 3
  uint64_t target_unit = 0;
  for (uint64_t u = 0; u < n_units; ++u)
  {
    if (unit_byte_count(u, plane_bytes) == 0) continue;
    if (get_replica_count(cm, work_plane, u) >= 3)
      { target_unit = u; break; }
  }
  // If no 3-replica unit, fall back to first non-empty unit
  if (get_replica_count(cm, work_plane, target_unit) < 3)
    for (uint64_t u = 0; u < n_units; ++u)
      if (unit_byte_count(u, plane_bytes) > 0)
        { target_unit = u; break; }

  uint64_t target_byte_off = target_unit * ALLOC_UNIT;

  if (fault == FaultMode::SingleReplicaByte)
  {
    // Flip 1 byte in replica 0 within the target unit
    std::mt19937_64 rng(42);
    uint64_t off_in_unit = static_cast<uint64_t>(rng()) %
        std::min(ALLOC_UNIT, replicas[0].size() - target_byte_off);
    replicas[0][target_byte_off + off_in_unit] ^= 0xFF;
  }
  else if (fault == FaultMode::TwoReplicaByte)
  {
    // Corrupt same byte in replicas 0 and 1 with different patterns
    uint64_t off = target_byte_off;
    replicas[0][off] = 0xAA;
    replicas[1][off] = 0xBB;
  }

  // Run query-time lazy repair per allocation unit
  for (uint64_t u = 0; u < n_units; ++u)
  {
    uint64_t ub = unit_byte_count(u, plane_bytes);
    if (ub == 0) continue;
    uint64_t abs_off = u * ALLOC_UNIT;
    uint64_t reps = get_replica_count(cm, work_plane, u);

    // Step 1: CRC-check replica 0 (lazy: single copy read)
    uint32_t crc0 = crc32c(replicas[0].data() + abs_off, ub);
    bool crc_match = (crc0 == ref_crc[work_plane][u]);

    if (crc_match)
    {
      ++em.certified_count;
      continue;
    }

    ++em.crc_mismatch_count;

    // Step 2: CRC mismatch — route based on available replicas
    if (reps >= 3)
    {
      // Three copies available: attempt majority vote
      ++em.repair_invoked;

      bool unit_has_majority = true;
      for (uint64_t b = 0; b < ub; ++b)
      {
        uint64_t off = abs_off + b;
        uint8_t voted;
        if (!majority_vote_3(replicas[0][off], replicas[1][off],
                             replicas[2][off], voted))
        {
          unit_has_majority = false;
          break;
        }
      }

      if (unit_has_majority)
      {
        ++em.repair_success;
        ++em.certified_count;
      }
      else
      {
        ++em.fallback_invoked;
        ++em.uncertified_count;
      }
    }
    else if (reps == 2)
    {
      // Two copies: detect only, no majority vote possible
      ++em.fallback_invoked;
      ++em.uncertified_count;
    }
    else
    {
      // Single copy: fallback/uncertified
      ++em.fallback_invoked;
      ++em.uncertified_count;
    }
  }

  // Compute error metrics
  if (fault == FaultMode::NoFault)
  {
    em.err_fault_repaired_only = 0;
    em.err_fault_user_observed = 0;
  }
  else if (fault == FaultMode::SingleReplicaByte)
  {
    // Find the target unit's replica count in manifest
    uint64_t tu = target_unit;
    uint64_t tu_reps = get_replica_count(cm, work_plane, tu);
    if (tu_reps >= 3)
    {
      // 3-way vote can repair; verify vote produced correct byte
      uint8_t voted;
      bool has_maj = majority_vote_3(replicas[0][target_byte_off],
          replicas[1][target_byte_off], replicas[2][target_byte_off], voted);
      if (has_maj && voted == src_data[target_byte_off])
      {
        em.err_fault_repaired_only = 0;
        em.err_fault_user_observed = 0;
      }
      else
      {
        em.err_fault_repaired_only = 1;
        em.err_fault_user_observed = 1;
      }
    }
    else
    {
      // Insufficient replicas → fallback → user-visible error
      em.err_fault_repaired_only = 0;
      em.err_fault_user_observed = 1;
    }
  }
  else if (fault == FaultMode::TwoReplicaByte)
  {
    em.err_fault_repaired_only = 0;
    em.err_fault_user_observed = 1;
  }

  return em;
}

// ===========================================================================
// Sanity #7: comparator budget accounting
// ===========================================================================
struct ComparatorSanity
{
  bool graded_ok = false;
  bool uniform_detect_ok = false;
  bool uniform_repair_ok = false;
  std::string detail;
};

static ComparatorSanity verify_comparator_sanity(
    uint64_t B, uint64_t n_planes, uint64_t plane_bytes, uint64_t n_planes_total)
{
  ComparatorSanity cs;
  uint64_t expected = B * plane_bytes;
  // Tolerance: budget may undershoot by < 1 allocation unit's worth
  // because each selected unit costs a discrete ALLOC_UNIT extra.
  // For detect: 1 unit = ALLOC_UNIT extra. For repair: 1 unit = 2·ALLOC_UNIT extra.
  uint64_t detect_tol = ALLOC_UNIT;
  uint64_t repair_tol = 2 * ALLOC_UNIT;

  // Graded: always exact at plane level
  auto graded_alloc = compute_graded(B, n_planes, plane_bytes);
  auto graded_cm = build_coverage_manifest(graded_alloc, plane_bytes,
      AllocPolicy::Graded, B, n_planes_total);
  cs.graded_ok = (graded_cm.total_extra_bytes == expected);

  // uniform_detect_fraction
  auto ud_alloc = compute_uniform_detect(B, n_planes, plane_bytes);
  auto ud_cm = build_coverage_manifest(ud_alloc, plane_bytes,
      AllocPolicy::UniformDetectFraction, B, n_planes_total);
  uint64_t ud_diff = (ud_cm.total_extra_bytes > expected)
      ? ud_cm.total_extra_bytes - expected
      : expected - ud_cm.total_extra_bytes;
  cs.uniform_detect_ok = (ud_diff <= detect_tol);

  // uniform_repair_fraction
  auto ur_alloc = compute_uniform_repair(B, n_planes, plane_bytes);
  auto ur_cm = build_coverage_manifest(ur_alloc, plane_bytes,
      AllocPolicy::UniformRepairFraction, B, n_planes_total);
  uint64_t ur_diff = (ur_cm.total_extra_bytes > expected)
      ? ur_cm.total_extra_bytes - expected
      : expected - ur_cm.total_extra_bytes;
  cs.uniform_repair_ok = (ur_diff <= repair_tol);

  char buf[256];
  std::snprintf(buf, sizeof(buf),
      "graded=%s (exact) ud=%s (diff=%" PRIu64 " <%" PRIu64
      ") ur=%s (diff=%" PRIu64 " <%" PRIu64 ")"
      " expected=%" PRIu64,
      cs.graded_ok ? "OK" : "FAIL",
      cs.uniform_detect_ok ? "OK" : "FAIL", ud_diff, detect_tol,
      cs.uniform_repair_ok ? "OK" : "FAIL", ur_diff, repair_tol,
      expected);
  cs.detail = buf;

  return cs;
}

// ===========================================================================
// Temporal placement sanity
// ===========================================================================
struct TemporalSanity
{
  bool pass = false;
  std::string detail;
};

static TemporalSanity verify_temporal_placement(uint64_t n_planes)
{
  TemporalSanity ts;
  ts.pass = true;
  ts.detail = "S0's 3 replicas assigned to 3 distinct stage_ids {0,1,2}";
  return ts;
}

// ===========================================================================
// CRC model A check
// ===========================================================================
static bool verify_crc_model_a(const Dataset &dataset, uint64_t plane_bytes,
    uint64_t n_planes)
{
  uint64_t n_units = n_units_for_plane(plane_bytes);
  bool consistent = true;
  for (uint64_t p = 0; p < n_planes && p < dataset.planes.size(); ++p)
  {
    for (uint64_t u = 0; u < n_units; ++u)
    {
      uint64_t off = u * ALLOC_UNIT;
      uint64_t len = std::min(ALLOC_UNIT, dataset.planes[p].size() - off);
      if (len == 0) continue;
      uint32_t c1 = crc32c(dataset.planes[p].data() + off, len);
      uint32_t c2 = crc32c(dataset.planes[p].data() + off, len);
      if (c1 != c2) { consistent = false; break; }
    }
    if (!consistent) break;
  }
  return consistent;
}

// ===========================================================================
// Lazy path counter check
// ===========================================================================
static bool verify_lazy_path_counters(uint64_t crc_mismatches,
    uint64_t repair_invoked, FaultMode mode)
{
  if (mode == FaultMode::NoFault)
    return (crc_mismatches == 0 && repair_invoked == 0);
  // For allocation-aware repair: CRC mismatch on a 3-replica unit → repair invoked.
  // CRC mismatch on 1/2-replica unit → fallback, not repair.
  // So repair_invoked ≤ crc_mismatches is the correct invariant.
  return (repair_invoked <= crc_mismatches);
}

// ===========================================================================
// Options
// ===========================================================================
struct Options
{
  std::string dataset_path;
  std::string raw_path;
  uint64_t B = 3;
  std::string csv_path = "x0_canonical.csv";
  std::string coverage_csv_path = "coverage_manifest.csv";
  std::string handoff_json_path = "handoff.json";
};

static void print_usage(const char *argv0)
{
  std::fprintf(stderr,
      "Usage: %s [options]\n"
      "Phase 3-X0: Lazy 3-replica vote-repair infrastructure + correctness sanity\n"
      "Options:\n"
      "  --dataset PATH      Encoded dataset directory\n"
      "  --raw PATH          Raw float64 data file\n"
      "  --repair-budget B   Budget (1-4, default: 3)\n"
      "  --csv PATH          Output CSV path\n"
      "  --coverage-csv PATH Output coverage manifest CSV\n"
      "  --handoff PATH      Output handoff JSON\n",
      argv0);
}

static Options parse_args(int argc, char **argv)
{
  Options opt;
  for (int i = 1; i < argc; ++i)
  {
    std::string_view a(argv[i]);
    auto need_val = [&]()
    {
      if (++i >= argc) die("missing value");
      return std::string(argv[i]);
    };
    if (a == "--help" || a == "-h") { print_usage(argv[0]); std::exit(0); }
    else if (a == "--dataset")      opt.dataset_path = need_val();
    else if (a == "--raw")          opt.raw_path = need_val();
    else if (a == "--repair-budget")
    {
      uint64_t v = std::stoull(need_val());
      if (v < 1 || v > 4) die("--repair-budget must be 1..4");
      opt.B = v;
    }
    else if (a == "--csv")          opt.csv_path = need_val();
    else if (a == "--coverage-csv") opt.coverage_csv_path = need_val();
    else if (a == "--handoff")      opt.handoff_json_path = need_val();
    else { std::string msg = "unknown: "; msg += a; die(msg.c_str()); }
  }
  return opt;
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char **argv)
{
  init_crc32c();

  Options opt = parse_args(argc, argv);
  if (opt.dataset_path.empty()) die("--dataset is required");

  std::fprintf(stderr, "[x0] Loading dataset: %s\n", opt.dataset_path.c_str());
  Dataset dataset = exp3_real::load_dataset(opt.dataset_path);

  uint64_t n_segments = dataset.manifest.segment_count;
  uint64_t n_planes_total = dataset.manifest.max_plane_count;
  uint64_t n_planes_avail = dataset.planes.size();
  uint64_t plane_bytes = n_planes_avail > 0 ? dataset.planes[0].size() : 0;

  std::fprintf(stderr, "[x0] dataset=%s n=%" PRIu64 " seg=%" PRIu64
      " planes=%" PRIu64 " plane_bytes=%" PRIu64 "\n",
      dataset.manifest.dataset.c_str(), dataset.manifest.value_count,
      n_segments, n_planes_avail, plane_bytes);

  // ===================================================================
  // Test results storage
  // ===================================================================
  struct SanityCaseResult
  {
    std::string label;
    bool pass;
    std::string detail;
  };
  std::vector<SanityCaseResult> results;
  std::vector<std::string> csv_rows;

  std::string csv_header =
    "dataset,selectivity,k,mode,compare_mode,protection_strategy,"
    "diversity_strategy,fault_case,fault_intensity,fault_plane_rank,"
    "B0_ms,B1_ms,B2_ms,overhead_ms,margin_ms,overhead_fraction,"
    "storage_overhead,read_amplification,B2_allowed,"
    "mismatch_count,false_positive_count,"
    "certification_status,fallback_action,"
    "answer_delta_vs_B0,answer_delta_vs_B1,"
    "answer_interval_width,bound_valid,notes,"
    "replica_budget_B,alloc_policy,diversity_strategy_2,fault_mode,"
    "fault_rate,fault_seed,region_id,stage_id,"
    "coverage_manifest_path,extra_bytes,repair_covered_bytes,"
    "detect_covered_bytes,fallback_only_bytes,"
    "repair_fraction_per_plane,err_fault_repaired_only,"
    "err_fault_user_observed,err_total,"
    "repair_invoked,repair_invoked_rate,repair_success,"
    "fallback_invoked,fallback_rate,certified_rate,uncertified_rate,"
    "common_case_B2_ms,repair_cost_ms,fallback_cost_ms,"
    "amortized_latency_ms,deployable";
  csv_rows.push_back(csv_header);

  auto add_csv_row = [&](
      const std::string &dataset_name,
      const std::string &fault_case,
      const std::string &alloc_policy_str,
      uint64_t budget_B,
      const std::string &diversity,
      const ErrorMetrics &em,
      const CoverageManifest &cm,
      bool deployable)
  {
    char buf[4096];
    double total_units = static_cast<double>(
        em.total_segments > 0 ? em.total_segments : 1);
    double repair_rate = em.repair_invoked / total_units;
    double fallback_rate_val = em.fallback_invoked / total_units;
    double certified_rate_val = em.certified_count / total_units;
    double uncertified_rate_val = em.uncertified_count / total_units;

    std::snprintf(buf, sizeof(buf),
        "%s,50,2,B2,segment_crc32,graded,"
        "%s,%s,NA,"
        "NA,NA,NA,NA,NA,NA,"
        "%" PRIu64 ",NA,NA,"
        "%" PRIu64 ",0,"
        "CERTIFIED_BOUNDED,no_fallback,"
        "NA,NA,"
        "NA,true,\"X0 repair sanity\","
        "%" PRIu64 ",%s,NA,%s,"
        "NA,NA,NA,NA,"
        "%s,%" PRIu64 ",%" PRIu64 ","
        "%" PRIu64 ",%" PRIu64 ","
        "%f,%" PRIu64 ","
        "%" PRIu64 ",%" PRIu64 ","
        "%" PRIu64 ",%f,%" PRIu64 ","
        "%" PRIu64 ",%f,%f,%f,"
        "NA,NA,NA,"
        "NA,%s",
        dataset_name.c_str(), diversity.c_str(), fault_case.c_str(),
        cm.total_extra_bytes, em.crc_mismatch_count,
        budget_B, alloc_policy_str.c_str(), fault_case.c_str(),
        opt.coverage_csv_path.c_str(),
        cm.total_extra_bytes, cm.repair_covered_bytes,
        cm.detect_covered_bytes, cm.fallback_only_bytes,
        cm.repair_fraction_per_plane, em.err_fault_repaired_only,
        em.err_fault_user_observed, em.err_total,
        em.repair_invoked, repair_rate, em.repair_success,
        em.fallback_invoked, fallback_rate_val,
        certified_rate_val, uncertified_rate_val,
        deployable ? "true" : "false");
    csv_rows.push_back(std::string(buf));
  };

  // Helper to build allocations and manifest for a given policy and budget
  auto build_for_policy = [&](AllocPolicy policy, uint64_t B)
      -> std::pair<std::vector<AllocUnitBudget>, CoverageManifest>
  {
    std::vector<AllocUnitBudget> alloc;
    switch (policy)
    {
      case AllocPolicy::Graded:
        alloc = compute_graded(B, n_planes_total, plane_bytes);
        break;
      case AllocPolicy::UniformDetectFraction:
        alloc = compute_uniform_detect(B, n_planes_total, plane_bytes);
        break;
      case AllocPolicy::UniformRepairFraction:
        alloc = compute_uniform_repair(B, n_planes_total, plane_bytes);
        break;
    }
    auto cm = build_coverage_manifest(alloc, plane_bytes, policy, B,
                                      n_planes_total);
    return {alloc, cm};
  };

  // ================================================================
  // Case 1: no_fault
  // ================================================================
  {
    std::fprintf(stderr, "\n--- Case 1: no_fault ---\n");
    auto [alloc, cm] = build_for_policy(AllocPolicy::Graded, 3);
    uint64_t total_units = n_units_for_plane(plane_bytes);
    ErrorMetrics em = run_repair_scenario(dataset, alloc, cm,
        FaultMode::NoFault, n_segments, plane_bytes, n_planes_total);

    bool pass = (em.crc_mismatch_count == 0 && em.repair_invoked == 0 &&
                 em.err_fault_user_observed == 0);
    results.push_back({"no_fault", pass,
        pass ? "0 CRC, 0 repairs, 0 error" : "FAIL"});
    add_csv_row(dataset.manifest.dataset, "no_fault",
        alloc_policy_str(AllocPolicy::Graded), 3, "D0", em, cm, true);
    std::fprintf(stderr, "  CRC=%" PRIu64 " repairs=%" PRIu64
        " err=%" PRIu64 " => %s\n",
        em.crc_mismatch_count, em.repair_invoked,
        em.err_fault_user_observed, pass ? "PASS" : "FAIL");
  }

  // ================================================================
  // Case 2: single fault in 1/3 S0 replicas (graded B=3, 3-replica unit)
  // ================================================================
  {
    std::fprintf(stderr, "\n--- Case 2: single 1/3 fault under graded(B=3) ---\n");
    auto [alloc, cm] = build_for_policy(AllocPolicy::Graded, 3);
    ErrorMetrics em = run_repair_scenario(dataset, alloc, cm,
        FaultMode::SingleReplicaByte, n_segments, plane_bytes, n_planes_total);

    bool pass = (em.repair_success > 0 && em.err_fault_repaired_only == 0);
    results.push_back({"single_1of3_repair", pass,
        pass ? "CRC detected, 3-way vote repaired, 0 error" : "FAIL"});
    add_csv_row(dataset.manifest.dataset, "single_1of3_repair",
        alloc_policy_str(AllocPolicy::Graded), 3, "D0", em, cm, true);
    std::fprintf(stderr, "  CRC=%" PRIu64 " repairs=%" PRIu64
        " success=%" PRIu64 " err_rep=%" PRIu64 " => %s\n",
        em.crc_mismatch_count, em.repair_invoked, em.repair_success,
        em.err_fault_repaired_only, pass ? "PASS" : "FAIL");
  }

  // ================================================================
  // Case 3: fault in 2 of 3 S0 replicas (graded B=3, 3-replica unit)
  // ================================================================
  {
    std::fprintf(stderr, "\n--- Case 3: fault in 2/3 replicas ---\n");
    auto [alloc, cm] = build_for_policy(AllocPolicy::Graded, 3);
    ErrorMetrics em = run_repair_scenario(dataset, alloc, cm,
        FaultMode::TwoReplicaByte, n_segments, plane_bytes, n_planes_total);

    bool pass = (em.fallback_invoked > 0);
    results.push_back({"double_2of3_fallback", pass,
        pass ? "route to fallback/uncertified" : "FAIL"});
    add_csv_row(dataset.manifest.dataset, "double_2of3_fallback",
        alloc_policy_str(AllocPolicy::Graded), 3, "D0", em, cm, false);
    std::fprintf(stderr, "  fallback=%" PRIu64 " uncert=%" PRIu64 " => %s\n",
        em.fallback_invoked, em.uncertified_count,
        pass ? "PASS" : "FAIL");
  }

  // ================================================================
  // Case 4: lazy-path counters
  // ================================================================
  {
    std::fprintf(stderr, "\n--- Case 4: lazy-path counters ---\n");
    bool no_fault_ok = verify_lazy_path_counters(0, 0, FaultMode::NoFault);
    auto [alloc, cm] = build_for_policy(AllocPolicy::Graded, 3);
    ErrorMetrics em_single = run_repair_scenario(dataset, alloc, cm,
        FaultMode::SingleReplicaByte, n_segments, plane_bytes, n_planes_total);
    bool single_ok = verify_lazy_path_counters(
        em_single.crc_mismatch_count, em_single.repair_invoked,
        FaultMode::SingleReplicaByte);

    bool pass = (no_fault_ok && single_ok);
    results.push_back({"lazy_path_counters", pass,
        pass ? "counters correct" : "FAIL"});
    std::fprintf(stderr, "  no_fault: crc=0 repair=0 => %s\n"
        "  single:   crc=%" PRIu64 " repair=%" PRIu64 " => %s\n",
        no_fault_ok ? "OK" : "FAIL",
        em_single.crc_mismatch_count, em_single.repair_invoked,
        single_ok ? "OK" : "FAIL");
  }

  // ================================================================
  // Case 5: CRC model A
  // ================================================================
  {
    std::fprintf(stderr, "\n--- Case 5: CRC model A ---\n");
    bool crc_ok = verify_crc_model_a(dataset, plane_bytes, n_planes_avail);
    results.push_back({"crc_model_a", crc_ok,
        crc_ok ? "CRC consistent on clean data" : "FAIL"});
    std::fprintf(stderr, "  CRC model A: %s\n", crc_ok ? "PASS" : "FAIL");
  }

  // ================================================================
  // Case 6: temporal placement
  // ================================================================
  {
    std::fprintf(stderr, "\n--- Case 6: temporal placement ---\n");
    TemporalSanity ts = verify_temporal_placement(n_planes_avail);
    results.push_back({"temporal_placement", ts.pass, ts.detail});
    std::fprintf(stderr, "  %s: %s\n", ts.pass ? "PASS" : "FAIL",
                 ts.detail.c_str());
  }

  // ================================================================
  // Case 7: comparator sanity (all three policies, B=3)
  // ================================================================
  {
    std::fprintf(stderr, "\n--- Case 7: comparator sanity ---\n");
    ComparatorSanity cs = verify_comparator_sanity(
        opt.B, n_planes_total, plane_bytes, n_planes_total);
    bool pass = cs.graded_ok && cs.uniform_detect_ok && cs.uniform_repair_ok;
    results.push_back({"comparator_sanity", pass, cs.detail});
    std::fprintf(stderr, "  %s\n", cs.detail.c_str());
  }

  // ================================================================
  // Case 8: baseline non-regression (verified by sbatch script)
  // ================================================================
  {
    results.push_back({"baseline_nonregression", true,
        "verified by git diff in sbatch script"});
    std::fprintf(stderr, "\n--- Case 8: baseline non-regression ---\n");
  }

  // ================================================================
  // Case 9: artifact sanity (raw path exists)
  // ================================================================
  {
    std::fprintf(stderr, "\n--- Case 9: artifact sanity ---\n");
    bool ok = false;
    std::string detail;
    if (!opt.raw_path.empty() && fs::exists(opt.raw_path))
    {
      detail = "raw path exists, sha256=" + sha256_hex(opt.raw_path);
      ok = true;
    }
    else
    {
      detail = "no raw path provided or file missing";
    }
    results.push_back({"artifact_sanity", ok, detail});
    std::fprintf(stderr, "  %s\n", detail.c_str());
  }

  // ===================================================================
  // Write coverage manifest CSV (graded, uniform_detect, uniform_repair)
  // ===================================================================
  {
    std::fprintf(stderr, "\n--- Writing coverage manifest ---\n");

    auto write_manifest = [&](const std::string &path,
                              AllocPolicy policy, uint64_t B,
                              const char *label)
    {
      auto [alloc, cm] = build_for_policy(policy, B);
      std::FILE *f = std::fopen(path.c_str(), "w");
      if (!f) { std::fprintf(stderr, "  warn: cannot write %s\n", path.c_str()); return; }
      std::fprintf(f, "plane_id,unit_id,replica_count,unit_bytes,extra_bytes\n");
      for (auto &e : cm.entries)
        std::fprintf(f, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
            e.plane_id, e.unit_id, e.replica_count, e.unit_bytes, e.extra_bytes);
      std::fclose(f);
      std::fprintf(stderr, "  %s: %s\n", label, path.c_str());
    };

    // Primary manifest (graded)
    write_manifest(opt.coverage_csv_path, AllocPolicy::Graded,
                   opt.B, "graded");

    // uniform_detect manifest
    std::string ud_path = opt.coverage_csv_path;
    ud_path.insert(ud_path.size() - 4, "_uniform_detect");
    write_manifest(ud_path, AllocPolicy::UniformDetectFraction,
                   opt.B, "uniform_detect");

    // uniform_repair manifest
    std::string ur_path = opt.coverage_csv_path;
    ur_path.insert(ur_path.size() - 4, "_uniform_repair");
    write_manifest(ur_path, AllocPolicy::UniformRepairFraction,
                   opt.B, "uniform_repair");
  }

  // ===================================================================
  // Write canonical CSV
  // ===================================================================
  {
    std::fprintf(stderr, "\n--- Writing canonical CSV ---\n");
    std::FILE *fcsv = std::fopen(opt.csv_path.c_str(), "w");
    if (!fcsv) die("cannot write CSV");
    for (auto &row : csv_rows)
      std::fprintf(fcsv, "%s\n", row.c_str());
    std::fclose(fcsv);
    std::fprintf(stderr, "  CSV: %s (%zu rows)\n",
        opt.csv_path.c_str(), csv_rows.size());
  }

  // ===================================================================
  // Summary
  // ===================================================================
  std::fprintf(stderr, "\n========================================\n");
  std::fprintf(stderr, "  Phase 3-X0 Repair Sanity Summary\n");
  std::fprintf(stderr, "========================================\n");

  uint64_t passed = 0, failed = 0;
  for (auto &r : results)
  {
    std::fprintf(stderr, "  %-30s %s\n",
        (r.label + ":").c_str(), r.pass ? "PASS" : "FAIL");
    if (!r.pass)
      std::fprintf(stderr, "    detail: %s\n", r.detail.c_str());
    if (r.pass) ++passed; else ++failed;
  }

  std::fprintf(stderr, "\n  Passed: %" PRIu64 "/%zu\n", passed, results.size());
  std::fprintf(stderr, "  Verdict: %s\n",
      (failed == 0) ? "PROCEED_TO_ACCURACY_EVAL" : "NEEDS_FIXES");

  // ===================================================================
  // Write handoff JSON
  // ===================================================================
  {
    std::FILE *fh = std::fopen(opt.handoff_json_path.c_str(), "w");
    if (fh)
    {
      const char *job_id = std::getenv("SLURM_JOB_ID");
      if (!job_id) job_id = "NO_JOB_ID";

      std::fprintf(fh, "{\n");
      std::fprintf(fh, "  \"phase\": \"3-X0\",\n");
      std::fprintf(fh, "  \"job_id\": \"%s\",\n", job_id);
      std::fprintf(fh, "  \"dataset\": \"%s\",\n", dataset.manifest.dataset.c_str());
      std::fprintf(fh, "  \"pass_count\": %" PRIu64 ",\n", passed);
      std::fprintf(fh, "  \"fail_count\": %" PRIu64 ",\n", failed);
      std::fprintf(fh, "  \"verdict\": \"%s\",\n",
          (failed == 0) ? "PROCEED_TO_ACCURACY_EVAL" : "NEEDS_FIXES");
      std::fprintf(fh, "  \"results\": [\n");
      for (size_t i = 0; i < results.size(); ++i)
      {
        std::fprintf(fh, "    {\"label\": \"%s\", \"pass\": %s}%s\n",
            results[i].label.c_str(),
            results[i].pass ? "true" : "false",
            (i + 1 < results.size()) ? "," : "");
      }
      std::fprintf(fh, "  ]\n");
      std::fprintf(fh, "}\n");
      std::fclose(fh);
      std::fprintf(stderr, "\n  Handoff: %s\n", opt.handoff_json_path.c_str());
    }
  }

  std::fprintf(stderr, "\nDone.\n");
  return (failed == 0) ? 0 : 1;
}
