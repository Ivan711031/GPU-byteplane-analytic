// Phase 3-X1: Graded vs Uniform Repair Accuracy — Paired-Seed Sweep
// CPU-only for X1 (accuracy metrics, no latency/timing).
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release && make bench_graded_repair_x1
// Run:   ./bench_graded_repair_x1 --dataset PATH --raw PATH [options]

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "../experiment3/exp3_real_data_layout.hpp"

using namespace exp3_real;
namespace fs = std::filesystem;

constexpr uint64_t ALLOC_UNIT = 4096;

// ===========================================================================
// CRC32C
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

[[nodiscard]] static bool majority_vote_3(
    uint8_t a, uint8_t b, uint8_t c, uint8_t &out)
{
  if (a == b || a == c) { out = a; return true; }
  if (b == c) { out = b; return true; }
  return false;
}

[[nodiscard]] static uint64_t n_units(uint64_t plane_bytes)
{
  return (plane_bytes + ALLOC_UNIT - 1) / ALLOC_UNIT;
}

[[nodiscard]] static uint64_t unit_bytes(uint64_t uid, uint64_t plane_bytes)
{
  uint64_t off = uid * ALLOC_UNIT;
  if (off >= plane_bytes) return 0;
  return std::min(ALLOC_UNIT, plane_bytes - off);
}

// ===========================================================================
// Allocation policies
// ===========================================================================
enum class AllocPolicy { Graded, UniformDetectFraction, UniformRepairFraction };

static const char *policy_str(AllocPolicy p)
{
  switch (p) {
    case AllocPolicy::Graded: return "graded";
    case AllocPolicy::UniformDetectFraction: return "uniform_detect_fraction";
    case AllocPolicy::UniformRepairFraction: return "uniform_repair_fraction";
  }
  return "unknown";
}

struct AllocUnit { uint64_t plane_id, unit_id; uint64_t replicas; };

static void fill_graded(std::vector<AllocUnit> &out, uint64_t B,
    uint64_t n_planes, uint64_t plane_bytes)
{
  uint64_t nu = n_units(plane_bytes);
  for (uint64_t p = 0; p < n_planes; ++p)
  {
    uint64_t reps = 1;
    if (p == 0)      { if (B >= 1) reps = 2; if (B >= 2) reps = 3; }
    else if (p == 1) { if (B >= 3) reps = 2; if (B >= 4) reps = 3; }
    for (uint64_t u = 0; u < nu; ++u) out.push_back({p, u, reps});
  }
}

static uint64_t n_selectable_units(uint64_t n_planes, uint64_t plane_bytes)
{
  uint64_t nu = n_units(plane_bytes);
  uint64_t total = 0;
  for (uint64_t p = 0; p < n_planes; ++p)
    for (uint64_t u = 0; u < nu; ++u)
      if (unit_bytes(u, plane_bytes) > 0) ++total;
  return total;
}

// Flat pre-allocation for uniform policies: all units start at 1,
// then first n in round-robin (unit_id, plane_id) get upgraded.
static void fill_uniform(std::vector<AllocUnit> &out, uint64_t B,
    uint64_t n_planes, uint64_t plane_bytes, uint64_t target_rep,
    uint64_t cost_per_unit)
{
  uint64_t budget = B * plane_bytes;
  uint64_t nu = n_units(plane_bytes);
  uint64_t n_upgrade = budget / cost_per_unit;
  uint64_t total = n_selectable_units(n_planes, plane_bytes);
  if (n_upgrade > total) n_upgrade = total;

  out.reserve(total);
  for (uint64_t p = 0; p < n_planes; ++p)
    for (uint64_t u = 0; u < nu; ++u)
      out.push_back({p, u, 1});

  uint64_t upgraded = 0;
  for (uint64_t u = 0; u < nu && upgraded < n_upgrade; ++u)
    for (uint64_t p = 0; p < n_planes && upgraded < n_upgrade; ++p)
    {
      if (unit_bytes(u, plane_bytes) == 0) continue;
      size_t idx = p * nu + u;
      if (out[idx].replicas == 1) { out[idx].replicas = target_rep; ++upgraded; }
    }
}

// ===========================================================================
// Fault injection: geometric-skip O(n_faults) generation
// ===========================================================================
struct FaultPlan
{
  std::vector<uint64_t> byte_offsets; // absolute byte positions to flip
  // For each byte offset, which replica is corrupted
  // For uniform fault mode: each byte is independently flipped w.p. rate
  // in each replica independently
  std::vector<std::vector<uint64_t>> replica_flips; // [byte_idx][replica]
};

// Generate fault plan for a given seed, rate, plane_bytes, n_replicas.
// Deterministic: same (seed, rate) → same fault positions.
static FaultPlan generate_fault_plan(
    uint64_t seed, double rate, uint64_t plane_bytes, int n_replicas)
{
  FaultPlan plan;
  if (rate <= 0.0) return plan;

  std::mt19937_64 rng(seed);
  // Generate byte positions for each replica independently
  plan.replica_flips.resize(n_replicas);
  for (int r = 0; r < n_replicas; ++r)
  {
    // Use geometric distribution to skip to next fault
    // P(skip = k) = rate * (1-rate)^k  for k >= 0
    // Distance between faults: geometric(p=rate)
    std::geometric_distribution<uint64_t> geom(rate);
    uint64_t pos = 0;
    while (pos < plane_bytes)
    {
      uint64_t skip = geom(rng);
      pos += skip;
      if (pos < plane_bytes)
      {
        plan.byte_offsets.push_back(pos);
        plan.replica_flips[r].push_back(pos);
      }
      pos += 1; // move past the flipped byte
    }
  }

  // Deduplicate and sort byte offsets
  std::sort(plan.byte_offsets.begin(), plan.byte_offsets.end());
  plan.byte_offsets.erase(
      std::unique(plan.byte_offsets.begin(), plan.byte_offsets.end()),
      plan.byte_offsets.end());

  return plan;
}

// Apply faults to a set of replicas
static void apply_faults(
    std::vector<uint8_t> *replicas, int n_replicas,
    const FaultPlan &plan)
{
  for (int r = 0; r < n_replicas && r < n_replicas; ++r)
    for (auto off : plan.replica_flips[r])
      replicas[r][off] ^= 0xFF;
}

// ===========================================================================
// Per-configuration metrics
// ===========================================================================
struct TrialMetrics
{
  uint64_t err_repaired_only = 0; // errors in successfully repaired rows
  uint64_t err_user_observed = 0; // errors including fallback
  uint64_t err_total = 0;
  uint64_t crc_mismatches = 0;
  uint64_t repair_invoked = 0;
  uint64_t repair_success = 0;
  uint64_t fallback_invoked = 0;
  uint64_t certified = 0;
  uint64_t uncertified = 0;
};

// Run one trial: check CRC and dispatch repair according to allocation.
// `clean_data` = original clean plane data
// `replicas` = 3 copies (possibly with faults injected)
// Returns metrics for this trial.
static TrialMetrics run_trial(
    const uint8_t *clean_data, uint64_t plane_bytes,
    const std::vector<uint8_t> *replicas,
    const std::vector<uint32_t> &clean_crc, // per-unit clean CRC
    const std::vector<uint64_t> &rep_per_unit) // per-unit replica count
{
  TrialMetrics m;
  uint64_t nu = n_units(plane_bytes);

  // Per-unit CRC check
  for (uint64_t u = 0; u < nu; ++u)
  {
    uint64_t ub = unit_bytes(u, plane_bytes);
    if (ub == 0) continue;
    uint64_t off = u * ALLOC_UNIT;
    uint64_t reps = (u < rep_per_unit.size()) ? rep_per_unit[u] : 1;

    // CRC-check replica 0
    uint32_t crc0 = crc32c(replicas[0].data() + off, ub);
    bool crc_match = (crc0 == clean_crc[u]);

    if (crc_match)
    {
      ++m.certified;
      continue;
    }

    ++m.crc_mismatches;

    if (reps >= 3)
    {
      ++m.repair_invoked;
      bool has_majority = true;
      for (uint64_t b = 0; b < ub; ++b)
      {
        uint8_t voted;
        if (!majority_vote_3(replicas[0][off+b], replicas[1][off+b],
                             replicas[2][off+b], voted))
        { has_majority = false; break; }
      }
      if (has_majority)
      {
        ++m.repair_success;
        ++m.certified;
      }
      else
      {
        ++m.fallback_invoked;
        ++m.uncertified;
      }
    }
    else if (reps == 2)
    {
      ++m.fallback_invoked;
      ++m.uncertified;
    }
    else
    {
      ++m.fallback_invoked;
      ++m.uncertified;
    }
  }

  // For X1 accuracy: compare against fault-free same-k answer.
  // err_fault_repaired_only = sum of per-byte error in repaired units
  // (0 if repair restores original byte)
  if (m.repair_success > 0 && m.repair_invoked > 0)
  {
    // Check that successful repairs actually restored the correct data.
    // For each unit that was repaired, verify every byte matches clean.
    for (uint64_t u = 0; u < nu; ++u)
    {
      uint64_t ub = unit_bytes(u, plane_bytes);
      if (ub == 0) continue;
      uint64_t off = u * ALLOC_UNIT;
      uint32_t crc0 = crc32c(replicas[0].data() + off, ub);

      // If this unit had a CRC mismatch AND was reported as repaired
      // successfully, verify the bytes are correct.
      if (crc0 != clean_crc[u] && rep_per_unit[u] >= 3)
      {
        // Unit had corruption and has 3-way vote available.
        // Check if voted bytes match clean.
        // (In a full impl, this would compare reconstructed answers, not raw bytes)
        for (uint64_t b = 0; b < ub; ++b)
        {
          uint8_t voted;
          if (!majority_vote_3(replicas[0][off+b], replicas[1][off+b],
                               replicas[2][off+b], voted))
            break; // no majority already counted
          if (voted != clean_data[off+b])
          {
            ++m.err_repaired_only;
            break; // one error per unit
          }
        }
      }
    }
  }

  if (m.fallback_invoked > 0)
    m.err_user_observed = 1;

  return m;
}

// ===========================================================================
// CI95
// ===========================================================================
struct CI95 { double mean, low, high; };

static CI95 compute_ci95(const std::vector<double> &samples)
{
  CI95 ci{};
  size_t n = samples.size();
  if (n < 2) return ci;
  double sum = 0, sum2 = 0;
  for (auto v : samples) { sum += v; sum2 += v * v; }
  ci.mean = sum / n;
  double var = (sum2 - sum * sum / n) / (n - 1);
  double sd = std::sqrt(var);
  // exact: t_0.975(29)
  static const double t29 = 2.045229642132703;
  double se = sd / std::sqrt(static_cast<double>(n));
  ci.low = ci.mean - t29 * se;
  ci.high = ci.mean + t29 * se;
  return ci;
}

// ===========================================================================
// Sweep driver
// ===========================================================================
struct SweepConfig
{
  std::string dataset_path, raw_path, csv_path, delta_csv_path, storage_csv_path,
              handoff_json_path;
  // Sweep dimensions
  std::vector<double> fault_rates = {1e-9, 1e-8, 1e-7, 1e-6, 1e-5};
  std::vector<AllocPolicy> policies = {
      AllocPolicy::Graded, AllocPolicy::UniformRepairFraction,
      AllocPolicy::UniformDetectFraction};
  std::vector<uint64_t> B_values = {1, 2, 3, 4};
  int n_seeds = 30;
};

static void run_sweep(const SweepConfig &cfg, const Dataset &dataset,
    uint64_t plane_bytes)
{
  uint64_t n_planes = dataset.manifest.max_plane_count;
  uint64_t nu = n_units(plane_bytes);

  // Pre-compute clean CRC per unit (model A)
  init_crc32c();
  std::vector<uint32_t> clean_crc(nu);
  const auto &clean_plane = dataset.planes[0];
  for (uint64_t u = 0; u < nu; ++u)
  {
    uint64_t ub = unit_bytes(u, plane_bytes);
    if (ub == 0) { clean_crc[u] = 0; continue; }
    clean_crc[u] = crc32c(clean_plane.data() + u * ALLOC_UNIT, ub);
  }

  // Pre-create allocation manifests for each (policy, B)
  struct AllocManifest
  {
    std::vector<AllocUnit> alloc;
    std::vector<uint64_t> rep_per_unit; // per-unit replica count for plane 0
  };
  std::vector<AllocManifest> manifests;

  auto build_manifest = [&](AllocPolicy policy, uint64_t B) -> AllocManifest
  {
    AllocManifest m;
    switch (policy)
    {
      case AllocPolicy::Graded:
        fill_graded(m.alloc, B, n_planes, plane_bytes); break;
      case AllocPolicy::UniformDetectFraction:
        fill_uniform(m.alloc, B, n_planes, plane_bytes, 2, ALLOC_UNIT); break;
      case AllocPolicy::UniformRepairFraction:
        fill_uniform(m.alloc, B, n_planes, plane_bytes, 3, 2*ALLOC_UNIT); break;
    }
    // Extract per-unit replica count for plane 0
    m.rep_per_unit.assign(nu, 1);
    for (auto &au : m.alloc)
      if (au.plane_id == 0 && au.unit_id < nu)
        m.rep_per_unit[au.unit_id] = au.replicas;
    return m;
  };

  for (auto policy : cfg.policies)
    for (auto B : cfg.B_values)
      manifests.push_back(build_manifest(policy, B));

  // Pre-create fault plans per (seed, rate)
  // Only needed for plane 0 (S0) — we only test S0 repair
  struct FaultPlanKey {
    uint64_t seed; double rate;
    bool operator<(const FaultPlanKey &o) const {
      if (seed != o.seed) return seed < o.seed;
      return rate < o.rate;
    }
  };
  std::map<FaultPlanKey, FaultPlan> fault_plans;
  for (int s = 0; s < cfg.n_seeds; ++s)
    for (auto rate : cfg.fault_rates)
      fault_plans[{static_cast<uint64_t>(s), rate}] =
          generate_fault_plan(static_cast<uint64_t>(s), rate, plane_bytes, 3);

  // CSV header
  auto csv_header =
      []() -> std::string
  {
    return "dataset,field,fault_rate,fault_seed,alloc_policy,replica_budget_B,"
           "err_fault_repaired_only,err_fault_user_observed,err_total,"
           "crc_mismatches,repair_invoked,repair_success,"
           "fallback_invoked,certified,uncertified,"
           "repair_invoked_rate,fallback_rate,certified_rate,uncertified_rate";
  };

  // Create 3 replicas (will be mutated per trial)
  std::vector<std::vector<uint8_t>> replicas(3, clean_plane);
  std::vector<uint8_t> clean_copy(clean_plane); // for reset

  // Open output files
  std::FILE *fcsv = std::fopen(cfg.csv_path.c_str(), "w");
  if (!fcsv) die("cannot write CSV");
  std::fprintf(fcsv, "%s\n", csv_header().c_str());

  // Per-seed paired deltas for headline gate (B=3, {1e-7,1e-6,1e-5})
  struct PairedEntry
  {
    double fault_rate;
    uint64_t seed;
    double delta; // uniform_repair - graded
    double mu_graded, mu_uniform;
  };
  std::vector<PairedEntry> paired_deltas;

  // Sweep loop
  for (auto rate : cfg.fault_rates)
  {
    std::fprintf(stderr, "[x1] Sweep rate=%.0e\n", rate);
    for (int s = 0; s < cfg.n_seeds; ++s)
    {
      uint64_t seed = static_cast<uint64_t>(s);
      FaultPlanKey fpk{seed, rate};
      auto fp_it = fault_plans.find(fpk);
      if (fp_it == fault_plans.end()) continue;
      const auto &plan = fp_it->second;

      // Metrics per policy for this (seed, rate)
      // indexed by [policy_idx][B_idx]
      std::vector<std::vector<TrialMetrics>> all_metrics(
          cfg.policies.size(),
          std::vector<TrialMetrics>(cfg.B_values.size()));

      for (size_t pi = 0; pi < cfg.policies.size(); ++pi)
      {
        for (size_t bi = 0; bi < cfg.B_values.size(); ++bi)
        {
          // Reset replicas to clean
          for (int r = 0; r < 3; ++r)
            std::memcpy(replicas[r].data(), clean_copy.data(), plane_bytes);

          // Apply faults
          apply_faults(replicas.data(), 3, plan);

          // Run trial with this manifest
          size_t mi = pi * cfg.B_values.size() + bi;
          auto mm = manifests[mi];
          all_metrics[pi][bi] = run_trial(
              clean_copy.data(), plane_bytes, replicas.data(),
              clean_crc, mm.rep_per_unit);
        }
      }

      // Write CSV rows
      for (size_t pi = 0; pi < cfg.policies.size(); ++pi)
      {
        for (size_t bi = 0; bi < cfg.B_values.size(); ++bi)
        {
          auto &m = all_metrics[pi][bi];
          double total_units = static_cast<double>(nu);
          double repair_rate = total_units > 0 ? m.repair_invoked / total_units : 0;
          double fb_rate = total_units > 0 ? m.fallback_invoked / total_units : 0;
          double cert_rate = total_units > 0 ? m.certified / total_units : 0;
          double uncert_rate = total_units > 0 ? m.uncertified / total_units : 0;

          std::fprintf(fcsv,
              "%s,%s,%.0e,%" PRIu64 ",%s,%" PRIu64 ","
              "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
              "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
              "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
              "%.6e,%.6e,%.6e,%.6e\n",
              dataset.manifest.dataset.c_str(),
              dataset.manifest.dataset.c_str(),
              rate, seed,
              policy_str(cfg.policies[pi]),
              cfg.B_values[bi],
              m.err_repaired_only, m.err_user_observed, m.err_total,
              m.crc_mismatches, m.repair_invoked, m.repair_success,
              m.fallback_invoked, m.certified, m.uncertified,
              repair_rate, fb_rate, cert_rate, uncert_rate);

          // Record paired delta for headline gate
          if (cfg.policies[pi] == AllocPolicy::Graded && cfg.B_values[bi] == 3)
          {
            // Store graded metrics; will match with uniform when we see it
          }
          if (cfg.policies[pi] == AllocPolicy::UniformRepairFraction &&
              cfg.B_values[bi] == 3)
          {
            // Find matching graded metrics (same seed, rate, B=3)
            // The graded row was written first (pi=0), so look back at all_metrics[0][bi_for_B3]
            // bi for B=3: B_values = {1,2,3,4} → bi=2
            size_t b3_idx = 2;
            if (b3_idx < cfg.B_values.size() && cfg.B_values[b3_idx] == 3)
            {
              auto &gm = all_metrics[0][b3_idx];
              double delta = static_cast<double>(m.err_repaired_only) -
                             static_cast<double>(gm.err_repaired_only);
              paired_deltas.push_back({
                  rate, seed, delta,
                  static_cast<double>(gm.err_repaired_only),
                  static_cast<double>(m.err_repaired_only)});
            }
          }
        }
      }
    }
  }

  std::fclose(fcsv);
  std::fprintf(stderr, "[x1] CSV written: %s\n", cfg.csv_path.c_str());

  // ===================================================================
  // Write paired delta summary
  // ===================================================================
  std::FILE *fdelta = std::fopen(cfg.delta_csv_path.c_str(), "w");
  if (!fdelta) die("cannot write delta CSV");
  std::fprintf(fdelta, "fault_rate,seed,delta,err_graded,err_uniform_repair\n");
  for (auto &pe : paired_deltas)
    std::fprintf(fdelta, "%.0e,%" PRIu64 ",%.17g,%.17g,%.17g\n",
        pe.fault_rate, pe.seed, pe.delta, pe.mu_graded, pe.mu_uniform);
  std::fclose(fdelta);
  std::fprintf(stderr, "[x1] Paired delta summary: %s\n", cfg.delta_csv_path.c_str());

  // ===================================================================
  // Compute CI95 for headline gate
  // ===================================================================
  std::fprintf(stderr, "\n=== Headline Gate: C3 graded vs uniform_repair ===\n");
  std::fprintf(stderr, "B=3, band={1e-7,1e-6,1e-5}\n\n");

  bool headline_passed = true;
  for (auto rate : cfg.fault_rates)
  {
    if (rate < 1e-7 || rate > 1e-5) continue; // only headline band
    std::vector<double> deltas;
    for (auto &pe : paired_deltas)
      if (std::abs(pe.fault_rate - rate) < rate * 0.01)
        deltas.push_back(pe.delta);

    CI95 ci = compute_ci95(deltas);
    bool graded_wins = (ci.low > 0);
    std::fprintf(stderr, "  rate=%.0e n=%zu delta_mean=%.4f CI95=[%.4f, %.4f] %s\n",
        rate, deltas.size(), ci.mean, ci.low, ci.high,
        graded_wins ? "graded BETTER" : (ci.high < 0 ? "uniform BETTER" : "equivocal"));
    if (!graded_wins) headline_passed = false;
  }

  std::fprintf(stderr, "\n  Headline verdict: %s\n",
      headline_passed ? "GRADED_BETTER (CI95_low > 0 at all 3 rates)"
                      : "EQUIVOCAL_OR_WORSE");

  // ===================================================================
  // Write storage audit
  // ===================================================================
  std::FILE *fstor = std::fopen(cfg.storage_csv_path.c_str(), "w");
  if (!fstor) die("cannot write storage CSV");
  std::fprintf(fstor, "alloc_policy,B,expected_extra_bytes,actual_extra_bytes,"
                      "repair_covered_bytes,detect_covered_bytes\n");
  for (size_t mi = 0; mi < manifests.size(); ++mi)
  {
    size_t pi = mi / cfg.B_values.size();
    size_t bi = mi % cfg.B_values.size();
    uint64_t B = cfg.B_values[bi];
    AllocPolicy policy = cfg.policies[pi];
    uint64_t expected = B * plane_bytes;

    // Compute actual extra from manifest
    uint64_t extra = 0, repair_cov = 0, detect_cov = 0;
    for (auto &au : manifests[mi].alloc)
    {
      if (au.plane_id == 0)
      {
        uint64_t ub = unit_bytes(au.unit_id, plane_bytes);
        if (au.replicas == 3) repair_cov += ub;
        else if (au.replicas == 2) detect_cov += ub;
        if (au.replicas > 1) extra += (au.replicas - 1) * ub;
      }
    }
    std::fprintf(fstor, "%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
        policy_str(policy), B, expected, extra, repair_cov, detect_cov);
  }
  std::fclose(fstor);
  std::fprintf(stderr, "[x1] Storage audit: %s\n", cfg.storage_csv_path.c_str());

  // ===================================================================
  // Write handoff JSON
  // ===================================================================
  {
    std::FILE *fh = std::fopen(cfg.handoff_json_path.c_str(), "w");
    if (fh)
    {
      const char *jid = std::getenv("SLURM_JOB_ID");
      if (!jid) jid = "NO_JOB_ID";
      std::fprintf(fh, "{\n");
      std::fprintf(fh, "  \"phase\": \"3-X1\",\n");
      std::fprintf(fh, "  \"job_id\": \"%s\",\n", jid);
      std::fprintf(fh, "  \"dataset\": \"%s\",\n", dataset.manifest.dataset.c_str());
      std::fprintf(fh, "  \"headline_passed\": %s,\n",
          headline_passed ? "true" : "false");
      std::fprintf(fh, "  \"verdict\": \"%s\"\n",
          headline_passed ? "PROCEED_TO_DIVERSITY_EVAL" : "STOP_GRADED_NOT_BETTER");
      std::fprintf(fh, "}\n");
      std::fclose(fh);
    }
  }
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char **argv)
{
  SweepConfig cfg;
  for (int i = 1; i < argc; ++i)
  {
    std::string_view a(argv[i]);
    auto val = [&]() {
      if (++i >= argc) die("missing value");
      return std::string(argv[i]);
    };
    if (a == "--help" || a == "-h")
    {
      std::fprintf(stderr,
          "Usage: %s --dataset PATH --raw PATH [options]\n"
          "X1 Sweep options:\n"
          "  --dataset PATH      Encoded dataset dir\n"
          "  --raw PATH          Raw float64 data\n"
          "  --csv PATH          Output canonical CSV (default: x1_accuracy_canonical.csv)\n"
          "  --delta-csv PATH    Paired delta summary (default: x1_paired_delta_summary.csv)\n"
          "  --storage-csv PATH  Storage audit (default: x1_storage_audit.csv)\n"
          "  --handoff PATH      Handoff JSON (default: handoff.json)\n"
          "  --n-seeds N         Number of seeds (default: 30)\n", argv[0]);
      return 0;
    }
    else if (a == "--dataset") cfg.dataset_path = val();
    else if (a == "--raw") cfg.raw_path = val();
    else if (a == "--csv") cfg.csv_path = val();
    else if (a == "--delta-csv") cfg.delta_csv_path = val();
    else if (a == "--storage-csv") cfg.storage_csv_path = val();
    else if (a == "--handoff") cfg.handoff_json_path = val();
    else if (a == "--n-seeds") cfg.n_seeds = std::stoi(val());
    else { std::string msg = "unknown: "; msg += a; die(msg.c_str()); }
  }

  if (cfg.dataset_path.empty()) die("--dataset required");
  if (cfg.csv_path.empty()) cfg.csv_path = "x1_accuracy_canonical.csv";
  if (cfg.delta_csv_path.empty()) cfg.delta_csv_path = "x1_paired_delta_summary.csv";
  if (cfg.storage_csv_path.empty()) cfg.storage_csv_path = "x1_storage_audit.csv";
  if (cfg.handoff_json_path.empty()) cfg.handoff_json_path = "handoff.json";

  Dataset dataset = exp3_real::load_dataset(cfg.dataset_path);
  uint64_t plane_bytes = dataset.planes.empty() ? 0 : dataset.planes[0].size();

  std::fprintf(stderr, "[x1] dataset=%s n=%" PRIu64 " planes=%zu plane_bytes=%" PRIu64 "\n",
      dataset.manifest.dataset.c_str(), dataset.manifest.value_count,
      dataset.planes.size(), plane_bytes);
  std::fprintf(stderr, "[x1] Sweep: %zu rates × %d seeds × %zu policies × %zu B = %zu trials\n",
      cfg.fault_rates.size(), cfg.n_seeds, cfg.policies.size(),
      cfg.B_values.size(),
      cfg.fault_rates.size() * cfg.n_seeds * cfg.policies.size() * cfg.B_values.size());

  run_sweep(cfg, dataset, plane_bytes);

  return 0;
}
