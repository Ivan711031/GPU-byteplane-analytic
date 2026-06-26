// Phase 3-X2: Spatial + Temporal Diversity Repair Evaluation
// CPU-only for the correctness/coverage aspect of diversity.
// D0 .. D3 diversity modes with cluster + temporal fault injection.
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release && make bench_graded_repair_x2
// Run:   ./bench_graded_repair_x2 --dataset PATH --raw PATH [options]

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// CRC32C
static uint32_t s_tab[256]; static bool s_init = false;
static void init_crc() {
  if (s_init) return;
  for (uint32_t i=0;i<256;++i) {uint32_t c=i;for(int j=0;j<8;++j)c=(c>>1)^(c&1?0x82F63B78u:0);s_tab[i]=c;}
  s_init=true;
}
static uint32_t crc32c(const uint8_t *d, uint64_t l) {
  uint32_t c=0xFFFFFFFFu; for(uint64_t i=0;i<l;++i) c=s_tab[(c^d[i])&0xFFu]^(c>>8); return c^0xFFFFFFFFu;
}

[[noreturn]] static void die(const char *m) { std::fprintf(stderr,"FATAL: %s\n",m); std::exit(2); }

[[nodiscard]] static bool majority_vote_3(uint8_t a, uint8_t b, uint8_t c, uint8_t &o) {
  if (a==b||a==c) {o=a;return true;} if (b==c) {o=b;return true;} return false;
}
[[nodiscard]] static uint64_t n_units(uint64_t pb) { return (pb+ALLOC_UNIT-1)/ALLOC_UNIT; }
[[nodiscard]] static uint64_t ubytes(uint64_t u, uint64_t pb) {
  uint64_t o=u*ALLOC_UNIT; if(o>=pb) return 0; return std::min(ALLOC_UNIT,pb-o);
}

enum class AllocPolicy { Graded, UniformDetectFraction, UniformRepairFraction };
static const char *ap_str(AllocPolicy p) {
  switch(p) {
    case AllocPolicy::Graded: return "graded";
    case AllocPolicy::UniformDetectFraction: return "uniform_detect_fraction";
    case AllocPolicy::UniformRepairFraction: return "uniform_repair_fraction";
  } return "unknown";
}

enum class DiversityMode { D0, D1, D2, D3 };
static const char *div_str(DiversityMode d) {
  switch(d) { case DiversityMode::D0: return "D0"; case DiversityMode::D1: return "D1"; case DiversityMode::D2: return "D2"; case DiversityMode::D3: return "D3"; } return "unknown";
}
enum class FaultMode { Cluster, Temporal };
static const char *fm_str(FaultMode f) {
  switch(f) { case FaultMode::Cluster: return "cluster"; case FaultMode::Temporal: return "temporal"; } return "unknown";
}

struct AllocUnit { uint64_t plane_id, unit_id; uint64_t replicas; };

static void fill_graded(std::vector<AllocUnit> &out, uint64_t B, uint64_t n_planes, uint64_t pb) {
  uint64_t nu=n_units(pb);
  for(uint64_t p=0;p<n_planes;++p) {
    uint64_t r=1; if(p==0){if(B>=1)r=2;if(B>=2)r=3;} else if(p==1){if(B>=3)r=2;if(B>=4)r=3;}
    for(uint64_t u=0;u<nu;++u) out.push_back({p,u,r});
  }
}

// === Diversity replica placement ===
// D0: all replicas same region (0), same stage (0)
// D1: replicas at different regions (0,1,2), same stage (0)
// D2: replicas same region (0), different stages (0,1,2)
// D3: different regions (0,1,2) AND different stages (0,1,2)
struct ReplicaPlacement { uint64_t region_id; uint64_t stage_id; };
static std::vector<ReplicaPlacement> placement_for(DiversityMode d, int n_replicas) {
  std::vector<ReplicaPlacement> p;
  for(int r=0;r<n_replicas;++r) {
    uint64_t region = (d==DiversityMode::D1||d==DiversityMode::D3) ? static_cast<uint64_t>(r) : 0;
    uint64_t stage  = (d==DiversityMode::D2||d==DiversityMode::D3) ? static_cast<uint64_t>(r) : 0;
    p.push_back({region, stage});
  }
  return p;
}

// === Fault injection ===
// Cluster: contiguous byte range [base, base+width) in global address space.
// A replica with region_id R is affected if its region overlaps [base, base+width).
// For simplicity, region 0 = [0, plane_bytes/3), region 1 = [plane_bytes/3, 2*plane_bytes/3), region 2 = [2*plane_bytes/3, plane_bytes).
struct ClusterPlan { uint64_t base, width; };
static ClusterPlan gen_cluster(uint64_t seed, double rate, uint64_t plane_bytes) {
  std::mt19937_64 rng(seed);
  uint64_t width = static_cast<uint64_t>(std::ceil(rate * static_cast<double>(plane_bytes)));
  if (width < 1) width = 1;
  if (width > plane_bytes) width = plane_bytes;
  uint64_t base = static_cast<uint64_t>(rng()) % (plane_bytes - width + 1);
  return {base, width};
}

// Temporal: corrupt replicas with a specific stage_id
struct TemporalPlan { uint64_t target_stage; double inner_rate; uint64_t plane_bytes; std::vector<uint64_t> offsets; };
static TemporalPlan gen_temporal(uint64_t seed, double rate, uint64_t plane_bytes) {
  std::mt19937_64 rng(seed);
  uint64_t target = static_cast<uint64_t>(rng()) % 3;
  std::geometric_distribution<uint64_t> geom(rate);
  std::vector<uint64_t> offs;
  uint64_t pos = 0;
  while(pos < plane_bytes) { pos += geom(rng); if(pos<plane_bytes) { offs.push_back(pos); pos+=1; } }
  return {target, rate, plane_bytes, offs};
}

struct AffectedMap {
  std::vector<uint64_t> unit_ids;  // which units have at least one fault across all replicas
  uint64_t crc_skip_count;         // how many units we can skip CRC for
};

// Apply diversity-aware faults and return which units are affected
static AffectedMap apply_diversity_faults(
    std::vector<uint8_t> *replicas, int n_replicas, uint64_t plane_bytes,
    const std::vector<ReplicaPlacement> &placement,
    FaultMode fmode, double rate, uint64_t seed)
{
  AffectedMap am;
  uint64_t nu = n_units(plane_bytes);
  std::vector<bool> unit_affected(nu, false);

  if (fmode == FaultMode::Cluster)
  {
    ClusterPlan cp = gen_cluster(seed, rate, plane_bytes);
    // For each replica, check if its region overlaps the cluster
    for (int r = 0; r < n_replicas; ++r)
    {
      uint64_t region_sz = plane_bytes / 3;
      uint64_t rbase = placement[r].region_id * region_sz;
      uint64_t rend = std::min(rbase + region_sz, plane_bytes);

      // Check overlap
      uint64_t ov_start = std::max(cp.base, rbase);
      uint64_t ov_end = std::min(cp.base + cp.width, rend);
      if (ov_start < ov_end)
      {
        // Corrupt bytes in overlap range
        for (uint64_t b = ov_start; b < ov_end; ++b)
          replicas[r][b] ^= 0xFF;
        // Mark affected units
        for (uint64_t u = 0; u < nu; ++u) {
          uint64_t us = u * ALLOC_UNIT, ue = us + ubytes(u, plane_bytes);
          if (us < ov_end && ue > ov_start) unit_affected[u] = true;
        }
      }
    }
  }
  else if (fmode == FaultMode::Temporal)
  {
    TemporalPlan tp = gen_temporal(seed, rate, plane_bytes);
    for (int r = 0; r < n_replicas; ++r)
    {
      if (placement[r].stage_id == tp.target_stage)
      {
        for (auto off : tp.offsets)
        {
          replicas[r][off] ^= 0xFF;
          uint64_t u = off / ALLOC_UNIT;
          if (u < nu) unit_affected[u] = true;
        }
      }
    }
  }

  for (uint64_t u = 0; u < nu; ++u)
    if (unit_affected[u]) am.unit_ids.push_back(u);
  am.crc_skip_count = nu - am.unit_ids.size();
  return am;
}

struct TrialMetrics {
  uint64_t crc_mismatches = 0, repair_invoked = 0, repair_success = 0;
  uint64_t fallback_invoked = 0, certified = 0, uncertified = 0;
  uint64_t any_replica_corrupted = 0, all_replicas_corrupted = 0;
};

static TrialMetrics run_trial_diversity(
    uint64_t plane_bytes,
    const uint8_t *clean_src,
    const std::vector<uint8_t> *replicas,
    const std::vector<uint32_t> &clean_crc,
    const std::vector<uint64_t> &rep_per_unit,
    const AffectedMap &am)
{
  TrialMetrics m;
  uint64_t nu = n_units(plane_bytes);
  // Use a set for O(1) lookup
  std::vector<bool> is_affected(nu, false);
  for (auto uid : am.unit_ids) if(uid < nu) is_affected[uid] = true;

  for (uint64_t u = 0; u < nu; ++u)
  {
    uint64_t ub = ubytes(u, plane_bytes);
    if (ub == 0) continue;
    uint64_t off = u * ALLOC_UNIT;

    // CRC: skip if unit has no faults (and no residual corruption from previous trial)
    bool crc_match;
    if (!is_affected[u]) {
      crc_match = true; // clean unit, CRC will match
    } else {
      uint32_t crc0 = crc32c(replicas[0].data() + off, ub);
      crc_match = (crc0 == clean_crc[u]);
    }

    if (crc_match) { ++m.certified; continue; }
    ++m.crc_mismatches;

    uint64_t reps = (u < rep_per_unit.size()) ? rep_per_unit[u] : 1;

    // After the vote, recompute CRC on voted data.
    // If it matches clean CRC, the vote correctly repaired the data.
    // If not (common-mode failure: all 3 replicas equally corrupted),
    // the vote succeeded but produced wrong data.

    if (reps >= 3) {
      ++m.repair_invoked;
      // Perform vote and build voted buffer for CRC verification
      std::vector<uint8_t> voted_data(ub);
      bool all_voted = true;
      for (uint64_t b = 0; b < ub; ++b) {
        if (!majority_vote_3(replicas[0][off+b], replicas[1][off+b],
                             replicas[2][off+b], voted_data[b]))
          { all_voted = false; break; }
      }
      if (all_voted) {
        // Verify: CRC of voted data must match clean CRC
        uint32_t voted_crc = crc32c(voted_data.data(), ub);
        if (voted_crc == clean_crc[u]) {
          ++m.repair_success; ++m.certified;
        } else {
          // Vote succeeded but produced wrong data (common-mode failure)
          ++m.fallback_invoked; ++m.uncertified;
        }
      } else {
        ++m.fallback_invoked; ++m.uncertified;
      }
    } else if (reps == 2) { ++m.fallback_invoked; ++m.uncertified; }
    else { ++m.fallback_invoked; ++m.uncertified; }
  }
  return m;
}

struct CI95 { double mean, low, high; };
static CI95 compute_ci95(const std::vector<double> &s) {
  CI95 ci{}; size_t n = s.size(); if(n<2) return ci;
  double sum=0,s2=0; for(auto v:s){sum+=v;s2+=v*v;}
  ci.mean=sum/n; double var=(s2-sum*sum/n)/(n-1); double sd=std::sqrt(var);
  double se=sd/std::sqrt(double(n));
  static const double t29=2.045229642132703;
  ci.low=ci.mean-t29*se; ci.high=ci.mean+t29*se;
  return ci;
}

// ===================================================================
// Main
// ===================================================================
int main(int argc, char **argv) {
  init_crc();

  std::string dataset_path, raw_path, csv_path="x2_diversity_canonical.csv",
              summary_path="x2_factorized_summary.csv", handoff_path="handoff.json";
  int n_seeds = 30;
  for (int i = 1; i < argc; ++i) {
    std::string_view a(argv[i]); auto val=[&](){if(++i>=argc)die("missing");return std::string(argv[i]);};
    if (a=="--help"||a=="-h") {
      std::fprintf(stderr,"Usage: %s --dataset PATH --raw PATH [options]\n",argv[0]); return 0;
    } else if (a=="--dataset") dataset_path=val();
    else if (a=="--raw") raw_path=val();
    else if (a=="--csv") csv_path=val();
    else if (a=="--summary-csv") summary_path=val();
    else if (a=="--handoff") handoff_path=val();
    else if (a=="--n-seeds") n_seeds=std::stoi(val());
    else { std::string m="unknown: ";m+=a;die(m.c_str()); }
  }
  if (dataset_path.empty()) die("--dataset required");

  Dataset dataset = exp3_real::load_dataset(dataset_path);
  uint64_t pb = dataset.planes.empty() ? 0 : dataset.planes[0].size();
  uint64_t n_planes = dataset.manifest.max_plane_count;
  uint64_t nu = n_units(pb);
  const auto &clean_src = dataset.planes[0];

  std::fprintf(stderr,"[x2] dataset=%s n=%" PRIu64 " plane_bytes=%" PRIu64 " units=%" PRIu64 "\n",
      dataset.manifest.dataset.c_str(), dataset.manifest.value_count, pb, nu);

  // Pre-compute clean CRC
  std::vector<uint32_t> clean_crc(nu);
  for(uint64_t u=0;u<nu;++u){uint64_t ub=ubytes(u,pb);if(ub>0)clean_crc[u]=crc32c(clean_src.data()+u*ALLOC_UNIT,ub);}

  // Pre-compute graded B=3 manifest
  std::vector<AllocUnit> alloc; fill_graded(alloc, 3, n_planes, pb);
  std::vector<uint64_t> rep_per_unit(nu, 1);
  for(auto &a : alloc) if(a.plane_id==0 && a.unit_id<nu) rep_per_unit[a.unit_id]=a.replicas;

  // Sweep dimensions
  std::vector<DiversityMode> divs = {DiversityMode::D0, DiversityMode::D1, DiversityMode::D2, DiversityMode::D3};
  std::vector<FaultMode> fmodes = {FaultMode::Cluster, FaultMode::Temporal};
  std::vector<double> rates = {1e-7, 1e-6, 1e-5};

  // Output CSV
  std::FILE *fcsv = std::fopen(csv_path.c_str(), "w");
  if (!fcsv) die("cannot write CSV");
  std::fprintf(fcsv, "dataset,diversity,fault_mode,fault_rate,fault_seed,"
      "crc_mismatches,repair_invoked,repair_success,fallback_invoked,certified,uncertified,"
      "repair_rate,fallback_rate,certified_rate,uncertified_rate\n");

  // Per (diversity, mode, rate) storage for factorized criterion
  struct CellKey { DiversityMode d; FaultMode m; double r;
    bool operator<(const CellKey &o) const {
      if (d != o.d) return d < o.d;
      if (m != o.m) return m < o.m;
      return r < o.r;
    }
  };
  std::map<CellKey, std::vector<double>> certified_rates;

  // Replicas
  std::vector<std::vector<uint8_t>> replicas(3, clean_src);
  std::vector<uint8_t> clean_copy(clean_src);

  for (auto dm : divs) {
    auto place = placement_for(dm, 3);
    for (auto fm : fmodes) {
      for (auto rate : rates) {
        std::fprintf(stderr, "[x2] %s %s rate=%.0e\n", div_str(dm), fm_str(fm), rate);
        for (int s = 0; s < n_seeds; ++s) {
          uint64_t seed = static_cast<uint64_t>(s);

          // Reset replicas
          for (int r = 0; r < 3; ++r)
            std::memcpy(replicas[r].data(), clean_copy.data(), pb);

          // Inject diversity-aware faults
          AffectedMap am = apply_diversity_faults(
              replicas.data(), 3, pb, place, fm, rate, seed);

          // Run trial with CRC optimization
          TrialMetrics m = run_trial_diversity(
              pb, clean_copy.data(), replicas.data(), clean_crc, rep_per_unit, am);

          double tu = static_cast<double>(nu);
          double rpr = tu>0?m.repair_invoked/tu:0, fbr = tu>0?m.fallback_invoked/tu:0;
          double crt = tu>0?m.certified/tu:0, uct = tu>0?m.uncertified/tu:0;

          std::fprintf(fcsv, "%s,%s,%s,%.0e,%" PRIu64 ","
              "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
              "%.6e,%.6e,%.6e,%.6e\n",
              dataset.manifest.dataset.c_str(), div_str(dm), fm_str(fm), rate, seed,
              m.crc_mismatches, m.repair_invoked, m.repair_success, m.fallback_invoked,
              m.certified, m.uncertified,
              rpr, fbr, crt, uct);

          certified_rates[{dm, fm, rate}].push_back(crt);
        }
      }
    }
  }
  std::fclose(fcsv);
  std::fprintf(stderr, "[x2] CSV: %s\n", csv_path.c_str());

  // ===================================================================
  // Factorized criterion
  // ===================================================================
  std::FILE *fsum = std::fopen(summary_path.c_str(), "w");
  if (!fsum) die("cannot write summary");
  std::fprintf(fsum, "fault_mode,fault_rate,group,mean_certified_rate,ci95_low,ci95_high\n");

  auto crit_fn = [](FaultMode fm) -> std::pair<std::vector<DiversityMode>, std::vector<DiversityMode>> {
    if (fm == FaultMode::Cluster)
      return {{DiversityMode::D1, DiversityMode::D3}, {DiversityMode::D0, DiversityMode::D2}};
    else
      return {{DiversityMode::D2, DiversityMode::D3}, {DiversityMode::D0, DiversityMode::D1}};
  };

  bool criterion_holds_cluster = true, criterion_holds_temporal = true;

  for (auto fm : fmodes) {
    auto [good, bad] = crit_fn(fm);
    for (auto rate : rates) {
      std::vector<double> gd, bd;
      for (auto d : good) {
        auto it = certified_rates.find({d, fm, rate});
        if (it != certified_rates.end())
          gd.insert(gd.end(), it->second.begin(), it->second.end());
      }
      for (auto d : bad) {
        auto it = certified_rates.find({d, fm, rate});
        if (it != certified_rates.end())
          bd.insert(bd.end(), it->second.begin(), it->second.end());
      }

      CI95 gci = compute_ci95(gd), bci = compute_ci95(bd);
      bool passes = (gci.low > bci.high); // good group strictly better (non-overlapping CI)

      std::fprintf(fsum, "%s,%.0e,good(%s),%.6f,%.6f,%.6f\n",
          fm_str(fm), rate, (fm==FaultMode::Cluster?"D1+D3":"D2+D3"),
          gci.mean, gci.low, gci.high);
      std::fprintf(fsum, "%s,%.0e,bad(%s),%.6f,%.6f,%.6f\n",
          fm_str(fm), rate, (fm==FaultMode::Cluster?"D0+D2":"D0+D1"),
          bci.mean, bci.low, bci.high);
      std::fprintf(fsum, "%s,%.0e,passes,%s\n",
          fm_str(fm), rate, passes ? "true" : "false");

      if (fm == FaultMode::Cluster && !passes) criterion_holds_cluster = false;
      if (fm == FaultMode::Temporal && !passes) criterion_holds_temporal = false;
    }
  }

  bool criterion_overall = criterion_holds_cluster && criterion_holds_temporal;
  std::fclose(fsum);
  std::fprintf(stderr, "\n=== Factorized Criterion ===\n");
  std::fprintf(stderr, "  cluster: D1/D3 > D0/D2 → %s\n", criterion_holds_cluster ? "PASS" : "FAIL");
  std::fprintf(stderr, "  temporal: D2/D3 > D0/D1 → %s\n", criterion_holds_temporal ? "PASS" : "FAIL");
  std::fprintf(stderr, "  Combined → %s\n", criterion_overall ? "PASS" : "FAIL");

  const char *verdict = criterion_overall ? "PROCEED_TO_PARETO" : "STOP_DIVERSITY_UNSUPPORTED";

  // Handoff JSON
  std::FILE *fh = std::fopen(handoff_path.c_str(), "w");
  if (fh) {
    const char *jid = std::getenv("SLURM_JOB_ID"); if(!jid) jid="NO_JOB";
    std::fprintf(fh,"{\"phase\":\"3-X2\",\"job_id\":\"%s\",\"dataset\":\"%s\""
        ",\"cluster_passes\":%s,\"temporal_passes\":%s"
        ",\"criterion_passes\":%s,\"verdict\":\"%s\"}\n",
        jid, dataset.manifest.dataset.c_str(),
        criterion_holds_cluster?"true":"false",
        criterion_holds_temporal?"true":"false",
        criterion_overall?"true":"false", verdict);
    std::fclose(fh);
  }

  std::fprintf(stderr, "\n  Verdict: %s\n", verdict);
  return criterion_overall ? 0 : 1;
}
