// NMR-C v2: K-Aware Policy Frontier Benchmark (k-sweep + graded/uniform)
//
// Measures under k=1,2,4,all sweep with per-policy replica protection:
//   P0_baseline_sum        → clean_k_answer (no faults, no vote)
//   P2_fused_vote_inline   → faulted_k_answer (policy replica + faults)
//   P4_raw_fp64_reference  → truth_raw (raw FP64 oracle)
//
// Policies: graded_B0..B16, uniform_full_r1/r2/r3
// Fault plans: external .fplan files (F1-F8 families)
//
// Each row outputs: truth_raw, clean_k_answer, faulted_k_answer
// plus error metrics and storage accounting.
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
//        && make bench_nmr_c_v2_k_sweep
// Run:   ./bench_nmr_c_v2_k_sweep --dataset PATH --raw PATH
//        --policy graded_B8 --fault-plan PATH [--k 1,2,4,all] [--csv PATH]

#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <chrono>
#include <string>
#include <string_view>
#include <vector>

#include "../experiment3/exp3_real_data_layout.hpp"
#include "nmr_c_v2_kernels.cuh"

using namespace exp3_real;

// ===========================================================================
// Error handling
// ===========================================================================
[[noreturn]] static void die(const char *m) { std::fprintf(stderr, "FATAL: %s\n", m); std::exit(2); }
static void cuda_check(cudaError_t e, const char *w) {
  if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error at %s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}

// ===========================================================================
// GPU timing via CUDA events
// ===========================================================================
static double measure_ms(std::function<void(cudaStream_t)> fn, int iters, cudaStream_t s) {
  for (int i = 0; i < 3; i++) fn(s);
  cuda_check(cudaStreamSynchronize(s), "warmup sync");
  cudaEvent_t start, stop;
  cuda_check(cudaEventCreate(&start), "ce"); cuda_check(cudaEventCreate(&stop), "ce");
  cuda_check(cudaEventRecord(start, s), "rec");
  for (int i = 0; i < iters; i++) fn(s);
  cuda_check(cudaEventRecord(stop, s), "rec"); cuda_check(cudaEventSynchronize(stop), "sync");
  float ms = 0;
  cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
  cuda_check(cudaEventDestroy(start), "d"); cuda_check(cudaEventDestroy(stop), "d");
  return ms / static_cast<double>(iters);
}

// ===========================================================================
// CPU reduction of device block arrays
// ===========================================================================
static uint64_t reduce_counts(const uint64_t *d_counts, uint64_t grid) {
  std::vector<uint64_t> h(grid);
  cuda_check(cudaMemcpy(h.data(), d_counts, grid * sizeof(uint64_t), cudaMemcpyDeviceToHost), "cpy counts");
  uint64_t total = 0;
  for (uint64_t i = 0; i < grid; ++i) total += h[i];
  return total;
}

static double reduce_sums(const double *d_sums, uint64_t grid) {
  std::vector<double> h(grid);
  cuda_check(cudaMemcpy(h.data(), d_sums, grid * sizeof(double), cudaMemcpyDeviceToHost), "cpy sums");
  double total = 0.0;
  for (uint64_t i = 0; i < grid; ++i) total += h[i];
  return total;
}

static std::vector<uint32_t> reduce_digests(const uint32_t *d_digests, int k, uint64_t grid) {
  std::vector<uint32_t> h(static_cast<size_t>(k) * grid);
  cuda_check(cudaMemcpy(h.data(), d_digests, static_cast<size_t>(k) * grid * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost), "cpy digests");
  std::vector<uint32_t> result(static_cast<size_t>(k), 0);
  for (int p = 0; p < k; ++p)
    for (uint64_t b = 0; b < grid; ++b)
      result[static_cast<size_t>(p)] += h[p * grid + b];
  return result;
}

// ===========================================================================
// Path label helpers
// ===========================================================================
enum class Path {
  P0, P1, P2, P3, P4,
};

static const char* path_label(Path p) {
  switch (p) {
    case Path::P0: return "P0_baseline_sum";
    case Path::P1: return "P1_fused_digest_inline";
    case Path::P2: return "P2_fused_vote_inline_all_read_planes";
    case Path::P3: return "P3_fused_vote_digest_inline_all_read_planes";
    case Path::P4: return "P4_raw_fp64_reference";
  }
  return "UNKNOWN";
}

static Path parse_path(const char *s) {
  std::string_view sv(s);
  if (sv == "P0") return Path::P0;
  if (sv == "P1") return Path::P1;
  if (sv == "P2") return Path::P2;
  if (sv == "P3") return Path::P3;
  if (sv == "P4") return Path::P4;
  die("unknown path (use P0,P1,P2,P3,P4)");
  return Path::P0;
}

// ===========================================================================
// Parse k values: comma-separated, supports "all"
// ===========================================================================
static std::vector<int> parse_k_sweep(const char *s, int max_planes) {
  std::vector<int> result;
  std::string tok;
  std::string_view sv(s);
  size_t pos = 0;
  while (pos <= sv.size()) {
    if (pos == sv.size() || sv[pos] == ',') {
      if (!tok.empty()) {
        if (tok == "all") result.push_back(max_planes);
        else {
          int v = std::stoi(tok);
          if (v < 1) die("k must be >= 1");
          result.push_back(v);
        }
        tok.clear();
      }
    } else {
      tok += sv[pos];
    }
    ++pos;
  }
  return result;
}

// ===========================================================================
// Policy parsing helpers (graded / uniform, from Issue #303)
// ===========================================================================
static std::vector<int> make_graded_extras(int B) {
  std::vector<int> extra(8, 0);
  int remaining = B;
  for (int p = 0; p < 8 && remaining > 0; ++p) {
    int give = remaining < 2 ? remaining : 2;
    extra[p] = give;
    remaining -= give;
  }
  return extra;
}

static std::vector<int> make_uniform_extras(int B) {
  int base_extra = B / 8;
  int rem = B % 8;
  std::vector<int> extra(8);
  for (int i = 0; i < 8; ++i)
    extra[i] = base_extra + (i < rem ? 1 : 0);
  return extra;
}

// Returns r_vector[p] = total replicas (1 + extras) per plane p
static std::vector<int> parse_policy(const std::string &name) {
  std::vector<int> extras(8, 0);
  if (name.rfind("graded_B", 0) == 0) {
    int B = std::stoi(name.substr(8));
    extras = make_graded_extras(B);
  } else if (name == "uniform_full_r1") {
    extras = make_uniform_extras(0);
  } else if (name == "uniform_full_r2") {
    extras = make_uniform_extras(8);
  } else if (name == "uniform_full_r3") {
    extras = make_uniform_extras(16);
  } else {
    die("unknown policy (use graded_B0..B16 or uniform_full_r1/r2/r3)");
  }
  std::vector<int> r_vector(8);
  for (int p = 0; p < 8; ++p) r_vector[p] = 1 + extras[p];
  return r_vector;
}

static const char* policy_type(const std::string &name) {
  if (name.rfind("graded_", 0) == 0) return "graded";
  return "uniform";
}

// ===========================================================================
// Fault injection helpers
// ===========================================================================
enum class FaultFamily { NONE, F2, F4, EXTERNAL };

struct FaultEntry {
  uint64_t offset;
  uint8_t  mask;
};

// GPU kernel: apply fault entries (XOR mask at offset)
__global__ void inject_fault_entries_kernel(uint8_t *buf, const FaultEntry *entries, uint64_t n_entries, uint64_t plane_bytes) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_entries) {
    uint64_t off = entries[i].offset;
    if (off < plane_bytes) buf[off] ^= entries[i].mask;
  }
}

// GPU kernel: inject F2 (localized multi-bit corruption) into a plane buffer
__global__ void inject_f2_fault(uint8_t *buf, uint64_t offset, uint64_t length) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) buf[offset + i] = ~buf[offset + i];
}

// GPU kernel: inject F4 (column-like repeated offset) into a plane buffer
__global__ void inject_f4_fault(uint8_t *buf, uint64_t stride, uint64_t offset, uint64_t count, uint64_t n) {
  uint64_t s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < count) {
    uint64_t pos = offset + s * stride;
    if (pos < n) buf[pos] = ~buf[pos];
  }
}

struct FaultPlanMeta {
  std::string family = "none";
  double rate = 0.0;
  int seed = 0;
};

// Parse .fplan text file: FAMILY=X RATE=Y SEED=Z N=W MAXP=V ENTRIES=U [K=W] on line 1,
// followed by "plane replica offset mask" per line.
// Returns per-plane-per-replica entry arrays.  plane_entries[p * 3 + r] for
// planes 0..MAXP-1 and replicas 0..2.
static FaultPlanMeta load_fault_plan(
    const std::string &path,
    int max_planes,
    std::vector<std::vector<FaultEntry>> &plane_entries)  // indexed by p*3+r
{
  FaultPlanMeta meta;
  plane_entries.clear();
  plane_entries.resize(static_cast<size_t>(max_planes) * 3);

  std::ifstream in(path);
  if (!in) die("cannot open fault plan file");

  std::string line;
  // Line 1: metadata header
  if (!std::getline(in, line)) die("fault plan file empty");
  {
    auto parse_key = [&](const char *key) -> std::string {
      auto pos = line.find(key);
      if (pos == std::string::npos) return "";
      pos += std::strlen(key);
      auto end = line.find(' ', pos);
      if (end == std::string::npos) end = line.size();
      return line.substr(pos, end - pos);
    };
    meta.family = parse_key("FAMILY=");
    auto rate_s = parse_key("RATE=");
    if (!rate_s.empty()) meta.rate = std::stod(rate_s);
    auto seed_s = parse_key("SEED=");
    if (!seed_s.empty()) meta.seed = std::stoi(seed_s);
  }

  // Parse optional ENTRIES=N from header for cross-validation
  uint64_t header_entries = UINT64_MAX;
  {
    auto pos = line.find("ENTRIES=");
    if (pos != std::string::npos) {
      pos += std::strlen("ENTRIES=");
      auto end = line.find(' ', pos);
      if (end == std::string::npos) end = line.size();
      header_entries = std::stoull(line.substr(pos, end - pos));
    }
  }

  // Subsequent lines: plane replica offset mask
  uint64_t lines_read = 0, skipped_parse = 0;
  uint64_t skipped_plane = 0, skipped_replica = 0, skipped_mask = 0;
  uint64_t total_parsed = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    lines_read++;
    int plane = 0, replica = 0;
    uint64_t offset = 0;
    int mask = 0;
    if (std::sscanf(line.c_str(), "%d %d %" SCNu64 " %d", &plane, &replica, &offset, &mask) != 4) {
      skipped_parse++;
      continue;
    }
    if (plane < 0 || plane >= max_planes) { skipped_plane++; continue; }
    if (replica < 0 || replica > 2)       { skipped_replica++; continue; }
    if (mask < 1 || mask > 255)           { skipped_mask++; continue; }
    FaultEntry fe{offset, static_cast<uint8_t>(mask)};
    plane_entries[static_cast<size_t>(plane) * 3 + static_cast<size_t>(replica)].push_back(fe);
    total_parsed++;
  }

  // Report entry cross-validation against header
  if (header_entries != UINT64_MAX && header_entries != total_parsed) {
    std::fprintf(stderr, "WARNING: fault plan %s header ENTRIES=%" PRIu64
                 " but parsed %" PRIu64 " valid entries\n",
                 path.c_str(), header_entries, total_parsed);
  }
  uint64_t total_skipped = skipped_parse + skipped_plane + skipped_replica + skipped_mask;
  if (total_skipped > 0) {
    std::fprintf(stderr, "WARNING: fault plan %s skipped %" PRIu64
                 " of %" PRIu64 " data lines (parse=%" PRIu64
                 " plane=%" PRIu64 " replica=%" PRIu64 " mask=%" PRIu64 ")\n",
                 path.c_str(), total_skipped, lines_read,
                 skipped_parse, skipped_plane, skipped_replica, skipped_mask);
  }
  return meta;
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char **argv) {
  std::string dataset_path, raw_path, csv_path = "nmr_c_v2_k_sweep.csv";
  std::string k_sweep_str = "1,2,4,all";
  std::string fault_plan_path;
  std::string policy_name;
  double threshold_val = 0.0;
  int iters = 50, block_threads = 256;
  FaultFamily fault_family = FaultFamily::NONE;
  int fault_p0 = 0, fault_p1 = 0;  // F2: offset, length; F4: stride, count
  bool run_all_paths = true;
  Path single_path = Path::P0;
  bool policy_mode = false;

  for (int i = 1; i < argc; ++i) {
    std::string_view a(argv[i]);
    auto val = [&]() { if (++i >= argc) die("missing value"); return std::string(argv[i]); };
    if (a == "--help" || a == "-h") {
      std::fprintf(stderr,
        "NMR-C v2: K-Aware Policy Frontier Benchmark (k-sweep + graded/uniform)\n\n"
        "Usage: %s --dataset PATH --raw PATH [options]\n\n"
        "Options:\n"
        "  --dataset PATH     Encoded dataset dir (required)\n"
        "  --raw PATH         Raw float64 file (required)\n"
        "  --policy NAME      Policy: graded_B0..B16 or uniform_full_r1/r2/r3\n"
        "  --fault-plan PATH  External .fplan file for F1-F8 families (required with --policy)\n"
        "  --path P0-P4       Run single path (default: all paths)\n"
        "  --k 1,2,4,all      Comma-separated k values (default: 1,2,4,all)\n"
        "  --threshold FP64   Threshold value (default: 0.0 = all rows)\n"
        "  --iters N          Iterations per measurement (default: 50)\n"
        "  --block-threads N  Threads per block (default: 256)\n"
        "  --fault-family F2|F4  Fault family for mechanism sanity (default: none)\n"
        "  --fault-params O,L    F2: offset,length  F4: stride,count\n"
        "  --csv PATH         Output CSV file (default: nmr_c_v2_k_sweep.csv)\n",
        argv[0]);
      return 0;
    } else if (a == "--dataset")     dataset_path = val();
    else if (a == "--raw")           raw_path = val();
    else if (a == "--path")          { single_path = parse_path(val().c_str()); run_all_paths = false; }
    else if (a == "--k")             k_sweep_str = val();
    else if (a == "--threshold")     threshold_val = std::stod(val());
    else if (a == "--iters")         iters = std::stoi(val());
    else if (a == "--block-threads") block_threads = std::stoi(val());
    else if (a == "--fault-family") {
      std::string f(val());
      if (f == "F2") fault_family = FaultFamily::F2;
      else if (f == "F4") fault_family = FaultFamily::F4;
      else die("unknown fault family (use F2 or F4)");
    }
    else if (a == "--fault-params") {
      std::string s(val());
      auto comma = s.find(',');
      if (comma == std::string::npos) die("--fault-params requires O,L or S,C");
      fault_p0 = std::stoi(s.substr(0, comma));
      fault_p1 = std::stoi(s.substr(comma + 1));
      if (fault_p0 < 0 || fault_p1 < 0)
        die("--fault-params values must be >= 0 (unsigned semantics)");
    }
    else if (a == "--policy")       { policy_name = val(); policy_mode = true; }
    else if (a == "--fault-plan")   fault_plan_path = val();
    else if (a == "--csv")           csv_path = val();
    else { std::string msg = "unknown: "; msg += a; die(msg.c_str()); }
  }
  if (dataset_path.empty()) die("--dataset required");
  if (raw_path.empty()) die("--raw required");
  if (policy_mode) {
    if (fault_plan_path.empty()) die("--policy requires --fault-plan");
    if (fault_family != FaultFamily::NONE)
      die("--policy cannot be used with --fault-family F2/F4 (use --fault-plan)");
  }

  cuda_check(cudaSetDevice(0), "dev");
  cudaDeviceProp prop;
  cuda_check(cudaGetDeviceProperties(&prop, 0), "prop");
  std::fprintf(stderr, "Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  // ---- Load external fault plan ----
  FaultPlanMeta fault_plan_meta;
  std::vector<std::vector<FaultEntry>> fault_plan_entries;
  std::vector<FaultEntry*> d_fault_bufs;

  // Load dataset
  Dataset ds = exp3_real::load_dataset(dataset_path);
  uint64_t n = ds.manifest.value_count;
  uint64_t max_planes = ds.manifest.max_plane_count;
  int PLANES = static_cast<int>(max_planes);
  std::string ds_name = ds.manifest.dataset;

  if (!fault_plan_path.empty()) {
    if (fault_family != FaultFamily::NONE)
      die("cannot use both --fault-family and --fault-plan");
    fault_family = FaultFamily::EXTERNAL;
    fault_plan_meta = load_fault_plan(fault_plan_path, PLANES, fault_plan_entries);
    std::fprintf(stderr, "Fault plan: family=%s rate=%.6e seed=%d\n",
                 fault_plan_meta.family.c_str(), fault_plan_meta.rate, fault_plan_meta.seed);

    d_fault_bufs.resize(static_cast<size_t>(PLANES) * 3, nullptr);
    for (int p = 0; p < PLANES; ++p) {
      for (int r = 0; r < 3; ++r) {
        const auto &entries = fault_plan_entries[static_cast<size_t>(p) * 3 + static_cast<size_t>(r)];
        if (entries.empty()) continue;
        size_t bytes = entries.size() * sizeof(FaultEntry);
        cuda_check(cudaMalloc(&d_fault_bufs[static_cast<size_t>(p) * 3 + static_cast<size_t>(r)], bytes),
                   "M fault entries");
        cuda_check(cudaMemcpy(d_fault_bufs[static_cast<size_t>(p) * 3 + static_cast<size_t>(r)],
                    entries.data(), bytes, cudaMemcpyHostToDevice), "H2D fault entries");
      }
    }
  }

  std::fprintf(stderr, "Dataset: %s n=%" PRIu64 " planes=%d\n", ds_name.c_str(), n, PLANES);

  // Parse k sweep
  std::vector<int> k_values = parse_k_sweep(k_sweep_str.c_str(), PLANES);
  // Clamp k to max planes
  for (auto &kv : k_values) if (kv > PLANES) kv = PLANES;

  // Path list
  std::vector<Path> paths;
  if (run_all_paths) paths = {Path::P0, Path::P1, Path::P2, Path::P3, Path::P4};
  else paths = {single_path};

  // Upload all planes to GPU
  std::vector<uint8_t*> d_planes(static_cast<size_t>(PLANES));
  for (int p = 0; p < PLANES; p++) {
    cuda_check(cudaMalloc(&d_planes[static_cast<size_t>(p)], ds.planes[static_cast<size_t>(p)].size()), "M plane");
    cuda_check(cudaMemcpy(d_planes[static_cast<size_t>(p)], ds.planes[static_cast<size_t>(p)].data(),
                ds.planes[static_cast<size_t>(p)].size(), cudaMemcpyHostToDevice), "H2D plane");
  }

  // Load raw FP64 data to GPU (for P4)
  double *d_raw = nullptr;
  uint64_t raw_count = 0;
  {
    uint64_t file_size = std::filesystem::file_size(raw_path);
    raw_count = file_size / sizeof(double);
    cuda_check(cudaMalloc(&d_raw, file_size), "M raw");
    std::ifstream raw_in(raw_path, std::ios::binary);
    if (!raw_in) die("cannot open raw file");
    constexpr size_t CHUNK = 256ull * 1024ull * 1024ull;
    std::vector<double> chunk(CHUNK / sizeof(double));
    uint64_t remaining = file_size;
    uint64_t offset = 0;
    while (remaining > 0) {
      size_t to_read = static_cast<size_t>(std::min<uint64_t>(remaining, CHUNK));
      raw_in.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(to_read));
      cuda_check(cudaMemcpy(reinterpret_cast<uint8_t*>(d_raw) + offset,
                  chunk.data(), to_read, cudaMemcpyHostToDevice), "H2D raw chunk");
      offset += to_read;
      remaining -= to_read;
    }
  }

  cudaStream_t st;
  cuda_check(cudaStreamCreate(&st), "st");

  // Open CSV
  bool csv_exists = std::filesystem::exists(csv_path);
  FILE *fcsv = std::fopen(csv_path.c_str(), "a");
  if (!fcsv) die("cannot open CSV for append");
  if (!csv_exists || std::ftell(fcsv) == 0) {
    if (policy_mode) {
      std::fprintf(fcsv, "dataset,n_rows,max_planes,k_values,policy_name,policy_type,"
                          "truth_raw,clean_k_answer,faulted_k_answer,"
                          "error_to_truth_abs,error_to_truth_rel,"
                          "error_vs_clean_k_abs,error_vs_clean_k_rel,"
                          "total_materialized_B,active_prefix_B,"
                          "r_vector,"
                          "fault_family,fault_rate,seed,"
                          "gpu_count,latency_ms,effective_bytes\n");
    } else {
      std::fprintf(fcsv, "dataset,n_rows,max_planes,k_values,path,latency_ms,effective_bytes,"
                          "effective_bandwidth_gb_s,gpu_count,gpu_sum,"
                          "cpu_count,cpu_sum,"
                          "replica_materialization_mode,protected_plane_count,"
                          "\"protected_plane_list\",fault_family,fault_params,"
                          "fault_rate,seed\n");
    }
  }

  // Precompute per-segment arrays from dataset metadata
  uint64_t segment_rows = ds.manifest.segment_size;
  uint64_t n_segments = ds.segments.size();
  std::vector<double> h_seg_base(n_segments);
  std::vector<double> h_seg_basis(n_segments * max_planes, 0.0);
  std::vector<uint32_t> h_seg_active(n_segments);
  std::vector<uint8_t> h_seg_threshold(n_segments * max_planes, 0);
  for (uint64_t s = 0; s < n_segments; ++s) {
    const auto &seg = ds.segments[static_cast<size_t>(s)];
    h_seg_base[static_cast<size_t>(s)] = seg.segment_base;
    h_seg_active[static_cast<size_t>(s)] = seg.active_plane_count;
    for (uint32_t p = 0; p < seg.active_plane_count; ++p)
      h_seg_basis[s * max_planes + p] = (p < seg.plane_basis.size()) ? seg.plane_basis[p] : 0.0;
    // threshold_bytes = 0 for threshold=0 (clean_no_fault, include all rows)
  }

  double *d_seg_base = nullptr;
  cuda_check(cudaMalloc(&d_seg_base, n_segments * sizeof(double)), "M seg_base");
  cuda_check(cudaMemcpy(d_seg_base, h_seg_base.data(), n_segments * sizeof(double),
              cudaMemcpyHostToDevice), "H2D seg_base");

  double *d_basis = nullptr;
  cuda_check(cudaMalloc(&d_basis, n_segments * max_planes * sizeof(double)), "M basis");
  cuda_check(cudaMemcpy(d_basis, h_seg_basis.data(), n_segments * max_planes * sizeof(double),
              cudaMemcpyHostToDevice), "H2D basis");

  uint32_t *d_seg_active = nullptr;
  cuda_check(cudaMalloc(&d_seg_active, n_segments * sizeof(uint32_t)), "M seg_active");
  cuda_check(cudaMemcpy(d_seg_active, h_seg_active.data(), n_segments * sizeof(uint32_t),
              cudaMemcpyHostToDevice), "H2D seg_active");

  uint8_t *d_threshold = nullptr;
  cuda_check(cudaMalloc(&d_threshold, n_segments * max_planes), "M thresh");
  cuda_check(cudaMemcpy(d_threshold, h_seg_threshold.data(), n_segments * max_planes,
              cudaMemcpyHostToDevice), "H2D thresh");

  uint32_t *d_active = d_seg_active;  // reset per-k; points to per-segment active plane counts

  // =======================================================================
  // Run the matrix
  // =======================================================================
  for (int k : k_values) {
    std::fprintf(stderr, "\n=== k=%d ===\n", k);

    int T = block_threads;
    uint64_t tiles_per_segment = (segment_rows + (static_cast<uint64_t>(T) * N2_ROWPACK16_WIDTH) - 1)
        / (static_cast<uint64_t>(T) * N2_ROWPACK16_WIDTH);
    uint64_t grid = tiles_per_segment * n_segments;
    uint64_t grid_raw = (raw_count + static_cast<uint64_t>(T) - 1) / static_cast<uint64_t>(T);

    // Per-segment active plane count = min(k, actual active planes)
    std::vector<uint32_t> h_active_k(n_segments);
    for (uint64_t s = 0; s < n_segments; ++s)
      h_active_k[static_cast<size_t>(s)] = std::min(static_cast<uint32_t>(k), h_seg_active[static_cast<size_t>(s)]);
    uint32_t *d_active_k = nullptr;
    cuda_check(cudaMalloc(&d_active_k, n_segments * sizeof(uint32_t)), "M active_k");
    cuda_check(cudaMemcpy(d_active_k, h_active_k.data(), n_segments * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "H2D active_k");
    d_active = d_active_k;

    // Block counts/sums
    uint64_t *d_counts = nullptr;
    double *d_sums = nullptr;
    cuda_check(cudaMalloc(&d_counts, grid * sizeof(uint64_t)), "M counts");
    cuda_check(cudaMalloc(&d_sums, grid * sizeof(double)), "M sums");

    // Raw FP64 counts/sums (for P4)
    uint64_t *d_raw_counts = nullptr;
    double *d_raw_sums = nullptr;
    cuda_check(cudaMalloc(&d_raw_counts, grid_raw * sizeof(uint64_t)), "M rc");
    cuda_check(cudaMalloc(&d_raw_sums, grid_raw * sizeof(double)), "M rs");

    // Digest buffer (for P1/P3: k planes * grid blocks)
    uint32_t *d_digests = nullptr;
    cuda_check(cudaMalloc(&d_digests, static_cast<size_t>(k) * grid * sizeof(uint32_t)), "M digests");

    // ---- Policy / r-per-plane setup ----
    std::vector<int> r_vector(PLANES, 3);  // default: r=3 for all planes (legacy mode)
    int total_materialized_B = 0;
    int active_prefix_B = 0;
    if (policy_mode) {
      r_vector = parse_policy(policy_name);
      for (int p = 0; p < k; ++p) {
        int r = r_vector[static_cast<size_t>(p)];
        active_prefix_B += r - 1;  // extra copies on read planes
      }
      for (int p = 0; p < PLANES; ++p) {
        total_materialized_B += r_vector[static_cast<size_t>(p)] - 1;  // extras on all planes
      }
      std::fprintf(stderr, "  Policy: %s total_B=%d prefix_B=%d\n",
                   policy_name.c_str(), total_materialized_B, active_prefix_B);
    }
    // Build r-vector string for CSV
    std::string r_vector_str = "[";
    for (int p = 0; p < PLANES; ++p) {
      if (p > 0) r_vector_str += ",";
      r_vector_str += std::to_string(r_vector[static_cast<size_t>(p)]);
    }
    r_vector_str += "]";

    // Replica sets for P2/P3: policy-driven per-plane copy count
    // r=1 → all 3 replicas same buffer (no extra copies); r=2 → 2 distinct;
    // r=3 → all 3 distinct.  Fault entries target specific replica indices.
    std::vector<uint8_t*> d_r0_bufs(k), d_r1_bufs(k), d_r2_bufs(k);
    N2RuntimeU8Subcolumns h_r0{}, h_r1{}, h_r2{};
    bool has_fault = (fault_family != FaultFamily::NONE);
    const uint64_t plane0_bytes = ds.planes[0].size();
    uint64_t fault_offset = static_cast<uint64_t>(fault_p0);
    uint64_t fault_length = static_cast<uint64_t>(fault_p1);
    if (fault_offset >= plane0_bytes) { fault_offset = 0; fault_length = 0; }
    if (fault_offset + fault_length > plane0_bytes) fault_length = plane0_bytes - fault_offset;

    for (int p = 0; p < k; ++p) {
      uint64_t p_bytes = ds.planes[static_cast<size_t>(p)].size();
      int r = r_vector[static_cast<size_t>(p)];  // total replicas: 1, 2, or 3

      // Allocate r distinct buffers, map into 3 prototype slots
      cuda_check(cudaMalloc(&d_r0_bufs[static_cast<size_t>(p)], p_bytes), "M r0");
      cuda_check(cudaMemcpy(d_r0_bufs[static_cast<size_t>(p)], d_planes[static_cast<size_t>(p)],
                  p_bytes, cudaMemcpyDeviceToDevice), "D2D r0");
      if (r >= 2) {
        cuda_check(cudaMalloc(&d_r1_bufs[static_cast<size_t>(p)], p_bytes), "M r1");
        cuda_check(cudaMemcpy(d_r1_bufs[static_cast<size_t>(p)], d_planes[static_cast<size_t>(p)],
                    p_bytes, cudaMemcpyDeviceToDevice), "D2D r1");
      }
      if (r >= 3) {
        cuda_check(cudaMalloc(&d_r2_bufs[static_cast<size_t>(p)], p_bytes), "M r2");
        cuda_check(cudaMemcpy(d_r2_bufs[static_cast<size_t>(p)], d_planes[static_cast<size_t>(p)],
                    p_bytes, cudaMemcpyDeviceToDevice), "D2D r2");
      }

      // Map ptrs: unused slots alias the best available buffer
      h_r0.ptrs[p] = d_r0_bufs[static_cast<size_t>(p)];
      h_r1.ptrs[p] = (r >= 2) ? d_r1_bufs[static_cast<size_t>(p)] : d_r0_bufs[static_cast<size_t>(p)];
      h_r2.ptrs[p] = (r >= 3) ? d_r2_bufs[static_cast<size_t>(p)] : h_r1.ptrs[p];

      // Fault injection
      if (has_fault) {
        if (fault_family == FaultFamily::EXTERNAL) {
          for (int rep = 0; rep < 3; ++rep) {
            if (rep >= r) continue;  // skip faults targeting non-existent replicas
            auto *d_entries = d_fault_bufs[static_cast<size_t>(p) * 3 + static_cast<size_t>(rep)];
            const auto &h_entries = fault_plan_entries[static_cast<size_t>(p) * 3 + static_cast<size_t>(rep)];
            if (d_entries && !h_entries.empty()) {
              uint64_t n = static_cast<uint64_t>(h_entries.size());
              int blk = static_cast<int>((n + 255) / 256);
              uint8_t *target = (rep == 0) ? d_r0_bufs[static_cast<size_t>(p)] :
                                (rep == 1) ? d_r1_bufs[static_cast<size_t>(p)] : d_r2_bufs[static_cast<size_t>(p)];
              inject_fault_entries_kernel<<<blk, 256>>>(target, d_entries, n, p_bytes);
            }
          }
        } else {
          uint64_t p_off = (fault_offset < p_bytes) ? fault_offset : 0;
          uint64_t p_len = (p_off + fault_length <= p_bytes) ? fault_length : (p_bytes > p_off ? p_bytes - p_off : 0);
          if (p_len > 0) {
            int fault_blk = static_cast<int>((p_len + 255) / 256);
            if (fault_family == FaultFamily::F2) {
              inject_f2_fault<<<fault_blk, 256>>>(d_r0_bufs[static_cast<size_t>(p)], p_off, p_len);
            } else if (fault_family == FaultFamily::F4) {
              uint64_t stride = static_cast<uint64_t>(fault_p0);
              if (stride == 0) die("F4 fault requires stride > 0");
              uint64_t p_count = (p_bytes + stride - 1) / stride;
              if (static_cast<uint64_t>(fault_p1) < p_count) p_count = static_cast<uint64_t>(fault_p1);
              inject_f4_fault<<<static_cast<int>((p_count + 255) / 256), 256>>>(
                  d_r0_bufs[static_cast<size_t>(p)], stride, 0, p_count, p_bytes);
            }
          }
        }
      }
    }
    if (has_fault) {
      cuda_check(cudaDeviceSynchronize(), "fault sync");
      if (fault_family == FaultFamily::EXTERNAL) {
        std::fprintf(stderr, "  Fault: %s rate=%.6e seed=%d applied to %d planes\n",
                     fault_plan_meta.family.c_str(), fault_plan_meta.rate, fault_plan_meta.seed, k);
      } else {
        std::fprintf(stderr, "  Fault: %s applied to all %d protected planes (replica 0)\n",
                     (fault_family == FaultFamily::F2) ? "F2" : "F4", k);
      }
    }
    std::fprintf(stderr, "  Replicas: policy-driven (r_vector=%s)\n", r_vector_str.c_str());

    // P0 subcolumns (only planes 0..k-1 matter, others can be null)
    N2RuntimeU8Subcolumns h_sc{};
    for (int p = 0; p < PLANES; ++p)
      h_sc.ptrs[p] = d_planes[static_cast<size_t>(p)];

    // =====================================================================
    // CPU oracle: compute count+sum from raw FP64 for correctness check
    // =====================================================================
    std::vector<double> raw_host(raw_count);
    {
      std::ifstream raw_in(raw_path, std::ios::binary);
      raw_in.read(reinterpret_cast<char*>(raw_host.data()),
                  static_cast<std::streamsize>(raw_count * sizeof(double)));
    }
    uint64_t cpu_count = 0;
    double cpu_sum = 0.0;
    for (double val : raw_host) {
      if (val > threshold_val) {
        ++cpu_count;
        cpu_sum += val;
      }
    }

    // Build read plane list string (all paths read 0..k-1)
    // But protected planes only exist for P2/P3 (voting paths).
    // CSV has "name_only" fields; actual protection status is in protected_plane_count.
    std::string read_plane_list;
    for (int p = 0; p < k; ++p) {
      if (p > 0) read_plane_list += ",";
      read_plane_list += std::to_string(p);
    }

    const int plane_stride = static_cast<int>(max_planes);

    if (policy_mode) {
      // ===================================================================
      // Policy mode: run P0 (clean_k) and P2 (faulted_k) once per k
      // ===================================================================
      double truth_raw = cpu_sum;

      // P0: clean-k baseline answer (no fault, no vote)
      {
        auto fn = [&](cudaStream_t s) {
          nmr_v2_p0_baseline_sum<<<static_cast<int>(grid), T, 0, s>>>(
              h_sc, n, segment_rows, tiles_per_segment,
              d_threshold, plane_stride,
              plane_stride, d_active,
              d_seg_base, d_basis, d_counts, d_sums);
        };
        measure_ms(fn, iters, st);
      }
      uint64_t clean_k_count = reduce_counts(d_counts, grid);
      double clean_k_answer = reduce_sums(d_sums, grid);

      // P2: faulted-k answer with policy replicas
      // Upload r_per_plane for kernel
      uint32_t *d_r_per_plane = nullptr;
      {
        std::vector<uint32_t> h_r(PLANES, 3);
        for (int p = 0; p < PLANES; ++p)
          h_r[static_cast<size_t>(p)] = static_cast<uint32_t>(r_vector[static_cast<size_t>(p)]);
        cuda_check(cudaMalloc(&d_r_per_plane, PLANES * sizeof(uint32_t)), "M rpp");
        cuda_check(cudaMemcpy(d_r_per_plane, h_r.data(), PLANES * sizeof(uint32_t),
                              cudaMemcpyHostToDevice), "H2D rpp");
      }
      {
        auto fn = [&](cudaStream_t s) {
          nmr_v2_p2_dispatch(k, h_r0, h_r1, h_r2,
              n, segment_rows, tiles_per_segment,
              d_threshold, plane_stride, plane_stride, d_active,
              d_seg_base, d_basis, d_counts, d_sums,
              d_r_per_plane,
              T, grid, s);
        };
        measure_ms(fn, iters, st);
      }
      cuda_check(cudaFree(d_r_per_plane), "F rpp");
      uint64_t faulted_k_count = reduce_counts(d_counts, grid);
      double faulted_k_answer = reduce_sums(d_sums, grid);

      // Error metrics
      double error_to_truth_abs = std::abs(faulted_k_answer - truth_raw);
      double error_to_truth_rel = (std::abs(truth_raw) > 1.0)
          ? error_to_truth_abs / std::abs(truth_raw) : error_to_truth_abs;
      double error_vs_clean_k_abs = std::abs(faulted_k_answer - clean_k_answer);
      double error_vs_clean_k_rel = (std::abs(clean_k_answer) > 1.0)
          ? error_vs_clean_k_abs / std::abs(clean_k_answer) : error_vs_clean_k_abs;

      // Write policy frontier CSV row
      const char *fault_label = (fault_family == FaultFamily::EXTERNAL)
          ? fault_plan_meta.family.c_str() : "NONE";
      double fault_rate_csv = (fault_family == FaultFamily::EXTERNAL)
          ? fault_plan_meta.rate : 0.0;
      int seed_csv = (fault_family == FaultFamily::EXTERNAL)
          ? fault_plan_meta.seed : -1;

      std::fprintf(fcsv, "%s,%" PRIu64 ",%d,%d,%s,%s,"
                          "%.6f,%.6f,%.6f,"
                          "%.6e,%.6e,"
                          "%.6e,%.6e,"
                          "%d,%d,"
                          "\"%s\","
                          "%s,%.6e,%d,"
                          "%" PRIu64 ",%.6f,%.0f\n",
                   ds_name.c_str(), n, PLANES, k,
                   policy_name.c_str(), policy_type(policy_name),
                   truth_raw, clean_k_answer, faulted_k_answer,
                   error_to_truth_abs, error_to_truth_rel,
                   error_vs_clean_k_abs, error_vs_clean_k_rel,
                   total_materialized_B, active_prefix_B,
                   r_vector_str.c_str(),
                   fault_label, fault_rate_csv, seed_csv,
                   faulted_k_count, 0.0, 0.0);
      std::fflush(fcsv);

      bool count_match = (clean_k_count == cpu_count);
      std::fprintf(stderr, "    truth=%.6f clean=%.6f faulted=%.6f"
                   " err_to_truth=%.6e err_vs_clean=%.6e"
                   " clean_count=%" PRIu64 " cpu_count=%" PRIu64 " match=%s\n",
                   truth_raw, clean_k_answer, faulted_k_answer,
                   error_to_truth_rel, error_vs_clean_k_rel,
                   clean_k_count, cpu_count, count_match ? "OK" : "MISMATCH");
    } else {
      // ===================================================================
      // Legacy timing mode: run all paths
      // ===================================================================
      for (Path path : paths) {
        std::fprintf(stderr, "  Path: %s\n", path_label(path));

        double latency_ms = 0.0;
        double effective_bytes = 0.0;
        uint64_t gpu_count = 0;
        double gpu_sum = 0.0;
        int protected_plane_count = 0;
        std::string replica_mode = "none";

        switch (path) {
        case Path::P0: {
          replica_mode = "none";
          protected_plane_count = 0;
          effective_bytes = static_cast<double>(n) * static_cast<double>(k);
          auto fn = [&](cudaStream_t s) {
            nmr_v2_p0_baseline_sum<<<static_cast<int>(grid), T, 0, s>>>(
                h_sc, n, segment_rows, tiles_per_segment,
                d_threshold, plane_stride,
                plane_stride, d_active,
                d_seg_base, d_basis, d_counts, d_sums);
          };
          latency_ms = measure_ms(fn, iters, st);
          gpu_count = reduce_counts(d_counts, grid);
          gpu_sum = reduce_sums(d_sums, grid);
          break;
        }
        case Path::P1: {
          replica_mode = "none";
          protected_plane_count = 0;
          effective_bytes = static_cast<double>(n) * static_cast<double>(k);
          auto fn = [&](cudaStream_t s) {
            nmr_v2_p1_fused_digest_inline<<<static_cast<int>(grid), T, 0, s>>>(
                h_sc, n, segment_rows, tiles_per_segment,
                d_threshold, plane_stride,
                plane_stride, d_active,
                d_seg_base, d_basis, d_counts, d_sums,
                d_digests, static_cast<int>(grid));
          };
          latency_ms = measure_ms(fn, iters, st);
          gpu_count = reduce_counts(d_counts, grid);
          gpu_sum = reduce_sums(d_sums, grid);
          { auto digests = reduce_digests(d_digests, k, grid);
            std::fprintf(stderr, "    digest plane0=%u\n", digests[0]); }
          break;
        }
        case Path::P2: {
          replica_mode = "identity_hbm";
          protected_plane_count = k;
          effective_bytes = static_cast<double>(n) * static_cast<double>(k) * 3.0;
          // Default r_per_plane = all 3 (legacy behavior)
          uint32_t h_def_rp[32] = {3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3};
          uint32_t *d_def_rp = nullptr;
          cuda_check(cudaMalloc(&d_def_rp, PLANES * sizeof(uint32_t)), "M defrp");
          cuda_check(cudaMemcpy(d_def_rp, h_def_rp, PLANES * sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D defrp");
          auto fn = [&](cudaStream_t s) {
            nmr_v2_p2_dispatch(k, h_r0, h_r1, h_r2,
                n, segment_rows, tiles_per_segment,
                d_threshold, plane_stride, plane_stride, d_active,
                d_seg_base, d_basis, d_counts, d_sums,
                d_def_rp,
                T, grid, s);
          };
          latency_ms = measure_ms(fn, iters, st);
          gpu_count = reduce_counts(d_counts, grid);
          gpu_sum = reduce_sums(d_sums, grid);
          cuda_check(cudaFree(d_def_rp), "F defrp");
          break;
        }
        case Path::P3: {
          replica_mode = "identity_hbm";
          protected_plane_count = k;
          effective_bytes = static_cast<double>(n) * static_cast<double>(k) * 3.0;
          auto fn = [&](cudaStream_t s) {
            nmr_v2_p3_dispatch(k, h_r0, h_r1, h_r2,
                n, segment_rows, tiles_per_segment,
                d_threshold, plane_stride, plane_stride, d_active,
                d_seg_base, d_basis, d_counts, d_sums,
                d_digests, static_cast<int>(grid),
                T, grid, s);
          };
          latency_ms = measure_ms(fn, iters, st);
          gpu_count = reduce_counts(d_counts, grid);
          gpu_sum = reduce_sums(d_sums, grid);
          { auto digests = reduce_digests(d_digests, k, grid);
            std::fprintf(stderr, "    digest plane0=%u\n", digests[0]); }
          break;
        }
        case Path::P4: {
          replica_mode = "none";
          protected_plane_count = 0;
          effective_bytes = static_cast<double>(raw_count) * sizeof(double);
          auto fn = [&](cudaStream_t s) {
            nmr_v2_p4_raw_fp64<<<static_cast<int>(grid_raw), T, 0, s>>>(
                d_raw, raw_count, threshold_val, d_raw_counts, d_raw_sums);
          };
          latency_ms = measure_ms(fn, iters, st);
          gpu_count = reduce_counts(d_raw_counts, grid_raw);
          gpu_sum = reduce_sums(d_raw_sums, grid_raw);
          break;
        }
        }

        double sec = latency_ms / 1000.0;
        double bw_gb_s = (sec > 0.0) ? (effective_bytes / sec / 1e9) : 0.0;

        const char *fault_label = (fault_family == FaultFamily::F2) ? "F2" :
                                  (fault_family == FaultFamily::F4) ? "F4" :
                                  (fault_family == FaultFamily::EXTERNAL) ? fault_plan_meta.family.c_str() : "NONE";
        char fault_params_str[256];
        if (fault_family == FaultFamily::F2 || fault_family == FaultFamily::F4) {
          std::snprintf(fault_params_str, sizeof(fault_params_str), "%d,%d", fault_p0, fault_p1);
        } else if (fault_family == FaultFamily::EXTERNAL) {
          std::snprintf(fault_params_str, sizeof(fault_params_str), "%s", fault_plan_path.c_str());
        } else {
          fault_params_str[0] = '\0';
        }
        double fault_rate_csv = (fault_family == FaultFamily::EXTERNAL) ? fault_plan_meta.rate : 0.0;
        int seed_csv = (fault_family == FaultFamily::EXTERNAL) ? fault_plan_meta.seed : -1;
        const char *prot_list_csv = (protected_plane_count > 0) ? read_plane_list.c_str() : "";
        std::fprintf(fcsv, "%s,%" PRIu64 ",%d,%d,%s,%.6f,%.0f,%.6f,%" PRIu64 ",%.6f,%" PRIu64 ",%.6f,%s,%d,\"%s\",%s,\"%s\",%.6e,%d\n",
                     ds_name.c_str(), n, PLANES, k, path_label(path),
                     latency_ms, effective_bytes, bw_gb_s,
                     gpu_count, gpu_sum,
                     cpu_count, cpu_sum,
                     replica_mode.c_str(), protected_plane_count,
                     prot_list_csv, fault_label, fault_params_str,
                     fault_rate_csv, seed_csv);
        std::fflush(fcsv);

        bool count_match = (gpu_count == cpu_count);
        std::fprintf(stderr, "    latency=%.6f ms  bw=%.3f GB/s  gpu_count=%" PRIu64 " cpu_count=%" PRIu64 " match=%s\n",
                     latency_ms, bw_gb_s, gpu_count, cpu_count, count_match ? "OK" : "MISMATCH");
      }
    }

    // Cleanup k-specific resources
    cuda_check(cudaFree(d_active_k), "F active_k");
    cuda_check(cudaFree(d_counts), "F counts");
    cuda_check(cudaFree(d_sums), "F sums");
    cuda_check(cudaFree(d_raw_counts), "F rc");
    cuda_check(cudaFree(d_raw_sums), "F rs");
    cuda_check(cudaFree(d_digests), "F digests");
    // Free fault replica buffers
    for (auto &buf : d_r0_bufs) if (buf) cuda_check(cudaFree(buf), "F r0buf");
    for (auto &buf : d_r1_bufs) if (buf) cuda_check(cudaFree(buf), "F r1buf");
    for (auto &buf : d_r2_bufs) if (buf) cuda_check(cudaFree(buf), "F r2buf");
    d_r0_bufs.clear(); d_r1_bufs.clear(); d_r2_bufs.clear();
  }

  // Cleanup
  for (auto &buf : d_fault_bufs) if (buf) cuda_check(cudaFree(buf), "F faultbuf");
  cuda_check(cudaStreamDestroy(st), "D st");
  if (d_raw) cuda_check(cudaFree(d_raw), "F raw");
  cuda_check(cudaFree(d_threshold), "F thresh");
  cuda_check(cudaFree(d_seg_base), "F seg_base");
  cuda_check(cudaFree(d_basis), "F basis");
  cuda_check(cudaFree(d_seg_active), "F seg_active");
  for (int p = 0; p < PLANES; p++) cuda_check(cudaFree(d_planes[static_cast<size_t>(p)]), "F plane");

  std::fclose(fcsv);
  std::fprintf(stderr, "\nDone. CSV: %s\n", csv_path.c_str());
  return 0;
}
