#include <cuda_runtime.h>

#include <unistd.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "rl_kernels_fixture.cuh"
#include "rl_common.cuh"
#include "rl_uint128.cuh"

namespace
{

[[nodiscard]] const char *cuda_err_str(cudaError_t err) { return cudaGetErrorString(err); }

[[noreturn]] void die(const char *msg)
{
  std::fprintf(stderr, "error: %s\n", msg);
  std::exit(2);
}

[[noreturn]] void die(const std::string &msg)
{
  die(msg.c_str());
}

void cuda_check(cudaError_t err, const char *where)
{
  if (err == cudaSuccess)
    return;
  std::fprintf(stderr, "cuda error at %s: %s\n", where, cuda_err_str(err));
  std::exit(2);
}

[[nodiscard]] bool parse_u64(std::string_view s, uint64_t &out)
{
  if (s.empty())
    return false;
  uint64_t value = 0;
  for (char c : s)
  {
    if (c < '0' || c > '9')
      return false;
    uint64_t digit = static_cast<uint64_t>(c - '0');
    if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10ull)
      return false;
    value = value * 10ull + digit;
  }
  out = value;
  return true;
}

// Read entire file into a uint8 vector.
[[nodiscard]] std::vector<uint8_t> read_binary(const std::string &path)
{
  std::FILE *f = std::fopen(path.c_str(), "rb");
  if (!f)
    die("cannot open: " + path);
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  if (sz < 0)
    die("ftell failed: " + path);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(static_cast<size_t>(sz));
  if (static_cast<long>(std::fread(buf.data(), 1, static_cast<size_t>(sz), f)) != sz)
    die("short read: " + path);
  std::fclose(f);
  return buf;
}

struct FaultEntry
{
  uint64_t offset;
  uint8_t mask;
};

struct FaultPlan
{
  int target_plane = -1;
  uint64_t fault_count = 0;
  int seed = 0;
  std::string fault_rate_str;
  std::vector<FaultEntry> entries;
};

FaultPlan parse_fault_plan(const std::string &path)
{
  std::vector<uint8_t> data = read_binary(path);
  data.push_back(0); // null terminator

  // Minimal JSON parse using a simple tokenizer.
  // We look for "target_plane", "entries", "offset", "mask".
  // This is not a full JSON parser; it's sufficient for the fixed-format fault plans.
  FaultPlan plan;
  plan.target_plane = -1;
  plan.fault_count = 0;

  // Use a simple state machine over the raw bytes.
  const char *p = reinterpret_cast<const char *>(data.data());
  const char *end = p + data.size();

  auto skip_ws = [&]() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) ++p; };
  auto expect = [&](char c) { skip_ws(); if (p >= end || *p != c) die("JSON parse error: expected " + std::string(1, c)); ++p; };
  auto match_key = [&](const char *key) -> bool {
    skip_ws();
    if (*p != '"') return false;
    const char *k = key;
    const char *s = p + 1;
    while (*k && s < end && *s == *k) { ++k; ++s; }
    if (*k != '\0' || s >= end || *s != '"') return false;
    p = s + 1;
    return true;
  };

  auto parse_int = [&]() -> int64_t {
    skip_ws();
    bool neg = false;
    if (*p == '-') { neg = true; ++p; }
    int64_t val = 0;
    while (p < end && *p >= '0' && *p <= '9')
    {
      val = val * 10 + (*p - '0');
      ++p;
    }
    return neg ? -val : val;
  };

  // Read the fault plan with a simple recursive descent for the specific structure.
  // { "metadata": { ... "target_plane": N, "actual_fault_count": N }, "entries": [...] }
  expect('{');
  while (p < end)
  {
    skip_ws();
    if (*p == '}') break;

    if (match_key("metadata"))
    {
      expect(':');
      expect('{');
      while (p < end)
      {
        skip_ws();
        if (*p == '}') { ++p; break; }

        if (match_key("target_plane"))
        {
          expect(':');
          plan.target_plane = static_cast<int>(parse_int());
        }
        else if (match_key("actual_fault_count"))
        {
          expect(':');
          plan.fault_count = static_cast<uint64_t>(parse_int());
        }
        else if (match_key("seed"))
        {
          expect(':');
          plan.seed = static_cast<int>(parse_int());
        }
        else if (match_key("fault_rate"))
        {
          expect(':');
          skip_ws();
          if (*p == '"') { ++p; while (p < end && *p != '"') { plan.fault_rate_str.push_back(*p); ++p; } if (p < end) ++p; }
          else { while (p < end && *p != ',' && *p != '}' && *p != '\n') ++p; }
        }
        else
        {
          // skip unknown key-value pair
          expect('"');
          while (p < end && *p != '"') ++p;
          if (p < end) ++p;
          expect(':');
          skip_ws();
          if (*p == '"') { ++p; while (p < end && *p != '"') ++p; if (p < end) ++p; }
          else if (*p == '{' || *p == '[') { /* skip, handled by brace matching */ }
          else { while (p < end && *p != ',' && *p != '}' && *p != '\n') ++p; }
        }
        skip_ws();
        if (*p == ',') ++p;
      }
    }
    else if (match_key("entries"))
    {
      expect(':');
      expect('[');
      while (p < end)
      {
        skip_ws();
        if (*p == ']') { ++p; break; }
        expect('{');
        uint64_t off = 0;
        uint8_t m = 0;
        while (p < end)
        {
          skip_ws();
          if (*p == '}') { ++p; break; }
          if (match_key("offset"))
          {
            expect(':');
            off = static_cast<uint64_t>(parse_int());
          }
          else if (match_key("mask"))
          {
            expect(':');
            m = static_cast<uint8_t>(parse_int());
          }
          else
          {
            expect('"');
            while (p < end && *p != '"') ++p;
            if (p < end) ++p;
            expect(':');
            skip_ws();
            if (*p == '"') { ++p; while (p < end && *p != '"') ++p; if (p < end) ++p; }
            else { while (p < end && *p != ',' && *p != '}' && *p != '\n') ++p; }
          }
          skip_ws();
          if (*p == ',') ++p;
        }
        plan.entries.push_back({off, m});
        skip_ws();
        if (*p == ',') ++p;
      }
    }
    else
    {
      // skip unknown key
      expect('"');
      while (p < end && *p != '"') ++p;
      if (p < end) ++p;
      expect(':');
      skip_ws();
      if (*p == '"') { ++p; while (p < end && *p != '"') ++p; if (p < end) ++p; }
      else if (*p == '{') { int depth = 1; ++p; while (p < end && depth > 0) { if (*p == '{') ++depth; if (*p == '}') --depth; ++p; } }
      else if (*p == '[') { int depth = 1; ++p; while (p < end && depth > 0) { if (*p == '[') ++depth; if (*p == ']') --depth; ++p; } }
      else { while (p < end && *p != ',' && *p != '}' && *p != '\n') ++p; }
    }
    skip_ws();
    if (*p == ',') ++p;
  }

  if (plan.target_plane < 0)
    die("fault plan missing target_plane");

  // Verify entry count matches.
  if (plan.entries.size() != static_cast<size_t>(plan.fault_count))
  {
    std::fprintf(stderr, "warning: fault plan entry count %zu != metadata fault_count %" PRIu64 "\n",
                 plan.entries.size(), plan.fault_count);
  }

  return plan;
}

struct ArtifactJson
{
  uint64_t n_rows;
  int scale;
  std::string dataset;
  std::string git_commit;
  unsigned __int128 clean_encoded_sum;
};

ArtifactJson parse_artifact_json(const std::string &dir)
{
  std::string path = dir + "/artifact.json";
  std::vector<uint8_t> data = read_binary(path);
  data.push_back(0);
  const char *p = reinterpret_cast<const char *>(data.data());

  ArtifactJson aj;
  aj.n_rows = 0;
  aj.scale = 100;
  aj.clean_encoded_sum = 0;

  // Minimal parse: find "n_rows", "scale", "clean_encoded_sum", "dataset", "git_commit"
  auto find_int = [&](const char *key) -> int64_t {
    const char *found = std::strstr(p, key);
    if (!found) return -1;
    found = std::strchr(found, ':');
    if (!found) return -1;
    ++found;
    while (*found == ' ' || *found == '\n' || *found == '\t') ++found;
    char *endp = nullptr;
    return std::strtoll(found, &endp, 10);
  };

  auto find_str = [&](const char *key) -> std::string {
    const char *found = std::strstr(p, key);
    if (!found) return "";
    found = std::strchr(found, ':');
    if (!found) return "";
    ++found;
    while (*found == ' ' || *found == '\n' || *found == '\t' || *found == '"') ++found;
    std::string val;
    while (*found && *found != '"' && *found != '\n' && *found != ',') {
      val.push_back(*found); ++found;
    }
    return val;
  };

  auto find_sum_str = [&]() -> unsigned __int128 {
    const char *found = std::strstr(p, "\"clean_encoded_sum\"");
    if (!found) return 0;
    found = std::strchr(found, ':');
    if (!found) return 0;
    ++found;
    while (*found == ' ' || *found == '\n' || *found == '\t') ++found;
    if (*found == '"') ++found;
    unsigned __int128 val = 0;
    while (*found >= '0' && *found <= '9')
    {
      val = val * 10 + (*found - '0');
      ++found;
    }
    return val;
  };

  int64_t n = find_int("n_rows");
  if (n > 0) aj.n_rows = static_cast<uint64_t>(n);
  int64_t s = find_int("scale");
  if (s > 0) aj.scale = static_cast<int>(s);
  aj.clean_encoded_sum = find_sum_str();
  aj.dataset = find_str("dataset");
  aj.git_commit = find_str("git_commit");
  return aj;
}

template <typename T>
[[nodiscard]] T ceil_div(T x, T y)
{
  return (x + y - T(1)) / y;
}

void format_u128(unsigned __int128 val, char *buf, size_t buf_size)
{
  if (buf_size < 2) return;
  char tmp[48];
  int pos = 47;
  tmp[pos] = '\0';
  if (val == 0)
  {
    buf[0] = '0';
    buf[1] = '\0';
    return;
  }
  while (val > 0)
  {
    tmp[--pos] = "0123456789"[val % 10];
    val /= 10;
  }
  std::memmove(buf, tmp + pos, 48 - pos);
  buf[48 - pos] = '\0';
}

struct Options
{
  int device = 0;
  uint64_t n = 1000;
  std::string artifact_dir;
  std::string fault_plan_path;
  bool run_clean = false;
  std::string csv_path = "reliability_fixture.csv";
  int block_threads = 256;
  double grid_mul = 1.0;
};

Options parse_options(int argc, char **argv)
{
  Options opts;
  for (int i = 1; i < argc; ++i)
  {
    std::string_view arg(argv[i]);
    auto next = [&]() -> std::string_view
    {
      if (i + 1 >= argc)
        die("missing value for " + std::string(arg));
      return std::string_view(argv[++i]);
    };

    if (arg == "--device")
      opts.device = std::atoi(std::string(next()).c_str());
    else if (arg == "--n")
    {
      auto val = next();
      if (!parse_u64(val, opts.n))
        die("invalid --n: " + std::string(val));
    }
    else if (arg == "--artifact-dir")
      opts.artifact_dir = std::string(next());
    else if (arg == "--fault-plan")
      opts.fault_plan_path = std::string(next());
    else if (arg == "--run-clean")
      opts.run_clean = true;
    else if (arg == "--csv")
      opts.csv_path = std::string(next());
    else if (arg == "--block")
      opts.block_threads = std::atoi(std::string(next()).c_str());
    else if (arg == "--grid-mul")
      opts.grid_mul = std::atof(std::string(next()).c_str());
    else if (arg == "--help" || arg == "-h")
    {
      std::printf(
        "Usage: bench_reliability_fixture [options]\n"
        "  --device N         GPU device (default: 0)\n"
        "  --n ROWS           row count (default: 1000)\n"
        "  --artifact-dir DIR plane artifact directory\n"
        "  --fault-plan PATH  fault plan JSON (omit for clean-only)\n"
        "  --run-clean        run clean SUM and validate against artifact metadata\n"
        "  --csv PATH         output CSV path\n"
        "  --block N          threads per block (default: 256)\n"
        "  --grid-mul M       grid = M * multiprocessor count (default: 1.0)\n"
      );
      std::exit(0);
    }
    else
      die("unknown option: " + std::string(arg));
  }
  return opts;
}

unsigned __int128 run_gpu_sum(const uint8_t *planes[8], uint64_t n,
                              int device, int block_threads, double grid_mul)
{
  int mp_count = 0;
  cuda_check(cudaDeviceGetAttribute(&mp_count, cudaDevAttrMultiProcessorCount, device),
             "cudaDeviceGetAttribute");
  int grid = std::max(1, static_cast<int>(std::ceil(static_cast<double>(mp_count) * grid_mul)));

  uint8_t *d_planes[8];
  for (int p = 0; p < 8; p++)
  {
    cuda_check(cudaMalloc(&d_planes[p], n), "cudaMalloc plane");
    cuda_check(cudaMemcpy(d_planes[p], planes[p], n, cudaMemcpyHostToDevice),
               "cudaMemcpy plane");
  }

  U8Planes dev_ptrs;
  for (int p = 0; p < 8; p++)
    dev_ptrs.ptrs[p] = d_planes[p];

  Uint128 *d_block_sums = nullptr;
  cuda_check(cudaMalloc(&d_block_sums, static_cast<size_t>(grid) * sizeof(Uint128)),
             "cudaMalloc block sums");

  sum_planes_rowpack16_u128<<<grid, block_threads>>>(dev_ptrs, n, d_block_sums);
  cuda_check(cudaGetLastError(), "kernel launch");
  cuda_check(cudaDeviceSynchronize(), "kernel sync");

  std::vector<Uint128> h_block_sums(static_cast<size_t>(grid));
  cuda_check(cudaMemcpy(h_block_sums.data(), d_block_sums,
                        static_cast<size_t>(grid) * sizeof(Uint128),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy block sums back");

  unsigned __int128 total = 0;
  for (int b = 0; b < grid; b++)
  {
    total += (static_cast<unsigned __int128>(h_block_sums[b].hi) << 64)
           | h_block_sums[b].lo;
  }

  for (int p = 0; p < 8; p++)
    cuda_check(cudaFree(d_planes[p]), "cudaFree plane");
  cuda_check(cudaFree(d_block_sums), "cudaFree block sums");

  return total;
}

} // anonymous namespace

int main(int argc, char **argv)
{
  Options opts = parse_options(argc, argv);

  cuda_check(cudaSetDevice(opts.device), "cudaSetDevice");

  // Load artifact metadata
  ArtifactJson art = parse_artifact_json(opts.artifact_dir);
  if (opts.n == 0) opts.n = art.n_rows;

  // Load clean plane data
  std::vector<uint8_t> clean_planes[8];
  for (int p = 0; p < 8; p++)
  {
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/plane_%d.bin", opts.artifact_dir.c_str(), p);
    clean_planes[p] = read_binary(path);
    if (clean_planes[p].size() < opts.n)
      die("plane file too small: " + std::string(path));
  }

  const uint8_t *run_planes[8];
  std::vector<uint8_t> faulted_copy;

  FaultPlan fault_plan;
  bool has_fault = false;

  if (!opts.fault_plan_path.empty())
  {
    fault_plan = parse_fault_plan(opts.fault_plan_path);
    has_fault = true;

    // Copy the target plane and apply XOR faults
    faulted_copy = clean_planes[fault_plan.target_plane];
    for (const auto &entry : fault_plan.entries)
    {
      if (entry.offset < opts.n)
        faulted_copy[entry.offset] ^= entry.mask;
    }

    for (int p = 0; p < 8; p++)
      run_planes[p] = (p == fault_plan.target_plane) ? faulted_copy.data() : clean_planes[p].data();
  }
  else
  {
    for (int p = 0; p < 8; p++)
      run_planes[p] = clean_planes[p].data();
  }

  // Run GPU SUM
  unsigned __int128 gpu_sum = run_gpu_sum(run_planes, opts.n, opts.device,
                                          opts.block_threads, opts.grid_mul);

  // Clean gate: when --run-clean and no fault, compare with metadata
  if (opts.run_clean && !has_fault)
  {
    if (gpu_sum != art.clean_encoded_sum)
    {
      char gpu_str[48], meta_str[48];
      format_u128(gpu_sum, gpu_str, sizeof(gpu_str));
      format_u128(art.clean_encoded_sum, meta_str, sizeof(meta_str));
      std::fprintf(stderr, "CLEAN GATE FAILED: gpu=%s meta=%s\n", gpu_str, meta_str);
      std::exit(2);
    }
    std::printf("clean gate passed: gpu_sum = artifact clean_encoded_sum\n");
  }

  // Oracle validation (faulted run)
  bool oracle_match = true;
  unsigned __int128 expected_sum = 0;
  if (has_fault)
  {
    // Compute oracle: clean_encoded_sum + sum over fault entries
    // signed_delta = int(new_byte) - int(old_byte)
    // sum_delta += signed_delta * plane_weight
    int p = fault_plan.target_plane;
    unsigned __int128 plane_weight = static_cast<unsigned __int128>(1) << (8 * (7 - p));

    // Compute using signed __int128 for delta arithmetic
    __int128 sum_delta_signed = 0;
    for (const auto &entry : fault_plan.entries)
    {
      if (entry.offset >= opts.n) continue;
      uint8_t old_byte = clean_planes[p][entry.offset];
      uint8_t new_byte = old_byte ^ entry.mask;
      __int128 delta = static_cast<__int128>(new_byte) - static_cast<__int128>(old_byte);
      sum_delta_signed += delta * static_cast<__int128>(plane_weight);
    }

    if (sum_delta_signed >= 0)
      expected_sum = art.clean_encoded_sum + static_cast<unsigned __int128>(sum_delta_signed);
    else
      expected_sum = art.clean_encoded_sum - static_cast<unsigned __int128>(-sum_delta_signed);

    oracle_match = (gpu_sum == expected_sum);
    if (!oracle_match)
    {
      char gpu_str[48], exp_str[48];
      format_u128(gpu_sum, gpu_str, sizeof(gpu_str));
      format_u128(expected_sum, exp_str, sizeof(exp_str));
      std::fprintf(stderr, "ORACLE MISMATCH: gpu=%s expected=%s\n", gpu_str, exp_str);
    }
  }

  // Compute damage fields
  // signed_damage = int(gpu) - int(clean)
  __int128 signed_damage = 0;
  unsigned __int128 abs_damage = 0;
  if (has_fault)
  {
    if (gpu_sum >= art.clean_encoded_sum)
    {
      signed_damage = static_cast<__int128>(gpu_sum - art.clean_encoded_sum);
      abs_damage = gpu_sum - art.clean_encoded_sum;
    }
    else
    {
      signed_damage = -static_cast<__int128>(art.clean_encoded_sum - gpu_sum);
      abs_damage = art.clean_encoded_sum - gpu_sum;
    }
  }

  int target_plane = has_fault ? fault_plan.target_plane : -1;
  unsigned __int128 plane_weight = (target_plane >= 0)
    ? (static_cast<unsigned __int128>(1) << (8 * (7 - target_plane)))
    : 0;
  unsigned __int128 normalized_damage = (plane_weight > 0) ? (abs_damage / plane_weight) : 0;

  // Compute plane nonzero stats
  unsigned plane_nonzero[8];
  for (int p = 0; p < 8; p++)
  {
    plane_nonzero[p] = 0;
    for (uint64_t i = 0; i < opts.n; i++)
      if (clean_planes[p][i] != 0) plane_nonzero[p]++;
  }

  // Write CSV
  std::FILE *csv_f = std::fopen(opts.csv_path.c_str(), "w");
  if (!csv_f) die("cannot open CSV: " + opts.csv_path);

  char buf_clean[48], buf_gpu[48], buf_expected[48], buf_signed[48], buf_abs[48];

  auto fmt_signed = [](__int128 val, char *buf) {
    if (val < 0) { buf[0] = '-'; format_u128(static_cast<unsigned __int128>(-val), buf + 1, 47); }
    else { format_u128(static_cast<unsigned __int128>(val), buf, 48); }
  };

  format_u128(art.clean_encoded_sum, buf_clean, sizeof(buf_clean));
  if (has_fault)
  {
    format_u128(gpu_sum, buf_gpu, sizeof(buf_gpu));
    format_u128(expected_sum, buf_expected, sizeof(buf_expected));
    fmt_signed(signed_damage, buf_signed);
    format_u128(abs_damage, buf_abs, sizeof(buf_abs));
  }
  else
  {
    format_u128(gpu_sum, buf_gpu, sizeof(buf_gpu));
    std::snprintf(buf_expected, sizeof(buf_expected), "%s", buf_clean);
    buf_signed[0] = '0'; buf_signed[1] = '\0';
    buf_abs[0] = '0'; buf_abs[1] = '\0';
  }

  char fault_rate_str[32] = "";
  int64_t fault_count_val = 0;
  int seed_val = 0;
  if (has_fault)
  {
    std::snprintf(fault_rate_str, sizeof(fault_rate_str), "%s", fault_plan.fault_rate_str.c_str());
    fault_count_val = static_cast<int64_t>(fault_plan.fault_count);
    seed_val = fault_plan.seed;
  }

  char pnz_str[128] = "";
  char pnf_str[128] = "";
  {
    char tmp[32];
    std::snprintf(pnz_str, sizeof(pnz_str), "%u", plane_nonzero[0]);
    std::snprintf(pnf_str, sizeof(pnf_str), "%.17g", static_cast<double>(plane_nonzero[0]) / opts.n);
    for (int p = 1; p < 8; p++)
    {
      std::snprintf(tmp, sizeof(tmp), "|%u", plane_nonzero[p]);
      std::strncat(pnz_str, tmp, sizeof(pnz_str) - std::strlen(pnz_str) - 1);
      std::snprintf(tmp, sizeof(tmp), "|%.17g", static_cast<double>(plane_nonzero[p]) / opts.n);
      std::strncat(pnf_str, tmp, sizeof(pnf_str) - std::strlen(pnf_str) - 1);
    }
  }

  // Compute SHA-256 checksums
  auto sha256_file = [](const std::string &path) -> std::string {
    char cmd[4096];
    std::snprintf(cmd, sizeof(cmd), "sha256sum \"%s\" 2>/dev/null | cut -d' ' -f1", path.c_str());
    std::FILE *fp = popen(cmd, "r");
    if (!fp) return "";
    char result[128] = "";
    if (std::fgets(result, sizeof(result), fp)) {
      char *nl = std::strchr(result, '\n');
      if (nl) *nl = '\0';
    }
    pclose(fp);
    return result;
  };

  std::string artifact_checksum = sha256_file(opts.artifact_dir + "/artifact.json");
  std::string fault_plan_checksum = has_fault ? sha256_file(opts.fault_plan_path) : "";

  // GPU name and hostname
  cudaDeviceProp cprop;
  cuda_check(cudaGetDeviceProperties(&cprop, opts.device), "cudaGetDeviceProperties");
  std::string gpu_name(cprop.name);
  char hname[256] = "";
  gethostname(hname, sizeof(hname) - 1);
  std::string hostname(hname);
  std::string slurm_job_id = std::getenv("SLURM_JOB_ID") ? std::getenv("SLURM_JOB_ID") : "";

  char norm_damage_str[48] = "";
  if (has_fault)
    format_u128(normalized_damage, norm_damage_str, sizeof(norm_damage_str));
  else
    std::snprintf(norm_damage_str, sizeof(norm_damage_str), "0");

  char plane_weight_str[48] = "";
  format_u128(plane_weight, plane_weight_str, sizeof(plane_weight_str));

  char damage_decoded[48];
  format_u128(abs_damage / art.scale, damage_decoded, sizeof(damage_decoded));

  std::fprintf(csv_f,
    "run_id,dataset,n_rows,scale,"
    "target_plane,plane_weight,plane_nonzero_count,plane_nonzero_fraction,"
    "fault_rate,fault_count,seed,fault_model,"
    "clean_encoded_sum,expected_corrupted_sum,gpu_corrupted_sum,"
    "signed_sum_damage_encoded,abs_sum_damage_encoded,"
    "normalized_abs_sum_damage,decoded_abs_sum_damage,"
    "oracle_match,artifact_id,fault_plan_id,"
    "artifact_checksum,fault_plan_checksum,"
    "git_commit,hostname,gpu_name,slurm_job_id,"
    "repro_command,validity_status\n");

  const char *dataset_name = art.dataset.empty() ? "tiny_fixture" : art.dataset.c_str();
  const char *run_id = has_fault ? "reliability_gpu_faulted" : "reliability_gpu_clean";

  // Build repro command
  char repro_cmd[4096];
  std::snprintf(repro_cmd, sizeof(repro_cmd),
    "bench_reliability_fixture --device %d --n %" PRIu64
    " --artifact-dir %s"
    "%s%s"
    " --csv <csv_path>",
    opts.device, opts.n,
    opts.artifact_dir.c_str(),
    has_fault ? " --fault-plan " : "",
    has_fault ? opts.fault_plan_path.c_str() : "");

  std::fprintf(csv_f,
    "%s,%s,%" PRIu64 ",%d,"
    "%d,%s,%s,%s,"
    "%s,%" PRId64 ",%d,%s,"
    "%s,%s,%s,"
    "%s,%s,%s,"
    "%s,%s,"
    "%s,%s,"
    "%s,%s,%s,%s,"
    "%s,%s,%s,%s\n",
    run_id, dataset_name, opts.n, art.scale,
    target_plane, plane_weight_str, pnz_str, pnf_str,
    fault_rate_str, fault_count_val, seed_val, "plane_targeted_random_byte_xor",
    buf_clean, buf_expected, buf_gpu,
    buf_signed, buf_abs, norm_damage_str,
    damage_decoded,
    oracle_match ? "true" : "false",
    opts.artifact_dir.c_str(), opts.fault_plan_path.c_str(),
    artifact_checksum.c_str(), fault_plan_checksum.c_str(),
    art.git_commit.c_str(), hostname.c_str(), gpu_name.c_str(), slurm_job_id.c_str(),
    repro_cmd, oracle_match ? "canonical" : "ORACLE_MISMATCH");

  std::fclose(csv_f);

  std::printf("gpu_sum=%s clean_sum=%s oracle_match=%s\n",
              buf_gpu, buf_clean, oracle_match ? "yes" : "NO");

  return oracle_match ? 0 : 2;
}
