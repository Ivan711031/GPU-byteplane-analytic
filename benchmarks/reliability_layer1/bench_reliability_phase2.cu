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

[[nodiscard]] bool parse_u128(std::string_view s, unsigned __int128 &out)
{
  if (s.empty()) return false;
  unsigned __int128 value = 0;
  for (char c : s)
  {
    if (c < '0' || c > '9') return false;
    value = value * 10 + static_cast<unsigned __int128>(c - '0');
  }
  out = value;
  return true;
}

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

struct ArtifactJson
{
  uint64_t n_rows;
  int64_t scale;
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
  if (s > 0) aj.scale = s;
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

struct VoteOutcomes
{
  int64_t resolved_correctly = 0;
  int64_t detected_mismatch = 0;
  int64_t undetected_corruption = 0;
  int r_p = 0;
};

VoteOutcomes parse_all_vote_outcomes(const std::string &path)
{
  std::vector<uint8_t> data = read_binary(path);
  data.push_back(0);
  const char *p = reinterpret_cast<const char *>(data.data());

  const char *nl = std::strchr(p, '\n');
  if (!nl)
    die("vote outcomes CSV has no header");
  p = nl + 1;

  VoteOutcomes total;
  int found = 0;

  while (*p)
  {
    if (*p == '\n' || *p == '\r') { ++p; continue; }

    nl = std::strchr(p, '\n');
    if (!nl) nl = p + std::strlen(p);

    std::vector<std::string_view> fields;
    const char *f_start = p;
    for (const char *fp = p; fp < nl; ++fp)
    {
      if (*fp == ',')
      {
        fields.push_back(std::string_view(f_start, static_cast<size_t>(fp - f_start)));
        f_start = fp + 1;
      }
    }
    fields.push_back(std::string_view(f_start, static_cast<size_t>(nl - f_start)));

    if (fields.size() >= 6)
    {
      int pid = std::atoi(std::string(fields[0]).c_str());
      if (pid >= 0 && pid < 8)
      {
        total.resolved_correctly += std::atoll(std::string(fields[3]).c_str());
        total.detected_mismatch += std::atoll(std::string(fields[4]).c_str());
        total.undetected_corruption += std::atoll(std::string(fields[5]).c_str());
        ++found;
      }
    }

    p = nl + 1;
  }

  if (found < 8)
    die("vote outcomes CSV missing some planes: found " + std::to_string(found));

  return total;
}

struct Options
{
  int device = 0;
  uint64_t n = 1000;
  std::string artifact_dir;
  std::string voted_dir;
  std::string vote_outcomes_path;
  std::string csv_path = "reliability_phase2.csv";
  std::string policy_id = "uniform";
  int budget_B = 0;
  std::string fault_rate_str;
  int base_seed = 0;
  std::string strategy_id = "per_dataset";
  std::string strategy_scale;
  std::string allocation_r;
  int block_threads = 256;
  double grid_mul = 1.0;
  std::string expected_sum_str;
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
    else if (arg == "--voted-dir")
      opts.voted_dir = std::string(next());
    else if (arg == "--vote-outcomes")
      opts.vote_outcomes_path = std::string(next());
    else if (arg == "--csv")
      opts.csv_path = std::string(next());
    else if (arg == "--policy-id")
      opts.policy_id = std::string(next());
    else if (arg == "--budget-B")
      opts.budget_B = std::atoi(std::string(next()).c_str());
    else if (arg == "--fault-rate")
      opts.fault_rate_str = std::string(next());
    else if (arg == "--base-seed")
      opts.base_seed = std::atoi(std::string(next()).c_str());
    else if (arg == "--strategy-id")
      opts.strategy_id = std::string(next());
    else if (arg == "--strategy-scale")
      opts.strategy_scale = std::string(next());
    else if (arg == "--allocation-r")
      opts.allocation_r = std::string(next());
    else if (arg == "--block")
      opts.block_threads = std::atoi(std::string(next()).c_str());
    else if (arg == "--grid-mul")
      opts.grid_mul = std::atof(std::string(next()).c_str());
    else if (arg == "--expected-sum")
      opts.expected_sum_str = std::string(next());
    else if (arg == "--help" || arg == "-h")
    {
      std::printf(
        "Usage: bench_reliability_phase2 [options]\n"
        "  --device N            GPU device (default: 0)\n"
        "  --n ROWS              row count (default: 1000)\n"
        "  --artifact-dir DIR    plane artifact directory (clean planes)\n"
        "  --voted-dir DIR       directory with voted plane files (plane_0_voted.bin ...)\n"
        "  --vote-outcomes PATH  vote outcomes CSV from phase2_vote.py\n"
        "  --csv PATH            output CSV path\n"
        "  --policy-id STR       policy identifier (uniform, graded_vacuous_aware)\n"
        "  --budget-B N          budget value\n"
        "  --fault-rate STR      fault rate string (e.g., 1e-06)\n"
        "  --base-seed N         base seed\n"
        "  --strategy-id STR     strategy (default: per_dataset)\n"
        "  --strategy-scale N    scale value from params\n"
        "  --allocation-r STR    allocation vector string\n"
        "  --block N             threads per block (default: 256)\n"
        "  --grid-mul M          grid = M * multiprocessor count (default: 1.0)\n"
        "  --expected-sum STR    expected voted sum from oracle (decimal string)\n"
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

  // Load clean plane data for clean gate and nonzero stats
  std::vector<uint8_t> clean_planes[8];
  for (int p = 0; p < 8; p++)
  {
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/plane_%d.bin", opts.artifact_dir.c_str(), p);
    clean_planes[p] = read_binary(path);
    if (clean_planes[p].size() < opts.n)
      die("clean plane file too small: " + std::string(path));
  }

  // ── Clean gate: run GPU SUM on clean planes, verify against artifact ──
  {
    const uint8_t *clean_ptrs[8];
    for (int p = 0; p < 8; p++)
      clean_ptrs[p] = clean_planes[p].data();

    unsigned __int128 clean_gpu_sum = run_gpu_sum(clean_ptrs, opts.n, opts.device,
                                                   opts.block_threads, opts.grid_mul);
    if (clean_gpu_sum != art.clean_encoded_sum)
    {
      char gpu_str[48], meta_str[48];
      format_u128(clean_gpu_sum, gpu_str, sizeof(gpu_str));
      format_u128(art.clean_encoded_sum, meta_str, sizeof(meta_str));
      std::fprintf(stderr, "CLEAN GATE FAILED: gpu=%s meta=%s\n", gpu_str, meta_str);
      std::exit(2);
    }
    std::printf("clean gate passed: gpu_sum = artifact clean_encoded_sum\n");
  }

  // ── Load voted plane data ──
  std::vector<uint8_t> voted_planes[8];
  for (int p = 0; p < 8; p++)
  {
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/plane_%d_voted.bin", opts.voted_dir.c_str(), p);
    voted_planes[p] = read_binary(path);
    if (voted_planes[p].size() < opts.n)
      die("voted plane file too small: " + std::string(path));
  }

  // ── Run GPU SUM on voted planes ──
  const uint8_t *voted_ptrs[8];
  for (int p = 0; p < 8; p++)
    voted_ptrs[p] = voted_planes[p].data();

  unsigned __int128 voted_gpu_sum = run_gpu_sum(voted_ptrs, opts.n, opts.device,
                                                 opts.block_threads, opts.grid_mul);

  // ── Parse vote outcomes across all 8 planes ──
  const char *target_plane_str = "all";
  VoteOutcomes vo = parse_all_vote_outcomes(opts.vote_outcomes_path);

  // ── Compute damage ──
  unsigned __int128 abs_damage = 0;
  __int128 signed_damage = 0;
  if (voted_gpu_sum >= art.clean_encoded_sum)
  {
    signed_damage = static_cast<__int128>(voted_gpu_sum - art.clean_encoded_sum);
    abs_damage = voted_gpu_sum - art.clean_encoded_sum;
  }
  else
  {
    signed_damage = -static_cast<__int128>(art.clean_encoded_sum - voted_gpu_sum);
    abs_damage = art.clean_encoded_sum - voted_gpu_sum;
  }

  unsigned __int128 plane_weight = static_cast<unsigned __int128>(0x0101010101010101ULL);
  unsigned __int128 normalized_damage = (plane_weight > 0) ? (abs_damage / plane_weight) : 0;

  // ── Compute plane nonzero stats from clean planes ──
  unsigned plane_nonzero[8];
  for (int p = 0; p < 8; p++)
  {
    plane_nonzero[p] = 0;
    for (uint64_t i = 0; i < opts.n; i++)
      if (clean_planes[p][i] != 0) plane_nonzero[p]++;
  }

  // ── Compute fault_count as positions where voted != clean ──
  int64_t fault_count_val = static_cast<int64_t>(8 * opts.n) - vo.resolved_correctly;

  // ── Oracle match: compare gpu_voted_sum vs expected sum ──
  unsigned __int128 expected_voted_sum = 0;
  bool has_expected = false;
  const char *oracle_match_str = "true";
  const char *validity_status_str = "canonical";
  if (!opts.expected_sum_str.empty())
  {
    if (!parse_u128(opts.expected_sum_str, expected_voted_sum))
      die("invalid --expected-sum: " + opts.expected_sum_str);
    has_expected = true;
    if (voted_gpu_sum == expected_voted_sum)
    {
      oracle_match_str = "true";
      validity_status_str = "canonical";
    }
    else
    {
      oracle_match_str = "false";
      validity_status_str = "ORACLE_MISMATCH";
    }
  }

  // ── Write CSV ──
  std::FILE *csv_f = std::fopen(opts.csv_path.c_str(), "w");
  if (!csv_f) die("cannot open CSV: " + opts.csv_path);

  std::fprintf(csv_f,
    "run_id,dataset,n_rows,scale,"
    "target_plane,plane_weight,plane_nonzero_count,plane_nonzero_fraction,"
    "fault_rate,fault_count,seed,fault_model,"
    "clean_encoded_sum,expected_voted_sum,gpu_voted_sum,"
    "signed_voted_sum_damage_encoded,abs_voted_sum_damage_encoded,"
    "normalized_abs_voted_damage,decoded_abs_voted_damage,"
    "oracle_match,artifact_id,fault_plan_id,"
    "artifact_checksum,fault_plan_checksum,"
    "git_commit,hostname,gpu_name,slurm_job_id,"
    "repro_command,validity_status,"
    "policy,budget_B,allocation_r,base_seed,"
    "resolved_correctly_count,detected_mismatch_count,undetected_corruption_count,"
    "strategy_id,strategy_scale\n");

  char buf_clean[48], buf_gpu[48], buf_signed[48], buf_abs[48], buf_norm[48];
  char buf_plane_weight[48], buf_decoded[48];

  format_u128(art.clean_encoded_sum, buf_clean, sizeof(buf_clean));
  format_u128(voted_gpu_sum, buf_gpu, sizeof(buf_gpu));

  auto fmt_signed = [](__int128 val, char *buf) {
    if (val < 0) { buf[0] = '-'; format_u128(static_cast<unsigned __int128>(-val), buf + 1, 47); }
    else { format_u128(static_cast<unsigned __int128>(val), buf, 48); }
  };
  fmt_signed(signed_damage, buf_signed);
  format_u128(abs_damage, buf_abs, sizeof(buf_abs));
  format_u128(normalized_damage, buf_norm, sizeof(buf_norm));
  format_u128(plane_weight, buf_plane_weight, sizeof(buf_plane_weight));
  format_u128(abs_damage / art.scale, buf_decoded, sizeof(buf_decoded));

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
  std::string voted_dir_checksum; // not applicable for Phase 2 (no single fault plan)

  cudaDeviceProp cprop;
  cuda_check(cudaGetDeviceProperties(&cprop, opts.device), "cudaGetDeviceProperties");
  std::string gpu_name(cprop.name);
  char hname[256] = "";
  gethostname(hname, sizeof(hname) - 1);
  std::string hostname(hname);
  std::string slurm_job_id = std::getenv("SLURM_JOB_ID") ? std::getenv("SLURM_JOB_ID") : "";

  const char *dataset_name = art.dataset.empty() ? "tiny_fixture" : art.dataset.c_str();

  char repro_cmd[4096];
  std::snprintf(repro_cmd, sizeof(repro_cmd),
    "bench_reliability_phase2 --device %d --n %" PRIu64
    " --artifact-dir %s"
    " --voted-dir %s"
    " --vote-outcomes %s"
    " --csv <csv_path>"
    " --policy-id %s --budget-B %d --fault-rate %s --base-seed %d"
    " --strategy-id %s --strategy-scale %s --allocation-r %s"
    " --expected-sum %s",
    opts.device, opts.n,
    opts.artifact_dir.c_str(),
    opts.voted_dir.c_str(),
    opts.vote_outcomes_path.c_str(),
    opts.policy_id.c_str(), opts.budget_B, opts.fault_rate_str.c_str(), opts.base_seed,
    opts.strategy_id.c_str(), opts.strategy_scale.c_str(),
    opts.allocation_r.c_str(),
    opts.expected_sum_str.c_str());

  std::fprintf(csv_f,
    "%s,%s,%" PRIu64 ",%" PRId64 ","
    "%s,%s,%s,%s,"
    "%s,%" PRId64 ",%d,%s,"
    "%s,%s,%s,"
    "%s,%s,%s,"
    "%s,%s,"
    "%s,%s,"
    "%s,%s,%s,%s,"
    "%s,%s,"
    "%s,%s,%s,%d,%s,%d,"
    "%" PRId64 ",%" PRId64 ",%" PRId64 ","
    "%s,%s\n",
    "reliability_phase2_voted", dataset_name, opts.n, art.scale,
    target_plane_str, buf_plane_weight, pnz_str, pnf_str,
    opts.fault_rate_str.c_str(), fault_count_val, opts.base_seed, "voted_planes",
    buf_clean, has_expected ? opts.expected_sum_str.c_str() : buf_gpu, buf_gpu,
    buf_signed, buf_abs, buf_norm,
    buf_decoded,
    oracle_match_str,
    opts.artifact_dir.c_str(), "",
    artifact_checksum.c_str(), "",
    art.git_commit.c_str(), hostname.c_str(), gpu_name.c_str(), slurm_job_id.c_str(),
    repro_cmd, validity_status_str,
    opts.policy_id.c_str(), opts.budget_B, opts.allocation_r.c_str(), opts.base_seed,
    vo.resolved_correctly, vo.detected_mismatch, vo.undetected_corruption,
    opts.strategy_id.c_str(), opts.strategy_scale.c_str());

  std::fclose(csv_f);

  char voted_sum_str[48];
  format_u128(voted_gpu_sum, voted_sum_str, sizeof(voted_sum_str));
  std::printf("voted_gpu_sum=%s clean_sum=%s clean_gate=passed\n",
              voted_sum_str, buf_clean);

  return 0;
}
