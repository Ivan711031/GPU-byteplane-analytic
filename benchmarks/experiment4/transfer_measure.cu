// transfer_measure.cu: Measure host-to-device HtoD copy time.
//
// Usage:
//   transfer_measure --artifact-dir <dir>     # plane_*.bin payload
//   transfer_measure --raw-file <path>         # single raw FP64 file
//
// Timing contract (same for both modes):
//   - read file(s) into a single contiguous host buffer
//   - allocate matching device memory
//   - one warmup cudaMemcpy HtoD
//   - one timed cudaMemcpy HtoD via CUDA events
//   - excludes file I/O, allocation, and kernel execution
//
// Output: one line to stdout:
//   H2D <total_bytes> <cudaMemcpy_ms>
//
// Returns 0 on success, 1 on error.

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void die(const char *msg) {
  std::fprintf(stderr, "transfer_measure: %s\n", msg);
  std::exit(1);
}

// Load a single file into host_buf.
static void load_file(const fs::path &path, std::vector<char> &host_buf) {
  auto size = fs::file_size(path);
  if (size == 0) die("empty file");
  std::ifstream in(path, std::ios::binary);
  if (!in) die("failed to open file");
  size_t offset = host_buf.size();
  host_buf.resize(offset + size);
  in.read(host_buf.data() + offset, static_cast<std::streamsize>(size));
  if (!in) die("failed to read file");
}

// Timing helper: warmup + timed cudaMemcpy HtoD, prints H2D <bytes> <ms>.
static void timed_h2d(const std::vector<char> &host_buf) {
  size_t total_bytes = host_buf.size();
  if (total_bytes == 0) die("zero-byte payload");

  char *d_buf = nullptr;
  auto cerr = cudaMalloc(&d_buf, total_bytes);
  if (cerr != cudaSuccess) die("cudaMalloc failed");

  // Warmup
  cerr = cudaMemcpy(d_buf, host_buf.data(), total_bytes, cudaMemcpyHostToDevice);
  if (cerr != cudaSuccess) die("cudaMemcpy warmup failed");

  // Timed copy with CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cerr = cudaMemcpy(d_buf, host_buf.data(), total_bytes, cudaMemcpyHostToDevice);
  if (cerr != cudaSuccess) die("cudaMemcpy timed failed");
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_buf);

  std::printf("H2D %zu %.3f\n", total_bytes, static_cast<double>(ms));
}

int main(int argc, char **argv) {
  if (argc < 3) die("usage: transfer_measure --artifact-dir <dir> | --raw-file <path>");

  std::string mode(argv[1]);
  fs::path target(argv[2]);

  std::vector<char> host_buf;

  if (mode == "--artifact-dir") {
    if (!fs::is_directory(target)) die("--artifact-dir requires a directory");

    std::vector<fs::path> plane_files;
    for (auto &entry : fs::directory_iterator(target)) {
      auto name = entry.path().filename().string();
      if (name.rfind("plane_", 0) == 0 && name.size() == 13 &&
          name.substr(9) == ".bin") {
        plane_files.push_back(entry.path());
      }
    }
    std::sort(plane_files.begin(), plane_files.end());
    if (plane_files.empty()) die("no plane_*.bin files found");

    for (auto &path : plane_files)
      load_file(path, host_buf);
  } else if (mode == "--raw-file") {
    if (!fs::is_regular_file(target)) die("--raw-file requires a regular file");
    load_file(target, host_buf);
  } else {
    die("unknown mode (use --artifact-dir or --raw-file)");
  }

  timed_h2d(host_buf);
  return 0;
}
