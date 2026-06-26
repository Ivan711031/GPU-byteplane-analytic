// Phase 3-Z0c: CRC implementation probe for sm_90 (H200)
// Uses inline PTX crc32.b32 (confirmed working: asm in kernel body, not via wrapper)
// Build: nvcc -arch=sm_90 -O3 -o z0c_crc_probe z0c_crc_probe.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

constexpr int ITERS = 1000;

__global__ void crc_ptx_kernel(const uint8_t *__restrict__ data, uint64_t n_bytes, uint32_t *__restrict__ out) {
    if (threadIdx.x > 0 || blockIdx.x > 0) return;
    uint32_t crc = 0xFFFFFFFFu;
    for (uint64_t i = 0; i < n_bytes; i++) {
        uint32_t b = data[i];
        asm("crc32.b32 %0, %0, %1;" : "+r"(crc) : "r"(b));
    }
    *out = crc ^ 0xFFFFFFFFu;
}

int main(int argc, char **argv) {
    uint64_t n_bytes = 4096;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bytes") == 0 && i+1 < argc)
            n_bytes = strtoull(argv[++i], nullptr, 10);
    }

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    uint8_t *d_data; uint32_t *d_out;
    cudaMalloc(&d_data, n_bytes);
    cudaMalloc(&d_out, sizeof(uint32_t));

    std::vector<uint8_t> h_data(n_bytes, 0x42);
    cudaMemcpy(d_data, h_data.data(), n_bytes, cudaMemcpyHostToDevice);

    printf("Bytes per CRC: %lu  ITERS=%d\n", n_bytes, ITERS);

    crc_ptx_kernel<<<1, 1>>>(d_data, n_bytes, d_out);
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < ITERS; i++)
        crc_ptx_kernel<<<1, 1>>>(d_data, n_bytes, d_out);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);

    uint32_t crc_val = 0;
    cudaMemcpy(&crc_val, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    double ns_per_byte = ms / ITERS * 1e6 / n_bytes;
    double per_crc_ms = ms / ITERS;

    printf("  PTX CRC32 available on sm_90 (H200): YES\n");
    printf("  Total:       %.3f ms\n", ms);
    printf("  Per CRC:     %.6f ms  (%.3f ns/byte)\n", per_crc_ms, ns_per_byte);
    printf("  Throughput:  %.1f GB/s\n", n_bytes / (per_crc_ms / 1000) / 1e9);
    printf("  CRC32:       0x%08X\n", crc_val);

    double proj_41133 = per_crc_ms * 41133;
    double with_read = proj_41133 + 0.40;
    printf("\n  Projected for 41133 units (full dataset):\n");
    printf("    HW CRC only:  %.3f ms\n", proj_41133);
    printf("    + baseline:   %.3f ms  (vs B0=0.870ms)\n", with_read);
    if (with_read < 0.870) printf("    => BW GATE PASS (estimate)\n");
    else printf("    => BW GATE FAIL (estimate)\n");

    cudaFree(d_data); cudaFree(d_out);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return 0;
}
