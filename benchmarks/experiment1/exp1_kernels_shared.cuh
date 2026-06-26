#pragma once

#include "exp1_scan_common.cuh"

__global__ void scan_planes_u8_shared128(const uint8_t *const *__restrict__ planes,
                                         int k,
                                         uint64_t n,
                                         unsigned long long *__restrict__ per_block_out)
{
  // Diagnostic path: each warp stages 128 physical bytes and consumes 32 logical rows.
  unsigned long long sum = 0;

  int lane = threadIdx.x & 31;
  int warp_in_block = threadIdx.x >> 5;
  int warps_per_block = blockDim.x >> 5;

  extern __shared__ uint32_t smem32[];
  uint32_t *warp_smem32 = smem32 + static_cast<size_t>(warp_in_block) * 32ull;

  uint64_t warp_global = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(warps_per_block) +
                         static_cast<uint64_t>(warp_in_block);
  uint64_t warp_stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(warps_per_block);

  uint64_t base = warp_global * 32ull;
  while (base < n)
  {
    uint64_t offset = base + static_cast<uint64_t>(lane) * 4ull;

    for (int p = 0; p < k; p++)
    {
      uint32_t pack = 0;
      if (offset + 3 < n)
      {
        pack = *reinterpret_cast<const uint32_t *>(planes[p] + offset);
      }
      else
      {
        if (offset + 0 < n)
          pack |= static_cast<uint32_t>(planes[p][offset + 0]) << 0;
        if (offset + 1 < n)
          pack |= static_cast<uint32_t>(planes[p][offset + 1]) << 8;
        if (offset + 2 < n)
          pack |= static_cast<uint32_t>(planes[p][offset + 2]) << 16;
        if (offset + 3 < n)
          pack |= static_cast<uint32_t>(planes[p][offset + 3]) << 24;
      }

      warp_smem32[lane] = pack;
      __syncwarp();

      uint64_t i = base + static_cast<uint64_t>(lane);
      if (i < n)
      {
        uint32_t ppack = warp_smem32[lane >> 2];
        uint32_t b = (ppack >> (8u * static_cast<uint32_t>(lane & 3))) & 0xffu;
        sum += static_cast<unsigned long long>(b);
      }
      __syncwarp();
    }

    base += warp_stride * 32ull;
  }

  block_reduce_store(sum, per_block_out);
}
