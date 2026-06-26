#pragma once

#include "exp1_scan_common.cuh"

__global__ void scan_planes_u16(const uint16_t *const *__restrict__ planes16,
                                int k,
                                uint64_t n,
                                unsigned long long *__restrict__ per_block_out)
{
  unsigned long long sum = 0;
  uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                 static_cast<uint64_t>(threadIdx.x);
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
  for (uint64_t i = tid; i < n; i += stride)
  {
    for (int p = 0; p < k; p++)
      sum += static_cast<unsigned long long>(planes16[p][i]);
  }
  block_reduce_store(sum, per_block_out);
}
