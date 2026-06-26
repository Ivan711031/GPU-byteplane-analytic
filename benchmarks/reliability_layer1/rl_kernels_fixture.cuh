#pragma once

#include "rl_uint128.cuh"
#include "rl_common.cuh"

#include <cstdlib>

__global__ void sum_planes_rowpack16_u128(const U8Planes planes,
                                          uint64_t n,
                                          Uint128 *per_block_out)
{
  Uint128 sum = {0, 0};
  uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                 static_cast<uint64_t>(threadIdx.x);
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
  uint64_t n16 = n / 16ull;

  for (uint64_t i16 = tid; i16 < n16; i16 += stride)
  {
    uint4 pack0 = reinterpret_cast<const uint4 *>(planes.ptrs[0])[i16];
    uint4 pack1 = reinterpret_cast<const uint4 *>(planes.ptrs[1])[i16];
    uint4 pack2 = reinterpret_cast<const uint4 *>(planes.ptrs[2])[i16];
    uint4 pack3 = reinterpret_cast<const uint4 *>(planes.ptrs[3])[i16];
    uint4 pack4 = reinterpret_cast<const uint4 *>(planes.ptrs[4])[i16];
    uint4 pack5 = reinterpret_cast<const uint4 *>(planes.ptrs[5])[i16];
    uint4 pack6 = reinterpret_cast<const uint4 *>(planes.ptrs[6])[i16];
    uint4 pack7 = reinterpret_cast<const uint4 *>(planes.ptrs[7])[i16];

    auto *b0 = reinterpret_cast<const uint8_t *>(&pack0);
    auto *b1 = reinterpret_cast<const uint8_t *>(&pack1);
    auto *b2 = reinterpret_cast<const uint8_t *>(&pack2);
    auto *b3 = reinterpret_cast<const uint8_t *>(&pack3);
    auto *b4 = reinterpret_cast<const uint8_t *>(&pack4);
    auto *b5 = reinterpret_cast<const uint8_t *>(&pack5);
    auto *b6 = reinterpret_cast<const uint8_t *>(&pack6);
    auto *b7 = reinterpret_cast<const uint8_t *>(&pack7);

    #pragma unroll
    for (int j = 0; j < 16; j++)
    {
      uint64_t row = reconstruct_u64(b0[j], b1[j], b2[j], b3[j],
                                     b4[j], b5[j], b6[j], b7[j]);
      sum = add_u64(sum, row);
    }
  }

  uint64_t tail_start = n16 * 16ull;
  for (uint64_t i = tail_start + tid; i < n; i += stride)
  {
    uint64_t row = reconstruct_u64(planes.ptrs[0][i], planes.ptrs[1][i],
                                   planes.ptrs[2][i], planes.ptrs[3][i],
                                   planes.ptrs[4][i], planes.ptrs[5][i],
                                   planes.ptrs[6][i], planes.ptrs[7][i]);
    sum = add_u64(sum, row);
  }

  sum = warp_reduce_u128(sum);
  __shared__ Uint128 warp_sums[32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0)
    warp_sums[warp] = sum;
  __syncthreads();

  if (warp == 0)
  {
    Uint128 block_sum = {0, 0};
    int warp_count = blockDim.x >> 5;
    for (int w = lane; w < warp_count; w += 32)
      block_sum = add_u128(block_sum, warp_sums[w]);
    block_sum = warp_reduce_u128(block_sum);
    if (lane == 0)
      per_block_out[blockIdx.x] = block_sum;
  }
}
