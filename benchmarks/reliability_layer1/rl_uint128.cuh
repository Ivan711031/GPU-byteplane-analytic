#pragma once

#include <cuda_runtime.h>

#include <cstdint>

struct Uint128
{
  uint64_t lo;
  uint64_t hi;
};

__device__ __host__ __forceinline__ Uint128 add_u64(Uint128 acc, uint64_t x)
{
  uint64_t new_lo = acc.lo + x;
  acc.hi += (new_lo < acc.lo) ? 1 : 0;
  acc.lo = new_lo;
  return acc;
}

__device__ __host__ __forceinline__ Uint128 add_u128(Uint128 a, Uint128 b)
{
  uint64_t new_lo = a.lo + b.lo;
  uint64_t carry = (new_lo < a.lo) ? 1 : 0;
  uint64_t new_hi = a.hi + b.hi + carry;
  return Uint128{new_lo, new_hi};
}

__device__ __forceinline__ Uint128 warp_reduce_u128(Uint128 v)
{
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
  {
    uint64_t shfl_lo = __shfl_down_sync(0xffffffff, v.lo, offset);
    uint64_t shfl_hi = __shfl_down_sync(0xffffffff, v.hi, offset);
    v = add_u128(v, Uint128{shfl_lo, shfl_hi});
  }
  return v;
}

__device__ __forceinline__ void block_reduce_store_u128(Uint128 v,
                                                         Uint128 *warp_sums,
                                                         Uint128 *per_block_out)
{
  v = warp_reduce_u128(v);

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0)
    warp_sums[warp] = v;
  __syncthreads();

  if (warp == 0)
  {
    int warp_count = blockDim.x >> 5;
    Uint128 block_sum = {0, 0};
    for (int w = lane; w < warp_count; w += 32)
      block_sum = add_u128(block_sum, warp_sums[w]);
    block_sum = warp_reduce_u128(block_sum);
    if (lane == 0)
      per_block_out[blockIdx.x] = block_sum;
  }
}
