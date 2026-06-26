#pragma once

#include <cuda_runtime.h>

#include <cstdint>

enum class ByteVariant
{
  Baseline,
  Ilp4,
};

template <typename T, int N>
struct PlanePointers
{
  const T *ptrs[N];
};

using U8Planes = PlanePointers<uint8_t, 8>;

__device__ __forceinline__ unsigned long long warp_reduce_sum_ull(unsigned long long v)
{
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

__device__ __forceinline__ void block_reduce_store(unsigned long long v,
                                                   unsigned long long *__restrict__ out)
{
  v = warp_reduce_sum_ull(v);
  __shared__ unsigned long long warp_sums[32]; // up to 1024 threads
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0)
    warp_sums[warp] = v;
  __syncthreads();

  if (warp == 0)
  {
    unsigned long long block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0ull;
    block_sum = warp_reduce_sum_ull(block_sum);
    if (lane == 0)
      out[blockIdx.x] = block_sum;
  }
}
