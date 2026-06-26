#pragma once

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

constexpr int EXP3_MAX_SUBCOLUMNS = 8;
constexpr int EXP3_RUNTIME_MAX_SUBCOLUMNS = 32;
constexpr int EXP3_ROWPACK16_WIDTH = 16;

// IEEE 754 double positive/negative infinity for device code.
static __device__ __forceinline__ double exp3_pos_inf() { return __longlong_as_double(0x7ff0000000000000ULL); }
static __device__ __forceinline__ double exp3_neg_inf() { return __longlong_as_double(0xfff0000000000000ULL); }

template <typename T, int N>
struct Exp3PointerSet
{
  const T *ptrs[N];
};

using Exp3U8Subcolumns = Exp3PointerSet<uint8_t, EXP3_MAX_SUBCOLUMNS>;
using Exp3RuntimeU8Subcolumns = Exp3PointerSet<uint8_t, EXP3_RUNTIME_MAX_SUBCOLUMNS>;

__host__ __device__ __forceinline__ uint64_t exp3_ceil_div_u64(uint64_t x, uint64_t y)
{
  return (x + y - 1ull) / y;
}

__device__ __forceinline__ unsigned int exp3_byte_sum_u32(uint32_t x)
{
  return (x & 0xffu) +
         ((x >> 8) & 0xffu) +
         ((x >> 16) & 0xffu) +
         ((x >> 24) & 0xffu);
}

__device__ __forceinline__ unsigned long long exp3_warp_reduce_sum_ull(unsigned long long v)
{
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

__device__ __forceinline__ unsigned long long
exp3_block_reduce_sum_ull(unsigned long long v, unsigned long long *__restrict__ warp_sums)
{
  v = exp3_warp_reduce_sum_ull(v);

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int warp_count = blockDim.x >> 5;

  if (lane == 0)
    warp_sums[warp] = v;
  __syncthreads();

  unsigned long long block_sum = 0ull;
  if (warp == 0)
  {
    block_sum = (lane < warp_count) ? warp_sums[lane] : 0ull;
    block_sum = exp3_warp_reduce_sum_ull(block_sum);
  }

  // Required when this helper is called repeatedly inside one block.
  __syncthreads();
  return block_sum;
}

__device__ __forceinline__ double exp3_warp_reduce_min_double(double v)
{
  for (int offset = 16; offset > 0; offset >>= 1)
    v = fmin(v, __shfl_down_sync(0xffffffff, v, offset));
  return v;
}

__device__ __forceinline__ double exp3_warp_reduce_max_double(double v)
{
  for (int offset = 16; offset > 0; offset >>= 1)
    v = fmax(v, __shfl_down_sync(0xffffffff, v, offset));
  return v;
}

__device__ __forceinline__ double exp3_block_reduce_min_double(double v, double *__restrict__ warp_vals)
{
  v = exp3_warp_reduce_min_double(v);

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int warp_count = blockDim.x >> 5;

  if (lane == 0)
    warp_vals[warp] = v;
  __syncthreads();

  double block_min = exp3_pos_inf();
  if (warp == 0)
  {
    block_min = (lane < warp_count) ? warp_vals[lane] : exp3_pos_inf();
    block_min = exp3_warp_reduce_min_double(block_min);
  }

  __syncthreads();
  return block_min;
}

__device__ __forceinline__ double exp3_block_reduce_max_double(double v, double *__restrict__ warp_vals)
{
  v = exp3_warp_reduce_max_double(v);

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int warp_count = blockDim.x >> 5;

  if (lane == 0)
    warp_vals[warp] = v;
  __syncthreads();

  double block_max = exp3_neg_inf();
  if (warp == 0)
  {
    block_max = (lane < warp_count) ? warp_vals[lane] : exp3_neg_inf();
    block_max = exp3_warp_reduce_max_double(block_max);
  }

  __syncthreads();
  return block_max;
}
