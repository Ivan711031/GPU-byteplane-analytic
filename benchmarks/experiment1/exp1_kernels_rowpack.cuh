#pragma once

#include "exp1_scan_common.cuh"

#include <cstdlib>

__device__ __forceinline__ unsigned int byte_sum_u32(uint32_t x)
{
  return (x & 0xffu) +
         ((x >> 8) & 0xffu) +
         ((x >> 16) & 0xffu) +
         ((x >> 24) & 0xffu);
}

template <int K>
__global__ void scan_planes_u8_rowpack4(const U8Planes planes,
                                        uint64_t n,
                                        unsigned long long *__restrict__ per_block_out)
{
  static_assert(K >= 1 && K <= 8, "K must be in [1, 8]");
  unsigned long long sum = 0;
  uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                 static_cast<uint64_t>(threadIdx.x);
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
  uint64_t n4 = n / 4ull;

  for (uint64_t i4 = tid; i4 < n4; i4 += stride)
  {
#pragma unroll
    for (int p = 0; p < K; p++)
    {
      const uint32_t *plane32 = reinterpret_cast<const uint32_t *>(planes.ptrs[p]);
      sum += static_cast<unsigned long long>(byte_sum_u32(plane32[i4]));
    }
  }

  uint64_t tail_start = n4 * 4ull;
  for (uint64_t i = tail_start + tid; i < n; i += stride)
  {
#pragma unroll
    for (int p = 0; p < K; p++)
      sum += static_cast<unsigned long long>(planes.ptrs[p][i]);
  }

  block_reduce_store(sum, per_block_out);
}

inline const void *scan_planes_u8_rowpack4_kernel_ptr(int k)
{
  switch (k)
  {
  case 1:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack4<1>);
  case 2:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack4<2>);
  case 3:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack4<3>);
  case 4:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack4<4>);
  case 5:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack4<5>);
  case 6:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack4<6>);
  case 7:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack4<7>);
  case 8:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack4<8>);
  default:
    std::abort();
  }
}

inline void launch_scan_planes_u8_rowpack4(int k,
                                           int grid,
                                           int block_threads,
                                           const U8Planes &planes,
                                           uint64_t n,
                                           unsigned long long *per_block_out)
{
  switch (k)
  {
  case 1:
    scan_planes_u8_rowpack4<1><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 2:
    scan_planes_u8_rowpack4<2><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 3:
    scan_planes_u8_rowpack4<3><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 4:
    scan_planes_u8_rowpack4<4><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 5:
    scan_planes_u8_rowpack4<5><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 6:
    scan_planes_u8_rowpack4<6><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 7:
    scan_planes_u8_rowpack4<7><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 8:
    scan_planes_u8_rowpack4<8><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  default:
    std::abort();
  }
}

template <int K>
__global__ void scan_planes_u8_rowpack16(const U8Planes planes,
                                         uint64_t n,
                                         unsigned long long *__restrict__ per_block_out)
{
  static_assert(K >= 1 && K <= 8, "K must be in [1, 8]");
  unsigned long long sum = 0;
  uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                 static_cast<uint64_t>(threadIdx.x);
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
  uint64_t n16 = n / 16ull;

  for (uint64_t i16 = tid; i16 < n16; i16 += stride)
  {
#pragma unroll
    for (int p = 0; p < K; p++)
    {
      const uint4 *plane128 = reinterpret_cast<const uint4 *>(planes.ptrs[p]);
      uint4 pack = plane128[i16];
      sum += static_cast<unsigned long long>(byte_sum_u32(pack.x));
      sum += static_cast<unsigned long long>(byte_sum_u32(pack.y));
      sum += static_cast<unsigned long long>(byte_sum_u32(pack.z));
      sum += static_cast<unsigned long long>(byte_sum_u32(pack.w));
    }
  }

  uint64_t tail_start = n16 * 16ull;
  for (uint64_t i = tail_start + tid; i < n; i += stride)
  {
#pragma unroll
    for (int p = 0; p < K; p++)
      sum += static_cast<unsigned long long>(planes.ptrs[p][i]);
  }

  block_reduce_store(sum, per_block_out);
}

inline const void *scan_planes_u8_rowpack16_kernel_ptr(int k)
{
  switch (k)
  {
  case 1:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack16<1>);
  case 2:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack16<2>);
  case 3:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack16<3>);
  case 4:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack16<4>);
  case 5:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack16<5>);
  case 6:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack16<6>);
  case 7:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack16<7>);
  case 8:
    return reinterpret_cast<const void *>(scan_planes_u8_rowpack16<8>);
  default:
    std::abort();
  }
}

inline void launch_scan_planes_u8_rowpack16(int k,
                                            int grid,
                                            int block_threads,
                                            const U8Planes &planes,
                                            uint64_t n,
                                            unsigned long long *per_block_out)
{
  switch (k)
  {
  case 1:
    scan_planes_u8_rowpack16<1><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 2:
    scan_planes_u8_rowpack16<2><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 3:
    scan_planes_u8_rowpack16<3><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 4:
    scan_planes_u8_rowpack16<4><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 5:
    scan_planes_u8_rowpack16<5><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 6:
    scan_planes_u8_rowpack16<6><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 7:
    scan_planes_u8_rowpack16<7><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 8:
    scan_planes_u8_rowpack16<8><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  default:
    std::abort();
  }
}
