#pragma once

#include "exp1_scan_common.cuh"

#include <cstdlib>

template <int K>
__global__ void scan_planes_u8_byte_unrolled(const U8Planes planes,
                                             uint64_t n,
                                             unsigned long long *__restrict__ per_block_out)
{
  static_assert(K >= 1 && K <= 8, "K must be in [1, 8]");
  unsigned long long sum = 0;
  uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                 static_cast<uint64_t>(threadIdx.x);
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
  for (uint64_t i = tid; i < n; i += stride)
  {
#pragma unroll
    for (int p = 0; p < K; p++)
      sum += static_cast<unsigned long long>(planes.ptrs[p][i]);
  }
  block_reduce_store(sum, per_block_out);
}

inline const void *scan_planes_u8_byte_unrolled_kernel_ptr(int k)
{
  switch (k)
  {
  case 1:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<1>);
  case 2:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<2>);
  case 3:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<3>);
  case 4:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<4>);
  case 5:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<5>);
  case 6:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<6>);
  case 7:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<7>);
  case 8:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_unrolled<8>);
  default:
    std::abort();
  }
}

inline void launch_scan_planes_u8_byte_unrolled(int k,
                                                int grid,
                                                int block_threads,
                                                const U8Planes &planes,
                                                uint64_t n,
                                                unsigned long long *per_block_out)
{
  switch (k)
  {
  case 1:
    scan_planes_u8_byte_unrolled<1><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 2:
    scan_planes_u8_byte_unrolled<2><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 3:
    scan_planes_u8_byte_unrolled<3><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 4:
    scan_planes_u8_byte_unrolled<4><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 5:
    scan_planes_u8_byte_unrolled<5><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 6:
    scan_planes_u8_byte_unrolled<6><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 7:
    scan_planes_u8_byte_unrolled<7><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 8:
    scan_planes_u8_byte_unrolled<8><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  default:
    std::abort();
  }
}

template <int K>
__global__ void scan_planes_u8_byte_ilp4(const U8Planes planes,
                                         uint64_t n,
                                         unsigned long long *__restrict__ per_block_out)
{
  static_assert(K >= 1 && K <= 8, "K must be in [1, 8]");
  unsigned long long sum0 = 0;
  unsigned long long sum1 = 0;
  unsigned long long sum2 = 0;
  unsigned long long sum3 = 0;

  uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                 static_cast<uint64_t>(threadIdx.x);
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
  uint64_t step = stride * 4ull;
  uint64_t i = tid;

  for (; i < n;)
  {
    uint64_t i1 = i + stride;
    uint64_t i2 = i1 + stride;
    uint64_t i3 = i2 + stride;
    if (i1 < i || i2 < i1 || i3 < i2 || i3 >= n)
      break;

#pragma unroll
    for (int p = 0; p < K; p++)
    {
      sum0 += static_cast<unsigned long long>(planes.ptrs[p][i]);
      sum1 += static_cast<unsigned long long>(planes.ptrs[p][i1]);
      sum2 += static_cast<unsigned long long>(planes.ptrs[p][i2]);
      sum3 += static_cast<unsigned long long>(planes.ptrs[p][i3]);
    }

    uint64_t next = i + step;
    if (next <= i)
      break;
    i = next;
  }

  for (; i < n;)
  {
#pragma unroll
    for (int p = 0; p < K; p++)
      sum0 += static_cast<unsigned long long>(planes.ptrs[p][i]);

    uint64_t next = i + stride;
    if (next <= i)
      break;
    i = next;
  }

  unsigned long long sum = sum0 + sum1 + sum2 + sum3;
  block_reduce_store(sum, per_block_out);
}

inline const void *scan_planes_u8_byte_ilp4_kernel_ptr(int k)
{
  switch (k)
  {
  case 1:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<1>);
  case 2:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<2>);
  case 3:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<3>);
  case 4:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<4>);
  case 5:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<5>);
  case 6:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<6>);
  case 7:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<7>);
  case 8:
    return reinterpret_cast<const void *>(scan_planes_u8_byte_ilp4<8>);
  default:
    std::abort();
  }
}

inline void launch_scan_planes_u8_byte_ilp4(int k,
                                            int grid,
                                            int block_threads,
                                            const U8Planes &planes,
                                            uint64_t n,
                                            unsigned long long *per_block_out)
{
  switch (k)
  {
  case 1:
    scan_planes_u8_byte_ilp4<1><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 2:
    scan_planes_u8_byte_ilp4<2><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 3:
    scan_planes_u8_byte_ilp4<3><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 4:
    scan_planes_u8_byte_ilp4<4><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 5:
    scan_planes_u8_byte_ilp4<5><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 6:
    scan_planes_u8_byte_ilp4<6><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 7:
    scan_planes_u8_byte_ilp4<7><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  case 8:
    scan_planes_u8_byte_ilp4<8><<<grid, block_threads>>>(planes, n, per_block_out);
    return;
  default:
    std::abort();
  }
}

inline const void *scan_planes_u8_byte_kernel_ptr(ByteVariant variant, int k)
{
  switch (variant)
  {
  case ByteVariant::Baseline:
    return scan_planes_u8_byte_unrolled_kernel_ptr(k);
  case ByteVariant::Ilp4:
    return scan_planes_u8_byte_ilp4_kernel_ptr(k);
  }
  std::abort();
}

inline void launch_scan_planes_u8_byte(ByteVariant variant,
                                       int k,
                                       int grid,
                                       int block_threads,
                                       const U8Planes &planes,
                                       uint64_t n,
                                       unsigned long long *per_block_out)
{
  switch (variant)
  {
  case ByteVariant::Baseline:
    launch_scan_planes_u8_byte_unrolled(k, grid, block_threads, planes, n, per_block_out);
    return;
  case ByteVariant::Ilp4:
    launch_scan_planes_u8_byte_ilp4(k, grid, block_threads, planes, n, per_block_out);
    return;
  }
  std::abort();
}
