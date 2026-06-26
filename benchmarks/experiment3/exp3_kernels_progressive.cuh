#pragma once

#include "exp3_common.cuh"

#include <cstdlib>

namespace exp3_detail
{

__device__ __forceinline__ unsigned long long exp3_sum_uint4_bytes(uint4 pack)
{
  return static_cast<unsigned long long>(exp3_byte_sum_u32(pack.x)) +
         static_cast<unsigned long long>(exp3_byte_sum_u32(pack.y)) +
         static_cast<unsigned long long>(exp3_byte_sum_u32(pack.z)) +
         static_cast<unsigned long long>(exp3_byte_sum_u32(pack.w));
}

template <int DEPTH>
__device__ __forceinline__ void add_scalar_row(const Exp3U8Subcolumns &subcolumns,
                                               uint64_t row,
                                               unsigned long long &sub0,
                                               unsigned long long &sub1,
                                               unsigned long long &sub2,
                                               unsigned long long &sub3,
                                               unsigned long long &sub4,
                                               unsigned long long &sub5,
                                               unsigned long long &sub6,
                                               unsigned long long &sub7)
{
  if constexpr (DEPTH >= 0)
    sub0 += static_cast<unsigned long long>(subcolumns.ptrs[0][row]);
  if constexpr (DEPTH >= 1)
    sub1 += static_cast<unsigned long long>(subcolumns.ptrs[1][row]);
  if constexpr (DEPTH >= 2)
    sub2 += static_cast<unsigned long long>(subcolumns.ptrs[2][row]);
  if constexpr (DEPTH >= 3)
    sub3 += static_cast<unsigned long long>(subcolumns.ptrs[3][row]);
  if constexpr (DEPTH >= 4)
    sub4 += static_cast<unsigned long long>(subcolumns.ptrs[4][row]);
  if constexpr (DEPTH >= 5)
    sub5 += static_cast<unsigned long long>(subcolumns.ptrs[5][row]);
  if constexpr (DEPTH >= 6)
    sub6 += static_cast<unsigned long long>(subcolumns.ptrs[6][row]);
  if constexpr (DEPTH >= 7)
    sub7 += static_cast<unsigned long long>(subcolumns.ptrs[7][row]);
}

template <int DEPTH>
__device__ __forceinline__ void add_rowpack16(const Exp3U8Subcolumns &subcolumns,
                                              uint64_t pack_index,
                                              unsigned long long &sub0,
                                              unsigned long long &sub1,
                                              unsigned long long &sub2,
                                              unsigned long long &sub3,
                                              unsigned long long &sub4,
                                              unsigned long long &sub5,
                                              unsigned long long &sub6,
                                              unsigned long long &sub7)
{
  if constexpr (DEPTH >= 0)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[0]);
    sub0 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (DEPTH >= 1)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[1]);
    sub1 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (DEPTH >= 2)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[2]);
    sub2 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (DEPTH >= 3)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[3]);
    sub3 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (DEPTH >= 4)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[4]);
    sub4 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (DEPTH >= 5)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[5]);
    sub5 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (DEPTH >= 6)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[6]);
    sub6 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (DEPTH >= 7)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[7]);
    sub7 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
}

__device__ __forceinline__ void add_scalar_row_runtime(const Exp3RuntimeU8Subcolumns &subcolumns,
                                                       uint64_t row,
                                                       int plane_limit,
                                                       unsigned long long *plane_sums)
{
  for (int plane = 0; plane < plane_limit; ++plane)
    plane_sums[plane] += static_cast<unsigned long long>(subcolumns.ptrs[plane][row]);
}

__device__ __forceinline__ void add_rowpack16_runtime(const Exp3RuntimeU8Subcolumns &subcolumns,
                                                      uint64_t pack_index,
                                                      int plane_limit,
                                                      unsigned long long *plane_sums)
{
  for (int plane = 0; plane < plane_limit; ++plane)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[plane]);
    plane_sums[plane] += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
}

template <int PLANE_LIMIT>
__device__ __forceinline__ void add_scalar_row_runtime_fixed(const Exp3RuntimeU8Subcolumns &subcolumns,
                                                             uint64_t row,
                                                             unsigned long long &sub0,
                                                             unsigned long long &sub1,
                                                             unsigned long long &sub2,
                                                             unsigned long long &sub3,
                                                             unsigned long long &sub4,
                                                             unsigned long long &sub5,
                                                             unsigned long long &sub6,
                                                             unsigned long long &sub7,
                                                             unsigned long long &sub8,
                                                             unsigned long long &sub9,
                                                             unsigned long long &sub10,
                                                             unsigned long long &sub11,
                                                             unsigned long long &sub12,
                                                             unsigned long long &sub13,
                                                             unsigned long long &sub14,
                                                             unsigned long long &sub15)
{
  static_assert(PLANE_LIMIT >= 1 && PLANE_LIMIT <= 16, "PLANE_LIMIT must be in [1, 16]");
  if constexpr (PLANE_LIMIT >= 1)
    sub0 += static_cast<unsigned long long>(subcolumns.ptrs[0][row]);
  if constexpr (PLANE_LIMIT >= 2)
    sub1 += static_cast<unsigned long long>(subcolumns.ptrs[1][row]);
  if constexpr (PLANE_LIMIT >= 3)
    sub2 += static_cast<unsigned long long>(subcolumns.ptrs[2][row]);
  if constexpr (PLANE_LIMIT >= 4)
    sub3 += static_cast<unsigned long long>(subcolumns.ptrs[3][row]);
  if constexpr (PLANE_LIMIT >= 5)
    sub4 += static_cast<unsigned long long>(subcolumns.ptrs[4][row]);
  if constexpr (PLANE_LIMIT >= 6)
    sub5 += static_cast<unsigned long long>(subcolumns.ptrs[5][row]);
  if constexpr (PLANE_LIMIT >= 7)
    sub6 += static_cast<unsigned long long>(subcolumns.ptrs[6][row]);
  if constexpr (PLANE_LIMIT >= 8)
    sub7 += static_cast<unsigned long long>(subcolumns.ptrs[7][row]);
  if constexpr (PLANE_LIMIT >= 9)
    sub8 += static_cast<unsigned long long>(subcolumns.ptrs[8][row]);
  if constexpr (PLANE_LIMIT >= 10)
    sub9 += static_cast<unsigned long long>(subcolumns.ptrs[9][row]);
  if constexpr (PLANE_LIMIT >= 11)
    sub10 += static_cast<unsigned long long>(subcolumns.ptrs[10][row]);
  if constexpr (PLANE_LIMIT >= 12)
    sub11 += static_cast<unsigned long long>(subcolumns.ptrs[11][row]);
  if constexpr (PLANE_LIMIT >= 13)
    sub12 += static_cast<unsigned long long>(subcolumns.ptrs[12][row]);
  if constexpr (PLANE_LIMIT >= 14)
    sub13 += static_cast<unsigned long long>(subcolumns.ptrs[13][row]);
  if constexpr (PLANE_LIMIT >= 15)
    sub14 += static_cast<unsigned long long>(subcolumns.ptrs[14][row]);
  if constexpr (PLANE_LIMIT >= 16)
    sub15 += static_cast<unsigned long long>(subcolumns.ptrs[15][row]);
}

template <int PLANE_LIMIT>
__device__ __forceinline__ void add_rowpack16_runtime_fixed(const Exp3RuntimeU8Subcolumns &subcolumns,
                                                            uint64_t pack_index,
                                                            unsigned long long &sub0,
                                                            unsigned long long &sub1,
                                                            unsigned long long &sub2,
                                                            unsigned long long &sub3,
                                                            unsigned long long &sub4,
                                                            unsigned long long &sub5,
                                                            unsigned long long &sub6,
                                                            unsigned long long &sub7,
                                                            unsigned long long &sub8,
                                                            unsigned long long &sub9,
                                                            unsigned long long &sub10,
                                                            unsigned long long &sub11,
                                                            unsigned long long &sub12,
                                                            unsigned long long &sub13,
                                                            unsigned long long &sub14,
                                                            unsigned long long &sub15)
{
  static_assert(PLANE_LIMIT >= 1 && PLANE_LIMIT <= 16, "PLANE_LIMIT must be in [1, 16]");
  if constexpr (PLANE_LIMIT >= 1)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[0]);
    sub0 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (PLANE_LIMIT >= 2)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[1]);
    sub1 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (PLANE_LIMIT >= 3)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[2]);
    sub2 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (PLANE_LIMIT >= 4)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[3]);
    sub3 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (PLANE_LIMIT >= 5)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[4]);
    sub4 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (PLANE_LIMIT >= 6)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[5]);
    sub5 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (PLANE_LIMIT >= 7)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[6]);
    sub6 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (PLANE_LIMIT >= 8)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[7]);
    sub7 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (PLANE_LIMIT >= 9)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[8]);
    sub8 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (PLANE_LIMIT >= 10)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[9]);
    sub9 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (PLANE_LIMIT >= 11)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[10]);
    sub10 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (PLANE_LIMIT >= 12)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[11]);
    sub11 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (PLANE_LIMIT >= 13)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[12]);
    sub12 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (PLANE_LIMIT >= 14)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[13]);
    sub13 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (PLANE_LIMIT >= 15)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[14]);
    sub14 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
  if constexpr (PLANE_LIMIT >= 16)
  {
    const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[15]);
    sub15 += exp3_sum_uint4_bytes(plane128[pack_index]);
  }
}

} // namespace exp3_detail

template <int DEPTH, int ITEMS_PER_THREAD>
__global__ void progressive_sum_rowpack16_u8subcols(Exp3U8Subcolumns subcolumns,
                                                    uint64_t n,
                                                    uint64_t segment_rows,
                                                    uint64_t tiles_per_segment,
                                                    const double *__restrict__ d_segment_base,
                                                    const double *__restrict__ d_subcolumn_basis,
                                                    double *__restrict__ d_partial_out)
{
  static_assert(DEPTH >= 0 && DEPTH < EXP3_MAX_SUBCOLUMNS, "DEPTH must be in [0, 7]");
  static_assert(ITEMS_PER_THREAD >= 1, "ITEMS_PER_THREAD must be positive");

  uint64_t tile_rows = static_cast<uint64_t>(blockDim.x) *
                       static_cast<uint64_t>(ITEMS_PER_THREAD) *
                       static_cast<uint64_t>(EXP3_ROWPACK16_WIDTH);
  uint64_t segment_id = static_cast<uint64_t>(blockIdx.x) / tiles_per_segment;
  uint64_t tile_in_segment = static_cast<uint64_t>(blockIdx.x) % tiles_per_segment;
  uint64_t segment_start = segment_id * segment_rows;
  uint64_t segment_end = segment_start + segment_rows;
  uint64_t tile_start = segment_start + tile_in_segment * tile_rows;

  if (tile_start >= n || tile_start >= segment_end)
  {
    if (threadIdx.x == 0)
      d_partial_out[blockIdx.x] = 0.0;
    return;
  }

  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > segment_end)
    tile_end = segment_end;
  if (tile_end > n)
    tile_end = n;
  uint64_t tile_rows_actual = tile_end - tile_start;

  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0)
    ++aligned_start;

  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull;
  uint64_t pack_count = (full_pack_end - aligned_start) / 16ull;

  unsigned long long sub0 = 0ull;
  unsigned long long sub1 = 0ull;
  unsigned long long sub2 = 0ull;
  unsigned long long sub3 = 0ull;
  unsigned long long sub4 = 0ull;
  unsigned long long sub5 = 0ull;
  unsigned long long sub6 = 0ull;
  unsigned long long sub7 = 0ull;

  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    exp3_detail::add_scalar_row<DEPTH>(subcolumns, row, sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7);
  }

  for (uint64_t pack = static_cast<uint64_t>(threadIdx.x);
       pack < pack_count;
       pack += static_cast<uint64_t>(blockDim.x))
  {
    exp3_detail::add_rowpack16<DEPTH>(subcolumns,
                                      first_pack + pack,
                                      sub0,
                                      sub1,
                                      sub2,
                                      sub3,
                                      sub4,
                                      sub5,
                                      sub6,
                                      sub7);
  }

  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    exp3_detail::add_scalar_row<DEPTH>(subcolumns, row, sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7);
  }

  __shared__ unsigned long long reduce_smem[EXP3_MAX_SUBCOLUMNS][32];
  unsigned long long sub0_block = exp3_block_reduce_sum_ull(sub0, reduce_smem[0]);
  unsigned long long sub1_block = 0ull;
  unsigned long long sub2_block = 0ull;
  unsigned long long sub3_block = 0ull;
  unsigned long long sub4_block = 0ull;
  unsigned long long sub5_block = 0ull;
  unsigned long long sub6_block = 0ull;
  unsigned long long sub7_block = 0ull;

  if constexpr (DEPTH >= 1)
    sub1_block = exp3_block_reduce_sum_ull(sub1, reduce_smem[1]);
  if constexpr (DEPTH >= 2)
    sub2_block = exp3_block_reduce_sum_ull(sub2, reduce_smem[2]);
  if constexpr (DEPTH >= 3)
    sub3_block = exp3_block_reduce_sum_ull(sub3, reduce_smem[3]);
  if constexpr (DEPTH >= 4)
    sub4_block = exp3_block_reduce_sum_ull(sub4, reduce_smem[4]);
  if constexpr (DEPTH >= 5)
    sub5_block = exp3_block_reduce_sum_ull(sub5, reduce_smem[5]);
  if constexpr (DEPTH >= 6)
    sub6_block = exp3_block_reduce_sum_ull(sub6, reduce_smem[6]);
  if constexpr (DEPTH >= 7)
    sub7_block = exp3_block_reduce_sum_ull(sub7, reduce_smem[7]);

  if (threadIdx.x == 0)
  {
    const double *basis = d_subcolumn_basis + segment_id * EXP3_MAX_SUBCOLUMNS;
    double partial = static_cast<double>(tile_rows_actual) * d_segment_base[segment_id];

    partial += static_cast<double>(sub0_block) * basis[0];
    if constexpr (DEPTH >= 1)
      partial += static_cast<double>(sub1_block) * basis[1];
    if constexpr (DEPTH >= 2)
      partial += static_cast<double>(sub2_block) * basis[2];
    if constexpr (DEPTH >= 3)
      partial += static_cast<double>(sub3_block) * basis[3];
    if constexpr (DEPTH >= 4)
      partial += static_cast<double>(sub4_block) * basis[4];
    if constexpr (DEPTH >= 5)
      partial += static_cast<double>(sub5_block) * basis[5];
    if constexpr (DEPTH >= 6)
      partial += static_cast<double>(sub6_block) * basis[6];
    if constexpr (DEPTH >= 7)
      partial += static_cast<double>(sub7_block) * basis[7];

    d_partial_out[blockIdx.x] = partial;
  }
}

inline const void *progressive_sum_rowpack16_kernel_ptr(int refinement_depth)
{
  switch (refinement_depth)
  {
  case 0:
    return reinterpret_cast<const void *>(progressive_sum_rowpack16_u8subcols<0, 1>);
  case 1:
    return reinterpret_cast<const void *>(progressive_sum_rowpack16_u8subcols<1, 1>);
  case 2:
    return reinterpret_cast<const void *>(progressive_sum_rowpack16_u8subcols<2, 1>);
  case 3:
    return reinterpret_cast<const void *>(progressive_sum_rowpack16_u8subcols<3, 1>);
  case 4:
    return reinterpret_cast<const void *>(progressive_sum_rowpack16_u8subcols<4, 1>);
  case 5:
    return reinterpret_cast<const void *>(progressive_sum_rowpack16_u8subcols<5, 1>);
  case 6:
    return reinterpret_cast<const void *>(progressive_sum_rowpack16_u8subcols<6, 1>);
  case 7:
    return reinterpret_cast<const void *>(progressive_sum_rowpack16_u8subcols<7, 1>);
  default:
    std::abort();
  }
}

inline void launch_progressive_sum_rowpack16(int refinement_depth,
                                             int grid,
                                             int block_threads,
                                             const Exp3U8Subcolumns &subcolumns,
                                             uint64_t n,
                                             uint64_t segment_rows,
                                             uint64_t tiles_per_segment,
                                             const double *d_segment_base,
                                             const double *d_subcolumn_basis,
                                             double *d_partial_out)
{
  switch (refinement_depth)
  {
  case 0:
    progressive_sum_rowpack16_u8subcols<0, 1><<<grid, block_threads>>>(
        subcolumns, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 1:
    progressive_sum_rowpack16_u8subcols<1, 1><<<grid, block_threads>>>(
        subcolumns, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 2:
    progressive_sum_rowpack16_u8subcols<2, 1><<<grid, block_threads>>>(
        subcolumns, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 3:
    progressive_sum_rowpack16_u8subcols<3, 1><<<grid, block_threads>>>(
        subcolumns, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 4:
    progressive_sum_rowpack16_u8subcols<4, 1><<<grid, block_threads>>>(
        subcolumns, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 5:
    progressive_sum_rowpack16_u8subcols<5, 1><<<grid, block_threads>>>(
        subcolumns, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 6:
    progressive_sum_rowpack16_u8subcols<6, 1><<<grid, block_threads>>>(
        subcolumns, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 7:
    progressive_sum_rowpack16_u8subcols<7, 1><<<grid, block_threads>>>(
        subcolumns, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  default:
    std::abort();
  }
}

__global__ void progressive_sum_rowpack16_u8subcols_runtime(Exp3RuntimeU8Subcolumns subcolumns,
                                                            int max_planes,
                                                            int refinement_depth,
                                                            uint64_t n,
                                                            uint64_t segment_rows,
                                                            uint64_t tiles_per_segment,
                                                            const double *__restrict__ d_segment_base,
                                                            const double *__restrict__ d_subcolumn_basis,
                                                            double *__restrict__ d_partial_out)
{
  uint64_t tile_rows = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(EXP3_ROWPACK16_WIDTH);
  uint64_t segment_id = static_cast<uint64_t>(blockIdx.x) / tiles_per_segment;
  uint64_t tile_in_segment = static_cast<uint64_t>(blockIdx.x) % tiles_per_segment;
  uint64_t segment_start = segment_id * segment_rows;
  uint64_t segment_end = segment_start + segment_rows;
  uint64_t tile_start = segment_start + tile_in_segment * tile_rows;

  if (tile_start >= n || tile_start >= segment_end)
  {
    if (threadIdx.x == 0)
      d_partial_out[blockIdx.x] = 0.0;
    return;
  }

  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > segment_end)
    tile_end = segment_end;
  if (tile_end > n)
    tile_end = n;
  uint64_t tile_rows_actual = tile_end - tile_start;

  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0)
    ++aligned_start;

  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull;
  uint64_t pack_count = (full_pack_end - aligned_start) / 16ull;

  int plane_limit = refinement_depth + 1;
  if (plane_limit > max_planes)
    plane_limit = max_planes;

  unsigned long long plane_sums[EXP3_RUNTIME_MAX_SUBCOLUMNS] = {};
  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    exp3_detail::add_scalar_row_runtime(subcolumns, row, plane_limit, plane_sums);
  }

  for (uint64_t pack = static_cast<uint64_t>(threadIdx.x);
       pack < pack_count;
       pack += static_cast<uint64_t>(blockDim.x))
  {
    exp3_detail::add_rowpack16_runtime(subcolumns, first_pack + pack, plane_limit, plane_sums);
  }

  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    exp3_detail::add_scalar_row_runtime(subcolumns, row, plane_limit, plane_sums);
  }

  __shared__ unsigned long long reduce_smem[EXP3_RUNTIME_MAX_SUBCOLUMNS][32];
  double partial = 0.0;
  if (threadIdx.x == 0)
    partial = static_cast<double>(tile_rows_actual) * d_segment_base[segment_id];

  const double *basis = d_subcolumn_basis + segment_id * max_planes;
  for (int plane = 0; plane < plane_limit; ++plane)
  {
    unsigned long long block_sum = exp3_block_reduce_sum_ull(plane_sums[plane], reduce_smem[plane]);
    if (threadIdx.x == 0)
      partial += static_cast<double>(block_sum) * basis[plane];
  }

  if (threadIdx.x == 0)
    d_partial_out[blockIdx.x] = partial;
}

template <int PLANE_LIMIT>
__global__ void progressive_sum_rowpack16_u8subcols_runtime_fixed(Exp3RuntimeU8Subcolumns subcolumns,
                                                                  int basis_stride,
                                                                  uint64_t n,
                                                                  uint64_t segment_rows,
                                                                  uint64_t tiles_per_segment,
                                                                  const double *__restrict__ d_segment_base,
                                                                  const double *__restrict__ d_subcolumn_basis,
                                                                  double *__restrict__ d_partial_out)
{
  static_assert(PLANE_LIMIT >= 1 && PLANE_LIMIT <= 16, "PLANE_LIMIT must be in [1, 16]");

  uint64_t tile_rows = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(EXP3_ROWPACK16_WIDTH);
  uint64_t segment_id = static_cast<uint64_t>(blockIdx.x) / tiles_per_segment;
  uint64_t tile_in_segment = static_cast<uint64_t>(blockIdx.x) % tiles_per_segment;
  uint64_t segment_start = segment_id * segment_rows;
  uint64_t segment_end = segment_start + segment_rows;
  uint64_t tile_start = segment_start + tile_in_segment * tile_rows;

  if (tile_start >= n || tile_start >= segment_end)
  {
    if (threadIdx.x == 0)
      d_partial_out[blockIdx.x] = 0.0;
    return;
  }

  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > segment_end)
    tile_end = segment_end;
  if (tile_end > n)
    tile_end = n;
  uint64_t tile_rows_actual = tile_end - tile_start;

  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0)
    ++aligned_start;

  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull;
  uint64_t pack_count = (full_pack_end - aligned_start) / 16ull;

  unsigned long long sub0 = 0ull;
  unsigned long long sub1 = 0ull;
  unsigned long long sub2 = 0ull;
  unsigned long long sub3 = 0ull;
  unsigned long long sub4 = 0ull;
  unsigned long long sub5 = 0ull;
  unsigned long long sub6 = 0ull;
  unsigned long long sub7 = 0ull;
  unsigned long long sub8 = 0ull;
  unsigned long long sub9 = 0ull;
  unsigned long long sub10 = 0ull;
  unsigned long long sub11 = 0ull;
  unsigned long long sub12 = 0ull;
  unsigned long long sub13 = 0ull;
  unsigned long long sub14 = 0ull;
  unsigned long long sub15 = 0ull;

  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    exp3_detail::add_scalar_row_runtime_fixed<PLANE_LIMIT>(subcolumns,
                                                           row,
                                                           sub0,
                                                           sub1,
                                                           sub2,
                                                           sub3,
                                                           sub4,
                                                           sub5,
                                                           sub6,
                                                           sub7,
                                                           sub8,
                                                           sub9,
                                                           sub10,
                                                           sub11,
                                                           sub12,
                                                           sub13,
                                                           sub14,
                                                           sub15);
  }

  for (uint64_t pack = static_cast<uint64_t>(threadIdx.x);
       pack < pack_count;
       pack += static_cast<uint64_t>(blockDim.x))
  {
    exp3_detail::add_rowpack16_runtime_fixed<PLANE_LIMIT>(subcolumns,
                                                          first_pack + pack,
                                                          sub0,
                                                          sub1,
                                                          sub2,
                                                          sub3,
                                                          sub4,
                                                          sub5,
                                                          sub6,
                                                          sub7,
                                                          sub8,
                                                          sub9,
                                                          sub10,
                                                          sub11,
                                                          sub12,
                                                          sub13,
                                                          sub14,
                                                          sub15);
  }

  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    exp3_detail::add_scalar_row_runtime_fixed<PLANE_LIMIT>(subcolumns,
                                                           row,
                                                           sub0,
                                                           sub1,
                                                           sub2,
                                                           sub3,
                                                           sub4,
                                                           sub5,
                                                           sub6,
                                                           sub7,
                                                           sub8,
                                                           sub9,
                                                           sub10,
                                                           sub11,
                                                           sub12,
                                                           sub13,
                                                           sub14,
                                                           sub15);
  }

  __shared__ unsigned long long reduce_smem[PLANE_LIMIT][32];
  unsigned long long sub0_block = exp3_block_reduce_sum_ull(sub0, reduce_smem[0]);
  unsigned long long sub1_block = 0ull;
  unsigned long long sub2_block = 0ull;
  unsigned long long sub3_block = 0ull;
  unsigned long long sub4_block = 0ull;
  unsigned long long sub5_block = 0ull;
  unsigned long long sub6_block = 0ull;
  unsigned long long sub7_block = 0ull;
  unsigned long long sub8_block = 0ull;
  unsigned long long sub9_block = 0ull;
  unsigned long long sub10_block = 0ull;
  unsigned long long sub11_block = 0ull;
  unsigned long long sub12_block = 0ull;
  unsigned long long sub13_block = 0ull;
  unsigned long long sub14_block = 0ull;
  unsigned long long sub15_block = 0ull;

  if constexpr (PLANE_LIMIT >= 2)
    sub1_block = exp3_block_reduce_sum_ull(sub1, reduce_smem[1]);
  if constexpr (PLANE_LIMIT >= 3)
    sub2_block = exp3_block_reduce_sum_ull(sub2, reduce_smem[2]);
  if constexpr (PLANE_LIMIT >= 4)
    sub3_block = exp3_block_reduce_sum_ull(sub3, reduce_smem[3]);
  if constexpr (PLANE_LIMIT >= 5)
    sub4_block = exp3_block_reduce_sum_ull(sub4, reduce_smem[4]);
  if constexpr (PLANE_LIMIT >= 6)
    sub5_block = exp3_block_reduce_sum_ull(sub5, reduce_smem[5]);
  if constexpr (PLANE_LIMIT >= 7)
    sub6_block = exp3_block_reduce_sum_ull(sub6, reduce_smem[6]);
  if constexpr (PLANE_LIMIT >= 8)
    sub7_block = exp3_block_reduce_sum_ull(sub7, reduce_smem[7]);
  if constexpr (PLANE_LIMIT >= 9)
    sub8_block = exp3_block_reduce_sum_ull(sub8, reduce_smem[8]);
  if constexpr (PLANE_LIMIT >= 10)
    sub9_block = exp3_block_reduce_sum_ull(sub9, reduce_smem[9]);
  if constexpr (PLANE_LIMIT >= 11)
    sub10_block = exp3_block_reduce_sum_ull(sub10, reduce_smem[10]);
  if constexpr (PLANE_LIMIT >= 12)
    sub11_block = exp3_block_reduce_sum_ull(sub11, reduce_smem[11]);
  if constexpr (PLANE_LIMIT >= 13)
    sub12_block = exp3_block_reduce_sum_ull(sub12, reduce_smem[12]);
  if constexpr (PLANE_LIMIT >= 14)
    sub13_block = exp3_block_reduce_sum_ull(sub13, reduce_smem[13]);
  if constexpr (PLANE_LIMIT >= 15)
    sub14_block = exp3_block_reduce_sum_ull(sub14, reduce_smem[14]);
  if constexpr (PLANE_LIMIT >= 16)
    sub15_block = exp3_block_reduce_sum_ull(sub15, reduce_smem[15]);

  if (threadIdx.x == 0)
  {
    const double *basis = d_subcolumn_basis + segment_id * basis_stride;
    double partial = static_cast<double>(tile_rows_actual) * d_segment_base[segment_id];

    partial += static_cast<double>(sub0_block) * basis[0];
    if constexpr (PLANE_LIMIT >= 2)
      partial += static_cast<double>(sub1_block) * basis[1];
    if constexpr (PLANE_LIMIT >= 3)
      partial += static_cast<double>(sub2_block) * basis[2];
    if constexpr (PLANE_LIMIT >= 4)
      partial += static_cast<double>(sub3_block) * basis[3];
    if constexpr (PLANE_LIMIT >= 5)
      partial += static_cast<double>(sub4_block) * basis[4];
    if constexpr (PLANE_LIMIT >= 6)
      partial += static_cast<double>(sub5_block) * basis[5];
    if constexpr (PLANE_LIMIT >= 7)
      partial += static_cast<double>(sub6_block) * basis[6];
    if constexpr (PLANE_LIMIT >= 8)
      partial += static_cast<double>(sub7_block) * basis[7];
    if constexpr (PLANE_LIMIT >= 9)
      partial += static_cast<double>(sub8_block) * basis[8];
    if constexpr (PLANE_LIMIT >= 10)
      partial += static_cast<double>(sub9_block) * basis[9];
    if constexpr (PLANE_LIMIT >= 11)
      partial += static_cast<double>(sub10_block) * basis[10];
    if constexpr (PLANE_LIMIT >= 12)
      partial += static_cast<double>(sub11_block) * basis[11];
    if constexpr (PLANE_LIMIT >= 13)
      partial += static_cast<double>(sub12_block) * basis[12];
    if constexpr (PLANE_LIMIT >= 14)
      partial += static_cast<double>(sub13_block) * basis[13];
    if constexpr (PLANE_LIMIT >= 15)
      partial += static_cast<double>(sub14_block) * basis[14];
    if constexpr (PLANE_LIMIT >= 16)
      partial += static_cast<double>(sub15_block) * basis[15];

    d_partial_out[blockIdx.x] = partial;
  }
}

inline void launch_progressive_sum_rowpack16_runtime(int refinement_depth,
                                                     int grid,
                                                     int block_threads,
                                                     const Exp3RuntimeU8Subcolumns &subcolumns,
                                                     int max_planes,
                                                     uint64_t n,
                                                     uint64_t segment_rows,
                                                     uint64_t tiles_per_segment,
                                                     const double *d_segment_base,
                                                     const double *d_subcolumn_basis,
                                                     double *d_partial_out)
{
  progressive_sum_rowpack16_u8subcols_runtime<<<grid, block_threads>>>(
      subcolumns,
      max_planes,
      refinement_depth,
      n,
      segment_rows,
      tiles_per_segment,
      d_segment_base,
      d_subcolumn_basis,
      d_partial_out);
}

inline void launch_progressive_sum_rowpack16_runtime_fixed(int refinement_depth,
                                                           int grid,
                                                           int block_threads,
                                                           const Exp3RuntimeU8Subcolumns &subcolumns,
                                                           int max_planes,
                                                           uint64_t n,
                                                           uint64_t segment_rows,
                                                           uint64_t tiles_per_segment,
                                                           const double *d_segment_base,
                                                           const double *d_subcolumn_basis,
                                                           double *d_partial_out)
{
  int plane_limit = refinement_depth + 1;
  if (plane_limit > max_planes)
    plane_limit = max_planes;

  switch (plane_limit)
  {
  case 1:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<1><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 2:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<2><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 3:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<3><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 4:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<4><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 5:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<5><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 6:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<6><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 7:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<7><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 8:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<8><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 9:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<9><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 10:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<10><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 11:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<11><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 12:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<12><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 13:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<13><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 14:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<14><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 15:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<15><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  case 16:
    progressive_sum_rowpack16_u8subcols_runtime_fixed<16><<<grid, block_threads>>>(
        subcolumns, max_planes, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
    return;
  default:
    std::abort();
  }
}

template <int K_READ, int PLANE_COUNT>
__global__ void progressive_sum_real_rowpack16_specialized(Exp3RuntimeU8Subcolumns subcolumns,
                                                           uint64_t n,
                                                           uint64_t segment_rows,
                                                           uint64_t tiles_per_segment,
                                                           const double *__restrict__ d_segment_base,
                                                           const double *__restrict__ d_subcolumn_basis,
                                                           double *__restrict__ d_partial_out)
{
  static_assert(PLANE_COUNT >= 1 && PLANE_COUNT <= 16, "PLANE_COUNT must be in [1, 16]");
  static_assert(K_READ >= 1 && K_READ <= PLANE_COUNT, "K_READ must be in [1, PLANE_COUNT]");

  uint64_t tile_rows = static_cast<uint64_t>(blockDim.x) *
                       static_cast<uint64_t>(EXP3_ROWPACK16_WIDTH);
  uint64_t segment_id = static_cast<uint64_t>(blockIdx.x) / tiles_per_segment;
  uint64_t tile_in_segment = static_cast<uint64_t>(blockIdx.x) % tiles_per_segment;
  uint64_t segment_start = segment_id * segment_rows;
  uint64_t segment_end = segment_start + segment_rows;
  uint64_t tile_start = segment_start + tile_in_segment * tile_rows;

  if (tile_start >= n || tile_start >= segment_end)
  {
    if (threadIdx.x == 0)
      d_partial_out[blockIdx.x] = 0.0;
    return;
  }

  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > segment_end)
    tile_end = segment_end;
  if (tile_end > n)
    tile_end = n;
  uint64_t tile_rows_actual = tile_end - tile_start;

  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0)
    ++aligned_start;

  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull;
  uint64_t pack_count = (full_pack_end - aligned_start) / 16ull;

  unsigned long long sub0 = 0ull;
  unsigned long long sub1 = 0ull;
  unsigned long long sub2 = 0ull;
  unsigned long long sub3 = 0ull;
  unsigned long long sub4 = 0ull;
  unsigned long long sub5 = 0ull;
  unsigned long long sub6 = 0ull;
  unsigned long long sub7 = 0ull;
  unsigned long long sub8 = 0ull;
  unsigned long long sub9 = 0ull;
  unsigned long long sub10 = 0ull;
  unsigned long long sub11 = 0ull;
  unsigned long long sub12 = 0ull;
  unsigned long long sub13 = 0ull;
  unsigned long long sub14 = 0ull;
  unsigned long long sub15 = 0ull;

  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    exp3_detail::add_scalar_row_runtime_fixed<K_READ>(subcolumns,
                                                      row,
                                                      sub0,
                                                      sub1,
                                                      sub2,
                                                      sub3,
                                                      sub4,
                                                      sub5,
                                                      sub6,
                                                      sub7,
                                                      sub8,
                                                      sub9,
                                                      sub10,
                                                      sub11,
                                                      sub12,
                                                      sub13,
                                                      sub14,
                                                      sub15);
  }

  for (uint64_t pack = static_cast<uint64_t>(threadIdx.x);
       pack < pack_count;
       pack += static_cast<uint64_t>(blockDim.x))
  {
    exp3_detail::add_rowpack16_runtime_fixed<K_READ>(subcolumns,
                                                     first_pack + pack,
                                                     sub0,
                                                     sub1,
                                                     sub2,
                                                     sub3,
                                                     sub4,
                                                     sub5,
                                                     sub6,
                                                     sub7,
                                                     sub8,
                                                     sub9,
                                                     sub10,
                                                     sub11,
                                                     sub12,
                                                     sub13,
                                                     sub14,
                                                     sub15);
  }

  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    exp3_detail::add_scalar_row_runtime_fixed<K_READ>(subcolumns,
                                                      row,
                                                      sub0,
                                                      sub1,
                                                      sub2,
                                                      sub3,
                                                      sub4,
                                                      sub5,
                                                      sub6,
                                                      sub7,
                                                      sub8,
                                                      sub9,
                                                      sub10,
                                                      sub11,
                                                      sub12,
                                                      sub13,
                                                      sub14,
                                                      sub15);
  }

  __shared__ unsigned long long reduce_smem[K_READ][32];
  unsigned long long sub0_block = exp3_block_reduce_sum_ull(sub0, reduce_smem[0]);
  unsigned long long sub1_block = 0ull;
  unsigned long long sub2_block = 0ull;
  unsigned long long sub3_block = 0ull;
  unsigned long long sub4_block = 0ull;
  unsigned long long sub5_block = 0ull;
  unsigned long long sub6_block = 0ull;
  unsigned long long sub7_block = 0ull;
  unsigned long long sub8_block = 0ull;
  unsigned long long sub9_block = 0ull;
  unsigned long long sub10_block = 0ull;
  unsigned long long sub11_block = 0ull;
  unsigned long long sub12_block = 0ull;
  unsigned long long sub13_block = 0ull;
  unsigned long long sub14_block = 0ull;
  unsigned long long sub15_block = 0ull;

  if constexpr (K_READ >= 2)
    sub1_block = exp3_block_reduce_sum_ull(sub1, reduce_smem[1]);
  if constexpr (K_READ >= 3)
    sub2_block = exp3_block_reduce_sum_ull(sub2, reduce_smem[2]);
  if constexpr (K_READ >= 4)
    sub3_block = exp3_block_reduce_sum_ull(sub3, reduce_smem[3]);
  if constexpr (K_READ >= 5)
    sub4_block = exp3_block_reduce_sum_ull(sub4, reduce_smem[4]);
  if constexpr (K_READ >= 6)
    sub5_block = exp3_block_reduce_sum_ull(sub5, reduce_smem[5]);
  if constexpr (K_READ >= 7)
    sub6_block = exp3_block_reduce_sum_ull(sub6, reduce_smem[6]);
  if constexpr (K_READ >= 8)
    sub7_block = exp3_block_reduce_sum_ull(sub7, reduce_smem[7]);
  if constexpr (K_READ >= 9)
    sub8_block = exp3_block_reduce_sum_ull(sub8, reduce_smem[8]);
  if constexpr (K_READ >= 10)
    sub9_block = exp3_block_reduce_sum_ull(sub9, reduce_smem[9]);
  if constexpr (K_READ >= 11)
    sub10_block = exp3_block_reduce_sum_ull(sub10, reduce_smem[10]);
  if constexpr (K_READ >= 12)
    sub11_block = exp3_block_reduce_sum_ull(sub11, reduce_smem[11]);
  if constexpr (K_READ >= 13)
    sub12_block = exp3_block_reduce_sum_ull(sub12, reduce_smem[12]);
  if constexpr (K_READ >= 14)
    sub13_block = exp3_block_reduce_sum_ull(sub13, reduce_smem[13]);
  if constexpr (K_READ >= 15)
    sub14_block = exp3_block_reduce_sum_ull(sub14, reduce_smem[14]);
  if constexpr (K_READ >= 16)
    sub15_block = exp3_block_reduce_sum_ull(sub15, reduce_smem[15]);

  if (threadIdx.x == 0)
  {
    const double *basis = d_subcolumn_basis + segment_id * PLANE_COUNT;
    double partial = static_cast<double>(tile_rows_actual) * d_segment_base[segment_id];

    partial += static_cast<double>(sub0_block) * basis[0];
    if constexpr (K_READ >= 2)
      partial += static_cast<double>(sub1_block) * basis[1];
    if constexpr (K_READ >= 3)
      partial += static_cast<double>(sub2_block) * basis[2];
    if constexpr (K_READ >= 4)
      partial += static_cast<double>(sub3_block) * basis[3];
    if constexpr (K_READ >= 5)
      partial += static_cast<double>(sub4_block) * basis[4];
    if constexpr (K_READ >= 6)
      partial += static_cast<double>(sub5_block) * basis[5];
    if constexpr (K_READ >= 7)
      partial += static_cast<double>(sub6_block) * basis[6];
    if constexpr (K_READ >= 8)
      partial += static_cast<double>(sub7_block) * basis[7];
    if constexpr (K_READ >= 9)
      partial += static_cast<double>(sub8_block) * basis[8];
    if constexpr (K_READ >= 10)
      partial += static_cast<double>(sub9_block) * basis[9];
    if constexpr (K_READ >= 11)
      partial += static_cast<double>(sub10_block) * basis[10];
    if constexpr (K_READ >= 12)
      partial += static_cast<double>(sub11_block) * basis[11];
    if constexpr (K_READ >= 13)
      partial += static_cast<double>(sub12_block) * basis[12];
    if constexpr (K_READ >= 14)
      partial += static_cast<double>(sub13_block) * basis[13];
    if constexpr (K_READ >= 15)
      partial += static_cast<double>(sub14_block) * basis[14];
    if constexpr (K_READ >= 16)
      partial += static_cast<double>(sub15_block) * basis[15];

    d_partial_out[blockIdx.x] = partial;
  }
}

template <int K_READ, int PLANE_COUNT>
inline void launch_progressive_sum_real_rowpack16_specialized_k(int grid,
                                                                int block_threads,
                                                                const Exp3RuntimeU8Subcolumns &subcolumns,
                                                                uint64_t n,
                                                                uint64_t segment_rows,
                                                                uint64_t tiles_per_segment,
                                                                const double *d_segment_base,
                                                                const double *d_subcolumn_basis,
                                                                double *d_partial_out)
{
  progressive_sum_real_rowpack16_specialized<K_READ, PLANE_COUNT><<<grid, block_threads>>>(
      subcolumns, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out);
}

template <int K_READ, int PLANE_COUNT>
inline void launch_progressive_sum_real_rowpack16_specialized_k_dispatch(int target_k,
                                                                          int grid,
                                                                          int block_threads,
                                                                          const Exp3RuntimeU8Subcolumns &subcolumns,
                                                                          uint64_t n,
                                                                          uint64_t segment_rows,
                                                                          uint64_t tiles_per_segment,
                                                                          const double *d_segment_base,
                                                                          const double *d_subcolumn_basis,
                                                                          double *d_partial_out)
{
  if (target_k == K_READ)
  {
    launch_progressive_sum_real_rowpack16_specialized_k<K_READ, PLANE_COUNT>(grid,
                                                                             block_threads,
                                                                             subcolumns,
                                                                             n,
                                                                             segment_rows,
                                                                             tiles_per_segment,
                                                                             d_segment_base,
                                                                             d_subcolumn_basis,
                                                                             d_partial_out);
    return;
  }
  if constexpr (K_READ < PLANE_COUNT)
  {
    launch_progressive_sum_real_rowpack16_specialized_k_dispatch<K_READ + 1, PLANE_COUNT>(target_k,
                                                                                          grid,
                                                                                          block_threads,
                                                                                          subcolumns,
                                                                                          n,
                                                                                          segment_rows,
                                                                                          tiles_per_segment,
                                                                                          d_segment_base,
                                                                                          d_subcolumn_basis,
                                                                                          d_partial_out);
    return;
  }
  int fallback_depth = target_k > 0 ? target_k - 1 : 0;
  launch_progressive_sum_rowpack16_runtime(fallback_depth,
                                           grid,
                                           block_threads,
                                           subcolumns,
                                           PLANE_COUNT,
                                           n,
                                           segment_rows,
                                           tiles_per_segment,
                                           d_segment_base,
                                           d_subcolumn_basis,
                                           d_partial_out);
}

inline void launch_progressive_sum_real_rowpack16_specialized(int refinement_depth,
                                                              int grid,
                                                              int block_threads,
                                                              const Exp3RuntimeU8Subcolumns &subcolumns,
                                                              int max_planes,
                                                              uint64_t n,
                                                              uint64_t segment_rows,
                                                              uint64_t tiles_per_segment,
                                                              const double *d_segment_base,
                                                              const double *d_subcolumn_basis,
                                                              double *d_partial_out)
{
  int k_read = refinement_depth + 1;
  if (k_read > max_planes)
    k_read = max_planes;

#define LAUNCH_CASE(N) do { \
    launch_progressive_sum_real_rowpack16_specialized_k_dispatch<1, N>(k_read, grid, block_threads, subcolumns, n, segment_rows, tiles_per_segment, d_segment_base, d_subcolumn_basis, d_partial_out); \
    return; \
  } while(0)

  switch (max_planes)
  {
  case 1: LAUNCH_CASE(1);
  case 2: LAUNCH_CASE(2);
  case 3: LAUNCH_CASE(3);
  case 4: LAUNCH_CASE(4);
  case 5: LAUNCH_CASE(5);
  case 6: LAUNCH_CASE(6);
  case 7: LAUNCH_CASE(7);
  case 8: LAUNCH_CASE(8);
  case 9: LAUNCH_CASE(9);
  case 10: LAUNCH_CASE(10);
  case 11: LAUNCH_CASE(11);
  case 12: LAUNCH_CASE(12);
  case 13: LAUNCH_CASE(13);
  case 14: LAUNCH_CASE(14);
  case 15: LAUNCH_CASE(15);
  case 16: LAUNCH_CASE(16);
  default:
    launch_progressive_sum_rowpack16_runtime(refinement_depth,
                                             grid,
                                             block_threads,
                                             subcolumns,
                                             max_planes,
                                             n,
                                             segment_rows,
                                             tiles_per_segment,
                                             d_segment_base,
                                             d_subcolumn_basis,
                                             d_partial_out);
    return;
  }
}
