#pragma once

#include "../../benchmarks/experiment3/exp3_common.cuh"
#include <cstdint>

namespace exp4_detail
{

__device__ __forceinline__ void extract_uint4_bytes(uint4 pack, uint8_t out[16])
{
  out[0]  = static_cast<uint8_t>(pack.x);
  out[1]  = static_cast<uint8_t>(pack.x >> 8);
  out[2]  = static_cast<uint8_t>(pack.x >> 16);
  out[3]  = static_cast<uint8_t>(pack.x >> 24);
  out[4]  = static_cast<uint8_t>(pack.y);
  out[5]  = static_cast<uint8_t>(pack.y >> 8);
  out[6]  = static_cast<uint8_t>(pack.y >> 16);
  out[7]  = static_cast<uint8_t>(pack.y >> 24);
  out[8]  = static_cast<uint8_t>(pack.z);
  out[9]  = static_cast<uint8_t>(pack.z >> 8);
  out[10] = static_cast<uint8_t>(pack.z >> 16);
  out[11] = static_cast<uint8_t>(pack.z >> 24);
  out[12] = static_cast<uint8_t>(pack.w);
  out[13] = static_cast<uint8_t>(pack.w >> 8);
  out[14] = static_cast<uint8_t>(pack.w >> 16);
  out[15] = static_cast<uint8_t>(pack.w >> 24);
}

// Compute 16-bit gt_mask and lt_mask directly from a uint4 and threshold byte,
// without extracting into an intermediate uint8_t[16] array.
// Byte order matches extract_uint4_bytes: pack.x → bits 0-3, pack.y → bits 4-7,
// pack.z → bits 8-11, pack.w → bits 12-15 (little-endian per component).
__device__ __forceinline__ void uint4_gt_lt_masks(
    uint4 pack, uint8_t thresh,
    uint16_t& gt_mask, uint16_t& lt_mask)
{
  gt_mask = 0;
  lt_mask = 0;

  // pack.x → byte indices 0-3
  uint32_t w = pack.x;
  if (static_cast<uint8_t>(w) > thresh) gt_mask |= static_cast<uint16_t>(1u << 0);
  if (static_cast<uint8_t>(w) < thresh) lt_mask |= static_cast<uint16_t>(1u << 0);
  if (static_cast<uint8_t>(w >> 8)  > thresh) gt_mask |= static_cast<uint16_t>(1u << 1);
  if (static_cast<uint8_t>(w >> 8)  < thresh) lt_mask |= static_cast<uint16_t>(1u << 1);
  if (static_cast<uint8_t>(w >> 16) > thresh) gt_mask |= static_cast<uint16_t>(1u << 2);
  if (static_cast<uint8_t>(w >> 16) < thresh) lt_mask |= static_cast<uint16_t>(1u << 2);
  if (static_cast<uint8_t>(w >> 24) > thresh) gt_mask |= static_cast<uint16_t>(1u << 3);
  if (static_cast<uint8_t>(w >> 24) < thresh) lt_mask |= static_cast<uint16_t>(1u << 3);

  // pack.y → byte indices 4-7
  w = pack.y;
  if (static_cast<uint8_t>(w) > thresh) gt_mask |= static_cast<uint16_t>(1u << 4);
  if (static_cast<uint8_t>(w) < thresh) lt_mask |= static_cast<uint16_t>(1u << 4);
  if (static_cast<uint8_t>(w >> 8)  > thresh) gt_mask |= static_cast<uint16_t>(1u << 5);
  if (static_cast<uint8_t>(w >> 8)  < thresh) lt_mask |= static_cast<uint16_t>(1u << 5);
  if (static_cast<uint8_t>(w >> 16) > thresh) gt_mask |= static_cast<uint16_t>(1u << 6);
  if (static_cast<uint8_t>(w >> 16) < thresh) lt_mask |= static_cast<uint16_t>(1u << 6);
  if (static_cast<uint8_t>(w >> 24) > thresh) gt_mask |= static_cast<uint16_t>(1u << 7);
  if (static_cast<uint8_t>(w >> 24) < thresh) lt_mask |= static_cast<uint16_t>(1u << 7);

  // pack.z → byte indices 8-11
  w = pack.z;
  if (static_cast<uint8_t>(w) > thresh) gt_mask |= static_cast<uint16_t>(1u << 8);
  if (static_cast<uint8_t>(w) < thresh) lt_mask |= static_cast<uint16_t>(1u << 8);
  if (static_cast<uint8_t>(w >> 8)  > thresh) gt_mask |= static_cast<uint16_t>(1u << 9);
  if (static_cast<uint8_t>(w >> 8)  < thresh) lt_mask |= static_cast<uint16_t>(1u << 9);
  if (static_cast<uint8_t>(w >> 16) > thresh) gt_mask |= static_cast<uint16_t>(1u << 10);
  if (static_cast<uint8_t>(w >> 16) < thresh) lt_mask |= static_cast<uint16_t>(1u << 10);
  if (static_cast<uint8_t>(w >> 24) > thresh) gt_mask |= static_cast<uint16_t>(1u << 11);
  if (static_cast<uint8_t>(w >> 24) < thresh) lt_mask |= static_cast<uint16_t>(1u << 11);

  // pack.w → byte indices 12-15
  w = pack.w;
  if (static_cast<uint8_t>(w) > thresh) gt_mask |= static_cast<uint16_t>(1u << 12);
  if (static_cast<uint8_t>(w) < thresh) lt_mask |= static_cast<uint16_t>(1u << 12);
  if (static_cast<uint8_t>(w >> 8)  > thresh) gt_mask |= static_cast<uint16_t>(1u << 13);
  if (static_cast<uint8_t>(w >> 8)  < thresh) lt_mask |= static_cast<uint16_t>(1u << 13);
  if (static_cast<uint8_t>(w >> 16) > thresh) gt_mask |= static_cast<uint16_t>(1u << 14);
  if (static_cast<uint8_t>(w >> 16) < thresh) lt_mask |= static_cast<uint16_t>(1u << 14);
  if (static_cast<uint8_t>(w >> 24) > thresh) gt_mask |= static_cast<uint16_t>(1u << 15);
  if (static_cast<uint8_t>(w >> 24) < thresh) lt_mask |= static_cast<uint16_t>(1u << 15);
}

// Convert a __vcmp[g|l]tu4 result (0xFF per byte-lane where true, 0x00 elsewhere)
// into a 4-bit mask (bit 0 = LSB byte, bit 3 = MSB byte).
__device__ __forceinline__ uint32_t mask4_from_vcmp_result(uint32_t cmp)
{
  // Each byte of cmp is 0xFF (true) or 0x00 (false).
  // Shift right 7 → each byte becomes 0x01 or 0x00.
  uint32_t t = cmp >> 7u;
  t &= 0x01010101u;
  // Pack the four bytes' bit 0 into bits 0-3.
  uint32_t bits = (t & 1u);
  bits |= (t >> 7u) & 2u;
  bits |= (t >> 14u) & 4u;
  bits |= (t >> 21u) & 8u;
  return bits;
}

// SIMD-accelerated version of uint4_gt_lt_masks.
// Same semantics and byte order: pack.x → bits 0-3, pack.y → bits 4-7,
// pack.z → bits 8-11, pack.w → bits 12-15 (little-endian per component).
// Uses __vcmpgtu4/__vcmpltu4 to reduce scalar byte-wise ALU operations.
__device__ __forceinline__ void uint4_gt_lt_masks_simd(
    uint4 pack, uint8_t thresh,
    uint16_t& gt_mask, uint16_t& lt_mask)
{
  // Broadcast threshold byte to all 4 byte lanes of a uint32.
  uint32_t tb = static_cast<uint32_t>(thresh) * 0x01010101u;

  gt_mask = 0;
  lt_mask = 0;

  // pack.x → byte indices 0-3
  {
    uint32_t gt_x = __vcmpgtu4(pack.x, tb);
    uint32_t lt_x = __vcmpltu4(pack.x, tb);
    gt_mask |= static_cast<uint16_t>(mask4_from_vcmp_result(gt_x) << 0);
    lt_mask |= static_cast<uint16_t>(mask4_from_vcmp_result(lt_x) << 0);
  }

  // pack.y → byte indices 4-7
  {
    uint32_t gt_y = __vcmpgtu4(pack.y, tb);
    uint32_t lt_y = __vcmpltu4(pack.y, tb);
    gt_mask |= static_cast<uint16_t>(mask4_from_vcmp_result(gt_y) << 4);
    lt_mask |= static_cast<uint16_t>(mask4_from_vcmp_result(lt_y) << 4);
  }

  // pack.z → byte indices 8-11
  {
    uint32_t gt_z = __vcmpgtu4(pack.z, tb);
    uint32_t lt_z = __vcmpltu4(pack.z, tb);
    gt_mask |= static_cast<uint16_t>(mask4_from_vcmp_result(gt_z) << 8);
    lt_mask |= static_cast<uint16_t>(mask4_from_vcmp_result(lt_z) << 8);
  }

  // pack.w → byte indices 12-15
  {
    uint32_t gt_w = __vcmpgtu4(pack.w, tb);
    uint32_t lt_w = __vcmpltu4(pack.w, tb);
    gt_mask |= static_cast<uint16_t>(mask4_from_vcmp_result(gt_w) << 12);
    lt_mask |= static_cast<uint16_t>(mask4_from_vcmp_result(lt_w) << 12);
  }
}

} // namespace exp4_detail

__global__ void progressive_filter_rowpack16_passive(
    Exp3RuntimeU8Subcolumns subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *__restrict__ d_threshold_bytes,
    int threshold_stride,
    const uint32_t *__restrict__ d_active_plane_count,
    uint64_t *__restrict__ d_block_counts)
{
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
      d_block_counts[blockIdx.x] = 0ull;
    return;
  }

  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > segment_end)
    tile_end = segment_end;
  if (tile_end > n)
    tile_end = n;

  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0)
    ++aligned_start;

  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull;
  uint64_t pack_count = (full_pack_end - aligned_start) / 16ull;

  uint32_t max_rounds = d_active_plane_count[segment_id];
  uint64_t thresh_offset = segment_id * static_cast<uint64_t>(threshold_stride);

  uint32_t local_count = 0u;

  // Scalar prefix rows (each thread handles a subset)
  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    bool active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)
      {
        qualified = true;
        active = false;
      }
      else if (row_byte < thresh_byte)
      {
        active = false;
      }
    }
    if (qualified)
      ++local_count;
  }

  // Rowpack16 main loop
  for (uint64_t pack = static_cast<uint64_t>(threadIdx.x);
       pack < pack_count;
       pack += static_cast<uint64_t>(blockDim.x))
  {
    uint64_t pack_idx = first_pack + pack;
    uint16_t active_mask = 0xFFFFu;
    uint16_t qualified_mask = 0x0000u;

    for (uint32_t round = 0; round < max_rounds && active_mask != 0; ++round)
    {
      const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[round]);
      uint4 pack_data = plane128[pack_idx];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];

      uint8_t vals[16];
      exp4_detail::extract_uint4_bytes(pack_data, vals);

      // Unrolled comparison for 16 rows
      #pragma unroll
      for (int i = 0; i < 16; ++i)
      {
        uint16_t bit = static_cast<uint16_t>(1u << i);
        if (active_mask & bit)
        {
          if (vals[i] > thresh_byte)
          {
            qualified_mask |= bit;
            active_mask &= static_cast<uint16_t>(~bit);
          }
          else if (vals[i] < thresh_byte)
          {
            active_mask &= static_cast<uint16_t>(~bit);
          }
        }
      }
    }

    local_count += static_cast<uint32_t>(__popc(qualified_mask));
  }

  // Scalar suffix rows
  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    bool active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)
      {
        qualified = true;
        active = false;
      }
      else if (row_byte < thresh_byte)
      {
        active = false;
      }
    }
    if (qualified)
      ++local_count;
  }

  // Block reduction
  __shared__ unsigned long long reduce_smem[32];
  unsigned long long block_count = exp3_block_reduce_sum_ull(
      static_cast<unsigned long long>(local_count), reduce_smem);

  if (threadIdx.x == 0)
    d_block_counts[blockIdx.x] = block_count;
}

__global__ void progressive_filter_rowpack16_predicated(
    Exp3RuntimeU8Subcolumns subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *__restrict__ d_threshold_bytes,
    int threshold_stride,
    const uint32_t *__restrict__ d_active_plane_count,
    uint64_t *__restrict__ d_block_counts)
{
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
      d_block_counts[blockIdx.x] = 0ull;
    return;
  }

  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > segment_end)
    tile_end = segment_end;
  if (tile_end > n)
    tile_end = n;

  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0)
    ++aligned_start;

  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull;
  uint64_t pack_count = (full_pack_end - aligned_start) / 16ull;

  uint32_t max_rounds = d_active_plane_count[segment_id];
  uint64_t thresh_offset = segment_id * static_cast<uint64_t>(threshold_stride);

  uint32_t local_count = 0u;

  // Scalar prefix rows are unchanged from the passive kernel so the A/B test
  // isolates the rowpack16 inner-loop branch structure.
  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    bool active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)
      {
        qualified = true;
        active = false;
      }
      else if (row_byte < thresh_byte)
      {
        active = false;
      }
    }
    if (qualified)
      ++local_count;
  }

  for (uint64_t pack = static_cast<uint64_t>(threadIdx.x);
       pack < pack_count;
       pack += static_cast<uint64_t>(blockDim.x))
  {
    uint64_t pack_idx = first_pack + pack;
    uint16_t active_mask = 0xFFFFu;
    uint16_t qualified_mask = 0x0000u;

    for (uint32_t round = 0; round < max_rounds && active_mask != 0; ++round)
    {
      const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[round]);
      uint4 pack_data = plane128[pack_idx];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];

      uint8_t vals[16];
      exp4_detail::extract_uint4_bytes(pack_data, vals);

      uint16_t gt_mask = 0x0000u;
      uint16_t lt_mask = 0x0000u;

      #pragma unroll
      for (int i = 0; i < 16; ++i)
      {
        uint16_t bit = static_cast<uint16_t>(1u << i);
        gt_mask |= (vals[i] > thresh_byte) ? bit : 0u;
        lt_mask |= (vals[i] < thresh_byte) ? bit : 0u;
      }

      uint16_t newly_qualified = static_cast<uint16_t>(active_mask & gt_mask);
      uint16_t newly_resolved = static_cast<uint16_t>(active_mask & (gt_mask | lt_mask));
      qualified_mask = static_cast<uint16_t>(qualified_mask | newly_qualified);
      active_mask = static_cast<uint16_t>(active_mask & ~newly_resolved);
    }

    local_count += static_cast<uint32_t>(__popc(qualified_mask));
  }

  // Scalar suffix rows are unchanged from the passive kernel.
  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    bool active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)
      {
        qualified = true;
        active = false;
      }
      else if (row_byte < thresh_byte)
      {
        active = false;
      }
    }
    if (qualified)
      ++local_count;
  }

  __shared__ unsigned long long reduce_smem[32];
  unsigned long long block_count = exp3_block_reduce_sum_ull(
      static_cast<unsigned long long>(local_count), reduce_smem);

  if (threadIdx.x == 0)
    d_block_counts[blockIdx.x] = block_count;
}

__global__ void progressive_filter_rowpack16_byte_mask(
    Exp3RuntimeU8Subcolumns subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *__restrict__ d_threshold_bytes,
    int threshold_stride,
    const uint32_t *__restrict__ d_active_plane_count,
    uint64_t *__restrict__ d_block_counts)
{
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
      d_block_counts[blockIdx.x] = 0ull;
    return;
  }

  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > segment_end)
    tile_end = segment_end;
  if (tile_end > n)
    tile_end = n;

  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0)
    ++aligned_start;

  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull;
  uint64_t pack_count = (full_pack_end - aligned_start) / 16ull;

  uint32_t max_rounds = d_active_plane_count[segment_id];
  uint64_t thresh_offset = segment_id * static_cast<uint64_t>(threshold_stride);

  uint32_t local_count = 0u;

  // Scalar prefix rows are unchanged from predicated so the A/B test
  // isolates the rowpack16 inner-loop instruction mix.
  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    bool active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)
      {
        qualified = true;
        active = false;
      }
      else if (row_byte < thresh_byte)
      {
        active = false;
      }
    }
    if (qualified)
      ++local_count;
  }

  // Rowpack16 main loop — no vals[16] array, direct masks from uint4
  for (uint64_t pack = static_cast<uint64_t>(threadIdx.x);
       pack < pack_count;
       pack += static_cast<uint64_t>(blockDim.x))
  {
    uint64_t pack_idx = first_pack + pack;
    uint16_t active_mask = 0xFFFFu;
    uint16_t qualified_mask = 0x0000u;

    for (uint32_t round = 0; round < max_rounds && active_mask != 0; ++round)
    {
      const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[round]);
      uint4 pack_data = plane128[pack_idx];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];

      uint16_t gt_mask = 0x0000u;
      uint16_t lt_mask = 0x0000u;
      exp4_detail::uint4_gt_lt_masks(pack_data, thresh_byte, gt_mask, lt_mask);

      uint16_t newly_qualified = static_cast<uint16_t>(active_mask & gt_mask);
      uint16_t newly_resolved = static_cast<uint16_t>(active_mask & (gt_mask | lt_mask));
      qualified_mask = static_cast<uint16_t>(qualified_mask | newly_qualified);
      active_mask = static_cast<uint16_t>(active_mask & ~newly_resolved);
    }

    local_count += static_cast<uint32_t>(__popc(qualified_mask));
  }

  // Scalar suffix rows are unchanged from predicated.
  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    bool active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)
      {
        qualified = true;
        active = false;
      }
      else if (row_byte < thresh_byte)
      {
        active = false;
      }
    }
    if (qualified)
      ++local_count;
  }

  __shared__ unsigned long long reduce_smem[32];
  unsigned long long block_count = exp3_block_reduce_sum_ull(
      static_cast<unsigned long long>(local_count), reduce_smem);

  if (threadIdx.x == 0)
    d_block_counts[blockIdx.x] = block_count;
}

__global__ void progressive_filter_rowpack16_byte_mask_interleave2(
    Exp3RuntimeU8Subcolumns subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *__restrict__ d_threshold_bytes,
    int threshold_stride,
    const uint32_t *__restrict__ d_active_plane_count,
    uint64_t *__restrict__ d_block_counts)
{
  uint64_t tile_rows = static_cast<uint64_t>(blockDim.x) *
                       static_cast<uint64_t>(EXP3_ROWPACK16_WIDTH) * 2ull;
  uint64_t segment_id = static_cast<uint64_t>(blockIdx.x) / tiles_per_segment;
  uint64_t tile_in_segment = static_cast<uint64_t>(blockIdx.x) % tiles_per_segment;
  uint64_t segment_start = segment_id * segment_rows;
  uint64_t segment_end = segment_start + segment_rows;
  uint64_t tile_start = segment_start + tile_in_segment * tile_rows;

  if (tile_start >= n || tile_start >= segment_end)
  {
    if (threadIdx.x == 0)
      d_block_counts[blockIdx.x] = 0ull;
    return;
  }

  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > segment_end)
    tile_end = segment_end;
  if (tile_end > n)
    tile_end = n;

  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0)
    ++aligned_start;

  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull;
  uint64_t pack_count = (full_pack_end - aligned_start) / 16ull;

  uint32_t max_rounds = d_active_plane_count[segment_id];
  uint64_t thresh_offset = segment_id * static_cast<uint64_t>(threshold_stride);

  uint32_t local_count = 0u;

  // Scalar prefix rows are unchanged from byte_mask.
  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    bool active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)
      {
        qualified = true;
        active = false;
      }
      else if (row_byte < thresh_byte)
      {
        active = false;
      }
    }
    if (qualified)
      ++local_count;
  }

  // Rowpack16 main loop — interleave2: each thread processes 2 consecutive packs per iteration.
  for (uint64_t base = static_cast<uint64_t>(threadIdx.x) * 2ull;
       base < pack_count;
       base += static_cast<uint64_t>(blockDim.x) * 2ull)
  {
    uint64_t pack_idx0 = first_pack + base;
    uint64_t pack_idx1 = first_pack + base + 1ull;
    bool valid1 = (base + 1ull < pack_count);

    uint16_t active_mask0 = 0xFFFFu;
    uint16_t active_mask1 = 0xFFFFu;
    uint16_t qualified_mask0 = 0x0000u;
    uint16_t qualified_mask1 = 0x0000u;

    for (uint32_t round = 0; round < max_rounds && (active_mask0 != 0 || (valid1 && active_mask1 != 0)); ++round)
    {
      const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[round]);
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];

      // Pack 0
      if (active_mask0 != 0)
      {
        uint4 pack_data0 = plane128[pack_idx0];
        uint16_t gt_mask0 = 0x0000u;
        uint16_t lt_mask0 = 0x0000u;
        exp4_detail::uint4_gt_lt_masks(pack_data0, thresh_byte, gt_mask0, lt_mask0);

        uint16_t newly_qualified0 = static_cast<uint16_t>(active_mask0 & gt_mask0);
        uint16_t newly_resolved0 = static_cast<uint16_t>(active_mask0 & (gt_mask0 | lt_mask0));
        qualified_mask0 = static_cast<uint16_t>(qualified_mask0 | newly_qualified0);
        active_mask0 = static_cast<uint16_t>(active_mask0 & ~newly_resolved0);
      }

      // Pack 1 (if valid)
      if (valid1 && active_mask1 != 0)
      {
        uint4 pack_data1 = plane128[pack_idx1];
        uint16_t gt_mask1 = 0x0000u;
        uint16_t lt_mask1 = 0x0000u;
        exp4_detail::uint4_gt_lt_masks(pack_data1, thresh_byte, gt_mask1, lt_mask1);

        uint16_t newly_qualified1 = static_cast<uint16_t>(active_mask1 & gt_mask1);
        uint16_t newly_resolved1 = static_cast<uint16_t>(active_mask1 & (gt_mask1 | lt_mask1));
        qualified_mask1 = static_cast<uint16_t>(qualified_mask1 | newly_qualified1);
        active_mask1 = static_cast<uint16_t>(active_mask1 & ~newly_resolved1);
      }
    }

    local_count += static_cast<uint32_t>(__popc(qualified_mask0));
    if (valid1)
      local_count += static_cast<uint32_t>(__popc(qualified_mask1));
  }

  // Scalar suffix rows are unchanged from byte_mask.
  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    bool active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)
      {
        qualified = true;
        active = false;
      }
      else if (row_byte < thresh_byte)
      {
        active = false;
      }
    }
    if (qualified)
      ++local_count;
  }

  __shared__ unsigned long long reduce_smem[32];
  unsigned long long block_count = exp3_block_reduce_sum_ull(
      static_cast<unsigned long long>(local_count), reduce_smem);

  if (threadIdx.x == 0)
    d_block_counts[blockIdx.x] = block_count;
}

// ---------------------------------------------------------------------------
// P2-2: Specialized kernels with compile-time K (1/4/8) so that the
// compiler can unroll the round loop and propagate constants.
// effective_rounds = min(K, d_active_plane_count[segment_id]) ensures
// correctness for segments whose depth < K.
// ---------------------------------------------------------------------------
template<int K>
__global__ void progressive_filter_rowpack16_byte_mask_specialized(
    Exp3RuntimeU8Subcolumns subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *__restrict__ d_threshold_bytes,
    int threshold_stride,
    const uint32_t *__restrict__ d_active_plane_count,
    uint64_t *__restrict__ d_block_counts)
{
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
      d_block_counts[blockIdx.x] = 0ull;
    return;
  }

  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > segment_end)
    tile_end = segment_end;
  if (tile_end > n)
    tile_end = n;

  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0)
    ++aligned_start;

  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull;
  uint64_t pack_count = (full_pack_end - aligned_start) / 16ull;

  uint32_t max_rounds = d_active_plane_count[segment_id];
  uint32_t effective_rounds = min(max_rounds, static_cast<uint32_t>(K));
  uint64_t thresh_offset = segment_id * static_cast<uint64_t>(threshold_stride);

  uint32_t local_count = 0u;

  // Scalar prefix rows — use effective_rounds for correctness.
  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    bool active = true;
    for (uint32_t round = 0; round < effective_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)
      {
        qualified = true;
        active = false;
      }
      else if (row_byte < thresh_byte)
      {
        active = false;
      }
    }
    if (qualified)
      ++local_count;
  }

  // Rowpack16 main loop — compiler sees K as compile-time constant and unrolls.
  for (uint64_t pack = static_cast<uint64_t>(threadIdx.x);
       pack < pack_count;
       pack += static_cast<uint64_t>(blockDim.x))
  {
    uint64_t pack_idx = first_pack + pack;
    uint16_t active_mask = 0xFFFFu;
    uint16_t qualified_mask = 0x0000u;

    #pragma unroll
    for (uint32_t round = 0; round < K && active_mask != 0; ++round)
    {
      if (round < effective_rounds)
      {
        const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[round]);
        uint4 pack_data = plane128[pack_idx];
        uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];

        uint16_t gt_mask = 0x0000u;
        uint16_t lt_mask = 0x0000u;
        exp4_detail::uint4_gt_lt_masks(pack_data, thresh_byte, gt_mask, lt_mask);

        uint16_t newly_qualified = static_cast<uint16_t>(active_mask & gt_mask);
        uint16_t newly_resolved = static_cast<uint16_t>(active_mask & (gt_mask | lt_mask));
        qualified_mask = static_cast<uint16_t>(qualified_mask | newly_qualified);
        active_mask = static_cast<uint16_t>(active_mask & ~newly_resolved);
      }
    }

    local_count += static_cast<uint32_t>(__popc(qualified_mask));
  }

  // Scalar suffix rows — use effective_rounds.
  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    bool active = true;
    for (uint32_t round = 0; round < effective_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)
      {
        qualified = true;
        active = false;
      }
      else if (row_byte < thresh_byte)
      {
        active = false;
      }
    }
    if (qualified)
      ++local_count;
  }

  __shared__ unsigned long long reduce_smem[32];
  unsigned long long block_count = exp3_block_reduce_sum_ull(
      static_cast<unsigned long long>(local_count), reduce_smem);

  if (threadIdx.x == 0)
    d_block_counts[blockIdx.x] = block_count;
}

// Explicit template instantiations so the compiler emits device code for K=1,4,8.
template __global__ void progressive_filter_rowpack16_byte_mask_specialized<1>(
    Exp3RuntimeU8Subcolumns, uint64_t, uint64_t, uint64_t,
    const uint8_t*, int, const uint32_t*, uint64_t*);
template __global__ void progressive_filter_rowpack16_byte_mask_specialized<4>(
    Exp3RuntimeU8Subcolumns, uint64_t, uint64_t, uint64_t,
    const uint8_t*, int, const uint32_t*, uint64_t*);
template __global__ void progressive_filter_rowpack16_byte_mask_specialized<8>(
    Exp3RuntimeU8Subcolumns, uint64_t, uint64_t, uint64_t,
    const uint8_t*, int, const uint32_t*, uint64_t*);

__global__ void progressive_filter_rowpack16_simd_masks(
    Exp3RuntimeU8Subcolumns subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *__restrict__ d_threshold_bytes,
    int threshold_stride,
    const uint32_t *__restrict__ d_active_plane_count,
    uint64_t *__restrict__ d_block_counts)
{
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
      d_block_counts[blockIdx.x] = 0ull;
    return;
  }

  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > segment_end)
    tile_end = segment_end;
  if (tile_end > n)
    tile_end = n;

  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0)
    ++aligned_start;

  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull;
  uint64_t pack_count = (full_pack_end - aligned_start) / 16ull;

  uint32_t max_rounds = d_active_plane_count[segment_id];
  uint64_t thresh_offset = segment_id * static_cast<uint64_t>(threshold_stride);

  uint32_t local_count = 0u;

  // Scalar prefix rows — identical to byte_mask
  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    bool active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)
      {
        qualified = true;
        active = false;
      }
      else if (row_byte < thresh_byte)
      {
        active = false;
      }
    }
    if (qualified)
      ++local_count;
  }

  // Rowpack16 main loop — SIMD masks instead of scalar byte comparisons
  for (uint64_t pack = static_cast<uint64_t>(threadIdx.x);
       pack < pack_count;
       pack += static_cast<uint64_t>(blockDim.x))
  {
    uint64_t pack_idx = first_pack + pack;
    uint16_t active_mask = 0xFFFFu;
    uint16_t qualified_mask = 0x0000u;

    for (uint32_t round = 0; round < max_rounds && active_mask != 0; ++round)
    {
      const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[round]);
      uint4 pack_data = plane128[pack_idx];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];

      uint16_t gt_mask = 0x0000u;
      uint16_t lt_mask = 0x0000u;
      exp4_detail::uint4_gt_lt_masks_simd(pack_data, thresh_byte, gt_mask, lt_mask);

      uint16_t newly_qualified = static_cast<uint16_t>(active_mask & gt_mask);
      uint16_t newly_resolved = static_cast<uint16_t>(active_mask & (gt_mask | lt_mask));
      qualified_mask = static_cast<uint16_t>(qualified_mask | newly_qualified);
      active_mask = static_cast<uint16_t>(active_mask & ~newly_resolved);
    }

    local_count += static_cast<uint32_t>(__popc(qualified_mask));
  }

  // Scalar suffix rows — identical to byte_mask
  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    bool active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)
      {
        qualified = true;
        active = false;
      }
      else if (row_byte < thresh_byte)
      {
        active = false;
      }
    }
    if (qualified)
      ++local_count;
  }

  __shared__ unsigned long long reduce_smem[32];
  unsigned long long block_count = exp3_block_reduce_sum_ull(
      static_cast<unsigned long long>(local_count), reduce_smem);

  if (threadIdx.x == 0)
    d_block_counts[blockIdx.x] = block_count;
}

__global__ void progressive_filter_rowpack16_prefetch(
    Exp3RuntimeU8Subcolumns subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *__restrict__ d_threshold_bytes,
    int threshold_stride,
    const uint32_t *__restrict__ d_active_plane_count,
    uint64_t *__restrict__ d_block_counts)
{
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
      d_block_counts[blockIdx.x] = 0ull;
    return;
  }

  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > segment_end)
    tile_end = segment_end;
  if (tile_end > n)
    tile_end = n;

  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0)
    ++aligned_start;

  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull;
  uint64_t pack_count = (full_pack_end - aligned_start) / 16ull;

  uint32_t max_rounds = d_active_plane_count[segment_id];
  uint64_t thresh_offset = segment_id * static_cast<uint64_t>(threshold_stride);

  uint32_t local_count = 0u;

  // Scalar prefix rows — identical to byte_mask
  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    bool active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)
      {
        qualified = true;
        active = false;
      }
      else if (row_byte < thresh_byte)
      {
        active = false;
      }
    }
    if (qualified)
      ++local_count;
  }

  // Rowpack16 main loop — byte_mask inner loop + prefetch next plane
  for (uint64_t pack = static_cast<uint64_t>(threadIdx.x);
       pack < pack_count;
       pack += static_cast<uint64_t>(blockDim.x))
  {
    uint64_t pack_idx = first_pack + pack;
    uint16_t active_mask = 0xFFFFu;
    uint16_t qualified_mask = 0x0000u;

    for (uint32_t round = 0; round < max_rounds && active_mask != 0; ++round)
    {
      const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[round]);
      uint4 pack_data = plane128[pack_idx];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];

      uint16_t gt_mask = 0x0000u;
      uint16_t lt_mask = 0x0000u;
      exp4_detail::uint4_gt_lt_masks(pack_data, thresh_byte, gt_mask, lt_mask);

      uint16_t newly_qualified = static_cast<uint16_t>(active_mask & gt_mask);
      uint16_t newly_resolved = static_cast<uint16_t>(active_mask & (gt_mask | lt_mask));
      qualified_mask = static_cast<uint16_t>(qualified_mask | newly_qualified);
      active_mask = static_cast<uint16_t>(active_mask & ~newly_resolved);

      // Prefetch next round's pack data into L1
      if (round + 1 < max_rounds)
      {
        const uint4 *next_plane = reinterpret_cast<const uint4 *>(subcolumns.ptrs[round + 1]);
        asm volatile("prefetch.global.L1 [%0];" :: "l"(&next_plane[pack_idx]));
      }
    }

    local_count += static_cast<uint32_t>(__popc(qualified_mask));
  }

  // Scalar suffix rows — identical to byte_mask
  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    bool active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)
      {
        qualified = true;
        active = false;
      }
      else if (row_byte < thresh_byte)
      {
        active = false;
      }
    }
    if (qualified)
      ++local_count;
  }

  __shared__ unsigned long long reduce_smem[32];
  unsigned long long block_count = exp3_block_reduce_sum_ull(
      static_cast<unsigned long long>(local_count), reduce_smem);

  if (threadIdx.x == 0)
    d_block_counts[blockIdx.x] = block_count;
}

__global__ void fixed_depth_filter_rowpack16_count(
    Exp3RuntimeU8Subcolumns subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *__restrict__ d_threshold_bytes,
    int threshold_stride,
    const uint32_t *__restrict__ d_active_plane_count,
    const uint8_t *__restrict__ d_segment_filter_mode,
    uint64_t *__restrict__ d_block_counts)
{
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
      d_block_counts[blockIdx.x] = 0ull;
    return;
  }

  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > segment_end)
    tile_end = segment_end;
  if (tile_end > n)
    tile_end = n;

  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0)
    ++aligned_start;

  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull;
  uint64_t pack_count = (full_pack_end - aligned_start) / 16ull;

  uint32_t max_rounds = d_active_plane_count[segment_id];
  uint8_t segment_mode = d_segment_filter_mode[segment_id];
  uint64_t thresh_offset = segment_id * static_cast<uint64_t>(threshold_stride);

  uint32_t local_count = 0u;

  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    int cmp = 0;
    for (uint32_t round = 0; round < max_rounds; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      if (segment_mode == 0 && cmp == 0)
      {
        uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
        if (row_byte > thresh_byte)
          cmp = 1;
        else if (row_byte < thresh_byte)
          cmp = -1;
      }
    }

    if (segment_mode == 1 || (segment_mode == 0 && cmp > 0))
      ++local_count;
  }

  for (uint64_t pack = static_cast<uint64_t>(threadIdx.x);
       pack < pack_count;
       pack += static_cast<uint64_t>(blockDim.x))
  {
    uint64_t pack_idx = first_pack + pack;
    uint16_t undecided_mask = 0xFFFFu;
    uint16_t qualified_mask = 0x0000u;

    for (uint32_t round = 0; round < max_rounds; ++round)
    {
      const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[round]);
      uint4 pack_data = plane128[pack_idx];

      if (segment_mode == 0)
      {
        uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
        uint8_t vals[16];
        exp4_detail::extract_uint4_bytes(pack_data, vals);

        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
          uint16_t bit = static_cast<uint16_t>(1u << i);
          if (undecided_mask & bit)
          {
            if (vals[i] > thresh_byte)
            {
              qualified_mask |= bit;
              undecided_mask &= static_cast<uint16_t>(~bit);
            }
            else if (vals[i] < thresh_byte)
            {
              undecided_mask &= static_cast<uint16_t>(~bit);
            }
          }
        }
      }
    }

    if (segment_mode == 1)
      local_count += 16u;
    else if (segment_mode == 0)
      local_count += static_cast<uint32_t>(__popc(qualified_mask));
  }

  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    int cmp = 0;
    for (uint32_t round = 0; round < max_rounds; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      if (segment_mode == 0 && cmp == 0)
      {
        uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
        if (row_byte > thresh_byte)
          cmp = 1;
        else if (row_byte < thresh_byte)
          cmp = -1;
      }
    }

    if (segment_mode == 1 || (segment_mode == 0 && cmp > 0))
      ++local_count;
  }

  __shared__ unsigned long long reduce_smem[32];
  unsigned long long block_count = exp3_block_reduce_sum_ull(
      static_cast<unsigned long long>(local_count), reduce_smem);

  if (threadIdx.x == 0)
    d_block_counts[blockIdx.x] = block_count;
}

inline void launch_progressive_filter_rowpack16_passive(
    int grid,
    int block_threads,
    const Exp3RuntimeU8Subcolumns &subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *d_threshold_bytes,
    int threshold_stride,
    const uint32_t *d_active_plane_count,
    uint64_t *d_block_counts)
{
  progressive_filter_rowpack16_passive<<<grid, block_threads>>>(
      subcolumns,
      n,
      segment_rows,
      tiles_per_segment,
      d_threshold_bytes,
      threshold_stride,
      d_active_plane_count,
      d_block_counts);
}

inline void launch_progressive_filter_rowpack16_predicated(
    int grid,
    int block_threads,
    const Exp3RuntimeU8Subcolumns &subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *d_threshold_bytes,
    int threshold_stride,
    const uint32_t *d_active_plane_count,
    uint64_t *d_block_counts)
{
  progressive_filter_rowpack16_predicated<<<grid, block_threads>>>(
      subcolumns,
      n,
      segment_rows,
      tiles_per_segment,
      d_threshold_bytes,
      threshold_stride,
      d_active_plane_count,
      d_block_counts);
}

inline void launch_progressive_filter_rowpack16_byte_mask(
    int grid,
    int block_threads,
    const Exp3RuntimeU8Subcolumns &subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *d_threshold_bytes,
    int threshold_stride,
    const uint32_t *d_active_plane_count,
    uint64_t *d_block_counts)
{
  progressive_filter_rowpack16_byte_mask<<<grid, block_threads>>>(
      subcolumns,
      n,
      segment_rows,
      tiles_per_segment,
      d_threshold_bytes,
      threshold_stride,
      d_active_plane_count,
      d_block_counts);
}

inline void launch_progressive_filter_rowpack16_byte_mask_interleave2(
    int grid,
    int block_threads,
    const Exp3RuntimeU8Subcolumns &subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *d_threshold_bytes,
    int threshold_stride,
    const uint32_t *d_active_plane_count,
    uint64_t *d_block_counts)
{
  progressive_filter_rowpack16_byte_mask_interleave2<<<grid, block_threads>>>(
      subcolumns,
      n,
      segment_rows,
      tiles_per_segment,
      d_threshold_bytes,
      threshold_stride,
      d_active_plane_count,
      d_block_counts);
}

inline void launch_progressive_filter_rowpack16_byte_mask_specialized_k1(
    int grid,
    int block_threads,
    const Exp3RuntimeU8Subcolumns &subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *d_threshold_bytes,
    int threshold_stride,
    const uint32_t *d_active_plane_count,
    uint64_t *d_block_counts)
{
  progressive_filter_rowpack16_byte_mask_specialized<1><<<grid, block_threads>>>(
      subcolumns,
      n,
      segment_rows,
      tiles_per_segment,
      d_threshold_bytes,
      threshold_stride,
      d_active_plane_count,
      d_block_counts);
}

inline void launch_progressive_filter_rowpack16_byte_mask_specialized_k4(
    int grid,
    int block_threads,
    const Exp3RuntimeU8Subcolumns &subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *d_threshold_bytes,
    int threshold_stride,
    const uint32_t *d_active_plane_count,
    uint64_t *d_block_counts)
{
  progressive_filter_rowpack16_byte_mask_specialized<4><<<grid, block_threads>>>(
      subcolumns,
      n,
      segment_rows,
      tiles_per_segment,
      d_threshold_bytes,
      threshold_stride,
      d_active_plane_count,
      d_block_counts);
}

inline void launch_progressive_filter_rowpack16_byte_mask_specialized_k8(
    int grid,
    int block_threads,
    const Exp3RuntimeU8Subcolumns &subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *d_threshold_bytes,
    int threshold_stride,
    const uint32_t *d_active_plane_count,
    uint64_t *d_block_counts)
{
  progressive_filter_rowpack16_byte_mask_specialized<8><<<grid, block_threads>>>(
      subcolumns,
      n,
      segment_rows,
      tiles_per_segment,
      d_threshold_bytes,
      threshold_stride,
      d_active_plane_count,
      d_block_counts);
}

inline void launch_progressive_filter_rowpack16_simd_masks(
    int grid,
    int block_threads,
    const Exp3RuntimeU8Subcolumns &subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *d_threshold_bytes,
    int threshold_stride,
    const uint32_t *d_active_plane_count,
    uint64_t *d_block_counts)
{
  progressive_filter_rowpack16_simd_masks<<<grid, block_threads>>>(
      subcolumns,
      n,
      segment_rows,
      tiles_per_segment,
      d_threshold_bytes,
      threshold_stride,
      d_active_plane_count,
      d_block_counts);
}

inline void launch_progressive_filter_rowpack16_prefetch(
    int grid,
    int block_threads,
    const Exp3RuntimeU8Subcolumns &subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *d_threshold_bytes,
    int threshold_stride,
    const uint32_t *d_active_plane_count,
    uint64_t *d_block_counts)
{
  progressive_filter_rowpack16_prefetch<<<grid, block_threads>>>(
      subcolumns,
      n,
      segment_rows,
      tiles_per_segment,
      d_threshold_bytes,
      threshold_stride,
      d_active_plane_count,
      d_block_counts);
}

inline void launch_fixed_depth_filter_rowpack16_count(
    int grid,
    int block_threads,
    const Exp3RuntimeU8Subcolumns &subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *d_threshold_bytes,
    int threshold_stride,
    const uint32_t *d_active_plane_count,
    const uint8_t *d_segment_filter_mode,
    uint64_t *d_block_counts)
{
  fixed_depth_filter_rowpack16_count<<<grid, block_threads>>>(
      subcolumns,
      n,
      segment_rows,
      tiles_per_segment,
      d_threshold_bytes,
      threshold_stride,
      d_active_plane_count,
      d_segment_filter_mode,
      d_block_counts);
}
