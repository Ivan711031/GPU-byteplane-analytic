#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Self-contained helper definitions copied from exp3_common.cuh and
// exp4_kernels_filter.cuh to keep this header standalone.
// ---------------------------------------------------------------------------

constexpr int FA_ROWPACK16_WIDTH = 16;
constexpr int FA_MAX_RUNTIME_PLANES = 32;

template <typename T, int N>
struct FaPointerSet
{
  const T *ptrs[N];
};

using FaRuntimeU8Subcolumns = FaPointerSet<uint8_t, FA_MAX_RUNTIME_PLANES>;

__host__ __device__ __forceinline__ uint64_t fa_ceil_div_u64(uint64_t x, uint64_t y)
{
  return (x + y - 1ull) / y;
}

__device__ __forceinline__ unsigned int fa_byte_sum_u32(uint32_t x)
{
  return (x & 0xffu) +
         ((x >> 8) & 0xffu) +
         ((x >> 16) & 0xffu) +
         ((x >> 24) & 0xffu);
}

__device__ __forceinline__ uint8_t fa_uint4_extract_byte(uint4 pack, int lane)
{
  uint32_t w;
  if (lane < 4)
    w = pack.x;
  else if (lane < 8)
    w = pack.y;
  else if (lane < 12)
    w = pack.z;
  else
    w = pack.w;
  return static_cast<uint8_t>(w >> ((lane & 3) * 8));
}

__device__ __forceinline__ unsigned long long fa_warp_reduce_sum_ull(unsigned long long v)
{
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

__device__ __forceinline__ unsigned long long
fa_block_reduce_sum_ull(unsigned long long v, unsigned long long *__restrict__ warp_sums)
{
  v = fa_warp_reduce_sum_ull(v);

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
    block_sum = fa_warp_reduce_sum_ull(block_sum);
  }

  __syncthreads();
  return block_sum;
}

__device__ __forceinline__ double fa_warp_reduce_sum_double(double v)
{
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

// ---------------------------------------------------------------------------
// uint4 byte-wise > and < masks (copied from exp4_detail)
// ---------------------------------------------------------------------------
namespace fa_detail
{

__device__ __forceinline__ void uint4_gt_lt_masks(
    uint4 pack, uint8_t thresh,
    uint16_t &gt_mask, uint16_t &lt_mask)
{
  gt_mask = 0;
  lt_mask = 0;

  // pack.x -> byte indices 0-3
  uint32_t w = pack.x;
  if (static_cast<uint8_t>(w) > thresh) gt_mask |= static_cast<uint16_t>(1u << 0);
  if (static_cast<uint8_t>(w) < thresh) lt_mask |= static_cast<uint16_t>(1u << 0);
  if (static_cast<uint8_t>(w >> 8)  > thresh) gt_mask |= static_cast<uint16_t>(1u << 1);
  if (static_cast<uint8_t>(w >> 8)  < thresh) lt_mask |= static_cast<uint16_t>(1u << 1);
  if (static_cast<uint8_t>(w >> 16) > thresh) gt_mask |= static_cast<uint16_t>(1u << 2);
  if (static_cast<uint8_t>(w >> 16) < thresh) lt_mask |= static_cast<uint16_t>(1u << 2);
  if (static_cast<uint8_t>(w >> 24) > thresh) gt_mask |= static_cast<uint16_t>(1u << 3);
  if (static_cast<uint8_t>(w >> 24) < thresh) lt_mask |= static_cast<uint16_t>(1u << 3);

  // pack.y -> byte indices 4-7
  w = pack.y;
  if (static_cast<uint8_t>(w) > thresh) gt_mask |= static_cast<uint16_t>(1u << 4);
  if (static_cast<uint8_t>(w) < thresh) lt_mask |= static_cast<uint16_t>(1u << 4);
  if (static_cast<uint8_t>(w >> 8)  > thresh) gt_mask |= static_cast<uint16_t>(1u << 5);
  if (static_cast<uint8_t>(w >> 8)  < thresh) lt_mask |= static_cast<uint16_t>(1u << 5);
  if (static_cast<uint8_t>(w >> 16) > thresh) gt_mask |= static_cast<uint16_t>(1u << 6);
  if (static_cast<uint8_t>(w >> 16) < thresh) lt_mask |= static_cast<uint16_t>(1u << 6);
  if (static_cast<uint8_t>(w >> 24) > thresh) gt_mask |= static_cast<uint16_t>(1u << 7);
  if (static_cast<uint8_t>(w >> 24) < thresh) lt_mask |= static_cast<uint16_t>(1u << 7);

  // pack.z -> byte indices 8-11
  w = pack.z;
  if (static_cast<uint8_t>(w) > thresh) gt_mask |= static_cast<uint16_t>(1u << 8);
  if (static_cast<uint8_t>(w) < thresh) lt_mask |= static_cast<uint16_t>(1u << 8);
  if (static_cast<uint8_t>(w >> 8)  > thresh) gt_mask |= static_cast<uint16_t>(1u << 9);
  if (static_cast<uint8_t>(w >> 8)  < thresh) lt_mask |= static_cast<uint16_t>(1u << 9);
  if (static_cast<uint8_t>(w >> 16) > thresh) gt_mask |= static_cast<uint16_t>(1u << 10);
  if (static_cast<uint8_t>(w >> 16) < thresh) lt_mask |= static_cast<uint16_t>(1u << 10);
  if (static_cast<uint8_t>(w >> 24) > thresh) gt_mask |= static_cast<uint16_t>(1u << 11);
  if (static_cast<uint8_t>(w >> 24) < thresh) lt_mask |= static_cast<uint16_t>(1u << 11);

  // pack.w -> byte indices 12-15
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

} // namespace fa_detail

// ---------------------------------------------------------------------------
// Main kernel: progressive filter + SUM aggregation
// ---------------------------------------------------------------------------
// For each row:
//   1. Evaluate predicate x > threshold using byte-plane comparison (same as
//      progressive_filter_rowpack16_byte_mask from Exp4).
//   2. If qualified, reconstruct approximate value from k planes:
//        value = segment_base + sum(plane_basis[p] * byte_value[p])
//   3. Accumulate: count += 1, sum += value
// ---------------------------------------------------------------------------
__global__ void progressive_filter_sum_rowpack16_byte_mask(
    FaRuntimeU8Subcolumns subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *__restrict__ d_threshold_bytes,
    int threshold_stride,
    int max_plane_count,
    const uint32_t *__restrict__ d_active_plane_count,
    const double *__restrict__ d_segment_base,
    const double *__restrict__ d_subcolumn_basis,
    uint64_t *__restrict__ d_block_counts,
    double *__restrict__ d_block_sums)
{
  uint64_t tile_rows = static_cast<uint64_t>(blockDim.x) *
                       static_cast<uint64_t>(FA_ROWPACK16_WIDTH);
  uint64_t segment_id = static_cast<uint64_t>(blockIdx.x) / tiles_per_segment;
  uint64_t tile_in_segment = static_cast<uint64_t>(blockIdx.x) % tiles_per_segment;
  uint64_t segment_start = segment_id * segment_rows;
  uint64_t segment_end = segment_start + segment_rows;
  uint64_t tile_start = segment_start + tile_in_segment * tile_rows;

  if (tile_start >= n || tile_start >= segment_end)
  {
    if (threadIdx.x == 0)
    {
      d_block_counts[blockIdx.x] = 0ull;
      d_block_sums[blockIdx.x] = 0.0;
    }
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
  double local_sum = 0.0;

  const double *seg_basis = d_subcolumn_basis + segment_id * max_plane_count;
  double seg_base = d_segment_base[segment_id];

  // -------------------------------
  // Scalar prefix rows
  // -------------------------------
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
    {
      double val = seg_base;
      for (uint32_t round = 0; round < max_rounds; ++round)
        val += seg_basis[round] * static_cast<double>(subcolumns.ptrs[round][row]);
      ++local_count;
      local_sum += val;
    }
  }

  // -------------------------------
  // Rowpack16 main loop
  // -------------------------------
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
      fa_detail::uint4_gt_lt_masks(pack_data, thresh_byte, gt_mask, lt_mask);

      uint16_t newly_qualified = static_cast<uint16_t>(active_mask & gt_mask);
      uint16_t newly_resolved = static_cast<uint16_t>(active_mask & (gt_mask | lt_mask));
      qualified_mask = static_cast<uint16_t>(qualified_mask | newly_qualified);
      active_mask = static_cast<uint16_t>(active_mask & ~newly_resolved);
    }

    local_count += static_cast<uint32_t>(__popc(qualified_mask));

    // Reconstruct values for qualified rows using coalesced uint4 loads
    if (qualified_mask != 0)
    {
      double this_pack_sum = 0.0;
      // Accumulate seg_base once per qualified row
      for (int i = 0; i < 16; ++i)
        if (qualified_mask & static_cast<uint16_t>(1u << i))
          this_pack_sum += seg_base;
      // For each plane: one coalesced uint4 load, then extract bytes for
      // qualified lanes from the register (avoiding scalar global reloads).
      for (uint32_t round = 0; round < max_rounds; ++round)
      {
        const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[round]);
        uint4 pack_data = plane128[pack_idx];
        double basis = seg_basis[round];
        for (int i = 0; i < 16; ++i)
          if (qualified_mask & static_cast<uint16_t>(1u << i))
            this_pack_sum += basis * static_cast<double>(fa_uint4_extract_byte(pack_data, i));
      }
      local_sum += this_pack_sum;
    }
  }

  // -------------------------------
  // Scalar suffix rows
  // -------------------------------
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
    {
      double val = seg_base;
      for (uint32_t round = 0; round < max_rounds; ++round)
        val += seg_basis[round] * static_cast<double>(subcolumns.ptrs[round][row]);
      ++local_count;
      local_sum += val;
    }
  }

  // -------------------------------
  // Block reductions
  // -------------------------------
  // Count reduction (unsigned long long)
  __shared__ unsigned long long reduce_smem[32];
  unsigned long long block_count = fa_block_reduce_sum_ull(
      static_cast<unsigned long long>(local_count), reduce_smem);

  // Sum reduction (double)
  __shared__ double reduce_dsmem[32];
  double local_double_sum = local_sum;
  local_double_sum = fa_warp_reduce_sum_double(local_double_sum);

  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int warp_count = (blockDim.x + 31) >> 5;

  if (lane == 0)
    reduce_dsmem[warp_id] = local_double_sum;
  __syncthreads();

  double block_sum = 0.0;
  if (warp_id == 0)
  {
    block_sum = (lane < warp_count) ? reduce_dsmem[lane] : 0.0;
    block_sum = fa_warp_reduce_sum_double(block_sum);
  }

  if (threadIdx.x == 0)
  {
    d_block_counts[blockIdx.x] = block_count;
    d_block_sums[blockIdx.x] = block_sum;
  }
}

// ---------------------------------------------------------------------------
// Raw FP64 baseline kernel: scan raw float64 data, count + sum rows where
// value > threshold. Provides ground-truth throughput for full-precision scan.
// ---------------------------------------------------------------------------
__global__ void raw_fp64_filter_sum(
    const double *__restrict__ d_data,
    uint64_t n,
    double threshold,
    uint64_t *__restrict__ d_block_counts,
    double *__restrict__ d_block_sums)
{
  uint64_t local_count = 0ull;
  double local_sum = 0.0;

  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                     static_cast<uint64_t>(threadIdx.x);
       i < n;
       i += static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x))
  {
    double val = d_data[i];
    if (val > threshold)
    {
      ++local_count;
      local_sum += val;
    }
  }

  __shared__ unsigned long long reduce_smem[32];
  unsigned long long block_count = fa_block_reduce_sum_ull(local_count, reduce_smem);

  __shared__ double reduce_dsmem[32];
  double warp_sum = fa_warp_reduce_sum_double(local_sum);

  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int warp_count = (blockDim.x + 31) >> 5;

  if (lane == 0)
    reduce_dsmem[warp_id] = warp_sum;
  __syncthreads();

  double block_sum = 0.0;
  if (warp_id == 0)
  {
    block_sum = (lane < warp_count) ? reduce_dsmem[lane] : 0.0;
    block_sum = fa_warp_reduce_sum_double(block_sum);
  }

  if (threadIdx.x == 0)
  {
    d_block_counts[blockIdx.x] = block_count;
    d_block_sums[blockIdx.x] = block_sum;
  }
}

// ---------------------------------------------------------------------------
// Deferred gather kernel 1: progressive filter COUNT only
// Outputs per-block qualified count (same filter logic as fused kernel, but
// no value reconstruction or sum accumulation).
// ---------------------------------------------------------------------------
__global__ void progressive_filter_count_rowpack16_byte_mask(
    FaRuntimeU8Subcolumns subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *__restrict__ d_threshold_bytes,
    int threshold_stride,
    int max_plane_count,
    const uint32_t *__restrict__ d_active_plane_count,
    uint64_t *__restrict__ d_block_counts)
{
  uint64_t tile_rows = static_cast<uint64_t>(blockDim.x) *
                        static_cast<uint64_t>(FA_ROWPACK16_WIDTH);
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
  if (tile_end > segment_end) tile_end = segment_end;
  if (tile_end > n) tile_end = n;

  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0) ++aligned_start;
  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull;
  uint64_t pack_count = (full_pack_end - aligned_start) / 16ull;

  uint32_t max_rounds = d_active_plane_count[segment_id];
  uint64_t thresh_offset = segment_id * static_cast<uint64_t>(threshold_stride);

  uint32_t local_count = 0u;

  // Scalar prefix rows
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
      if (row_byte > thresh_byte) { qualified = true; active = false; }
      else if (row_byte < thresh_byte) { active = false; }
    }
    if (qualified) ++local_count;
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

      uint16_t gt_mask = 0x0000u;
      uint16_t lt_mask = 0x0000u;
      fa_detail::uint4_gt_lt_masks(pack_data, thresh_byte, gt_mask, lt_mask);

      uint16_t newly_qualified = static_cast<uint16_t>(active_mask & gt_mask);
      uint16_t newly_resolved = static_cast<uint16_t>(active_mask & (gt_mask | lt_mask));
      qualified_mask = static_cast<uint16_t>(qualified_mask | newly_qualified);
      active_mask = static_cast<uint16_t>(active_mask & ~newly_resolved);
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
      if (row_byte > thresh_byte) { qualified = true; active = false; }
      else if (row_byte < thresh_byte) { active = false; }
    }
    if (qualified) ++local_count;
  }

  // Block reduction for count
  __shared__ unsigned long long reduce_smem_count[32];
  unsigned long long block_count = fa_block_reduce_sum_ull(
      static_cast<unsigned long long>(local_count), reduce_smem_count);

  if (threadIdx.x == 0)
    d_block_counts[blockIdx.x] = block_count;
}

// ---------------------------------------------------------------------------
// Deferred gather kernel 2: progressive filter SCATTER
// Same filter logic, writes qualified row indices to d_qualified_indices
// using d_block_offsets for positioning. Must be called after exclusive
// prefix sum of d_block_counts to compute d_block_offsets.
// ---------------------------------------------------------------------------
__global__ void progressive_filter_scatter_rowpack16_byte_mask(
    FaRuntimeU8Subcolumns subcolumns,
    uint64_t n,
    uint64_t segment_rows,
    uint64_t tiles_per_segment,
    const uint8_t *__restrict__ d_threshold_bytes,
    int threshold_stride,
    int max_plane_count,
    const uint32_t *__restrict__ d_active_plane_count,
    const uint64_t *__restrict__ d_block_offsets,
    uint64_t *__restrict__ d_qualified_indices)
{
  uint64_t tile_rows = static_cast<uint64_t>(blockDim.x) *
                        static_cast<uint64_t>(FA_ROWPACK16_WIDTH);
  uint64_t segment_id = static_cast<uint64_t>(blockIdx.x) / tiles_per_segment;
  uint64_t tile_in_segment = static_cast<uint64_t>(blockIdx.x) % tiles_per_segment;
  uint64_t segment_start = segment_id * segment_rows;
  uint64_t segment_end = segment_start + segment_rows;
  uint64_t tile_start = segment_start + tile_in_segment * tile_rows;

  if (tile_start >= n || tile_start >= segment_end) return;

  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > segment_end) tile_end = segment_end;
  if (tile_end > n) tile_end = n;

  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0) ++aligned_start;
  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull;
  uint64_t pack_count = (full_pack_end - aligned_start) / 16ull;

  uint32_t max_rounds = d_active_plane_count[segment_id];
  uint64_t thresh_offset = segment_id * static_cast<uint64_t>(threshold_stride);

  constexpr int kMaxPerThread = 256;
  uint64_t thread_indices[kMaxPerThread];
  int thread_count = 0;

  // Scalar prefix rows
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
      if (row_byte > thresh_byte) { qualified = true; active = false; }
      else if (row_byte < thresh_byte) { active = false; }
    }
    if (qualified && thread_count < kMaxPerThread)
      thread_indices[thread_count++] = row;
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

      uint16_t gt_mask = 0x0000u;
      uint16_t lt_mask = 0x0000u;
      fa_detail::uint4_gt_lt_masks(pack_data, thresh_byte, gt_mask, lt_mask);

      uint16_t newly_qualified = static_cast<uint16_t>(active_mask & gt_mask);
      uint16_t newly_resolved = static_cast<uint16_t>(active_mask & (gt_mask | lt_mask));
      qualified_mask = static_cast<uint16_t>(qualified_mask | newly_qualified);
      active_mask = static_cast<uint16_t>(active_mask & ~newly_resolved);
    }

    uint64_t pack_row_base = pack_idx * 16ull;
    for (int i = 0; i < 16; ++i)
    {
      uint16_t bit = static_cast<uint16_t>(1u << i);
      if ((qualified_mask & bit) && thread_count < kMaxPerThread)
        thread_indices[thread_count++] = pack_row_base + static_cast<uint64_t>(i);
    }
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
      if (row_byte > thresh_byte) { qualified = true; active = false; }
      else if (row_byte < thresh_byte) { active = false; }
    }
    if (qualified && thread_count < kMaxPerThread)
      thread_indices[thread_count++] = row;
  }

  // Intra-block prefix sum to determine write positions
  __shared__ unsigned int smem_counts[1024];
  __shared__ uint64_t smem_base;

  smem_counts[threadIdx.x] = static_cast<unsigned int>(thread_count);
  __syncthreads();

  unsigned int prefix_sum = 0;
  for (int i = 0; i < threadIdx.x; ++i)
    prefix_sum += smem_counts[i];

  if (threadIdx.x == 0)
    smem_base = d_block_offsets[blockIdx.x];
  __syncthreads();

  uint64_t write_base = smem_base + static_cast<uint64_t>(prefix_sum);
  for (int i = 0; i < thread_count; ++i)
    d_qualified_indices[write_base + static_cast<uint64_t>(i)] = thread_indices[i];
}

// ---------------------------------------------------------------------------
// Deferred gather kernel 3: gather raw FP64 values at qualified indices
// and accumulate count + sum.
// ---------------------------------------------------------------------------
__global__ void raw_fp64_gather_sum(
    const double *__restrict__ d_raw_data,
    const uint64_t *__restrict__ d_qualified_indices,
    uint64_t qualified_count,
    uint64_t *__restrict__ d_gather_count,
    double *__restrict__ d_gather_sum)
{
  uint64_t local_count = 0ull;
  double local_sum = 0.0;

  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
                     static_cast<uint64_t>(threadIdx.x);
       i < qualified_count;
       i += static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x))
  {
    uint64_t row_idx = d_qualified_indices[i];
    double val = d_raw_data[row_idx];
    ++local_count;
    local_sum += val;
  }

  __shared__ unsigned long long reduce_smem_g[32];
  unsigned long long block_count = fa_block_reduce_sum_ull(local_count, reduce_smem_g);

  __shared__ double reduce_dsmem_g[32];
  double warp_sum = fa_warp_reduce_sum_double(local_sum);

  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int warp_count = (blockDim.x + 31) >> 5;

  if (lane == 0) reduce_dsmem_g[warp_id] = warp_sum;
  __syncthreads();

  double block_sum = 0.0;
  if (warp_id == 0)
  {
    block_sum = (lane < warp_count) ? reduce_dsmem_g[lane] : 0.0;
    block_sum = fa_warp_reduce_sum_double(block_sum);
  }

  if (threadIdx.x == 0)
  {
    d_gather_count[blockIdx.x] = block_count;
    d_gather_sum[blockIdx.x] = block_sum;
  }
}
