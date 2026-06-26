#pragma once

#include "exp3_common.cuh"

#include <cfloat>
#include <cstdlib>

// Progressive MIN/MAX extrema kernel (runtime variant).
//
// For each block (tile of rows within a segment), computes:
//   min_prefix = minimum prefix value across rows in the tile
//   max_prefix = maximum prefix value across rows in the tile
//
// where prefix(k) = segment_base + sum(byte[p] * basis[p] for p in 0..plane_limit-1).
//
// Output: d_extrema_out[2*blockIdx.x]     = min_prefix for this block's tile
//         d_extrema_out[2*blockIdx.x + 1] = max_prefix for this block's tile
//
// Empty tiles write (INFINITY, -INFINITY) as neutral elements for host-side reduction.

__global__ void progressive_extrema_rowpack16_runtime(Exp3RuntimeU8Subcolumns subcolumns,
                                                      int max_planes,
                                                      int refinement_depth,
                                                      uint64_t n,
                                                      uint64_t segment_rows,
                                                      uint64_t tiles_per_segment,
                                                      const double *__restrict__ d_segment_base,
                                                      const double *__restrict__ d_subcolumn_basis,
                                                      double *__restrict__ d_extrema_out)
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
    {
      d_extrema_out[2 * blockIdx.x] = exp3_pos_inf();
      d_extrema_out[2 * blockIdx.x + 1] = exp3_neg_inf();
    }
    return;
  }

  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > segment_end)
    tile_end = segment_end;
  if (tile_end > n)
    tile_end = n;

  int plane_limit = refinement_depth + 1;
  if (plane_limit > max_planes)
    plane_limit = max_planes;

  double base = d_segment_base[segment_id];
  const double *basis = d_subcolumn_basis + segment_id * max_planes;

  double thread_min = exp3_pos_inf();
  double thread_max = exp3_neg_inf();

  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    double prefix = base;
    for (int plane = 0; plane < plane_limit; ++plane)
      prefix += static_cast<double>(subcolumns.ptrs[plane][row]) * basis[plane];

    if (prefix < thread_min)
      thread_min = prefix;
    if (prefix > thread_max)
      thread_max = prefix;
  }

  __shared__ double warp_vals[32];
  double block_min = exp3_block_reduce_min_double(thread_min, warp_vals);
  double block_max = exp3_block_reduce_max_double(thread_max, warp_vals);

  if (threadIdx.x == 0)
  {
    d_extrema_out[2 * blockIdx.x] = block_min;
    d_extrema_out[2 * blockIdx.x + 1] = block_max;
  }
}

inline void launch_progressive_extrema_rowpack16_runtime(int refinement_depth,
                                                         int grid,
                                                         int block_threads,
                                                         const Exp3RuntimeU8Subcolumns &subcolumns,
                                                         int max_planes,
                                                         uint64_t n,
                                                         uint64_t segment_rows,
                                                         uint64_t tiles_per_segment,
                                                         const double *d_segment_base,
                                                         const double *d_subcolumn_basis,
                                                         double *d_extrema_out)
{
  progressive_extrema_rowpack16_runtime<<<grid, block_threads>>>(
      subcolumns,
      max_planes,
      refinement_depth,
      n,
      segment_rows,
      tiles_per_segment,
      d_segment_base,
      d_subcolumn_basis,
      d_extrema_out);
}