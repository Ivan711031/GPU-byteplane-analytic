#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Shared constants and helpers
// ---------------------------------------------------------------------------
constexpr int N2_ROWPACK16_WIDTH = 16;
constexpr int N2_MAX_RUNTIME_PLANES = 32;

template <typename T, int N>
struct N2PointerSet
{
  const T *ptrs[N];
};

using N2RuntimeU8Subcolumns = N2PointerSet<uint8_t, N2_MAX_RUNTIME_PLANES>;

__device__ __forceinline__ uint64_t n2_ceil_div_u64(uint64_t x, uint64_t y)
{
  return (x + y - 1ull) / y;
}

__device__ __forceinline__ unsigned long long n2_warp_reduce_sum_ull(unsigned long long v)
{
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

__device__ __forceinline__ double n2_warp_reduce_sum_double(double v)
{
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

__device__ __forceinline__ unsigned long long
n2_block_reduce_sum_ull(unsigned long long v, unsigned long long *__restrict__ warp_sums)
{
  v = n2_warp_reduce_sum_ull(v);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int warp_count = blockDim.x >> 5;
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  unsigned long long block_sum = 0ull;
  if (warp == 0)
  {
    block_sum = (lane < warp_count) ? warp_sums[lane] : 0ull;
    block_sum = n2_warp_reduce_sum_ull(block_sum);
  }
  __syncthreads();
  return block_sum;
}

__device__ __forceinline__ unsigned int n2_warp_reduce_sum_uint(unsigned int v)
{
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

// ---------------------------------------------------------------------------
// Branchless uint4 byte extraction. Array index → uniform path, no warp divergence.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t n2_uint4_extract_byte(uint4 pack, int lane)
{
  uint32_t words[4] = {pack.x, pack.y, pack.z, pack.w};
  return static_cast<uint8_t>(words[lane >> 2] >> ((lane & 3) * 8));
}

__device__ __forceinline__ uint32_t n2_sum4bytes_u32(uint32_t w)
{
  return (w & 0xFFu) + ((w >> 8) & 0xFFu) + ((w >> 16) & 0xFFu) + ((w >> 24) & 0xFFu);
}

__device__ __forceinline__ uint32_t n2_uint4_sum_bytes(uint4 pack)
{
  return n2_sum4bytes_u32(pack.x) + n2_sum4bytes_u32(pack.y) +
      n2_sum4bytes_u32(pack.z) + n2_sum4bytes_u32(pack.w);
}

// Full-qualified fast path + sparse predicated-add fallback
__device__ __forceinline__ double n2_uint4_masked_scaled_sum(uint4 pack, uint16_t mask, double scale)
{
  if (mask == 0xFFFFu)
    return scale * static_cast<double>(n2_uint4_sum_bytes(pack));
  double acc = 0.0;
  for (int i = 0; i < 16; ++i)
    if (mask & static_cast<uint16_t>(1u << i))
      acc += scale * static_cast<double>(n2_uint4_extract_byte(pack, i));
  return acc;
}

// ---------------------------------------------------------------------------
// uint4 byte-wise > and < masks
// Uses simple unsigned byte comparison; CUDA predicates 'if' as branchless setp.
// ---------------------------------------------------------------------------
namespace n2_detail
{

__device__ __forceinline__ void uint4_gt_lt_masks(
    uint4 pack, uint8_t thresh,
    uint16_t &gt_mask, uint16_t &lt_mask)
{
  gt_mask = 0; lt_mask = 0;
  uint32_t w;
  #define N2_CMP(b, idx) do {                               \
    if (static_cast<uint8_t>(b) > thresh) gt_mask |= (1u << (idx));  \
    if (static_cast<uint8_t>(b) < thresh) lt_mask |= (1u << (idx));  \
  } while(0)
  w = pack.x;
  N2_CMP(w, 0); N2_CMP(w>>8, 1); N2_CMP(w>>16, 2); N2_CMP(w>>24, 3);
  w = pack.y;
  N2_CMP(w, 4); N2_CMP(w>>8, 5); N2_CMP(w>>16, 6); N2_CMP(w>>24, 7);
  w = pack.z;
  N2_CMP(w, 8); N2_CMP(w>>8, 9); N2_CMP(w>>16, 10); N2_CMP(w>>24, 11);
  w = pack.w;
  N2_CMP(w, 12); N2_CMP(w>>8, 13); N2_CMP(w>>16, 14); N2_CMP(w>>24, 15);
  #undef N2_CMP
}

} // namespace n2_detail

// ---------------------------------------------------------------------------
// Branchless byte-level majority vote (8-bit per lane)
// For each byte: majority of (va, vb, vc). Tie-break -> max value.
// Uses XOR-based equality detection + bitwise selection; no control-flow branches.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t n2_byte_majority3(uint8_t va, uint8_t vb, uint8_t vc)
{
  // Equality flags (0x01 if equal, 0x00 if different)
  uint8_t x_ab = (va ^ vb);
  uint8_t x_ac = (va ^ vc);
  uint8_t x_bc = (vb ^ vc);
  // Convert to full-byte masks (0xFF or 0x00) using signed negation
  // -(int8_t)(x==0) = 0xFF when x==0, 0x00 otherwise
  uint8_t eq_ab = -(int8_t)(x_ab == 0);
  uint8_t eq_ac = -(int8_t)(x_ac == 0);
  uint8_t eq_bc = -(int8_t)(x_bc == 0);
  // sel_a: a matches b or c
  uint8_t sel_a = eq_ab | eq_ac;
  // sel_b: a is odd-one-out but b matches c
  uint8_t sel_b = eq_bc & ~eq_ab;
  // sel_tie: all three different
  uint8_t sel_tie = ~(sel_a | sel_b);
  // Max for tie-break (predicated select, branchless in CUDA)
  uint8_t mx = va;
  mx = (vb > mx) ? vb : mx;
  mx = (vc > mx) ? vc : mx;
  return (va & sel_a) | (vb & sel_b) | (mx & sel_tie);
}

__device__ __forceinline__ uint32_t n2_u32_byte_majority_vote_r3(uint32_t wa, uint32_t wb, uint32_t wc)
{
  uint32_t result = 0;
  #pragma unroll
  for (int i = 0; i < 4; ++i)
  {
    uint32_t shift = static_cast<uint32_t>(i) * 8u;
    uint8_t vote = n2_byte_majority3(
        static_cast<uint8_t>(wa >> shift),
        static_cast<uint8_t>(wb >> shift),
        static_cast<uint8_t>(wc >> shift));
    result |= static_cast<uint32_t>(vote) << shift;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Vectorized byte-level majority vote across 3 replicas (16 bytes of uint4)
// Calls branchless n2_byte_majority3 per byte. 0 branches, 0 warp divergence.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint4 n2_uint4_byte_majority_vote_r3(uint4 a, uint4 b, uint4 c)
{
  return make_uint4(
      n2_u32_byte_majority_vote_r3(a.x, b.x, c.x),
      n2_u32_byte_majority_vote_r3(a.y, b.y, c.y),
      n2_u32_byte_majority_vote_r3(a.z, b.z, c.z),
      n2_u32_byte_majority_vote_r3(a.w, b.w, c.w));
}

// ---------------------------------------------------------------------------
// Scalar row majority vote (3 replicas → 1 byte). Branchless.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t n2_scalar_vote_r3(const uint8_t *r0, const uint8_t *r1, const uint8_t *r2, uint64_t row)
{
  return n2_byte_majority3(r0[row], r1[row], r2[row]);
}

// ---------------------------------------------------------------------------
// P0: baseline byte-plane progressive filter + SUM (k-flexible)
// Identical logic to progressive_filter_sum_rowpack16_byte_mask but
// self-contained. k = d_active_plane_count[segment_id] (read set size).
// ---------------------------------------------------------------------------
__global__ void nmr_v2_p0_baseline_sum(
    N2RuntimeU8Subcolumns subcolumns,
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
                       static_cast<uint64_t>(N2_ROWPACK16_WIDTH);
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
  double local_sum = 0.0;

  const double *seg_basis = d_subcolumn_basis + segment_id * max_plane_count;
  double seg_base = d_segment_base[segment_id];

  // Scalar prefix
  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false, active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)      { qualified = true; active = false; }
      else if (row_byte < thresh_byte) { active = false; }
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
      uint16_t gt_mask = 0x0000u, lt_mask = 0x0000u;
      n2_detail::uint4_gt_lt_masks(pack_data, thresh_byte, gt_mask, lt_mask);
      uint16_t newly_qualified = static_cast<uint16_t>(active_mask & gt_mask);
      uint16_t newly_resolved = static_cast<uint16_t>(active_mask & (gt_mask | lt_mask));
      qualified_mask = static_cast<uint16_t>(qualified_mask | newly_qualified);
      active_mask = static_cast<uint16_t>(active_mask & ~newly_resolved);
    }

    local_count += static_cast<uint32_t>(__popc(qualified_mask));

    if (qualified_mask != 0)
    {
      double this_pack_sum = 0.0;
      for (int i = 0; i < 16; ++i)
        if (qualified_mask & static_cast<uint16_t>(1u << i))
          this_pack_sum += seg_base;
      for (uint32_t round = 0; round < max_rounds; ++round)
      {
        const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[round]);
        uint4 pack_data = plane128[pack_idx];
        double basis = seg_basis[round];
        for (int i = 0; i < 16; ++i)
          if (qualified_mask & static_cast<uint16_t>(1u << i))
            this_pack_sum += basis * static_cast<double>(n2_uint4_extract_byte(pack_data, i));
      }
      local_sum += this_pack_sum;
    }
  }

  // Scalar suffix
  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false, active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)      { qualified = true; active = false; }
      else if (row_byte < thresh_byte) { active = false; }
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

  // Block reductions
  __shared__ unsigned long long red_smem[32];
  unsigned long long block_count = n2_block_reduce_sum_ull(
      static_cast<unsigned long long>(local_count), red_smem);

  __shared__ double red_dsmem[32];
  double local_double_sum = n2_warp_reduce_sum_double(local_sum);
  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int warp_count = (blockDim.x + 31) >> 5;
  if (lane == 0) red_dsmem[warp_id] = local_double_sum;
  __syncthreads();
  double block_sum = 0.0;
  if (warp_id == 0)
  {
    block_sum = (lane < warp_count) ? red_dsmem[lane] : 0.0;
    block_sum = n2_warp_reduce_sum_double(block_sum);
  }

  if (threadIdx.x == 0)
  {
    d_block_counts[blockIdx.x] = block_count;
    d_block_sums[blockIdx.x] = block_sum;
  }
}

// ---------------------------------------------------------------------------
// P2: fused vote inline for all read planes (branchless + cached)
//
// d_r_per_plane[round] = total replicas per plane (1, 2, or 3).
//   r=1: read r0 only, no vote, no detection
//   r=2: read r0, r1; compare per-byte; disagreement → degrade to prefix-only partial answer
//   r=3: 3-way byte majority vote (full recovery under single-fault assumption)
//
// replicas_r0/r1/r2: each is a pointer set keyed by plane index.
// Only planes 0..protected_planes-1 have valid replica pointers.
// max_rounds (= k, the read set size) ≤ protected_planes ≤ 8.
// ---------------------------------------------------------------------------
__global__ void nmr_v2_p2_fused_vote_inline(
    N2RuntimeU8Subcolumns replicas_r0,
    N2RuntimeU8Subcolumns replicas_r1,
    N2RuntimeU8Subcolumns replicas_r2,
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
    double *__restrict__ d_block_sums,
    const uint32_t *__restrict__ d_r_per_plane)
{
  uint64_t tile_rows = static_cast<uint64_t>(blockDim.x) *
                       static_cast<uint64_t>(N2_ROWPACK16_WIDTH);
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
  double local_sum = 0.0;

  const double *seg_basis = d_subcolumn_basis + segment_id * max_plane_count;
  double seg_base = d_segment_base[segment_id];

  // Scalar prefix — per-plane r-aware with partial-sum on r=2 detect
  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    uint32_t active_sentinel = 0xFFFFFFFFu;
    int r2_detect_round = static_cast<int>(max_rounds);
    for (uint32_t round = 0; round < max_rounds; ++round)
    {
      uint32_t rp = d_r_per_plane[round];
      uint8_t voted;
      if (rp == 1) {
        voted = replicas_r0.ptrs[round][row];
      } else if (rp == 2) {
        uint8_t v0 = replicas_r0.ptrs[round][row];
        uint8_t v1 = replicas_r1.ptrs[round][row];
        if (v0 != v1) { r2_detect_round = static_cast<int>(round); break; }
        voted = v0;
      } else {
        voted = n2_scalar_vote_r3(replicas_r0.ptrs[round], replicas_r1.ptrs[round],
                                  replicas_r2.ptrs[round], row);
      }
      uint8_t tb = d_threshold_bytes[thresh_offset + round];
      int is_gt = (voted > tb);
      int is_lt = (voted < tb);
      qualified |= (is_gt & active_sentinel);
      active_sentinel &= ~(is_gt | is_lt);
    }
    if (qualified || r2_detect_round < static_cast<int>(max_rounds))
    {
      int eff_rounds = r2_detect_round < static_cast<int>(max_rounds) ? r2_detect_round : static_cast<int>(max_rounds);
      double val = seg_base;
      for (int round = 0; round < eff_rounds; ++round)
      {
        uint32_t rp = d_r_per_plane[round];
        uint8_t voted;
        if (rp == 1) {
          voted = replicas_r0.ptrs[round][row];
        } else if (rp == 2) {
          voted = replicas_r0.ptrs[round][row];
        } else {
          voted = n2_scalar_vote_r3(replicas_r0.ptrs[round], replicas_r1.ptrs[round],
                                    replicas_r2.ptrs[round], row);
        }
        val += seg_basis[round] * static_cast<double>(voted);
      }
      ++local_count;
      local_sum += val;
    }
  }

  // Rowpack16 main loop — split-phase: vote all rounds into cache → filter →
  // reconstruct. No early exit on global reads ensures complete cache.
  // Cache (vc[8] = 128B) lives in local memory but avoids 3× global re-reads.
  constexpr int kMaxCache = 8;
  for (uint64_t pack = static_cast<uint64_t>(threadIdx.x);
       pack < pack_count;
       pack += static_cast<uint64_t>(blockDim.x))
  {
    uint64_t pack_idx = first_pack + pack;

    // Phase 1: vote all rounds → cache (sequential coalesced reads)
    // r=2: also compute per-byte disagree_mask for detection
    uint4 vc[kMaxCache];
    uint16_t r2_disagree[kMaxCache] = {0};
#pragma unroll
    for (uint32_t round = 0; round < kMaxCache; ++round)
      if (round < max_rounds) r2_disagree[round] = 0;
    for (uint32_t round = 0; round < max_rounds; ++round)
    {
      uint32_t rp = d_r_per_plane[round];
      if (rp == 1) {
        const uint4 *p0 = reinterpret_cast<const uint4 *>(replicas_r0.ptrs[round]);
        vc[round] = p0[pack_idx];
      } else if (rp == 2) {
        const uint4 *p0 = reinterpret_cast<const uint4 *>(replicas_r0.ptrs[round]);
        const uint4 *p1 = reinterpret_cast<const uint4 *>(replicas_r1.ptrs[round]);
        uint4 a = p0[pack_idx], b = p1[pack_idx];
        vc[round] = a;
        // Compute per-byte XOR to find disagreement lanes
        uint32_t x_xor = a.x ^ b.x;
        uint32_t y_xor = a.y ^ b.y;
        uint32_t z_xor = a.z ^ b.z;
        uint32_t w_xor = a.w ^ b.w;
        uint16_t dm = 0;
        if (x_xor & 0x000000FFu) dm |= 1;    if (x_xor & 0x0000FF00u) dm |= 2;
        if (x_xor & 0x00FF0000u) dm |= 4;    if (x_xor & 0xFF000000u) dm |= 8;
        if (y_xor & 0x000000FFu) dm |= 16;   if (y_xor & 0x0000FF00u) dm |= 32;
        if (y_xor & 0x00FF0000u) dm |= 64;   if (y_xor & 0xFF000000u) dm |= 128;
        if (z_xor & 0x000000FFu) dm |= 256;  if (z_xor & 0x0000FF00u) dm |= 512;
        if (z_xor & 0x00FF0000u) dm |= 1024; if (z_xor & 0xFF000000u) dm |= 2048;
        if (w_xor & 0x000000FFu) dm |= 4096; if (w_xor & 0x0000FF00u) dm |= 8192;
        if (w_xor & 0x00FF0000u) dm |= 16384; if (w_xor & 0xFF000000u) dm |= 32768;
        r2_disagree[round] = dm;
      } else {
        const uint4 *p0 = reinterpret_cast<const uint4 *>(replicas_r0.ptrs[round]);
        const uint4 *p1 = reinterpret_cast<const uint4 *>(replicas_r1.ptrs[round]);
        const uint4 *p2 = reinterpret_cast<const uint4 *>(replicas_r2.ptrs[round]);
        vc[round] = n2_uint4_byte_majority_vote_r3(p0[pack_idx], p1[pack_idx], p2[pack_idx]);
      }
    }

    // Phase 2: apply r=2 disagreement masks (detected lanes → partial answer)
    // For lanes where r2 disagrees at round R: answer = seg_base + planes 0..R-1
    uint16_t r2_detected_mask = 0;
    for (uint32_t round = 0; round < max_rounds; ++round) {
      if (d_r_per_plane[round] == 2 && r2_disagree[round] != 0) {
        uint16_t new_detected = r2_disagree[round] & ~r2_detected_mask;
        if (new_detected) {
          r2_detected_mask |= new_detected;
          int popc_d = __popc(new_detected);
          double sp = seg_base * static_cast<double>(popc_d);
          for (uint32_t prev = 0; prev < round; ++prev)
            sp += n2_uint4_masked_scaled_sum(vc[prev], new_detected, seg_basis[prev]);
          local_count += static_cast<uint32_t>(popc_d);
          local_sum += sp;
        }
      }
    }

    // Phase 3: filter from cache for non-r2-detected lanes (early exit from cache)
    uint16_t active_mask = 0xFFFFu & ~r2_detected_mask;
    uint16_t qualified_mask = 0x0000u;
    for (uint32_t round = 0; round < max_rounds && active_mask != 0; ++round)
    {
      uint8_t thr = d_threshold_bytes[thresh_offset + round];
      uint16_t gt = 0, lt = 0;
      n2_detail::uint4_gt_lt_masks(vc[round], thr, gt, lt);
      qualified_mask |= static_cast<uint16_t>(active_mask & gt);
      active_mask    &= static_cast<uint16_t>(~(active_mask & (gt | lt)));
    }

    uint32_t qualified_count = static_cast<uint32_t>(__popc(qualified_mask));
    local_count += qualified_count;

    // Phase 4: reconstruct with full-qualified fast path
    if (qualified_mask != 0)
    {
      double sp = seg_base * static_cast<double>(qualified_count);
      for (uint32_t round = 0; round < max_rounds; ++round)
        sp += n2_uint4_masked_scaled_sum(vc[round], qualified_mask, seg_basis[round]);
      local_sum += sp;
    }
  }

  // Scalar suffix — per-plane r-aware with partial-sum on r=2 detect
  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    uint32_t active_sentinel = 0xFFFFFFFFu;
    int r2_detect_round = static_cast<int>(max_rounds);
    for (uint32_t round = 0; round < max_rounds; ++round)
    {
      uint32_t rp = d_r_per_plane[round];
      uint8_t voted;
      if (rp == 1) {
        voted = replicas_r0.ptrs[round][row];
      } else if (rp == 2) {
        uint8_t v0 = replicas_r0.ptrs[round][row];
        uint8_t v1 = replicas_r1.ptrs[round][row];
        if (v0 != v1) { r2_detect_round = static_cast<int>(round); break; }
        voted = v0;
      } else {
        voted = n2_scalar_vote_r3(replicas_r0.ptrs[round], replicas_r1.ptrs[round],
                                  replicas_r2.ptrs[round], row);
      }
      uint8_t tb = d_threshold_bytes[thresh_offset + round];
      int is_gt = (voted > tb);
      int is_lt = (voted < tb);
      qualified |= (is_gt & active_sentinel);
      active_sentinel &= ~(is_gt | is_lt);
    }
    if (qualified || r2_detect_round < static_cast<int>(max_rounds))
    {
      int eff_rounds = r2_detect_round < static_cast<int>(max_rounds) ? r2_detect_round : static_cast<int>(max_rounds);
      double val = seg_base;
      for (int round = 0; round < eff_rounds; ++round)
      {
        uint32_t rp = d_r_per_plane[round];
        uint8_t voted;
        if (rp == 1) {
          voted = replicas_r0.ptrs[round][row];
        } else if (rp == 2) {
          voted = replicas_r0.ptrs[round][row];
        } else {
          voted = n2_scalar_vote_r3(replicas_r0.ptrs[round], replicas_r1.ptrs[round],
                                    replicas_r2.ptrs[round], row);
        }
        val += seg_basis[round] * static_cast<double>(voted);
      }
      ++local_count;
      local_sum += val;
    }
  }

  // Block reductions
  __shared__ unsigned long long red_smem[32];
  unsigned long long block_count = n2_block_reduce_sum_ull(
      static_cast<unsigned long long>(local_count), red_smem);

  __shared__ double red_dsmem[32];
  double local_double_sum = n2_warp_reduce_sum_double(local_sum);
  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int warp_count = (blockDim.x + 31) >> 5;
  if (lane == 0) red_dsmem[warp_id] = local_double_sum;
  __syncthreads();
  double block_sum = 0.0;
  if (warp_id == 0)
  {
    block_sum = (lane < warp_count) ? red_dsmem[lane] : 0.0;
    block_sum = n2_warp_reduce_sum_double(block_sum);
  }

  if (threadIdx.x == 0)
  {
    d_block_counts[blockIdx.x] = block_count;
    d_block_sums[blockIdx.x] = block_sum;
  }
}

// ---------------------------------------------------------------------------
// P1: fused digest inline — P0 baseline + per-block SUM32 plane digest
//
// Same filter+sum as P0, plus computes per-block partial SUM32 digests for
// each read plane. d_block_plane_digests[p * grid + blockIdx.x] stores the
// partial sum of plane p for this block. Host-side reduction produces final.
// ---------------------------------------------------------------------------
__global__ void nmr_v2_p1_fused_digest_inline(
    N2RuntimeU8Subcolumns subcolumns,
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
    double *__restrict__ d_block_sums,
    uint32_t *__restrict__ d_block_plane_digests,
    int digest_stride)
{
  uint64_t tile_rows = static_cast<uint64_t>(blockDim.x) *
                       static_cast<uint64_t>(N2_ROWPACK16_WIDTH);
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
  double local_sum = 0.0;

  const double *seg_basis = d_subcolumn_basis + segment_id * max_plane_count;
  double seg_base = d_segment_base[segment_id];

  // Per-plane digest accumulators (per-thread, per-plane)
  uint32_t plane_digests[N2_MAX_RUNTIME_PLANES] = {0};

  // Scalar prefix
  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false, active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      plane_digests[round] += static_cast<uint32_t>(row_byte);
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)      { qualified = true; active = false; }
      else if (row_byte < thresh_byte) { active = false; }
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
      // Accumulate digest: sum all 16 bytes in uint4 (4x byte_sum_u32)
      plane_digests[round] += ((pack_data.x & 0xffu) + ((pack_data.x >> 8) & 0xffu) +
                               ((pack_data.x >> 16) & 0xffu) + ((pack_data.x >> 24) & 0xffu));
      plane_digests[round] += ((pack_data.y & 0xffu) + ((pack_data.y >> 8) & 0xffu) +
                               ((pack_data.y >> 16) & 0xffu) + ((pack_data.y >> 24) & 0xffu));
      plane_digests[round] += ((pack_data.z & 0xffu) + ((pack_data.z >> 8) & 0xffu) +
                               ((pack_data.z >> 16) & 0xffu) + ((pack_data.z >> 24) & 0xffu));
      plane_digests[round] += ((pack_data.w & 0xffu) + ((pack_data.w >> 8) & 0xffu) +
                               ((pack_data.w >> 16) & 0xffu) + ((pack_data.w >> 24) & 0xffu));
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      uint16_t gt_mask = 0x0000u, lt_mask = 0x0000u;
      n2_detail::uint4_gt_lt_masks(pack_data, thresh_byte, gt_mask, lt_mask);
      uint16_t newly_qualified = static_cast<uint16_t>(active_mask & gt_mask);
      uint16_t newly_resolved = static_cast<uint16_t>(active_mask & (gt_mask | lt_mask));
      qualified_mask = static_cast<uint16_t>(qualified_mask | newly_qualified);
      active_mask = static_cast<uint16_t>(active_mask & ~newly_resolved);
    }

    local_count += static_cast<uint32_t>(__popc(qualified_mask));

    if (qualified_mask != 0)
    {
      double this_pack_sum = 0.0;
      for (int i = 0; i < 16; ++i)
        if (qualified_mask & static_cast<uint16_t>(1u << i))
          this_pack_sum += seg_base;
      for (uint32_t round = 0; round < max_rounds; ++round)
      {
        const uint4 *plane128 = reinterpret_cast<const uint4 *>(subcolumns.ptrs[round]);
        uint4 pack_data = plane128[pack_idx];
        double basis = seg_basis[round];
        for (int i = 0; i < 16; ++i)
          if (qualified_mask & static_cast<uint16_t>(1u << i))
            this_pack_sum += basis * static_cast<double>(n2_uint4_extract_byte(pack_data, i));
      }
      local_sum += this_pack_sum;
    }
  }

  // Scalar suffix
  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false, active = true;
    for (uint32_t round = 0; round < max_rounds && active; ++round)
    {
      uint8_t row_byte = subcolumns.ptrs[round][row];
      plane_digests[round] += static_cast<uint32_t>(row_byte);
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      if (row_byte > thresh_byte)      { qualified = true; active = false; }
      else if (row_byte < thresh_byte) { active = false; }
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

  // Block reductions for count + sum
  __shared__ unsigned long long red_smem[32];
  unsigned long long block_count = n2_block_reduce_sum_ull(
      static_cast<unsigned long long>(local_count), red_smem);

  __shared__ double red_dsmem[32];
  double local_double_sum = n2_warp_reduce_sum_double(local_sum);
  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int warp_count = (blockDim.x + 31) >> 5;
  if (lane == 0) red_dsmem[warp_id] = local_double_sum;
  __syncthreads();
  double block_sum = 0.0;
  if (warp_id == 0)
  {
    block_sum = (lane < warp_count) ? red_dsmem[lane] : 0.0;
    block_sum = n2_warp_reduce_sum_double(block_sum);
  }

  // Block reduction for per-plane digests
  __shared__ uint32_t digest_red[32 * N2_MAX_RUNTIME_PLANES];
  for (uint32_t p = 0; p < max_rounds; ++p)
  {
    uint32_t psum = n2_warp_reduce_sum_uint(plane_digests[p]);
    if (lane == 0) digest_red[warp_id + p * 32] = psum;
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    d_block_counts[blockIdx.x] = block_count;
    d_block_sums[blockIdx.x] = block_sum;
    for (uint32_t p = 0; p < max_rounds; ++p)
    {
      uint32_t fin = 0;
      for (int w = 0; w < warp_count; ++w)
        fin += digest_red[w + p * 32];
      d_block_plane_digests[p * digest_stride + blockIdx.x] = fin;
    }
  }
}

// ---------------------------------------------------------------------------
// P3: fused vote + digest inline for all read planes (branchless + cached)
//
// Combines P2 (vote from replicas) + P1 (per-plane digest on voted bytes).
// Each read plane has 3 replicas; voted bytes feed filter+sum AND digest.
// Uses branchless vote and caches voted uint4 packs to avoid double-load.
// ---------------------------------------------------------------------------
__global__ void nmr_v2_p3_fused_vote_digest_inline(
    N2RuntimeU8Subcolumns replicas_r0,
    N2RuntimeU8Subcolumns replicas_r1,
    N2RuntimeU8Subcolumns replicas_r2,
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
    double *__restrict__ d_block_sums,
    uint32_t *__restrict__ d_block_plane_digests,
    int digest_stride)
{
  uint64_t tile_rows = static_cast<uint64_t>(blockDim.x) *
                       static_cast<uint64_t>(N2_ROWPACK16_WIDTH);
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
  double local_sum = 0.0;

  const double *seg_basis = d_subcolumn_basis + segment_id * max_plane_count;
  double seg_base = d_segment_base[segment_id];

  uint32_t plane_digests[N2_MAX_RUNTIME_PLANES] = {0};

  // Scalar prefix — branchless vote + digest
  for (uint64_t row = tile_start + static_cast<uint64_t>(threadIdx.x);
       row < aligned_start;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    uint32_t active_sentinel = 0xFFFFFFFFu;
    for (uint32_t round = 0; round < max_rounds; ++round)
    {
      uint8_t voted = n2_scalar_vote_r3(replicas_r0.ptrs[round], replicas_r1.ptrs[round],
                                        replicas_r2.ptrs[round], row);
      plane_digests[round] += static_cast<uint32_t>(voted);
      uint8_t tb = d_threshold_bytes[thresh_offset + round];
      int is_gt = (voted > tb);
      int is_lt = (voted < tb);
      qualified |= (is_gt & active_sentinel);
      active_sentinel &= ~(is_gt | is_lt);
    }
    if (qualified)
    {
      double val = seg_base;
      for (uint32_t round = 0; round < max_rounds; ++round)
      {
        uint8_t voted = n2_scalar_vote_r3(replicas_r0.ptrs[round], replicas_r1.ptrs[round],
                                          replicas_r2.ptrs[round], row);
        val += seg_basis[round] * static_cast<double>(voted);
      }
      ++local_count;
      local_sum += val;
    }
  }

  // Rowpack16 main loop — vote+digest ALL rounds (correctness), filter + reconstruct from cache
  constexpr int kMaxCache = 8;
  for (uint64_t pack = static_cast<uint64_t>(threadIdx.x);
       pack < pack_count;
       pack += static_cast<uint64_t>(blockDim.x))
  {
    uint64_t pack_idx = first_pack + pack;

    // ---- Vote every round → cache + digest (all rounds required for digest correctness) ----
    uint4 voted_cache[kMaxCache];
    for (uint32_t round = 0; round < max_rounds; ++round)
    {
      const uint4 *p0 = reinterpret_cast<const uint4 *>(replicas_r0.ptrs[round]);
      const uint4 *p1 = reinterpret_cast<const uint4 *>(replicas_r1.ptrs[round]);
      const uint4 *p2 = reinterpret_cast<const uint4 *>(replicas_r2.ptrs[round]);
      uint4 vp = n2_uint4_byte_majority_vote_r3(p0[pack_idx], p1[pack_idx], p2[pack_idx]);
      voted_cache[round] = vp;
      plane_digests[round] += ((vp.x & 0xffu) + ((vp.x >> 8) & 0xffu) + ((vp.x >> 16) & 0xffu) + ((vp.x >> 24) & 0xffu));
      plane_digests[round] += ((vp.y & 0xffu) + ((vp.y >> 8) & 0xffu) + ((vp.y >> 16) & 0xffu) + ((vp.y >> 24) & 0xffu));
      plane_digests[round] += ((vp.z & 0xffu) + ((vp.z >> 8) & 0xffu) + ((vp.z >> 16) & 0xffu) + ((vp.z >> 24) & 0xffu));
      plane_digests[round] += ((vp.w & 0xffu) + ((vp.w >> 8) & 0xffu) + ((vp.w >> 16) & 0xffu) + ((vp.w >> 24) & 0xffu));
    }

    // ---- Filter from cache (can early exit) ----
    uint16_t active_mask = 0xFFFFu;
    uint16_t qualified_mask = 0x0000u;
    for (uint32_t round = 0; round < max_rounds && active_mask != 0; ++round)
    {
      uint8_t thresh_byte = d_threshold_bytes[thresh_offset + round];
      uint16_t gt_mask = 0x0000u, lt_mask = 0x0000u;
      n2_detail::uint4_gt_lt_masks(voted_cache[round], thresh_byte, gt_mask, lt_mask);
      qualified_mask |= static_cast<uint16_t>(active_mask & gt_mask);
      active_mask    &= static_cast<uint16_t>(~(active_mask & (gt_mask | lt_mask)));
    }

    local_count += static_cast<uint32_t>(__popc(qualified_mask));

    // ---- Reconstruct from cache with full-qualified fast path ----
    if (qualified_mask != 0)
    {
      int popc = __popc(qualified_mask);
      double this_pack_sum = seg_base * static_cast<double>(popc);
      if (qualified_mask == 0xFFFFu) {
        for (uint32_t round = 0; round < max_rounds; ++round) {
          uint4 vp = voted_cache[round]; double bs = seg_basis[round]; uint32_t w;
          w = vp.x; this_pack_sum += bs * (double)((w&0xFFu)+((w>>8)&0xFFu)+((w>>16)&0xFFu)+((w>>24)&0xFFu));
          w = vp.y; this_pack_sum += bs * (double)((w&0xFFu)+((w>>8)&0xFFu)+((w>>16)&0xFFu)+((w>>24)&0xFFu));
          w = vp.z; this_pack_sum += bs * (double)((w&0xFFu)+((w>>8)&0xFFu)+((w>>16)&0xFFu)+((w>>24)&0xFFu));
          w = vp.w; this_pack_sum += bs * (double)((w&0xFFu)+((w>>8)&0xFFu)+((w>>16)&0xFFu)+((w>>24)&0xFFu));
        }
      } else {
        for (uint32_t round = 0; round < max_rounds; ++round)
        {
          uint4 vp = voted_cache[round];
          double basis = seg_basis[round];
          for (int i = 0; i < 16; ++i)
            if (qualified_mask & static_cast<uint16_t>(1u << i))
              this_pack_sum += basis * static_cast<double>(n2_uint4_extract_byte(vp, i));
        }
      }
      local_sum += this_pack_sum;
    }
  }

  // Scalar suffix — branchless vote + digest
  for (uint64_t row = full_pack_end + static_cast<uint64_t>(threadIdx.x);
       row < tile_end;
       row += static_cast<uint64_t>(blockDim.x))
  {
    bool qualified = false;
    uint32_t active_sentinel = 0xFFFFFFFFu;
    for (uint32_t round = 0; round < max_rounds; ++round)
    {
      uint8_t voted = n2_scalar_vote_r3(replicas_r0.ptrs[round], replicas_r1.ptrs[round],
                                        replicas_r2.ptrs[round], row);
      plane_digests[round] += static_cast<uint32_t>(voted);
      uint8_t tb = d_threshold_bytes[thresh_offset + round];
      int is_gt = (voted > tb);
      int is_lt = (voted < tb);
      qualified |= (is_gt & active_sentinel);
      active_sentinel &= ~(is_gt | is_lt);
    }
    if (qualified)
    {
      double val = seg_base;
      for (uint32_t round = 0; round < max_rounds; ++round)
      {
        uint8_t voted = n2_scalar_vote_r3(replicas_r0.ptrs[round], replicas_r1.ptrs[round],
                                          replicas_r2.ptrs[round], row);
        val += seg_basis[round] * static_cast<double>(voted);
      }
      ++local_count;
      local_sum += val;
    }
  }

  // Block reductions
  __shared__ unsigned long long red_smem[32];
  unsigned long long block_count = n2_block_reduce_sum_ull(
      static_cast<unsigned long long>(local_count), red_smem);

  __shared__ double red_dsmem[32];
  double local_double_sum = n2_warp_reduce_sum_double(local_sum);
  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int warp_count = (blockDim.x + 31) >> 5;
  if (lane == 0) red_dsmem[warp_id] = local_double_sum;
  __syncthreads();
  double block_sum = 0.0;
  if (warp_id == 0)
  {
    block_sum = (lane < warp_count) ? red_dsmem[lane] : 0.0;
    block_sum = n2_warp_reduce_sum_double(block_sum);
  }

  __shared__ uint32_t digest_red[32 * N2_MAX_RUNTIME_PLANES];
  for (uint32_t p = 0; p < max_rounds; ++p)
  {
    uint32_t psum = n2_warp_reduce_sum_uint(plane_digests[p]);
    if (lane == 0) digest_red[warp_id + p * 32] = psum;
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    d_block_counts[blockIdx.x] = block_count;
    d_block_sums[blockIdx.x] = block_sum;
    for (uint32_t p = 0; p < max_rounds; ++p)
    {
      uint32_t fin = 0;
      for (int w = 0; w < warp_count; ++w)
        fin += digest_red[w + p * 32];
      d_block_plane_digests[p * digest_stride + blockIdx.x] = fin;
    }
  }
}

// ---------------------------------------------------------------------------
// P4: Raw FP64 baseline — filter + sum on raw double values
// ---------------------------------------------------------------------------
__global__ void nmr_v2_p4_raw_fp64(
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

  __shared__ unsigned long long red_smem[32];
  unsigned long long block_count = n2_block_reduce_sum_ull(local_count, red_smem);

  __shared__ double red_dsmem[32];
  double warp_sum = n2_warp_reduce_sum_double(local_sum);
  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int warp_count = (blockDim.x + 31) >> 5;
  if (lane == 0) red_dsmem[warp_id] = warp_sum;
  __syncthreads();
  double block_sum = 0.0;
  if (warp_id == 0)
  {
    block_sum = (lane < warp_count) ? red_dsmem[lane] : 0.0;
    block_sum = n2_warp_reduce_sum_double(block_sum);
  }

  if (threadIdx.x == 0)
  {
    d_block_counts[blockIdx.x] = block_count;
    d_block_sums[blockIdx.x] = block_sum;
  }
}

// =========================================================================
// Template<K> P2: compile-time K — per-plane r-aware
// =========================================================================
template<int K>
__global__ void nmr_v2_p2_kernel(
    N2RuntimeU8Subcolumns replicas_r0,
    N2RuntimeU8Subcolumns replicas_r1,
    N2RuntimeU8Subcolumns replicas_r2,
    uint64_t n, uint64_t segment_rows, uint64_t tiles_per_segment,
    const uint8_t *__restrict__ d_threshold_bytes, int threshold_stride, int max_plane_count,
    const uint32_t *__restrict__ d_active_plane_count,
    const double *__restrict__ d_segment_base, const double *__restrict__ d_subcolumn_basis,
    uint64_t *__restrict__ d_block_counts, double *__restrict__ d_block_sums,
    const uint32_t *__restrict__ d_r_per_plane)
{
  uint64_t tile_rows = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(N2_ROWPACK16_WIDTH);
  uint64_t segment_id = static_cast<uint64_t>(blockIdx.x) / tiles_per_segment;
  uint64_t tile_in_segment = static_cast<uint64_t>(blockIdx.x) % tiles_per_segment;
  uint64_t seg_start = segment_id * segment_rows, seg_end = seg_start + segment_rows;
  uint64_t tile_start = seg_start + tile_in_segment * tile_rows;
  if (tile_start >= n || tile_start >= seg_end) {
    if (threadIdx.x == 0) { d_block_counts[blockIdx.x] = 0ull; d_block_sums[blockIdx.x] = 0.0; }
    return;
  }
  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > seg_end) tile_end = seg_end;
  if (tile_end > n) tile_end = n;
  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0) ++aligned_start;
  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull, pack_count = (full_pack_end - aligned_start) / 16ull;

  uint32_t max_rounds = d_active_plane_count[segment_id];
  uint64_t thresh_offset = segment_id * static_cast<uint64_t>(threshold_stride);
  uint32_t local_count = 0u; double local_sum = 0.0;
  const double *seg_basis = d_subcolumn_basis + segment_id * max_plane_count;
  double seg_base = d_segment_base[segment_id];

  for (uint64_t row = tile_start + threadIdx.x; row < aligned_start; row += blockDim.x) {
    bool qualified = false; uint32_t as = 0xFFFFFFFFu;
    int r2dr = static_cast<int>(max_rounds);
    #pragma unroll
    for (uint32_t r = 0; r < K; ++r) {
      if (r >= max_rounds) break;
      uint32_t rp = d_r_per_plane[r];
      uint8_t voted;
      if (rp == 1) {
        voted = replicas_r0.ptrs[r][row];
      } else if (rp == 2) {
        uint8_t v0 = replicas_r0.ptrs[r][row];
        uint8_t v1 = replicas_r1.ptrs[r][row];
        if (v0 != v1) { r2dr = static_cast<int>(r); break; }
        voted = v0;
      } else {
        voted = n2_scalar_vote_r3(replicas_r0.ptrs[r], replicas_r1.ptrs[r], replicas_r2.ptrs[r], row);
      }
      uint8_t tb = d_threshold_bytes[thresh_offset + r];
      int gt = (voted > tb), lt = (voted < tb);
      qualified |= (gt & as); as &= ~(gt | lt);
    }
    if (qualified || r2dr < static_cast<int>(max_rounds)) {
      int ers = r2dr < static_cast<int>(max_rounds) ? r2dr : static_cast<int>(max_rounds);
      double val = seg_base;
      #pragma unroll
      for (int r = 0; r < K; ++r) {
        if (r >= ers) break;
        uint32_t rp = d_r_per_plane[r];
        uint8_t voted;
        if (rp == 1) { voted = replicas_r0.ptrs[r][row]; }
        else if (rp == 2) { voted = replicas_r0.ptrs[r][row]; }
        else { voted = n2_scalar_vote_r3(replicas_r0.ptrs[r], replicas_r1.ptrs[r], replicas_r2.ptrs[r], row); }
        val += seg_basis[r] * static_cast<double>(voted);
      }
      ++local_count; local_sum += val;
    }
  }

  for (uint64_t pack = threadIdx.x; pack < pack_count; pack += blockDim.x) {
    uint64_t pack_idx = first_pack + pack;
    uint4 vc[K];
    uint16_t r2d_m[K] = {0};
    #pragma unroll
    for (uint32_t r = 0; r < K; ++r) {
      if (r >= max_rounds) break;
      uint32_t rp = d_r_per_plane[r];
      if (rp == 1) {
        const uint4 *p0 = reinterpret_cast<const uint4 *>(replicas_r0.ptrs[r]);
        vc[r] = p0[pack_idx];
      } else if (rp == 2) {
        const uint4 *p0 = reinterpret_cast<const uint4 *>(replicas_r0.ptrs[r]);
        const uint4 *p1 = reinterpret_cast<const uint4 *>(replicas_r1.ptrs[r]);
        uint4 a = p0[pack_idx], b = p1[pack_idx];
        vc[r] = a;
        uint32_t x_x = a.x ^ b.x, y_x = a.y ^ b.y, z_x = a.z ^ b.z, w_x = a.w ^ b.w;
        uint16_t dm = 0;
        if (x_x & 0x000000FFu) dm |= 1; if (x_x & 0x0000FF00u) dm |= 2;
        if (x_x & 0x00FF0000u) dm |= 4; if (x_x & 0xFF000000u) dm |= 8;
        if (y_x & 0x000000FFu) dm |= 16; if (y_x & 0x0000FF00u) dm |= 32;
        if (y_x & 0x00FF0000u) dm |= 64; if (y_x & 0xFF000000u) dm |= 128;
        if (z_x & 0x000000FFu) dm |= 256; if (z_x & 0x0000FF00u) dm |= 512;
        if (z_x & 0x00FF0000u) dm |= 1024; if (z_x & 0xFF000000u) dm |= 2048;
        if (w_x & 0x000000FFu) dm |= 4096; if (w_x & 0x0000FF00u) dm |= 8192;
        if (w_x & 0x00FF0000u) dm |= 16384; if (w_x & 0xFF000000u) dm |= 32768;
        r2d_m[r] = dm;
      } else {
        const uint4 *p0 = reinterpret_cast<const uint4 *>(replicas_r0.ptrs[r]);
        const uint4 *p1 = reinterpret_cast<const uint4 *>(replicas_r1.ptrs[r]);
        const uint4 *p2 = reinterpret_cast<const uint4 *>(replicas_r2.ptrs[r]);
        vc[r] = n2_uint4_byte_majority_vote_r3(p0[pack_idx], p1[pack_idx], p2[pack_idx]);
      }
    }

    // r2 partial-plane: accumulate contributions from detected lanes
    uint16_t r2_det = 0;
    for (uint32_t rd = 0; rd < max_rounds; ++rd) {
      if (d_r_per_plane[rd] == 2 && r2d_m[rd] != 0) {
        uint16_t nd = r2d_m[rd] & ~r2_det;
        if (nd) {
          r2_det |= nd;
          int pn = __popc(nd);
          double sp = seg_base * static_cast<double>(pn);
          for (uint32_t pv = 0; pv < rd; ++pv)
            sp += n2_uint4_masked_scaled_sum(vc[pv], nd, seg_basis[pv]);
          local_count += static_cast<uint32_t>(pn);
          local_sum += sp;
        }
      }
    }

    uint16_t active_mask = 0xFFFFu & ~r2_det, qualified_mask = 0;
    #pragma unroll
    for (uint32_t r = 0; r < K && active_mask != 0; ++r) {
      if (r >= max_rounds) break;
      uint8_t thr = d_threshold_bytes[thresh_offset + r];
      uint16_t gt = 0, lt = 0;
      n2_detail::uint4_gt_lt_masks(vc[r], thr, gt, lt);
      qualified_mask |= static_cast<uint16_t>(active_mask & gt);
      active_mask &= static_cast<uint16_t>(~(active_mask & (gt | lt)));
    }

    uint32_t qualified_count = static_cast<uint32_t>(__popc(qualified_mask));
    local_count += qualified_count;

    if (qualified_mask) {
      double sp = seg_base * static_cast<double>(qualified_count);
      #pragma unroll
      for (uint32_t r = 0; r < K; ++r) {
        if (r >= max_rounds) break;
        sp += n2_uint4_masked_scaled_sum(vc[r], qualified_mask, seg_basis[r]);
      }
      local_sum += sp;
    }
  }

  for (uint64_t row = full_pack_end + threadIdx.x; row < tile_end; row += blockDim.x) {
    bool qualified = false; uint32_t as = 0xFFFFFFFFu;
    int r2dr = static_cast<int>(max_rounds);
    #pragma unroll
    for (uint32_t r = 0; r < K; ++r) {
      if (r >= max_rounds) break;
      uint32_t rp = d_r_per_plane[r];
      uint8_t voted;
      if (rp == 1) { voted = replicas_r0.ptrs[r][row]; }
      else if (rp == 2) {
        uint8_t v0 = replicas_r0.ptrs[r][row];
        uint8_t v1 = replicas_r1.ptrs[r][row];
        if (v0 != v1) { r2dr = static_cast<int>(r); break; }
        voted = v0;
      } else { voted = n2_scalar_vote_r3(replicas_r0.ptrs[r], replicas_r1.ptrs[r], replicas_r2.ptrs[r], row); }
      uint8_t tb = d_threshold_bytes[thresh_offset + r];
      int gt = (voted > tb), lt = (voted < tb);
      qualified |= (gt & as); as &= ~(gt | lt);
    }
    if (qualified || r2dr < static_cast<int>(max_rounds)) {
      int ers = r2dr < static_cast<int>(max_rounds) ? r2dr : static_cast<int>(max_rounds);
      double val = seg_base;
      #pragma unroll
      for (int r = 0; r < K; ++r) {
        if (r >= ers) break;
        uint32_t rp = d_r_per_plane[r];
        uint8_t voted;
        if (rp == 1) { voted = replicas_r0.ptrs[r][row]; }
        else if (rp == 2) { voted = replicas_r0.ptrs[r][row]; }
        else { voted = n2_scalar_vote_r3(replicas_r0.ptrs[r], replicas_r1.ptrs[r], replicas_r2.ptrs[r], row); }
        val += seg_basis[r] * static_cast<double>(voted);
      }
      ++local_count; local_sum += val;
    }
  }

  __shared__ unsigned long long rm[32];
  unsigned long long bc = n2_block_reduce_sum_ull(static_cast<unsigned long long>(local_count), rm);
  __shared__ double rdm[32];
  double lds = n2_warp_reduce_sum_double(local_sum);
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, wc = (blockDim.x + 31) >> 5;
  if (lane == 0) rdm[wid] = lds; __syncthreads();
  double bs = 0.0;
  if (wid == 0) { bs = (lane < wc) ? rdm[lane] : 0.0; bs = n2_warp_reduce_sum_double(bs); }
  if (threadIdx.x == 0) { d_block_counts[blockIdx.x] = bc; d_block_sums[blockIdx.x] = bs; }
}

// =========================================================================
// Template<K> P3
// =========================================================================
template<int K>
__global__ void nmr_v2_p3_kernel(
    N2RuntimeU8Subcolumns replicas_r0, N2RuntimeU8Subcolumns replicas_r1, N2RuntimeU8Subcolumns replicas_r2,
    uint64_t n, uint64_t segment_rows, uint64_t tiles_per_segment,
    const uint8_t *__restrict__ d_threshold_bytes, int threshold_stride, int max_plane_count,
    const uint32_t *__restrict__ d_active_plane_count,
    const double *__restrict__ d_segment_base, const double *__restrict__ d_subcolumn_basis,
    uint64_t *__restrict__ d_block_counts, double *__restrict__ d_block_sums,
    uint32_t *__restrict__ d_block_plane_digests, int digest_stride)
{
  uint64_t tile_rows = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(N2_ROWPACK16_WIDTH);
  uint64_t segment_id = static_cast<uint64_t>(blockIdx.x) / tiles_per_segment;
  uint64_t tile_in_segment = static_cast<uint64_t>(blockIdx.x) % tiles_per_segment;
  uint64_t seg_start = segment_id * segment_rows, seg_end = seg_start + segment_rows;
  uint64_t tile_start = seg_start + tile_in_segment * tile_rows;
  if (tile_start >= n || tile_start >= seg_end) {
    if (threadIdx.x == 0) { d_block_counts[blockIdx.x] = 0ull; d_block_sums[blockIdx.x] = 0.0; }
    return;
  }
  uint64_t tile_end = tile_start + tile_rows;
  if (tile_end > seg_end) tile_end = seg_end;
  if (tile_end > n) tile_end = n;
  uint64_t aligned_start = tile_start;
  while (aligned_start < tile_end && (aligned_start & 15ull) != 0) ++aligned_start;
  uint64_t full_pack_end = aligned_start + ((tile_end - aligned_start) / 16ull) * 16ull;
  uint64_t first_pack = aligned_start / 16ull, pack_count = (full_pack_end - aligned_start) / 16ull;

  uint32_t max_rounds = d_active_plane_count[segment_id];
  uint64_t thresh_offset = segment_id * static_cast<uint64_t>(threshold_stride);
  uint32_t local_count = 0u; double local_sum = 0.0;
  const double *seg_basis = d_subcolumn_basis + segment_id * max_plane_count;
  double seg_base = d_segment_base[segment_id];
  uint32_t plane_digests[N2_MAX_RUNTIME_PLANES] = {0};

  for (uint64_t row = tile_start + threadIdx.x; row < aligned_start; row += blockDim.x) {
    bool qualified = false; uint32_t as = 0xFFFFFFFFu;
    #pragma unroll
    for (uint32_t r = 0; r < K; ++r) {
      if (r >= max_rounds) break;
      uint8_t voted = n2_scalar_vote_r3(replicas_r0.ptrs[r], replicas_r1.ptrs[r], replicas_r2.ptrs[r], row);
      plane_digests[r] += static_cast<uint32_t>(voted);
      uint8_t tb = d_threshold_bytes[thresh_offset + r];
      int gt = (voted > tb), lt = (voted < tb);
      qualified |= (gt & as); as &= ~(gt | lt);
    }
    if (qualified) {
      double val = seg_base;
      #pragma unroll
      for (uint32_t r = 0; r < K; ++r) {
        if (r >= max_rounds) break;
        val += seg_basis[r] * static_cast<double>(n2_scalar_vote_r3(replicas_r0.ptrs[r], replicas_r1.ptrs[r], replicas_r2.ptrs[r], row));
      }
      ++local_count; local_sum += val;
    }
  }

  for (uint64_t pack = threadIdx.x; pack < pack_count; pack += blockDim.x) {
    uint64_t pack_idx = first_pack + pack;
    uint4 vc[K];
    #pragma unroll
    for (uint32_t r = 0; r < K; ++r) {
      if (r >= max_rounds) break;
      const uint4 *p0 = reinterpret_cast<const uint4 *>(replicas_r0.ptrs[r]);
      const uint4 *p1 = reinterpret_cast<const uint4 *>(replicas_r1.ptrs[r]);
      const uint4 *p2 = reinterpret_cast<const uint4 *>(replicas_r2.ptrs[r]);
      uint4 vp = n2_uint4_byte_majority_vote_r3(p0[pack_idx], p1[pack_idx], p2[pack_idx]);
      vc[r] = vp;
      plane_digests[r] += ((vp.x&0xffu)+((vp.x>>8)&0xffu)+((vp.x>>16)&0xffu)+((vp.x>>24)&0xffu));
      plane_digests[r] += ((vp.y&0xffu)+((vp.y>>8)&0xffu)+((vp.y>>16)&0xffu)+((vp.y>>24)&0xffu));
      plane_digests[r] += ((vp.z&0xffu)+((vp.z>>8)&0xffu)+((vp.z>>16)&0xffu)+((vp.z>>24)&0xffu));
      plane_digests[r] += ((vp.w&0xffu)+((vp.w>>8)&0xffu)+((vp.w>>16)&0xffu)+((vp.w>>24)&0xffu));
    }

    uint16_t active_mask = 0xFFFFu, qualified_mask = 0;
    #pragma unroll
    for (uint32_t r = 0; r < K && active_mask != 0; ++r) {
      if (r >= max_rounds) break;
      uint8_t thr = d_threshold_bytes[thresh_offset + r];
      uint16_t gt = 0, lt = 0;
      n2_detail::uint4_gt_lt_masks(vc[r], thr, gt, lt);
      qualified_mask |= static_cast<uint16_t>(active_mask & gt);
      active_mask &= static_cast<uint16_t>(~(active_mask & (gt | lt)));
    }

    local_count += static_cast<uint32_t>(__popc(qualified_mask));

    if (qualified_mask) {
      double sp = seg_base * static_cast<double>(__popc(qualified_mask));
      if (qualified_mask == 0xFFFFu) {
        #pragma unroll
        for (uint32_t r = 0; r < K; ++r) {
          if (r >= max_rounds) break;
          uint4 vp = vc[r]; double bs = seg_basis[r]; uint32_t w;
          w = vp.x; sp += bs * (double)((w&0xFFu)+((w>>8)&0xFFu)+((w>>16)&0xFFu)+((w>>24)&0xFFu));
          w = vp.y; sp += bs * (double)((w&0xFFu)+((w>>8)&0xFFu)+((w>>16)&0xFFu)+((w>>24)&0xFFu));
          w = vp.z; sp += bs * (double)((w&0xFFu)+((w>>8)&0xFFu)+((w>>16)&0xFFu)+((w>>24)&0xFFu));
          w = vp.w; sp += bs * (double)((w&0xFFu)+((w>>8)&0xFFu)+((w>>16)&0xFFu)+((w>>24)&0xFFu));
        }
      } else {
        #pragma unroll
        for (uint32_t r = 0; r < K; ++r) {
          if (r >= max_rounds) break;
          uint4 vp = vc[r]; double bs = seg_basis[r];
          #pragma unroll
          for (int i = 0; i < 16; ++i)
            if (qualified_mask & static_cast<uint16_t>(1u << i))
              sp += bs * static_cast<double>(n2_uint4_extract_byte(vp, i));
        }
      }
      local_sum += sp;
    }
  }

  for (uint64_t row = full_pack_end + threadIdx.x; row < tile_end; row += blockDim.x) {
    bool qualified = false; uint32_t as = 0xFFFFFFFFu;
    #pragma unroll
    for (uint32_t r = 0; r < K; ++r) {
      if (r >= max_rounds) break;
      uint8_t voted = n2_scalar_vote_r3(replicas_r0.ptrs[r], replicas_r1.ptrs[r], replicas_r2.ptrs[r], row);
      plane_digests[r] += static_cast<uint32_t>(voted);
      uint8_t tb = d_threshold_bytes[thresh_offset + r];
      int gt = (voted > tb), lt = (voted < tb);
      qualified |= (gt & as); as &= ~(gt | lt);
    }
    if (qualified) {
      double val = seg_base;
      #pragma unroll
      for (uint32_t r = 0; r < K; ++r) {
        if (r >= max_rounds) break;
        val += seg_basis[r] * static_cast<double>(n2_scalar_vote_r3(replicas_r0.ptrs[r], replicas_r1.ptrs[r], replicas_r2.ptrs[r], row));
      }
      ++local_count; local_sum += val;
    }
  }

  __shared__ unsigned long long rm[32];
  unsigned long long bc = n2_block_reduce_sum_ull(static_cast<unsigned long long>(local_count), rm);
  __shared__ double rdm[32];
  double lds = n2_warp_reduce_sum_double(local_sum);
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, wc = (blockDim.x + 31) >> 5;
  if (lane == 0) rdm[wid] = lds; __syncthreads();
  double bs = 0.0;
  if (wid == 0) { bs = (lane < wc) ? rdm[lane] : 0.0; bs = n2_warp_reduce_sum_double(bs); }

  __shared__ uint32_t dr[32 * N2_MAX_RUNTIME_PLANES];
  for (uint32_t p = 0; p < max_rounds; ++p) {
    uint32_t ps = n2_warp_reduce_sum_uint(plane_digests[p]);
    if (lane == 0) dr[wid + p * 32] = ps;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    d_block_counts[blockIdx.x] = bc; d_block_sums[blockIdx.x] = bs;
    for (uint32_t p = 0; p < max_rounds; ++p) {
      uint32_t fin = 0;
      for (int w = 0; w < wc; ++w) fin += dr[w + p * 32];
      d_block_plane_digests[p * digest_stride + blockIdx.x] = fin;
    }
  }
}

// =========================================================================
// Dispatch: K≤4 → template<K> (register cache, low regs), K>4 → non-templated
// =========================================================================
inline void nmr_v2_p2_dispatch(int k,
    N2RuntimeU8Subcolumns r0, N2RuntimeU8Subcolumns r1, N2RuntimeU8Subcolumns r2,
    uint64_t n, uint64_t seg_rows, uint64_t tiles_per_seg,
    const uint8_t *thresh, int thresh_stride, int max_planes, const uint32_t *active,
    const double *base, const double *basis, uint64_t *counts, double *sums,
    const uint32_t *d_r_per_plane,
    int T, uint64_t grid, cudaStream_t st)
{
  if (k <= 4) {
    switch (k) {
      case 1: nmr_v2_p2_kernel<1><<<(int)grid, T, 0, st>>>(r0,r1,r2, n,seg_rows,tiles_per_seg, thresh,thresh_stride,max_planes, active,base,basis, counts,sums,d_r_per_plane); break;
      case 2: nmr_v2_p2_kernel<2><<<(int)grid, T, 0, st>>>(r0,r1,r2, n,seg_rows,tiles_per_seg, thresh,thresh_stride,max_planes, active,base,basis, counts,sums,d_r_per_plane); break;
      case 3: nmr_v2_p2_kernel<3><<<(int)grid, T, 0, st>>>(r0,r1,r2, n,seg_rows,tiles_per_seg, thresh,thresh_stride,max_planes, active,base,basis, counts,sums,d_r_per_plane); break;
      default: nmr_v2_p2_kernel<4><<<(int)grid, T, 0, st>>>(r0,r1,r2, n,seg_rows,tiles_per_seg, thresh,thresh_stride,max_planes, active,base,basis, counts,sums,d_r_per_plane); break;
    }
  } else {
    nmr_v2_p2_fused_vote_inline<<<(int)grid, T, 0, st>>>(r0,r1,r2, n,seg_rows,tiles_per_seg, thresh,thresh_stride,max_planes, active,base,basis, counts,sums,d_r_per_plane);
  }
}

inline void nmr_v2_p3_dispatch(int k,
    N2RuntimeU8Subcolumns r0, N2RuntimeU8Subcolumns r1, N2RuntimeU8Subcolumns r2,
    uint64_t n, uint64_t seg_rows, uint64_t tiles_per_seg,
    const uint8_t *thresh, int thresh_stride, int max_planes, const uint32_t *active,
    const double *base, const double *basis, uint64_t *counts, double *sums,
    uint32_t *digests, int dig_stride,
    int T, uint64_t grid, cudaStream_t st)
{
  if (k <= 4) {
    switch (k) {
      case 1: nmr_v2_p3_kernel<1><<<(int)grid, T, 0, st>>>(r0,r1,r2, n,seg_rows,tiles_per_seg, thresh,thresh_stride,max_planes, active,base,basis, counts,sums, digests,dig_stride); break;
      case 2: nmr_v2_p3_kernel<2><<<(int)grid, T, 0, st>>>(r0,r1,r2, n,seg_rows,tiles_per_seg, thresh,thresh_stride,max_planes, active,base,basis, counts,sums, digests,dig_stride); break;
      case 3: nmr_v2_p3_kernel<3><<<(int)grid, T, 0, st>>>(r0,r1,r2, n,seg_rows,tiles_per_seg, thresh,thresh_stride,max_planes, active,base,basis, counts,sums, digests,dig_stride); break;
      default: nmr_v2_p3_kernel<4><<<(int)grid, T, 0, st>>>(r0,r1,r2, n,seg_rows,tiles_per_seg, thresh,thresh_stride,max_planes, active,base,basis, counts,sums, digests,dig_stride); break;
    }
  } else {
    nmr_v2_p3_fused_vote_digest_inline<<<(int)grid, T, 0, st>>>(r0,r1,r2, n,seg_rows,tiles_per_seg, thresh,thresh_stride,max_planes, active,base,basis, counts,sums, digests,dig_stride);
  }
}
