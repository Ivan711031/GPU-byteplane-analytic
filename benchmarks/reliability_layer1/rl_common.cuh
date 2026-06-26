#pragma once

#include <cuda_runtime.h>

#include <cstdint>

template <typename T, int N>
struct PlanePointers
{
  const T *ptrs[N];
};

using U8Planes = PlanePointers<uint8_t, 8>;

__host__ __device__ __forceinline__ uint64_t reconstruct_u64(uint8_t byte0, uint8_t byte1, uint8_t byte2, uint8_t byte3,
                                                             uint8_t byte4, uint8_t byte5, uint8_t byte6, uint8_t byte7)
{
  return (static_cast<uint64_t>(byte0) << 56) |
         (static_cast<uint64_t>(byte1) << 48) |
         (static_cast<uint64_t>(byte2) << 40) |
         (static_cast<uint64_t>(byte3) << 32) |
         (static_cast<uint64_t>(byte4) << 24) |
         (static_cast<uint64_t>(byte5) << 16) |
         (static_cast<uint64_t>(byte6) <<  8) |
         (static_cast<uint64_t>(byte7));
}

__host__ __device__ __forceinline__ Uint128 uint64_to_u128(uint64_t x)
{
  return Uint128{x, 0};
}
