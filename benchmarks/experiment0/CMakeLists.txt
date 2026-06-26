cmake_minimum_required(VERSION 3.24)

project(gpu_byteplane_scan_experiment0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(bench_hbm_bw bench_hbm_bw.cu)

target_compile_options(bench_hbm_bw PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_hbm_bw PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

add_executable(p3d_checksum_timing p3d_checksum_timing.cu)

target_compile_options(p3d_checksum_timing PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -arch=sm_90>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(p3d_checksum_timing PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

