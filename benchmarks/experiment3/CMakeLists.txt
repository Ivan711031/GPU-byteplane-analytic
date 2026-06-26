cmake_minimum_required(VERSION 3.24)

project(gpu_byteplane_scan_experiment3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(bench_progressive_aggregation
  bench_progressive_aggregation.cu
  exp3_real_data_layout.cpp
)

add_executable(bench_raw_fp64_sum
  bench_raw_fp64_sum.cu
)

target_include_directories(bench_progressive_aggregation PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(bench_raw_fp64_sum PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(bench_progressive_aggregation PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -lineinfo --ptxas-options=-v>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_progressive_aggregation PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

add_executable(export_encoded_dev_layout
  export_encoded_dev_layout.cpp
  exp3_real_data_layout.cpp
  ../../buff_encoder/buff_codec.cpp
)

target_include_directories(export_encoded_dev_layout PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../buff_encoder
)

set_target_properties(export_encoded_dev_layout PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

target_compile_options(export_encoded_dev_layout PRIVATE
  -O3
)
