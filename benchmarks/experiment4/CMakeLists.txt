cmake_minimum_required(VERSION 3.24)

project(gpu_byteplane_scan_experiment4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ptxas spill/local-memory warning helper for experiment4 CUDA targets
set(_EXP4_CUDA_OPTIONS --use_fast_math -O3 -lineinfo --ptxas-options=-v,--warn-on-spills)

add_executable(bench_progressive_filter
  bench_progressive_filter.cu
  ../experiment3/exp3_real_data_layout.cpp
  ../../buff_encoder/buff_codec.cpp
)

add_executable(bench_raw_fp64_count
  bench_raw_fp64_count.cu
)

# byteplane_count_gt static library
add_library(byteplane_count_gt STATIC
  byteplane_count_gt.cu
  ../experiment3/exp3_real_data_layout.cpp
  ../../buff_encoder/buff_codec.cpp
)

add_executable(byteplane_count_gt_cli
  byteplane_count_gt_cli.cu
)

add_executable(byteplane_count_gt_batch_cli
  byteplane_count_gt_batch_cli.cu
)

target_link_libraries(byteplane_count_gt_cli PRIVATE
  byteplane_count_gt
)

target_link_libraries(byteplane_count_gt_batch_cli PRIVATE
  byteplane_count_gt
)

target_include_directories(bench_progressive_filter PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
  ${CMAKE_CURRENT_SOURCE_DIR}/../../buff_encoder
)

target_include_directories(byteplane_count_gt PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
  ${CMAKE_CURRENT_SOURCE_DIR}/../../buff_encoder
)

target_include_directories(byteplane_count_gt_cli PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(byteplane_count_gt_batch_cli PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(bench_raw_fp64_count PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(bench_progressive_filter PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_EXP4_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_compile_options(byteplane_count_gt PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_EXP4_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_compile_options(byteplane_count_gt_cli PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_EXP4_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_compile_options(byteplane_count_gt_batch_cli PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_EXP4_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_compile_options(bench_raw_fp64_count PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_EXP4_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_threshold_prep_decomp
  bench_threshold_prep_decomp.cu
)

target_link_libraries(bench_threshold_prep_decomp PRIVATE
  byteplane_count_gt
)

target_include_directories(bench_threshold_prep_decomp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(bench_threshold_prep_decomp PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_EXP4_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_progressive_filter PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(byteplane_count_gt PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(byteplane_count_gt_cli PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(byteplane_count_gt_batch_cli PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(bench_raw_fp64_count PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(bench_threshold_prep_decomp PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
