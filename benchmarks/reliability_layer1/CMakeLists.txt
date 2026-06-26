cmake_minimum_required(VERSION 3.24)

project(gpu_reliability_layer1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(bench_reliability_fixture bench_reliability_fixture.cu)

target_include_directories(bench_reliability_fixture PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(bench_reliability_fixture PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_reliability_fixture PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

add_executable(bench_reliability_phase2 bench_reliability_phase2.cu)

target_include_directories(bench_reliability_phase2 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(bench_reliability_phase2 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_reliability_phase2 PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
