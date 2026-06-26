cmake_minimum_required(VERSION 3.24)

project(gpu_byteplane_scan_experiment1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(bench_byteplane_scan bench_byteplane_scan.cu)
add_executable(bench_contiguous_baseline bench_contiguous_baseline.cu)

function(configure_exp1_target target_name)
  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )

  set_target_properties(${target_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
  )
endfunction()

configure_exp1_target(bench_byteplane_scan)
configure_exp1_target(bench_contiguous_baseline)
