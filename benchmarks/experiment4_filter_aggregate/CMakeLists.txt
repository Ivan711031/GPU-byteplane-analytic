cmake_minimum_required(VERSION 3.24)

project(gpu_byteplane_scan_filter_aggregate LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(_FA_CUDA_OPTIONS --use_fast_math -O3 -lineinfo --ptxas-options=-v,--warn-on-spills)

add_executable(bench_filter_aggregate
  bench_filter_aggregate.cu
  ../experiment3/exp3_real_data_layout.cpp
  ../../buff_encoder/buff_codec.cpp
)

target_include_directories(bench_filter_aggregate PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment4
  ${CMAKE_CURRENT_SOURCE_DIR}/../../buff_encoder
)

target_compile_options(bench_filter_aggregate PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_filter_aggregate PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(buff_cpu_proxy
  buff_cpu_proxy.cpp
  ../experiment3/exp3_real_data_layout.cpp
  ../../buff_encoder/buff_codec.cpp
)

target_include_directories(buff_cpu_proxy PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment4
  ${CMAKE_CURRENT_SOURCE_DIR}/../../buff_encoder
)

target_compile_options(buff_cpu_proxy PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(buff_cpu_proxy PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(buff_cpu_proxy_native
  buff_cpu_proxy.cpp
  ../experiment3/exp3_real_data_layout.cpp
  ../../buff_encoder/buff_codec.cpp
)

target_include_directories(buff_cpu_proxy_native PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment4
  ${CMAKE_CURRENT_SOURCE_DIR}/../../buff_encoder
)

target_compile_options(buff_cpu_proxy_native PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)

set_target_properties(buff_cpu_proxy_native PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_nmr_overhead
  bench_nmr_overhead.cu
  ../experiment3/exp3_real_data_layout.cpp
  ../../buff_encoder/buff_codec.cpp
)

target_include_directories(bench_nmr_overhead PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment4
  ${CMAKE_CURRENT_SOURCE_DIR}/../../buff_encoder
)

target_compile_options(bench_nmr_overhead PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_nmr_overhead PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_graded_nmr
  bench_graded_nmr.cu
  ../experiment3/exp3_real_data_layout.cpp
  ../../buff_encoder/buff_codec.cpp
)

target_include_directories(bench_graded_nmr PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment4
  ${CMAKE_CURRENT_SOURCE_DIR}/../../buff_encoder
)

target_compile_options(bench_graded_nmr PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_graded_nmr PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_graded_repair
  bench_graded_repair.cu
  ../experiment3/exp3_real_data_layout.cpp
  ../../buff_encoder/buff_codec.cpp
)

target_include_directories(bench_graded_repair PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment4
  ${CMAKE_CURRENT_SOURCE_DIR}/../../buff_encoder
)

target_compile_options(bench_graded_repair PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_graded_repair PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_graded_repair_x1
  bench_graded_repair_x1.cu
  ../experiment3/exp3_real_data_layout.cpp
  ../../buff_encoder/buff_codec.cpp
)

target_include_directories(bench_graded_repair_x1 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment4
  ${CMAKE_CURRENT_SOURCE_DIR}/../../buff_encoder
)

target_compile_options(bench_graded_repair_x1 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_graded_repair_x1 PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_graded_repair_x2
  bench_graded_repair_x2.cu
  ../experiment3/exp3_real_data_layout.cpp
  ../../buff_encoder/buff_codec.cpp
)

target_include_directories(bench_graded_repair_x2 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment4
  ${CMAKE_CURRENT_SOURCE_DIR}/../../buff_encoder
)

target_compile_options(bench_graded_repair_x2 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_graded_repair_x2 PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_graded_repair_timing
  bench_graded_repair_timing.cu
  ../experiment3/exp3_real_data_layout.cpp
  ../../buff_encoder/buff_codec.cpp
)

target_include_directories(bench_graded_repair_timing PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment4
  ${CMAKE_CURRENT_SOURCE_DIR}/../../buff_encoder
)

target_compile_options(bench_graded_repair_timing PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_graded_repair_timing PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_z0_bandwidth_derisk
  bench_z0_bandwidth_derisk.cu
  ../experiment3/exp3_real_data_layout.cpp
  ../../buff_encoder/buff_codec.cpp
)

target_include_directories(bench_z0_bandwidth_derisk PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment4
  ${CMAKE_CURRENT_SOURCE_DIR}/../../buff_encoder
)

target_compile_options(bench_z0_bandwidth_derisk PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS} -arch=sm_90>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_z0_bandwidth_derisk PROPERTIES
  CUDA_ARCHITECTURES "90"
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_z0b_bandwidth_derisk
  bench_z0b_bandwidth_derisk.cu
  ../experiment3/exp3_real_data_layout.cpp
)

target_include_directories(bench_z0b_bandwidth_derisk PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
)

target_compile_options(bench_z0b_bandwidth_derisk PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_z0b_bandwidth_derisk PROPERTIES
  CUDA_ARCHITECTURES "90"
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_z0d_integrity_derisk
  bench_z0d_integrity_derisk.cu
  ../experiment3/exp3_real_data_layout.cpp
)

target_include_directories(bench_z0d_integrity_derisk PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
)

target_compile_options(bench_z0d_integrity_derisk PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_z0d_integrity_derisk PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_z0e_sum32_sweep
  bench_z0e_sum32_sweep.cu
  ../experiment3/exp3_real_data_layout.cpp
)

target_include_directories(bench_z0e_sum32_sweep PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
)

target_compile_options(bench_z0e_sum32_sweep PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_z0e_sum32_sweep PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_z0f_parallel_sum32
  bench_z0f_parallel_sum32.cu
  ../experiment3/exp3_real_data_layout.cpp
)

target_include_directories(bench_z0f_parallel_sum32 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
)

target_compile_options(bench_z0f_parallel_sum32 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_z0f_parallel_sum32 PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_z_g1_gpu_smoke
  bench_z_g1_gpu_smoke.cu
  ../experiment3/exp3_real_data_layout.cpp
)

target_include_directories(bench_z_g1_gpu_smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
)

target_compile_options(bench_z_g1_gpu_smoke PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_z_g1_gpu_smoke PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_nmr_r0_gpu_smoke
  bench_nmr_r0_gpu_smoke.cu
  ../experiment3/exp3_real_data_layout.cpp
)

target_include_directories(bench_nmr_r0_gpu_smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
)

target_compile_options(bench_nmr_r0_gpu_smoke PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_nmr_r0_gpu_smoke PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_z_ext_digest_sweep
  bench_z_ext_digest_sweep.cu
  ../experiment3/exp3_real_data_layout.cpp
)

target_include_directories(bench_z_ext_digest_sweep PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
)

target_compile_options(bench_z_ext_digest_sweep PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_z_ext_digest_sweep PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_nmr_b_e2e_pipeline
  bench_nmr_b_e2e_pipeline.cu
  ../experiment3/exp3_real_data_layout.cpp
)

target_include_directories(bench_nmr_b_e2e_pipeline PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
)

target_compile_options(bench_nmr_b_e2e_pipeline PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_nmr_b_e2e_pipeline PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_nmr_c_ncu_profile
  bench_nmr_c_ncu_profile.cu
  ../experiment3/exp3_real_data_layout.cpp
)

target_include_directories(bench_nmr_c_ncu_profile PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment4
)

target_compile_options(bench_nmr_c_ncu_profile PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_nmr_c_ncu_profile PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

add_executable(bench_nmr_c_v2_k_sweep
  bench_nmr_c_v2_k_sweep.cu
  ../experiment3/exp3_real_data_layout.cpp
)

target_include_directories(bench_nmr_c_v2_k_sweep PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment3
  ${CMAKE_CURRENT_SOURCE_DIR}/../experiment4
)

target_compile_options(bench_nmr_c_v2_k_sweep PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${_FA_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(bench_nmr_c_v2_k_sweep PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

