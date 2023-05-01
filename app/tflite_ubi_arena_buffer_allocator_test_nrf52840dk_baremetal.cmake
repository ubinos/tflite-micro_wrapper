#
# Copyright (c) 2023 Sung Ho Park and CSOS
#
# SPDX-License-Identifier: Apache-2.0
#

# ubinos_config_info {"name_base": "tflite_ubi_arena_buffer_allocator_test", "build_type": "cmake_ubinos", "app": true}

set_cache(PROJECT_TOOLCHAIN_C_STD "GNU11" STRING)
set_cache(PROJECT_TOOLCHAIN_CXX_STD "GNU++14" STRING)
set_cache(PROJECT_TOOLCHAIN_CXX_W_NO_CXX14_COMPAT FALSE BOOL)

set_cache(UBINOS__BSP__CMSIS_INCLUDE_DIR "${PROJECT_LIBRARY_DIR}/CMSIS_5/CMSIS/Core/Include" PATH)

set_cache(NRF5SDK__BSP_DEFINES_ONLY TRUE BOOL)
set_cache(NRF5SDK__NRFX_POWER_ENABLED FALSE BOOL)

include(${PROJECT_UBINOS_DIR}/config/ubinos_nrf52840dk_baremetal.cmake)
include(${PROJECT_LIBRARY_DIR}/CMSIS_5_wrapper/config/cmsis_5.cmake)
include(${PROJECT_LIBRARY_DIR}/tflite-micro_wrapper/config/tflite_micro.cmake)
include(${PROJECT_LIBRARY_DIR}/googletest_wrapper/config/googletest.cmake)

####

set(INCLUDE__APP TRUE)
set(APP__NAME "tflite_ubi_arena_buffer_allocator_test")

get_filename_component(_tmp_source_dir "${CMAKE_CURRENT_LIST_DIR}/${APP__NAME}" ABSOLUTE)

include_directories(${_tmp_source_dir})

file(GLOB_RECURSE _tmp_sources
    "${_tmp_source_dir}/*.c"
    "${_tmp_source_dir}/*.cpp"
    "${_tmp_source_dir}/*.cc"
    "${_tmp_source_dir}/*.S")

set(PROJECT_APP_SOURCES ${PROJECT_APP_SOURCES} ${_tmp_sources})
