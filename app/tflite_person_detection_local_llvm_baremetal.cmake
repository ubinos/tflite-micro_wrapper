#
# Copyright (c) 2022 Sung Ho Park and CSOS
#
# SPDX-License-Identifier: Apache-2.0
#

# ubinos_config_info {"name_base": "tflite_person_detection", "build_type": "cmake_ubinos", "app": true}

set_cache(PROJECT_TOOLCHAIN_C_STD "GNU11" STRING)
set_cache(PROJECT_TOOLCHAIN_CXX_STD "GNU++14" STRING)
set_cache(PROJECT_TOOLCHAIN_CXX_W_NO_CXX14_COMPAT FALSE BOOL)

set_cache(TFLITE_MICRO__INCLUDE_CMSIS_NN FALSE BOOL)
set_cache(TFLITE_MICRO__PERSON_DETECTION_IMAGE_PROVIDER_TYPE "TEST01" STRING)

include(${PROJECT_UBINOS_DIR}/config/ubinos_local_llvm_baremetal.cmake)
include(${PROJECT_LIBRARY_DIR}/tflite-micro_wrapper/config/tflite_micro.cmake)

####

set(INCLUDE__APP TRUE)
set(APP__NAME "tflite_person_detection")

get_filename_component(_tmp_source_dir "${CMAKE_CURRENT_LIST_DIR}/${APP__NAME}" ABSOLUTE)

include_directories(${_tmp_source_dir})

file(GLOB_RECURSE _tmp_sources
    "${_tmp_source_dir}/*.c"
    "${_tmp_source_dir}/*.cpp"
    "${_tmp_source_dir}/*.cc"
    "${_tmp_source_dir}/*.S")

set(PROJECT_APP_SOURCES ${PROJECT_APP_SOURCES} ${_tmp_sources})
