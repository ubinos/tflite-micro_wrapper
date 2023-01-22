#
# Copyright (c) 2023 Sung Ho Park and CSOS
#
# SPDX-License-Identifier: Apache-2.0
#

# ubinos_config_info {"name_base": "tflite_micro_recording_micro_allocator_test", "build_type": "cmake_ubinos", "app": true}

set_cache(PROJECT_TOOLCHAIN_C_STD "GNU11" STRING)
set_cache(PROJECT_TOOLCHAIN_CXX_STD "GNU++14" STRING)
set_cache(PROJECT_TOOLCHAIN_CXX_W_NO_CXX14_COMPAT FALSE BOOL)

set_cache(TFLITE_MICRO__INCLUDE_CMSIS_NN FALSE BOOL)

include(${PROJECT_UBINOS_DIR}/config/ubinos_local_llvm_baremetal.cmake)
include(${PROJECT_LIBRARY_DIR}/tflite-micro_wrapper/config/tflite_micro.cmake)
include(${PROJECT_LIBRARY_DIR}/googletest_wrapper/config/googletest.cmake)

####

set(INCLUDE__APP TRUE)
set(APP__NAME "tflite_micro_recording_micro_allocator_test")

get_filename_component(_tmp_source_dir "${TFLITE_MICRO__BASE_DIR}/tensorflow/lite/micro" ABSOLUTE)

set(PROJECT_APP_SOURCES ${PROJECT_APP_SOURCES} ${_tmp_source_dir}/recording_micro_allocator_test.cc)
set(PROJECT_APP_SOURCES ${PROJECT_APP_SOURCES} ${_tmp_source_dir}/testing/test_conv_model.cc)

add_definitions("-DTF_LITE_STATIC_MEMORY")
add_definitions("-DTF_LITE_MCU_DEBUG_LOG")
