#
# Copyright (c) 2023 Sung Ho Park and CSOS
#
# SPDX-License-Identifier: Apache-2.0
#

# ubinos_config_info {"name_base": "tflite_micro", "build_type": "cmake_ubinos"}

set_cache(PROJECT_TOOLCHAIN_C_STD "GNU11" STRING)
set_cache(PROJECT_TOOLCHAIN_CXX_STD "GNU++14" STRING)
set_cache(PROJECT_TOOLCHAIN_CXX_W_NO_CXX14_COMPAT FALSE BOOL)

set_cache(TFLITE_MICRO__INCLUDE_CMSIS_NN FALSE BOOL)

include(${PROJECT_UBINOS_DIR}/config/ubinos_local_llvm_baremetal.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/tflite_micro.cmake)
