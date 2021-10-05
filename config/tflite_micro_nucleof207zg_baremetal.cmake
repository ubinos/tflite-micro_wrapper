#
# Copyright (c) 2019 Sung Ho Park and CSOS
#
# SPDX-License-Identifier: Apache-2.0
#

# {ubinos_config_type: [buildable, cmake, lib]}

set_cache(PROJECT_TOOLCHAIN_C_STD "GNU11" STRING)
set_cache(PROJECT_TOOLCHAIN_CXX_STD "GNU++14" STRING)
set_cache(PROJECT_TOOLCHAIN_CXX_W_NO_CXX14_COMPAT FALSE BOOL)

include(${PROJECT_LIBRARY_DIR}/CMSIS_5_wrapper/config/cmsis_5.cmake)
include(${PROJECT_UBINOS_DIR}/config/ubinos_nucleof207zg_baremetal.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/tflite_micro.cmake)

