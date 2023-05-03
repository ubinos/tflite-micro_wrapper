#
# Copyright (c) 2023 Sung Ho Park and CSOS
#
# SPDX-License-Identifier: Apache-2.0
#

# ubinos_config_info {"name_base": "tflite_ubi_heap_micro_allocator_test", "build_type": "cmake_ubinos", "app": true}

set_cache(PROJECT_TOOLCHAIN_C_STD "GNU11" STRING)
set_cache(PROJECT_TOOLCHAIN_CXX_STD "GNU++14" STRING)
set_cache(PROJECT_TOOLCHAIN_CXX_W_NO_CXX14_COMPAT FALSE BOOL)

set_cache(UBINOS__BSP__CMSIS_INCLUDE_DIR "${PROJECT_LIBRARY_DIR}/CMSIS_5/CMSIS/Core/Include" PATH)

set_cache(UBINOS__BSP__LINKSCRIPT_FILE "${PROJECT_UBINOS_DIR}/source/ubinos/bsp/arch/arm/cortexm/nrf52/xxaa/flash_bdh_align16.ld" PATH)

set_cache(UBINOS__UBIK__TICK_TYPE "RTC" STRING)
set_cache(UBINOS__UBIK__TICK_PER_SEC 1024 STRING)

set_cache(UBINOS__UBICLIB__EXCLUDE_CLI FALSE BOOL)

set_cache(UBINOS__UBICLIB__EXCLUDE_HEAP_DMPM FALSE BOOL)
set_cache(UBINOS__UBICLIB__EXCLUDE_HEAP_DMPM_DIR_ON_OFF FALSE BOOL)
set_cache(UBINOS__UBICLIB__EXCLUDE_HEAP_DMPM_MEMORY_READY_CHECK FALSE BOOL)

set_cache(UBINOS__UBICLIB__EXCLUDE_HEAP_ALGORITHM__BESTFIT FALSE BOOL)
set_cache(UBINOS__UBICLIB__EXCLUDE_HEAP_ALGORITHM__FIRSTFIT FALSE BOOL)
set_cache(UBINOS__UBICLIB__EXCLUDE_HEAP_ALGORITHM__NEXTFIT FALSE BOOL)
set_cache(UBINOS__UBICLIB__EXCLUDE_HEAP_ALGORITHM__PGROUP FALSE BOOL)
set_cache(UBINOS__UBICLIB__EXCLUDE_HEAP_ALGORITHM__GROUP FALSE BOOL)
set_cache(UBINOS__UBICLIB__EXCLUDE_HEAP_ALGORITHM__BBUDDY FALSE BOOL)
set_cache(UBINOS__UBICLIB__EXCLUDE_HEAP_ALGORITHM__WBUDDY FALSE BOOL)

# set_cache(UBINOS__UBICLIB__HEAP_DIR0_ALGORITHM "GROUP" STRING)
# set_cache(UBINOS__UBICLIB__HEAP_DIR0_LOCKTYPE "MUTEX" STRING)
# set_cache(UBINOS__UBICLIB__HEAP_DIR0_M 8 STRING)
# set_cache(UBINOS__UBICLIB__HEAP_DIR0_FBLCOUNT 226 STRING)
# set_cache(UBINOS__UBICLIB__HEAP_DIR0_FBLBM_BUFSIZE 36 STRING)

set_cache(UBINOS__UBICLIB__HEAP_DIR0_ALGORITHM "BESTFIT" STRING)
set_cache(UBINOS__UBICLIB__HEAP_DIR0_LOCKTYPE "MUTEX" STRING)
set_cache(UBINOS__UBICLIB__HEAP_DIR0_M 2 STRING)
set_cache(UBINOS__UBICLIB__HEAP_DIR0_FBLCOUNT 2 STRING)
set_cache(UBINOS__UBICLIB__HEAP_DIR0_FBLBM_BUFSIZE 4 STRING)

set_cache(UBINOS__UBICLIB__HEAP_DIR1_ALGORITHM "BESTFIT" STRING)
set_cache(UBINOS__UBICLIB__HEAP_DIR1_LOCKTYPE "MUTEX" STRING)
set_cache(UBINOS__UBICLIB__HEAP_DIR1_M 2 STRING)
set_cache(UBINOS__UBICLIB__HEAP_DIR1_FBLCOUNT 2 STRING)
set_cache(UBINOS__UBICLIB__HEAP_DIR1_FBLBM_BUFSIZE 4 STRING)

set_cache(UBINOS__UBICLIB__HEAP_DEFAULT_DIR 1 STRING)
set_cache(UBINOS__UBICLIB__HEAP_ALIGNMENT 16 STRING)

set_cache(NRF5SDK__BSP_DEFINES_ONLY TRUE BOOL)
set_cache(NRF5SDK__NRFX_POWER_ENABLED FALSE BOOL)

# set_cache(UBINOS__BSP__DTTY_TYPE "EXTERNAL" STRING)
# set_cache(NRF5SDK__DTTY_NRF_UART_ENABLE TRUE BOOL)
# set_cache(NRF5SDK__UART_ENABLED TRUE BOOL)
# set_cache(NRF5SDK__NRFX_UARTE0_ENABLED TRUE BOOL)

set_cache(UBINOS__BSP__STACK_SIZE 0x8000 STRING)

include(${PROJECT_UBINOS_DIR}/config/ubinos_nrf52840dk.cmake)
include(${PROJECT_LIBRARY_DIR}/CMSIS_5_wrapper/config/cmsis_5.cmake)
include(${PROJECT_LIBRARY_DIR}/seggerrtt_wrapper/config/seggerrtt.cmake)
include(${PROJECT_LIBRARY_DIR}/nrf5sdk_wrapper/config/nrf5sdk.cmake)
include(${PROJECT_LIBRARY_DIR}/nrf5sdk_extension/config/nrf5sdk_extension.cmake)
include(${PROJECT_LIBRARY_DIR}/tflite-micro_wrapper/config/tflite_micro.cmake)
include(${PROJECT_LIBRARY_DIR}/googletest_wrapper/config/googletest.cmake)

####

set(INCLUDE__APP TRUE)
set(APP__NAME "tflite_ubi_heap_micro_allocator_test")

get_filename_component(_tmp_source_dir "${CMAKE_CURRENT_LIST_DIR}/${APP__NAME}" ABSOLUTE)
string(TOLOWER ${UBINOS__BSP__BOARD_VARIATION_NAME} _temp_board_model)
string(TOLOWER ${UBINOS__BSP__NRF52_SOFTDEVICE_NAME} _temp_softdevice_name)

include_directories(${_tmp_source_dir}/arch/arm/cortexm/${_temp_board_model}/${_temp_softdevice_name}/config)
include_directories(${_tmp_source_dir}/arch/arm/cortexm/${_temp_board_model})
include_directories(${_tmp_source_dir})

file(GLOB_RECURSE _tmp_sources
    "${_tmp_source_dir}/*.c"
    "${_tmp_source_dir}/*.cpp"
    "${_tmp_source_dir}/*.cc"
    "${_tmp_source_dir}/*.S")

set(PROJECT_APP_SOURCES ${PROJECT_APP_SOURCES} ${_tmp_sources})
