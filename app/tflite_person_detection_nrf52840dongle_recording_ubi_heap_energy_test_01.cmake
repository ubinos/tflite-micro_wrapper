#
# Copyright (c) 2023 Sung Ho Park and CSOS
#
# SPDX-License-Identifier: Apache-2.0
#

# ubinos_config_info {"name_base": "tflite_person_detection", "build_type": "cmake_ubinos", "app": true}

set_cache(UBINOS__UBIK__TICK_RTC_SLEEP_IDLE FALSE BOOL)
set_cache(UBINOS__UBIK__TICK_RTC_TICKLESS_IDLE FALSE BOOL)
set_cache(UBINOS__UBICLIB__EXCLUDE_HEAP_DMPM TRUE BOOL)
set_cache(UBINOS__UBICLIB__EXCLUDE_HEAP_DMPM_DIR_ON_OFF TRUE BOOL)

include(${CMAKE_CURRENT_LIST_DIR}/tflite_person_detection_nrf52840dongle_recording_ubi_heap.cmake)
