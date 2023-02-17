#
# Copyright (c) 2023 Sung Ho Park and CSOS
#
# SPDX-License-Identifier: Apache-2.0
#

# ubinos_config_info {"name_base": "tflite_person_detection", "build_type": "cmake_ubinos", "app": true}

set_cache(TFLITE_MICRO__INTERPRETER_TYPE "RECORDING_MICRO" STRING)

include(${CMAKE_CURRENT_LIST_DIR}/tflite_person_detection_nrf52840dk_baremetal.cmake)
