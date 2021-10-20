#
# Copyright (c) 2021 Sung Ho Park and CSOS
#
# SPDX-License-Identifier: Apache-2.0
#

set(INCLUDE__TFLITE_MICRO TRUE)
set(PROJECT_UBINOS_LIBRARIES ${PROJECT_UBINOS_LIBRARIES} tflite-micro_wrapper)

set_cache_default(TFLITE_MICRO__BASE_DIR "${PROJECT_LIBRARY_DIR}/tflite-micro" STRING "tflite-micro project base dir")

