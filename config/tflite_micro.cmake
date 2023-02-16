#
# Copyright (c) 2021 Sung Ho Park and CSOS
#
# SPDX-License-Identifier: Apache-2.0
#

set(INCLUDE__TFLITE_MICRO TRUE)
set(PROJECT_UBINOS_LIBRARIES ${PROJECT_UBINOS_LIBRARIES} tflite-micro_wrapper)

set_cache_default(TFLITE_MICRO__BASE_DIR "${PROJECT_LIBRARY_DIR}/tflite-micro" STRING "tflite-micro project base dir")
set_cache_default(TFLITE_MICRO__KISSFFT_BASE_DIR "${PROJECT_LIBRARY_DIR}/kissfft" STRING "kissfft project base dir")
set_cache_default(TFLITE_MICRO__FLATBUFFERS_BASE_DIR "${PROJECT_LIBRARY_DIR}/flatbuffers" STRING "flatbuffers project base dir")
set_cache_default(TFLITE_MICRO__GEMMLOWP_BASE_DIR "${PROJECT_LIBRARY_DIR}/gemmlowp" STRING "gemmlowp project base dir")
set_cache_default(TFLITE_MICRO__RUY_BASE_DIR "${PROJECT_LIBRARY_DIR}/ruy" STRING "ruy project base dir")

set_cache_default(TFLITE_MICRO__TF_LITE_DISABLE_X86_NEON TRUE BOOL "Disable x86 NEON")
set_cache_default(TFLITE_MICRO__TF_LITE_STATIC_MEMORY TRUE BOOL "Use TFLite staic memory")
set_cache_default(TFLITE_MICRO__TF_LITE_MCU_DEBUG_LOG TRUE BOOL "Use TFLite MCU debug log")
set_cache_default(TFLITE_MICRO__TF_LITE_USE_CTIME FALSE BOOL "Use TFLite CTime")

set_cache_default(TFLITE_MICRO__INCLUDE_CMSIS_NN TRUE BOOL "Include ARM CMSIS NN")

set_cache_default(TFLITE_MICRO__PERSON_DETECTION_IMAGE_PROVIDER_TYPE "DUMMY" STRING "Image provider type of person detection app [DUMMY | TEST01]")

set_cache_default(FLATBUFFERS_LOCALE_INDEPENDENT FALSE BOOL "FlatBuffers locale independent")

add_definitions(-DFLATBUFFERS_LOCALE_INDEPENDENT=$<BOOL:${FLATBUFFERS_LOCALE_INDEPENDENT}>)
