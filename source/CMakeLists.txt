cmake_minimum_required(VERSION 3.3)

include("${CMAKE_CURRENT_LIST_DIR}/../../ubinos/config/common.cmake")

project(tflite-micro_wrapper)
project_begin()
project_add_library(ubinos)
project_add_library(CMSIS_5_wrapper)
project_end()

