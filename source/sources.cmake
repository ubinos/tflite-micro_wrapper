if(INCLUDE__TFLITE_MICRO)

    get_filename_component(_tmp_lib_src_dir "${TFLITE_MICRO__BASE_DIR}" ABSOLUTE)

    include_directories(${_tmp_lib_src_dir})

    if(TFLITE_MICRO__TF_LITE_DISABLE_X86_NEON)
        add_definitions("-DTF_LITE_DISABLE_X86_NEON")
    endif()
    if(TFLITE_MICRO__TF_LITE_STATIC_MEMORY)
        add_definitions("-DTF_LITE_STATIC_MEMORY")
    endif()
    if(TFLITE_MICRO__TF_LITE_MCU_DEBUG_LOG)
        add_definitions("-DTF_LITE_MCU_DEBUG_LOG")
    endif()
    if(TFLITE_MICRO__TF_LITE_USE_CTIME)
        add_definitions("-DTF_LITE_USE_CTIME")
    endif()
    if(TFLITE_MICRO__TF_LITE_SHOW_MEMORY_USE)
        add_definitions("-DTF_LITE_SHOW_MEMORY_USE")
    endif()

    if(TFLITE_MICRO__INCLUDE_CMSIS_NN)
        add_definitions("-DCMSIS_NN")
    endif()

    get_filename_component(_tmp_lib_src_dir "${TFLITE_MICRO__BASE_DIR}/tensorflow/lite/c" ABSOLUTE)
    file(GLOB_RECURSE _tmp_sources
        "${_tmp_lib_src_dir}/*.c"
        "${_tmp_lib_src_dir}/*.cc")
    list(FILTER _tmp_sources EXCLUDE REGEX "_test.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark_8bit.cc$")
    set(PROJECT_SOURCES ${PROJECT_SOURCES} ${_tmp_sources})

    get_filename_component(_tmp_lib_src_dir "${TFLITE_MICRO__BASE_DIR}/tensorflow/lite/core" ABSOLUTE)
    file(GLOB_RECURSE _tmp_sources
        "${_tmp_lib_src_dir}/*.c"
        "${_tmp_lib_src_dir}/*.cc")
    list(FILTER _tmp_sources EXCLUDE REGEX "_test.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark_8bit.cc$")
    set(PROJECT_SOURCES ${PROJECT_SOURCES} ${_tmp_sources})

    get_filename_component(_tmp_lib_src_dir "${TFLITE_MICRO__BASE_DIR}/tensorflow/lite/kernels" ABSOLUTE)
    file(GLOB_RECURSE _tmp_sources
        "${_tmp_lib_src_dir}/*.c"
        "${_tmp_lib_src_dir}/*.cc")
    list(FILTER _tmp_sources EXCLUDE REGEX "_test.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark_8bit.cc$")
    set(PROJECT_SOURCES ${PROJECT_SOURCES} ${_tmp_sources})

    get_filename_component(_tmp_lib_src_dir "${TFLITE_MICRO__BASE_DIR}/tensorflow/lite/schema" ABSOLUTE)
    file(GLOB_RECURSE _tmp_sources
        "${_tmp_lib_src_dir}/*.c"
        "${_tmp_lib_src_dir}/*.cc")
    list(FILTER _tmp_sources EXCLUDE REGEX "_test.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark_8bit.cc$")
    set(PROJECT_SOURCES ${PROJECT_SOURCES} ${_tmp_sources})

    get_filename_component(_tmp_lib_src_dir "${TFLITE_MICRO__BASE_DIR}/tensorflow/lite/micro" ABSOLUTE)
    file(GLOB _tmp_sources
        "${_tmp_lib_src_dir}/*.c"
        "${_tmp_lib_src_dir}/*.cc")
    list(FILTER _tmp_sources EXCLUDE REGEX "_test.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark_8bit.cc$")
    set(PROJECT_SOURCES ${PROJECT_SOURCES} ${_tmp_sources})

    get_filename_component(_tmp_lib_src_dir "${TFLITE_MICRO__BASE_DIR}/tensorflow/lite/micro/kernels" ABSOLUTE)
    file(GLOB _tmp_sources
        "${_tmp_lib_src_dir}/*.c"
        "${_tmp_lib_src_dir}/*.cc")
    list(FILTER _tmp_sources EXCLUDE REGEX "_test.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark_8bit.cc$")
    set(PROJECT_SOURCES ${PROJECT_SOURCES} ${_tmp_sources})

    if(TFLITE_MICRO__INCLUDE_CMSIS_NN)
        get_filename_component(_tmp_lib_src_dir "${TFLITE_MICRO__BASE_DIR}/tensorflow/lite/micro/kernels/cmsis_nn" ABSOLUTE)
        file(GLOB_RECURSE _tmp_sources
            "${_tmp_lib_src_dir}/*.c"
            "${_tmp_lib_src_dir}/*.cc")
        list(FILTER _tmp_sources EXCLUDE REGEX "_test.cc$")
        list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark.cc$")
        list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark_8bit.cc$")
        set(PROJECT_SOURCES ${PROJECT_SOURCES} ${_tmp_sources})
        set_source_files_properties(${_tmp_lib_src_dir}/conv.cc PROPERTIES COMPILE_FLAGS "-Wno-sign-compare -Wno-enum-compare")
        set_source_files_properties(${_tmp_lib_src_dir}/depthwise_conv.cc PROPERTIES COMPILE_FLAGS "-Wno-sign-compare -Wno-enum-compare")
        set_source_files_properties(${_tmp_lib_src_dir}/fully_connected.cc PROPERTIES COMPILE_FLAGS "-Wno-sign-compare -Wno-enum-compare")
        set_source_files_properties(${_tmp_lib_src_dir}/pooling.cc PROPERTIES COMPILE_FLAGS "-Wno-sign-compare -Wno-enum-compare")
        set_source_files_properties(${_tmp_lib_src_dir}/softmax.cc PROPERTIES COMPILE_FLAGS "-Wno-sign-compare -Wno-enum-compare")
    endif()

    get_filename_component(_tmp_lib_src_dir "${TFLITE_MICRO__BASE_DIR}/tensorflow/lite/micro/memory_planner" ABSOLUTE)
    file(GLOB_RECURSE _tmp_sources
        "${_tmp_lib_src_dir}/*.c"
        "${_tmp_lib_src_dir}/*.cc")
    list(FILTER _tmp_sources EXCLUDE REGEX "_test.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark_8bit.cc$")
    set(PROJECT_SOURCES ${PROJECT_SOURCES} ${_tmp_sources})

    get_filename_component(_tmp_lib_src_dir "${TFLITE_MICRO__BASE_DIR}/tensorflow/lite/micro/arena_allocator" ABSOLUTE)
    file(GLOB_RECURSE _tmp_sources
        "${_tmp_lib_src_dir}/*.c"
        "${_tmp_lib_src_dir}/*.cc")
    list(FILTER _tmp_sources EXCLUDE REGEX "_test.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark_8bit.cc$")
    set(PROJECT_SOURCES ${PROJECT_SOURCES} ${_tmp_sources})

    get_filename_component(_tmp_lib_src_dir "${TFLITE_MICRO__BASE_DIR}/tensorflow/lite/micro/tflite_bridge" ABSOLUTE)
    file(GLOB_RECURSE _tmp_sources
        "${_tmp_lib_src_dir}/*.c"
        "${_tmp_lib_src_dir}/*.cc")
    list(FILTER _tmp_sources EXCLUDE REGEX "_test.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark.cc$")
    list(FILTER _tmp_sources EXCLUDE REGEX "_benchmark_8bit.cc$")
    set(PROJECT_SOURCES ${PROJECT_SOURCES} ${_tmp_sources})

    get_filename_component(_tmp_lib_src_dir "${TFLITE_MICRO__BASE_DIR}/tensorflow/lite/micro/testing" ABSOLUTE)
    set(PROJECT_SOURCES ${PROJECT_SOURCES} ${_tmp_lib_src_dir}/test_conv_model.cc)

    get_filename_component(_tmp_src_dir "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
    include_directories(${_tmp_src_dir})
    file(GLOB_RECURSE _tmp_sources
        "${_tmp_src_dir}/*.c"
        "${_tmp_src_dir}/*.cc")
    set(PROJECT_SOURCES ${PROJECT_SOURCES} ${_tmp_sources})

    include_directories("${TFLITE_MICRO__KISSFFT_BASE_DIR}")
    include_directories("${TFLITE_MICRO__FLATBUFFERS_BASE_DIR}/include")
    include_directories("${TFLITE_MICRO__GEMMLOWP_BASE_DIR}")
    include_directories("${TFLITE_MICRO__RUY_BASE_DIR}")

    set(PROJECT_SOURCES ${PROJECT_SOURCES} "${TFLITE_MICRO__KISSFFT_BASE_DIR}/kiss_fft.c")
    set(PROJECT_SOURCES ${PROJECT_SOURCES} "${TFLITE_MICRO__KISSFFT_BASE_DIR}/tools/kiss_fftr.c")

endif(INCLUDE__TFLITE_MICRO)

