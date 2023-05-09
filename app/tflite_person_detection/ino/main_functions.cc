/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ubinos_config.h>

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "models/person_detect_model_data.h"

#if   (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__MICRO)
#include "tensorflow/lite/micro/micro_interpreter.h"
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_MICRO)
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__UBI_MICRO)
#include "tensorflow/lite/micro/ubi_micro_interpreter.h"
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_MICRO)
#include "tensorflow/lite/micro/recording_ubi_micro_interpreter.h"
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__UBI_HEAP_MICRO)
#include "tensorflow/lite/micro/ubi_heap_micro_interpreter.h"
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_HEAP_MICRO)
#include "tensorflow/lite/micro/recording_ubi_heap_micro_interpreter.h"
#else
#error "Unsupported TFLITE_MICRO__INTERPRETER_TYPE"
#endif

#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
#if   (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__MICRO)
tflite::MicroInterpreter* interpreter = nullptr;
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_MICRO)
tflite::RecordingMicroInterpreter* interpreter = nullptr;
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__UBI_MICRO)
tflite::UbiMicroInterpreter* interpreter = nullptr;
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_MICRO)
tflite::RecordingUbiMicroInterpreter* interpreter = nullptr;
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__UBI_HEAP_MICRO)
tflite::UbiHeapMicroInterpreter* interpreter = nullptr;
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_HEAP_MICRO)
tflite::RecordingUbiHeapMicroInterpreter* interpreter = nullptr;
#else
#error "Unsupported TFLITE_MICRO__INTERPRETER_TYPE"
#endif
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

#if (    !(TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__UBI_HEAP_MICRO) \
      && !(TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_HEAP_MICRO)    )
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 136 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
#endif

#if   (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_MICRO)
const tflite::RecordingMicroAllocator* allocator = nullptr;
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_MICRO)
const tflite::RecordingUbiMicroAllocator* allocator = nullptr;
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_HEAP_MICRO)
const tflite::RecordingUbiHeapMicroAllocator* allocator = nullptr;
#endif

#if (    (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_MICRO) \
      || (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_MICRO) \
      || (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_HEAP_MICRO)    )
int inference_count_static = 0;
#endif
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D_INT8());
  micro_op_resolver.AddConv2D(tflite::Register_CONV_2D_INT8());
  micro_op_resolver.AddDepthwiseConv2D(
      tflite::Register_DEPTHWISE_CONV_2D_INT8());
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax(tflite::Register_SOFTMAX_INT8());

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
#if   (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__MICRO)
  static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_MICRO)
  static tflite::RecordingMicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__UBI_MICRO)
  static tflite::UbiMicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_MICRO)
  static tflite::RecordingUbiMicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__UBI_HEAP_MICRO)
  static tflite::UbiHeapBufferAllocator static_buffer_allocator;
  static tflite::UbiHeapMicroAllocator static_allocator(&static_buffer_allocator);
  static tflite::UbiHeapMicroInterpreter static_interpreter(model, micro_op_resolver, &static_allocator);
#elif (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_HEAP_MICRO)
  static tflite::UbiHeapBufferAllocator static_buffer_allocator;
  static tflite::RecordingUbiHeapMicroAllocator static_allocator(&static_buffer_allocator);
  static tflite::RecordingUbiHeapMicroInterpreter static_interpreter(model, micro_op_resolver, &static_allocator);
#else
#error "Unsupported TFLITE_MICRO__INTERPRETER_TYPE"
#endif
      
  interpreter = &static_interpreter;

#if (    (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_MICRO) \
      || (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_MICRO) \
      || (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_HEAP_MICRO)    )
  allocator = &(interpreter->GetMicroAllocator());

  MicroPrintf("After create interpreter");
  allocator->PrintAllocations();
#endif

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

#if (    (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_MICRO) \
      || (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_MICRO) \
      || (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_HEAP_MICRO)    )
  MicroPrintf("After allocate tensors");
  allocator->PrintAllocations();
#endif

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Get image from provider.
  if (kTfLiteOk !=
      GetImage(kNumCols, kNumRows, kNumChannels, input->data.int8)) {
    MicroPrintf("Image capture failed.");
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

#if (    (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_MICRO) \
      || (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_MICRO) \
      || (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_HEAP_MICRO)    )
  if (inference_count_static <= 0)
  {
    MicroPrintf("After invoke (%d)", inference_count_static);
    allocator->PrintAllocations();
  }
#endif

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t person_score = output->data.uint8[kPersonIndex];
  int8_t no_person_score = output->data.uint8[kNotAPersonIndex];
  RespondToDetection(person_score, no_person_score);

#if (    (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_MICRO) \
      || (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_MICRO) \
      || (TFLITE_MICRO__INTERPRETER_TYPE == TFLITE_MICRO__INTERPRETER_TYPE__RECORDING_UBI_HEAP_MICRO)    )
  inference_count_static += 1;
#endif
}
