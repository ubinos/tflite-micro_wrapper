#include "tensorflow/lite/micro/ubi_micro_interpreter.h"

namespace tflite {

tflite::UbiMicroInterpreter::UbiMicroInterpreter(const Model* model, const MicroOpResolver& op_resolver, tflite::UbiMicroAllocator* allocator, MicroResourceVariables* resource_variables, MicroProfilerInterface* profiler)
    : MicroInterpreter(model, op_resolver, allocator, resource_variables, profiler) {
}

}  // namespace tflite
