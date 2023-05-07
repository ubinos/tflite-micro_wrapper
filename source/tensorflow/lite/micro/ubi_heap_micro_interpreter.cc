#include "tensorflow/lite/micro/ubi_heap_micro_interpreter.h"

namespace tflite {

tflite::UbiHeapMicroInterpreter::UbiHeapMicroInterpreter(const Model* model, const MicroOpResolver& op_resolver, tflite::UbiHeapMicroAllocator* allocator, MicroResourceVariables* resource_variables, MicroProfilerInterface* profiler)
      : MicroInterpreter(model, op_resolver, allocator, resource_variables, profiler),
        ubi_heap_micro_allocator_(*allocator)
{
}

const UbiHeapMicroAllocator& tflite::UbiHeapMicroInterpreter::GetMicroAllocator() const {
    return ubi_heap_micro_allocator_;
}

}  // namespace tflite
