#include "tensorflow/lite/micro/ubi_micro_interpreter.h"

namespace tflite {

tflite::UbiMicroInterpreter::UbiMicroInterpreter(const Model* model, const MicroOpResolver& op_resolver, tflite::UbiMicroAllocator* allocator, MicroResourceVariables* resource_variables, MicroProfilerInterface* profiler)
      : MicroInterpreter(model, op_resolver, allocator, resource_variables, profiler),
        ubi_micro_allocator_(*allocator)
{
}

tflite::UbiMicroInterpreter::UbiMicroInterpreter(const Model* model, const MicroOpResolver& op_resolver, uint8_t* tensor_arena, size_t tensor_arena_size, MicroResourceVariables* resource_variables, MicroProfilerInterface* profiler)
      : MicroInterpreter(model, op_resolver,
            UbiMicroAllocator::Create(tensor_arena, tensor_arena_size),
            resource_variables, profiler),
        ubi_micro_allocator_(
            static_cast<const UbiMicroAllocator&>(allocator()))
{
}

const UbiMicroAllocator& tflite::UbiMicroInterpreter::GetMicroAllocator() const {
    return ubi_micro_allocator_;
}

}  // namespace tflite
