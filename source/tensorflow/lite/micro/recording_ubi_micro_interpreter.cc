#include "tensorflow/lite/micro/recording_ubi_micro_interpreter.h"

namespace tflite {

tflite::RecordingUbiMicroInterpreter::RecordingUbiMicroInterpreter(const Model* model, const MicroOpResolver& op_resolver, tflite::RecordingUbiMicroAllocator* allocator, MicroResourceVariables* resource_variables, MicroProfilerInterface* profiler)
      : UbiMicroInterpreter(model, op_resolver, allocator, resource_variables, profiler),
        recording_ubi_micro_allocator_(*allocator)
{
}

tflite::RecordingUbiMicroInterpreter::RecordingUbiMicroInterpreter(const Model* model, const MicroOpResolver& op_resolver, uint8_t* tensor_arena, size_t tensor_arena_size, MicroResourceVariables* resource_variables, MicroProfilerInterface* profiler)
      : UbiMicroInterpreter(model, op_resolver,
            RecordingUbiMicroAllocator::Create(tensor_arena, tensor_arena_size),
            resource_variables, profiler),
        recording_ubi_micro_allocator_(static_cast<const RecordingUbiMicroAllocator&>(allocator()))
{
}

const RecordingUbiMicroAllocator& tflite::RecordingUbiMicroInterpreter::GetMicroAllocator() const
{
    // TODO - implement RecordingUbiMicroInterpreter.GetMicroAllocator
    return recording_ubi_micro_allocator_;
}

}  // namespace tflite
