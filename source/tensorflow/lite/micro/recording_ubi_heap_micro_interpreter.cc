#include "tensorflow/lite/micro/recording_ubi_heap_micro_interpreter.h"

namespace tflite {

tflite::RecordingUbiHeapMicroInterpreter::RecordingUbiHeapMicroInterpreter(const Model* model, const MicroOpResolver& op_resolver, tflite::RecordingUbiHeapMicroAllocator* allocator, MicroResourceVariables* resource_variables, MicroProfilerInterface* profiler)
      : UbiHeapMicroInterpreter(model, op_resolver, allocator, resource_variables, profiler),
        recording_ubi_heap_micro_allocator_(*allocator)
{
}

const RecordingUbiHeapMicroAllocator& tflite::RecordingUbiHeapMicroInterpreter::GetMicroAllocator() const
{
    // TODO - implement RecordingUbiHeapMicroInterpreter.GetMicroAllocator
    return recording_ubi_heap_micro_allocator_;
}

}  // namespace tflite
