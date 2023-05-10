#ifndef RECORDINGUBIHEAPMICROINTERPRETER_H
#define RECORDINGUBIHEAPMICROINTERPRETER_H

#include "tensorflow/lite/micro/ubi_heap_micro_interpreter.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/recording_ubi_heap_micro_allocator.h"

namespace tflite {
    class RecordingUbiHeapMicroInterpreter : public tflite::UbiHeapMicroInterpreter {

    private:
        RecordingUbiHeapMicroAllocator& recording_ubi_heap_micro_allocator_;

    public:
        RecordingUbiHeapMicroInterpreter(const Model* model, const MicroOpResolver& op_resolver, tflite::RecordingUbiHeapMicroAllocator* allocator, MicroResourceVariables* resource_variables = nullptr, MicroProfilerInterface* profiler = nullptr);

        TF_LITE_REMOVE_VIRTUAL_DELETE

        RecordingUbiHeapMicroAllocator& GetMicroAllocator() const;
    };
}

#endif
