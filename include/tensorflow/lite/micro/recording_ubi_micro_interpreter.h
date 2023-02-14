#ifndef RECORDINGUBIMICROINTERPRETER_H
#define RECORDINGUBIMICROINTERPRETER_H

#include "tensorflow/lite/micro/ubi_micro_interpreter.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/recording_ubi_micro_allocator.h"

namespace tflite {
    class RecordingUbiMicroInterpreter : public tflite::UbiMicroInterpreter {

    public:
        RecordingUbiMicroInterpreter(const Model* model, const MicroOpResolver& op_resolver, tflite::RecordingUbiMicroAllocator* allocator, MicroResourceVariables* resource_variables = nullptr, MicroProfilerInterface* profiler = nullptr);

        RecordingUbiMicroInterpreter(const Model* model, const MicroOpResolver& op_resolver, uint8_t* tensor_arena, size_t tensor_arena_size, MicroResourceVariables* resource_variables = nullptr, MicroProfilerInterface* profiler = nullptr);

        const RecordingUbiMicroAllocator& GetMicroAllocator() const;

    private:
        const RecordingUbiMicroAllocator& recording_ubi_micro_allocator_;
    };
}

#endif
