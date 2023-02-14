#ifndef UBIMICROINTERPRETER_H
#define UBIMICROINTERPRETER_H

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/ubi_micro_allocator.h"

namespace tflite {
    class UbiMicroInterpreter : public tflite::MicroInterpreter {

    public:
        UbiMicroInterpreter(const Model* model, const MicroOpResolver& op_resolver, tflite::UbiMicroAllocator* allocator, MicroResourceVariables* resource_variables = nullptr, MicroProfilerInterface* profiler = nullptr);

        UbiMicroInterpreter(const Model* model, const MicroOpResolver& op_resolver, uint8_t* tensor_arena, size_t tensor_arena_size, MicroResourceVariables* resource_variables = nullptr, MicroProfilerInterface* profiler = nullptr);

        const UbiMicroAllocator& GetMicroAllocator() const;

    private:
        const UbiMicroAllocator& ubi_micro_allocator_;
    };
}

#endif
