#ifndef UBIMICROINTERPRETER_H
#define UBIMICROINTERPRETER_H

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/ubi_micro_allocator.h"

namespace tflite {
    class UbiMicroInterpreter : public MicroInterpreter {

    private:
        const UbiMicroAllocator& ubi_micro_allocator_;

    public:
        UbiMicroInterpreter(const Model* model, const MicroOpResolver& op_resolver, UbiMicroAllocator* allocator, MicroResourceVariables* resource_variables = nullptr, MicroProfilerInterface* profiler = nullptr);

        UbiMicroInterpreter(const Model* model, const MicroOpResolver& op_resolver, uint8_t* tensor_arena, size_t tensor_arena_size, MicroResourceVariables* resource_variables = nullptr, MicroProfilerInterface* profiler = nullptr);

        TF_LITE_REMOVE_VIRTUAL_DELETE

        const UbiMicroAllocator& GetMicroAllocator() const;
    };
}

#endif
