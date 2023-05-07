#ifndef UBIHEAPMICROINTERPRETER_H
#define UBIHEAPMICROINTERPRETER_H

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/ubi_heap_micro_allocator.h"

namespace tflite {
    class UbiHeapMicroInterpreter : public MicroInterpreter {

    private:
        const UbiHeapMicroAllocator& ubi_heap_micro_allocator_;

    public:
        UbiHeapMicroInterpreter(const Model* model, const MicroOpResolver& op_resolver, UbiHeapMicroAllocator* allocator, MicroResourceVariables* resource_variables = nullptr, MicroProfilerInterface* profiler = nullptr);

        TF_LITE_REMOVE_VIRTUAL_DELETE

        const UbiHeapMicroAllocator& GetMicroAllocator() const;
    };
}

#endif
