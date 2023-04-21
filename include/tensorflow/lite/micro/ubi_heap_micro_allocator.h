#ifndef UBIHEAPMICROALLOCATOR_H
#define UBIHEAPMICROALLOCATOR_H

#include "tensorflow/lite/micro/arena_allocator/ubi_heap_buffer_allocator.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_allocator.h"

namespace tflite {
    class UbiHeapMicroAllocator : public MicroAllocator {


    public:
        /**
         * Creates a UbiHeapMicroAllocator instance using the provided UbiHeapBufferAllocator instance and the MemoryPlanner.
         * This allocator instance will use the UbiHeapBufferAllocator instance to manage allocations internally.
         */
        static UbiHeapMicroAllocator* Create(UbiHeapBufferAllocator* memory_allocator, MicroMemoryPlanner* memory_planner);

        static UbiHeapMicroAllocator* Create(UbiHeapBufferAllocator* memory_allocator);

        /**
         * Returns the fixed amount of memory overhead of UbiMicroAllocator.
         */
        static size_t GetDefaultTailUsage(bool is_memory_planner_given);

    protected:
        UbiHeapMicroAllocator(IPersistentBufferAllocator* persistent_buffer_allocator, INonPersistentBufferAllocator* non_persistent_buffer_allocator, MicroMemoryPlanner* memory_planner;

        UbiHeapMicroAllocator(UbiHeapBufferAllocator* memory_allocator, MicroMemoryPlanner* memory_planner);

        TF_LITE_REMOVE_VIRTUAL_DELETE
    };
}

#endif
