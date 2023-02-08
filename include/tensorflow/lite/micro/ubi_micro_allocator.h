#ifndef UBIMICROALLOCATOR_H
#define UBIMICROALLOCATOR_H

#include "tensorflow/lite/micro/arena_allocator/ubi_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_allocator.h"

namespace tflite {
    class UbiMicroAllocator : public tflite::MicroAllocator {

    public:
        /**
         * Creates a MicroAllocator instance using the provided SingleArenaBufferAllocator instance and the MemoryPlanner.
         * This allocator instance will use the SingleArenaBufferAllocator instance to manage allocations internally.
         */
        static UbiMicroAllocator* Create(UbiArenaBufferAllocator* memory_allocator, MicroMemoryPlanner* memory_planner);

        static UbiMicroAllocator* Create(UbiArenaBufferAllocator* memory_allocator);

        static UbiMicroAllocator* Create(uint8_t* tensor_arena, size_t arena_size);

        static UbiMicroAllocator* Create(uint8_t* tensor_arena, size_t arena_size, MicroMemoryPlanner* memory_planner);

        /**
         * Returns the fixed amount of memory overhead of UbiMicroAllocator.
         */
        static size_t GetDefaultTailUsage(bool is_memory_planner_given);

    protected:
        UbiMicroAllocator(IPersistentBufferAllocator* persistent_buffer_allocator, INonPersistentBufferAllocator* non_persistent_buffer_allocator, MicroMemoryPlanner* memory_planner);

        TF_LITE_REMOVE_VIRTUAL_DELETE
    };
}

#endif
