#ifndef UBIHEAPMICROALLOCATOR_H
#define UBIHEAPMICROALLOCATOR_H

#include "tensorflow/lite/micro/arena_allocator/ubi_heap_buffer_allocator.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"

namespace tflite {
    class UbiHeapMicroAllocator : public MicroAllocator {


    public:
        UbiHeapMicroAllocator(IPersistentBufferAllocator* persistent_buffer_allocator, INonPersistentBufferAllocator* non_persistent_buffer_allocator, MicroMemoryPlanner* memory_planner);

        UbiHeapMicroAllocator(UbiHeapBufferAllocator* memory_allocator, MicroMemoryPlanner* memory_planner);

        UbiHeapMicroAllocator(UbiHeapBufferAllocator* memory_allocator);

        /**
         * ubinos_lang_config {"override": true}
         */
        virtual ~UbiHeapMicroAllocator() override;

        TF_LITE_REMOVE_VIRTUAL_DELETE

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

        /**
         * Returns a pointer pointing to the start of the overlay memory, which is used for activation tensors and scratch buffers by kernels at Invoke stage.
         */
        virtual uint8_t* GetOverlayMemoryAddress() const;

    private:
        GreedyMemoryPlanner default_memory_planner_;

        /**
         * ubinos_lang_config {"init_value": nullptr}
         */
        INonPersistentBufferAllocator* non_persistent_heap_buffer_allocator_ = nullptr;
    };
}

#endif
