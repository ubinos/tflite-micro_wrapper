#ifndef RECORDINGUBIHEAPBUFFERALLOCATOR_H
#define RECORDINGUBIHEAPBUFFERALLOCATOR_H

#include "tensorflow/lite/micro/arena_allocator/ubi_heap_buffer_allocator.h"
#include "tensorflow/lite/micro/compatibility.h"

namespace tflite {
    /**
 * Utility class used to log allocations of a UbiHeapBufferAllocator.
 * Should only be used in debug/evaluation settings or unit tests to evaluate allocation usage.
 */
    class RecordingUbiHeapBufferAllocator : public tflite::UbiHeapBufferAllocator {

    private:
        size_t requested_head_bytes_;
        size_t requested_tail_bytes_;
        size_t used_bytes_;
        size_t alloc_count_;

    public:
        RecordingUbiHeapBufferAllocator();

        /**
         * ubinos_lang_config {"override": true}
         */
        virtual ~RecordingUbiHeapBufferAllocator() override;

        TF_LITE_REMOVE_VIRTUAL_DELETE

        /**
         * Creates a new RecordingSingleArenaBufferAllocaton
         */
        static tflite::RecordingUbiHeapBufferAllocator* Create();

        /**
         * Returns the number of bytes requested from the head or tail.
         */
        virtual size_t GetRequestedBytes() const;

        /**
         * Returns the number of bytes actually allocated from the head or tail.
         * This value will be >= to the number of requested bytes due to padding and alignment.
         * 
         * ubinos_lang_config {"override": true}
         */
        virtual size_t GetUsedBytes() const override;

        /**
         * Returns the number of alloc calls from the head or tail.
         * 
         * ubinos_lang_config {"override": true}
         */
        virtual size_t GetAllocatedCount() const override;

        /**
         * Resizes a buffer that is previously returned by the AllocateResizableBuffer.
         * In current implementation, it Adjusts the head (lowest address and moving upwards) memory allocation to a given size.
         * Calls to this method will also invalidate all temporary allocation values (it sets the location of temp space at the end of the head section).
         * This call will fail if a chain of allocations through AllocateTemp() have not been cleaned up with a call to ResetTempAllocations().
         * 
         * ubinos_lang_config {"override": true}
         */
        virtual TfLiteStatus ResizeBuffer(uint8_t* resizable_buf, size_t size, size_t alignment) override;

        /**
         * Allocates persistent memory. The persistent buffer is never freed.
         * 
         * ubinos_lang_config {"override": true}
         */
        virtual uint8_t* AllocatePersistentBuffer(size_t size, size_t alignment) override;

        /**
         * ubinos_lang_config {"override": true}
         */
        virtual TfLiteStatus DeallocatePersistentBuffer(uint8_t* buf) override;
    };
}

#endif
