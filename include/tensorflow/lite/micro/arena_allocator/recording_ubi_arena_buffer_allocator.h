#ifndef RECORDINGUBIARENABUFFERALLOCATOR_H
#define RECORDINGUBIARENABUFFERALLOCATOR_H

#include "tensorflow/lite/micro/arena_allocator/ubi_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/compatibility.h"

namespace tflite {
    class RecordingUbiArenaBufferAllocator : public tflite::UbiArenaBufferAllocator {

    private:
        size_t requested_head_bytes_;
        size_t requested_tail_bytes_;
        size_t used_bytes_;
        size_t alloc_count_;

    public:
        RecordingUbiArenaBufferAllocator(uint8_t* buffer_head, size_t buffer_size);

        /**
         * ubinos_lang_config {"override": true}
         */
        virtual ~RecordingUbiArenaBufferAllocator() override;

        /**
         * Creates a new RecordingSingleArenaBufferAllocaton from a given buffer head and size.
         */
        static tflite::RecordingUbiArenaBufferAllocator* Create(uint8_t* buffer_head, size_t buffer_size);

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
         */
        virtual size_t GetAllocatedCount() const;

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
    };
}

#endif
