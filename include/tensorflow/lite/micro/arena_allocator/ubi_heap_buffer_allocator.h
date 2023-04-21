#ifndef UBIHEAPBUFFERALLOCATOR_H
#define UBIHEAPBUFFERALLOCATOR_H

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/arena_allocator/ibuffer_allocator.h"
#include "tensorflow/lite/micro/compatibility.h"

namespace tflite {
    class UbiHeapBufferAllocator : public IPersistentBufferAllocator, public INonPersistentBufferAllocator {

    private:
        /**
         * The combination of the checksum of outstanding temporary buffer pointers and the count of outstanding temporary buffer provide a low cost mechanism to audit temporary buffers' allocation and deallocation.
         * XOR Check sum for outstanding temp buffers.
         * If all temp buffers are deallocated or no temp buffers are allocated, temp_buffer_ptr_check_sum_ == nullptr.
         *
         * ubinos_lang_config {"init_value": 0}
         */
        intptr_t temp_buffer_ptr_check_sum_ = 0;

        /**
         * ubinos_lang_config {"init_value": 0}
         */
        int temp_buffer_count_ = 0;

    public:
        UbiHeapBufferAllocator();

        /**
         * ubinos_lang_config {"override": true}
         */
        virtual ~UbiHeapBufferAllocator() override;

        TF_LITE_REMOVE_VIRTUAL_DELETE

        /**
         * Creates a new UbiHeapBufferAllocator.
         */
        static UbiHeapBufferAllocator* Create();

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
         * Signals that all temporary allocations can be reclaimed.
         * TFLM calls this API when it knows that all temporary buffers that it requested has been deallocated. The goal of API is to facilitate implementations of INonPersistentBufferAllocator can reuse buffer with some reasonable complexity.
         *
         * ubinos_lang_config {"override": true}
         */
        virtual TfLiteStatus ResetTempAllocations() override;

        /**
         * Reserves the size of the overlay memory.
         * This overlay is reserved for the kernels at Invoke stage.
         * This is referred to as the overlay because before Invoke state, the same memory can be used for temp buffers.
         * The layout of the memory is planned by the memory planner separately at Invoke stage.
         *
         * ubinos_lang_config {"override": true}
         */
        virtual TfLiteStatus ReserveNonPersistentOverlayMemory(size_t size, size_t alignment) override;

        /**
         * Returns true if all temporary buffers are already deallocated.
         *
         * ubinos_lang_config {"override": true}
         */
        virtual bool IsAllTempDeallocated() override;

        /**
         * Returns the number of used bytes in the allocator.
         * This number takes in account any temporary allocations.
         */
        virtual size_t GetUsedBytes() const;

        /**
         * Returns the number of max used bytes in the allocator.
         * This number takes in account any temporary allocations.
         */
        virtual size_t GetUsedBytesMax() const;

        /**
         * Returns a pointer pointing to the start of the overlay memory, which is used for activation tensors and scratch buffers by kernels at Invoke stage.
         *
         * ubinos_lang_config {"override": true}
         */
        virtual uint8_t* GetOverlayMemoryAddress() const override;

        /**
         * Returns the size of non-persistent buffer in use.
         *
         * ubinos_lang_config {"override": true}
         */
        virtual size_t GetNonPersistentUsedBytes() const override;

        /**
         * Returns the max size of non-persistent buffer in use.
         */
        virtual size_t GetNonPersistentUsedBytesMax() const;

        /**
         * Returns the size of all persistent allocations in bytes.
         *
         * ubinos_lang_config {"override": true}
         */
        virtual size_t GetPersistentUsedBytes() const override;

        /**
         * Returns the max size of all persistent allocations in bytes.
         */
        virtual size_t GetPersistentUsedBytesMax() const;

        /**
         * Returns the number of bytes available with a given alignment.
         * This number takes in account any temporary allocations.
         *
         * ubinos_lang_config {"override": true}
         */
        virtual size_t GetAvailableMemory(size_t alignment) const override;

        /**
         * Signals that a temporary buffer is no longer needed.
         *
         * ubinos_lang_config {"override": true}
         */
        virtual void DeallocateTemp(uint8_t* buf) override;

        /**
         * ubinos_lang_config {"override": true}
         */
        virtual TfLiteStatus DeallocateResizableBuffer(uint8_t* resizable_buf) override;

        /**
         * Allocates a temporary buffer. This buffer is not resizable.
         *
         * ubinos_lang_config {"override": true}
         */
        virtual uint8_t* AllocateTemp(size_t size, size_t alignment) override;

        /**
         * Returns a buffer that is resizable viable ResizeBuffer().
         * Only one resizable buffer is currently supported.
         *
         * ubinos_lang_config {"override": true}
         */
        virtual uint8_t* AllocateResizableBuffer(size_t size, size_t alignment) override;

        /**
         * Allocates persistent memory. The persistent buffer is never freed.
         *
         * ubinos_lang_config {"override": true}
         */
        virtual uint8_t* AllocatePersistentBuffer(size_t size, size_t alignment) override;
    };
}

#endif
