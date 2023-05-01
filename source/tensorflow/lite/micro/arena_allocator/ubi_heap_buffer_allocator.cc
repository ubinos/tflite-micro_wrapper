#if defined(UBINOS_BSP_PRESENT)

#include <ubinos.h>
#include <ubinos/ubiclib/heap.h>

#include "tensorflow/lite/micro/arena_allocator/ubi_heap_buffer_allocator.h"

#include <cstddef>
#include <cstdint>
#include <new>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

tflite::UbiHeapBufferAllocator::UbiHeapBufferAllocator() {
  TfLiteStatus status;
  status = ResizeBuffer(nullptr, 4, 1);
  ubi_assert(status == kTfLiteOk);
  status = ResizeBuffer(nullptr, 0, 1);
  ubi_assert(status == kTfLiteOk);
}

tflite::UbiHeapBufferAllocator::~UbiHeapBufferAllocator() {
  TfLiteStatus status;
  ResetTempAllocations();
  if (resizable_buffer_size_ > 0) {
    status = ResizeBuffer(nullptr, 0, 1);
    ubi_assert(status == kTfLiteOk);
  }
}

tflite::UbiHeapBufferAllocator* tflite::UbiHeapBufferAllocator::Create() {
  UbiHeapBufferAllocator tmp = UbiHeapBufferAllocator();

  // Allocate enough bytes from the buffer to create a
  // UbiHeapBufferAllocator. The new instance will use the current adjusted
  // tail buffer from the tmp allocator instance.
  uint8_t* allocator_buffer = tmp.AllocatePersistentBuffer(
      sizeof(UbiHeapBufferAllocator), alignof(UbiHeapBufferAllocator));
  // Use the default copy constructor to populate internal states.
  return new (allocator_buffer) UbiHeapBufferAllocator(tmp);
}

TfLiteStatus tflite::UbiHeapBufferAllocator::ResizeBuffer(uint8_t* resizable_buf, size_t size, size_t alignment) {
  TfLiteStatus status;
  uint8_t* result;
  int r;

  do
  {
    if (resizable_buf != nullptr && resizable_buf != resizable_buffer_) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
      MicroPrintf(
          "Failed to resize resizable memory (wrong address). Requested: %u, %u, %u",
          resizable_buf, size, alignment);
#endif
      status = kTfLiteError;
      break;
    }

    if (!IsAllTempDeallocated()) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
      MicroPrintf(
          "Failed to resize resizable memory (temporary buffer allocated). Requested: %u, %u, %u",
          resizable_buf, size, alignment);
#endif
      status = kTfLiteError;
      break;
    }

    if (size == 0) {
      if (resizable_buffer_size_ > 0) {
        r = heap_free(NULL, resizable_buffer_);
        ubi_assert(r == 0);

        resizable_buffer_size_ = 0;
      }
    }
    else
    {
      if (resizable_buffer_size_ > 0) {
        result = reinterpret_cast<uint8_t*>(heap_resize(NULL, resizable_buffer_, size)); // Use normal direction heap
        if (result == nullptr) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
          MicroPrintf(
              "Failed to resize resizable memory (resize fail). Requested: %u, %u",
              size, alignment);
#endif
          status = kTfLiteError;
          break;
        }
      }
      else {
        result = reinterpret_cast<uint8_t*>(heap_malloc(NULL, size, 0)); // Use normal direction heap
        if (result == nullptr) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
          MicroPrintf(
              "Failed to resize resizable memory (allocation fail). Requested: %u, %u",
              size, alignment);
#endif
          status = kTfLiteError;
          break;
        }
      }

      if (result != AlignPointerUp(result, alignment)) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
        MicroPrintf(
            "Failed to resize resizable memory (alignment fail). Requested: %u, %u",
            size, alignment);
#endif
        r = heap_free(NULL, result);
        ubi_assert(r == 0);
        status = kTfLiteError;
        break;
      }

      resizable_buffer_ = result;
      resizable_buffer_size_ = size;
    }

    status = kTfLiteOk;
    break;
  } while (1);

  return status;
}

TfLiteStatus tflite::UbiHeapBufferAllocator::ResetTempAllocations() {
  int r;
  void * first_ptr = heap_get_first_allocated_block(NULL, 0);
  if (first_ptr != NULL) {
    void * next_ptr = heap_get_next_allocated_block(NULL, first_ptr);
    while (next_ptr != NULL)
    {
      r = heap_free(NULL, next_ptr);
      ubi_assert(r == 0);
      next_ptr = heap_get_next_allocated_block(NULL, first_ptr);
    }
    if (resizable_buffer_size_ == 0) {
      r = heap_free(NULL, first_ptr);
      ubi_assert(r == 0);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus tflite::UbiHeapBufferAllocator::ReserveNonPersistentOverlayMemory(size_t size, size_t alignment) {
  return ResizeBuffer(resizable_buffer_, size, alignment);
}

bool tflite::UbiHeapBufferAllocator::IsAllTempDeallocated() {
  unsigned int count;
  unsigned int ncount;
  unsigned int rcount;
  int r;

  r = heap_getallocatedcount_ext(NULL, &count, &ncount, &rcount);
  ubi_assert(r == 0);

  if ((resizable_buffer_size_ == 0 && ncount == 0) || (resizable_buffer_size_ > 0 && ncount == 1)) {
    return true;
  }
  else {
    return false;
  }
}

size_t tflite::UbiHeapBufferAllocator::GetUsedBytes() const {
  return GetPersistentUsedBytes() + GetNonPersistentUsedBytes();
}

size_t tflite::UbiHeapBufferAllocator::GetUsedBytesMax() const {
  return GetPersistentUsedBytesMax() + GetNonPersistentUsedBytesMax();
}

uint8_t* tflite::UbiHeapBufferAllocator::GetOverlayMemoryAddress() const {
  return resizable_buffer_;
}

size_t tflite::UbiHeapBufferAllocator::GetNonPersistentUsedBytes() const {
  int r;
  unsigned int size;
  unsigned int nsize;
  unsigned int rsize;

  r = heap_getrequestedsize_ext(NULL, &size, &nsize, &rsize);
  ubi_assert(r == 0);

  return reinterpret_cast<size_t>(nsize);
}

size_t tflite::UbiHeapBufferAllocator::GetNonPersistentUsedBytesMax() const {
  int r;
  unsigned int size;
  unsigned int nsize;
  unsigned int rsize;

  r = heap_getrequestedsizemax_ext(NULL, &size, &nsize, &rsize);
  ubi_assert(r == 0);

  return reinterpret_cast<size_t>(nsize);
}

size_t tflite::UbiHeapBufferAllocator::GetPersistentUsedBytes() const {
  int r;
  unsigned int size;
  unsigned int nsize;
  unsigned int rsize;

  r = heap_getrequestedsize_ext(NULL, &size, &nsize, &rsize);
  ubi_assert(r == 0);

  return reinterpret_cast<size_t>(rsize);
}

size_t tflite::UbiHeapBufferAllocator::GetPersistentUsedBytesMax() const {
  int r;
  unsigned int size;
  unsigned int nsize;
  unsigned int rsize;

  r = heap_getrequestedsizemax_ext(NULL, &size, &nsize, &rsize);
  ubi_assert(r == 0);

  return reinterpret_cast<size_t>(rsize);
}

size_t tflite::UbiHeapBufferAllocator::GetAvailableMemory(size_t alignment) const {
  unsigned int size;
  int r;

  r = heap_getexpandablesize(NULL, &size);
  ubi_assert(r == 0);

  return reinterpret_cast<size_t>(size);
}

void tflite::UbiHeapBufferAllocator::DeallocateTemp(uint8_t* buf) {
  int r;

  r = heap_free(NULL, buf);
  ubi_assert(r == 0);
}

TfLiteStatus tflite::UbiHeapBufferAllocator::DeallocateResizableBuffer(uint8_t* resizable_buf) {
  return ResizeBuffer(resizable_buf, 0, 1);
}

uint8_t* tflite::UbiHeapBufferAllocator::AllocateTemp(size_t size, size_t alignment) {
  int r;
  uint8_t* result = reinterpret_cast<uint8_t*>(heap_malloc(NULL, size, 0)); // Use normal direction heap
  if (result == nullptr) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
    MicroPrintf(
        "Failed to allocate temporary memory (allocation fail). Requested: %u, %u",
        size, alignment);
#endif
    return nullptr;
  }

  if (result != AlignPointerUp(result, alignment)) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
    MicroPrintf(
        "Failed to align temporary memory (alignment fail). Requested: %u, %u",
        size, alignment);
#endif
    r = heap_free(NULL, result);
    ubi_assert(r == 0);
    return nullptr;
  }

  return result;
}

uint8_t* tflite::UbiHeapBufferAllocator::AllocateResizableBuffer(size_t size, size_t alignment) {
  if (ResizeBuffer(resizable_buffer_, size, alignment) == kTfLiteOk) {
    return resizable_buffer_;
  }
  return nullptr;
}

uint8_t* tflite::UbiHeapBufferAllocator::AllocatePersistentBuffer(size_t size, size_t alignment) {
  int r;
  uint8_t* result = reinterpret_cast<uint8_t*>(heap_malloc(NULL, size, 1)); // Use reverse direction heap
  if (result == nullptr) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
    MicroPrintf(
        "Failed to allocate persistent memory (allocation fail). Requested: %u, %u",
        size, alignment);
#endif
    return nullptr;
  }

  if (result != AlignPointerDown(result, alignment)) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
    MicroPrintf(
        "Failed to allocate persistent memory (alignment fail). Requested: %u, %u",
        size, alignment);
#endif
    r = heap_free(NULL, result);
    ubi_assert(r == 0);
    return nullptr;
  }

  return result;
}

}  // namespace tflite

#endif /* defined(UBINOS_BSP_PRESENT) */
