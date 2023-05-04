#if defined(UBINOS_BSP_PRESENT)

#include <ubinos_config.h>

#if (INCLUDE__UBINOS__UBICLIB == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP_FLAG == 1)

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
#include "tensorflow/lite/micro/micro_arena_constants.h"

#define TFL_HEAP_DIR_PERSISTENT 1
#define TFL_HEAP_DIR_NONPERSISTENT 0

namespace tflite {

tflite::UbiHeapBufferAllocator::UbiHeapBufferAllocator() {
  TfLiteStatus status;
  status = ResizeBuffer(nullptr, 4, 1);
  ubi_assert(status == kTfLiteOk);
  status = ResizeBuffer(nullptr, 0, 1);
  ubi_assert(status == kTfLiteOk);
}

tflite::UbiHeapBufferAllocator::~UbiHeapBufferAllocator() {
  ResetTempAllocations();
  ResetPersistentAllocations();
  ResizeBuffer(nullptr, 0, 1);
}

tflite::UbiHeapBufferAllocator* tflite::UbiHeapBufferAllocator::Create() {
  return new UbiHeapBufferAllocator();
}

size_t tflite::UbiHeapBufferAllocator::GetUsedBytes() const {
  return GetPersistentUsedBytes() + GetNonPersistentUsedBytes();
}

size_t tflite::UbiHeapBufferAllocator::GetUsedBytesMax() const {
  return GetPersistentUsedBytesMax() + GetNonPersistentUsedBytesMax();
}

size_t tflite::UbiHeapBufferAllocator::GetNonPersistentUsedBytes() const {
  return resizable_buffer_size_ + temp_buffer_size_;
}

size_t tflite::UbiHeapBufferAllocator::GetNonPersistentUsedBytesMax() const {
  return nonpersistent_buffer_size_max_;
}

size_t tflite::UbiHeapBufferAllocator::GetPersistentUsedBytes() const {
  return persistent_buffer_size_;
}

size_t tflite::UbiHeapBufferAllocator::GetPersistentUsedBytesMax() const {
  return persistent_buffer_size_max_;
}

size_t tflite::UbiHeapBufferAllocator::GetAvailableMemory(size_t alignment) const {
  unsigned int size;
  int r;

  r = heap_getexpandablesize(NULL, &size);
  ubi_assert(r == 0);

  return reinterpret_cast<size_t>(size - heap_get_block_overhead(NULL) - MicroArenaBufferAlignment());
}

uint8_t* tflite::UbiHeapBufferAllocator::AllocateResizableBuffer(size_t size, size_t alignment) {
  if (ResizeBuffer(resizable_buffer_, size, alignment) == kTfLiteOk) {
    return resizable_buffer_;
  }
  return nullptr;
}

TfLiteStatus tflite::UbiHeapBufferAllocator::DeallocateResizableBuffer(uint8_t* resizable_buf) {
  return ResizeBuffer(resizable_buf, 0, 1);
}

TfLiteStatus tflite::UbiHeapBufferAllocator::ReserveNonPersistentOverlayMemory(size_t size, size_t alignment) {
  return ResizeBuffer(resizable_buffer_, size, alignment);
}

uint8_t* tflite::UbiHeapBufferAllocator::GetOverlayMemoryAddress() const {
  return resizable_buffer_;
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

      r = heap_set_flag(NULL, result, HEAP_FLAG_NO__TFL_RESIZABLE, 1);
      ubi_assert(r == 0);

      resizable_buffer_ = result;
      resizable_buffer_size_ = size;
      if (nonpersistent_buffer_size_max_ < (resizable_buffer_size_ + temp_buffer_size_)) {
        nonpersistent_buffer_size_max_ = resizable_buffer_size_ + temp_buffer_size_;
      }
    }

    status = kTfLiteOk;
    break;
  } while (1);

  return status;
}

uint8_t* tflite::UbiHeapBufferAllocator::AllocateTemp(size_t size, size_t alignment) {
  int r;
  uint8_t* result;
  unsigned int block_size;

  do
  {
    result = reinterpret_cast<uint8_t*>(heap_malloc(NULL, size, TFL_HEAP_DIR_NONPERSISTENT));
    if (result == nullptr) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
      MicroPrintf(
          "Failed to allocate persistent memory (allocation fail). Requested: %u, %u",
          size, alignment);
#endif
      break;
    }

    if (result != AlignPointerDown(result, alignment)) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
      MicroPrintf(
          "Failed to allocate persistent memory (alignment fail). Requested: %u, %u",
          size, alignment);
  #endif
      r = heap_free(NULL, result);
      ubi_assert(r == 0);
      result = nullptr;
      break;
    }

    r = heap_set_flag(NULL, result, HEAP_FLAG_NO__TFL_TEMP, 1);
    ubi_assert(r == 0);

    r = heap_getblocksize(NULL, result, &block_size);
    ubi_assert(r == 0);

    temp_buffer_size_ += block_size;
    if (nonpersistent_buffer_size_max_ < (resizable_buffer_size_ + temp_buffer_size_)) {
      nonpersistent_buffer_size_max_ = resizable_buffer_size_ + temp_buffer_size_;
    }

    break;
  } while(1);

  return result;
}

void tflite::UbiHeapBufferAllocator::DeallocateTemp(uint8_t* buf) {
  int r;
  unsigned int block_size;

  r = heap_get_flag(NULL, buf, HEAP_FLAG_NO__TFL_TEMP);
  ubi_assert(r == 1);

  r = heap_getblocksize(NULL, buf, &block_size);
  ubi_assert(r == 0);

  r = heap_free(NULL, buf);
  ubi_assert(r == 0);

  temp_buffer_size_ -= block_size;
}

TfLiteStatus tflite::UbiHeapBufferAllocator::ResetTempAllocations() {
  int r;
  void * cur_block;
  void * prev_block;
  unsigned int block_size;
  TfLiteStatus status;

  status = kTfLiteError;
  cur_block = heap_get_last_allocated_block(NULL, TFL_HEAP_DIR_NONPERSISTENT);
  do
  {
    if (cur_block == NULL) {
      status = kTfLiteOk;
      break;
    }

    prev_block = heap_get_prev_allocated_block(NULL, cur_block);

    r = heap_get_flag(NULL, cur_block, HEAP_FLAG_NO__TFL_TEMP);
    if (r == 1) {
      r = heap_getblocksize(NULL, cur_block, &block_size);
      ubi_assert(r == 0);

      r = heap_free(NULL, cur_block);
      ubi_assert(r == 0);

      temp_buffer_size_ -= block_size;
    }

    cur_block = prev_block;
  } while (1);

  return status;
}

bool tflite::UbiHeapBufferAllocator::IsAllTempDeallocated() {
  if (temp_buffer_size_ == 0) {
    return true;
  }
  else {
    return false;
  }
}

uint8_t* tflite::UbiHeapBufferAllocator::AllocatePersistentBuffer(size_t size, size_t alignment) {
  int r;
  uint8_t* result;
  unsigned int block_size;

  do
  {
    result = reinterpret_cast<uint8_t*>(heap_malloc(NULL, size, TFL_HEAP_DIR_PERSISTENT));
    if (result == nullptr) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
      MicroPrintf(
          "Failed to allocate persistent memory (allocation fail). Requested: %u, %u",
          size, alignment);
#endif
      break;
    }

    if (result != AlignPointerDown(result, alignment)) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
      MicroPrintf(
          "Failed to allocate persistent memory (alignment fail). Requested: %u, %u",
          size, alignment);
  #endif
      r = heap_free(NULL, result);
      ubi_assert(r == 0);
      result = nullptr;
      break;
    }

    r = heap_set_flag(NULL, result, HEAP_FLAG_NO__TFL_PERSISTENT, 1);
    ubi_assert(r == 0);

    r = heap_getblocksize(NULL, result, &block_size);
    ubi_assert(r == 0);

    persistent_buffer_size_ += block_size;
    if (persistent_buffer_size_max_ < persistent_buffer_size_) {
      persistent_buffer_size_max_ = persistent_buffer_size_;
    }

    break;
  } while(1);

  return result;
}

TfLiteStatus tflite::UbiHeapBufferAllocator::ResetPersistentAllocations() {
  int r;
  void * cur_block;
  void * prev_block;
  unsigned int block_size;
  TfLiteStatus status;

  status = kTfLiteError;
  cur_block = heap_get_last_allocated_block(NULL, TFL_HEAP_DIR_PERSISTENT);
  do
  {
    if (cur_block == NULL) {
      status = kTfLiteOk;
      break;
    }

    prev_block = heap_get_prev_allocated_block(NULL, cur_block);

    r = heap_get_flag(NULL, cur_block, HEAP_FLAG_NO__TFL_PERSISTENT);
    if (r == 1) {
      r = heap_getblocksize(NULL, cur_block, &block_size);
      ubi_assert(r == 0);

      r = heap_free(NULL, cur_block);
      ubi_assert(r == 0);

      persistent_buffer_size_ -= block_size;
    }

    cur_block = prev_block;
  } while (1);

  return status;
}

}  // namespace tflite

#endif /* (INCLUDE__UBINOS__UBICLIB == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP_FLAG == 1) */

#endif /* defined(UBINOS_BSP_PRESENT) */
