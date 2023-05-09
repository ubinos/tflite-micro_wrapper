#if defined(UBINOS_BSP_PRESENT)

#include <ubinos_config.h>

#if (INCLUDE__UBINOS__UBICLIB == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP_FLAG == 1)

#include <ubinos.h>
#include <ubinos/ubiclib/heap.h>

#include "tensorflow/lite/micro/arena_allocator/recording_ubi_heap_buffer_allocator.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

#include <new>

namespace tflite {

tflite::RecordingUbiHeapBufferAllocator::RecordingUbiHeapBufferAllocator()
    : requested_head_bytes_(0),
      requested_tail_bytes_(0),
      used_bytes_(0),
      alloc_count_(0) {}

tflite::RecordingUbiHeapBufferAllocator::~RecordingUbiHeapBufferAllocator() {}

tflite::RecordingUbiHeapBufferAllocator* tflite::RecordingUbiHeapBufferAllocator::Create() {
  return new RecordingUbiHeapBufferAllocator();
}

size_t tflite::RecordingUbiHeapBufferAllocator::GetRequestedBytes() const {
  return requested_head_bytes_ + requested_tail_bytes_;
}

size_t tflite::RecordingUbiHeapBufferAllocator::GetUsedBytes() const {
  return used_bytes_;
}

size_t tflite::RecordingUbiHeapBufferAllocator::GetAllocatedCount() const {
  return alloc_count_;
}

TfLiteStatus tflite::RecordingUbiHeapBufferAllocator::ResizeBuffer(uint8_t* resizable_buf, size_t size, size_t alignment) {
  size_t previous_size = GetNonPersistentUsedBytes();
  TfLiteStatus status = UbiHeapBufferAllocator::ResizeBuffer(resizable_buf, size, alignment);
  if (status == kTfLiteOk) {
    size_t current_size = GetNonPersistentUsedBytes();
    current_size > previous_size ? used_bytes_ += (current_size - previous_size) : used_bytes_ -= (previous_size - current_size);
    requested_head_bytes_ = size;
  }
  return status;
}

uint8_t* tflite::RecordingUbiHeapBufferAllocator::AllocatePersistentBuffer(size_t size, size_t alignment) {
  size_t previous_size = GetPersistentUsedBytes();
  uint8_t* result = UbiHeapBufferAllocator::AllocatePersistentBuffer(size, alignment);
  if (result != nullptr) {
    size_t current_size = GetPersistentUsedBytes();
    used_bytes_ += (current_size - previous_size);
    requested_tail_bytes_ += size;
    alloc_count_++;
  }
  return result;
}

TfLiteStatus tflite::RecordingUbiHeapBufferAllocator::DeallocatePersistentBuffer(uint8_t* buf) {
  size_t previous_size = GetPersistentUsedBytes();
  size_t previous_requested_size = GetRequestedBytes();
  TfLiteStatus status = UbiHeapBufferAllocator::DeallocatePersistentBuffer(buf);
  if (status == kTfLiteOk) {
    size_t current_size = GetPersistentUsedBytes();
    size_t current_requested_size = GetRequestedBytes();
    used_bytes_ -= (previous_size - current_size);
    requested_tail_bytes_ -= (previous_requested_size - current_requested_size);
    alloc_count_--;
  }
  return status;
}

}  // namespace tflite

#endif /* (INCLUDE__UBINOS__UBICLIB == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP_FLAG == 1) */

#endif /* defined(UBINOS_BSP_PRESENT) */
