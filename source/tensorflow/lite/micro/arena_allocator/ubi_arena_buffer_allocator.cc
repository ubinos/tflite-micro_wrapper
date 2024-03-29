#include "tensorflow/lite/micro/arena_allocator/ubi_arena_buffer_allocator.h"

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

tflite::UbiArenaBufferAllocator::UbiArenaBufferAllocator(uint8_t* buffer_head, uint8_t* buffer_tail)
    : buffer_head_(buffer_head),
      buffer_tail_(buffer_tail),
      head_(buffer_head),
      tail_(buffer_tail),
      temp_(buffer_head_),
      head_max_(head_),
      tail_min_(tail_),
      temp_max_(temp_) {}

tflite::UbiArenaBufferAllocator::UbiArenaBufferAllocator(uint8_t* buffer_head, size_t buffer_size)
    : UbiArenaBufferAllocator(buffer_head, buffer_head + buffer_size) {}

tflite::UbiArenaBufferAllocator::~UbiArenaBufferAllocator() {}

tflite::UbiArenaBufferAllocator* tflite::UbiArenaBufferAllocator::Create(uint8_t* buffer_head, size_t buffer_size) {
  TFLITE_DCHECK(buffer_head != nullptr);
  UbiArenaBufferAllocator tmp =
      UbiArenaBufferAllocator(buffer_head, buffer_size);

  // Allocate enough bytes from the buffer to create a
  // UbiArenaBufferAllocator. The new instance will use the current adjusted
  // tail buffer from the tmp allocator instance.
  uint8_t* allocator_buffer = tmp.AllocatePersistentBuffer(
      sizeof(UbiArenaBufferAllocator), alignof(UbiArenaBufferAllocator));
  // Use the default copy constructor to populate internal states.
  return new (allocator_buffer) UbiArenaBufferAllocator(tmp);
}

TfLiteStatus tflite::UbiArenaBufferAllocator::ResizeBuffer(uint8_t* resizable_buf, size_t size, size_t alignment) {
  // Only supports one resizable buffer, which starts at the buffer head.
  uint8_t* expect_resizable_buf = AlignPointerUp(buffer_head_, alignment);
  if (head_ != temp_ || resizable_buf != expect_resizable_buf) {
    MicroPrintf(
        "Internal error: either buffer is not resizable or "
        "ResetTempAllocations() is not called before ResizeBuffer().");
    return kTfLiteError;
  }

  uint8_t* const aligned_result = AlignPointerUp(buffer_head_, alignment);
  const size_t available_memory = tail_ - aligned_result;
  if (available_memory < size) {
    MicroPrintf(
        "Failed to resize buffer. Requested: %u, available %u, missing: %u",
        size, available_memory, size - available_memory);
    return kTfLiteError;
  }

  head_ = aligned_result + size;
  head_max_ = std::max(head_max_, head_);
  head_used_size_ = head_ - buffer_head_;
  head_used_size_max_ = std::max(head_used_size_max_, head_used_size_);

  temp_ = head_;
  temp_max_ = std::max(temp_max_, temp_);
  temp_used_size_ = temp_ - buffer_head_;
  temp_used_size_max_ = std::max(temp_used_size_max_, temp_used_size_);

  return kTfLiteOk;
}

TfLiteStatus tflite::UbiArenaBufferAllocator::ResetTempAllocations() {
  // TODO(b/209453859): enable error check based on IsAllTempDeallocated after
  // all AllocateTemp have been paird with DeallocateTemp
  if (!IsAllTempDeallocated()) {
    MicroPrintf(
        "All temp buffers must be freed before calling ResetTempAllocations()");
    return kTfLiteError;
  }
  temp_ = head_;
  temp_max_ = std::max(temp_max_, temp_);
  temp_used_size_ = temp_ - buffer_head_;
  temp_used_size_max_ = std::max(temp_used_size_max_, temp_used_size_);
  return kTfLiteOk;
}

TfLiteStatus tflite::UbiArenaBufferAllocator::ReserveNonPersistentOverlayMemory(size_t size, size_t alignment) {
  uint8_t* expect_resizable_buf = AlignPointerUp(buffer_head_, alignment);
  return ResizeBuffer(expect_resizable_buf, size, alignment);
}

bool tflite::UbiArenaBufferAllocator::IsAllTempDeallocated() {
  if (temp_buffer_count_ != 0 || temp_buffer_ptr_check_sum_ != 0) {
    MicroPrintf(
        "Number of allocated temp buffers: %d. Checksum passing status: %d",
        temp_buffer_count_, !temp_buffer_ptr_check_sum_);
    return false;
  }
  return true;
}

size_t tflite::UbiArenaBufferAllocator::GetUsedBytes() const {
  return GetPersistentUsedBytes() + GetNonPersistentUsedBytes();
}

size_t tflite::UbiArenaBufferAllocator::GetUsedBytesMax() const {
  return GetPersistentUsedBytesMax() + GetNonPersistentUsedBytesMax();
}

uint8_t* tflite::UbiArenaBufferAllocator::GetOverlayMemoryAddress() const {
  return buffer_head_;
}

size_t tflite::UbiArenaBufferAllocator::GetNonPersistentUsedBytes() const {
  return std::max(head_ - buffer_head_, temp_ - buffer_head_);
}

size_t tflite::UbiArenaBufferAllocator::GetNonPersistentUsedBytesMax() const {
  return std::max(head_max_ - buffer_head_, temp_max_ - buffer_head_);
}

size_t tflite::UbiArenaBufferAllocator::GetPersistentUsedBytes() const {
  return buffer_tail_ - tail_;
}

size_t tflite::UbiArenaBufferAllocator::GetPersistentUsedBytesMax() const {
  return buffer_tail_ - tail_min_;
}

size_t tflite::UbiArenaBufferAllocator::GetAvailableMemory(size_t alignment) const {
  uint8_t* const aligned_temp = AlignPointerUp(temp_, alignment);
  uint8_t* const aligned_tail = AlignPointerDown(tail_, alignment);
  return aligned_tail - aligned_temp;
}

void tflite::UbiArenaBufferAllocator::DeallocateTemp(uint8_t* buf) {
  temp_buffer_ptr_check_sum_ ^= (reinterpret_cast<intptr_t>(buf));
  temp_buffer_count_--;
}

TfLiteStatus tflite::UbiArenaBufferAllocator::DeallocateResizableBuffer(uint8_t* resizable_buf) {
  return ResizeBuffer(resizable_buf, 0, 1);
}

uint8_t* tflite::UbiArenaBufferAllocator::AllocateTemp(size_t size, size_t alignment) {
  uint8_t* const aligned_result = AlignPointerUp(temp_, alignment);
  const size_t available_memory = tail_ - aligned_result;
  if (available_memory < size) {
    MicroPrintf(
        "Failed to allocate temp memory. Requested: %u, "
        "available %u, missing: %u",
        size, available_memory, size - available_memory);
    return nullptr;
  }
  temp_ = aligned_result + size;
  temp_max_ = std::max(temp_max_, temp_);
  temp_used_size_ = temp_ - buffer_head_;
  temp_used_size_max_ = std::max(temp_used_size_max_, temp_used_size_);

  temp_buffer_ptr_check_sum_ ^= (reinterpret_cast<intptr_t>(aligned_result));
  temp_buffer_count_++;
  return aligned_result;
}

uint8_t* tflite::UbiArenaBufferAllocator::AllocateResizableBuffer(size_t size, size_t alignment) {
  // Only supports one resizable buffer, which starts at the buffer head.
  uint8_t* expect_resizable_buf = AlignPointerUp(buffer_head_, alignment);
  if (ResizeBuffer(expect_resizable_buf, size, alignment) == kTfLiteOk) {
    return expect_resizable_buf;
  }
  return nullptr;
}

uint8_t* tflite::UbiArenaBufferAllocator::AllocatePersistentBuffer(size_t size, size_t alignment) {
  uint8_t* const aligned_result = AlignPointerDown(tail_ - size, alignment);
  if (aligned_result < head_) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
    const size_t missing_memory = head_ - aligned_result;
    MicroPrintf(
        "Failed to allocate tail memory. Requested: %u, "
        "available %u, missing: %u",
        size, size - missing_memory, missing_memory);
#endif
    return nullptr;
  }
  tail_ = aligned_result;
  tail_min_ = std::min(tail_min_, tail_);
  tail_used_size_ = buffer_tail_ - tail_;
  tail_used_size_max_ = std::max(tail_used_size_max_, tail_used_size_);

  return aligned_result;
}

uint8_t* tflite::UbiArenaBufferAllocator::head() const {
    return head_;
}

uint8_t* tflite::UbiArenaBufferAllocator::tail() const {
    return tail_;
}

size_t tflite::UbiArenaBufferAllocator::GetBufferSize() const {
    return buffer_tail_ - buffer_head_;
}

}  // namespace tflite
