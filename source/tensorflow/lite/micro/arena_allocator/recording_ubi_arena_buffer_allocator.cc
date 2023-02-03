#include "tensorflow/lite/micro/arena_allocator/recording_ubi_arena_buffer_allocator.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

#include <new>

namespace tflite {

tflite::RecordingUbiArenaBufferAllocator::RecordingUbiArenaBufferAllocator(uint8_t* buffer_head, size_t buffer_size)
    : UbiArenaBufferAllocator(buffer_head, buffer_size),
      requested_head_bytes_(0),
      requested_tail_bytes_(0),
      used_bytes_(0),
      alloc_count_(0) {}

tflite::RecordingUbiArenaBufferAllocator::~RecordingUbiArenaBufferAllocator() {}

tflite::RecordingUbiArenaBufferAllocator* tflite::RecordingUbiArenaBufferAllocator::Create(uint8_t* buffer_head, size_t buffer_size) {
  TFLITE_DCHECK(buffer_head != nullptr);
  RecordingUbiArenaBufferAllocator tmp =
      RecordingUbiArenaBufferAllocator(buffer_head, buffer_size);

  uint8_t* allocator_buffer = tmp.AllocatePersistentBuffer(
      sizeof(RecordingUbiArenaBufferAllocator),
      alignof(RecordingUbiArenaBufferAllocator));
  // Use the default copy constructor to populate internal states.
  return new (allocator_buffer) RecordingUbiArenaBufferAllocator(tmp);
}

size_t tflite::RecordingUbiArenaBufferAllocator::GetRequestedBytes() const {
  return requested_head_bytes_ + requested_tail_bytes_;
}

size_t tflite::RecordingUbiArenaBufferAllocator::GetUsedBytes() const {
  return used_bytes_;
}

size_t tflite::RecordingUbiArenaBufferAllocator::GetAllocatedCount() const {
  return alloc_count_;
}

TfLiteStatus tflite::RecordingUbiArenaBufferAllocator::ResizeBuffer(uint8_t* resizable_buf, size_t size, size_t alignment) {
  const uint8_t* previous_head = head();
  TfLiteStatus status =
      UbiArenaBufferAllocator::ResizeBuffer(resizable_buf, size, alignment);
  if (status == kTfLiteOk) {
    used_bytes_ += head() - previous_head;
    requested_head_bytes_ = size;
  }
  return status;
}

uint8_t* tflite::RecordingUbiArenaBufferAllocator::AllocatePersistentBuffer(size_t size, size_t alignment) {
  const uint8_t* previous_tail = tail();
  uint8_t* result =
      UbiArenaBufferAllocator::AllocatePersistentBuffer(size, alignment);
  if (result != nullptr) {
    used_bytes_ += previous_tail - tail();
    requested_tail_bytes_ += size;
    alloc_count_++;
  }
  return result;
}

}  // namespace tflite
