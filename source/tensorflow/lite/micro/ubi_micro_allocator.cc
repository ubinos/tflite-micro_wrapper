#include "tensorflow/lite/micro/ubi_micro_allocator.h"

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/arena_allocator/ubi_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

UbiMicroAllocator* tflite::UbiMicroAllocator::Create(UbiArenaBufferAllocator* memory_allocator, MicroMemoryPlanner* memory_planner) {
  TFLITE_DCHECK(memory_allocator != nullptr);
  TFLITE_DCHECK(memory_planner != nullptr);

  uint8_t* allocator_buffer = memory_allocator->AllocatePersistentBuffer(
      sizeof(UbiMicroAllocator), alignof(UbiMicroAllocator));
  UbiMicroAllocator* allocator = new (allocator_buffer)
      UbiMicroAllocator(memory_allocator, memory_allocator, memory_planner);
  return allocator;
}

tflite::UbiMicroAllocator::UbiMicroAllocator(IPersistentBufferAllocator* persistent_buffer_allocator, INonPersistentBufferAllocator* non_persistent_buffer_allocator, MicroMemoryPlanner* memory_planner)
    : MicroAllocator(persistent_buffer_allocator, non_persistent_buffer_allocator, memory_planner) {
}

}  // namespace tflite
