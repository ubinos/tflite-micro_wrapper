#include "tensorflow/lite/micro/ubi_micro_allocator.h"

#include <cstddef>
#include <cstdint>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/memory_planner/micro_memory_planner.h"
#include "tensorflow/lite/micro/micro_allocation_info.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

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

UbiMicroAllocator* tflite::UbiMicroAllocator::Create(UbiArenaBufferAllocator* memory_allocator) {
  TFLITE_DCHECK(memory_allocator != nullptr);

  uint8_t* memory_planner_buffer = memory_allocator->AllocatePersistentBuffer(
      sizeof(GreedyMemoryPlanner), alignof(GreedyMemoryPlanner));
  GreedyMemoryPlanner* memory_planner =
      new (memory_planner_buffer) GreedyMemoryPlanner();

  uint8_t* allocator_buffer = memory_allocator->AllocatePersistentBuffer(
      sizeof(UbiMicroAllocator), alignof(UbiMicroAllocator));
  UbiMicroAllocator* allocator = new (allocator_buffer)
      UbiMicroAllocator(memory_allocator, memory_allocator, memory_planner);

  return allocator;
}

UbiMicroAllocator* tflite::UbiMicroAllocator::Create(uint8_t* tensor_arena, size_t arena_size) {
  uint8_t* aligned_arena =
      AlignPointerUp(tensor_arena, MicroArenaBufferAlignment());
  size_t aligned_arena_size = tensor_arena + arena_size - aligned_arena;
  UbiArenaBufferAllocator* memory_allocator =
      UbiArenaBufferAllocator::Create(aligned_arena, aligned_arena_size);

  // By default create GreedyMemoryPlanner.
  // If a different MemoryPlanner is needed, use the other api.
  uint8_t* memory_planner_buffer = memory_allocator->AllocatePersistentBuffer(
      sizeof(GreedyMemoryPlanner), alignof(GreedyMemoryPlanner));
  GreedyMemoryPlanner* memory_planner =
      new (memory_planner_buffer) GreedyMemoryPlanner();

  return Create(memory_allocator, memory_planner);
}

UbiMicroAllocator* tflite::UbiMicroAllocator::Create(uint8_t* tensor_arena, size_t arena_size, MicroMemoryPlanner* memory_planner) {
  uint8_t* aligned_arena =
      AlignPointerUp(tensor_arena, MicroArenaBufferAlignment());
  size_t aligned_arena_size = tensor_arena + arena_size - aligned_arena;
  UbiArenaBufferAllocator* memory_allocator =
      UbiArenaBufferAllocator::Create(aligned_arena, aligned_arena_size);

  return Create(memory_allocator, memory_planner);
}

size_t UbiMicroAllocator::GetDefaultTailUsage(bool is_memory_planner_given) {
  size_t total_size = AlignSizeUp<UbiArenaBufferAllocator>() +
                      AlignSizeUp<UbiMicroAllocator>() +
                      AlignSizeUp<MicroBuiltinDataAllocator>() +
                      AlignSizeUp<SubgraphAllocations>();
  if (!is_memory_planner_given) {
    total_size += AlignSizeUp<GreedyMemoryPlanner>();
  }
  return total_size;
}

tflite::UbiMicroAllocator::UbiMicroAllocator(IPersistentBufferAllocator* persistent_buffer_allocator, INonPersistentBufferAllocator* non_persistent_buffer_allocator, MicroMemoryPlanner* memory_planner)
    : MicroAllocator(persistent_buffer_allocator, non_persistent_buffer_allocator, memory_planner) {
}

}  // namespace tflite
