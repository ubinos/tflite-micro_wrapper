#if defined(UBINOS_BSP_PRESENT)

#include <ubinos_config.h>

#if (INCLUDE__UBINOS__UBICLIB == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP_FLAG == 1)

#include <ubinos.h>

#include "tensorflow/lite/micro/ubi_heap_micro_allocator.h"

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

tflite::UbiHeapMicroAllocator::UbiHeapMicroAllocator(UbiHeapBufferAllocator* memory_allocator, MicroMemoryPlanner* memory_planner)
    : MicroAllocator(memory_allocator, memory_allocator, memory_planner), heap_memory_allocator_(memory_allocator) {
}

tflite::UbiHeapMicroAllocator::UbiHeapMicroAllocator(UbiHeapBufferAllocator* memory_allocator)
    : MicroAllocator(memory_allocator, memory_allocator, &default_memory_planner_), heap_memory_allocator_(memory_allocator) {
}

tflite::UbiHeapMicroAllocator::~UbiHeapMicroAllocator() {
}

UbiHeapMicroAllocator* tflite::UbiHeapMicroAllocator::Create(UbiHeapBufferAllocator* memory_allocator, MicroMemoryPlanner* memory_planner) {
  TFLITE_DCHECK(memory_allocator != nullptr);
  TFLITE_DCHECK(memory_planner != nullptr);

  return new UbiHeapMicroAllocator(memory_allocator, memory_planner);
}

UbiHeapMicroAllocator* tflite::UbiHeapMicroAllocator::Create(UbiHeapBufferAllocator* memory_allocator) {
  TFLITE_DCHECK(memory_allocator != nullptr);

  return new UbiHeapMicroAllocator(memory_allocator);
}

size_t tflite::UbiHeapMicroAllocator::GetDefaultTailUsage(bool is_memory_planner_given) {
  // MicroBuiltinDataAllocator + SubgraphAllocations
  return heap_get_block_allocated_size_min(NULL) * 2;
}

uint8_t* tflite::UbiHeapMicroAllocator::GetOverlayMemoryAddress() const {
  return heap_memory_allocator_->GetOverlayMemoryAddress();
}

const UbiHeapBufferAllocator* tflite::UbiHeapMicroAllocator::GetSimpleMemoryAllocator() const {
  return heap_memory_allocator_;
}

}  // namespace tflite

#endif /* (INCLUDE__UBINOS__UBICLIB == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP_FLAG == 1) */

#endif /* defined(UBINOS_BSP_PRESENT) */
