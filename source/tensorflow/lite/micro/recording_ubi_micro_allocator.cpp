#include "tensorflow/lite/micro/recording_uib_micro_allocator.h"

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/arena_allocator/recording_ubi_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/uib_micro_allocator.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

static tflite::RecordingUbiMicroAllocator* tflite::RecordingUbiMicroAllocator::Create(uint8_t* tensor_arena, size_t arena_size) {
  RecordingUbiArenaBufferAllocator* simple_memory_allocator =
      RecordingUbiArenaBufferAllocator::Create(tensor_arena, arena_size);
  TFLITE_DCHECK(simple_memory_allocator != nullptr);

  uint8_t* memory_planner_buffer =
      simple_memory_allocator->AllocatePersistentBuffer(
          sizeof(GreedyMemoryPlanner), alignof(GreedyMemoryPlanner));
  GreedyMemoryPlanner* memory_planner =
      new (memory_planner_buffer) GreedyMemoryPlanner();

  uint8_t* allocator_buffer = simple_memory_allocator->AllocatePersistentBuffer(
      sizeof(RecordingUbiMicroAllocator), alignof(RecordingUbiMicroAllocator));
  RecordingUbiMicroAllocator* allocator = new (allocator_buffer)
      RecordingUbiMicroAllocator(simple_memory_allocator, memory_planner);
  return allocator;
}

static size_t tflite::RecordingUbiMicroAllocator::GetDefaultTailUsage() {
  // RecordingMicroAllocator inherits from MicroAllocator and its tail usage is
  // similar with MicroAllocator with SingleArenaBufferAllocator and
  // MicroAllocator being replaced.
  return UbiMicroAllocator::GetDefaultTailUsage(/*is_memory_planner_given=*/false) +
         AlignSizeUp<RecordingUbiArenaBufferAllocator>() - AlignSizeUp<UbiArenaBufferAllocator>() +
         AlignSizeUp<RecordingUbiMicroAllocator>() - AlignSizeUp<UbiMicroAllocator>();
}

RecordedAllocation tflite::RecordingUbiMicroAllocator::GetRecordedAllocation(RecordedAllocationType allocation_type) const {
    RecordedAllocationType allocation_type) const {
  switch (allocation_type) {
    case RecordedAllocationType::kTfLiteEvalTensorData:
      return recorded_tflite_eval_tensor_data_;
    case RecordedAllocationType::kPersistentTfLiteTensorData:
      return recorded_persistent_tflite_tensor_data_;
    case RecordedAllocationType::kPersistentTfLiteTensorQuantizationData:
      return recorded_persistent_tflite_tensor_quantization_data_;
    case RecordedAllocationType::kPersistentBufferData:
      return recorded_persistent_buffer_data_;
    case RecordedAllocationType::kTfLiteTensorVariableBufferData:
      return recorded_tflite_tensor_variable_buffer_data_;
    case RecordedAllocationType::kNodeAndRegistrationArray:
      return recorded_node_and_registration_array_data_;
    case RecordedAllocationType::kOpData:
      return recorded_op_data_;
  }
  MicroPrintf("Invalid allocation type supplied: %d", allocation_type);
  return RecordedAllocation();
}

const RecordingUbiArenaBufferAllocator* tflite::RecordingUbiMicroAllocator::GetSimpleMemoryAllocator() const {
  return recording_memory_allocator_;
}

void tflite::RecordingUbiMicroAllocator::PrintAllocations() const {
    // TODO - implement RecordingUbiMicroAllocator.PrintAllocations
    throw new UnsupportedOperationException();
}

void*  tflite::RecordingUbiMicroAllocator::AllocatePersistentBuffer(size_t bytes) {
    // TODO - implement RecordingUbiMicroAllocator.AllocatePersistentBuffer
    throw new UnsupportedOperationException();
}

TfLiteStatus tflite::RecordingUbiMicroAllocator::AllocateNodeAndRegistrations(const Model* model, SubgraphAllocations* subgraph_allocations) {
    // TODO - implement RecordingUbiMicroAllocator.AllocateNodeAndRegistrations
    throw new UnsupportedOperationException();
}

TfLiteStatus tflite::RecordingUbiMicroAllocator::AllocateTfLiteEvalTensors(const Model* model, SubgraphAllocations* subgraph_allocations) {
    // TODO - implement RecordingUbiMicroAllocator.AllocateTfLiteEvalTensors
    throw new UnsupportedOperationException();
}

TfLiteStatus tflite::RecordingUbiMicroAllocator::AllocateVariables(const SubGraph* subgraph, TfLiteEvalTensor* eval_tensors) {
    // TODO - implement RecordingUbiMicroAllocator.AllocateVariables
    throw new UnsupportedOperationException();
}

TfLiteTensor* tflite::RecordingUbiMicroAllocator::AllocatePersistentTfLiteTensorInternal() {
    // TODO - implement RecordingUbiMicroAllocator.AllocatePersistentTfLiteTensorInternal
    throw new UnsupportedOperationException();
}

TfLiteStatus tflite::RecordingUbiMicroAllocator::PopulateTfLiteTensorFromFlatbuffer(const Model* model, TfLiteTensor* tensor, int tensor_index, int subgraph_index, bool allocate_temp) {
    // TODO - implement RecordingUbiMicroAllocator.PopulateTfLiteTensorFromFlatbuffer
    throw new UnsupportedOperationException();
}

tflite::RecordingUbiMicroAllocator::RecordingUbiMicroAllocator(tflite::RecordingUbiArenaBufferAllocator* memory_allocator, MicroMemoryPlanner* memory_planner) {
    // TODO - implement RecordingUbiMicroAllocator.RecordingUbiMicroAllocator
    throw new UnsupportedOperationException();
}

void tflite::RecordingUbiMicroAllocator::PrintRecordedAllocation(RecordedAllocationType allocation_type, const char* allocation_name, const char* allocation_description) const {
    // TODO - implement RecordingUbiMicroAllocator.PrintRecordedAllocation
    throw new UnsupportedOperationException();
}

RecordedAllocation tflite::RecordingUbiMicroAllocator::SnapshotAllocationUsage() const {
    // TODO - implement RecordingUbiMicroAllocator.SnapshotAllocationUsage
    throw new UnsupportedOperationException();
}

void tflite::RecordingUbiMicroAllocator::RecordAllocationUsage(const RecordedAllocation& snapshotted_allocation, RecordedAllocation& recorded_allocation) {
    // TODO - implement RecordingUbiMicroAllocator.RecordAllocationUsage
    throw new UnsupportedOperationException();
}

}  // namespace tflite
