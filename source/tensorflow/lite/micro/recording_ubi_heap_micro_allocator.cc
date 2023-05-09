#if defined(UBINOS_BSP_PRESENT)

#include <ubinos_config.h>

#if (INCLUDE__UBINOS__UBICLIB == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP_FLAG == 1)

#include <ubinos.h>

#include "tensorflow/lite/micro/recording_ubi_heap_micro_allocator.h"

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/arena_allocator/ubi_heap_buffer_allocator.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/ubi_heap_micro_allocator.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

size_t tflite::RecordingUbiHeapMicroAllocator::GetDefaultTailUsage() {
  return UbiHeapMicroAllocator::GetDefaultTailUsage(/*is_memory_planner_given=*/false);
}

tflite::RecordingUbiHeapMicroAllocator::RecordingUbiHeapMicroAllocator(UbiHeapBufferAllocator* recording_memory_allocator, MicroMemoryPlanner* memory_planner)
    : UbiHeapMicroAllocator(recording_memory_allocator, memory_planner), recording_memory_allocator_(recording_memory_allocator) {
}

tflite::RecordingUbiHeapMicroAllocator::RecordingUbiHeapMicroAllocator(UbiHeapBufferAllocator* recording_memory_allocator)
    : UbiHeapMicroAllocator(recording_memory_allocator), recording_memory_allocator_(recording_memory_allocator) {
}

tflite::RecordingUbiHeapMicroAllocator::~RecordingUbiHeapMicroAllocator() {
}

RecordingUbiHeapMicroAllocator* tflite::RecordingUbiHeapMicroAllocator::Create(UbiHeapBufferAllocator* recording_memory_allocator) {
  TFLITE_DCHECK(recording_memory_allocator != nullptr);

  return new RecordingUbiHeapMicroAllocator(recording_memory_allocator);
}

RecordedAllocation tflite::RecordingUbiHeapMicroAllocator::GetRecordedAllocation(RecordedAllocationType allocation_type) const {
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

void tflite::RecordingUbiHeapMicroAllocator::PrintAllocations() const {
  MicroPrintf("[RecordingMicroAllocator] Arena allocation total (max.) %d (%d) bytes",
              recording_memory_allocator_->GetUsedBytes(), recording_memory_allocator_->GetUsedBytesMax());
  MicroPrintf("[RecordingMicroAllocator] Arena allocation head (max.) %d (%d) bytes",
              recording_memory_allocator_->GetNonPersistentUsedBytes(), recording_memory_allocator_->GetNonPersistentUsedBytesMax());
  MicroPrintf("[RecordingMicroAllocator] Arena allocation tail (max.) %d (%d) bytes",
              recording_memory_allocator_->GetPersistentUsedBytes(), recording_memory_allocator_->GetPersistentUsedBytesMax());

  PrintRecordedAllocation(RecordedAllocationType::kTfLiteEvalTensorData,
                          "TfLiteEvalTensor data", "allocations");
  PrintRecordedAllocation(RecordedAllocationType::kPersistentTfLiteTensorData,
                          "Persistent TfLiteTensor data", "tensors");
  PrintRecordedAllocation(
      RecordedAllocationType::kPersistentTfLiteTensorQuantizationData,
      "Persistent TfLiteTensor quantization data", "allocations");
  PrintRecordedAllocation(RecordedAllocationType::kPersistentBufferData,
                          "Persistent buffer data", "allocations");
  PrintRecordedAllocation(
      RecordedAllocationType::kTfLiteTensorVariableBufferData,
      "TfLiteTensor variable buffer data", "allocations");
  PrintRecordedAllocation(RecordedAllocationType::kNodeAndRegistrationArray,
                          "NodeAndRegistration struct",
                          "NodeAndRegistration structs");
  PrintRecordedAllocation(RecordedAllocationType::kOpData,
                          "Operator runtime data", "OpData structs");
}

void*  tflite::RecordingUbiHeapMicroAllocator::AllocatePersistentBuffer(size_t bytes) {
  RecordedAllocation allocations = SnapshotAllocationUsage();
  void* buffer = UbiHeapMicroAllocator::AllocatePersistentBuffer(bytes);
  RecordAllocationUsage(allocations, recorded_persistent_buffer_data_);

  return buffer;
}

void tflite::RecordingUbiHeapMicroAllocator::PrintRecordedAllocation(RecordedAllocationType allocation_type, const char* allocation_name, const char* allocation_description) const {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  RecordedAllocation allocation = GetRecordedAllocation(allocation_type);
  if (allocation.used_bytes > 0 || allocation.requested_bytes > 0) {
    MicroPrintf(
        "[RecordingMicroAllocator] '%s' used %d bytes with alignment overhead "
        "(requested %d bytes for %d %s)",
        allocation_name, allocation.used_bytes, allocation.requested_bytes,
        allocation.count, allocation_description);
  }
#endif
}

TfLiteStatus tflite::RecordingUbiHeapMicroAllocator::AllocateNodeAndRegistrations(const Model* model, SubgraphAllocations* subgraph_allocations) {
  RecordedAllocation allocations = SnapshotAllocationUsage();

  TfLiteStatus status =
      UbiHeapMicroAllocator::AllocateNodeAndRegistrations(model, subgraph_allocations);

  RecordAllocationUsage(allocations,
                        recorded_node_and_registration_array_data_);

  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs()->size();
       subgraph_idx++) {
    // The allocation count in SingleArenaBufferAllocator will only be 1. To
    // provide better logging, decrement by 1 and add in the actual number of
    // operators used in the graph: The allocation for this recording will
    // always be 1. This is because the parent class mallocs one large
    // allocation for the number of nodes in the graph (e.g.
    // sizeof(NodeAndRegistration) * num_nodes). To prevent extra overhead and
    // potential for fragmentation, manually adjust the accounting by
    // decrementing by 1 and adding the actual number of nodes used in the
    // graph:
    if (model->subgraphs()->Get(subgraph_idx)->operators()) {
      recorded_node_and_registration_array_data_.count +=
          model->subgraphs()->Get(subgraph_idx)->operators()->size() - 1;
    } else {
      recorded_node_and_registration_array_data_.count -= 1;
    }
  }
  return status;
}

TfLiteStatus tflite::RecordingUbiHeapMicroAllocator::AllocateTfLiteEvalTensors(const Model* model, SubgraphAllocations* subgraph_allocations) {
  RecordedAllocation allocations = SnapshotAllocationUsage();

  TfLiteStatus status =
      UbiHeapMicroAllocator::AllocateTfLiteEvalTensors(model, subgraph_allocations);

  RecordAllocationUsage(allocations, recorded_tflite_eval_tensor_data_);

  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs()->size();
       subgraph_idx++) {
    // The allocation for this recording will always be 1. This is because the
    // parent class mallocs one large allocation for the number of tensors in
    // the graph (e.g. sizeof(TfLiteEvalTensor) * num_tensors). To prevent extra
    // overhead and potential for fragmentation, manually adjust the accounting
    // by decrementing by 1 and adding the actual number of tensors used in the
    // graph:
    recorded_tflite_eval_tensor_data_.count +=
        model->subgraphs()->Get(subgraph_idx)->tensors()->size() - 1;
  }
  return status;
}

TfLiteStatus tflite::RecordingUbiHeapMicroAllocator::AllocateVariables(const SubGraph* subgraph, TfLiteEvalTensor* eval_tensors) {
  RecordedAllocation allocations = SnapshotAllocationUsage();

  TfLiteStatus status =
      UbiHeapMicroAllocator::AllocateVariables(subgraph, eval_tensors);

  RecordAllocationUsage(allocations,
                        recorded_tflite_tensor_variable_buffer_data_);
  return status;
}

TfLiteTensor* tflite::RecordingUbiHeapMicroAllocator::AllocatePersistentTfLiteTensorInternal() {
  RecordedAllocation allocations = SnapshotAllocationUsage();

  TfLiteTensor* result =
      UbiHeapMicroAllocator::AllocatePersistentTfLiteTensorInternal();

  RecordAllocationUsage(allocations, recorded_persistent_tflite_tensor_data_);
  return result;
}

TfLiteStatus tflite::RecordingUbiHeapMicroAllocator::PopulateTfLiteTensorFromFlatbuffer(const Model* model, TfLiteTensor* tensor, int tensor_index, int subgraph_index, bool allocate_temp) {
  RecordedAllocation allocations = SnapshotAllocationUsage();

  TfLiteStatus status = UbiHeapMicroAllocator::PopulateTfLiteTensorFromFlatbuffer(
      model, tensor, tensor_index, subgraph_index, allocate_temp);

  RecordAllocationUsage(allocations,
                        recorded_persistent_tflite_tensor_quantization_data_);
  return status;
}

RecordedAllocation tflite::RecordingUbiHeapMicroAllocator::SnapshotAllocationUsage() const {
  return {/*requested_bytes=*/recording_memory_allocator_->GetRequestedBytes(),
          /*used_bytes=*/recording_memory_allocator_->GetUsedBytes(),
          /*count=*/recording_memory_allocator_->GetAllocatedCount()};
}

void tflite::RecordingUbiHeapMicroAllocator::RecordAllocationUsage(const RecordedAllocation& snapshotted_allocation, RecordedAllocation& recorded_allocation) {
  recorded_allocation.requested_bytes +=
      recording_memory_allocator_->GetRequestedBytes() -
      snapshotted_allocation.requested_bytes;
  recorded_allocation.used_bytes +=
      recording_memory_allocator_->GetUsedBytes() -
      snapshotted_allocation.used_bytes;
  recorded_allocation.count +=
      recording_memory_allocator_->GetAllocatedCount() -
      snapshotted_allocation.count;
}

}  // namespace tflite

#endif /* (INCLUDE__UBINOS__UBICLIB == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP == 1) && !(UBINOS__UBICLIB__EXCLUDE_HEAP_FLAG == 1) */

#endif /* defined(UBINOS_BSP_PRESENT) */
