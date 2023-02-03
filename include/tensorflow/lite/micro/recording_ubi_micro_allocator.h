#ifndef RECORDINGUBIMICROALLOCATOR_H
#define RECORDINGUBIMICROALLOCATOR_H

namespace tflite {

#include "tensorflow/lite/micro/arena_allocator/recording_ubi_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/ubi_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"

    /**
 * Utility subclass of MicroAllocator that records all allocations inside the arena.
 * A summary of allocations can be logged through the ErrorReporter by invoking LogAllocations().
 * This special allocator requires an instance of RecordingSingleArenaBufferAllocator to capture allocations in the head and tail.
 * Arena allocation recording can be retrieved by type through the GetRecordedAllocation() function.
 * This class should only be used for auditing memory usage or integration testing.
 */
class RecordingUbiMicroAllocator : public tflite::UbiMicroAllocator {

    private:
        const RecordingUbiArenaBufferAllocator* recording_memory_allocator_;
        RecordedAllocation recorded_tflite_eval_tensor_data_;
        RecordedAllocation recorded_persistent_tflite_tensor_data_;
        RecordedAllocation recorded_persistent_tflite_tensor_quantization_data_;
        RecordedAllocation recorded_persistent_buffer_data_;
        RecordedAllocation recorded_tflite_tensor_variable_buffer_data_;
        RecordedAllocation recorded_node_and_registration_array_data_;
        /**
         * TODO(b/187993291): Re-enable OpData allocating tracking.
         */
        RecordedAllocation recorded_op_data_;

    public:
        /**
         * Creates a MicroAllocator instance from a given tensor arena.
         * This arena will be managed by the created instance.
         * The GreedyMemoryPlanner will by default be used and created on the arena.
         * Note: Please use alignas(16) to make sure tensor_arena is 16 bytes aligned, otherwise some head room will be wasted.
         * TODO(b/157615197): Cleanup constructor + factory usage.
         */
        static RecordingUbiMicroAllocator* Create(uint8_t* tensor_arena, size_t arena_size);

        /**
         * Returns the fixed amount of memory overhead of RecordingMicroAllocator.
         */
        static size_t GetDefaultTailUsage();

        /**
         * Returns the recorded allocations information for a given allocation type.
         */
        RecordedAllocation GetRecordedAllocation(RecordedAllocationType allocation_type) const;

        const RecordingUbiArenaBufferAllocator* GetSimpleMemoryAllocator() const;

        /**
         * Logs out through the ErrorReporter all allocation recordings by type defined in RecordedAllocationType.
         */
        void PrintAllocations() const;

        /**
         * ubinos_lang_config {"override": true}
         */
        void*  AllocatePersistentBuffer(size_t bytes);

    protected:
        /**
         * ubinos_lang_config {"override": true}
         */
        TfLiteStatus AllocateNodeAndRegistrations(const Model* model, SubgraphAllocations* subgraph_allocations);

        /**
         * ubinos_lang_config {"override": true}
         */
        TfLiteStatus AllocateTfLiteEvalTensors(const Model* model, SubgraphAllocations* subgraph_allocations);

        /**
         * ubinos_lang_config {"override": true}
         */
        TfLiteStatus AllocateVariables(const SubGraph* subgraph, TfLiteEvalTensor* eval_tensors);

        /**
         * TODO(b/162311891): Once all kernels have been updated to the new API drop this method.
         * It is only used to record TfLiteTensor persistent allocations.
         * 
         * ubinos_lang_config {"override": true}
         */
        TfLiteTensor* AllocatePersistentTfLiteTensorInternal();

        /**
         * TODO(b/162311891): Once all kernels have been updated to the new API drop this function since all allocations for quantized data will take place in the temp section.
         * 
         * ubinos_lang_config {"override": true}
         */
        TfLiteStatus PopulateTfLiteTensorFromFlatbuffer(const Model* model, TfLiteTensor* tensor, int tensor_index, int subgraph_index, bool allocate_temp);

    private:
        RecordingUbiMicroAllocator(tflite::RecordingUbiArenaBufferAllocator* memory_allocator, MicroMemoryPlanner* memory_planner);

        TF_LITE_REMOVE_VIRTUAL_DELETE

        void PrintRecordedAllocation(RecordedAllocationType allocation_type, const char* allocation_name, const char* allocation_description) const;

        RecordedAllocation SnapshotAllocationUsage() const;

        void RecordAllocationUsage(const RecordedAllocation& snapshotted_allocation, RecordedAllocation& recorded_allocation);
    };
}

#endif
