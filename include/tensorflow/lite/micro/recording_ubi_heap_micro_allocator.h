#ifndef RECORDINGUBIHEAPMICROALLOCATOR_H
#define RECORDINGUBIHEAPMICROALLOCATOR_H

#include "tensorflow/lite/micro/arena_allocator/ubi_heap_buffer_allocator.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/ubi_heap_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"

namespace tflite {

class RecordingUbiHeapMicroAllocator : public tflite::UbiHeapMicroAllocator {

    private:
        const UbiHeapBufferAllocator* recording_memory_allocator_;

        /**
         * ubinos_lang_config {"init_value": "{}"}
         */
        RecordedAllocation recorded_tflite_eval_tensor_data_ = {};
        /**
         * ubinos_lang_config {"init_value": "{}"}
         */
        RecordedAllocation recorded_persistent_tflite_tensor_data_ = {};
        /**
         * ubinos_lang_config {"init_value": "{}"}
         */
        RecordedAllocation recorded_persistent_tflite_tensor_quantization_data_ = {};
        /**
         * ubinos_lang_config {"init_value": "{}"}
         */
        RecordedAllocation recorded_persistent_buffer_data_ = {};
        /**
         * ubinos_lang_config {"init_value": "{}"}
         */
        RecordedAllocation recorded_tflite_tensor_variable_buffer_data_ = {};
        /**
         * ubinos_lang_config {"init_value": "{}"}
         */
        RecordedAllocation recorded_node_and_registration_array_data_ = {};

        /**
         * TODO(b/187993291): Re-enable OpData allocating tracking.
         *
         * ubinos_lang_config {"init_value": "{}"}
         */
        RecordedAllocation recorded_op_data_ = {};

    public:
        RecordingUbiHeapMicroAllocator(UbiHeapBufferAllocator* memory_allocator, MicroMemoryPlanner* memory_planner);

        RecordingUbiHeapMicroAllocator(UbiHeapBufferAllocator* memory_allocator);

        /**
         * ubinos_lang_config {"override": true}
         */
        virtual ~RecordingUbiHeapMicroAllocator() override;

        TF_LITE_REMOVE_VIRTUAL_DELETE

        static RecordingUbiHeapMicroAllocator* Create(UbiHeapBufferAllocator* memory_allocator);

        /**
         * Returns the fixed amount of memory overhead of RecordingMicroAllocator.
         */
        static size_t GetDefaultTailUsage();

        /**
         * Returns the recorded allocations information for a given allocation type.
         */
        RecordedAllocation GetRecordedAllocation(RecordedAllocationType allocation_type) const;

        /**
         * Logs out through the ErrorReporter all allocation recordings by type defined in RecordedAllocationType.
         */
        void PrintAllocations() const;

        /**
         * ubinos_lang_config {"override": true}
         */
        void* AllocatePersistentBuffer(size_t bytes) override;

    protected:
        /**
         * ubinos_lang_config {"override": true}
         */
        TfLiteStatus AllocateNodeAndRegistrations(const Model* model, SubgraphAllocations* subgraph_allocations) override;

        /**
         * ubinos_lang_config {"override": true}
         */
        TfLiteStatus AllocateTfLiteEvalTensors(const Model* model, SubgraphAllocations* subgraph_allocations) override;

        /**
         * ubinos_lang_config {"override": true}
         */
        TfLiteStatus AllocateVariables(const SubGraph* subgraph, TfLiteEvalTensor* eval_tensors) override;

        /**
         * TODO(b/162311891): Once all kernels have been updated to the new API drop this method.
         * It is only used to record TfLiteTensor persistent allocations.
         *
         * ubinos_lang_config {"override": true}
         */
        TfLiteTensor* AllocatePersistentTfLiteTensorInternal() override;

        /**
         * TODO(b/162311891): Once all kernels have been updated to the new API drop this function since all allocations for quantized data will take place in the temp section.
         *
         * ubinos_lang_config {"override": true}
         */
        TfLiteStatus PopulateTfLiteTensorFromFlatbuffer(const Model* model, TfLiteTensor* tensor, int tensor_index, int subgraph_index, bool allocate_temp) override;

    private:
        void PrintRecordedAllocation(RecordedAllocationType allocation_type, const char* allocation_name, const char* allocation_description) const;

        RecordedAllocation SnapshotAllocationUsage() const;

        void RecordAllocationUsage(const RecordedAllocation& snapshotted_allocation, RecordedAllocation& recorded_allocation);
 };

}  // namespace tflite

#endif
