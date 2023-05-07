#include <ubinos.h>
#include <ubinos/ubiclib/heap.h>

#include "tensorflow/lite/micro/arena_allocator/ubi_heap_buffer_allocator.h"

#include <cstdint>

#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/memory_helpers.h"

TF_LITE_MICRO_TESTS_BEGIN

// TF_LITE_MICRO_TEST(TestEnsureHeadSizeSimpleAlignment) {
//   tflite::UbiHeapBufferAllocator allocator;

//   uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, 1);
//   TF_LITE_MICRO_EXPECT(resizable_buf != nullptr);

//   TF_LITE_MICRO_EXPECT_EQ(
//       kTfLiteOk,
//       allocator.ResizeBuffer(resizable_buf, /*size=*/100, /*alignment=*/1));
//   TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(100),
//                           allocator.GetNonPersistentUsedBytes());

//   TF_LITE_MICRO_EXPECT_EQ(
//       kTfLiteOk,
//       allocator.ResizeBuffer(resizable_buf, /*size=*/10, /*alignment=*/1));
//   TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(10),
//                           allocator.GetNonPersistentUsedBytes());

//   TF_LITE_MICRO_EXPECT_EQ(
//       kTfLiteOk,
//       allocator.ResizeBuffer(resizable_buf, /*size=*/1000, /*alignment=*/1));
//   TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(1000),
//                           allocator.GetNonPersistentUsedBytes());
// }

TF_LITE_MICRO_TEST(TestAdjustHeadSizeMisalignment) {
  tflite::UbiHeapBufferAllocator allocator;
  size_t size;
  size_t alignment = tflite::MicroArenaBufferAlignment();
  size_t block_overhead = heap_get_block_overhead(NULL);
  // size_t arena_size = allocator.GetAvailableMemory(1);

  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, alignment);
  TF_LITE_MICRO_EXPECT(resizable_buf != nullptr);

  // First head adjustment of 100 bytes (aligned alignment):
  size = 100;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, size, alignment));

  // Offset alignment can lead to allocation within 8 byte range of
  // requested bytes based to arena alignment at runtime:
  TF_LITE_MICRO_EXPECT_GE(allocator.GetNonPersistentUsedBytes(), size);
  TF_LITE_MICRO_EXPECT_LE(allocator.GetNonPersistentUsedBytes(), size + block_overhead + alignment - 1);

  size = 10;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, size, alignment));
  TF_LITE_MICRO_EXPECT_GE(allocator.GetNonPersistentUsedBytes(), size);
  TF_LITE_MICRO_EXPECT_LE(allocator.GetNonPersistentUsedBytes(), size + block_overhead + alignment - 1);

  size = 1000;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, size, alignment));
  TF_LITE_MICRO_EXPECT_GE(allocator.GetNonPersistentUsedBytes(), size);
  TF_LITE_MICRO_EXPECT_LE(allocator.GetNonPersistentUsedBytes(), size + block_overhead + alignment - 1);
}

TF_LITE_MICRO_TEST(TestAdjustHeadSizeMisalignedHandlesCorrectBytesAvailable) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t alignment = tflite::MicroArenaBufferAlignment();
  size_t block_overhead = heap_get_block_overhead(NULL);
  size_t arena_size = allocator.GetAvailableMemory(alignment);

  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, alignment);
  TF_LITE_MICRO_EXPECT(resizable_buf != nullptr);

  // First head adjustment of 100 bytes (aligned alignment):
  size_t size = 100;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, size, alignment));

  // allocator.GetAvailableMemory() should also report the actual amount of
  // memory available based on a requested offset (alignment):
  size_t aligned_available_bytes =
      allocator.GetAvailableMemory(alignment);
  TF_LITE_MICRO_EXPECT_LE(aligned_available_bytes, arena_size - (size + block_overhead));
  TF_LITE_MICRO_EXPECT_GE(aligned_available_bytes, arena_size - (size + block_overhead + alignment));

  size = 10;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, size, alignment));
  aligned_available_bytes = allocator.GetAvailableMemory(alignment);
  TF_LITE_MICRO_EXPECT_LE(aligned_available_bytes, arena_size - (size + block_overhead));
  TF_LITE_MICRO_EXPECT_GE(aligned_available_bytes, arena_size - (size + block_overhead + alignment));

  size = 100;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, size, alignment));
  aligned_available_bytes = allocator.GetAvailableMemory(alignment);
  TF_LITE_MICRO_EXPECT_LE(aligned_available_bytes, arena_size - (size + block_overhead));
  TF_LITE_MICRO_EXPECT_GE(aligned_available_bytes, arena_size - (size + block_overhead + alignment));
}

TF_LITE_MICRO_TEST(TestGetAvailableMemory) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t alignment = tflite::MicroArenaBufferAlignment();
  size_t block_overhead = heap_get_block_overhead(NULL);
  size_t arena_size = allocator.GetAvailableMemory(alignment);

  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, alignment);
  TF_LITE_MICRO_EXPECT(resizable_buf != nullptr);

  size_t size = 100;
  allocator.ResizeBuffer(resizable_buf, size,
                         alignment);
  allocator.AllocatePersistentBuffer(size,
                                     alignment);

  size_t aligned_available_bytes =
        allocator.GetAvailableMemory(alignment);
  TF_LITE_MICRO_EXPECT_LE(aligned_available_bytes, arena_size - (size + block_overhead) * 2);
  TF_LITE_MICRO_EXPECT_GE(aligned_available_bytes, arena_size - (size + block_overhead + alignment) * 2);
}

TF_LITE_MICRO_TEST(TestGetAvailableMemoryWithTempAllocations) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t alignment = tflite::MicroArenaBufferAlignment();
  size_t block_overhead = heap_get_block_overhead(NULL);
  size_t arena_size = allocator.GetAvailableMemory(alignment);

  size_t size = 100;
  uint8_t* temp = allocator.AllocateTemp(size,
                                         alignment);

  size_t aligned_available_bytes =
        allocator.GetAvailableMemory(alignment);
  TF_LITE_MICRO_EXPECT_LE(aligned_available_bytes, arena_size - (size + block_overhead));
  TF_LITE_MICRO_EXPECT_GE(aligned_available_bytes, arena_size - (size + block_overhead + alignment));

  // Reset temp allocations and ensure GetAvailableMemory() is back to the
  // starting size:
  allocator.DeallocateTemp(temp);
  allocator.ResetTempAllocations();

  TF_LITE_MICRO_EXPECT_EQ(allocator.GetAvailableMemory(alignment), arena_size);
}

TF_LITE_MICRO_TEST(TestGetUsedBytes) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t alignment = tflite::MicroArenaBufferAlignment();
  size_t block_overhead = heap_get_block_overhead(NULL);

  TF_LITE_MICRO_EXPECT_EQ(allocator.GetUsedBytes(), static_cast<size_t>(0));
  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, alignment);
  TF_LITE_MICRO_EXPECT(resizable_buf != nullptr);

  constexpr size_t size = 100;
  allocator.ResizeBuffer(resizable_buf, size,
                         alignment);
  allocator.AllocatePersistentBuffer(size,
                                     alignment);

  TF_LITE_MICRO_EXPECT_GE(allocator.GetUsedBytes(), size * 2);
  TF_LITE_MICRO_EXPECT_LE(allocator.GetUsedBytes(), (size + block_overhead + alignment - 1) * 2);
}

TF_LITE_MICRO_TEST(TestGetUsedBytesTempAllocations) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t alignment = tflite::MicroArenaBufferAlignment();
  size_t block_overhead = heap_get_block_overhead(NULL);

  constexpr size_t size = 100;
  uint8_t* temp = allocator.AllocateTemp(size,
                                         alignment);

  TF_LITE_MICRO_EXPECT_GE(allocator.GetUsedBytes(), size );
  TF_LITE_MICRO_EXPECT_LE(allocator.GetUsedBytes(), size + block_overhead + alignment - 1);

  // Reset temp allocations and ensure GetUsedBytes() is back to the starting
  // size:
  allocator.DeallocateTemp(temp);
  allocator.ResetTempAllocations();

  TF_LITE_MICRO_EXPECT_EQ(allocator.GetUsedBytes(), static_cast<size_t>(0));
}

TF_LITE_MICRO_TEST(TestJustFits) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t alignment = tflite::MicroArenaBufferAlignment();

  size_t size = allocator.GetAvailableMemory(alignment);
  uint8_t* result = allocator.AllocatePersistentBuffer(size, alignment);
  TF_LITE_MICRO_EXPECT(nullptr != result);
}

TF_LITE_MICRO_TEST(TestAligned) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t alignment = tflite::MicroArenaBufferAlignment();

  uint8_t* result = allocator.AllocatePersistentBuffer(1, alignment);
  TF_LITE_MICRO_EXPECT(nullptr != result);

  result = allocator.AllocatePersistentBuffer(16, alignment);
  TF_LITE_MICRO_EXPECT(nullptr != result);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(0),
                          reinterpret_cast<std::uintptr_t>(result) & 3);
}

TF_LITE_MICRO_TEST(TestMultipleTooLarge) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t alignment = tflite::MicroArenaBufferAlignment();

  size_t size = 768;
  uint8_t* result = allocator.AllocatePersistentBuffer(size, alignment);
  TF_LITE_MICRO_EXPECT(nullptr != result);

  size = allocator.GetAvailableMemory(alignment) + 768;
  result = allocator.AllocatePersistentBuffer(size, alignment);
  TF_LITE_MICRO_EXPECT(nullptr == result);
}

TF_LITE_MICRO_TEST(TestTempAllocations) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t alignment = tflite::MicroArenaBufferAlignment();
  size_t block_overhead = heap_get_block_overhead(NULL);
  size_t size = 100;
  uint8_t* temp1 = allocator.AllocateTemp(size, alignment);
  TF_LITE_MICRO_EXPECT(nullptr != temp1);

  uint8_t* temp2 = allocator.AllocateTemp(size, alignment);
  TF_LITE_MICRO_EXPECT(nullptr != temp2);

  // Expect that the next micro allocation is size + block_overhead bytes away from each other.
  TF_LITE_MICRO_EXPECT_EQ(temp2 - temp1, (int) tflite::AlignSizeUp(size + block_overhead, alignment));
}

TF_LITE_MICRO_TEST(TestResetTempAllocations) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t alignment = tflite::MicroArenaBufferAlignment();

  size_t size = 100;

  uint8_t* temp1 = allocator.AllocateTemp(size, alignment);
  TF_LITE_MICRO_EXPECT(nullptr != temp1);

  allocator.DeallocateTemp(temp1);
  allocator.ResetTempAllocations();

  uint8_t* temp2 = allocator.AllocateTemp(size, alignment);
  TF_LITE_MICRO_EXPECT(nullptr != temp2);

  // Reset temp allocations should have the same start address:
  TF_LITE_MICRO_EXPECT_EQ(temp2 - temp1, 0);
}

TF_LITE_MICRO_TEST(TestEnsureHeadSizeWithoutResettingTemp) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t alignment = tflite::MicroArenaBufferAlignment();

  size_t size = 100;

  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, alignment);
  TF_LITE_MICRO_EXPECT(resizable_buf != nullptr);

  uint8_t* temp = allocator.AllocateTemp(size, alignment);
  TF_LITE_MICRO_EXPECT(nullptr != temp);

  // Adjustment to head should fail since temp allocation was not followed by a
  // call to ResetTempAllocations().
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          allocator.ResizeBuffer(resizable_buf, size, alignment));

  allocator.DeallocateTemp(temp);
  allocator.ResetTempAllocations();

  // Reduce head size back to zero.
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          allocator.ResizeBuffer(resizable_buf, 0, alignment));

  // The most recent head allocation should be in the same location as the
  // original temp allocation pointer.
  TF_LITE_MICRO_EXPECT(temp == allocator.GetOverlayMemoryAddress());
}

TF_LITE_MICRO_TEST(TestIsAllTempDeallocated) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t alignment = tflite::MicroArenaBufferAlignment();

  size_t size = 100;

  uint8_t* temp1 = allocator.AllocateTemp(size, alignment);
  TF_LITE_MICRO_EXPECT(allocator.IsAllTempDeallocated() == false);

  uint8_t* temp2 = allocator.AllocateTemp(size, alignment);
  TF_LITE_MICRO_EXPECT(allocator.IsAllTempDeallocated() == false);

  allocator.DeallocateTemp(temp1);
  TF_LITE_MICRO_EXPECT(allocator.IsAllTempDeallocated() == false);

  allocator.DeallocateTemp(temp2);
  TF_LITE_MICRO_EXPECT(allocator.IsAllTempDeallocated() == true);
}

TF_LITE_MICRO_TESTS_END
