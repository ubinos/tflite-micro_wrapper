#include <ubinos.h>
#include <ubinos/ubiclib/heap.h>

#include "tensorflow/lite/micro/arena_allocator/ubi_heap_buffer_allocator.h"

#include <cstdint>

#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/memory_helpers.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestEnsureHeadSizeSimpleAlignment) {
  tflite::UbiHeapBufferAllocator allocator;

  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, 1);
  TF_LITE_MICRO_EXPECT(resizable_buf != nullptr);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, /*size=*/100, /*alignment=*/1));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(100),
                          allocator.GetNonPersistentUsedBytes());

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, /*size=*/10, /*alignment=*/1));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(10),
                          allocator.GetNonPersistentUsedBytes());

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, /*size=*/1000, /*alignment=*/1));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(1000),
                          allocator.GetNonPersistentUsedBytes());
}

TF_LITE_MICRO_TEST(TestAdjustHeadSizeMisalignment) {
  tflite::UbiHeapBufferAllocator allocator;
  size_t allocation_size;
  size_t alignment = tflite::MicroArenaBufferAlignment();
  // size_t block_overhead = heap_get_block_overhead(NULL);
  // size_t arena_size = allocator.GetAvailableMemory(1);

  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, alignment);
  TF_LITE_MICRO_EXPECT(resizable_buf != nullptr);

  // First head adjustment of 100 bytes (aligned alignment):
  allocation_size = 100; // tflite::AlignSizeUp(100 + block_overhead, alignment) - block_overhead;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, allocation_size, alignment));

  // Offset alignment can lead to allocation within 8 byte range of
  // requested bytes based to arena alignment at runtime:
  TF_LITE_MICRO_EXPECT_GE(allocator.GetNonPersistentUsedBytes(), allocation_size);
  TF_LITE_MICRO_EXPECT_LE(allocator.GetNonPersistentUsedBytes(), allocation_size + alignment - 1);

  allocation_size = 10; // tflite::AlignSizeUp(10 + block_overhead, alignment) - block_overhead;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, allocation_size, alignment));
  TF_LITE_MICRO_EXPECT_GE(allocator.GetNonPersistentUsedBytes(), allocation_size);
  TF_LITE_MICRO_EXPECT_LE(allocator.GetNonPersistentUsedBytes(), allocation_size + alignment - 1);

  allocation_size = 1000; // tflite::AlignSizeUp(1000 + block_overhead, alignment) - block_overhead;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, allocation_size, alignment));
  TF_LITE_MICRO_EXPECT_GE(allocator.GetNonPersistentUsedBytes(), allocation_size);
  TF_LITE_MICRO_EXPECT_LE(allocator.GetNonPersistentUsedBytes(), allocation_size + alignment - 1);
}

TF_LITE_MICRO_TEST(TestAdjustHeadSizeMisalignedHandlesCorrectBytesAvailable) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t block_overhead = heap_get_block_overhead(NULL);
  size_t arena_size = allocator.GetAvailableMemory(1);

  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, 16);
  TF_LITE_MICRO_EXPECT(resizable_buf != nullptr);

  // First head adjustment of 100 bytes (aligned 16):
  size_t size = tflite::AlignSizeUp(100 + block_overhead, tflite::MicroArenaBufferAlignment()) - block_overhead;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, /*size=*/size, /*alignment=*/16));

  // allocator.GetAvailableMemory() should also report the actual amount of
  // memory available based on a requested offset (16):
  size_t aligned_available_bytes =
      allocator.GetAvailableMemory(/*alignment=*/16);
  TF_LITE_MICRO_EXPECT_LE(aligned_available_bytes, arena_size - size - block_overhead);
  TF_LITE_MICRO_EXPECT_GE(aligned_available_bytes, arena_size - size - block_overhead - 16);

  size = tflite::AlignSizeUp(10 + block_overhead, tflite::MicroArenaBufferAlignment()) - block_overhead;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, size, /*alignment=*/16));
  aligned_available_bytes = allocator.GetAvailableMemory(/*alignment=*/16);
  TF_LITE_MICRO_EXPECT_LE(aligned_available_bytes, arena_size - size - block_overhead);
  TF_LITE_MICRO_EXPECT_GE(aligned_available_bytes, arena_size - size - block_overhead - 16);

  size = tflite::AlignSizeUp(1000 + block_overhead, tflite::MicroArenaBufferAlignment()) - block_overhead;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, size, /*alignment=*/16));
  aligned_available_bytes = allocator.GetAvailableMemory(/*alignment=*/16);
  TF_LITE_MICRO_EXPECT_LE(aligned_available_bytes, arena_size - size - block_overhead);
  TF_LITE_MICRO_EXPECT_GE(aligned_available_bytes, arena_size - size - block_overhead - 16);
}

TF_LITE_MICRO_TEST(TestGetAvailableMemory) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t block_overhead = heap_get_block_overhead(NULL);
  size_t arena_size = allocator.GetAvailableMemory(1);

  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, 1);
  TF_LITE_MICRO_EXPECT(resizable_buf != nullptr);

  size_t allocation_size = tflite::AlignSizeUp(100 + block_overhead, tflite::MicroArenaBufferAlignment()) - block_overhead;;
  allocator.ResizeBuffer(resizable_buf, /*size=*/allocation_size,
                         /*alignment=*/1);
  allocator.AllocatePersistentBuffer(/*size=*/allocation_size,
                                     /*alignment=*/1);

  TF_LITE_MICRO_EXPECT_EQ(allocator.GetAvailableMemory(/*alignment=*/1),
                          arena_size - (allocation_size + block_overhead) * 2);
}

TF_LITE_MICRO_TEST(TestGetAvailableMemoryWithTempAllocations) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t block_overhead = heap_get_block_overhead(NULL);
  size_t arena_size = allocator.GetAvailableMemory(1);

  size_t allocation_size = tflite::AlignSizeUp(100 + block_overhead, tflite::MicroArenaBufferAlignment()) - block_overhead;;
  uint8_t* temp = allocator.AllocateTemp(/*size=*/allocation_size,
                                         /*alignment=*/1);

  TF_LITE_MICRO_EXPECT_EQ(allocator.GetAvailableMemory(/*alignment=*/1),
                          arena_size - (allocation_size + block_overhead));

  // Reset temp allocations and ensure GetAvailableMemory() is back to the
  // starting size:
  allocator.DeallocateTemp(temp);
  allocator.ResetTempAllocations();

  TF_LITE_MICRO_EXPECT_EQ(allocator.GetAvailableMemory(/*alignment=*/1),
                          arena_size);
}

TF_LITE_MICRO_TEST(TestGetUsedBytes) {
  tflite::UbiHeapBufferAllocator allocator;

  TF_LITE_MICRO_EXPECT_EQ(allocator.GetUsedBytes(), static_cast<size_t>(0));
  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, 1);
  TF_LITE_MICRO_EXPECT(resizable_buf != nullptr);

  constexpr size_t allocation_size = 100;
  allocator.ResizeBuffer(resizable_buf, /*size=*/allocation_size,
                         /*alignment=*/1);
  allocator.AllocatePersistentBuffer(/*size=*/allocation_size,
                                     /*alignment=*/1);

  TF_LITE_MICRO_EXPECT_EQ(allocator.GetUsedBytes(), allocation_size * 2);
}

TF_LITE_MICRO_TEST(TestGetUsedBytesTempAllocations) {
  tflite::UbiHeapBufferAllocator allocator;

  constexpr size_t allocation_size = 100;
  uint8_t* temp = allocator.AllocateTemp(/*size=*/allocation_size,
                                         /*alignment=*/1);

  TF_LITE_MICRO_EXPECT_EQ(allocator.GetUsedBytes(), allocation_size);

  // Reset temp allocations and ensure GetUsedBytes() is back to the starting
  // size:
  allocator.DeallocateTemp(temp);
  allocator.ResetTempAllocations();

  TF_LITE_MICRO_EXPECT_EQ(allocator.GetUsedBytes(), static_cast<size_t>(0));
}

TF_LITE_MICRO_TEST(TestJustFits) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t size = allocator.GetAvailableMemory(1);
  uint8_t* result = allocator.AllocatePersistentBuffer(size, 1);
  TF_LITE_MICRO_EXPECT(nullptr != result);
}

TF_LITE_MICRO_TEST(TestAligned) {
  tflite::UbiHeapBufferAllocator allocator;

  uint8_t* result = allocator.AllocatePersistentBuffer(1, 1);
  TF_LITE_MICRO_EXPECT(nullptr != result);

  result = allocator.AllocatePersistentBuffer(16, 4);
  TF_LITE_MICRO_EXPECT(nullptr != result);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(0),
                          reinterpret_cast<std::uintptr_t>(result) & 3);
}

TF_LITE_MICRO_TEST(TestMultipleTooLarge) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t size = 768;
  uint8_t* result = allocator.AllocatePersistentBuffer(size, 1);
  TF_LITE_MICRO_EXPECT(nullptr != result);

  size = allocator.GetAvailableMemory(1) + 768;
  result = allocator.AllocatePersistentBuffer(size, 1);
  TF_LITE_MICRO_EXPECT(nullptr == result);
}

TF_LITE_MICRO_TEST(TestTempAllocations) {
  tflite::UbiHeapBufferAllocator allocator;

  size_t block_overhead = heap_get_block_overhead(NULL);
  size_t size = tflite::AlignSizeUp(100 + block_overhead, tflite::MicroArenaBufferAlignment()) - block_overhead;
  uint8_t* temp1 = allocator.AllocateTemp(size, 1);
  TF_LITE_MICRO_EXPECT(nullptr != temp1);

  uint8_t* temp2 = allocator.AllocateTemp(size, 1);
  TF_LITE_MICRO_EXPECT(nullptr != temp2);

  // Expect that the next micro allocation is size + block_overhead bytes away from each other.
  TF_LITE_MICRO_EXPECT_EQ(temp2 - temp1, (int) (size + block_overhead));
}

TF_LITE_MICRO_TEST(TestResetTempAllocations) {
  tflite::UbiHeapBufferAllocator allocator;

  uint8_t* temp1 = allocator.AllocateTemp(100, 1);
  TF_LITE_MICRO_EXPECT(nullptr != temp1);

  allocator.DeallocateTemp(temp1);
  allocator.ResetTempAllocations();

  uint8_t* temp2 = allocator.AllocateTemp(100, 1);
  TF_LITE_MICRO_EXPECT(nullptr != temp2);

  // Reset temp allocations should have the same start address:
  TF_LITE_MICRO_EXPECT_EQ(temp2 - temp1, 0);
}

TF_LITE_MICRO_TEST(TestEnsureHeadSizeWithoutResettingTemp) {
  tflite::UbiHeapBufferAllocator allocator;

  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, 1);
  TF_LITE_MICRO_EXPECT(resizable_buf != nullptr);

  uint8_t* temp = allocator.AllocateTemp(100, 1);
  TF_LITE_MICRO_EXPECT(nullptr != temp);

  // Adjustment to head should fail since temp allocation was not followed by a
  // call to ResetTempAllocations().
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          allocator.ResizeBuffer(resizable_buf, 100, 1));

  allocator.DeallocateTemp(temp);
  allocator.ResetTempAllocations();

  // Reduce head size back to zero.
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          allocator.ResizeBuffer(resizable_buf, 0, 1));

  // The most recent head allocation should be in the same location as the
  // original temp allocation pointer.
  TF_LITE_MICRO_EXPECT(temp == allocator.GetOverlayMemoryAddress());
}

TF_LITE_MICRO_TEST(TestIsAllTempDeallocated) {
  tflite::UbiHeapBufferAllocator allocator;

  uint8_t* temp1 = allocator.AllocateTemp(100, 1);
  TF_LITE_MICRO_EXPECT(allocator.IsAllTempDeallocated() == false);

  uint8_t* temp2 = allocator.AllocateTemp(100, 1);
  TF_LITE_MICRO_EXPECT(allocator.IsAllTempDeallocated() == false);

  allocator.DeallocateTemp(temp1);
  TF_LITE_MICRO_EXPECT(allocator.IsAllTempDeallocated() == false);

  allocator.DeallocateTemp(temp2);
  TF_LITE_MICRO_EXPECT(allocator.IsAllTempDeallocated() == true);
}

TF_LITE_MICRO_TESTS_END
