#include "tensorflow/lite/micro/arena_allocator/ubi_heap_buffer_allocator.h"

#include <cstdint>

#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

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

  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, 4);
  TF_LITE_MICRO_EXPECT(resizable_buf != nullptr);

  // First head adjustment of 100 bytes (aligned 12):
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, /*size=*/100, /*alignment=*/4));

  // Offset alignment of 12 can lead to allocation within 8 byte range of
  // requested bytes based to arena alignment at runtime:
  TF_LITE_MICRO_EXPECT_GE(allocator.GetNonPersistentUsedBytes(), 100);
  TF_LITE_MICRO_EXPECT_LE(allocator.GetNonPersistentUsedBytes(), 100 + 3);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, /*size=*/10, /*alignment=*/4));
  TF_LITE_MICRO_EXPECT_GE(allocator.GetNonPersistentUsedBytes(), 10);
  TF_LITE_MICRO_EXPECT_LE(allocator.GetNonPersistentUsedBytes(), 100 + 3);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      allocator.ResizeBuffer(resizable_buf, /*size=*/1000, /*alignment=*/4));
  TF_LITE_MICRO_EXPECT_GE(allocator.GetNonPersistentUsedBytes(), 1000);
  TF_LITE_MICRO_EXPECT_LE(allocator.GetNonPersistentUsedBytes(), 1000 + 3);
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
