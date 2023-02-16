/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ubinos.h>

#include "image_provider.h"
#include "model_settings.h"

#if (TFLITE_MICRO__PERSON_DETECTION_IMAGE_PROVIDER_TYPE == TFLITE_MICRO__PERSON_DETECTION_IMAGE_PROVIDER_TYPE__TEST01)
#include "testdata/no_person_image_data.h"
#include "testdata/person_image_data.h"

#include <string.h>

static int _g_get_image_count = 0;
#endif

TfLiteStatus GetImage(int image_width, int image_height, int channels,
                      int8_t* image_data) {
#if   (TFLITE_MICRO__PERSON_DETECTION_IMAGE_PROVIDER_TYPE == TFLITE_MICRO__PERSON_DETECTION_IMAGE_PROVIDER_TYPE__DUMMY)
  for (int i = 0; i < image_width * image_height * channels; ++i) {
    image_data[i] = 0;
  }
  return kTfLiteOk;
#elif (TFLITE_MICRO__PERSON_DETECTION_IMAGE_PROVIDER_TYPE == TFLITE_MICRO__PERSON_DETECTION_IMAGE_PROVIDER_TYPE__TEST01)
  if (_g_get_image_count % 2 == 0)
  {
    memcpy(image_data, g_person_image_data, image_width * image_height * channels);
  }
  else
  {
    memcpy(image_data, g_no_person_image_data, image_width * image_height * channels);
  }
  _g_get_image_count++;
  return kTfLiteOk;
#else
#error "Unsupported TFLITE_MICRO__PERSON_DETECTION_IMAGE_PROVIDER_TYPE"
#endif
}
