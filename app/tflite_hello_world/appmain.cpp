/*
 * Copyright (c) 2021 Sung Ho Park and CSOS
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ubinos.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ino/main_functions.h"

static void tflite_func(void * arg);

int appmain(int argc, char * argv[])
{
    int r;
    (void) r;

    r = task_create(NULL, tflite_func, NULL, task_getlowestpriority(), 0, "tflite");
    ubi_assert(r == 0);

    ubik_comp_start();

    return 0;
}

static void tflite_func(void * arg)
{
    printf("\n\n\n");
    printf("================================================================================\n");
    printf("tflite_hello_world (build time: %s %s)\n", __TIME__, __DATE__);
    printf("================================================================================\n");
    printf("\n");

    setup();
    for (unsigned int i = 0; ; i++)
    {
        loop();
    }
}

