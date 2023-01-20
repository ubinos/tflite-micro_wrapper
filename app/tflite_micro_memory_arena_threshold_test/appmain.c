/*
 * Copyright (c) 2023 Sung Ho Park and CSOS
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ubinos.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gtest_main.h"

int appmain(int argc, char * argv[])
{
    int r;

    r = gtest_main(argc, argv);

    return r;
}
