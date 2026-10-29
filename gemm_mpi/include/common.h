#ifndef COMMON_H

#define COMMON_H

#define _XOPEN_SOURCE 700
#include "hthread_host.h"
#include <string.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>
#include <omp.h>
#include <sys/types.h> 
#include <fcntl.h>

#define VECTOR_LENGTH 16        // 向量长度
#define threads 24
#define ERROR_THRESHOLD 0.0000001

// 定义复数类型
/*
typedef struct {
    double real;
    double imag;
} Complex;
 */

#endif