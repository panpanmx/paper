#include "../include/common.h"
#include "../include/operator.h"
#include "hthread_host.h"
#include <omp.h>


// 从二进制文件中获取矩阵的大小（字节数）
void get_complex_matrix_dim(int* m, int* n, const char* filename) {
    FILE* file = fopen(filename, "rb");  // 以二进制模式读取文件
    if (file == NULL) {
        printf("无法打开文件 %s\n", filename);
    }

    // 读取矩阵的维度 m 和 n
    size_t result_m = fread(m, sizeof(int), 1, file);
    if (result_m != 1) {
        fprintf(stderr, "Error reading matrix dimensions for m.\n");
    }

    size_t result_n = fread(n, sizeof(int), 1, file);
    if (result_n != 1) {
        fprintf(stderr, "Error reading matrix dimensions for n.\n");
    }

    fclose(file);
}

// 从文件中读取原矩阵
int read_matrix(const char *filename, double *matrix, size_t size) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Could not open file %s for reading\n", filename);
        return EXIT_FAILURE;
    }

    size_t result = fread(matrix, sizeof(double), size, file);
    fclose(file);

    if (result != size) {
        fprintf(stderr, "Error reading from file %s\n", filename);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int pread_matrix(const char *filename, double *matrix, size_t size, off_t offset) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Could not open file %s for reading\n", filename);
        return EXIT_FAILURE;
    }

    ssize_t result = pread(fd, matrix, size, offset);

    if (result != size) {
        fprintf(stderr, "Error reading from file %s\n", filename);
        close(fd);
        return EXIT_FAILURE;
    }
    close(fd);
    return EXIT_SUCCESS;
}

// 读出文件中矩阵的转置矩阵
int read_transpose_matrix(const char *filename, double *matrix, size_t rows, size_t cols) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Could not open file %s for reading\n", filename);
        return EXIT_FAILURE;
    }

    // 读取原矩阵
    double *temp_matrix = (double *)malloc(rows * cols * sizeof(double));
    if (!temp_matrix) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return EXIT_FAILURE;
    }

    size_t result = fread(temp_matrix, sizeof(double), rows * cols, file);
    fclose(file);

    if (result != rows * cols) {
        fprintf(stderr, "Error reading from file %s\n", filename);
        free(temp_matrix);
        return EXIT_FAILURE;
    }

    // 转置矩阵
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[j * rows + i] = temp_matrix[i * cols + j];
        }
    }

    free(temp_matrix);
    return EXIT_SUCCESS;
}

// 将矩阵写入文件的函数
void write_matrix_to_file(const char *filename, double *matrix, int rows, int cols) {
    FILE *file = fopen(filename, "a");
    if (file == NULL) {
        perror("Failed to open file");
        return;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%lf ", matrix[i * cols + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

// 使用CPU计算不分块的GEMM
void gemm_cpu_no_block(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                       int M, int N, int K) {

    // 直接三层循环计算
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            double A_real = A_x[m * K + k];
            double A_imag = A_y[m * K + k];

            for (int n = 0; n < N; n++) {
                double B_real = B_x[k * N + n];
                double B_imag = B_y[k * N + n];
                //double B_real = B_x[n * K + k];  // 访问转置后的实部
                //double B_imag = B_y[n * K + k];  // 访问转置后的虚部
                // 计算C矩阵的实部和虚部
                C_x[m * N + n] += A_real * B_real - A_imag * B_imag;
                C_y[m * N + n] += A_real * B_imag + A_imag * B_real;
            }
        }
    }
}



void block_dma_test(double *A_x, double *A_y, double *B_x, double *B_y, double *C_x, double *C_y,
                    int M, int K, int N, int num_threads, int tg_id){
    unsigned long args[10];
    args[0] = M;
    args[1] = K;
    args[2] = N;
    args[3] = num_threads;
    args[4] = (unsigned long)A_x;
    args[5] = (unsigned long)A_y;
    args[6] = (unsigned long)B_x;
    args[7] = (unsigned long)B_y;
    args[8] = (unsigned long)C_x;
    args[9] = (unsigned long)C_y;

    hthread_group_exec(tg_id, "transfer_test", 4, 6, args);        // 在指定线程组上执行给定的函数
    hthread_group_wait(tg_id);
}

// 生成 m x n 大小的随机实部矩阵
double* generate_real_matrix(int m, int n) {
    double* matrix = (double*)malloc(m * n * sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = (double)(rand() % 100) / 10.0;  // 随机生成实部
        }
    }
    return matrix;
}

// 生成 m x n 大小的随机虚部矩阵
double* generate_imag_matrix(int m, int n) {
    double* matrix = (double*)malloc(m * n * sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = (double)(rand() % 100) / 10.0;  // 随机生成虚部
        }
    }
    return matrix;
}

// 将实部矩阵存储为二进制文件
void save_matrix_to_binary(double* matrix, int m, int n, const char* filename) {
    FILE* file = fopen(filename, "wb");  // 以二进制模式打开文件
    if (file == NULL) {
        printf("无法打开文件 %s\n", filename);
        return;
    }

    // 写入矩阵的维度 m 和 n
    fwrite(&m, sizeof(int), 1, file);
    fwrite(&n, sizeof(int), 1, file);

    // 写入矩阵的数据
    fwrite(matrix, sizeof(double), m * n, file);

    fclose(file);
}

void gemm_dsp_AM_SM_d_opt(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                          int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads,
                          int tg_id){
    unsigned long args[14];
    args[0] = num_threads;
    args[1] = M_block;
    args[2] = K_block;
    args[3] = N_block;
    args[4] = M;
    args[5] = K;
    args[6] = N;
    args[7] = (unsigned long)N_thread;
    args[8] = (unsigned long)A_x;
    args[9] = (unsigned long)A_y;
    args[10] = (unsigned long)B_x;
    args[11] = (unsigned long)B_y;
    args[12] = (unsigned long)C_x;
    args[13] = (unsigned long)C_y;

    hthread_group_exec(tg_id, "matrix_mul_asd_opt", 7, 7, args);
    hthread_group_wait(tg_id);
}


// 使用DSP单缓冲计算GEMM（寄存器分块与指令优化版本）
void gemm_dsp_AM_SM_s_opt(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                          int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads,
                          int tg_id){
    unsigned long args[14];
    args[0] = num_threads;
    args[1] = M_block;
    args[2] = K_block;
    args[3] = N_block;
    args[4] = M;
    args[5] = K;
    args[6] = N;
    args[7] = (unsigned long)N_thread;
    args[8] = (unsigned long)A_x;
    args[9] = (unsigned long)A_y;
    args[10] = (unsigned long)B_x;
    args[11] = (unsigned long)B_y;
    args[12] = (unsigned long)C_x;
    args[13] = (unsigned long)C_y;

    hthread_group_exec(tg_id, "matrix_mul_as_opt", 7, 7, args);
    hthread_group_wait(tg_id);
}

// 使用DSP三缓冲流水线计算GEMM（寄存器分块与指令优化版本）
void gemm_dsp_AM_SM_pipeline_opt(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                                 int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads,
                                 int tg_id){
    unsigned long args[14];
    args[0] = num_threads;
    args[1] = M_block;
    args[2] = K_block;
    args[3] = N_block;
    args[4] = M;
    args[5] = K;
    args[6] = N;
    args[7] = (unsigned long)N_thread;
    args[8] = (unsigned long)A_x;
    args[9] = (unsigned long)A_y;
    args[10] = (unsigned long)B_x;
    args[11] = (unsigned long)B_y;
    args[12] = (unsigned long)C_x;
    args[13] = (unsigned long)C_y;

    hthread_group_exec(tg_id, "matrix_mul_pipeline_opt", 7, 7, args);
    hthread_group_wait(tg_id);
}

