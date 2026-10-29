#ifndef OPERATOR_H

#define OPERATOR_H

#include "./common.h"

void get_complex_matrix_dim(int* m, int* n, const char* filename);

int read_matrix(const char *filename, double *matrix, size_t size);

int pread_matrix(const char *filename, double *matrix, size_t size, off_t offset);

int read_transpose_matrix(const char *filename, double *matrix, size_t rows, size_t cols);

void write_matrix_to_file(const char *filename, double *matrix, int rows, int cols);

void block_dma_test(double *A_x, double *A_y, double *B_x, double *B_y, double *C_x, double *C_y,
                    int M, int K, int N, int num_threads, int tg_id);

void gemm_cpu_no_block(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                       int M, int N, int K);

void gemm_cpu(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
              int M_block, int K_block, int N_block, int M, int N, int K);

void gemm_dsp_AM_SM_gs(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                      int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads,
                      int b_id, int tg_id);

void gemm_dsp_AM_SM_s(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                      int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads,
                      int tg_id);

void gemm_dsp_AM_SM_d(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                      int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads,
                      int tg_id);

void gemm_dsp_AM_SM_triple(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                      int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads,
                      int tg_id);

void gemm_dsp_AM_SM_pipeline(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                      int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads,
                      int tg_id);
                      
void gemm_dsp_AM_SM_pipeline2(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                      int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads,
                      int tg_id);

void gemm_dsp_AM_SM_gd(int b_id, double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                      int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads,
                      int tg_id);

void gemm_dsp_AM(double* reduce_x, double* reduce_y,
                double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                int M_block, int K_block, int N_block, int M, int N, int K, int* M_thread, int num_threads,
                int tg_id);

double* generate_real_matrix(int m, int n);
double* generate_imag_matrix(int m, int n);

void save_matrix_to_binary(double* matrix, int m, int n, const char* filename);

void gemm_dsp_B_s(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                      int M_block, int K_block, int N_block, int M, int N, int K, int* M_thread, int num_threads,
                      int tg_id);
//GSM
void gemm_dsp_G(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                      int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads,
                      int b_id, int tg_id);

//实数
void gemm_cpu_real(double* C, double* A, double* B, int M, int N, int K);

void gemm_dsp_real_pipeline(double* C, double* A, double* B, int M_block, int K_block, int N_block, int M, int K, int N, int* N_thread, int num_threads, int tg_id);

void gemm_dsp_real_as(double* C, double* A, double* B, int M_block, int K_block, int N_block, int M, int K, int N, int* N_thread, int num_threads, int tg_id);
void gemm_dsp_AM_SM_d_opt(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y,
                          int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads,
                          int tg_id);

void gemm_dsp_AM_SM_d_opt_2d(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y, int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads, int tg_id);
void gemm_dsp_AM_SM_d_opt_2d_v2(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y, int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads, int tg_id);
void gemm_dsp_AM_SM_d_opt_v2(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y, int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads, int tg_id);
void gemm_dsp_AM_SM_d_opt_v3(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y, int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads, int tg_id);
void gemm_dsp_AM_SM_d_opt_v4(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y, int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads, int tg_id);
void gemm_dsp_AM_SM_d_opt_v5(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y, int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads, int tg_id);
void gemm_dsp_AM_SM_s_opt(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y, int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads, int tg_id);
void gemm_dsp_AM_SM_pipeline_opt(double* C_x, double* C_y, double* A_x, double* A_y, double* B_x, double* B_y, int M_block, int K_block, int N_block, int M, int N, int K, int* N_thread, int num_threads, int tg_id);

#endif

