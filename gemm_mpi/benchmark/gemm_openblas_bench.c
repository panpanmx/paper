#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <cblas.h>
#include <openblas_config.h>

#include "../include/common.h"
#include "../include/operator.h"

static double get_wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void run_openblas_4dgemm_ld(int M, int N, int K,
                                   const double *Ar, const double *Ai, int lda,
                                   const double *Br, const double *Bi, int ldb,
                                   double *Cr, double *Ci, int ldc) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, Ar, lda, Br, ldb, 0.0, Cr, ldc);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, -1.0, Ai, lda, Bi, ldb, 1.0, Cr, ldc);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, Ar, lda, Bi, ldb, 0.0, Ci, ldc);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, Ai, lda, Br, ldb, 1.0, Ci, ldc);
}

static void run_openblas_4dgemm(int M, int N, int K,
                                const double *Ar, const double *Ai,
                                const double *Br, const double *Bi,
                                double *Cr, double *Ci) {
    run_openblas_4dgemm_ld(M, N, K, Ar, Ai, K, Br, Bi, N, Cr, Ci, N);
}

static void run_openblas_1zgemm(int M, int N, int K,
                                const double *Ar, const double *Ai,
                                const double *Br, const double *Bi,
                                double *Cr, double *Ci) {
    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;
    double *A_cplx = (double *)malloc(sizeA * 2 * sizeof(double));
    double *B_cplx = (double *)malloc(sizeB * 2 * sizeof(double));
    double *C_cplx = (double *)malloc(sizeC * 2 * sizeof(double));
    if (!A_cplx || !B_cplx || !C_cplx) {
        if (A_cplx) free(A_cplx);
        if (B_cplx) free(B_cplx);
        if (C_cplx) free(C_cplx);
        return;
    }
    for (size_t i = 0; i < sizeA; i++) {
        A_cplx[2 * i]     = Ar[i];
        A_cplx[2 * i + 1] = Ai[i];
    }
    for (size_t i = 0; i < sizeB; i++) {
        B_cplx[2 * i]     = Br[i];
        B_cplx[2 * i + 1] = Bi[i];
    }
    double alpha[2] = {1.0, 0.0};
    double beta[2]  = {0.0, 0.0};
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A_cplx, K, B_cplx, N, beta, C_cplx, N);
    for (size_t i = 0; i < sizeC; i++) {
        Cr[i] = C_cplx[2 * i];
        Ci[i] = C_cplx[2 * i + 1];
    }
    free(A_cplx);
    free(B_cplx);
    free(C_cplx);
}

// Function to run DSP with gemm_dsp_AM_SM_pipeline_opt
static double run_dsp_gemm(int scale, int num_threads,
                           const double *Ac_x, const double *Ac_y,
                           const double *Bc_x, const double *Bc_y,
                           double *dsp_res_x, double *dsp_res_y) {
    int dev_id = 0;
    int M = scale, K = scale, N = scale;
    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;

    int M_block = 56;
    int K_block = 32;
    int N_block = 64;

    double dsp_time = 0.0;

    if (scale <= 8192) {
        int tg_id = hthread_group_create(dev_id, num_threads);
        hthread_group_wait(tg_id);

        double *Ad_x = (double *)hthread_malloc(dev_id, sizeA * sizeof(double), HT_MEM_RO);
        double *Ad_y = (double *)hthread_malloc(dev_id, sizeA * sizeof(double), HT_MEM_RO);
        double *Bd_x = (double *)hthread_malloc(dev_id, sizeB * sizeof(double), HT_MEM_RO);
        double *Bd_y = (double *)hthread_malloc(dev_id, sizeB * sizeof(double), HT_MEM_RO);
        double *local_Cd_x = (double *)hthread_malloc(dev_id, sizeC * sizeof(double), HT_MEM_RO);
        double *local_Cd_y = (double *)hthread_malloc(dev_id, sizeC * sizeof(double), HT_MEM_RO);

        memcpy(Ad_x, Ac_x, sizeA * sizeof(double));
        memcpy(Ad_y, Ac_y, sizeA * sizeof(double));
        memcpy(Bd_x, Bc_x, sizeB * sizeof(double));
        memcpy(Bd_y, Bc_y, sizeB * sizeof(double));
        memset(local_Cd_x, 0, sizeC * sizeof(double));
        memset(local_Cd_y, 0, sizeC * sizeof(double));

        if (num_threads == 24) {
            int *N_thread = (int *)hthread_malloc(dev_id, num_threads * sizeof(int), HT_MEM_RO);
            int N_per_thread = N / num_threads;
            int N_remaining_cols = N % num_threads;
            for (int t = 0; t < num_threads; ++t) {
                N_thread[t] = N_per_thread + (t < N_remaining_cols ? 1 : 0);
            }

            if (scale <= 2048) {
                gemm_dsp_AM_SM_pipeline_opt(local_Cd_x, local_Cd_y, Ad_x, Ad_y, Bd_x, Bd_y,
                                            M_block, K_block, N_block, M, N, K, N_thread, num_threads, tg_id);
                memset(local_Cd_x, 0, sizeC * sizeof(double));
                memset(local_Cd_y, 0, sizeC * sizeof(double));
            }

            double t0 = MPI_Wtime();
            gemm_dsp_AM_SM_pipeline_opt(local_Cd_x, local_Cd_y, Ad_x, Ad_y, Bd_x, Bd_y,
                                        M_block, K_block, N_block, M, N, K, N_thread, num_threads, tg_id);
            dsp_time = MPI_Wtime() - t0;
            hthread_free(N_thread);
        } else {
            // Single-core (num_threads == 1): use sliceN = 512
            int sliceN = (scale < 512) ? scale : 512;
            int *N_thread = (int *)hthread_malloc(dev_id, 1 * sizeof(int), HT_MEM_RO);
            N_thread[0] = sliceN;

            double t0 = MPI_Wtime();
            for (int col = 0; col < N; col += sliceN) {
                gemm_dsp_AM_SM_pipeline_opt(&local_Cd_x[col], &local_Cd_y[col],
                                            Ad_x, Ad_y,
                                            &Bd_x[col], &Bd_y[col],
                                            M_block, K_block, N_block, M, N, K, N_thread, 1, tg_id);
            }
            dsp_time = MPI_Wtime() - t0;
            hthread_free(N_thread);
        }

        if (dsp_res_x) memcpy(dsp_res_x, local_Cd_x, sizeC * sizeof(double));
        if (dsp_res_y) memcpy(dsp_res_y, local_Cd_y, sizeC * sizeof(double));

        hthread_group_destroy(tg_id);
        hthread_free(Ad_x);
        hthread_free(Ad_y);
        hthread_free(Bd_x);
        hthread_free(Bd_y);
        hthread_free(local_Cd_x);
        hthread_free(local_Cd_y);
    } else {
        // scale == 16384: Tile into 4 blocks of 8192 x 16384 x 8192
        int tg_id = hthread_group_create(dev_id, num_threads);
        hthread_group_wait(tg_id);

        int subM = 8192;
        int subK = 16384;
        int subN = 8192;
        size_t sub_sizeA = (size_t)subM * subK;
        size_t sub_sizeB = (size_t)subK * subN;
        size_t sub_sizeC = (size_t)subM * subN;

        double *Ad_x = (double *)hthread_malloc(dev_id, sub_sizeA * sizeof(double), HT_MEM_RO);
        double *Ad_y = (double *)hthread_malloc(dev_id, sub_sizeA * sizeof(double), HT_MEM_RO);
        double *Bd_x = (double *)hthread_malloc(dev_id, sub_sizeB * sizeof(double), HT_MEM_RO);
        double *Bd_y = (double *)hthread_malloc(dev_id, sub_sizeB * sizeof(double), HT_MEM_RO);
        double *Cd_x = (double *)hthread_malloc(dev_id, sub_sizeC * sizeof(double), HT_MEM_RO);
        double *Cd_y = (double *)hthread_malloc(dev_id, sub_sizeC * sizeof(double), HT_MEM_RO);

        double *subB_x = (double *)malloc(sub_sizeB * sizeof(double));
        double *subB_y = (double *)malloc(sub_sizeB * sizeof(double));

        dsp_time = 0.0;
        if (num_threads == 24) {
            int *N_thread = (int *)hthread_malloc(dev_id, num_threads * sizeof(int), HT_MEM_RO);
            int N_per_thread = subN / num_threads;
            int N_remaining_cols = subN % num_threads;
            for (int t = 0; t < num_threads; ++t) {
                N_thread[t] = N_per_thread + (t < N_remaining_cols ? 1 : 0);
            }

            for (int i_t = 0; i_t < 2; i_t++) {
                int row_offset_A = i_t * subM;
                memcpy(Ad_x, &Ac_x[(size_t)row_offset_A * K], sub_sizeA * sizeof(double));
                memcpy(Ad_y, &Ac_y[(size_t)row_offset_A * K], sub_sizeA * sizeof(double));

                for (int j_t = 0; j_t < 2; j_t++) {
                    int col_offset_B = j_t * subN;
                    for (int r = 0; r < subK; r++) {
                        memcpy(&subB_x[r * subN], &Bc_x[(size_t)r * N + col_offset_B], subN * sizeof(double));
                        memcpy(&subB_y[r * subN], &Bc_y[(size_t)r * N + col_offset_B], subN * sizeof(double));
                    }
                    memcpy(Bd_x, subB_x, sub_sizeB * sizeof(double));
                    memcpy(Bd_y, subB_y, sub_sizeB * sizeof(double));
                    memset(Cd_x, 0, sub_sizeC * sizeof(double));
                    memset(Cd_y, 0, sub_sizeC * sizeof(double));

                    double t0 = MPI_Wtime();
                    gemm_dsp_AM_SM_pipeline_opt(Cd_x, Cd_y, Ad_x, Ad_y, Bd_x, Bd_y,
                                                M_block, K_block, N_block, subM, subN, subK, N_thread, num_threads, tg_id);
                    dsp_time += (MPI_Wtime() - t0);

                    if (dsp_res_x) {
                        for (int r = 0; r < subM; r++) {
                            memcpy(&dsp_res_x[(size_t)(row_offset_A + r) * N + col_offset_B], &Cd_x[r * subN], subN * sizeof(double));
                            memcpy(&dsp_res_y[(size_t)(row_offset_A + r) * N + col_offset_B], &Cd_y[r * subN], subN * sizeof(double));
                        }
                    }
                }
            }
            hthread_free(N_thread);
        } else {
            // Single-core (num_threads == 1) for 16384: benchmark 1 tile of 8192 x 16384 x 8192 with sliceN = 512, scaled x4
            int sliceN = 512;
            int *N_thread = (int *)hthread_malloc(dev_id, 1 * sizeof(int), HT_MEM_RO);
            N_thread[0] = sliceN;

            int i_t = 0; // tile 0 (8192 x 16384 x 8192)
            int row_offset_A = i_t * subM;
            memcpy(Ad_x, &Ac_x[(size_t)row_offset_A * K], sub_sizeA * sizeof(double));
            memcpy(Ad_y, &Ac_y[(size_t)row_offset_A * K], sub_sizeA * sizeof(double));

            int j_t = 0;
            int col_offset_B = j_t * subN;
            for (int r = 0; r < subK; r++) {
                memcpy(&subB_x[r * subN], &Bc_x[(size_t)r * N + col_offset_B], subN * sizeof(double));
                memcpy(&subB_y[r * subN], &Bc_y[(size_t)r * N + col_offset_B], subN * sizeof(double));
            }
            memcpy(Bd_x, subB_x, sub_sizeB * sizeof(double));
            memcpy(Bd_y, subB_y, sub_sizeB * sizeof(double));
            memset(Cd_x, 0, sub_sizeC * sizeof(double));
            memset(Cd_y, 0, sub_sizeC * sizeof(double));

            double t0 = MPI_Wtime();
            for (int col = 0; col < subN; col += sliceN) {
                gemm_dsp_AM_SM_pipeline_opt(&Cd_x[col], &Cd_y[col],
                                            Ad_x, Ad_y,
                                            &Bd_x[col], &Bd_y[col],
                                            M_block, K_block, N_block, subM, subN, subK, N_thread, 1, tg_id);
            }
            dsp_time = (MPI_Wtime() - t0) * 4.0;
            hthread_free(N_thread);
        }

        free(subB_x);
        free(subB_y);
        hthread_group_destroy(tg_id);
        hthread_free(Ad_x);
        hthread_free(Ad_y);
        hthread_free(Bd_x);
        hthread_free(Bd_y);
        hthread_free(Cd_x);
        hthread_free(Cd_y);
    }

    return dsp_time;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        printf("Usage: %s <realA> <imagA> <realB> <imagB> <scale>\n", argv[0]);
        return 1;
    }

    MPI_Init(&argc, &argv);

    int scale = atoi(argv[5]);
    int M = scale, K = scale, N = scale;

    double total_flops = 8.0 * (double)M * (double)K * (double)N;
    double gflops_unit = total_flops * 1e-9;

    printf("==========================================================================================\n");
    printf("【实验轮次：Scale %d x %d】\n", M, N);
    printf("  [Matrix Scale: %d x %d x %d] (Complex FP64, Total FLOPs = %.2f GFLOPs)\n", M, K, N, gflops_unit);
    printf("  Data Files: %s | %s\n", argv[1], argv[2]);
    printf("==========================================================================================\n");

    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;

    double *Ac_x = (double *)malloc(sizeA * sizeof(double));
    double *Ac_y = (double *)malloc(sizeA * sizeof(double));
    double *Bc_x = (double *)malloc(sizeB * sizeof(double));
    double *Bc_y = (double *)malloc(sizeB * sizeof(double));
    double *local_Cc_x = (double *)malloc(sizeC * sizeof(double));
    double *local_Cc_y = (double *)malloc(sizeC * sizeof(double));
    double *dsp_res_x  = (double *)malloc(sizeC * sizeof(double));
    double *dsp_res_y  = (double *)malloc(sizeC * sizeof(double));

    if (!Ac_x || !Ac_y || !Bc_x || !Bc_y || !local_Cc_x || !local_Cc_y || !dsp_res_x || !dsp_res_y) {
        printf("Error: host memory allocation failed for scale %d\n", scale);
        MPI_Finalize();
        return 1;
    }

    double t_load_start = get_wall_time();
    if (read_matrix(argv[1], Ac_x, sizeA) != EXIT_SUCCESS) return 1;
    if (read_matrix(argv[2], Ac_y, sizeA) != EXIT_SUCCESS) return 1;
    if (read_matrix(argv[3], Bc_x, sizeB) != EXIT_SUCCESS) return 1;
    if (read_matrix(argv[4], Bc_y, sizeB) != EXIT_SUCCESS) return 1;
    double t_load = get_wall_time() - t_load_start;
    int num_tiles = (scale <= 8192) ? 1 : 4;
    int tile_m = (scale <= 8192) ? scale : 8192;
    int tile_k = scale;
    int tile_n = (scale <= 8192) ? scale : 8192;
    printf("  [Data Load] Loaded binary matrix data in %.4f s (Tiles=%d, Tile shape=%dx%dx%d)\n\n",
           t_load, num_tiles, tile_m, tile_k, tile_n);

    // Open DSP device 0
    int dev_id = 0;
    hthread_dev_open(dev_id);
    hthread_dat_load(dev_id, "kernel_pipeline.dat");

    // 1. Group B: DSP 24 Cores (Single Cluster)
    double t_dsp_24 = run_dsp_gemm(scale, 24, Ac_x, Ac_y, Bc_x, Bc_y, dsp_res_x, dsp_res_y);
    double gflops_dsp_24 = gflops_unit / t_dsp_24;

    // 2. OpenBLAS CPU Multi-core benchmarks
    // CPU 4 Cores (Cluster-Eq)
    openblas_set_num_threads(4);
    double t_cpu_4d_4 = 0.0, gflops_cpu_4d_4 = 0.0, speedup_24_vs_4 = 0.0;
    if (scale <= 8192) {
        double t_cpu4_start = get_wall_time();
        run_openblas_4dgemm(M, N, K, Ac_x, Ac_y, Bc_x, Bc_y, local_Cc_x, local_Cc_y);
        t_cpu_4d_4 = get_wall_time() - t_cpu4_start;
        gflops_cpu_4d_4 = gflops_unit / t_cpu_4d_4;
        speedup_24_vs_4 = t_cpu_4d_4 / t_dsp_24;
    } else {
        // scale == 16384: benchmark 1 tile of 8192x16384x8192 and scale
        double t_tile_start = get_wall_time();
        run_openblas_4dgemm_ld(8192, 8192, 16384, Ac_x, Ac_y, 16384, Bc_x, Bc_y, 16384, local_Cc_x, local_Cc_y, 16384);
        double t_tile = get_wall_time() - t_tile_start;
        t_cpu_4d_4 = t_tile * 4.0;
        gflops_cpu_4d_4 = gflops_unit / t_cpu_4d_4;
        speedup_24_vs_4 = t_cpu_4d_4 / t_dsp_24;
    }

    // CPU 16 Cores (Node)
    openblas_set_num_threads(16);
    double t_cpu_4d_16 = 0.0, gflops_cpu_4d_16 = 0.0;
    if (scale <= 8192) {
        double t_cpu16_start = get_wall_time();
        run_openblas_4dgemm(M, N, K, Ac_x, Ac_y, Bc_x, Bc_y, local_Cc_x, local_Cc_y);
        t_cpu_4d_16 = get_wall_time() - t_cpu16_start;
        gflops_cpu_4d_16 = gflops_unit / t_cpu_4d_16;
    } else {
        // scale == 16384: compute tile (8192 x 8192 x 16384) into top-left of local_Cc_x and scale x4
        double t_cpu16_start = get_wall_time();
        run_openblas_4dgemm_ld(8192, 8192, 16384, Ac_x, Ac_y, 16384, Bc_x, Bc_y, 16384, local_Cc_x, local_Cc_y, 16384);
        double t_tile = get_wall_time() - t_cpu16_start;
        t_cpu_4d_16 = t_tile * 4.0;
        gflops_cpu_4d_16 = gflops_unit / t_cpu_4d_16;
    }
    double speedup_24_vs_16 = t_cpu_4d_16 / t_dsp_24;
    double speedup_node_96_vs_16 = speedup_24_vs_16 * 4.0;

    // CPU 16 Cores (1x zgemm) if scale <= 8192
    double t_cpu_1z_16 = 0.0, gflops_cpu_1z_16 = 0.0, speedup_24_vs_16z = 0.0, speedup_node_96_vs_16z = 0.0;
    if (scale <= 8192) {
        openblas_set_num_threads(16);
        double t_z16_start = get_wall_time();
        run_openblas_1zgemm(M, N, K, Ac_x, Ac_y, Bc_x, Bc_y, local_Cc_x, local_Cc_y);
        t_cpu_1z_16 = get_wall_time() - t_z16_start;
        gflops_cpu_1z_16 = gflops_unit / t_cpu_1z_16;
        speedup_24_vs_16z = t_cpu_1z_16 / t_dsp_24;
        speedup_node_96_vs_16z = speedup_24_vs_16z * 4.0;
    }

    printf("  --- [Group B: Single Cluster 24 DSP Cores vs 4 CPU Cores (Cluster-Eq) & 16 CPU Cores (Node)] (%dx%d) ---\n", M, N);
    printf("  [DSP 24 Cores] gemm_dsp_AM_SM_pipeline_opt : Time = %10.6f s | Throughput = %8.2f GFLOPS\n", t_dsp_24, gflops_dsp_24);
    printf("  [CPU  4 Cores] OpenBLAS (4x dgemm)         : Time = %10.6f s | Throughput = %8.2f GFLOPS | Speedup(DSP24/CPU4)  = %.2fx\n",
           t_cpu_4d_4, gflops_cpu_4d_4, speedup_24_vs_4);
    printf("  [CPU 16 Cores] OpenBLAS (4x dgemm)         : Time = %10.6f s | Throughput = %8.2f GFLOPS | Speedup(DSP24/CPU16) = %.2fx | Speedup(4-Cluster 96DSP/CPU16) = %.2fx\n",
           t_cpu_4d_16, gflops_cpu_4d_16, speedup_24_vs_16, speedup_node_96_vs_16);
    if (scale <= 8192) {
        printf("  [CPU 16 Cores] OpenBLAS (1x zgemm)         : Time = %10.6f s | Throughput = %8.2f GFLOPS | Speedup(DSP24/CPU16) = %.2fx | Speedup(4-Cluster 96DSP/CPU16) = %.2fx\n",
               t_cpu_1z_16, gflops_cpu_1z_16, speedup_24_vs_16z, speedup_node_96_vs_16z);
    }

    // 3. Group A: Single-Core Benchmarks (1 DSP Core vs 1 CPU Core)
    double t_dsp_1 = run_dsp_gemm(scale, 1, Ac_x, Ac_y, Bc_x, Bc_y, NULL, NULL);
    double gflops_dsp_1 = gflops_unit / t_dsp_1;

    // CPU 1 Core (4x dgemm)
    openblas_set_num_threads(1);
    double t_cpu_4d_1 = 0.0, gflops_cpu_4d_1 = 0.0, speedup_1_vs_1d = 0.0;
    if (scale <= 4096) {
        double t_cpu1_start = get_wall_time();
        run_openblas_4dgemm(M, N, K, Ac_x, Ac_y, Bc_x, Bc_y, local_Cc_x, local_Cc_y);
        t_cpu_4d_1 = get_wall_time() - t_cpu1_start;
        gflops_cpu_4d_1 = gflops_unit / t_cpu_4d_1;
        speedup_1_vs_1d = t_cpu_4d_1 / t_dsp_1;
    } else if (scale == 8192) {
        // Benchmark 1/4 slice (2048 x 8192 x 8192) on 1 core to prevent excessive serial runtime
        double *slice_C_x = (double *)malloc((size_t)2048 * 8192 * sizeof(double));
        double *slice_C_y = (double *)malloc((size_t)2048 * 8192 * sizeof(double));
        double t_cpu1_start = get_wall_time();
        run_openblas_4dgemm(2048, 8192, 8192, Ac_x, Ac_y, Bc_x, Bc_y, slice_C_x, slice_C_y);
        double t_sub = get_wall_time() - t_cpu1_start;
        t_cpu_4d_1 = t_sub * 4.0;
        gflops_cpu_4d_1 = gflops_unit / t_cpu_4d_1;
        speedup_1_vs_1d = t_cpu_4d_1 / t_dsp_1;
        free(slice_C_x); free(slice_C_y);
    } else {
        // scale == 16384: benchmark 1 sub-tile (2048 x 2048 x 16384) on 1 core and scale by 64x
        double *slice_C_x = (double *)malloc((size_t)2048 * 2048 * sizeof(double));
        double *slice_C_y = (double *)malloc((size_t)2048 * 2048 * sizeof(double));
        double t_sub_start = get_wall_time();
        run_openblas_4dgemm_ld(2048, 2048, 16384, Ac_x, Ac_y, 16384, Bc_x, Bc_y, 16384, slice_C_x, slice_C_y, 2048);
        double t_sub = get_wall_time() - t_sub_start;
        t_cpu_4d_1 = t_sub * 64.0;
        gflops_cpu_4d_1 = gflops_unit / t_cpu_4d_1;
        speedup_1_vs_1d = t_cpu_4d_1 / t_dsp_1;
        free(slice_C_x); free(slice_C_y);
    }

    // CPU 1 Core (1x zgemm) if scale <= 8192
    double t_cpu_1z_1 = 0.0, gflops_cpu_1z_1 = 0.0, speedup_1_vs_1z = 0.0;
    if (scale <= 4096) {
        openblas_set_num_threads(1);
        double t_z1_start = get_wall_time();
        run_openblas_1zgemm(M, N, K, Ac_x, Ac_y, Bc_x, Bc_y, local_Cc_x, local_Cc_y);
        t_cpu_1z_1 = get_wall_time() - t_z1_start;
        gflops_cpu_1z_1 = gflops_unit / t_cpu_1z_1;
        speedup_1_vs_1z = t_cpu_1z_1 / t_dsp_1;
    } else if (scale == 8192) {
        openblas_set_num_threads(1);
        double *slice_C_x = (double *)malloc((size_t)2048 * 8192 * sizeof(double));
        double *slice_C_y = (double *)malloc((size_t)2048 * 8192 * sizeof(double));
        double t_z1_start = get_wall_time();
        run_openblas_1zgemm(2048, 8192, 8192, Ac_x, Ac_y, Bc_x, Bc_y, slice_C_x, slice_C_y);
        double t_sub = get_wall_time() - t_z1_start;
        t_cpu_1z_1 = t_sub * 4.0;
        gflops_cpu_1z_1 = gflops_unit / t_cpu_1z_1;
        speedup_1_vs_1z = t_cpu_1z_1 / t_dsp_1;
        free(slice_C_x); free(slice_C_y);
    }

    printf("\n  --- [Group A: Single Core 1 DSP Core (gemm_dsp_AM_SM_pipeline_opt) vs 1 CPU Core (OpenBLAS)] (%dx%d) ---\n", M, N);
    printf("  [DSP  1 Core ] gemm_dsp_AM_SM_pipeline_opt : Time = %10.6f s | Throughput = %8.2f GFLOPS\n", t_dsp_1, gflops_dsp_1);
    printf("  [CPU  1 Core ] OpenBLAS (4x dgemm)         : Time = %10.6f s | Throughput = %8.2f GFLOPS | Speedup(DSP1/CPU1_4dgemm) = %.2fx\n",
           t_cpu_4d_1, gflops_cpu_4d_1, speedup_1_vs_1d);
    if (scale <= 8192) {
        printf("  [CPU  1 Core ] OpenBLAS (1x zgemm)         : Time = %10.6f s | Throughput = %8.2f GFLOPS | Speedup(DSP1/CPU1_zgemm)  = %.2fx\n",
               t_cpu_1z_1, gflops_cpu_1z_1, speedup_1_vs_1z);
    }

    // 4. Accuracy Check (DSP vs OpenBLAS)
    double err_x = 0.0, err_y = 0.0;
    size_t check_count = 0;
    if (scale <= 8192) {
        check_count = sizeC;
        for (size_t i = 0; i < sizeC; i++) {
            double diff_x = fabs(dsp_res_x[i] - local_Cc_x[i]);
            double diff_y = fabs(dsp_res_y[i] - local_Cc_y[i]);
            if (diff_x > err_x) err_x = diff_x;
            if (diff_y > err_y) err_y = diff_y;
        }
    } else {
        // scale == 16384: verify tile 0 (8192 x 8192)
        check_count = (size_t)8192 * 8192;
        for (int r = 0; r < 8192; r++) {
            for (int c = 0; c < 8192; c++) {
                size_t idx = (size_t)r * N + c;
                double diff_x = fabs(dsp_res_x[idx] - local_Cc_x[idx]);
                double diff_y = fabs(dsp_res_y[idx] - local_Cc_y[idx]);
                if (diff_x > err_x) err_x = diff_x;
                if (diff_y > err_y) err_y = diff_y;
            }
        }
    }
    printf("\n  [Accuracy Check] Max Diff (DSP gemm_dsp_AM_SM_pipeline_opt vs OpenBLAS, verified %zu elements): err_x = %.3e, err_y = %.3e\n\n",
           check_count, err_x, err_y);

    hthread_dev_close(dev_id);

    free(Ac_x); free(Ac_y); free(Bc_x); free(Bc_y);
    free(local_Cc_x); free(local_Cc_y); free(dsp_res_x); free(dsp_res_y);

    MPI_Finalize();
    return 0;
}
