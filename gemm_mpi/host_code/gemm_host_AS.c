#include "../include/common.h"
#include "../include/operator.h"
#include <stdlib.h> 

int main(int argc, char **argv)
{
    char hostname[100];
    int commSize, myRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);//commSize总进程数
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    gethostname(hostname,sizeof(hostname));
    //printf("Hello world from process %d of %d: host: %s\n", myRank, commSize, hostname);

    /*** 主机打开计算设备、创建DSP线程组 ***/
    int dev_id = myRank % 4;
    // int num_threads = atoi(argv[5]);                     // 24      一个线程组的线程数 / 计算单元 / 核数
    int num_threads = 24;

    hthread_dev_open(dev_id);
    hthread_dat_load(dev_id, "kernel_AS.dat");

    int tg_id = hthread_group_create(dev_id, num_threads);      // 在指定 DSP 簇上创建 DSP 线程组
	// printf("tg_id = %d \n", tg_id);      // tg_id = 0
	hthread_group_wait(tg_id);

	int b_id = hthread_barrier_create(dev_id);
    int lock_id = hthread_rwlock_create(dev_id);

    int M, K, N;
    get_complex_matrix_dim(&M, &K, argv[1]);
    get_complex_matrix_dim(&K, &N, argv[3]);
    if(myRank == 0) printf("M = %d, K = %d, N = %d\n", M, K, N);

    double* Ac_x = (double*)malloc(M * K * sizeof(double));
    double* Ac_y = (double*)malloc(M * K * sizeof(double));

    double* Bc_x = (double*)malloc(K * N * sizeof(double));
	double* Bc_y = (double*)malloc(K * N * sizeof(double));

	double* Bd_x = (double*)hthread_malloc(dev_id, K * N * sizeof(double), HT_MEM_RO);
	double* Bd_y = (double*)hthread_malloc(dev_id, K * N * sizeof(double), HT_MEM_RO);

	double* Cc_x = NULL;
	double* Cc_y = NULL;

	double* Cd_x = NULL;
	double* Cd_y = NULL;

    //对B进行划分
    int N_per_thread = N / num_threads;      // B基本cols
    int N_remaining_cols = N % num_threads;       // B剩余cols

    if(myRank == 0) printf("N_remaining_cols : %d\n", N_remaining_cols);

    // 存储每个线程的行数
    int * N_thread = (int*)hthread_malloc(dev_id, num_threads * sizeof(int), HT_MEM_RO);
    // 分配行数
    for (int t = 0; t < num_threads; ++t) {
        N_thread[t] = N_per_thread + (t < N_remaining_cols ? 1 : 0);
        // if(myRank == 0) printf("N_thread[%d] : %d\n", t, N_thread[t]);
    }

    int local_A_src[commSize];
    int local_C_src[commSize];
    int local_A_counts[commSize];
    int local_C_counts[commSize];

    //对A进行划分
    int M_per_process = M / commSize;     // A基本行数
    int M_remaining_rows = M % commSize;       // A剩余行数

    int M_last_sum;
    int local_M[commSize];

    for (int t = 0; t < commSize; t++) {
        M_last_sum = t * M_per_process + (t < M_remaining_rows ? t : M_remaining_rows);
        local_M[t] = (M_per_process + (t < M_remaining_rows ? 1 : 0));
        local_A_src[t] = K * M_last_sum;
        local_C_src[t] = N * M_last_sum;
        local_A_counts[t] = K * local_M[t];
        local_C_counts[t] = N * local_M[t];

        // if(myRank == 0){
        //     printf("local_M[%d] : %d\n", t, local_M[t]);
        //     printf("M_last_sum[%d] : %d\n", t, M_last_sum);
        // } 
    }

    if(myRank == 0) {
        printf("M_remaining_rows : %d\n", M_remaining_rows);

        Cc_x = (double*)malloc(M * N * sizeof(double));
        Cc_y = (double*)malloc(M * N * sizeof(double));

        Cd_x = (double*)hthread_malloc(dev_id, M * N * sizeof(double), HT_MEM_RW);
        Cd_y = (double*)hthread_malloc(dev_id, M * N * sizeof(double), HT_MEM_RW);
    }

    double* local_Ac_x = (double*)malloc(local_A_counts[myRank] * sizeof(double));
    double* local_Ac_y = (double*)malloc(local_A_counts[myRank] * sizeof(double));

    double* local_Ad_x = (double*)hthread_malloc(dev_id , local_A_counts[myRank] * sizeof(double), HT_MEM_RO);
    double* local_Ad_y = (double*)hthread_malloc(dev_id , local_A_counts[myRank] * sizeof(double), HT_MEM_RO);

    // ??????? C ?????
    double* local_Cc_x = (double*)malloc(local_C_counts[myRank] * sizeof(double));
    double* local_Cc_y = (double*)malloc(local_C_counts[myRank] * sizeof(double));

    double* local_Cd_x = (double*)hthread_malloc(dev_id , local_C_counts[myRank] * sizeof(double), HT_MEM_RW);
    double* local_Cd_y = (double*)hthread_malloc(dev_id , local_C_counts[myRank] * sizeof(double), HT_MEM_RW);

    double io_start_time, io_end_time, io_time_cost, max_io_time_cost;
    double t_sub0, t_sub1, only_io_time = 0.0, only_comm_time = 0.0, max_only_io = 0.0, max_only_comm = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    io_start_time = MPI_Wtime();

    t_sub0 = MPI_Wtime();
    if(myRank == 0) {
        if(read_matrix(argv[1], Ac_x, M * K) != EXIT_SUCCESS) return EXIT_FAILURE;
        if(read_matrix(argv[2], Ac_y, M * K) != EXIT_SUCCESS) return EXIT_FAILURE;
        if(read_matrix(argv[3], Bc_x, K * N) != EXIT_SUCCESS) return EXIT_FAILURE;
        if(read_matrix(argv[4], Bc_y, K * N) != EXIT_SUCCESS) return EXIT_FAILURE;
    }
    t_sub1 = MPI_Wtime();
    only_io_time += (t_sub1 - t_sub0);

    t_sub0 = MPI_Wtime();
    MPI_Bcast(Bc_x, K * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Bc_y, K * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(Ac_x, M * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Ac_y, M * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    memcpy(Bd_x, Bc_x, K * N * sizeof(double));
    memcpy(Bd_y, Bc_y, K * N * sizeof(double));
    t_sub1 = MPI_Wtime();
    only_comm_time += (t_sub1 - t_sub0);

    t_sub0 = MPI_Wtime();
    if(pread_matrix(argv[1], local_Ac_x, local_A_counts[myRank] * sizeof(double), local_A_src[myRank] * sizeof(double)) != EXIT_SUCCESS) return EXIT_FAILURE; 
    if(pread_matrix(argv[2], local_Ac_y, local_A_counts[myRank] * sizeof(double), local_A_src[myRank] * sizeof(double)) != EXIT_SUCCESS) return EXIT_FAILURE; 

    memcpy(local_Ad_x, local_Ac_x, local_A_counts[myRank] * sizeof(double));
    memcpy(local_Ad_y, local_Ac_y, local_A_counts[myRank] * sizeof(double));
    t_sub1 = MPI_Wtime();
    only_io_time += (t_sub1 - t_sub0);

    io_end_time = MPI_Wtime();
    io_time_cost = io_end_time - io_start_time;
    MPI_Reduce(&only_io_time, &max_only_io, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&only_comm_time, &max_only_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&io_time_cost, &max_io_time_cost, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(myRank == 0) {
        printf("time: io = %lf\n", max_only_io);
        printf("time: comm = %lf\n", max_only_comm);
        printf("time: comm_io = %lf\n", max_io_time_cost);
    }

    int M_block = 56;
    int K_block = 32;
    int N_block = 64;       // 必须是16的倍数


//     if(myRank == 0){
//         printf("M_block = %d, K_block = %d, N_block = %d.\n", M_block, K_block, N_block);
//     }

    double start_time, end_time, time_cost, max_time_cost;

    // Initialize C to zero
    for (int i = 0; i < local_C_counts[myRank]; i++) {
        local_Cc_x[i] = 0.0;
        local_Cc_y[i] = 0.0;
    }

    //** CPU运算 **
//     start_time = MPI_Wtime();
//     gemm_cpu_no_block(local_Cc_x, local_Cc_y, local_Ac_x, local_Ac_y, Bc_x, Bc_y, local_M[myRank], N, K);
//     end_time = MPI_Wtime();
//     time_cost = end_time - start_time;
//     MPI_Reduce(&time_cost, &max_time_cost, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
// 
//     if(myRank == 0){
//         printf("time: cpu = %lf\n", max_time_cost);
//     }

//     // Initialize C to zero
    for (int i = 0; i < local_C_counts[myRank]; i++) {
        local_Cd_x[i] = 0.0;
        local_Cd_y[i] = 0.0;
    }

//     /*** DSP AM + SM 运算 ***/
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    // gemm_dsp_AM_SM_pipeline(local_Cd_x, local_Cd_y, local_Ad_x, local_Ad_y, Bd_x,Bd_y,
    //                     M_block, K_block, N_block, local_M[myRank], N, K, N_thread, num_threads, tg_id);
// //    gemm_dsp_AM_SM_gs(lock_id, local_Cd_x, local_Cd_y, local_Ad_x, local_Ad_y, Bd_x, Bd_y,
// //                     M_block, K_block, N_block, local_M[myRank], N, K, N_thread, num_threads, tg_id);
// //    gemm_dsp_AM_SM_d(local_Cd_x, local_Cd_y, local_Ad_x, local_Ad_y, Bd_x, Bd_y,
// //                     M_block, K_block, N_block, local_M[myRank], N, K, N_thread, num_threads, tg_id);
    gemm_dsp_AM_SM_d_opt(local_Cd_x, local_Cd_y, local_Ad_x, local_Ad_y, Bd_x, Bd_y,
                    M_block, K_block, N_block, local_M[myRank], N, K, N_thread, num_threads, tg_id);
// //    gemm_dsp_AM_SM_gs(local_Cd_x, local_Cd_y, local_Ad_x, local_Ad_y, Bd_x, Bd_y,
// //                     M_G, K_G, N_G, M_block, local_M[myRank], N, K, N_thread, num_threads, tg_id);
    end_time = MPI_Wtime();

    time_cost = end_time - start_time;
    MPI_Reduce(&time_cost, &max_time_cost, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(myRank == 0){
        printf("time: AM_SM_d_opt = %lf\n", max_time_cost);
    }

//     double error_x_local = 0.0;
//     double error_y_local = 0.0;
//     for(int i = 0; i < local_M[myRank] * N; i++){
//         error_x_local += abs(local_Cd_x[i] - local_Cc_x[i]);
//         error_y_local += abs(local_Cd_y[i] - local_Cc_y[i]);
//     }
    
//     double error_x_global;
//     double error_y_global;
    
//     MPI_Reduce(&error_x_local, &error_x_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//     MPI_Reduce(&error_y_local, &error_y_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
//     if (myRank == 0) {
//         printf("error_x_global = %le   error_y_global = %le\n", error_x_global, error_y_global);
//     }


    /*** 主机关闭计算设备、销毁DSP线程组 ***/
    hthread_group_destroy(tg_id);        // 释放缓冲区和上下文

    hthread_barrier_destroy(b_id);
    hthread_rwlock_destroy(lock_id);
    
    hthread_free(Ac_x);
    hthread_free(Ac_y);
    hthread_free(Bc_x);
    hthread_free(Bc_y);
    hthread_free(Bd_x);
    hthread_free(Bd_y);
    if(myRank == 0){
        free(Cc_x);
        free(Cc_y);
        hthread_free(Cd_x);
        hthread_free(Cd_y);
    }
    hthread_free(local_Ac_x);
    hthread_free(local_Ac_y);
    hthread_free(local_Ad_x);
    hthread_free(local_Ad_y);
    hthread_free(local_Cc_x);
    hthread_free(local_Cc_y);
    hthread_free(local_Cd_x);
    hthread_free(local_Cd_y);
	hthread_dev_close(dev_id);

    MPI_Finalize();

}

