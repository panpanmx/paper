#include "../include/common.h"
#include "../include/operator.h"
#include <stdlib.h> 

int main(int argc, char **argv)
{
    char hostname[100];
    int commSize, myRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    //gethostname(hostname,sizeof(hostname));
    //printf("Hello world from process %d of %d: host: %s\n", myRank, commSize, hostname);

    /* 主机打开计算设备、创建DSP线程组 ***/
    int dev_id = myRank % 4;
    //int num_threads = atoi(argv[5]);                   // 24      一个线程组的线程数 / 计算单元 / 核数
    int num_threads = 24;

    hthread_dev_open(dev_id);
    // hthread_dat_load(dev_id, "kernel_pipeline.dat");
    hthread_dat_load(dev_id, "kernel_AS.dat");

    int tg_id = hthread_group_create(dev_id, num_threads);      // 在指定 DSP 簇上创建 DSP 线程组
	// printf("tg_id = %d \n", tg_id);      // tg_id = 0
	hthread_group_wait(tg_id);


    int b_id = hthread_barrier_create(dev_id);
    
    int lock_id = hthread_rwlock_create(dev_id);

    int M, K, N;
    get_complex_matrix_dim(&M, &K, argv[1]);
    get_complex_matrix_dim(&K, &N, argv[3]);
    if(myRank == 0){
        printf("M = %d, K = %d, N = %d, ", M, K, N);
        printf("节点数:%d, ", (commSize/4));
        printf("进程数:%d, ", commSize);
        printf("线程数:%d\n", num_threads);
    } 
/*    if (loaded_n1 != loaded_m2) {
        printf("Error: Matrices cannot be multiplied. Matrix 1's columns (%d) do not match Matrix 2's rows (%d).\n", loaded_n1, loaded_m2);
        return -1; // 返回错误码
    }   */

    double* Ac_x = (double*)malloc(M * K * sizeof(double));
    double* Ac_y = (double*)malloc(M * K * sizeof(double));

    double* Ad_x = (double*)hthread_malloc(dev_id, M * K * sizeof(double), HT_MEM_RO);
    double* Ad_y = (double*)hthread_malloc(dev_id, M * K * sizeof(double), HT_MEM_RO);

    double* Bc_x = (double*)malloc(K * N * sizeof(double));
	double* Bc_y = (double*)malloc(K * N * sizeof(double));

	double* Bd_x = (double*)hthread_malloc(dev_id, K * N * sizeof(double), HT_MEM_RO);
	double* Bd_y = (double*)hthread_malloc(dev_id, K * N * sizeof(double), HT_MEM_RO);

	double* local_Cc_x = (double*)malloc(M * N * sizeof(double));
	double* local_Cc_y = (double*)malloc(M * N * sizeof(double));

	double* local_Cd_x = (double*)hthread_malloc(dev_id, M * N * sizeof(double), HT_MEM_RO);
	double* local_Cd_y = (double*)hthread_malloc(dev_id, M * N * sizeof(double), HT_MEM_RO);

    int N_per_thread = N / num_threads;      // B基本cols
    int N_remaining_cols = N % num_threads;       // B剩余cols

    //if(myRank == 0) printf("N_remaining_cols : %d\n", N_remaining_cols);

    // 存储每个线程的行数
    //unsigned long M_thread[num_threads];
    int * N_thread = (int*)hthread_malloc(dev_id, num_threads * sizeof(int), HT_MEM_RO);
    // 分配行数
    for (int t = 0; t < num_threads; ++t) {
        N_thread[t] = N_per_thread + (t < N_remaining_cols ? 1 : 0);
        // if(myRank == 0) printf("N_thread[%d] : %d\n", t, N_thread[t]);
    }

    if(myRank == 0) {
        if(read_matrix(argv[1], Ac_x, M * K) != EXIT_SUCCESS) return EXIT_FAILURE;
        if(read_matrix(argv[2], Ac_y, M * K) != EXIT_SUCCESS) return EXIT_FAILURE;

        // if(read_matrix(argv[1], Ad_x, M * K) != EXIT_SUCCESS) return EXIT_FAILURE;
        // if(read_matrix(argv[2], Ad_y, M * K) != EXIT_SUCCESS) return EXIT_FAILURE;

        if(read_matrix(argv[3], Bc_x, K * N) != EXIT_SUCCESS) return EXIT_FAILURE;
        if(read_matrix(argv[4], Bc_y, K * N) != EXIT_SUCCESS) return EXIT_FAILURE;

        // if(read_matrix(argv[3], Bd_x, K * N) != EXIT_SUCCESS) return EXIT_FAILURE;
        // if(read_matrix(argv[4], Bd_y, K * N) != EXIT_SUCCESS) return EXIT_FAILURE;
    }

     MPI_Bcast(Ac_x, M * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     MPI_Bcast(Ac_y, M * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //  MPI_Bcast(Ad_x, M * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //  MPI_Bcast(Ad_y, M * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

     MPI_Bcast(Bc_x, K * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     MPI_Bcast(Bc_y, K * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //  MPI_Bcast(Bd_x, K * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //  MPI_Bcast(Bd_y, K * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for(int i=0;i<M * K;i++){
        // Ac_x[i]=Ac_x[i]*0.90000000009;
        // Ac_y[i]=Ac_y[i]*0.79999999999;
        Ad_x[i]=Ac_x[i];
        Ad_y[i]=Ac_y[i];
    }

    for(int i=0;i<K * N;i++){
        // Bc_x[i]=Bc_x[i]*0.900000000009;
        // Bc_y[i]=Bc_y[i]*0.799999999999;
        Bd_x[i]=Bc_x[i];
        Bd_y[i]=Bc_y[i];
    }

    // 最大分块
    // int M_block = 56;
    // int K_block = 32;
    // int N_block = 64;       // 必须是16的倍数
    
    int M_block = 56;
    int K_block = 32;
    int N_block = 64;       // 必须是16的倍数


    int M_G = 48;
    int K_G = 256;
    int N_G = 64;

    //if(myRank == 0){
     //   printf("M_block = %d, K_block = %d, N_block = %d.\n", M_block, K_block, N_block);
    //}

    double start_time=0, end_time=0, time_cost=0, max_time_cost=0;

    // Initialize C to zero
    for (int i = 0; i < M * N; i++) {
        local_Cc_x[i] = 0.0;
        local_Cc_y[i] = 0.0;
    }

    // CPU运算 
    // start_time = MPI_Wtime();
    // //gemm_cpu(local_Cc_x, local_Cc_y, Ac_x, Ac_y, Bc_x, Bc_y, M_block, K_block, N_block, M, N, K);
    gemm_cpu_no_block(local_Cc_x, local_Cc_y, Ac_x, Ac_y, Bc_x, Bc_y, M, N, K);
    // end_time = MPI_Wtime();
    // time_cost = end_time - start_time;
    // MPI_Reduce(&time_cost, &max_time_cost, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // if(myRank == 0){
    //     printf("time: cpu = %lf\n", max_time_cost);

    //     //write_matrix_to_file("../resultASR/MPI_C_x_CPU.txt", local_Cc_x, M, N);
    //     //write_matrix_to_file("../resultASR/MPI_C_y_CPU.txt", local_Cc_y, M, N);
    // }

                 
    /*** DSP AM + SM 运算 ***/
    //gettimeofday(&t3_AM_SM_s, NULL);
    for(int i=0;i<5;i++){
        // Initialize C to zero
        for (int i = 0; i < M * N; i++) {
            local_Cd_x[i] = 0.0;
            local_Cd_y[i] = 0.0;
        }
        start_time = MPI_Wtime();
        gemm_dsp_AM_SM_s(local_Cd_x, local_Cd_y, Ad_x, Ad_y, Bd_x, Bd_y,
                     M_block, K_block, N_block, M, N, K, N_thread, num_threads, tg_id);
        end_time = MPI_Wtime();

        time_cost = end_time - start_time;
        MPI_Reduce(&time_cost, &max_time_cost, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        double error_x_local = 0.0;
        double error_y_local = 0.0;
        for(int i = 0; i < M * N; i++){
            error_x_local = fmax(error_x_local,abs(local_Cd_x[i] - local_Cc_x[i]));
            error_y_local = fmax(error_y_local,abs(local_Cd_y[i] - local_Cc_y[i]));
        }
        
        double error_x_global;
        double error_y_global;
        
        MPI_Reduce(&error_x_local, &error_x_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&error_y_local, &error_y_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
        if (myRank == 0) {
            printf("error_x_global = %le   error_y_global = %le\n", error_x_global, error_y_global);
            printf("time = %lfs\n", max_time_cost);
        }

    }

//    gemm_cpu(local_Cd_x, local_Cd_y, Ad_x, Ad_y, Bd_x, Bd_y, M_block, K_block, N_block, M, N, K);
    
    // gemm_dsp_AM_SM_d(local_Cd_x, local_Cd_y, Ad_x, Ad_y, Bd_x, Bd_y,
    //                 M_block, K_block, N_block, M, N, K, N_thread, num_threads, tg_id);
    
    // gemm_dsp_AM_SM_triple(local_Cd_x, local_Cd_y, Ad_x, Ad_y, Bd_x, Bd_y,
    //                 M_block, K_block, N_block, M, N, K, N_thread, num_threads, tg_id);
    // gemm_dsp_AM_SM_pipeline(local_Cd_x, local_Cd_y, Ad_x, Ad_y, Bd_x, Bd_y,
    //                 M_block, K_block, N_block, M, N, K, N_thread, num_threads, tg_id);
    // gemm_dsp_AM_SM_gs(local_Cd_x, local_Cd_y, Ad_x, Ad_y, Bd_x, Bd_y,
    //                  M_block, K_block, N_block, M, N, K, N_thread, num_threads, b_id, tg_id);

    // gemm_dsp_G(local_Cd_x, local_Cd_y, Ad_x, Ad_y, Bd_x, Bd_y,
    //                  M_block, K_block, N_block, M, N, K, N_thread, num_threads, b_id, tg_id);


    // if(myRank == 0){
        // int err = 0;
        // 在主进程中打印计算结果
        // for (int i = 0; i < M * N; i++) {
        //     //printf("local_Cc_x[%d] = %4f\n", i, local_Cc_x[i]);
        //     if(abs(local_Cd_x[i] - local_Cc_x[i]) > ERROR_THRESHOLD)    {
        //     printf("Rerr[%d] = %d\n", i, err);
        //     err++;
        //     }
        //     if(abs(local_Cd_y[i] - local_Cc_y[i]) > ERROR_THRESHOLD) {
        //     printf("Yerr[%d] = %d\n", i, err);
        //     err++;
        //     }
        // }

        
        // printf("err = %d    ", err);
        // if(err==0){
        //     printf("计算结果验证正确 ");
        // }
    // }

    
    

    hthread_barrier_destroy(b_id);
    hthread_rwlock_destroy(lock_id);
    //主机关闭计算设备、销毁DSP线程组 ***/
    hthread_group_destroy(tg_id);        // 释放缓冲区和上下文
    hthread_free(Ac_x);
    hthread_free(Ac_y);
    hthread_free(Ad_x);
    hthread_free(Ad_y);
    hthread_free(Bc_x);
    hthread_free(Bc_y);
    hthread_free(Bd_x);
    hthread_free(Bd_y);
    hthread_free(local_Cc_x);
    hthread_free(local_Cc_y);
    hthread_free(local_Cd_x);
    hthread_free(local_Cd_y);
	hthread_dev_close(dev_id);

    MPI_Finalize();

}