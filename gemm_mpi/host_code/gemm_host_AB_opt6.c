#include "../include/common.h"
#include "../include/operator.h"
#include <stdlib.h>  
#include <string.h>

int main(int argc, char **argv)
{
    int commSize, myRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    const char *kernel_choice = "gemm_dsp_AM_SM_pipeline_opt";
    int num_threads = 24;
    int N_block = 64;

    for (int i = 5; i < argc; i++) {
        if (strstr(argv[i], "pipeline") != NULL) {
            kernel_choice = "gemm_dsp_AM_SM_pipeline_opt";
        } else if (strstr(argv[i], "2d_v2") != NULL || strstr(argv[i], "matrix_mul_asd_opt_2d_v2") != NULL) {
            kernel_choice = "matrix_mul_asd_opt_2d_v2";
        } else if (strstr(argv[i], "2d") != NULL || strstr(argv[i], "matrix_mul_asd_opt_2d") != NULL) {
            kernel_choice = "matrix_mul_asd_opt_2d";
        } else if (strstr(argv[i], "d_opt_v3") != NULL) {
            kernel_choice = "gemm_dsp_AM_SM_d_opt_v3";
        } else if (strstr(argv[i], "d_opt") != NULL || strstr(argv[i], "asd_opt") != NULL) {
            kernel_choice = "gemm_dsp_AM_SM_d_opt";
        } else if (atoi(argv[i]) > 0) {
            int val = atoi(argv[i]);
            if (val <= 24) {
                num_threads = val;
            } else {
                N_block = val;
            }
        }
    }

    // 1. 硬件初始化 
    int dev_id = myRank % 4;
    hthread_dev_open(dev_id);
    if (strstr(kernel_choice, "pipeline") != NULL) {
        hthread_dat_load(dev_id, "kernel_pipeline.dat");
    } else {
        hthread_dat_load(dev_id, "kernel_AS.dat");
    }
    int tg_id = hthread_group_create(dev_id, num_threads);
    hthread_group_wait(tg_id);
    int lock_id = hthread_rwlock_create(dev_id);

    /*** 2. 矩阵维度与分块参数 ***/
    int M, K, N;
    get_complex_matrix_dim(&M, &K, argv[1]);
    get_complex_matrix_dim(&K, &N, argv[3]);
    if(myRank == 0) {
        printf("M = %d, K = %d, N = %d\n", M, K, N);
        printf("运行内核: %s, 线程数: %d, N_block: %d\n", kernel_choice, num_threads, N_block);
        fflush(stdout);
    }

    int is_pipeline = (strstr(kernel_choice, "pipeline") != NULL);
    int is_1d_kernel = is_pipeline || (strstr(kernel_choice, "matrix_mul_asd_opt") != NULL && strstr(kernel_choice, "2d") == NULL) || (strstr(kernel_choice, "d_opt") != NULL && strstr(kernel_choice, "2d") == NULL);
    int is_v3 = (strstr(kernel_choice, "v3") != NULL);
    int is_2d_v1 = (strcmp(kernel_choice, "matrix_mul_asd_opt_2d") == 0);

    int M_block = 56, K_block = 32;
    if (!is_pipeline && N_block == 64) {
        N_block = 128;
    }

    /*** 3. 划分逻辑 (2D MPI 分布 + DSP 簇内 3x8 100% 均衡网格划分) ***/
    int group_num = 2;
    int group_size = commSize / group_num;
    int group_id = myRank / group_size;
    int group_rank = myRank % group_size;

    // B 矩阵列划分
    int base_cols = N / group_num;
    int remainder = N % group_num;
    int start_col[group_num + 1];
    start_col[0] = 0;
    for (int i = 0; i < group_num; i++) {
        start_col[i + 1] = start_col[i] + base_cols + (i < remainder ? 1 : 0);
    }
    int local_cols = start_col[group_id + 1] - start_col[group_id];

    // 线程任务量分配
    int *N_thread = (int*)hthread_malloc(dev_id, num_threads * sizeof(int), HT_MEM_RO);
    if (is_1d_kernel) {
        int chunk_64 = 64;
        int total_chunks = local_cols / chunk_64;
        int rem_elems = local_cols % chunk_64;
        int base_chunks = total_chunks / num_threads;
        int rem_chunks = total_chunks % num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int chunks = base_chunks + (t < rem_chunks ? 1 : 0);
            N_thread[t] = chunks * chunk_64 + (t == num_threads - 1 ? rem_elems : 0);
        }
    } else {
        for (int t = 0; t < num_threads; ++t) {
            N_thread[t] = local_cols / 8;
        }
    }
    int local_A_src[commSize];
    int local_C_src[commSize];
    int local_A_counts[commSize];
    int local_C_counts[commSize];

    int groupm = M / (commSize / group_num);
    int M_per_process = (M - groupm) / (commSize / group_num - 1);
    int M_remaining_rows = (M - groupm) % (commSize / group_num - 1);
    if(myRank == 0){
        printf("节点数:%d\n", (commSize/4));
        printf("groupm: %d\n", groupm);
        printf("M_per_process: %d\n", M_per_process);
        printf("M_remaining_rows: %d\n", M_remaining_rows);
    }
    int M_last_sum = 0;
    int local_M[commSize];
    int groupt = 0;
    for (int t = 0; t < commSize; t++, groupt++) {
        if(t % group_size == 0){
            M_last_sum = 0;
            groupt = 0;
            local_M[t] = groupm;
        }
        else{
            local_M[t] = (M_per_process + (groupt <= M_remaining_rows ? 1 : 0));
        }
        local_A_src[t] = K * M_last_sum;
        local_C_src[t] = N * M_last_sum;
        local_A_counts[t] = K * local_M[t];
        local_C_counts[t] = N * local_M[t];
        M_last_sum += local_M[t];
    }

    /*** 4. 内存分配 (仅保留 DSP 侧内存) ***/
    double* local_Ad_x = (double*)hthread_malloc(dev_id, local_A_counts[myRank] * sizeof(double), HT_MEM_RO);
    double* local_Ad_y = (double*)hthread_malloc(dev_id, local_A_counts[myRank] * sizeof(double), HT_MEM_RO);
    
    double* local_Bd_x = (double*)hthread_malloc(dev_id, K * local_cols * sizeof(double), HT_MEM_RO);
    double* local_Bd_y = (double*)hthread_malloc(dev_id, K * local_cols * sizeof(double), HT_MEM_RO);

    double* local_Cd_x = (double*)hthread_malloc(dev_id, local_C_counts[myRank] * sizeof(double), HT_MEM_RW);
    double* local_Cd_y = (double*)hthread_malloc(dev_id, local_C_counts[myRank] * sizeof(double), HT_MEM_RW);
    memset(local_Cd_x, 0, local_C_counts[myRank] * sizeof(double));
    memset(local_Cd_y, 0, local_C_counts[myRank] * sizeof(double));

    // 5. 数据并行读取与分发
    double start_time, start_time1, end_time, end_time1, time_cost, time_cost1;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm group_comm;
    MPI_Comm_split(MPI_COMM_WORLD, group_id, myRank, &group_comm);

    MPI_Request *dist_reqs = NULL;
    int dist_req_count = 0;
    MPI_Datatype *coltype = NULL; 
    if (myRank == 0) dist_reqs = (MPI_Request*)malloc(2 * group_num * sizeof(MPI_Request));
    
    MPI_Request recv_reqs[2];
    
    double t_sub0, t_sub1, only_io_time = 0.0, only_comm_time = 0.0;
    start_time = MPI_Wtime();
    double *Bc_x = NULL, *Bc_y = NULL;
    if (myRank == 0) {
        Bc_x = (double*)malloc(K * N * sizeof(double));
        Bc_y = (double*)malloc(K * N * sizeof(double));
        t_sub0 = MPI_Wtime();
        if(read_matrix(argv[3], Bc_x, K * N) != EXIT_SUCCESS) return EXIT_FAILURE;
        if(read_matrix(argv[4], Bc_y, K * N) != EXIT_SUCCESS) return EXIT_FAILURE;
        t_sub1 = MPI_Wtime();
        only_io_time += (t_sub1 - t_sub0);

        t_sub0 = MPI_Wtime();
        coltype = (MPI_Datatype*)malloc(group_num * sizeof(MPI_Datatype));

        for (int g = 0; g < group_num; g++) {
            int g_start = start_col[g];
            int g_cols = start_col[g + 1] - g_start;
            
            if (g == 0) {
                for (int i = 0; i < K; i++) {
                    memcpy(&local_Bd_x[i * g_cols], &Bc_x[i * N + g_start], g_cols * sizeof(double));
                    memcpy(&local_Bd_y[i * g_cols], &Bc_y[i * N + g_start], g_cols * sizeof(double));
                }
            } else {
                MPI_Type_vector(K, g_cols, N, MPI_DOUBLE, &coltype[g]);
                MPI_Type_commit(&coltype[g]);
                MPI_Isend(&Bc_x[g_start], 1, coltype[g], g * group_size, 0, MPI_COMM_WORLD, &dist_reqs[dist_req_count++]);
                MPI_Isend(&Bc_y[g_start], 1, coltype[g], g * group_size, 1, MPI_COMM_WORLD, &dist_reqs[dist_req_count++]);
            }
        }
        t_sub1 = MPI_Wtime();
        only_comm_time += (t_sub1 - t_sub0);
    } else if (group_rank == 0) {
        t_sub0 = MPI_Wtime();
        MPI_Irecv(local_Bd_x, K * local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &recv_reqs[0]);
        MPI_Irecv(local_Bd_y, K * local_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &recv_reqs[1]);
        t_sub1 = MPI_Wtime();
        only_comm_time += (t_sub1 - t_sub0);
    }

    t_sub0 = MPI_Wtime();
    if (pread_matrix(argv[1], local_Ad_x, local_A_counts[myRank] * sizeof(double),
                    local_A_src[myRank] * sizeof(double)) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    if (pread_matrix(argv[2], local_Ad_y, local_A_counts[myRank] * sizeof(double),
                    local_A_src[myRank] * sizeof(double)) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    t_sub1 = MPI_Wtime();
    only_io_time += (t_sub1 - t_sub0);

    t_sub0 = MPI_Wtime();
    if (myRank == 0) {
        if (dist_req_count > 0) MPI_Waitall(dist_req_count, dist_reqs, MPI_STATUSES_IGNORE);
        for (int g = 1; g < group_num; g++) {
            MPI_Type_free(&coltype[g]);
        }
    } else if (group_rank == 0) {
        MPI_Waitall(2, recv_reqs, MPI_STATUSES_IGNORE);
    }

    MPI_Request bcast_requests[2];
    size_t b_alloc_bytes = (size_t)K * local_cols * sizeof(double);
    double* host_temp_x = (double*)malloc(b_alloc_bytes);
    double* host_temp_y = (double*)malloc(b_alloc_bytes);
    if (host_temp_x == NULL || host_temp_y == NULL) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (group_rank == 0) {
        memcpy(host_temp_x, local_Bd_x, b_alloc_bytes);
        memcpy(host_temp_y, local_Bd_y, b_alloc_bytes);
    }

    MPI_Ibcast(host_temp_x, K * local_cols, MPI_DOUBLE, 0, group_comm, &bcast_requests[0]);
    MPI_Ibcast(host_temp_y, K * local_cols, MPI_DOUBLE, 0, group_comm, &bcast_requests[1]);

    MPI_Waitall(2, bcast_requests, MPI_STATUSES_IGNORE);

    memcpy(local_Bd_x, host_temp_x, b_alloc_bytes);
    memcpy(local_Bd_y, host_temp_y, b_alloc_bytes);

    free(host_temp_x);
    free(host_temp_y);
    t_sub1 = MPI_Wtime();
    only_comm_time += (t_sub1 - t_sub0);

    end_time = MPI_Wtime();
    time_cost = end_time - start_time;
    double leader_io = (group_rank == 0) ? only_io_time : 0.0;
    double member_io = (group_rank != 0) ? only_io_time : 0.0;
    double leader_comm = (group_rank == 0) ? only_comm_time : 0.0;
    double member_comm = (group_rank != 0) ? only_comm_time : 0.0;
    double leader_time = (group_rank == 0) ? time_cost : 0.0;
    double member_time = (group_rank != 0) ? time_cost : 0.0;
    double max_leader_io, max_member_io, max_leader_comm, max_member_comm, max_leader_time, max_member_time;
    MPI_Reduce(&leader_io, &max_leader_io, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&member_io, &max_member_io, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&leader_comm, &max_leader_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&member_comm, &max_member_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&leader_time, &max_leader_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&member_time, &max_member_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(myRank == 0){
        printf("Max leader io = %lf\n", max_leader_io);
        printf("Max member io = %lf\n", max_member_io);
        printf("Max leader comm = %lf\n", max_leader_comm);
        printf("Max member comm = %lf\n", max_member_comm);
        printf("Max leader time = %lf\n", max_leader_time);
        printf("Max leader time = %lf\n", max_member_time);
    }
    if (myRank == 0) {
        free(Bc_x); free(Bc_y);
        free(dist_reqs);
    }
    
    /*** 6. DSP 计算 ***/
    MPI_Barrier(MPI_COMM_WORLD);
    start_time1 = MPI_Wtime();
    if (is_pipeline) {
        gemm_dsp_AM_SM_pipeline_opt(local_Cd_x, local_Cd_y, local_Ad_x, local_Ad_y, local_Bd_x, local_Bd_y,
                                    M_block, K_block, N_block, local_M[myRank], local_cols, K, N_thread, num_threads, tg_id);
    } else if (is_v3) {
        gemm_dsp_AM_SM_d_opt_v3(local_Cd_x, local_Cd_y, local_Ad_x, local_Ad_y, local_Bd_x, local_Bd_y,
                                M_block, K_block, N_block, local_M[myRank], local_cols, K, N_thread, num_threads, tg_id);
    } else if (is_1d_kernel) {
        gemm_dsp_AM_SM_d_opt(local_Cd_x, local_Cd_y, local_Ad_x, local_Ad_y, local_Bd_x, local_Bd_y,
                             M_block, K_block, N_block, local_M[myRank], local_cols, K, N_thread, num_threads, tg_id);
    } else if (is_2d_v1) {
        gemm_dsp_AM_SM_d_opt_2d(local_Cd_x, local_Cd_y, local_Ad_x, local_Ad_y, local_Bd_x, local_Bd_y,
                                M_block, K_block, N_block, local_M[myRank], local_cols, K, N_thread, num_threads, tg_id);
    } else {
        gemm_dsp_AM_SM_d_opt_2d_v2(local_Cd_x, local_Cd_y, local_Ad_x, local_Ad_y, local_Bd_x, local_Bd_y,
                                   M_block, K_block, N_block, local_M[myRank], local_cols, K, N_thread, num_threads, tg_id);
    }
    end_time1 = MPI_Wtime();
    time_cost1 = end_time1 - start_time1;
    double leader_comp = (group_rank == 0) ? time_cost1 : 0.0;
    double member_comp = (group_rank != 0) ? time_cost1 : 0.0;
    double max_leader, max_member, max_comp_time;
    MPI_Reduce(&leader_comp, &max_leader, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&member_comp, &max_member, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_cost1, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(myRank == 0){
        printf("Kernel: %s (N_block=%d)\n", kernel_choice, N_block);
        printf("time: %s = %lf s\n", kernel_choice, max_comp_time);
        printf("Max leader comp = %lf\n", max_leader);
        printf("Max member comp = %lf\n", max_member);
        fflush(stdout);
    }

    if (group_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&group_comm);
    }
    hthread_group_destroy(tg_id);
    hthread_rwlock_destroy(lock_id);
    
    hthread_free(local_Ad_x);
    hthread_free(local_Ad_y);
    hthread_free(local_Bd_x);
    hthread_free(local_Bd_y);
    hthread_free(local_Cd_x);
    hthread_free(local_Cd_y);
    hthread_free(N_thread);

    hthread_dev_close(dev_id);
    MPI_Finalize();
    return 0;
}
