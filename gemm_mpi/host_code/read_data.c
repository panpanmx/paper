#include "../include/common.h"
#include "../include/operator.h"
#include <stdlib.h> 
// int main() {
//     const char *filename = "/vol8/home/njust_ztc/hnu/yhy/gemm_mpi/data/A.dat";
//     FILE *fp = fopen(filename, "r");
//     if (!fp) {
//         perror("无法打开文件");
//         return -1;
//     }

//     // --- 第一步：统计行数 ---
//     size_t rowCount = 0;
//     {
//         int ch;
//         while ((ch = fgetc(fp)) != EOF) {
//             if (ch == '\n') rowCount++;
//         }
//         // 最后一行如果没有换行符，也算一行
//         fseek(fp, -1, SEEK_END);
//         if (fgetc(fp) != '\n') rowCount++;
//     }

//     // --- 第二步：统计第一行的数字个数 ---
//     rewind(fp);
//     size_t numCount = 0;
//     double tmp;
//     while (fscanf(fp, "%lf", &tmp) == 1) {
//         numCount++;
//         int c = fgetc(fp);
//         if (c == '\n' || c == EOF) break;
//         ungetc(c, fp); // 把不是换行的字符放回去
//     }

//     fclose(fp);

//     if (numCount % 2 != 0) {
//         fprintf(stderr, "格式错误：第一行的数字不是偶数\n");
//         return -1;
//     }

//     size_t colCount = numCount / 2;

//     if (rowCount != colCount) {
//         fprintf(stderr, "警告：文件不是方阵 (%zu x %zu)\n", rowCount, colCount);
//     }

//     printf("矩阵维度: %zu x %zu\n", rowCount, colCount);
//     return 0;
// } 
// int main(int argc, char **argv)
// {
//     int commSize, myRank;
//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
//     MPI_Comm_size(MPI_COMM_WORLD, &commSize);

//     /*** 主机打开计算设备、创建DSP线程组 ***/
//     int dev_id = myRank % 4;
//     int num_threads = atoi(argv[5]);                     // 24      一个线程组的线程数 / 计算单元 / 核数
//     hthread_dev_open(dev_id);
//     hthread_dat_load(dev_id, "kernel_AS.dat");

//     int tg_id = hthread_group_create(dev_id, num_threads);      // 在指定 DSP 簇上创建 DSP 线程组
// 	// printf("tg_id = %d \n", tg_id);      // tg_id = 0
// 	hthread_group_wait(tg_id);

//     int M, K, N;
//     get_complex_matrix_dim(&M, &K, argv[1]);
//     get_complex_matrix_dim(&K, &N, argv[3]);
//     if(myRank == 0) printf("M = %d, K = %d, N = %d\n", M, K, N);
   
//     int group_num = 4;              // 列块数量（分组数）
//     int group_size = commSize / group_num;   // 每组进程数量

//     // 计算每组的列范围
//     int base_cols = N / group_num;         // 每组至少拥有的列数
//     int remainder = N % group_num;         // 多余的列
//     int start_col[group_num + 1];
//     start_col[0] = 0;
//     for (int i = 0; i < group_num; i++) {
//         int cols_in_group = base_cols + (i < remainder ? 1 : 0);
//         start_col[i + 1] = start_col[i] + cols_in_group;
//     }

//     // 确定当前进程所属的组
//     int group_id = myRank / group_size;
//     int group_rank = myRank % group_size;
//     int cols_start = start_col[group_id];
//     int cols_end   = start_col[group_id + 1];
//     int local_cols = cols_end - cols_start;

//     // root 进程读取矩阵 B
//     double *Bc_x = NULL;
//     double *Bc_y = NULL;
//     if (myRank == 0)
//     {
//         Bc_x = (double*)malloc(K * N * sizeof(double));
// 	    Bc_y = (double*)malloc(K * N * sizeof(double));
        
//         if(read_matrix(argv[3], Bc_x, K * N) != EXIT_SUCCESS) return EXIT_FAILURE;
//         if(read_matrix(argv[4], Bc_y, K * N) != EXIT_SUCCESS) return EXIT_FAILURE;
//     }

//     // 每个组的根进程分配对应的列块
//     MPI_Comm group_comm;
//     MPI_Comm_split(MPI_COMM_WORLD, group_id, myRank, &group_comm);

//     double* local_Bc_x = (double*)malloc(K * local_cols * sizeof(double));
//     double* local_Bc_y = (double*)malloc(K * local_cols * sizeof(double));

//     double start_time, end_time, time_cost, max_time_cost;
//     start_time = MPI_Wtime();
//     // 根进程发送对应列块给各组的组内根
//     if (myRank == 0) {
//         for (int g = 0; g < group_num; g++) {
//             int g_start = start_col[g];
//             int g_end = start_col[g + 1];
//             int g_cols = g_end - g_start;

//             MPI_Datatype coltype;
//             MPI_Type_vector(N, g_cols, N, MPI_DOUBLE, &coltype);
//             MPI_Type_commit(&coltype);

//             if (g == 0) {
//                 // 自己这一组（组0）
//                 for (int j = 0; j < g_cols; j++){
//                     for (int i = 0; i < N; i++){
//                         local_Bc_x[i * g_cols + j] = Bc_x[i * N + g_start + j];
//                         local_Bc_y[i * g_cols + j] = Bc_y[i * N + g_start + j];
//                     } 
//                 }                          
//             } else {
//                 // 发送给各组的组内根（rank = g * group_size）
//                 MPI_Send(&Bc_x[g_start], 1, coltype, g * group_size, 0, MPI_COMM_WORLD);
//                 MPI_Send(&Bc_y[g_start], 1, coltype, g * group_size, 0, MPI_COMM_WORLD);
//             }
//             MPI_Type_free(&coltype);
//         }
//     }

//     // 各组内根接收列块
//     if (group_rank == 0 && myRank != 0) {
//         MPI_Status status;
//         MPI_Recv(local_Bc_x, N * local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
//         MPI_Recv(local_Bc_y, N * local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
//     }

//     // 组内广播，让每个进程持有相同列块
//     MPI_Bcast(local_Bc_x, N * local_cols, MPI_DOUBLE, 0, group_comm);
//     MPI_Bcast(local_Bc_y, N * local_cols, MPI_DOUBLE, 0, group_comm);

//     end_time = MPI_Wtime();
//     time_cost = end_time - start_time;
//     MPI_Reduce(&time_cost, &max_time_cost, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//     if(myRank == 0){
//         printf("time: cpu = %lf\n", max_time_cost);
//     }

//     // --------------------------
//     // 测试输出
//     // --------------------------
//     if (myRank == 0)
//         printf("N=%d, group_num=%d, remainder=%d\n", N, group_num, remainder);
//     if (myRank == 0 || myRank == group_size || myRank == 2 * group_size){
//         printf("myRank %d in Group %d: cols [%d, %d), local_cols=%d\n",
//                myRank, group_id, cols_start, cols_end, local_cols);
//     }


//     // 主机关闭计算设备、销毁DSP线程组
//     hthread_group_destroy(tg_id);
//     if(myRank == 0){
//         hthread_free(Bc_x);
//         hthread_free(Bc_y);
//     }
//     hthread_free(local_Bc_x);
//     hthread_free(local_Bc_y);
//     MPI_Comm_free(&group_comm);
//     hthread_dev_close(dev_id);
//     MPI_Finalize();
//     return 0;
// }

int main(){
    int m=8192;
    int n=8192;
    int k=8192;
    // double *real_matrix = (double*)malloc(n * n * sizeof(double));
    // double *imag_matrix = (double*)malloc(n * n * sizeof(double));
    const char *real_filename = "/vol8/home/njust_ztc/hnu/jpp/gemm_mpi/test_data/realH8192.bin";
    const char *imag_filename = "/vol8/home/njust_ztc/hnu/jpp/gemm_mpi/test_data/imagH8192.bin";
    double *real_matrix = generate_real_matrix(m, k);
    double *imag_matrix = generate_imag_matrix(k, n);
    save_matrix_to_binary(real_matrix, m, k, real_filename);
    save_matrix_to_binary(imag_matrix, k, n, imag_filename);
    free(real_matrix);
    free(imag_matrix);
    real_matrix = NULL;
    imag_matrix = NULL;

}