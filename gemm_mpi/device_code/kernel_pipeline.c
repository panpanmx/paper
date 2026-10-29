#include <compiler/m3000.h>
#include "hthread_device.h"

#define v_num 16

// ==============================================================================
// 4x4 向量寄存器分块计算宏（支持指定目标 C 向量缓冲区 CX_A / CY_A）
// 累加器全程驻留 32 个向量寄存器，双发射 FMA 算子融合
// ==============================================================================
#define COMPUTE_K_BLOCK_OPT_C(AX_S, AY_S, BX_A, BY_A, CX_A, CY_A, K_CUR) \
do { \
    int m_k = 0; \
    for (; m_k + 3 < M_cur; m_k += 4) { \
        int c0_base = (m_k + 0) * N_block; \
        int c1_base = (m_k + 1) * N_block; \
        int c2_base = (m_k + 2) * N_block; \
        int c3_base = (m_k + 3) * N_block; \
        lvector double cx0_0 = vec_ld(c0_base + 0 * v_num, CX_A); \
        lvector double cx0_1 = vec_ld(c0_base + 1 * v_num, CX_A); \
        lvector double cx0_2 = vec_ld(c0_base + 2 * v_num, CX_A); \
        lvector double cx0_3 = vec_ld(c0_base + 3 * v_num, CX_A); \
        lvector double cy0_0 = vec_ld(c0_base + 0 * v_num, CY_A); \
        lvector double cy0_1 = vec_ld(c0_base + 1 * v_num, CY_A); \
        lvector double cy0_2 = vec_ld(c0_base + 2 * v_num, CY_A); \
        lvector double cy0_3 = vec_ld(c0_base + 3 * v_num, CY_A); \
        lvector double cx1_0 = vec_ld(c1_base + 0 * v_num, CX_A); \
        lvector double cx1_1 = vec_ld(c1_base + 1 * v_num, CX_A); \
        lvector double cx1_2 = vec_ld(c1_base + 2 * v_num, CX_A); \
        lvector double cx1_3 = vec_ld(c1_base + 3 * v_num, CX_A); \
        lvector double cy1_0 = vec_ld(c1_base + 0 * v_num, CY_A); \
        lvector double cy1_1 = vec_ld(c1_base + 1 * v_num, CY_A); \
        lvector double cy1_2 = vec_ld(c1_base + 2 * v_num, CY_A); \
        lvector double cy1_3 = vec_ld(c1_base + 3 * v_num, CY_A); \
        lvector double cx2_0 = vec_ld(c2_base + 0 * v_num, CX_A); \
        lvector double cx2_1 = vec_ld(c2_base + 1 * v_num, CX_A); \
        lvector double cx2_2 = vec_ld(c2_base + 2 * v_num, CX_A); \
        lvector double cx2_3 = vec_ld(c2_base + 3 * v_num, CX_A); \
        lvector double cy2_0 = vec_ld(c2_base + 0 * v_num, CY_A); \
        lvector double cy2_1 = vec_ld(c2_base + 1 * v_num, CY_A); \
        lvector double cy2_2 = vec_ld(c2_base + 2 * v_num, CY_A); \
        lvector double cy2_3 = vec_ld(c2_base + 3 * v_num, CY_A); \
        lvector double cx3_0 = vec_ld(c3_base + 0 * v_num, CX_A); \
        lvector double cx3_1 = vec_ld(c3_base + 1 * v_num, CX_A); \
        lvector double cx3_2 = vec_ld(c3_base + 2 * v_num, CX_A); \
        lvector double cx3_3 = vec_ld(c3_base + 3 * v_num, CX_A); \
        lvector double cy3_0 = vec_ld(c3_base + 0 * v_num, CY_A); \
        lvector double cy3_1 = vec_ld(c3_base + 1 * v_num, CY_A); \
        lvector double cy3_2 = vec_ld(c3_base + 2 * v_num, CY_A); \
        lvector double cy3_3 = vec_ld(c3_base + 3 * v_num, CY_A); \
        for (int k_k = 0; k_k < (K_CUR); k_k++) { \
            int b_base = k_k * N_block; \
            lvector double bx0 = vec_ld(b_base + 0 * v_num, BX_A); \
            lvector double bx1 = vec_ld(b_base + 1 * v_num, BX_A); \
            lvector double bx2 = vec_ld(b_base + 2 * v_num, BX_A); \
            lvector double bx3 = vec_ld(b_base + 3 * v_num, BX_A); \
            lvector double by0 = vec_ld(b_base + 0 * v_num, BY_A); \
            lvector double by1 = vec_ld(b_base + 1 * v_num, BY_A); \
            lvector double by2 = vec_ld(b_base + 2 * v_num, BY_A); \
            lvector double by3 = vec_ld(b_base + 3 * v_num, BY_A); \
            double a_r, a_i; \
            lvector double ax_v, ay_v, nay_v; \
            a_r = AX_S[(m_k + 0) * K_block + k_k]; a_i = AY_S[(m_k + 0) * K_block + k_k]; \
            ax_v = vec_svbcast(a_r); ay_v = vec_svbcast(a_i); nay_v = vec_svbcast(-a_i); \
            cx0_0 = vec_mula(ax_v, bx0, cx0_0); cx0_0 = vec_mula(nay_v, by0, cx0_0); \
            cy0_0 = vec_mula(ax_v, by0, cy0_0); cy0_0 = vec_mula(ay_v,  bx0, cy0_0); \
            cx0_1 = vec_mula(ax_v, bx1, cx0_1); cx0_1 = vec_mula(nay_v, by1, cx0_1); \
            cy0_1 = vec_mula(ax_v, by1, cy0_1); cy0_1 = vec_mula(ay_v,  bx1, cy0_1); \
            cx0_2 = vec_mula(ax_v, bx2, cx0_2); cx0_2 = vec_mula(nay_v, by2, cx0_2); \
            cy0_2 = vec_mula(ax_v, by2, cy0_2); cy0_2 = vec_mula(ay_v,  bx2, cy0_2); \
            cx0_3 = vec_mula(ax_v, bx3, cx0_3); cx0_3 = vec_mula(nay_v, by3, cx0_3); \
            cy0_3 = vec_mula(ax_v, by3, cy0_3); cy0_3 = vec_mula(ay_v,  bx3, cy0_3); \
            a_r = AX_S[(m_k + 1) * K_block + k_k]; a_i = AY_S[(m_k + 1) * K_block + k_k]; \
            ax_v = vec_svbcast(a_r); ay_v = vec_svbcast(a_i); nay_v = vec_svbcast(-a_i); \
            cx1_0 = vec_mula(ax_v, bx0, cx1_0); cx1_0 = vec_mula(nay_v, by0, cx1_0); \
            cy1_0 = vec_mula(ax_v, by0, cy1_0); cy1_0 = vec_mula(ay_v,  bx0, cy1_0); \
            cx1_1 = vec_mula(ax_v, bx1, cx1_1); cx1_1 = vec_mula(nay_v, by1, cx1_1); \
            cy1_1 = vec_mula(ax_v, by1, cy1_1); cy1_1 = vec_mula(ay_v,  bx1, cy1_1); \
            cx1_2 = vec_mula(ax_v, bx2, cx1_2); cx1_2 = vec_mula(nay_v, by2, cx1_2); \
            cy1_2 = vec_mula(ax_v, by2, cy1_2); cy1_2 = vec_mula(ay_v,  bx2, cy1_2); \
            cx1_3 = vec_mula(ax_v, bx3, cx1_3); cx1_3 = vec_mula(nay_v, by3, cx1_3); \
            cy1_3 = vec_mula(ax_v, by3, cy1_3); cy1_3 = vec_mula(ay_v,  bx3, cy1_3); \
            a_r = AX_S[(m_k + 2) * K_block + k_k]; a_i = AY_S[(m_k + 2) * K_block + k_k]; \
            ax_v = vec_svbcast(a_r); ay_v = vec_svbcast(a_i); nay_v = vec_svbcast(-a_i); \
            cx2_0 = vec_mula(ax_v, bx0, cx2_0); cx2_0 = vec_mula(nay_v, by0, cx2_0); \
            cy2_0 = vec_mula(ax_v, by0, cy2_0); cy2_0 = vec_mula(ay_v,  bx0, cy2_0); \
            cx2_1 = vec_mula(ax_v, bx1, cx2_1); cx2_1 = vec_mula(nay_v, by1, cx2_1); \
            cy2_1 = vec_mula(ax_v, by1, cy2_1); cy2_1 = vec_mula(ay_v,  bx1, cy2_1); \
            cx2_2 = vec_mula(ax_v, bx2, cx2_2); cx2_2 = vec_mula(nay_v, by2, cx2_2); \
            cy2_2 = vec_mula(ax_v, by2, cy2_2); cy2_2 = vec_mula(ay_v,  bx2, cy2_2); \
            cx2_3 = vec_mula(ax_v, bx3, cx2_3); cx2_3 = vec_mula(nay_v, by3, cx2_3); \
            cy2_3 = vec_mula(ax_v, by3, cy2_3); cy2_3 = vec_mula(ay_v,  bx3, cy2_3); \
            a_r = AX_S[(m_k + 3) * K_block + k_k]; a_i = AY_S[(m_k + 3) * K_block + k_k]; \
            ax_v = vec_svbcast(a_r); ay_v = vec_svbcast(a_i); nay_v = vec_svbcast(-a_i); \
            cx3_0 = vec_mula(ax_v, bx0, cx3_0); cx3_0 = vec_mula(nay_v, by0, cx3_0); \
            cy3_0 = vec_mula(ax_v, by0, cy3_0); cy3_0 = vec_mula(ay_v,  bx0, cy3_0); \
            cx3_1 = vec_mula(ax_v, bx1, cx3_1); cx3_1 = vec_mula(nay_v, by1, cx3_1); \
            cy3_1 = vec_mula(ax_v, by1, cy3_1); cy3_1 = vec_mula(ay_v,  bx1, cy3_1); \
            cx3_2 = vec_mula(ax_v, bx2, cx3_2); cx3_2 = vec_mula(nay_v, by2, cx3_2); \
            cy3_2 = vec_mula(ax_v, by2, cy3_2); cy3_2 = vec_mula(ay_v,  bx2, cy3_2); \
            cx3_3 = vec_mula(ax_v, bx3, cx3_3); cx3_3 = vec_mula(nay_v, by3, cx3_3); \
            cy3_3 = vec_mula(ax_v, by3, cy3_3); cy3_3 = vec_mula(ay_v,  bx3, cy3_3); \
        } \
        vec_st(cx0_0, c0_base + 0 * v_num, CX_A); vec_st(cx0_1, c0_base + 1 * v_num, CX_A); \
        vec_st(cx0_2, c0_base + 2 * v_num, CX_A); vec_st(cx0_3, c0_base + 3 * v_num, CX_A); \
        vec_st(cy0_0, c0_base + 0 * v_num, CY_A); vec_st(cy0_1, c0_base + 1 * v_num, CY_A); \
        vec_st(cy0_2, c0_base + 2 * v_num, CY_A); vec_st(cy0_3, c0_base + 3 * v_num, CY_A); \
        vec_st(cx1_0, c1_base + 0 * v_num, CX_A); vec_st(cx1_1, c1_base + 1 * v_num, CX_A); \
        vec_st(cx1_2, c1_base + 2 * v_num, CX_A); vec_st(cx1_3, c1_base + 3 * v_num, CX_A); \
        vec_st(cy1_0, c1_base + 0 * v_num, CY_A); vec_st(cy1_1, c1_base + 1 * v_num, CY_A); \
        vec_st(cy1_2, c1_base + 2 * v_num, CY_A); vec_st(cy1_3, c1_base + 3 * v_num, CY_A); \
        vec_st(cx2_0, c2_base + 0 * v_num, CX_A); vec_st(cx2_1, c2_base + 1 * v_num, CX_A); \
        vec_st(cx2_2, c2_base + 2 * v_num, CX_A); vec_st(cx2_3, c2_base + 3 * v_num, CX_A); \
        vec_st(cy2_0, c2_base + 0 * v_num, CY_A); vec_st(cy2_1, c2_base + 1 * v_num, CY_A); \
        vec_st(cy2_2, c2_base + 2 * v_num, CY_A); vec_st(cy2_3, c2_base + 3 * v_num, CY_A); \
        vec_st(cx3_0, c3_base + 0 * v_num, CX_A); vec_st(cx3_1, c3_base + 1 * v_num, CX_A); \
        vec_st(cx3_2, c3_base + 2 * v_num, CX_A); vec_st(cx3_3, c3_base + 3 * v_num, CX_A); \
        vec_st(cy3_0, c3_base + 0 * v_num, CY_A); vec_st(cy3_1, c3_base + 1 * v_num, CY_A); \
        vec_st(cy3_2, c3_base + 2 * v_num, CY_A); vec_st(cy3_3, c3_base + 3 * v_num, CY_A); \
    } \
    for (; m_k < M_cur; m_k++) { \
        int c0_base = m_k * N_block; \
        lvector double cx0_0 = vec_ld(c0_base + 0 * v_num, CX_A); \
        lvector double cx0_1 = vec_ld(c0_base + 1 * v_num, CX_A); \
        lvector double cx0_2 = vec_ld(c0_base + 2 * v_num, CX_A); \
        lvector double cx0_3 = vec_ld(c0_base + 3 * v_num, CX_A); \
        lvector double cy0_0 = vec_ld(c0_base + 0 * v_num, CY_A); \
        lvector double cy0_1 = vec_ld(c0_base + 1 * v_num, CY_A); \
        lvector double cy0_2 = vec_ld(c0_base + 2 * v_num, CY_A); \
        lvector double cy0_3 = vec_ld(c0_base + 3 * v_num, CY_A); \
        for (int k_k = 0; k_k < (K_CUR); k_k++) { \
            int b_base = k_k * N_block; \
            lvector double bx0 = vec_ld(b_base + 0 * v_num, BX_A); \
            lvector double bx1 = vec_ld(b_base + 1 * v_num, BX_A); \
            lvector double bx2 = vec_ld(b_base + 2 * v_num, BX_A); \
            lvector double bx3 = vec_ld(b_base + 3 * v_num, BX_A); \
            lvector double by0 = vec_ld(b_base + 0 * v_num, BY_A); \
            lvector double by1 = vec_ld(b_base + 1 * v_num, BY_A); \
            lvector double by2 = vec_ld(b_base + 2 * v_num, BY_A); \
            lvector double by3 = vec_ld(b_base + 3 * v_num, BY_A); \
            double a_r = AX_S[m_k * K_block + k_k]; \
            double a_i = AY_S[m_k * K_block + k_k]; \
            lvector double ax_v = vec_svbcast(a_r); \
            lvector double ay_v = vec_svbcast(a_i); \
            lvector double nay_v = vec_svbcast(-a_i); \
            cx0_0 = vec_mula(ax_v, bx0, cx0_0); cx0_0 = vec_mula(nay_v, by0, cx0_0); \
            cy0_0 = vec_mula(ax_v, by0, cy0_0); cy0_0 = vec_mula(ay_v,  bx0, cy0_0); \
            cx0_1 = vec_mula(ax_v, bx1, cx0_1); cx0_1 = vec_mula(nay_v, by1, cx0_1); \
            cy0_1 = vec_mula(ax_v, by1, cy0_1); cy0_1 = vec_mula(ay_v,  bx1, cy0_1); \
            cx0_2 = vec_mula(ax_v, bx2, cx0_2); cx0_2 = vec_mula(nay_v, by2, cx0_2); \
            cy0_2 = vec_mula(ax_v, by2, cy0_2); cy0_2 = vec_mula(ay_v,  bx2, cy0_2); \
            cx0_3 = vec_mula(ax_v, bx3, cx0_3); cx0_3 = vec_mula(nay_v, by3, cx0_3); \
            cy0_3 = vec_mula(ax_v, by3, cy0_3); cy0_3 = vec_mula(ay_v,  bx3, cy0_3); \
        } \
        vec_st(cx0_0, c0_base + 0 * v_num, CX_A); vec_st(cx0_1, c0_base + 1 * v_num, CX_A); \
        vec_st(cx0_2, c0_base + 2 * v_num, CX_A); vec_st(cx0_3, c0_base + 3 * v_num, CX_A); \
        vec_st(cy0_0, c0_base + 0 * v_num, CY_A); vec_st(cy0_1, c0_base + 1 * v_num, CY_A); \
        vec_st(cy0_2, c0_base + 2 * v_num, CY_A); vec_st(cy0_3, c0_base + 3 * v_num, CY_A); \
    } \
} while (0)

// ==============================================================================
// matrix_mul_pipeline_opt:
// 三缓冲流水线 GEMM 计算内核（优化版）
// 保留三缓冲（C矩阵三缓冲旋转 + A/B双缓冲），计算部分采用 4x4 向量寄存器分块
// ==============================================================================
__global__ void matrix_mul_pipeline_opt(
        int coreNum,
        int M_block,
        int K_block,
        int N_block,
        int M,
        int K,
        int N,
        int *N_thread,
        double *A_x,
        double *A_y,
        double *B_x,
        double *B_y,
        double *C_x,
        double *C_y)
{
    int core_id = get_thread_id();
    if (core_id < coreNum)
    {
        if (N_thread[core_id] <= 0) return;
        int col_offset = 0;
        for (int i = 0; i < core_id; i++) {
            col_offset += N_thread[i];
        }
        double *Bx_fixed = &B_x[col_offset];
        double *By_fixed = &B_y[col_offset];
        double *Cx_fixed = &C_x[col_offset];
        double *Cy_fixed = &C_y[col_offset];

        double *Ax_s1 = scalar_malloc(M_block * K_block * sizeof(double));
        double *Ay_s1 = scalar_malloc(M_block * K_block * sizeof(double));
        double *Ax_s2 = scalar_malloc(M_block * K_block * sizeof(double));
        double *Ay_s2 = scalar_malloc(M_block * K_block * sizeof(double));

        lvector double *Bx_a1 = vector_malloc(K_block * N_block * sizeof(double));
        lvector double *By_a1 = vector_malloc(K_block * N_block * sizeof(double));
        lvector double *Bx_a2 = vector_malloc(K_block * N_block * sizeof(double));
        lvector double *By_a2 = vector_malloc(K_block * N_block * sizeof(double));

        lvector double *Cx_a1 = vector_malloc(M_block * N_block * sizeof(double));
        lvector double *Cy_a1 = vector_malloc(M_block * N_block * sizeof(double));
        lvector double *Cx_a2 = vector_malloc(M_block * N_block * sizeof(double));
        lvector double *Cy_a2 = vector_malloc(M_block * N_block * sizeof(double));
        lvector double *Cx_a3 = vector_malloc(M_block * N_block * sizeof(double));
        lvector double *Cy_a3 = vector_malloc(M_block * N_block * sizeof(double));

        double *Axs_src = NULL;
        double *Ays_src = NULL;
        double *Bxa_src = NULL;
        double *Bya_src = NULL;
        double *Cxa_src1 = NULL;
        double *Cya_src1 = NULL;
        double *Cxa_src2 = NULL;
        double *Cya_src2 = NULL;
        double *Cxa_src3 = NULL;
        double *Cya_src3 = NULL;

        unsigned int ch1 = 0, ch2 = 0, ch3 = 0, ch4 = 0, ch5 = 0, ch6 = 0, ch7 = 0, ch8 = 0;
        unsigned int chc1 = 0, chc2 = 0, chc3 = 0, chc4 = 0, chc5 = 0, chc6 = 0;
        unsigned int chendx = 0, chendy = 0;

        int K_cur1 = 0, K_cur2 = 0;
        int N_start = 0, N_end = 0;
        int base_idx = 0, idx = 0;

        for (int i = 0; i < M; i += M_block) {
            int M_cur = (i + M_block > M) ? (M - i) : M_block;
            int N_cur1 = 0, N_cur2 = 0, N_cur3 = 0;
            for (int j = 0; ; j += N_block)
            {
                if (j > 0) {
                    // 写回 buffer3
                    if (j > N_block) {
                        chc5 = dma_p2p(Cx_a3, M_cur, N_cur3 * sizeof(double), (N_block - N_cur3) * sizeof(double),
                                       Cxa_src3, M_cur, N_cur3 * sizeof(double), (N - N_cur3) * sizeof(double), false, 0);
                        chc6 = dma_p2p(Cy_a3, M_cur, N_cur3 * sizeof(double), (N_block - N_cur3) * sizeof(double),
                                       Cya_src3, M_cur, N_cur3 * sizeof(double), (N - N_cur3) * sizeof(double), false, 0);
                        dma_wait(chc3);
                        dma_wait(chc4);
                        if (j > N_thread[core_id]) {
                            chendx = chc5;
                            chendy = chc6;
                            break;
                        }
                    }

                    // 加载 buffer2
                    if (j < N_thread[core_id]) {
                        N_cur2 = (j + N_block > N_thread[core_id]) ? (N_thread[core_id] - j) : N_block;
                        Cxa_src2 = Cx_fixed + (i * N + j);
                        Cya_src2 = Cy_fixed + (i * N + j);
                        chc3 = dma_p2p(Cxa_src2, M_cur, N_cur2 * sizeof(double), (N - N_cur2) * sizeof(double),
                                       Cx_a2, M_cur, N_cur2 * sizeof(double), (N_block - N_cur2) * sizeof(double), false, 0);
                        chc4 = dma_p2p(Cya_src2, M_cur, N_cur2 * sizeof(double), (N - N_cur2) * sizeof(double),
                                       Cy_a2, M_cur, N_cur2 * sizeof(double), (N_block - N_cur2) * sizeof(double), false, 0);
                    }

                    // 计算 buffer1 (使用 4x4 寄存器分块优化内核)
                    dma_wait(chc1);
                    dma_wait(chc2);
                    N_start = N_cur1 / v_num;
                    N_end = N_block / v_num;
                    for (int q = 0; ; q += K_block) {
                        if (q > 0) {
                            if (q < K) {
                                K_cur2 = (q + K_block > K) ? (K - q) : K_block;

                                if (N_cur1 != N_block) {
                                    for (int i_a = 0; i_a < K_cur2; i_a++) {
                                        base_idx = i_a * (N_block / v_num);
                                        for (int j_a = N_start; j_a < N_end; j_a++) {
                                            idx = base_idx + j_a;
                                            Bx_a2[idx] = 0;
                                            By_a2[idx] = 0;
                                        }
                                    }
                                }

                                Axs_src = A_x + (i * K + q);
                                Ays_src = A_y + (i * K + q);
                                Bxa_src = Bx_fixed + (q * N + (j - N_block));
                                Bya_src = By_fixed + (q * N + (j - N_block));

                                ch5 = dma_p2p(Axs_src, M_cur, K_cur2 * sizeof(double), (K - K_cur2) * sizeof(double),
                                              Ax_s2, M_cur, K_cur2 * sizeof(double), (K_block - K_cur2) * sizeof(double), false, 0);
                                ch6 = dma_p2p(Ays_src, M_cur, K_cur2 * sizeof(double), (K - K_cur2) * sizeof(double),
                                              Ay_s2, M_cur, K_cur2 * sizeof(double), (K_block - K_cur2) * sizeof(double), false, 0);
                                ch7 = dma_p2p(Bxa_src, K_cur2, N_cur1 * sizeof(double), (N - N_cur1) * sizeof(double),
                                              Bx_a2, K_cur2, N_cur1 * sizeof(double), (N_block - N_cur1) * sizeof(double), false, 0);
                                ch8 = dma_p2p(Bya_src, K_cur2, N_cur1 * sizeof(double), (N - N_cur1) * sizeof(double),
                                              By_a2, K_cur2, N_cur1 * sizeof(double), (N_block - N_cur1) * sizeof(double), false, 0);
                            }

                            dma_wait(ch1);
                            dma_wait(ch2);
                            dma_wait(ch3);
                            dma_wait(ch4);

                            COMPUTE_K_BLOCK_OPT_C(Ax_s1, Ay_s1, Bx_a1, By_a1, Cx_a1, Cy_a1, K_cur1);

                            if (q >= K) break;
                            q += K_block;
                        }
                        if (q < K) {
                            K_cur1 = (q + K_block > K) ? (K - q) : K_block;

                            if (N_cur1 != N_block) {
                                for (int i_a = 0; i_a < K_cur1; i_a++) {
                                    base_idx = i_a * (N_block / v_num);
                                    for (int j_a = N_start; j_a < N_end; j_a++) {
                                        idx = base_idx + j_a;
                                        Bx_a1[idx] = 0;
                                        By_a1[idx] = 0;
                                    }
                                }
                            }

                            Axs_src = A_x + (i * K + q);
                            Ays_src = A_y + (i * K + q);
                            Bxa_src = Bx_fixed + (q * N + (j - N_block));
                            Bya_src = By_fixed + (q * N + (j - N_block));

                            ch1 = dma_p2p(Axs_src, M_cur, K_cur1 * sizeof(double), (K - K_cur1) * sizeof(double),
                                          Ax_s1, M_cur, K_cur1 * sizeof(double), (K_block - K_cur1) * sizeof(double), false, 0);
                            ch2 = dma_p2p(Ays_src, M_cur, K_cur1 * sizeof(double), (K - K_cur1) * sizeof(double),
                                          Ay_s1, M_cur, K_cur1 * sizeof(double), (K_block - K_cur1) * sizeof(double), false, 0);
                            ch3 = dma_p2p(Bxa_src, K_cur1, N_cur1 * sizeof(double), (N - N_cur1) * sizeof(double),
                                          Bx_a1, K_cur1, N_cur1 * sizeof(double), (N_block - N_cur1) * sizeof(double), false, 0);
                            ch4 = dma_p2p(Bya_src, K_cur1, N_cur1 * sizeof(double), (N - N_cur1) * sizeof(double),
                                          By_a1, K_cur1, N_cur1 * sizeof(double), (N_block - N_cur1) * sizeof(double), false, 0);
                        }
                        if (q > 0) {
                            dma_wait(ch5);
                            dma_wait(ch6);
                            dma_wait(ch7);
                            dma_wait(ch8);

                            COMPUTE_K_BLOCK_OPT_C(Ax_s2, Ay_s2, Bx_a2, By_a2, Cx_a1, Cy_a1, K_cur2);

                            if (q >= K) break;
                        }
                    }

                    // 写回 buffer1
                    chc1 = dma_p2p(Cx_a1, M_cur, N_cur1 * sizeof(double), (N_block - N_cur1) * sizeof(double),
                                   Cxa_src1, M_cur, N_cur1 * sizeof(double), (N - N_cur1) * sizeof(double), false, 0);
                    chc2 = dma_p2p(Cy_a1, M_cur, N_cur1 * sizeof(double), (N_block - N_cur1) * sizeof(double),
                                   Cya_src1, M_cur, N_cur1 * sizeof(double), (N - N_cur1) * sizeof(double), false, 0);
                    if (j >= N_thread[core_id]) {
                        chendx = chc1;
                        chendy = chc2;
                        break;
                    }

                    // 计算 buffer2 (使用 4x4 寄存器分块优化内核)
                    dma_wait(chc3);
                    dma_wait(chc4);
                    N_start = N_cur2 / v_num;
                    N_end = N_block / v_num;
                    for (int q = 0; ; q += K_block) {
                        if (q > 0) {
                            if (q < K) {
                                K_cur2 = (q + K_block > K) ? (K - q) : K_block;

                                if (N_cur2 != N_block) {
                                    for (int i_a = 0; i_a < K_cur2; i_a++) {
                                        base_idx = i_a * (N_block / v_num);
                                        for (int j_a = N_start; j_a < N_end; j_a++) {
                                            idx = base_idx + j_a;
                                            Bx_a2[idx] = 0;
                                            By_a2[idx] = 0;
                                        }
                                    }
                                }

                                Axs_src = A_x + (i * K + q);
                                Ays_src = A_y + (i * K + q);
                                Bxa_src = Bx_fixed + (q * N + j);
                                Bya_src = By_fixed + (q * N + j);

                                ch5 = dma_p2p(Axs_src, M_cur, K_cur2 * sizeof(double), (K - K_cur2) * sizeof(double),
                                              Ax_s2, M_cur, K_cur2 * sizeof(double), (K_block - K_cur2) * sizeof(double), false, 0);
                                ch6 = dma_p2p(Ays_src, M_cur, K_cur2 * sizeof(double), (K - K_cur2) * sizeof(double),
                                              Ay_s2, M_cur, K_cur2 * sizeof(double), (K_block - K_cur2) * sizeof(double), false, 0);
                                ch7 = dma_p2p(Bxa_src, K_cur2, N_cur2 * sizeof(double), (N - N_cur2) * sizeof(double),
                                              Bx_a2, K_cur2, N_cur2 * sizeof(double), (N_block - N_cur2) * sizeof(double), false, 0);
                                ch8 = dma_p2p(Bya_src, K_cur2, N_cur2 * sizeof(double), (N - N_cur2) * sizeof(double),
                                              By_a2, K_cur2, N_cur2 * sizeof(double), (N_block - N_cur2) * sizeof(double), false, 0);
                            }

                            dma_wait(ch1);
                            dma_wait(ch2);
                            dma_wait(ch3);
                            dma_wait(ch4);

                            COMPUTE_K_BLOCK_OPT_C(Ax_s1, Ay_s1, Bx_a1, By_a1, Cx_a2, Cy_a2, K_cur1);

                            if (q >= K) break;
                            q += K_block;
                        }
                        if (q < K) {
                            K_cur1 = (q + K_block > K) ? (K - q) : K_block;

                            if (N_cur2 != N_block) {
                                for (int i_a = 0; i_a < K_cur1; i_a++) {
                                    base_idx = i_a * (N_block / v_num);
                                    for (int j_a = N_start; j_a < N_end; j_a++) {
                                        idx = base_idx + j_a;
                                        Bx_a1[idx] = 0;
                                        By_a1[idx] = 0;
                                    }
                                }
                            }

                            Axs_src = A_x + (i * K + q);
                            Ays_src = A_y + (i * K + q);
                            Bxa_src = Bx_fixed + (q * N + j);
                            Bya_src = By_fixed + (q * N + j);

                            ch1 = dma_p2p(Axs_src, M_cur, K_cur1 * sizeof(double), (K - K_cur1) * sizeof(double),
                                          Ax_s1, M_cur, K_cur1 * sizeof(double), (K_block - K_cur1) * sizeof(double), false, 0);
                            ch2 = dma_p2p(Ays_src, M_cur, K_cur1 * sizeof(double), (K - K_cur1) * sizeof(double),
                                          Ay_s1, M_cur, K_cur1 * sizeof(double), (K_block - K_cur1) * sizeof(double), false, 0);
                            ch3 = dma_p2p(Bxa_src, K_cur1, N_cur2 * sizeof(double), (N - N_cur2) * sizeof(double),
                                          Bx_a1, K_cur1, N_cur2 * sizeof(double), (N_block - N_cur2) * sizeof(double), false, 0);
                            ch4 = dma_p2p(Bya_src, K_cur1, N_cur2 * sizeof(double), (N - N_cur2) * sizeof(double),
                                          By_a1, K_cur1, N_cur2 * sizeof(double), (N_block - N_cur2) * sizeof(double), false, 0);
                        }
                        if (q > 0) {
                            dma_wait(ch5);
                            dma_wait(ch6);
                            dma_wait(ch7);
                            dma_wait(ch8);

                            COMPUTE_K_BLOCK_OPT_C(Ax_s2, Ay_s2, Bx_a2, By_a2, Cx_a2, Cy_a2, K_cur2);

                            if (q >= K) break;
                        }
                    }

                    j += N_block;
                    // 加载 buffer3
                    if (j > 2 * N_block) {
                        dma_wait(chc5);
                        dma_wait(chc6);
                    }
                    if (j < N_thread[core_id]) {
                        N_cur3 = (j + N_block > N_thread[core_id]) ? (N_thread[core_id] - j) : N_block;
                        Cxa_src3 = Cx_fixed + (i * N + j);
                        Cya_src3 = Cy_fixed + (i * N + j);
                        chc5 = dma_p2p(Cxa_src3, M_cur, N_cur3 * sizeof(double), (N - N_cur3) * sizeof(double),
                                       Cx_a3, M_cur, N_cur3 * sizeof(double), (N_block - N_cur3) * sizeof(double), false, 0);
                        chc6 = dma_p2p(Cya_src3, M_cur, N_cur3 * sizeof(double), (N - N_cur3) * sizeof(double),
                                       Cy_a3, M_cur, N_cur3 * sizeof(double), (N_block - N_cur3) * sizeof(double), false, 0);

                        // 计算 buffer3 (使用 4x4 寄存器分块优化内核)
                        dma_wait(chc5);
                        dma_wait(chc6);
                        N_start = N_cur3 / v_num;
                        N_end = N_block / v_num;
                        for (int q = 0; ; q += K_block) {
                            if (q > 0) {
                                if (q < K) {
                                    K_cur2 = (q + K_block > K) ? (K - q) : K_block;

                                    if (N_cur3 != N_block) {
                                        for (int i_a = 0; i_a < K_cur2; i_a++) {
                                            base_idx = i_a * (N_block / v_num);
                                            for (int j_a = N_start; j_a < N_end; j_a++) {
                                                idx = base_idx + j_a;
                                                Bx_a2[idx] = 0;
                                                By_a2[idx] = 0;
                                            }
                                        }
                                    }

                                    Axs_src = A_x + (i * K + q);
                                    Ays_src = A_y + (i * K + q);
                                    Bxa_src = Bx_fixed + (q * N + j);
                                    Bya_src = By_fixed + (q * N + j);

                                    ch5 = dma_p2p(Axs_src, M_cur, K_cur2 * sizeof(double), (K - K_cur2) * sizeof(double),
                                                  Ax_s2, M_cur, K_cur2 * sizeof(double), (K_block - K_cur2) * sizeof(double), false, 0);
                                    ch6 = dma_p2p(Ays_src, M_cur, K_cur2 * sizeof(double), (K - K_cur2) * sizeof(double),
                                                  Ay_s2, M_cur, K_cur2 * sizeof(double), (K_block - K_cur2) * sizeof(double), false, 0);
                                    ch7 = dma_p2p(Bxa_src, K_cur2, N_cur3 * sizeof(double), (N - N_cur3) * sizeof(double),
                                                  Bx_a2, K_cur2, N_cur3 * sizeof(double), (N_block - N_cur3) * sizeof(double), false, 0);
                                    ch8 = dma_p2p(Bya_src, K_cur2, N_cur3 * sizeof(double), (N - N_cur3) * sizeof(double),
                                                  By_a2, K_cur2, N_cur3 * sizeof(double), (N_block - N_cur3) * sizeof(double), false, 0);
                                }

                                dma_wait(ch1);
                                dma_wait(ch2);
                                dma_wait(ch3);
                                dma_wait(ch4);

                                COMPUTE_K_BLOCK_OPT_C(Ax_s1, Ay_s1, Bx_a1, By_a1, Cx_a3, Cy_a3, K_cur1);

                                if (q >= K) break;
                                q += K_block;
                            }
                            if (q < K) {
                                K_cur1 = (q + K_block > K) ? (K - q) : K_block;

                                if (N_cur3 != N_block) {
                                    for (int i_a = 0; i_a < K_cur1; i_a++) {
                                        base_idx = i_a * (N_block / v_num);
                                        for (int j_a = N_start; j_a < N_end; j_a++) {
                                            idx = base_idx + j_a;
                                            Bx_a1[idx] = 0;
                                            By_a1[idx] = 0;
                                        }
                                    }
                                }

                                Axs_src = A_x + (i * K + q);
                                Ays_src = A_y + (i * K + q);
                                Bxa_src = Bx_fixed + (q * N + j);
                                Bya_src = By_fixed + (q * N + j);

                                ch1 = dma_p2p(Axs_src, M_cur, K_cur1 * sizeof(double), (K - K_cur1) * sizeof(double),
                                              Ax_s1, M_cur, K_cur1 * sizeof(double), (K_block - K_cur1) * sizeof(double), false, 0);
                                ch2 = dma_p2p(Ays_src, M_cur, K_cur1 * sizeof(double), (K - K_cur1) * sizeof(double),
                                              Ay_s1, M_cur, K_cur1 * sizeof(double), (K_block - K_cur1) * sizeof(double), false, 0);
                                ch3 = dma_p2p(Bxa_src, K_cur1, N_cur3 * sizeof(double), (N - N_cur3) * sizeof(double),
                                              Bx_a1, K_cur1, N_cur3 * sizeof(double), (N_block - N_cur3) * sizeof(double), false, 0);
                                ch4 = dma_p2p(Bya_src, K_cur1, N_cur3 * sizeof(double), (N - N_cur3) * sizeof(double),
                                              By_a1, K_cur1, N_cur3 * sizeof(double), (N_block - N_cur3) * sizeof(double), false, 0);
                            }
                            if (q > 0) {
                                dma_wait(ch5);
                                dma_wait(ch6);
                                dma_wait(ch7);
                                dma_wait(ch8);

                                COMPUTE_K_BLOCK_OPT_C(Ax_s2, Ay_s2, Bx_a2, By_a2, Cx_a3, Cy_a3, K_cur2);

                                if (q >= K) break;
                            }
                        }
                    }

                    // 写回 buffer2
                    chc3 = dma_p2p(Cx_a2, M_cur, N_cur2 * sizeof(double), (N_block - N_cur2) * sizeof(double),
                                   Cxa_src2, M_cur, N_cur2 * sizeof(double), (N - N_cur2) * sizeof(double), false, 0);
                    chc4 = dma_p2p(Cy_a2, M_cur, N_cur2 * sizeof(double), (N_block - N_cur2) * sizeof(double),
                                   Cya_src2, M_cur, N_cur2 * sizeof(double), (N - N_cur2) * sizeof(double), false, 0);
                    dma_wait(chc1);
                    dma_wait(chc2);
                    if (j >= N_thread[core_id]) {
                        chendx = chc3;
                        chendy = chc4;
                        break;
                    }
                    j += N_block;
                }

                // 加载 buffer1
                if (j < N_thread[core_id]) {
                    N_cur1 = (j + N_block > N_thread[core_id]) ? (N_thread[core_id] - j) : N_block;
                    Cxa_src1 = Cx_fixed + (i * N + j);
                    Cya_src1 = Cy_fixed + (i * N + j);
                    chc1 = dma_p2p(Cxa_src1, M_cur, N_cur1 * sizeof(double), (N - N_cur1) * sizeof(double),
                                   Cx_a1, M_cur, N_cur1 * sizeof(double), (N_block - N_cur1) * sizeof(double), false, 0);
                    chc2 = dma_p2p(Cya_src1, M_cur, N_cur1 * sizeof(double), (N - N_cur1) * sizeof(double),
                                   Cy_a1, M_cur, N_cur1 * sizeof(double), (N_block - N_cur1) * sizeof(double), false, 0);
                }
            }
            dma_wait(chendx);
            dma_wait(chendy);
        }

        scalar_free(Ax_s1);
        scalar_free(Ay_s1);
        scalar_free(Ax_s2);
        scalar_free(Ay_s2);

        vector_free(Bx_a1);
        vector_free(By_a1);
        vector_free(Bx_a2);
        vector_free(By_a2);

        vector_free(Cx_a1);
        vector_free(Cy_a1);
        vector_free(Cx_a2);
        vector_free(Cy_a2);
        vector_free(Cx_a3);
        vector_free(Cy_a3);
    }
}