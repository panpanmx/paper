#include <compiler/m3000.h>
#include "hthread_device.h"

#define v_num 16
#define GSM_SIZE 300001

__gsm__ double Ax_g[GSM_SIZE] = {0};
__gsm__ double Ay_g[GSM_SIZE] = {0};


#define COMPUTE_K_BLOCK_OPT(AX_S, AY_S, BX_A, BY_A, K_CUR) \
do { \
    int m_k = 0; \
    for (; m_k + 3 < M_cur; m_k += 4) { \
        int c0_base = (m_k + 0) * N_block; \
        int c1_base = (m_k + 1) * N_block; \
        int c2_base = (m_k + 2) * N_block; \
        int c3_base = (m_k + 3) * N_block; \
        lvector double cx0_0 = vec_ld(c0_base + 0 * v_num, Cx_a); \
        lvector double cx0_1 = vec_ld(c0_base + 1 * v_num, Cx_a); \
        lvector double cx0_2 = vec_ld(c0_base + 2 * v_num, Cx_a); \
        lvector double cx0_3 = vec_ld(c0_base + 3 * v_num, Cx_a); \
        lvector double cy0_0 = vec_ld(c0_base + 0 * v_num, Cy_a); \
        lvector double cy0_1 = vec_ld(c0_base + 1 * v_num, Cy_a); \
        lvector double cy0_2 = vec_ld(c0_base + 2 * v_num, Cy_a); \
        lvector double cy0_3 = vec_ld(c0_base + 3 * v_num, Cy_a); \
        lvector double cx1_0 = vec_ld(c1_base + 0 * v_num, Cx_a); \
        lvector double cx1_1 = vec_ld(c1_base + 1 * v_num, Cx_a); \
        lvector double cx1_2 = vec_ld(c1_base + 2 * v_num, Cx_a); \
        lvector double cx1_3 = vec_ld(c1_base + 3 * v_num, Cx_a); \
        lvector double cy1_0 = vec_ld(c1_base + 0 * v_num, Cy_a); \
        lvector double cy1_1 = vec_ld(c1_base + 1 * v_num, Cy_a); \
        lvector double cy1_2 = vec_ld(c1_base + 2 * v_num, Cy_a); \
        lvector double cy1_3 = vec_ld(c1_base + 3 * v_num, Cy_a); \
        lvector double cx2_0 = vec_ld(c2_base + 0 * v_num, Cx_a); \
        lvector double cx2_1 = vec_ld(c2_base + 1 * v_num, Cx_a); \
        lvector double cx2_2 = vec_ld(c2_base + 2 * v_num, Cx_a); \
        lvector double cx2_3 = vec_ld(c2_base + 3 * v_num, Cx_a); \
        lvector double cy2_0 = vec_ld(c2_base + 0 * v_num, Cy_a); \
        lvector double cy2_1 = vec_ld(c2_base + 1 * v_num, Cy_a); \
        lvector double cy2_2 = vec_ld(c2_base + 2 * v_num, Cy_a); \
        lvector double cy2_3 = vec_ld(c2_base + 3 * v_num, Cy_a); \
        lvector double cx3_0 = vec_ld(c3_base + 0 * v_num, Cx_a); \
        lvector double cx3_1 = vec_ld(c3_base + 1 * v_num, Cx_a); \
        lvector double cx3_2 = vec_ld(c3_base + 2 * v_num, Cx_a); \
        lvector double cx3_3 = vec_ld(c3_base + 3 * v_num, Cx_a); \
        lvector double cy3_0 = vec_ld(c3_base + 0 * v_num, Cy_a); \
        lvector double cy3_1 = vec_ld(c3_base + 1 * v_num, Cy_a); \
        lvector double cy3_2 = vec_ld(c3_base + 2 * v_num, Cy_a); \
        lvector double cy3_3 = vec_ld(c3_base + 3 * v_num, Cy_a); \
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
        vec_st(cx0_0, c0_base + 0 * v_num, Cx_a); vec_st(cx0_1, c0_base + 1 * v_num, Cx_a); \
        vec_st(cx0_2, c0_base + 2 * v_num, Cx_a); vec_st(cx0_3, c0_base + 3 * v_num, Cx_a); \
        vec_st(cy0_0, c0_base + 0 * v_num, Cy_a); vec_st(cy0_1, c0_base + 1 * v_num, Cy_a); \
        vec_st(cy0_2, c0_base + 2 * v_num, Cy_a); vec_st(cy0_3, c0_base + 3 * v_num, Cy_a); \
        vec_st(cx1_0, c1_base + 0 * v_num, Cx_a); vec_st(cx1_1, c1_base + 1 * v_num, Cx_a); \
        vec_st(cx1_2, c1_base + 2 * v_num, Cx_a); vec_st(cx1_3, c1_base + 3 * v_num, Cx_a); \
        vec_st(cy1_0, c1_base + 0 * v_num, Cy_a); vec_st(cy1_1, c1_base + 1 * v_num, Cy_a); \
        vec_st(cy1_2, c1_base + 2 * v_num, Cy_a); vec_st(cy1_3, c1_base + 3 * v_num, Cy_a); \
        vec_st(cx2_0, c2_base + 0 * v_num, Cx_a); vec_st(cx2_1, c2_base + 1 * v_num, Cx_a); \
        vec_st(cx2_2, c2_base + 2 * v_num, Cx_a); vec_st(cx2_3, c2_base + 3 * v_num, Cx_a); \
        vec_st(cy2_0, c2_base + 0 * v_num, Cy_a); vec_st(cy2_1, c2_base + 1 * v_num, Cy_a); \
        vec_st(cy2_2, c2_base + 2 * v_num, Cy_a); vec_st(cy2_3, c2_base + 3 * v_num, Cy_a); \
        vec_st(cx3_0, c3_base + 0 * v_num, Cx_a); vec_st(cx3_1, c3_base + 1 * v_num, Cx_a); \
        vec_st(cx3_2, c3_base + 2 * v_num, Cx_a); vec_st(cx3_3, c3_base + 3 * v_num, Cx_a); \
        vec_st(cy3_0, c3_base + 0 * v_num, Cy_a); vec_st(cy3_1, c3_base + 1 * v_num, Cy_a); \
        vec_st(cy3_2, c3_base + 2 * v_num, Cy_a); vec_st(cy3_3, c3_base + 3 * v_num, Cy_a); \
    } \
    for (; m_k < M_cur; m_k++) { \
        int c0_base = m_k * N_block; \
        lvector double cx0_0 = vec_ld(c0_base + 0 * v_num, Cx_a); \
        lvector double cx0_1 = vec_ld(c0_base + 1 * v_num, Cx_a); \
        lvector double cx0_2 = vec_ld(c0_base + 2 * v_num, Cx_a); \
        lvector double cx0_3 = vec_ld(c0_base + 3 * v_num, Cx_a); \
        lvector double cy0_0 = vec_ld(c0_base + 0 * v_num, Cy_a); \
        lvector double cy0_1 = vec_ld(c0_base + 1 * v_num, Cy_a); \
        lvector double cy0_2 = vec_ld(c0_base + 2 * v_num, Cy_a); \
        lvector double cy0_3 = vec_ld(c0_base + 3 * v_num, Cy_a); \
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
        vec_st(cx0_0, c0_base + 0 * v_num, Cx_a); vec_st(cx0_1, c0_base + 1 * v_num, Cx_a); \
        vec_st(cx0_2, c0_base + 2 * v_num, Cx_a); vec_st(cx0_3, c0_base + 3 * v_num, Cx_a); \
        vec_st(cy0_0, c0_base + 0 * v_num, Cy_a); vec_st(cy0_1, c0_base + 1 * v_num, Cy_a); \
        vec_st(cy0_2, c0_base + 2 * v_num, Cy_a); vec_st(cy0_3, c0_base + 3 * v_num, Cy_a); \
    } \
} while (0)

__global__ void matrix_mul_asd_opt(
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

        lvector double *Cx_a = vector_malloc(M_block * N_block * sizeof(double));
        lvector double *Cy_a = vector_malloc(M_block * N_block * sizeof(double));

        double *Axs_src = NULL;
        double *Ays_src = NULL;
        double *Bxa_src = NULL;
        double *Bya_src = NULL;
        double *Cxa_src = NULL;
        double *Cya_src = NULL;

        unsigned int ch, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8;

        for (int i = 0; i < M; i += M_block) {
            int M_cur = (i + M_block > M) ? (M - i) : M_block;

            for (int j = 0; j < N_thread[core_id]; j += N_block) {
                int N_cur = (j + N_block > N_thread[core_id]) ? (N_thread[core_id] - j) : N_block;

                Cxa_src = Cx_fixed + (i * N + j);
                Cya_src = Cy_fixed + (i * N + j);

                ch = dma_p2p(Cxa_src, M_cur, N_cur * sizeof(double), (N - N_cur) * sizeof(double),
                             Cx_a, M_cur, N_cur * sizeof(double), (N_block - N_cur) * sizeof(double), false, 0);
                dma_wait(ch);

                ch = dma_p2p(Cya_src, M_cur, N_cur * sizeof(double), (N - N_cur) * sizeof(double),
                             Cy_a, M_cur, N_cur * sizeof(double), (N_block - N_cur) * sizeof(double), false, 0);
                dma_wait(ch);

                int K_cur1, K_cur2;
                int N_start = N_cur / v_num;
                int N_end = N_block / v_num;
                for (int q = 0; ; q += K_block) {
                    if (q > 0) {
                        if (q < K) {
                            K_cur2 = (q + K_block > K) ? (K - q) : K_block;

                            if (N_cur != N_block) {
                                for (int i_a = 0; i_a < K_cur2; i_a++) {
                                    int base_idx = i_a * (N_block / v_num);
                                    for (int j_a = N_start; j_a < N_end; j_a++) {
                                        int idx = base_idx + j_a;
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
                            ch7 = dma_p2p(Bxa_src, K_cur2, N_cur * sizeof(double), (N - N_cur) * sizeof(double),
                                          Bx_a2, K_cur2, N_cur * sizeof(double), (N_block - N_cur) * sizeof(double), false, 0);
                            ch8 = dma_p2p(Bya_src, K_cur2, N_cur * sizeof(double), (N - N_cur) * sizeof(double),
                                          By_a2, K_cur2, N_cur * sizeof(double), (N_block - N_cur) * sizeof(double), false, 0);
                        }

                        dma_wait(ch1);
                        dma_wait(ch2);
                        dma_wait(ch3);
                        dma_wait(ch4);

                        COMPUTE_K_BLOCK_OPT(Ax_s1, Ay_s1, Bx_a1, By_a1, K_cur1);

                        if (q >= K) break;
                        q += K_block;
                    }
                    if (q < K) {
                        K_cur1 = (q + K_block > K) ? (K - q) : K_block;

                        if (N_cur != N_block) {
                            for (int i_a = 0; i_a < K_cur1; i_a++) {
                                int base_idx = i_a * (N_block / v_num);
                                for (int j_a = N_start; j_a < N_end; j_a++) {
                                    int idx = base_idx + j_a;
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
                        ch3 = dma_p2p(Bxa_src, K_cur1, N_cur * sizeof(double), (N - N_cur) * sizeof(double),
                                      Bx_a1, K_cur1, N_cur * sizeof(double), (N_block - N_cur) * sizeof(double), false, 0);
                        ch4 = dma_p2p(Bya_src, K_cur1, N_cur * sizeof(double), (N - N_cur) * sizeof(double),
                                      By_a1, K_cur1, N_cur * sizeof(double), (N_block - N_cur) * sizeof(double), false, 0);
                    }
                    if (q > 0) {
                        dma_wait(ch5);
                        dma_wait(ch6);
                        dma_wait(ch7);
                        dma_wait(ch8);

                        COMPUTE_K_BLOCK_OPT(Ax_s2, Ay_s2, Bx_a2, By_a2, K_cur2);

                        if (q >= K) break;
                    }
                }
                ch = dma_p2p(Cx_a, M_cur, N_cur * sizeof(double), (N_block - N_cur) * sizeof(double),
                             Cxa_src, M_cur, N_cur * sizeof(double), (N - N_cur) * sizeof(double), false, 0);
                dma_wait(ch);

                ch = dma_p2p(Cy_a, M_cur, N_cur * sizeof(double), (N_block - N_cur) * sizeof(double),
                             Cya_src, M_cur, N_cur * sizeof(double), (N - N_cur) * sizeof(double), false, 0);
                dma_wait(ch);
            }
        }

        scalar_free(Ax_s1);
        scalar_free(Ay_s1);
        scalar_free(Ax_s2);
        scalar_free(Ay_s2);

        vector_free(Bx_a1);
        vector_free(By_a1);
        vector_free(Bx_a2);
        vector_free(By_a2);

        vector_free(Cx_a);
        vector_free(Cy_a);
    }
}


// ==============================================================================
// matrix_mul_as_opt:
// 无缓冲 / 单缓冲 GEMM 计算内核（优化版）
// 保留单缓冲同步 DMA 行为，计算部分采用 4x4 向量寄存器分块与累加器常驻优化 (COMPUTE_K_BLOCK_OPT)
// ==============================================================================
__global__ void matrix_mul_as_opt(
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

        double *Ax_s = scalar_malloc(M_block * K_block * sizeof(double));
        double *Ay_s = scalar_malloc(M_block * K_block * sizeof(double));

        lvector double *Bx_a = vector_malloc(K_block * N_block * sizeof(double));
        lvector double *By_a = vector_malloc(K_block * N_block * sizeof(double));

        lvector double *Cx_a = vector_malloc(M_block * N_block * sizeof(double));
        lvector double *Cy_a = vector_malloc(M_block * N_block * sizeof(double));

        double *Axs_src = NULL;
        double *Ays_src = NULL;
        double *Bxa_src = NULL;
        double *Bya_src = NULL;
        double *Cxa_src = NULL;
        double *Cya_src = NULL;

        unsigned int ch;
        unsigned int ch1, ch2, ch3, ch4;

        for (int i = 0; i < M; i += M_block) {
            int M_cur = (i + M_block > M) ? (M - i) : M_block;

            for (int j = 0; j < N_thread[core_id]; j += N_block) {
                int N_cur = (j + N_block > N_thread[core_id]) ? (N_thread[core_id] - j) : N_block;

                Cxa_src = Cx_fixed + (i * N + j);
                Cya_src = Cy_fixed + (i * N + j);

                ch = dma_p2p(Cxa_src, M_cur, N_cur * sizeof(double), (N - N_cur) * sizeof(double),
                             Cx_a, M_cur, N_cur * sizeof(double), (N_block - N_cur) * sizeof(double), false, 0);
                dma_wait(ch);

                ch = dma_p2p(Cya_src, M_cur, N_cur * sizeof(double), (N - N_cur) * sizeof(double),
                             Cy_a, M_cur, N_cur * sizeof(double), (N_block - N_cur) * sizeof(double), false, 0);
                dma_wait(ch);

                int N_start = N_cur / v_num;
                int N_end = N_block / v_num;

                for (int q = 0; q < K; q += K_block) {
                    int K_cur = (q + K_block > K) ? (K - q) : K_block;

                    if (N_cur != N_block) {
                        for (int i_a = 0; i_a < K_cur; i_a++) {
                            int base_idx = i_a * (N_block / v_num);
                            for (int j_a = N_start; j_a < N_end; j_a++) {
                                int idx = base_idx + j_a;
                                Bx_a[idx] = 0;
                                By_a[idx] = 0;
                            }
                        }
                    }

                    Axs_src = A_x + (i * K + q);
                    Ays_src = A_y + (i * K + q);

                    Bxa_src = Bx_fixed + (q * N + j);
                    Bya_src = By_fixed + (q * N + j);

                    ch1 = dma_p2p(Axs_src, M_cur, K_cur * sizeof(double), (K - K_cur) * sizeof(double),
                                  Ax_s, M_cur, K_cur * sizeof(double), (K_block - K_cur) * sizeof(double), false, 0);

                    ch2 = dma_p2p(Ays_src, M_cur, K_cur * sizeof(double), (K - K_cur) * sizeof(double),
                                  Ay_s, M_cur, K_cur * sizeof(double), (K_block - K_cur) * sizeof(double), false, 0);

                    ch3 = dma_p2p(Bxa_src, K_cur, N_cur * sizeof(double), (N - N_cur) * sizeof(double),
                                  Bx_a, K_cur, N_cur * sizeof(double), (N_block - N_cur) * sizeof(double), false, 0);

                    ch4 = dma_p2p(Bya_src, K_cur, N_cur * sizeof(double), (N - N_cur) * sizeof(double),
                                  By_a, K_cur, N_cur * sizeof(double), (N_block - N_cur) * sizeof(double), false, 0);

                    dma_wait(ch1);
                    dma_wait(ch2);
                    dma_wait(ch3);
                    dma_wait(ch4);

                    COMPUTE_K_BLOCK_OPT(Ax_s, Ay_s, Bx_a, By_a, K_cur);
                }

                ch = dma_p2p(Cx_a, M_cur, N_cur * sizeof(double), (N_block - N_cur) * sizeof(double),
                             Cxa_src, M_cur, N_cur * sizeof(double), (N - N_cur) * sizeof(double), false, 0);
                dma_wait(ch);

                ch = dma_p2p(Cy_a, M_cur, N_cur * sizeof(double), (N_block - N_cur) * sizeof(double),
                             Cya_src, M_cur, N_cur * sizeof(double), (N - N_cur) * sizeof(double), false, 0);
                dma_wait(ch);
            }
        }

        scalar_free(Ax_s);
        scalar_free(Ay_s);

        vector_free(Bx_a);
        vector_free(By_a);

        vector_free(Cx_a);
        vector_free(Cy_a);
    }
}