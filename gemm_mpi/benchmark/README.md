# OpenBLAS 基线对比评测模块 (Benchmark Suite)

本目录为学术论文的**工业级基准对比（Baseline Comparison）**与**数值精度校验（Accuracy Validation）**提供完整的评测程序与实验原始数据。

---

## 1. 目录文件结构

```
E:\personal\论文\gemm_mpi\benchmark/
├── gemm_openblas_bench.c    # OpenBLAS vs MT-3000 DSP 综合基准测试驱动源码
├── openblas_vs_asd.log      # 全规模（512 ~ 16384）对比实验原始日志与准确率报告
├── Makefile                 # 自动化编译规则文件（链接 OpenBLAS、Hthread 与 OpenMP）
└── README.md                # 本说明文档
```

---

## 2. 评测方案与实验设计（Paper Baseline Design）

在学术论文的“实验评估与对比分析”章节中，基线对比遵循严谨的对照实验设计：

### 2.1 对比平台与软件栈
* **异构加速端（DSP）**：国产银河天河 MT-3000 自主 DSP 矩阵加速核，运行最终优化内核 `gemm_dsp_AM_SM_pipeline_opt`（三缓冲流水线 + $4\times 4$ 向量寄存器分块 + 双发射 FMA 融合）。
* **通用处理器端（CPU）**：国产飞腾 FT-2000+ 多核 ARM64 处理器，运行业界广泛采用的成熟工业级高优化数学库 **OpenBLAS**。

### 2.2 对照组划分
1. **Group A：单核计算密度对比（Single Core Efficiency）**
   * **测试对象**：1 个 MT-3000 DSP 核心 vs 1 个 FT-2000+ CPU 核心。
   * **OpenBLAS 算子实现**：
     * **4x DGEMM**：分拆为 4 次实数双精度矩阵乘法：
       $$\text{Re}(C) = A_r B_r - A_i B_i, \quad \text{Im}(C) = A_r B_i + A_i B_r$$
     * **1x ZGEMM**：调用原生复数双精度接口 `cblas_zgemm`。
   * **论文核心结论**：DSP 单核在规模 $\ge 2048$ 时稳定维持在 **48.4 ~ 49.2 GFLOPS**，相对 CPU 单核稳定保持 **6.1x ~ 7.3x 的显著加速比**，证明了 DSP 硬件宽向量与专用片上 SRAM 高吞吐流水线的单核性能优势。

2. **Group B：簇级与节点级加速比对比（Cluster / Node Scale-up）**
   * **测试对象**：
     * DSP 单簇 24 核心（对应单卡设备算力）；
     * CPU 等效簇 4 核心（OpenMP 4 线程）；
     * CPU 整节点 16 核心（OpenMP 16 线程）。
   * **论文核心结论**：
     * 相比 CPU 4 核，DSP 24 核取得 **3.5x ~ 9.3x** 的加速比；
     * 相比 CPU 16 核，单簇 DSP 取得 **1.6x ~ 6.6x** 的加速比；
     * 换算到整节点（4 簇共 96 个 DSP 核心）对标 CPU 16 核，整节点异构加速比达到 **7.9x ~ 26.3x**。

3. **数值精度绝对误差校验（Accuracy Check）**
   * 对比方式：将 DSP 的输出矩阵实部与虚部结果，与 CPU OpenBLAS 标准计算结果逐元素进行绝对差值比较：
     $$\text{err}_x = \max |C_{x,\text{DSP}} - C_{x,\text{BLAS}}|, \quad \text{err}_y = \max |C_{y,\text{DSP}} - C_{y,\text{BLAS}}|$$
   * 检验结果（见 [`openblas_vs_asd.log`](file:///E:/personal/论文/gemm_mpi/benchmark/openblas_vs_asd.log)）：
     * $512 \times 512$：$\text{err}_x = 7.28 \times 10^{-12}, \ \text{err}_y = 1.13 \times 10^{-10}$
     * $8192 \times 8192$：$\text{err}_x = 1.89 \times 10^{-10}, \ \text{err}_y = 7.57 \times 10^{-09}$（全量 $6.71 \times 10^7$ 复数元素比对）
     * $16384 \times 16384$：$\text{err}_x = 1.05 \times 10^{-09}, \ \text{err}_y = 2.18 \times 10^{-08}$
   * 结论：最大误差远低于单精度阈值，严格处于标准 FP64 累加截断误差范围内，兼具极致算力与高数值精度。

---

## 3. 编译与运行指引

在超算环境编译测试驱动：
```bash
cd benchmark
make
```

运行单次测试（例如矩阵规模 $N=8192$）：
```bash
./gemm_openblas_bench 8192
```
