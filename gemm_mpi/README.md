# MT-3000 异构超算平台高性能分布式复数 GEMM 优化工程

本项目是面向国产银河天河 MT-3000 异构超算系统（FT-2000+ 多核 ARM64 处理器 + MT-3000 自主 DSP 矩阵加速核）设计实现的高性能分布式双精度复数矩阵乘法系统（Complex Matrix Multiplication, CGEMM / ZGEMM）。

本文档对本工程的软件架构、核心算法创新、各模块源码分工及基线对照设计进行系统性说明，旨在为后续基于本代码撰写学术论文（系统结构、算法设计、消融实验与扩展性评测）提供权威、精准的代码级知识索引与技术映射。

---

## 1. 核心算法设计与最终版本代码（论文主推方案）

在分布式异构超算架构下，大规模双精度复数矩阵乘法面临三大严峻瓶颈：
1. **千节点通信风暴**：传统 1D 分布式广播算法下，随着节点数扩至百节点以上，全局广播矩阵 $B$ 引发严重的网络拥塞与通信延迟瓶颈。
2. **片上存储带宽墙与访存停顿**：DSP 核心与片上 SRAM/DDR 之间的数据搬运开销远大于计算耗时，且 RAW（Read-After-Write）数据冒险导致严重的流水线气泡。
3. **异构计算核心负载不均**：多核 DSP 的向量宽度对齐约束容易在边缘块产生非对齐访存与核心计算失衡。

针对上述瓶颈，本工程设计并实现了**全链路软硬件协同优化方案**。最终版本代码由**主机端最终主控**与**设备端最终内核**协同构成：

```
+-------------------------------------------------------------------------------------------------------+
|                                        最终版本系统架构方案                                            |
+-------------------------------------------------------------------------------------------------------+
|                                                                                                       |
|  【主机端分布式主控】: host_code/gemm_host_AB_opt6.c                                                     |
|  ┌─────────────────────────────────┐   ┌──────────────────────────────────┐   ┌────────────────────┐  |
|  │  2D 混合网格拓扑 (P/2 x 2)       │   │   非阻塞重叠通信与异步 pread      │   │ 3x8 DSP 核心网格   │  |
|  │  (B矩阵列拆分 + 组内A矩阵行划分) │ + │   (MPI_Type_vector + MPI_Ibcast) │ + │ 100% 均衡无气泡划分│  |
|  └─────────────────────────────────┘   └──────────────────────────────────┘   └────────────────────┘  |
|                                                     │                                                 |
|                                                     ▼ (hthread_group_exec / 参数封装传递)             |
|                                                                                                       |
|  【加速端计算内核】: device_code/kernel_pipeline.c  (对应算子接口: gemm_dsp_AM_SM_pipeline_opt)         |
|  ┌─────────────────────────────────┐   ┌──────────────────────────────────┐   ┌────────────────────┐  |
|  │  三缓冲流水线 (Triple Buffering)│   │ 4x4 向量寄存器分块 (Reg Tiling)  │   │ 双发射 FMA 指令融合│  |
|  │  (C 矩阵 3 级旋转 + A/B 双缓冲) │ + │ (32 个向量寄存器驻留累加)        │ + │ (vec_mula 消除气泡)│  |
|  └─────────────────────────────────┘   └──────────────────────────────────┘   └────────────────────┘  |
|                                                                                                       |
+-------------------------------------------------------------------------------------------------------+
```

### 1.1 主机端最终版本代码：`host_code/gemm_host_AB_opt6.c`

该程序是目前工程中最完善的 MPI 分布式主控代码，核心技术特性包括：

1. **二维混合进程拓扑（2D Process Grid）**：
   * 采用 `group_num = 2` 的 $P/2 \times 2$ 混合网格结构。
   * **矩阵 $B$ 按列划分为两半**，分别下发给两个进程组；组内各进程则对**矩阵 $A$ 按行划分**，避免了所有进程都重复接收并持有整个 $B$ 矩阵副本。
   * 在千节点（512 节点，2048 进程）规模下，将单次广播通信量由整矩阵降至子矩阵，并从全局广播转为组内组播，彻底粉碎了超算千节点级别的“全广播通信风暴”。
2. **非阻塞异步重叠通信（Async Non-blocking Comm）**：
   * 采用 `pread_matrix` 进行分布式存储并行切片 I/O。
   * 利用 MPI 派生数据类型 `MPI_Type_vector` 结合非阻塞传输原语 `MPI_Isend` / `MPI_Irecv` 进行列跨步数据重组与直发。
   * 组内使用异步组播 `MPI_Ibcast`，实现通信与本地硬件初始化的时间重叠。
3. **3x8 DSP 核心网格 100% 均衡任务分配**：
   * 针对单设备内 24 个自主 DSP 核心，采用 $3 \times 8$ 网格拓扑与严格的 64 元素向量对齐划分算法。
   * 彻底消除了因非对齐导致的补零开销与 DMA 碎片，各 DSP 线程任务负载严格对称，算力利用率达到 100%。
4. **统一内核调度入口**：
   * 默认直接调度执行最高性能的流水线内核 `gemm_dsp_AM_SM_pipeline_opt`，同时支持通过参数回退调用消融内核以执行基线对比。

---

### 1.2 设备端最终版本内核：`device_code/kernel_pipeline.c`
> **对应主机端调用接口**：`gemm_dsp_AM_SM_pipeline_opt(...)`（定义于 `host_code/operator.c`）  
> **DSP 设备端执行函数**：`matrix_mul_pipeline_opt`（定义于 `device_code/kernel_pipeline.c`）

该内核代表了针对 MT-3000 DSP 微架构所能达到的最高极限性能，核心优化策略包括：

1. **三缓冲流水线（Triple Buffering DMA Pipeline）**：
   * **C 矩阵三级流水旋转**：在片上向量内存（AM）开辟 3 组 C 向量缓冲，分别用于：
     * Buffer 1：当前块高吞吐算子并行计算；
     * Buffer 2：下一块 C 矩阵历史数据的预加载（DMA P2P 读取）；
     * Buffer 3：上一块计算完成 C 矩阵结果的异步回写（DMA P2P 写透回 DDR）。
   * **A/B 矩阵双缓冲（Ping-Pong Double Buffering）**：标量内存（SM）与向量内存（AM）采用交替乒乓机制，计算当前块的同时利用 DMA 预取下一块，数据搬运延迟被计算彻底隐藏。
2. **$4 \times 4$ 向量寄存器分块（Register Tiling）**：
   * 设计宏算子 `COMPUTE_K_BLOCK_OPT_C`，将中间累加器持续锁死驻留在 DSP 的 **32 个物理向量寄存器**中（`cx0_0 ~ cx3_3` 与 `cy0_0 ~ cy3_3`）。
   * 避免频繁将中间累加结果读写片上 SRAM，将访存压力完全限制在寄存器级别。
3. **双发射 FMA 指令融合与无停顿流水线**：
   * 针对双精度复数运算公式：
     $$\text{Re}(C) += A_r \cdot B_r - A_i \cdot B_i$$
     $$\text{Im}(C) += A_r \cdot B_i + A_i \cdot B_r$$
   * 采用硬件矢量广播指令 `vec_svbcast` 与乘累加融合原语 `vec_mula`，将实部、虚部的 4 次实数乘加重排为双发射对称结构，消除了前向依赖引起的 RAW 流水线停顿。

---

## 2. 实验对比与消融分析代码（Baseline & Ablation Files）

为支撑学术论文中严谨的**消融实验（Ablation Study）**与**对比分析（Comparative Evaluation）**，工程中专门保留了以下对照基线代码：

| 源码文件 | 对比维度 / 实验目的 | 关键区别与论文作用 |
| :--- | :--- | :--- |
| **`host_code/gemm_host_AS.c`** | **分布式通信对比（1D vs 2D）** | **1D 全局广播基线**：所有进程平铺分布，采用全局 `MPI_Bcast` 广播整块 $B$ 矩阵与 $A$ 矩阵。用于论文中论证当超算集群节点扩展到 64 ~ 512 节点时，1D 广播通信发生雪崩式延迟增长，从而反衬出 2D 划分算法在强扩展（Strong Scaling）中的巨大优势。 |
| **`host_code/gemm_host_ASR.c`** | **CPU 基线与正确性校验** | **纯 CPU 串行/多核基准**：内置 `gemm_cpu_no_block` 纯 CPU 计算逻辑。一方面用于单节点上的绝对数值误差验证（Error $\le 10^{-14}$）；另一方面用于产出 CPU 串行/朴素实现的执行耗时基线。 |
| **`device_code/kernel_AS.c`** | **DSP 内核消融实验（逐级递进）** | 内置两个关键对比核函数：<br>1. `matrix_mul_as_opt`：**单缓冲（Single Buffer）基线**，计算与 DMA 搬运完全串行阻塞；<br>2. `matrix_mul_asd_opt`：**双缓冲（Double Buffer）基线**，具备双缓冲重叠但尚未引入 C 矩阵三缓冲旋转与 $4\times 4$ 深度寄存器分块。<br>用于论文中绘制“单缓冲 $\to$ 双缓冲 $\to$ 三缓冲流水线+寄存器分块”的算力逐步攀升消融柱状图。 |
| **`benchmark/gemm_openblas_bench.c`** | **工业级标准库基线对比（BLAS Baseline）** | **OpenBLAS 对比与全精度检验驱动**：在 FT-2000+ CPU 上运行 OpenBLAS（支持单核、多核 4x DGEMM 与 1x ZGEMM），并与最终版 DSP 流水线内核进行端到端吞吐率对比及全量矩阵元素级数值误差验证（误差见 `openblas_vs_asd.log`）。 |

---

## 3. 项目完整目录树与文件清单

```
E:\personal\论文\gemm_mpi/
├── README.md                     # 本项目全景技术文档与论文指导索引（本文件）
├── rollback.sh                   # 远程集群工作区版本回滚与备份脚本
├── .gitignore                    # Git 忽略配置
│
├── host_code/                    # 主机端源码目录（FT-2000+ CPU，负责 MPI 与硬件调用）
│   ├── gemm_host_AB_opt6.c       # ★【最终版本主控】2D 拓扑 + 非阻塞通信 + 3x8 均衡网格划分
│   ├── gemm_host_AS.c            # ☆【消融基线】1D 全广播通信基线程序
│   ├── gemm_host_ASR.c           # ☆【校验基线】含 CPU 校验与 1D 历史基准程序
│   ├── operator.c                # 主机端 DSP 算子驱动库（封装 hthread 运行时加载与调用）
│   ├── read_data.c               # 矩阵数据切片读取底层 I/O 辅助工具函数
│   └── Makefile                  # 主机端 MPICH 交叉编译 Makefile
│
├── device_code/                  # 设备端加速内核源码目录（MT-3000 自主 DSP 矢量核）
│   ├── kernel_pipeline.c         # ★【最终版本内核】包含 matrix_mul_pipeline_opt 三缓冲流水线内核
│   ├── kernel_AS.c               # ☆【消融基线内核】包含 matrix_mul_as_opt / matrix_mul_asd_opt 对照内核
│   └── Makefile                  # DSP 矢量编译器交叉编译 Makefile
│
├── benchmark/                    # 工业级基准对比与精度评测套件
│   ├── README.md                 # OpenBLAS 对比评测方案与性能指标解析文档
│   ├── gemm_openblas_bench.c     # OpenBLAS (4x dgemm / 1x zgemm) vs DSP 性能与精度对比测试源码
│   ├── openblas_vs_asd.log       # 512 ~ 16384 全规模对比实验原始日志（含单核、簇级、节点级及绝对误差）
│   └── Makefile                  # 测试驱动交叉编译 Makefile
│
├── include/                      # 项目公共头文件
│   ├── common.h                  # 全局通用宏定义、常量配置
│   └── operator.h                # DSP 主机调用接口与核心算子声明
│
└── bin/                          # 集群执行环境与运行时依赖
    ├── batch_run.sh              # 强扩展与弱扩展自动化批处理 Slurm 评测脚本
    ├── run.sh                    # 基础运行脚本
    ├── run2.sh                   # 备用运行脚本
    ├── lib/                      # MT-3000 异构运行时动态库（libmpi.so.40 等）
    └── libs/                     # 基础系统动态链接库（libc.so.6, libstdc++.so.6 等）
```

---

## 4. 论文撰写结构与代码模块映射指南

后续使用 Gemini 或其他大模型基于本代码库辅助撰写论文时，推荐的章节素材对应关系如下：

### 4.1 引言与背景（Introduction & Background）
* **硬件背景**：银河天河 MT-3000 异构超级计算机节点拓扑（FT-2000+ 主机处理器通过 PCIe 互连 4 块 MT-3000 矩阵加速 DSP，每块 DSP 拥有 24 个自主矢量核心）。
* **技术难点**：超算超多节点（2048 进程）下的分布式通信拥塞、异构计算存储层级深导致的带宽瓶颈、复数浮点乘累加的指令开销。

### 4.2 算法与系统设计（Methodology & Architecture）
* **分布式 2D 拓扑与非阻塞优化**：
  * 参照代码：[`host_code/gemm_host_AB_opt6.c`](file:///E:/personal/论文/gemm_mpi/host_code/gemm_host_AB_opt6.c)
  * 重点阐述：$P/2 \times 2$ 混合网格、`MPI_Type_vector` 列拆分重构、`MPI_Ibcast` 组内并行广播、`pread_matrix` 并行并行文件预读。
* **DSP 核内多级流水线与寄存器分块**：
  * 参照代码：[`device_code/kernel_pipeline.c`](file:///E:/personal/论文/gemm_mpi/device_code/kernel_pipeline.c)
  * 重点阐述：C 矩阵三级缓冲轮转（计算-预取-写回三维隐藏）、$4\times 4$ 向量寄存器分块（32 寄存器驻留累加）、`vec_mula` 指令重排消除 RAW 数据相关。
* **任务分配均衡性算法**：
  * 重点阐述：针对 24 核 DSP 的 64 列模数整除与非对齐边缘块消除算法，达到 100% 算力利用率。

### 4.3 实验评估设计（Performance Evaluation）
* **实验一：单核/单卡内核级消融实验（Kernel Ablation）**：
  * 对比对象：`matrix_mul_as_opt`（单缓冲） vs `matrix_mul_asd_opt`（双缓冲） vs `matrix_mul_pipeline_opt`（三缓冲流水线）。
* **实验二：分布式强扩展/弱扩展实验（Scaling Benchmarks）**：
  * 对比对象：`gemm_host_AS.c`（1D 模式） vs `gemm_host_AB_opt6.c`（2D 模式）。
  * 规模跨度：1 节点（4 进程）扩展至 512 节点（2048 进程）。
* **实验三：工业级基线对比（BLAS Baseline Comparison）**：
  * 对比对象：本工程 DSP 优化实现 vs CPU OpenBLAS / 平台厂商 MTBLAS。
  * 评测指标：GFLOPS 算力、能效比、不同规模矩阵下的端到端加速比。
