# 基于 MT-3000 异构超算的高性能分布式复数 GEMM 研究与论文工程

本项目面向国产银河天河 MT-3000 异构超算平台（FT-2000+ 多核 ARM64 处理器 + MT-3000 自主 DSP 矩阵加速核），包含高性能分布式复数矩阵乘法（CGEMM / ZGEMM）的完整优化算法实现、基线对比测试套件以及学术论文编写工程。

---

## 1. 仓库目录导航

```
.
├── gemm_mpi/                  # 核心算法实现与分布式评测源码工程
│   ├── README.md              # 算法架构、最终版实现与消融实验说明
│   ├── host_code/             # 主机端主控源码 (gemm_host_AB_opt6.c 等)
│   ├── device_code/           # DSP 加速内核源码 (kernel_pipeline.c 等)
│   ├── benchmark/             # OpenBLAS 工业级基线对比与精度测试套件
│   ├── include/               # 项目公共头文件
│   └── bin/                   # 集群运行调度与批处理评测脚本
│
├── 论文初稿/                  # 学术论文主工程与初稿文件
│   ├── bare_jrnl_new_sample4.tex     # 论文 LaTeX 主源文件
│   └── bare_jrnl_new_sample4_v2.txt  # 论文初稿文本与内容草稿
│
├── 会议论文模板/              # IEEE 官方标准会议模板套件
│   └── IEEE-conference-template-062824/
│       ├── IEEEtran.cls       # IEEE 样式类文件
│       ├── IEEE-conference-template-062824.tex  # 模板范例源文件
│       └── IEEE-conference-template-062824.pdf  # 格式参考 PDF
│
├── 实验结果.xlsx              # 实验全量原始数据表（强扩展/弱扩展/单核/簇级加速比）
├── .gitignore                 # Git 忽略配置（已过滤编译临时文件与大二进制文件）
└── README.md                  # 本全局导航文档
```

---

## 2. 核心技术创新点（论文主推）

1. **2D 混合进程拓扑与异步重叠通信 (`gemm_host_AB_opt6.c`)**：
   * 采用 $P/2 \times 2$ 二维混合网格，矩阵 $B$ 列分发 + 组内矩阵 $A$ 行划分，彻底解决超算千节点全广播网络拥塞；
   * 非阻塞数据通信（`MPI_Isend` / `MPI_Irecv` / `MPI_Ibcast`）与非阻塞切片读取（`pread_matrix`）全链路重叠。
2. **DSP 核内三缓冲流水线与向量寄存器分块 (`kernel_pipeline.c`)**：
   * C 矩阵三级流水旋转（计算、DMA 预取、DMA 写透异步重叠）+ A/B 矩阵双缓冲；
   * $4 \times 4$ 向量寄存器分块，32 个向量累加器全周期驻留寄存器堆；
   * 双发射 FMA 乘累加指令融合 (`vec_mula`)，消除 RAW 数据冒险。
3. **严格的工业级与跨平台对比 (`benchmark/`)**：
   * 与飞腾 FT-2000+ CPU 上的多核 OpenBLAS（4x DGEMM 与 1x ZGEMM）在 512~16384 全规模下进行吞吐率对比与绝对数值精度验证（误差 $\le 10^{-8}$）。
