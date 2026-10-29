# 基于 MT-3000 多域异构超算的高性能分布式复数 GEMM 研究与直接投稿论文工程

本项目面向国产天河 **MT-3000** 多域异构超算处理器（16 核 ARMv8 通用主控域 GP Zone + 4 簇 $\times$ 24 核自主 VLIW/SIMD DSP 矩阵加速域 ACC Zone），包含大规模高性能分布式双精度复数矩阵乘法（`ZGEMM`）的跨层软硬件协同优化实现、工业级 OpenBLAS 对比基准测试套件，以及面向计算机体系结构与高性能计算顶级会议（**SC / IEEE IPDPS / IEEE CLUSTER / ICPP**）和顶级期刊（**IEEE TPDS / IEEE TC**）标准的**可直接提交（Direct-Submission Ready）LaTeX 论文工程**。

---

## 1. 仓库目录导航

```
.
├── gemm_mpi/                  # 核心算法实现与分布式评测源码工程
│   ├── README.md              # 算法架构、最终版实现与消融实验说明
│   ├── host_code/             # 主机端主控源码 (gemm_host_AB_opt6.c, operator.c, read_data.c 等)
│   ├── device_code/           # DSP 加速内核源码 (kernel_pipeline.c, kernel_AS.c 等)
│   ├── benchmark/             # OpenBLAS 工业级基线对比与 FP64 精度测试套件 (openblas_vs_asd.log)
│   ├── include/               # 项目公共头文件
│   └── bin/                   # 集群运行调度与批处理评测脚本
│
├── 论文初稿/                  # 学术论文直接投稿主工程（双版本齐备，0 Warning / 0 Type-3 字体）
│   ├── README.md                     # 投稿模式切换、一键编译与源码-图表核验指南
│   ├── paper_conference.tex          # IEEE 顶级会议直接投稿版主源文件 ([conference]，默认双盲匿名)
│   ├── paper_conference.pdf          # 已编译会议版 PDF（10 页技术正文 + 第 11 页均衡参考文献）
│   ├── bare_jrnl_new_sample4.tex     # IEEE Transactions 顶级期刊直接投稿版主源文件 ([journal])
│   ├── bare_jrnl_new_sample4.pdf     # 已编译期刊版 PDF（IEEE TPDS / IEEE TC 格式，双栏均衡）
│   ├── bare_jrnl_new_sample4_v2.txt  # 与期刊版主源文件完全同步的纯文本备份
│   ├── references.bib                # 25 篇经权威数据库逐条核验的 BibTeX 文献库
│   ├── IEEEtran.cls                  # IEEE 官方标准样式类文件 (v1.8b)
│   ├── generate_figures.py           # 从真实实验数据一键生成全部 7 张矢量 PDF 插图的 Python 脚本
│   └── figures/                      # 7 张出版级全矢量 PDF 插图 (fig1..fig7.pdf，100% 嵌入 TrueType/Type-1)
│
├── 会议论文模板/              # IEEE 官方标准会议模板参考套件
│   └── IEEE-conference-template-062824/
│       ├── IEEEtran.cls                         # IEEE 样式类文件
│       ├── IEEE-conference-template-062824.tex  # 模板范例源文件
│       └── IEEE-conference-template-062824.pdf  # 格式参考 PDF
│
├── 实验结果.xlsx              # 1~512 节点（96~49,152 DSP 核）全套实测原始数据（8192/16384 强扩展、弱扩展、DMA 缓冲消融、I/O 与单核/簇级基准）
├── AGENTS.md                  # ★ AI Agent 运行手册：仓库全局上下文索引、不可违背红线约束与标准构建命令
├── DECISIONS.md               # ★ 架构与论文决策记录 (ADR-001 ~ ADR-008)：硬件建模、分块/对齐定理与双版本排版规范
├── VERIFICATION.md            # ★ 自动化核验准入套件：图表生成、双盲/定稿四模式零警告编译、PDF 字体与实测指标核查表
├── ROADMAP.md                 # ★ 项目里程碑 (M1~M5) 与后续演进路线图 (R1~R5：自适应 2.5D 分组、GSM 协同预取、AE 复现包)
├── .gitignore                 # Git 忽略配置（已过滤 LaTeX 编译临时文件与大二进制文件）
└── README.md                  # 本全局导航文档
```

---

## 2. 核心架构与算法创新点（论文主线）

1. **面向非对称显式片上存储的 14 通道三缓冲 DMA 与 $4 \times 4$ 寄存器驻留双发射 FMA 微内核 (`kernel_pipeline.c`)**：
   * 针对 MT-3000 单核 **64 KB 标量存储（SM）+ 768 KB 向量阵列存储（AM）+ 64 个 1024-bit（$16 \times$ FP64）物理向量寄存器**的非对称架构，构造 $(M_b, K_b, N_b)=(56, 32, 64)$ 紧致分块（占用 56 KB SM 与 232 KB AM），通过 14 条物理点对点异步 DMA 通道实现 $A$/$B$ 双缓冲乒乓预取与 $C$ 矩阵“预取—计算—写回”三级循环流水线；
   * 在内层 $K_b$ 循环中将 32 个物理向量累加器全周期驻留寄存器堆并交错调度双发射 `vec_mula` FMA 指令，消除 **86.1%（$7.20\times$）** 的 AM 向量访存流量；
   * **实测单核性能达 49.19 GFLOPS**（达双发射向量理论峰值 76.8 GFLOPS 的 **64.1%**），较同节点 ARMv8 CPU 单核优化 OpenBLAS（`4x DGEMM` 与 `1x ZGEMM`）加速 **$6.12\times\text{--}7.28\times$**；在固定同一 $4\times 4$ 32-VR 微内核的控制变量消融中，异步 DMA 缓冲将单核吞吐率提升 **$1.422\times$**（由 33.79 至 **48.05 GFLOPS**，耗时缩短 **29.69%**）；**单簇 24 核性能达 243.44 GFLOPS**（近 **100%** 饱和 240.5 GFLOPS 簇级 DDR4 Roofline 上限），较同配比 4 核 CPU OpenBLAS 快 **$9.33\times$**、较整节点 16 核 CPU 快达 **$6.57\times$**；**整节点 96 核（4 簇 $2\times 2$）性能达 956.17 GFLOPS**（达 962.0 GFLOPS 整节点 DDR4 Roofline 的 **99.4%**），且全规模 FP64 逐元素无穷范数误差 $\le 2.18 \times 10^{-8}$。

2. **$P/2 \times 2$ 驻留 $A$ / 列面板广播 $B$ 的二维混合拓扑与两阶段通信-I/O 全异步重叠引擎 (`gemm_host_AB_opt6.c`)**：
   * 针对大规模强扩展下 1D 行划分引发的全网广播风暴与局部行数 $M_{\mathrm{local}} < M_b$ 导致的 **DDR 流式带宽下界（DDR-Streaming Floor）**，通过 `MPI_Comm_split` 构建 $P/2 \times 2$ 二维进程网格，利用 `MPI_Type_vector` 实现零拷贝列面板分发，严格配合 DSP 侧 `chunk_64 = 64` 向量对齐约束证明 $G=2$ 为 $N=8192$ 下的唯一最优列分组度；
   * 将跨组异步 `MPI_Isend`/`MPI_Irecv`、组内 `MPI_Ibcast` 与底层偏移切片并行 `pread` 全链路重叠：在 128 节点上将成员进程读取耗时压缩至 **0.0406 s（$93.5\times$ 加速）**，在 64 节点上隐藏主进程 **82.5%（2.23 s）** 的文件 I/O 延迟；
   * 在 **512 节点（2,048 MPI 进程、49,152 个 DSP 核心）**超大规模强扩展与弱扩展评测中：
     * **$N=8192$ 强扩展**：彻底消除 1D 全局广播风暴，将通信时间由 102.25 s 降至 **10.32 s（$9.90\times$ 加速，端到端总时间加速 $9.86\times$）**，并将内核聚合吞吐率从 1D 基线的 44.78 TFLOPS 翻倍提升至 **89.92 TFLOPS（0.0489 s，较 1 节点提速 $118.20\times$，较 1D 基线提速 $2.01\times$）**；
     * **$N=16384$（35.184 TFLOPs）极限规模强扩展**：在 2 节点实现 **112.80% 超线性并行效率（$2.256\times$）**，256 节点达 **166.82 TFLOPS（66.26% 效率）**，512 节点聚合算力突破至 **218.79 TFLOPS（0.1608 s，$222.49\times$ 加速，达 $8192^3$ 规模的 $2.43\times$）**，与理论推导的 DDR 流式下界定理（预测 212.3 TFLOPS，误差 **2.9%**）精确吻合，同时 4.30 GB $B$ 面板组播全程无风暴、成员并行 `pread` 仅耗时 **0.1117–0.1235 s（$37.9\times$ 加速）**；
     * **1–512 节点多粒度等颗粒度弱扩展**：在单节点 $2048^3$ 基准下扩展至 512 节点（总工作量 35.184 TFLOPs），三级流水线内核 `pipeline_opt` 维持 **92.21% 弱扩展并行效率（67.24 TFLOPS，0.5233 s）**，并凭借循环 $C$ 分块流水吸收集群内存干扰，在 512 节点规模上较纯双缓冲 `d_opt`（89.64%）进一步提升 **+2.57 个百分点（吞吐 +2.77%）**。

---

## 3. AI Agent 友好协作与自动化核验套件（Agent Harness）

为支持 AI 编程/科研智能体（Antigravity、Claude Code、Codex、Cursor 等）与人类协作者零幻觉、高可靠地维护代码库与双版本论文，仓库根目录内置了完整的结构化 Harness 文档体系：

| Harness 文档 | 核心职责与适用场景 |
| :--- | :--- |
| **[`AGENTS.md`](AGENTS.md)** | **智能体首要入口手册**：定义权威数据源映射表（Single Source of Truth）、6 条硬性护栏红线（零虚构数据、MT-3000 单芯片多域权威架构术语、64 元素向量对齐与片上容量上界、会议版 **10.0 页正文硬红线**、双版本同步、分支保护）以及标准构建脚本。 |
| **[`DECISIONS.md`](DECISIONS.md)** | **架构与手稿决策日志（`ADR-001`–`ADR-008`）**：记录实虚分离平面布局、$(56,32,64)$ 非对称分块与 32 寄存器驻留、14 通道混合缓冲抗抖动、$P/2 \times 2$ 对齐唯一性定理、零拷贝 `MPI_Type_vector` + `pread` 重叠、$N=16384$ 1–4 节点内存初始化协议及双版本排版决策的推导依据。 |
| **[`VERIFICATION.md`](VERIFICATION.md)** | **可执行质量门禁（Gate 1–Gate 4）**：提供一键复现 7 张矢量图 Type-42 字体检查、会议版/期刊版 $\times$ 双盲/定稿四模式零警告编译检查、300-DPI 视觉防遮挡审计，以及与 `实验结果.xlsx` / `openblas_vs_asd.log` 逐项锚定的核心指标核对表。 |
| **[`ROADMAP.md`](ROADMAP.md)** | **里程碑与后续演进路线图**：汇总已交付的 `M1`–`M5` 里程碑，并规划 `R1`（$N \ge 16384$ 自适应 $G \in \{2,4,8\}$ 列分组与 2.5D SUMMA）、`R2`（片上 `GSM`/`HBSM` 核间协同预取与任意维度免补零尾块）、`R3`（混合精度与分布式复数矩阵分解）、`R4`（CI 自动编译门禁）与 `R5`（顶会/顶刊 AE 复现徽章包）。 |

---

## 4. 快速复现与编译命令

```powershell
# 1. 从实测原始数据一键生成全部 7 张出版级矢量 PDF 图表 (fig1..fig7.pdf)
python "论文初稿\generate_figures.py"

# 2. 编译 IEEE 顶级会议版 (10.0 页技术正文 + 第 11 页均衡参考文献) 与 IEEE Transactions 期刊版
Push-Location "论文初稿"
pdflatex -interaction=nonstopmode paper_conference.tex; bibtex paper_conference; pdflatex -interaction=nonstopmode paper_conference.tex; pdflatex -interaction=nonstopmode paper_conference.tex
pdflatex -interaction=nonstopmode bare_jrnl_new_sample4.tex; bibtex bare_jrnl_new_sample4; pdflatex -interaction=nonstopmode bare_jrnl_new_sample4.tex; pdflatex -interaction=nonstopmode bare_jrnl_new_sample4.tex
Pop-Location
```
