# 学术论文直接投稿套件（IEEE Conference & IEEE Transactions 双版本）

本目录包含面向计算机体系结构与高性能计算一流会议（**SC / IEEE IPDPS / IEEE CLUSTER / ICPP**）及一流期刊（**IEEE TPDS / IEEE TC**）标准打造、可直接提交（Direct-Submission Ready）的完整 LaTeX 论文工程、矢量图表生成脚本与已编译并通过 **IEEE PDF eXpress** 字体/版式审计的 PDF 成稿。

> **AI Agent 协作提示**：修改本目录下的 `.tex`、`generate_figures.py` 或 `references.bib` 前，请先阅读仓库根目录的 [`../AGENTS.md`](../AGENTS.md)、[`../DECISIONS.md`](../DECISIONS.md) 与 [`../VERIFICATION.md`](../VERIFICATION.md)，严格遵守双版本同步、会议版 **10.0 页正文红线**及零警告/零 Type-3 字体门禁。

---

## 1. 交付文件清单

| 文件路径 | 类型 | 用途与投稿目标说明 |
| :--- | :--- | :--- |
| `paper_conference.tex` | 主源文件 (Conf) | **IEEE 顶级会议直接投稿版**（`\documentclass[conference]{IEEEtran}`，默认开启双盲匿名 `\anonymoustrue`，正文严格控制在 **10.0 页以内**，第 11 页为均衡双栏参考文献） |
| `paper_conference.pdf` | 编译成稿 (Conf) | 11 页 US Letter（10 页技术正文 + 1 页均衡参考文献），**0 Warning / 0 Overfull Box / 0 Type-3 字体 / 100% 字体嵌入** |
| `bare_jrnl_new_sample4.tex` | 主源文件 (Journal) | **IEEE Transactions 顶级期刊直接投稿版**（`\documentclass[journal]{IEEEtran}`，含标准页眉 `\markboth`、首字下沉 `\IEEEPARstart` 与单盲/双盲一键切换开关） |
| `bare_jrnl_new_sample4.pdf` | 编译成稿 (Journal) | IEEE TPDS / IEEE TC 格式直接投稿 PDF，**0 Warning / 0 Overfull Box / 0 Type-3 字体 / 100% 字体嵌入** |
| `bare_jrnl_new_sample4_v2.txt` | 同步纯文本备份 | 与 `bare_jrnl_new_sample4.tex` 完全同步的纯文本备份（消除旧版草稿遗留问题） |
| `references.bib` | BibTeX 文献库 | 25 篇经 DBLP / IEEE Xplore / ACM DL 逐条核验的真实学术文献（0 BibTeX 警告） |
| `IEEEtran.cls` | 官方文档类 | IEEE 官方标准样式类文件（v1.8b），支持目录独立编译 |
| `generate_figures.py` | 绘图复现脚本 | 从 `../实验结果.xlsx` 与 `../gemm_mpi/benchmark/openblas_vs_asd.log` 真实测量数据自动生成全部 7 张出版级矢量 PDF 图（强制 TrueType Type-42 字体嵌入） |
| `figures/fig1..fig7.pdf` | 矢量插图套件 | 7 张全矢量 PDF 插图（架构内存层次图、2D 通信拓扑与重叠时序图、14 通道三缓冲与 $4\times 4$ 微内核图、OpenBLAS 吞吐与精度对比图、含消融轨迹的实测 Roofline 模型图、$8192^3$ 512 节点强扩展与通信重叠图、DMA 缓冲消融 + $16384^3$ 218.79 TFLOPS 极限强扩展 + 1–512 节点弱扩展综合评测图） |

---

## 2. 双盲评审（Double-Blind）与定稿署名（Camera-Ready）一键切换

两份主源文件（`paper_conference.tex` 与 `bare_jrnl_new_sample4.tex`）顶部第 8–9 行均内置了单行布尔开关：

```latex
\newif\ifanonymous
\anonymoustrue   % 默认：双盲匿名投稿模式（输出 Anonymous Author(s)，无任何身份泄露）
% \anonymousfalse % 切换为定稿 / 单盲投稿模式：取消注释此行并填入真实作者姓名、单位与邮箱
```

* **投递 SC / IEEE IPDPS / IEEE CLUSTER / ICPP 双盲审稿**：保持默认 `\anonymoustrue`，直接上传 `paper_conference.pdf`。
* **投递单盲期刊或录用后 Camera-Ready 定稿**：将第 9 行改为 `\anonymousfalse`，在 `\else ... \fi` 块中填入真实作者与基金致谢信息后重新编译即可。

---

## 3. 一键复现图表与编译指南

### 3.1 重新生成全部 7 张出版级矢量 PDF 图表
```powershell
python generate_figures.py
```

### 3.2 编译 IEEE 顶级会议版（`paper_conference.pdf`）
```powershell
pdflatex -interaction=nonstopmode paper_conference.tex
bibtex paper_conference
pdflatex -interaction=nonstopmode paper_conference.tex
pdflatex -interaction=nonstopmode paper_conference.tex
```

### 3.3 编译 IEEE Transactions 顶级期刊版（`bare_jrnl_new_sample4.pdf`）
```powershell
pdflatex -interaction=nonstopmode bare_jrnl_new_sample4.tex
bibtex bare_jrnl_new_sample4
pdflatex -interaction=nonstopmode bare_jrnl_new_sample4.tex
pdflatex -interaction=nonstopmode bare_jrnl_new_sample4.tex
```

### 3.4 IEEE PDF eXpress 字体嵌入合规性校验
```powershell
pdffonts paper_conference.pdf
pdffonts bare_jrnl_new_sample4.pdf
```
全部字体对象均为 `Type 1` 或 `CID TrueType` 且 `emb=yes, sub=yes`，无任何 `Type 3` 位图字体。

---

## 4. 论文核心章节与源码/实测数据严格映射表

| 论文章节 / 图表 / 算法 | 对应工程源码或原始实验数据文件 | 核心核验指标 |
| :--- | :--- | :--- |
| **Sec. III & Alg. 1 & Fig. 2**（$P/2 \times 2$ 2D 混合拓扑与两阶段重叠通信） | `../gemm_mpi/host_code/gemm_host_AB_opt6.c`、`operator.c`、`read_data.c` | `MPI_Comm_split`、`MPI_Type_vector` 零拷贝列面板分发、异步 `Isend`/`Irecv` + `Ibcast` 与 `pread` 重叠、`chunk_64 = 64` 向量对齐 |
| **Sec. IV & Alg. 2 & Fig. 3**（14 通道三缓冲 DMA 与 $4\times 4$ 寄存器驻留微内核） | `../gemm_mpi/device_code/kernel_pipeline.c` | `ch1..8` A/B 双缓冲（56 KB SM + 64 KB AM）+ `chc1..6` C 循环三缓冲（168 KB AM）；32 个物理向量累加器驻留，消除 **86.1%（7.20x）** AM 访存指令 |
| **Sec. IV-C/D & Fig. 5**（分块容量证明、Roofline 与强扩展 DDR 流式下界定理） | `../gemm_mpi/device_code/kernel_pipeline.c`、`../实验结果.xlsx` | $(56,32,64)$ 分块算术强度 $\mathcal{I}_{\mathrm{DMA}}=14.83$ FLOPs/B，理论簇级 DDR4 Roofline 上限 **240.5 GFLOPS**（实测 **243.44 GFLOPS**）；精准预测 512 节点 $N=8192$（86.8 vs. 实测 **89.92 TFLOPS**）与 $N=16384$（212.3 vs. 实测 **218.79 TFLOPS**，误差 **2.9%**） |
| **Sec. V-B & Table III & Fig. 4**（节点内单核/单簇 vs. 多核 OpenBLAS 与 FP64 精度） | `../gemm_mpi/benchmark/gemm_openblas_bench.c`、`openblas_vs_asd.log` | 单核 **49.19 GFLOPS**（64.1% 峰值，较单核 OpenBLAS 快 **6.12x–7.28x**）；单簇 **243.44 GFLOPS**（较同配比 4 核 CPU 快 **9.33x**，较整节点 16 核快达 **6.57x**）；FP64 最大绝对误差 $\le 2.18 \times 10^{-8}$ |
| **Sec. V-C & Table IV & Fig. 5 & Fig. 7(a)**（片上 DMA 单/双/三缓冲微架构消融） | `../实验结果.xlsx`（`实验6_双缓冲与流水线优化`） | 固定同一 $4\times 4$ 32-VR 微内核：单核异步双缓冲由 33.79 跃升至 **48.05 GFLOPS（1.422x 加速，耗时缩短 29.69%）**；单簇 24 核达 **224.79 GFLOPS（1.144x）**；整节点 96 核（4 簇 $2\times 2$）在 $N=16384$ 下达 **956.17 GFLOPS（1.141x，达 962.0 GFLOPS 整节点 DDR4 Roofline 的 99.4%）** |
| **Sec. V-D & Table V & Fig. 6 & Fig. 7(b)**（1–512 节点 $8192^3$ 与 $16384^3$ 极限强扩展） | `../实验结果.xlsx`（`Sheet1`–`Sheet4`、`实验5_更大矩阵规模16384强扩展`） | $N=8192$ 下 512 节点通信耗时由 102.25 s 降至 **10.32 s（9.90x 加速，端到端 9.86x）**、吞吐 **89.92 TFLOPS（较 1D 基线 2.01x）**；$N=16384$（35.184 TFLOPs）下 2 节点实现 **112.80% 超线性效率（2.256x）**，256 节点达 **166.82 TFLOPS（66.26% 效率）**，512 节点（49,152 DSP 核）聚合算力突破至 **218.79 TFLOPS（0.1608 s，222.49x 加速）** 且 4.30 GB $B$ 面板组播全程无风暴（39.58–47.34 s）、成员 `pread` 仅 **0.1117–0.1235 s（37.9x 加速）** |
| **Sec. V-E & Table V & Fig. 7(c)**（1–512 节点多粒度等颗粒度弱扩展与抗抖动） | `../实验结果.xlsx`（`实验4_弱扩展实验`） | 单节点基准 $2048^3$（68.72 GFLOPs/node）扩展至 512 节点（总工作量 35.184 TFLOPs）：`pipeline_opt` 耗时稳定在 0.4825–**0.5233 s**，在 512 节点维持 **92.21% 弱扩展并行效率（67.24 TFLOPS）**，凭借 $C$ 分块三级循环流水线吸收跨节点内存干扰，较纯双缓冲 `d_opt`（89.64%，65.38 TFLOPS）高出 **+2.57 个百分点（吞吐 +2.77%）**；$1024^3$ 基准 512 节点保持 **80.07%** 效率，$512^3$ 极细粒度基准 128 节点保持 **87.51%** 效率 |
