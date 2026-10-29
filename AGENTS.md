# AGENTS.md — AI Coding & Research Agent Operating Manual

This file is the authoritative entry point and guardrail harness for autonomous coding and research agents working in this repository. Read this document **first** before modifying any source code, figure script, experimental table, or LaTeX manuscript.

---

## 1. Repository Mission & Dual-Track Architecture

This repository houses a full-stack HPC systems research project targeting the **Tianhe MT-3000** multi-zone heterogeneous processor (16-core ARMv8 General-Purpose Zone + 4 Acceleration Clusters $\times$ 24 VLIW/SIMD DSP cores = 96 DSP cores/node, scaled up to **512 nodes / 2,048 MPI ranks / 49,152 DSP cores**):

1. **System & Kernel Implementation (`gemm_mpi/`)**: Distributed double-precision complex matrix multiplication (`ZGEMM`) combining an intra-core **14-channel tri-buffered DMA + $4 \times 4$ 32-vector-register-resident dual-FMA micro-kernel** with an inter-node **$P/2 \times 2$ hybrid 2D MPI topology + asynchronous `pread` overlap engine**.
2. **Direct-Submission Academic Manuscript Suite (`论文初稿/`)**: Synchronized **IEEE Conference** (`[conference]`, SC / IPDPS / CLUSTER / ICPP) and **IEEE Transactions** (`[journal]`, IEEE TPDS / TC) LaTeX manuscripts, reproducible vector figure generator, and verified BibTeX library.

---

## 2. Fast Context Map (Canonical Sources of Truth)

| Domain | Canonical File(s) | Role & Agent Rule |
| :--- | :--- | :--- |
| **Raw Cluster Data** | `实验结果.xlsx` | **Ground truth** for all 1–512 node experiments (1D vs. 2D $8192^3$ strong scaling, communication/I/O breakdown, $16384^3$ extreme strong scaling to `218.79 TFLOPS`, 1–512 node weak scaling, and DMA single/double/tri-buffering ablation). **Read-only.** |
| **Intra-Node & BLAS Data** | `gemm_mpi/benchmark/openblas_vs_asd.log` | **Ground truth** for single-core (`49.19 GFLOPS`), 24-core cluster (`243.44 GFLOPS`), multi-threaded ARMv8 OpenBLAS (`4x DGEMM` & `1x ZGEMM`), and FP64 accuracy ($\le 2.18 \times 10^{-8}$) across $N \in [512, 16384]$. **Read-only.** |
| **Production Host MPI** | `gemm_mpi/host_code/gemm_host_AB_opt6.c`, `operator.c`, `read_data.c` | Canonical 2D $P/2 \times 2$ MPI coordinator (`MPI_Comm_split`, `MPI_Type_vector`, `MPI_Isend`/`Irecv`, `MPI_Ibcast`, offset-sliced `pread`, $3 \times 8$ DSP core partitioning). |
| **Production DSP Kernel** | `gemm_mpi/device_code/kernel_pipeline.c` | Canonical 14-channel tri-buffered DMA + $4 \times 4$ 32-VR `vec_mula` micro-kernel (`matrix_mul_pipeline_opt`). |
| **Ablation Baselines** | `gemm_mpi/host_code/gemm_host_AS.c`, `gemm_host_ASR.c`, `device_code/kernel_AS.c`, `benchmark/gemm_openblas_bench.c` | 1D global-broadcast baseline, CPU verification baseline, single/double-buffer DSP ablation kernels, and OpenBLAS benchmark driver. |
| **Conference Manuscript** | `论文初稿/paper_conference.tex` $\to$ `paper_conference.pdf` | IEEE `[conference]` edition. Default `\anonymoustrue` (double-blind). **Hard limit: Sections I–VII $\le 10.0$ pages** in both anonymous and camera-ready modes. |
| **DAC Manuscript** | `论文初稿/paper_dac.tex` $\to$ `paper_dac.pdf` | ACM/IEEE DAC edition. Default `\anonymoustrue` (double-blind). **Hard limit: Sections I–VII $\le 6.0$ pages** + 1 page exclusively references (7 pages total). |
| **Journal Manuscript** | `论文初稿/bare_jrnl_new_sample4.tex` $\to$ `bare_jrnl_new_sample4.pdf` & `bare_jrnl_new_sample4_v2.txt` | IEEE `[journal]` edition (continuous references, balanced final page columns). Always keep `_v2.txt` byte-identical to `.tex`. |
| **Figure Pipeline** | `论文初稿/generate_figures.py` $\to$ `论文初稿/figures/fig1..fig7.pdf` | Generates all 7 vector PDF figures directly from real experimental measurements with Type-42 TrueType font embedding (`pdf.fonttype = 42`). |
| **Design Decisions** | `DECISIONS.md` | Architectural & manuscript decision records (`ADR-001`–`ADR-008`). Consult before altering algorithms, tiling, or hardware terminology. |
| **Verification Gates** | `VERIFICATION.md` | Executable compilation, font-embedding, page-budget, and data-audit checks required before finishing any task. |
| **Project Roadmap** | `ROADMAP.md` | Completed milestones (`M1`–`M5`) and upcoming research/engineering tracks (`R1`–`R5`). |

---

## 3. Non-Negotiable Invariants ("Red Lines")

1. **Zero Hallucinated Metrics or Citations**:
   - Every latency, throughput (GFLOPS/TFLOPS), speedup, efficiency percentage, or numerical error reported anywhere in `.tex`, `.py`, or `.md` **must** trace directly to `实验结果.xlsx` or `gemm_mpi/benchmark/openblas_vs_asd.log`.
   - Every entry in `论文初稿/references.bib` must be a real, verifiable publication with exact authors, venue, year, volume/pages, and zero BibTeX warnings.
2. **Authoritative MT-3000 Hardware Model (See `DECISIONS.md` `ADR-006`)**:
   - Model each MT-3000 node accurately as an **integrated multi-zone heterogeneous SoC**: 16-core ARMv8 GP Zone (@ 2.0 GHz) + 4 ACC Clusters $\times$ 24 VLIW/SIMD DSP cores (@ 1.2 GHz, 96 DSP cores/node; 76.8 GFLOPS/core peak, 1,843.2 GFLOPS/cluster peak).
   - Per-core software-managed scratchpads: **64 KB Scalar Memory (`SM`)**, **768 KB Vector Array Memory (`AM`)**, **64 $\times$ 1024-bit ($16 \times$ FP64) vector registers**, **16 point-to-point DMA channels/core** (14 active in our pipeline), and **16.22 GB/s** measured cluster DDR4 DMA STREAM Triad bandwidth (`240.5 GFLOPS` cluster Roofline ceiling at $\mathcal{I}_{\mathrm{DMA}} = 14.83\text{ FLOPs/B}$).
   - Do **not** re-introduce legacy draft wording describing MT-3000 as an external PCIe card attached to an FT-2000+ host CPU.
3. **Hardware Alignment & Scratchpad Budget Invariants (`gemm_mpi/`)**:
   - Preserve `chunk_64 = 64` (4 vectors $\times$ 16 FP64) column alignment across $3 \times 8$ DSP core partitioning (`N_cluster = N / G` must satisfy `(N / G) % (8 * 64) == 0` for zero-padding execution).
   - Preserve per-core scratchpad capacity bounds: $(M_b, K_b, N_b) = (56, 32, 64)$ uses **56 KB SM** ($\le 64\text{ KB}$) and **232 KB AM** ($64\text{ KB } B\text{ ping-pong} + 168\text{ KB } C\text{ tri-buffer} \le 768\text{ KB}$).
4. **Dual-Manuscript Synchronization & Conference 10.0-Page Budget**:
   - Whenever updating prose, equations, tables (`Tables I–V`), or figures (`Figs. 1–7`), update **both** `论文初稿/paper_conference.tex` and `论文初稿/bare_jrnl_new_sample4.tex` (plus `bare_jrnl_new_sample4_v2.txt`).
   - `paper_conference.pdf` **must** complete `Section I` through `Section VII` within **Pages 1–10.0** (with Page 11 holding balanced two-column References) under **both** `\anonymoustrue` and `\anonymousfalse`, while keeping `\anonymoustrue` as the committed default on line 9.
5. **IEEE PDF eXpress & Zero-Warning Build Gate**:
   - Every manuscript build must finish with **0 LaTeX/BibTeX warnings**, **0 Overfull boxes**, **0 Undefined citations/references**, and **`pdffonts` reporting 0 `Type 3` fonts** (`100% emb=yes, sub=yes`).
6. **Git Branch Safety**:
   - Never force-push or push directly to `main` unless explicitly requested by the user. Commit and push work on dedicated topic branches (e.g., `docs/ieee-paper-submission`, `feat/*`) so changes can be reviewed via Pull Request.

---

## 4. Quick-Start Agent Commands

### 4.1 Regenerate All 7 Vector PDF Figures
```powershell
Set-Location "论文初稿"
python generate_figures.py
```

### 4.2 Full 4-Pass LaTeX Build (Conference, DAC & Journal)
```powershell
Set-Location "论文初稿"
# 1. IEEE Conference Edition (10+1 pages)
pdflatex -interaction=nonstopmode paper_conference.tex
bibtex paper_conference
pdflatex -interaction=nonstopmode paper_conference.tex
pdflatex -interaction=nonstopmode paper_conference.tex

# 2. ACM/IEEE DAC Edition (6+1 pages)
pdflatex -interaction=nonstopmode paper_dac.tex
bibtex paper_dac
pdflatex -interaction=nonstopmode paper_dac.tex
pdflatex -interaction=nonstopmode paper_dac.tex

# 3. IEEE Transactions Edition (+ sync _v2.txt)
Copy-Item bare_jrnl_new_sample4.tex bare_jrnl_new_sample4_v2.txt -Force
pdflatex -interaction=nonstopmode bare_jrnl_new_sample4.tex
bibtex bare_jrnl_new_sample4
pdflatex -interaction=nonstopmode bare_jrnl_new_sample4.tex
pdflatex -interaction=nonstopmode bare_jrnl_new_sample4.tex
```

### 4.3 Cross-Compile C / MPI / HThread Binaries (On MT-3000 Linux Compile Node)
```bash
cd gemm_mpi/device_code && make clean && make
cd ../host_code && make clean && make
cd ../benchmark && make clean && make
```

For full verification gates (including camera-ready `\anonymousfalse` page-budget testing, `pdffonts` Type-3 audit, and 300-DPI visual page inspection), follow **`VERIFICATION.md`**.
