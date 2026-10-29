# ROADMAP.md — Project Milestones & Future Research Roadmap

This document tracks completed engineering/publication milestones and outlines the structured roadmap for future algorithmic extensions, hardware co-design generalizations, and artifact evaluation on MT-3000 heterogeneous supercomputers.

---

## 1. Completed Milestones (`M1`–`M5`)

- [x] **M1 — Intra-Core 14-Channel Tri-Buffered DMA & $4 \times 4$ Register-Resident Micro-Kernel (`gemm_mpi/device_code/kernel_pipeline.c`)**
  - Designed split-complex $(56, 32, 64)$ scratchpad tiling (`56 KB SM`, `232 KB AM`), 32-VR accumulator residency (`86.1%` / `7.20x` AM vector traffic reduction), and 14-channel $A/B$ ping-pong + $C$ cyclic tri-buffering (`49.19 GFLOPS` single core, `243.44 GFLOPS` 24-core cluster, `956.17 GFLOPS` 96-core full node).
- [x] **M2 — Industrial OpenBLAS Benchmark & FP64 Numerical Verification Suite (`gemm_mpi/benchmark/`)**
  - Verified element-wise infinity-norm accuracy ($\le 2.18 \times 10^{-8}$) and `6.12x–9.33x` throughput speedups against single- and multi-threaded ARMv8 OpenBLAS (`4x DGEMM` and `1x ZGEMM`) across $N \in [512, 16384]$.
- [x] **M3 — $P/2 \times 2$ Hybrid 2D MPI Topology & Async `pread` Overlap Engine (`gemm_mpi/host_code/gemm_host_AB_opt6.c`)**
  - Eliminated 512-node 1D `MPI_Bcast` storms (`102.25 s` $\to$ `10.32 s`, `9.90x`) via zero-copy `MPI_Type_vector` strided panel dispatch, subgroup `MPI_Ibcast`, and offset-sliced parallel `pread` (`93.5x` member I/O speedup, `82.5%` leader I/O hiding).
- [x] **M4 — Comprehensive 512-Node (49,152 DSP Cores) Multi-Regime Supercomputer Evaluation (`实验结果.xlsx`)**
  - Completed all 6 experimental suites: single-core/cluster BLAS comparison, DMA single/double/tri-buffering ablation, $8192^3$ 1D vs. 2D strong scaling (`89.92 TFLOPS`), $16384^3$ extreme-scale strong scaling (**`218.79 TFLOPS`**, `222.49x` speedup, `112.80%` 2-node superlinear efficiency), and 1–512 node multi-granularity weak scaling (**`92.21%` efficiency** at 512 nodes with `+2.77%` tri-buffer jitter resilience).
- [x] **M5 — Direct-Submission IEEE Conference & IEEE Transactions Manuscript Suite (`论文初稿/`)**
  - Authored and verified `paper_conference.tex`/`.pdf` (strict $\le 10.0$-page technical body + balanced Page 11 References) and `bare_jrnl_new_sample4.tex`/`.pdf` (`0` warnings, `0` overfull boxes, `0` Type-3 fonts, 7 embedded-font vector figures `fig1..fig7.pdf`, 25 verified BibTeX references).

---

## 2. Next-Phase Research & Engineering Roadmap (`R1`–`R5`)

### R1 — Adaptive Column-Grouping ($G \in \{2, 4, 8\}$) & 2.5D SUMMA Generalization
* **Motivation**: By our alignment theorem (`ADR-004`), $N=8192$ restricts zero-padding column grouping to $G=2$ ($8192 / (8 \times 2) = 512 = 8 \times 64$). However, at $N=16384$ and $N=32768$, $G=4$ and $G=8$ also satisfy $N / (8G) \equiv 0 \pmod{64}$ while preserving $\ge 4$ `AM` tiles/core! Raising $G$ from $2$ to $4$ at $N=16384$ on 512 nodes doubles $M_{\mathrm{local}}$ from $16$ to $32$, lifting $\mathcal{I}_{\mathrm{eff}}$ from $6.40$ to $10.04\text{ FLOPs/B}$ (projected $\sim 340\text{ TFLOPS}$ across 512 nodes).
* **Deliverables**:
  - Parameterize `group_num` ($G$) dynamically in `gemm_host_AB_opt6.c` as $G^\star(N, P) = \max \{ G \mid P \bmod G = 0,\; N \bmod (512 G) = 0,\; N / (512 G) \ge 2 \}$.
  - Extend 2D stationary-$A$ / broadcast-$B$ into a 2.5D replication-aware SUMMA pipeline overlapping outer-$K$ panel broadcasts with DSP kernel execution.

### R2 — On-Chip GSM (`6 MB`) / HBSM (`48 MB`) Cooperative Prefetching & Arbitrary Matrix Shapes
* **Motivation**: Current cluster scaling saturates the `16.22 GB/s` cluster DDR4 DMA channel (`243.44 GFLOPS` vs. `240.5 GFLOPS` Roofline). Exploiting cluster-shared `GSM`/`HBSM` as an intermediate software-managed staging level across the $3 \times 8$ core grid can amplify external DDR reuse before per-core `SM`/`AM` DMA fan-out.
* **Deliverables**:
  - Inter-core cooperative `DDR -> GSM -> SM/AM` broadcast/multicast DMA staging in `device_code/`.
  - Masked vector tail micro-kernels (`vec_mula` predication / fringe-tile cleanup) relaxing the `N % 512 == 0` alignment requirement to arbitrary $(M, K, N)$ dimensions with zero host padding.

### R3 — Mixed-Precision Complex Kernels & Distributed Factorizations (`ZTRSM` / `ZHERK` / `ZGETRF`)
* **Motivation**: Quantum chemistry, electromagnetics (MoM/FDTD), and radar STAP workloads combine `ZGEMM` with triangular solves and LU/Cholesky/QR factorizations, where `ZGEMM` accounts for $>85\%$ of FLOPs.
* **Deliverables**:
  - Port the 14-channel tri-buffered register-resident template to single-precision `CGEMM` ($32 \times$ FP32 per 1024-bit VR) and mixed-precision iterative refinement.
  - Expose a standard CBLAS/ScaLAPACK-compatible `mt3000_zgemm` / `mt3000_pzgemm` wrapper API for drop-in scientific application integration.

### R4 — Automated CI Lint & Manuscript Guardrail Workflow (`.github/workflows/`)
* **Deliverables**:
  - Add a lightweight GitHub Actions workflow running `python generate_figures.py`, TeX Live `pdflatex`/`bibtex` builds for both `\anonymoustrue` and `\anonymousfalse`, `pdffonts` zero-Type-3 checks, and Conference 10.0-page body enforcement on every Pull Request.

### R5 — Peer-Review Submission, Reproducibility Artifact Evaluation (AE), & Camera-Ready Release
* **Deliverables**:
  - Submit `paper_conference.pdf` (double-blind default) to target venue (SC / IEEE IPDPS / IEEE CLUSTER / ICPP) or `bare_jrnl_new_sample4.pdf` to IEEE TPDS / IEEE TC.
  - Package self-contained Slurm batch scripts (`gemm_mpi/bin/batch_run.sh`), log parsers, and `generate_figures.py` for ACM/IEEE **Artifacts Available & Artifacts Functional / Reproduced** badges.
