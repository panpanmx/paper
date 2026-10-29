# DECISIONS.md — Architecture & Manuscript Decision Records (ADR)

This document records the core hardware-software co-design, distributed algorithm, experimental methodology, and manuscript engineering decisions in this project so that contributors and AI agents preserve key invariants and avoid design regressions.

---

## ADR-001: Split-Complex Planar Layout over Interleaved Complex Storage
* **Status**: Accepted & Implemented (`gemm_mpi/host_code/`, `gemm_mpi/device_code/kernel_pipeline.c`)
* **Context**: Standard BLAS `ZGEMM` stores complex matrices as interleaved `(re, im)` pairs (`double complex`). On MT-3000's 1024-bit ($16 \times$ FP64) SIMD units, loading interleaved elements requires expensive vector de-interleaving/permutation shuffles before executing independent real/imaginary FMA streams.
* **Decision**: Store and transmit matrices in **split-complex planar layout** $(A_r, A_i)$, $(B_r, B_i)$, and $(C_r, C_i)$, evaluating $\Re(C) \mathrel{+}= A_r B_r - A_i B_i$ and $\Im(C) \mathrel{+}= A_r B_i + A_i B_r$ directly via scalar-to-vector broadcast (`vec_svbcast`) of $A$ scalars from `SM` and contiguous 16-FP64 vector loads of $B$ rows from `AM`.
* **Consequences**: Zero SIMD shuffle overhead, 100% 128-byte DMA/vector alignment, and symmetric dual-issue `vec_mula` FMA utilization achieving `49.19 GFLOPS` on a single DSP core (`64.1%` of theoretical dual-FMA peak) with $\le 2.18 \times 10^{-8}$ infinity-norm deviation vs. FP64 OpenBLAS.

---

## ADR-002: Asymmetric Scratchpad Tiling $(M_b, K_b, N_b) = (56, 32, 64)$ & $4 \times 4$ (32-VR) Register Residency
* **Status**: Accepted & Implemented (`gemm_mpi/device_code/kernel_pipeline.c`)
* **Context**: Each MT-3000 DSP core provides an asymmetric software-managed scratchpad hierarchy: **64 KB Scalar Memory (`SM`)**, **768 KB Vector Array Memory (`AM`)**, and **64 $\times$ 1024-bit physical vector registers (`VR`)**, with no hardware L1/L2 data cache coherence over `AM`/`SM`.
* **Decision**:
  1. Choose macro-tile $(M_b, K_b, N_b) = (56, 32, 64)$: double-buffered $(A_r, A_i)$ occupies $2 \times 2 \times 56 \times 32 \times 8 = 56\text{ KB} \le 64\text{ KB SM}$ (`87.5%` utilization); double-buffered $(B_r, B_i)$ + tri-buffered $(C_r, C_i)$ occupies $(2 \times 2 \times 32 \times 64 + 3 \times 2 \times 56 \times 64) \times 8 = 64\text{ KB} + 168\text{ KB} = 232\text{ KB} \le 768\text{ KB AM}$.
  2. Partition each $(56, 64)$ output tile into $4 \times 4$ micro-tiles (4 scalar rows $\times$ 4 vectors of 16 FP64 columns), locking **32 physical vector accumulators** (`cx0_0..cx3_3`, `cy0_0..cy3_3`) in the 64-VR file across the entire $K_b=32$ loop while leaving headroom for $B$ operand prefetch vectors and scalar broadcasts.
* **Consequences**: Raises DMA arithmetic intensity to $\mathcal{I}_{\mathrm{DMA}} = 14.83\text{ FLOPs/B}$ (saturating the 24-core cluster's `16.22 GB/s` DDR4 DMA Roofline ceiling of `240.5 GFLOPS` at **`243.44 GFLOPS`**) and eliminates **$86.1\%$ ($7.20\times$)** of `AM` vector load/store traffic inside the $K$-loop.

---

## ADR-003: 14-Channel Hybrid DMA Pipeline ($A/B$ Ping-Pong + $C$ Cyclic Tri-Buffering)
* **Status**: Accepted & Implemented (`gemm_mpi/device_code/kernel_pipeline.c`, `Sec. IV-B`, `Sec. V-C`, `Sec. V-E`)
* **Context**: Unlike read-only $A$ and $B$ tiles, updating $C \mathrel{+}= A B$ across outer tiles requires **both** a DMA read of historical $(C_r, C_i)$ from DDR into `AM` and a DMA write-back of accumulated $(C_r, C_i)$ to DDR. Pure 2-way ping-pong buffering decouples $A/B$ prefetching on a quiet single node, but exposes read-after-write DMA synchronization stalls when shared DDR4 memory controllers experience multi-cluster/multi-node contention jitter.
* **Decision**: Dedicate **14 independent point-to-point hardware DMA channels per core**: 8 channels (`ch1..8`) for $A/B$ ping-pong double buffering plus 6 channels (`chc1..6`) rotating three $(C_r, C_i)$ buffers across `(Prefetch t+1, Compute t, Write-back t-1)`.
* **Consequences**: Controlled ablation (`Table IV`) confirms that async buffering boosts single-core throughput by **$1.422\times$** (`33.79` $\to$ `48.05 GFLOPS`, `-29.69%` wall time) and 96-core full-node throughput to **`956.17 GFLOPS`** (`99.4%` of the 4-cluster DDR4 Roofline). Under 512-node weak scaling (`Table V`, $2048^3$/node), cyclic $C$ tri-buffering (`pipeline_opt`) absorbs cross-node memory jitter and achieves **`92.21%` efficiency (`67.24 TFLOPS`)**, outperforming pure double buffering (`d_opt`, `89.64%`, `65.38 TFLOPS`) by **`+2.57` percentage points (`+2.77%` throughput)**.

---

## ADR-004: $P/2 \times 2$ (`G = 2`) Hybrid 2D Process Grid & Vector-Alignment Theorem
* **Status**: Accepted & Implemented (`gemm_mpi/host_code/gemm_host_AB_opt6.c`, `Sec. III-A`, `Sec. IV-D`)
* **Context**: Under 1D row-block partitioning across $P = 2048$ MPI ranks (512 nodes $\times$ 4 clusters), global `MPI_Bcast` of the full $B$ matrix triggers network contention storms (`102.25 s` at $N=8192$), while shrinking per-rank rows to $M_{\mathrm{local}} = N / P = 4 \ll M_b = 56$ collapses effective arithmetic intensity to the **DDR-Streaming Floor** ($\mathcal{I}_{\mathrm{eff}}(4) = 2.67\text{ FLOPs/B}$, capping 512-node 1D throughput at `44.78 TFLOPS`). However, each 24-core DSP cluster partitions its local columns $N / G$ across an 8-column core grid (`3 x 8` cores) where every core requires multiples of `chunk_64 = 64` columns ($4 \times 16$-FP64 vectors).
* **Decision**: Configure $G = 2$ ($P/2 \times 2$ process grid via `MPI_Comm_split`), which is the unique $G > 1$ divisor at $N = 8192$ satisfying both $N / (8G) = 512 \equiv 0 \pmod{64}$ (zero tail padding across all 24 DSP cores) and $\ge 2$-tile `AM` pipeline priming ($8 \text{ tiles/core}$), while doubling $M_{\mathrm{local}}$ at every cluster count ($4 \to 8$ at $N=8192$; $8 \to 16$ at $N=16384$).
* **Consequences**: At 512 nodes ($N=8192$), eliminates the MPI broadcast storm (**`102.25 s` $\to$ `10.32 s`, $9.90\times$ speedup**) and lifts kernel throughput by **$2.01\times$** (`44.78` $\to$ **`89.92 TFLOPS`**). At $N=16384$ ($M_{\mathrm{local}}=16, \mathcal{I}_{\mathrm{eff}}=6.40\text{ FLOPs/B}$), scales cleanly to **`218.79 TFLOPS`** (`222.49x` speedup over 1 node), matching our analytical Roofline prediction (`212.3 TFLOPS`) within **`2.9%`**.

---

## ADR-005: Zero-Copy Strided `MPI_Type_vector` + Two-Phase Non-Blocking Comm-I/O Overlap
* **Status**: Accepted & Implemented (`gemm_mpi/host_code/gemm_host_AB_opt6.c`, `read_data.c`)
* **Context**: Extracting column panels $B_{:, g}$ from row-major files on Lustre/parallel storage and distributing them across subgroup communicators can serialize file I/O, host memory repacking, and MPI collectives.
* **Decision**: Define strided MPI derived datatypes (`MPI_Type_vector`) to dispatch non-contiguous column panels directly via `MPI_Isend`/`MPI_Irecv` from Rank 0 to Subgroup-1 leader without temporary host repacking buffers, and overlap inter-leader point-to-point transfer + intra-group `MPI_Ibcast` with lock-free offset-sliced `pread` of local $A_r, A_i$ row slices across all member ranks.
* **Consequences**: Accelerates member-rank file reads by up to **$93.5\times$** (`0.0406 s` at 128 nodes, $N=8192$; `0.1117–0.1235 s` / $37.9\times$ at 256–512 nodes, $N=16384$) and hides up to **$82.5\%$ (`2.23 s`)** of Rank 0's I/O latency behind non-blocking communication.

---

## ADR-006: Authoritative Single-Chip Multi-Zone MT-3000 Architectural Modeling
* **Status**: Accepted & Enforced (`论文初稿/paper_conference.tex`, `bare_jrnl_new_sample4.tex`, `figures/fig1_arch_overview.pdf`)
* **Context**: Early informal notes referred to an external FT-2000+ host connected over PCIe to discrete MT-3000 cards. Published architectural literature (e.g., Lu et al., *CCF THPC* 2022; Tianhe-2A/3 MT-3000 specifications) defines **MT-3000** as a single-chip multi-zone heterogeneous processor integrating a 16-core ARMv8 GP Zone and a 96-core ($4 \times 24$) DSP ACC Zone sharing unified node memory domains via high-bandwidth on-chip interconnects and DMA engines.
* **Decision**: Enforce the official single-chip multi-zone MT-3000 terminology (`GP Zone` + `4 ACC Clusters x 24 VLIW/SIMD DSP cores`, `hthread` programming interface) uniformly across all manuscripts, figures, and verification checks.
* **Consequences**: 100% factual alignment with peer-reviewed supercomputing literature (`lu2022mt3000` and related citations in `references.bib`).

---

## ADR-007: In-Memory Data Initialization for 1–4 Node $N=16384$ Strong-Scaling Points
* **Status**: Accepted & Documented (`实验结果.xlsx`, `Table V` footnote `$^{\S}$`, `Sec. V-D`)
* **Context**: At $N = 16384$, staging the full unpartitioned split-complex $B$ matrix (`4.30 GB`) on Rank 0 alongside four local clusters' DSP buffers (`~20 GiB` aggregate split-complex $(A, B, C)$ footprint on a single node) exceeds single-node host RAM capacity during file-staged I/O, whereas at $\ge 8$ nodes ($M_{\mathrm{local}} \le 1024$) per-node `A/C` footprint drops to $\le 3.0\text{ GiB}$ and full parallel `pread` + subgroup broadcast executes normally.
* **Decision**: Measure 1–4 node $N=16384$ kernel baselines via direct in-memory initialization (`'---'` reported for I/O and MPI columns in `Table V`), and run the full end-to-end parallel `pread` + 2D subgroup MPI pipeline across 8–512 nodes, explicitly documenting this protocol in `Table V` and `Section V-D`.
* **Consequences**: Complete scientific transparency and apples-to-apples kernel strong-scaling baselines from 1 to 512 nodes (`0.983` $\to$ `218.79 TFLOPS`), while also explaining the **2-node superlinear speedup (`2.256x`, `112.80%` efficiency)** caused by halving per-cluster `A/C` working sets across shared DDR4 controllers.

---

## ADR-008: Synchronized Dual-Edition Manuscript Engineering & Strict 10-Page Conference Cap
* **Status**: Accepted & Enforced (`论文初稿/paper_conference.tex`, `bare_jrnl_new_sample4.tex`)
* **Context**: Top HPC conferences (SC, IPDPS, CLUSTER, ICPP) enforce a strict **10-page technical body limit** (excluding references) under double-blind review, while IEEE Transactions journals (TPDS, TC) use continuous `[journal]` layout without forced page breaks before references.
* **Decision**:
  1. Maintain both `paper_conference.tex` (`[conference]`) and `bare_jrnl_new_sample4.tex` (`[journal]` + `_v2.txt` mirror) against a single shared `references.bib` (25 verified entries) and `generate_figures.py` (`fig1..fig7.pdf`).
  2. Provide a single-line `\ifanonymous` toggle on line 9 (`\anonymoustrue` default) in both manuscripts.
  3. Order Section V floats sequentially (`Tables I–V`, `Figs. 1–7`), place `\clearpage\IEEEtriggeratref{14}` before References in `paper_conference.tex` (so `Sections I–VII` finish strictly on **Page 10.0** and Page 11 renders symmetric 13-vs-12 References in both anonymous and camera-ready modes), and use continuous `\IEEEtriggeratref{21}` without `\clearpage` in `bare_jrnl_new_sample4.tex` for symmetric Page 11 journal column alignment.
* **Consequences**: Both manuscripts remain 100% direct-submission ready at all times with zero manual layout rework when switching between conference double-blind review, conference camera-ready, or journal submission.
