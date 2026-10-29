#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducible Publication-Grade Vector Figure Generator for MT-3000 Complex GEMM Paper.
Generates ALL SIX IEEE PDF eXpress compliant vector PDFs (Type 42 TrueType fonts, zero Type 3 fonts):
  - Fig. 1 (fig1_arch_overview.pdf): MT-3000 Multi-Zone Architecture & 4-Tier DMA Memory Hierarchy
  - Fig. 2 (fig2_2d_comm_timeline.pdf): P/2 x 2 Hybrid 2D Process Grid & Overlapped Comm-I/O Timeline
  - Fig. 3 (fig3_tri_pipeline_microkernel.pdf): 14-Ch Tri-Buffered DMA Pipeline & 4x4 32-VR Micro-Kernel
  - Fig. 4 (fig4_intra_node_openblas.pdf): Intra-Node OpenBLAS Comparison & FP64 Numerical Accuracy
  - Fig. 5 (fig5_roofline_model.pdf): Empirical Cluster & Single-Core Roofline Model
  - Fig. 6 (fig6_strong_scaling_512nodes.pdf): 1-512 Nodes (96-49,152 DSP Cores) Strong-Scaling Evaluation

All empirical data points match verified supercomputer logs verbatim:
  - ../实验结果.xlsx
  - ../gemm_mpi/benchmark/openblas_vs_asd.log
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

# Strict IEEE publication styling (Type-42 TrueType embedding, STIX/Times serif math)
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.titlesize": 8.4,
    "axes.labelsize": 7.9,
    "xtick.labelsize": 7.1,
    "ytick.labelsize": 7.1,
    "legend.fontsize": 6.6,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.45,
    "lines.linewidth": 1.35,
    "lines.markersize": 4.2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.025,
})

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Unified Colorblind-Safe IEEE Palette
C_NAVY      = "#0B3C5D"   # Primary: Proposed DSP / 2D Hybrid
C_NAVY_LT   = "#EBF2F7"
C_NAVY_MID  = "#D0E1EE"
C_TEAL      = "#00796B"   # Secondary: Group 1 / ZGEMM / Overlap
C_TEAL_LT   = "#E6F4F1"
C_AMBER     = "#D97706"   # Accent: 4x DGEMM / B-Tile / Store
C_AMBER_LT  = "#FEF5E7"
C_RED       = "#C0392B"   # Alert / 1D Baseline
C_RED_LT    = "#FDEDEC"
C_GREEN     = "#1E8449"   # Roofline / Compute / Parallel I/O
C_GREEN_LT  = "#EAFAF1"
C_PURPLE    = "#6C3483"   # Imaginary / C-Tile
C_PURPLE_LT = "#F4ECF7"
C_SLATE     = "#4A5568"   # Structural outlines
C_SLATE_LT  = "#F4F6F8"


# ==============================================================================
# HELPER PRIMITIVES FOR VECTOR SCHEMATICS (1.0 UNIT = 1.0 PHYSICAL INCH)
# ==============================================================================
def new_canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def draw_box(ax, x, y, w, h, fc="white", ec=C_SLATE, lw=0.75, r=0.04, ls="-", zorder=2, alpha=1.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder, alpha=alpha
    )
    ax.add_patch(patch)
    return patch


def draw_header_card(ax, x, y, w, h, title, header_h=0.23, fc=C_SLATE_LT,
                     header_fc=C_NAVY, ec=C_NAVY, title_fs=7.3, zorder=2):
    draw_box(ax, x, y, w, h, fc=fc, ec=ec, lw=0.85, r=0.045, zorder=zorder)
    draw_box(ax, x, y + h - header_h, w, header_h, fc=header_fc, ec=ec, lw=0.85, r=0.045, zorder=zorder + 1)
    ax.add_patch(Rectangle((x + 0.008, y + h - header_h), w - 0.016, header_h * 0.45,
                           facecolor=header_fc, edgecolor="none", zorder=zorder + 1))
    ax.text(x + w / 2.0, y + h - header_h / 2.0 - 0.004, title,
            ha="center", va="center", color="white", fontsize=title_fs,
            fontweight="bold", zorder=zorder + 2)


def draw_arrow(ax, x1, y1, x2, y2, color=C_NAVY, lw=0.95, style="-|>", ms=7.5, ls="-", rad=0.0, zorder=8):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=ms,
        color=color, linewidth=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=zorder
    )
    ax.add_patch(arr)
    return arr


# ==============================================================================
# FIG. 1: MT-3000 Multi-Zone Architecture & 4-Tier DMA Memory Hierarchy
# ==============================================================================
def make_fig1_arch_overview():
    W, H = 7.16, 2.74
    fig, ax = new_canvas(W, H)

    # ---------------- 1. LEFT: GP ZONE & FABRIC (x: 0.03 .. 2.13) ----------------
    draw_header_card(ax, 0.03, 0.05, 2.10, 2.64,
                     "MT-3000 GP Zone (16 ARMv8 Cores)",
                     header_h=0.23, fc=C_SLATE_LT, header_fc=C_SLATE, ec=C_SLATE, title_fs=7.1)

    # 16 ARMv8 Core Grid
    draw_box(ax, 0.10, 1.87, 1.96, 0.54, fc="white", ec=C_SLATE, lw=0.65, r=0.03, zorder=3)
    ax.text(1.08, 2.30, "16 ARMv8 Host CPU Cores (Linux OS)", ha="center", va="center",
            fontsize=6.5, fontweight="bold", color="#1A202C", zorder=4)
    for r_i in range(2):
        for c_i in range(8):
            cx = 0.155 + c_i * 0.234
            cy = 1.92 + r_i * 0.145
            draw_box(ax, cx, cy, 0.20, 0.115, fc="#DCE4EC", ec=C_SLATE, lw=0.45, r=0.015, zorder=4)
            ax.text(cx + 0.10, cy + 0.057, f"C{r_i*8+c_i}", ha="center", va="center",
                    fontsize=5.0, color="#1A202C", zorder=5)

    # Host Co-Design Engine
    draw_box(ax, 0.10, 0.90, 1.96, 0.89, fc=C_NAVY_LT, ec=C_NAVY, lw=0.8, r=0.035, zorder=3)
    ax.text(1.08, 1.68, "Host 2D Hybrid Orchestration (4 Ranks/Node)",
            ha="center", va="center", fontsize=6.35, fontweight="bold", color=C_NAVY, zorder=4)
    bullets = [
        r"$\bullet$ $P/2 \times 2$ Grid ($\mathrm{dev\_id} = \mathrm{rank}\;\mathrm{mod}\;4$)",
        r"$\bullet$ Zero-Copy $\mathtt{MPI\_Type\_vector}$ ($B$ Panels)",
        r"$\bullet$ Async $\mathtt{Isend/Irecv}$ + Subgroup $\mathtt{Ibcast}$",
        r"$\bullet$ 64-Col Vector-Aligned Dispatch ($\mathtt{chunk\_64}$)"
    ]
    for idx, btxt in enumerate(bullets):
        ax.text(0.15, 1.50 - idx * 0.165, btxt, ha="left", va="center",
                fontsize=5.85, color="#111111", zorder=4)

    # Bottom: NIC & Lustre PFS
    draw_box(ax, 0.10, 0.12, 0.94, 0.62, fc=C_TEAL_LT, ec=C_TEAL, lw=0.7, r=0.03, zorder=3)
    ax.text(0.57, 0.62, "High-Speed NIC", ha="center", va="center",
            fontsize=6.2, fontweight="bold", color=C_TEAL, zorder=4)
    ax.text(0.57, 0.37, "2-Phase Non-Blocking\nPanel-B Collectives\n(512 Nodes / 2K Ranks)",
            ha="center", va="center", fontsize=5.35, color="#111111", zorder=4)

    draw_box(ax, 1.12, 0.12, 0.94, 0.62, fc=C_GREEN_LT, ec=C_GREEN, lw=0.7, r=0.03, zorder=3)
    ax.text(1.59, 0.62, "Parallel File Sys.", ha="center", va="center",
            fontsize=6.2, fontweight="bold", color=C_GREEN, zorder=4)
    ax.text(1.59, 0.37, "Offset-Addressed\nConcurrent pread(A)\n(Zero MPI on Matrix A)",
            ha="center", va="center", fontsize=5.35, color="#111111", zorder=4)

    draw_arrow(ax, 0.57, 0.74, 0.57, 0.90, color=C_TEAL, lw=0.85, style="<->", ms=5.5)
    draw_arrow(ax, 1.59, 0.74, 1.59, 0.90, color=C_GREEN, lw=0.85, style="-|>", ms=6.5)

    # ---------------- 2. CENTER: ACC ZONE (x: 2.31 .. 5.01) ----------------
    draw_header_card(ax, 2.31, 0.05, 2.70, 2.64,
                     "MT-3000 ACC Zone (4 DSP Clusters, 11.6 TFLOPS)",
                     header_h=0.23, fc="#FAFBFD", header_fc=C_NAVY, ec=C_NAVY, title_fs=7.1)

    # 4 Autonomous Cluster Tabs
    cl_w = 0.615
    for c_idx in range(4):
        cx = 2.38 + c_idx * 0.645
        is_active = (c_idx == 0)
        draw_box(ax, cx, 2.15, cl_w, 0.25,
                 fc=C_NAVY if is_active else C_NAVY_MID,
                 ec=C_NAVY, lw=0.75 if is_active else 0.55, r=0.025, zorder=4)
        ax.text(cx + cl_w / 2, 2.275, f"Cluster {c_idx}\n(24 DSPs)",
                ha="center", va="center", fontsize=5.6, fontweight="bold",
                color="white" if is_active else C_NAVY, zorder=5)

    # Expanded Cluster 0 Zoom Container
    draw_box(ax, 2.38, 0.12, 2.56, 1.97, fc="white", ec=C_NAVY, lw=0.8, r=0.035, zorder=3)
    ax.text(3.66, 1.99, "Autonomous DSP Cluster (24 Cores @ 1.2 GHz, 1.84 TFLOPS Dual-FMA)",
            ha="center", va="center", fontsize=5.65, fontweight="bold", color=C_NAVY, zorder=4)

    # 24 DSP Core Array Strip
    for d_i, lbl in enumerate(["DSP 0", "DSP 1", "DSP 2", "...", "DSP 22", "DSP 23"]):
        dx = 2.44 + d_i * 0.412
        active_d = (d_i == 0)
        draw_box(ax, dx, 1.67, 0.38, 0.21,
                 fc=C_TEAL if active_d else "#E2E8F0",
                 ec=C_TEAL if active_d else C_SLATE, lw=0.65, r=0.02, zorder=4)
        ax.text(dx + 0.19, 1.775, lbl, ha="center", va="center",
                fontsize=5.6, fontweight="bold", color="white" if active_d else "#1A202C", zorder=5)

    # Zoomed Single DSP Core Microarchitecture
    draw_box(ax, 2.44, 0.52, 2.44, 1.07, fc="#F8FAFC", ec=C_TEAL, lw=0.8, r=0.03, zorder=4)
    ax.text(3.66, 1.49, "Single DSP Acceleration Core (76.8 GFLOPS FP64 Peak)",
            ha="center", va="center", fontsize=6.1, fontweight="bold", color=C_TEAL, zorder=5)

    # Left inside core: SPU + 64 KB SM + SVR
    draw_box(ax, 2.50, 0.58, 0.98, 0.81, fc=C_AMBER_LT, ec=C_AMBER, lw=0.7, r=0.025, zorder=5)
    ax.text(2.99, 1.29, "SPU + 64 KB SM", ha="center", va="center",
            fontsize=6.0, fontweight="bold", color="#9A4D00", zorder=6)
    ax.text(2.99, 1.05, "A Ping-Pong (56 KB)\n87.5% Occupancy", ha="center", va="center",
            fontsize=5.45, color="#111111", zorder=6)
    draw_box(ax, 2.55, 0.63, 0.88, 0.23, fc="white", ec=C_AMBER, lw=0.6, r=0.02, zorder=6)
    ax.text(2.99, 0.745, "SVR: vec_svbcast()", ha="center", va="center",
            fontsize=5.45, fontweight="bold", color="#9A4D00", zorder=7)

    # Right inside core: 16-Lane VPU + 32 Pinned VRs + 768 KB AM
    draw_box(ax, 3.58, 0.58, 1.24, 0.81, fc=C_NAVY_LT, ec=C_NAVY, lw=0.7, r=0.025, zorder=5)
    ax.text(4.20, 1.29, "16-Lane FP64 VPU", ha="center", va="center",
            fontsize=6.0, fontweight="bold", color=C_NAVY, zorder=6)
    ax.text(4.20, 1.05, "Dual-Issue vec_mula\n32 VRs Pinned (4x4 Tile)", ha="center", va="center",
            fontsize=5.45, color="#111111", zorder=6)
    draw_box(ax, 3.63, 0.63, 1.14, 0.23, fc=C_PURPLE_LT, ec=C_PURPLE, lw=0.6, r=0.02, zorder=6)
    ax.text(4.20, 0.745, "768 KB AM (232 KB B+C)", ha="center", va="center",
            fontsize=5.45, fontweight="bold", color=C_PURPLE, zorder=7)

    draw_arrow(ax, 3.44, 0.75, 3.62, 1.03, color=C_AMBER, lw=0.85, style="-|>", ms=6.5)

    # Bottom inside cluster: 14-Channel Hardware P2P DMA Engine
    draw_box(ax, 2.44, 0.17, 2.44, 0.28, fc=C_GREEN_LT, ec=C_GREEN, lw=0.75, r=0.025, zorder=4)
    ax.text(3.66, 0.31, "14-Channel Async P2P DMA Engine (ch1..8 A/B + chc1..6 Cx3)",
            ha="center", va="center", fontsize=5.8, fontweight="bold", color=C_GREEN, zorder=5)

    # Host-to-ACC Hthread Dispatch Arrow
    draw_arrow(ax, 2.13, 1.35, 2.31, 1.35, color=C_NAVY, lw=1.0, style="-|>", ms=7)
    ax.text(2.22, 1.62, "Hthread\nDispatch", ha="center", va="center",
            fontsize=5.3, fontweight="bold", color=C_NAVY, rotation=90)

    # ---------------- 3. RIGHT: 4-TIER MEMORY HIERARCHY (x: 5.19 .. 7.13) ----------------
    draw_header_card(ax, 5.19, 0.05, 1.94, 2.64,
                     "4-Tier Explicit Memory Hierarchy",
                     header_h=0.23, fc=C_SLATE_LT, header_fc=C_TEAL, ec=C_TEAL, title_fs=7.1)

    tiers = [
        (1.87, C_NAVY_LT, C_NAVY, "Tier 1: Vector RegFile (64 VRs)",
         r"32 VRs Pinned ($4\times 64$ Complex $C$)" + "\n" + r"Zero AM Traffic Across $K_b = 32$"),
        (1.30, C_PURPLE_LT, C_PURPLE, "Tier 2: Core SM (64K) + AM (768K)",
         "~921.6 GB/s Internal Vector BW\n86.1% AM L/S Reduction (7.20x)"),
        (0.73, C_AMBER_LT, C_AMBER, "Tier 3: GSM (6M) / HBSM (48M)",
         "On-Chip Shared Cluster Scratchpad\nFast Kernel Args & Group Sync"),
        (0.14, C_GREEN_LT, C_GREEN, "Tier 4: Off-Chip DDR4 (32 GB)",
         r"16.22 GB/s Direct Cluster DMA BW" + "\n" + r"$\mathcal{I}=14.83$ F/B $\rightarrow$ 243.4 GF/s (~100%)"),
    ]
    for ty, tfc, tec, ttitle, tdesc in tiers:
        draw_box(ax, 5.26, ty, 1.80, 0.51, fc=tfc, ec=tec, lw=0.75, r=0.03, zorder=3)
        ax.text(6.16, ty + 0.39, ttitle, ha="center", va="center",
                fontsize=5.95, fontweight="bold", color=tec, zorder=4)
        ax.text(6.16, ty + 0.17, tdesc, ha="center", va="center",
                fontsize=5.4, color="#111111", zorder=4)

    # Arrows linking ACC Core/DMA to Memory Tiers
    draw_arrow(ax, 4.88, 1.15, 5.25, 2.11, color=C_NAVY, lw=0.8, style="<->", ms=6.5)
    draw_arrow(ax, 4.88, 0.75, 5.25, 1.54, color=C_PURPLE, lw=0.8, style="<->", ms=6.5)
    draw_arrow(ax, 4.94, 0.31, 5.25, 0.37, color=C_GREEN, lw=1.05, style="<->", ms=7)
    ax.text(5.10, 0.54, "DMA", ha="center", va="center",
            fontsize=5.2, fontweight="bold", color=C_GREEN, rotation=90)

    out_path = os.path.join(OUT_DIR, "fig1_arch_overview.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print("Saved:", out_path)


# ==============================================================================
# FIG. 2: P/2 x 2 Hybrid 2D Process Grid & Overlapped Comm-I/O Timeline
# ==============================================================================
def make_fig2_2d_comm_timeline():
    W, H = 7.16, 2.60
    fig, ax = new_canvas(W, H)

    # Panel Titles
    ax.text(1.73, 2.50, "(a) Stationary-A, Panel-B 2D Grid (P/2 x 2) & Vector Alignment",
            ha="center", va="center", fontsize=7.9, fontweight="bold", color="#111111")
    ax.text(5.34, 2.50, "(b) 1D Global Broadcast vs. 2D Overlapped Execution Timeline",
            ha="center", va="center", fontsize=7.9, fontweight="bold", color="#111111")

    # ---------------- PANEL (a): LEFT (x: 0.03 .. 3.43) ----------------
    draw_box(ax, 0.03, 0.04, 3.40, 2.36, fc="#FAFBFD", ec="#CBD5E1", lw=0.7, r=0.04, zorder=1)

    # Top-Right: Rank 0 Zero-Copy B Matrix Split
    ax.text(2.41, 2.29, r"Matrix $B \in \mathbb{C}^{K \times N}$ at Rank 0 (Zero-Copy Vector Split)",
            ha="center", va="center", fontsize=6.0, fontweight="bold", color="#111111", zorder=3)
    draw_box(ax, 1.52, 1.84, 0.87, 0.36, fc=C_NAVY_LT, ec=C_NAVY, lw=0.8, r=0.02, zorder=3)
    ax.text(1.955, 2.02, r"Panel $B_0$ ($K \times \frac{N}{2}$)" + "\nStays at Leader 0",
            ha="center", va="center", fontsize=5.45, fontweight="bold", color=C_NAVY, zorder=4)

    draw_box(ax, 2.45, 1.84, 0.89, 0.36, fc=C_TEAL_LT, ec=C_TEAL, lw=0.8, r=0.02, zorder=3)
    ax.text(2.895, 2.02, r"Panel $B_1$ ($K \times \frac{N}{2}$)" + "\nMPI_Type_vector",
            ha="center", va="center", fontsize=5.45, fontweight="bold", color=C_TEAL, zorder=4)

    # Phase 1 Isend arrow from Panel B1 to Group 1 Leader
    draw_arrow(ax, 2.895, 1.84, 2.895, 1.62, color=C_TEAL, lw=0.95, style="-|>", ms=6.5)
    ax.text(3.16, 1.73, "Phase 1\nIsend", ha="center", va="center",
            fontsize=5.2, fontweight="bold", color=C_TEAL)

    # Left: Matrix A on Parallel File System (x: 0.10 .. 1.04, y: 0.60 .. 1.72)
    draw_box(ax, 0.10, 0.60, 0.94, 1.13, fc=C_GREEN_LT, ec=C_GREEN, lw=0.8, r=0.03, zorder=3)
    ax.text(0.57, 1.63, r"PFS: Matrix $A$", ha="center", va="center",
            fontsize=6.0, fontweight="bold", color=C_GREEN, zorder=4)
    row_labels = [r"$A_0$ ($\frac{2M}{P}\times K$)", r"$A_1$ ($\frac{2M}{P}\times K$)", r"$\vdots$", r"$A_{P/2-1}$"]
    row_ys = [1.33, 1.09, 0.89, 0.65]
    row_hs = [0.20, 0.20, 0.16, 0.20]
    for rl, ry, rh in zip(row_labels, row_ys, row_hs):
        draw_box(ax, 0.15, ry, 0.84, rh, fc="white", ec=C_GREEN, lw=0.55, r=0.015, zorder=4)
        ax.text(0.57, ry + rh / 2, rl, ha="center", va="center", fontsize=5.4, color="#111111", zorder=5)

    for ry in [1.43, 1.19, 0.75]:
        draw_arrow(ax, 1.00, ry, 1.50, ry, color=C_GREEN, lw=0.85, style="-|>", ms=6.5)
    ax.text(1.25, 1.56, "Offset pread\n(Zero MPI)", ha="center", va="center",
            fontsize=5.25, fontweight="bold", color=C_GREEN)

    # Center-Right: P/2 x 2 Process Grid producing Disjoint C_{r,g}
    ax.text(1.955, 1.59, "Group 0 (Even)", ha="center", va="center",
            fontsize=5.8, fontweight="bold", color=C_NAVY)
    ax.text(2.895, 1.59, "Group 1 (Odd)", ha="center", va="center",
            fontsize=5.8, fontweight="bold", color=C_TEAL)

    g0_cells = [("Rank 0", r"$C_{0,0}=A_0 B_0$"),
                ("Rank 2", r"$C_{1,0}=A_1 B_0$"),
                ("Ibcast", r"Subgroup Ibcast($B_0$)"),
                ("Rank P-2", r"$C_{P/2-1,0}$")]
    g1_cells = [("Rank 1", r"$C_{0,1}=A_0 B_1$"),
                ("Rank 3", r"$C_{1,1}=A_1 B_1$"),
                ("Ibcast", r"Subgroup Ibcast($B_1$)"),
                ("Rank P-1", r"$C_{P/2-1,1}$")]

    for (r0, c0), (r1, c1), ry, rh in zip(g0_cells, g1_cells, row_ys, row_hs):
        is_ib = (r0 == "Ibcast")
        draw_box(ax, 1.52, ry, 0.87, rh, fc=C_NAVY_LT if is_ib else "white",
                 ec=C_NAVY, lw=0.65, r=0.015, ls="--" if is_ib else "-", zorder=4)
        ax.text(1.955, ry + rh / 2, f"{r0}: {c0}" if not is_ib else c0,
                ha="center", va="center", fontsize=5.15, color=C_NAVY, fontweight="bold", zorder=5)

        draw_box(ax, 2.45, ry, 0.89, rh, fc=C_TEAL_LT if is_ib else "white",
                 ec=C_TEAL, lw=0.65, r=0.015, ls="--" if is_ib else "-", zorder=4)
        ax.text(2.895, ry + rh / 2, f"{r1}: {c1}" if not is_ib else c1,
                ha="center", va="center", fontsize=5.15, color=C_TEAL, fontweight="bold", zorder=5)

    # Bottom Strip of Panel (a): 24-Core chunk_64 Vector Alignment
    draw_box(ax, 0.10, 0.10, 3.24, 0.43, fc=C_AMBER_LT, ec=C_AMBER, lw=0.75, r=0.025, zorder=3)
    ax.text(1.72, 0.44, r"Intra-Cluster 24-Core Vector-Aligned Partitioning ($\mathtt{chunk\_64} = 4 \times 16$ FP64 Cols)",
            ha="center", va="center", fontsize=5.7, fontweight="bold", color="#9A4D00", zorder=4)
    segs = [("Core 0: 64k Cols", 0.15, 0.74), ("Core 1: 64k Cols", 0.93, 0.74),
            ("All Multiples of 64", 1.71, 0.84), ("Core 23: Tail", 2.59, 0.69)]
    for stxt, sx, sw in segs:
        draw_box(ax, sx, 0.14, sw, 0.19, fc="white", ec=C_AMBER, lw=0.55, r=0.015, zorder=4)
        ax.text(sx + sw / 2, 0.235, stxt, ha="center", va="center", fontsize=5.1, color="#111111", zorder=5)

    # ---------------- PANEL (b): RIGHT (x: 3.55 .. 7.13) ----------------
    draw_box(ax, 3.55, 0.04, 3.58, 2.36, fc="#FAFBFD", ec="#CBD5E1", lw=0.7, r=0.04, zorder=1)

    # --- Top Half: 1D Flat Baseline Timeline (y: 1.51 .. 2.33) ---
    draw_box(ax, 3.62, 1.51, 3.44, 0.82, fc=C_RED_LT, ec=C_RED, lw=0.75, r=0.03, zorder=2)
    ax.text(5.34, 2.23, "1D Flat Global Broadcast Baseline (512 Nodes: 102.35 s E2E)",
            ha="center", va="center", fontsize=6.1, fontweight="bold", color=C_RED, zorder=3)

    ax.text(3.96, 2.01, "Rank 0", ha="center", va="center", fontsize=5.5, fontweight="bold", color="#222222")
    ax.text(3.96, 1.80, "Ranks 1..P", ha="center", va="center", fontsize=5.5, fontweight="bold", color="#222222")
    ax.text(3.96, 1.60, "DSP Cores", ha="center", va="center", fontsize=5.5, fontweight="bold", color="#222222")

    draw_box(ax, 4.31, 1.93, 0.72, 0.16, fc="#F5B7B1", ec=C_RED, lw=0.6, r=0.015, zorder=4)
    ax.text(4.67, 2.01, "Read Full A,B", ha="center", va="center", fontsize=5.15, color="#78281F", fontweight="bold", zorder=5)

    draw_box(ax, 4.31, 1.72, 0.72, 0.16, fc="#E5E7EB", ec="#9CA3AF", lw=0.55, r=0.015, ls="--", zorder=4)
    ax.text(4.67, 1.80, "Blocked (Idle)", ha="center", va="center", fontsize=5.05, color="#4B5563", zorder=5)

    draw_box(ax, 5.06, 1.72, 1.50, 0.37, fc=C_RED, ec="#78281F", lw=0.7, r=0.02, zorder=4)
    ax.text(5.81, 1.905, "Global Bcast(100% B) + Scatterv(A)\n512N Comm Storm: 102.25 s!",
            ha="center", va="center", fontsize=5.35, color="white", fontweight="bold", zorder=5)

    draw_box(ax, 6.59, 1.53, 0.42, 0.16, fc="#F5B7B1", ec=C_RED, lw=0.6, r=0.015, zorder=4)
    ax.text(6.80, 1.61, "0.098s", ha="center", va="center", fontsize=5.15, color="#78281F", fontweight="bold", zorder=5)

    # --- Bottom Half: Proposed 2D Overlapped Timeline (y: 0.10 .. 1.44) ---
    draw_box(ax, 3.62, 0.10, 3.44, 1.34, fc=C_NAVY_LT, ec=C_NAVY, lw=0.8, r=0.03, zorder=2)
    ax.text(5.34, 1.34, "Proposed 2D Overlapped Pipeline (512 Nodes: 10.38 s E2E, 9.86x Faster)",
            ha="center", va="center", fontsize=6.1, fontweight="bold", color=C_NAVY, zorder=3)

    lanes_2d = [("Leader 0", 1.10), ("Leader 1", 0.86), ("Members", 0.62), ("24x4 DSPs", 0.36)]
    for ltxt, ly in lanes_2d:
        ax.text(3.96, ly, ltxt, ha="center", va="center", fontsize=5.45, fontweight="bold", color="#1A202C")

    draw_box(ax, 4.31, 1.02, 0.43, 0.16, fc=C_AMBER_LT, ec=C_AMBER, lw=0.6, r=0.015, zorder=4)
    ax.text(4.525, 1.10, "Read B", ha="center", va="center", fontsize=5.1, fontweight="bold", color="#9A4D00", zorder=5)

    draw_box(ax, 4.77, 1.02, 0.80, 0.16, fc=C_TEAL_LT, ec=C_TEAL, lw=0.65, r=0.015, zorder=4)
    ax.text(5.17, 1.10, "Isend(B1) || pread(A0)", ha="center", va="center", fontsize=5.0, fontweight="bold", color=C_TEAL, zorder=5)

    draw_box(ax, 4.31, 0.78, 1.26, 0.16, fc=C_GREEN_LT, ec=C_GREEN, lw=0.65, r=0.015, zorder=4)
    ax.text(4.94, 0.86, "Irecv(B1) || pread(A0) (82.5% Hidden)",
            ha="center", va="center", fontsize=5.0, fontweight="bold", color=C_GREEN, zorder=5)

    draw_box(ax, 4.31, 0.54, 0.70, 0.16, fc=C_GREEN_LT, ec=C_GREEN, lw=0.65, r=0.015, zorder=4)
    ax.text(4.66, 0.62, r"pread($A_r$) 0.04s", ha="center", va="center", fontsize=5.0, fontweight="bold", color=C_GREEN, zorder=5)

    draw_box(ax, 5.04, 0.54, 0.53, 0.16, fc="white", ec=C_NAVY, lw=0.55, r=0.015, ls="--", zorder=4)
    ax.text(5.305, 0.62, "Post Ibcast", ha="center", va="center", fontsize=4.95, color=C_NAVY, zorder=5)

    draw_box(ax, 5.61, 0.54, 0.78, 0.64, fc=C_NAVY, ec=C_NAVY, lw=0.75, r=0.02, zorder=4)
    ax.text(6.00, 0.86, "Phase 2:\nSubgroup Parallel\nIbcast(B0) || Ibcast(B1)\n(50% Size, 9.90x)",
            ha="center", va="center", fontsize=5.0, color="white", fontweight="bold", zorder=5)

    draw_box(ax, 6.42, 0.27, 0.59, 0.21, fc=C_TEAL, ec=C_NAVY, lw=0.7, r=0.02, zorder=4)
    ax.text(6.715, 0.375, "0.0489 s\n89.9 TFLOPS", ha="center", va="center",
            fontsize=5.0, color="white", fontweight="bold", zorder=5)

    draw_arrow(ax, 4.31, 0.18, 7.01, 0.18, color=C_SLATE, lw=0.8, style="-|>", ms=6.5)
    ax.text(5.36, 0.195, "Wall-Clock Time (Schematic Scale)", ha="center", va="bottom",
            fontsize=5.05, color=C_SLATE)

    out_path = os.path.join(OUT_DIR, "fig2_2d_comm_timeline.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print("Saved:", out_path)


# ==============================================================================
# FIG. 3: 14-Channel Tri-Buffered DMA Pipeline & 4x4 32-VR Micro-Kernel
# ==============================================================================
def make_fig3_tri_pipeline_microkernel():
    W, H = 7.16, 2.66
    fig, ax = new_canvas(W, H)

    ax.text(1.86, 2.56, "(a) 14-Channel Asynchronous DMA Space-Time Pipeline",
            ha="center", va="center", fontsize=7.9, fontweight="bold", color="#111111")
    ax.text(5.47, 2.56, "(b) 4x4 Register-Resident Dual-FMA Micro-Kernel (32 VRs)",
            ha="center", va="center", fontsize=7.9, fontweight="bold", color="#111111")

    # ---------------- PANEL (a): LEFT (x: 0.03 .. 3.70) ----------------
    draw_box(ax, 0.03, 0.04, 3.67, 2.43, fc="#FAFBFD", ec="#CBD5E1", lw=0.7, r=0.04, zorder=1)

    # Top Sub-Box: Middle-Tier C Triple Buffering across N_b Steps
    draw_box(ax, 0.10, 1.28, 3.53, 1.13, fc="white", ec=C_PURPLE, lw=0.75, r=0.03, zorder=2)
    ax.text(1.865, 2.31, r"Middle Tier: $C$-Matrix Cyclic Triple Buffering ($\mathtt{chc1..chc6}$, 168 KB AM)",
            ha="center", va="center", fontsize=5.95, fontweight="bold", color=C_PURPLE, zorder=3)

    col_xs = [0.76, 1.32, 1.88, 2.44, 3.00]
    col_w = 0.53
    step_hdrs = ["Prologue", "Step j=1", "Step j=2", "Step j=3", "Epilogue"]
    for cx, sh in zip(col_xs, step_hdrs):
        ax.text(cx + col_w / 2, 2.14, sh, ha="center", va="center",
                fontsize=5.3, fontweight="bold", color="#333333")

    draw_box(ax, 1.86, 1.33, 1.13, 0.74, fc=C_PURPLE_LT, ec=C_PURPLE, lw=0.7, ls="--", r=0.02, zorder=2)

    lane_ys = [1.88, 1.63, 1.38]
    lane_names = ["DMA Load\n(DDR to AM)", "16-VPU\nAccumulate", "DMA Store\n(AM to DDR)"]
    for ly, ln in zip(lane_ys, lane_names):
        ax.text(0.42, ly + 0.09, ln, ha="center", va="center", fontsize=5.1, fontweight="bold", color="#222222")

    c_load  = [("Load C0: B1", C_TEAL_LT, C_TEAL), ("Load C1: B2", C_TEAL_LT, C_TEAL),
               ("Load C2: B3", C_TEAL_LT, C_TEAL), ("Load C3: B1", C_TEAL_LT, C_TEAL), None]
    c_comp  = [None, ("Comp C0 [B1]", C_NAVY, "white"), ("Comp C1 [B2]", C_NAVY, "white"),
               ("Comp C2 [B3]", C_NAVY, "white"), ("Comp C3 [B1]", C_NAVY, "white")]
    c_store = [None, None, ("Store C0 [B1]", C_AMBER_LT, "#9A4D00"),
               ("Store C1 [B2]", C_AMBER_LT, "#9A4D00"), ("Store C2..3", C_AMBER_LT, "#9A4D00")]

    for idx, cx in enumerate(col_xs):
        if c_load[idx]:
            txt, fc, tc = c_load[idx]
            draw_box(ax, cx, lane_ys[0], col_w, 0.18, fc=fc, ec=C_TEAL, lw=0.6, r=0.015, zorder=4)
            ax.text(cx + col_w / 2, lane_ys[0] + 0.09, txt, ha="center", va="center",
                    fontsize=4.95, fontweight="bold", color=tc, zorder=5)
        if c_comp[idx]:
            txt, fc, tc = c_comp[idx]
            draw_box(ax, cx, lane_ys[1], col_w, 0.18, fc=fc, ec=C_NAVY, lw=0.6, r=0.015, zorder=4)
            ax.text(cx + col_w / 2, lane_ys[1] + 0.09, txt, ha="center", va="center",
                    fontsize=4.95, fontweight="bold", color=tc, zorder=5)
        if c_store[idx]:
            txt, fc, tc = c_store[idx]
            draw_box(ax, cx, lane_ys[2], col_w, 0.18, fc=fc, ec=C_AMBER, lw=0.6, r=0.015, zorder=4)
            ax.text(cx + col_w / 2, lane_ys[2] + 0.09, txt, ha="center", va="center",
                    fontsize=4.95, fontweight="bold", color=tc, zorder=5)

    draw_arrow(ax, 2.145, 1.63, 2.145, 1.18, color=C_NAVY, lw=0.95, style="-|>", ms=6.5)

    # Bottom Sub-Box: Inner-Tier A/B Ping-Pong Double Buffering
    draw_box(ax, 0.10, 0.10, 3.53, 1.08, fc="white", ec=C_NAVY, lw=0.75, r=0.03, zorder=2)
    ax.text(1.865, 1.08, r"Inner Tier: $A/B$ Ping-Pong Along $K_b=32$ ($\mathtt{ch1..8}$: 56 KB SM + 64 KB AM)",
            ha="center", va="center", fontsize=5.85, fontweight="bold", color=C_NAVY, zorder=3)

    k_lanes = [("SM DMA (A)", 0.76), ("AM DMA (B)", 0.52), ("16-VPU FMA", 0.28)]
    for kn, ky in k_lanes:
        ax.text(0.43, ky + 0.085, kn, ha="center", va="center", fontsize=5.15, fontweight="bold", color="#222222")

    k_cols = [0.78, 1.48, 2.18, 2.88]
    k_w = 0.66
    a_seq = ["DMA A[0]: s1", "DMA A[1]: s2", "DMA A[2]: s1", "DMA A[3]: s2"]
    b_seq = ["DMA B[0]: a1", "DMA B[1]: a2", "DMA B[2]: a1", "DMA B[3]: a2"]
    f_seq = ["Wait(A0, B0)", "FMA(s1, a1)", "FMA(s2, a2)", "FMA(s1, a1)"]

    for idx, kx in enumerate(k_cols):
        draw_box(ax, kx, 0.76, k_w, 0.17, fc=C_AMBER_LT, ec=C_AMBER, lw=0.55, r=0.015, zorder=4)
        ax.text(kx + k_w / 2, 0.845, a_seq[idx], ha="center", va="center", fontsize=5.0, fontweight="bold", color="#9A4D00", zorder=5)

        draw_box(ax, kx, 0.52, k_w, 0.17, fc=C_GREEN_LT, ec=C_GREEN, lw=0.55, r=0.015, zorder=4)
        ax.text(kx + k_w / 2, 0.605, b_seq[idx], ha="center", va="center", fontsize=5.0, fontweight="bold", color=C_GREEN, zorder=5)

        is_wait = (idx == 0)
        draw_box(ax, kx, 0.28, k_w, 0.17, fc="#E5E7EB" if is_wait else C_NAVY,
                 ec=C_SLATE if is_wait else C_NAVY, lw=0.55, r=0.015, zorder=4)
        ax.text(kx + k_w / 2, 0.365, f_seq[idx], ha="center", va="center",
                fontsize=5.05, fontweight="bold", color="#374151" if is_wait else "white", zorder=5)

    ax.text(1.865, 0.175, r"$\mathtt{dma\_wait()}$ Hoisted Outside $K_b=32$ Loop $\rightarrow$ Zero Stall Cycles Inside FMA Body",
            ha="center", va="center", fontsize=5.35, fontweight="bold", color=C_TEAL, zorder=4)

    # ---------------- PANEL (b): RIGHT (x: 3.82 .. 7.13) ----------------
    draw_box(ax, 3.82, 0.04, 3.31, 2.43, fc="#FAFBFD", ec="#CBD5E1", lw=0.7, r=0.04, zorder=1)

    # Top Vector Load Array from 768 KB AM
    draw_box(ax, 4.74, 1.96, 2.31, 0.44, fc=C_GREEN_LT, ec=C_GREEN, lw=0.75, r=0.025, zorder=3)
    ax.text(5.895, 2.30, r"768 KB AM: 8 $\times$ $\mathtt{vec\_ld}$ per $p \in [0, 31]$ (64 Complex Cols)",
            ha="center", va="center", fontsize=5.65, fontweight="bold", color=C_GREEN, zorder=4)

    vx_list = [4.79, 5.36, 5.93, 6.50]
    vw = 0.52
    for v_i, vx in enumerate(vx_list):
        draw_box(ax, vx, 2.00, vw, 0.19, fc="white", ec=C_GREEN, lw=0.55, r=0.015, zorder=4)
        ax.text(vx + vw / 2, 2.095, f"bx{v_i}, by{v_i} (16)", ha="center", va="center",
                fontsize=5.05, fontweight="bold", color="#111111", zorder=5)
        draw_arrow(ax, vx + vw / 2, 1.99, vx + vw / 2, 1.84, color=C_GREEN, lw=0.8, style="-|>", ms=6)

    # Left Scalar Broadcast Column from 64 KB SM
    draw_box(ax, 3.89, 0.68, 0.72, 1.16, fc=C_AMBER_LT, ec=C_AMBER, lw=0.75, r=0.025, zorder=3)
    ax.text(4.25, 2.02, "64 KB SM\nvec_svbcast", ha="center", va="center",
            fontsize=5.55, fontweight="bold", color="#9A4D00", zorder=4)

    ry_list = [1.56, 1.28, 1.00, 0.72]
    rh_cell = 0.23
    for r_i, ry in enumerate(ry_list):
        draw_box(ax, 3.94, ry, 0.62, rh_cell, fc="white", ec=C_AMBER, lw=0.55, r=0.015, zorder=4)
        ax.text(4.25, ry + rh_cell / 2, f"Row i+{r_i}\n{{ar, ai, -ai}}", ha="center", va="center",
                fontsize=4.95, fontweight="bold", color="#9A4D00", zorder=5)
        draw_arrow(ax, 4.57, ry + rh_cell / 2, 4.76, ry + rh_cell / 2, color=C_AMBER, lw=0.8, style="-|>", ms=6)

    # Center 4x4 Grid of 32 Pinned Physical Vector Accumulators
    draw_box(ax, 4.74, 0.68, 2.31, 1.16, fc=C_NAVY_LT, ec=C_NAVY, lw=0.8, r=0.025, zorder=3)
    for r_i, ry in enumerate(ry_list):
        for v_i, vx in enumerate(vx_list):
            draw_box(ax, vx, ry, vw, rh_cell, fc="white", ec=C_NAVY, lw=0.6, r=0.015, zorder=4)
            ax.text(vx + vw / 2, ry + rh_cell / 2,
                    f"cx{r_i}_{v_i}, cy{r_i}_{v_i}\n(2 x 16 FP64)",
                    ha="center", va="center", fontsize=4.85, fontweight="bold", color=C_NAVY, zorder=5)

    # Bottom Formula & Traffic-Elimination Summary
    draw_box(ax, 3.89, 0.10, 3.16, 0.51, fc="white", ec=C_NAVY, lw=0.75, r=0.025, zorder=3)
    ax.text(5.47, 0.48,
            r"Dual-Issue FMA: $\mathtt{cx} += a_r \mathbf{b}_x + (-a_i)\mathbf{b}_y$,  "
            r"$\mathtt{cy} += a_r \mathbf{b}_y + a_i \mathbf{b}_x$ (64 FMAs/step)",
            ha="center", va="center", fontsize=5.5, fontweight="bold", color=C_NAVY, zorder=4)
    ax.text(5.47, 0.24,
            r"32/64 VRs Pinned (Zero Spilling) | AM Vector L/S: $2{,}304 \rightarrow 320$ ($-86.1\%$, $7.20\times$ Cut)",
            ha="center", va="center", fontsize=5.35, fontweight="bold", color=C_RED, zorder=4)

    out_path = os.path.join(OUT_DIR, "fig3_tri_pipeline_microkernel.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print("Saved:", out_path)


# ==============================================================================
# DATASET 1 & FIG. 4: Intra-Node OpenBLAS Comparison & Accuracy
# ==============================================================================
scales = np.array([512, 1024, 2048, 4096, 8192, 16384])
scale_labels = ["512", "1K", "2K", "4K", "8K", "16K"]

dsp_1c_gflops      = np.array([44.16, 45.77, 49.19, 48.39, 49.16, 49.16])
cpu_1c_4d_gflops   = np.array([ 7.22,  7.07,  6.80,  6.78,  6.76,  7.20])
cpu_1c_1z_gflops   = np.array([ 7.10,  7.69,  7.63,  7.74,  7.60, np.nan])
speedup_1c_4d      = np.array([ 6.12,  6.48,  7.24,  7.13,  7.28,  6.83])

dsp_24c_gflops     = np.array([80.98, 139.73, 145.96, 173.05, 243.44, 199.21])
cpu_4c_4d_gflops   = np.array([23.32,  28.38,  22.84,  24.41,  26.08,  26.62])
cpu_16c_4d_gflops  = np.array([17.23,  51.88,  86.15,  92.28,  44.56,  30.34])
cpu_16c_1z_gflops  = np.array([40.99,  82.29,  92.49,  85.51,  89.65, np.nan])

err_real = np.array([7.276e-12, 1.728e-11, 4.366e-11, 7.458e-11, 1.892e-10, 1.051e-09])
err_imag = np.array([1.128e-10, 3.274e-10, 9.750e-10, 2.794e-09, 7.567e-09, 2.177e-08])


def make_fig4_intra_node():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.48))
    plt.subplots_adjust(wspace=0.38)
    x = np.arange(len(scales))

    # --- Panel (a): Single-Core GFLOPS + Speedup ---
    ax = axes[0]
    w = 0.27
    ax.bar(x - w, dsp_1c_gflops, width=w, color=C_NAVY, edgecolor="black", linewidth=0.45,
           label="MT-3000 DSP (1 Core)", zorder=3)
    ax.bar(x, np.nan_to_num(cpu_1c_1z_gflops, nan=0.0), width=w, color=C_TEAL, edgecolor="black",
           linewidth=0.45, label="CPU 1C (1x ZGEMM)", zorder=3)
    ax.bar(x + w, cpu_1c_4d_gflops, width=w, color=C_AMBER, edgecolor="black", linewidth=0.45,
           label="CPU 1C (4x DGEMM)", zorder=3)

    ax.text(x[-1], 1.5, "OOM", ha="center", va="bottom", fontsize=5.4, color=C_TEAL, rotation=90, fontweight="bold")

    for idx, (g, sp) in enumerate(zip(dsp_1c_gflops, speedup_1c_4d)):
        ax.text(x[idx] - w, g + 1.2, f"{sp:.1f}x", ha="center", va="bottom",
                fontsize=5.6, fontweight="bold", color=C_NAVY)

    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.set_xlabel("Matrix Dimension ($M=N=K$)")
    ax.set_ylabel("Throughput (GFLOPS)")
    ax.set_ylim(0, 74)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.set_title("(a) Single-Core Compute Density", fontweight="bold", pad=5)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94,
              edgecolor="#CCCCCC", borderpad=0.22, labelspacing=0.16, handletextpad=0.35, fontsize=5.85)

    # --- Panel (b): Cluster (24 DSP) vs Multi-Core CPU ---
    ax = axes[1]
    ax.plot(x, dsp_24c_gflops, marker="o", color=C_NAVY, linewidth=1.6,
            label="DSP 1 Cluster (24C)", zorder=5)
    ax.plot(x[:-1], cpu_16c_1z_gflops[:-1], marker="s", color=C_TEAL, linewidth=1.35,
            label="CPU 16C (1x ZGEMM)", zorder=4)
    ax.plot(x, cpu_16c_4d_gflops, marker="^", color=C_AMBER, linewidth=1.3, linestyle="--",
            label="CPU 16C (4x DGEMM)", zorder=4)
    ax.plot(x, cpu_4c_4d_gflops, marker="d", color=C_SLATE, linewidth=1.2, linestyle=":",
            label="CPU 4C (Cluster-Eq)", zorder=3)

    ax.axhline(240.5, color=C_GREEN, linestyle="-.", linewidth=0.95, zorder=2)
    ax.text(5.10, 250.0, "Roofline: 240.5", ha="right", va="bottom",
            fontsize=5.65, color=C_GREEN, fontweight="bold")

    ax.annotate("243.4 GF/s\n(9.33x vs 4C)", xy=(4, 243.44), xytext=(0.45, 196),
                fontsize=5.75, fontweight="bold", color=C_NAVY,
                arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=0.75))
    ax.annotate("2x2 Tiled\n(6.57x vs 16C)", xy=(5, 199.21), xytext=(3.15, 126),
                fontsize=5.65, fontweight="bold", color=C_NAVY,
                arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.set_xlabel("Matrix Dimension ($M=N=K$)")
    ax.set_ylabel("Throughput (GFLOPS)")
    ax.set_ylim(0, 338)
    ax.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_title("(b) Single-Cluster vs. Multi-Core CPU", fontweight="bold", pad=5)
    ax.legend(loc="upper left", ncol=1, frameon=True, facecolor="white",
              framealpha=0.94, edgecolor="#CCCCCC", borderpad=0.22, labelspacing=0.14, fontsize=5.6)

    # --- Panel (c): Numerical Precision Verification ---
    ax = axes[2]
    ax.semilogy(x, err_imag, marker="o", color=C_PURPLE, linewidth=1.4,
                label=r"Imag Max $|\Delta C_{\mathrm{Im}}|$", zorder=4)
    ax.semilogy(x, err_real, marker="s", color=C_NAVY, linewidth=1.4, linestyle="--",
                label=r"Real Max $|\Delta C_{\mathrm{Re}}|$", zorder=4)

    ax.axhline(1.19e-7, color=C_RED, linestyle=":", linewidth=0.95, zorder=2)
    ax.text(0.12, 1.75e-7, r"FP32 $\epsilon_{\mathrm{mach}}$ ($1.19\times 10^{-7}$)",
            fontsize=5.8, color=C_RED, fontweight="bold")

    ref_slope = 2.0e-13 * scales
    ax.semilogy(x, ref_slope, color=C_SLATE, linestyle="-.", linewidth=0.85,
                label=r"$\mathcal{O}(K \epsilon_{\mathrm{FP64}})$ Slope", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.set_xlabel("Matrix Dimension ($M=N=K$)")
    ax.set_ylabel(r"Max Element-Wise Error ($\ell_\infty$)")
    ax.set_ylim(1e-12, 1e-6)
    ax.grid(True, which="both", linestyle="--", alpha=0.38, zorder=0)
    ax.set_title("(c) FP64 Numerical Accuracy vs. BLAS", fontweight="bold", pad=5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94,
              edgecolor="#CCCCCC", borderpad=0.22, labelspacing=0.16, fontsize=5.8)

    out_path = os.path.join(OUT_DIR, "fig4_intra_node_openblas.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print("Saved:", out_path)


# ==============================================================================
# DATASET 2 & FIG. 5: Empirical Roofline Model (Single-Column 3.50 in)
# ==============================================================================
def make_fig5_roofline():
    fig, ax = plt.subplots(figsize=(3.50, 2.55))

    ai = np.logspace(-0.22, 1.82, 400)
    bw_ddr = 16.22
    peak_1core = 76.8
    peak_cluster = 24 * peak_1core  # 1843.2 GFLOPS

    roof_cluster = np.minimum(peak_cluster, bw_ddr * ai)
    roof_1core   = np.minimum(peak_1core, bw_ddr * ai)

    ax.loglog(ai, roof_cluster, color=C_NAVY, linewidth=1.6,
              label="24-Core Cluster Roofline (16.22 GB/s DDR)")
    ax.loglog(ai, roof_1core, color=C_TEAL, linewidth=1.25, linestyle="--",
              label="1-Core Dual-FMA Ceiling (76.8 GFLOPS)")

    ax.axvline(14.83, color="#999999", linestyle=":", linewidth=0.85, zorder=1)
    ax.text(16.0, 8.2, "Tile (56,32,64)\n" + r"$\mathcal{I}=14.83$ F/B",
            fontsize=5.8, color="#222222")

    # 1. Single-core sustained point
    ax.scatter([14.83], [49.19], color=C_TEAL, marker="s", s=34, edgecolor="black", linewidth=0.5, zorder=6)
    ax.annotate("1-Core Sustained\n(49.2 GF/s, 64.1%)", xy=(14.83, 49.19), xytext=(16.5, 29.0),
                fontsize=5.65, color=C_TEAL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=0.7))

    # 2. 24-Core Cluster trajectory
    ai_traj = np.array([5.15, 9.35, 9.75, 13.05, 14.83])
    gf_traj = np.array([80.98, 139.73, 145.96, 173.05, 243.44])
    ax.plot(ai_traj, gf_traj, color=C_GREEN, linestyle="-", linewidth=1.15, marker="o",
            markersize=4.0, markeredgecolor="black", markeredgewidth=0.45, zorder=6,
            label="24-Core Pipeline Trajectory ($N$: 512$\\to$8K)")

    ax.annotate("$N$=512\n(1 tile, no C-rot.)", xy=(5.15, 80.98), xytext=(4.85, 31.0),
                fontsize=5.5, color="#222222",
                arrowprops=dict(arrowstyle="->", color="#444444", lw=0.65))
    ax.annotate("$N$=8192: 243.4 GFLOPS\n(~100% DDR DMA Ceiling)", xy=(14.83, 243.44), xytext=(2.25, 455.0),
                fontsize=5.85, fontweight="bold", color=C_NAVY,
                arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=0.8))

    # 3. 512-Node (P=2048) per-cluster operating points: 1D (8K) -> 2D (8K) -> 2D (16K)
    ax.scatter([1.88], [21.87], color=C_RED, marker="^", s=36, edgecolor="black", linewidth=0.5, zorder=7)
    ax.scatter([3.56], [43.91], color=C_AMBER, marker="D", s=32, edgecolor="black", linewidth=0.5, zorder=7)
    ax.scatter([6.40], [106.83], color=C_PURPLE, marker="*", s=58, edgecolor="black", linewidth=0.45, zorder=8)
    ax.annotate("", xy=(3.56, 43.91), xytext=(1.88, 21.87),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.95))
    ax.annotate("", xy=(6.40, 106.83), xytext=(3.56, 43.91),
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=0.95))
    ax.annotate("1D 8K @ 512N\n($M_{\\mathrm{loc}}$=4, 44.8 TF/s)", xy=(1.88, 21.87), xytext=(1.45, 7.2),
                fontsize=5.4, color=C_RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.6))
    ax.annotate("2D 8K @ 512N\n($M_{\\mathrm{loc}}$=8, 89.9 TF/s)", xy=(3.56, 43.91), xytext=(0.66, 46.0),
                fontsize=5.4, color="#9A4D00", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#9A4D00", lw=0.6))
    ax.annotate("2D 16K @ 512N\n($M_{\\mathrm{loc}}$=16, 218.8 TF/s)", xy=(6.40, 106.83), xytext=(0.68, 175.0),
                fontsize=5.5, color=C_PURPLE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=0.65))

    ax.set_xlim(0.62, 66)
    ax.set_ylim(6.0, 2950)
    ax.set_xlabel(r"Operational Arithmetic Intensity $\mathcal{I}_{\mathrm{DMA}}$ (FLOPs/Byte)")
    ax.set_ylabel("Sustained FP64 Performance (GFLOPS)")
    ax.grid(True, which="both", linestyle="--", alpha=0.38, zorder=0)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94,
              edgecolor="#CCCCCC", borderpad=0.22, labelspacing=0.18, fontsize=5.75)

    out_path = os.path.join(OUT_DIR, "fig5_roofline_model.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print("Saved:", out_path)


# ==============================================================================
# DATASET 3 & FIG. 6: 1 to 512 Nodes Strong Scaling (实验结果.xlsx)
# ==============================================================================
nodes       = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
procs       = nodes * 4
dsp_cores   = procs * 24

t_kern_1d   = np.array([4.5354, 2.3148, 1.2773, 0.6532, 0.3629, 0.3319, 0.1222, 0.1017, 0.0984, 0.0982])
t_kern_2d   = np.array([5.7812, 2.8848, 1.4777, 0.7639, 0.3877, 0.2133, 0.1284, 0.0647, 0.0538, 0.0489])
tflops_1d   = np.array([0.97, 1.90, 3.44, 6.73, 12.12, 13.25, 36.00, 43.24, 44.70, 44.78])
tflops_2d   = np.array([0.76, 1.52, 2.98, 5.76, 11.34, 20.62, 34.26, 68.01, 81.77, 89.92])
eff_1d      = np.array([100.0, 98.0, 88.75, 86.75, 78.12, 42.72, 58.02, 34.84, 18.01,  9.02])
eff_2d      = np.array([100.0, 100.0, 97.75, 94.62, 93.19, 84.72, 70.36, 69.84, 41.98, 23.09])

t_comm_1d   = np.array([9.0148, 13.9390, 18.0232, 18.8795, 20.2766, 24.2608, 32.2132, 26.9565, 36.0212, 102.2459])
t_comm_2d   = np.array([2.8916,  5.0928,  7.1000,  7.6984,  9.7332,  8.3339, 10.2557,  8.7471, 10.4307,  10.3231])
t_e2e_1d    = np.array([14.4051, 16.5016, 19.6237, 19.5836, 20.7317, 24.6883, 32.3580, 30.1049, 36.1225, 102.3500])
t_e2e_2d    = np.array([10.4492,  9.0629,  9.3791,  8.5446, 10.1578,  8.5675, 10.4476,  8.8499, 10.4872,  10.3819])

t_io_1d     = np.array([4.9328, 2.3293, 2.2159, 4.0208, 3.9320, 3.9662, 3.8780, 3.7947, 3.9273, 1.8929])
t_mio_2d    = np.array([1.7782, 1.2621, 0.8289, 0.3816, 0.1992, 0.1156, 0.3060, 0.0406, 4.0284, 3.0916])

t_lio_2d    = np.array([2.5993, 2.4597, 2.6256, 0.9160, 2.5751, 0.8924, 2.6963, 0.7940, 0.8873, 0.9292])
t_lcomm_2d  = np.array([3.0886, 4.0690, 6.4899, 6.2871, 8.2855, 7.3434, 9.1613, 7.5485, 8.4671, 8.8500])
t_lcomb_2d  = np.array([4.4771, 5.7975, 7.5522, 7.2037, 8.9763, 8.1901, 9.6323, 7.8068, 9.0925, 9.7596])


def _fmt_cores(c):
    if c < 1000:
        return str(c)
    if c < 10000:
        return f"{c / 1000.0:.1f}K"
    return f"{int(round(c / 1000.0))}K"


def make_fig6_strong_scaling():
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.92))
    plt.subplots_adjust(wspace=0.27, hspace=0.46)
    x = np.arange(len(nodes))
    x_labels = [f"{n}\n({_fmt_cores(c)})" for n, c in zip(nodes, dsp_cores)]

    # --- Panel (a): Aggregate Kernel Throughput (TFLOPS) ---
    ax = axes[0, 0]
    ax.axvspan(6.5, 9.35, color="#EEF2F6", alpha=0.95, zorder=0)
    ax.text(7.92, 4.5, "DDR-Streaming\nFloor Regime\n($M_{\\mathrm{loc}} < M_b$)",
            ha="center", va="bottom", fontsize=5.65, color="#4A5568", style="italic")

    ax.plot(x, tflops_2d, marker="o", color=C_NAVY, linewidth=1.65,
            label="Proposed 2D Hybrid ($P/2 \\times 2$)", zorder=5)
    ax.plot(x, tflops_1d, marker="s", color=C_RED, linewidth=1.4, linestyle="--",
            label="1D Flat Global Broadcast", zorder=4)

    ax.annotate("89.92 TFLOPS\n(118.2x vs 1N, 2.01x vs 1D)", xy=(9, 89.92), xytext=(1.6, 73.0),
                fontsize=5.9, fontweight="bold", color=C_NAVY,
                arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=0.8))
    ax.annotate("1D Floor:\n44.78 TFLOPS", xy=(9, 44.78), xytext=(6.65, 25.5),
                fontsize=5.75, fontweight="bold", color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.75))

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=6.1)
    ax.set_xlabel("Compute Nodes (Total DSP Cores)", labelpad=3)
    ax.set_ylabel("Aggregate Kernel Throughput (TFLOPS)")
    ax.set_ylim(0, 114)
    ax.grid(True, linestyle="--", alpha=0.45, zorder=1)
    ax.set_title("(a) Aggregate DSP Kernel Throughput Scaling", fontweight="bold", pad=5)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94,
              edgecolor="#CCCCCC", fontsize=6.0)

    # --- Panel (b): Kernel Strong-Scaling Parallel Efficiency (%) ---
    ax = axes[0, 1]
    ax.plot(x, eff_2d, marker="o", color=C_NAVY, linewidth=1.65,
            label="Proposed 2D Hybrid Efficiency", zorder=5)
    ax.plot(x, eff_1d, marker="s", color=C_RED, linewidth=1.4, linestyle="--",
            label="1D Flat Baseline Efficiency", zorder=4)

    ax.annotate("93.2% @ 16N\n(1,536 Cores)", xy=(4, 93.19), xytext=(1.3, 66.0),
                fontsize=5.9, fontweight="bold", color=C_NAVY,
                arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=0.75))
    ax.annotate("69.8% vs 34.8%\n(2.00x @ 128N)", xy=(7, 69.84), xytext=(5.0, 91.0),
                fontsize=5.9, fontweight="bold", color=C_NAVY,
                arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=0.75))

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=6.1)
    ax.set_xlabel("Compute Nodes (Total DSP Cores)", labelpad=3)
    ax.set_ylabel("Kernel Parallel Efficiency (%)")
    ax.set_ylim(0, 114)
    ax.grid(True, linestyle="--", alpha=0.45, zorder=1)
    ax.set_title("(b) Strong-Scaling Parallel Efficiency ($8192^3$)", fontweight="bold", pad=5)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.94,
              edgecolor="#CCCCCC", fontsize=6.0)

    # --- Panel (c): MPI Communication & End-to-End Time ---
    ax = axes[1, 0]
    ax.plot(x, t_e2e_1d, marker="^", color=C_RED, linewidth=1.55, linestyle="-",
            label="1D End-to-End Total Time", zorder=5)
    ax.plot(x, t_comm_1d, marker="s", color="#E67E22", linewidth=1.25, linestyle=":",
            label="1D Pure MPI Broadcast Time", zorder=4)
    ax.plot(x, t_e2e_2d, marker="o", color=C_NAVY, linewidth=1.65, linestyle="-",
            label="Proposed 2D End-to-End Time", zorder=6)
    ax.plot(x, t_comm_2d, marker="d", color=C_TEAL, linewidth=1.25, linestyle="--",
            label="Proposed 2D Group Comm Time", zorder=4)

    ax.annotate("512N Comm Storm:\n102.25 s Comm / 102.35 s E2E", xy=(9, 102.35), xytext=(4.2, 74.0),
                fontsize=5.95, fontweight="bold", color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.8))
    ax.annotate("10.38 s E2E (9.86x Speedup)\n10.32 s Comm (9.90x Speedup)", xy=(9, 10.38), xytext=(3.6, 44.0),
                fontsize=5.85, fontweight="bold", color=C_NAVY,
                arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=6.1)
    ax.set_xlabel("Compute Nodes (Total DSP Cores)", labelpad=3)
    ax.set_ylabel("Wall-Clock Latency (s)")
    ax.set_ylim(0, 126)
    ax.grid(True, linestyle="--", alpha=0.45, zorder=1)
    ax.set_title("(c) Elimination of 512-Node Communication Storm", fontweight="bold", pad=5)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94,
              edgecolor="#CCCCCC", fontsize=5.85, labelspacing=0.16)

    # --- Panel (d): Parallel Sliced pread I/O & Leader Comm-I/O Overlap ---
    ax = axes[1, 1]
    w = 0.34
    unoverlapped_sum = t_lio_2d + t_lcomm_2d
    ax.bar(x - w/2, unoverlapped_sum, width=w, color="#D5D8DC", edgecolor="#566573",
           linewidth=0.5, label="Leader Unoverlapped (I/O + Comm)", zorder=2)
    ax.bar(x + w/2, t_lcomb_2d, width=w, color=C_TEAL, edgecolor="black",
           linewidth=0.5, label="Leader Actual Overlapped Wall Time", zorder=3)
    ax.plot(x, t_mio_2d, marker="o", color=C_AMBER, linewidth=1.4,
            label="2D Member Sliced pread I/O", zorder=5)
    ax.plot(x, t_io_1d, marker="x", color=C_RED, linewidth=1.15, linestyle="--",
            label="1D Root Serialized Read I/O", zorder=4)

    ax.annotate("2.23 s Hidden\n(82.5% of I/O)", xy=(6 - w/2, 11.86), xytext=(6.2, 13.3),
                fontsize=5.7, fontweight="bold", color=C_TEAL,
                arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=0.7))
    ax.annotate("0.041 s pread\n(93.5x vs 1D)", xy=(7, 0.0406), xytext=(3.1, 1.9),
                fontsize=5.65, fontweight="bold", color="#9A4D00",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=C_AMBER, lw=0.55, alpha=0.94),
                arrowprops=dict(arrowstyle="->", color="#9A4D00", lw=0.7), zorder=8)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=6.1)
    ax.set_xlabel("Compute Nodes (Total DSP Cores)", labelpad=3)
    ax.set_ylabel("I/O & Overlapped Latency (s)")
    ax.set_ylim(0, 16.5)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=1)
    ax.set_title("(d) Sliced Parallel pread & Leader Comm-I/O Overlap", fontweight="bold", pad=5)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94,
              edgecolor="#CCCCCC", fontsize=5.65, labelspacing=0.14)

    out_path = os.path.join(OUT_DIR, "fig6_strong_scaling_512nodes.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print("Saved:", out_path)


# ==============================================================================
# DATASETS 4, 5, 6 & FIG. 7: DMA Buffering Ablation, 16384^3 Extreme Strong
# Scaling (218.79 TFLOPS), and 1-512 Node Weak Scaling (92.21% Efficiency)
# ==============================================================================
# Experiment 6: DMA Buffering Ablation (s_opt vs d_opt vs pipeline_opt)
abl_scales_96 = ["512", "1K", "2K", "4K", "8K", "16K"]
gf_96_s   = np.array([222.17, 462.10, 501.07, 613.76, 783.47, 838.03])
gf_96_d   = np.array([227.34, 467.17, 512.10, 632.00, 895.84, 956.17])
gf_96_tri = np.array([228.85, 465.81, 512.07, 630.40, 893.32, 953.54])

sp_1c_d   = np.array([1.068, 1.070, 1.138, 1.171, 1.422])  # 512..8192
sp_96_d   = np.array([1.023, 1.011, 1.022, 1.030, 1.143, 1.141])

# Experiment 5: 16384^3 Extreme-Scale Strong Scaling (1 to 512 Nodes, 35.184 TFLOPs)
t_kern_16k  = np.array([35.779546, 15.862078, 9.810461, 5.023064, 2.595670,
                        1.300730,  0.711197, 0.422338, 0.210917, 0.160816])
tflops_16k  = np.array([0.983, 2.218, 3.586, 7.005, 13.555,
                        27.050, 49.472, 83.309, 166.816, 218.787])
eff_16k     = np.array([100.00, 112.80, 91.17, 89.04, 86.15,
                        85.96, 78.61, 66.19, 66.26, 43.45])

# Experiment 4: 1 to 512 Nodes Weak Scaling (512^3, 1024^3, 2048^3)
weak_2k_tri = np.array([100.00, 96.45, 89.78, 93.21, 93.10, 92.55, 92.80, 92.81, 92.59, 92.21])
weak_2k_dbl = np.array([100.00, 96.45, 89.74, 93.56, 93.77, 92.66, 93.00, 92.89, 92.67, 89.64])
weak_1k_tri = np.array([100.00, 92.62, 88.36, 90.75, 83.55, 82.17, 77.00, 89.46, 81.55, 80.07])
weak_512_tri = np.array([100.00, 94.15, 98.86, 91.20, 90.77, 88.68, 53.97, 87.51, 61.74, 55.39])


def make_fig7_ablation_extreme_weak():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.38))
    plt.subplots_adjust(wspace=0.55)

    # --- Panel (a): DMA Buffering Ablation (Single vs Double vs Tri-Buffered) ---
    ax = axes[0]
    xa = np.arange(len(abl_scales_96))
    w = 0.25
    ax.bar(xa - w, gf_96_s, width=w, color="#BDC3C7", edgecolor="#4A5568",
           linewidth=0.45, label="96C Single-Buf (s_opt)", zorder=2)
    ax.bar(xa, gf_96_d, width=w, color=C_TEAL, edgecolor="black",
           linewidth=0.45, label="96C Double-Buf (d_opt)", zorder=3)
    ax.bar(xa + w, gf_96_tri, width=w, color=C_NAVY, edgecolor="black",
           linewidth=0.45, label="96C Tri-Buf (pipeline_opt)", zorder=3)

    ax.set_xticks(xa)
    ax.set_xticklabels(abl_scales_96, fontsize=6.2)
    ax.set_xlabel("Matrix Scale $N$ ($M=N=K$)")
    ax.set_ylabel("1-Node (96 DSPs) GFLOPS", labelpad=2)
    ax.set_ylim(0, 1480)
    ax.grid(axis="y", linestyle="--", alpha=0.42, zorder=0)
    ax.set_title("(a) Scratchpad DMA Buffering Ablation", fontweight="bold", pad=5, fontsize=7.5)

    ax_r = ax.twinx()
    ax_r.plot(xa[:5], sp_1c_d, color=C_RED, marker="o", markersize=3.5, linewidth=1.25,
              linestyle="-", label="1-Core Async Gain (1.42x)", zorder=6)
    ax_r.plot(xa, sp_96_d, color=C_AMBER, marker="s", markersize=3.2, linewidth=1.1,
              linestyle="--", label="96-Core Async Gain (1.14x)", zorder=5)
    ax_r.set_ylabel("Async Speedup vs. Single-Buf", color=C_RED, fontsize=6.6, labelpad=2)
    ax_r.tick_params(axis="y", labelcolor=C_RED, labelsize=6.0, pad=1.5)
    ax_r.set_ylim(0.95, 1.68)

    ax_r.annotate("1C: 1.42x\n(-29.7% Stall)", xy=(4, 1.422), xytext=(1.65, 1.31),
                  fontsize=5.2, fontweight="bold", color=C_RED, ha="center",
                  arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.65))
    ax.annotate("956.2 GF/s\n(1.14x)", xy=(5, 956.17), xytext=(4.65, 1145),
                fontsize=5.2, fontweight="bold", color=C_NAVY, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=0.65))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, facecolor="white",
              framealpha=0.94, edgecolor="#CCCCCC", fontsize=4.95, borderpad=0.18,
              labelspacing=0.12, handlelength=1.4, handletextpad=0.35)

    # --- Panel (b): Extreme-Scale 16384^3 vs 8192^3 Strong Scaling ---
    ax = axes[1]
    xb = np.arange(len(nodes))
    xb_lbl = ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512"]
    ax.plot(xb, tflops_16k, marker="o", markersize=3.6, color=C_PURPLE, linewidth=1.55,
            label="2D $16384^3$ TFLOPS (Ours)", zorder=5)
    ax.plot(xb, tflops_2d, marker="s", markersize=3.2, color=C_NAVY, linewidth=1.25,
            label="2D $8192^3$ TFLOPS (Ours)", zorder=4)
    ax.plot(xb, tflops_1d, marker="^", markersize=3.0, color=C_RED, linewidth=1.05, linestyle=":",
            label="1D $8192^3$ Floor (44.8 TF/s)", zorder=3)

    ax.set_xticks(xb)
    ax.set_xticklabels(xb_lbl, fontsize=5.8)
    ax.set_xlabel("Compute Nodes (Up to 49,152 DSPs)")
    ax.set_ylabel("Sustained Kernel TFLOPS", labelpad=2)
    ax.set_ylim(0, 265)
    ax.grid(True, linestyle="--", alpha=0.42, zorder=0)
    ax.set_title("(b) Extreme-Scale $16384^3$ Strong Scaling", fontweight="bold", pad=5, fontsize=7.5)

    ax_b2 = ax.twinx()
    ax_b2.plot(xb, eff_16k, marker="d", markersize=2.9, color=C_GREEN, linewidth=1.1, linestyle="--",
               label="$16384^3$ Parallel Eff. (%)", zorder=4)
    ax_b2.set_ylabel("Parallel Efficiency (%)", color=C_GREEN, fontsize=6.6, labelpad=2)
    ax_b2.tick_params(axis="y", labelcolor=C_GREEN, labelsize=6.0, pad=1.5)
    ax_b2.set_ylim(0, 132)

    ax.annotate("218.79 TFLOPS\n(222.5x, 0.161 s)", xy=(9, 218.79), xytext=(3.65, 182.0),
                fontsize=5.3, fontweight="bold", color=C_PURPLE,
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=0.65))
    ax_b2.annotate("112.8% Superlinear\n(2.26x @ 2N)", xy=(1, 112.8), xytext=(1.75, 115.5),
                   fontsize=5.15, fontweight="bold", color=C_GREEN,
                   arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=0.65))

    hb1, lb1 = ax.get_legend_handles_labels()
    hb2, lb2 = ax_b2.get_legend_handles_labels()
    ax.legend(hb1 + hb2, lb1 + lb2, loc="center left", bbox_to_anchor=(0.0, 0.50),
              frameon=True, facecolor="white", framealpha=0.94, edgecolor="#CCCCCC",
              fontsize=4.95, borderpad=0.18, labelspacing=0.13, handlelength=1.4, handletextpad=0.35)

    # --- Panel (c): 1 to 512 Nodes Weak Scaling (96 to 49,152 DSP Cores) ---
    ax = axes[2]
    ax.axhline(100.0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)
    ax.plot(xb, weak_2k_tri, marker="o", markersize=3.5, color=C_NAVY, linewidth=1.55,
            label="$2048^3$ Tri-Buf (92.2% @ 512N)", zorder=6)
    ax.plot(xb, weak_2k_dbl, marker="x", markersize=3.4, color=C_TEAL, linewidth=1.1, linestyle="--",
            label="$2048^3$ Double-Buf (89.6% @ 512N)", zorder=5)
    ax.plot(xb, weak_1k_tri, marker="s", markersize=3.2, color=C_AMBER, linewidth=1.25,
            label="$1024^3$ Tri-Buf (80.1% @ 512N)", zorder=4)
    ax.plot(xb, weak_512_tri, marker="^", markersize=3.1, color=C_RED, linewidth=1.1, linestyle="-.",
            label="$512^3$ Tri-Buf (87.5% @ 128N)", zorder=3)

    ax.annotate("92.21% @ 512N\n(67.24 TF/s, +2.8% vs Dbl)", xy=(9, 92.21), xytext=(2.2, 99.0),
                fontsize=5.25, fontweight="bold", color=C_NAVY,
                arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=0.65))
    ax.annotate("80.07% @ 512N\n(0.079 s)", xy=(9, 80.07), xytext=(4.6, 68.5),
                fontsize=5.25, fontweight="bold", color="#9A4D00",
                arrowprops=dict(arrowstyle="->", color="#9A4D00", lw=0.65))

    ax.set_xticks(xb)
    ax.set_xticklabels(xb_lbl, fontsize=5.8)
    ax.set_xlabel("Compute Nodes (Up to 49,152 DSPs)")
    ax.set_ylabel("Weak-Scaling Efficiency (%)", labelpad=2)
    ax.set_ylim(42, 110)
    ax.grid(True, linestyle="--", alpha=0.42, zorder=0)
    ax.set_title("(c) 512-Node Weak-Scaling Efficiency", fontweight="bold", pad=5, fontsize=7.5)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.94,
              edgecolor="#CCCCCC", fontsize=4.95, borderpad=0.18, labelspacing=0.13,
              handlelength=1.4, handletextpad=0.35)

    out_path = os.path.join(OUT_DIR, "fig7_ablation_extreme_weak.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    make_fig1_arch_overview()
    make_fig2_2d_comm_timeline()
    make_fig3_tri_pipeline_microkernel()
    make_fig4_intra_node()
    make_fig5_roofline()
    make_fig6_strong_scaling()
    make_fig7_ablation_extreme_weak()
