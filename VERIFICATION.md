# VERIFICATION.md — Automated Build, Font, Page-Budget & Data Audit Harness

Every modification to `论文初稿/` or `gemm_mpi/` **must** pass all applicable verification gates below before committing or opening/updating a Pull Request.

---

## Gate 1: Vector Figure Regeneration & Font Embedding Check

Run `generate_figures.py` and verify that all **7 vector PDFs** (`fig1`–`fig7`) exist and embed **zero `Type 3` bitmap fonts**:

```powershell
$env:Path += ";C:\Users\Administrator\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
python "论文初稿\generate_figures.py"

Get-ChildItem "论文初稿\figures\fig*.pdf" | ForEach-Object {
    $bad = pdffonts $_.FullName | Select-String "Type 3| no "
    Write-Output "$($_.Name): BadFonts=$($bad.Count)"
}
```
* **Pass Criterion**: All 7 files (`fig1_arch_overview.pdf` through `fig7_ablation_extreme_weak.pdf`) output `BadFonts=0`.

---

## Gate 2: Four-Variant LaTeX Build, Zero-Warning & Page-Budget Gate

Both `paper_conference.tex` and `bare_jrnl_new_sample4.tex` must compile cleanly in **both** `\anonymoustrue` (default double-blind) and `\anonymousfalse` (camera-ready / single-blind) modes, and `bare_jrnl_new_sample4_v2.txt` must stay synchronized with `bare_jrnl_new_sample4.tex`:

```powershell
$env:Path += ";C:\Users\Administrator\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
Push-Location "论文初稿"

# Sync plain-text mirror
Copy-Item bare_jrnl_new_sample4.tex bare_jrnl_new_sample4_v2.txt -Force

foreach ($stem in @("paper_conference", "bare_jrnl_new_sample4")) {
    pdflatex -interaction=nonstopmode "$stem.tex" | Out-Null
    bibtex $stem | Out-Null
    pdflatex -interaction=nonstopmode "$stem.tex" | Out-Null
    $log = pdflatex -interaction=nonstopmode "$stem.tex"
    $issues = Select-String -Path "$stem.log" -Pattern "LaTeX Warning|Package .* Warning|Overfull|Undefined"
    $badFonts = pdffonts "$stem.pdf" | Select-String "Type 3| no "
    $pages = (pdfinfo "$stem.pdf" | Select-String "^Pages:").ToString().Trim()
    Write-Output "[$stem] $pages | Warnings/Overfull=$($issues.Count) | BadFonts=$($badFonts.Count)"
}
Pop-Location
```

### Camera-Ready (`\anonymousfalse`) Shadow Test
Before committing manuscript edits, copy `paper_conference.tex` and `bare_jrnl_new_sample4.tex` to a temporary scratch folder (outside the repo), replace `\anonymoustrue` with `\anonymousfalse`, compile 4 passes, and verify:
1. `paper_conference.pdf` still has **11 total pages**, with `Section VII (Conclusion)` ending strictly on **Page 10** and `References` starting at the top of **Page 11**.
2. `Warnings/Overfull=0` and `BadFonts=0` in both camera-ready builds.
3. Confirm `\anonymoustrue` remains restored on line 9 of both `.tex` files in the working tree.

---

## Gate 3: 300-DPI Visual Layout & Overlap Inspection

Whenever figure layouts (`generate_figures.py`), tables (`Tables I–V`), or float placements change, rasterize the affected pages/figures via `pdftocairo` into a temporary directory outside the repository and visually inspect every page:

```powershell
$env:Path += ";C:\Users\Administrator\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
New-Item -ItemType Directory -Force -Path "$env:TEMP\pp_paper_audit" | Out-Null
pdftocairo -png -r 150 "论文初稿\paper_conference.pdf" "$env:TEMP\pp_paper_audit\conf_p"
pdftocairo -png -r 150 "论文初稿\bare_jrnl_new_sample4.pdf" "$env:TEMP\pp_paper_audit\jrnl_p"
```
* **Visual Checklist**:
  - [ ] Float numbering appears in strict sequential order (`Tables I–V`, `Figs. 1–7`).
  - [ ] Dual-axis (`twinx`) subplots in `Fig. 4`, `Fig. 6`, and `Fig. 7` have zero colliding y-axis titles, legends, or data callouts.
  - [ ] `paper_conference.pdf` Page 10 ends cleanly at `Section VII` without trailing orphan lines spilling onto Page 11; Page 11 displays two balanced Reference columns (`[1]–[13]` left, `[14]–[25]` right).
  - [ ] `bare_jrnl_new_sample4.pdf` final page displays two balanced columns at the bottom margin.

---

## Gate 4: Canonical Metric Cross-Check Matrix

Any edit touching performance numbers in Abstract, Introduction, Section IV-D, Section V, or Conclusion must preserve exact agreement with these canonical anchor values:

| Metric Anchor | Canonical Value | Source File & Location |
| :--- | :--- | :--- |
| Single DSP Core Peak Sustained (`pipeline_opt`) | `49.19 GFLOPS` (`64.1%` of `76.8 GFLOPS` dual-FMA peak; `6.12x–7.28x` vs. 1C OpenBLAS) | `gemm_mpi/benchmark/openblas_vs_asd.log` ($N=4096$) |
| Single 24-Core Cluster Peak Sustained | `243.44 GFLOPS` (`~100%` of `240.5 GFLOPS` DDR4 Roofline; `9.33x` vs. 4C, `6.57x` vs. 16C OpenBLAS) | `gemm_mpi/benchmark/openblas_vs_asd.log` ($N=16384$) |
| FP64 Element-Wise Infinity-Norm Error | $\le 2.18 \times 10^{-8}$ across $N \in [512, 16384]$ (`2.70e-10` at $N=512$) | `gemm_mpi/benchmark/openblas_vs_asd.log` |
| DMA Ablation (`s_opt` $\to$ `d_opt` / `pipeline_opt`) | 1C: `33.79` $\to$ `48.05 GFLOPS` (`1.422x`, `-29.69%` time); 24C: `196.53` $\to$ `224.79 GFLOPS` (`1.144x`); 96C Full Node ($N=16384$): `838.03` $\to$ `956.17 GFLOPS` (`1.141x`, `99.4%` node Roofline) | `实验结果.xlsx` (`实验6_双缓冲与流水线优化`) |
| 512-Node (49,152 DSPs) $N=8192$ 2D Strong Scaling | Comm: `102.25 s` $\to$ `10.32 s` (`9.90x`; end-to-end `9.86x`); Kernel: `89.92 TFLOPS` (`0.0489 s`, `118.20x` vs. 1N, `2.01x` vs. 1D `44.78 TFLOPS`); Member `pread`: `0.0406 s` @ 128N (`93.5x`), `82.5%` leader I/O hidden @ 64N | `实验结果.xlsx` (`Sheet1`–`Sheet4`) |
| 512-Node (49,152 DSPs) $N=16384$ Extreme Strong Scaling | 2N: `2.218 TFLOPS` (`2.256x` superlinear, `112.80%` eff.); 256N: `166.82 TFLOPS` (`66.26%` eff.); 512N: **`218.79 TFLOPS`** (`0.1608 s`, `222.49x` speedup, `2.9%` vs. `212.3 TFLOPS` Roofline theorem); Member `pread`: `0.1117–0.1235 s` (`37.9x`) | `实验结果.xlsx` (`实验5_更大矩阵规模16384强扩展`) |
| 1–512 Node Multi-Granularity Weak Scaling | $2048^3$/node @ 512N: `pipeline_opt` **`92.21%` eff. (`67.24 TFLOPS`, `0.5233 s`)** vs. `d_opt` `89.64%` (`65.38 TFLOPS`, `+2.57` pp / `+2.77%` jitter resilience); $1024^3$ @ 512N: `80.07%`; $512^3$ @ 128N: `87.51%` | `实验结果.xlsx` (`实验4_弱扩展实验`) |
