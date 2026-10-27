# AWQ v6 review record

Reviewed source: 2306.00978v6, 15 pages. Reading date: 2026-10-27.

## Source accuracy review

- Verified PDF title, all ten authors, affiliations, first publication / venue / v6 dates, arXiv DOI and exact chapter order. No appendix or independent training recipe in v6.
- Matched all 10 Figures and 10 Tables to original HTML and original PDF pages. Visually inspected method, numerical charts, captions and original table evidence.
- Rebuilt equation groups (1)–(5) from PDF. Corrected the HTML's corrupted “diag” in equation (4). Distinguished exact scaling identity from quantized approximation, signed versus absolute rounding residual and activation storage versus quantization.
- Verified salient channels are weight input columns. Separated observational 1% FP16 experiment from final AWQ, fake quantization from compressed export and scale search from gradient-based training.
- Audited fixed official commit d6e797a42b9ef7778de8ee2352116e0f48a78d61: alpha grid excludes 1.0, clipping exclusions, calibration defaults, cached-input order, folding/GQA guards, qweight layout, metadata/padding and GEMV/GEMM dispatch.
- Recomputed PPL / CIDEr gaps, baseline comparisons, distribution-shift column differences, throughput ratios and group storage overhead. Three storage formats have separate labels; KV remains FP16.
- Preserved source anomalies: Figure 2 versus Table 1 values, RTN OPT-1.3B values in Tables 1/3, outdated ablation pointer, Table 7 caption/columns and nonzero task degradation.
- Dated external state 2026-10-27. AAAC, Schur Replay and Humming claims are primary author reports, not local reproductions. Explicitly separated quantization solve, kernel microbenchmarks, TTFT and inference throughput.

Independent widget verification observed the expected missing-module failure before implementation. A hand-calculated toy example initially rounded 1.2/0.475 incorrectly; the exact test caught it. Corrected the example, expected output and prose to Q(1.2)=1.425 before release. Verified output-error bounds, 32-code pack/unpack roundtrip, three metadata formats, and GQA/MHA KV sizes.

## Formatting / rendering review

The release candidate follows original sections 1–6 before critique and external synthesis. Each original figure/table is adjacent to interpretation, source and caveats. Original tables preserve complete rows/columns, direction, emphasis and captions; full PDF pages back up exact layout and footnotes. Native MathML includes copyable LaTeX; generated SVG and Mermaid are labeled Agent-created.

Browser validation passed at 375/390/430/1440/1920 widths. Captured and inspected top, scaling mechanism, packing, capacity, Table 7, equation (4), and references screenshots. All 20 embedded original figure / PDF-page evidence images loaded. All 16 checked local stylesheet/script/diagram/reference routes returned HTTP 200; no console errors. Browser navigation canceled some pending lazy requests, which completed normally when explicitly decoded.

Actual render review caught unsupported `aligned` / `arg` tokens in the MathML converter. Fixed by rendering each equation row separately and converting argmin to a supported operator name; verified eight MathML rows in six equation groups contain no literal alignment/command tokens. Dense mobile diagrams now have their own horizontal scroll region instead of shrinking labels. The page itself has no horizontal overflow.

Real mouse operations verified packing stage/reset and figure zoom; Home/ArrowRight updated the scale; capacity selects verified MHA KV=2 GiB and packed-zero weights=3,636,718,750 bytes. Escape closes the modal. Collapsible mobile TOC, active desktop TOC, reduced motion and three no-JS numerical fallbacks are verified.

Feishu draft profile passed. Created and fetched the same document: 10 original images, 3 native Mermaid whiteboards, 19 complete tables (10 numbered original table groups plus explanatory comparison tables), 6 LaTeX formula groups and 14 top-level headings. Exported and visually inspected all three whiteboards; inspected the actual browser document. Resolved yinhongyi@bytedance.com to the authenticated document owner and verified `full_access`, edit=true, manage_public=true.

Production verification and commit are recorded in the final publishing report after deployment.
