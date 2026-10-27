# Two-pass Review · 2026-10-27

## Pass 1 · Formatting / Readability

- Built the final static HTML with native MathML and local fonts; checked article, original visual inventory, chapter order, captions, tables, source disclosures, SVG/Mermaid sources and no-JS alternatives.
- Real Chrome checked 375/390/430/1440/1920 px. No page-level horizontal overflow. Wide tables/formulas/diagrams scroll within their own regions.
- Increased formula font size after observing small glyph heights. Removed duplicate `table-N` IDs from PDF evidence figures. Kept the site navigation in English.
- Inspected top, method, probability, cost, equations, table, code, bottom and full Figure 6 zoom screenshots; screenshots live outside publication assets.
- Exported all three actual Feishu Mermaid whiteboards to JPEG and checked their rendering. No empty placeholder boards.

## Pass 2 · Primary-source Accuracy

- Rechecked original 13-page PDF, original chapter tree, all 9 figures, 8 tables and Algorithm 1, plus fixed official v1 source.
- Corrected §2 to Preliminaries, §5 to Related Work; placed Mixtral Table 3 in §4.1 and gpt-fast in §4.2.
- Corrected Figure 4: left is speedup, right is accuracy; both x axes are epochs.
- Corrected Figure 8: rows are T=0/1 and columns are speed/τ/0-α/1-α; input ablation is not a separate loss/noise ablation.
- Expanded Figure 3 PDF crop to retain the full last caption line.
- Distinguished original soft-distribution CE from hard-token CE; q is a notation alias for original hat-p.
- Preserved Algorithm 1 original and documented scalar residual misprint; correct operation is full-vector positive part and normalization. Fixed v1 code confirms this.
- Verified conditional proposals for sampling without replacement, depth positions, ancestor masks, commit path KV and fallback feature lifecycle.
- Counted actual default draft nodes: 25 non-root, depth5, plus anchor=26; variable name “63” is not node count.
- Recomputed Table4 FP16/INT4 incremental ratios1.82/1.50, Table5 acceptance increases, Table6 relative changes; distinguished TPS reciprocal from measured TPOT.
- Corrected MT-Bench headline scope to2.13–3.07×;3.76× is HumanEval.
- Separated paper/code noise and precision settings, and high-batch tree-disabled exception.
- Checked dated EAGLE2/3, SpecBench, DFlash-v2 and EAGLE4 author sources; DFlash2 in EAGLE4 is not DFlash paper v2. No universal SOTA claim.
- All reported model performance is source-reported; no local model benchmark. Analytical widget results are explicitly labeled.

## Mathematical validation

8 meaningful checks pass: mass reconstruction; identical/disjoint support; conditional proposals including exhaustive support; sequential two-candidate exactness; ancestry/depth and cycle guard; KV/features byte accounting; measured ratios and total-cycle break-even.

## Feishu mirror

Created https://bytedance.larkoffice.com/docx/WG8fdNptKoLukIxhNjYcOiTzn7g.
Read back18 original image resources,3 whiteboards,8 formula rows and final insights. Current creator is尹宏轶/Hongyi Yin; contact search matched required email yinhongyi@bytedance.com to the authenticated open_id. Edit and manage_public checks both returned true. Figure4 correction patched in place without recreating the document.
