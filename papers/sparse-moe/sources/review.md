# Two review passes — 2026-10-27

## Formatting and readability

Chapter order follows original §1–6 and appendices A–G before engineering/research synthesis. Full page has 22 navigable sections, 4 original figures, all 9 complete tables and 22 numbered MathML equations. Original evidence is visibly distinguished from three simplified SVG/Mermaid diagrams and two mechanism/resource interactions.

Checked rendered screenshots at desktop and mobile: hero, routing mechanism, multilingual table and current-state comparison. Original table rows/cells/bold marks compared programmatically with extracted source. Image SHA-256 matches original visual manifest. Captions are preserved as separate, copyable text outside the images. Long tables/equations/code have their own overflow containers. Fixed mobile router heading wrap and TOC resize behavior. Formula and SVG source are downloadable.

Final screenshot review identified arXiv's split Eq. (8) MathML with literal OPEN/CLOSE placeholders. Joined the two original rows inside one probability fence, preserving the original equation and raw extracted JSON. Static check verifies exactly 22 displayed math elements and no OPEN/CLOSE pseudotext. Tiny positive resource quantities retain significant digits instead of rounding to zero.

## Source-grounded content accuracy

Checked original PDF pages 1–19 against headline claims, all tables, all equation indices and notes. Corrected parameter terminology to explicitly exclude BOTH embedding and output softmax where original columns do. Added precise C.2 width/expert-count variants, original first-level branching16, output importance sampling and eos evaluation. Added matched dense baseline identities.

Preserved nuanced findings:

- LSTM backbone; Figure1 top2 vs major LM top4.
- KeepTopK precedes softmax. Renormalized top1 task-router gradient collapses within fixed selection.
- Importance, conditional smooth Load and actual counts differ. Selected/unselected thresholds use global k+1/k.
- Appendix E exact-four wording conflicts with strict batchwise F; zero coverage and noninteger quota unspecified.
- Multilingual is noisy-top-k flat512/top2/hidden8192, not single-pair hierarchy2048/top2×2/hidden2048.
- 137.7B largest model is not best perplexity; 68.9B/28.9 vs largest29.2.
- Original 18% claim vs visible-table 19.31% recalculation, 6% forward nonsoftmax arithmetic vs12.71% estimated GPU-hours.
- Historical tokenized Moses BLEU, unequal hardware/data/budget, no variance or causal proof of Korean degradation.
- Attention appendix changes score function, not equivalent algebra or Transformer scaled attention.
- Current sources dated 2026-10-27; Qwen official2026-08-26 and DeepEPV2.5 used. Benchmark protocol/scale/openness distinguished; no absolute global-SOTA assertion.

Mathematical unit checks verified TopK normalization, top1 invariance, balanced-Importance counterexample, conditional thresholds, strict-batchwise zero-coverage, expected expert assignments, expert parameters and forward-traffic arithmetic. No training or GPU performance experiments were run.

Browser and production evidence are recorded separately in sources and publishing-report.md after verification; they are not inferred from source inspection.

## Caption separation correction — 2026-10-28

Recropped all four figures and nine table screenshots directly from the original PDF. Captions are excluded from every image and appear once as text beside the corresponding figure/table. Verified PDF caption bounding boxes do not intersect the crop, PNG pixels match the original PDF render, and complete embedded chart image bounds remain inside the new crop. Figure 2 now also preserves the complete upper edge of both plots. Reviewed both visual inventories for axes, legends, table borders and all rows.
