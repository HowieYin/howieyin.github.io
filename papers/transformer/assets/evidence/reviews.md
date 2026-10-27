# Two review passes — 2026-10-27

## Readability and presentation

Reviewed the chapter-ordered narrative from Abstract through original §1–7 and the Attention Visualizations appendix. The method defines Q/K/V, key-axis softmax, source/target shapes, right-shift and teacher forcing before extending to caches. Five original figures sit beside their explanations; four complete tables include the paper's row/column spans, blank cells and bold, with original PDF views available for comparison. Ten formula panels contain eleven selectable MathML expressions. Four additional SVG diagrams and two interactive components are explicitly marked agent-created.

Reviewed actual Chromium screenshots for desktop and mobile hero, original architecture, attention interaction, long ablation table, positional equation, cache/FLOPs and ending sections. Wide formulas and tables remain locally scrollable; page width remains constrained. Original white-background figures are not inverted or cropped. Changed TOC highlighting to track the last passed heading, so jumping into the middle of a section still identifies the active section. Corrected QA's screenshot scroll behavior to instant; the original site uses smooth scrolling.

## Primary-source accuracy

Cross-checked all fifteen PDF pages, official arXiv v7 tables and captions, selected official Tensor2Tensor routines, and dated external primary sources. Retained all four tables with their original metric units and inherited blanks. Confirmed base/big dimensions, six layers per stack, post-LN, ReLU, embedding sharing/scaling, sin/cos convention, Adam and 4000 warmup, batch token counts, checkpoint averaging, beam search and parsing-specific output length.

Kept the EN–FR 41.8 vs 41.0 conflict visible; recomputed both consequence paths. EN–DE +2.04 is BLEU points. Parsing 91.3/92.7 does not beat all listed baselines. Table 2 cost is a literature proxy and merged across the two Transformer language columns; it is not a hardware measurement. Table 3 is development-set/no-checkpoint-averaging, whereas Table 2 is test-set/averaged-weights. Attention visualizations establish observed patterns rather than causal head specialization. Position shift linearity does not guarantee content-dependent logits are relative-only or length extrapolation succeeds.

KV and FLOPs are labeled derivations: six decoder layers at 4096 tokens, eight 64-dimensional heads, two bytes gives 48 MiB self and 6 MiB cross at source length 512; one materialized A is 256 MiB. Encoder projection/mixing/FFN at n=512 sum to 3.758 GFLOPs, excluding other operations. Independent numerical model checks passed known two-token results, causal support, huge-score softmax stability, cache unit arithmetic and per-op FLOPs. Official current base preset uses pre-LN and 8000 warmup, so it is not a paper-exact reproduction.

WMT26 sources are dated and separate human cESA from self-reported GEMBA, single models from API/agent systems, and constrained from all-system top statistical clusters. No direct ranking or score improvement is claimed against WMT2014 BLEU.

## Supplemental Feishu verification

Created a chapter-ordered document with five original images, nine tables (four original plus five explanatory), four whiteboards and a final insights section. Readback confirms resource counts and source order. The XML importer reported removing an unsupported outer article wrapper while retaining content; readback confirmed all substantive content. Email resolution matched yinhongyi@bytedance.com to the current account. Member listing shows full_access; edit and manage_public checks returned true. Website publication remains the primary requested deliverable.
