# EAGLE-3 review record

Reading version: 2503.01840v3, 2025-04-23. Review date: 2026-10-27.

## Pass 1 — source accuracy

- Checked original chapter order, full 12-page PDF, TeX, appendix and bibliography.
- Preserved all 7 original figures and 5 table crops; complete table transcriptions retain headers, groups, original bold and all 24 Table 1 data rows.
- Recomputed speed-ratio changes, elapsed-time reductions, throughput reciprocals, parameter counts and KV sizing; derived results labeled as calculations.
- Kept the DSL8B/CNN greedy exception (3.53 vs 3.52), Table 5 RTX3090/A100 conflict and batch/maximum-throughput ambiguity explicit.
- Distinguished conditional acceptance from chain survival and strict distribution equivalence from bitwise outputs.
- Checked current official code at cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b. Corrected training source path to eagle/traineagle3/cnets.py and HASS bibliography title.
- Training exposes free hidden states, while later token embeddings use shifted teacher data. Joint attention softmax and no-detach state feedback checked directly.
- DFlash and EAGLE-4 comparisons labeled external, dated and setup-specific; no claim of independently measured or globally best results.

## Pass 2 — rendering and interaction

- Real Chrome rendering checked at 1440/1920 desktop and 375/390/430 mobile widths, including top, method, experiments and references screenshots.
- All 7 figures, 5 complete table transcriptions and 107 MathML expressions loaded. No page-wide horizontal overflow, missing anchors or browser/network errors.
- Fixed long inline mathematics with local scrolling and made the mobile contents tab narrow enough to avoid obscuring text.
- Verified image zoom, Escape, focus restoration, mobile contents, native no-JavaScript rendering and reduced-motion behavior.
- Verified training mask branch isolation; all 48 Table 1 E2/E3 comparisons; the DSL8B/CNN exception; analytical default 2.4696×; acceptance=0; reset controls and invalid cost input.
- Checked the archive, homepage and reciprocal FlashAttention/GQA links.
- Preserved original source figure subplots, stacking Figures 1/4 in source order with an explicit caption note.
- Standalone workspace uses `_shared`; the static Pages copy uses `shared` because the existing Jekyll pipeline excludes underscore directories. Both use local fonts, styles and MathML; no runtime CDN.
- The repository has no local framework build command. Article generation, static serving and real browser QA form the local build check; GitHub Pages performs the existing remote Jekyll build.

Production build, deployment and online screenshot verification are recorded in the final publishing report.
