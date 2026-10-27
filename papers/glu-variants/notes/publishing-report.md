# Publishing report — GLU Variants Improve Transformer

Reading: https://howieyin.github.io/papers/glu-variants/

Source: 2002.05202v1.pdf; research cutoff 2026-10-27.

Completed: original five-page source and LaTeX, six equation groups, all four original tables and captions, chapter-ordered explanation, tensor/gradient/budget/IO reconstruction, experiment audit, historical and current primary-source research, two review passes.

Local and production Chrome rendering QA passed at 375 / 390 / 430 / 1440 / 1920px. Article widths: 343 / 358 / 398 / 880 / 880px; document width equals viewport at each size. Four original screenshots decoded, six original equation groups rendered; fonts loaded.

Real mouse: negative-gate preset / carried sign flip / budget reset / original-table zoom / mobile TOC anchor. Real keyboard: gate range Home / ArrowRight and zoom Escape. Select updates tested with synthetic input events. Complete GLUE table programmatically scrolled 600px in a 356px viewport across 983px content. No-JS gate and budget equivalents present; reduced motion supported; no automatic playback.

After merging the latest website updates, 187 local resources / links / anchors resolved. Repository content uses plain static HTML with no local framework build command. GitHub Pages runs the existing managed Jekyll build and deployment; both passed.

Numerical QA: all X/W/V/W2 analytic gradients verified by float64 central differences, max absolute error<5.2e-10; packed / unpacked projection difference2.22e-16; shared widget arithmetic and original table values / bold counts verified.

Publishing status: published and verified on 2026-10-27.

- Content commit: `6fe662b`; integration/deployment commit: `6d3653fcc3026f6f050b0f91578a66a816b1a72a`.
- Latest remote readings preserved through a normal merge; no force push or deployment setting changes.
- Pages build, status reporting and deploy: success. Workflow: https://github.com/HowieYin/howieyin.github.io/actions/runs/40488588631
- Production page returned HTTP 200. CSS, JavaScript, local fonts, four original images, six equation groups, three generated SVG diagrams and both explorers rendered successfully. Console errors: 0; failed resource responses: 0.
- Screenshots inspected for desktop hero, tensor/gradient reconstruction and experiments, and mobile interaction and research tables. Original-table zoom, TOC active anchor and full-table scroll passed.
- Home `/`, archive `/papers/`, and Transformer backlink returned HTTP 200 and contain the new reading. Archive retains all 15 readings present at verification.
- Detailed machine-readable results: `production-render-qa.json`. Full production screenshots retained with the local research artifacts.

No model training or GPU/HTP latency benchmark was performed; arithmetic, logical IO and literature results retain their stated scope.
