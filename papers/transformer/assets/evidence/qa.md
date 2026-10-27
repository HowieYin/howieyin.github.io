# Final local rendering QA — 2026-10-27

Served the actual static repository with Python HTTP server; no application build tool is present or required. Chromium tested 375, 390, 430, 1440 and 1920 px widths.

Passed: page width containment, valid and unique anchors, loaded five original figures and four PDF table images, four selectable source tables, eleven MathML expressions, four custom SVG diagrams, local assets and links, current-section TOC, causal/scaling toggles, known toy outputs, keyboard range Home, cache growth/reset, KV-sharing select input, animation play/pause, figure modal and Escape, reduced-motion single-step behavior, and no-JavaScript reading fallback. Console errors and failed page/resource requests: zero.

Manual screenshot review covered desktop/mobile hero, architecture, attention, wide ablation table, position formula, cache/FLOPs and closing analysis. Long equations and tables scroll within their containers; original figures preserve white backgrounds, aspect ratio and all source content. Screenshots and detailed automation logs remain in the temporary research directory.

Four original table structures also passed a cell-by-cell comparison against official arXiv v7 HTML, including text, row spans, column spans and blank cells: 6/12/22/13 rows respectively. Bold values were compared against the PDF; the whole base training-cost number is emphasized as in the source.

Supplemental Feishu content readback found five images, four whiteboards, nine tables and chapter order. Exported all four whiteboard previews and inspected them; no blank boards or missing nodes. Current user resolved by email and has full_access, edit and manage_public.
