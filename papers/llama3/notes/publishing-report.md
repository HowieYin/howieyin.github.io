# Llama 3 v3 — Publication record

Paper: The Llama 3 Herd of Models · 2407.21783v3 · 92 pages.

Published reading: https://howieyin.github.io/papers/llama3/

Feishu mirror: https://bytedance.larkoffice.com/docx/VCl1dEIupoYKhFx2BYccgpagn0C

## Content and local checks

Chinese chapter-ordered reading covers PDF §1–10, with 30 original figures (38 panels), 35 full numbered tables (36 panels), all three original display-equation instances, six explicitly reconstructed derivations, two Mermaid diagrams and three interactive explainers.

Two review passes cover readability and primary-source accuracy. Source discrepancies are preserved and explained. Related research is dated 2026-10-29. No local GPU/NPU training or model benchmark is claimed.

Local and offline browser checks passed at 375, 390, 430, 1440 and 1920 pixels: images, formulas, table scrolling, TOC, dark/light themes, zoom, Escape and interaction arithmetic. JSON and screenshot evidence is retained locally.

## Release verification

Feishu readback passed: 40 images (38 original panels plus two explanatory diagrams), 40 tables (36 original panels plus four explanatory/metadata tables), all 471 submitted formula expressions and both Mermaid sources. The service additionally converted 15 existing superscript footnote markers into native formulas; normalized table cell contents match exactly. The end-of-document references were read back.

The requested account yinhongyi@bytedance.com matches the document owner. Collaborator readback confirms full_access; separate edit and permission-management checks both returned true.

The original site generator includes this reading across the existing six-topic catalog; concurrent publications were retained. The homepage and Training & Scaling archive include this reading; GQA, Scaling Laws and ViT link back to it.

Production verification passed on 2026-10-29.

- Content commit: 8d10d15. Integrated publication commit: 475d208.
- Verified deployed commit: 233697271563194e632afd2c90887bd54fa88b31.
- GitHub Actions: https://github.com/HowieYin/howieyin.github.io/actions/runs/40875183473 — build, deploy and status reporting succeeded.
- Production browser QA: 375/390/430/1440/1920 pixels, all passed. All 30 original figures/38 panels, 35 original tables/36 panels, nine display-formula groups, two diagrams and three interactive components were present.
- Dark/light, TOC, zoom/Escape, wide-table scroll, formula rendering, KV/CP/confusion-matrix arithmetic and base-rate changes passed without document overflow or browser errors.
- Online navigation/source checks: 23 passed, covering homepage, paper archive and category navigation at all five widths, three backlinks, the original PDF and linked audit/source files.
- Desktop/mobile screenshots visually reviewed. Local evidence is in notes/qa-screenshots; published JSON records are adjacent to this report.
- All 471 submitted Feishu formulas, 40 images and 40 tables survived readback; the online presentation assessment also passed. The requested owner has full_access, edit and management permissions.

Concurrent pushes were merged without force-pushing or replacing existing readings. A stale Pages status-report job whose deployment was already cancelled was terminated to unblock the latest build. Earlier superseded builds were cancelled by GitHub; the verified deployment above succeeded.

Only release records are updated after these checks; article HTML, CSS, JavaScript and original assets remain identical to the verified content deployment.
