# Two review passes — 2026-10-27

## Source-grounded accuracy
Primary five-page v1 PDF and full LaTeX source checked in source order.
- Six original equation groups preserved; separate derived forward/gradient forms labeled Interpretation.
- Original visual inventory: zero Figures, four Tables, no appendix.
- All four tables' complete row/column values equal the extracted source data, and original bold counts are 4/15/13/2. All four original screenshot crops visually inspected, with full captions and no cut-off headers.
- Architecture verified against paper and original T5 gin chain: encoder12/decoder12, D768,H03072,H2048. FFN-only count113,246,208 is not total model count.
- Long/short steps, batch, lengths, optimizer, dropout, joint fine-tuning, frozen embeddings and task-best dev checkpoint protocol retained. Precision marked not reported.
- Improvements recalculated; GEGLU best pretraining, ReGLU best GLUE aggregate, SwiGLU best SuperGLUE aggregate, Bilinear best SQuAD EM and ReGLU best F1.
- Parenthetical short-run statistic not defined; visible columns cannot reconstruct reported aggregates; old Raffel standard deviations do not establish this study's significance.
- 2026 follow-ups bounded by model/data/hardware/precision/M/sparsity conditions; no universal SOTA declaration.
- Numeric check: all X/W/V/W2 gradient entries centre-differenced; maximum errors below 5.2e-10. Packed versus unpacked error2.22e-16. Widget model parameter/FLOPs/capacity/rounding/gate examples passed.

## Formatting/readability
- Core narrative follows original §1→§2→§3.1→§3.2→§3.3→§4; extensions after conclusion.
- Metadata, one-sentence/paragraph/five-point summaries, mathematical notation and tensor definitions included.
- Four source tables reflow without deleting columns; author bold retained; original screenshots available in expandable zoom view.
- Three static SVG diagrams have explicit generated attribution, Mermaid source and prose equivalents.
- Gate/budget widgets describe purpose, controls, values and limits; no automatic motion; no-JS fallback.
- Local MathML, CSS and small JS avoid external rendering dependencies.
- Responsive real-render checks, console/resources/zoom/navigation and production follow-up recorded in publishing-report.md after completion.
