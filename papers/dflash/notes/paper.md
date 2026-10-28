# DFlash v2 — method reconstruction

Primary evidence: arXiv 2602.06036v2 PDF13 pages, arXiv HTML and TeX.
Structure: Abstract; Introduction; Related Work; Preliminaries; Method (Inference / Training); Experiments (Instruct / Reasoning / Serving / Long context / Ablations); Conclusion; Acknowledgements / Impact; A.1–A.5.

- B=anchor+ B−1 masks; τ includes target bonus, proposal count γ=B−1.
- H_t=RMSNorm(W_c concat five target layer states). Context enters each draft layer's own K/V projections; no target-original-KV sharing.
- Q and FFN update only block tokens; attention is bidirectional within draft block, causal in target verification.
- Training context stops before anchor, blocks are mutually isolated; clean target forward does not justify seeing future states.
- Draft uses shared frozen target embedding/head. Weighted CE prioritizes early predictions; gamma here is loss decay, distinct from proposal count.
- Official snapshot deterministic argmax draft, target multinomial when T>0; prefix equality realizes delta-proposal exact sampling, not general p/q acceptance for random draft.
- Crop both target and draft cache correctly; discard bidirectional block KV and regenerate history from verified target hidden.
- Appendix projection40MiB does not include persistent draftKV or Transformer weights.

Equations: original (1)–(4), A.3 fusion, Q/K/V, projection bytes; additional prefix-survival, weighted CE and KV accounting marked Interpretation.
All5figures and13tables retained; source provenance in sources/asset-manifest.json.
