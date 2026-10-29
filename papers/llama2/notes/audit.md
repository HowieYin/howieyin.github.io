# Llama 2 v2 — two review passes

Primary: 2307.09288v2, 77 pages. Review date: 2026-10-29.

## Pass 1 — content accuracy

Cross-checked primary PDF/TeX/HTML, all 33 figure records, 52 table records and five original equation groups. Preserved caption/axis/legend contents and full native tables. No conference/DOI invented.

Corrected the initial composition of A.5: all four annotator tests belong to A.5.4; A.5.1 is annotation instruction, A.5.2 lists prohibited negative behaviors, A.5.3 is manual QA. The final reading follows this primary-source placement.

Verified: 2T per size; GQA only34/70; 7B/13B MHA; 27,540 SFT; 512 RM pairs=1024 rows; reward threshold<0.15 (strict); safety aux recall threshold0.5 separately; branch uses Safety if tagged OR low score; only final assistant loss in GAtt, history still attends; repeated version restarts from base stated by source; best-of-K is training selection, not speculative sampling.

Verified metrics: ChatGPT wins36/ties31.5/loss32.5 → ties-half51.75, ties-excluded52.55; F1(0.89,0.55)=0.67986 and confusion counts/accuracy unavailable; 70B KV4KBF16=1.25GiB, counterfactual MHA10GiB; TP8 MQA replication matches GQA physical capacity. Table16/17 context gains/costs and Table18 enlarged FFN noted. Safety100% refers to full pool, not final mixture. GPU-hours/CO2 offsets separated from wall clock and absence of emissions.

Source ambiguities retained: prose30B vs Table1 34B; Table25 maj1@1; TruthfulQA rounded relative gain; Table40/41 prose cross-reference handled by actual captions. Auxiliary safety loss formula/weight remains undisclosed rather than invented.

## Pass 2 — readability and presentation

Chapter order: Introduction→Pretraining→Fine-tuning→Safety→Discussion→Related Work→Conclusion→A.1–A.7→independent engineering/audit/current research/critique. Every original figure/table has adjacent explanation; full long samples use expandable native details. Original captions are outside images; no redrawn image replaces original evidence.

Two original Mermaid diagrams explain feedback dependencies and GAtt mask semantics; three interactive widgets explain analytic margin, GAtt and compact/replicated KV. Formula uses native MathML with original LaTeX; all source references and related readings are clickable. Navigation is English and existing site style retained.

Actual desktop/mobile local, offline and production QA evidence is retained in qa-*.json. No model training or GPU benchmark was performed; all model performance is paper-reported.
