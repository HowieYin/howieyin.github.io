# Two review passes — Llama3 v3

## Pass 1: formatting and readability

- Chapter order follows PDF§1–10, with the extended research and personal interpretations afterwards.
- Full30figures/38panels; captions/interpretations in text outside images; original author PDF/PNG retained and credited.
- Full35numberedtables/36panels; table35has two panels. All source row counts, cell text and colspan/rowspan checked against originalHTML. Bold/italic/underline remain classes.
- All3originaldisplayequationinstances retained (CI appears twice), plus6explicitly agent-reconstructed formula groups.
- Two Mermaid diagrams rendered as SVG/PNG; source .mmd retained. Three meaningful interactive components.
- Fixed arXiv rotation-wrapper widths causing mobile document overflow; removed unsupported`\rotate` macro artifacts while retaining model names.
- Local5width browserQA375/390/430/1440/1920: all expected contents, images, formulas, anchors, dark/light, zoom/Escape, wide-table scroll and interaction arithmetic passed. Screenshots visually inspected for top, long table, CP and classification component.
- Local server socket backlog fixed for simultaneous image fetch; failures were local preview serving, not missing assets.

## Pass 2: primary-source accuracy

- PDF92pages, v3date2024-11-23 vs paperdate2024-07-23; actualLlama3.1 results. TeXappendices not included in specifiedPDF.
- Checked fullsource/Table3 architecture, Table4 configuration arithmetic, GQA cache and alltablecells/spans.
- Found/corrected draft speechPM error: sixattentionheads/hidden864, not sixlayers or unsupported500Mparameter claim.
- Found scaling discrepancy: reportedA.29/alpha.53 at3.8e25 computes10.46365T vs author16.55Tprediction; both preserved with unknownfullcoefficients/normalization.
- Preserved252M/40Mtoken quantities with doubts; Table5counts419/percent94.9% and faultyGPU35.32%vs30.1%, GPUcategory63.96%vs58.7%.
- Distinguished Table21EM/F1/ROUGE/recall, En.QA40530.5<70B36.7; originalGPT4EnMC72.0vsTable2 72.1.
- Checked FP8 row-bound/format convention via datedofficialkernel snapshot; quantattention/firstlastskip, H100requestconditions and RMproxy limits.
- Checked int8safety per-language recall drops; AUC/TPR/FPR and unprovidedcounts/base rate. Interactive counts are explicitly analytic expectations.
- Officialfollowups verifiedLlama3.2 beforev3revision, Llama4totalvsactive, OLMo3fullflow, DeepSeekV3route, latest2026MetaMuseWonderclaim. Emptyresearchblog and estimatedaggregator scores excluded from quantitativeSOTA.
- No localGPU/NPUmodeltraining/benchmark is claimed.

## Release validation

Feishu readback and online presentation assessment passed; requested account has full_access and edit/manage permissions. GitHub Pages build/deploy succeeded. Production five-width browser QA and navigation/source checks passed. See publishing-report.md and the adjacent JSON evidence.
