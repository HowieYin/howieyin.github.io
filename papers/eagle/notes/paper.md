# Paper source and structure

2401.15077v3,13 pages,ICML2024. Full original chapter tree is retained in paper-structure.md.
The completed chapter-ordered reading is ../index.md; full primary-source dossier is research-dossier.md.
All original visuals are indexed in ../sources/visual-manifest.json, including PDF page/crop provenance and captions.

Core reconstruction: F1:i + T2:i+1 → embedding/concat → FC2D→D → single decoder → predicted f_i+1 → frozen target head → q_i+2. Target tree forward uses ancestry/depth; exact conditional verification commits only accepted path features and KV.
