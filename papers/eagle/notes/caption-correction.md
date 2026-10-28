# Caption correction · 2026-10-28

Request: keep explanatory captions outside images as selectable text.

All nine original figures, eight PDF tables and Algorithm 1 were recropped from the unchanged v3 PDF at 4 pixels per PDF point. The image files contain the plots, diagrams, table data or algorithm body; complete original captions are rendered separately as HTML/MathML text. Figure legends, axes, module labels, units and table data remain in the images.

Figure 1's full upper legend was restored. Table 2's unrelated trailing body paragraph was removed. Each crop was checked against the caption's PDF coordinates; all 18 crops have zero caption overlap. The bounds and image hashes are recorded in `caption-crop-audit.json`.

The page presents ten visible figure/algorithm captions and eight visible table captions. The optional PDF table views also retain their separate full captions. Chinese explanations remain immediately beside each visual. Caption text is selectable, including its inline mathematical symbols.

Local real Chrome QA at 375, 390, 430, 1440 and 1920 px passed: no horizontal page overflow, all 18 images loaded, all visible captions and eight formula rows rendered, all table columns remained available, and image zoom/Escape worked. The three analytical interactions, keyboard slider, reduced motion and three no-JavaScript fallbacks also passed. No runtime errors or non-canceled network failures occurred; navigation cancellations during reload were recorded separately.

Feishu mirror: https://bytedance.larkoffice.com/docx/WG8fdNptKoLukIxhNjYcOiTzn7g

All 18 mirror images were replaced and read back at revision 18. Their dimensions match the corrected assets, and their full captions are visible text outside the image resources. Existing Chinese explanations, formula rows and Mermaid boards were preserved.

Website: https://howieyin.github.io/papers/eagle/

Production validation is recorded separately after deployment; the local checks above are not a claim of production validation.
