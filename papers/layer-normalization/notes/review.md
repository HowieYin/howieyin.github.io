# Review record — 2026-10-27

## Pass 1 — formatting / readability

- Rendered the final HTML in Chrome156 at 375,390,430,1440,1920 CSS pixels. Desktop article880px; mobile343/358/398px. No whole-document horizontal overflow.
- Inspected desktop/mobile screenshots of top, mechanism, table, wide Fisher formula and ending. Retained6 paperfigures+3 tableimages all loaded; source captions intact.
- Fixed matrix rowspacing `[1.5]` exposed by LaTeXML; restored dropped standalone vertical ellipses. SourceTeX preserved. Fixed inline r/rho/dagger table symbols.
- Mobile figure/controls use readable sizes, localized horizontal scrolling for formulas/tables/interactiveSVG. Added focusable regions; real keyboard horizontal scrolling confirmed.
- Responsive Contents expands desktop/collapses mobile; current-section marker follows scroll. Mouse presets/zoom/checkbox and range keyboard checked. Select state updates checked with input events.
- No console/runtime errors; three no-JS text fallbacks, native MathML, reduced-motion verified. No auto-playing animation.
- OriginalMermaid sources retained and three corresponding SVG flow diagrams rendered. Feishu counterpart includes three nativeMermaid boards,9 images,8 full tables,35 numberedformula labels.

## Pass 2 — source accuracy

- Read supplied14-pagev1 with supplement. Original metadata,6 figures,3 tables,35 equations enumerated against PDF/TeX.
- Corrected extraction so Eq8–14 and17–31 contain their own numberedrows rather than duplicated wholeequationgroups. Eq6 retains aligned precedingrow.
- Recalculated OE R1+1.9pp/+4.08%,image+1.1pp/+2.91%; samebudget skip-thoughtMSE−.021/−7.05%,MR+2.2pp/+2.85%; DRAW−.27nats/−.328%.
- Preserved unreportedR5blanks,all originalbold/notes,†differentbudget,figureunitsandcurvecolors. No digitized inventedfinalscores.
- Gain1/bias0 and eps1e−5 checked in authorcode. LSTMconcat4H/GRUconcat2H;18H/12H/DRAW2H addedparams checked.
- Scopequalified positiveδ,epsilon,inactivationcommonshift vsrawinputshift,FisherGLM,validationvstest,NLLvsbound.
- Contemporarysources fixedcommits and datedcurrentupdate; Qwen3.8officialchecked2026-10-27. No universalnormSOTA or end-to-endscorecausalnormclaim.
- Numericmodelverifiedpopulationstats,axes,sign,shift,epsilon,constants,finite-differencebackward and IOexamples. No empiricaltrainingorhardwareperformancereproductionclaimed.

## Publishing checks

- StaticHTML architecture retained. Only newarticle/home/archive/Transformerrelatedlink changed.
- LocalbrowserandstaticQA evidence saved in sources. GitHubPages and productionQA verification recorded after deployment in local publishing report.
