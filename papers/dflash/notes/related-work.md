# External sources reviewed — cutoff 2026-10-27

Primary papers read beyond abstracts where needed:
2211.17192v2 exact sampling;2503.01840v3 EAGLE-3;2503.09573v3 blockdiffusion;2504.18583v4 PARD;2507.11851 futurefeatures;2510.02358v1 DiffuSpec;2511.00606v2 SpecDiff-2.
Later:2605.29707v1 Domino;2607.05147v1 DSpark;2610.07659v2 DLoop;2610.13708v1 DHiSparse.
Official blogs: IncoAI DFlash2 Aug18;vLLMAMD Aug23.

Evidence boundaries:
- EAGLE-3 officialup-to6.5×, not universal2–3ceiling.
- PARD own3.67× andSpecDiff-2own5.5×; differconditions.
- Domino5.49Transformers/5.8SGL; no commonmatrix withDFlash.
- DSparkproduction60–85%per-user speed atmatchedthroughput; no DFlashTable1ratio.
- DFlash2meanτ4.92→5.97 (+21.34%);selector+2M/.6%cycle,conv+16.5M/.7%;officialblog not paper v2.
- DLoopwithin-own-tableQ8 DFlash4.15→4.71 (+13.49%speed),τ6.08→7.40;own8benchmarks differfromoriginal7.
- DHiSparseGLM5.3/100K/C128793outputtok/s/GPU1.45HiSparse;notdenseQwenbenchmark.

Source links and full citations in index.md. No universalSOTA rank without same hardware/model/workload/precision/sampling/scheduler.
