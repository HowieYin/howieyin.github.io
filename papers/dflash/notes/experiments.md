# Experiment audit

800K target-regenerated NemotronV2/CodeAlpaca samples; ablation100K.
6epochs AdamW6e−4, clip1, cosine warmup.04; seq3072/Coder4096;512anchors.
Five target features; draft5/Coder8; B16/LLaMA10.
H200 default; singleB200 serving. Table3FA4/Spec-v2 overlap; Table5FlashInfer/Spec-v1 no overlap for both DFlash and EAGLE.

Recalculated from source:
- Table1 Q3-8B T0:4.86/2.02=2.406 (EAGLE60);4.86/1.76=2.761(EAGLE16).
- T1:4.03/1.88=2.144;4.03/1.68=2.399.
- Figure1 compares tree60; caption >2.5overall is overbroad.
- Table3Q8mathC1:1175/230=5.109;C32:16076/5694=2.823.
- Table4longcontext onlyτ; no measured longcontextspeedup.
- Table5GSMC1 DFlashτ4.32<EAGLE60τ4.55 but2.4>1.9speed.
- Table6 mean3L3.657,5L3.673,8L3.610; chat3Lfastest, uncertainty unknown.
- Table7 feature3→5 offline payload+66.7%.
- Table8 bigtrain→smalltest works better than reverse;35.7%fullaccept text-only, no histogram asset.
- Table9 everycellτ/speedup; sameKV AR GSM4.8/2.4 vs diffusion4.2/3.3.
- Table10 nofeature3.73mathT0; don't claim strict2–3cap.
- Table11 Q3.5-9Bmath MTP1.7 vs D3.5;τ6.7vs7.3.
- Table12 vLLMchatC32only1.3speed.
- Table13 math+13.56%speed,+14.17%τ;code+18.54%speed.

Missing: errorbars, full per-tableprecision/batchsettings, actual trainingtokens/GPUhours, explicit answerqualityregression, peakmemory/per-optrace, unified diffusionbaseline retraining, p99latency.
Local mechanistic model test passes; no localGPU reproduction or claim of remeasured benchmark speed.
