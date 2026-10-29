# Experiment audit — author reports vs local verification

No GPU/NPU model training or model benchmark was run locally. All performance results are paper/vendor reports. Local checks cover source recovery, arithmetic, code contracts, diagrams and browser interactions.

## Training and comparison

- 405B dense, 15.6T tokens, 3.8e25 FLOPs; H100 80GB, TP/CP/PP/DP, sequence 8K or128K, AdamW schedule in §3.4. Betas/full data corpus not published.
- Table4: 8192→16384 short-context per-GPU throughput430→400TFLOPs/s; long-context380. MFU43/41/38 differs from >90% effective time.
- Closed baselines use different data/compute/context, some results drawn from other reports or reruns. No common training-budget causal architecture comparison.
- Standard95%Wald CI reflects test-set sampling only. Prompt formats, label orders, answer permutations, judge/seed are additional variance.

## Outcomes and boundaries

- Table2:405B MMLU87.3 vsGPT4o89.1/Sonnet89.9; GPQA51.1 vsSonnet59.4. Strong but no blanket dominance.
- Table18:HumanEval89.0→HumanEval+82.3; different test coverage changes rankings.
- Table21:metrics mixEM/F1/ROUGE/recall. En.QA40530.5<70B36.7, not monotonic by size. Needle≈100 is retrieval, not full reasoning.
- Table15:overlap threshold selected by gain; no clean-counterfactual retraining. Unavailable estimates are unknown.
- Table27:int8 multilingual/tool recall drops despite some stable/improved aggregate metrics.
- Table28:indirectTPR71.4/FPR1.0/AUC.996; counts/base rate not provided.
- Fig26:RM distribution over100Kresponses; not paired logit or rare-error accuracy.
- Fig27:4096input/256output H100 service curves; max50%prefill improvement is setup-specific.
- Table29:externalOCR baseline and multimodal total parameter count require caveats.
- Table30:WUPS/GPTjudge/subtitle conditions, max64frames temporal coverage.
- Table31:WER/CER/language coverage differ; Table33emptyresponse assumption affects safety.
- Table34/35:TNaccuracy/PMpreference different metrics; no complete firstaudio/real-time performance.

## Proposed meaningful reproduction

1. Fix held-out business truth/test data independent of generator.
2. Pair BF16→quantized converted/exported/runtime logits and per-layer outputs; measure saturation/underflow and rare OCR/numeric errors.
3. Record TTFT/TPOT plus queue, encoder, executor, collective waits; distinguish cold RPC and steady runner.
4. Measure model weights/self-KV/cross-KV/page slack/temporaries plus CPU peak and device dma-buf separately.
5. Report integerTP/FP/TN/FN, Precision/Recall/F1/FPR/Accuracy and language/capability subgroups at fixedthreshold.
