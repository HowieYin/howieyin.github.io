# DeepSeek-V2 v5 — 研究证据与数值审计基线

研究截止 2026-10-29。正文按 v5，2024-06-19；首版 2024-05-07，作者仓库首次发布 2024-05-06。技术报告，无已核验会议 venue。52 页，5 编号图（Figure 1 两个独立向量 panel），37 表，47 编号公式。所有证据已提取到 sources/extracted.json，作者 TeX 完整保留到 sources/tex。

## 源码与维度

官方 HF config/modeling 已下载，sources/config.json / modeling_deepseek.py。V2: 60 layers, hidden5120, H128, dc512, dq1536, nope128, rope64, value128, first dense width12288, remaining59 MoE, Ns2 Nr160 K6 expertwidth1536, vocab102400 (论文概述100K)。RMS norm on both compressed latent vectors; routed_scaling_factor16, no top-k renormalization. topk_group3 among n_group8 using maximum expert score per group then top6 among chosen groups. HF aux_loss_alpha.001 is inference/reference config; not paper training .003/.05/.02.

HF reference eager path expands kv_b_proj, constructs [B,H,S,192] keys and [B,H,S,128] values, then past_key_value.update caches expanded states. Thus model architecture alone doesn't promise compressed runtime cache. vLLM current source verified has DeepseekV2MLAAttention / MultiHeadLatentAttentionWrapper, use_mla selection and latent cache; obtain kernel hardware support separately.

DeepSeek-67B baseline config: https://huggingface.co/deepseek-ai/deepseek-llm-67b-base/raw/main/config.json ; 95 layers, 64 query heads, 8 KV heads (GQA), hidden8192, head128, BF16. It is dense FFN but NOT MHA. Actual baseline KV =95*2*8*128=194560 elements/token, 389120 BF16 bytes.

V2 compressed cache=(512+64)*60=34560 elements/token,69120 BF16 bytes=67.5KiB. At128Ki tokens:8.4375GiB BF16,3.1640625GiB nominal6-bit. Same baseline precision:82.2368% fewer elements vs67B. 6-bit V2 vs BF16 baseline:93.3388% fewer nominal bytes, which matches93.3% headline; don't claim exact deployed metadata/layout independently verified. Same V2 geometry MHA=2*128*128*60=1966080 elements ->98.2421875% savings. These three denominators differ; Table9 ablation is its own model configs (860.2K vs34.6K, about96%), not final V2 MHA hypothetical.

MLA absorption derivation: qC^T UK_i c = (UK_i^T qC)^T c; weighted value Σ_j a_j UV_i c_j= UV_i Σ_j a_j c_j. UK_i shape128×512, qC128, effective query512; latent aggregate512; UV_i128×512, O_i5120×128. Keep per-head weighted latent aggregate since weights differ by head. Full fused W^O_i W^UV_i shape5120×512 can grow weight memory; practical implementations may factorize query/output projections instead of storing huge fused matrices. RoPE naive q^T R_t^T R_j UK_i c_j prevents fixed absorption; separate positional64dim shared K solves. Cache contains normalized cKV512 + rotated shared kR64, not cQ and not expanded128 head copies.

Attention arithmetic one decode query (excluding projections): expanded QK+PV2H*S*(192+128)=81920S FLOPs; absorbed latent2H*S*(576+512)=278528S FLOPs,3.4× arithmetic but sharply lower cache bandwidth if reusable kernel. Not measured latency. Each expert3*d*m=23592960 weights; 2shared+6routed=188743680 active FFN weights/layer. Total expert resident162*...=3822059520 weights/MoE layer. 236B BF16 raw weight≈472GB decimal,FP8≈236GB. Active21B is not resident21B. TP can replicate shared latent cache; don't divide aggregate cache blindly by TP.

## 实验设置与边界

8.1T tokens, Chinese token count~12% greater than English (ratio1.12, not12% corpus); 4K pretrain; AdamW(.9,.95,.1),LR2.4e-4,warmup2K, step.316 at60/90% tokens,clip1;batch2304→9216 first225B tokens.16-way zero-bubble PP,8-way EP,ZeRO1,no TP, H800 NVLink/NVSwitch within8GPU node, IB across. α=.003/.05/.02. Train token drop capacity1, 10% sequences protected; eval no drops. No exact GPU count or training dtype/memory/seed coverage reported; don't invent.

YaRN s40, α1 β32 target160K; train1000 steps32K batch576; advertised/evaluated128K. sqrt(t)=.0707ln(s)+1. NIAH retrieval success does not prove all complex128K tasks.

Training172.8K vs300.6K GPUh per trillion tokens=42.515% savings; 8.1T implied1.39968M GPUh, extrapolation not independently logged total. Throughput>50K generated tokens/s and>100K inputtokens/s per8H800 node;5.76× maximum service throughput under actual67B traffic distribution, V2FP8 weights and average6bitKV, baseline exact service quantization/batch/TTFT/TPOT/p95 not reported. Can't say single-request latency5.76×.

SFT1.5M:1.2M helpful .3M safety,2epochs,LR5e-6. GRPO equations32–36 are response-level summary, not full token-level production implementation. Oldpolicy sampling, group standardized reward, clipped policy ratio, nonnegative sampledKL estimator. Zero reward variance needs epsilon/skip guard; no value given. No critic doesn't mean no reward/ref/old model memory. Reasoning RM then helpful/safety/rule mixed RM. hybridparallel, vLLM largebatch generation, CPU offload.

Headline evidence Table2: V2MMLU78.5 vsLlama70B78.9, MATH43.6 best listed, HellaSwag84.2 vs87.9 andDS67B86.3; not universal improvements. Table3 SFT→RL: HumanEval76.8→81.1,+4.3pp; LiveCodeBench28.7→32.5,+3.8pp;MATH52.7→53.9,+1.2pp; BBH81.3→79.7,-1.6pp;MMLU78.4→77.8,-.6pp;CEval80.9→78.0,-2.9pp;IFEval64.1→63.8. TriviaQA/NQ chat formatting caveat original caption, DS67BChat values borrowed previous release.

Table4 RLMTBench8.97 vsLlama8.95 numerical near-tie not confident win; AlpacaLC30.0→38.9,+8.9pp29.67%relative. Table5 overall7.74→7.91,+.17; language8.36 vsGPT4preview8.29, reasoning7.45 belowGPT4preview7.73;GPT4-0613 judge; not general human preference guarantee.

Lite27layers hidden2048 H16 dc512 no querycompression,rope64, expert1408 Ns2 Nr64 K6,15.7Btotal2.4Bactive;5.7T vsold2T,32K context; all layer experts on same device so onlyexpertbalance.001. LR4.2e-4 decay80/90%,batch4608. Table6 LiteMMLU58.3 vs45.0 and48.2 partly2.85× trainingtokens, can't isolateMLA. Table7HumanEval57.3 vs45.7 butMBPP45.8 below46.2.

Table8 7B1.33T equalapprox params by changing layercounts: MMLUMQA37.9 GQA41.2 MHA45.2; non-attention depth confound. Table9Small MHA/MLA1.33T:CMMLU52.3→53.4 butCEval51.6→50.9; large25.0→21.5Bactive,250.8→247.4Btotal,420Btraining not8.1T;MLA smallcache15.6Kvs110.6K, large34.6Kvs860.2K. Table10 low annotatoragreement420 moralitems, no controlledfiltered/unfiltered training, so does not establish causal debiasing superiority. Table11 RLcomposite83.35 belowErnie85.60/GLM84.24, accuracy84.54 higher thanErnie84.38; don't reproduce “beats all Chinese models” for every metric.

EvaluationBPBtokenizer-normalized bits/byte; multiplechoiceAcc andEM/codePass@1 cannot inventTP/FP/TN/FN;DROPF1 is answer-token overlap notclassificationF1. Full promptsTables12–37 preserved with comment on PPL-vs-generation, CoTfewshots, answerextractor and WinoGrandeprefixselection.
