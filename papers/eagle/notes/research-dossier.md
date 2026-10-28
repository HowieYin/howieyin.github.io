# EAGLE v3 研究证据

研究日期 2026-10-27。本文原文 13 页已完整阅读；PDF、arXiv HTML 与 v1 官方实现交叉核对。外部来源是独立扩展，不能补写为原论文的实验。

## 原文与元信息

- EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty，Yuhui Li / Fangyun Wei / Chao Zhang / Hongyang Zhang。
- Peking University / Microsoft Research / University of Waterloo / Vector Institute。
- 首次提交 2024-01-26，v2 2024-02-04，v3 2025-03-04。ICML 2024，PMLR 235:28935–28948，会议 2024-07-21–27；不能把 v3 日期误写成首发年。
- https://arxiv.org/pdf/2401.15077v3
- https://proceedings.mlr.press/v235/li24bt.html
- arXiv DOI 10.48550/arXiv.2401.15077 是仓储 DOI。

## 方法与公式

- feature 是 LM head 前 hidden state，不是“倒数第二个 Transformer block”。v1 ge_data.py 185–190 取 outs_big.hidden_states[-1]；ea_model.py 125–145 从 outputs[0] 取 feature。
- input F1:i 为 B×S×D；shifted tokens T2:i+1 为 B×S；embedding 同 B×S×D；cat B×S×2D；FC 2D→D；single decoder D→D；f_hat_(i+1) 经共享冻结 head 到 q_(i+2)。重点 token/feature 索引错位。
- SmoothL1(f_(i+1),Draft(T2:i+1,F1:i))；soft p_(i+2) 与 q_(i+2) cross entropy；L=Lreg+.1 Lcls；这不是 next-ground-truth-token hard-label CE。
- Paper train noise U(-.1,.1)。v1 train.py 124–131 实现 amplitude=.2×512/S，即 half-range=.1×512/S，与论文固定噪声不同；S=2048 得 ±.025。v1 当前训练 BF16、max_len2048、bs4、20 epochs；不能当成所有论文实验统一训练设置。
- Acceptance min(1,p(x)/q(x))；residual normalized [p−q]+ 为全词表向量。质量保证来自 exact verifier，不来自高 feature 精度。
- Algorithm 1 原稿错误 p←norm(max(0,p(t_i)−q_i(t_i))) 为标量；保留原图、另明确正确 vector update p←norm(max(0,p−q_i))。代码 utils.py 396–405 确为 vector subtraction。
- without replacement cnets.py 708–724 q_i(x)=q(x)/(1−sum previous candidates' q mass)，previous support zero；utils.py 398–405 zeros prior candidate support and normalizes before residual. 不可把 k 个 candidates 当独立同分布原 q。
- v1 branch commit 4a9cf3a1f6cd4a294e6d30a4e7c77cba246d7ca5。main 已是 EAGLE2/3，复现原论文选 v1。
- `mc_sim_7b_63` 实际 25 non-root draft nodes，depth counts 4/8/8/3/2，depth5；variable name 63 不是真实节点数。加 target-sampled root anchor 是26 verification entries。Figure6 10-node/3-step 是说明图，不是全部实验树；A1说5 draft forwards。
- target verification tree must mask to shared prefix+self+ancestors; pos=prefix position+depth not packed index; commit only accepted branch KV/features, discard siblings. root token sampled by target needs matching feature; at rejection fallback token initially lacks its target feature; it is evaluated next round.
- code note utils.py terminal unadjusted non-greedy sample_p uses raw Softmax, unlike prior processed branch; requires consistency test of all processors, do not extrapolate T=1 benchmark to arbitrary temperatures. Report as source-inspection concern, not measured model failure.

## 全部实验数据与结论边界

- All8tables HTML retained. Table1 T0 Llama70B HumanEval3.52 τ4.42/GSM3.03 τ3.93/Alpaca2.97 τ3.77; T1 2.92 τ3.76/2.74 τ3.58/2.65 τ3.47。
- Fig1 MT-Bench T0 Eagle Vicuna7/13/33=2.90/3.07/2.95，Llama7/13/70=2.78/3.03/3.01。Medusa corresponding1.97/1.92/1.94/1.64/1.51/1.45；Lookahead1.00标values chart sources copied reports, not full same-environment retest.
- Fig2 T1 Eagle2.13/2.32/2.40/2.22/2.68/2.67。N/A not zero or1;7B draft too costly for13B, tiny instruct mismatch at time.
- Paper sentence “as shown Figure2 and AppendixB” for alpha points actually Table2; “shown Figure4” gpt-fast actually Table4。Conclusion says MT-bench2.1–3.8 but Fig1/2 actual2.13–3.07;3.76 from HumanEval Table1, so distinguish scope error.
- 68k ShareGPT dialogues; LR3e-5 AdamW betas .9/.95 gradclip.5; trainable params .24/.37/.56/.99B at7/13/33/70B, Mixtral .28B;2–4B trainingtokens is processed exposure not necessarily2–4B unique tokens。4×A10040G1–2days70B in intro;GPU hours96–192derived not exact run.
- Table2 τ vs n-alpha: chain-measured alpha, tree τ. V7B T0τ3.94 alpha .79/.74/.72/.73/.67; Llama70 τ3.81 alpha.75/.69/.65/.64/.64。n predicted feature count not token position correctness count.
- Table3 Mixtral1.50 τ3.25 alpha.67/.62/.61/.64/.63;expert union can >2 under verification; no measured routing breakdown.
- Table4 RTX3090 Llama7 MT-Bench T0 baselineHF FP16 24.5;gpt-fast55.1FP16/106.9INT4;Eagle+gpt-fast100.2/160.4; increment1.8185FP16 and1.5005INT4。160.4/24.5=6.547 is combined quant+compile+Eagle not Eagle-alone. Decode TPS not TTFT or universal peakTPS.
- Table5 chain/tree τ V7=3.20/3.94,V13=3.23/3.98,V33=2.97/3.68,Llama7=3.00/3.62,Llama13=3.18/3.90,Llama70=3.12/3.81;relative V7+23.125%,Llama70+22.115%.
- Table6 fixed2.78τ3.62,target-generated2.88τ3.75;+.10 speed ratio,+3.597% relative,+.13τ,+3.591%relative。One Llama7 greedy benchmark insufficient general data-sensitivity claim.
- Table7 batch1/2/3/4 Vicuna7 2.90/2.87/2.65/2.76;Llama70 3.01/2.81/2.50/2.40;max throughput ratios1.97/1.99 under memory budgets. Vicuna309024G baseline maxbs8 vsEagle7;Llama4A10040G bs5 vs4;FP16. Tree disabled atbs7 to improve throughput. “all same tree”A1 inconsistent highbatch special case; explain exception.
- Missing timings hardware per every table, prompt/outputlength, rawtimingvariance, TP layout, prefill accounting, EOS handling, allkernel/softwareversions not fullyspecified; do not invent. No taskaccuracy reported;theory isn't implementationqualityverification.

## 扩展一手资料

1. Fast Inference from Transformers via Speculative Decoding, 2022preprint/ICML2023, https://arxiv.org/abs/2211.17192 . Exact acceptance/residual proof.
2. SpecInfer https://arxiv.org/abs/2305.09781 ; 2023preprint/ASPLOS2024; EAGLE acknowledges inherited tree verification, not new lossless proof.
3. Medusa https://arxiv.org/abs/2401.10774 ; contemporary multihead route; distinguish Medusa1 frozen and Medusa2 jointfinetune nonidentical targets, non-greedy thresholds not exact verifier.
4. EAGLE2 v2 https://arxiv.org/html/2406.16858v2 ; 2024-06-30, EMNLP2024 README verified; same draft weights, calibrated confidence,pathscore,dynamic expansion/rerank;20–40%overEagle1 reported.
5. EAGLE3 v3 https://arxiv.org/html/2503.01840v3 ;2025-04-23version,initialMarch2025,NeurIPS2025 README; removes feature-regloss,low/mid/highfeaturefusion,training-time autoregressivefeedback;up6.5×andSGLangbs64+1.38× authorreports.
6. SpecBench https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md ; three repeats same devices PyTorch2.5.1. A10080G/CUDA11.5 Vicuna13BFP16T0bs1 Eagle2.16/Eagle2 2.46/Eagle3 3.02 overall; meanaccepted3.64/4.43/5.71; +39.81%Eagle3/Eagle1.3090 V7 Eagle2.03,Eagle2 2.19,SAMD[E2]2.38;cannot “EAGLE currentlyfastest” from README.
7. DFlash v2 https://arxiv.org/html/2602.06036v2 ;2026-05-28version,initialFeb2026;one-passblockdiffusiondrafter,targetfeatureKV injection;Table1Qwen3-8B T0 avgEagle3(60)2.02,DFlash(16)4.86;H200,2Koutput,thinkingoff;2.406 relative. Table5 controlledsame data Llama3.1-8B SGLangSpecv1/nooverlap B200,GSMbs1Eagle3(60)1.9 τ4.55,DFlash10 2.4 τ4.32;bs32 .6 vs1.6;acceptancelength aloneinsufficient.
8. EAGLE4 official announcement2026-10-20 https://blog.eagle.ai/blog/introducing-eagle-4 ; actual released PDF https://github.com/eagle-hq/EAGLE-4/blob/main/EAGLE4_Paper.pdf downloaded/readmethod&results. No verified arXiv/venue yet.
   - Hardwarecost warmup lookup,costawarewidth/depth/verifybudget;LM head lowrankcandidate retrieval fullvocab then exactheadrows shortlisted. Arithmetic O(Dr+Vr+KD) notO(VD), topK/gather extra;fullW retained.
   - Figure2 base574us65.4%head;r512326us37.9%,r128255us17.9%;43.2/55.6%total reductions;not measured locally.
   - T>0 harness fixes sampledround depthinadvance based previousroundstop,avoidscurrenttokenselectionbias. Important lossless boundary.
   - Table2SPEEDqualitativeQwen3.8-27B 4B300TP4T0maxoutput4096sameengineMegaKerneldisabled:vanilla240.5/21203 atC1/C512,E4 863.8/28340,DFlash2 672.6/16641,DSpark618.6/14229,MTP534.5/21518;E4 speed3.59/1.34,relativeDFlash21.2843/1.7030. T1E4608.6/21005 vs vanilla234.9/20307 =>2.59/1.03. **Do not mix DFlash2 with original DFlash-v2.**
   - HumanEvalFigure3 has different complete engine/backends,not allsameengine;can only compare deployedsystems,cannotpurealgoattribution.
   - CPU4bitGemma4E2B AWSc5.xlarge4vCPU8GBllama.cppT0single: GSM17.2→49.3,Math50016.7→48.8,HumanEval16.6→45.8,MT17.2→39.3. Phone MotorolaRazr70Ultra8Elite16GB sixCPUthreadsT0cool:Human22.8→82.6,long23.3→74.7;USB356→166mJ/token53.4%reduction;notNPU,notVmHWM/dmabufpeaknotsteadythermal.
   - drafter134MB extraresidentCPU.27GBphone.26GB;nottotalprocesspeak. Headline paper count layer params excludes fusion/head/embedding.
9. Current SGLang https://docs.sglang.ai/advanced_features/speculative_decoding.html ; vLLM https://docs.vllm.ai/en/latest/features/speculative_decoding/ ;live sources saved,avoidversionoldcapabilityclaim. Selection needsexactcheckpoint,scheduler,kernel,TP,precision,concurrency,fulltargetdistribution.

## 页面实现与验收

沿用现有 dark static HTML、DM Sans / IBM Plex Mono、英文栏导航、/papers/eagle/。目标三套交互：shiftedtoken+branchmask；exact probability mass；实测行导出的成本/理想容量。全部标记 Agent-created，无编造测量。原始 PNG、caption、HTML 表、PDF 都完整。
