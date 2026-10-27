# EAGLE-3 — source-grounded research dossier

Research date: 2026-10-27. Requested version: arXiv 2503.01840v3, 2025-04-23, 12 PDF pages. Authors: Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang. Affiliations: Peking University, Microsoft Research, University of Waterloo, Vector Institute. First arXiv release 2025-03-03. Official repository and NeurIPS proceedings confirm NeurIPS 2025; do not replace requested v3 with the later camera-ready.

## Primary structure and coverage

Abstract; §1 Introduction; §2 Preliminaries (§2.1 Speculative Sampling, §2.2 EAGLE and EAGLE-2); §3 EAGLE-3 (§3.1 Inference Pipeline, §3.2 Draft Model Training); §4 Experiments (§4.1 Effectiveness, §4.2 Ablation Study, §4.3 SGLang, §4.4 vLLM); §5 Related Work; §6 Conclusion; acknowledgements; references; Appendix A Implementation Details. Preserve chapter order in the main narrative. Seven figures, five tables, no numbered equations. The two sampling expressions in §2.1 are original; other math is an explicitly labeled reconstruction.

## Mechanism

EAGLE1/2 optimize feature regression plus token prediction. Regressing a teacher feature supports recurrent feedback but constrains a small drafter to a particular representation. Removing regression alone raises first-step acceptance but makes self-feedback distribution mismatch severe. Training-time test explicitly rolls the drafter's unconstrained hidden output back into the drafter over multiple rounds. It does not mean running inference benchmarks only during training or modifying the target weights. Removing feature regression also permits fusing low/middle/high teacher features. Fusion g=W_f[l;m;h], [B,S,3k]→[B,S,k]. Feedback a has k dimensions but need not equal any teacher feature. One-token-advanced embedding tells the model the actual sampled token.

Figure5 prefix How can: target prefill produces root I but has no teacher feature g_I yet. Draft uses g_can and e_I→a_I→do; next draft uses a_I and e_do→a_do→it. Paper wording “I has not been checked” means it has not been fed through target as an input, not that a target-sampled root is untrusted. §3.1 says input to draft “Step1” in its Step2 sentence; label as textual indexing typo.

Standard stochastic sampler uses p=target,q=proposal: accept min(1,p(x)/q(x)); reject samples normalized positive p−q; keep accepted prefix; discard rejected suffix; when all proposals pass add a target bonus. Distribution equivalence is exact under correct algorithm; not bitwise equality under identical seed, not a guarantee for arbitrary numerical kernels or altered thresholds. Tree proposal logic requires its own valid sequential conditional sampler, not blindly applying a chain q to an arbitrary tree.

Tree mask allows committed prefix, self and ancestors only, not siblings. EAGLE2 builds context-dependent trees using draft confidence. EAGLE3 reuses tree algorithm. Training mask has dense lower-triangular original-key block, diagonal blocks for generated rounds; branches at different original anchors must remain isolated. Native original attention and a previous simulated-round key use one joint softmax: do not separately normalize dense and diagonal pieces then sum.

## Mathematical caveats

Full column rank W maps hidden state injectively to complete logits, but this does not prove top states contain only next-token information, nor make logits-to-softmax injective. Treat richer intermediate features as empirical motivation, not a theorem.

For a chain with conditional acceptance α_i, E[K]=Σ_{i=1..d}Π_{j=1..i}α_j; expected progress including target root/correction/bonus is 1+E[K], barring truncation. The paper τ describes tokens generated/cycle and loosely equates accepted drafts; official update adds accept_length+1. State counting convention before using speedup. Flat conditional α does not imply flat probability an entire chain survives. Avoid reading plots as fitted power-law exponents.

Analytical S≈τ·t_AR/(T_draft+T_verify+T_misc); throughput and walltime denominators vary by framework. A speedup ratio increase S3/S2−1 differs from elapsed-time reduction 1−S2/S3.

## Complete results and audits

Table1 has 24 method/model rows: greedy 14, temperature1 10; 28 total HTML rows including two header rows and two temperature dividers. Six dataset/mean groups × speedup and τ, plus model/method. V13B greedy E2 mean4.22 τ4.83, E3 5.51 τ6.62; L31 8B 3.23/4.11→4.44/6.23; L33 70B2.85/3.78→4.12/5.88; DSL8B3.26/3.92→4.16/5.84. Temperature1 V13B3.76/4.41→4.65/5.67; L31 2.80/3.62→3.45/4.92; L33 2.65/3.54→3.95/5.66; DSL2.77/3.54→3.52/4.89. Max6.47 speedup and τ7.54 occur V13B HumanEval greedy. DSL8B greedy CNN/DM E3 speed3.52 < E2 3.53 despite higher τ4.92>3.33: directly refutes the prose “highest speedup on all tasks.” One hundredth is not a statistically meaningful regression without variance, but claim must be scoped. Greedy L33 mean gain44.56%; temp1 L33 mean49.06%, so prose20–40% is an approximation that doesn't cover all rows.

Table2 E2 MT3.16/4.05,GSM3.39/4.24; remove fea con3.82/5.37,3.77/5.22; plus fusion4.40/6.13,4.48/6.23. Sequential rather than full-factorial ablation; first change combines free state and rollout training. No isolation of each component, layer selection or order effects.

Table3 H100 singleGPU SGLang0.4.4,L31 8B,MTbench,chain3. Batch2,4,8,16,24,32,48,56,64. E1 [1.40,1.38,1.23,1.02,.93,.94,.88,.99,.99]; E3[1.81,1.82,1.62,1.48,1.39,1.32,1.38,1.34,1.38]. E1, NOT E2. Author/SGLang measured; current docs repeat same values, not independent replication.

Table4 H100 batch1: vanilla158.34,E2 244.10,E3 373.25 tokens/s. Derived2.3573× vanilla,1.5291×E2. Inverse aggregate bs1 throughput ≈6.3168,4.097,2.6792ms/token; not per-token TPOT distribution or TTFT measurement.

Table5 body RTX3090 but caption A100; preserve contradiction. vLLM version not specified. chain2. Batch2,4,8,16,24,32,48,56. E1[1.30,1.25,1.21,1.10,1.03,.93,.82,.71]; E3[1.75,1.68,1.58,1.49,1.42,1.36,1.21,1.01]. Body says “maximum throughput improvement at24/56” but numerical ratio max at2; interpret largest tested batch retaining >1×, not optimum throughput. Table has ratios so aggregate tokens/s optimum cannot be inferred.

Setup AdamW betas(.9,.95),clip.5,lr5e−5. ShareGPT≈68K entries,UltraChat200K≈464K entries, teacher regenerates responses; reasoning adds OpenThoughts114k-math. 8× is data-entry approximate532/68=7.82, not a declared precise training-token budget. Hardware/precision/training epochs/compute budget/main Table1 measurement protocol and per-row uncertainty incomplete in v3. Max405B/671B not evaluated dueGPU constraints. No quality benchmarks reported because strict verifier intended to preserve target distribution. No TP/FP/TN/FN taxonomy suitable for this task; explain acceptance metrics instead.

Appendix native HF+PyTorch with preallocated KV; PLD/Lookahead/Medusa/Hydra default weights/settings. E1 Llama3 trainedShareGPT. E2 candidate nodes60(8B),50(13B),48(70B),depth6,top10 expansions; E3 same total nodes,depth8. Therefore increased training data AND changed tree depth vs baseline; not compute-controlled isolate of objective. No original latency per-op, energy, TTFT, long-context sweep or confidence interval.

## Official code audit

SafeAILab/EAGLE main SHA cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b (current inspected, NOT proven original benchmark commit). Files in /tmp/eagle3-research: cnets.py,ea_model.py,utils.py,train-cnets.py,train-main.py,train-config.json,modeling_llama_kv.py,train-modeling_llama_kv.py.

Feature hooks before blocks idx2, L//2, L−3: L32 gives hidden states after2,16,29 completed blocks (not outputs of zero-based2,16,29). Only selected tuple then final state may also returned; first three used.

Inference cnets Model fc3k→k; separately RMSNorm embedding/state then concatenate2k; q/k/v Linear input2k directly. Paper describes extra2k→k FC; that standalone FC is commented in code. Q outk,K/Voutkvheads×head_dim, O k→k, SwiGLU intermediate14336. residual carries state. Norm before lm_head; unconstrained raw a fed back. Separate trainable head [Vd,k], draft vocab32K vs target128256 and fixed embedding from target; remaps d2t,t2d. Do not claim shared frozen target LMHead in current code.

Training self.length7; target and embedding frozen; CE against target soft distribution restricted to draft vocab; mask excludes teacher argmax outside subset. Sum0.8^r losses; vloss list unused, no feature term. hidden_states=hidden_states_out without detach; future loss backpropagates recurrent state. Target features no_grad. Token IDs, logits and masks shifted left each round, rather than feeding sampled predicted token IDs: rollout simulates state exposure while token embeddings follow shifted teacher data. Native dense QK plus same-position generated Q·K; combined softmax; sparse diagonal avoids full anchor crossproducts.

Parameter sizing must be clearly config-derived, not paper result. k4096,hq32,hkv8,dh128,m14336,V128256,Vd32000. Fusion3k²=50,331,648; Q2k²=33,554,432; eachK/V2k×1024=8,388,608;O k²16,777,216; MLP3km176,160,768; headkVd131,072,000; embeddingkV525,336,576; norms count inspect. Active trainable excluding embedding~424.7M; unique loaded draftinclcopiedembedding~950M (~1.77GiB FP16) unless shared; decoder-only differs. KVdraft2 B S hkv dh bytes one layer vs targetL layers, but temporary tree KV and feature buffers increase peak. Fusion feature raw3BSkbytes, one fused BSk; do not require retaining all layers. No unconditional hardware or NPU speed prediction.

## Extension evidence and links

1. Original https://arxiv.org/abs/2503.01840v3 and PDF/source; DOI10.48550/arXiv.2503.01840.
2. Venue https://openreview.net/forum?id=4exx1hUffq and https://proceedings.neurips.cc/paper_files/paper/2025/file/c7b5a35ea98b62512a869c19ea7b03cb-Paper-Conference.pdf verified discovery; venue DOI not invented.
3. Official repo https://github.com/SafeAILab/EAGLE ; repo project link https://sites.google.com/view/eagle-llm (not yet fetched).
4. Exact stochastic foundation https://arxiv.org/abs/2211.17192 (2022preprint,ICML2023oral); https://arxiv.org/abs/2302.01318 complementary.
5. EAGLE1 https://arxiv.org/abs/2401.15077 Jan26 2024; v3Mar4 2025;ICML2024 official repo.
6. EAGLE2 https://arxiv.org/abs/2406.16858 Jun24 2024;EMNLP2024.
7. HASS https://arxiv.org/abs/2408.15766 Aug28 2024;ICLR2025, feature loss retained with context alignment.
8. FRSpec https://arxiv.org/abs/2502.14856 Feb20 2025; large-vocabulary headcost; author75%headcostreduction,1.12×overE2 not endtoend75%.
9. DFlash https://arxiv.org/abs/2602.06036v2 May28 2026(firstFeb5);ICML2026; fullHTML /tmp/eagle3-research/dflash.html; local12?PDF in paper/2602.06036v2.pdf. Single-pass block diffusion, target-context KV injection, typically5layers,16block vs E3 one-layer serial. Read relevant tables and hardware before reporting2.5×; this is author's setup-specific result not universal or old6.47 denominator. Project https://dflash.z-lab.ai/ ; code https://github.com/z-lab/dflash .
10. SpecForge https://github.com/sgl-project/SpecForge; README saved; EAGLE repo recommends since2025-07-23. Current training tools reduce feature extraction overhead; no claim replicated our GPU training.
11. SGLang https://docs.sglang.ai/advanced_features/speculative_decoding.html (redirectdocs.sglang.io/docs/advanced_features/speculative_decoding) includes E3,E2,MTP,DFlash,adaptive etc; default recommendation not global SOTA. Table4 values repeated.
12. Speculators https://vllm-project.github.io/2026/05/28/speculators-v050.html ; RedHat authorblog, online/offline hidden extraction nativevLLM, DFlash support, >=vLLM0.20; no new numeric claim without image reading.
13. EAGLE4 https://blog.eagle.ai/blog/introducing-eagle-4 Oct20 2026; official repo https://github.com/eagle-hq/EAGLE-4 createdOct20, licenseNOASSERTION/sourceavailable not assumeApache; PDF downloaded /tmp/eagle3-research/eagle4.pdf and text eagle4.txt. Paper title “One Flexible Drafter for All Batch Sizes and Hardware”. New efficient lowrank candidate head + exactcandidate scores, runtime hardware/batch-specific costprofiling and dynamic tree (EAGLEHarness). MatchedQwen3.8-27B4×B300TP4,T0,SPEED-Bench qualitative, engineMegaKerneldisabled: vanilla240.5,DSpark618.6,DFlash2 672.6,E4 863.8tok/s atC1; C512aggregate21203/14229/16641/28340. E4/DFlash2=1.2843 C1,1.7030 C512. Avoid generalSOTA “better thanDFlash”; version2 and engine differ from ICMLDFlash.
14. E4 CPUauthorreport 4bitGemma4E2Bit llama.cpp AWSc5.xlarge4vCPU8GB: E4mean2.71×; phone Snapdragon8Elite,Razr70Ultra,6threads,HumanEval22.8→82.6tok/s,cool,USBenergy356→166mJ/tokenlongform; no NPU/QNN implication. Memorydraft134MB,totalextraresident.27GBserver/.26GBphone; inspect paper details before cite. This is latestauthorrelease not peerreviewed confirmed replication.

## Delivery and QA decisions

User explicitly requests AGENT.md + PUBLISHING.md and publication to Paper Readings; follow existing site's /papers/eagle-3/ route instead of hypothetical /paper-readings/. Isolated clone /tmp/eagle3-research/site is clean main. Website pure staticHTML local fonts, dark colors, English navigation, Chinese body. README preview pythonhttpserver; Pages legacy Jekyll main root, no localbuild script. Reuse shared assets/article-navigation.{css,js} and localmathML rather than introduce CDN dependencies. Workspace finaldir /Users/bytedance/project/read_papers/paper-readings/eagle-3. Save intermediate scripts/cache/screenshots outside workspace; durable final research/review/QA summary and required supporting sources may be in notes/sources. Need update homepage+papersarchive+related reading backlinks (FlashAttention,GQA appropriate). User authorized commit+push by publishing request. Do not create Feishu instead of requested website or stall for permission.

Need two distinct reviews: formatting/render and sourceaccuracy. Desktop1440/1920,mobile375/390/430; inspect top/method/experiments/bottom; original7figures+5tables, fulltableHTML, math,code,interactive,zoom,Escape/focus,TOC,anchors,resource/console, noJS/reducedmotion. Online repeat after Pages build/deploy success and index/backlinks check. Original visual assets must preserve full subplots, original tableheaders,rowlabels,bold,etc. Generated visuals must labeled and not replace originals. Proposed interactions: branch mask explorer with teacher-key triangle/generated-key diagonals; observed result selector using exactTable1/Table3 values; chain acceptance/cost calculator analytic demonstration explicitlynotGPUbenchmark. No need anatomy physical metaphor or unnecessary inlinewidget because deliverablepage is requested.
