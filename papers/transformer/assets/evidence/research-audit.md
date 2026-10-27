# Primary research audit — 2026-10-27

Read all 15 pages of local `1706.03762v7.pdf`, including references and Attention Visualizations. Version date is 2023-08-02, initial submission 2017-06-12, venue NIPS 2017. PDF creation timestamp 2024 is not the publication date. Metadata/captions match official arXiv HTML v7.

## Structure and evidence

Abstract; 1 Introduction; 2 Background; 3 Model Architecture (3.1 stacks; 3.2 attention including 3.2.1–3; 3.3 FFN; 3.4 embedding; 3.5 positions); 4 Why Self-Attention; 5 Training (5.1–4); 6 Results (6.1–3); 7 Conclusion; References; Attention Visualizations.

Preserved 5 original figures and 4 tables, including figure/table captions, at 288 dpi from PDF. HTML tables preserved without deleting blank cells, colspans, rows or math. Table 3 blank settings inherit base, including big dk=dv=64. Fig 4 visually inspected: full head-5 panel and two its-only head-5/head-6 arcs to Law/application; all source token labels present.

## Source inconsistencies and conclusions

- Table 2 and abstract EN–FR 41.8; §6.1 prose 41.0. Do not silently reconcile. Against best single MoE 40.56, improvement is +1.24 under table value and +0.44 under prose value. Against ensemble 41.29, +0.51 or −0.29 respectively.
- EN–DE big 28.4 versus strongest ensemble ConvS2S 26.36: +2.04 BLEU (7.739% arithmetic ratio, not percentage accuracy). Base 27.3 beats same ensemble by +0.94.
- EN–FR base 38.1 does not beat 40.56 MoE; introductory “base surpasses all” phrasing pertains to EN–DE evidence, not both tasks.
- Table 2 Transformer cost cell spans both languages, one estimate per configuration. Do not invent separate EN–FR FLOP measurements. Cost estimated from wall time × GPUs × assumed sustained TFLOPS (P100 9.5), not per-op FLOP traces, energy, kernel benchmarks or TTFT/TPOT.
- Big/base costs: 2.3e19 / 3.3e18 = 6.97×. EN–DE big approximately GNMT single cost, but 3.35× less than ConvS2S ensemble. EN–FR big 1.2e20/2.3e19=5.22× less than MoE under this proxy.
- Base step 0.4s ×100k=11.11h versus stated ~12h; big 1s ×300k=3.472 days versus stated ~3.5; rounded, not an inconsistency requiring correction.
- Table 3 dev newstest2013, no checkpoint averaging; Table 2 test newstest2014, last5/20 checkpoints averaged. Dev25.8 cannot be compared as same setting with test27.3.
- Table 4 WSJ-only Transformer91.3 below Dyer91.7; semi Transformer92.7 below multi-task Luong93.0 and generative Dyer93.3. §6.3 prose “all except RNNG” overlooks Luong row; scope to same listed semi-supervised group. TP/FP/FN for labeled spans are not published; TN/accuracy are not appropriate substitute for parsing F1.
- Residual is post-LN `LayerNorm(x+Dropout(Sublayer(x)))`, ReLU FFN. Current official tensor2tensor master `transformer_base()` delegates v3→v2 with pre-LN, attention/relu dropout .1, 8000 warmup. Treat as evolved implementation, not paper exact reproduction.
- 2017 sin/cos positions are additive, relative shift is a linear rotation of coordinate pair, not a proof final QK depends only on difference or guarantees extrapolation. Table 3 learned PE25.7 vs sinusoidal25.8 only supports near-equivalence on dev.
- Table1 self-attention O(n²d) is mixing term; complete block also projections O(nd²) and FFN O(nd*dff). O(1) refers sequence-axis dependency stages per layer, not full inference. Convolution log_k(n) path presumes dilation; contiguous stack needs O(n/k).
- Attention heatmaps in Fig3–5 are selected qualitative evidence; do not claim causal head specialization without intervention.

## Method reconstruction values

6 encoder + 6 decoder layers, D512, dff2048, h8, dk=dv64; big D1024,dff4096,h16,dk=dv64, Pdrop.3 (EN–FR .1),300k steps,213M; base65M.
Approximate major base parameter matrices with shared ~37k vocabulary: V*D +6*(4D²+2D*dff)+6*(8D²+2D*dff)=62.98M; rounding/vocabulary/bias/config differences mean not exact reconciliation to65M.
Source-target lengths ns,nt independent; cross attention Q:[B,h,nt,dk],K:[B,h,ns,dk],V:[B,h,ns,dv],A:[B,h,nt,ns].
Training target `[BOS,y1,…,y(m−1)]` predicts `[y1,…,ym]` under j≤i causal mask; all outputs computed in parallel from gold prefix. Inference unknown prefixes remain serial.
KV extension (engineering, not paper measured): self cache bytes=2*Ld*B*T*hKV*dhead*bytes; original base6layers,T4096,8heads,64dim,FP16=48MiB; cross for ns512=6MiB. FP16 is scenario, no claim paper used mixed precision.
Dense materialized per-layer A base h8,T4096,FP16=256MiB; FlashAttention removes materialization, not quadratic exact compute or autoregression.
Encoder block FLOPs (multiply+add=2): projections8*n*D²; mixing4*n²*D; FFN4*n*D*dff. n512,D512,dff2048 =1.074G +.537G +2.147G=3.758G. Excludes softmax,LN,dropout,vocabulary head,backward.

## External primary-source findings

Successful retrieval URLs, dates and content hashes are preserved in the source manifest. Raw HTML/PDF/code and parsed text were collected during research; final figures, tables and source evidence accompany the reading.
Bahdanau2014 alignment removes fixed-vector bottleneck but RNN serial remains; ConvS2S2017 training parallel with convolutional paths; BERT2018 bidirectional encoder; T52019 encoder-decoder text-to-text; GPT32020 autoregressive decoder scaling; MQA2019 KV sharing; GQA2023 intermediate KV-head count; RoFormer2021 Q/K rotation; FlashAttention2022 exact IO-aware tiled attention; Mamba22024 state-space duality alternative; DeepSeek-V32024 MLA+MoE not current SOTA assertion.

As of2026-10-27, official WMT26 Findings at https://www2.statmt.org/wmt26/pdf/2026.wmt-1.48.pdf:
- §6.4 Wayfinder best overall, top statistical cluster17/23; Gemini3.1Pro15/23; TRIVE14/23; Lumen13/23 when compared against all systems.
- Constrained Lumen top cluster20/23; 14B Qwen3 backbone from AppendixB.13; final weights released; constrained limit total<20B and permissive research licensing (including teacher distillation).
- English→German §6.2 Wayfinder79.1/198docs, TRIVE78.5/198, Gemini3.1Pro78.2/198, GPT5.5 77.1/131; all rank range1–4; no significant strict unique winner. Dynamic evaluation yields differing assessed counts.
- Wayfinder uses API-based ensemble/agent pipeline, parameter count not disclosed; cannot compare as single 65M/213M model.
- North Small Translate official arXiv2609.13916v1 Table1/4 218Btotal/25Bactive, self-reported WMT26 GEMBA83.6 (agentic84.4), LLM judge; distinct from official human cESA ranking. Not percentage translation accuracy. This is deliberate example of why aggregator leaderboard “83.6% WMT26 #1” should not be copied.
- Modern held-out tasks/domains, instructions, data budgets, size and metrics differ from WMT2014. No defensible single worldwide Transformer SOTA ranking from these sources.

## Final research and rendering verification

All five original figures were inspected in full. All four HTML table structures preserve source cells and spans, with bold compared against the PDF. Eleven MathML expressions rendered successfully. Independent attention/causal/cache/FLOPs checks passed; local Chromium QA at 375, 390, 430, 1440 and 1920 px passed figure loading, table and formula overflow, interaction, keyboard, animation, reduced-motion, no-JavaScript and resource checks. Two review passes and supplemental Feishu resource/permission verification are documented separately. Production deployment verification is recorded in the publishing report.
