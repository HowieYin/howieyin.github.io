# Research ledger

Read date / external-source cutoff: 2026-10-27. Primary artifact: 2002.05202v1.pdf, 5 pages. Full PDF and source main.tex/main.bbl preserved.

## Paper structure and provenance

Abstract → 1 Introduction (Eq1–3) → 2 Gated Linear Units and Variants (Eq4–6) → 3 Experiments on T5 → 3.1 Model Architecture → 3.2 Pre-Training and Perplexity Results → 3.3 Fine-Tuning (Tables2–4) → 4 Conclusions → References. Table1 occurs on page2 after3.1; Tables3–4 float onto page4 after conclusion but belong to3.3. No original Figure, appendix, separate supplement or dedicated project page was found in the official source. Do not fabricate absent figures.

Author: Noam Shazeer, Google. arXiv submission 2020-02-12 19:57:13 UTC; PDF title date February14,2020. No verified conference/journal venue. Identifier 2002.05202v1; arXiv DOI10.48550/arXiv.2002.05202 is repository DOI, not proof of venue.

## Verified primary claims

- Bias-free ordinary FFN: max(xW1,0)W2; GELU=xΦ(x), Swishβ=xσ(βx).
- GLU family: f(xW+b)⊗(xV+c), f=sigmoid / linear / ReLU / GELU / Swish. FFN omits biases and adds W2. SwiGLU FFN fixes β=1, does not learn β.
- W,V ∈ R[D,H], W2∈R[H,D]. X∈R[B,S,D]; flatten M=BS. Gate/up/product each M×H. Original ⊗ means elementwise product, not Kronecker product.
- Matched budget: 2D H0 =3D H, so H=2H0/3. D768, H03072, H2048 gives4,718,592 weights and9,437,184 leading GEMM FLOPs/token/FFN. Twelve encoder + twelve decoder FFNs gives113,246,208 FFN weights; total model params not reported. Official original T5 gin chain confirms num_layers12; do not “correct” to modern HF6.
- Large-M activation/materialization can increase despite equal GEMM FLOPs: ordinary unfused ReLU4MH0b (GEMM write, activation read/write, output GEMM read); producer-fused ReLU2MH0b. Separate gate/up GEMMs plus fused SiLU-product and output GEMM costs6MHb=4MH0b; extra separate SiLU adds2MHb. Counts depend on fusion/cache/training and exclude weights/input/output. Explain rather than claim universal slowdown.
- C4 span filling, batch128, encoder512 tokens and target114.524288steps;4 short runs/variant65536steps. Adafactor inverse-sqrt LR, final10% linear decay. No pretraining dropout.32-core TPUv2 ~0.15s/step (not variant latency benchmark). Precision not reported.
- Fine-tune once/variant131072steps, LR1e-3, ~65536 input tokens/step; examples-proportional joint SQuAD+GLUE+SuperGLUE mixture; dropout0.1 layer outputs/FFN hidden/attention; embedding frozen. Report dev best checkpoint per task, not one guaranteed shared best checkpoint. No own long-run or fine-tune std/confidence interval.
- Table1 GEGLU1.633 best, SwiGLU1.636 vs ReLU1.677. Deltas0.044/0.041 lower; exp(-delta) gives ~4.30%/~4.02% perplexity reduction IF natural log assumed; paper does not explicitly define log base.
- Table1 brackets: source only states4 short runs measuring interrun variability; does not explicitly label parenthetical statistic. Retain as “括号离散度（统计量未明示）”, no confident SD/CI/significance claim.
- Table2 GLUE average ReGLU84.67 best vs ReLU83.80 (+0.87); SwiGLU84.36(+0.56); GEGLU84.12(+0.32). MRPC ReLU is competitive/best; GEGLU RTE79.42 <ReLU80.14.
- Table3 SuperGLUE SwiGLU74.56 best vs ReLU72.76(+1.80). SwiGLU MultiRCF175.56 <76.93, EM38.72<39.14, WiC67.24<67.71. GEGLU MultiRC77.43/41.03.
- Table4 SQuAD EM Bilinear83.82 best vs83.18(+0.64); F1 ReGLU91.18 vs90.87(+0.31). GEGLU83.55/91.12; Swi83.42/91.03. GLU82.88/90.69 worse.
- Tables2/3 Score Average cannot be reconstructed as simple mean of displayed best task metrics: ReLU grouped visible GLUE84.498125 vs printed83.80, SGLUE74.601875 vs72.76. Preserve printed aggregate; no invented explanation, note checkpoint/aggregation details insufficient to resolve. Original per-task checkpoint selection is potential explanation, not established cause.
- Raffel rows and stddev in Tables2–4 belong to separate earlier protocol: pretraining dropout + separate per-task finetuning vs joint mixture now. Cannot treat delta as solely GLU nor use those SDs for this study's significance.
- Conclusion explicitly offers no explanation for success. Gradient/feature interaction are own reconstruction/hypotheses, not theorem showing SwiGLU superior.

## External verified sources

Sources acquired via official arXiv pages/full HTML into /private/tmp/glu-work; acquisition.json records status. GitHub accessed using gh.

- GLU:1612.08083 (2016 preprint / ICML2017 gated conv LM).
- GELU:1606.08415 (2016); Swish:1710.05941 (2017).
- T5.1.1 actual google-research/text-to-text-transfer-transformer gin config names GEGLU, H2048,no training dropout. Original bi_bert_base gin confirms D768,num_layers12,H3072,heads12,d_kv64. Related code, not exact8-variant experiment checkpoint reproduction.
- mT5:2010.11934 (2020/NAACL2021) multilingual adoption route; no direct same-protocol comparison.
- PaLM2204.02311 §2 explicitly cites Shazeer2020 and SwiGLU; decoder-only.
- Llama2302.13971 §2 cites SwiGLU, H=(2/3)4D. Meta model.py computes shrink, multiplier and multiple_of rounding; w2(silu(w1(x))*w3(x)). Modern widths not universal exact8D/3.
- HF T5 T5DenseGatedActDense maps wi_0/wi_1/wo. PyTorch SiLU=xσ(x), nn.GLU uses firsthalf*sigmoid(secondhalf); nn.GLU is not SwiGLU. Source API via gh; docs URLs403, don't retry them.
- Official PyTorch packed-projection tutorial github pytorch/tutorials intermediate_source/transformer_building_blocks.py shows packed SwiGLUFFN; example A100 ~1.08x, not portable claim. Packing combines input projections but does not eliminate product tensor itself.
- Scaling FP8 training to trillion-token LLMs2409.12517v2:7B,up to2Ttokens,256Gaudi2;34% throughput includes Smooth-SwiGLU + FP8 optimizer. Table3 performance measured8Gaudi2:12.65 BF16→16.89 samples/s(+33.52%). Stable vs basic FP8 diverges. Whole-MLP rescaling preserves real arithmetic behavior; not necessarily activation itself identical, quantized finite precision differs.
- Towards Fully FP8 GEMM LLM Training at Scale2505.20524v1: reinforces gate/outlier quantization concern.
- Tile-Level Activation Overlap for Efficient LLM Inference2607.02521v1 (2026Jul): H10080GB,CUDA13,CUTLASS3,PyTorch2.11;Qwen2.5 0.5/1.5/14/72B, M256–4096 BF16. Up-to2.47x vs eager,0.5B M256;72B M4096=0.99x. Scope upstream GEMM+SwiGLU fused operation, not TTFT/TPOT or end-to-end model; not M=1 decode nor non-Nvidia hardware. Numerical “zero mismatches” author relative reference not proof higher FP32 accuracy.
- Prox2607.27591v1 (2026Jul):10models6families; train-free channelmask proxy then selected exact3projections. Singlebatch A6000 main end-to-end1.51–1.99x at60–70%sparsity; quality decreases at high sparsity. ExampleQwen3-8B aggregate76.1dense→68.6at70%; not exact SwiGLU faster. Throughput vsTEAL no consistent dominance.
- Weakening Neurons2609.18612v1 (2026Sep): analyzes GLU reading/writing vectors, negative gates meaningful; own9model study not proof universal mechanism or activation SOTA.
- PowLU2605.25704 (2026May):Ling MoE7.9Btotal600Btokens Table2 MMLU53.95Swi→54.92Pow, AGIEval31.75→31.13.124Btotal800B Table3 MMLU69.10→69.14,MMLUPro40.75→40.12,GSM8K70.81→69.90. No across-task universal winner; activated params/checkpoint reproducibility require qualification. Table4 47Mactive29.8Btokens loss1.910Swi vs1.912–1.914Pow. Avoid claim PowLU always wins.

## Review/engineering direction

Two widgets: (1) gate vs carried branch scalar values f(a)*b with activation selector, negative-gate and a≠b presets; optionally sample curve static fallbacks; no empirical accuracy implication. (2) width/matched-vs-fixed parameter/GEMM FLOPs/logical traffic calculator; M=1 decode vs largeM input matrix setting; no latency projection. Numerical validate packed/unpacked outputs, bias-free shapes and chain-rule gradients. Proposed GPU/HTP profiling only, never claim trained/benchmarked.

References should link actual sources and original tables. Core sections follow1→2→3.1→3.2→3.3→4; engineering/extensions after. Source MathML available. All source tables have no underline or ↑↓; retain exact bold choices, including tie not consistently bolded. Add your own clear metric arrows only in analysis, not as original formatting.
