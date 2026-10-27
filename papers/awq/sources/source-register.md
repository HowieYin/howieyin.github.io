# AWQ source register

Accessed / reviewed: 2026-10-27. All numerical performance and large-model quality evidence is reported by the cited primary source; no independent large-model reproduction is claimed.

| Source | Role / supported claims |
|---|---|
| [AWQ v6 PDF](https://arxiv.org/pdf/2306.00978v6) | Primary 15-page source: title, sections, original 10 figures, 10 tables, 5 equation groups and experiment protocol |
| [arXiv record](https://arxiv.org/abs/2306.00978v6) | Version date, authors, DOI and submission history |
| [arXiv original HTML](https://arxiv.org/html/2306.00978v6) | Original vector/raster figure files and selectable table markup; equation (4) corrected using PDF |
| [AWQ project](https://hanlab.mit.edu/projects/awq) | Official project / code / slides links and publication context |
| [llm-awq fixed revision](https://github.com/mit-han-lab/llm-awq/tree/d6e797a42b9ef7778de8ee2352116e0f48a78d61) | Full quantize modules / TinyChat execution: alpha, clipping, graph folds, qweight packing, metadata, GEMV/GEMM |
| [Author slides](https://www.dropbox.com/scl/fi/dtnp6h6y1mnp7g036axu6/AWQ-slide.pdf?rlkey=ffgh50hxhx8dmsnjiu8kef0ou&dl=0) | 37-slide official mechanism and systems explanation; older numbers not merged into v6 tables |
| [GPTQ](https://arxiv.org/abs/2210.17323v2) / [code](https://github.com/IST-DASLab/gptq) | Prior second-order weight-only quantization and complementarity |
| [SmoothQuant](https://arxiv.org/abs/2211.10438v7) | Prior scaling and W8A8 objective |
| [QuaRot](https://arxiv.org/abs/2404.00456v2) | Orthogonal rotation / W4A4KV4 route |
| [SpinQuant](https://arxiv.org/abs/2405.16406v4) | Learned rotations and training-cost difference |
| [QServe](https://arxiv.org/abs/2405.04532v3) / [project](https://hanlab.mit.edu/projects/qserve) / [code](https://github.com/mit-han-lab/omniserve) | W4A8KV4, large batch serving and system-specific throughput |
| [Marlin README](https://github.com/IST-DASLab/marlin) / [paper](https://arxiv.org/abs/2408.11743) | W4A16 linear kernels; A10, group128, small/mid batch and hardware caveats |
| [TinyChat 2.0](https://hanlab.mit.edu/blog/tinychat20) | 2024-12-12 TTFT, FlashAttention, history reuse and NVILA vision bottleneck |
| [Transformers AWQ](https://huggingface.co/docs/transformers/quantization/awq) | Checkpoint format / import configuration |
| [AutoAWQ README](https://github.com/casper-hansen/AutoAWQ) | Explicit deprecation and last tested Torch/Transformers versions |
| [LLM Compressor examples](https://github.com/vllm-project/llm-compressor/tree/main/examples/awq) | Current AWQ migration destination documented in vLLM |
| [AAAC v1](https://arxiv.org/html/2605.08692v1) | 2026-05 activation-weighted codebooks, Table2/5 values, metadata and no throughput proof |
| [Schur Replay v1](https://arxiv.org/html/2609.36654v1) | 2026-09 NVFP4 W4A4, weighted recovery, per-layer quantization solve / projection distinction |
| [Humming announcement](https://vllm.ai/blog/2026-10-21-humming) / [code](https://github.com/vllm-project/humming) | 2026-10-21 current W4A16 dispatch/repacking, Hopper/Blackwell cold L2 warm kernel microbenchmarks |

Original asset inventory and exact captions: [visual-manifest.json](visual-manifest.json).

Generated visualizations: `assets/generated/pipeline.{svg,mmd}`, `runtime.{svg,mmd}`, `timeline.{svg,mmd}`. They are explanatory creations, not replacements for original figures. Interactive arithmetic: `interactive/math.cjs` and `interactive/reading.js`; scaling toy uses the paper's simplified unclipped grid, unsigned packing illustrates ARM NEON only, capacity calculation excludes padding / exceptions / runtime peak.
