# GLU Variants supporting note

[完整精读](../index.html)

## 技术演进与相关工作：2016–2026 的几条路线

**Before this paper。** 2016 年 GELU 与 GLU 的预印本分别探索平滑激活和门控卷积；2017 年 Swish 提供 `xσ(βx)`，Transformer 的普通 FFN 则通常用 ReLU。它们分别改变非线性形状与结构。

**This paper，2020。** 把 GLU 变体放入 Transformer FFN，以 2/3 hidden width 对齐主要预算，显示 GEGLU / SwiGLU 的去噪收益。

**After this paper。** T5.1.1 官方配置采用 GEGLU，mT5 把相关架构带到多语言；PaLM 与 Llama 的 decoder-only 系列采用 SwiGLU。后来的 kernel 融合、FP8 稳定训练和稀疏门控是在这一结构上处理不同的工程或精度问题，不能合成一条“后者全面替代前者”的单线。

<figure class="generated-figure"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 400" role="img" aria-label="从激活和 GLU 来源到采用与工程分支，2016–2026"><style>text{font:14px sans-serif;fill:#d5e1f3}.small{font-size:12px;fill:#9fb1c9}.box{fill:#182335;stroke:#49617f}.gate{stroke:#90b8f4}.up{stroke:#d9b779}.line{fill:none;stroke:#8eafdf;stroke-width:1.5}</style><text x="30" y="45">2016–17</text><rect class="box " x="155" y="15" width="585" height="65" rx="6"/><text x="171" y="41">GELU / GLU / Swish</text><text class="small" x="171" y="64">Smooth activation and multiplicative gating, separate routes</text><path class="line" d="M70 58v47"/><text x="30" y="140">2020</text><rect class="box " x="155" y="110" width="585" height="65" rx="6"/><text x="171" y="136">GLU Variants → T5.1.1 / mT5</text><text class="small" x="171" y="159">Budget-matched FFN comparison; GEGLU adoption</text><path class="line" d="M70 153v47"/><text x="30" y="235">2022–23</text><rect class="box " x="155" y="205" width="585" height="65" rx="6"/><text x="171" y="231">PaLM / Llama</text><text class="small" x="171" y="254">SwiGLU adoption in decoder-only language models</text><path class="line" d="M70 248v47"/><text x="30" y="330">2024–26</text><rect class="box " x="155" y="300" width="585" height="65" rx="6"/><text x="171" y="326">FP8 / tile fusion / sparse gates / alternatives</text><text class="small" x="171" y="349">Numerical stability, IO, approximation and new functions</text></svg><figcaption>从激活和 GLU 来源到采用与工程分支，2016–2026 · Agent-created / Simplified Diagram。[<a href="assets/generated/timeline.mmd">Mermaid source</a>]</figcaption></figure>

| 一手来源 | 年份 | 核心想法 / 优势 | 限制 | 与本文关系 |
|---|---:|---|---|---|
| [GELU](https://arxiv.org/abs/1606.08415) | 2016 | `xΦ(x)`，平滑保留小负值 | 单路激活不直接构成两路交互 | GEGLU 的 gate 函数 |
| [Gated Conv LM](https://arxiv.org/abs/1612.08083) | 2016 / ICML 2017 | sigmoid 门控卷积特征 | 当时不是本文的 T5 对照 | GLU 的直接来源 |
| [Swish](https://arxiv.org/abs/1710.05941) | 2017 | `xσ(βx)`，可设置 β | 不是独立的两路 gate 架构 | SwiGLU 的函数来源 |
| [T5](https://arxiv.org/abs/1910.10683) / [T5.1.1 config](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/models/gin/models/t5.1.1.base.gin) | 2019 / 2020 | 统一 text-to-text；后续实际采用 GEGLU / H=2048 | 配置不等于八变体完整重现 | 实验底座与落地 |
| [mT5](https://arxiv.org/abs/2010.11934) | 2020 / NAACL 2021 | 多语言 text-to-text | 不提供同协议的八 gate 比较 | 多语言延伸 |
| [PaLM](https://arxiv.org/abs/2204.02311) | 2022 | 大规模 decoder-only 明确采用 SwiGLU | 与 T5 的数据、规模和目标不同 | 后续采用，非同条件复现 |
| [Llama](https://arxiv.org/abs/2302.13971) / [Meta implementation](https://github.com/meta-llama/llama/blob/main/llama/model.py) | 2023 | SwiGLU，hidden 缩小并做硬件倍数 rounding | 现代模型 H 可能有 multiplier | 广泛部署路线 |
| [Scaling FP8 Training](https://arxiv.org/abs/2409.12517) | 2024 | Smooth-SwiGLU 等处理 FP8 稳定性 | 总吞吐提升不能全归于 gate | 低精度的隐藏 trade-off |
| [Fully FP8 GEMM Training](https://arxiv.org/abs/2505.20524) | 2025 | 进一步研究全 FP8 GEMM 的稳定方案 | 训练设置与算子/硬件特定 | 乘法 outlier 与数值工程 |
| [PowLU](https://arxiv.org/abs/2605.25704) | 2026 | 新 gate 路线，在 MoE 中比较 | 各任务混合胜负，非统一最佳 | 当代替代函数研究 |
| [Tile-level overlap](https://arxiv.org/abs/2607.02521) | 2026 | 重叠 GEMM tile 与 SwiGLU activation | 主测大 M、NVIDIA H100 | 融合与 IO 路线 |
| [Prox](https://arxiv.org/abs/2607.27591) | 2026 | 代理选择 channel 后仅算部分精确投影 | 稀疏率越大质量越可能下降 | 近似稀疏加速路线 |
| [Weakening Neurons](https://arxiv.org/abs/2609.18612) | 2026 | 分析 GLU 读取/写入向量及负 gate | 研究观察非普适机制定理 | 机制解释的后续证据 |



## Current State：截至 2026-10-27，怎样读“最好”？

**External Source / 可比性优先。** 未找到统一公开基准，能在相同数据、训练 tokens、参数、实现、精度和硬件下，证明某种 gate 是所有 Transformer 的绝对 SOTA。下表分开列质量、算子效率、低精度稳定性与近似稀疏速度；每项都有独立比较对象。

| 方向 / 来源 | Benchmark、scale 与条件 | 报告结果 | 开放性 / 可比边界 |
|---|---|---|---|
| 结构采用：T5.1.1 / PaLM / Llama | encoder-decoder 与 decoder-only 各自训练体系 | GEGLU / SwiGLU 是有公开采用证据的常见选择 | T5 / Llama 实现公开；采用不等于同条件证明最优 |
| FP8：[Scaling FP8](https://arxiv.org/html/2409.12517v2) | 7B，最多 2T tokens，训练集群 256 Gaudi2；Table 3 吞吐测 8 Gaudi2 | BF16 12.65 → FP8 16.89 samples/s，约 +33.52%；基本 FP8 的不稳定被缓解 | 方案包括 Smooth-SwiGLU 与 FP8 optimizer，34% 不能只归于 gate；不是端侧 INT4 证据 |
| 算子融合：[Tile-level overlap](https://arxiv.org/html/2607.02521v1) | H100 80GB，BF16，Qwen2.5 0.5 / 1.5 / 14 / 72B 的 FFN 上游 GEMM+SwiGLU；M=256–4096 | 对 eager 最多 2.47×，出现在 0.5B / M=256；72B / M=4096 为 0.99× | 论文披露 CUDA13 / CUTLASS3 / PyTorch2.11；未将可复现完整仓库作为本次证据；非整模型、非 M=1 decode |
| 稀疏：[Prox](https://arxiv.org/html/2607.27591v1) | 10 models / 6 families；A6000 单 batch，60–70% channel sparsity | 主要整模型示例 1.51–1.99×；Qwen3-8B aggregate 76.1 dense → 68.6 at 70% | 以论文表为依据，完整 checkpoint 复现未核验；近似删通道有质量损失，不是 exact SwiGLU 更快 |
| 替代 gate：[PowLU](https://arxiv.org/html/2605.25704) | Ling MoE 7.9B total / 600B tokens 与 124B total / 800B tokens | 7.9B MMLU 53.95 → 54.92，AGIEval 31.75 → 31.13；124B MMLU 69.10 → 69.14，GSM8K 70.81 → 69.90 | 不以 total 参数替代 active 参数；完整训练 checkpoint 开放性未核验；混合胜负 |

**为什么不能直接排行。** Tile-level overlap 优化保持函数的执行路径，Prox 则改变实际计算的 channel 集；PowLU 改变 gate 函数；FP8 训练改变数值表示和稳定策略。这些方法回答不同问题，速度数字的分母也不同。Tile 论文的 “zero mismatches” 只表示与其参考路径的对照，不等于对 FP32 真值更准确，也不是模型 accuracy 指标。

**补充细节。** PowLU 论文还报告 47M active、29.8B tokens 的小规模 loss：SwiGLU 1.910，对比 PowLU 配置 1.912–1.914。因此“新 gate 在某表有分数提升”不能扩写成“训练 loss 全面胜出”。Prox 对 TEAL 的 throughput 也不是所有条件都更快。对于端侧 HTP、INT4 和 batch=1 decode，本次这些来源都不足以给出迁移后的收益数字。
