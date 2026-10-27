## Related Work & Technology Evolution：2018—2026

下表的速度均为各来源自报上限或范围，**不形成跨条件排名**。年份取最早公开时间；正文保持 Medusa 2024 年历史语境，后续工作另列。

| 路线 / 来源 | 时间 | 核心变化 | 优点 | 限制 / 与 Medusa 的关系 |
|---|---|---|---|---|
| Blockwise Parallel Decoding [2] | 2018-11 | 多未来头 + scoring model 最长前缀 | 奠定并行预测再回退思想 | MT/图像设置；Medusa 明确继承这一骨架 |
| Exact Speculative Sampling [3] | 2022-11 / ICML 2023 | 独立 drafter + rejection/residual | 保持固定 target 采样分布 | 两模型和 draft 串行成本；接受保证需正确实现 |
| Medusa [1] | 2024-01 | 独立未来 heads、固定树、M1/M2 | 复用 target hidden，结构浅 | 缺少草稿依赖；typical 不无损；需训练 |
| EAGLE [4] | 2024-01 | feature-level 自回归并引入移位 token 条件 | 建模 token 依赖；论文报告 2.7–3.5× | draft 仍有多步依赖；不能直接对比旧 Medusa baseline |
| Hydra [5] | 2024-02 | sequentially dependent heads | 针对 Medusa 头间独立性；报告最高 1.31× Medusa | 增加依赖计算；结果属其测试条件 |
| ReDrafter [6] | 2024-03 | recurrent draft head、beam tree、前缀去重 | 少量共享参数；GPU/Apple MLX 两类实现 | 报告最高 2.8× H100、2.3× Apple，硬件条件不同 |
| EAGLE-2 [7] | 2024-06 | context-dependent 动态树 | 根据在线置信度分配预算；报告 3.05–4.26× | 构树与调度成本；相比静态树更动态 |
| EAGLE-3 [8] | 2025-03 | 多层特征融合，训练模拟 test-time draft | 去掉 feature regression 约束；报告最高 6.5× | 训练/系统配置决定效果；最高值不是通用比例 |
| DFlash [9] | 2026-02 | 条件于 target 特征的轻量 block diffusion drafter | 单次并行提出 token block；严格验证 | 需专用 drafter；报告 >6× 和最高 2.5× EAGLE-3 属自身条件 |
| DFlash / SpecV2 系统更新 [10] | 2026-06 | KV feature 注入、异步准备与 host 清理 | 将接受器/prepare/sync 纳入优化 | kernel、concurrency、训练与硬件决定可比性 |

{{DIAGRAM:evolution}}

这张 **Agent-created / Simplified Mermaid diagram** 表示技术关系，不表示单一路线取代所有其他路线。未来 token 依赖、在线预算分配、并行草稿生成和异步 runtime 是不同轴，可以有组合；箭头不是跨论文性能胜负。


## Current State / SOTA：截至 2026-10-27 的条件化判断

本节重新检索并核验原始论文与官方引擎资料。当前公开路线已超出 Medusa 的独立 heads：feature-conditioned draft、动态树、native MTP、并行 block drafter 与 speculative runtime overlap 都在演进。**没有统一 target、硬件、质量、采样和负载的公共结果，就不宣布一个全局 SOTA。**

**较强的同条件证据 [10]。** 2026-06-15 的 ZLab / Modal / SGLang 联合官方文章，给出相同 Qwen3-4B target、相同训练数据、5 层 drafter 的 EAGLE-3 与 DFlash 对照：

| Benchmark | EAGLE-3 acc. length / speedup | DFlash acc. length / speedup |
|---|---:|---:|
| GSM8K | 4.2 / 2.1× | 4.2 / 3.3× |
| HumanEval | 4.3 / 2.2× | 4.0 / 3.2× |
| MT-Bench | 3.1 / 1.4× | 3.0 / 2.2× |

接受长度相近却速度更高，说明串行 drafter 与 runtime 开销是关键；不能只看 accepted tokens 排名。相对于 EAGLE-3 的比值分别约 1.57、1.45、1.57。该结果比跨论文“最高 6.5× 与最高 6×”互比更有意义；文章正文未把此表所有独立硬件条件单列到每一格，因此仍应按该文章完整配置理解。

**大模型实例。** 同文 Qwen3.5-397B-A17B BF16，8×B200，HumanEval greedy、thinking、最大输出 4096、200 cases、concurrency=1，DFlash block 16 对 AR 报告 >4.3×，对 7-step MTP 约 1.5×。这些条件与 Medusa 的 Vicuna/默认引擎结果完全不同，不列为直接胜负。

**服务工程。** SpecV2 清理 host stop-token 等逻辑，并将 prepare/allocation 与 GPU 工作重叠；Qwen3-8B、单 B200、concurrency=32 的报告从约 11.4k 到 15.3k token/s（约 +33%）。这正说明优化应沿调用链测量：draft/verify 之外，CPU 准备和同步也能决定结果。其 feature→KV 注入和 radix prefix sharing 是新 runtime 设计，不能当成原 Medusa 的机制。

**实际采用。** Hugging Face TGI 文档仍明确支持 Medusa heads 和 n-gram speculation [11]；SGLang 官方更新讨论 EAGLE/DFlash/SpecV2 [10]；vLLM 2026-08-23 的 AMD GPU 官方文章讨论 MI300X/MI355X 上 native MTP、EAGLE-3、DFlash 等路线和 workload 差异 [12]。这些是实现采用证据，不是相同 benchmark 的质量/性能排名。没有读取到的 NVIDIA 旧文档返回 404，未用它作为支持证据。


## References & Source Record

**原论文与资产。** [1] Cai et al., ICML 2024，[v3 PDF](sources/2401.10774v3.pdf)、[PMLR](https://proceedings.mlr.press/v235/cai24b.html)、[完整图表来源清单](sources/visual-manifest.json)、[源文本](sources/paper-text.txt)。PDF SHA-256：`93d98f2e858c87ee04be440ee81ab8ad93700652ed759227d25fcf75cfdd5ef0`。23 Figures 与 8 Tables 均逐号保留。原图由作者源 PDF/PNG 高分辨率渲染，原表由对应原 PDF 页面完整裁取；多 panel 按原顺序组合，未删 legend、坐标轴、脚注。

**相关论文。**

- [2] Stern, Shazeer & Uszkoreit. *Blockwise Parallel Decoding for Deep Autoregressive Models*. NeurIPS 2018. [arXiv:1811.03115](https://arxiv.org/abs/1811.03115)。
- [3] Leviathan, Kalman & Matias. *Fast Inference from Transformers via Speculative Decoding*. ICML 2023. [arXiv:2211.17192v2](https://arxiv.org/html/2211.17192v2)，尤其 Algorithm / §2.3 与 Theorem 3.5。
- [4] Li et al. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*. [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)。
- [5] Ankner et al. *Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding*. [arXiv:2402.05109](https://arxiv.org/abs/2402.05109)；[code](https://github.com/zankner/Hydra)。
- [6] Cheng et al. *Recurrent Drafter for Fast Speculative Decoding in Large Language Models*. [arXiv:2403.09919](https://arxiv.org/abs/2403.09919)。
- [7] Li et al. *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*. [arXiv:2406.16858](https://arxiv.org/abs/2406.16858)。
- [8] Li et al. *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*. [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)。
- [9] *DFlash: Block Diffusion for Flash Speculative Decoding*. ICML 2026. [arXiv:2602.06036v2](https://arxiv.org/abs/2602.06036v2)。

**官方代码、项目与引擎资料。**

- [Medusa official code](https://github.com/FasterDecoding/Medusa)，本次检查 commit `e2a5d20c048a9b0a4092e6933c34313687422518`。重点文件：`medusa/model/medusa_model.py`、`utils.py`、`kv_cache.py`。[作者 project/blog](https://sites.google.com/view/medusa-llm) 为早期介绍，与最终 v3 的 M2/recipe 范围需区分。
- [10] ZLab / Modal / SGLang. *Next-Generation Speculative Decoding: DFlash and Speculative Decoding V2*. 2026-06-15. [官方文章](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2)。
- [11] Hugging Face TGI. *Speculation*. [官方文档](https://huggingface.co/docs/text-generation-inference/conceptual/speculation)，访问 2026-10-27。
- [12] vLLM. *Speculative Decoding on AMD GPUs*. 2026-08-23. [官方文章](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus)。

外部资料核验截止 2026-10-27。上述来源用于机制、历史和当下公开结果的解释；除 Tables 6–8 的代数查询以及页面交互外，没有将外部 benchmark 冒充本地实验。[来源与核验记录](sources/source-ledger.json) 提供版本、用途与可比性说明。

