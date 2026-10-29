# Extended research — verified through 2026-10-29

扩展研究区分“本文原结果”“后续公开模型”“最新官方声明”。来源优先 arXiv、作者模型卡与官方实现；下面的性能均为各作者报告，未本地复测。

## Before / This Paper / After

| Work / source | Date | Core idea / advantage | Limitation | Relation to Llama 3 |
|---|---|---|---|---|
| [Transformer](https://arxiv.org/abs/1706.03762) | 2017 | attention与FFN构成可并行序列模型 | dense attention长序列成本高 | 语言主干基础 |
| [RMSNorm](https://arxiv.org/abs/1910.07467)、[GLU Variants](https://arxiv.org/abs/2002.05202)、[RoFormer](https://arxiv.org/abs/2104.09864) | 2019–2021 | 简化归一化、门控FFN、旋转位置编码 | 各有归约/投影/长位置外推成本 | 成熟组件组合，并非本文逐个首创 |
| [Chinchilla](https://arxiv.org/abs/2203.15556) | 2022-03 | 固定compute下联合选参数与tokens，约等比例扩展 | 最优依赖data/recipe，未直接优化未来服务成本 | 本文IsoFLOPs与405B选择；小模型则过度训练 |
| [Flamingo](https://arxiv.org/abs/2204.14198) | 2022-04 | 连接pretrained视觉/语言，支持交错多模态 | 接口成本和冻结底座能力边界 | 视觉cross-attention路线的前作 |
| [BEST-RQ](https://arxiv.org/abs/2202.01855) | 2022-02 | 固定随机projection/codebook生成masked speech目标 | label设计与感知架构仍影响效果 | speech encoder自监督目标 |
| [GQA](https://arxiv.org/abs/2305.13245) | 2023-05 | query组共享KV，减少缓存/带宽 | 不等于减少query attention FLOPs同样倍数 | 所有Llama3尺寸8个KV heads |
| [DPO](https://arxiv.org/abs/2305.18290) | 2023-05 | reference-relative偏好分类损失，简化policy对齐 | 数据偏好、reference与mask影响结果 | 加chosen NLL并mask格式tokens |
| The Llama 3 Herd of Models | 2024-07 / v3 2024-11 | dense405B、15.6T、4D系统与后训练闭环 | 数据/内部judge/全部研究接口未完整开放 | 本文 |
| [DeepSeek-V3 v2](https://arxiv.org/html/2412.19437v2) | 2024-12 / v2 2025-02 | 671B total/37B active、MLA/MoE、14.8T、FP8训练与DualPipe | routing/通信复杂，公开预算不含所有研究试错 | 另一条效率路线；不能按37B计算全部权重存储 |
| [Olmo 3](https://arxiv.org/abs/2512.13961v2) | 2025-12 / v2 2026-04 | 7B/32B，发布full model flow、数据与阶段依赖 | 尺寸/推理设置不同于405B | 提升“全链路可复现”程度，与仅开权重不同 |

Chinchilla原文在400多个70M–16B模型、5–500B tokens实验中估計compute optimum。Llama3采用自己的数据与规模点重新拟合，而不是机械套20tokens/parameter。本文N*(C)是tokens，DeepSeek-V3的total/active是参数；符号意义与成本口径都要分开。

## Official implementation audit

Meta `llama-models` snapshot：`0e0b8c519242d5833d8c11bffc1232b77ad7f301`；读取 `models/llama3/model.py`、Llama3.1 `MODEL_CARD.md`/`prompt_format.md` 与Llama4 model card。FBGEMM snapshot：`d4b20c45449d302a5984066e3836b0f91a8f20f0`；读取 `fbgemm_gpu/experimental/gen_ai/src/quantize/quantize.cu`。检索日固定snapshot，不声称这些main代码等同2024内部训练实现。

- **GQA cache。**参考attention缓存以local KV heads保存，repeat_kv按query组扩展供attention使用。cache容量不应按扩展后的query heads重复计算。
- **Tool protocol。**Llama3.1格式文档说明EOM用于executor交接，EOT终止turn，ipython是tool结果role。格式与runtime都要匹配；官方文档中某些自定义prompt例子只支持single call，不据此否定论文更广的function评测能力。
- **FP8 scales。**内核明确对row_absmax取min(scale_ub)，再除格式最大值求scale；CUDA E4M3最大448。1200约束range估计，允许大值饱和换小值分辨率。当前内核有多种路径/格式，部署时要检查实际调用路径。
- **Open boundary。**语言checkpoint与参考实现可查；15T专有混合语料、全偏好/安全数据、NCCLX生产运行栈和所有speech/video研究权重并非此仓库全套公开。

源代码链接：

https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/model.py

https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3_1/prompt_format.md

https://github.com/pytorch/FBGEMM/blob/d4b20c45449d302a5984066e3836b0f91a8f20f0/fbgemm_gpu/experimental/gen_ai/src/quantize/quantize.cu

## Released models after the paper's July date

**2024-09-25 Llama3.2。**Meta正式发布11B/90B vision与1B/3B text。视觉继续冻结语言底座和encoder/cross-attention接口思路；小文本模型采用结构化剪枝与蒸馏、对接ExecuTorch等端侧路线。发布时间早于指定v3修订、晚于论文内日期，因此不能说“v3之后才第一次发布vision”。正式产品名称/参数也不应与本文405B视觉研究模块混为一谈。

官方发布：https://about.fb.com/ltam/news/2024/09/llama-3-2-revolucionando-la-ia-y-la-vision-de-vanguardia-con-modelos-abiertos-y-personalizables/

**2025-04-05 Llama4。**官方模型卡写Scout17B active/109B total、16experts、10M context、约40T tokens；Maverick17B active/400B total、128experts、1M context、约22T tokens。它们是early-fusion native multimodal MoE，结构已不同于本文dense+compositional路线。模型卡说明报告benchmark用BF16；不能拿INT4部署条件直接给BF16精度结果背书。Behemoth288B active是当时预告，不能写为正式发布。

模型卡：https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama4/MODEL_CARD.md

## Current state / SOTA evidence boundary

**截至2026-10-29，不能继续把Llama4称为Meta最新模型。**本次检索并读取Meta 2026-10-12官方新闻：Muse Wonder与Wonder Full用于agent tasks，通过Meta Model API/Muse Code提供，强调Active Memory/Focus。官方称达到前沿表现；取得的官方新闻正文没有足以与本文Table2同条件对比的完整benchmark、参数与开权重信息。链接的research blog本次抓取为空，故不采用第三方转述的“57.5 SWE-Marathon”等值作为已核验原始成绩。

官方最新来源：https://about.fb.com/news/2026/10/introducing-muse-wonder-models/

搜索亦发现2026聚合榜含明确“Estimated”的scores。这类估计不进入当前SOTA定量结论；第三方重复数字不构成独立复测。当前任务只给已核验的证据范围，不能据此指定全球唯一SOTA。

| 可比较范围 | Verified model / scale | Metric / result | Open status / fairness |
|---|---|---|---|
| 本文2024实验 | Llama3.1 405B dense | MMLU87.3、GPQA51.1、HumanEval89.0、HumanEval+82.3 | 开权重；与当时closed baseline的data/compute不一致 |
| 后续作者评测 | DeepSeek-V3 671B total/37B active | v2报告MMLU88.5、GPQA59.1，14.8T；官方训练2.788M H800hours | checkpoint公开；作者/版本/推理budget不同，不与2024表做统一排行榜 |
| 全链路开放路线 | Olmo3 7B/32B，Think/Instruct | 2026-04 v2宣称Think32B是当时最强fully-open thinking model | full model flow公开；“fully-open”子集与全球frontier不是同一集合，本文不复核其全球排名 |
| 最新Meta官方声明 | Muse Wonder / Wonder Full，参数本次未确认 | agentic frontier claim；取得正文无完整同条件表 | API产品；本次未核验开权重、推理budget及可比benchmark |

技术方向已从本文的dense规模与compositional接口，扩展到MoE/MLA、FP8训练、长推理/agent系统和更完整训练流程开放。可复用的原则仍是同一张账：训练FLOPs、实际权重内存、KV、通信、推理token预算、数据与业务错误。任务SOTA要在相同metric、版本、hardware、precision、latency/compute约束下判断。
