# Medusa：一次权重读取，能提交多少个 Token？

**Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**

Tianle Cai、Yuhong Li（共同一作）、Zhengyang Geng、Hongwu Peng、Jason D. Lee、Deming Chen、Tri Dao。机构：Princeton University、Together AI、UIUC、Carnegie Mellon University、University of Connecticut。ICML 2024，PMLR 235，5209–5235。初稿 2024-01-19；本次精读严格采用 **arXiv 2401.10774v3，2024-06-14，27 页**。阅读与更新：2026-10-27。

[Paper PDF](sources/2401.10774v3.pdf) · [arXiv](https://arxiv.org/abs/2401.10774v3) · [Publisher](https://proceedings.mlr.press/v235/cai24b.html) · [DOI](https://doi.org/10.48550/arXiv.2401.10774) · [Official code](https://github.com/FasterDecoding/Medusa) · [Project / author blog](https://sites.google.com/view/medusa-llm)


## TL;DR：用更多候选，换更少串行轮次

**一句话：** Medusa 在同一个 LLM 的最后隐藏状态上增加多个未来位置预测头，把候选组织成树，一次主模型 forward 验证多条路径，再只提交合法的最长前缀。

**一段话：** 小 batch 的 autoregressive decode 经常花大量时间读权重，却只生成一个 token。Medusa 利用这部分算力余量，多算一些可能被接受的未来 token，以减少主模型串行调用。Medusa-1 冻结 backbone，Medusa-2 联合训练 backbone 和预测头；tree attention 避免候选之间污染上下文，稀疏树减少无效验证，typical acceptance 则用分布放宽换接受长度。论文的 Vicuna-7B/13B Medusa-2 在 MT-Bench 达到约 2.83× 相对默认基线的 wall-time speedup。该结果有明确模型、实现和质量条件，不能直接外推为今天所有服务或端侧 NPU 的收益。

五个关键点：

1. **省的是主模型轮次，不是每轮 FLOPs。** 多 token 验证让矩阵更大，但有望摊薄权重读取。
2. **多个 head 共享输入，不是彼此自回归。** 它们预测未来位置的边缘分布，缺少草稿 token 之间的条件依赖。
3. **tree mask 和 position ID 都属于正确性。** 同层兄弟不可互看，RoPE 位置取路径深度，不能取展平索引。
4. **加速率与 speedup 是两种量。** 前者是 token/轮，后者还要除以每轮额外成本。
5. **质量保证要拆开判断。** 冻结、联合训练、greedy verification、严格 rejection sampling、typical acceptance 的保证各不相同。

<div class="callout interpretation"><strong>阅读目标 / Interpretation</strong><p>面向熟悉 Transformer、KV cache 与 TTFT/TPOT 的工程师。先按原论文章节重构方法，再读全部附录、23 张原图和 8 张完整原表，最后讨论 2026 年技术演进和可迁移实现。青绿色提示为本精读的分析，黄色提示为限制或原文需澄清之处；原论文、外部资料和分析分别标注。所有 GPU 速度数字均来自来源报告，本次没有复跑 GPU benchmark。</p></div>


## 1. Introduction：权重很大，每轮产出很少

**作者的问题。** LLM 的生成是串行依赖链：生成新 token 后，才能更新上下文并运行下一步。KV cache 缓存已计算的 K/V，免去重算历史，却不免去每一层投影和 MLP 的权重读取。小 batch 时，单 token GEMV/GEMM 的算术强度偏低，GPU 算力往往等待内存。

**已有方案。** 增大 batch 可以让多个请求共用权重读取，但会增加 KV 占用、调度约束和用户延迟。标准 speculative decoding 则先由小 draft model 生成多步，再由 target model 并行验证。它要求便宜而准确的草稿模型，也引入两套模型的加载、缓存与分布匹配问题。Medusa 将 drafter 融入 target 的末端表示，沿用 2018 年 blockwise parallel decoding 的思想 [2]，增加 tree verification 和训练策略。

**核心观察。** 某个未来 token 的 top-1 预测可能不够准，top-k 的覆盖率却更高。通过一次 forward 验证多种组合，有机会接受更长路径。代价是验证所有候选，因此关键任务变成：把有限验证预算放到最可能贡献接受长度的位置。

**贡献如何分工。** heads 负责生成，tree attention 负责共享候选前缀，Medusa-2 负责提高可预测性，typical acceptance 负责放宽接受条件，自蒸馏负责补训练数据，稀疏树负责控制工作量。它们分别改变计算、训练和输出契约，不能当成一个天然无损的整体。

{{FIG:1}}

**Figure 1 深读。** 从左到右读：LLM 的同一个最后隐藏状态进入原 LM head 与多个 Medusa heads；不同颜色对应不同预测距离；各 head 的高分 token 组成树；主模型以 tree mask 验证；接受器选出最长合法前缀。最后一个已接受节点对应的 hidden/logits 能为下一轮提供草稿，循环复用主模型输出。图里的并行箭头表示候选可同时处理，不能理解为最终输出 token 取消了自回归条件验证。真正减少的是串行验证轮数。


## 2. Methodology：候选、验证与接受形成状态闭环

方法输入是已提交前缀、对应 KV cache、最后位置的 LM/Medusa logits 和树配置；输出是新提交 token、压实后的 KV、下一轮 logits。主模型仍是最终条件概率的计算者。将“已提交状态”和“候选临时状态”分开，是读懂整个方法的起点。

{{DIAGRAM:flow}}

此图为 **Agent-created / Simplified Mermaid diagram**。主模型树验证产生临时节点状态，接受后才把一条路径变成真实历史。被拒节点不是下一轮历史；只重写 token 序列而忘记修正 KV 长度，会造成逻辑污染。

### 2.1.1 Medusa Heads：同时预测多个未来位置

原 LM head 用 h_t 预测 y_(t+1)；第 k 个新增 head 预测 y_(t+k+1)，k 从 1 到 K。因此第一个 Medusa head 预测的是“下下个”token。论文定义：

{{EQ:head}}

其中 h_t 是最后隐藏状态，d 是隐藏宽度，V 是词表大小；p_t^(k) 是 V 维概率向量。W₁ 先把表示变换为 d 维，SiLU 加非线性，再与原 hidden 相加；W₂ 把残差表示投影到词表；softmax 归一化。SiLU(z)=z·sigmoid(z)，残差允许预测头从主模型表示出发逐步偏移。

**Tensor orientation 的澄清。** 原文同时写 W₂∈R^(d×V) 和左乘 W₂·h；若 h 是列向量，应使用 V×d。工程上采用行向量时，hidden 为 [B,T,d]，W₁ 为 [d,d]，W₂ 为 [d,V]，计算是 `(SiLU(h @ W1) + h) @ W2`；PyTorch `Linear(d,V).weight` 存 [V,d]。这是记号方向问题，原式保留，部署 shape 另行明示。

初始化 W₁=0、W₂ 从 LM head 复制，在无偏置的公式下所有新头起初等同于 next-token head，再训练成不同预测距离。官方代码快照的 `ResBlock` 使用带 bias 的 Linear，只将 weight 清零；bias 未同步归零时，不能从公式推出初始化输出严格相同。这里是源码检查发现的实现差异，未做运行验证。

**为什么预测不一定一致？** head 1 和 head 2 都只看到 h_t。head 2 没看到 head 1 最终提出的 token，因此不能直接表示 `p(y_(t+3)|prefix,y_(t+1),y_(t+2))`。例如 “New” 后同时猜 “York” 和 “City” 很自然，但若某条候选在中间选成了 “Jersey”，同一个远期 “City” 可能不再合适。tree verification 会用每条实际路径重新计算 target 条件分布。

**参数与内存不应忽略。** 忽略 bias，一个 head 新增 d²+dV 参数；K 个是 K(d²+dV)。以 d=4096、V=32000、K=5 为代数示例，共 739,246,080 参数，FP16 权重约 1.377 GiB。浅层结构确实比完整第二模型简单，但“大词表×多头”仍昂贵；训练梯度、优化器状态、激活与 logits 另算。

官方实现会先在树验证的多个 hidden 上计算 head 输出，再取被接受终点给下一轮使用；全量临时 logits 可以达到 [K,B,q,V]。工程上是否能只对最终需要的 hidden 做投影、跳过冗余头，必须在依赖链允许时单独评估，不能把它当作原论文已经证明的优化。

### 2.1.2 Tree Attention：共享祖先，隔离兄弟

第 k 个 head 取 top-s_k；完整笛卡尔积得到多条路径。论文给出新增 Medusa 节点数量：

{{EQ:nodes}}

这里 q_M 只统计 Medusa 节点；原 LM head 的根 token 在实际验证中占一个额外位置。s₁=2、s₂=3 时，有 6 个叶子、2+6=8 个 Medusa 节点，物理 forward 为 9 个新位置。逐条复制 6 条“根+两步”路径会处理 18 个位置；树共享根与第一层前缀，避免重复这些位置。

{{FIG:2}}

**Figure 2 深读。** 先看左侧拓扑，再对照右侧 mask。每个查询节点只能看完整 committed prefix、根、自己的祖先和自身；兄弟与其他分支不可见。注意“同一个 token ID”在不同父节点下仍是不同状态，因为历史不同。mask 控制信息流，position IDs 则沿深度递增：`prefix_length + depth`，根 depth=0；同层节点可以有相同 RoPE 位置，不能用节点展平序号代替。

标准展平因果三角 mask 会让后面的分支读到前面不相关分支，验证概率便不再对应候选路径。反之，一个只允许祖先的 mask，也不一定对应真正稀疏 kernel：通用 attention 可能先计算完整矩阵再屏蔽，语义稀疏与计算稀疏是两回事。

统一工程 shape：Q=[B,H_q,q,D_h]；K/V=[B,H_kv,S+q,D_h]；展开或分组后的 score=[B,H_q,q,S+q]。S 是旧前缀长度，q 包括物理根。与普通 decode 的 q=1 相比，投影/MLP 工作量近似随 q 增加，attention 还要访问 S 个历史状态和树内节点。GQA/MQA 改变 KV head 数，却不消除每个 query 的注意力工作。

{{DEMO:tree}}

**KV cache 的提交机制。** 临时 KV 在旧长度后保存 q 个位置；选出路径后 gather 这些位置，将其压实到旧前缀末尾，再把所有 layer 的逻辑长度统一改成 `old_length + accepted_count`。官方代码的 `accept_length` 是额外 Medusa 前缀长度，提交包含根的 `accept_length+1` 个位置。未接受的候选可保留在物理缓冲里，但长度指针必须保证它们不可见。

FP16 下，额外候选 KV 字节为 `2 × L × B × q × H_kv × D_h × 2`。代数示例 L=32、B=1、q=65、H_kv=32、D_h=128 为 32.5 MiB；若 H_kv=8，为 8.125 MiB。这里没有加 head 权重、激活、attention workspace 和完整词表 logits。跨步拒绝后通常不需要回算完整旧前缀，但需要正确维护 cache、RoPE 位置和最后 logits。

### 2.2.1 Medusa-1：冻结 backbone，只学习未来头

给定训练 token y_(t+k+1)，每个 head 用 cross-entropy 学习该位置；论文 Eq.1：

{{EQ:loss1}}

λ_k=0.8^k。它降低远期、较难预测 head 在总损失中的权重。训练时对所有有效 t 与 k 聚合，边界位置和 padding 必须 mask；不能在样本末尾读越界标签，也不能把未来 head 训练成同一个 y_(t+1)。输入 hidden 应对 backbone stop-gradient，仅 head 参数更新。

**为什么便宜？** 不保存 backbone 反向所需的整套激活，也不为其建立完整优化器状态；可用量化 backbone 产生表示。**仍有什么成本？** 主模型 forward、K 个 d→V 投影、head 反向、head optimizer 和 [B,T,V] logits。论文举例：60k ShareGPT 样本、Vicuna-7B、单 A100 PCIe，Medusa-1 训练约 5 小时；这是报告条件，不是每个新模型的固定预算。

冻结只说明训练没有更新 backbone 参数。若推理使用不同量化精度、不同 attention 数值路径，或接受规则放宽输出分布，都不能仅凭“冻结”推出完全相同结果。严谨的无损契约要同时指定目标模型、数值计算和验证器。

### 2.2.2 Medusa-2：调整表示，让未来更容易被预测

论文 Eq.2：

{{EQ:loss2}}

L_LM 是 backbone next-token cross-entropy。**按公式，λ₀ 乘的是 Medusa loss**，而不是 LM loss。正文部分文字称“backbone loss 的 weight”，容易误读。将 head loss 加到共享表示上，使 hidden 既支持下一步，又支持多步预测；它会改变 backbone，因此即使严格验证，保证的也是更新后模型。

三项训练策略共同控制扰动：先训练 heads 的 warmup；backbone 与 head 差分学习率；next-token loss 与未来 loss 联合约束。实测主配置先 Medusa-1 再 Medusa-2，以 QLoRA 更新 backbone；head 学习率是 backbone 的 4 倍。原文 warmup 描述在步骤和参数名上不完全一致，复现时应以 Appendix B.3 的两阶段配置为锚点，而不是随意补出一个未给出的 schedule。

**直觉。** 冻结 backbone 是在固定表示上拟合未来；联合训练则改变这套表示，让未来头能更好“读出来”。它提高接受长度的同时，也引入质量保持任务。质量接近的 GPT-4 judge 均分是经验支持，不等价于全输入、全分布的数学不变性。

### 2.2.3 Head 数量：训练上限与实际执行分开

作者经验上建议训练 5 个 head，再按树优化选用 3 或 4 个。远期越不确定，节点预算越容易浪费。若不需要的 head 被真正跳过，其推理开销可以去掉；如果 forward 仍构建所有 head logits 再丢弃，就存在实际开销。“忽略 head”必须落实到执行路径和内存分配。

### 2.3.1 Typical Acceptance：质量接近与分布相同是两种要求

原规则为：

{{EQ:typical}}

p_original 是沿真实候选前缀的目标分布；H 是自然对数熵；ε 是硬阈值；δ·exp(−H) 是熵自适应阈值。满足条件的 token 连续组成前缀，首次失败处截断，所有路径中选最长前缀。论文用 greedy 生成第一个根 token并无条件接受，保证每轮有进展；后续 token 用 typical acceptance。

exp(−H) 是“有效支持集大小”的倒数直觉：若 V 个 token 均匀，H=ln V，exp(−H)=1/V。高熵时阈值降低，更多候选可能合格。但某个具体 candidate 的概率也随温度变化，因此温度增大并不数学保证接受长度单调增加。T=0 应明确分支到 greedy，而不是直接计算 logits/T。

**关键校正 / 原文 §2.3.1 的例子。** 论文称 draft 与 target 相同，独立采样仍会拒绝。这描述的是独立抽样后比对 token，不是标准的严格 speculative rejection sampling [3]。严格接受概率为：

{{EQ:rejection}}

q_D 是该条件下 draft 分布，p 是 target 分布。如果 p=q_D，接受率就是 1；拒绝后从归一化 `(p−q_D)_+` 残差分布采样，才能保持目标分布。独立抽样的相同概率是 Σp_i²，均匀 4 类时只有 1/4。温度可能影响实际 draft-target 差异和系统开销，但不能用独立匹配反例证明标准 rejection 本身低效。

典型接受避免了严格残差校正，并偏好树中最长合法路径。这种输出选择会改变概率质量分配，因而不保证与 target 的原采样分布一致。其价值是经验质量/速度折中。不能把它与 Meister 等人的 locally typical sampling 简单视为同一算法；这里使用概率下限及路径接受，原文直接借鉴的是 truncation / η-sampling 方向。

{{DEMO:acceptance}}

这个代数实验把两种接受含义摆在同一分布上：严格 rejection 在 p=q_D 时始终接受；独立匹配会因分布平坦而下降；typical acceptance 只决定允许哪些 token，不给出输出分布无损证明。

| 配置 | 与什么保持一致 | 保证条件 / 限制 |
|---|---|---|
| Medusa-1 + greedy verifier | 固定 target 的 greedy token 路径 | target 数值语义一致；每步验证到首个 mismatch |
| Medusa-1 + 严格 rejection | 固定 target 的采样分布 | 正确的 proposal、accept/reject 与残差；不是同 seed 路径必然相同 |
| Medusa-1 + typical | 经验质量可能接近 | 冻结不恢复分布等价 |
| Medusa-2 + 严格验证 | 更新后的 target | 不是微调前的原模型 |
| Medusa-2 + typical | 更新模型的放宽输出策略 | 同时有模型变化和采样契约变化 |

### 2.3.2 Self-Distillation：没有原始 SFT 数据时怎么办

先从领域相近的公开 prompts 出发，让原模型生成回答；多轮可以逐轮喂入 seed 用户问题，也可以由具备两角色生成能力的模型 self-talk。Medusa-1 用生成的 hard token 标签训练 heads，避免只依赖不匹配的外部回答。

Medusa-2 若连 backbone 也只用生成 token 做 SFT，可能丢失原分布细节；论文改用 teacher 对整个词表的概率约束：

{{EQ:distill}}

这是 KL(teacher || student)。逐词 KL 对 [B,T,V] 概率求和，再沿有效 token 聚合；teacher stop-gradient。它同时保存正确 token 的相对概率和其他合理候选，而不只是让 student 增大某个抽到的 hard label。不要颠倒 KL 方向。

**共享权重技巧。** student 使用 LoRA，关闭 adapter 即得到 teacher。这样省掉第二份完整 backbone 权重；不表示省掉 teacher forward、soft logits、额外激活生命周期和 adapter/optimizer 内存。若 teacher 也是量化版，保持的是量化 teacher 的分布，不能据此认为恢复了未量化原模型。作者建议蒸馏时优先非量化 LoRA。

### 2.3.3 Optimized Tree：按 rank 的边际命中率分配预算

令 a_k^(i) 为 head k 的第 i 名 token 单独命中率，即 `top-i accuracy − top-(i−1) accuracy`。它不是累积 top-i accuracy。独立假设下，一个 rank 路径的权重为：

{{EQ:tree-score}}

所有深度的节点都算入 I；一个节点被加入时，贡献是其路径概率乘积。用 frontier greedy，每次从连接现有树的可扩展节点中取最大权重，直到预算耗尽。必须 prefix-closed：选远期节点前，它的祖先应已存在。

**为何比均匀树好？** head 深度和 rank 的命中率不平衡。把第 1 head 的第 1 名后面多扩几条，常常比平等扩展低概率早期分支更划算。校准误差、条件相关性与分布漂移则可能改变优先级；树是离线固定的预算分配，不是线上每个上下文都自适应。

**目标函数的边界。** 对 greedy 的单一正确路径、准确概率和独立近似，它可以作为接受长度的代理。typical 允许多条路径重叠且最终取最长时，求和并不严格等于最长接受长度期望；不能把此近似目标当成真实系统最优性的证明。


## 4. Discussion：适用范围由瓶颈决定

作者总结：没有独立完整 draft 模型的 Medusa 更易集成，通过共享 hidden、树验证和轻量训练提升 LLM 生成效率。**工程分析：** 它仍需要 tree mask、特殊 position IDs、候选索引、接受规则、KV 重排和下一轮 logits 管理；“结构简单”相对于两套完整模型部署成立，不能理解为无需修改 decoding runtime。

可用前提是 target 验证额外工作的边际成本较低，而且 heads 的接受贡献足够大。高 batch、长上下文、昂贵 attention、很小 target、受限词表投影或慢 CPU/GPU 同步都可能压低收益。训练适配需求也使它不等于“即插即用支持所有模型”。

Impact Statement 讨论更低成本、更可获得的生成能力。论文没有实际能耗或端侧电池测量；减少 wall time 不必然同比减少功耗，不能把成本、能耗和吞吐当作互相等价的实测指标。


## Appendix A. Related Work：同一目标有多条优化轴

### A.1 LLM Inference：KV、量化、IO 与并行生成

论文讨论 attention IO 优化、KV 减量/压缩、量化和推理引擎，这些分别减少中间矩阵读写、每 token 历史状态、权重字节或调度浪费。Medusa 改变每次读权重后的有效产出，理论上可与这些路线组合，但收益通常不能相乘：一种优化可能让另一个瓶颈突出。

例如 GQA 缩小 KV，使 attention bandwidth 下降；weight-only 量化压缩权重，使 CPU top-k、head 输出投影或候选 gather 的比例上升；FlashAttention 免去显式 score 存储，但仍要满足 tree mask 的语义。PagedAttention 管理 KV page 不等于自动理解一棵非连续候选树。

### A.2 Sampling：接受集合与输出分布

论文的 typical acceptance 借鉴概率截断方向，核心是“概率够高则候选可接受”。strict speculative sampling 的目标是完全匹配 fixed target 分布，而本文扩展选择接受较合理的最长路径。两类设计解决的输出契约不同。本文贡献在系统/质量折中上有意义，其合理性应由任务评估支持，而不是错误地用“p=q 也会拒绝”来证明。


## Appendix B. Experimental Settings：复现需要哪些具体数值

### B.1 Baseline 与速度协议

主要 baseline 使用默认 Hugging Face 路径；标准 speculative baseline 还尝试不同 draft 大小和草稿长度，并使用编译降低 draft overhead，Appendix D 有具体结果。不同模型、sampling、dtype、GPU 和版本对每张主表的对应披露不够完整。附录算子 profiling 的 A100/A40/A6000 不自动证明主实验也分别用了这些配置。

### B.2 所有模型的共同训练配置

| 项目 | 原论文报告 |
|---|---|
| 框架 / optimizer | Axolotl；8-bit AdamW |
| LR schedule | cosine + warmup |
| Heads | 5 个，每个 1 层；λ_k=0.8^k |
| 联合训练 | LoRA 或 QLoRA；heads LR 为 backbone 4× |
| LoRA 范围 | backbone 全部 Linear，含 LM head |
| LoRA 参数 | rank=32，α=16，dropout=0.05 |

### B.3 Vicuna-7B/13B

原文配置为 global batch 64、backbone peak LR 5e−4、heads 2e−3、warmup 40 steps、4-bit backbone；先 Medusa-1 初始化，再 Medusa-2 QLoRA，Eq.2 的 λ₀=0.2。模型基于 Vicuna v1.5，训练 2 epochs，sequence length 4096。数据与模板边界应对齐其 ShareGPT 配置，避免未来位置标签跨对话边界。

### B.4 Vicuna-33B / Zephyr-7B

采用自蒸馏，约 100k 样本、sequence length 2048、global batch 128；backbone LR 1e−4、head LR 4 倍、warmup 20 steps、λ₀=0.01。这里使用逐渐增大的 sine schedule 代替主实验两阶段方案，但文中 θ₀ 的命名没有充分定义。该缺口标为复现需核对，不能自行补全精确 schedule。这里训练策略与 B.3 不同，应按模型列分别记录。


## Appendix C. Sparse Tree：预计算拓扑，让运行时只用索引

{{FIG:6}}

**Figure 6 深读。** 每个节点表示某个 head 的 top-k rank；四层深度对应 4 个活跃未来 heads，64 个候选节点由校准命中率决定。红线标出一条正确未来路径；树在早期高概率分支扩展更多。读图时看“扩展落在哪些父节点”比只数叶子更重要。正文的虚拟根与 runtime LM 根有计数差别，部署时应检查实际 buffer length。

稀疏树使用 rank 路径，而不把具体 token 词汇固化；运行时 top-k 换成真实 token ID。可预先生成 `tree_attn_mask`、`tree_indices`、`tree_position_ids`、`retrieve_indices`。树节点存储顺序与路径逻辑顺序可以不同，retrieve 索引负责映射。

官方实现 pipeline 中，candidate selection 读取已有 logits，tree decoding 产生所有节点 hidden，再由 posterior evaluation 选 best path，最后 update inference inputs 更新 tokens 和 KV。这条链路是移植时最值得先核验的地方。

