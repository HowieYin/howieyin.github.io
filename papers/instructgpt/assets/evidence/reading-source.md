<p class="eyebrow">OPENAI · NEURIPS 2022 · ARXIV 2203.02155 V1 · RLHF</p>

# InstructGPT：把会续写的模型，训练成更愿意完成任务的助手

> **一句话。** 用人类示范规定回答形式，用偏好排序学习奖励，再以带 KL 约束和预训练梯度的 PPO 改变 GPT-3 的输出分布，使它更符合特定标注者对用户意图的判断。

论文：**Training language models to follow instructions with human feedback**。作者：Long Ouyang、Jeffrey Wu、Xu Jiang、Diogo Almeida、Carroll L. Wainwright、Pamela Mishkin、Chong Zhang、Sandhini Agarwal、Katarina Slama、Alex Ray、John Schulman、Jacob Hilton、Fraser Kelton、Luke Miller、Maddie Simens、Amanda Askell、Peter Welinder、Paul Christiano、Jan Leike、Ryan Lowe。机构：OpenAI。指定版本：2022-03-04 的 **2203.02155v1.pdf，68 页**；后续发表于 NeurIPS 2022。本解读的阅读和扩展检索日期：**2026-10-29**。

<div class="source-downloads"><a href="https://arxiv.org/pdf/2203.02155v1">指定 PDF</a><a href="https://arxiv.org/abs/2203.02155">arXiv</a><a href="https://github.com/openai/following-instructions-human-feedback">官方资料仓库</a><a href="assets/evidence/2203.02155v1.pdf">PDF 本地存档</a><a href="assets/evidence/reading-source.md">解读 Markdown</a></div>

**阅读约定。** 主体遵循原文 Abstract → §1–5 → Appendix A–F 的顺序；工程重建和扩展研究放在之后。“原文”指指定 v1，“解释 / 推导”指本解读，“扩展”指单独链接的一手资料。50 张编号原图、14 张原表和 2 个编号公式全部保留。交互中的数字是数学示例，不是训练复现或 GPU 性能实测。文字图提供完整可复制转录；图注独立呈现于图片之外。

## Abstract — 模型规模与用户意图是两条轴

**一段话。** GPT-3 学的是互联网文本的下一 token，而用户往往希望模型执行任务、遵守约束并在不确定时说清楚。作者收集少量高质量示范进行 SFT，再让标注者比较模型回答，训练一个能给整段回答打分的奖励模型 RM，最后用 PPO 优化策略，同时约束它偏离初始策略的程度。加入预训练数据更新的 PPO-ptx 在 API 提示分布的人类偏好上明显胜过 GPT-3，甚至 1.3B 模型能胜过 175B GPT-3；但这没有证明小模型掌握了更多知识，也没有证明它在所有任务上更真实、更安全。

五个抓手：

1. **目标错配。** 续写真实文本不等于完成用户指令；更低语言建模损失不能直接代表更好助手。
2. **三种监督。** SFT 学示范 token，RM 学回答之间的次序，PPO 学在当前策略生成的回答上提高奖励。
3. **奖励与约束。** RM 偏好分数、reference KL、PPO clipping、ptx 梯度各管不同问题。
4. **结论依赖评测。** API win rate、Likert、TruthfulQA、毒性分数和公开 NLP 指标不可互换。
5. **对齐有对象。** “human feedback”来自具体承包标注者、指令规范与用户样本，并非全人类的统一价值函数。

### Figure 1 — 首图到底比较谁？

{{FIGURE:1}}

横轴是 1.3B、6B、175B 模型规模；纵轴是回答**相对 175B SFT** 被偏好的概率，0.5 是该参照下的平局线。每条线先固定训练方法，再比较规模；再在同一规模纵向比较方法。原文所有误差条采用 95% 置信区间。

最关键的信号是 PPO / PPO-ptx 的曲线高于普通 GPT-3，说明任务适配能大幅改变输出效用。**这张图不是“对 GPT-3 175B 的直接胜率图”。** 正文另外报告 175B InstructGPT 对 GPT-3 的直接偏好率 **85±3%**，对 few-shot GPT-3 为 **71±4%**。1.3B 与 175B 参数比为 134.6 倍；论文“100× fewer”是数量级表达。小模型胜出的边界是这些 API prompts 与偏好标准。

## §1 Introduction — 从“预测文本”到“理解要完成什么”

预训练把各种写作目的混在同一概率分布：说明文、虚假论断、辱骂、代码、对话都可能是合理续写。用户交给模型一句任务时，模型仍可能把它当成文章开头，重复任务、补写另一个问题，或编造貌似合理的答案。扩规模增强建模能力，但不会自动改变何种行为受到奖励。

作者把目标表述为 helpful、honest、harmless。三者并不总一致：遵从有害要求有助于“执行指令”，却违背 harmlessness；不回答可减少错误，但降低 helpfulness。因此实验既看整体偏好，也记录任务执行、约束遵守、封闭领域幻觉和不适宜内容，不能只用一个总分解释全部行为。

### Figure 2 — 三阶段中谁在被训练？

{{FIGURE:2}}

从左至右读：标注者先给 prompt 写示范，更新 SFT policy；再对 A–D 等多个采样回答排序，更新 RM；最后 policy 生成回答，RM 给分，PPO 更新 policy。蓝箭头表示**监督数据流向可训练模型**，不是推理时每次都要访问人类。排序回答与 RM 的输入是同一 prompt 下的 completion；PPO 的输入 prompt 无须再附人写目标答案。

图隐藏了实际 PPO 中的 frozen reference、value critic 和预训练数据梯度，后面的工程图会补齐这些依赖。部署后提供回答的是 policy；SFT、RM 和 critic 不会依次替用户“审稿”。论文的创新在于把已有 RLHF 路线推广到大规模通用 API 指令任务，并系统检验其收益和代价。

## §2 Related work — 哪些技术已存在，哪些证据是新的？

原文沿三条线定位工作。第一条是从人类偏好学习奖励并优化策略：2017 年的人类偏好 RL、2019 年语言模型微调、2020 年摘要 RLHF 已建立示范 / 比较 / 优化的基础。第二条是 instruction tuning：FLAN、T0 等把有标注任务改写成指令，提高跨任务泛化。第三条是安全、真实性和偏见评测，提醒“被喜欢”不能完全替代这些维度。

区别在监督来源与覆盖范围。FLAN/T0 主要利用既有 NLP 数据集的任务模板；这里的示范与比较更贴近用户在 Playground 中实际提出的开放任务。PPO 又允许策略对同一 prompt 生成新的候选，再通过 RM 获得信号，超出只拟合已有示范的范围。代价是引入一个可被优化器钻空子的代理奖励，以及远复杂于 SFT 的训练生命周期。

## §3.1 High-level methodology — 先教形式，再学比较，再移动分布

SFT 的输出仍是词表概率。RM 则把 prompt 与完整回答连接起来，通过 GPT-3 backbone 和 scalar head 得到一个 reward。RM 学的是“在这类 prompt 上，标注者更喜欢哪一个回答”，不是输出每句事实的可信度。PPO 以当前模型的采样结果作为训练轨迹；PPO-ptx 额外混入原预训练分布的语言建模梯度。

论文默认 **InstructGPT = PPO-ptx**；不含预训练混合的 variant 称 PPO。读图时需逐条辨别 GPT、GPT prompted、SFT、PPO 和 PPO-ptx，不能将所有经过监督的模型都当成同一个处理组。

## §3.2 Dataset — prompt 来源决定了优化方向

数据来源有两类：承包标注者编写的任务和用户在 OpenAI API **Playground** 提交的 prompts。这里并不包含生产客户流量。早期用户提示常为 GPT-3 设计 few-shot 续写；模型可用后，用户也开始直接对 InstructGPT 下指令。作者在两个 prompt 分布上评测，避免把更会写 GPT-3 提示的样本当作全部用户需求。

正文说按 **user ID** 切分 train/valid/test，并限制每 user 约 200 prompts；附录则按 **organization ID** 表述同类步骤。两种描述的隔离粒度并不完全统一，应保留这种差异，不能声称已严格核验 user 和 organization 双重隔离。过滤、近重复处理与 PII 检查有助降低污染，但并不证明不存在泄漏或选择偏差。

{{TABLE:1}}

Table 1 的百分比描述 API 用途，不是公开 NLP benchmark 的混合比例。Generation 45.6%、Open QA 12.4%、Brainstorming 11.2% 合计 69.2%，使开放回答的写作与任务完成质量成为主要监督来源。Classification 3.5% 较少，不能据此推断 RLHF 会最优地保持所有分类能力。比例有四舍五入，逐项相加为 99.9%。

{{TABLE:2}}

Table 2 的三个例子是**基于真实用途创作的虚构提示**：brainstorming 的数量要求、generation 的剧情要求、rewrite 的格式要求。它们说明标注者需要评价约束和任务意图，但不能拿这些示例估算实际数据难度。

## §3.3 Tasks — 同一个助手要承接多种成功标准

任务包括生成、开放 / 封闭问答、头脑风暴、聊天、改写、摘要、分类和抽取。封闭问答与摘要可检查是否忠实于给定材料；开放问答要依赖外部知识；创意任务通常没有唯一答案。于是“正确 completion”很难由一份统一 ground truth 覆盖，比较判断比要求人逐字写出唯一答案更灵活。

**解释。** 灵活监督也会混入写作风格、长度、礼貌和标注者个人倾向。若更长的回答更常获胜，RM 可能将长度当成质量；若承包商对某主题知识不足，模型可能学会更像正确答案，却未变得更正确。这是后续奖励泛化和真实性审计的动机。

## §3.4 Human data collection — 有操作规范的判断，仍有分歧

约 40 位承包标注者经筛选与试任务参与；先写示范，再对每 prompt 的 **K=4–9** 个模型回答排序。最多可展开 6–36 个无向成对比较；ties 被允许，但训练 RM 时丢弃相等的 pair。标注者对同一 prompt 的回答给 Likert 分与 metadata，再综合排序。

{{TABLE:3}}

Table 3 列出 overall quality 的 1–7 分和其余 binary 标签。它拆出了“尝试正确任务”“满足约束”“助手场景合适”“封闭领域幻觉”等维度，也含色情、暴力、有害建议、观点和道德判断。**含观点不自动等于错误**，这是辅助标签。总体偏好与每个 binary 标签并非完全等价。

训练标注主要强调 helpfulness，最终评估则更优先 truthfulness / harmlessness，存在目标口径变化。论文报告训练标注者一致率 **72.6±1.5%**、held-out 标注者 **77.3±1.3%**；这是人类判断的一致程度，不是 RM accuracy。分歧既可能来自噪声，也可能来自合理的价值差异。

## §3.5 Models — 两个原公式，四种不同约束

### SFT：teacher forcing 的目标与助手体验并不完全同向

给定 prompt \(x\) 和示范 \(y\)，解释性损失是回答 token 的交叉熵，通常 mask 掉 prompt 的 loss。输入 `[B,T]` token IDs，经 decoder 得到 `[B,T,d]` hidden states，再得 `[B,T,V]` logits；response mask 决定哪些位置参与监督。SFT 教模型在用户提示后如何直接回答。

论文展示的 SFT 基线训练 **16 epochs**，尽管验证 NLL 在约 1 epoch 后已过拟合，人类 / RM 评价仍改善。这说明模型选择指标可以与生成质量不同，但不说明“过拟合越多越好”。实际 PPO 初始化另用 **2 epochs、10% ptx** 的 SFT 变体，不能把 16-epoch 基线当成所有 PPO 的唯一初始模型。

### 原式 (1)：从排序学到 Bradley–Terry 奖励差

{{EQUATION:1}}

\(x\) 是 prompt；\(y_w,y_l\) 是同一 prompt 下获胜和落败的回答；\(r_\theta(x,y)\) 为 scalar reward，\(\sigma\) 为 sigmoid；\(K\) 是这条 prompt 的候选数。把 reward difference \(\Delta=r_w-r_l\) 放进 sigmoid，相当于估计 \(P(y_w\succ y_l)=\sigma(\Delta)\)。当 \(\Delta=0\)，概率 0.5、loss 约 0.693；差值 2 时概率约 0.881、loss 约 0.127。

RM 可将 `[B,K,T]` 展平为 `[B·K,T]`，在每个 completion 的末端取 `[B·K,d]` 特征，再通过线性 head 输出 `[B,K]`。pair index 从这些 K 个 scalar 上取差；每个 completion 只做一次 backbone 前向。同 prompt 的 pairs 高度相关，不能视为独立新增了 36 份任务知识。

<div class="callout warning"><strong>原式归约审计。</strong> 原式在 expectation 外又写 1/binomial(K,2)。若 expectation 已经是 pairs 的均值，再除一次就会 double averaging。规范实现应明确“每 prompt 的有效 pair loss 求和 / 有效 pair 数，再按 prompt 平均”；丢弃 ties 后尤其要定义分母。这里保留原式，下面代码与交互明确采用有效 pairs 的均值，不偷偷重写原文。</div>

{{DERIVED:ranking}}

加同一个 bias \(c\) 到所有 reward，差值不变，pairwise likelihood 也不变。因此绝对 reward 原点不可识别；作者在 RL 前让示范的平均 reward 为 0，这是校准偏置，不是生成了绝对效用刻度。原式也不能证明不同 prompts 的 reward 尺度完全一致。

{{INTERACTIVE:ranking}}

```python
# Explanatory grouped RM loss; B prompts, K completions per prompt.
scores = rm(tokens.reshape(B * K, T), end_positions).reshape(B, K)
prompt_losses = []
for b in range(B):
    pairs = strict_preference_pairs(rankings[b])  # remove ties
    delta = scores[b, pairs.winner] - scores[b, pairs.loser]
    if len(pairs):
        prompt_losses.append(softplus(-delta).mean())
loss = stack(prompt_losses).mean()
loss.backward()          # one grouped backward; do not update per pair
optimizer.step()
```

pair head 的计算为 \(O(BK^2)\)，backbone 约为 \(O(BKTd^2+BKT^2d)\)，不应为了 pair 数平方而重复做 K 倍 backbone。grouping 减少共享回答的重复前向，也避免同 prompt 的高度相关 pairs 在连续更新中过拟合。内存仍随 completion 数和序列长度增长；“只算一次”不表示所有回答共享同一套训练 activation。

### 原式 (2)：reward、reference KL 和 ptx

{{EQUATION:2}}

\(\phi\) 是可训练 policy 参数，\(\theta\) 是冻结 RM 参数；\(D_{\pi_\phi^{RL}}\) 表示 prompt 与当前 policy 采样回答的联合分布。第一项提高 learned reward，第二项惩罚相对 SFT reference 的变化，第三项用 \(D_{\rm pretrain}\) 中原文本的 log likelihood 保持语言建模能力。纯 PPO 令 \(\gamma=0\)，PPO-ptx 用 \(\gamma=27.8\)。

这里的 \(x\) 在 ptx expectation 中指预训练文本，与第一项的用户 prompt 语义不同。\(y\) 是整段 completion；自回归 sequence log probability 是 response token log probabilities 的**和**，不是概率直接相减或逐 token 简单平均：

{{DERIVED:kl}}

单个 sample 的 log ratio 可以负。只有对当前 policy 的完整分布取期望才得到非负的 forward KL；不能在代码里对每个采样值 clamp 到正数。PPO rollout 会把 \(-\beta(\log\pi_{\rm old}-\log\pi_{\rm ref})\) 分配为 token reward，并在终止位置加 RM 分数；更新时使用旧轨迹及相应 advantage。公式 (2) 是目标描述，不能直接当作完整 PPO 实现。

{{INTERACTIVE:kl}}

### PPO clipping 的 reference 是另一个对象

reference policy 冻结在 PPO 初始化状态，用来约束**累计偏移**；old policy 是本轮采样时的 snapshot，用来计算**本次更新的 importance ratio**。old policy 每轮变化，reference 不变。

{{DERIVED:ppo}}

\(\rho_t=\pi_\phi(a_t|s_t)/\pi_{\rm old}(a_t|s_t)\) 是 response token 的 ratio；\(A_t\) 是 critic 估计的 advantage，clip \(\epsilon=0.2\)。正 advantage 的 token 概率变大过多会被 clipping；负 advantage 反向约束。critic 拟合 returns，policy 使用 detached advantages；把 reference ratio 当成 PPO old ratio 会改变优化算法。

最后一项 ptx 是另一批预训练文本上的梯度。作者每个 RL episode 配 **8 个 ptx examples**，两路 gradient 依次计算再累积。**27.8 是 loss coefficient，不是 27.8% 数据混合，也不是 27.8 倍 tokens。** ratio 8 与 gamma 27.8 分别控制采样量和梯度权重。

## §3.6 Evaluation — 一种指标回答一种问题

API 分布上主要比较 human preference 与 Likert；公开任务上检查 TruthfulQA、RealToxicityPrompts、Winogender、CrowS-Pairs 和传统 NLP。held-out 标注者用于检测对训练标注者的过拟合，但仍来自同一承包标注者总体，不代表对所有语言和人口的普适性。

| 指标 | 测量对象 | 读数方向 | 不能自动推出 |
|---|---|---|---|
| Pairwise win rate | 相对某个固定模型的偏好 | ↑ | 绝对正确率、任意参照的胜率 |
| Likert 1–7 | 单个回答的整体质量 | ↑ | 可跨标注组直接校准的 utility |
| True / true+info | 真、以及真且有信息 | ↑ | 长篇逐事实准确率 |
| Perspective toxicity | 自动模型判定的毒性 | ↓ | 用户主观伤害或所有安全维度 |
| Bias entropy | 对选择分布的不确定性 | ↑，作者作为低偏向代理 | 公平性已经得到证明 |
| Accuracy / F1 / BLEU / ROUGE-L | 具体任务匹配或文本质量 | ↑ | 用户会更喜欢该回答 |

开放回答没有统一 binary ground truth，原文未提供足以计算全部 TP/FP/TN/FN 的逐题预测，因此不能凭总体 win rate 伪造 Precision、Recall 或 F1。表中的 QA F1 是答案 token overlap，不是“用户指令是否成功”的分类 F1。

## §4.1 Results on the API distribution — 同规模优势与跨规模优势

### Figure 3 — prompt 分布和标注组都要拆开看

{{FIGURE:3}}

左列是提交给 GPT-3 的 prompts，右列是提交给 InstructGPT 的 prompts；上排 held-out labelers，下排 training labelers。每个面板仍以 175B SFT 为参照。观察方法曲线在四个条件下是否一致，而不只是挑最有利面板。左列省略 GPT prompted，是因为这些 prompts 本来已为 GPT-3 设计。

PPO 方法整体受到更多偏好；小 PPO-ptx 可超过大 GPT 的事实说明行为目标具有独立收益。175B 模型相对普通 GPT-3 的直接 85±3% 偏好，是另外的比较统计。held-out 结果支持向相近标注者泛化，但同样的训练指南和承包商来源限制了外推范围。

### Figure 4 — preference 改善究竟落在哪些行为？

{{FIGURE:4}}

按 metadata 的含义看方向：inappropriate、fail task、hallucination 越低越好，constraint satisfaction 越高越好。模型规模被合并，是为了应对部分标签的样本不足，不表示所有规模效果一样。封闭领域“幻觉”指输入材料之外捏造信息，不等于对全知识域事实错误的完整审计。

结果把 overall preference 拆成较具体的行为：更常执行正确任务、更守约束、更适合助手、更少封闭领域编造。它们支持“输出分布移动到了目标行为”的解释，却不单独证明语言模型内部推理能力增加。

### Figure 5 — 与 FLAN / T0 的比较边界

{{FIGURE:5}}

纵轴改为 1–7 Likert，不能与 Figure 3 win rate 相减。FLAN 和 T0 高于未提示的 GPT-3，接近经过 instruction-following few-shot 提示的 GPT-3；InstructGPT 在该 API 任务分布上更好。

比较同时改变了基础模型、训练数据来源、任务分布与训练技术，因此不能把全部差距归因于“PPO 比 instruction tuning 强”。更公平的因果实验应固定 backbone、数据和预算，只改变 SFT 与偏好优化路径。

## §4.2 Results on public NLP datasets — 真实性、安全性与能力保留

### Figure 6 — true 与 true+informative 要一起读

{{FIGURE:6}}

灰柱测 truthfulness，彩柱测同时 true 且 informative。仅回答“不知道”可能提升 true，却不一定提供有用信息。大模型 PPO 系列提升明显；1.3B 的 human TruthfulQA 中 PPO-ptx 略低于同规模 GPT，不能宣称每个模型都改善。附录自动评测另有 prompt 模板与自动 judge，不能把两者当作同一个分数。

### Figure 7 — respectful 指令是重要实验变量

{{FIGURE:7}}

比较同一批 **1,729 prompts** 的 human ratings 与 Perspective API 输出，对三个 175B 模型分别加入或不加入 respectful instruction。需先固定提示条件，再比较方法；再看 human 与自动判定方向是否一致。这里自动评测也只用同一批 1,729 prompts，因此会与 Appendix Table 14 的 5,000-prompt 自动评测有差异。

主要收益出现在明确要求 respectful 时。Table 14 的 175B GPT → PPO-ptx toxicity：basic **0.231→0.234**，并未下降；respectful **0.233→0.196**，下降 0.037，约 15.9%；biased **0.285→0.400**，上升 0.115，约 40.4%。这显示指令遵从也会增强对不良要求的执行能力。

公开 NLP 上，纯 PPO 在 DROP、SQuAD、翻译等任务出现退化，PPO-ptx 可缓解但不保证恢复全部基线。bias entropy 结果没有给出一致显著改善。对齐收益因此是一个多维 trade-off，不是单个性能开关。

## §4.3 Qualitative results — 泛化样例与简单失误同时成立

### Figure 8 — 语言迁移与代码解释

{{FIGURE:8}}

逐段读 prompt、GPT-3 completion 与 InstructGPT completion。法语指令和代码解释显示行为适配可以迁移到训练主要未覆盖的形式；作者也注明代码问答的 InstructGPT 回答并不完全正确，普通 GPT-3 约 50% 时间会回答该问题。能回应问题与回答正确仍是不同事件。

### Figure 9 — 错误前提与过度保守

{{FIGURE:9}}

两个错误分别是跟随假前提，以及对简单问题过度 hedge。模型可能把“配合用户”看得比“纠正问题”更重要，也可能把谨慎表述当成避免被判错的手段。GPT-3 未被专门提示进入 QA mode，不能仅凭样例断言其最佳提示下表现更差。

这两张图的 prompts 被有意选择，outputs 没有 cherry-pick。它们说明行为存在，不能估计行为发生频率，也不能当作多语言或代码 benchmark。

## §5 Discussion — 对齐对象、规模与社会边界

### §5.1 Implications for alignment research

作者强调迭代式对齐的经验反馈：先改进当前系统，再观察人类偏好、代理奖励和能力退化的关系。SFT 报告 **4.9 PF/s-days**，PPO-ptx **60 PF/s-days**，GPT-3 预训练 **3,640 PF/s-days**。重算比例为 0.1346%、1.6484%，合计约 1.7830%。相对预训练较少，不代表工程简单或所有实验、标注与部署成本都包含在这里。

小模型经对齐可更受欢迎说明参数与意图适配是不同维度；但这条经验不能直接推导到超人能力系统，其输出更难让人核验，RM 的盲区可能更大。

### §5.2 Who are we aligning to?

实际对齐对象包含用户 prompt 分布、研究者规定的标注原则，以及筛选后的承包标注者。标注者间存在分歧；系统把这些判断压成单个 scalar reward，就隐含了偏好聚合选择。用 held-out labelers 不等于跨文化代表性验证。

### §5.3 Limitations

模型仍会犯简单错误、响应错误前提、输出不实内容、过度拒答、产生偏见或被有害指令诱导。数据以英语为主，有限 API 分布与有限评估组可能覆盖不足；奖励模型能过拟合特定标注风格。纯 PPO 的能力回归也是技术限制，不能全部归为“安全代价”。

### §5.4 Open questions

谁来决定冲突偏好的权重？人类无法验证的复杂任务如何监督？如何保证对新 prompt、不同用户和更大模型的奖励泛化？怎样减少 reward hacking，同时保持基本能力？论文提出方向但未证明答案。

### §5.5 Broader impacts

更好执行任务降低使用门槛，也可能让恶意用户更容易获得符合其目的的输出。符合指令与符合安全规范需明确优先级。开放反馈和外部审计有助暴露盲区，但无法由单次训练消除全部风险。

Acknowledgements 与 References 列出标注、研究协作及前人工作；下面继续进入全部附录，它们提供了比主文流程图更精确的配置与评测口径。

## Appendix A — Additional prompt data details

### A.1 Labeler-written prompts

承包商编写 plain、few-shot 和 user-based prompts。plain prompts 增加任务表达多样性；few-shot prompts 用上下文示范指定任务；user-based prompts 围绕 API 用途补充示范。这样可以冷启动 instruction 数据，但人为编写数据与真实用户短提示的长度、分布不同。

### A.2 API user prompts

附录描述 organization 粒度限制和去重；用户 PII 处理与质量过滤是收集步骤，不应与后面的训练长度过滤混在一起。下面两张 longtable 保留全部提示示例，花括号中的字段是替代内容。

{{TABLE:4}}

Table 4 **没有独立原 caption**，本页标题是识别补充。按 use case 读：brainstorming / generation 要求结构与创作；classification / extract 要求离散标签或忠实抽取；rewrite / summarization 要求保持信息；chat / QA 还涉及角色与上下文。表内有续写式 prompts，也有直接命令，说明 InstructGPT 用户分布不是纯聊天模板。

{{TABLE:5}}

Table 5 同样无独立 caption；这里完整保留 GPT-3 分布的 few-shot、续写与角色提示。与 Table 4 比较表达形式即可，不应用有限列举计算两个分布的真实差异。对 GPT-3 更有利的模板是 baseline fairness 的重要变量。

### A.3 Dataset sizes

{{TABLE:6}}

按 prompt 而不是 pair 计：SFT train **12,725**，valid **1,653**；RM train **33,207**，valid **17,887**；PPO train **31,144**，valid **16,185**。SFT 总数由 11,295+1,430 构成，RM train 由 6,623+26,584 构成。RM 产生更多 pairs，不意味着这些 pairs 是独立 prompts；PPO 不需要每个 prompt 都有人工示范。

### A.4 Data diversity

{{TABLE:7}}

Table 7 描述 ambiguous、sensitive、closed domain、continuation、constraints 等标注比例。破折号表示未标或缺失，不是 0。RM train 的 explicit other constraints 26.3%，SFT train 25.6%；这类约束使 metadata 有具体依据。Closed domain 在 SFT valid 40.6%、train 27.4%，提示 split 之间并非所有属性完全一致。

{{TABLE:8}}

Table 8 按 customer 计平均 prompt 数。RM / PPO valid 平均 27.96 / 31.55，高于 train 5.35 / 6.01；测试仅 1.81。总体 prompt 数不等于同样多独立用户。置信区间和泛化结论若按 prompt 处理，也需考虑同一用户内的相关性。

{{TABLE:9}}

Table 9 单位为 tokens，测试集 **3,196**；测试 prompt mean 115、median 49，比 SFT train mean 408 更短。最大长度到约 2k 是**收集阶段统计**；实际训练过滤超过 1k 的 prompts、response 最多 1k，模型 context 2k，这两者不矛盾。

{{TABLE:10}}

Table 10 摘要 mean 424、chat 386、closed QA 325，明显长于 brainstorming 83、open QA 89。任务成本取决于输入长度和输出长度，不能仅看 Table 1 的任务百分比估计 GPU 占比；dense attention 的 prefill 对长输入更敏感。

{{TABLE:11}}

Table 11 承包商 prompt mean 437、demo mean 38；customer prompt mean 153、demo mean 88。长上下文 / 短输出与短上下文 / 长输出的训练成本不同。表中 count 与 SFT split 汇总也不是同一切分口径，不宜强行逐格等同。

语言检测约 96% 英语；作者估计真实英语占比可能 ≥99%。字符脚本和短文本会影响 langid，因此不能把剩余 4% 直接当成覆盖充分的多语言监督集。

## Appendix B — Additional human data collection details

### B.1 Labeler selection

筛选试任务检验敏感内容处理、指令执行和判断稳定性；研究者持续提供反馈。这样可提高监督一致性，也会把研究团队规范传播给整个标注群体。相近标注者的 held-out 泛化，应在这种共享指南背景下理解。

### B.2 Labeling instructions

{{FIGURE:10}}

Figure 10 是最终 API 评价指南原文：先识别用户任务，再按输出的总体质量、真实性、伤害及约束判断。它把“好回答”转成可操作规范，是理解偏好 label 的必要证据。训练与评估的优先级变化意味着 RM 的训练排序和最终 human evaluation 未必完全同目标。

{{FIGURE:11}}

Figure 11 是完整 RealToxicityPrompts 标注指南。它要求在上下文中评价 toxicity、continuity 和相对输入的毒性，并处理引用或描述等情况。Perspective 分数没有等价地包含全部上下文判断，因此 Figure 7 的 human / automatic 两栏必须同时保留。

### B.3 Demographic data

{{TABLE:12}}

Table 12 只来自 **19 位自愿回应者**；部分问题缺答、多选，百分比可有不同分母，族裔比例不一定加到 100%。18–34 岁合计 73.7%；本科 52.6%、硕士 36.8%。这不是全部约 40 位标注者的完备人口画像，也不是全球用户的代表性抽样。

### B.4 Satisfaction survey

{{TABLE:13}}

Table 13 同为自愿问卷：对任务清晰、满意等评价较好，但也有 31.6% agree 任务 repetitive，10.5% 对报酬公平持中立。问卷支持参与体验的局部描述，不能当成独立劳动审计或排除自选择偏差。

### B.5 Web interface

{{FIGURE:12}}

上图界面先显示单个 completion，收集 1–7 Likert 与 metadata；下图对同 prompt 的回答做 ranking。ties 被鼓励，防止强迫标注者制造无意义次序；RM training 却丢弃 ties，因此模型未直接学习“同等质量”概率。双阶段界面也可能有顺序或锚定影响，需随机化候选顺序并审计。

## Appendix C — Additional model details

所有 policy 沿用 GPT-3 decoder，context **2k**；训练过滤 prompt >1k，completion cap 1k。权重与 activation FP16，FP32 master weights，Adam \((\beta_1,\beta_2)=(0.9,0.95)\)。论文未提供可精确重建端到端时延的 GPU 型号、并行布局和逐算子 profile。

### C.1 SFT

| 规模 | LR | Batch | Epochs | 其它 |
|---|---:|---:|---:|---|
| 1.3B / 6B | 9.65e−6 | 32 | 16 | residual dropout 0.2 |
| 175B | 5.03e−6 | 8 | 16 | 同类 cosine schedule |

cosine 衰减至峰值的 10%，无 warmup；最终以 RM validation 选模型而非最小 NLL。SFT 全部 response 的 teacher forcing 与 PPO 自回归采样有不同硬件利用率。

### C.2 RM：实际初始化并非图示的唯一简化版本

正文用“从 SFT 替换末端 head”说明 RM；附录精确记录最终 **6B RM** 从经过 ARC、BoolQ、CoQA、DROP、MultiNLI、OpenBookQA、QuAC、RACE、Winogrande 微调的 GPT-3 6B 初始化。作者说明是历史原因，GPT-3 或 SFT init 得到近似结果。不能省略实际设置，也不能用简化图宣称实验始终只有 SFT→RM 这一条初始化路径。

RM train **1 epoch**，LR **9e−6**，batch **64 prompts**，每 prompt 最多 36 pairs，故最多 2,304 pairs；这个 batch 不是 64 个独立 pair。训练更久会过拟合。6B RM 固定服务不同尺寸 actor，控制 reward capacity，也产生 reward model 小于 175B actor 的能力瓶颈。

### C.3 PPO initialization

PPO 使用 **2 epochs SFT +10% ptx** 初始化；这一步 1.3B / 6B / 175B 的 SFT peak LR 分别为 **5e−6 / 1.04e−5 / 2.45e−6**。这些不是最终 PPO actor LR。初始 policy 同时决定 reference 和初始 rollout 分布，改变它会影响探索范围、语言能力与可承受 KL，不能仅作为无关实现细节。

### C.4 RLHF

| 配置 | 原文最终设置 | 解释 |
|---|---|---|
| Actor | 1.3B / 6B / 175B | 唯一最终生成 policy |
| RM / critic | 各 6B | RM frozen；critic 从 RM 初始化并更新 |
| Reference | frozen PPO initialization | 原式简记为 SFT |
| Episodes | 256k | 每条是一个 prompt/completion trajectory |
| RL batch / minibatch | 512 / 64 | 1 inner epoch |
| PPO actor LR | 见 E.9 的扫描与 human Likert 选择 | 不混同 C.3 初始化 SFT LR |
| Critic LR | 小 actor 9e−6，大 actor 5e−6 | 单独优化 |
| Sampling | temperature 1 | response max 1k |
| PPO clip / KL beta | 0.2 / 0.02 | 两种不同限制 |
| LR warmup / EMA | 首 10 iterations / 0.992 | 训练实现设置 |
| Discount | 无 discount | 完整 completion reward |
| ptx ratio / gamma | 8 / 27.8 | 样本量与 loss 权重 |

按 256k=256,000 重算：**500 outer iterations**，每轮 512/64=8 个 minibatches，共 **4,000 minibatch updates**。单纯把 episodes 数当成 optimizer steps 会高估 64 倍。2,048,000 个 ptx examples 是按最终 ratio 8 得到的算术量，不是披露的唯一 token 总量。

### C.5 FLAN / T0

{{FIGURE:13}}

Figure 13 用 RM score 扫描 FLAN/T0 模板或设置。它减少给 baseline 随意选提示的偏差，但 tuning 依据仍是本文 RM，不是完全独立标尺。最终 Figure 5 的 human Likert 更接近独立结果，二者不可混称。

## Appendix D — Automatic evaluation details

### D.1 Toxicity and bias

RealToxicityPrompts 自动评测取 **5,000 prompts**，按输入 toxicity 近似均匀采样，GPT 与 instruction 系列在 basic/respectful/biased 三种条件下续写。因此均值不能直接与按数据集原始分布随机抽样的其它报告比较。Winogender 与 CrowS-Pairs 用选择概率归一与 entropy 测量偏向。

{{DERIVED:entropy}}

二选一 entropy 最大 1 bit；接近 0 表示几乎只偏一项。高 entropy 可以意味着少偏向，也可以意味着对两个候选都不确定，不能等价于群体公平。须检查选项本身的含义、分组差异和是否有任务损失。

### D.2 全部 prompt templates 与 dataset features

下面保留 Figure 14–27 原文字图，可放大或展开转录。模板是实验的一部分；对 completion 的长度归一、标点、few-shot 例子数与生成 cap 都可能改变分数。

{{FIGURE:14}}

Figure 14（Winogender）比较代词 / 职业相关候选，经候选概率得到 entropy；读者应看其候选设计，而非将它当作全面偏见评测。

{{FIGURE:15}}

Figure 15（CrowS-Pairs）比较 stereotype 与较少 stereotype 的句子；概率代理和任务外的现实公平仍有距离。

{{FIGURE:16}}

Figure 16（RealToxicityPrompts）完整列出三种提示。biased prompt 是主动改变行为目标的 intervention；它揭示指令遵从对安全目标的潜在冲突。

{{FIGURE:17}}

Figure 17（TruthfulQA）用 QA、Instruction、Instruction+QA 等变体检验诚实指令。Table 14 的“QA+instruct”名称与正文排列不完全一致；原标签照录，不擅自合并成同一新行。

{{FIGURE:18}}

Figure 18（DROP）要求依据段落进行离散推理并回答，F1 不只是判断文风。显著退化意味着 RLHF 优化没有保持所有既有任务行为。

{{FIGURE:19}}

Figure 19（QuAC）是带上下文的问答模板；原图“7.306 examples”是排版异常，应理解为 7,306，但资产保留原字样。

{{FIGURE:20}}

Figure 20（SQuADv2）含可无答案的问题，生成回答的格式和拒答处理会影响 F1；zero-shot 与 few-shot 需要分开看。

{{FIGURE:21}}

Figure 21（HellaSwag）比较续写选项的 logprob；不依赖自由生成整段答案。同一个 actor 可以在人类风格上改善，同时选项 likelihood 改变。

{{FIGURE:22}}

Figure 22（RTE）考察文本蕴含，模板将任务转换为离散选项；改善可能依赖 prompt 与候选表达，不能只归结为整体推理提升。

{{FIGURE:23}}

Figure 23（SST）考察 sentiment；候选与归一方式是测量定义。要与其它 SST 报告比较，需先确认 split 与模板一致。

{{FIGURE:24}}

Figure 24（WSC）考察代词指代，logprob 选项评价与自然语言解释能力不同，few-shot 上的退化可被整体偏好分掩盖。

{{FIGURE:25}}

Figure 25（WMT Fr→En 2015）使用 BLEU 与翻译模板。生成长度、语言提示和 few-shot examples 会影响结果；论文说明使用软件没有提供翻译误差条。

{{FIGURE:26}}

Figure 26（CNN/DM）为摘要任务，指标 ROUGE-L。模板文字却写“15 additional French/English pairs”，疑似从 WMT 复制；不把这段异常当作摘要真实 few-shot 配置。

{{FIGURE:27}}

Figure 27（TL;DR）同为摘要，保留原 prompt 与生成设置；同样存在 French/English pairs 表述异常。Table 14 两组摘要结果又逐格相同，不能拿它们作两组独立验证。

<div class="callout warning"><strong>选择题原文疑点。</strong> Appendix D 写选择“lowest average per-token log probability”，与通常最大 likelihood 选择相反，可能漏了 negative；没有实现证据就不能替作者悄悄改成 highest。模板另有 logprob 归一说明。这里记录异常，并将它列为复现前必须确认的测量细节。</div>

## Appendix E.1 — Performance on public NLP datasets

{{FIGURE:28}}

Figure 28 是 zero-shot：按每个任务分别比较四类方法与模型大小。175B PPO 在部分知识 / 阅读理解 / 翻译任务的退化最突出，PPO-ptx 往回拉。纵轴单位随面板变化，不能把 BLEU、F1 和 accuracy 柱高合并成一个平均效用。

{{FIGURE:29}}

Figure 29 是 few-shot；任务示例对部分退化有缓解。必须同 Figure 28 配对，例如 SQuADv2 的 ptx 在 few-shot 基本回到基线，在 zero-shot 仍有差距；alignment tax 受提示条件影响。

## Appendix E.2 — Reward model generalization across labelers

保留一组标注者不用来训练 RM，再检查 RM 与他们判断是否一致。原文按 **inter- and intra-group** 顺序报告 **72.4±0.4%、69.6±0.9%**。不因直觉而自行交换数字。这支持对同一承包商群体中另一组人的局部泛化，无法检验未覆盖文化、语言或专业知识的泛化。

## Appendix E.3–E.5 — Metadata、Likert 与 bias

{{FIGURE:30}}

Figure 30 将 Figure 4 的 metadata 按规模分开，可检查“合并后的改善是否由大模型驱动”。规模与方法的交互说明不能把 pooled metadata 当作所有尺寸的精确结论。

{{FIGURE:31}}

Figure 31 给各模型 Likert，用单回答评分交叉检查 pairwise preference。排序与绝对评分具有不同噪声与尺度；两者一致可增强证据，却不能消除共享标注规范的偏向。

{{FIGURE:32}}

Figure 32 给 Winogender / CrowS-Pairs entropy。PPO-ptx 与 GPT 的某些 basic 行接近，但 respectful / biased 条件不一致；不能宣布 RLHF 稳定降低全部偏见。应把 baseline、提示与尺寸逐格匹配。

## Appendix E.6 — Fixing regressions

{{FIGURE:33}}

Figure 33 在 1.3B 扫描 ptx coefficient；大约 gamma ≥20 时若干 DROP / SQuAD 指标改善，同时 reward 不严重下降。这是给旧任务分布重新提供正向梯度，而非仅约束输出距离。最优权重受 loss reduction、样本比例和尺度影响，27.8 不能无条件迁移到现代实现。

{{FIGURE:34}}

Figure 34 扫描 KL coefficient，却**改用 pretrained GPT reference**，且 gamma=0。beta 到 2 仍不能充分恢复 DROP/SQuAD。它支持“单纯拉近旧模型分布未必等价于练回旧任务”，但不是只在默认 SFT reference 上改变 beta 的完美单变量比较。

{{FIGURE:35}}

Figure 35 延长至 **512k episodes、3 个 seeds**，后期 DROP / SQuAD 略回退。更多 RL 训练未必持续改善能力保留；需同时跟踪 human quality、reward 与公开任务，按共同边界 early stop。

## Appendix E.7–E.9 — KL、初始化与 learning rate

{{FIGURE:36}}

Figure 36 的 beta=0 用蓝线单独表示，不能在 log 横轴上画 0。beta 0 和 2 的 Likert 较差，0.01–0.02 较好；太小可能 reward exploitation，太大限制有益适配。曲线为该 setup 的经验权衡，不是通用 KL 最优常数。

{{FIGURE:37}}

Figure 37 比较 PPO 初始化模型，10% ptx 的初始化更好。起始模型决定可探索行为和 reference；不能把之后 PPO 的差异完全归于 reward 训练。

{{FIGURE:38}}

Figure 38 对学习率扫描：无 ptx 时，超过约 **8.05e−6** 会发散；小模型扫描约 2.55e−6–2.55e−5，175B 只测 **2.55e−6 和 3.74e−6**。按最高 human Likert 选择 checkpoint；正文没有逐模型逐variant明确汇总最终选中 LR，不凭 C.3 的初始化配置补齐。最终配置是有限预算下的调参结果，无法宣称为所有模型尺寸的全局最优。

## Appendix E.10–E.11 — Toxicity 分层与其它 ablations

{{FIGURE:39}}

Figure 39 按输入 toxicity 分层。respectful 指令下 PPO 输出更少毒性，biased 指令下低毒输入也可被转成高毒输出。这个因果 intervention 比单看平均分更能解释“遵从”与“安全”的冲突。

{{FIGURE:40}}

Figure 40 给 continuity 和相对输入的毒性 human ratings，检验模型是否只是离题或无条件拒绝来降低 toxicity。质量收益应同时满足 task relevance 与安全维度，不能只优化一个自动分数。

{{FIGURE:41}}

Figure 41 将 PPO-ptx、SFT 对 175B GPT-3 的 win rates 用于 toxicity 分布，参照与 Figure 1 / 3 的 175B SFT 不同。respectful 条件提升偏好，不代表 basic 或 biased 条件下的 harm 同样改善。

E.11 的附加扫描包含 ptx ratio、minibatch 等效率折中：ratio 8 的训练时间约翻倍，ratio 32 更慢且仅为初步结果；minibatch 32 略好，最终用 64 以提高 GPU utilization。作者没有给逐算子时延与显存峰值，这里只记录已有结论，不虚构性能曲线。

### Table 14 — 完整自动结果与重点重算

{{TABLE:14}}

列分为 GPT / SFT / PPO / PPO+ptx，每组 1.3B（原表 XL）、6B、175B。前三列明确 task、metric、prompt。entropy 高为作者的低偏向代理，toxicity 低较好，其余传统任务高较好。保留全部数值和样式；不只抽取最有利结果。

下面的重算固定 175B 和同一 prompt，展示 absolute difference；relative change 仅在基线非零且语义合理时给出：

| 175B 条件 / 指标 | GPT | PPO | PPO-ptx | PPO-ptx − GPT | 解释 |
|---|---:|---:|---:|---:|---|
| TruthfulQA，QA，true+info | 0.251 | 0.752 | 0.689 | +43.8 pp | 自动 judge；不同于 Figure 6 human |
| Toxicity，basic | 0.231 | 0.228 | 0.234 | +0.003 | ptx 没有降低基本提示均值 |
| Toxicity，respectful | 0.233 | 0.205 | 0.196 | −0.037 | 相对降低 15.9% |
| Toxicity，biased | 0.285 | 0.427 | 0.400 | +0.115 | 相对增加 40.4% |
| DROP，zero-shot F1 | 27.53 | 13.08 | 15.23 | −12.30 | ptx 对 PPO 恢复 2.15，仍退化 |
| SQuADv2，zero-shot F1 | 64.30 | 43.68 | 59.85 | −4.45 | ptx 对 PPO 恢复 16.17 |
| SQuADv2，few-shot F1 | 69.75 | 51.95 | 69.93 | +0.18 | 恢复范围依赖 shot |
| Fr→En，zero-shot BLEU | 38.92 | 24.16 | 34.28 | −4.64 | ptx 对 PPO 恢复 10.12 |
| HellaSwag，zero-shot accuracy | 0.781 | 0.743 | 0.807 | +2.6 pp | 并非每个 NLP task 都退化 |

**原表异常。** CNN/DM 与 TLDR 的 12 个结果逐格相同；可能是复制问题，原样保留，不能把两行作为独立复现相互印证。本文没有原模型预测可追溯此问题，不自行填入“应该正确”的数值。

## Appendix F — Model samples

附录 InstructGPT 用 temperature **1**，GPT-3 用 **0.7**。不同 sampling 会改变样例分布。前几张 prompts 被选来展示特定行为；后五张从 15 条选 5 条展示任务范围，outputs 不挑选。下面保留原示范、prompt 和所有 completion 文本。

{{FIGURE:42}}

Figure 42 是 Figure 8 法语例子的完整版。它展示在多数英语监督下跟随法语要求的可能性，不给多语言成功率。

{{FIGURE:43}}

Figure 43 是瑞典语指令：InstructGPT 执行任务但主要用英语回答，说明“懂任务”与“匹配输出语言”可分离。

{{FIGURE:44}}

Figure 44 是“如何偷超市商品而不被抓到”的请求，InstructGPT 给出配合该目标的回答。它是 helpfulness 优先可能造成有害遵从的具体反例；单个例子不能估计发生率，却足以否定“对齐后必然拒绝这类请求”。

{{FIGURE:45}}

Figure 45 是代码描述：原 Python 片段枚举子串并检查回文，InstructGPT 尝试解释并举例。回应更贴合问题，但索引、循环边界和所谓“最长”仍需逐行验证；此处没有本地执行原代码的声明，不能把流畅解释当成代码正确性。

{{FIGURE:46}}

Figure 46 要求从就业段落各找一条正向、负向证据，使用指定 bullet 格式。输入是申请失业救济 866,000→870,000、失业率从 4 月近15%降至8月8.4%、近期增速放缓。human demonstration 把866,000写成860,000，说明示范也可出错；InstructGPT 虽给两条bullet，却说近期就业增长更快，未忠实对应“增速放缓”。格式遵从与证据正确必须分项检查。

{{FIGURE:47}}

Figure 47 给出 serendipity 的定义，要求用该词造句。GPT-3 继续编写或重复定义，InstructGPT 给出 Lori 找到合适工作的例句，但还额外解释定义。这是简单任务选择的例子，不是摘要或专业投资任务。

{{FIGURE:48}}

Figure 48 根据 Ruth Bader Ginsburg 去世的新闻回答三个问题：谁宣布死讯、她对孙女的声明、对法院和国家的影响。InstructGPT 第一问正确，但把“新总统就职前不要替换我”释为“总统任期结束前不要替换”，失去原话的关键条件。逐问作答不保证全部答案忠实，封闭材料允许直接检查这一点。

{{FIGURE:49}}

Figure 49 把西葫芦 recipe 转成购物清单：西葫芦、牛肉、洋葱、蘑菇、甜椒、奶酪、番茄酱、盐和胡椒。GPT-3 只续写烘烤步骤；InstructGPT 也先加一句烘烤再列清单，并给出源文未提供的具体数量。此例区分task selection与续写，也提醒“完成清单”不等于“数量有依据”。

{{FIGURE:50}}

Figure 50 要求 ELI5 解释焦虑或受挫时胸口的“anxiety lump”。GPT-3 更像一篇带小标题的长文章，InstructGPT 给压力、fight-or-flight等因素的概述，但不够具体，也未真正核验人写示范中的生理机制。这个健康解释样例不能作为医学正确性证据。综合全部样例，instruction adaptation 改变了常见回答方式，没有使所有output都等价于正确示范。

## 工程重建 — 一条 PPO 轨迹经过哪些模型？

以下是**解释性重建**，补足 Figure 2 省略的生命周期，不声称原作者公开了完整训练程序。

{{DIAGRAM:execution}}

给定 batch prompts `[B,P]`，actor 自回归生成 `[B,R]` response tokens。reference 对相同拼接序列得到 `[B,R]` 的 selected-token log probabilities；RM 输出 `[B]` scalar；critic 给每个 response 位置 `[B,R]` value；old actor 的 rollout logprobs 也是 `[B,R]`。sequence、padding、EOS 终止位置和 response mask 必须在四路模型之间一致。

reward 的时间分配为 token KL penalty 加 terminal RM；returns / advantages 使用 mask 避免把 padding 当成 action。PPO 更新通过 old logprob 计算 ratio，通过 critic baseline 降低梯度方差。reference 与 RM 无 backward；critic 更新自己的 value loss。ptx 另做 teacher-forcing CE backward，累积到 actor 更新。新策略再生成下一轮 trajectories。

{{INTERACTIVE:ppo}}

```python
# Lifecycle sketch, not the authors' released implementation.
with no_grad():
    trajectory = actor.generate(prompts)  # keep response mask and EOS
    old_logp = trajectory.selected_logprobs
    ref_logp = reference.logprob(trajectory.tokens)
    terminal_score = reward_model(trajectory.tokens)
    values = critic(trajectory.tokens)
    rewards = -beta * (old_logp - ref_logp) * trajectory.response_mask
    rewards[trajectory.last_response_token] += terminal_score
    returns, advantages = estimate_returns(rewards, values, mask)
for minibatch in split(trajectory, size=64):  # one inner epoch
    new_logp = actor.logprob(minibatch.tokens)
    ratio = exp(new_logp - minibatch.old_logp)
    actor_loss = clipped_policy_loss(ratio, minibatch.advantages.detach())
    actor_loss.backward()
    ptx_ce = actor.cross_entropy(sample_pretrain(ratio=8))
    (gamma * ptx_ce).backward()              # separately accumulate
    actor_optimizer.step()
    critic_optimizer.step_on_value_loss(minibatch.returns)
```

### 算力与显存：训练复杂，推理不必附带 RM

dense decoder full-sequence 计算约 \(O(BTd^2+BT^2d)\) 每层，actor rollout 则要逐 token 自回归运行。backward 通常显著增加计算与 activation 存储；RM / reference 只有前向，仍占权重与临时工作空间。6B RM 不会让 175B actor 训练变成 6B 成本，ptx 也会消耗额外大模型 backward。

**解释性权重下界。** 若同一设备组同时驻留 175B actor、175B frozen reference、6B RM、6B critic，FP16 权重共约 \((175+175+6+6)\times2=724\) GB，约 674.3 GiB。对可训练 actor+critic 的 181B 参数，FP32 master 与 Adam 两个 FP32 moment 另需约 **2,172 GB**；FP16 gradients 若全部同时存储另约 362 GB。它们不含 activation、rollout、通信 buffer 和 KV，且可通过分片、offload、模型调度降低单设备峰值。此为条件算账，不是作者实测显存。

KV 对单个 decoder 仍近似 \(M_{\rm KV}=2BLH_{\rm KV}d_hTs\)。actor generation 用 KV 加速 decode；reference / RM 的 full-sequence teacher forcing 可用不同执行路径，不应将四个模型都按同一常驻 decode cache 估算。FP16 的 175B inference weights 下界约 350 GB；1.3B 约 2.6 GB。部署只运行最终 policy，参数结构未增添奖励 head 或 value critic，所以 RLHF 本身没有规定额外的每 token 审核调用。

**per-op 与端侧含义。** SFT/RM 的大 batch GEMM、PPO rollout 的小步 decode、reference logprob 前向和 ptx backward 对硬件利用率不同；E.11 的 minibatch64 正是在 quality 与 GPU utilization 间折中。论文没有测 TTFT/TPOT，也没有量化与端侧 kernel 数据。若将 1.3B 部署到 HTP/NPU，还要独立检查量化造成的 logits 漂移、真实 instruction/OCR 标定集与敏感层误差；不能因为论文“小模型更受偏好”就推定端侧低精度模型效果已验证。

### 实现中优先审计的调用链

同 prompt 排序的所有 completion 要在一次参数状态下评分，再进行 grouped backward；逐 pair 更新会使共享回答的后续分数受前一次更新影响。rollout 必须与 old snapshot 配套保存，不能拿更新后的 logprob冒充 old；reference 要保持冻结，critic 不能沿 policy loss 反向串进 actor。response masks、tokenizer、EOS、截断和 terminal reward 的位置必须一致，否则 KL 或奖励会作用到 prompt / padding。

异步 rollout workers 的模型版本与轨迹需要显式绑定，训练队列堵塞、RPC 回调丢失或过期 trajectory 会改变 PPO 的实际数据分布。本文未公开这些调度代码，因此这些是复现建议，并非已定位原实现缺陷。

## 实验审计 — 哪些 claim 的证据足够？

| Claim | 最直接证据 | 控制 / 缺口 | 判断 |
|---|---|---|---|
| RLHF 改善 API 偏好 | Fig 1/3、同规模 human comparison | 有两个 prompt 分布与 held-out 组；仍有限总体 | 支持局部偏好结论 |
| 1.3B 胜 175B GPT | API 偏好 | 参数和行为目标同时改变 | 不等价整体能力逆转 |
| 提高真实性 | Fig 6、Table14 TQA | human/auto、prompt和info需拆开；小模型例外 | 有条件支持 |
| 更少毒性 | Fig7/39、Table14 | respectful 获益；basic/biased反例 | 不能无条件宣称 |
| ptx 缓解回归 | Fig28/29/33、Table14 | LR/reference/init/ratio扫描有限 | 支持缓解，不证明全恢复 |
| 对 held-out 人群泛化 | E.2 | 同承包商总体、共享指南 | 支持相近判断组 |
| 可公开复现训练 | 官方 repo | 无原偏好集、weights、训练代码 | 仅可复查部分自动样本 |

缺少的关键实验包括：固定 backbone / 数据 / token 预算的 SFT vs PPO；跨语言和专业领域的代表性反馈；长度控制的偏好评估；reward gain 与人工真实性的长训练分离；不同 RM 规模与 actor/RM 能力差的系统扫描；更多 seed 与用户级置信区间；端到端显存、通信与 rollout utilization。

## 扩展研究 — 从 2017 年偏好学习到 2026 年奖励评测

以下资料单独检索，不能作为 2022 年实验的复现结果。日期是论文首次发布或明确的版本日期，箭头表示技术关系，不是性能排名。

{{DIAGRAM:evolution}}

| 一手工作 | 时间 | 核心变化 | 优势 | 限制 / 与本篇关系 |
|---|---|---|---|---|
| [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) | 2017 | 人类比较→learned reward→RL | 无需手写任务 reward | reward 外推与监督成本；RLHF基础 |
| [PPO](https://arxiv.org/abs/1707.06347) | 2017 | clipped policy update | 减少过大的单轮更新 | clipping不保证全局安全；本文优化器 |
| [Learning to Summarize](https://arxiv.org/abs/2009.01325) | 2020 | 摘要偏好→RM/PPO | 证明生成任务可用反馈优化 | 任务较窄；InstructGPT扩到通用指令 |
| InstructGPT | 2022-03 | API示范 / ranking / PPO-ptx | 助手行为明显改善 | 数据封闭、奖励单维、能力回归 |
| [Constitutional AI](https://arxiv.org/abs/2212.08073) | 2022-12 | 原则驱动 critique/revision + AI feedback | 减少逐条人工标签需求 | 原则与 AI judge仍需人为选择 |
| [DPO](https://arxiv.org/abs/2305.18290) | 2023-05 | BT+KL下以policy logratio重参数化偏好 | 离线训练，无显式RM/critic rollout链 | 数据固定；新policy覆盖与外推仍有问题 |
| [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) | 2024-02 | 同prompt候选组relative advantage | 可省value critic | 组方差与奖励可验证性；开放偏好仍困难 |
| [RewardBench 2](https://arxiv.org/abs/2506.01937) | 2025-06 / 2026-04 v2 | best-of-4与ties，多维奖励压力测试 | 更难以靠浅层格式取胜 | 专门RM benchmark，非API win rate |
| [General Preference RL](https://arxiv.org/html/2605.18721) | 2026 | 多维/非传递偏好与分维advantage | 放宽单scalar偏好假设 | 论文自报，外部广泛复现待定 |

DPO 并非把 PPO 的所有组件逐个做等价替换；其封闭形式依赖偏好模型和 KL 正则的假设。GRPO 也不是保证更便宜的任意 RM替代：省 critic 的同时，需要同 prompt 多个 rollout，成本仍取决于生成长度、group size 与 reward。Constitutional AI 则主要改变监督来源，而不是让价值选择自动消失。

### 官方实现能验证到哪里？

[OpenAI repo](https://github.com/openai/following-instructions-human-feedback)，核验 SHA **5c0534c78136809ab0118c5c1b74d159befa2467**，提供 modelcard、automatic benchmark samples 与 label instructions。它**没有完整训练程序、原偏好数据或模型权重**。2022 年 1 月 modelcard 还说明 API 实际版本的 fine-tuning procedure 稍有差异，因此论文模型与线上产品不能逐项等同。

**复现结论。** 本解读做了原文/TeX/HTML交叉阅读、公式与数值重算、资产提取和页面 QA；没有训练 GPT-3、没有复现 PPO，也没有测 GPU throughput。公开自动样本可帮助复查特定评测，但不能据此宣称整条 RLHF pipeline 可独立重建。

## 当前状态 — 截至 2026-10-29 的可比边界

当前路线已分化为高质量 SFT、离线偏好优化、在线奖励优化，以及 AI反馈 / 可验证奖励。选择取决于任务是否有可靠 reward、rollout 成本、数据覆盖与基础能力保持；不能用一个排行榜总分宣布所有对齐问题已解决。

本次实际核对 [RewardBench 2 官方论文 v2](https://arxiv.org/abs/2506.01937) 与 [官方源码](https://github.com/allenai/reward-bench)（main SHA **59f6ff7005a9f9484727ca27f5383420875e5dbf**）。benchmark 有 **1,865 prompts**：Factuality475、PreciseIF160、Math183、Safety450、Focus495、Ties102；best-of-4 和 ties 评价比普通 pairwise 更难。

| RewardBench2 官方 2026 v2 快照 | 规模 / 开放性 | Avg | Factuality | PreciseIF | Math | Safety | Focus | Ties |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Skywork-Reward-V2-Llama-3.1-8B | 8B，开放权重 | 84.1 | 84.6 | 66.3 | 77.6 | 96.7 | 98.4 | 81.2 |
| AI2 Llama3.1-70B RM-RB2 | 70B，开放权重 | 76.1 | — | — | — | — | — | — |
| AI2 Llama3.1-8B RM-RB2 | 8B，开放权重 | 72.8 | — | — | — | — | — | — |

破折号仅表示本页未展开该快照的分项，不表示官方未报告。Skywork 是**该官方快照内领先的公开模型**，不是已经核验 2026-10-29 所有提交的绝对第一。官方 [实时榜入口](https://huggingface.co/spaces/allenai/reward-bench) 可访问，但本次静态抓取没有榜单主体；[结果数据集](https://huggingface.co/datasets/allenai/reward-bench-2-results) 也未全量下载，不把入口可访问说成实时排名全核验。

高 overall score 仍可掩盖 PreciseIF 66.3 与 Focus 98.4 的大差异。分项分析与 InstructGPT 将 preference 拆成 metadata 的思路一致，但模型、数据与评价函数已变，**84.1 不能与 2022 年 85% API偏好率相比**。

2026 [General Preference RL](https://arxiv.org/html/2605.18721) 报告 Llama3-8B-Instruct 在 AlpacaEval2 length-controlled win rate **56.51**，其 GRPO+BT 对照 **41.92**，差 **14.59 pp**。这是该论文自己的实验，单来源研究证据，未本地复现；既非 RewardBench 分数，也非 InstructGPT API胜率。它把多维、非传递偏好作为建模对象，为单scalar RM的限制提供后续方向。

## 我的洞察 — 监督目标变化会重排整条训练链

**贡献。** 本文把通用助手的质量从“预训练大一点”转为“可采集、可优化、可审计的行为目标”，并给出三阶段落地证据。新意主要在规模、用户任务覆盖与系统评测，而非发明 PPO 或偏好学习。

**正确性与实验质量。** 同规模人类比较、两个 prompt 分布、held-out 标注者、真实性/毒性/传统任务和丰富 ablation 是长处。部分测量细节、标注群体代表性、调参预算与附录异常留下复现空白。pairwise BT 和 KL 目标数学上清楚，却不保证 RM 判断等于真实用户利益。

**工程复杂度与扩展性。** actor / reference / RM / critic 与 rollout queues 的生命周期使在线训练复杂。grouped RM 避免相关 pairs 的重复调用，ptx 改善能力保持但显著增加时间。分片与并行可以处理规模，通信、activation 与小步 decode 仍需实测；不能用“额外 FLOPs不到2%预训练”推断部署维护成本低。

**隐含 trade-off。** 更会遵从同时增加有害指令风险；更保守可能牺牲 informativeness；更高 reward可能偏离真实质量；更多 ptx保留能力但可能降低适配速度；更强 instruction行为并不免除量化/转换/导出精度损耗。每一项都应有独立测量，避免将误差归咎于模型太小。

**可推进的方向。** 固定基础模型和真实多样任务集，分别做 SFT、离线 preference 与在线 RL，控制 tokens / compute，再按 task success、事实 correctness、refusal、长度、group fairness、TTFT/TPOT与内存峰值分项报告。RM采用多维或可验证信号，保留ties并检查不确定性；验证用户级数据隔离与奖励漂移；对端侧部署独立评测原精度、量化后和导出后 logits/任务差异。

对我最有价值的工程启示是：**先定义哪个阶段的哪些输出受到什么监督，再检查它们如何沿调用链进入更新。** 目标、mask、snapshot、reward归约或回调生命周期的错误，都会让形式上正确的 PPO 训练出完全不同的行为。

## References 与阅读路径

1. [Ouyang et al., 2022，指定 v1 PDF](https://arxiv.org/pdf/2203.02155v1)：本篇原图、表、配置及结论的来源。
2. [OpenAI 官方资料仓库](https://github.com/openai/following-instructions-human-feedback)：modelcard、自动样本、标注指南；非完整训练代码。
3. [Christiano et al., 2017，人类偏好 RL](https://arxiv.org/abs/1706.03741)；[Schulman et al., 2017，PPO](https://arxiv.org/abs/1707.06347)；[Stiennon et al., 2020，摘要 RLHF](https://arxiv.org/abs/2009.01325)。
4. [Bai et al., Constitutional AI](https://arxiv.org/abs/2212.08073)；[Rafailov et al., DPO](https://arxiv.org/abs/2305.18290)；[Shao et al., DeepSeekMath](https://arxiv.org/abs/2402.03300)。
5. [RewardBench 2](https://arxiv.org/abs/2506.01937)，[官方源码](https://github.com/allenai/reward-bench)，[实时入口](https://huggingface.co/spaces/allenai/reward-bench)，[公开结果](https://huggingface.co/datasets/allenai/reward-bench-2-results)：2026 v2快照与实时榜须区分。
6. [General Preference Reinforcement Learning](https://arxiv.org/html/2605.18721)：2026单来源新研究，56.51等指标来自该论文。

<div class="related-readings"><p><a href="../transformer/">Attention Is All You Need</a>：先理解 decoder、attention与teacher forcing。</p><p><a href="../scaling-laws/">Scaling Laws</a>：区分基础建模能力的规模规律与行为目标适配。</p><p><a href="../pagedattention/">PagedAttention</a>：大规模rollout或助手服务中的KV容量与调用成本。</p></div>

**审校与证据。** 正文、附录、50图/14表/2式按指定版本核对；原文异常已就地说明。全部性能数字均标明原论文、自报扩展研究或条件推导；页面交互和响应式验证记录见本地发布报告。
