# EAGLE-2：把验证预算分给更可能走通的路径

同一个 draft model，不同上下文需要不同的树。EAGLE-2 用路径置信度分配候选节点，再让原模型并行验证；关键成本仍落在 draft、LM head、树注意力与 KV 提交的整条链路上。

<p class="reading-meta">FULL PAPER READING · 2406.16858v2 · UPDATED 2026-10-28</p>

**论文**：EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees。**作者**：Yuhui Li、Fangyun Wei、Chao Zhang、Hongyang Zhang。**机构**：Peking University、Microsoft Research、University of Waterloo、Vector Institute。

**版本**：2024-06-30 的 arXiv v2，12 页；后发表于 EMNLP 2024，pp. 7421–7432，DOI 10.18653/v1/2024.emnlp-main.422。下面按 v2 的章节顺序精读，最终会议版和后续工作另作来源标记。

[PDF v2](https://arxiv.org/pdf/2406.16858v2) · [arXiv v2](https://arxiv.org/abs/2406.16858v2) · [会议版 / DOI](https://aclanthology.org/2024.emnlp-main.422/) · [官方代码](https://github.com/SafeAILab/EAGLE) · [项目页面](https://sites.google.com/view/eagle-llm)

**来源约定**：**Paper** 表示 v2 原文；**External Source** 表示前作、后续论文或官方实现；**Interpretation** 表示本文的推导、工程重构或评审判断。原始图表完整保留；补充图与交互均标记 Agent-created。

**阅读起点与目标**：适合已了解 Transformer decode、softmax 与 KV Cache 的工程研究者；读完可重构 EAGLE-2 的候选扩展、路径筛选和验证提交，并判断论文加速比能否迁移到自己的 runtime。

## 阅读入口：一句话、一段话、五个关键点

**一句话**：EAGLE-2 把 EAGLE 的固定 draft tree 改成按当前上下文的路径价值动态扩展、全局筛选的树，在沿用原验证机制的条件下提高每轮有效产出。

**一段话**：自回归 decode 每输出一个 token，就要执行一次大模型。Speculative decoding 先用小模型提出候选，再让大模型一次验证多个位置。EAGLE 已经用 feature 预测降低 draft 成本，但固定树形仍会把有限验证预算浪费在低概率分支上。EAGLE-2 观察到接受率依赖上下文，且 EAGLE draft confidence 在实验中接近接受率，于是用从根到节点的 confidence 连乘估计路径价值：逐层只继续展开最有价值的前沿，最后从全部已探索节点中选出高价值子树。收益来自候选预算的重新分配；不需要为 EAGLE-2 重新训练一个新模型。

1. **问题**：固定位置、固定宽深的候选树无法适应“此处容易预测，彼处有歧义”的变化。
2. **观察**：接受率随 query 变化；Vicuna 7B 的置信度分箱均值接近接受率分箱均值。
3. **方法**：Expansion 用路径 value 选择继续生成的前沿；Reranking 在全部候选里选最终验证节点。
4. **结果**：v2 的 MT-bench 图示约 3.05–4.26×；Table 1 全任务区间为 2.48–5.00×。不能把图示区间当作全表范围。
5. **边界**：confidence 校准是经验依据，严格无损来自正确验证；更长接受长度不自动等于更高吞吐，也不代表少存一份 target 权重。

## Abstract：改善的是树的分配策略

**Paper**：摘要给出相对普通自回归约 3.05–4.26×，比 EAGLE 快约 20%–40%；不需要额外训练。

**Interpretation**：这些数字是论文的概括口径，必须回到模型、任务和温度逐格比较。3.05–4.26× 对应主图所展示的 MT-bench 结果子集；全表还包括 CNN/DM、Natural Questions 等更低加速的任务以及 HumanEval 的 5.00×。非 greedy 条件下，个别相对 EAGLE 的增益超过 40%。最终 EMNLP 版摘要采用 “up to 5x / 1.3x” 表述，不能倒灌进 v2。

### Figure 1 — 温度 1 下的无损加速比较

{{fig1}}

**目的与读法**：按横轴模型逐组比较粉色 EAGLE-2、蓝色 EAGLE 和紫色标准 speculative sampling；纵轴是相对同一原模型普通生成的 speedup，越高越好。Vicuna 使用 Vicuna-68M 作为标准 draft；LLaMA2-Chat 没有对应合适 draft，标为 N/A，不能当作 0。

**关键观察**：四个模型的 EAGLE-2 均高于 EAGLE，例如 Vicuna 13B 是 3.80× 对 2.32×，相对提高 63.79%。图中没有温度 1 的 Medusa/Hydra 等结果，因为作者只比较保持目标分布的采样机制。

**解释边界**：图证明所测条件下的系统速度差异；没有显示 GPU 型号、单轮 draft/verify 耗时或方差。此处的“无损”指理论输出分布，不是两次随机运行逐 token 完全一样。

## 1. Introduction：固定树把算力花在了不值得验证的分支

**What / Why**：大模型的单 token decode 通常受内存访问与低并行度限制。一次验证多位置可以提高计算利用率，但要付出 draft 与多候选验证成本。EAGLE 已提高 draft 的预测效率；EAGLE-2 进一步问：每轮的树是不是都应该长得一样？

**How / Evidence**：作者用数学题中的歧义位置解释预算分配，并在多个模型和任务上与 EAGLE 比较。贡献是利用上下文置信度动态确定树形，保持 draft model 与 lossless 验证路线。

**Interpretation**：方法可以看成一个在线候选调度器。它改变“哪批节点进入昂贵验证”，并不修改大模型的知识、最终输出头或目标分布。

### Figure 2 — greedy 模式，覆盖更大的 target

{{fig2}}

**如何阅读**：先看温度为 0，再在同一模型内比较柱子；EAGLE-2 的 7 个粉色柱子是 3.62、4.26、3.43、4.21、3.51、3.46、3.29×。颜色代表不同系统，N/A 仍表示未比较。

**Components / Data flow**：Vicuna 的标准 draft 是 68M；LLaMA2-Chat 70B、LLaMA3-Instruct 70B 的标准 speculative baseline 分别使用各自系列的 7B、8B。横跨模型比较时，target 与 draft 尺寸同时变化。

**Why it matters**：动态树的收益并不局限于一个小 target，但 greedy 的候选匹配条件也不能替代温度 1 的分布修正。Figure 2 中部分标准 speculative sampling 柱子没有在 Table 2 列出，不能称 Table 2 为该图所有 baseline 的完整数值表。

## 2. Preliminaries：先分清提案、预测与验证

### 2.1 Speculative Sampling：正确性由验证保障

**Paper**：小模型自回归产生 draft，大模型并行检查；拒绝之后修正采样分布。树状候选用专门 attention mask 保持每条路径的因果关系。

**Interpretation · 符号**：在已接受前缀 h 下，P(x|h) 为 target 对词表 x 的分布，Q(x|h) 为真正提出候选的 proposal。二者形状为 V；单 token x 的值是标量。下面是标准 speculative sampling 的重构公式，不是 v2 新增公式。

{{accept-equation}}

**每一步**：从 Q 采到 x 后，以 a(x) 接受；拒绝时从正部残差 R 采样。Z 为残差总质量，若 P=Q 则拒绝概率为 0，无须使用 R。以 P=(.7,.3)、Q=(.4,.6) 为例，候选 A 必接受，B 以 .5 接受；拒绝质量 .3 全部补到 A。

{{mass-equation}}

接受部分给出 min(P,Q)，拒绝后的补偿给出 P−min(P,Q)，两者相加恢复 P。证明依赖正确的条件 proposal 和残差更新，与 confidence 是否校准无关。

**工程含义**：若树构建时用 deterministic top-k、重排或无放回候选，不能直接把“网络 softmax Q”冒充实际提案分布套上 P/Q。要使用对应候选生成方式的修正机制。greedy 可以匹配 target argmax；随机采样需按实际提案更新残差。

**External Source · 官方代码核对**：固定快照中的 `utils.py` 在非 greedy 路径对按顺序检查的确定候选使用 `qx=1`，拒绝后将该候选的残差质量置零并归一化；greedy 路径比较 target argmax 并用连续匹配长度选择路径。这体现的是与实现相匹配的修正，不能简化成“confidence 就是接受概率”。[固定版本 utils.py](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/utils.py)

### 2.2 EAGLE：预测 feature，利用下一 token 消除不确定性

### Figure 3 — Feature draft 与 target verification

{{fig3}}

**目的**：解释 EAGLE-2 继承的 EAGLE 基础。先读左侧 drafting：标准小模型依赖 token，EAGLE 使用 target feature 加位移 token 信息预测下一 feature；再读右侧 verification：大模型检查候选，勾叉表明路径能接受到哪里。

**Components**：tᵢ 是 token embedding；fᵢ 是输出头之前用于 draft 的 target feature。原文以 “second-to-top-layer / before LM head” 描述，实际读取哪个 hidden state 必须结合实现，不能理解为最后两个 Transformer 层都要一起运行。

**Data flow / Why it matters**：feature 比离散 token 包含更多状态，但下一 feature 也受下一 token 影响，所以把下一步 token 信息前移对齐。图为简化在 draft 阶段画成链，verification 画树；不能据此说 EAGLE 只能生成链。

### 继承结构、Tensor Shape 与训练边界

**External Source · EAGLE 前作与 legacy EAGLE-2 实现**：设 B 为 batch、S 为已处理位置、d 为 target hidden size、V 为词表。输入 feature `[B,S,d]` 与位移 token id `[B,S]`；embedding 得 `[B,S,d]`，concat 成 `[B,S,2d]`，FC 回 `[B,S,d]`，一个 decoder layer 预测下一 feature，复用 target LM head 生成 `[B,S,V]` logits。

{{feature-diagram}}

**训练输入 / 输出**：冻结 target，离线收集 feature 与对齐 token；draft 预测 feature，再映射到目标词表。EAGLE 前作使用 SmoothL1 feature regression 与权重 .1 的 soft-label cross entropy。这不是 EAGLE-2 新 loss。

{{training-equation}}

**External Source**：EAGLE 前作报告 7B/13B/70B target 对应约 .24B/.37B/.99B trainable draft 参数、68K ShareGPT 样本、AdamW 学习率 3×10⁻⁵、β=(.9,.95)、gradient clip .5。EAGLE-2 沿用已有 draft，不在本文重新训练；上述设置来自前作，不能填入 v2 未披露的测评硬件栏。[EAGLE v3](https://arxiv.org/abs/2401.15077v3)

**优化目标区分**：训练让 draft feature/输出分布靠近 target；EAGLE-2 的运行时树选择则让有限节点预算落在更可能被接受的路径上。它们属于两个不同层次。

### Figure 4 — 同样预算，树形随上下文变化

{{fig4}}

**如何阅读**：左侧 EAGLE 对 “10+2” 和 “10+2=” 均给固定分叉；右侧 EAGLE-2 在歧义位置保留 “=” 与 “+”，在确定位置把预算放到 “1→2” 的更深路径。

**关键观察**：宽度用于覆盖不确定性，深度用于利用确定性。EAGLE-2 通过 value 排序产生这种结果，而不是手写“数学题必须走深”的规则。

**My interpretation / Caveat**：示例说明设计动机，未提供算式 query 的独立耗时统计；最终树只在已探索候选里筛选，不能保证找到所有可能 token 路径中的最优树。

## 3. Observations：两条经验前提

### 3.1 Context-Dependent Acceptance Rates

{{fig5}}

**Figure 5 解决的问题**：位置 P1–P6 是否具有稳定、通用的接受率？左图定义二叉树位置，右图横轴是同一组位置、纵轴是接受率，每个点代表一个 query。

**读法**：同一个位置竖向分散，说明 query 差异；不同位置总体分布也有差异，说明深度与分支 rank 的影响仍然存在。

**Interpretation**：固定树通常利用平均位置统计；当同一位置从近 0 到近 1 都可能出现，仅按位置分配候选就会错配预算。图支持 context dependence，不能把所有差异直接归因为文本难度，因为路径前缀、token rank 等因素共同影响。

### 3.2 Well-Calibrated Draft Model

{{fig6}}

**Figure 6 读法**：横轴是 draft confidence 分箱，柱高是该区间 token 的平均接受率；红色虚线是 y=x。target 是 Vicuna 7B，论文观察来自 Alpaca。极低 confidence 区间接受率约 .04，极高区间约 .98。

**核心观察**：confidence 越高，平均接受率越高，并且大体接近对角线。这样无需调用 target 就能用 cheap proxy 估计候选价值。

**Caveat**：分箱平均校准不等于单 token 相等，也不等于在任意模型、温度、任务或深度都校准。图没有置信区间、样本数、ECE/Brier score 或跨域校准曲线。`c=.8` 不能保证此候选恰好以 .8 被接受，更不是 `min(1,P/Q)` 的另一种写法。

## 4. Context-Aware Dynamic Draft Tree：把局部置信度变成路径价值

### 原公式逐步拆解

{{value-equation}}

**Paper · 原文未编号公式**：Vᵢ 为节点 i 的 value；Path(root,i) 为从根到 i 的路径；pⱼ 为路径节点的条件接受率；cⱼ 为 draft confidence；根节点已由 target 确定，value 设为 1。

**Shape**：每个 cⱼ/pⱼ/Vᵢ 都是标量；如果候选池有 N 个节点，存储 `[N]` 的 score、parent index 和 depth。c 来自 draft logits 的 softmax；p 是未知、要估计的接受行为，不是上文 target 词表分布 P。

**推导直觉**：节点 i 被实际接受，要求祖先先被接受，再接受 i。连乘应理解为概率链式分解中的条件概率，不需要假设这些事件相互独立。用 c 替代每个条件 p 才是经验近似。

**Toy example**：`It→is→a→good` 的边 confidence 为 .6、.8、.7，value 是 .6×.8×.7=.336。另一支 `It→has→to` 的局部 .7 看起来很高，但整条路径只有 .2×.7=.14。用局部 confidence 排序会忽略“走到这里”的代价。

**工程含义**：用 log value 把连乘变加法，避免深路径数值下溢；c=0 对应 −∞。因为 0≤c≤1，parent value≥child value，按 value 排序通常自然保留祖先。相等时必须保证浅层优先，不能无条件相信任意 top-k 实现的 tie 顺序。

### 4.1 Expansion：每层只扩展高价值前沿

**Input / Output**：输入已验证 prefix、root token、draft 状态；输出比最终预算更大的候选池及其 parent/score。每步在当前层选择 k 个高 value 节点，给 draft model 作为并行输入；每个 parent 生成 b 个候选 child，累积路径 value。

**What changed**：EAGLE 的固定树先决定哪些位置有资格生成；EAGLE-2 把资格交给当前 score。它依然是逐层 draft，不能称“一次小模型 forward 就生成整棵树”。

**官方实现对照 · External Source**：`cnets1.py::topK_genrate` 将 log-softmax child score 与 parent score 相加；每层 k parents × k children 形成 k² 候选，再取 k 个作为下层前沿。未进入前沿的候选仍可留在全局池中供 reranking。论文 Fig. 7 的 k=2 是示意，附录扩展宽度是 10。

### 4.2 Reranking：全局预算与祖先闭包

**Paper**：扩展完成后，从所有候选中保留 value 最大的 m 个，展平成一维序列并生成树 attention mask。m 是最终验证预算，和扩展前沿 k 不是同一个量。

### Figure 7 — 完整算法、展平顺序与注意力

{{fig7}}

**如何阅读**：从上到下：橙色节点表示本层选择展开的 top-2，绿色是新生成 children；中间蓝色表示最终 top-8；然后是展平序列与 mask。

**Components / 数值**：`is=.6`、`has=.2`；第二层 `a=.48`、`the=.06`、`to=.14`、另一 `a=.02`。展开 a(.48) 和 to(.14)，产生 good=.336、nice=.048、be=.084、do=.028。原图将末层值四舍五入成 .34、.05、.08、.03。

**最终 top-8**：保留 It、is、has、a(.48)、the、to、good、be；删去 a(.02)、nice、do。按 value 选出集合后，按拓扑/原候选顺序展平，得到 `It is has a the to good be`；展平顺序不是 value 的递减顺序。

**Data flow**：每个 query 只访问共享已验证 prefix、自己的祖先和自己。`be` 看 It/has/to/be，不能看 is/a/good；`good` 看 It/is/a/good，不能看另一分支。否则验证的 target 条件分布会被兄弟 token 污染。

**Position ids**：位置取 prefix length + 相对 depth，而不是数组下标。`is` 与 `has` 深度相同，应使用相同的相对 position；一维展平只是为了批量执行，不改变树的因果位置。

{{tree-widget}}

**交互解释**：调低 `It→is`，它整条子树的 value 都下降；提高 `It→has`，to/be 路径会更有竞争力。节点预算变小后，mask 跟随实际保留集合变化。组件采用根在内的 m，与 Figure 7 一致；官方配置是否把 root 单独计数要读实现，不能直接互换数值。

### 可复现伪代码

```python
# 输入 prefix/root + draft state；输出候选 IDs、parents、depth 与 mask
pool = [root(value=1)]
frontier = [root]
for level in range(depth):
    children = draft_expand(frontier, branching=b)
    for child in children:
        child.log_value = child.parent.log_value + log(child.confidence)
    pool.extend(children)
    frontier = top_k(children, key=(log_value, shallow_first), k=k)
selected = top_m(pool, key=(log_value, shallow_first), m=m)
selected = topological_order(selected)
mask = shared_prefix_plus_ancestors_and_self(selected)
position_ids = prefix_length + relative_depth(selected)
target_logits, target_features, temporary_kv = target_verify(selected, mask, position_ids)
accepted_path, next_root = corrected_acceptance(target_logits, actual_proposals)
commit_only_accepted_path(temporary_kv, accepted_path)
```

**条件**：浅层优先解决 confidence=1 或 0 时的 ties；如果引入长度奖励、外部重标定或 score 噪声后破坏 parent≥child，就要显式做祖先闭包，不能直接 top-m。上述伪代码用于机制解释，真实 sampling correction 还需要按路径记录提案和残差状态。

### 执行链路、KV 提交与成本

{{cycle-diagram}}

**External Source · 当前固定实现**：target 并行验证后，`update_inference_inputs` 根据被接受路径 gather 临时 KV，把它搬到连续 prefix 位置，更新 current length；同时更新 draft 所需 feature 与下一 root。落选节点的临时 KV 不应作为未来 prefix 留存。官方路径含 batch=1 假设，不宜直接套到连续批处理服务。

**长度口径**：代码 `new_token += accept_length + 1`，root 是由 target 确定的 token，另外计 1。v2 将 τ 描述为每轮 generated tokens，并称 average acceptance length；阅读和复现时应明确是否包含 root/bonus，避免拿 draft-only acceptance 对齐另一系统的 generated-per-cycle。

**Interpretation · 逻辑成本重构**：

| 环节 | 典型形状 / 操作 | 成本与瓶颈 |
|---|---|---|
| Draft FC/decoder | 前沿 `[B,k,d]`；concat `[B,k,2d]` | 深度 D 轮顺序依赖，小网络 kernel/RPC 开销可能明显 |
| Draft LM head | `[B,k,d] × [d,V] → [B,k,V]` | 每层约 BkdV MAC；大词表和 logits 带宽不可忽略 |
| 候选池与选择 | O(Dkb) 个 child；官方 b=k | top-k、索引构造、同步、parent lookup |
| Target verify | m 个位置；hidden `[B,m,d]`、logits `[B,m,V]` | 全部 target 权重仍需驻留；验证更宽有时更划算，有时增加延迟 |
| Tree attention | mask 逻辑 `[B,m,S+m]` | 稠密执行仍可做被 mask 的计算，稀疏树语义不等于稀疏 kernel |
| KV commit | gather 被接受路径 → 连续 prefix | 索引/拷贝和 cache length 更新必须与 token/features 一致 |

{{cost-equation}}

L 为 target 层数、h_kv 为 KV heads、d_h 为 head dim、s 为元素字节数；临时新 KV 为 `2LBmh_kv d_h s` 字节，不含已有 prefix、draft KV、workspace、logits 或 allocator 峰值。稠密 attention 逻辑计算约 O(LBm(S+m)d)；若 kernel 真正只计算祖先，理想相关项约 O(LBm(S+D)d)，不是本文已测实现保证。

{{cost-widget}}

{{speed-equation}}

τ 为同口径每轮产出，t_AR 为普通单 token decode 时间。分母包含 draft、target verify、select/mask 和 commit。增大 m 可能提高 τ，也可能抬高验证耗时；最大化 value 的子树并不等于硬件上的 latency-optimal 树。这里是结构性成本模型，没有制造 per-op 实测时间。

## 5. Experiments：逐表检查收益、代价与可比性

### Setup 审计

| 项目 | v2 可以确认 | 未披露 / 不能自行补全 |
|---|---|---|
| Target | Vicuna 7B/13B；LLaMA2-Chat 7B/13B/70B；LLaMA3-Instruct 8B/70B | 精确 checkpoint hash、权重格式 |
| Task | MT-bench、HumanEval、GSM8K、Alpaca、CNN/DM、Natural Questions | 全部采样子集、prompt/output 长度分布 |
| Sampling | temperature 0、1；部分方法只比较 greedy | top-p/top-k 等完整组合与 RNG 对齐 |
| Metrics | 相对 AR speedup ↑；平均每轮长度 τ ↑ | 方差/误差条、端到端 p50/p99、TTFT/TPOT 分解 |
| Draft | 沿用 EAGLE；附录说明预算、深度和扩展宽度 | EAGLE-2 无新训练 tokens/loss/optimizer |
| Hardware / runtime | v2 没有足以复现的完整配置 | GPU、precision、batch、TP、kernel、warmup、重复次数 |

**Baseline fairness**：温度 1 只和 lossless 采样比较是合理边界；温度 0 可比较更广。但模型间不同 draft 大小、实现优化程度与未公开 runtime 使横向结果无法当作统一硬件成本排序。没有 precision / batch 信息，就不能宣称所有 baseline 的资源预算完全公平。

### 5.1 Effectiveness · Table 1：完整 32 个数据行

{{table1}}

**如何阅读**：两个左侧字段 Model / Method，随后 6 个任务与 Mean，每组含 Speedup、τ，共 16 列。Temperature=0 有 22 个数据行，Temperature=1 有 10 个数据行；另有标题与分组行。原截图保留 bold 与完整表格布局，caption 以图片外的文字呈现；网页表保留所有行列。

**最强 baseline**：同模型、同温度下 EAGLE 是主要最强可比 baseline。Mean speedup 的核算如下，绝对差单位为“相对 AR 的倍数差”，不是毫秒或百分点。

| Target | Temp | EAGLE | EAGLE-2 | 绝对差 | 相对 EAGLE 提升 |
|---|---:|---:|---:|---:|---:|
| Vicuna 13B | 0 | 2.95× | 4.04× | 1.09× | 36.95% |
| LLaMA2-Chat 13B | 0 | 3.09× | 4.10× | 1.01× | 32.69% |
| Vicuna 7B | 0 | 2.78× | 3.39× | .61× | 21.94% |
| LLaMA2-Chat 7B | 0 | 2.78× | 3.43× | .65× | 23.38% |
| Vicuna 13B | 1 | 2.39× | 3.65× | 1.26× | 52.72% |
| LLaMA2-Chat 13B | 1 | 2.64× | 3.88× | 1.24× | 46.97% |
| Vicuna 7B | 1 | 2.18× | 2.94× | .76× | 34.86% |
| LLaMA2-Chat 7B | 1 | 2.32× | 3.15× | .83× | 35.78% |

**Claim audit**：greedy Mean 增益约 22%–37%，温度 1 约 35%–53%，所以摘要 “20–40%” 不是每个格子的边界。全任务 EAGLE-2 speedup 最低 2.48×（V7、temp 1、NQ），最高 5.00×（L2 13B、temp 0、HumanEval）。

**细节校正**：正文提到 “EAGLE up to 5x” 有命名含混；对应 Table 1 的 5.00× 属于 EAGLE-2。PLD 在 CNN/DM 的突出成绩是它自身强项，不代表击败 EAGLE-2：V7 是 2.53× vs 2.94×，V13 是 2.42× vs 3.40×。

**成本与 trade-off**：长度上升与 speedup 上升支持调度有效，但原表没给 select/verify 的额外时间和内存。Mean 是任务加速比的摘要，不等于将所有请求串起来后的总体吞吐倍率。

### Table 2 — 大 target 与 LLaMA3

{{table2}}

**条件**：MT-bench，temperature=0。LLaMA2-Chat 70B 有 PLD/Lookahead/EAGLE/EAGLE-2 四行；另两组各两行，总计 8 个数据行。

| Target | Speedup EAGLE → EAGLE-2 | 相对增益 | τ EAGLE → EAGLE-2 |
|---|---|---:|---|
| LLaMA2-Chat 70B | 3.01 → 3.51× | 16.61% | 3.81 → 4.48 |
| LLaMA3-Instruct 70B | 2.83 → 3.29× | 16.25% | 3.62 → 4.16 |
| LLaMA3-Instruct 8B | 2.72 → 3.46× | 27.21% | 3.65 → 4.53 |

**Interpretation**：dynamic tree 在更大 target 上仍有收益，但提升并非所有尺寸都 20%–40%。不能把 8B vs 70B 的倍率差解释为只由 target 大小造成，预算、模型可预测性和 runtime 也不同。

### 5.2 Ablation · Table 3：value 比 reranking 更关键

{{table3}}

**条件与方向**：Vicuna 7B、temperature=0；MT-bench/GSM8K，Speedup 与 τ 均越大越好。4 种 variant × 2 个任务 × 2 个指标完整保留。

**5.2.1 Value and Confidence Score**：MT 去掉 value、直接用局部 confidence，3.62→3.21×，相对 full 下降 11.33%；τ 4.98→4.39。GSM 更明显：3.63→2.93×，下降 19.28%；τ 4.97→3.96。祖先成功概率是主要信号。

**5.2.2 Reranking**：MT 去掉全局 rerank，3.62→3.48×，下降 3.87%；GSM 3.63→3.50×，下降 3.58%。最后筛选的额外收益较小但两个任务方向一致。

**同时移除**：MT 2.81×/3.92，GSM 2.85×/3.93；full 分别提高 28.83%/27.37%。不能把 value 与 rerank 的百分比相加：两个组件相互影响，也没有误差条来判断统计显著性。

### 如果作为 reviewer，还缺哪些实验？

- 逐轮记录 draft forward、LM head、top-k、mask construction、verify、KV gather/commit 的 latency 与 GPU 利用率；报告 TTFT、TPOT、p50/p99。
- 固定总节点预算，仅改变树形；再固定硬件、探索 m/depth/k 的 Pareto 曲线，核对“多接受”是否抵消“多验证”。
- 分任务、深度、rank、temperature 测 calibration 与置信区间；测试失配时静态树是否更稳。
- 在长 context、continuous batching、高 concurrency、不同 target 架构与量化精度下复现；报告临时 KV 与峰值显存。
- 随机模式检验输出分布与实际 proposal 修正；把 greedy 输出一致与统计分布一致分开验收。

## 6. Related Work：2024 年时它与谁相邻

**Paper**：相关工作涵盖 speculative decoding、检索型草稿、Lookahead、Medusa/Hydra、多分支验证，以及 Sequoia 的树形优化。EAGLE-2 与前作共享 feature draft，新增的是 context-aware 树生成和筛选。

| 工作 | 时间 | 核心路线 | 优势 | 限制 / 与 EAGLE-2 的关系 |
|---|---|---|---|---|
| Speculative Decoding / Sampling | 2022 preprint；2023 ICML | 小模型提案、大模型验证与残差修正 | 目标分布可保持；一般性框架 | 小模型成本/兼容性；EAGLE-2 继承验证思想 |
| PLD / prompt lookup | 2023 | 复用 prompt/文本匹配片段 | 无额外模型、复制任务强 | 自由续写适用性受重复度限制 |
| Lookahead | 2023–2024 | 并行 token 候选构建与验证 | 不需要单独训练同类 draft | 候选质量与执行成本需权衡 |
| Medusa | 2024 | target feature 上的多头候选 + tree verify | 草稿开销低、多个未来位置并行 | 随机接受变体可改变分布；不可和 lossless 结果混排 |
| Hydra | 2024 | 顺序相关的多头草稿 | 强化未来 token 间依赖 | 训练/候选与验证预算不同；v2 引作相关路线 |
| Sequoia | 2024 | 用统计模型与硬件成本优化树 | 明确考虑接受率/验证成本 | 先验/硬件模型与 query 实时信号不同 |
| EAGLE | 2024-01 | feature autoregression + token shift | 高质量 cheap draft | 静态树；EAGLE-2 直接沿用同一 draft |
| EAGLE-2 | 2024-06 | query confidence 的路径 value + rerank | 无重训；预算随上下文变化 | 校准、候选池覆盖、kernel 与批处理决定收益 |

**来源时间边界**：Sequoia 现有 arXiv v3 是 2025 修订，不能把修订中的所有系统结论当成 2024 年 EAGLE-2 已比较过的内容。上表历史定位遵循 v2 引用语境。

## 7. Conclusion：有效的调度，不是免费的多 token

**Paper**：context-dependent acceptance 与 calibrated confidence 使动态树成为可行设计；EAGLE-2 相对 EAGLE 提高速度，且不需要新训练。

**Interpretation**：论文最有价值的抽象是将“节点存在的资格”从固定模板转向路径成功概率。正确性与效率分别由 target verifier 和候选调度负责。校准好坏主要影响候选利用率；验证/位置/KV 链路正确与否影响输出正确性。

## Appendix A：实现参数不能与当前默认值混用

**Paper**：7B/8B、13B、70B target 的 draft token budget 分别为 60、50、48；depth=6；每轮 expansion 选择 10 个节点。不要用 Figure 7 的 top-2/top-8 当论文实验配置。

**Paper · Baseline 实现**：standard speculative sampling 使用 HuggingFace Transformers assisted generation；PLD、Lookahead、Medusa、Hydra 使用默认设置与官方权重；EAGLE 的 Vicuna/LLaMA2 draft 使用官方权重，LLaMA3-Instruct draft 由 ShareGPT 训练。后者是适配 target 的 draft 准备，不是 EAGLE-2 动态树新增的训练步骤。

**External Source · 固定代码快照**：2026-02-20 的 commit `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b` 用于核对当前官方实现。`ea_model.py` 的 `use_eagle3` 默认 True；显式 False 才进入 `cnets1.Model` 的 legacy EAGLE-2 路径。`cnets.Model` 已是 EAGLE-3 的融合与独立输出头，不能据此重构 EAGLE-2。

| 代码位点 | 对应机制 | 复现时要核对 |
|---|---|---|
| `cnets1.Model` | feature+embedding、FC 2d→d、一个 decoder | target hidden state、token shift、shared head |
| `cnets1.topK_genrate` | log value、k² children、全局 top-k | 默认 total=63、depth=5、top=8 不同于附录 |
| candidates / parent lookup | 选节点、按原索引排、searchsorted parent | 祖先闭包与 parent 是否存在 |
| tree mask / positions | identity + parent ancestors；行和−1 得 depth | 不能把 flat offset 当 position id |
| `evaluate_posterior` | greedy 连续匹配 / sampling correction | proposal 的生成方式、残差更新顺序 |
| `update_inference_inputs` | 路径 KV gather、长度、下一 root/features | batch=1 假设、同步开销、EOS/max tokens |

**鲁棒性观察 · Interpretation**：当前 `torch.topk` 未显式规定相等 score 时浅层优先。对 c=1/0 的 tie，数学闭包需要稳定祖先顺序；这是应补充验证的边界，不是在本文复现出一个 runtime bug。深度循环的初始展开与 root 计数还会改变配置口径。

**复现最小记录**：target/draft hash、EAGLE-2 route、Torch/CUDA/kernel、GPU/TP、precision、batch/concurrency、context/output 分布、temperature/seed、m/depth/k、root 计数、warmup、重复次数；测 accepted path 与 KV 内容一致，再测 per-op timing。本文没有执行 GPU benchmark 或新增训练。

## 技术演进：从草稿质量到预算、词表和并行草稿

{{timeline-diagram}}

**External Source**：2022–2023，speculative sampling 建立“廉价提案 + 正确验证”的基础；2024 年 EAGLE 提高 feature draft 质量，EAGLE-2 加入 context-aware 候选分配；2025 年 EAGLE-3 减少 feature regression 约束并增加训练时多步模拟，FR-Spec 优化巨大词表输出头；2026 年 PEAGLE、DFlash、JetSpec 等向并行/分块草稿推进，EAGLE-3.1 则解决长上下文 draft attention 漂移。

**箭头语义**：问题的延续与相关路线，不表示每项都是 EAGLE-2 的直接后继，也不构成在同一测评条件下的性能排序。

## 截至 2026-10-27：当前强路线与可比性审计

**External Source · 最新检索**：下面使用论文或官方作者/框架报告。指标、模型、精度、GPU、并发与 baseline 不统一，不能据这些数字给出一个全场景 SOTA 冠军。

| 工作 / 来源 | 日期与公开性 | 公开报告条件与数值 | 与 EAGLE-2 的可比性 |
|---|---|---|---|
| EAGLE-3 | 2025-03；官方开源 | v3 摘要 up to 6.5×、约 1.4× EAGLE-2；多层 feature fusion、直接 token 训练、training-time test | 作者对照比跨论文数字有价值；不等于在本文 v2 全任务同硬件复跑 |
| FR-Spec | 2025-02/03；公开论文/实现 | 128K 词表；报告 LM-head 计算降约 75%、平均比 EAGLE-2 快 1.12× | 说明词表头可限制实际速度；不是新树策略的独立 ablation |
| SGLang EAGLE 文档 | 2026-10-27 读取；框架开源 | Llama3.1-8B、H100、MT-bench；AR/E2/E3=158.34/244.10/373.25 TPS，对 AR 1.54×/2.36× | serving 与 v2 图示不同；不可把论文 4× 作为部署保底 |
| EAGLE-3.1 | 2026-05-26；官方博客/实现 | normalization + postnorm feedback；长 context 接受率 up to 2×；Kimi K2.6 NVFP4、TP4、GB200、coding，C1/C4/C16 output TPS 对 AR=2.03/1.71/1.66× | 量化 MoE/并发与 v2 不同；作者自报告，不能归因于树 alone |
| PEAGLE | 2026-03-13；vLLM 官方报告 | 一次 forward K 位置；4-layer head；GPT-OSS-20B、1 B200；对 EAGLE-3 最好约 1.55× MT /1.55× HumanEval /1.69× SPEED | 扫描 K=3/5/7、C1…64；training/head 优化共同变化 |
| DFlash / DFlash v2 | 2026-02，v2 05-28；06-15 SGLang 报告；开源 | block diffusion + target feature KV injection；Qwen3-8B Transformers 报告 6.1×。Qwen3.5-397B-A17B BF16、8 B200、HumanEval、greedy thinking、max-new=4096、C1，>4.3× AR、约 1.5× MTP | block=16 调优；跨后端和任务不可对照 v2；低 draft 延迟是重要解释 |
| JetSpec | 2026-06，v5 10-03；公开论文/官方代码 | Qwen3 dense/MoE、H100、m=256；报告 MATH up to 9.64×、chat 4.58×；representation flow | 大预算/自报告；不能称所有 sampled branches 都由一次 forward 精确条件化 |

**EAGLE-3 的机制差异**：移除显式 feature regression，使用多层 target feature 融合并在训练中模拟测试阶段多步草稿，直接优化 token。不要把 EAGLE-3 的 3d 融合与独立 LM head 描写成 EAGLE-2 的结构。[论文 v3](https://arxiv.org/abs/2503.01840v3)

**一个说明 τ 不够的对照**：DFlash v2 报告在相同 5-layer Qwen3-4B draft 比较中，MT-bench 的 EAGLE-3 τ=3.1/speedup=1.4×，DFlash τ=3.0/speedup=2.2×；GSM8K τ 都约 4.2，speedup 仍是 2.1× vs 3.3×。接受长度接近，草稿执行成本不同，速度可以差很大。[DFlash v2 原论文](https://arxiv.org/html/2602.06036v2)

**文档快照冲突**：SGLang 的 speculative decoding 文档把 SpecV2 描写为 experimental，而 2026-06 新博客采用默认启用的口径。部署应以实际 release、配置和代码核对；本文记录两个来源的差异，不宣称某默认值普遍成立。

**行业使用判断**：EAGLE-2/EAGLE-3 已进入 SGLang 等主流开源 serving 栈。今天选择路线应同时问 target 是否有匹配 draft、长 context 是否校准、并发下收益如何、head/verify/KV/kernel 能否充分并行。论文单请求加速不是线上多请求吞吐承诺。

## 我的判断、开放问题与可行改进

**以下全部为 Interpretation，与论文结论分开。**

**Strengths**：新增训练成本为零；路径 value 的直觉明确；完整消融说明祖先成功概率的重要性；把 tree shape 从固定结构变成数据依赖调度，让方法容易插入已有 EAGLE 验证系统。

**Weaknesses**：校准证据集中于一个 target/任务；hardware/runtime 披露不足；没有 per-op、内存、高并发或长上下文曲线。Novelty 是相对前作的增量，但调度抽象具有复用价值。

**Correctness**：path value 属于效率估计；无损要依赖条件提案修正、祖先隔离、depth-based position 和 KV 提交。即使 value 全错，正确 verifier 仍可保持分布，但可能失去加速。

**Engineering complexity**：小模型顺序循环、GPU top-k、tree-index 构建、临时 KV 与 path gather 比固定模板更复杂。若在 Native runner 中集成，应收口 `draft→select→verify→commit` 的状态机；上层只接收最终 token，避免把候选生命周期暴露给业务层。

**Scalability / Generalization**：m 增大使临时 KV、logits、mask 与 verify 上升；D 增大使 draft 序列依赖加深；target 更大或不同 attention/quantization 的收益取决于实际 kernel。MoE target 的逐节点路由/通信也可能改变最佳树，v2 未隔离这一因素。

**Hidden trade-offs**：只追求 ΣVᵢ 会忽略新增一层 draft 与 target 批次形状的成本；固定最终预算下，选更深路径通常提高潜在产出，但 per-layer launch、LM head 和 position/KV 处理也能成为新的瓶颈。

**Possible improvements**：

- **成本感知在线停止**：比较新增 frontier 的预计有效 token 收益与实测 marginal draft/verify cost；按 context length、并发和 target kernel 选择 m/D/k。
- **按上下文分组校准**：用深度、rank、domain、temperature 的 held-out 数据拟合 c→p，检查是否保持 parent≥child；失配时回退稳定策略。
- **词表与 logits 优化**：减少 draft head 的无效 vocabulary 计算，评估频率子词表的 coverage 与精确验证路径；FR-Spec 提供可查的实现方向。
- **复用树索引与融合提交**：把 selection、parent remap、mask 与 KV path copy 尽量减少同步，但必须先验证 position/ancestor/cache length 不变量。
- **并行草稿与动态树结合**：研究并行表示预测如何保留跨深度依赖，再让 context-aware budget 选择要验证的分支。不能以预测未来 marginal 的方式直接替代 branch-conditioned proposal。

**最想验证的问题**：在相同训练、target、预算和 kernel 下，静态树、EAGLE-2 value 树、成本感知树哪个得到最佳 TPOT？value 信号是否在长 context、量化 target、OCR/VLM 多模态前缀上仍校准？这些都需要独立实验，本精读不声称已证明。

## 相关精读与参考资料

**Next Reading**：[EAGLE-3：让草稿模型学会接住自己的输出](https://howieyin.github.io/papers/eagle-3/)。动态树之外，继续看自由状态、多层特征融合和 training-time test 如何提高多步草稿的可靠性。

[DFlash：并行扩散草稿与目标特征 KV](https://howieyin.github.io/papers/dflash/) · [EAGLE：先知道下一个 token，再外推它的特征](https://howieyin.github.io/papers/eagle/)

[Transformer：注意力与因果位置](https://howieyin.github.io/papers/transformer/) · [Sparsely-Gated MoE：路由与执行链路](https://howieyin.github.io/papers/sparse-moe/)

**Original Paper**：

- [S1 · arXiv 2406.16858v2 PDF](https://arxiv.org/pdf/2406.16858v2)；[v2 HTML](https://arxiv.org/html/2406.16858v2)。7 图、3 表、主公式与附录的主依据。
- [S2 · EMNLP 2024 最终版](https://aclanthology.org/2024.emnlp-main.422/)，用于 venue/DOI 核验。

**Previous / related papers**：

- [S3 · EAGLE, 2401.15077v3](https://arxiv.org/abs/2401.15077v3)，继承的 feature draft、训练与参数。
- [S4 · Fast Inference from Transformers via Speculative Decoding, 2211.17192v2](https://arxiv.org/abs/2211.17192v2)，标准采样修正。
- [S5 · Sequoia, 2402.12374](https://arxiv.org/abs/2402.12374)；[Medusa, 2401.10774](https://arxiv.org/abs/2401.10774)；[Hydra, 2402.05109](https://arxiv.org/abs/2402.05109)。

**Official implementation**：

- [S6 · EAGLE repo 固定快照](https://github.com/SafeAILab/EAGLE/tree/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b)；[cnets1.py](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/cnets1.py)；[ea_model.py](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/ea_model.py)；[utils.py](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/utils.py)。

**Follow-up / current state · accessed 2026-10-27**：

- [S7 · EAGLE-3, 2503.01840v3](https://arxiv.org/abs/2503.01840v3)。
- [S8 · FR-Spec, 2502.14856v2](https://arxiv.org/abs/2502.14856v2)。
- [S9 · SGLang speculative decoding docs](https://sgl-project.github.io/advanced_features/speculative_decoding.html)。
- [S10 · EAGLE-3.1 官方博客, 2026-05-26](https://vllm.ai/blog/2026-05-26-eagle-3-1)。
- [S11 · PEAGLE 官方博客, 2026-03-13](https://vllm.ai/blog/2026-03-13-p-eagle)。
- [S12 · DFlash v2, 2602.06036v2](https://arxiv.org/html/2602.06036v2)；[SGLang DFlash v2, 2026-06-15](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2)。
- [S13 · JetSpec, 2606.18394v5, 2026-10-03](https://arxiv.org/html/2606.18394v5)。

**Changelog**：2026-10-27，首次全文精读；以 v2 为准，补充当前官方代码与 2026 技术路线。2026-10-28，重新裁切全部 7 张原图和 3 张原表，将 caption 与子图说明移至图片外的可选文字，保留图内标注与完整数据。本文未运行 GPU benchmark、未重新训练 draft。
