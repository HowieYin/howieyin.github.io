# Sparsely-Gated MoE：容量可以稀疏激活，成本仍在整条链路

把“模型拥有多少知识容量”和“每个 token 使用多少计算”分开，是这篇论文真正改变的事情。

<p class="reading-meta">2017 · arXiv v1 · MoE / LLM / Training / Systems<br>精读与资料核验：2026-10-27 · 全文 19 页 · 4 张原图 · 9 张完整原表 · 22 个编号公式</p>

[原论文 v1](https://arxiv.org/abs/1701.06538v1) · [原始 PDF](assets/original/1701.06538v1.pdf) · [历史官方代码对照](https://github.com/tensorflow/tensor2tensor/blob/v1.15.7/tensor2tensor/utils/expert_utils.py) · [相关精读：Transformer](../transformer/)

## 阅读坐标与论文元数据

**Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**，Noam Shazeer、Azalia Mirhoseini、Krzysztof Maziarz、Andy Davis、Quoc Le、Geoffrey Hinton、Jeff Dean；机构为 Google Brain 与 Jagiellonian University。本次严格读取 2017-01-23 提交的 **1701.06538v1**；题头为“Under review as a conference paper at ICLR 2017”，以该版本为依据，不把后来的文稿措辞回填。DOI：[10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)。未确认独立项目页与该版本对应的完整训练发行包。

适合已了解矩阵乘法、softmax、反向传播和 LSTM 的读者。读完应能重建 token 到专家再回到主干的链路，区分三种负载量，核算专家参数与数据移动，并判断实验究竟支持了什么。正文按原论文 §1–6、附录 A–G 排列，之后补充工程重构、批判性分析与截至 2026-10-27 的后续研究。

页面约定：**Paper** 为原论文事实；**Interpretation / 推导** 为这里的解释与算账；**External Source** 为后续原始论文或官方资料。交互只演示机制和静态资源量，不模拟精度或推理速度。

## TL;DR：一句话、一段话与五个关键点

**一句话：** 用可训练的稀疏路由让每个 token 只访问少数专家，以近似固定的专家计算扩大模型容量，但必须同时解决路由学习、小 batch、负载不均和跨设备数据移动。

**一段话：** 论文在循环语言模型与翻译模型中插入大量独立前馈专家，用带可学习噪声的 Top-K gate 选择并混合少数输出；用 Importance 与平滑 Load 的辅助损失抑制专家垄断；把数据并行的 token 汇到模型并行的专家上，以扩大有效专家 batch。它在近似相同的每步算术预算下观察到更低 perplexity，也训练了最高约 137.7B 总参数的模型。新增参数依然要保存、分片、更新和读取；最终速度由整条执行链路决定。

- **容量与激活量分离。** 专家总参数近似随专家数增长，单个 token 的专家 MAC 主要随 k 增长。
- **路由并非“先 softmax 再随便屏蔽”。** 主方法先对带噪 logits 做 KeepTopK，再在选中集合内 softmax。
- **均衡有三个量。** gate 权重和 Importance、实际 token 计数、可微的条件概率 Load，不能混用。
- **实验最有力的是同预算容量曲线。** 1B 词数据集同预算 LSTM 的 PPL 44.7 到 MoE 的 34.1；最大规模不是最佳质量。
- **两处常见误读。** 原始骨干是 LSTM；附录 F 的 batchwise gate 不保证每个 token 恰好 k 个专家。

## §1 Introduction：为什么扩大容量会撞上计算预算

### 问题、观察与核心贡献

**Paper。** 神经网络通常把更多参数与更多计算绑定：加宽、加深意味着每个输入都经过新增计算。Conditional computation 想让参数只在需要时激活，但实际收益受训练可行性和硬件执行方式约束。论文点名的障碍包括动态选择难求梯度、专家 batch 太小、网络通信、负载不均，以及模型规模与数据需求。

**直觉。** 可以把专家看作一批不同参数的函数。token 仍有统一的表示空间，router 根据表示决定调用哪几个函数。容量不是每次都翻遍整座图书馆，而是拥有更多可被选择的书架。这个比喻解释参数复用，不意味着专家天然对应可解释主题。

此前 Jacobs 等的 mixture、Jordan 与 Jacobs 的层级 mixture，以及 deep mixture 和条件计算工作已存在。本文的创新在于把**稀疏、可训练、可分布式执行的大规模 MoE 层**落到实际语言任务，不能称它“发明了 MoE”。它也早于 2017 年 Transformer 论文，本身没有 self-attention、RoPE 或 Transformer KV cache。

### Figure 1：MoE 在循环语言模型里的位置

{{fig1}}

从下往上读：输入进入下层 LSTM，gate 从当前输入表示选择图中少数专家，各专家执行前馈计算，输出乘 gate 权重后求和，结果送上层 LSTM。图中的专家连线强调选择性执行，循环连接则属于 LSTM。此图示范 **k=2**，不能据此把全部语言模型实验写成 top-2；后面的主要 LM 实验使用 top-4。

图的证据功能是说明网络连接，而不是展示速度测量。它没有画 token 打包、跨设备传输和梯度通信；这些是实现中不能忽略的隐含链路。我的理解是：MoE 改变的是这一层所访问的参数子集，序列状态仍由上下两层 LSTM 维护。

## §2 Structure：带噪 Top-K 怎样成为可训练的稀疏层

### 输入、专家、输出与张量形状

记专家数量为 n，输入 x∈Rᵈ，专家输出 Eᵢ(x)∈Rᵈᵒᵘᵗ，gate G(x)∈Rⁿ。对 N 个 token，输入 X 为 N×d，gate 矩阵为 N×n，合并结果为 N×d_out。每个专家是有独立参数的前馈网络，而非一个 token 的身份标签。

{{eq1}}

Eq. (1) 的关键是零权重专家可以不执行。输出在共同的 d_out 空间相加，因此专家不需要把中间 hidden 向量互相拼接。对专家参数，梯度只通过被执行的路径传播；对 gate，任务梯度依赖被选专家输出的相对价值。

{{eq2}}

Eq. (2) 是 dense softmax gate：W_g∈Rᵈˣⁿ，xW_g 为 n 个 logits，所有专家权重通常非零，不能直接带来稀疏计算。它建立对照，后续主方法才加入噪声与截断。

{{eq3}}

{{eq4}}

{{eq5}}

Eq. (3)–(5) 的顺序是：计算 clean logits；用 W_noise∈Rᵈˣⁿ 预测每个专家的噪声尺度 σᵢ=softplus((xW_noise)ᵢ)>0；独立标准高斯噪声乘尺度再相加；保留最大的 k 个 logits，把其余设为 −∞；对剩下的 logits 做 softmax。**不能把 −∞ 改成 0**，否则原本应被删除的专家会重新得到概率。

噪声让早期路由有探索机会，也让附录 A 可以构造平滑的命中概率。可学习尺度允许模型调节这种扰动。Top-K 集合切换仍然不连续，噪声并没有使硬选择处处可微；可训练性来自选中集合内部的 softmax 梯度，以及辅助均衡目标。

### 一个数值例子与 top-1 梯度陷阱

logits=[2,1,0,−1]，k=2，不加噪时得到 [0.7311,0.2689,0,0]。选中专家输出分别为 a、b，则 y=0.7311a+0.2689b。提高第一个 logit 会增加 a 的占比；是否有利由任务损失决定。

但 k=1 时，在这个**截断后重新归一化**的定义下，选中专家的权重恒为 1。在选择集合不变的区域，gate 对选中 logit 的导数为 0，单靠任务损失无法训练 router。论文明确强调 k>1 的作用。后来的 Switch top-1 使用不同的 gate 处理，不能拿 Switch 的成功反证这里的导数结论。

```python
# 教学伪代码：主论文 §2，X: [N,d]，不是分布式实现。
clean = X @ Wg                           # [N,E]
std = softplus(X @ Wnoise)               # [N,E]
noisy = clean + normal_like(clean) * std
indices = topk(noisy, k, axis=-1).indices # [N,k]
gates = softmax(gather(noisy, indices))   # [N,k]
packed = group_tokens_by_expert(X, indices)
expert_outputs = run_selected_experts(packed)
Y = weighted_scatter_sum(expert_outputs, gates)
```

group / gather / scatter 需要保留 token 原始索引和 gate 对应关系；先汇总专家输出再错配权重，会破坏数学定义。主 gate 的正常行总和为 1，这不代表平滑 Load 估计在每次噪声实现中也严格总和为 k。

### 交互：观察截断、噪声与权重

{{router-widget}}

默认固定六个 clean logits，只修改 k、噪声幅度或随机种子。高斯数值仅用于机制演示，幅度滑块不等于已训练 W_noise。蓝色专家是本轮非零 gate；负的 logit 也可能被选中，权重由相对差决定。切到 k=1 后反复换噪声，可以看到专家身份变化，但选中权重一直为 1。

## §3 Performance：稀疏算术如何变成有效吞吐

### Shrinking batch：专家越多，每个 GEMM 越容易变小

如果 N 个 token 各选 k 个专家且完全均衡，每个专家平均处理 Nk/n 个 token。增加 n 并保持 N、k 不变，会减小专家 GEMM 的 M 维，降低计算利用率；它还让每个专家每步看到的训练样本更少。平均数不是最热专家的峰值，实际分布还决定等待和 activation 容量。

**Paper。** 一方面扩大 batch，另一方面结合数据并行与专家模型并行，让不同数据分片的 token 汇到相同专家。MoE 自身没有 recurrence，因此在下层 LSTM 已产生各时间步表示后，可以合并多个时间步送专家。这个前提不能直接迁移到自回归 decode：未来 token 尚未生成，无法预先攒入同一 batch。

### 网络带宽：多参数可以少通信，但路由不是免费

专家的 hidden 宽度增加参数和算术，专家输入输出宽度却不必同比增长。这改善了算术与传输数据量的比值，是本文认为 MoE 适合分布式执行的理由。然而每个 token 的跨设备输入分发和输出返回都需要通信；具体有多少取决于专家放置、k、局部命中和层级聚合。

{{pipeline-diagram}}

这是按数学依赖重构的执行图，现代术语 dispatch / combine 帮助定位阶段；原论文没有证明其实现使用今天的 NCCL all-to-all API。把 experts 换成空算子测出的 routing+pack+communication 时间，以及实际专家 GEMM 时间，才可以回答开销发生在哪一环。

## §4 Expert Utilization：Importance 平衡为何仍可能忙闲不均

### 正反馈怎样使少数专家垄断

若某专家早期稍好，gate 会更多选择它；它因此得到更多训练，又更好，其他专家则缺少样本。这个正反馈同时损害质量和吞吐。均衡辅助损失把“专家充分利用”加入训练目标，但不能用它推导每个专家被严格平均分配。

{{eq6}}

{{eq7}}

Importance(X)∈Rⁿ 是一个 batch 内每个专家 **gate 权重之和**。CV 为标准差除以均值，CV² 无量纲；w_importance 控制该项与任务损失的权衡。若每个 token 的 gate 和为 1，则所有专家 Importance 的总和为 N。

实际 token 数为 Cᵢ=Σₓ1[G(x)ᵢ>0]，它决定本专家的行数和执行工作量。反例：四个 token 的 gate 分别为 [2/3,1/3,0]、[2/3,0,1/3]、[0,1/2,1/2]、[0,1/2,1/2]。三个专家的 Importance 都是 4/3，CV=0；token 数却是 [2,3,3]。**gate 权重小，并不会让一个已经分发的 token 免费计算。**

附录 A 的 Load 用平滑选择概率近似计数，针对的是这个缺口。联合目标可写为任务损失+L_importance+L_load，这是机制重构；没有把辅助损失值当成最终任务指标。Table 6 的实验证据放在附录 A 的原始位置。

## §5 Experiments：质量收益与预算边界

### §5.1 One Billion Word：相同每步预算下增加专家

**问题。** 如果每个 token 仍使用少数专家，更多参数能否改善语言建模？数据集约 829M 训练词，词表 793,471，句子被打乱；这是词级 benchmark，不是现代长文档或指令评测。主要结构为 embedding512→LSTM512→MoE(512→1024→512)→LSTM512→输出词表。细节与完整结果在附录 C。

### Figure 2：容量曲线和算术预算曲线要分开看

{{fig2}}

左图横轴是参数容量、纵轴是 test perplexity，越低越好；比较的是约 8M ops/timestep 的模型。随专家数增大，MoE 曲线下降，而并非让每个 token 计算所有新增参数。右图横轴变成算术预算，比较 LSTM 与约 4B 专家参数的 MoE。左右图的控制变量不同，不能把它们拼成统一的 latency 曲线。

最直接的同预算比较来自 Table 7：LSTM 44.7、4096 专家 MoE 34.1，相对降低 **23.71%**。与 4 专家模型的 45.0 相比下降 24.22%。这是同一类训练预算下的容量收益；最强历史 LSTM 经 100 epochs 得到 30.6，则不能当成与 MoE 10 epochs 的 34.1 同训练量的对照。

### Table 1：高容量 MoE 的算术预算与训练时间

{{table1}}

高计算 MoE 达到 PPL **28.0**，对表中 10 epochs 历史 baseline 34.7 下降 **19.31%**。附录 C.2 写“18%”；这里保留原作者措辞的来源，同时明确按可见表值重算为 19.31%，没有悄悄把不一致抹平。

最省算术配置的 8.9M/151M=**5.89%**，接近摘要的 6%。原论文 ops/timestep 统计一次乘加为一次、只计 forward 且排除输出 softmax。训练 TFLOPS/GPU 则含 backward 和 sampled softmax，一次乘加算两次。两者不能直接相除或混写成端到端 FLOPs。

同表时间给出 15h×16 GPU=240 GPUh，对历史 59h×32 GPU=1888 GPUh 为 **12.71%**；这与 5.89% 算术占比不同。硬件、实现与训练配置仍需对齐，不能据此声称现代 decode 快 17 倍。

### §5.2 100 Billion Word：数据增大后，容量拐点在哪里

**Paper。** 用私有 Google News 100B 词语料进一步扩大专家数，最多 131,072 个；采用层级路由，固定全局约 2.5M 词 batch，以约 8M ops/timestep 比较。模型最大约 **137.7B 总参数**，不是 137.7B active parameters。完整表在附录 D。

### Figure 3：最大容量不等于最佳质量

{{fig3}}

沿专家数量读曲线，比较 10B / 100B 训练量与不同模型的 test PPL。更大数据支撑了更大容量，但曲线并非永远单调。Table 8 中 65,536 专家模型为 **28.9**，131,072 专家为 **29.2**；相对 100B 训练的 4-LSTM 47.0，最优点下降 **38.51%**。

固定 batch、k=4 时，两者每叶专家平均 token assignment 从 **152.59** 降至 **76.29**。训练效率从 0.72 到 0.30 TFLOPS/GPU，下降 **58.33%**。这支持“容量扩展有系统拐点”，但论文没有逐算子测量来证明下降唯一由小 batch 引起。语料也无法按公开资料原样复现。

### §5.3 单语言对翻译：BLEU 更高，但结构与硬件同时变化

MoE 分别插在 encoder 与 decoder 的 LSTM 层间，最多每层 2048 个专家，层级 top-2×top-2。主干层数、attention 实现、训练与硬件均不同于 GNMT，因此这是系统组合的结果，不能当成只替换 FFN 的干净消融。附录 E 与 F 还有一个需要揭示的路由配置冲突。

{{table2}}

Table 2：WMT’14 En→Fr，MoE 3 天 40.35、6 天 **40.56**，GNMT 39.22、GNMT+RL 39.92。分别高 **1.34** 与 **0.64 BLEU**。这些是绝对分差，不能与相对 PPL 改善混写成统一百分比。

{{table3}}

Table 3：En→De，**26.03 对 24.91**，高 **1.12 BLEU**。最强 baseline 应保留其训练预算与模型名称，不只摘 MoE 行。

{{table4}}

Table 4：生产 En→Fr，**36.57 对 35.56**，高 **1.01 BLEU**，但数据私有。表中 85M/214M=**39.72%** forward ops；MoE 用 64 K40、GNMT 用 96 K80，不能把 wallclock 比直接解释为算法单独贡献。该版本使用 tokenized Moses `multi-bleu.pl`，没有现代 sacreBLEU signature；跨时代分数不具备直接可比性。

### §5.4 多语言翻译：共享容量能缓解任务干扰

12 个翻译方向共同训练时，小 dense 模型面临容量竞争。**这里的配置改为 flat 512 专家、top-2、专家 hidden8192，每个 MoE 层如此**，不能照搬上一节的 hierarchical2048 / hidden2048 / 四叶路径。总参数 8.7B，102M ops/timestep。

{{table5}}

Table 5 的开发集 PPL 4.14→3.35，相对下降 **19.08%**。MoE 超过 multilingual GNMT 的 11/12 方向，超过独立 monolingual 模型的 8/12 方向。反例 En→Ko 是 **16.62 对 18.41**，差 **−1.79 BLEU**。作者提出 oversampling 与过拟合解释，但没有针对性消融验证，不能写成已证实的单一因果结论。

训练硬件亦不一致：GNMT-Multi 为 96 K20、21 天，MoE 为 64 K40、12 天。保留参数、算术、时间和全部 12 个方向，才能看见总体收益与局部退步同时存在。没有多随机种子、显著性检验或置信区间，微小分差应审慎解释。

## §6 Conclusion：它证明了什么，又留下什么

**Paper。** 稀疏 MoE 可以在较低的每步算术预算下扩大参数容量，并在语言建模和翻译上获得质量收益；规模化的关键是可训练 gate、专家利用率和分布式执行。

**Interpretation。** 最可靠的历史结论是：参数容量不必与每个 token 的专家算术严格绑定。本文没有证明固定总内存、固定网络带宽、batch1 decode、任意专家数下都能保持同样效率。它提出了一条能扩展的路线，也用最大模型的效率下降暴露了这条路线的边界。

## 附录 A Load：用条件概率给“是否选中”一条梯度路径

### Eq. (8)–(11)：重新采样本专家的噪声，其他专家固定

硬计数不可微。对 token x 与专家 i，固定其他专家已抽出的噪声，只重新抽 i 的噪声；定义 τᵢ 为排除 i 后，其余 logits 的第 k 大值。i 能进 Top-K，当且仅当自己的带噪 logit 超过 τᵢ。

{{eq8}}

{{eq9}}

若本专家 clean logit 为 μᵢ、噪声尺度为 σᵢ，P(x,i)=Pr[μᵢ+εσᵢ>τᵢ]=Φ((μᵢ−τᵢ)/σᵢ)，Φ 是标准高斯累积分布。Eq. (9) 的符号来自 1−Φ((τ−μ)/σ)=Φ((μ−τ)/σ)。P 为标量，每个 batch 的矩阵为 N×n，μ 和 σ 提供平滑的梯度路径。

一个实现易错点：若 i 本来在 Top-K 中，排除 i 后阈值为全局**第 k+1 大**；若 i 未入选，阈值为全局**第 k 大**。例 H=[5,4,3,2]、k=2，各专家阈值依次 [3,3,4,4]，绝不能给所有专家共用阈值 4。这里是条件估计，并非重新采样全部噪声的直接 Monte Carlo 频率；单次实现 ΣᵢP(x,i) 不要求精确等于 k。

{{eq10}}

{{eq11}}

Load(X)∈Rⁿ 是 batch 内选择概率之和，单位接近 token assignment。L_load 用 CV² 抑制负载偏差。它仍是平滑估计，真正的内存容量与最慢专家时间要看实际计数分布。W_g 与 W_noise 零初始化时 σ=softplus(0)=ln2，不等于零噪声；这使初始路由具有随机分散性。

### Table 6：两个均衡损失是否有必要

{{table6}}

没有均衡项时 PPL=39.8，Importance CV=3.04、Load CV=3.01，最大/平均负载=17.80。Importance 权重 .2、Load0 时 PPL35.6，但 Load CV=.17；只用 Load.2 时 PPL35.7、Importance CV=.22。两者 .1/.1 时 PPL35.6、CV .06/.05、最大/平均1.14，说明权重和与工作量的平衡目标互补。

加大到1/1后 CV 更低（.03/.02），PPL35.7 并没有继续改善；最小 CV 不是唯一目标。指标是 batch 平均，没有方差区间；这个表证明失衡治理重要，不足以证明 .1 是跨任务最优系数。

## 附录 B Hierarchical MoE：先选组，再选组内专家

### 架构与 Eq. (12)–(14)

平坦路由需为所有 n 个专家打分。层级方案有 a 个组、每组 b 个叶专家，共 ab 个专家；第一层 gate 选组，第二层 gate 只在被选组内选叶。输出 gate 是两层权重的乘积。top-2×top-2 在标准逐 token 路由下有四个叶路径，组间专家参数不共享。

{{hierarchy-diagram}}

{{eq12}}

Eq. (12)：Eᵢ,ⱼ(x) 输出在相同 d_out 空间。第一层选择会限制第二层的候选，错过组时不能直接访问该组最好的专家，这是减少 router 候选代价的 trade-off。平坦打分代价约 Ndab，层级约 Nda+Nk_primary·db，不含 Top-K、数据移动与专家计算。

{{eq13}}

Eq. (13) 的 Importance_H 为 a×b 的权重和，可以直接由 gate 乘积累加。只在组内平衡并不能解决组间流量偏斜，因此一二级均衡需一起考虑。

{{eq14}}

X⁽ⁱ⁾ 是实际送到组 i 的 token 集合。Eq. (14) 用 primary 的平滑 Load 乘 local 条件比例，替代不可微的纯叶计数，使第一层也获得均衡梯度。若组为空，分母为零，工程实现必须 guard；不能不加处理直接翻译公式。层级降低 gate 开销，却新增了调度层级和粗粒度选择瓶颈。

## 附录 C One Billion Word：复原结构、训练与完整结果

### C.1：张量与训练链路

输入 IDs B×T，经 embedding512 成 B×T×512，下层 LSTM512 产生表示；合并可用时间步为 N×512 送 MoE，专家 512→1024 ReLU→512；MoE 输出再加 sigmoid、dropout 与 residual，接上层 LSTM512，投影词表 793,471。输出 softmax 与大词表会影响完整成本，不能因 ops 表排除它们就当不存在。

全局 batch 约 300K 词，16 张 K40，27K 步对应约 10 epochs。Adam，warmup1000，后续 inverse square root；dropout 以 .1 网格调，Importance/Load 系数 .1。非 softmax 层的输出均做 dropout 与残差；输出 softmax 用 importance sampling 高效训练。PPL 是指数化的平均负对数似然，越低越好，评估包括句末符号。论文没有建立现代 FP16/BF16 的精度设定，保持“未明确报告”。专家数多时需要跨数据分片汇聚 token，不应误用每卡 batch 算全局专家平均负载。

标准层级 LM 的首层 branching=16，对应16 GPU；普通 MoE 使用 k=4，层级每层 k=2。MoE-4 会执行全部4个专家，训练约18小时，因为专家计算只在4/16 GPU上，其他同预算模型约12–16小时。dense 对照还包括单专家 hidden4096 的 wide、4层 hidden1024 的 deep、4×LSTM512，以及重跑的 projected LSTM2048→512；不仅比较一个弱基线。

### C.2：同容量改变算术，完整 Table 7

{{table7}}

表分组分别对应同预算对照、历史高计算 baseline 与高容量 MoE 的计算扫描。8.9M 配置从 4 专家45.0 到32专家39.7、256专家35.7、1024层级34.6、4096层级34.1；wide LSTM46.1、deep LSTM45.7 并没有复制同样的容量收益。256 flat35.7 与 hierarchical36.0 提示层级不是无代价替代，单点不能给出普适优劣。

4096 配置总参数5.1B，其中**排除 embedding 与输出 softmax**后为4303.4M；单专家权重约 2×512×1024=1,048,576，4096 专家共4,294,967,296。差额来自 gate 与主干等部分。不能用专家数乘单专家参数后声称已得到“完整模型参数”。

高计算配置的 embedding 与 MoE 输入输出改为1024。MoE-34M 用 LSTM1024，1024个层级专家、hidden2048；MoE-143M 用 LSTM4096并投影1024，256个层级专家、hidden8192。两者专家本体均约4B，分别 PPL31.3 / 28.0。高计算143M配置比8.9M配置34.1更好，说明少激活并不消灭任务所需计算；它同时改变主干与专家形状，不能把改善全部归因于增大 k。

## 附录 D 100 Billion Word：参数、优化器和专家 batch 的账本

### 固定 batch 的扩展方式

一般32 GPU，65,536专家64 GPU，131,072专家128 GPU；层级首层 branching 随规模为32、32、64、128、256、256。每叶仍为512→1024→512，全局2.5M词 batch 保持固定，训练一遍100B词。规模扩大让存储分片可行，但并未同比增加每专家 batch。

专家 activation 可重算以换显存。优化器设 β₁=0，第二矩采用矩阵行列的 factored estimates，而不是每个元素完整维护二阶状态；这预示了后来的 [Adafactor](https://arxiv.org/abs/1804.04235)。它减少部分优化器状态，不删除权重本体，也不能把训练显存等同于参数字节数。

{{table8}}

最大模型**排除 embedding 与输出 softmax**后为137,577.6M，专家本体2×512×1024×131,072=**137,438,953,472** 参数，完整模型约137.7B。最佳65,536专家则约68.9B总参数、PPL28.9。表中 Kneser–Ney 分别用13B/130B词，而 neural对应10B/100B，训练量并不相同，原文 footnote 4 的预算区别应一起读。

### 交互：容量、平均专家 batch 与通信上界

{{cost-widget}}

这里 d 是专家输入/输出宽度，h 是 hidden；只计两层矩阵权重，不含 bias、gate、LSTM、embedding、optimizer 与 activation。存储精度选项只是字节假设，不能说明原论文用了该精度，更不能表示硬件支持。前向流量按全部路由远端、每条输入与输出各传一次计算，未含 backward、metadata、层级中间传输与 transport 复制。

## 附录 E Machine Translation：配置例外与 Figure 4 / Table 9

### 单语言对与多语言不能合写成一套模型

单语言对采用3层 encoder LSTM与2层 decoder LSTM；内部2048维、投影到512。encoder的MoE位于第2/3层间，decoder位于第1/2层间；专家 hidden2048，每个约2M参数，sigmoid与residual。32K共享 wordpiece，沿用 GNMT beam，并采用附录 G 修改过的 attention。与文中 GNMT9/8层对照时，需要保留主干变化。

warmup2000→constant8000→inverse square root；single-pair dropout .4，16K词/GPU，均衡系数 .01。多语言则 flat512 / top2 / hidden8192，总参数8.7B、102M ops；single-pair为85M。数值看似都称 MoE，实际容量分配与计算配置不同。

**原文措辞冲突。** E 的描述说每个输入使用恰好四个专家，但同段指定了附录 F 的 strict batchwise gate。F 保证的是每个专家接收固定配额，单 token 的专家数可变。这里明确保留此不一致；只有标准逐 token 的两级 top-2 才能直接保证四叶，不能替原作者虚构一个未写出的 coverage 修复。

### Figure 4：训练词数曲线避免了哪种误判

{{fig4}}

横轴是已经处理的词数，纵轴是 perplexity；左为 WMT En→Fr，右为生产 En→Fr。曲线比较专家规模，除 no-experts 外保持85M ops/timestep。作者特别指出开头的差异来自 batch size，因此前期更低 PPL 不能直接读成“每个优化步学得更好”，更不能把 processed words 改成 seconds。

图说明容量优势在训练轨迹上出现，但不独立隔离 gate、attention、batch 与模型深度。它与主文 BLEU 表测量的对象也不同：训练/开发困惑度曲线不是测试集翻译分数。

### Table 9：专家是否形成了语义分工

{{table9}}

原表从2048个 encoder专家中挑出3个，按训练 batch 内 gate 权重降序展示高分上下文，粗体词保留。可看出 researcher/innovation、leadership短语与rapidity相关的局部模式。这支持“存在可观察分工”的定性解释，不能证明所有专家解耦、每个专家都有稳定主题，或移除专家会造成特定语义能力缺失。

更扎实的验证应在独立数据上统计主题/语言与路由的关系，控制 token频率，并做专家屏蔽或重路由的因果干预。只展示高 gate 样本容易产生选择偏差。

## 附录 F Strict Batchwise Gating：固定列配额不等于固定行配额

### Eq. (15)–(18)：选择方向发生了变化

{{eq15}}

{{eq16}}

这里先对全体专家做 softmax得 scores S∈Rᴺˣⁿ，再乘 mask并逐 token重新归一化。Eq. (16) 分母必须大于0。它与主方法的 noisy logits Top-K不是同一算法描述，应分别实现。

{{eq17}}

Eq. (17) 给出0/1选择 mask，与 Eq. (5) 的保留 logit / −∞不同。标准 token-choice为每一行选k个专家；接下来的 batchwise改成每一列选m个token。

{{eq18}}

设 m=kN/n，训练对每个专家列保留最大的m个 scores。每列处理量固定，整个矩阵的 assignment数为nm=kN，但**行数可变**。N=4、n=2、k=1、m=2时，如果两个专家都喜欢 token0/1，就可能出现行 coverage=[2,2,0,0]。token2/3没有专家，Eq. (16) 分母为0。v1没有完整说明零覆盖fallback与m不是整数时的处理；教学交互把零覆盖标出，不能暗中归一化成有效路径。

从系统角度，这是把 expert batch的确定性放在前面，把 token的可变计算与覆盖风险放在后面。它与2022年的 Expert Choice有机制上的亲缘关系，但二者不是完全相同的方法，不能用同名简写省掉各自训练与容量约束。

### Eq. (19)–(20)：推理为何需要学阈值

{{eq19}}

训练有大 batch可排名，推理尤其单 token时没有同样的列竞争。作者学习每专家阈值Tᵢ，通过 score>Tᵢ决定mask，近似训练的 batchwise选择。阈值不是逐token Top-K，因此不保证严格均衡或严格k，也可能出现不同于训练的负载和覆盖。

{{eq20}}

Eq. (20) 以训练mask为目标，惩罚阈值mask与它的差异：false positive时score−T>0，梯度下降提高阈值；false negative时score−T<0，梯度下降降低阈值。忽略mask切换点的导数，此方向可由分段线性项直接核算。原式x记号不够明确，重构时将它理解为对应的score行，而非原输入hidden向量。

可验证的部署指标应包括 mask agreement、每token覆盖分布、零路径比例、专家峰值负载，以及batch变化前后的质量。论文没有报告这些，所以不能宣称阈值推理与训练gate完全等价。

## 附录 G Attention：为矩阵乘法改写的双线性交互

{{eq21}}

原 GNMT score对encoder状态xᵢ与decoder状态yⱼ先投影、相加、tanh，再以V求和。如果源长S、目标长T、attention hidden为r，直接形成pairwise hidden可能需要S×T×r的中间激活。

{{eq22}}

新score把tanh分别放在两侧：P=tanh(XU)∈Rˢˣʳ、Q=tanh(YW)∈Rᵀˣʳ，A=(P⊙V)Qᵀ∈Rˢˣᵀ。于是可以预计算投影，再用GEMM；避免显式S×T×r激活，但S×T score与后续softmax仍在。两式非等价代数变换，而是改变了score函数。

这是2017版本为实现效率所做的设计，不是Transformer的scaled dot-product attention，也没有1/√d。翻译实验的baseline审计必须把该改动算进去，否则会把整套系统质量和效率全归因到MoE。

## 工程重构：参数、MAC、内存与 per-op 调用链

### 统一账本与估算边界

两层专家 d→h→d，权重P_expert≈2dh；n个专家P_total≈2ndh；每token的激活专家权重与专家MAC≈2kdh。这里的“active参数”只针对这一层专家，不包括每token都会访问的dense主干。若现代专家使用SwiGLU三矩阵，参数项近似3dh，不能照搬2dh。

训练还要存梯度、优化器状态、activation与workspace。FP32专家本体137.44B参数约549.76GB十进制，仅是静态权重；不同分片与精度会改变每卡常驻量。没有报告足够的内存峰值与精度，不能从论文反推出准确显存曲线。

前向路由数据移动上界为 N·k·(d_in+d_out)·s bytes，s为每标量字节数；全部远端、输入输出各一次、layer-wide。d_in=d_out=512、k=4、s=2时每token8192 bytes；N=2.5M时20.48GB十进制。注意该2字节只是估算假设；这个公式不计路由metadata、反向、层级内转发、padding与协议开销。

### 从调用链定位真实损耗

| 阶段 | 主要操作 / 形状 | 可能瓶颈 | 应记录的证据 |
|---|---|---|---|
| Router | XW_g、XW_noise：N×d×n | 平坦专家多时dense打分；低batch启动 | 两个GEMM、Top-K、softmax分开计时 |
| Pack | token→expert分组，索引与gate | scatter/gather、小kernel、host同步 | 排序/排列/元数据时间、同步点 |
| Dispatch | 向专家设备发N×k份输入 | RDMA/NVLink带宽、拥塞、负载偏斜 | 远端比例、传输字节、等待尾部 |
| Expert | Nᵢ×d→h→d | 小Nᵢ、权重带宽、GEMM利用率 | 每专家batch直方图、GEMM时间 |
| Combine | 结果返回、weighted scatter-add | 原索引恢复与归约、重复数据移动 | combine时间、有效带宽 |
| Main path | sigmoid、dropout/residual、LSTM/输出 | sparse层之外的固定开销 | 完整step、softmax、backward拆分 |

端到端critical path取决于阶段重叠与最慢专家，不能把每环孤立峰值吞吐直接相加成速度保证。公平测量应固定硬件、训练token量、质量目标与batch，既报算术，也报每op和整个step时间。

### Prefill、Decode、KV 与端侧 offload

原MoE前馈层没有自己的Transformer KV cache；原模型保持LSTM recurrent state。现代MoE Transformer中，专家数改变FFN参数容量，attention KV还由层数、KV头、head维与序列长决定，除非同时改attention架构。低active参数不意味着小KV。

大batch或prefill能汇聚更多token，提高专家GEMM效率；batch1 decode要对少量选中专家读取大量权重，可能受内存带宽或cold expert miss支配。把专家offload到CPU/flash时，除了active MAC，应测**TTFT、TPOT、常驻总权重、专家cache命中、冷启动权重读取、PCIe/存储传输与峰值物理内存**。本次没有GPU/NPU训练或推理基准，不给未经测量的端侧加速数字。

### 官方历史实现对照与复现范围

[Tensor2Tensor v1.15.7 `expert_utils.py`](https://github.com/tensorflow/tensor2tensor/blob/v1.15.7/tensor2tensor/utils/expert_utils.py) 中的 `noisy_top_k_gating`、`_prob_in_top_k`、`cv_squared` 与 `SparseDispatcher` 能对照主公式、阈值选择和打包/合并。该实现噪声std有额外1e−2、训练开噪声而推理关闭；这些应作为**版本化实现细节**标注，不能说全部逐字来自v1公式，也不能宣称它就是本文实验的原始完整checkpoint。

复现分三层：数学机制可用本页确定性例子核验；公开1B Word任务仍需重建全部训练配置；私有100B News与生产翻译不能按公开资料原样重跑。本文的数学计算已独立检查，性能和质量结果均来自论文报告。

## 批判性审阅：强项、隐藏代价与缺失实验

**强项。** Novelty在于把条件计算从概念变成高容量语言模型；公式能重建，均衡消融清楚暴露垄断风险；同每步算术预算的容量对照有说服力；同时讨论小batch与网络带宽，说明作者知道稀疏选择不自动等于高吞吐。

**弱项。** 硬件、attention、主干深度与训练预算在跨系统对照中变化；私有语料限制可复现性；没有随机种子方差、详细显存峰值和per-op profiling；附录F的零覆盖、非整数配额与训练推理一致性没有完整交代；专家分工的证据是选择性的定性上下文。

**隐藏trade-off。** 更多专家同时增加总存储、router打分、碎片化与每专家样本稀释；层级降低候选搜索，却可能提前排除正确组；均衡有利吞吐，也会约束自由分工；严格专家配额把不确定性转移到token覆盖。对硬件的依赖体现在batch、网络拓扑与内存带宽，不能脱离这些条件评估scalability。

如果重新评审，我会要求以下补充实验：

1. 固定专家规模、算术、硬件和训练量，分别消融噪声、k、层级与两种损失；报告多seed均值/方差。
2. 对F逐token coverage、零覆盖、train/inference mask agreement与batch变化做测量，明示fallback。
3. 做router、pack、dispatch、expert、combine、backward的时间/字节分解，联同专家计数尾部分布。
4. 固定质量目标测训练成本；固定服务负载测TTFT/TPOT、吞吐和峰值内存，区分prefill/decode。
5. 对跨语言路由与专家主题做holdout统计和因果干预，测试共享容量是否真在减少负迁移。

这些是这里提出的验证方向，不能当成原作者已完成的实验证据。

## Related Work Matrix：每条路线解决哪一个瓶颈

| 工作 / 原始来源 | 年份 | 核心方法 | 优势 | 代价 / 边界 | 与本文关系 |
|---|---:|---|---|---|---|
| Jacobs / Jordan mixture（见原文参考文献） | 1991 / 1994 | 混合与层级专家 | 选择多个参数函数 | 早期规模与执行条件不同 | 思想前史，不是本文之后才有 |
| [GShard](https://arxiv.org/abs/2006.16668) | 2020 | Transformer MoE、自动分片 | 把稀疏容量扩展到大规模TPU | 通信、容量与硬件依赖 | 专家容量与系统分片 |
| [Switch Transformer](https://arxiv.org/abs/2101.03961) | 2021 | 简化top-1与稳定训练 | 降低路由/专家计算复杂度 | capacity与token丢弃等约束 | gate formulation有差别 |
| [GLaM](https://arxiv.org/abs/2112.06905) | 2021 / 2022 | 大规模稀疏语言模型 | 质量与计算/能耗比较 | 不同数据协议不能横比 | 通用语言能力扩展 |
| [ST-MoE](https://arxiv.org/abs/2202.08906) | 2022 | 稳定训练与迁移 | 处理稀疏训练/微调可靠性 | 稳定性需配套设计 | 超越“能训大”的问题 |
| [Expert Choice](https://arxiv.org/abs/2202.09368) | 2022 | 专家选择token | 固定expert bucket、可变token计算 | 覆盖与任务约束要处理 | 与附录F机制相近，非同一算法 |
| [MegaBlocks](https://arxiv.org/abs/2211.15841) | 2022 / 2023 | dropless block-sparse执行 | 减少丢token与padding取舍 | 需要高效sparse/grouped kernel | 对付碎片batch和动态形状 |
| [Adafactor](https://arxiv.org/abs/1804.04235) | 2018 | factored二阶状态 | 降低optimizer存储 | 近似估计及优化超参 | 本文D已有行列状态前兆 |
| [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | 2024 | 细粒度专家、共享专家 | 减少冗余、改善组合分工 | 更复杂路由/执行 | 改造“专家怎样切” |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 2024 / 2025 | aux-loss-free balancing、FP8与EP | 大规模容量/通信协同 | 训练设施与数值稳定性要求 | 均衡方法与系统栈发展 |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | 2025–2026 | dispatch/combine通信库 | 优化MoE数据移动 | GPU/网络/软件要求严格 | 直接回应§3的通信问题 |

GShard报告600B级模型、2048 TPUv3、约4天训练；GLaM报告1.2T家族；ST-MoE讨论269B稀疏模型与约32B dense compute可比。这些数值只标明各自研究尺度，不能当成同平台速度/质量排行榜。

## Technology Evolution：从条件计算到容量、路由和系统共同设计

{{timeline-diagram}}

1991/1994的mixture与层级思想、2013的deep mixture、2015的conditional computation构成前史；2017本文把稀疏路由、均衡与分布式训练汇在一起。2020 GShard推进Transformer与分片，2021 Switch简化选择，2022出现稳定迁移、expert-choice和dropless执行的并行分支。2024细粒度/共享专家改变容量组织，2025–2026把均衡、通信、动态复制与异构容量继续合在系统层面优化。

箭头表示问题的延续与技术关系，并不宣称每篇工作严格从前一篇继承代码。MoE、attention sparsity、KV压缩、optimizer状态压缩是不同轴；[Transformer](../transformer/)、[GQA](../gqa/)、[FlashAttention](../flashattention/) 的相关精读可分别补齐注意力、KV和IO视角。

## Current State：截至 2026-10-27 的可核验进展

### 模型规模、开放性与 benchmark 条件

| 一手来源 | 报告规模 | 当前资料中的指标 | 开放性 / 严格可比性 |
|---|---|---|---|
| [DeepSeek-V3 v2](https://arxiv.org/html/2412.19437v2) | 671B总参数、37B active | 原报告语言/推理/代码任务套件 | 开放权重；原协议与2026报告不同 |
| [DeepSeek-V4报告](https://arxiv.org/html/2606.19348)、[官方发布](https://api-docs.deepseek.com/news/news260424) | Pro1.6T/49B active；Flash284B/13B active | Table6 ProMax：GPQA90.1、LiveCodeBench93.5、SWE-bench Verified80.6 | 官方链接开放权重；自报结果，不能当独立全球排名 |
| [Qwen3.8-Flash-Next 官方文章](https://qwen.ai/blog?id=qwen3.8-flash-next)（2026-08-26） | 125B主模型+51B n-gram，合176B存储；6B activated为作者口径 | GPQA91.7；SWE-bench Pro62.5 | [权重](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)与[技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)公开；脚手架与benchmark修订需一起读 |

Qwen文章的SWE-bench Pro比较含Qwen3.7Plus55.8、DeepSeekV4Flash0731 56.0、Opus4.6Max53.4；其说明使用修订benchmark与Claude Code（temperature1、top_p.95、256K上下文），其中Opus官方分数并非同一次重跑。不能直接拿这些数去比较DeepSeek-V4另一张表的SWE Pro55.4，更不能把SWE Verified和Pro混为同一benchmark。上述GPQA也来自不同报告协议，不作分差排名。

Qwen模型native context为262,144，扩展到1M依赖YaRN等设置，不能写成原生1M。51B n-gram容量使用host offload/prefetch，属于确定性索引访问的存储容量，不等同于router动态选择的expert FFN。它再次提醒：今天的“total / active”口径要逐部件读，而不是只看一个B数字。

**结论边界。** 这次检索确定了2026年仍在扩展稀疏与异构容量，也取得若干官方报告的benchmark进展；没有获得统一脚手架、统一budget、独立复测覆盖全体候选的10月全球评测，因此不宣称某模型是绝对SOTA。

### 当前执行栈：DeepEP 已不是旧的 V1 文档

2026-10-27读取的[DeepEP官方README](https://github.com/deepseek-ai/DeepEP)标为 **V2.5**：统一EPBuffer接口，DeepJIT、NCCL Gin，要求Linux、Hopper或更新GPU、CUDA13.1+、PyTorch2.10+、NCCL2.32.3+；说明移除NVSHMEM与V1。它还讨论动态专家复制、权重prefetch、gradient reduce。不能继续套用旧V1的依赖与API说明。

通信仍会消耗SM资源，优化all-to-all不意味着计算完全无干扰。[MegaBlocks当前README](https://github.com/databricks/megablocks)在Hopper推荐grouped GEMM。它们共同表明今日收益依靠kernel、调度、网络和数值栈，而非只把dense层改成if分支。本次核验资料与接口，不复现厂商性能数字。

## My Insights：把稀疏性理解为访问预算

**观点一：应把MoE至少拆成四本账。** 总参数账决定存储与更新，active参数账决定本token理论访问，expert batch账决定GEMM效率，数据移动账决定分布式与offload可行性。任一本账优秀，都不能单独推出端到端效率。这比“总参数大、active小”更接近实际优化决策。

**观点二：均衡指标必须与目标资源对应。** Importance服务于权重贡献，平滑Load服务于路由训练，actual count/峰值服务于设备容量与尾部耗时。若目标是TPOT，至少还要看batch1专家权重读入与cache miss；若目标是训练吞吐，还要看最热expert和通信排队。用一个CV覆盖全部目标容易误判。

**观点三：容量扩展与样本密度共同决定有效容量。** 同全局batch下，expert数翻倍会减半每步样本量。更大语料帮助参数被训练，却不自动修复每步的小矩阵问题。可以设计实验把batch随expert数增长，再比较同质量下的总训练成本，以区分统计效率与系统效率。

**观点四：gate的可训练性和服务可执行性应一起审计。** 主文top-1梯度退化与附录F零覆盖是两个不同问题，一个属于训练导数，一个属于服务路径定义。将它们与阈值推理、负载尾部、fallback行为放在同一版本化契约里，才能避免公式正确却部署链路不完整。

**建议方向。** 建一个可重复的MoE账本与per-op trace：输入模型配置、precision、expert放置和batch；输出理论参数/字节，再与router/pack/dispatch/GEMM/combine实测对照；进一步加入专家cache和TTFT/TPOT。这里的计算器只是第一层数学账本，后续的真实profiling能检验理论节省是否兑现。

## References、资料清单与更新记录

**Original Paper。** [arXiv1701.06538v1](https://arxiv.org/abs/1701.06538v1)，[版本固定PDF](https://arxiv.org/pdf/1701.06538v1)，[本地原始PDF](assets/original/1701.06538v1.pdf)。全部原图表含原caption；[视觉清单与SHA-256](sources/visual-manifest.json)，[完整公式MathML/TeX](sources/equations.json)。

**Related Papers。** [GShard](https://arxiv.org/abs/2006.16668)、[Switch](https://arxiv.org/abs/2101.03961)、[GLaM](https://arxiv.org/abs/2112.06905)、[ST-MoE](https://arxiv.org/abs/2202.08906)、[Expert Choice](https://arxiv.org/abs/2202.09368)、[MegaBlocks](https://arxiv.org/abs/2211.15841)、[Adafactor](https://arxiv.org/abs/1804.04235)、[DeepSeekMoE](https://arxiv.org/abs/2401.06066)、[DeepSeek-V3 v2](https://arxiv.org/html/2412.19437v2)、[DeepSeek-V4报告](https://arxiv.org/html/2606.19348)。

**Official Code / Updates。** [Tensor2Tensor版本固定实现](https://github.com/tensorflow/tensor2tensor/blob/v1.15.7/tensor2tensor/utils/expert_utils.py)、[MegaBlocks](https://github.com/databricks/megablocks)、[DeepEP](https://github.com/deepseek-ai/DeepEP)、[DeepSeek-V4官方发布](https://api-docs.deepseek.com/news/news260424)、[Qwen3.8-Flash-Next官方文章](https://qwen.ai/blog?id=qwen3.8-flash-next)、[官方技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)、[官方权重](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。

以上外部资料检索日期均为 **2026-10-27**，用途分别为历史关系、机制比较或当前状态，不将链接数量当作独立实验证据。正文解释、简化图与交互由本次精读创建。原作者数值均保留协议和预算边界；未运行模型训练、GPU benchmark或质量复测。

2026-10-27：初版精读；完整覆盖§1–6与A–G，保留4图9表22公式，增加路由交互、容量/通信账本、实验公平性审计与当前资料核验。
