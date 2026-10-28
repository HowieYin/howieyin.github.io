# Layer Normalization：一条样本内部，谁决定尺度？

<p class="reading-meta">CLOSE READING / 2016 → 2026 · NORMALIZATION · RNN · TRAINING · SYSTEMS</p>

**一句话：**Layer Normalization（LN）用同一条样本、同一层的隐藏维度计算均值与方差，使归一化不依赖同 batch 的其他样本，并把激活的公共偏移与整体幅度从后续计算中分离。

这篇 2016 年论文的主要实验对象是 RNN，而今天读者最熟悉的使用场景是 Transformer。把这两个时代接起来，需要同时看清三个问题：**统计量沿哪个轴计算，归一化放在网络的什么位置，以及它在硬件上花了多少代价。**它们分别属于算子定义、架构设计和系统实现，不能互相替代。

本文按原论文 Abstract、§1–7、Supplementary Material 的顺序精读，再补现代实现、技术演进与个人判断。6 张原 Figure、3 张完整 Table 和 35 个编号公式均保留；图注以可复制文字放在图外，表格另附原始截图。标为 **Paper** 的内容来自原论文；**External Source** 来自相关论文或官方实现；**Interpretation / 推导** 是本文解释。所有工程容量计算都是分析模型，未运行 GPU/NPU 性能基准。

| 元数据 | 已核验信息 |
|---|---|
| Title | Layer Normalization |
| Authors | Jimmy Lei Ba · Jamie Ryan Kiros · Geoffrey E. Hinton |
| Institution | University of Toronto；Hinton 同时署名 Google Inc. |
| Version / date | arXiv:1607.06450v1 · 2016-07-21 · 14 页，含附录 |
| Publication | 本文读取 arXiv 预印本；不补写未经核验的会议归属 |
| Primary source | [arXiv v1](https://arxiv.org/abs/1607.06450v1) · [原 PDF](https://arxiv.org/pdf/1607.06450v1) · [本地保存 PDF](sources/1607.06450v1.pdf) |
| DOI | [10.48550/arXiv.1607.06450](https://doi.org/10.48550/arXiv.1607.06450)，为 arXiv DOI |
| Official code | [ryankiros/layer-norm](https://github.com/ryankiros/layer-norm)；核对 commit `7a9a1bf7c6553b5a89949c5c860bdfc0937af483` |
| Reading / research cutoff | 2026-10-27 |

## Abstract — 建立全局图景

**一段话。**Batch Normalization（BN）对每个特征跨样本估计统计量，在小 batch、变长序列和循环网络中会遇到统计噪声与时间步处理问题。LN 改为在一条样本内部跨隐藏单元计算统计量，再用可学习的逐特征 gain/bias 恢复表达能力。训练与推理使用同样的输入统计计算，不需要 running mean/variance。论文用检索、阅读理解、句子表征、生成模型、手写轨迹和 MNIST MLP 展示更快收敛的例子，并从不变性和 Fisher 几何解释参数化的作用。它对 CNN 的初步结果却不如 BN，因此证据支持的是有条件的适用性。

五个关键点：

1. **统计边界从 batch 移到 sample。**同 batch 的另一条样本变化，不改变本样本 LN 的输出。
2. **共享统计量，独立仿射参数。**均值/方差是每条向量两个标量，gain/bias 是每个隐藏单元各自的参数。
3. **RNN 的价值是当前步就能计算。**无需为所有可能时间步预先建立推理统计。
4. **最强证据偏向训练收敛。**更少更新次数到达较好验证结果，不等于归一化 kernel 更快。
5. **今日采用需要分层讨论。**LN/RMSNorm 改统计，Pre-LN/Post-LN 改残差位置，fused kernel 改执行方式。

## 1 Introduction — Batch 的统计，为什么会成为约束？

**Paper。**深层网络中，前层参数更新会改变后层输入分布。作者沿用当时的 internal covariate shift（ICS）解释，认为归一化有助于稳定训练。BN 很有效，但 mini-batch 的大小与组成会进入一个样本的计算；数据并行可能需要额外协调；RNN 不同时间步的输入分布也不同。

**Why。**如果一个样本需要借助其他样本才能归一化，那么 batch=1、长度不同的序列和在线推理都会更难处理。RNN 还有共享权重跨时间复用的特征：同一算子面对不同历史状态，不适合简单套用固定的某一个时间步统计。

**Interpretation。**LN 的观察是：一条隐藏向量本身也有“中心”和“尺度”。无需估计整个人群的分布，先规定这条向量内部的坐标系，就能消除一部分参数与激活幅度变化。

但应区分**论文的历史动机**与**已证明的机制**。后来的 Santurkar 等人（2018）对 BN 的研究强调优化平滑性，挑战“减少 ICS 就是主要原因”的解释。[S7] 不能因此把任何深网络的稳定训练都直接归因于 LN 降低 ICS，也不能把那篇 BN 实验当作对 LN 的完整证明。

## 2 Background — 先把三个对象拆开

{{eq1}}

式 (1) 中，输入隐藏向量 `hˡ ∈ Rᵈ`，该层权重 `Wˡ ∈ Rᴴˣᵈ`；第 i 行 `wᵢˡ ∈ Rᵈ` 与输入内积得到标量 `aᵢˡ`，所有单元组成 `aˡ ∈ Rᴴ`。加偏置 `bᵢˡ` 后经非线性 `f` 产生下一层输出。**a 是归一化前的 preactivation；h 是非线性后的状态；W 是产生它们的参数。**后面的不变性不能把三者混用。

{{eq2}}

式 (2) 的 BN 对第 i 个特征计算 `μᵢˡ、σᵢˡ`，期望沿数据分布 `x∼P(x)`，实际训练使用 mini-batch 估计。gain `gᵢˡ` 调整标准化激活的幅度，式 (1) 中的 bias 再决定非线性工作点。若 batch 含 M 条样本，前馈情形统计张量为 `H`，不是每条样本一个数。CNN 常跨 batch 与空间位置统计每个 channel；这是另一种轴选择。

原式用理想总体统计，没有显示数值稳定项 ε。工程实现通常加在**方差内、开方之前**：`sqrt(var + ε)`。训练 BN 与推理 BN 的统计来源可不同；LN 下一节会去掉这一步切换。

为什么不是直接把每个 `wᵢ` 归一化？那是 WeightNorm（WN）路线：`gᵢ wᵢᵀx / ‖wᵢ‖`。它控制参数长度，却不直接控制当前输入引起的激活分布；输入 x 的尺度仍会传到输出。[S3]

## 3 Layer normalization — 改的是归约轴

{{eq3}}

式 (3) 对**同一层、同一条样本的 H 个单元**求均值与总体方差。`a∈Rᴴ → μ,σ∈R`，随后把两个标量广播给 H 个位置。这里分母是 H，现代 PyTorch 对应 `correction=0`，不是样本方差的 H−1。

把正文与式 (5) 拼起来，完整计算是：`a → μ → a−μ → v=mean((a−μ)²) → rstd=1/sqrt(v+ε) → z=(a−μ)·rstd → y=g⊙z+b → f(y)`。`g,b∈Rᴴ`，每个特征一对可学习参数。均值和方差共享，参数不共享。

### 重建 Tensor Shape、训练与推理

在现代 token 张量 `X∈Rᴮˣᵀˣᴰ` 上，常见 LayerNorm 的归约轴是**最后一个 D**：对每个 `(b,t)` 独立计算。`μ、rstd` 都是 `B×T×1`，`γ、β` 是 `D`，广播结果还是 `B×T×D`。B、T 不参与这个统计。若把 T 也纳入一个 causal decoder 的归一化，当前 token 会依赖未来 token 的激活。

{{axes-diagram}}

**训练。**前向按当前输入算统计量；任务头继续使用原来的损失，例如检索 ranking loss、分类交叉熵或生成负对数似然。LN 不引入一个新的专属任务 objective。反向传播经过均值和方差，也更新 γ、β。训练常保存每行 mean/rstd，并保留反向所需激活。

**推理。**依旧读取当前输入、求同样的统计量、做仿射。没有 running mean/variance，没有新 KV cache。RNN 的 `(hₜ,cₜ)` 是循环状态；Transformer 的 KV 是 attention 状态；它们都不能算成“LN 自己的缓存”。一个向量的 LN 代价 O(D)，所有 token O(BTD)，单层仿射参数 2D。D×D 投影约 O(BTD²)，但低 FLOPs 不保证低延迟。

### Toy example：标准化输出也有边界

取 `a=[1,2,3,4]`，`μ=2.5`，`v=1.25`。忽略 ε 时，`z≈[−1.341641,−0.447214,0.447214,1.341641]`，其均值为 0、总体方差为 1。若 `g=2,b=3`，输出均值变成 3、方差变成 4。若 g/b 随特征变化，均值甚至不能简单写为某个统一 b。

因此，**“LN 输出总是零均值、单位方差”只适用于仿射之前，且 ε=0、方差非零的标准化向量。**ε>0 时标准化方差为 `v/(v+ε)`；输入全相等时 z=0，最终输出就是 β。

{{norm-widget}}

交互固定使用 4 维 toy 向量。调整共同偏移 c 时 LN 会抵消它；换 RMSNorm 就不会。将尺度 δ 调到负数，LN 的方向翻转；让 δ 接近零且保留 ε，尺度不变性开始明显失效。最后的“单个特征偏移”按钮只改一维，说明公共偏移与任意 feature shift 是不同条件。条形长度显示数值，不表示训练质量。

### 3.1 Layer normalized recurrent neural networks

{{eq4}}

式 (4) 中 `aᵗ=Wₕhᵗ⁻¹+Wₓxᵗ`，`aᵗ,hᵗ,g,b∈Rᴴ`，`μᵗ,σᵗ` 是当前样本在 t 步的标量，g/b 跨时间共享。每步都可即时计算，所以训练到一个长度后，推理到更长长度也不需要一张更长的 BN running-stat 表。

这解决的是统计量的定义与复用问题。它**没有证明任意长序列泛化、没有消除 BPTT 的长链路依赖，也没有保证所有梯度不爆炸**。从式 (4) 的普通 RNN 到真正 LSTM/GRU，还必须确定归一化哪条分支、在哪个门前后插入；附录给出了不同方案。

## 4 Related work — 同时代有哪些替代路径？

**Paper。**本节联系 BN、Recurrent BN、WN、自然梯度以及保持信号尺度的初始化/约束方法。BN 使用样本群体统计；WN 重参数化单个权重向量；LN 使用当前样本的跨单元统计。它们控制的对象不同，不能仅凭名字中都有 “norm” 当作同一个操作。

Recurrent BN（Cooijmans 等，2016）把 BN 用到 hidden-to-hidden 转换，并研究时间步统计与 gain 初始化。[S4] 所以“BN 完全不能用于 RNN”是过度表述。LN 的优势是减少 batch 和时间统计管理，在本文多个循环任务上更容易沿用已有配置。

自然梯度用 Fisher 几何纠正参数空间中不同方向的度量；LN 则改变模型的参数化与输入尺度。作者下一节用 Fisher 分析说明二者的联系，并未声称 LN 等价于计算自然梯度矩阵逆。

## 5 Analysis — 不变性与参数几何

{{eq5}}

式 (5) 把 BN、LN、WN 都写为 `hᵢ=f(gᵢ(aᵢ−μᵢ)/σᵢ+bᵢ)`。LN 的 `μᵢ=μ,σᵢ=σ` 对一条向量共享；BN 的统计依数据分布、按特征独立；WN 可视作 `μᵢ=0,σᵢ=‖wᵢ‖₂`。形式相似，统计依赖关系不同。

### 5.1 Invariance — Table 1 中每个 Yes 到底需要什么？

{{table1}}

**读法。**行依次是整体权重缩放、权重公共平移、单个权重向量缩放、整个数据集缩放、整个数据集平移、单条样本缩放。列为 BN、WN、LN。原表中 LN 是 `Yes / Yes / No / Yes / No / Yes`；WN 的 dataset re-scaling 是 **No**。这是数学性质表，不是实测 accuracy 表，也不存在数值“提升多少”。

{{eq6}}

式 (6) 令 `W′=δW+1γᵀ`，`W,W′∈Rᴴˣᵈ`，`1∈Rᴴ`，此处 γ 是 `d` 维**权重变换向量**，不是现代 LN 的 gain γ。对固定输入 x，`a′=δa+1(γᵀx)`；第二项对 H 个单元完全相同，减均值时消失。若 δ>0，且忽略 ε，`μ′=δμ+γᵀx, σ′=δσ`，标准化结果不变。整层的 f 与 g/b 也相同，故输出相同。

如果 δ<0，真实标准差是 `|δ|σ`，z 翻转符号；δ=0 时无 ε 的定义退化。若只放大一个 `wᵢ`，它会改变该单元相对其余单元的位置及共享方差，LN 不保持不变；WN 控制每条权重的长度，因此可抵消单行正缩放。若 bias 被放在归一化前，变换时还必须一起处理它，不能默认式 (6) 原封不动成立。

{{eq7}}

式 (7) 是单条输入 `x′=δx`，δ>0 且 ε=0 时 `a′=δa`，LN 的同一条样本统计跟着变化，因此输出相同。BN 若只改变一个样本，会改变整组统计而影响其他样本，不具备这种逐样本不变性；对全数据集一致正缩放，则理想 BN 统计也同步变化。

**需要纠正的三处边界。**

- 正文说所有方法对 dataset re-scaling 不变，但 WN 的 `g·wᵀ(δx)/‖w‖` 仍乘 δ。原 Table 1 的 WN=No 与直接计算一致；正文 “all methods” 过宽。
- 结论提到 per-training-case feature shifting。严格抵消的是激活向量的公共偏移 `a+c1`。任意原输入 `x+q` 会产生 `a+Wq`，通常 `Wq` 并非常量向量；Table 1 的 dataset re-centering=No 并不矛盾。
- 现代 ε>0 时 `LN(δa)` 的分母为 `sqrt(δ²v+ε)`，无法完全消去 δ。共同平移在实数计算中仍可抵消，浮点数上大公共偏移可能先把小差异舍入掉。

### 5.2 Geometry — 同样大的参数更新，函数不一定同样变化

{{eq8}}

{{eq9}}

式 (8) 用相近预测分布的 KL 距离作局部二次近似；θ 是参数向量，δ 在此处是参数增量，不再是前面的缩放常数。式 (9) 给出 Fisher：对输入分布和模型条件输出分布取 score 外积期望，`F∈Rᴾˣᴾ`。这是 model Fisher，不是把某次真实标签的梯度外积就当作精确总体矩阵。

**直觉。**参数动 0.01，在不同尺度、不同方向上会造成不同预测变化。训练对参数使用统一学习率，不代表在函数空间迈出相同的一步。归一化的一个作用是改变这种“参数移动 → 分布变化”的局部关系。

### 5.2.1 Riemannian metric 与 GLM 的范围

{{eq10}}

{{eq11}}

式 (10) 用指数族 generalized linear model 做可分析对象：`a=wᵀx`；φ 是 dispersion；η 是 log-partition；c 不依赖 w/b。式 (11) 中 `f=η′`，条件均值是 `f(a+b)`，条件方差是 `φf′(a+b)`。x 为 d 维，w 同维，y 可先理解为一个标量输出。

{{eq12}}

扩展到 H 个输出，式 (12) 将 `Cov(y|x)∈Rᴴˣᴴ` 与增广输入二阶矩 `[[xxᵀ,x],[xᵀ,1]]∈R⁽ᵈ⁺¹⁾ˣ⁽ᵈ⁺¹⁾` 做 Kronecker product，得到 `H(d+1)×H(d+1)` 的 Fisher；φ² 在分母。它揭示输入尺度如何进入曲率。

### 5.2.2 Normalized GLM — 标准差怎样进入权重方向？

{{eq13}}

{{eq14}}

归一化后，每个单元参数按 `[wᵢ,bᵢ,gᵢ]` 排列，块大小 `(d+2)×(d+2)`，整矩阵 `H(d+2)` 方阵。式 (13) 的 weight–weight 块出现 `gᵢgⱼ/(σᵢσⱼ)·χᵢχⱼᵀ`；weight–bias、weight–gain、gain–gain 块分别描述两种参数方向间的耦合。式 (14) 的 `χᵢ∈Rᵈ` 来自链式法则：直接的 x 项，减去 mean 导数，再减去 std 导数。

**工程含义。**若整层权重共同正缩放，使 σ 同比例增长而标准化激活不变，那么权重方向对函数的敏感度可随尺度下降，weight–weight 度量含约 `1/δ²` 的因子。这里不是 `1/δ` 的 Fisher 变化，也不是 LN 下独立放大单行权重就一定令其统计同比增长。作者把这种有效学习率降低类比为 early stopping；这是一种解释，不能推成对 Adam、任意 loss、任意深网络的收敛或泛化定理。

LN 的共享统计意味着不同单元的参数还会交叉影响统计量。式 (13) 是作者给出的分析表达，完整网络反向传播必须对真实计算图求导，不能只看一个对角 Fisher 块就忽略所有跨单元依赖。

### 推导：Jacobian 给出更直接的局部图景

{{jacobian-equations}}

设 `a∈Rᴰ`，`P=I−11ᵀ/D`，`r=sqrt(v+ε)`，`z=Pa/r`。本文推导的标准化 Jacobian 是 `J=(P−zzᵀ/D)/r`；仿射输出的 Jacobian 再左乘 `diag(g)`。

`J1=0`：公共偏移方向被投掉。ε=0、非零方差时 `Jz=0`：纯径向放大也被投掉，D>1 时秩为 D−2。ε>0 时 `Jz=(ε/r³)z`，径向影响没有完全消失。D=1 时所有输入信息都被去掉；D=2 且 ε=0 时只剩两种标准化符号方向，所以 LN 对很窄的向量可能过于强烈。

令上游梯度为 u，`q=u⊙g`，反向是 `da=(q−mean(q)−z·mean(qz))/r`；`dg` 累加 `u⊙z`，`db` 累加 u，后两者沿 batch/token 累计。它说明为什么 backward 同样需要 reduction。我们对这个表达与 ε>0 的 toy 前向做了有限差分核对；这是数值正确性检查，不是训练性能复现。

## 6 Experiments — 证据更接近“更快到达”，不是“算子更快”

**Paper 的默认配置。**除特别说明，gain 初始化为 1，bias 初始化为 0。多个实验沿用 baseline 的初始 hyperparameters，以观察加 LN 是否直接有益。完整硬件、精度、训练 token 数、seed 方差和 wall-clock 对照并未全部报告；缺失字段不能用现代训练经验补齐。

### 6.1 Order-embeddings：图文检索

MSCOCO；句子用 GRU 编码，图像用预训练 VGG 与 10-crop embedding；作者的 Theano baseline 与 LN 版本使用相同架构和 Adam 初始超参。每 300 次更新验证；按 recall 选择 checkpoint；测试为 5 个 split 平均，每个 split 1,000 图像、5,000 captions。

{{fig1}}

**Figure 1 如何读。**三块面板依次是 image retrieval 的 R@1、R@5、R@10，纵轴越大越好，横轴单位为 300 次 iteration。蓝色 LN 与绿色 baseline 比较，在相同更新数上 LN 更早进入较高召回区域。它回答训练过程是否更快进入有效表征；图中没有 FLOPs 或 kernel latency 轴，也没有置信区间可用于精确检验差距显著性。

作者说 LN 约用 baseline 60% 的时间到达最好结果。该陈述属于这个实验的训练过程，不是一个现代 GPU 上“LN 让所有模型训练耗时减少 40%”的结论。

{{table2}}

**Table 2 如何读。**caption retrieval 是给图找句子；image retrieval 是给句子找图。每边 R@1、R@5、R@10 越高越好，mean rank Mr 越低越好。表中 Sym 和引用 OE 的 R@5 留空，表示未报告，不能补成 0 或估计值。

对 LN 干预最公平的比较是 **OE (ours) → OE + LN**：

| Metric | Baseline → LN | 绝对变化 | 相对变化 |
|---|---|---:|---:|
| Caption R@1 ↑ | 46.6 → 48.5 | +1.9 pp | +4.08% |
| Image R@1 ↑ | 37.8 → 38.9 | +1.1 pp | +2.91% |
| Caption mean rank ↓ | 5.2 → 5.1 | −0.1 | −1.92% |
| Image mean rank ↓ | 7.9 → 7.6 | −0.3 | −3.80% |

最强 baseline 随指标变化，例如引用 OE 的 caption R@1 46.7 略高于 ours 46.6；不能只挑最弱行计算提升。作者关于 RNN embeddings 的历史最优结果有范围限制；另一种方法使用单一 split，协议不同。gain/bias 增加参数且新增归一化计算，但表格没有对应 latency/显存量化。

### 6.2 Attentive reader：阅读理解中的 LSTM

CNN corpus，实体匿名化并截断为 4 句话；模型为单向 LSTM attentive reader。比较普通 LSTM、BN-LSTM、BN-everywhere 与只在 LSTM 使用 LN 的版本。BN 曲线来自 Recurrent BN 作者提供结果，归一化位置和调参不能视为完全相同的单变量控制实验。

{{fig2}}

**Figure 2 如何读。**横轴是 thousands of training steps，纵轴为 validation error，越低越好。蓝 LSTM、绿 BN-LSTM、红 BN-everywhere 与青色 LN 展示不同收敛轨迹；LN 更早达到较低验证错误。这个图是验证集，不是 test accuracy；末端读数约在 0.49 附近，但本文不把视觉读取当作精确表格值。

作者观察 gain=1 比 0.1 更好，并认为不必像某些 recurrent BN 设置那样使用小 gain。这支持本文配置的选择，不证明所有模型都对 gain 初始化不敏感。官方 README 曾提供 `dim=240,batch=64,lr=8e−5,Adam` 的运行命令，同时提醒仓库设置已经变化；这些属于**代码证据**，不能无条件标为原论文完全固定协议。

### 6.3 Skip-thought vectors：收敛与最终句子表征

BookCorpus，encoder/decoder GRU，encoder dimension 2400，Theano；同超参比较。每 50k updates 保存模型，主要预算 1M。作者用 CNMeM 后未观察到显著 per-step 变慢，这是当时实现的报告，本文未复测，也不是所有硬件都免费。

{{fig3}}

**Figure 3 如何读。**六面板为 SICK Pearson×100、SICK MSE×100、MR、CR、SUBJ、MPQA。除了 MSE 越低越好，其余越高越好。绿色为作者 baseline，蓝色为 LN；灰色参考原 skip-thought 结果。蓝曲线通常更早达到较好区域。

阴影是下游分类器 10-fold cross-validation 的波动，不是 10 个独立预训练 seed 的置信区间。图中 SICK MSE 乘 100，Table 3 没乘 100；跨图表核对必须还原尺度。

{{table3}}

**Table 3 的公平比较。**SICK 的 r 为 Pearson，ρ 为 Spearman，MSE 为均方误差；后四列为分类 accuracy（%）。`Ours + LN` 与 `Ours` 都是 1M iterations；带 † 的 LN 训练一个月约 1.7M iterations，预算不同。

同预算的 SICK MSE `0.298→0.277`，绝对 −0.021，相对下降 7.05%；MR `77.3→79.5`，+2.2 pp，相对 +2.85%；CR、SUBJ 分别 +0.8 pp，MPQA +1.1 pp。† 进一步提升多数指标，但 MR 79.4 低于 1M LN 的 79.5，原表粗体也保留这一点。不能把 † 的改进全归为同预算 LN 优势。

原 skip-thought 论文 decoder 在每个时间步接收 encoder 条件，公开实现却没有；作者指出这是 Original 与 Ours 差异的可能原因。因此 Original→LN 混有实现变化，**Ours→同预算 LN** 才更接近所要测试的干预。

### 6.4 DRAW：生成 bound 的改善幅度不大

固定二值 MNIST（Larochelle/Murray split），50k train、10k validation、10k test；64 glimpses、256 维 LSTM，Adam 默认配置、batch 128。DRAW 的 LN 仅放在 cell 输出进入 tanh 的位置，附录有明确公式，并不是把所有门都归一化。

{{fig4}}

**Figure 4 如何读。**横轴 epochs 0–100，纵轴 test variational bound，越低越好；蓝 baseline、绿 WN、红 LN。红线较快下降，作者说达到相近 bound 的收敛速度几乎翻倍。它没有比较两个归一化 kernel 的吞吐，也没有证明最终生成分布质量提升同样大。

正文另报告训练 200 epochs 后 baseline 82.36 nats、LN 82.09 nats，改善 0.27 nats，相对 0.328%。这是 variational upper bound，不能称为精确 NLL；末端数值来自正文，不是图的 100-epoch 端点。原图在 PDF 中因浮动排版晚于 Figure 5 出现，本文按 §6.4 的概念顺序讨论。

### 6.5 Handwriting sequence generation：长轨迹、小 batch

IAM-OnDB：221 writers、12,179 lines，平均轨迹约 700 steps，输入文字超过 25 characters。模型有 3 层各 400 单元 LSTM，约 3.7M parameters；输入轨迹 3 维；输出 20 个二维 Gaussian mixture components。字符/window 输入为 57 维，attention window 为 10 Gaussians×3 parameters=30 维；Adam，batch=8，训练切片长度 500。

{{fig5}}

**Figure 5 如何读。**横轴是每 200 次更新计一个单位，且为对数尺度；纵轴是 negative log-likelihood，越低越好。baseline/LN 的 train/test 四条曲线分别展示优化与泛化趋势。LN 更早达到 baseline 接近最终的区域，最明确的证据仍是更新效率。

连续密度可以大于 1，所以 NLL 可以为负，这不违反离散概率小于等于 1。图中未给新生成样例，也没有表列最终精确数值；本文不制造一个额外 final score。该任务说明在小 batch、较长循环链路中 LN 有用，不等于 700 步长度外推已经被严格测过。

### 6.6 Permutation-invariant MNIST：这里是 MLP，不是 pixel RNN

784–1000–1000–10 的前馈网络，55k train，Adam；分别 batch=128 和 4。LN 只用于 hidden layers，不作用于最后 softmax；BN baseline 用于所有层，且小 batch 时作者使用 unbiased variance。这两个设置差异要计入公平性。

{{fig6}}

**Figure 6 如何读。**左列 batch 128，右列 batch 4；上排 train NLL、对数纵轴，下排 test error（%）。右上最能展示小 batch 下归一化与优化的差异；下排说明训练曲线更快下降不必带来同等幅度 test error 改善，各方案大致在 1% 量级附近。

这里 permutation-invariant 指固定像素输入交给全连接网络，不是把 784 pixels 当作时间步喂给 sequential RNN。LN 的统计不依赖 batch size，但 SGD 的噪声、学习率和最终泛化仍依赖 batch；“batch-robust normalization”不能推成所有 batch 下训练结果一致。

### 6.7 Convolutional networks：作者主动报告了失败边界

这是定性 preliminary experiments，没有附 CNN 数值表或新 Figure。作者说 BN 比 LN 和 baseline 更好，假设空间边界以及大量非活跃单元使同层单元的分布不一致，从而影响 LN。它给出值得探索的原因，缺少控制空间位置、活跃比例与归约轴的消融，不能当作已验证因果。

**Interpretation。**LN 默认把一个统计域内的单元作为同一参照群体。CNN 中 channel 和 spatial 的分布异质性可能很强；“不依赖 batch”解决了一种约束，却未自动选择最适合视觉结构的统计域。后来的 GroupNorm 在每个样本的 channel groups 内归一化，是另一种取舍。[S6]

### 实验审计总账

| Setting | 已报告的关键配置 | 主要 evidence | 尚缺的验证 |
|---|---|---|---|
| Order-embeddings | MSCOCO；GRU/VGG；Adam；5 test splits | retrieval recall / rank 与更新曲线 | 完整硬件/precision，成本、seed variance |
| Attentive reader | CNN corpus 4 sentences；uni-LSTM | validation error 曲线 | 同位置/同预算归一化对照、精确 test 值 |
| Skip-thought | BookCorpus；2400 encoder；1M updates | 7 metrics；† 1.7M | 多预训练 seed；同预算长训 baseline |
| DRAW | 64 glimpses；256 LSTM；B128；200 epochs | variational bound | exact NLL、生成质量、放置位置消融 |
| Handwriting | 3×400 LSTM；~3.7M；B8；length500 | train/test NLL 曲线 | 长度外推、per-step latency、样例 |
| MNIST MLP | 784–1000–1000–10；B128/B4 | train NLL / test error | 对齐输出层 norm 与方差 estimator |
| CNN | preliminary qualitative statement | BN 更好 | 完整数值、结构和统计域消融 |

论文没有系统拆分 mean-only、variance-only、gain-only、bias-only、LN placement 与 optimizer 的 factorial ablation；也没有给现代 LLM 的 TTFT、TPOT、p95 或端侧峰值内存。多个任务的积极结果是广度证据，但不是六个独立受控实验共同证明所有架构最优。

## 7 Conclusion — 贡献成立，措辞需要收紧

**Paper。**LN 是简单的逐样本归一化，易用于循环网络，在不同任务上加快训练，训练/推理计算一致。作者强调它对权重和输入尺度的若干不变性，以及在神经网络参数几何上的作用。

本文认为最稳固的贡献是**提出并验证了一种不依赖 mini-batch 的激活归一化边界**。它的成功不要求接受每一句 ICS 或几何解释。原文有关 arbitrary feature shifting、所有方法 dataset scaling、附录初始化的宽泛/矛盾表述，应依据公式和官方代码具体澄清；前面已经就地标明，下一节继续核对附录。

## Supplementary Material — RNN 分支的具体位置

### LN 的正式定义与初始化笔误

{{eq15}}

{{eq16}}

式 (15) 的输入 z 是 D 维向量，α、β 同为 D 维；式 (16) 的 μ、σ 是标量。**公式中 α 乘标准化输入，所以是 gain；β 在最后加，所以是 bias。**附录紧邻文字却把二者称呼写反，并说 α 置零、β 置一。

正文默认 gain=1/bias=0，官方 Theano 的 `s=ones,b=zeros` 与之相符，函数还加入 `1e−5`。复现应采用公式语义和代码初始化。若真的照那句文字把 gain=0，初始输出将只剩 β，输入路径的局部梯度会被 gain 阻断，明显是另一种初始化行为。[S1]

### LSTM：输入与循环投影先各自 LN，再相加

{{eq17}}

{{eq18}}

{{eq19}}

标准 LSTM 的式 (17) 先形成 f/i/o/g 四个门的 preactivations。输入 `xₜ∈Rᵈ`，状态 `hₜ₋₁,cₜ₋₁∈Rᴴ`；`Wₕ∈R⁴ᴴˣᴴ,Wₓ∈R⁴ᴴˣᵈ,b∈R⁴ᴴ`。式 (18) 的 sigmoid forget/input gate 混合旧 cell 和候选；式 (19) 的 output gate 读取 tanh(cell)。此处 σ(·) 是 sigmoid，不能与标准差 σ 混读。

{{eq20}}

{{eq21}}

{{eq22}}

式 (20) 把 hidden projection 与 input projection **分别归一化后相加**；式 (21) cell update 形式保持不变；式 (22) 在 cell 进入 tanh 前再用第三个 LN。前两个 LN 的统计域为 concatenated **4H**，不是四个门各自 H。这在原作者实现中可核验，切分发生在 LN 之后。

{{lstm-diagram}}

因此，三个 gain/bias 参数组分别长 `4H、4H、H`，新增仿射参数 `2(4H+4H+H)=18H`，不含已有 b。独立 gate LN 改变了门之间的公共统计，`LN(A)+LN(B)` 与 `LN(A+B)` 也通常不同；做移植时应优先核对这两个区别。

### GRU：重置门作用于已归一化循环候选分支

{{eq23}}

{{eq24}}

{{eq25}}

式 (23) 产生 update zₜ 与 reset rₜ 两个门，拼接为 2H；式 (24) 的 candidate 先算输入 W xₜ 与循环 U hₜ₋₁，并用 reset gate 逐元素控制循环分支；式 (25) 用 update gate 混合旧状态与 candidate。注意不同框架的 GRU reset 放置及 z 定义可能不同，需按这里的公式对应。

{{eq26}}

{{eq27}}

{{eq28}}

式 (26) 两条 gate projection 各有一个 2H LN；式 (27) 输入 candidate 与循环 candidate 各有一个 H LN，reset gate 在循环 LN 之后相乘；式 (28) 状态混合保持原式。统计域大小 `2H、2H、H、H`，新增参数 `2(2H+2H+H+H)=12H`。candidate 中 reset 位置不变，但它读到的是归一化后的循环向量。

### DRAW：不是前面的完整 LN-LSTM

{{eq29}}

{{eq30}}

{{eq31}}

式 (29) 的 4H gate projection 保持标准 LSTM；式 (30) cell update 保持原式；式 (31) 只归一化 H 维 cell readout 再 tanh。新增参数 2H。把 Figure 4 的结果归为“对全部 LSTM 门加 LN 的收益”会错误解释论文的实际干预。

### Gain 方向的 Fisher 几何

{{eq32}}

式 (32) 讨论 BN 只沿 gain 更新的局部 KL；δg 是 H 维，`vecop` 将参数块展开。作者把度量写成只含条件输出 covariance 的形式。更一般地，gain–gain 块应包含 `E[Cov(yᵢ,yⱼ|x) zᵢzⱼ]/φ²`。标准化变量有单位边际方差，不自动意味着 covariance 与 z² 独立，也不自动消除所有交叉项；此简化需要额外条件，不能无条件使用。

{{eq33}}

式 (33) 为 LN 保留共享 σ 下 `(aᵢ−μ)(aⱼ−μ)/σ²` 的 H×H 矩阵，展示 gain 方向仍依赖样本内相对激活及条件 covariance。它解释与 BN 统计边界的差异，不意味着 LN 总能给出更好的曲率条件数。

{{eq34}}

式 (34) 对 WN 使用 `aᵢaⱼ/(‖wᵢ‖‖wⱼ‖)`。该尺度来自权重，不消去输入的幅度。每条 weight 的长度与方向重参数化，可以使 gain 更新对应普通模型中特定的径向更新。

{{eq35}}

式 (35) 就是在普通 Fisher 中代入 `δwᵢ=δgᵢ wᵢ/‖wᵢ‖` 与 j 的对应方向；右侧为 covariance 加权的交叉项。原式左侧第一向量中 j 项出现 `‖wᵢ‖`，与右侧、另一个向量中的 `‖wⱼ‖` 不一致；对称的归一化方向应使用 j 的范数。这里保留原式，并明确指出这一处疑似下标笔误，而非静默修改源公式。

## 现代实现 — 从 O(D) 到 TTFT、TPOT 和数值精度

### 可复核的实现，而不是只调用库

下面是现代语义的 reference code；`normalized_shape=D`，统计 fp32、方差 correction=0、ε 加在 variance 中，再返回原 dtype。它用于解释语义，真实高性能 kernel 会融合并行执行，不能据此推断速度。

```python
def layer_norm_reference(x, gamma, beta, eps=1e-5):
    # x: [B, T, D]; affine: [D]
    a = x.float()
    mean = a.mean(dim=-1, keepdim=True)
    var = ((a - mean) ** 2).mean(dim=-1, keepdim=True)
    z = (a - mean) * (var + eps).rsqrt()
    return (z * gamma.float() + beta.float()).to(x.dtype)
```

PyTorch LayerNorm 支持最后 `normalized_shape` 的多个维度，默认 γ=1、β=0，训练/推理都使用当前输入统计。[S12] Triton 官方 tutorial 每行读取、fp32 accumulation、计算 mean/variance、做仿射并写回；训练保存 mean/rstd，backward 对输入与 affine 梯度做归约。[S13] tutorial 有 feature row 64KB 的特定限制，它不是 LN 数学定义的容量上限。

**数值。**避免低精度累加以及直接用 `E[x²]−E[x]²` 处理大公共偏移造成相消。稳定实现可用 centered two-pass 或 Welford；输入在 fp16/bf16 中已经丢掉的相对差异，后转 fp32 也无法恢复。低方差、很小 ε、大 gain 会放大精度误差；不同 ε、归约顺序、affine dtype 都可能造成导出后 logits 漂移。

**量化与端侧。**LN 的 μ/σ 依赖每次输入，不能一般性折叠成固定 W/b；冻结统计的 BN 才有常见 folding 条件。RMSNorm 同样输入依赖。部署时检查轴、ε、gain/bias、cast、低方差与饱和；使用真实多样标定数据，并按 per-op 与整链路定位精度损耗。本文没有实测 HTP/NPU 映射，不能宣称某个整数 kernel 与浮点 LN 完全等价。

### IO 账本：为什么算术少，仍能占 Decode 时间？

{{io-diagram}}

令 `N=BTD`，元素占 s bytes，忽略低阶统计与 affine cache。理想 row-fused kernel 主激活 read+write 约 **2Ns**。一个 mean→center→square→variance→affine 的五 kernel 物化链，分别是 `N+2N+2N+N+2N=8N` 元素流量。这里 affine 重用中心化向量，参数读另计；它是明确指定实现的分析模型，不代表所有 unfused 框架都会正好读 8N。

{{cost-widget}}

默认 `B=1,T=2048,D=4096,FP16`：一处 LN 理想主激活 32 MiB；五 kernel 模型 128 MiB；保存 fp32 mean+rstd 16 KiB；该处 γ/β 若 fp32 存储占 32 KiB。单 token 一处 LN read+write 为 16 KiB；32 个 block 每个 2 LN 加 final，共 65 处，则 1.015625 MiB/token。这是理想主激活流量，未计 cold affine 参数、workspace、缓存行为与真实 DRAM 事务。

**Prefill / TTFT。**行数 BT 大，LN 可有大量并行；投影/attention 常占主导，但物化中间量与数据转换增加 IO。**Decode / TPOT。**每步只有 B 行，单个 LN 的工作很小，launch、reduction 同步、残差/量化边界可能更突出；几十次小算子重复会影响总链路。归一化本身不读取历史 KV，不能把随上下文增长的 attention 成本也归给 LN。

若 D 维被 tensor parallel 分片，标准 LN 的统计要覆盖完整 D，可能需要跨 shard 聚合局部和/平方量；若每个 rank 已持有完整 hidden 或按 token 做 sequence parallel，则通信条件不同。不能笼统说 LN 总是零通信，亦不能把 BN 的跨样本同步等同于所有并行方式的 LN。

准确的优化实验应逐项报告 per-op latency、launch 数、IO、dtype 与误差，再看完整 TTFT/TPOT；理想 `bytes/bandwidth` 不是测得 latency。本文所有容量数可以交互复核，未给出虚构的速度曲线。

## 技术演进 — 改统计、改位置、改执行，三条并行线

{{timeline-diagram}}

| Paper / route | Year | Core idea | Advantage | Limitation | 与 LN 的关系 |
|---|---:|---|---|---|---|
| [Batch Normalization](https://arxiv.org/abs/1502.03167) | 2015 | 按特征跨样本估计统计 | CNN 常用；冻结后可 folding | batch 噪声、train/eval 差异 | LN 的直接背景 [S2] |
| [Weight Normalization](https://arxiv.org/abs/1602.07868) | 2016 | 权重方向/长度重参数化 | 无跨样本统计 | 不直接规范当前激活尺度 | 不变性对照 [S3] |
| [Recurrent BN](https://arxiv.org/abs/1603.09025) | 2016 | 循环转换加 BN | 在 RNN 中验证 BN 可行 | 时间统计、gain 与 batch 设置 | 同时代替代 [S4] |
| Layer Normalization | 2016 | 单样本跨隐藏单元统计 | B1、变长 RNN 方便 | reduction、轴选择与信息丢失 | 本文 |
| [Transformer](https://arxiv.org/abs/1706.03762) | 2017 | attention/FFN 子层后的残差 LN | 移入序列并行架构 | 原 Post-LN 深层优化需关注 | 拓展采用场景 [S5] |
| [Group Normalization](https://arxiv.org/abs/1803.08494) | 2018 | 单样本 channel groups+空间统计 | 小 batch 视觉任务 | group 数与统计域仍需选择 | 另一种样本内边界 [S6] |
| [RMSNorm](https://arxiv.org/abs/1910.07467) | 2019 | 仅均方根，无减均值 | 简化统计与参数 | 不具公共偏移不变性 | 修改算子定义 [S8] |
| [On LN in Transformer](https://arxiv.org/abs/2002.04745) | 2020 | 比较 Pre/Post-LN 初始化梯度 | 解释 warmup/位置影响 | mean-field 与实验范围限制 | 修改残差位置 [S9] |
| [DeepNet / DeepNorm](https://arxiv.org/abs/2203.00555) | 2022 | 残差缩放+初始化 | 支持很深 Transformer | 配方与结构关联 | 联合调整架构/优化 [S10] |
| [nGPT](https://arxiv.org/abs/2410.01131) | 2024–25 | 多种向量在单位球面约束 | 作者报告更少训练 steps | 模型/序列条件不同 | 更广的几何路线 [S11] |
| [Triton LayerNorm](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) | ongoing | row fusion、fp32 reduction | 减少物化与 launch | 资源/shape 限制 | 执行方式 [S13] |

GroupNorm 仅 1 group 时，在特定 axes 下统计可与 LayerNorm 相同，但其 affine 常按 channel，而 LayerNorm 可逐 normalized element；不能无条件写成两者模块完全等价。

**RMSNorm。**`y=g⊙x/sqrt(mean(x²)+ε)`；保留正缩放的近似不变性，不减均值。原论文报告 7–64% runtime savings，是其当时实验范围，不能写成现代 LLM 普遍提速。pRMSNorm 用子集估计均方根，可能产生额外估计误差。[S8] 训练好的 LN checkpoint 换 RMSNorm 会改变函数，不能因为形状相同就假设无需训练。

**位置。**Post-LN 为 `LN(x+F(x))`；Pre-LN 为 `x+F(LN(x))`。前者把残差流也归一化，后者保留一条显式恒等路径；final norm 等细节还需单独检查。Xiong 等分析初始化时 Post-LN 输出附近梯度较大，并在实验中展示 Pre-LN 可减少 warmup 依赖。[S9] 这不是“所有大模型不需要 warmup”。DeepNorm 和 nGPT 的模型效果/训练步骤改进也不是 LN kernel 的速度数据。

## 当前状态 — 截至 2026-10-27 的证据边界

这里不宣布一个脱离任务和训练配置的“归一化 SOTA”。目前核验的公开来源包含模型采用与任务结果，但没有同一模型、数据、token 预算、优化器、硬件下，覆盖所有主流归一化方案的最新统一排名。

**官方实现样本。**Qwen3 的 Transformers 代码在固定 commit 中使用 fp32 均方根计算，包括 input/post-attention/final RMSNorm 以及按 head dimension 的 Q/K normalization。[S14] token hidden norm 归约 D，Q/K norm 归约 head dimension；名称相似不意味着统计轴相同。

**最新检索样本。**Qwen Team 2026-08-26 的 Qwen3.8-Flash-Next 官方介绍明确列出 **Zero-Centered RMSNorm**、norm weight decay、attention output gate 和 normalized MoE router initialization 作为稳定性方案。[S15] “Zero-Centered” 说的是 gain 参数化概念，不能据词面理解成减去 activation mean；未据此补写未经核对的具体 `(1+γ)` 代码。

| 官方当前样本 | 模型规模 / 开放状态 | Benchmark / metric | 作者报告 | 可比性 |
|---|---|---|---:|---|
| Qwen3.8-Flash-Next | 主模型 125B，active 6B，另有 51B Ngram；open weights | GPQA Diamond，accuracy % | 91.7 | 模型整体结果；不是 LN/RMSNorm swap 对照 |
| 同一模型 | 同上 | LiveCodeBench v6，官方表 score % | 91.9 | 按官方各任务协议；不能从分数归因 norm |

官方介绍还涉及长上下文、稀疏注意力及不同 128K/256K 协议，进一步限制跨模型直接比较。上述分数仅展示当前方案的采用背景，**不证明 RMSNorm 比 LN 多了多少 accuracy，也不能把稀疏 attention 的加速归为 norm 贡献**。要比较归一化，需要独立控制 mean removal、gain 参数化、weight decay、位置和融合实现。

## Critique — 简单算子，隐含选择并不少

**Novelty / strengths。**统计域从跨样本转到样本内部，是一个清晰可复用的设计；不需要推理 running-stat 表；循环任务上的广泛收敛证据具有实际价值。新增参数规模线性，操作定义简单，官方代码让 RNN 放置位置可以核查。

**Correctness / weaknesses。**正缩放和 ε 边界被原文省略；Table 1 与正文、附录符号/初始化存在矛盾；部分 Fisher 简化缺少明确依赖条件。实验对训练效率的支持强于统一泛化收益，缺少多 seed、成本和系统消融；CNN 结果提示统计域异质性。

**Hidden trade-offs。**LN 去掉公共幅度/偏移，可能也丢掉任务有用的信息；g/b 可重建特征标定，但不能恢复被投掉的输入自由度。推理没有 running-state，不意味着没有输入依赖计算；低 FLOPs 不意味着没有显存流量或小 batch launch 代价。减少 batch 耦合不意味着消除所有并行通信。

**Scalability / reproducibility。**数学可扩展到大 D，实际 kernel 受寄存器、shared memory、dtype、layout、并行 shard 约束。作者代码主要是 Python 2.7/Theano/Torch，核验函数与固定 commit 并不等于已复现所有训练；旧模型下载是否可用也未完整验证。

我希望补的实验是：同一 RNN 的 gate-concat 与 per-gate、两分支 LN 与相加后 LN、cell-only 与 full LN-LSTM；相同预算的 mean-only/RMS/LN 配合 gain/bias；小 batch MLP 中对齐输出层与 variance estimator；CNN 按空间/通道统计域控制；现代 LLM 中同时检查 loss、gradient、logits drift 与 TTFT/TPOT。

## References — 原论文、外部证据与复核记录

1. **[S1 / Primary]** Ba, Kiros, Hinton. *Layer Normalization*, arXiv:1607.06450v1 (2016)。[全文](https://arxiv.org/abs/1607.06450v1)；[官方代码固定版本](https://github.com/ryankiros/layer-norm/tree/7a9a1bf7c6553b5a89949c5c860bdfc0937af483)。
2. **[S2]** Ioffe, Szegedy. *Batch Normalization*, 2015。[arXiv](https://arxiv.org/abs/1502.03167)。
3. **[S3]** Salimans, Kingma. *Weight Normalization*, 2016。[arXiv](https://arxiv.org/abs/1602.07868)。
4. **[S4]** Cooijmans et al. *Recurrent Batch Normalization*, 2016。[arXiv](https://arxiv.org/abs/1603.09025)；后续 v5 修订为 2017，本文年代指初始路线。
5. **[S5]** Vaswani et al. *Attention Is All You Need*, 2017。[原文](https://arxiv.org/abs/1706.03762)；[本站 Transformer 精读](../transformer/)。
6. **[S6]** Wu, He. *Group Normalization*, ECCV 2018。[arXiv](https://arxiv.org/abs/1803.08494)。
7. **[S7]** Santurkar, Tsipras, Ilyas, Madry. *How Does Batch Normalization Help Optimization?*, NeurIPS 2018。[arXiv](https://arxiv.org/abs/1805.11604)。
8. **[S8]** Biao Zhang, Rico Sennrich. *Root Mean Square Layer Normalization*, NeurIPS 2019。[arXiv](https://arxiv.org/abs/1910.07467)；[官方代码](https://github.com/bzhangGo/rmsnorm)。
9. **[S9]** Xiong et al. *On Layer Normalization in the Transformer Architecture*, ICML 2020。[arXiv](https://arxiv.org/abs/2002.04745)。
10. **[S10]** Hongyu Wang et al. *DeepNet: Scaling Transformers to 1,000 Layers*, 2022。[arXiv](https://arxiv.org/abs/2203.00555)。
11. **[S11]** Loshchilov et al. *nGPT: Normalized Transformer with Representation Learning on the Hypersphere*, 2024，v2 2025。[arXiv](https://arxiv.org/abs/2410.01131v2)。
12. **[S12]** PyTorch LayerNorm 官方文档与 v2.11.0 normalization 实现。[API](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)。
13. **[S13]** Triton 官方 Layer Normalization tutorial。[说明](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)；[固定 commit 源码](https://github.com/triton-lang/triton/blob/3785150e4d097f32c11af3aa955b78f80e2c16a8/python/tutorials/05-layer-norm.py)。
14. **[S14]** Hugging Face Transformers，Qwen3 modeling 官方实现。[固定 commit](https://github.com/huggingface/transformers/blob/a672eb5f474b8b30173e9f96eab66adf292da635/src/transformers/models/qwen3/modeling_qwen3.py)。
15. **[S15]** Qwen Team. *Qwen3.8-Flash-Next*, 2026-08-26。[官方介绍](https://qwen.ai/blog?id=qwen3.8-flash-next)；于 2026-10-27 用正文抓取与 Chrome DOM 双核对，[归一化证据存档](sources/qwen38-normalization-evidence.json)。

原图与表截图来源为 supplied `1607.06450v1.pdf`，288 dpi；图像保留坐标轴、图例和面板标签，原 caption 与中文图注以文字放在图外。[visual manifest](sources/visual-manifest.json) 记录页码、图像与图注各自的 bbox，以及完整图注文本。[全部公式 MathML/TeX](sources/equations.json) 按编号保存；[原始 TeX](sources/paper.tex) 与 [bibliography](sources/paper.bbl) 用于二次核对。现代工程图均为 Agent-created Simplified Diagram，Mermaid source 可下载，不替代原 Figure。

## My insights / future directions — 把三个边界分别测清楚

**我的判断。**LN 最值得记住的是统计边界，而不是一个“让训练稳定”的口号。先问一条输出向量的参照群体是谁，再问这些单元是否应共享均值与尺度；这个问题能解释它为何适合 token/RNN，也能解释 CNN 的失败与 GroupNorm 的出现。

对推理工程，我会把研究拆成三个互相独立的对照：**统计是否减均值，残差分支在哪归一化，执行图是否融合。**第一类影响函数与训练；第二类影响信息/梯度通路；第三类影响 IO 和 launch。把它们一起改完再报告一个 TPOT，会很难判断收益或精度损失来自哪里。

一个可执行的下一步是给同一 checkpoint 做逐层对照：固定 prompt 与生成路径，记录 LN 输入均值、方差、低方差比例、gain 分布、reference/kernel 最大误差和 logits 漂移；再测 Prefill/Decode 的 per-op 耗时。替换 RMSNorm 的实验则必须另列训练适配预算，不能混进“等价 kernel 优化”。

对训练研究，最有价值的控制是把 ε、gain 参数化、norm weight decay、residual scaling、Q/K norm 各自隔离，并同时报告优化轨迹和泛化指标。当前模型把这些工具联合使用说明它们仍重要，但联合成功不等于每一项的独立因果都已经被证明。
