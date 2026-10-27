# RMSNorm：保留尺度约束，拿掉均值中心化

**Root Mean Square Layer Normalization**

Biao Zhang、Rico Sennrich · University of Edinburgh / University of Zurich · NeurIPS 2019\
精读版本：1910.07467v1，14 页（正文、参考文献与完整附录）。arXiv 提交：2019-10-16。阅读与更新：2026-10-27。

[Paper PDF](assets/original/1910.07467v1.pdf) · [arXiv](https://arxiv.org/abs/1910.07467) · [DOI](https://doi.org/10.48550/arXiv.1910.07467) · [Official Code](https://github.com/bzhangGo/rmsnorm) · [Original TeX](sources/paper.tex)


## TL;DR：归一化需要约束什么？

<div class="callout"><strong>一句话：</strong>RMSNorm 用每个 token 的均方根约束激活尺度，省去 LayerNorm 的均值中心化，在论文测试的多种模型中保持相近质量，并降低训练耗时。</div>

一段话理解：LayerNorm 同时把向量拉到零均值、单位标准差。作者提出，真正关键的可能是尺度约束；均值归零在许多任务里可以省略。RMSNorm 直接除以二阶矩的平方根，保留正比例缩放不变性，但失去平移不变性。pRMSNorm 再只用固定前缀估计分母。论文用梯度分析解释这一取舍，用翻译、实体填空、图文检索和图像分类检查质量与速度。今天它已进入 Llama 等模型的实现，工程瓶颈也从“少一个统计量”走向归约、融合、带宽与 launch 开销。

五个关键点：

1. **RMS 不是标准差。** RMS² = variance + mean²；均值不为零时，两种归一化会产生不同方向与尺度的输出。
2. **不变性有条件。** 全向量正比例缩放、无 epsilon、同样的 gain/bias 才得到严格等式；负比例缩放会翻转符号，单个神经元缩放也不保持输出。
3. **梯度不是逐维独立的。** RMS 的分母让同一 token 的所有通道通过一个点积耦合；高效 backward 不需要生成 D×D Jacobian。
4. **6.25% 指参与分母估计的元素。** 所有元素仍参与输出、gain 与后续层计算。pRMSNorm 没有把整个算子或模型缩小到 1/16。
5. **7%–64% 是历史训练耗时降幅范围。** 最高 63.9% 属于 Theano 图文检索的 pRMSNorm；不能换算成现代 LLM 的 TTFT／TPOT 保证。

适合已有矩阵运算与反向传播基础、想理解归一化机制及 per-op 性能的读者。主线严格按原文 §1–§7、附录 A.1–A.4 展开，之后再做独立审计与扩展研究。**全部 7 张 Figure 和 10 个 Table 编号均保留；Table 9 原本就是曲线图。** 原论文事实、外部来源和本文解释在正文明确区分。


## 1. Introduction：每步更有效，也可能每步更慢

**原文 §1，PDF pp.1–2。** 深层模型的输入分布会随着前层参数更新而变化。LayerNorm 通过局部统计改善训练，尤其适合 batch 统计不稳定的 RNN。但论文的起点是一个系统问题：收敛步数减少，不必然等于墙钟时间减少。

{{fig:1}}

**Figure 1 — 步数与时间是两种横轴。** 两个面板展示 GRU RNNSearch 的前 10k 训练步骤；纵轴都是 training loss，越低越好。(a) 以 step 为横轴，Baseline 到 7.0 时 LayerNorm 到约 5.4；(b) 以 seconds 为横轴，同样时间里 LayerNorm 只到约 5.9。曲线的组成是同一模型的无归一化与 LayerNorm 两种训练过程，没有额外网络数据流箭头。先对齐纵轴，再对齐横轴单位，才能读出“优化效果被运行开销抵消了一部分”。

**关键观察：** 这张图证明要检查 time-to-quality，而非只比较 loss/step。它并没有直接测 RMSNorm，也没有证明所有 LayerNorm kernel 都很慢；后续实验才比较新方法。RNN 的时间依赖会使许多小归一化沿序列串行发生，归约与调度开销反复进入关键路径。

作者的假设是：LayerNorm 的 re-scaling invariance 更关键，而 re-centering invariance 的贡献可以较小。贡献有三层：提出 RMSNorm；证明输出与梯度的尺度性质；探索 pRMSNorm，并跨任务／框架验证质量与训练时间。应把“中心化可以省略”理解为一个得到实验支持的设计假设，不能提升成所有模型的必要充分条件定理。


## 2. Related Work：统计对象比“有没有 Norm”更重要

**原文 §2，PDF p.2。** BatchNorm 用 batch 的统计值，所以 mini-batch 大小、样本相关性与训练／推理统计切换都会影响行为。LayerNorm 对单样本的某一层特征归一化，避免依赖 batch；这让它适合循环网络与序列模型。WeightNorm 则重参数化权重，把向量方向和长度分开，不直接计算每个 token 的激活统计。

原文还讨论了 GroupNorm、对归一化解释的争论，以及不同范数的探索。这里最容易混淆的是三种“尺度”：权重向量的范数、单个 token 的特征范数、一个 batch 中同一特征的方差。它们的轴不同，不变性也不同，不能只凭名称里的 Norm 相互替换。

**本文解释：** RMSNorm 的改动很小，却有明确的机制选择：依然归一化激活，依然每个样本独立，依然训练 gain；删除的是均值中心化。它不依赖新增 loss、辅助网络或数据标注。2019 年的价值在于把一个常见组合中的两个功能拆开验证；今天看起来“只少一次 mean”是因为这个选择后来变得常见。


## 3. Background：先重构 LayerNorm 的输入和输出

**原文 §3，PDF p.3。**

{{eq:1}}

W ∈ Rⁿˣᵐ，x ∈ Rᵐ，a = Wx ∈ Rⁿ，b ∈ Rⁿ，f 为逐元素激活。第 i 行权重与输入做点积生成 aᵢ，再加 bᵢ、进入 f。**论文的归一化对象是线性求和结果 a**，不是权重 W，也不是先加 bias 后的任意张量。

{{eq:2}}

{{eq:3}}

g ∈ Rⁿ 是可学习 gain，初始值为 1；b 初始值通常为 0。μ、σ 都是单个向量上的标量；除法和 gain 相乘广播到 n 个元素。原公式没有 epsilon，σ² 的除数是 n，属于总体方差。

计算顺序是：a → 均值 μ → a−μ → 平均平方差 → 开方 σ → 除法 → gain → bias → f。理想中心化核心的均值为 0、方差为 1；**经过不均匀 gain 和 bias 后，最终输出一般没有这两个性质**。

现代 Transformer 通常把输入直接写作 X ∈ Rᴮˣᵀˣᴰ，将 B、T 展平为 M=B×T 行，每行在 D 上归一化。此时论文 n 对应 D，m 是前一线性层输入维；不要和 sequence length 混用。RNN 中相同操作位于每个 timestep 的门控映射内，具体 placement 比符号本身更重要。

<div class="callout interpretation"><strong>Toy Example：</strong>a=[1,3] 时 μ=2、σ=1，LayerNorm 的无 affine 核心为 [−1,1]。RMS=√5，RMSNorm 为 [0.4472,1.3416]。两者均控制整体尺度，但保留的均值、符号与方向不同。</div>


## 4. RMSNorm：把半径固定，允许均值保留

**原文 §4，PDF pp.3–5。**

{{eq:4}}

先逐元素平方 aᵢ，再对 n 求平均、开方得到一个 r=RMS(a)，用同一个 r 除每个元素，最后乘 gain。输入／输出均为 [n]，统计量为 scalar，g 为 [n]；batched 版本统计量为 [M,1]，gain 为 [D]。

由于 sum(aᵢ²)=n r²，在 r>0、无 epsilon 时 z=a/r 满足 sum(zᵢ²)=n，因此位于半径 √n 的球面。**球面约束成立在 gain/bias 之前**。它并没有把每个元素压到 [−1,1]：一个分量可以达到 √n，其余为零。用 L2 norm 会得到半径 1，比 RMSNorm 的每维典型量级小 √n；论文的 L2 实验没有重新调好所有补偿因素，不能据此宣称 L2 在任何网络中都无效。

由二阶矩展开，RMS²(a)=σ²(a)+μ²(a)。μ=0 时 RMSNorm 与 LayerNorm 的核心相同；μ≠0 时分母更大，且 RMSNorm 不减均值。RMSNorm 不强制 variance=1，而是强制 second moment=1。

{{diagram:flow}}

示意图将两条路径的统计量与 affine 操作并排展开。箭头表示张量依赖；这不是 2019 年的 kernel profiler trace，也不意味着实现一定启动图中那么多个 kernel。编译器可以把多个节点融合进同一个程序。

### 4.1 Invariance Analysis：哪些变化能被抵消？

{{eq:5}}

v 是 f 内部的 affine 后向量。统一将 W 或 x 乘 δ>0，a 与 RMS(a) 同时乘 δ，分子分母抵消；g、b 与 f 保持不变时输出保持不变。

{{eq:6}}

<div class="callout warning"><strong>公式边界：</strong>原 Eq.6 写 RMS(αx)=αRMS(x)，严格说要限制 α≥0；一般形式是 RMS(αx)=|α|RMS(x)。α&lt;0 时无 affine 核心翻转符号；α=0 且不加 epsilon 时分母为零。本文保留原式，并明确这些条件。</div>

{{eq:7}}

这里 W′=δW、a′=W′x。缩放某一行 wᵢ 会只改变一个 aᵢ，分母则由全部元素决定，其他分量也受到影响，因此单行／逐通道不同倍数缩放不具备同样不变性。

{{table:1}}

**Table 1 — 按列定义扰动。** 每列是对权重矩阵、单行权重、整数据集或单训练样本的某类缩放／中心化；✓／✗ 不是质量好坏。LayerNorm 与 RMSNorm 唯一不同的这一列是 weight-matrix re-centering，而非“所有输入平移”都受保护。把 W 的每行减去同一个向量会在 a 的各分量上形成共同偏移，LayerNorm 可以消掉；将 x 加任意向量则经 W 生成非均匀偏移，所以表中 dataset re-centering 仍为 ✗。

此表来自原论文分析，隐含非退化输入、统一正尺度和理想统计。现实实现的 epsilon 会破坏严格缩放等式：sqrt(δ²r²+ε) 不能约掉 δ。r²远大于ε时才近似成立。这个差异也解释了接近零的输入为什么需要单独测试。

{{interactive:norm}}

实验台把相同输入同时送入 LN、RMSNorm 与固定前缀 pRMSNorm，展示每个输出分量、分母和样本数。它用于理解代数，不拟合训练结果。先增加均值偏移看 LN 保持输出、RMS 改变；再改变正尺度看 epsilon=0 时 RMS 保持；最后选择前缀反例观察 pRMSNorm 的分母失真。

### 4.2 Gradient Analysis：尺度约束如何进入更新？

{{eq:8}}

设 q=∂L/∂v ∈ Rⁿ，u=g⊙q ∈ Rⁿ，r=RMS(a)。bias 的梯度为 q，gain 的梯度为 q⊙a/r。讨论缩放不变时，作者也隐含下游计算及上游 q 不变；如果别的支路使用未归一化 a，不能单独套这个局部结论。

{{eq:9}}

原 Eq.9 用 Kronecker product、diag 和下标求和表达 W 的梯度。这里 R ∈ Rⁿˣⁿ 是 a/r 对 a 的 Jacobian；I 为单位矩阵，aaᵀ 为 outer product，xᵀ ∈ R¹ˣᵐ。原式的 [·]ᵢ 索引约定不够清晰，因此工程实现使用下面等价、shape 明确的重构：

$$
R=\frac{I}{r}-\frac{aa^T}{nr^3},\qquad
\frac{\partial L}{\partial a}=\frac{u}{r}-\frac{a(a^Tu)}{nr^3},\qquad
\frac{\partial L}{\partial W}=\left(\frac{\partial L}{\partial a}\right)x^T.
$$

第一项是各维梯度按 1/r 缩放；第二项减掉沿 a 方向的分量。因为 aaᵀ/(nr²) 是径向投影，R 会消掉改变整个向量长度的局部扰动，保留切向方向变化。**无需创建 n×n 矩阵：**先算一个点积 aᵀu，再做向量运算，backward 是 Θ(n)，W 梯度再由 [n,1]×[1,m] 生成 [n,m]。

{{eq:10}}

δ>0 时 R′=R/δ。缩放 x 会让 R 的 1/δ 与 outer product 中 x 的 δ 抵消，W 梯度不变；缩放 W 时 x 不变，W 梯度缩小到 1/δ。固定学习率 η 的 vanilla SGD 下，参数相对长度的有效变化还会按约 1/δ² 缩小，这就是“implicit learning-rate adaptation”的直观来源。

<div class="callout warning"><strong>不能外推：</strong>这种局部梯度代数不保证全网络平滑、不保证不会梯度爆炸，也不是 Adam、weight decay、residual branch 和有限精度下完整训练动态的证明。原文把它解释为有助稳定收敛，实验提供的是经验支持。</div>

带内部 epsilon 的实际 backward 中令 r=sqrt(mean(a²)+ε)，仍是 u/r−a(aᵀu)/(nr³)，但 Ra 不再严格为零。这是数值稳定与精确不变性之间的取舍。该梯度已用独立的中心差分验证。


## 5. pRMSNorm：少看分母，仍然处理整个向量

**原文 §5，PDF p.5。** 作者提出用前 k 个元素估计 RMS：

$$
\overline{\mathrm{RMS}}(a)=\sqrt{\frac{1}{k}\sum_{i=1}^{k}a_i^2},
\qquad k=\lceil n p\rceil,\qquad
z_i=\frac{a_i}{\overline{\mathrm{RMS}}(a)}\quad(i=1,\ldots,n).
$$

这里 p 用 [0,1] 比例表示；p=6.25% 即 0.0625。论文文本说 “first-p%”，不是随机抽样，不是 top-k，也不是每个元素各自用局部统计。输出 shape 仍 [n]，gain 长度仍 n。平方归约从 n 项降到 k 项；读取和写出所有输出元素的工作仍在。

固定前缀也具有正尺度同次性，所以 Table 1 的理想不变性保持。但“同次性”只保证统一缩放可抵消，**不保证分母接近完整 RMS**。即使元素是 i.i.d.，对均方的估计无偏也不代表开方后的 RMS 无偏；固定前缀若具有不同分布，还会引入系统性偏差。原文承认估计不精确，小样本可能导致梯度不稳定；正文 “small m” 与此段定义的 k/n 不一致，应理解为小估计集合。

一个确定性反例：a=[1,1,100,100]，p=50%，前缀 RMS=1，完整 RMS≈70.714。pRMS 输出后两维都是 100，完整 RMS 输出约 1.414；小分母放大输出以及梯度。LLM 的 outlier 通道、MoE 分布差异或非随机 hidden layout 让“前缀代表整体”的假设尤其需要验证。

**本文补充推导：** 设 s=sumᵢ aᵢuᵢ，pRMS 的 backward 为 daⱼ=uⱼ/r−1{j≤k}aⱼs/(kr³)。所有输出的梯度都进入 s，但只有前缀输入还承担分母导数项。前缀与后缀受到不同训练作用；交换通道排列也可能改变输出，完整 RMS 则对通道置换保持等变性。

因此 pRMSNorm 的工程收益取决于统计归约是否真正是瓶颈。切片可能引入 kernel、copy 或不合适的布局，理论算术减少不能自动转为 wall-clock 加速。论文中 pRMS 比完整 RMS 更慢的 case 正是这条证据。


## 6. Experiments：先看质量，再看统一口径的时间

**原文 §6，PDF pp.5–9。** 未另说明时使用 TITAN X Pascal，计时是 3 次运行的均值与标准差。Transformer 改用 Tesla V100，CIFAR 改用 RTX 2080 Ti。论文横跨 TensorFlow、Theano、PyTorch，多样性有帮助，但不同框架 baseline 的细节也有差异，不能跨表把时间直接相除。

本文统一使用 **耗时降幅=(tLN−tnew)/tLN**，加速倍数=tLN/tnew。原表括号的百分比按前者保留。更低的 time 与 error 更好，更高的 BLEU 与 Recall 更好；未标注方向的原表不擅自改写，方向在解释中补充。

### 6.1 Machine Translation：RNN 与 Transformer 的收益不相同

**WMT14 En→De；原文 §6.1，PDF pp.6–7。** RNNSearch 在 recurrent connections 与 feed-forward layers 中加入归一化；Transformer 使用 base setting、训练 300k steps。配置详情见附录 A.1。

{{fig:2}}

**Figure 2 — TensorFlow RNNSearch 的开发集 BLEU。** 横轴为 training steps（每 30k 评估一次），纵轴为 newstest2013 SacreBLEU。先比较达到同一 BLEU 的步数，再比较平台值：LayerNorm、RMSNorm 与 pRMSNorm 比 Baseline 更快收敛，L2-Norm 较弱。图中没有时间横轴，因此不能仅凭曲线宣布实际更快；需要 Table 2。

{{table:2}}

**Table 2 — 相近质量，更低训练耗时。** 完整 RMS 相对 LN 的 Test14 为 −0.2 BLEU，Test17 为 +0.1；501 s 相对 665 s，降幅 24.66%，加速 1.327×。pRMS 的 Test14 持平、Test17 为 −0.5；493 s 降幅 25.86%，加速 1.349×。二者都比无归一化 Baseline 的 399 s 慢。作者称收敛步数约减少 50%，需要把每步成本与达标步数结合，才得到训练预算意义。

L2 不是最强 baseline：它只有 20.7／22.0 BLEU，且论文脚注承认前人只对最后一层使用 L2 并调尺度，本实验不是该方法的完整复现。不能将差结果归因于范数本身而忽略 placement 与调参。

{{fig:3}}

**Figure 3 — p 的消融。** 横轴是 partial ratio，纵轴是开发集 BLEU；TensorFlow Nematus 以 10% 步长改变 p。各点接近说明此 RNNSearch 配置对 p 不太敏感，但图没有多 seed 质量方差，也不是所有比率都充分测试。6.25% 是随后采用的设置，不能宣称图中每个点都直接证明 6.25% 最优。原 §6.1 的文字误指 Figure 7，PDF 实际对应本图；TeX 使用了重复 label。

{{table:3}}

**Table 3 — 框架依赖明显。** Theano 中 RMS 的 988→652 s 为 34.01% 降幅、1.515×，Test14 +0.2、Test17 −0.6 BLEU；pRMS 的 658 s 比完整 RMS 多 6 s，却获得 22.7／24.0 BLEU。PyTorch 中 RMS 的 857→763 s 为 10.97%、1.123×，Test14 −0.3、Test17 +0.2；Test17 最佳反而是无归一化 Baseline 的 24.7。这更支持“总体质量相近，时间可改善”，不支持“所有数据集指标全面胜出”。

{{table:4}}

**Table 4 — Transformer 的成本占比更低。** RMS 的 248→231 s 为 6.85%、1.074×；pRMS 的 225 s 为 9.27%、1.102×。RMS Test14 +0.2 BLEU、Test17 持平。Baseline 的 “−” 表示训练失败且 BLEU 为 0，210 s 只能用于说明无 Norm 的执行时间，不能作为同质量最快方案。RNN 归一化沿时间串行，Transformer 更易并行，这是作者对收益差异的解释；缺少独立 kernel trace，尚不能把全部差异精确归因给某一次 reduction。

{{table:5}}

**Table 5 — 均值不归零，也能更稳定。** M=mean，S=standard deviation；列 1–4 是 token 位置，ALL 是所有位置的平均。Baseline 的 S 从 7.35 跌到约 2.3–2.7，LN 与 RMS 基本维持约 1.5；RMS 的 M 从 −0.40 逐步到 −0.74，而 LN 也不是 0。这些测量是在指定 GRU hidden-to-hidden 表示上、包含模型上下文，不能误当成“裸 LN 核心输出仍不为零”的反证。它支持 RMS 在此模型中稳定统计，但没有完整分布尾部、跨层或跨 seed 的诊断。

{{fig:4}}

**Figure 4 — 将权重初始化中心移到 0.2。** 横轴 training steps，纵轴 dev BLEU；RMS 曲线比 LN 稳定，两者都低于正常初始化结果。先读纵轴平台，再看波动，不要把某个局部高点当最终质量。一个异常中心的 case 只能检验有限的 robustness，不能证明对任意 mean drift 都更稳；毕竟 RMS 在代数上仍不具备平移不变性。

### 6.2 CNN/Daily Mail Reading Comprehension：实体填空，不是摘要任务

**原文 §6.2，PDF pp.7–8。** 标题提 CNN/Daily Mail，实际实验训练在 CNN corpus；问题答案是 passage 中匿名化的实体。使用双向 attentive reader，top4 预处理，每篇最多 4 句。归一化只施于 LSTM，隐藏维度 240。

{{fig:5}}

**Figure 5 — 看验证错误率的下降。** 横轴 training steps，纵轴 validation error，越低越好。BatchNorm-LSTM 与 BatchNorm-Everywhere 均加快收敛，LN 与 RMS 曲线更低且相近，pRMS 稍弱。legend 区分归一化方法及 BatchNorm placement；不把后者当同一个算子。图提供收敛轨迹，没有正式完整 test 指标表，也没有 TP/FP/TN/FN。

{{table:6}}

**Table 6 — 每 100 steps 的时间。** RMS 392→333 s，降幅 15.05%、1.177×；pRMS 330 s，15.82%、1.188×。相对无 Norm 的 315 s 仍有开销。原脚注明确 BatchNorm 基于 cuDNN，不能直接把其 345／348 s 与其余非同等优化实现比较，进而宣布算法复杂度排名。

### 6.3 Image-Caption Retrieval：收益很大，也更受旧实现影响

**原文 §6.3，PDF pp.8–9。** COCO 图文双向检索。OE（order embedding）用 GRU 编码文本、预训练 VGG 的 10crop 特征表示图像，以 pairwise ranking loss 学联合空间。Caption retrieval 是 image→caption，image retrieval 是 caption→image。

{{fig:6}}

**Figure 6 — 三个 K，同一个训练过程。** 左到右为 Recall@1、@5、@10，横轴为 training steps，纵轴为 recall；每 300 steps 评估。先比较同一面板内方法，再跨面板看难度：K 越大，允许命中候选更多，Recall 通常越高。LN 与 RMS 较早到达更高曲线，RMS 在验证集略优。图不含 mean rank，也不能替代 Table 7 的测试集结果。

{{table:7}}

**Table 7 — 保留两方向的所有指标。** R@K 是正确候选进入 top-K 的查询比例，越高越好；Mean r 越低越好，不是 median rank。上半部来自 existing work，下半部才是本文自己复现的对照，前者存在实现与随机性差异，不应优先拿来算因果提升。

以 This Work 的 LN 为公平对照：RMS caption R@1 47.9→48.7，+0.8 个百分点，相对 +1.67%；image R@1 38.4→39.0，+0.6 个百分点，相对 +1.56%。Caption R@10 +0.3，但 image R@10 86.7→86.3 为 −0.4；caption Mean r 5.3 持平，image Mean r 7.5 持平。pRMS 的 caption R@1 反而是 46.8。没有一个方法在全部列严格占优；原文“better generalization”应读成总体观察。

每个测试 split 含 1000 图像、5000 captions，报告 5 个测试集合的平均；这不是 5 个训练 seed。Recall@K 的检索定义不提供完整二分类 confusion matrix，不能从表中虚构 TP/FP/TN/FN、Precision 或 F1。

{{table:8}}

**Table 8 — 摘要最高数字的来源。** LN 12.02→RMS 7.12 s，为 40.77% 耗时降低、1.688×；pRMS 4.34 s 为 63.89%、2.770×。单位是每 100 steps。Baseline 仅 2.11 s，所以 pRMS 仍约 2.06×于无 Norm 时间。原文 Table 8 没有内核融合、精度、带宽、launch 或 profiler 细节；旧 Theano RNN 上的巨大 overhead 不应直接迁移到今天单 kernel 的 RMS/LN 比较。

### 6.4 CIFAR-10 Classification：partial approximation 的代价显现

**原文 §6.4，PDF p.9。** Modified ConvPool-CNN-C；归一化沿图像 width/height，gain 和 bias 沿 channel。这里不是今天 LLM 最后一维 token RMSNorm 的同样轴。模型较小，p 调到 **12.5%**，不是前面通用的 6.25%。

{{table:9}}

**Table 9（原文编号）— 实际为训练曲线。** 横轴 epoch，纵轴 training error，越低越好。BN 收敛最好，各种 Norm 比 Baseline 快；训练曲线上的优势不保证测试 error 同样更低。原文 §6.4 叫它 “Figure 9”，但 PDF caption 及 TeX table 环境实际使它编号为 Table 9；本文按 PDF 原样保留，不自行重排后面的 Table 10。

{{table:10}}

**Table 10 — 最强质量对照是 BatchNorm。** BN test error 8.25%，WeightNorm 8.28%，均优于 RMS 8.83%；RMS 优于 LN 10.49%。RMS 相对 LN 降 1.66 个百分点，error 相对减少 15.82%；时间 39→31 s，降幅 20.51%、1.258×。RMS 相对无 Norm 的 8.96% 只改善 **0.13 个百分点**，相对减少 1.45%，不能写成原文正文的 “0.013%”。

pRMS 10.37% 比 RMS 多 1.54 个百分点的错误，只省 1 s/epoch（相对 RMS 3.23%）；原文所谓 “further 2.6%” 是二者相对 LN 降幅 23.1% 与 20.5% 的差，约 2.6 个百分点，不是 pRMS 相对 RMS 的时间降幅。相同表还显示 Baseline 学习率 0.0003、其余 0.003，比较质量时必须承认这一混杂因素。


## 7. Conclusion and Future Work：是可用替代，也存在明确边界

**原文 §7，PDF p.9。** 作者认为 RMSNorm 保留有用的尺度不变性，省去中心化的开销；NLP 任务中质量相近，时间改善依赖框架、硬件、架构及其他模块成本。pRMS 的理论省算没有稳定兑现为更快，因此低层实现优化留待未来。

作者称 drop-in replacement，指结构形状与部署接口容易替换。**已有 LayerNorm checkpoint 的函数一般会改变**：a/r 与 (a−μ)/σ 不相同，bias 的保留也不保证等价，所以普通 pretrained LLM 不能只换类名就声称质量无损。

原文展望其他范数与 BatchNorm 简化，也预期能与低精度和 kernel fusion 叠加。这是合理方向；但实际叠加幅度仍取决于融合后哪个成本消失、哪个成本留在关键路径。以下独立审计和 2026 年更新会继续验证这点。


## Appendix A：配置细节决定实验能否复现

### A.1 Machine Translation

**原附录 A.1，PDF pp.13–14。** WMT14 4.5M aligned sentence pairs；newstest2013 选择模型，newstest2014／2017 测试；32k BPE merge；最大似然训练、label smoothing 0.1。Case-sensitive detokenized SacreBLEU，原签名：

```text
BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt14+tok.13a+version.1.2.12
BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt17+tok.13a+version.1.2.12
```

RNNSearch：最大 source/target length=80，embedding=512，hidden=1024，batch=80，Adam lr=1e−4。一般权重为标准正态缩放 0.01，方阵随机正交。Transformer：D=512，FFN=2048，8 heads，batch=25000 source/target tokens，Adam β₁=0.9、β₂=0.98，inverse-square-root schedule，warmup=4000，Xavier 初始化，300k steps。

{{fig:7}}

**Figure 7 — Theano RNNSearch 的 WeightNorm 比较。** 横轴 steps，纵轴 dev BLEU。WeightNorm 收敛慢于 LN／RMS／pRMS；附录给出 test BLEU 21.7／23.5，质量未超过本文主要归一化方案。PyTorch 官方 WeightNorm API 的尝试导致 OOM，作者未给完整 memory trace；这是一条具体实现失败记录，不能等价于 WeightNorm 必然比 RMS 使用更多激活内存。

### A.2 CNN/Daily Mail Reading Comprehension

使用公开 Theano 模型，CNN 的 top4 setting；只在 LSTM 内使用 RMS 来匹配 LN placement，hidden=240，Adam batch=64、lr=8e−5，Xavier 初始化，方阵进一步正交。未披露总参数、全训练 tokens、完整 test accuracy 与质量 seed 方差。不能把任务标题扩大成 CNN+Daily Mail 上全部配置的验证。

### A.3 Image-Caption Retrieval

VGGNet 10crop 图像特征；word embedding=300，GRU hidden=1024；不同参数使用 Xavier、Gaussian、orthogonal 初始化。Adam batch=128、lr=1e−3。训练 ranking loss，不改变 loss 以适配 RMS。最终 R@K 为 5 个 split 的平均，每 split 1000 images／5000 captions。

### A.4 CIFAR-10 Classification

沿 width、height 归一化，沿 channel affine；Adam batch=100。Baseline lr=0.0003，其余 lr=0.003；pRMS=12.5%。原文未完整披露 RMS 的累加精度、epsilon 放置、warmup、CUDA/cuDNN 版本和所有 task 的 batch timing 边界。因此复现时必须使用代码补足，同时保留代码与论文不一致之处。
