# Attention Is All You Need

用内容决定信息流：从 Transformer 原论文，到训练并行与逐 Token 推理。

## Paper Metadata

**作者**：Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Łukasz Kaiser、Illia Polosukhin。**机构**：Google Brain、Google Research、University of Toronto。**会议**：NIPS 2017（现称 NeurIPS）。

**首次提交**：2017-06-12。**本文阅读版本**：1706.03762v7，2023-08-02，15 页，包括 References 与 Attention Visualizations。2017 是论文发表年，2023 是这个修订版本的日期。阅读日期：**2026-10-27**；排版更新日期：**2026-10-28**。

[原论文 PDF](assets/original/1706.03762v7.pdf) · [arXiv](https://arxiv.org/abs/1706.03762) · [arXiv HTML v7](https://arxiv.org/html/1706.03762v7) · [DOI](https://doi.org/10.48550/arXiv.1706.03762) · [官方 Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)

本文按 Abstract → §1–7 → Attention Visualizations 的顺序精读，再展开实现、后续研究与当前评测。**原论文**表示 PDF 内证据；**外部来源**表示后续论文或官方资料；**分析 / 推导**表示本文重建。原图 5 张、原表 4 张全部保留，图题、表题和原文说明均以可选择文字放在图片外，图片保留完整图形、组件标签与表格内容；原表还提供可选择文字的版本。额外图示和交互均标明来源。

## Abstract / TL;DR：先建立全局

**一句话**：Transformer 把序列内的信息交换改成注意力，把每个位置的特征加工交给 FFN，在翻译任务上证明了无需循环和卷积也能构建有效的 encoder–decoder。

**一段话**：以前的翻译模型通常逐位置运行 RNN，或者堆叠卷积扩大感受野，再通过 attention 读取源句。Transformer 让每个位置直接根据 Q 与 K 的相似度选择 V，以多头机制同时保留多种读取关系，用位置编码表达顺序。训练时已知整条目标句，右移输入加因果 mask 可以并行计算各位置；生成时下一个词还未知，仍需一步步产生。原论文在 WMT14 英德上得到 28.4 BLEU；英法表格为 41.8，但正文有 41.0 的冲突。它的历史贡献是建立一种通用序列架构，不能由这些实验推出今天所有模型或硬件上的性能结论。

**五个关键点**：

1. Attention 负责跨位置通信；FFN 负责单位置的非线性变换，二者相互补充。
2. 多头不是重复同一个注意力图，而是先用不同投影构建不同的 Q/K/V 子空间。
3. 因果 mask 配合目标输入右移，既防止训练泄漏，又允许 teacher forcing 并行。
4. 短信息路径带来建模优势，但全连接注意力的序列混合计算为二次增长。
5. 原文的训练时间、BLEU、PPL 与解析 F1 都有明确设置；不能直接换算成推理延迟、准确率或现代榜单排名。

**推荐阅读路径**：先看 [Figure 1 与数据流](#architecture)，再用 [注意力交互](#attention-lab) 理解式 (1)，看 [Table 2](#table-2) 校准历史贡献。关注部署时，继续看 [KV 与 FLOPs](#cache-lab) 和 [复现边界](#implementation)。

## 1. Introduction：论文究竟移除了什么瓶颈？

**原论文，p.1–2。** 问题是 sequence transduction：给定源序列，生成另一条序列，如英语到德语翻译。当时的高质量模型以 RNN/LSTM/GRU encoder–decoder 为主，attention 已经广泛存在，Transformer 并不是注意力机制的发明。

RNN 的核心依赖是 `h_t = f(h_(t−1), x_t)`。即使所有输入 token 已知，计算第 t 个隐藏状态仍要等待前一状态。这限制训练时沿序列长度维度的并行；长序列又受单设备内存约束，难以简单靠更大 batch 抵消。

作者的观察是：**让每个位置直接读取其他位置，可以同时缩短通信路径和解除层内位置计算的循环依赖。** “All You Need” 要结合全文读：架构还需要 embedding、位置编码、FFN、残差、LayerNorm、输出投影、loss 与优化器；它移除的是作为序列建模骨架的 recurrence 与 convolution。

**证据与边界。** 论文通过两个翻译任务和一个句法解析任务验证这条路线，报告更高 BLEU 和较低的估计训练成本。它没有给出消除每个模块的完整因果实验，也没有证明 self-attention 对所有任务都最优。序列维度训练并行与自回归推理串行必须分开理解。

## 2. Background：从固定路径变成内容寻址

**原论文，p.2；外部来源 [R2–R3]。** 在经典 encoder–decoder 中，编码器把源句变成状态，解码器依据目标前缀生成下一词。Bahdanau attention 允许 decoder 动态读取源句不同位置，缓解压缩成一个固定向量的瓶颈，但编码器和解码器的 RNN 仍沿时间步串行。

ConvS2S 使用卷积，在已知序列上可以并行计算。代价是一个局部卷积层只能读取邻近 token；远距离信息要跨越多个层。论文也讨论 Extended Neural GPU、ByteNet 等路线，并指出长距离路径会随着距离增加。

Transformer 把读取关系交给输入内容：同一句中，一个 token 可以在一层里与很远的 token 建立连接。作者同时承认，加权平均可能降低有效分辨率；多头用不同子空间减轻这个问题。**分析**：这是从“固定拓扑逐步传消息”转向“每层动态决定读哪些位置”。短路径有利于表达，不能直接等同于梯度稳定或推理更快。

## 3. Model Architecture：按真实数据流重建

<span id="architecture"></span>

### 3.1 Encoder and Decoder Stacks

[[FIGURE:1]]

**这张图解决什么问题？** Figure 1 说明 encoder 如何产生源句表示，以及 decoder 如何同时读取目标历史与源句。阅读时从左右下方的 embedding 开始，沿纵向箭头向上，再跟随 encoder 输出通往 decoder 中间 attention 的横向箭头。

**组件与数据流**：左边是 6 层 encoder，每层为 self-attention → Add & Norm → FFN → Add & Norm。右边是 6 层 decoder，每层先 masked self-attention，再 encoder–decoder attention，最后 FFN，每个子层后都有残差与归一化。右下角的 **Outputs (shifted right)** 是训练时的目标前缀输入，右上角才是预测分布，不能把两者当作同一条未右移序列。

**原论文配置**：base 的模型宽度 D = 512、FFN 隐层 2048、8 头、每头 d_k = d_v = 64。big 为 D = 1024、FFN 4096、16 头，每头仍是 64；6 层指 encoder 和 decoder 各 6 层。所有残差分支的外部维度保持 D，便于相加。原版使用 **post-LN**：

[[EQ:residual]]

**符号与运算**：x 为 `[B,n,D]`，Sublayer 输出相同 shape，dropout 后逐元素加上 x，再沿最后的特征维进行 LayerNorm。归一化不是跨 token 混合。论文没有用现代 decoder 中常见的 pre-LN、RMSNorm、SwiGLU，也没有在这篇论文里使用 RoPE。

**关键观察**：decoder 中间的 attention 不是 encoder self-attention 的复制；它的 Q 来自目标，K/V 来自源句。这条桥接通道使输出与输入对齐。**分析**：把图拆成“目标历史读取”“源句读取”“单位置加工”，比把整个 decoder 当作黑盒更容易定位实现问题。

[[DIAGRAM:flow]]

| 张量 / 阶段 | Shape | 信息来自哪里 |
|---|---|---|
| 源 token ID | `[B,n_s]` | 完整输入句 |
| Encoder 输入 / 输出 | `[B,n_s,D]` | embedding + PE，经 6 层编码 |
| 右移目标 ID | `[B,n_t]` | BOS 与正确目标前缀 |
| Decoder 隐状态 | `[B,n_t,D]` | 因果目标历史与源句 |
| Cross-attention Q | `[B,h,n_t,d_k]` | decoder 当前子层输入 |
| Cross-attention K / V | `[B,h,n_s,d_k]` / `[B,h,n_s,d_v]` | encoder 最后一层输出 |
| Cross 权重 A | `[B,h,n_t,n_s]` | 每个目标位置对源位置的分布 |
| 词表 logits | `[B,n_t,Vocab]` | decoder 输出 × 共享 embedding 的转置 |

源句与目标句长度 n_s、n_t 独立；矩阵通常不是方阵。Padding mask 必须分别跟随对应序列长度。

### 3.2 Attention：Q、K、V 不是三个固定含义的单词

**原论文，p.3–5。** Query 是“当前位置用什么特征发起读取”，Key 是“其他位置用什么特征被检索”，Value 是“读到后实际带走什么内容”。它们是可学习投影后的向量，不必具有可直接命名的人类语义。Self-attention 时三者都源于同一隐藏序列，但通常使用不同权重；cross-attention 时来源不同。

#### 3.2.1 Scaled Dot-Product Attention — 式 (1)

[[FIGURE:2]]

**如何读 Figure 2？** 左图从 Q/K 的 MatMul 开始，经 Scale、可选 Mask、SoftMax，最后与 V 做 MatMul。右图把输入分别投影成 h 组 Q/K/V，各组独立 attention，再 concat 与输出 Linear。图中输入排列为 V、K、Q；不要据横向排列误认三个投影的数据来源。

[[EQ:attention]]

**符号与 shape**：暂时省略 B、h。Q 为 `[n_q,d_k]`，K 为 `[n_k,d_k]`，V 为 `[n_k,d_v]`；QKᵀ 为 `[n_q,n_k]`，softmax 沿 key 轴逐行归一化，输出为 `[n_q,d_v]`。如果加 mask，先将不允许位置的分数设为 −∞，再 softmax；将输出清零无法修复已发生的信息泄漏。

**逐步计算**：QKᵀ 给出每个 query 与每个 key 的内容相似度；除以 √d_k 控制 logit 尺度；softmax 将一行变成非负且和为 1 的权重；AV 把该 query 的读取结果组合出来。权重行不是词表概率，也不是 token 的预测置信度。

**为什么缩放？** 原文脚注假设 q、k 各分量独立、均值为 0、方差为 1，则点积有 d_k 个项，方差为 d_k。除以 √d_k 后方差回到 1，避免维度增大时 softmax 过度饱和，使梯度变小。真实训练特征未必独立同分布，所以这是动机而非严格分布定理。它也不等同于数值稳定 softmax：实现还要减去每行最大值。

**Toy example / 分析**：Q = K = `[[1,0],[0,1]]`，V = `[[10,0],[0,20]]`。第一行缩放后 logits 是 `[0.7071,0]`，权重约 `[0.6698,0.3302]`，输出约 `[6.698,6.605]`；启用 causal mask 后第一行只能读第一个 value，变成 `[10,0]`。没有“选中一个 key”这一步，输出通常是连续加权和。

<span id="attention-lab"></span>
[[INTERACTIVE:attention]]

**工程含义 / 推导**：单头两次混合 MatMul 的乘加 FLOPs 为 `2·n_q·n_k·d_k + 2·n_q·n_k·d_v`，尚未包括 Q/K/V 投影与 softmax。方阵 self-attention 为二次序列增长；cross-attention 为源长 × 目标长。一个 query 如果整行都被 mask，普通 softmax 会遇到未定义结果；padding/query 有效性要在实际实现中处理。

#### 3.2.2 Multi-Head Attention — 未编号核心公式

[[EQ:multihead]]

W_i^Q、W_i^K 为 `[D,d_k]`，W_i^V 为 `[D,d_v]`，每个 head 输出 `[B,n_q,d_v]`。concat 得到 `[B,n_q,h·d_v]`，W^O 为 `[h·d_v,D]`，把多个读取结果重新混合回残差需要的 D 维。

原版取 `h·d_k = h·d_v = D`。多头把一个宽 attention 分成多个窄 attention，主要投影与混合算术量大致相同；**不能因此说实际耗时严格相同**，因为 kernel shape、launch、转置与 memory layout 会改变硬件效率。注意力权重存储还带有 h 这个维度。

**为什么多个头？** 一个 softmax 行只能给出一种归一化读取分布；多个头各自计算分布，再做输出混合，能同时保留不同的关系。Table 3 的单头结果低 0.9 BLEU，是任务内支持；32 头反而下降，说明更多头不是单调收益。附录展示头之间的不同图案，但没有证明每个头承担固定语言学功能。

#### 3.2.3 三种 attention 使用方式

| 类型 | Q 来源 | K/V 来源 | Mask | 目的 |
|---|---|---|---|---|
| Encoder self-attention | encoder 当前隐藏状态 | 同一层输入 | 源 padding；无因果限制 | 双向理解源句 |
| Decoder self-attention | decoder 当前隐藏状态 | 同一层输入 | 目标 padding + 因果 | 只读取可用目标前缀 |
| Encoder–decoder attention | decoder self-attention 后状态 | encoder 最终输出 | 源 padding | 把目标位置与源句对齐 |

**分析**：调试 attention 时先查 Q/K/V 来源、mask 广播维度和 key 轴 softmax，再查数值。把 cross-attention 当成 self-attention，或把源 padding 广播到目标轴，都可能不报 shape 错却改变含义。

### 3.3 Position-wise Feed-Forward Networks — 式 (2)

[[EQ:ffn]]

x 为 `[B,n,D]`，W₁ 为 `[D,d_ff]`、b₁ 为 `[d_ff]`；ReLU 后为 `[B,n,d_ff]`；W₂ 为 `[d_ff,D]`、b₂ 为 `[D]`，输出恢复 `[B,n,D]`。同一层中所有位置共享 FFN 权重，不同层有不同权重。等价描述是两次 kernel size 为 1 的卷积，但这不引入相邻位置通信。

**直觉 / 分析**：attention 先把别的位置的信息运来，FFN 再在本位置进行非线性组合。如果只做注意力加权平均，位置间能交流，却缺少足够的逐位置变换能力。

**计算 / 推导**：两个线性层约 `4·B·n·D·d_ff` FLOPs。在 base、n = 512、B = 1 时是 2.147 GFLOPs；这比 QKᵀ 与 AV 合计的 0.537 GFLOPs 大得多。因此只优化 attention 未必显著改善短句端到端耗时。

### 3.4 Embeddings and Softmax

**原论文，p.5。** 输入、输出 embedding 把 token 映射到 D 维，输出端 Linear 与 Softmax 生成词表分布。论文共享两个 embedding 层与 pre-softmax 线性变换的权重，并将 embedding 乘以 √D。这个缩放发生在 embedding 查表之后，与 attention 的 1/√d_k 是两件事。

**参数重建 / 推导**：以约 37k 共享词表估算，主要矩阵参数为：

[[EQ:parameters]]

分别对应共享 embedding、6 层 encoder、6 层 decoder；decoder 每层有 self 与 cross 两套 attention。代入 D = 512、d_ff = 2048 得约 **62.98M**，接近论文报告的 **65M**，但不是精确复原。词表“约 37k”的取整、实际配置、bias 与 LayerNorm 等都会影响差异；不能把近似值当成源代码精确参数统计。big 原文为 **213M**。

词表投影单独有 `2·B·n_t·D·Vocab` 的乘加成本，常被只讨论 attention 的估算漏掉；增量解码时也可能显著占比。

### 3.5 Positional Encoding：有顺序，但尚不是 RoPE

[[EQ:position]]

pos 是位置，i 是通道对编号 `0…D/2−1`；每一对通道共享角频率 `ω_i = 10000^(−2i/D)`。PE shape 是 `[n,D]`，广播到 batch 后与 embedding **相加**。偶数通道用 sin，奇数通道用 cos；波长从 2π 延伸到接近 10000·2π。

**为什么选择 sin/cos？** 对固定偏移 Δ，由和角公式可把 `PE(pos+Δ)` 表成 `PE(pos)` 的一个只依赖 Δ 的二维线性变换。每个通道对 `[sin θ, cos θ]` 的变换矩阵是 `[[cos δ,sin δ],[-sin δ,cos δ]]`。这让网络有机会学到相对偏移关系。

**边界 / 分析**：相对位移的线性可表示性，**不等于最终 QKᵀ 只依赖位置差**。这里 attention 用的是对 `embedding + PE` 的投影，展开点积会有内容–位置交叉项。作者选择正弦编码还出于外推到更长序列的期待；Table 3 的 25.8 与 learned PE 的 25.7 只支持该开发集上效果接近，没有提供系统性的长度外推实验证明。

后来的 [RoFormer 精读](../roformer/) 在 Q/K 上旋转，构造相对位置点积性质；[VRoPE 精读](../vrope/) 进一步处理视频空间与时间坐标。两者都是后续路线，不能倒写成 2017 原版使用的方法。

## 4. Why Self-Attention：复杂度表需要读脚注

[[TABLE:1]]

**比较对象与指标**：Table 1 比较不同层类型的每层复杂度、最低串行操作数量与最大信息路径。n 是序列长，d 是表示宽度，k 是卷积核宽度，r 是局部 attention 邻域。它不是 BLEU 表，也不是测量出的 kernel 耗时表。

Self-attention 给出 `O(n²d)`、`O(1)` 串行阶段和 `O(1)` 路径；RNN 为 `O(nd²)`、`O(n)` 阶段和 `O(n)` 路径。因此当 n < d 时，混合部分的理论计算量有利；更核心的区别是所有位置能在一层直接联系。

**两个容易漏掉的限制**：第一，表中的 attention 成本突出序列混合项，完整 block 还有 `O(nd²)` 的投影和 `O(nd·d_ff)` 的 FFN。第二，convolution 的 `O(log_k n)` 路径对应扩张卷积；普通连续卷积需要 `O(n/k)` 层，论文 p.7 明确区分了两者。

**局部化 trade-off**：限制在大小 r 的邻域后，混合成本成为 `O(rnd)`，但全局信息传播路径增长为 `O(n/r)`。论文把它列作未来工作，不能当作本实验实际训练的窗口 attention。

**分析**：O(1) 是单层沿已知序列的位置依赖深度，不是整个网络只需一个操作，更不是任意长度文本能一次性生成。实际模型仍有 L 层依赖，自回归推理仍有 T 步依赖。Tensor shape、memory IO 与硬件并行度必须加入分析。

## 5. Training：可复原的设置与缺失信息

### 5.1 Training Data and Batching

**原论文，p.7。** WMT14 英德约 **4.5M 句对**，BPE，共享源/目标约 **37k** 词表；英法约 **36M 句对**，使用 **32k word-piece** 词表。按近似句长组 batch，减少 padding；每批约 **25k 源 token + 25k 目标 token**，不是 25k 个句子。

**实验审计**：论文报告了数据规模、tokenizer 类型和近似 batch token 数，但没有统一给出训练语料的完整有效 token 总数、各长度区间分布和所有 baseline 的同预算训练版本。将 batch token 数 × step 数作为累计处理 token 的粗估时，还必须区分源/目标、padding 与跨设备统计口径，不能说它就是去重训练集 token 数。

### 5.2 Hardware and Schedule

使用一台机器的 **8 张 NVIDIA P100**。base 每 step 约 **0.4 s**，训练 **100k steps，约 12 小时**；big 每 step 约 **1.0 s**，训练 **300k steps，约 3.5 天**。0.4×100k = 11.11 小时，1×300k = 3.47 天，与原文近似时间同一量级。

**边界**：原文训练成本用估计的持续单精度能力作代理，并没有提供完整 mixed precision recipe、通信 trace、显存峰值、功耗、TTFT 或 TPOT。后文 FP16 Cache 例子是工程场景，不是声称论文当时以 FP16 训练。

### 5.3 Optimizer — 式 (3)

[[EQ:schedule]]

Adam 参数为 β₁ = 0.9、β₂ = 0.98、ε = 10⁻⁹。s 为 optimizer step，w 为 warmup_steps = **4000**，D 为模型宽度。s≤w 时学习率线性增加；s>w 时按 s⁻¹ᐟ² 衰减。base 在 s = 4000 的峰值约 **0.000699**，s = 100000 时约 **0.000140**。

**直觉 / 分析**：先避免训练初期大步更新，再随训练推进减小步长；D⁻¹ᐟ² 提供宽度相关缩放。它不是 cosine schedule，也不是只写一个固定 learning rate 就能复现。实际实现的 step 从 0 还是 1 开始、额外倍率与梯度累计的 optimizer step 定义都需要核对。

### 5.4 Regularization 与训练目标重建

子层输出在残差相加前 dropout；encoder / decoder 的 embedding 与 PE 相加后也 dropout。base 的 P_drop = **0.1**，Table 3 的 big 为 **0.3**，英法 big 特例仍是 **0.1**。Label smoothing ε_ls = **0.1**；原文指出它提高 BLEU，却使 perplexity 更差。

[[EQ:loss]]

**推导，非原文编号公式。** 给定源 x，目标序列 y，训练最大化正确目标前缀条件下的 next-token 概率，等价于最小化上述交叉熵。P 是模型词表分布，q 为 label-smoothed 目标分布；可用 `q(correct)=1−ε`，其余有效词项共享 ε。实际实现须确认 padding/BOS 等特殊 token 是否排除，以及采用 ε/V 还是 ε/(V−1) 的约定。训练 loss 不一定等于评测普通 hard-label NLL，因此 PPL 变化不能单独决定翻译质量。

[[DIAGRAM:training]]

**右移与 mask / 分析**：目标 `[y₁,y₂,y₃,EOS]` 的 decoder 输入是 `[BOS,y₁,y₂,y₃]`。第 i 行只能读 j≤i 的输入；当前位置看到的是前一个目标 token，而非该行要预测的答案。所有正确前缀已知，能一次计算各行 logits；这是 teacher forcing。若不右移，即使 mask 是下三角，也会让当前位置读到答案。

## 6. Results：按每张表核对结论

### 6.1 Machine Translation — Table 2

[[TABLE:2]]

**读表方式**：数据为 WMT14，测试集 **newstest2014**；BLEU 越高越好，Training Cost 的 FLOPs 越低越好。原表没画 ↑/↓，本文只解释方向而不改原貌。空白表示未报告，不是 0。single 与 ensemble 是不同成本设置。base 用最后 **5 个 checkpoint** 的权重平均，big 用最后 **20 个**；权重平均产生单个模型，不等于运行多个模型再集成预测。

解码采用 **beam size 4**、length penalty **α = 0.6**，在 dev 上选择；最长输出设为输入长 + 50，允许提前结束。BLEU 是翻译与参考文本的 n-gram 匹配指标，并带 brevity penalty；不能称为“28.4% 翻译准确率”。tokenization 与评测脚本会影响分数，后续 SacreBLEU [R13] 强调可复现的评测签名。

| 对比 / 本文复算 | 绝对差 | 数值比值 | 可以支持的结论 |
|---|---:|---:|---|
| EN–DE big 28.4 vs ConvS2S ensemble 26.36 | **+2.04 BLEU** | +7.74% 的分数算术比 | 当时英德超过最强表内 ensemble；不是准确率提升 7.74% |
| EN–DE base 27.3 vs 同上 26.36 | +0.94 BLEU | +3.57% | base 也超过该英德 ensemble |
| EN–FR big 41.8 vs MoE single 40.56 | +1.24 BLEU | +3.06% | 按表格与摘要值，超过最强表内 single |
| EN–FR big 41.8 vs ConvS2S ensemble 41.29 | +0.51 BLEU | +1.24% | 仅按 41.8 的口径 |
| EN–FR base 38.1 vs MoE single 40.56 | −2.46 BLEU | −6.07% | base 并没有同时胜过两个语言任务的所有方法 |

**原文冲突**：摘要与 Table 2 写 **41.8**，§6.1 正文写 **41.0**，本文不擅自纠正。如果按 41.0，则相对 MoE single 仅 **+0.44 BLEU**，相对 ConvS2S ensemble 是 **−0.29**。因此英法是否超过最强 ensemble 取决于冲突的取值。英德 28.4 的证据没有这个冲突。

**训练成本如何得来？** 原文是 `wall time × GPU 数 × 估计持续 TFLOPS`，脚注使用 K80 2.8、K40 3.7、M40 6.0、P100 9.5 TFLOPS。Transformer 两行的 cost 单元格横跨 EN–DE / EN–FR 两列，分别是 **3.3×10¹⁸** 与 **2.3×10¹⁹**；它不是两个独立语言任务的逐算子测量值。

**成本复算与公平性**：big/base 约 **6.97×**；英德 big 与 GNMT single 的代理成本相同，约为 ConvS2S ensemble 的 **1/3.35**；英法与 MoE single 相比约 **1/5.22**。这些是文献估计之间的比值，各模型的数据、硬件、优化、模型规模和训练时间并非统一控制；它支持“历史上的质量与估计成本组合很强”，不能当作严格同硬件的 FLOP/能效测量。

### 6.2 Model Variations — Table 3

[[TABLE:3]]

**设置**：英德 **newstest2013 开发集**，beam 同 §6.1，但**没有 checkpoint averaging**。PPL 为 **per-wordpiece**，不能与按词统计的 perplexity 直接比较。未列设置继承 base；big 的空白 d_k / d_v 因而为 **64 / 64**。Table 3 base dev 25.8 与 Table 2 test 27.3 不是同一评测设置。

**(A) Head 数**：保持总宽度与主要计算量，h = 1 / 4 / 8 / 16 / 32 时 BLEU 为 24.9 / 25.5 / 25.8 / 25.8 / 25.4。单头低 **0.9**，16 头没有再提高，32 头退 **0.4**。支持多头有效和每头宽度存在 trade-off，不支持“头数越多越好”。

**(B) Key 维度**：d_k 从 64 降为 16 / 32，BLEU 25.1 / 25.4，参数 58M / 60M；d_v 未列，仍继承 64。作者推测点积兼容性判定不容易；这并没有直接实验证明一个更复杂的打分函数必然更好。

**(C) 宽度与深度**：N = 2 / 4 / 8 的 BLEU 为 23.7 / 25.3 / 25.5，参数 36M / 50M / 80M；8 层并未超过 6 层 base。D = 256 / 1024 得到 24.5 / 26.0，参数 28M / 168M。d_ff = 1024 / 4096 为 25.4 / 26.2，参数 53M / 90M。d_ff 增大是较明显的收益来源，同时直接增加 FFN 成本；“更大总更好”是超出这张表的概括。

**(D) 正则化**：移除 dropout 使 BLEU 从 25.8 降到 **24.6**；P_drop = 0.2 为 25.5。去掉 label smoothing 后 **PPL 4.67 更低**，但 BLEU **25.3 更差**；ε_ls = 0.2 得 PPL 5.47、BLEU 25.7。这个结果清楚展示 loss / calibration 与任务指标之间的取舍。

**(E) 位置编码**：learned PE 得 25.7，sinusoidal 25.8，单次差距仅 **0.1 BLEU**。未提供多 seed 方差或显著性，不能宣称正弦必然优于 learned。

**big**：213M、300k steps、dev BLEU 26.4 / PPL 4.33，对比 base 65M、100k。参数约 **3.28×**，训练步数 **3×**，多项超参同时改变；这不是单因素“只加宽度”的消融。

**缺失实验 / 分析**：建议补多 seed、同预算 head/FFN 对比、长句分桶、长度外推、去位置编码、严格匹配的 pre/post-LN，以及头干预实验。原表是有价值的探索，但不足以对每一模块给出稳定、独立的因果排序。

### 6.3 English Constituency Parsing — Table 4

[[TABLE:4]]

**任务与设置**：将英语句子转换成线性化的 constituency tree，输出有强结构约束且往往比输入长。4 层 Transformer，D = **1024**；WSJ 约 **40k 训练句**、**16k** 词表；半监督高置信语料约 **17M 句**、**32k** 词表。WSJ Section 22 调 dropout、学习率和 beam，Section 23 测试；解析任务最大输出设为输入长 + **300**，beam **21**、length penalty **0.3**（p.10）。

**读表与结论**：F1 衡量 labeled constituents / spans 匹配，越高越好。WSJ-only 的 Transformer **91.3**，低于 Dyer **91.7**，差 **−0.4 F1**；semi-supervised **92.7**，比表内其他同组最高 **92.1** 高 **+0.6**，但低于 multi-task Luong **93.0** 和 generative Dyer **93.3**。训练范式、额外语料不同，不能合成一个公平的纯架构排行榜。

**原文 claim 审计**：§6.3 说除 RNNG 外胜过全部先前模型，忽略了表中 Luong 93.0；更稳妥的结论是**在表内半监督组表现最好，并显示迁移到结构输出任务的可行性**。不能据三个有限任务宣称全面通用性。

**指标细分 / 分析**：若以 labeled span 为元素，TP 是预测和标注同时出现的 span，FP 是多预测的 span，FN 是漏掉的 span；Precision = TP/(TP+FP)，Recall = TP/(TP+FN)，F1 = 2PR/(P+R)。论文未公布计数或独立 P/R，无法从一个 F1 唯一反推出它们。TN 与普通分类 accuracy 不适合作为这类解析表的替代指标，也不能编造混淆矩阵。

## 7. Conclusion：成立的贡献与没有回答的问题

**原论文，p.10。** 作者总结 Transformer 完全以 attention 构建序列转导模型，在翻译上得到强结果且训练更快，并扩展到 constituency parsing。未来方向包括图像、音频、视频、局部 attention 和减少生成的串行性。

**分析**：历史上最重要的贡献是把 attention、位置表示、逐位置 FFN 与残差组织成一个可扩展的序列架构。今天的 LLM/VLM 延续了这个骨架，但常常修改归一化、位置编码、KV 表示、FFN、训练目标、并行与内核实现。阅读原论文能建立数据流基准，同时避免把后续优化归给 2017 实验。

## Attention Visualizations：原附录三张图逐项读

### Figure 3 — making … more difficult：一跳连接长依赖

[[FIGURE:3]]

**目的与阅读顺序**：这张图展示 encoder 第 **5/6 层**中，query **making** 的注意力。上、下两排是同一句 token；从上排 making 沿彩色连线看下排被读取位置。不同颜色为不同 head，透明度和色块显示注意力强弱；句末 `<EOS>` / `<pad>` 也属于原图内容。

**观察**：多个头指向远处 **more / difficult**，对应 “making … more difficult” 的关系，也有头指向自己或较近位置。其价值是直观说明不同 head 可以跨越插入的短语，在一层直接联系远处词项。

**解释边界**：这是筛选出的单词、单句、单层 attention，未说明这些连接对翻译结果的必要性，也不能推出全模型头都自动变成长依赖检测器。注意力图是读取模式，不是完整因果归因。

### Figure 4 — its：指代与不同读取关系

[[FIGURE:4]]

**如何读**：上半部分保留 head **5** 的整句 attention；下半部分只保留 **its** 发出的关系，并展示 head **5 / 6**。原句是 “The Law will never be perfect, but its application should be just …”。跟随 its 的连线，看它与 **Law / application** 的联系；不要把下半部分误当成整个 layer 的所有 heads。

**观察与意义**：两个头对 its 都很尖锐，但关注的关联不同，符合指代与名词短语结构的直觉。原 caption 使用 “apparently involved”，是定性描述。

**分析**：把这张图理解为“多种读取关系可以并存”最稳妥。要证明模型通过这些头完成 anaphora resolution，需要屏蔽/替换头或改变指代语句的干预，并测量输出变化。

### Figure 5 — 结构相关模式：不同头并不重复

[[FIGURE:5]]

**组件与阅读**：上下两个完整面板来自 encoder 第 **5/6 层**的不同头，上面绿色、下面红色。同一原句的源/目标 token 排列、标点、EOS/PAD 全部保留。对比每个位置发出的线，而不是只比较颜色。

**观察**：上图可见跨词组、偏向 Law、application、missing 等位置的聚合；下图不少位置保持较强自连接，也出现较局部的其他联系。它说明两个头学到不同分布，不证明每条边等同于句法树依存边。

**外部来源 [R12]**：后续研究 “Attention is not Explanation” 进一步提醒，attention weights 未必能作为预测解释的充分依据。这不否定原图的观察价值，而是区分可视化发现与干预后的机制证据。

## Engineering Reconstruction：KV、计算与端到端性能

<span id="implementation"></span>

### 从训练矩阵到增量 decoder

**推导，原论文没有这些推理 benchmark。** Encoder 先处理完整源句一次。Decoder 每层保存过去目标 token 的 self-attention K/V；每个新 token 只生成当前 Q/K/V，把新的 K/V append，再读取历史。Cross-attention 的 K/V 由不变的 encoder 输出在各 decoder 层各自投影一次，可重复使用。

[[DIAGRAM:cache]]

**每步数据流**：当前 token embedding → decoder layer 1 的新 self Q/K/V → 读取该层历史 cache → 源 K/V cross 读取 → FFN → 后续层 → 词表 logits → 选 token → 下一步。各层 cache 存的是该层输入所生成的 K/V，不能把上一层的 cache 当成本层数据复用。

[[EQ:cache]]

L_d 是 decoder 层数，B 是 batch，T 是缓存目标长度，h_KV 是 KV head 数，d_head 是每头宽度，b 是每元素字节，2 代表 K 与 V。原版 MHA 的 h_KV = h。Cross cache 把 T 换成 n_s；beam search 还要处理不同前缀 cache 的复制、索引与潜在共享，下面按 B=1、单序列估算。

**base 场景 / 推导**：6 层、T = 4096、8 KV heads、64 维、FP16 得 self cache **48 MiB**；源长 512 的 cross cache **6 MiB**。它们不包括 encoder 输出、权重、allocator、工作区与 activation。8 头、T = 4096、FP16 的单层物化 self attention 矩阵 A 为 **256 MiB**。Cache 和 A 是不同数据结构：前者线性随 T 增长，后者在 prefill 可二次增长。

<span id="cache-lab"></span>
[[INTERACTIVE:cache]]

**后续技术边界 [R7–R10]**：MQA/GQA 减少 h_KV，有利于 cache 容量和 decode 的读取带宽；MLA 压缩 KV 的表示，公式需要按其实际 latent 结构重建。FlashAttention 分块、维护在线 softmax，以减少大 attention 矩阵的 HBM 写回；它保持 exact attention 的数学结果（浮点舍入允许差异），不消除二次混合算术，也不消除 token 生成的串行依赖。交互里的 KV heads 调整是后续技术的容量示意，不是 2017 模型的一项消融。

### 一层 encoder 的 per-op 成本

[[EQ:flops]]

约定乘法 + 加法算 **2 FLOPs**，B=1，h·d_k=h·d_v=D。投影含 Q/K/V/O 四个线性层；mixing 含 QKᵀ 和 AV；FFN 含 W₁/W₂。n = 512、D = 512、d_ff = 2048 时：**1.074G + 0.537G + 2.147G = 3.758 GFLOPs**。

mixing / projection = n/(2D)，mixing / FFN = n/d_ff；base 的两处交点分别为 n=1024、n=2048。长序列的注意力二次项最终占主导，短序列 FFN 和投影同样值得关注。这个估算不是 decoder 全层成本：decoder 还包括 cross-attention，完整训练还有 backward，完整生成还有词表投影、softmax、LN、采样等。

**TTFT / TPOT 分析**：翻译式 encoder–decoder 的 TTFT 至少包括 encoder、首个 decoder step、调度与数据准备；TPOT 受逐步 decoder、权重/KV 读取、词表头和输出策略影响。小 batch 的 decode 常受带宽与 kernel launch 限制，不能由训练 FLOPs 推出毫秒耗时。若关注 NPU/HTP，还要测动态 shape、transpose、softmax/LN 的算子支持、设备 fallback 与同步链路。

**量化边界**：把 b 从 2 改成 1 只代表理想 KV 存储大小减半。实际 INT8/INT4 带有 scale、zero point、group alignment 与解量化成本，可能改变 attention logit 精度。本文没有提供量化质量或吞吐实测；应在转换/导出环节检查 QK 的数值差异、mask 与 softmax，再用真实翻译/OCR等任务评测。

### 复现伪代码与官方代码核对

```python
# Paper-style structure; x:[B,n,D]. Helpers include masks and stable softmax.
def encoder_layer(x, source_padding):
    x = layer_norm(x + dropout(mha(x, x, x, source_padding)))
    return layer_norm(x + dropout(relu(x @ W1 + b1) @ W2 + b2))

def decoder_layer(y, enc, causal_and_target_padding, source_padding):
    y = layer_norm(y + dropout(mha(y, y, y, causal_and_target_padding)))
    y = layer_norm(y + dropout(mha(y, enc, enc, source_padding)))
    return layer_norm(y + dropout(relu(y @ W1 + b1) @ W2 + b2))

# labels:[B,n_t], excluding padding in loss
decoder_ids = shift_right(labels)  # BOS + labels[:-1]
logits = decode(decoder_ids, encode(source_ids)) @ embedding.T
loss = smoothed_cross_entropy(logits, labels)
```

**输入 / 输出**：encoder 接收源隐藏表示并输出同 shape；decoder 接收右移目标表示和 encoder memory；最终 logits 按词表轴归一化并对有效 label 计算 loss。`mha` 内部必须进行独立投影、split heads、按 key 轴 softmax 与合并头。上面省略初始化、embedding 缩放、PE、cache、训练态切换和 bias 细节，不能直接当成完整可运行训练脚本。

**外部来源，官方代码 [R14]，2026-10-27 检索**：当前 Tensor2Tensor `transformer_base()` 调用 v3，v3 又从 v2 构建；v2 把 `layer_preprocess_sequence` 设为 `"n"`、`layer_postprocess_sequence` 设为 `"da"`，且 warmup **8000**，attention / ReLU dropout 各 **0.1**。这已是演进的 **pre-LN** 配置，不能直接声称 `master + transformer_base` 精确复现论文的 post-LN / 4000 warmup。

官方 attention 文件能核对 split/combine heads、dot-product attention 与 lower-triangle bias 等操作。复现要固定 commit、hyperparameter set、词表、预处理与 BLEU 签名；仅链接 “官方仓库” 不足以固定历史实验。阅读源码时尤其要跟踪实际 preset 的调用链。

## Related Work & Technology Evolution

相关精读：[Sparsely-Gated MoE：容量可以稀疏激活，成本仍在整条链路](../sparse-moe/)。2017 年的原始 MoE 放在 LSTM 中；后续与 Transformer 结合时，专家容量、路由通信与 attention / KV 成本需要分别计算。

相关精读：[EAGLE-2：把验证预算分给更可能走通的路径](../eagle2/)。多候选并行验证仍依赖因果 attention、按树深度确定的位置，以及只提交被接受路径的 KV；一维展平不能让兄弟分支互相可见。

[[DIAGRAM:evolution]]

以下是外部研究与本文分析。时间线分成架构、位置、KV 与 IO 多条路线；它们解决不同问题，没有形成一个每一步都“取代上一代”的单一排行。

| 工作 / 年份 / 来源 | 核心想法 | 优势 | 限制 | 与本论文的关系 |
|---|---|---|---|---|
| Bahdanau attention · 2014 [R2] | decoder 动态对齐源状态 | 解除固定向量读取瓶颈 | RNN 仍串行 | attention 的前史 |
| ConvS2S · 2017 [R3] | 卷积序列转导 + attention | 训练位置可并行 | 全局联系需多层 | 同期替代路线 |
| Transformer · 2017 [R1] | self / cross attention + FFN | 内容通信、短路径、训练并行 | 全局混合二次；生成串行 | 本文骨架 |
| BERT · 2018 [R4] | 双向 Transformer encoder + 预训练 | 上下文表征 | 不直接是自回归生成器 | encoder 路线 |
| T5 · 2019 [R5] | text-to-text encoder–decoder | 统一多任务接口 | 编码器和解码器成本 | 保留原型两栈 |
| GPT-3 · 2020 [R6] | 大规模 autoregressive decoder | 通用生成、in-context learning | 数据/计算大；逐步生成 | decoder 路线 |
| MQA · 2019 [R7] | 多 Q heads 共享一组 K/V | 更小 cache / 更少读取 | 质量与共享取舍 | 改变 KV 结构 |
| RoFormer · 2021 [R8] | 在 Q/K 上旋转编码位置 | 相对位置点积结构 | 长度外推仍需评测 | 修改 additive PE |
| FlashAttention · 2022 [R9] | IO-aware tiled exact attention | 少物化、减少 HBM IO | 不改二次算术 | 改内核与存储 |
| GQA · 2023 [R10] | 多组 query 共享 KV | MHA 与 MQA 的折中 | 仍需模型训练/转换评测 | KV head 维度演进 |
| Mamba-2 · 2024 [R11] | structured state-space duality | 长序列高效路线 | 不等价于全局 softmax attention | 替代/混合架构方向 |
| DeepSeek-V3 · 2024 [R15] | MLA + MoE 等训练系统 | 压缩 KV、稀疏参数使用 | 实现与分布式复杂 | 组合多条优化路线 |

**与站内知识连接**：[RoFormer](../roformer/) 展开 additive PE 与 rotary 的区别；[VRoPE](../vrope/) 展开多模态位置轴；[MQA](../mqa/) 展开共享 KV、增量解码与带宽成本。Transformer 本篇提供这些后续工作的 attention、shape 和因果训练基础。

[GLU Variants 精读](../glu-variants/) 接着研究 FFN：两路投影相乘、2/3 隐层宽度的预算匹配，以及 GEGLU / SwiGLU 的实验与执行成本。

## Current State：截至 2026-10-27 怎样判断“最好”？

**外部来源 [R16–R17]，本轮检索。** Transformer 是架构族，无法为整族定义一个全球 SOTA 分数。为了与原论文任务关联，这里查看最新 **WMT26 General MT 官方 Findings**，同时保留模型规模、系统形态和评测口径。

**官方跨语言结果，§6.4**：Wayfinder 在与全部系统比较的 **23 个语言方向中 17 个**进入最高统计组，Gemini 3.1 Pro **15/23**，TRIVE **14/23**，Lumen **13/23**。只比较 constrained submissions 时，Lumen 为 **20/23**；该受限赛道要求总参数低于 **20B** 且满足许可条件，Appendix B.13 说明其 Qwen3 **14B** backbone，最终权重发布。这些计数是“在多少方向进入并列最高组”，不是单一准确率。

| WMT26 英语→德语，官方 §6.2 | Human cESA | 评估文档数 | 排名区间 |
|---|---:|---:|---|
| Wayfinder | 79.1 | 198 | 1–4 |
| TRIVE | 78.5 | 198 | 1–4 |
| Gemini 3.1 Pro | 78.2 | 198 | 1–4 |
| GPT-5.5 | 77.1 | 131 | 1–4 |

**如何解读**：评测动态分配导致样本数不同，排名区间相同意味着不能根据这几项小数宣称唯一显著冠军。Wayfinder 使用 API ensemble / agent pipeline，官方没有披露总参数量；它不是一个可与原版 65M/213M 直接并列的单模型。托管 API 与最终权重发布也要区分，不能把系统可调用当作模型开源。

**一个口径陷阱**：North Small Translate [R17] 的论文 Table 1 / 4 报 **218B 总参数 / 25B 激活参数**，自报 WMT26 **GEMBA 83.6**、agentic **84.4**。这是 LLM judge 口径；不是上表官方 human cESA，也不是“翻译准确率 83.6%”。没有同一评测集、同一计分器和同一系统限制，就不能把数字拼成一个榜。

**与 2017 的可比性**：WMT14 的 newstest2014 / BLEU 与 WMT26 的文档级多领域数据 / human cESA、系统规模、数据预算、指令和 agent 流程均不同。当前证据能说明翻译系统已扩展到大模型与组合系统，不能计算“从 28.4 到 79.1 提升了多少 BLEU”。SOTA 段落属于有日期的补充研究，不改变原论文的历史结论。

## Critique：贡献、代价与下一步实验

**创新性与正确性**：attention 本身已有前史，创新是构造完整无循环、无卷积的序列转导骨架并给出强实验；公式、shape 和信息流清楚。位置外推与可解释性主要是动机/定性观察，原文个别结果表述还存在冲突与过度概括。

**实验质量与泛化**：两个翻译任务、一个结构预测任务，以及 head、宽度、FFN、dropout、label smoothing、PE 消融，足以建立可行性。基线来自不同文献，缺少多 seed、统一预算和部分模块的独立对照；小数据解析表现也并非绝对最高。

**工程复杂度与复现**：结构比很多循环模型规整，易用大矩阵并行；mask、右移、归一化顺序、词表共享和 decode cache 却都可能产生静默错误。如今官方默认 preset 已变，必须固定历史配置。

**扩展性、计算与内存**：序列内信息路径短、架构便于加宽加深；长上下文的全局混合、attention activation、decoder cache 与词表头成为不同瓶颈。头数改变可能不改主 FLOPs，却改存储与 kernel 效率。

**硬件依赖**：论文的效率优势建立在 P100 与训练任务上。端侧的算子支持、动态长度、数据布局、精度和 CPU/device 边界会决定真正耗时。成本公式与能力提升必须结合实际 trace，不应自动迁移到 HTP/NPU。

**隐藏取舍**：并行训练依赖已知目标前缀；生成仍逐步。Label smoothing 改善 BLEU 同时恶化 PPL。多头增加表示能力，却可能增加小矩阵和存储压力。Sinusoidal 无需学习位置表，却不能自动保证长度外推。

**未答问题与可行改进**：同训练预算下哪种 head/FFN 配比最优？长句与罕见词是否同样受益？不同 heads 被屏蔽后质量怎样？位置外推究竟在哪些长度失效？建议补同预算、多 seed、长度分桶、干预、同设备的 prefill/decode trace 与真实峰值内存实验；MQA/GQA、RoPE、FlashAttention 是不同层面的后续改进，需分别验证效果。

## Takeaways、阅读审校与可下载证据

**我的理解**：把 Transformer 记成三件相互配合的事：**attention 搬运信息，FFN 加工信息，位置和 mask 限定信息关系**。再把训练、prefill 和 decode 拆开，就能从公式追踪到实际算子、cache 与任务指标。原版结构是分析后续模型的一把标尺，不能代替后续模型的实际配置与性能证据。

两轮审校分别检查“读者能否按 Q/K/V、右移和表格复算主结论”与“逐项回到 PDF/官方源核对数字和限定条件”。确认全部 15 页、5 图、4 表；保留 41.8/41.0 冲突，修正解析表 claim 范围，区分 dev/test 和平均 checkpoint，标记 FLOPs/KV 推导与当前评测口径。交互模型还独立验证已知矩阵结果、因果第一行、稳定 softmax、MiB 换算和 per-op FLOPs。

[研究与审校记录](assets/evidence/research-audit.md) · [来源检索清单](assets/evidence/manifest.json) · [图表完整性清单](assets/evidence/inventory.json) · [交互模型校验](assets/evidence/model-checks.json) · [Markdown 讲解源稿](assets/evidence/reading-source.md) · [BibTeX](assets/evidence/paper.bib)

**更新记录**：2026-10-27，首次精读发布，按 v7 逐页核对并补充 WMT26 官方评测与工程推导。

2026-10-28，重新裁切全部 5 张原图与 4 张原表截图，移除图片中的图题、表题及说明；原文 caption 和中文解读以图外文字呈现，放大查看也显示可选择的说明。保留全部面板、连线、token 标签、表头、单元格及表格边线。

相关精读：[Scaling Laws：规模、训练与预算分配](../scaling-laws/)。在 dense Transformer 数据流之上，进一步区分参数量、唯一数据、处理 token、训练 FLOPs 与临界 batch；平均 loss 的规模规律仍需与推理延迟和内存账本分别评估。

**Related Reading：**[RMSNorm：保留尺度约束，拿掉均值中心化](../rmsnorm/)。原始 Transformer 的 LayerNorm 同时中心化与缩放；这篇精读拆开验证两种作用，并审计归约、融合和端到端耗时。

## References：原论文、后续论文与官方资料

- **[R1]** Vaswani et al. *Attention Is All You Need*, NIPS 2017，1706.03762v7。[PDF](assets/original/1706.03762v7.pdf)，https://arxiv.org/abs/1706.03762 。全篇主证据；图 1–5、表 1–4、式 (1)–(3)。
- **[R2]** Bahdanau, Cho, Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate*, 2014 / ICLR 2015。https://arxiv.org/abs/1409.0473
- **[R3]** Gehring et al. *Convolutional Sequence to Sequence Learning*, 2017。https://arxiv.org/abs/1705.03122
- **[R4]** Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2018 / NAACL 2019。https://arxiv.org/abs/1810.04805
- **[R5]** Raffel et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, 2019 / JMLR 2020。https://arxiv.org/abs/1910.10683
- **[R6]** Brown et al. *Language Models are Few-Shot Learners*, 2020。https://arxiv.org/abs/2005.14165
- **[R7]** Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need*, 2019。https://arxiv.org/abs/1911.02150
- **[R8]** Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021。https://arxiv.org/abs/2104.09864
- **[R9]** Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022。https://arxiv.org/abs/2205.14135
- **[R10]** Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023。https://arxiv.org/abs/2305.13245
- **[R11]** Dao, Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*, 2024。https://arxiv.org/abs/2405.21060
- **[R12]** Jain, Wallace. *Attention is not Explanation*, NAACL 2019。https://aclanthology.org/N19-1357/
- **[R13]** Post. *A Call for Clarity in Reporting BLEU Scores*, WMT 2018。https://aclanthology.org/W18-6319/
- **[R14]** TensorFlow / Tensor2Tensor 官方源码，检索于 2026-10-27：[模型与配置](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)，[Attention 实现](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py)。快照哈希见 manifest。
- **[R15]** DeepSeek-AI. *DeepSeek-V3 Technical Report*, 2024。https://arxiv.org/abs/2412.19437
- **[R16]** *Findings of the WMT 2026 General Machine Translation Shared Task*，§6.2 / §6.4 与 Appendix B.13。https://www2.statmt.org/wmt26/pdf/2026.wmt-1.48.pdf
- **[R17]** *North Small Translate* 论文，2026，2609.13916v1，Table 1 / 4。https://arxiv.org/html/2609.13916v1

上述扩展来源在本轮研究中实际检索，URL、状态、日期与内容哈希见清单。对原论文所有引用的完整书目，请见 PDF pp.10–12 的 References；本文只列对讲解与扩展调研直接使用的来源。

相关阅读：[Layer Normalization 精读](../layer-normalization/)，继续核对归一化轴、Pre/Post-LN 的区别、RNN 分支与推理 IO 成本。
