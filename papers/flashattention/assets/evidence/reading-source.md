# FlashAttention：算得更多，为什么反而更快？

**一次改变 Attention 执行方式的工作：保留 dense softmax 的数学结果，把二次规模的中间矩阵留在芯片上。**

## Paper Metadata

| 项目 | 核验信息 |
|---|---|
| 原标题 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness |
| 作者 | Tri Dao · Daniel Y. Fu · Stefano Ermon · Atri Rudra · Christopher Ré |
| 机构 | Stanford University；University at Buffalo, SUNY |
| 发表 | NeurIPS 2022 |
| 版本 | arXiv v1：2022-05-27；本次精读 v2：2022-06-23 提交，PDF 封面标注 June 24, 2022 |
| arXiv / DOI | [2205.14135](https://arxiv.org/abs/2205.14135) · [10.48550/arXiv.2205.14135](https://doi.org/10.48550/arXiv.2205.14135) |
| 论文 | [本地保留的完整 v2 PDF](assets/original/2205.14135v2.pdf) · [arXiv PDF](https://arxiv.org/pdf/2205.14135) |
| 官方代码 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)；当前 main 已包含后续代际，不能视为 2022 版本快照 |
| 作者项目资料 | [Tri Dao：FlashAttention-4 技术博客](https://tridao.me/blog/2026/flash4/)；作为后续资料，不冒充原论文项目页 |
| 阅读 / 更新 | 2026-10-27；原文 34 页，含附录 A–E、8 张 Figure、21 张 Table、Algorithms 0–5、6 个编号公式 |

正文先按原论文章节顺序展开，再单独给出扩展调研与个人判断。**Paper fact** 来自本地 v2；**External source** 指向后续一手资料；**Interpretation** 是本文重构、计算或审计结果。原图和原表完整保留，教学图明确标记为 Agent-created Visualization。

## TL;DR：从一句话走到五个关键点

**一句话。** FlashAttention 通过分块融合和 online softmax 避免把完整 Attention 矩阵写入 HBM，并在反向阶段重计算概率，以更少的数据搬运完成同一个 dense Attention。

**一段话。** 常规 Attention 把 `QKᵀ → mask → softmax → dropout → PV` 拆成多次操作，巨大的 `N×N` 中间矩阵反复落到显存。FlashAttention 发现这些中间结果并不是最终输出：只要维护每一行的最大值、归一化和输出累积量，就能逐块处理全部 Key，得到全局 softmax 的结果。反向传播也只需少量前向统计量，通过重新生成小块概率计算梯度。它减少 HBM 流量与激活存储，而不是减少所有 Query–Key 配对。论文由此获得 kernel 加速和端到端训练加速，并让部分长序列任务变得可训练。

1. **瓶颈是执行链路。** FLOPs 不能独立决定时间；HBM 读写、launch、同步和低效逐元素操作都可能主导。
2. **数学关键是可合并的 softmax 状态。** 两个块的分母不能各算各的；要在共同最大值下重标定，再合并分子、分母。
3. **Memory 与 IO 是两个问题。** 避免保存 `N²` 激活带来线性存储；理论 IO 项 `N²d²/M` 在固定硬件上仍随 `N²` 增长。
4. **重计算可以是优化。** Figure 2 中算术量增加约 12.9%，HBM 流量减少约 9.16 倍，时间减少约 5.71 倍，三者来自同一图中的特定设置。
5. **结果要分层读。** 7.6× 是特定 Attention 前向 kernel；BERT 的 20.0→17.4 分钟是 13% 时间下降；长上下文的精度提升并不等于单独替换 kernel 的因果效果。

## 1. Introduction：为什么更少 FLOPs 的方法未必更快？

### 1.1 原论文的出发点

Transformer 的序列长度增长会让 Attention 出现二次规模的计算与存储。2022 年的主要替代路线通常修改 Attention：稀疏化、低秩近似、核函数线性化。作者观察到，这些方法理论复杂度更低，却不一定获得对应的实际加速。一部分原因是忽视 GPU 的分层存储，把多个低算术强度操作变成了频繁的 HBM 访问。

作者提出的目标更具体：**在不近似 dense Attention 的条件下，减少 HBM 与片上 SRAM 之间的数据交换。** 因此这是一篇算法与系统协同设计论文。它不引入新的训练目标，不增加模型参数，也不要求更换语言建模 loss。

### Figure 1 — 从硬件层次到分块执行

{{fig:1}}

**这张图解决什么问题？** 把“Attention 很慢”拆成具体的数据路径。先看左侧 GPU memory hierarchy，再看中间分块算法，最后看右侧速度比较。

**Components。** 左侧 HBM 大但相对慢，SRAM 小但带宽高。图中的 A100 约 20 MB SRAM 是多个 SM 的合计；每个 SM 约 192 KB，不能把总量当成单个 kernel 任意分配的共享池。中间虚线大方框代表没有被实体化到 HBM 的 `N×N` 矩阵。红箭头遍历 K/V 块，蓝箭头遍历 Q 块；输出与行统计量在块之间更新。

**Data flow。** 将 K/V tile 搬到片上，加载一块 Q 与当前输出统计，在 SRAM 中计算 score、softmax 更新和 `PV`，将更新后的输出写回 HBM。这里展示的是 v1 的循环次序，不能用 FA2 的实现反向替换它。

**Key observation。** 右侧最高 7.6× 的前向加速，是指定 GPT-2 Attention 配置下相对 PyTorch 的结果。它不表示整模型、反向、所有长度、所有 GPU 都有同样加速。

**Interpretation。** 读这张图时应追问“哪个张量必须落显存”。FlashAttention 改变的是中间量的生命周期：完整 `S` 与 `P` 只在数学表达里存在，小块在片上生成、使用、丢弃。

### 1.2 贡献与适用边界

原论文贡献包括 dense exact Attention 的 IO-aware 算法、HBM 访问量分析、block-sparse 扩展，以及 BERT/GPT-2/LRA/长文档/路径任务实验。三层结果要分开：

| 层次 | 作者要证明什么 | 证据在哪里 | 能推出的结论 |
|---|---|---|---|
| 算子 | 同一个 dense Attention 可以减少 IO | Algorithm 1；Proposition 1–3；Figures 1–3 | 融合与重计算能改善特定执行链路 |
| 整模型 | kernel 优化转化为训练加速 | Tables 1–4 | 在指定模型、GPU、实现与设置下时间下降 |
| 任务能力 | 更长序列能提高部分任务成绩 | Tables 5–6 | 更低内存成本开放了新的训练配置 |

**Caveat。** Block-sparse 版本对所选 sparse mask 是 exact，对完整 dense Attention 则属于改变算子的近似路线。标题中的 exact 主要指 dense 算法的数学等价性；浮点归约顺序改变，不能保证逐 bit 一致。

## 2. Background：沿着一次 Attention 调用重构问题

### 2.1 符号、shape 与目标

先按一个 batch、一个 head 分析。原文令 `Q,K,V ∈ ℝ^(N×d)`，输出 `O ∈ ℝ^(N×d)`；`N` 是序列长度，`d` 是 head dimension。完整实现通常有 `B` 个 batch 和 `H` 个 head，逻辑 shape 为 `[B,H,N,d]`，布局与 stride 另行处理。

{{core:attention}}

`S=QKᵀ` 与 `P=softmax(S)` 都是 `[N,N]`，softmax 沿 Key 轴逐行归一化，`O=PV` 是 `[N,d]`。原文基础表达省略缩放；实际 scaled dot-product Attention 应用 `τ=1/√d`，还可能加 bias、causal/padding mask 与 dropout。附录 B.4 给出包含这些扩展的完整算法。

每行的含义是：Query `qᵢ` 与所有 Key 计算相关性，将分数变成总和为 1 的权重，再加权 Value。输入投影、输出投影、LayerNorm、MLP 与残差不在这个核心 kernel 中。讨论内存或速度时必须明确是否包含这些外围操作。

### 2.2 常规实现的逐操作成本

| 操作 | 输入 → 输出 shape | 主要计算 | HBM / 生命周期问题 |
|---|---|---|---|
| `QKᵀ` | `[N,d]×[d,N]→[N,N]` | 约 `2N²d` FLOPs | Tensor Core 友好，但完整 score 被写回 |
| scale / mask | `[N,N]→[N,N]` | `Θ(N²)` | 低算术强度；融合不好时再读写一次 |
| softmax | `[N,N]→[N,N]` | max、exp、sum、divide | 多次行归约，读 score、写 probability |
| dropout | `[N,N]→[N,N]` | RNG、筛选、缩放 | 训练期可能保存或重放随机 mask |
| `PV` | `[N,N]×[N,d]→[N,d]` | 约 `2N²d` FLOPs | 再读大概率矩阵 |
| backward | `dO,Q,K,V,P` | 多个 GEMM、归约 | 通常要读保存的 `P`、生成 `dP/dS` |

这些行描述逻辑操作，并不假定每个库都产生相同数量的显存读写。已有融合 softmax 可以减少其中一些往返；FlashAttention 的新意在于跨越 GEMM 与非 GEMM 的边界，一起控制 `S/P` 的生命周期。

{{algorithm:0}}

原 Algorithm 0 完整列出标准执行路径，作为后面的 memory-efficient 方案参照。

**Interpretation：计算量与 roofline。** 单个 matmul 常能利用 Tensor Core；softmax、mask、dropout 更容易受带宽或标量吞吐限制。一条链路的总耗时还包括 launch 和同步，所以只对照 `4N²d` 的 FLOPs 会漏掉瓶颈。端到端收益又受 Amdahl 定律限制：如果 Attention 原来占比为 `f`，算子加速为 `s`，其他部分不变时总加速约为 `1/[(1−f)+f/s]`。

### 2.3 Kernel fusion 与存储层次

HBM 用于放整个模型、输入输出与大张量；SRAM / shared memory 和寄存器用于当前计算块。论文将容量抽象成 `M` 个 scalar words，而不是固定字节数。现实分配还要考虑 dtype、FP32 累积量、线程寄存器、共享内存布局、对齐、bank conflict 和占用率。

Fusion 的价值是让生产者的输出直接被片上的消费者读取。但 softmax 的分母依赖整行所有 Key：为什么只拿一块还能给出全局正确的权重？这正是下一节的数学难点。

### 2.4 Approximate Attention 的参照意义

低秩 Linformer、随机特征 Performer、局部和其他 sparse 方法减少配对或近似概率，是另一条技术路线。它们的误差、任务适配与计算量不同。原论文把它们放进速度比较，说明 IO-aware dense Attention 可以具有竞争力；这并不使它们与 dense exact 方法成为完全同一问题上的公平竞赛。

## 3. FlashAttention：从可合并状态到可执行 kernel

### 3.1 Tiling 与 online softmax：必须保留什么？

对一个 score 向量 `x`，稳定 softmax 使用 `m=max(x)`，`ℓ=Σ exp(x−m)`。将它分成两个块 `a,b`，分别获得 `(mₐ,ℓₐ)` 与 `(mᵦ,ℓᵦ)`。合并时取共同的 `m=max(mₐ,mᵦ)`，把两个分母都重标定：

{{core:merge}}

这不是近似。`exp(x−mₐ) exp(mₐ−m)=exp(x−m)`，所以新分母就是全体元素在相同基准下的指数和。为了计算 `PV`，还要合并加权分子 `u=Σ exp(x−m)v`；最终 `O=u/ℓ`。

**关键区别。** 各块独立 softmax 后直接平均，会把每块强制分配相同质量，丢掉块与块的相对 score 尺度。正确状态必须包含全局归一化信息。Max shift 不改变概率，但保证指数不因大正分数溢出。

### 3.2 按原 Algorithm 1 重构一次更新

设 Q tile 有 `Bᵣ` 行，K/V tile 有 `B꜀` 行。shape 为：

| 片上量 | Shape | 用途 |
|---|---|---|
| `Qᵢ` | `[Bᵣ,d]` | 当前 Query 块 |
| `Kⱼ,Vⱼ` | 各 `[B꜀,d]` | 当前 Key / Value 块 |
| `Sᵢⱼ=QᵢKⱼᵀ` | `[Bᵣ,B꜀]` | 局部 score |
| `mᵢ,ℓᵢ` | 各 `[Bᵣ]` | 已处理 Key 的逐行统计 |
| `Oᵢ` | `[Bᵣ,d]` | 原 v1 保存的归一化输出 |
| `P̃ᵢⱼ` | `[Bᵣ,B꜀]` | 以局部最大值为基准的未归一化指数 |

块内先算 `m̃=rowmax(Sᵢⱼ)`，`P̃=exp(Sᵢⱼ−m̃)`，`ℓ̃=rowsum(P̃)`。然后取 `m′=max(mᵢ,m̃)`，`α=exp(mᵢ−m′)`，`β=exp(m̃−m′)`，更新 `ℓ′=αℓᵢ+βℓ̃`。由于旧 `Oᵢ` 已被旧分母归一化，必须乘回 `ℓᵢ`：

{{core:output}}

除法与乘法按行广播到 `[Bᵣ,d]`。一个特别容易抄错的点是把 `αOᵢ` 当成旧分子；这少了 `ℓᵢ`，会从第二个块开始产生错误。现代实现可能保留未归一化的累积输出，直到最后才除分母；那是等价的状态选择，但应与本篇 v1 公式区分。

### 教学实验 — 一行 softmax 如何跨块合并？

{{interactive:softmax}}

**Purpose。** 观察每次新块到来时，旧输出为什么需要重新缩放。控制块宽、分数偏移与 mask；单步或播放遍历，比较 online 输出与独立 dense 计算。固定 Value 是教学数据，不是模型 attention map，也不是性能测试。

默认例子 `scores=[1,2,3,4]`、`values=[10,20,30,40]`。第一块 `[1,2]` 给出输出约 17.310586；第二块 `[3,4]` 的独立输出约 37.310586。错误地取二者平均得到 27.310586；全局正确结果约 **34.926527**。合并在最大值 4 下重标定第一块，保留第二块更大的概率质量，因此得到 dense 结果。

**边界处理。** 如果整行被 mask，score 为 `−∞`，不能直接计算 `−∞−(−∞)`。教学实现把空块表示为零质量状态，全空行返回零输出并显式标记；实际 kernel 应遵循框架关于全 mask 行的契约。初始化 `m=−∞,ℓ=0,O=0` 同样需要避免 `0×NaN`。

### 3.3 Loop order、读写与计算路径

{{diagram:flow}}

原 Algorithm 1 在外层遍历 K/V 块：先把一个 K/V tile 放进 SRAM，然后内层遍历全部 Q 块。每次内层加载 `Qᵢ,Oᵢ,mᵢ,ℓᵢ`，片上算 score、归约、指数和输出，写回更新后的 O 与统计量。Q/O 会随 K/V 外循环多次读写；这是论文 IO 分析的来源之一。不能解释成“所有输入只读一次”。

原文理论选择 `B꜀=⌈M/(4d)⌉`、`Bᵣ=min(⌈M/(4d)⌉,d)`。它用于证明块能装进片上及计算 IO 的量级；不是可以不看 GPU 型号、dtype 和寄存器就直接照抄的最优 CUDA 配置。现实 head dimension 与 tile 宽度经常受到 Tensor Core 对齐和 occupancy 约束。

```python
# Agent reconstruction of v1 forward; one head, no dropout.
# Q/K/V: [N,d], O: [N,d], m/l: [N].
O, l, m = zeros(N, d), zeros(N), full(N, -inf)
for K_j, V_j in key_value_tiles:
    load_to_sram(K_j, V_j)
    for Q_i in query_tiles:
        S = scale * (Q_i @ K_j.T) + bias_and_mask
        mt = rowmax(S)
        Pt = exp_with_empty_row_guard(S - mt)
        lt = rowsum(Pt)
        mn = maximum(m_i, mt)
        alpha, beta = safe_rescale(m_i, mt, mn)
        ln = alpha * l_i + beta * lt
        numerator = alpha[:, None] * l_i[:, None] * O_i
        numerator += beta[:, None] * (Pt @ V_j)
        O_i = divide_or_zero(numerator, ln[:, None])
        write_back(O_i, mn, ln)
```

**逐操作解读。** `QKᵀ` 和 `P̃V` 仍然做全量 dense 配对；mask、max、exp、sum 与重缩放在 tile 上执行。主要节约来自 score/probability 不落 HBM，代价包括额外的统计更新、重缩放，以及对 Q/O 的多轮访问。这个伪代码帮助理解数据流，不宣称是可直接达到论文速度的实现。

### 3.4 Backward：把保存概率改为重生成概率

前向结束保留 Q/K/V、O 和逐行归一化统计。反向重新计算当前 `QᵢKⱼᵀ`，用全局统计恢复当前块的真实概率；利用 `dO` 计算 `dV,dP,dS,dQ,dK`，而不用从 HBM 读取完整 P。附录 B 将推导完整展开，见后面的六个原编号公式。

{{diagram:backward}}

这里有两种重计算，不能混为一谈：常规 activation checkpointing 可以重新跑一段网络；FlashAttention 在 Attention 的片上执行路径内重生成一个局部 P tile，立刻用于梯度。前者可能重新发生大量 HBM 往返，后者主动避免它们。

**为什么增加 FLOPs 仍更快？** 若从显存取 P 的时间比在 Tensor Core 上重新算 score 更贵，重算就值得；减少中间矩阵读写也让融合链路更完整。这个判断依赖硬件与负载，不是“所有重计算都更快”的定理。

### Figure 2 — 66.6→75.2 GFLOPs，41.7→7.3 ms

{{fig:2}}

先看左侧统计，再看右侧曲线。左侧在给定 GPT-2 medium 设置下比较标准实现与 FlashAttention：batch 64、sequence 1024、head 64，A100，forward + backward。FLOPs 为 66.6 与 75.2 GFLOPs，HBM 读写为 40.3 与 4.4 GB，时间为 41.7 与 7.3 ms。

**重算结果。** `(75.2−66.6)/66.6≈12.9%` 更多算术；`40.3/4.4≈9.16×` 更少 HBM 流量；`41.7/7.3≈5.71×` 加速。对应时间下降约 82.5%。这些是单图中同一对结果；不要把 Figure 1 的 7.6× 拼到这里。

中间图改变 block size，横轴是块宽而非 SRAM KB：块变大，扫 Q 的轮数减少，访问下降、前向时间下降；超过约 256 后其他瓶颈开始限制收益。右图比较随序列增长的 HBM 访问。关键意义是实际测量支持 IO 分析的解释方向。局限是这不是跨所有 GPU、shape 的因果消融，也不能用理论 IO 比值精确预测每个 kernel 的时间。

### 3.5 IO complexity：线性存储不等于线性计算

**Paper fact。** Proposition 1：dense 算法输出正确，计算量 `O(N²d)`，除输入输出外的统计量存储为 `O(N)`。如果把 Q/K/V/O 也算上，整体张量存储是 `O(Nd)`；原文固定 d 时常写成对 N 线性。`S/P` 的二次激活被消除了，但所有 score 仍要计算。

**Proposition 2。** 在 `d≤M≤Nd` 的容量范围内，标准 Attention 需要 `Θ(Nd+N²)` HBM accesses；FlashAttention 的主项为：

{{core:io}}

粗略推导是 K/V 外循环约 `Nd/M` 次，每轮内层扫 Q/O 的 `Θ(Nd)` 量，因此主项 `Θ(N²d²/M)`。完整算法还存在输入输出流量与常数项；论文给出的容量范围使主项覆盖相应规模。

**固定硬件审计。** 将 N 加倍而 M、d 不变，主项依然增加四倍。论文的“subquadratic HBM accesses”是在把 M 作为可变问题参数的算法表述下理解；固定 GPU 上不应把它写成低于二次的 N 复杂度。若 M 能随 N 扩大，IO 项才会体现相应的不同增长。

**Proposition 3 的强度。** 作者证明不存在一个算法，对范围内的每一个 M 都统一取得 `o(N²d²/M)` 的访问量。证明使用 `M=Θ(Nd)`，此时读输入已需 `Ω(Nd)`。它排除的是“对所有 M 统一的小 o 改善”，并未给每一个固定 M 都建立独立、紧的下界；不能扩大成任意容量下的唯一最优 kernel。

### 教学实验 — 激活、KV Cache 与 IO proxy

{{interactive:cost}}

**Purpose。** 调整序列、head dimension、batch/head、精度与假设可用 SRAM，分别观察二次矩阵、线性张量、持久 KV Cache 和理论 IO 主项。输出是由公式算出的容量与代理量，不是实测显存、延迟或 TFLOPs。

图中传统方案的单个 `S` 或 `P` 是 `BHN²×bytes`；FlashAttention 的 Q/K/V/O 合计近似 `4BHN d×bytes`，还需行统计和 workspace。二者不是完整训练峰值的公平相减，因为传统方案也包含 Q/K/V/O；比较意图是定位二次中间量。实际 Figure 3/Table 21 才给出论文测量。

KV Cache 用另一条公式 `2×B×L×N×Hkv×d×bytes`，包括 L 层的 K/V，不包含 Q/O。GQA 的 `Hkv` 可以少于 query head 数；教学面板为简单 MHA 令二者相等。FlashAttention 不改变这条持久存储公式。

### 3.6 Block-sparse FlashAttention：哪些 tile 可以跳过？

对块级 sparse mask，将不允许的 Q–K 块直接跳过，其余块继续使用相同的 online softmax。非零块占比 `s∈[0,1]` 时，Proposition 4 给出 `Θ(Nd+sN²d²/M)` IO。这里 `s` 表示非零比例，而不是“稀疏比例”。

mask 的数学语义是被排除 score 设为 `−∞`，权重为零；不是把 score 乘以 0，因为 `exp(0)=1` 仍会参与分母。causal/padding mask 保持指定 dense 算子的语义，额外 block sparsity 则改变 Attention pattern。前者不是论文意义上的近似模型，后者相对于全 dense pattern 是另一种算子。

**Trade-off。** Sparse 版本少算 tile，但会增加索引、非规则调度与块对齐成本；短序列下未必更快。论文附录自身就有反例，后面逐表说明。

## 4. Experiments：把 kernel、训练与任务收益分开审计

### 4.1 Faster model training：BERT 与 GPT-2

**实验问题。** Attention 的收益能否穿过其余模型操作，降低到达同一目标的训练时间？这里要看训练设置与评估目标，而不只看单个 forward。

### Table 1 — BERT-large 到达 MLPerf 目标的时间

{{table:1}}

三行完整比较 HuggingFace、NVIDIA MLPerf 和 FlashAttention。最快 baseline 是 NVIDIA MLPerf，20.0±1.5 分钟；FlashAttention 为 17.4±1.4 分钟。HF 的 32.9±2.6 分钟不是最强 baseline。

**重算。** 相对 NVIDIA：绝对少 2.6 分钟，时间减少 13.0%，速度提高约 14.94%；原文约 15% 加速可以由此理解。“15% faster”不应不加区分地写成“时间下降 15%”。

**Setup。** 8×A100 80 GB，按 MLPerf 指定 checkpoint 初始化，目标 validation accuracy 72%，最多 7100 steps，10 次运行。batch 448，LAMB，learning rate 3.75e−3，AMP O2 FP16；验证样本 10000。该实验测达到阈值的时间，不是固定训练 token 的吞吐。

**Fairness / caveat。** 与优化过的 MLPerf baseline 比比与 HF 更有说服力；方差区间不是统计显著性检验。不要把 BERT 的训练阶段写成从随机初始化完成所有预训练，更不要将 8 卡结果解释为单卡端侧性能。

### Table 2 — GPT-2：相近 perplexity，训练时间不同

{{table:2}}

GPT-2 small 在 1K context 下，HF / Megatron-LM / FlashAttention 的 OpenWebText perplexity 都为 18.2，训练时间分别为 9.5 / 4.7 / 2.7 天。对 HF 加速 3.52×，对 Megatron 加速 1.74×。GPT-2 medium 的 perplexity 为 14.2 / 14.3 / 14.3，时间为 21.0 / 11.5 / 6.9 天；对 HF 加速 3.04×，对 Megatron 加速 1.67×。

**How to read。** Perplexity 越低越好，时间越短越好。Dense kernel 的目标是保持模型结果，同时减少执行开销；medium 的 0.1 perplexity 差异不能被称为严格相同或精度提升。

**Setup。** 8×A100 40 GB，OpenWebText，sequence 1024，effective batch 512，400K steps，AdamW，weight decay 0.1；small/medium 的 LR 为 6e−4 / 1.5e−4，0.5% validation split，AMP。训练时间表中的 approximate day 与后文学习曲线提供同方向证据。

**Comparison。** 对不同库的结果包含其 Attention fusion 与整体训练栈差异；比较不应只归因于减少 launch 个数。论文没有逐 token 的 TTFT/TPOT，也没有端侧 QNN/HTP 测量。

### Table 3 — LRA：效率竞争不意味着同一个 Attention

{{table:3}}

完整表保留 ListOps、Text、Retrieval、Image、Pathfinder、Average、Speedup。标准 Transformer / FlashAttention / block-sparse 的 Average 为 59.3 / 59.8 / 59.6；后两者 speedup 为 2.4× / 2.8×，是跨五个任务的几何平均，不是所有单任务的同一加速倍数。

**Baselines。** 表中的 Linformer、Reformer、BigBird、Performer、Linear Attention、Local Attention 都有自己的近似或 pattern 选择。不同任务最佳行会改变，不存在用 Average 一列替代整个比较的理由。Dense FA 与标准 Transformer 的约 0.5 点 Average 差异是结果接近的证据，不应归因于 kernel 创造更好的函数。

**Audit。** 作者使用原论文、官方实现与复现结果中的最佳成绩；速度实验中部分库的 mixed precision 支持不一致。Local Attention 用 FP32，Performer 未使用 mixed precision。这些细节影响效率公平性，尤其不能把它们都描述成相同精度、相同优化水平。

### 4.2 Longer sequence length：效果提升来自哪一步？

### Table 4 — 更长 GPT-2 context：perplexity 与成本

{{table:4}}

GPT-2 small 的 FlashAttention 1K / 2K / 4K context 得到 perplexity 18.2 / 17.6 / 17.5，时间 2.7 / 3.0 / 3.6 天。最重要的参照是 Megatron 1K 的 18.2、4.7 天：FA 4K 用时减少约 23.4%，加速约 1.31×，同时 perplexity 低 0.7。

**Trade-off。** 在 FA 内部把 1K 扩到 4K，时间增加 0.9 天，即约 33.3%，换取更低 perplexity。不能只写“更长且更快”而省略参照对象。

**因果边界。** Kernel 本身不改变 Attention 数学函数，长度改变才改变可见上下文；但长度可能同时改变每步 token 数、训练总 token 与数据打包。文中提供 effective batch 与 steps，却不足以将此表提升为严格 iso-token、iso-compute 的长度因果实验。较长 context 是否在同一 token budget 下改善 perplexity，还需要单独控制。

### Table 5 — 长文档分类：micro-F1 不随长度单调上升

{{table:5}}

MIMIC-III 在 512→16384 tokens 下 micro-F1 从 52.8→57.1，绝对 +4.3 个百分点；相对约 +8.14%。ECtHR 在 512→8192 下从 72.2→80.7，绝对 +8.5 点；相对约 +11.77%，但 16384 时回落到 79.2。

摘要中的 “6.4 points” 是这两个提升 4.3 与 8.5 的平均值，不是每个数据集都提升 6.4，也不是百分之 6.4。最强长度配置不同，说明更长上下文的收益依赖任务与训练。

**Metric 审计。** 原文只报告聚合 micro-F1，没有 TP/FP/TN/FN。F1 的表达为 `2TP/(2TP+FP+FN)`；一个 F1 数字无法唯一恢复 confusion counts、accuracy、precision 或 recall。因此不能补造这些指标，也不能把 micro-F1 称为 accuracy。

**Setup / fairness。** 这是使用不同序列长度的文档任务训练结果，论文附录给出任务配置。长输入会改变截断、有效信息与计算成本；它证明低内存 kernel 有助于研究长文本任务，不单独证明一个算子替换直接提高分类能力。

### Table 6 — Path-X / Path-256：训练可行性与成绩

{{table:6}}

FlashAttention 在 16K 的 Path-X 达到 61.4%；block-sparse 为 56.0%，并在 64K 的 Path-256 达到 63.1%。表中的 × 表示 OOM 或未超随机，不是所有 baseline 都 OOM。任务是路径二分类，随机基线约为 50%。

**Evidence。** 作者提出第一个在 Path-X 和 Path-256 上有非随机表现的 Transformer。需要保留配置背景：先在较低分辨率 Path-64 训练 200 epochs，再在目标任务训练 200 epochs；Path-X 多训练 200 epochs 还能增益约 4 点。curriculum、训练时长与 Attention pattern 都可能贡献。

**Interpretation。** 更低存储使长序列的训练配置可行，这有实质价值；但表格不是仅换 kernel、其他训练完全不变的能力消融。Dense FA 与 sparse FA 的 Path-X 分差也不能简单归因于 IO，因为 pattern 已改变。

### 4.3 Benchmarking Attention：长度、pass 与 baseline 共同决定结果

### Figure 3 — 时间曲线与内存曲线是两个证据

{{fig:3}}

左图是 forward + backward 时间，右图是额外内存。先辨认图例中 exact 与 approximate 方法，再看交叉点和随 N 的增长，不要直接按曲线排名当作精度等价竞赛。

**Key observation。** Dense FA 的时间仍随长度明显增长，内存曲线却近似线性；这与算术二次、激活线性的分析一致。某些近似算法在部分长度时间更低；FA 的价值是 exact 结果与效率的组合。左图的 cross-over 指不同方法的速度排序会随 N 改变。

**Setup。** 微基准用随机 Q/K/V，batch 16、8 heads、head dimension 64，A100 40 GB；不包括 QKV/output projection。计时重复 100 次，分别测试 forward、backward、combined，dropout 为 0 或 0.1，mask 为 padding/causal/无。padding 长度在 `N−20` 到 N 之间随机选取。

**公平性边界。** 表中的不支持长度、实现 mask bug、mixed precision 差异都影响比较。combined 是独立测量，不必严格等于 separately measured forward + backward。不同库运行到的最大长度也不同，dash 不等于一律 OOM。

### 4.4 Ablation：这篇论文实际消融了什么？

Figure 2 直接支持增加重计算、减少 IO 的解释；Figures 5–8 比较长度、head dimension、mask、dropout 与 GPU；Tables 9–20 分解 pass 和配置。但缺少一个把 tiling、跨 GEMM fusion、online softmax、recompute、loop order 逐个隔离的完整组件消融。

因此可以说证据显示一整套 IO-aware 执行策略有效；不能说实验已单独量化每个组件贡献。Sparse 方案也不是只减少 IO 的消融，因为 Attention operator 改变。

## 5. Discussion and Limitations：作者承认了哪些问题？

**Paper fact。** GPU 优化实现需要工程投入；算法性能依赖硬件与实现；exact dense Attention 仍有二次计算量。作者将 IO-awareness 视为其他算子也可以考虑的设计原则，而不是对所有工作负载的性能保证。

**Interpretation。** 论文打开的是中间激活存储和数据搬运的瓶颈。对超长 N，dense 配对算术仍然昂贵；对 decode，query 很短，occupancy 与 KV 带宽变成另外的问题；对小 N，既有高度融合 kernel 的优势可能压过分块开销。

面向工程使用，应分别报告算法语义、工作负载、硬件、dtype、软件版本、mask/dropout 支持与测量范围。省略这些条件，“FlashAttention 更快、更省”就缺少可以复现的含义。

## Appendix A — Related Work：放回 2022 年的知识背景

原附录依次讨论硬件效率、模型效率、IO-aware 算法与 Attention 计算，不把后续 FA2–FA4 写回历史。

**Hardware efficiency。** Compiler、kernel fusion、GPU 上的高效矩阵乘法为系统路线提供基础。FlashAttention 的区别是依照 Attention 的数学结构设计状态和存储路径，跨越传统库调用的边界；不是仅让一个单独 GEMM 更快。

**Model efficiency。** Sparse Attention、low-rank 与 kernel feature approximation 主要改变所计算的配对或概率表达。原文以任务成绩与真实运行时间比较它们，但模型误差和库支持仍需单独考虑。

**IO-awareness。** 理论模型显式考虑两级存储之间的通信，不只算 arithmetic。图示中的 M 让算法能根据快速存储容量选择 tile，延续外存算法和数值线性代数里“计算与通信一起设计”的传统。

**Memory-efficient exact Attention。** Online normalizer、Rabe & Staats 等先行工作已揭示 softmax/Attention 可分块而不保存整个概率矩阵。论文不应被简化为“首次发现分块 softmax”；它的突出贡献是把数学重构、GPU kernel 与实测 IO/训练效果组合起来。

## Appendix B — Algorithms：完整保留并重推梯度

### B.1 Memory-efficient forward pass：先确定归一化轴

{{algorithm:1}}

这一节讨论 forward 的分块归约。Algorithm 1 保存归一化 O 与 `(m,ℓ)`，外 KV、内 Q 对应正文重构。按本文约定，`Pᵢⱼ` 是第 i 个 Query 对第 j 个 Key 的概率，所以 softmax 沿每一行的 Key 轴归一化。原附录个别文字把 rows/columns 混称；这里根据公式与 `PV` 的 shape 统一按行解释，不改变原算法展示。初始化和块更新按行广播，第一个空统计状态要安全处理。

### B.2 Memory-efficient backward pass：六个原编号公式

推导令 `sᵢⱼ=qᵢᵀkⱼ`，暂时省略 scale、mask、dropout，与原附录一致。`qᵢ,kⱼ,vⱼ,oᵢ,doᵢ` 都是 d 维向量，`Pᵢⱼ,dPᵢⱼ,Dᵢ,Lᵢ` 是 scalar；`dQ,dK,dV,dO` 都为 `[N,d]`。下面的公式直接保留原文 MathML 和 LaTeX。

#### Equation (1) — 概率与行分母

{{eq:1}}

`Lᵢ` 是未 shift 的指数和，不要与后续库常保存的 `LSEᵢ=logΣ exp(sᵢⱼ)` 混淆。原公式用于推导，数值实现必须使用 max shift；等价地 `Pᵢⱼ=exp(sᵢⱼ−mᵢ)/ℓᵢ`。归约读完一行 Key，分母是 scalar，概率 tile 可在 backward 临时恢复。

#### Equation (2) — 从概率到输出

{{eq:2}}

`oᵢ` 是所有 Value 的线性组合，输出不要求保留每一个权重。工程上 score/probability tile 的消费者就是 `PV`；计算后可以丢弃，而 O 在 HBM 中保留。这正是 forward fusion 的入口。

#### Equation (3) — Value 的梯度

{{eq:3}}

一个 `vⱼ` 被全部 Query 的输出使用，所以累积 `Σᵢ Pᵢⱼ doᵢ`。矩阵形式 `dV=PᵀdO`：`[N,N]×[N,d]→[N,d]`，算术 `Θ(N²d)`。反向 tile 中先恢复 P，再与 `dOᵢ` 相乘并累积 `dVⱼ`，不需要完整 P 存储。

#### Equation (4) — 把一行归约压缩成一个内积

{{eq:4}}

`dPᵢⱼ=doᵢᵀvⱼ`。直接算 `Dᵢ=Σⱼ Pᵢⱼ dPᵢⱼ` 看似要遍历全部 Key；交换求和得 `Dᵢ=doᵢᵀ(ΣⱼPᵢⱼvⱼ)=doᵢᵀoᵢ`。所以 D 可以由前向输出与上游梯度做 `[N,d]` 的逐行内积，成本 `Θ(Nd)`，不用存 `[N,N]` 的 `dP`。这是一个真实减少中间依赖的代数变换，不是只给操作换名称。

#### Equation (5) — Query 的梯度

{{eq:5}}

Softmax Jacobian 收缩给出 `dSᵢⱼ=Pᵢⱼ(dPᵢⱼ−Dᵢ)`，再由 `S=QKᵀ` 得 `dQ=dSK`。shape 为 `[Bᵣ,B꜀]×[B꜀,d]→[Bᵣ,d]`。每个 Query 块累积来自所有 Key 块的贡献；减去 D 表示 score 变化不仅影响本位置，也会通过分母影响整行概率。

#### Equation (6) — Key 的梯度

{{eq:6}}

同理 `dK=dSᵀQ`，shape 为 `[B꜀,Bᵣ]×[Bᵣ,d]→[B꜀,d]`。K/V 外循环里可在片上累积它们的梯度；不同 loop order 会改变 dQ 或 dK 的多次写回模式、并行归约与同步需求。

**Scale 修正。** 如果 `S=τQKᵀ`，则最终 `dQ=τdSK`、`dK=τdSᵀQ`；`dV` 没有这个 τ。原基础推导省略它，完整 Algorithm 4 补上。抄代码时忘掉 τ 是数学错误，不是普通浮点误差。

### B.3 FlashAttention：完整 Forward

{{algorithm:2}}

Algorithm 2 加入 scale、mask、dropout 等完整前向语义。dropout 发生在归一化概率之后，使用 `Z/(1−p)` 的随机因子。为融合可对未归一化分子应用 dropout，再用原来的 ℓ 归一化；**ℓ 本身不能对幸存元素重新归一化**，否则变成另一种 operator。输出通常不再满足“权重之和严格为 1”的样本级条件，但期望保持。

### B.4 FlashAttention：完整 Backward 与 RNG replay

{{algorithm:3}}

Algorithm 3 是标准 Attention backward 参照，显式使用完整 P，并把多个梯度中间量写回 HBM。它帮助辨认优化版本省去了哪些生命周期。

{{algorithm:4}}

Algorithm 4 读取前向统计，在每个 tile 中重算 score/probability，再计算 `dV,dP,dS,dQ,dK`。它重放相同 dropout mask 以计算梯度。保存 RNG state 比保存完整 `[N,N]` mask 更省，但不仅要求相同 seed：元素与随机序列的映射、counter offset、tile 次序也要能重现。如果 forward/backward 随机索引不一致，梯度对应的就不是同一次前向。

**可复现教学验证。** [数值验证脚本](assets/evidence/verify_mechanism.py) 使用独立 dense stable softmax 对照不同块宽、大分数与空 mask；进一步以 central finite difference 检查 Q/K/V 梯度，包括 scale、mask、dropout 重放。误差阈值与结果在 [验证记录](assets/evidence/mechanism-results.json)；这些小规模 FP64 验证不代表 GPU 速度、模型收敛或所有 kernel 支持矩阵。

### B.5 Comparison with Rabe and Staats

先行工作提出 memory-efficient exact Attention，但原文指出只降低峰值存储并不保证 wall time 改善。FlashAttention 把 backward 重计算与片上 fusion 联合设计，避免重计算过程再产生大规模 HBM 往返。比较应针对论文当时的版本；Rabe & Staats 当前 arXiv 页面含 2022-10 的 v3，不能把那个版本的实测倒填到本地 June 2022 v2 的历史对比。

## Appendix C — Proofs：每个 Proposition 证明到哪一步？

### C.1 Proposition 1：正确性与存储

证明的关键是不变式：处理前 j 个 Key 块后，m 是这些 score 的最大值，ℓ 是相同 shift 下的指数和，O 是这些 Value 按全局部分 softmax 加权的结果。合并新块时采用共同最大值，不变式持续成立；全部块遍历完即获得完整 softmax Attention。

计算仍覆盖 `N²` 配对，两次主要 matmul 给 `O(N²d)`。HBM 不保存 score/probability，只保留输入、输出和每行统计；因此辅助统计 `O(N)`，输入输出 `O(Nd)`。片上的 `[Bᵣ,B꜀]` workspace 用快速存储容量约束控制。

### C.2 Proposition 2：通信次数来自嵌套循环

令 K/V 块数 `T꜀≈N/B꜀=Θ(Nd/M)`，Q 块数 `Tᵣ≈N/Bᵣ`。外循环每一轮扫完整 Q/O，访问 `Θ(Nd)`；乘外循环数得主项 `Θ(N²d²/M)`。这里 M 以 words 计数，并采用原文 tile 选择；现实多 SM 并行、cache 与共享内存设置会影响常数和路径，但不改变证明如何计量。

### C.3 Proposition 3：避免把量词读错

当 `M=Θ(Nd)`，主项降至 `Θ(Nd)`。任何正确读取一般 Q/K/V 输入的算法都需要相应数据访问，所以不能存在对全容量范围统一达到更小阶的方案。这是一个关于“所有 M”的排除结论，不是所有 fixed-M 实例上都独立证明紧下界。算法在特定硬件上仍能通过更好调度、压缩常数、并行或低精度改变运行时间。

### C.4 Proposition 4：Sparse 的 s 与输入下界

非零 tile 比例为 s，配对处理主项乘 s；输入与输出仍需 `Θ(Nd)`。当 s 很小时，不能只写 `sN²d²/M` 而忘掉线性项。若 sparsity 随长度改变，复杂度也要把 `s(N)` 明确写出。选择 sparse pattern 是模型语义决策，不是一个无损存储压缩操作。

## Appendix D — Extensions：改 mask、bias 与跨设备有什么含义？

### D.1 Block-sparse FlashAttention

{{algorithm:5}}

Algorithm 5 只遍历选定非零块，保留 softmax merge 与完整 forward 扩展。对选定 pattern 是 exact；跳过的 blocks 不是事后把已生成的 probability 归零。

### D.2 Potential extensions

**Paper fact。** 原附录提出三个方向：Multi-GPU Attention，将其他 GPU 的 HBM 视为另一层存储；Sparse MLP，稀疏权重虽然减少算术，却可能转为 memory-bound；Kernel machine learning，利用 `N×N` kernel 元素可由两个低维输入重新计算的结构，减少大 kernel 矩阵读写。文中引用 KeOps 作为减少 IO 的先例。它没有把 dense MLP 普遍称为 memory-bound。

**Interpretation：Attention 的其他接口。** 同一思想也提示如何处理位置、mask 与 cross-attention，但下面是本文工程分析，不是原 D.2 已实现并测试的三项扩展。共同原则是 score tile 在进入 softmax 前完成变换，然后沿正确的有效 Key 集合合并状态；可片上生成的 bias/mask 通常比读取完整 `[N,N]` 矩阵更有利。

**Relative position / bias。** 若 bias 由位置索引生成，可在 tile 中构造；若用户传入一个完整 dense bias，算法避免 S/P 落显存的优势仍存在，但 bias 自身的二次读取或存储不能凭空消失。不是所有自定义 Attention 都会得到同样的 IO。

**Cross-attention。** Query 与 KV 长度可以不同，score 是 `[Nq,Nkv]`。正文等长公式用于原文分析；扩展时配对量是 `NqNkv`，循环边界和 mask 要相应调整，不能强行把二者都记为 N。

**Multi-GPU。** 合并 softmax 状态的代数思想可以服务于跨设备分块，但通信量、设备调度与数值归约另需设计。原论文单 GPU kernel 的 SRAM/HBM 论证，不能自动证明多 GPU 的通信效率；后续 Ring Attention 专门研究这个问题。

## Appendix E — Experimental Details：完整图表与可复现边界

### E.1–E.3 BERT、GPT-2 与 LRA：训练设置如何约束结论

附录补充 BERT、GPT-2、LRA、长文档与路径任务。BERT 采用 MLPerf 初始化和达标时间；GPT-2 按固定训练设置比较不同库；LRA 的任务长度、模型配置和实现来源不同；路径任务包含分辨率 curriculum。本文已在每个主表旁记录关键条件，避免只列一个总平均速度。

**Training tokens。** 对 Table 2 可以由给定 steps、effective batch 与序列长度估算名义 token，但实际有效 token 还受数据打包和 padding 影响。跨 Table 4 长度配置，不能仅凭 steps 相同就认定 token 或计算预算相同。原文没有完整逐任务训练 token 明细，这项应记为未充分报告。

**Task details。** BERT 的目标是 masked language modeling validation accuracy；GPT-2 使用 GPT-2 BPE tokenizer 和语言建模 loss，各模型使用同一次随机划分的 validation set。长文档分类 follow Dai et al. 的超参数，原附录没有完整独立展开模型参数、batch、LR 与训练 token，不能擅自补齐。LRA follow 官方任务及 Nyströmformer 复现配置；任务间不是同一个模型、长度与训练预算。

**Path curriculum。** Path-64 checkpoint 取自 200 epochs，位置 embedding 按网格复制以适配较大分辨率；目标任务训练含 1 epoch 线性 warmup、cosine LR decay。Path-X 按 validation accuracy 选择最佳 checkpoint 后再训练 200 epochs，约 +4 点后开始过拟合。这一过程是理解 Table 6 因果边界的关键。

### Figure 4 — 学习曲线：时间与收敛不能互相替代

{{fig:4}}

先看横轴 training steps、纵轴 validation perplexity，再按 small/medium 分组比较 HF 与 FA 曲线。对应模型的曲线接近，支持 kernel 替换不明显破坏训练结果。横轴不是 wall time，因此曲线本身不能证明训练时间快多少；时间证据来自 Table 2。单次或汇总曲线也不能推出所有随机种子严格一致。

### E.4 FMHA comparison：已有融合 kernel 是必要的 baseline

### Table 7 — 短序列不是始终占优

{{table:7}}

这是 A100-SXM4-80GB、batch 64、16 heads、head dimension 64 的短序列测试，包含 mask 与 dropout。128 / 256 / 512 时，Apex FMHA 的 forward+backward 为 0.27 / 0.81 / 2.95 ms，FA 为 0.28 / 0.75 / 2.81 ms。

**重算。** 128 时 FA 时间多约 3.7%；256、512 时分别少约 7.4%、4.7%。FA 的 backward 三个长度都比 FMHA 更慢；combined 的后两项获益来自 forward。这个表直接否定“所有长度、所有 pass 都更快”。

**Interpretation。** 很短序列时完整 Attention 可能已经适合现有融合策略，分块统计和重算的代价相对变大。Benchmark 必须包括强融合 baseline，而不只拿多个独立 PyTorch 操作比较。

### E.5 GPU、mask、dropout 与 head dimension

### Figure 5 — A100，head dimension 64

{{fig:5}}

按序列长度从左到右，按四种颜色区分 dropout + masking、dropout、masking、无两者。加速随长度和配置变化，说明被融合的低算术强度操作数量影响收益。纵轴是相对指定 PyTorch baseline 的 speedup，不是绝对毫秒或硬件利用率。

### Figure 6 — A100，head dimension 128

{{fig:6}}

读法与 Figure 5 相同，但 head dimension 改为 128。曲线不应与上一图当成同 shape 的重复测量：d 变大会改变 tile 容量、matmul 成本与 occupancy。图显示仍有加速，但不能仅凭柱高推出独立的 d 因果关系，因为最优 kernel 配置可能也发生变化。

### Figure 7 — RTX 3090

{{fig:7}}

消费级 GPU 的测试支持方法具有一定跨硬件价值。依然要按 mask/dropout 与长度阅读；最高柱不是一条全平台保证。A100 与 RTX 3090 的片上资源、带宽、Tensor Core 和软件环境不同，原文没有把它们归一成统一 roofline 实验。

### Figure 8 — T4 与 forward-only

{{fig:8}}

上下两个 panel 分别提供 combined 和 forward-only 视角；先识别 panel，再比较同一配置。只看前向不包含反向重计算，因此与训练时间不能直接互换。T4 的柱图说明跨代 GPU 仍可能收益，也进一步说明应该对目标设备单独 profile。

### E.6 Full speed and memory tables：21 张表的完整性说明

Table 8 是原文导航表，Tables 9–20 是四组 dropout/mask 条件下的 forward/backward/combined；Table 21 为内存。所有行列、单位、粗体、下划线和原始 caption 保留。可选文字表与 PDF 原表同时提供；折叠只是控制页面长度，不删除内容。

{{table:8}}

**共同 Setup。** Tables 9–21 用 A100 40 GB、batch 16、8 heads、d=64；时间单位 ms，内存单位 MB。表中的 1k、2k 等为原文序列标签。粗体/下划线标识最佳/次佳，不能跨 exact/approximate 语义直接当作质量等价排名。

**Dash 审计。** Megatron-LM 测到约 2K，OpenAI sparse 实现约 4K，Longformer/BigBird 约 8K；部分 backward 有 mask 支持问题。dash 可能是 unsupported、OOM 或实现约束，原文并未给每一个空格唯一统一的失败原因。

### Tables 9–11 — Dropout + masking

{{table:9}}

**Table 9 / Forward。** 比较带 dropout 和 mask 的前向，各实现是否把逐元素路径融合到 matmul 至关重要。沿每一行看长度增长，再在同长度下比较 exact baseline；不要把 Linformer/Local 的最低数字解释成它们计算了同一个 dense Attention。

{{table:10}}

**Table 10 / Backward。** 重计算的代价与减少激活读取的收益都在这里体现。某些库只有一部分长度可以运行 backward，因此只在共同支持的列比较。表单独计时，不要求数值和 Table 9 严格加和到下一表。

{{table:11}}

**Table 11 / Combined。** 1K 时 PyTorch 8.29 ms、FA 2.55 ms，加速约 3.25×；Linformer 1.60 ms 则比 FA 更快，但属于近似路线。64K 时 dense FA 9341.3 ms、sparse FA 64.11 ms，约 145.7× 的差异包含大量被跳过的配对，不能宣传为 dense exact kernel 的无损加速。

### Tables 12–14 — Masking、无 dropout

{{table:12}}

**Table 12 / Forward。** 关闭 dropout 后，RNG 与随机缩放消失；比较本表与 Table 9 可观察配置相关变化，但两个设置中的最佳方案不必相同。mask 仍参与有效 Key 集合，不能把其开销或语义省略。

{{table:13}}

**Table 13 / Backward。** 不需要重放 dropout RNG，但依旧要在相同 mask 下重算概率。Backward 支持矩阵不等于 forward 支持矩阵，空项不补值、不推断时间。

{{table:14}}

**Table 14 / Combined。** 这是许多 causal 训练场景更接近的参考组，但“masking”包含论文具体设置，不能拿此表替代现代任意 mask、GQA、paged KV 或 cross-attention 的测试。读取一整行才能看到长度与交叉点。

### Tables 15–17 — Dropout、无 masking

{{table:15}}

**Table 15 / Forward。** 没有 mask 的完整 dense 配对，仍有 dropout；与 masking 组相比，有效计算量及 mask 处理不同。FA 在不显式保存 P 的条件下应用随机权重，但输出分母依然来自完整未 dropout softmax。

{{table:16}}

**Table 16 / Backward。** 需要恢复 forward 对应的 dropout mask。实现速度之外，还应检验 RNG replay 的数学一致性；单纯 benchmark 成功不是梯度正确性的证明。

{{table:17}}

**Table 17 / Combined。** 128 tokens 时 sparse FA 0.91 ms，dense FA 0.35 ms，sparse 多约 160% 时间。这个短序列反例说明非规则调度可以比减少配对更贵；“sparse 在每个长度都更快”的描述不被原表支持。

### Tables 18–20 — 无 dropout、无 masking

{{table:18}}

**Table 18 / Forward。** 最简 dense operator，融合收益主要来自 score/probability 的片上生命周期，而不是额外 dropout/mask 的消除。适合判断最基础的 IO-aware 路径，但不能自动代表训练的完整成本。

{{table:19}}

**Table 19 / Backward。** 即便没有 mask/dropout，backward 仍需重生成 P、做多个梯度 matmul 与跨 tile 累积，不能只拿 forward 的倍数外推。

{{table:20}}

**Table 20 / Combined。** 512 时 sparse FA 0.82 ms、dense FA 0.73 ms，sparse 多约 12.3% 时间，是另一个短序列反例。中长序列会出现不同排序；按列和 exact/approximate 分组理解，比只摘最大倍数更可靠。

### Table 21 — Memory：线性曲线与真实 MB

{{table:21}}

4K 时 PyTorch 17024 MB，FA 836 MB，约 20.36×；64K 时 FA 13376 MB、Linformer 26252 MB，后者约是 FA 的 1.96 倍。64K 的 sparse FA 13384 MB，与 dense 13376 MB 接近但不完全相同，不能写成 byte-identical。

**How to read。** 这是指定 batch/head/dtype、实现与 allocator 测量下的 Attention 内存，不是整个 Transformer 训练峰值。线性存储方案也会因临时 workspace、allocator、不同精度产生不同常数；图中线性规律支持算法，但不保证任意框架里的同一 MB 值。

**工程启发。** 真正复现实验时要区分 allocated/reserved、峰值与当前值、算子额外内存与整个模型的峰值，并将 GPU allocator、workspace、保存激活和持久 KV Cache 分开。单看一个工具的“GPU memory used”不能定位每个 operator 的损耗。

## 扩展调研 — 从 online normalizer 到 2026 年的实现

### 时间线：保留历史，再看后续变化

{{diagram:evolution}}

| 工作 / 时间 | 核心思路 | 优势 | 边界 | 与本文的关系 |
|---|---|---|---|---|
| [Online normalizer，2018](https://arxiv.org/abs/1805.02867) | 一次遍历更新稳定 softmax 的 max/sum | 少遍历输入 | 不等于完整高效 Attention kernel | 可合并统计的数学先行工作 |
| [Rabe & Staats，2021](https://arxiv.org/abs/2112.05682) | Exact Attention 的分块、低内存求值与梯度 | 消除二次存储的理论可能 | 当前页面含后续修订；不能将最新实测倒填到 2022 年之前 | Memory-efficient exact 路线 |
| FlashAttention，2022 | Tiling + online softmax + backward recompute + IO 模型 | 保持 dense 语义，获得 GPU/训练实测收益 | 二次算术、硬件依赖 | 本次原论文 |
| [FlashAttention-2，2023；ICLR 2024](https://arxiv.org/abs/2307.08691) | 改善工作划分、序列并行与 warp 内任务分配，减少非 matmul 开销 | 论文约 2×，A100 约 50–73% 峰值利用率 | 数字有指定 shape/精度 | 说明 v1 的 IO-aware 方案仍需进一步调度优化 |
| [Ring Attention，2023](https://arxiv.org/abs/2310.01889) | 分布式 blockwise Attention，环式传递 K/V 并重叠计算 | 扩大多设备上下文规模 | 网络通信、拓扑、并行配置 | 从片上存储走向设备间通信 |
| [Flash-Decoding，2023-10](https://pytorch.org/blog/flash-decoding/) | Split KV，分别计算局部 Attention，再合并 log-sum-exp | query 很短时增加并行度 | 必须测 decode，不等于训练 kernel | 使用相同可合并思想解决 occupancy |
| [FlashAttention-3，2024](https://arxiv.org/abs/2407.08608) | Hopper 上异步数据/矩阵流水、warp specialization、低精度支持 | H100 FP16 最高约 740 TFLOPs/s；FP8 约 1.2 PFLOPs/s | 不同 dtype 与误差不能混比 | 将硬件特性进一步写入调度 |
| [FlexAttention，2024-08](https://pytorch.org/blog/flexattention/) | 编译自定义 score modification 与 block mask | 兼顾可定制性与融合 | 自定义模式性能不同 | 降低手写专用 kernel 的门槛 |
| [FlashAttention-4，2026-03](https://arxiv.org/abs/2603.05451) | Blackwell 专属流水与 CuTe DSL 实现 | 论文 B200 BF16 1613 TFLOPs/s，约 71% 峰值利用率 | 作者博客另报 1605；保持各自来源，不混作一个精确数字 | 继续缩短 Tensor Core 与非 matmul 路径的失衡 |

**External source。** FA4 论文报告相对 cuDNN 9.13 最高约 1.3×、相对指定 Triton 实现最高约 2.7×，CuTe DSL 编译时间改善约 20–30×。这些都是相应版本和 benchmark 的结果，不是 2022 FA 的成绩，也不是端到端 LLM 的通用倍数。

### Current state — 截至 2026-10-27，如何谈“SOTA”？

没有一个跨训练/推理、GPU、dtype、mask 和误差约束的全局冠军。本文按 workload 列出核验到的一手资料，而不把单个峰值当作所有场景的 SOTA。

| 场景 / 一手来源 | Hardware / precision | 报告指标 | 模型 / shape 与开源状态 | 可比性与证据强度 |
|---|---|---|---|---|
| [FA4 论文](https://arxiv.org/abs/2603.05451)，2026-03 | B200，BF16 | 峰值约 1613 TFLOPs/s，71% | Attention kernel benchmark；官方库开放 | 与同硬件、dtype、shape 的 kernel 才能比较；峰值不是 TTFT |
| [FlexAttention + FA4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)，2026-03-05 | Blackwell；可编译 score/mask | 可定制 fused Attention 的实测性能 | PyTorch 官方，开放实现 | 性能受 score_mod/block mask 支配，不同 pattern 不能无条件比较 |
| [Low-Precision FA4](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)，2026-09-16，更新 09-23 | Blackwell，MXFP8 | LLM shape 前向约 2.85 PFLOPs/s、反向约 2 PFLOPs/s；内部 workload 约 2.54/1.58 | Meta GEM 训练采用为官方自报；[lp_fa4 代码](https://github.com/facebookresearch/ads_model_kernel_library/tree/main/lp_fa4) 开放 | 低精度量化与缩放参与端到端链路；不能与 BF16 只按峰值排名 |
| [SageAttention3](https://arxiv.org/abs/2505.11594)，最新 v3 为 2026-01-14 | RTX 5090，FP4 | 报告最高 1038 TOPS，约 5× 指定 baseline | 论文及开放实现；包含生成模型场景 | 改变量化误差，论文也讨论 pretraining 收敛变慢；不是 exact BF16 Attention |
| [MpFA](https://arxiv.org/abs/2609.33135)，2026-09-27 | B200，QK FP4 / PV FP8 | 自报最高 2.81× 端到端对 FA4 BF16；约 2% kernel overhead 恢复约 62.5% 的准确率损失 | Llama-3.1-8B / Qwen3-14B，16K–128K；预印本页面是本次直接核验依据 | 单一作者来源，缺独立复现；不开出“已确认全局 SOTA”结论 |

**指标纪律。** FP4 的 TOPS、FP8 的 PFLOPs/s、BF16 的 TFLOPs/s 不能只看数字大小。需要同样问题规模、计算计数、mask、精度、误差标准与测量边界。量化输入是否已生成、量化/scale 的 producer fusion 是否计入，都会改变实际收益。

**现阶段主流入口。** 应优先使用框架的 [scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 或官方 [flash-attention](https://github.com/Dao-AILab/flash-attention) 接口，由实际支持矩阵决定后端。参考 [Triton fused attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) 可以学习 kernel 构造，但教程曲线不能替代目标业务测量。当前 README 随版本变化，实验要记录 commit/版本而不是只写“main”。

## My insights / future directions — 面向推理链路的个人判断

### 1. Prefill、decode 与 KV Cache 是三个不同的成本

Prefill 的 `Nq≈Nkv≈N`，score 是二次矩阵，FlashAttention 的优势直接对应原问题。Decode 通常 `Nq=1,Nkv=N`，Attention 矩阵已不再是 `N×N`；此时关注 KV 读取带宽、head 数、batch、split-KV 并行、归约成本与 launch。

KV Cache 是多层持续保存的 K/V，容量 `2BLNHkv d×bytes`。FlashAttention 避免中间概率，却不自动减少 persistent cache。MQA/GQA、KV quantization、paged layout、windowed pattern 与 FlashAttention 可以组合，但各自改的是不同成本。

可继续阅读 [MQA：多个读头，共用一份历史](https://howieyin.github.io/papers/mqa/)：共享 KV 改变持久表示的份数，FlashAttention 改变 Attention 中间量的生命周期；两者可以组合，解码速度还取决于实际 KV 复用与并行划分。

**TTFT / TPOT。** 原论文没有提供这些业务指标。TTFT 包含 tokenize、投影、prefill、调度及首 token 路径；TPOT 包含 decode、KV 访问、采样和调度。应以相同 prompt、output length、batch 和 warmup 测整链路，同时 profile per-op。不能拿训练 Attention 倍数预测端侧 TTFT。

### 2. 与 RoPE / VRoPE 的关系：语义与执行正交

RoPE 在 Attention 前旋转 Q/K；VRoPE 决定多模态坐标如何映射到这些旋转。FlashAttention 执行旋转后的 QKᵀ 与 softmax。它不决定位置坐标，也不修复多模态位置设计。可片上融合的位置处理可能影响额外 IO，但必须保持同一位置语义。

相关阅读：[RoFormer：旋转与相对内积](../roformer/)；[VRoPE：视频位置与坐标分配](../vrope/)。它们回答“score 应表达什么”，本篇回答“同一 score 如何高效计算”。

### 3. 正确性检查要穿过 forward / backward 的调用链

最容易产生明显错误的位置是 mask 的 `−∞` 语义、空行、scale 漏乘、O 合并漏掉旧 ℓ、dropout RNG replay，以及 tensor layout/stride 误读。混合精度下还需区分指数/归约累积 dtype、数值容差与随机误差。减少保存激活不是忽略中间状态，而是找到能够重建它的最小状态。

生产验证应比较 output 与梯度，并区分 causal、padding、全空行、非整 tile 长度、不同 d、dropout、GQA/cross-attention。小规模 dense reference 是正确性工具，真实 kernel benchmark 是性能工具，两者缺一不可。

### 4. 对论文的综合评价

| 维度 | 判断 | 依据 / 未解问题 |
|---|---|---|
| Novelty | 强在数学、IO 模型和 GPU 实现的组合 | online softmax、低内存 Attention 有先行工作；跨 GEMM fusion 与训练实测有突出价值 |
| Correctness | 核心合并与梯度推导清晰 | exact 是实数数学意义；浮点、空 mask、dropout 实现契约需另外验证 |
| Experiment quality | kernel、模型与任务覆盖广 | 最强 fused baseline 揭示短序列边界；部分精度/实现不完全对齐 |
| Engineering complexity | 高 | tile、寄存器、shared memory、归约、RNG 与后向累积一起约束 |
| Scalability | 缓解激活和 IO，未消除二次算术 | fixed M 下 N 复杂度仍二次；多 GPU 通信和 decode 需专门方案 |
| Generalization | 算法思想适用多种 Attention | 任意 dense bias、复杂 pattern、目标 GPU 会改变实际收益 |
| Reproducibility | 公式、算法和完整表支持复核 | 当前代码已演进；需要固定原始版本、软件栈与失败项原因 |

**Strengths。** 直指真实执行瓶颈；用可合并状态维护 exact Attention；把重计算设计成带宽优化；同时给出理论、kernel 和端到端证据。

**Weaknesses。** 完整组件消融不足；部分 baseline 精度和支持矩阵不一致；长上下文任务混入数据量、curriculum 与训练预算变化；最佳峰值不代表所有 shape。

**Hidden trade-offs。** 更少 HBM 换更多算术和复杂 kernel；更长 context 换更大配对成本；sparse 换语义与调度；低精度后继换额外误差与量化链路。

**如果补实验。** 我会增加 iso-token 与 iso-compute 的长度实验；同硬件/精度/强融合 baseline 的 per-op 时间与流量；forward/backward 分离的短长度曲线；全 mask 和 dropout 梯度验证；prefill/decode 的 TTFT/TPOT；以及实际 allocated/reserved/peak 内存的分解。

**Possible directions。** 把 score modification 与 producer fusion 联合优化；根据 query/KV 长度和 batch 动态选择并行划分；把 KV 量化/解码、位置变换、tile load 的误差与 IO 成本一起分析。2026 的低精度结果尤其说明：量化前后转换和 scale 路径的成本不能被隐藏在输入准备阶段。

## References & provenance

原论文：[arXiv 2205.14135](https://arxiv.org/abs/2205.14135) · [完整 v2 PDF](assets/original/2205.14135v2.pdf) · [DOI](https://doi.org/10.48550/arXiv.2205.14135)。

相关论文：[Online normalizer](https://arxiv.org/abs/1805.02867) · [Rabe & Staats](https://arxiv.org/abs/2112.05682) · [FA2](https://arxiv.org/abs/2307.08691) · [FA3](https://arxiv.org/abs/2407.08608) · [FA4](https://arxiv.org/abs/2603.05451) · [Ring Attention](https://arxiv.org/abs/2310.01889) · [SageAttention3](https://arxiv.org/abs/2505.11594) · [MpFA](https://arxiv.org/abs/2609.33135)。

官方实现与资料：[flash-attention](https://github.com/Dao-AILab/flash-attention) · [Flash-Decoding](https://pytorch.org/blog/flash-decoding/) · [FlexAttention](https://pytorch.org/blog/flexattention/) · [FlexAttention + FA4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/) · [Low-Precision FA4](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/) · [FA4 作者博客](https://tridao.me/blog/2026/flash4/) · [Triton 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) · [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。

核验与资产来源：[一手资料 manifest / 日期与哈希](assets/evidence/sources-manifest.json) · [8 图 / 21 表 manifest](assets/evidence/visual-manifest.json) · [数值验证](assets/evidence/mechanism-results.json)。原始图表为 v2 PDF 的高分辨率完整区域；可选文字表和原编号公式来自 arXiv HTML；教学图、推导、重算与审计为本文创建。所有后续信息以 **2026-10-27** 为检索截止日，不隐含持续更新。

**Changelog。** 2026-10-27：完成 v2 全文与附录精读，保留全图表，加入可合并 softmax / IO 教学组件、两轮审校与后续研究快照。
