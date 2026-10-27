# MQA：多个读头，共用一份历史

**Fast Transformer Decoding: One Write-Head is All You Need**

Noam Shazeer · Google · 2019 · arXiv preprint  
原文：1911.02150v1，9 页。arXiv 提交于 2019-11-06，PDF 首页日期为 2019-11-07；线上记录仅有 v1。精读与扩展调研：2026-10-27。

[Paper PDF](assets/original/1911.02150v1.pdf) · [arXiv](https://arxiv.org/abs/1911.02150) · [DOI](https://doi.org/10.48550/arXiv.1911.02150) · [Original TeX](assets/evidence/main.tex) · [Implementation reference](https://github.com/tensorflow/mesh/blob/e6798a2610a2c2f4c4cd236d8214422cb1ecc00a/mesh_tensorflow/transformer/attention.py)

## TL;DR：把写入历史的表示与读取历史的视角分开

<div class="callout"><strong>一句话：</strong>MQA 保留多个 query head，却只产生一组共享 K/V；减少解码时反复读取历史状态的字节数，以小幅质量代价换取更快的增量推理。</div>

一段话理解：训练时，整段目标序列已知，许多 query 能在同一次矩阵运算中复用 K/V。逐 token 生成时，当前 token 决定下一个 token，时间维度的并行消失；每一步都要读取历史 K/V。MHA 为每个 head 保存一份历史，MQA 则让所有 query head 读同一份历史。它改变的是模型的 K/V 参数化和持久状态，保留了多个 query 及各自的 attention 分布。论文在等总参数量的翻译和语言建模实验中验证质量，在大 batch TPUv2 解码实验中验证速度。

五个关键点：

1. **删掉 K/V 的 head 维，保留 Q 的 head 维。** 历史 token 数量、因果关系、全局可见范围均未因此减少。
2. **KV 容量缩小 h 倍，不等于整层 FLOPs 或整模型耗时缩小 h 倍。** QK、softmax、加权 V 仍要为每个 query head 执行。
3. **等参数比较包含 FFN 扩宽。** 翻译模型把 dff 从 4096 调至 5440；这会影响质量和计算，不能忽略。
4. **约 12 倍是特定 decoder 速度结果。** Table 2 的 46 → 3.8 μs 是 batch=1024 时的摊销成本；实际 greedy step 为 47 → 3.9 ms。
5. **GQA 把二选一变成连续设计空间。** 后续工作用若干 KV group 平衡质量、cache 和并行；MLA 则进一步改变持久状态的表示。

本文主线严格按原论文 §1 → §2 → §3 → §4 → §5 展开；之后单独讨论实验审计、相关工作、当前状态和个人判断。原文没有 Figure、没有附录、没有编号公式，包含 **3 张原始表、6 段代码**。所有表均保留 PDF 原貌与可选择的完整行列；下面新增示意图均标为 Agent-created。

## 1. Introduction：缓存解决了重算，却留下反复读取

**原文 §1，PDF p.1。** Transformer 用 attention 传递序列内及序列间的信息。相比 RNN，它在训练中容易沿序列长度并行；但这种优势不能直接搬到 autoregressive inference。

先区分两类成本。没有 KV cache 时，每个新 token 都重算历史 token 的 K/V。加入 cache 后，这些投影只计算一次并保存；然而当前 query 仍要和历史 key 比较、按权重聚合历史 value。缓存降低重算，历史状态的读取仍随长度增长。

作者的观察是：现代计算硬件的算力相对于带宽很高。decode 只产生一个新 query，却搬运大量 K/V，算术密度不够高。这里的核心问题是 **每搬一个字节能完成多少有效计算**，不能只看模型 FLOPs。

此前可以缩短序列、使用 local attention 或压缩 memory positions。MQA 从另一个轴切入：不减少历史位置，而减少每个位置保存的表示份数。这是本文新颖性所在。

<div class="callout interpretation"><strong>阅读解释：</strong>标题里的 “write-head” 指把当前 token 投影成 K/V 并写入历史的表示通路；多个 query 是读取视角。这是模型机制的比喻，并非硬件只有一次内存 store，也不表示 K 和 V 是同一个 tensor。</div>

## 2. Background: Neural Attention：先把运算和字节对应起来

**原文 §2，PDF pp.1–4。** 作者用 einsum 给出方法，因此理解字母如何消失，比记住函数名更重要。左侧所有轴的并集定义乘法，右侧没有出现的轴被求和；广播是数学语义，不要求实际生成一个巨大的广播 tensor。

| 符号 | 原论文含义 | 本文使用边界 |
|---|---|---|
| b | 不相互交互的序列 batch | 与同一序列中的 h 个 query head 不同 |
| n | 一次处理的 query 位置数，或生成总长度 | decode 单步只有一个新位置 |
| m | memory / KV 位置数 | 自注意力单步为可见历史长度 |
| d | 输入与输出 hidden dimension | 原论文简化令二者相同 |
| h | query head 数 | MQA 中仍然保留 |
| k、v | 每个 key、value 的维度 | 复杂度简化时 k=v=d/h |
| X、M | query 来源和 memory 来源 | self-attention 中通常来自同一序列；cross-attention 来源不同 |
| Pq、Pk、Pv、Po | 可学习投影 | 本文矩阵方向遵循原论文 einsum |

### 2.1 Dot-Product Attention：一个 query 如何读历史

**原文 §2.1，PDF pp.1–2。** q 的 shape 为 [k]，K 为 [m,k]，V 为 [m,v]。

$$
\ell_j=\sum_{a=1}^{k}q_aK_{ja},\qquad
w_j=\frac{\exp(\ell_j)}{\sum_{r=1}^{m}\exp(\ell_r)},\qquad
y_c=\sum_{j=1}^{m}w_jV_{jc}.
$$

上式是对原代码的数学重写，非原论文编号公式。第一次 contraction 消掉 k，得到每个历史位置的 logit；softmax 沿 m 归一化；第二次消掉 m，得到 [v] 的输出。单 query 的 QK 与加权 V 分别是 Θ(mk) 与 Θ(mv)，而 K/V 都必须提供给计算。

{{listing:1}}

作者省略常见的 1/√k scaling，并在 §2.2 说明可把常数折入 Pq 或 Pk。解释论文时应保留这一约定；迁移一个已有 checkpoint 时，不能随意删掉其 scaling。工程里 softmax 还应先减去最大 logit，避免指数溢出。

### 2.2 Multi-head Attention：读视角和写表示原本一一配对

**原文 §2.2，PDF p.2。** MHA 的每个 head 都有独立 query、key、value 和输出投影。

$$
q_i=xP_{q,i},\quad K_i=MP_{k,i},\quad V_i=MP_{v,i},
\quad o_i=\operatorname{softmax}(q_iK_i^\top)V_i,\quad
y=\sum_{i=1}^{h}o_iP_{o,i}^\top.
$$

这里 q=[h,k]，K=[h,m,k]，V=[h,m,v]，o=[h,v]，y=[d]。原代码的 Po=[h,d,v]，所以每个 head 输出右乘 Po 的转置。它与常见的 concat(heads) 后乘一个 Wᴼ 是等价的分块写法。

{{listing:2}}

保留 h 个 query head 的作用，是允许同一个 token 以多个方向提问。保留 h 组 K/V 的作用，是允许历史 token 为每个 head 写入不同的索引特征和被取回的内容。MQA 后面要放弃的是后一种自由度，而不是前一种。

### 2.3 Batched MHA：训练与 prefill 为什么更容易复用

**原文 §2.3，PDF p.3。** 同时加入 batch b 和 query 长度 n。

| 运算 | einsum | 输出 shape | 工程含义 |
|---|---|---|---|
| Query projection | bnd,hdk→bhnk | [b,h,n,k] | 每个 query 位置各自产生 h 个读头 |
| Key projection | bmd,hdk→bhmk | [b,h,m,k] | 一段 memory 一次写完各 head 的 K |
| Value projection | bmd,hdv→bhmv | [b,h,m,v] | 与 K 配对的历史内容 |
| QK scores | bhnk,bhmk→bhnm | [b,h,n,m] | 同一 K 可被 n 个 query 复用 |
| Weighted values | bhnm,bhmv→bhnv | [b,h,n,v] | 每个 query/head 仍有独立结果 |
| Output projection | bhnv,hdv→bnd | [b,n,d] | 合并各 head |

{{listing:3}}

mask 的 shape 可写为 [b,h,n,m]；实现可用可广播形式，不必存满。因果 self-attention 让未来位置为 −∞；packed 训练样本还应阻止不同句子/文档间错误交互。论文提到句子拼接，但没有完整公开其 packing mask 配置，不能替作者补出具体实现。

### 2.3.1 Batched 性能分析：每个权重跨 bn 个 token 摊销

**原文 §2.3.1，PDF p.3。** 作者限定 m=n、k=v=d/h、n≤d，并计入中间 tensors 的大小。

{{original:batched-compute}}

{{original:batched-memory}}

{{original:batched-ratio}}

这里 memory 是访问元素规模的估计，并非带 bytes、读写次数和 cache 层级的完整 DRAM trace。bnd 来自输入、投影及输出；bhn² 来自 logits/weights；d² 来自投影参数。

把分式展开可得 O(1/d + n/(dk) + 1/(bn))。因为 n≤d、k≤d，作者把前两项粗略上界为 O(1/k)。**大 bn 能摊薄投影权重的读取，多个 query 能复用同一历史状态。**

这是未融合实现的分析。FlashAttention 后来通过分块与在线 softmax 避免把完整 score/weight 矩阵写回 HBM；这会改变 IO 账，但不会把 MQA 的持久 KV sharing 自动替代掉。[R5]

### 2.4 Incremental MHA：一步只添一项，却重新读很多项

**原文 §2.4，PDF pp.3–4。** 已知目标序列的训练可以用 teacher forcing 并行。生成时，位置 t 的输出决定 token t+1，后者决定下一层输入，因而普通 autoregressive decoding 不能把所有未来 query 提前算出来。

单步 x=[b,d]，q=[b,h,k]；旧 cache 为 K=[b,h,m,k]、V=[b,h,m,v]。写入本步投影后长度为 m+1，对全部可见历史算 attention，得到 [b,d]。

{{listing:4}}

原样保留的代码有两个实质笔误：投影当前 token 时使用未定义的 `M`，应该使用参数 `x`；最后输出投影使用 `O`，此前变量是小写 `o`。这里直接运行会失败，不能当作可执行参考。

原文用 `concat` 描述增长。真实 runner 通常预分配 cache 或使用 page/block allocator，按 token index 写新槽位。每步复制全部旧 cache 会额外产生 Θ(m) 的读写；这并非 attention 机制必需的成本。

### 2.4.1 Incremental 性能分析：batch 只能解决其中一项

**原文 §2.4.1，PDF p.4。** 生成 n 步，在相同简化假设下：

{{original:incremental-compute}}

{{original:incremental-memory}}

{{original:incremental-ratio}}

第一项 bn²d 来自 ∑ₜ btd：cache 随历史增长，每一步再次被读。第二项 nd² 来自每步重读投影权重。除以计算量后得到 n/d 和 1/b。

增大 batch b 能让更多序列共享参数读取，降低 1/b；不同序列的 KV 通常仍不同，增大 batch 不会消掉 n/d。而 cache 越大，又越限制可容纳的 batch，形成容量与带宽的耦合。

<div class="callout warning"><strong>适用范围：</strong>这里 Θ(bnd²) 依赖 n≤d。一般情况下应保留 attention 项 Θ(bn²d)，总量为 Θ(bnd²+bn²d)。长上下文 n≫d 时，不能继续宣称计算随 n 线性增长。</div>

原文此处写 “bhmk=bn²”，在给定假设下并不普遍成立：因为 hk=d，单个 K 的规模应为 bmd，最终 m=n 时为 bnd；累积读取才是 Θ(bn²d)。本文使用前后性能分析一致的 bmd / bn²d，不传播这个维度笔误。

## 3. Multi-Query Attention：只删除写入端的 h

**原文 §3，PDF pp.5–6。** MQA 把 Pk 从 [h,d,k] 改为 [d,k]，Pv 从 [h,d,v] 改为 [d,v]。Q 和 Po 都保留 h。

$$
Q=\operatorname{einsum}(\text{bnd,hdk}\rightarrow\text{bhnk}),\quad
K=\operatorname{einsum}(\text{bmd,dk}\rightarrow\text{bmk}),\quad
V=\operatorname{einsum}(\text{bmd,dv}\rightarrow\text{bmv}).
$$

这是原代码的 shape 重构。随后：

$$
\ell_{binj}=\sum_a Q_{bina}K_{bja},\quad
W=\operatorname{softmax}_j(\ell+\mathrm{mask}),\quad
O_{binc}=\sum_j W_{binj}V_{bjc}.
$$

不要把 h 从 scores 或 weights 中删除。即便所有 head 读取相同 K/V，不同 Q 也能给历史位置分配不同权重，得到不同输出。**共享历史表示 ≠ 共享 attention 分布。**

{{diagram:execution}}

{{listing:5}}

MQA 的训练目标没有另加蒸馏项或 cache 正则项，仍是原任务的预测训练。被分享的 K/V 投影参数接收所有 query head 汇合的梯度，因此需要在训练中学出“可被多个读头共同使用”的历史表示。

这同时解释了质量风险：MHA 可以为不同 head 学出独立索引和取回空间；MQA 把这些写表示限制在同一个 k/v 维空间。多个 query 能表达不同检索方向，却不能凭空恢复被删除的独立 K/V 子空间。

### 共享 K/V 的具体例子

下面是固定二维向量的 **教学构造**，不是训练所得 attention map。K 分别为 (1,0)、(0,1)、(−1,0)，V 使用同样三组数值只是为了容易读图；真实 MQA 的 K 和 V 有不同投影，通常不相等。

令 Q₁=(2,0)、Q₂=(−2,0)。无 scaling 的 logits 为 [2,0,−2] 和 [−2,0,2]，softmax 约为 [0.867,0.117,0.016] 及其逆序。两个读头共用同一份 cache，输出的第一维却分别约为 +0.851 与 −0.851。

{{interactive:queries}}

### Incremental MQA：cache 的时间轴也随之移动

{{listing:6}}

原文 §3 的 incremental listing 同样误用了 `M` 与 `O`，并把 [b,k] 的新 key 在 axis=2 扩维和拼接。MQA cache 是 [b,m,k]，正确的新 key 应为 [b,1,k]，**时间轴是 axis=1**。MHA 的时间轴是 axis=2；不能仅删除 einsum 字母而不修改 cache 操作。

以下为修正版教学实现，可用于理解 shape。它用 NumPy、稳定 softmax、原文未缩放的 logits；`concatenate` 只用于验证，不是高效生产 cache allocator。

```python
def mqa_step(x, cache_k, cache_v, pq, pk, pv, po):
    # x: [b,d]; cache_k/cache_v: [b,m,k/v]
    q = np.einsum("bd,hdk->bhk", x, pq)
    cache_k = np.concatenate((cache_k, (x @ pk)[:, None, :]), axis=1)
    cache_v = np.concatenate((cache_v, (x @ pv)[:, None, :]), axis=1)
    scores = np.einsum("bhk,bmk->bhm", q, cache_k)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    o = np.einsum("bhm,bmv->bhv", weights, cache_v)
    y = np.einsum("bhv,hdv->bd", o, po)
    return y, cache_k, cache_v
```

训练/批量 prefill 与逐步 decode 必须给出同样的因果结果。本文用独立的显式 head 循环核对完整因果计算，再比较逐步缓存计算，float64 最大误差分别为 8.89×10⁻¹⁶ 以下与 4.45×10⁻¹⁶ 以下。[Validation](assets/evidence/mechanism-checks.json) 只验证代数与 shape，不复现论文 BLEU、PPL 或 TPU 性能。

### 3.1 Performance Analysis：h 倍节省具体发生在哪

**原文 §3.1，PDF p.6。**

{{original:mqa-compute}}

{{original:mqa-memory}}

{{original:mqa-ratio}}

bnd 是本步激活/输出，bn²k 是历史 K/V 累积读取，nd² 是仍然存在的投影权重成本。由于 k=d/h，历史读取项从 n/d 变成 n/(dh)；1/b 没有消失。

**存储。** 若一层一条序列缓存 m 个 token，MHA 需 m·h·(k+v) 个元素，MQA 需 m·(k+v)。令 k=v、层数 L、每元素 s bytes，得到：

$$
C_{\mathrm{KV}}=2bLm\,h_{\mathrm{KV}}k\,s,\qquad
h_{\mathrm{KV}}=h\ \text{(MHA)},\quad 1\ \text{(MQA)}.
$$

上式是本文补充的精确容量账。它不含 padding、allocator metadata、量化 scales、beam/prefix sharing 或额外工作区。单步写入为 2bLhKVks，与 m 无关；每步完整逻辑读取近似 C_KV，随 m 线性增长。

**计算。** 固定 b、m、h、k，QK 与 weighted V 的乘加计算合计约 4bhmk FLOPs/层，MQA 不改变它。K/V 投影的 FLOPs 确实减少，但 Q、输出投影和 h 份 softmax 都还存在。完整层复杂度的阶不变，常数可以改变。

**算术密度。** 在理想的共享读取与 k=v 条件下，QK/PV 计算除以 KV read bytes 约为 2h/(hKV·s) FLOPs/byte。BF16 下 MHA 约为 1，MQA 为 h；这里没有计入权重、激活及 softmax。如果 kernel 给每个 query head 从 DRAM 再读一遍共享 K/V，而不能在片上有效复用，实际流量可能远大于这个模型。

{{interactive:cache}}

默认例子 b=1、L=32、h=32、k=128、m=4096、BF16：MHA 仅 KV 为 2 GiB；GQA-8 为 512 MiB；MQA 为 64 MiB。**这是精确 shape 计算的例子，不是论文模型配置，也不是硬件实测。** 切换 dtype 只改变容量账，不提供量化质量结论。

### 训练、prefill、decode 与 cross-attention

| 阶段 | 需要什么 query | KV 如何产生与使用 | MQA 主要改变 |
|---|---|---|---|
| Teacher-forced training | 一段目标位置已知 | 整段 K/V，一次批量注意力，反向累积梯度 | K/V 投影及激活规模；attention score 仍有 h |
| Prompt prefill | 所有 prompt 位置 | 计算并写 prompt KV，因果并行 | 初始 cache 容量与写流量；TTFT 需单独测 |
| Self-attention decode | 当前新 token | 每步写一项，读可见历史 | 最直接的重复读取节省 |
| Encoder-decoder attention | 当前 decoder query | encoder memory 的 K/V 可预计算，随后反复读取 | 源序列 KV sharing 同样减少读取 |

论文实际翻译模型把 encoder self-attention、decoder self-attention、cross-attention **全部** 换成 MQA。Table 2 不是“只改 decoder self-attention”的隔离实验。

## 4. Experiments and Results：质量、参数和速度分开验

**原文 §4，PDF pp.6–8。** 本节保留原论文子章节顺序。Table 3 虽与 Table 1、2 一起排在 p.8，但讨论属于 §4.2，因此放在质量分析旁。

### 4.1 Experimental Setup：等参数不是等架构

| 项目 | WMT 2014 EN-DE | Billion-Word LM |
|---|---|---|
| 架构 | 标准 6 层 encoder + 6 层 decoder Transformer | 6 层 decoder-only Transformer |
| dmodel；h；dk=dv | 1024；8；128 | 1024；8；128 |
| MHA → MQA dff | 4096 → 5440 | 8192 → 9088 |
| 总参数 | 所有变体约 211M | 所有变体约 192M |
| 训练步数 | 100,000，约 20 epochs | 136K，10 epochs |
| 训练 batch | 128 例，input/target 各 256 token | 64K tokens，原文未进一步展开 |
| 训练硬件与耗时 | 32-core TPUv3，约 2 小时/模型 | 32-core TPUv3，约 3 小时/模型 |
| 位置与 embedding | learned positional embeddings；token/output embedding sharing | 原文未完整展开对应细节 |
| 评估指标 | dev ln(PPL)、dev BLEU；test greedy/beam-4 BLEU | dev per-word PPL |
| 推理/训练计时硬件 | 单 TPUv2，8 cores | 无 LM 速度表 |

原文写翻译模型 “6 layers”，结合标准 encoder-decoder 配置理解为每侧 6 层；下面 FFN 参数守恒也与此一致。它没有独立给出模型配置文件来进一步复核。

翻译模型有 18 个 attention 模块：6 个 encoder self-attention、6 个 decoder self-attention、6 个 cross-attention。去掉每层 K/V 的 7 份重复 head，节省：

$$
\Delta P_{\mathrm{attn}}=18\cdot2\cdot1024\cdot128\cdot(8-1)=33{,}030{,}144.
$$

12 个 FFN 各有两个矩阵，扩宽 5440−4096=1344，增加 12·2·1024·1344=33,030,144，正好抵消。LM 对应节省 11,010,048，扩宽 9088−8192=896 同样抵消。以上是按主要矩阵的参数账；原文没有逐项交代 bias 参数与约 211M/192M 的细粒度舍入。

因此实验有一个明确优点：没有靠减少总参数量取得速度。也有一个归因边界：**MQA sharing 与把容量重新分配给 FFN 同时发生**；它们对质量的贡献没有在正文表中完全分离。TeX 中有被注释掉的同 dff 行，但不属于已发表表格结果，本文不把它们当正式证据。

local 变体只让 decoder self-attention 看当前与此前 31 个位置，总窗口 32；encoder 和 cross-attention 不截断。减 h、减 dk/dv 的替代变体使 h·k=128，并扩宽 FFN，以保持总参数。这些变体的 query 总宽度比 MQA 的 1024 小，不能说两类方法保留了相同的 attention 表达能力。

### 4.2 Model Quality — Table 1：共享 KV 比削弱读头更稳

{{table:1}}

**如何阅读。** ln(PPL) 越低越好；BLEU 越高越好。原表没有添加 ↑↓，本文解释方向而不改写原表。dev 用 WMT13，test 用 WMT14；test 两个数字对应 greedy 与 beam=4、α=0.6。部分替代变体 test 列留空，表示原文没有报告，不表示 0。原有 bold 完整保留。

**主比较。** MHA → MQA，dev ln(PPL) 1.424 → 1.439，差 +0.015。还原 PPL 的相对变化为 exp(0.015)−1≈**+1.51%**。dev BLEU 26.7 → 26.5，下降 **0.2 个 BLEU 点**；test greedy 27.7 → 27.5，同样下降 0.2。test beam-4 28.4 → 28.5，增加 0.1。

这支持“质量接近”，不支持“MQA 更准确”。没有多随机种子、置信区间或 bootstrap，0.1 BLEU 不能被当作统计显著优势。相对 dev BLEU 变化约 −0.75%，但 BLEU 不是线性效用尺度，优先看绝对点差。

**替代基线。** h=2、k=v=64 是列出的削减变体中 dev ln(PPL) / BLEU 最好的，仍为 1.480 / 26.2，落后 MQA 的 1.439 / 26.5。h=1 或 h=8、k=16 的 dev BLEU 为 25.8。更合理的解释是：在这组预算下，保留多个宽 query，让它们共享 KV，比同时削弱 query 视角更划算。

**local 交互。** MQA local 的 dev ln(PPL)=1.437，略好于 global MQA 的 1.439；test beam=28.2 却低于 global 28.5。它说明可组合，不能说明 local 无质量代价。

评估命令原样保留：`sacrebleu -t wmt13 -l en-de -tok intl` 与 `sacrebleu -t wmt14 -l en-de -tok intl`。原文没有 sacrebleu 版本和完整 signature；其数值不能与使用其他 tokenizer、case、参考集的 BLEU 直接拼成 SOTA 排名。

### 4.2 Model Quality — Table 3：语言建模上出现相同趋势

{{table:3}}

**指标。** 这里是 dev **per-word** perplexity，越低越好；与翻译表的 per-subword ln(PPL) 不是同一个尺度。PPL 是负 log-likelihood 的指数变换，不是分类准确率，不能从这三张表推出 TP/FP/TN/FN、Precision、Recall、F1。

MHA → MQA：29.9 → 30.2，绝对 +0.3，相对约 **+1.00%**。最好的削减变体为 h=8、k=v=16，PPL=30.9；它比 MQA 高 0.7，约 2.32%。其余削减变体为 31.0–31.2。

这个结果为翻译之外的任务提供了补充证据。但只有一个较小的 6 层语言模型和一个 benchmark，且 Billion-Word 主要由被打乱的独立句子构成，无法证明长文档、多模态、复杂推理或巨大模型的质量不受影响。论文没有提供该任务的加速比。

### 4.3 Speed — Table 2：摊销微秒与 wall time 的区别

{{table:2}}

**条件。** 单 TPUv2（8 cores），greedy inference batch=1024（每 core 128），source/target 各 128 token。self-attention memory 固定 padding 到 128；local 模型为 32。每一步执行固定 shape，所以作者测得每步耗时恒定。训练计时用 input/target 各 256 token，不能被表 caption 的“sequence length 128”误导。

| 从原文及 Table 2 重算 | MHA | MQA | 正确解读 |
|---|---:|---:|---|
| Greedy encoder wall time | 222 ms | 195 ms | 整个输入 batch 一次编码 |
| Greedy decoder step wall time | 47 ms | 3.9 ms | 一步为每条序列各生成一个 token |
| Greedy decoder 摊销成本 | 46 μs | 3.8 μs | wall time ÷ 1024，不是单请求延迟 |
| Decoder speedup | — | ≈12.1× | rounded 46/3.8；raw 47/3.9≈12.05× |
| Encoder+decoder 摊销总成本 | 47.7 μs | 5.3 μs | 等 source/target 长度下约 9.0× |
| Beam-4 decoder 摊销成本 | 203 μs | 32 μs | 约 6.34×；不同于 greedy |

greedy decoder 的时间下降约 91.7%，beam-4 约 84.2%。把 encoder 纳入后 greedy 全程约为 222+128·47=6238 ms 与 195+128·3.9=694.2 ms，约 **8.99×**。这与表中舍入数算出的 9.0×一致，不包含完整线上服务调度、网络或 tokenization。

**TPOT。** 在这个 batch 设置下，一条序列相邻 token 的 step 间隔约为 47 / 3.9 ms，不是 46 / 3.8 μs。**Throughput。** batch 同时产出 1024 个 token，故可以由 batch/step 推得吞吐，但不能由此推出 b=1 的速度。**TTFT。** 论文没有现代服务定义下的独立 TTFT 测量，encoder wall time也不能直接替代端到端 TTFT。

**local 的作用。** MHA decoder 46 → 23，MQA 3.8 → 3.3；在 MQA 下再缩窗口的收益较小。它与“共享后 KV 成本不再是唯一瓶颈”的解释一致，但论文没有 per-op profiling，不能据此断言余下多少时间来自 FFN、softmax、参数读取或 beam 处理。

**训练。** 433 → 425 ms/step，改善约 1.85%；表中 13.2 → 13.0 μs。原文以 32,768 作分母，实际一步同时有 32,768 input 和 32,768 target token。因此它相当于每个 input/target 配对或每 target token 的成本；若把两侧 token 都计入，baseline 是约 6.61 μs/物理 token。原文措辞 “per (input-token + target-token)” 存在口径歧义，本文显式说明，不悄悄改掉表中数值。

<div class="callout interpretation"><strong>为什么加速可以超过 h=8？</strong>h 倍是单份 KV 容量的精确比例，并非整模型时间上界。这里还改变多个 attention 通路的投影、shape、可能的 kernel 效率及 FFN 分配。论文没有把 12 倍分解到算子或带宽 counter，不能把 12.1× 全归因于“少读 8 倍 KV”，也不能用它证明理想带宽模型失效。</div>

## 5. Conclusion：证据支持的是一个有效的架构取舍

**原文 §5，PDF p.8。** 作者提出 MQA，把增量推理中的 KV memory-bandwidth 要求大幅降低，认为它能帮助 attention 模型进入推理性能敏感的应用。

在本文设置中，这个结论有两个证据支点：等总参数模型在两个任务上质量接近；大 batch TPUv2 翻译 decoder 明显加速。它没有证明任意 MHA checkpoint 可无损转换，没有证明所有任务/硬件/batch 都同样受益，也没有证明固定窗口或大模型训练稳定性。

原论文 References 在 pp.8–9，连接 Bahdanau attention、Transformer、长序列压缩、average attention 和 time-restricted attention。没有单独的 related-work 章节，也没有附录。下面内容均为本文扩展分析。

## 6. Experiment Audit：哪些结论足够，哪些仍缺证据

| 审计维度 | 原文确实提供 | 需要保留的边界 |
|---|---|---|
| Novelty / mechanism | 保留 query heads、共享 K/V；容量账清晰 | 模型参数化变化，不是对任意 MHA 的等价加速 |
| Baseline fairness | 同任务、训练步数、硬件；总参数对齐 | FFN 扩宽使注意力与 FFN 贡献耦合 |
| Quality | 翻译 dev/test、多 decoding；LM dev PPL | 无多种子、显著性、长上下文或大型模型 |
| Speed | 8-core TPUv2；greedy/beam；窗口对比 | batch=1024、长度128、固定 padding；无 b=1 sweep |
| Precision | 未明确报告 | 不能凭 TPU 型号写成已证实 BF16 |
| Training recipe | 步数、batch、规模与总时长 | learning rate、optimizer、dropout、label smoothing 未完整给出 |
| Compute accounting | 理论 Θ 分析；FFN 的重新分配可核算 | 无 measured FLOPs、HBM bytes、per-op time |
| Reproducibility | 原 TeX、示意代码；说明用 t2t / Mesh TensorFlow | 配置链接仍是 “to be added before publication” |
| Generalization | 两种语言任务 | 没有 VLM、端侧 NPU 或 long-document 证据 |
| Capacity/scalability | KV 存储按 head 因子下降 | 没有实际峰值显存、最大 batch 或 tensor-parallel 测量 |

**优点。** 方法非常小，却把模型设计与 inference IO 直接连起来；作者没有只报最快的一个结果，而同时给 global/local、greedy/beam 与削减 attention 的替代路线。

**弱点。** 主性能点非常依赖大 batch、短固定长度与 TPU 实现；超参数发布承诺没有兑现到当前 v1 文本；质量差异缺乏统计估计；代码有会影响执行的错误。它应被视为强机制与初步实证，不是今天完整的 serving benchmark。

**隐藏成本。** 少 KV head 可能限制表示；等参数时 FFN 变宽；共享 KV 的高效 kernel 需要跨 query head 复用；head-axis TP 常需要复制单组 KV；量化 cache 还涉及 scale/zero-point、误差和混合 dtype，不是只乘一个 byte 数。

**我希望补的实验。** b∈{1,8,64,1024}、长度从 128 到长上下文的 sweep；相同 FFN 与相同参数两组控制；逐层/逐算子 time 与读写 bytes；多种子 BLEU/PPL；native shared-KV kernel 对比物化 repeat；完整 TTFT/TPOT/throughput 与峰值 cache；TP 与 beam/prefix sharing 场景。这些是建议，本文没有声称运行了它们。

## 7. Related Work：从共享写头到不同的持久状态

扩展资料优先原论文、作者/组织代码，检索截止 **2026-10-27**。下面区分历史机制、后续实验和系统实现，不把不同任务的数字混在一起。

### 7.1 Before / Concurrent：减少位置与减少 head 是两个轴

2017 Transformer 的 MHA 是本文的出发点。2018 年 long-sequence summarization、Average Attention Network 和 time-restricted attention 等路线改变 memory positions、聚合方式或可见区域；MQA 在保留全局历史位置的情况下减少 head multiplicity。local 与 MQA 的组合实验正是验证这种区别。[R1, R2]

2019 年的证据应放回当时背景：主要是 200M 级语言任务、TPU、较短序列。不能用今天的巨大 decoder-only LLM benchmark 替换作者原本的问题；也不能因为今天普遍有 KV cache，就忽略“cache 仍要读”在当时模型设计中的价值。

### 7.2 After：PaLM、GQA 与 Llama

**PaLM（2022）。** 其 Architecture 章节明确写 K/V 为 [1,head_dim]，Q 保持多个 head，并与 RoPE 一同使用，报告其设置下质量与训练速度大体中性、decode 更经济。这提供规模化应用证据，但不是对 2019 年实验的相同配置复现。[R3]

**GQA（2023，EMNLP）。** 设置 G 个 KV head，每组 h/G 个 query。G=h 恢复 MHA，G=1 恢复 MQA。另提出把 MHA checkpoint 的 K/V heads 按组 mean-pool，然后用原预训练计算约 5% 的 uptraining 适应新表示。T5 实验中选择 G=8 为质量/速度折中。[R4]

这回答了 MQA 原文没有回答的迁移问题：sharing 不是一个可以无损删除的 tensor 轴。平均投影只能给出初始化，后续训练用于恢复能力；“5%”是该研究方案的预算与结果，不是所有模型的通用承诺。

**Llama 3（2024）。** 官方模型论文明确采用 8 个 KV heads 的 GQA 以改善 decode 并缩小 cache。这说明实际部署常选中间点，不意味着所有层、所有新模型都必须 G=8。[R6]

### 7.3 Orthogonal：FlashAttention 与 PagedAttention

FlashAttention 通过分块、在线 softmax 和片上复用减少 attention 计算过程的 IO，保持 exact attention 语义。MQA 改的是模型中 K/V 的表示自由度。二者可以共存：一个减少“历史要存多少”，另一个优化“拿到历史后怎样计算”。[R5]

PagedAttention / vLLM 处理动态 KV 分配、碎片、复制与共享，提高可用 batch 和吞吐。它不把 MHA checkpoint 改成 MQA；MQA/GQA 缩小 block payload，paging 管理这些 blocks。prefix cache reuse 又是跨请求的语义共享，与单模型多 query heads 共享不是一回事。[R7]

### 7.4 MLA：cache 里不一定要显式存 K 和 V

DeepSeek-V2（2024）的 MLA 保存联合低秩 latent 与单独的 RoPE key，通过吸收上投影，decode 可直接在 latent 上计算。若 latent 宽度 r、位置通道宽度 dr，每层每 token 状态为 r+dr，而 MQA 是 k+v；谁更小取决于实际宽度，不能只凭“latent”命名判断。[R8]

关键区别是：MQA 让每个 head 共享同一显式 K/V 空间；MLA 让持久状态共享低秩 latent，可保留 head-specific 上投影。矩阵吸收与位置旋转之间有条件：一般 RoPE 不能随意穿过任意 learned projection，DeepSeek 用 decoupled positional path 避开这一点。可与站内 [RoFormer 精读](../roformer/) 的旋转与相对位置分析对照。

### 7.5 Related-work Matrix

| 工作 | 年份 | 核心机制 | 优势 | 限制 / 与 MQA 的关系 |
|---|---:|---|---|---|
| Transformer [R2] | 2017 | 每个 head 独立 Q/K/V | 表达自由度高 | cache 与 head 数成正比，MQA 的基线 |
| MQA [R1] | 2019 | 多 Q，共用一组 K/V | 极小改动，KV 减 h 倍 | 写表示受限；TP 可能复制 |
| PaLM [R3] | 2022 | 大模型采用 MQA + RoPE | 规模化应用 | 多项架构同时改变，无同设置因果归因 |
| FlashAttention [R5] | 2022 | exact attention IO tiling | 少中间矩阵读写 | 不减少模型规定的 KV head 数；可组合 |
| GQA [R4] | 2023 | 多 KV groups + checkpoint uptraining | 控制质量/速度折中 | 多于 MQA 的 cache；转换需适应 |
| PagedAttention [R7] | 2023 | 分块动态分配与共享 | 降碎片、扩大 batch | 存储管理层；不改变 attention 参数化 |
| Llama 3 [R6] | 2024 | 8 KV-head GQA | 开放模型应用证据 | 不能由采用本身证明全局最优 |
| DeepSeek-V2 MLA [R8] | 2024 | joint latent + decoupled RoPE | 紧凑持久状态，head-specific 上投影 | kernel、矩阵吸收与并行更复杂 |
| GTA / GLA [R9] | 2025 | tied state / grouped latent，hardware-aware kernel | 平衡 IO 与 TP，给出模型和 serving 实验 | 硬件、kernel 版本、规模依赖 |
| GQLA [R10] | 2026 | 同权重允许 latent / grouped cache 两种执行 | 硬件适配的研究方向 | v1 硬件性能主要是 roofline；实际 kernel benchmark 待补 |
| GVA [R11] | 2026 | 缓存 grouped V，用 learned map 重构 content K | 进一步少 cache，保留不同索引映射 | 350M 单规模，decode 与 serving 性能未报告 |

{{diagram:evolution}}

## 8. Current State / SOTA：截止 2026-10-27 的证据地图

“最快 attention”至少需要指定硬件、batch、上下文、query length、TP、dtype、模型质量和比较 kernel 版本。本文没有找到能把 MQA、GQA、MLA 及最新变体在全部这些条件下统一排名的可靠公开证据，因而不指定一个跨设置的 SOTA 冠军。以下是已检索一手材料中的具体进展。

**GLA / GTA（2025）。** “Hardware-Efficient Attention for Fast Decoding” 在 1.47B、FineWeb-Edu 模型设置中，报告 GTA PPL=10.12 vs GQA=10.20；GLA 下游平均 accuracy=60.0% vs MLA=59.1%，PPL=10.21 vs 10.25。optimized decode kernel 在部分条件下快于 FlashMLA 1.2–2×，特别是 query length≥2；原比较明确绑定 **2025-03-28 版本的 FlashMLA**，线上 serving 最多约 2×也是其特定实验结果。[R9]

2026-10-27 检查的官方仓库仍提供 modeling，kernel 指向 FlashAttention 的 Hopper 路径，要求 Hopper GPUs、CUDA≥12.3，建议12.8。这里的“开源”指可检查源码，不表示本地 Apple CPU 具备对应 GPU 环境，也不表示论文的 peak speed 能直接迁移到 HTP/NPU。

**GQLA（2026-05，v1）。** 提出同一训练参数可走 compact latent 的 MQA-absorb path，或 grouped expanded-cache 的 GQA path。其 Llama-3-8B 转换例子报告 latent path cache 为原 GQA 的28.125%。但 Limitations 明确：H100/H20 的硬件结论主要来自 roofline；真实 kernel 对比仍待做，继续预训练结果部分仍为预测。[R10] 本文不把摘要里的 near-ideal MTP gain 当作已实测吞吐。

**GVA（2026-09，v1）。** 在约350M、30B FineWeb-Edu token 的实验中，DRoPE-16 variant 五任务平均 accuracy=44.35，GQA=44.36，MLA=43.88；按三种 random seed 的 benchmark 结果取平均。理论持久状态比 matched GQA 少约45–47%。但同一论文明确没有报告 fused decode throughput、serving peak memory 或 batch capacity，kernel 正在测试，release 仍是计划。[R11]

这组新材料再次说明：**紧凑状态、质量接近和真正低延迟是三个需要分别验证的命题。** 不能把 GVA 的44.35与 GLA 的60.0横向比高低：模型规模、训练预算、任务集合、评测协议不同。

## 9. Implementation Notes：从公式走到 runner

### 数据布局与 per-op 检查

1. **投影与 cache write。** 把 hKV 从 h 调至1或G，需要匹配训练权重。native runner 持有 [L,B,hKV,T,k] 或等价布局；不能靠裁掉已有 checkpoint 的 head 实现数学等价。
2. **QK。** 按 query head 找所属 KV group；MQA 全部映射到0。同组 KV tile 读入后应被多个 Q 复用，避免物化 h 份 `repeat`。
3. **softmax。** 仍对每个 query/head 在可见 token 轴独立归一化；local mask、padding mask 和 causal boundary 必须一致。
4. **weighted V 与 output。** 每个 query head 的聚合不同，输出投影仍保留 h。K/V 共享不意味着输出可省为一个 head。
5. **cache 生命周期。** 用已分配 token slot 更新 K/V；真实长度与可见窗口分开，不能把 padding 容量误当有效位置。prefix/beam sharing 需要独立处理所有权与写入语义。

Mesh TensorFlow 的固定 commit 是实现参考，不是被恢复的2019实验 release。其 `memory_heads_dims` 与 query heads 分离，可以表达 MQA/GQA；`shared_kv=True` 则有另一个含义：K 和 V 共享投影/表示。[R12] **跨 head 共享 K/V 与让 K=V 是不同操作。**

### RoPE、量化与 tensor parallel

MQA 可以与普通 RoPE 组合：按 token position 旋转 Q 与共享 K，K 在写 cache 时旋转一次，V 保持原样。同组 query heads 使用相容的位置约定。RoPE 不改变 token slot 数；多模态 position IDs 的数值跨度也不代表 KV 更短，可对照 [VRoPE 精读](../vrope/)。

cache INT8/INT4 需要独立评估 K 的 logit 误差和 V 的聚合误差。容量缩小是 shape 与 bytes 的结论，精度是否保住取决于真实校准、scale 粒度、异常值与各层敏感性，本文交互不生成“位宽 → 准确率”的虚假曲线。

head-axis tensor parallel 分割 query heads 时，MQA 的单组 KV 常要在参与设备上复制，或通过其他方式传输。GQA 可按 group 分片，但若 TP 超过 KV group 数也可能复制。总 cluster cache、per-device cache 和通信必须分别计数；单卡 h 倍节省不能直接套到每卡。

### 面向端侧与服务端的测量

端侧 b=1 时参数读取的1/b项很大，小模型/短上下文也可能被 dispatch、RPC、转置或 kernel 不支持拖慢。应把 TTFT、稳定 TPOT、各 op latency、实际 memory traffic、cache 峰值放在同一组 workload 下观察。对 HTP/NPU，要确认 shared-head broadcast 由计算 kernel 直接支持，而不是退回 host repeat 或额外 tensor copy。

服务端 b 较大时，MQA/GQA 节省的 cache 可能容纳更多请求，改变吞吐；为了归因，先比较固定 batch，再比较固定 memory budget 下的最大 batch。两种结果都有效，但回答的问题不同。

本文没有执行训练或 accelerator benchmark；[CPU mechanism evidence](assets/evidence/mechanism-checks.json) 与 [可运行验证程序](assets/evidence/verify_mechanism.py) 只覆盖因果全序列/逐步等价、参数守恒及论文数字重算。

## 10. My Takeaways / Open Questions

<div class="callout interpretation"><strong>个人判断：</strong>MQA 最持久的贡献，是把“读历史的视角数”与“写历史的表示数”解耦，让 inference IO 成为模型架构的一等约束。今天看 GQA、MLA 或 grouped latent，仍然可以用这两个数与 cache 内容来还原机制。</div>

我认为阅读这篇论文时最值得记住的是三个账：**状态账**决定每 token 保留多少元素；**运算账**决定各 query 如何访问与聚合它们；**执行账**决定这些共享能否在真实 kernel 和硬件上兑现。三者分开，才能解释为什么 cache 省32倍而模型未必快32倍，也能识别“理论更紧凑、实测尚未完成”的新论文。

两个值得继续追的方向：第一，给 query 更灵活的检索空间，能否在小的 shared state 上恢复 MHA 的表示自由度？MLA/GVA 的 per-head map 属于这一方向。第二，同一套训练权重能否提供不同持久 cache 表示和执行路径，按硬件/并行拓扑选择？这需要同时考虑转换精度、cache migration、kernel 与 serving 调度，不能只看 roofline。

对实际选型，我会先保留任务质量约束，再决定 KV groups、dtype 和 cache layout，最后用目标设备的真实 TTFT/TPOT 与容量验证。MQA 是可取的设计点，GQA 是可调折中，latent 是不同的状态表示；它们的选择应由模型训练和系统测量共同决定。

## 11. References & Source Record

**Original Paper**

- **[R1]** Noam Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need*. 2019. [arXiv:1911.02150v1](https://arxiv.org/abs/1911.02150v1). [PDF](assets/original/1911.02150v1.pdf) · [TeX](assets/evidence/main.tex) · [完整原表与公式清单](assets/evidence/inventory.json)。

**Related Papers / Primary Sources**

- **[R2]** Vaswani et al. *Attention Is All You Need*. 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)。原论文其余前置工作见 [original bibliography](assets/evidence/main.bib)，包含 Bahdanau attention、Liu et al. long-sequence summarization、Zhang et al. Average Attention Network、Povey et al. time-restricted attention。
- **[R3]** Chowdhery et al. *PaLM: Scaling Language Modeling with Pathways*. 2022. [arXiv:2204.02311v5](https://arxiv.org/html/2204.02311v5)，Architecture。
- **[R4]** Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. EMNLP 2023. [arXiv:2305.13245v3](https://arxiv.org/html/2305.13245v3)，§2及checkpoint/uptraining/groups实验。
- **[R5]** Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)。
- **[R6]** Grattafiori et al. *The Llama 3 Herd of Models*. 2024. [arXiv:2407.21783v3](https://arxiv.org/html/2407.21783v3)，模型架构中的8 KV heads。
- **[R7]** Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) · [vLLM](https://github.com/vllm-project/vllm)。
- **[R8]** DeepSeek-AI et al. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. 2024. [arXiv:2405.04434v5](https://arxiv.org/html/2405.04434v5)，MLA 方法。
- **[R9]** Zadouri, Strauss, Dao. *Hardware-Efficient Attention for Fast Decoding*. 2025. [arXiv:2505.21487v1](https://arxiv.org/html/2505.21487v1) · [Official code, pinned commit](https://github.com/Dao-AILab/grouped-latent-attention/tree/17889dbb4612182a4b7b92315ca11bf76ddaadef)。
- **[R10]** Fanxu Meng. *GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding*. 2026. [arXiv:2605.15250v1](https://arxiv.org/html/2605.15250v1)，Methods 与 Limitations。
- **[R11]** Vishesh Tripathi, Abhay Kumar, Ramsha Khan. *Grouped Value Attention: Efficient KV Caching via On-Demand Key Reconstruction*. 2026. [arXiv:2609.13285v1](https://arxiv.org/html/2609.13285v1)，Abstract、Results 与 Limitations。

**Implementation / Additional Resources**

- **[R12]** TensorFlow Mesh TensorFlow [attention.py，固定 commit](https://github.com/tensorflow/mesh/blob/e6798a2610a2c2f4c4cd236d8214422cb1ecc00a/mesh_tensorflow/transformer/attention.py)，`memory_heads_dims`、`shared_kv`。
- [Source retrieval record](assets/evidence/retrieval.json) · [CPU validation](assets/evidence/mechanism-checks.json) · [Reading source](assets/evidence/reading-source.md)。
- 站内延伸：[RoFormer](../roformer/)：旋转与共享 K 的位置语义；[VRoPE](../vrope/)：多模态位置跨度与 KV token 数的区别。

**Updated 2026-10-27：**首次精读发布；包含全部3张原表与6段原代码、公式重构、数字核对、交互教学与截至当日的一手扩展调研。未恢复原论文缺失的完整实验配置；没有声称复现实验性能。
