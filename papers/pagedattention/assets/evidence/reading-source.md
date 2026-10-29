<p class="eyebrow">SOSP 2023 · LLM SERVING · PAGED KV CACHE · 2309.06180 V1</p>

# PagedAttention：把 KV Cache 的容量，变成可用的并发

**Efficient Memory Management for Large Language Model Serving with PagedAttention**

PagedAttention 用逻辑块表连接不连续的物理 KV 块，按需分配、共享前缀并在写入时复制；vLLM 再把释放出来的容量交给迭代调度器，容纳更多同时活跃的请求。

Woosuk Kwon、Zhuohan Li（共同一作）、Siyuan Zhuang、Ying Sheng、Lianmin Zheng、Cody Hao Yu、Joseph E. Gonzalez、Hao Zhang、Ion Stoica。机构：UC Berkeley；Ying Sheng 另属 Stanford University，Cody Hao Yu 为独立研究者，Hao Zhang 为 UC San Diego。

**SOSP 2023** · v1 发布 **2023-09-12** · DOI **10.1145/3600006.3613165**<br>
精读版本：**2309.06180v1.pdf，16 页** · 阅读 / 更新 / 扩展调研截至：**2026-10-29**

<div class="source-downloads"><a href="https://arxiv.org/pdf/2309.06180v1">Paper PDF · v1</a><a href="https://arxiv.org/abs/2309.06180">arXiv</a><a href="https://doi.org/10.1145/3600006.3613165">ACM DOI</a><a href="https://github.com/vllm-project/vllm">Official code</a><a href="https://sky.cs.berkeley.edu/project/vllm/">Project</a><a href="https://blog.vllm.ai/2023/06/20/vllm.html">Author blog</a></div>

## 阅读地图与 TL;DR

**一句话。** 论文把操作系统分页的思想搬到 LLM KV Cache：历史 token 的逻辑顺序保持连续，存储位置可以分散，attention kernel 通过块表找到每一段历史。

**一段话。** 生成结束时间未知，连续 KV 缓冲往往提前预留远大于实际长度的空间，还受分配器碎片影响。PagedAttention 把每条序列切成固定 token 数的逻辑块，映射到可复用的物理块；只为已经需要的 token 分配空间，相同因果前缀通过引用计数共享。vLLM 联合迭代调度、抢占与融合 kernel，使更多请求在同一批里使用同一组模型权重。论文在特定模型、硬件和长度分布下报告相对强基线的 2–4× 概括性吞吐改善；这一倍数不能直接迁移到今天的 vLLM、短上下文或计算受限的负载。

**五个关键点：**

1. **节省的是预留和碎片，不是每个 token 的 KV 字节数。** OPT-13B 一个 FP16 KV token 仍约 800 KiB；分页不把它压缩。
2. **注意力仍然是精确的全历史 causal attention。** 每轮 decode 遍历有效历史块；分页不提供滑窗、稀疏或近似计算。
3. **共享靠语义与生命周期。** 相同 token 文本不保证 KV 相同；完整因果前缀、位置与模型配置一致才有共享基础。共享尾块被修改时才 COW。
4. **更大的可用 batch 可以抵消更慢的单个 kernel。** 原 attention 微基准慢 20–26%；系统吞吐增加来自容量与调度收益，不能解读为该 kernel 本身更快。
5. **2023 原型与 2026 实现分开读。** 论文的预定义前缀共享、FCFS 与 CPU swap 策略，不等于今天的自动前缀缓存、混合 KV 管理器与所有 attention backend。

正文按原论文 **Abstract → §1–§10** 顺序展开。其后单列工程重建、扩展研究与我的判断。v1 没有 Appendix；全文共有 **19 张编号原图、1 张编号原表、4 个编号公式**，全部保留。生成示意图、交互、算术重算均明确标识。

<div class="callout">证据范围：实验性能均为论文报告；本次没有本地 GPU 推理复现。本地验证覆盖公式与表格核对、确定性算术、页面交互、资源和多端显示。曲线未标出的数据不伪造为精确数字。</div>

## Abstract — 容量问题怎样变成吞吐问题

作者观察到，LLM 服务的有效 batch 往往先被 KV Cache 的可分配容量限制。模型参数常驻，但 KV 随请求和生成长度增长；连续分配的预留与碎片使大量显存无法承载实际 token。PagedAttention 在 attention 的访存路径中加入块映射，vLLM 在服务层管理这些块，并在多输出请求间共享它们。

摘要说 vLLM 相比当时 FasterTransformer 和 Orca 达到 2–4× 吞吐提升，尤其长序列、多输出场景收益明显。这是跨实验的概括，不是所有横轴位置上的固定倍数。正文还有最高 22× 相对 FasterTransformer 的结果，但其基线调度器不同，不能全部归因于分页。

重要贡献可以拆成三层：**地址组织**让物理块按需增长，**生命周期管理**让分支共享与复制可控，**迭代调度**把新增容量转为活跃请求。只实现其中一个 CUDA kernel，并不自动得到论文的系统结果。

## §1 Introduction — KV Cache 是动态工作集

### Figure 1 — 常驻权重与增长的 KV

{{FIGURE:1}}

**如何阅读。** 左图把 A100 上服务 13B 模型的内存分成常驻参数、按请求分配的 KV 与短暂激活。右图从可用 batch 的增长方向比较 KV 占用：传统系统较早触及容量边界，vLLM 更接近实际 token 的增长轨迹。

这不是说参数不重要，而是同一模型的权重大小固定，服务并发的主要可调部分是 KV。OPT-13B 背景例子中，KV 约 800 KB/token，单个 2048-token 请求约 1.6 GB；精确二进制单位在后面重算。请求数量乘上每请求长度，远比一次 forward 的临时张量更容易持续积累。

作者的洞察是：即便设备还有很多“已预留但未存 token”的显存，服务端也可能无法接纳新请求。GPU 上的数据没有物理改变，**对容量的组织方式**已经改变吞吐上限。右图表达这一因果方向，并不是对任意硬件拟合出的函数。

### Figure 2 — “分配了”与“有效使用了”的差距

{{FIGURE:2}}

条形图表示 §6.2 实验期间 KV 管理区域的平均组成。Token 是有效 KV；Reserved 是为未来增长预留；Internal fragmentation 是分配粒度造成的块内空余；External fragmentation & others 是外部碎片及其它开销。分母是图中对应的 KV 管理空间，不能把 96.3% 写成整张 A100 的模型加激活总利用率。

| 系统 | Token | Reserved | Internal fragmentation | External fragmentation & others |
|---|---:|---:|---:|---:|
| Orca (Max) | 20.4% | 13.3% | 57.3% | 8.9% |
| Orca (Pow2) | 26.8% | 17.9% | 13.6% | 41.6% |
| Orca (Oracle) | 38.2% | 25.2% | 0% | 36.6% |
| vLLM | 96.3% | 图未逐项标值 | 图未逐项标值 | 图未逐项标值 |

以上为**原图标注的数字转录**；前三行因一位小数舍入不必正好等于 100%。vLLM 剩余合计 3.7% 可以由 100−96.3 算出，但不能据此拆出每类精确占比。“低于 4% 浪费”是这项实验的观测；固定块设计提供的是每条未共享序列最多一个未填满尾块的结构性质。

## §2 Background — 自回归、attention 与 batching

### §2.1 Transformer：先确定需要缓存什么

{{EQUATION:1}}

式 (1) 把序列概率分解为各 token 在已有前缀条件下的概率乘积。\(x_{<i}\) 是历史 token；生成服务逐步增加这个条件，而非一次知道整个输出。训练可以并行处理已知目标序列，在线推理必须等前一轮采样结果才能确定下一轮输入。

{{EQUATION:2}}

式 (2) 用列向量约定：输入 \(x_i\in\mathbb{R}^{d_{\rm model}}\)，投影矩阵 \(W_q,W_k,W_v\in\mathbb{R}^{d\times d_{\rm model}}\)，输出每个 head 的 \(q_i,k_i,v_i\in\mathbb{R}^{d}\)。多头在实现中另有 head 维。历史 K/V 不依赖未来 token，因此可缓存；本轮 Q 与当前输入有关，用完即可。

{{EQUATION:3}}

式 (3) 的分子是当前 Q 与历史 K 的缩放点积再取指数，分母在有效历史 token 上归一化；输出是 V 的加权和。单 head 的 scores 为长度 \(i\) 的向量，输出仍为 \(d\) 维。因果 mask 排除未来；序列最后一个物理块中尚未写入的槽也必须排除。

因此 **KV Cache 存储顺序可以变，逻辑 token 顺序不能变**。地址转换不能改位置、前缀或 mask，否则不再计算同一模型的 attention。精确是数学语义上的精确；不同归约顺序与浮点 kernel 不保证输出逐 bit 相同。

### §2.2 两个阶段：prefill 不是“很多次 decode”

Prefill 对完整 prompt 并行计算各层 K/V，并产生用于采样第一个输出 token 的 logits。Decode 每轮只输入新得到的 token，却要读取已有历史的 K/V。论文的 prefill 可以使用常规 attention kernel，例如 FlashAttention；PagedAttention 重点解决增量访问非连续 KV。

一个容易混淆的时间点：**刚采样出的第一个输出 token 还没有自己的 KV**。它的 KV 要等下一轮把这个 token 输入模型才生成。这里的“已有 token 数”应指已经执行过模型计算、已存 KV 的数量。

计算形状不同也导致性能不同：prefill 有较大 GEMM，decode 低 batch 时更容易受权重与 KV 的访存限制。只报告统一平均“每 token 延迟”容易遮蔽两阶段差异。

### §2.3 batching：权重读取如何被多个请求摊薄

在一个迭代中把不同请求的当前 token 一起执行，可复用权重读取并提高矩阵计算利用率。请求输入、输出长度各不相同；固定整请求 batch 会等最长请求完成，iteration-level batching 则每轮淘汰结束的请求并接纳新请求。

论文承接 Orca 的迭代调度思想。PagedAttention 的贡献是让迭代调度不再被过大的连续预留限制；二者互补。有效活跃请求增加不一定等比例提高吞吐：batch 很大、上下文很长时，KV 读取和计算也可能成为新瓶颈。

## §3 Memory Challenges — 三种浪费的来源

### §3.1 Existing memory management 与 Figure 3

{{FIGURE:3}}

从左到右读各请求占据的连续区间：实心 token 格是真正存有 KV 的位置，其后的空余是尚未使用但已经排他分配的空间；不同大小连续区间之间还可能存在不能满足下一次申请的孔洞。

**Reserved** 源于输出终点未知，系统先留未来位置；**internal fragmentation** 源于分配器把申请向上取整；**external fragmentation** 源于空闲总额虽足够，却找不到足够大的连续区间。论文分析基于它为 Orca 自行实现的 buddy allocator 假设，而非已验证 Orca 原实现的内部行为。

一个输出 25 token 的请求，在 Pow2 例子中预留为 32。这减少 Max=2048 的过量预留，但并未变成 token 级按需分配，更未消除外部碎片。Oracle 预知真实输出长度，也仍须为完整输出排他保留连续区间。**预知长度是研究基线的优势，不是部署时能普遍获得的信息。**

图中特意提醒：同一个词在不同位置、不同前缀下的 KV 可能不同。共享不能只按词面去重，必须遵守自回归条件。

## §4 Method — 地址映射、共享与调度一起工作

### Figure 4 — vLLM 的服务链路

{{FIGURE:4}}

请求先进入 scheduler，再由 KV Cache manager 分配和跟踪块。Scheduler 产出本次迭代的 token 与块表；worker 执行模型，PagedAttention kernel 根据块表访问 GPU KV；采样结果返回给调度器，进入下一轮。

图中控制路径与数据路径要分开看：调度器决定谁参与及块的归属，GPU kernel 计算实际张量。块表是它们之间的契约。如果释放发生在 GPU 尚未读完旧块时，即使数学公式正确，下一请求也可能读到覆盖后的数据；后文会把这类生命周期约束标为工程推论。

### §4.1 PagedAttention — 固定 token 块，完整历史归一化

{{FIGURE:5}}

每个逻辑块存 \(B\) 个 token 的 K/V。物理块可以不连续；图中逻辑块仍按历史顺序组成完整 attention 输入。箭头表示查表后的读取，不表示只计算一个局部窗口。

单 head 可把块写成 \(K_j,V_j\in\mathbb{R}^{d\times B}\)，Q 为 \(d\times1\)，scores \(q_i^\top K_j/\sqrt d\) 为 \(1\times B\)。每块分子可以并行计算，但归一化分母要跨所有有效块合并。

{{EQUATION:4}}

<div class="callout warning">原式审计：v1 的 PDF 与 TeX 把式 (4) 的分母写成先将一个块的 logits 求和、再 exp。按字面不能得到式 (3) 的逐 token softmax。上方保留原式；下面单列修正推导，不静默改写作者公式。</div>

**修正后的数学形式（本精读推导）。** 令 \(s_j=q_i^\top K_j/\sqrt d\)，\(\mathbf1\in\mathbb R^{B\times1}\)，并把无效位置设为 \(-\infty\)：

{{DERIVED:block-softmax}}

与原式的区别是 **exp 先逐元素作用于 logits，再乘 \(\mathbf1\) 求和**。如果两个块各有两个有效 token，四个 logits 全为 0：正确分母为 4，每个权重为 1/4；原式字面分母为 2，各权重为 1/2，总和变成 2。这个反例不需要任何性能测量。

原 TeX 还让式 (3)、(4) 使用相同的 equation label，部分正文交叉引用因此错位。设计意图仍是全局 dense attention；审计应把排版问题与算法意图区分开。

<details><summary>展开：跨块稳定归约为什么仍然等价</summary>

对一个块计算局部最大值 \(m_b\)、指数和 \(\ell_b\)、未归一化加权和 \(u_b\)。已有累积状态为 \((m,\ell,u)\)，合并时取 \(m'=\max(m,m_b)\)，分别用 \(e^{m-m'}\) 与 \(e^{m_b-m'}\) 缩放旧状态和新块，再相加，最终输出 \(u/\ell\)。

{{DERIVED:online-softmax}}

这解释了非连续块如何保持全局 softmax；它是为理解而给出的稳定归约推导，并非声称 2023 kernel 逐行使用这段算法。不能对每块单独 softmax 后直接相加，因为各块本来的归一化质量不同。

</details>

### §4.2 KV Cache manager — 逻辑连续，物理可复用

{{FIGURE:6}}

Figure 6 中，逻辑块表依次指向物理块 **7、1、3**。相邻逻辑块并不要求相邻物理地址；kernel 用“逻辑块号 → 物理块号”，再加块内 offset 找到 token 槽。

物理 KV 池本身可以一次申请为连续的大区域，再切成等容量块。这里的“非连续”是**某条请求拥有的块**不连续，并非每个 block 都通过一次独立 CUDA 分配得到。固定块便于 free list 回收，也避免大连续申请造成的外部碎片。

只要需要新块时才分配，每条未共享序列最多一个尾块未满，因此最多浪费 \(B-1\) 个槽。较大的 B 会减少表项和搬运次数，同时增大尾块浪费与 COW 复制量；不存在无需看 workload 的最佳块大小。

### §4.3 Decoding — 一个具体的分配时序

{{FIGURE:7}}

两个请求各维护逻辑块表，并从公共物理池取得不同块。图中的不同 token、块表箭头与物理位置共同说明：调度器可以把长度不同的序列放进同一迭代，无需把它们整理成连续大缓冲。

按 Figure 6/7 的思路取 \(B=4\)、prompt 长度 7。Prefill 后已有 7 个 KV，表为 `[7,1]`，第二块已用 3 槽；下一轮把首个输出 token 输入模型，填满第 8 槽，**无需新块**；再下一轮才分配物理块 3，表变为 `[7,1,3]`。分配时机取决于将要写入的 KV 槽，而不是采样后的字符串长度显示。

下图把调度、分配、GPU 完成和释放串起来，强调控制路径的依赖。

{{DIAGRAM:execution}}

### §4.4 多输出与共享 — fork、append、free

**Parallel sampling。** 相同 prompt 产生多个独立输出，每个序列共享已经计算好的 prompt KV；只有分叉后的后缀需要各自存储。

{{FIGURE:8}}

Figure 8 从共享状态读到追加状态：fork 后，两条逻辑块表指向同一批物理块，引用计数增加。若尾块尚未填满且引用数大于 1，一条分支追加时先分配新块、复制旧块内容、改自己的映射，再递减旧块引用计数。另一分支若已经独占旧块，可直接原位写入，不必再复制一次。

完整只读前缀块不因后续生成而复制。COW 的单位是块，不是整个 prompt；代价受 B 影响。下面的交互可逐步查看分配、fork、COW 和释放，状态是确定性的语义示例，不是 GPU benchmark。

{{INTERACTIVE:blocks}}

**Beam search。** 宽度 \(k\) 的 beam 每轮从 \(k|V|\) 个候选扩展里保留 top-k。父分支可能保留多个子分支，也可能完全被淘汰。

{{FIGURE:9}}

Figure 9 的候选 0、3 被淘汰，对应物理块 **2、4、5、8** 在引用耗尽后回收；新块为 **9–12**。公共前缀块 **0、1、3** 继续共享，选中的不同分支还可能共享块 **6 或 7**。应沿“候选的父节点 → 保留序列的新块表 → 引用变化”读图，而不能把 beam 重排理解为搬运所有历史 KV。

图展示共享与 remapping 如何减少复制；top-k 的计算、logits 与采样本身并未因此消失。宽 beam 仍增加活跃序列、后缀 KV 和候选筛选成本。

**Shared prefix。** 论文讨论服务提供方预定义并缓存的公共前缀，例如翻译用 few-shot 示例。

{{FIGURE:10}}

Figure 10 先给共同翻译示例，再连接不同用户输入；相同前缀的 K/V 可被多个请求引用，减少重复计算与存储。读图时共享范围应到分叉点截止，而不是把相同末尾词或相同句子局部也当作可直接复用。

这与现代 APC 的自动查找有关，但不是同一实现。论文这里没有完整介绍基于父 hash、token 块及 LoRA/多模态等额外键的缓存体系。Mixed decoding methods 说明上述策略可组合；共享状态必须由同一块管理器协调。

### §4.5 Scheduling & preemption — 内存不足时的选择

原型使用 **FCFS**，优先服务早到请求；内存不足时优先抢占最新请求。一次 parallel sampling 或 beam 请求形成一个 sequence group，采用 gang scheduling，组内分支一起安排，以免一部分输出阻塞整个请求。

Dense attention 每轮仍读取所有历史，因此论文采用 **all-or-nothing eviction**：恢复生成前，整条序列所需历史都必须可访问。它不像 CPU 程序只访问少数最近页面，不能直接套用普通 LRU page working-set 的直觉。

两条恢复路线是：

- **Swap。** 把被抢占组的 KV 搬到 **CPU RAM**，恢复时搬回 GPU。论文不是 SSD 换页。抢占后暂停接纳新请求，直到被抢占组完成；其 CPU swap 池按 GPU KV 池容量界定。
- **Recompute。** 保存 token IDs、丢弃 KV，恢复时把 prompt 加已生成 token 作为一次 prefill 重算。它不重新采样已生成输出，也不是逐 token decode 回放。

这些是 2023 原型策略。CPU 侧 token/块表与 GPU 侧操作完成之间的同步、异步搬运、今天的 chunked prefill 或 KV connector，应另外检查实际实现。

### §4.6 Distributed execution — 统一块号，不等于复制同一数据

Tensor Parallelism 把各 attention head 的 KV 分给不同 GPU。Scheduler 广播本轮 token IDs 与块表，各 rank 使用相同逻辑物理块号，从各自本地 KV 池访问对应 head。块号相同表示同一组 token 槽，**不同 rank 内存中的字节不相同**。

模型层继续进行既有 collective，例如 NCCL all-reduce；无须每层再协商一次块分配。这让集中管理与并行模型计算分离。不过当 GQA 的 KV head 数少于 TP rank，可能存在 KV 复制，内存不能无条件除以 TP。论文实验主要为 OPT 的 MHA，不能直接外推所有 GQA/MLA 配置。

## §5 Implementation — 三类融合 kernel 与块 API

### §5.1 Kernel-level optimizations

原型约 **8.5K 行 Python + 2K 行 C++/CUDA**，服务接口用 FastAPI 并兼容 OpenAI API，模型执行基于 PyTorch、Hugging Face Transformers，分布式通信使用 NCCL。

| 操作 | 输入 → 输出 | 为什么融合 | 仍然付出的成本 |
|---|---|---|---|
| KV write | 本轮 K/V + slot mapping → 物理块池 | 合并 reshape 与 scatter 写入 | 地址计算、非连续目标与实际 KV 写入 |
| PagedAttention | Q + KV 池 + 块表 + 有效长度 → attention 输出 | 在 kernel 内读取块并归约，避免完整 gather 到连续临时 KV | 块表读取、分支、变长与跨块归约 |
| Block copy | 多个 source/destination 块对 → 复制后的块 | 多块 COW 复制合并到一次 launch | 被复制 KV 字节仍要读写 |

原文具体说明：block read/attention kernel 由 FasterTransformer attention kernel 改造，给每个 KV 块安排一个 warp 读取，以支持 coalesced access；同时支持 batch 内变长序列。融合 copy 避免对不连续的每个小块分别调用 `cudaMemcpyAsync`，降低多次小搬运的调用开销，但不消除数据移动。

注意三个不同的“块”：**KV cache block** 是 token 管理粒度，**CUDA thread block** 是执行调度单位，**warp** 是线程执行组。不能把 B=16 token 误解为一个 CUDA block 只有 16 个线程，也不能把 context partition 大小当 TP GPU 数量。

### §5.2 fork / append / free 的重建

下面是为说明不变量写的伪代码；不是官方实现的直接摘录。

```python
def fork(parent):
    child.table = list(parent.table)       # 只复制映射
    for block in child.table:
        refcount[block] += 1
    return child

def prepare_append(seq):                  # 为下一次 KV 写入准备
    if seq.kv_len % B == 0:
        seq.table.append(allocate_block())
    elif refcount[seq.table[-1]] > 1:
        old = seq.table[-1]
        new = allocate_block()
        enqueue_copy(old, new)            # 写依赖 copy 完成
        seq.table[-1] = new
        refcount[old] -= 1
    return slot_mapping(seq.table, seq.kv_len)

def free_after_device_completion(seq):
    for block in seq.table:
        refcount[block] -= 1
        if refcount[block] == 0:
            return_to_pool(block)
```

输入是当前序列的块表、KV 长度与引用计数；输出是可写槽位或回收的物理块。Fork 的 CPU 工作随块表项数增长，append 通常只改尾部；COW 的 GPU IO 随块字节数增长。实际实现还要处理异步执行、异常回滚、cache residency 与 allocator 接口。

**工程推论。** “refcount 变为 0”与“GPU 已不再引用该内存”必须共同成立才能安全复用；copy、write、attention read 之间的执行顺序也必须保证。上面显式的 device completion 是理解生命周期的约束，不冒充论文列出的某个同步 API。

## §6 Evaluation — 读清基线、指标与负载

### §6.1 Setup 与 Table 1

{{TABLE:1}}

**如何读。** 列是模型规模，行是硬件、总显存、参数内存、可用于 KV 的内存与最大 token 槽位；这张表描述实验容量配置，不是吞吐排名，没有 ↑/↓ 指标。175B 原表的参数内存为 **346 GB**，保留原值，不改成按 2 bytes 估算的 350 GB。

KV 份额按原表为 **30.0%、13.125%、41.25%**（算术重算）。175B 使用更多总显存，KV 空间相对宽裕；这有助于理解后面的短序列实验为什么更容易转为计算受限。表中的 15.7K/9.7K/60.1K 是容量 token 槽，不是每请求最大 context，也不是 tokens/s。

硬件是 GCP A2 的 A100：13B 用单 A100，66B 用 4×A100，175B 用 8×A100-80GB。基本、多输出与聊天实验使用 OPT；共享前缀实验使用 LLaMA-13B。论文以 FP16 KV/参数容量作背景，但没有为每一曲线列出完整 kernel、精度与并行配置清单，不能补写成统一完整配置。

这是服务系统论文，**没有训练新模型**。Training tokens、训练 loss、optimizer、learning rate 不适用；优化目标是给定模型与设备下的服务吞吐/延迟，不是模型准确率。Batch 是 scheduler 动态形成，不能把 Fig. 13 的平均 batch 当固定启动 batch。

### Figure 11 — 长度分布决定缓存压力

{{FIGURE:11}}

左为 (a) ShareGPT，右为 (b) Alpaca；横轴 input/output token length，纵轴频率。ShareGPT 长度更长且更分散，Alpaca 更集中于短输入输出。不能只拿“平均请求数”判断两者 GPU 压力。

| Trace | 平均 prompt tokens | 平均 output tokens | 到达过程与时长 |
|---|---:|---:|---|
| ShareGPT | 161.31 | 337.99 | 从长度数据构造 Poisson arrivals；通常 1 小时 |
| Alpaca | 19.31 | 58.45 | 同上；175B 实验因成本限制为 15 分钟 |

ShareGPT/Alpaca 的平均输入长度比约 **8.353×**，输出长度比约 **5.782×**（算术）。这些数据没有原始请求到达 timestamps，作者用 Poisson 过程合成到达；不是现实流量的时间回放，也未验证 burst 或多租户热点前缀。

### Metric 与 baseline fairness

论文使用 **normalized latency**：先对每个请求算端到端延迟除以它的输出 token 数，再对请求取均值。

{{DERIVED:latency}}

端到端延迟包含排队与 prefill。它既不是 TTFT，也不是稳态 TPOT；\(\frac1N\sum_i L_i/O_i\) 一般也不等于 \(\sum_i L_i/\sum_i O_i\)。短输出更容易被排队和 prefill 放大。横轴 requests/s，纵轴 normalized latency；曲线发散意味着这个到达率下服务跟不上。

| 基线 | 作者实际采用的版本 / 假设 | 它帮助比较什么 | 比较边界 |
|---|---|---|---|
| FasterTransformer | FT 执行后端 + 作者定制 request-level batch scheduler | 当时整请求 batching 的服务性能 | 与 vLLM 调度方式不同，不能视为只换分页 kernel |
| Orca (Max) | 作者自实现 iteration batching，假设 buddy；每请求按最大长度 2048 预留 | 无准确输出长度先验的连续管理 | 不是对 Orca 官方源码的复现 |
| Orca (Pow2) | 用真实输出长度向上取 2 的幂 | 减少 internal fragmentation 的强基线 | 仍用了事后真实长度，不是无先验在线算法 |
| Orca (Oracle) | 预知真实输出长度并按需预留连续区间 | 理想长度预测的连续管理基线 | 保留未来预留/外部碎片，无法普遍在线部署 |
| vLLM | 块分配、迭代调度、PagedAttention、多序列共享 | 完整新系统 | 系统收益与单因素 paging 收益有区别 |

作者在相同模型/硬件/trace 下比较，控制了很多重要变量，但 baseline 重实现与调度差异仍限制因果归因。没有模型质量训练对比，因为数学 attention 语义不变；这也不等于实验已检验所有 backend 的逐 bit 一致性。

### §6.2 Basic sampling — Figure 12

{{FIGURE:12}}

上排是 ShareGPT、下排是 Alpaca；各 panel 比较 OPT-13B/66B/175B。先在相同 normalized latency 位置读横轴支持的到达率，再看高负载下曲线是否发散。不能在不同 latency 水平随意取两点相除。

ShareGPT 上，正文报告 vLLM 支持率相对 Orca (Oracle) 提高 **1.7–2.7×**，相对 Orca (Max) 提高 **2.7–8×**；相对 FasterTransformer 最高 **22×**。这与摘要 2–4× 的概括作用不同。

Alpaca 更短，尤其 OPT-175B 的 KV 容量相对宽裕，系统更容易受计算吞吐限制，收益缩小。这是关键反例：当连续管理已经能容纳足够 batch，分页减少浪费不一定继续提升可承载速率。

### Figure 13 — 吞吐改善的中间证据是可用 batch

{{FIGURE:13}}

左 (a) 为 OPT-13B ShareGPT 2 requests/s，右 (b) 为 Alpaca 30 requests/s。纵轴是**平均同时被 batched 的请求数**，不是 speedup，也不是固定 batch size。

| Trace / 到达率 | Orca Max | Orca Pow2 | Orca Oracle | vLLM |
|---|---:|---:|---:|---:|
| ShareGPT · 2 req/s | 7 | 9.81 | 13.62 | 30.42 |
| Alpaca · 30 req/s | 7 | 43.24 | 72.75 | 132.44 |

vLLM 相对 Oracle 的平均 batch 比分别为 **2.233×、1.820×**；相对 Max 为 **4.346×、18.920×**。这支持“更有效利用容量 → 更大 batch”的解释，但 batch 比不等于最终吞吐比。Alpaca 的大 batch 比还受到请求短、到达率高与 baseline 容量策略的共同影响。

### §6.3 Parallel sampling & beam search — Figures 14–15

{{FIGURE:14}}

Figure 14 上排 parallel generation、下排 beam search，分别考察宽度/输出数量 **2、4、6**，模型为 OPT-13B、负载为 Alpaca。相对基本采样的约 **1.3×**，beam=6 时相对 Oracle 的支持率收益可到 **2.3×**。共享比例越大，复制连续历史的 baseline 越吃亏；同时更多后缀仍增加计算与存储，不能说多输出近乎免费。

{{FIGURE:15}}

Figure 15 左 (a) parallel generation，右 (b) beam search；纵轴为共享节省的 KV 块数占**关闭共享时块数**的比例。它不是全 GPU 显存下降比例，也不是吞吐改善比例。

| Alpaca · OPT-13B | 2 outputs/beams | 4 outputs/beams | 6 outputs/beams |
|---|---:|---:|---:|
| Parallel sampling | 6.09% | 8.53% | 9.79% |
| Beam search | 37.56% | 53.13% | 55.16% |

论文正文对 ShareGPT 给出范围：parallel **16.2–30.5%**，beam **44.3–66.3%**，没有这张图的逐档精确值，此处不补齐。Alpaca prompt 短，parallel 在长独立后缀中共享的占比较低；beam 在生成途中继续共享公共祖先，因此可有更大收益。Figure 15 给出内存机制的证据，Figure 14 给出系统性能证据，两者不应互换单位。

### §6.4 Shared prefix — Figure 16

{{FIGURE:16}}

LLaMA-13B 的 WMT16 English→German 翻译负载，共享 few-shot 前缀。Panel (a) 是 1 个例子、**80 tokens**，(b) 是 5 个例子、**341 tokens**；正文报告相对 Oracle 支持率分别提高 **1.67×、3.58×**。

随着前缀变长，省掉的 prefill 与公共 KV 更大。但这个干预同时减少**计算与存储**，因此不能把这两个倍数标成“仅分页”的增益。前缀由服务提供方预定义并缓存，实验没有验证现代 APC 的 hash 开销、命中分布、淘汰与安全隔离。

### §6.5 Chatbot — Figure 17

{{FIGURE:17}}

多轮聊天取 ShareGPT 的历史，将历史拼接为下一轮 prompt；OPT-13B，最终最多 **1024-token prompt + 1024-token output**。论文这里**轮间不保留 KV**，所以不能把结果解读为跨轮缓存命中的测量。

vLLM 可支持约 **2×** Orca 的请求率。三种 Orca 策略较接近，与长 prompt、buddy 分配粒度导致区间大小趋同有关。这项实验说明可变长对话下管理优势仍存在，却没有直接回答今天聊天场景的 prefix reuse、长会话 cache retention 或 first-token SLO。

## §7 Ablation — 更慢的 kernel，更大的系统收益

### §7.1 Kernel microbenchmark 与 §7.2 block size

{{FIGURE:18}}

Figure 18 左 (a) 比较 attention 微基准：context **64/128/256**，batch **8/32**。PagedAttention 相对 FasterTransformer attention kernel 的延迟增加 **20–26%**，归因于块表访问、额外分支与处理变长序列等成本。这个比例属于单个 attention operation，不能写成整个模型延迟增加 20–26%，也不能说该论文的 PagedAttention kernel 比 FT 更快。

右 (b) 扫描 \(B=1,2,4,8,16,32,64,128,256\)。ShareGPT 的最佳区域大致为 **16–128**，Alpaca **16/32** 较好，过大 block 时短序列性能急剧变差。2023 原型默认 B=16，是特定 workload 下的折中。

小块减少尾部浪费，但增加 block table、调度与搬运操作；大块减少元数据，却放大尾块空余与 COW 复制。这一消融把 kernel 效率与容量放进同一个系统曲线，不能只看某个最佳 B 就推广所有 context、GQA、backend。

### §7.3 Recomputation versus swapping — Figure 19

{{FIGURE:19}}

左 (a) 是不同 B 下重算与 swap 的开销，右 (b) 是 OPT-13B、ShareGPT、相同到达率下的服务性能。小块导致大量 PCIe 小搬运，swap 的调用/传输开销更大；recompute 在图中约为 40 ms、随 B 较平稳。较大块使 swap 更高效；B=16–64 范围内两种策略的端到端表现较接近。

<div class="callout warning">文字与图的歧义：原文称 “the recomputation overhead is never higher than 20% of the swapping's latency”。按字面应为 recompute ≤ 0.2×swap，但图中大 B 的重算约 40 ms、swap 总时延约 32 ms，并且作者随后说大块时 swap 更好，无法支持这个字面不等式。本精读保留该原句并标明不一致；不将其转述为已证明的上界，也不凭曲线填出精确表格。</div>

可操作的解释是：比较必须把 payload 和搬运次数一起算。Swap 近似有双向 \(2M\) KV 数据穿越链路，recompute 要重跑模型对历史 token 的 prefill；前者依赖带宽/传输粒度，后者依赖模型、长度与 GPU。论文没有给出可跨硬件使用的统一阈值。

## §8 Discussion — 分页不包办所有优化

原文先讨论**适用条件**：输出长度未知，导致动态申请；性能又受 GPU 内存容量限制。DNN 训练的 tensor shape 通常静态，可提前规划分配；非 LLM 的 DNN 服务若主要 compute-bound，节省容量也不提高吞吐，反而可能被间接寻址与非连续访问拖慢。因此不能把分页推广为所有 GPU workload 的通用加速。

接着作者强调**LLM 特定语义**：完整历史都要在 GPU 才能继续 dense attention，因此是 all-or-nothing swap；被丢弃的 KV 可以从 token 重算，普通 OS 页无法普遍这样恢复；内存访问与 attention 融合则缓解了间接地址开销。分页解决 GPU KV 的分配与共享，激活、采样、模型权重容量和算力不会自动改善。

从论文自己的消融可以看出，它主动接受一部分地址间接访问的开销，换取更好的利用率。规模扩大后，控制路径、块表与每轮通信也要考虑；固定块抽象能简化管理，不代表 host scheduler 无限可扩展。

这里最值得保留的设计原则是：**逻辑生命周期与物理存储解耦**。它使 sharing、COW、eviction 有统一表达。但不同硬件可采用不同地址机制，后续 vAttention 用虚拟地址连续、物理按需映射就是另一条路线。

## §9 Related Work — 前身与互补路线

原论文依次梳理**通用 model serving、Transformer 专用服务、Orca 对比、内存优化**。Clipper、TensorFlow Serving、Nexus、InferLine、Clockwork 等研究 batching/caching/placement/scheduling；DVABatch、REEF、Shepherd、AlpaServe 分别涉及批组织、抢占与并行模型复用。作者认为它们没有充分利用 LLM 的自回归 token 状态。

Transformer 专用系统从 kernel、batching、model parallelism 与参数共享优化推理；Orca 最相关，改善每轮请求调度，而 vLLM 增加可容纳工作集。Swapping/recomputation 也用于训练，但训练目标与 online serving 不同；FlexGen 面向有限显存的权重/token state offload，不以低延迟在线服务为主要目标；OLLA 优化 tensor 生命周期和位置，不提供同样的细粒度在线 KV 块管理。

与 **FlashAttention** 的关系尤其容易混淆。FlashAttention 通过 IO-aware tiling 避免显式存巨大的 attention score 矩阵，重点是计算时 HBM↔SRAM 的数据流；PagedAttention 为持久化 KV 提供块映射和生命周期管理。一个管临时 attention IO，一个管请求之间持续增长的 KV。Prefill 可以使用 FlashAttention，decode 可以使用理解 paged KV layout 的 kernel。

MQA/GQA 则减少 KV head 数，从源头降低每 token 字节；分页减少“有效 KV 之外”的浪费。两者可组合，不能把它们归成同一种压缩方法。原论文没有全面评测今天的 GQA/MLA、低精度 KV、稀疏 attention 与混合模型。

## §10 Conclusion — 成果的最小可信表述

PagedAttention 让非连续 KV 存储仍能执行完整 attention，vLLM 用块管理、共享与调度将节省的容量转为更多活跃请求。论文的证据链是：**更高有效 KV 比例（Fig. 2）→ 更大平均 batch（Fig. 13）→ 特定延迟水平下更高可承载请求率（Fig. 12/14/16/17）**，并用 Fig. 18/19 展示 kernel、块大小与恢复策略的代价。

这条链足以支持其服务系统贡献；不足以支持“任何单请求都更快”“attention 算量变小”或“今天仍固定领先 2–4×”。v1 随后有 Acknowledgement 与 64 条参考文献，没有附录或编号算法表。完整 PDF 与作者 TeX/Bib 已保存在本地 sources，参考脉络在下面的扩展研究中按一手资料核验。

## 工程重建 — KV、IO、TP 与性能拆账

<p class="eyebrow">AGENT INTERPRETATION · 与论文原实验分开</p>

### 每 token 的 KV 与尾块浪费

令 L 为层数，\(H_{\rm kv}\) 为 KV head 数，\(d_h\) 为 head dimension，s 为每元素字节，T 为已经计算 KV 的 token 数：

{{DERIVED:memory}}

OPT-13B 的背景配置 \(L=40,\ H_{\rm kv}d_h=5120,\ s=2\)：每 token **819,200 bytes = 800 KiB**，2048 token 是 **1.5625 GiB = 1.6777216 GB**。论文写 800 KB / 1.6 GB 是近似；计算器与精确推导使用 bytes、KiB、GiB，避免十进制/二进制混用。

若 B=16、T=17，要占两块 32 槽，尾块空 15 槽；这是短序列反例，浪费比例 **46.875%**。T=2048 时正好整块，没有尾部空余。故“每序列尾块最多 B−1 槽”不保证所有负载都小于 4%。

公共 prompt 的完整块可以共享，分叉点的未满块进入 COW。为了给出不依赖时序的估算，计算器假定 N 个活跃分支都已有独立后缀，**只共享完整前缀块**。它计入尾部槽，但不把 allocator 元数据、激活、workspace 或权重隐藏在 KV 小计中。

{{INTERACTIVE:memory}}

### per-op 代价如何组合

| 路径 | 主要规模 | 分页改变了什么 | 不能从论文推出什么 |
|---|---|---|---|
| KV append | 每新增 token 写入约 \(2LH_{\rm kv}d_hs\) bytes | 改变目的地址与块申请时机 | 字节数被压缩或写入免费 |
| Decode attention | 每序列扫描有效 T；计算/读 KV 随 T 增长 | 分块间接寻址、可融合读取 | 从 dense \(O(T)\) 变成稀疏/滑窗 |
| KV COW | 一次复制的 payload 约 B×每 token KV | 通常只复制共享尾块 | 与 B 无关、没有读写 IO |
| Fork / beam remap | CPU 块表项数约 \(\lceil T/B\rceil\) | 表复制代替整个历史 KV 复制 | host 调度成本消失 |
| Swap out + in | 双向 payload 约 2M，时间至少受 2M/BW 约束 | 固定块可批量安排 | 实际时延等于纯带宽下界 |
| Recompute | 历史 T 的 prefill，重做投影/FFN/attention | 用计算替代 KV 搬运 | 只重算 attention 或只保存最后一块 |
| Tensor parallel | 计算与 collective，KV 依 head 分片 | 同块表索引各 rank 本地 shard | 所有 GQA 都恰好把 KV 除以 TP |

若纯作 payload 算术，OPT-13B、B=16，一块跨所有层的 KV 为 **12.5 MiB**；复制要读写这份 payload，内存总线流量下界约 **25 MiB**。2048-token KV 为 1.5625 GiB，完整 swap out/in 的双向 payload 为 **3.125 GiB**。这不是实测时间；链路同步、传输粒度、copy 合并与有效带宽都影响时延。

### 为何慢 20–26% 的 attention operation 仍能赢

设一轮总时间由权重/FFN 等 \(t_{\rm other}\)、attention \(t_{\rm attn}\)、scheduler/通信/采样构成。即便新 attention 为 \(1.2t_{\rm attn}\)，总轮时延也不会简单乘 1.2；attention 占比与 batch/context 决定影响。更重要的是，轮内处理的活跃请求数量也变了。

吞吐的粗略关系是“每轮有效 token 数 / 轮时延”。更多 batch 可摊薄参数读取，也会增加 KV 流量与计算。Fig. 13 展示分子增长，Fig. 18a 展示一部分分母增加，Fig. 12 展示综合结果。这里不能用 batch 比除以 1.2 伪造最终 speedup，因为 kernel 形状和其它操作随 batch 改变。

### 命中检查：为什么前缀中途不同，后面的词相同也无效

现代 APC 的 documented full-block 模式可用父 hash 串联各块；缓存键还需考虑模型/adapter、多模态内容及隔离域。下面示例只演示这一模式，B=4、缓存了 12-token 因果前缀。修改一个 token 或隔离 salt，观察**连续可复用前缀**长度。

{{INTERACTIVE:prefix}}

若第 5 个 token 不同，第一个满块可命中，后续块的父前缀已不同，即使第 9–12 个 token 完全相同，也不能恢复为 12-token 命中。若第 2 个 token 不同，则第一个满块也不命中。改变 salt 会隔离整个缓存。这是机制示例，不是论文 Fig. 16 的 APC 实测，也不是对 2026 所有 partial-block 功能的穷尽。

### 今天应如何测量

| 指标 | 精确定义 / 作用 | 为什么不能用 normalized latency 替代 |
|---|---|---|
| TTFT | 到达请求至第一个输出 token，可另拆排队与 prefill | normalized latency 把前段成本摊到输出长度 |
| TPOT | 常用为 first→last 输出间隔 / (O−1)，O>1 | 不包括首 token，论文指标包括端到端 |
| ITL | 相邻输出 token 的时间间隔分布 | 均值不能揭示 chunked prefill 或抢占尖峰 |
| p95 / p99 | 每请求/每 token 的尾部分位，明确总体 | 原文只给主要平均曲线 |
| Goodput | 满足 TTFT、TPOT 等 SLO 的请求率 | 饱和吞吐不保证业务请求及时完成 |
| KV occupancy | 有效 token、已分配槽、cached ref=0、free、峰值 | “已分配”不等于有请求在用 |

合理的复现实验固定 checkpoint、GPU/TP、KV dtype、backend、prompt/output 分布与到达过程，再分别开关按需分配、共享、缓存、chunked prefill 和恢复策略。记录每轮 scheduler、block allocator、COW/copy、attention、GEMM、NCCL 与 sampling；只看总吞吐很难定位哪个调用造成回退。

## 扩展研究 — 2022→2026 的互补与替代路线

<p class="eyebrow">PRIMARY-SOURCE RESEARCH · AS OF 2026-10-29</p>

### 技术关系：这是一组不同瓶颈，不是线性换代

{{DIAGRAM:evolution}}

2022 的 Orca 让迭代调度成为核心；2023 PagedAttention 把内存容量释放给这种调度。后续工作既改 kernel，也改 prefill/decode 组织与前缀缓存。vAttention 则挑战“必须用软件块表寻址”的具体实现，而非否定物理按需分配的价值。

| 工作 / 一手出处 | 时间 | 主要问题与做法 | 优点 / 报告证据 | 边界与本论文关系 |
|---|---|---|---|---|
| [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) | OSDI 2022 | Iteration-level + selective batching | 请求可在每个迭代加入/退出 | 按需 KV 块管理是另一维；本论文用自实现基线 |
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | SOSP 2023 | 非连续 KV、块共享、COW | 96.3% 有效 KV，特定场景 2–4× 概括 | 有软件间接寻址成本，非 attention 压缩 |
| [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104) | 2023-12；2024 版本 | 程序化 LLM 调用与跨调用前缀 KV 复用 | 特定任务报告最高 6.4× | 系统/工作负载不同，不能作 2026 统一排名 |
| [DeepSpeed-FastGen](https://arxiv.org/abs/2401.08671) | 2024-01 | Dynamic SplitFuse 混合 prompt/decode | 相对历史 vLLM 报告最高 2.3× effective throughput | 关注批组织与 tail，不能归为仅 allocator 提升 |
| [DistServe](https://arxiv.org/abs/2401.09670) | 2024-01；OSDI 2024 | Prefill/decode 分离与资源规划 | 特定 SLO 下最高 7.4× requests；>90% 请求满足 SLO | KV 传输和两池资源是额外成本，不与 SOSP 倍数叠乘 |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | 2024-03；OSDI 2024 | Chunked prefill 与 stall-free 调度 | 特定配置 2.6×/3.7×/5.6× | 分别对应 Mistral-7B 单 A100、Yi-34B 两 A100、Falcon-180B PP；非统一模型 |
| [vAttention](https://arxiv.org/abs/2405.04437) | 2024-05；ASPLOS 2025 | CUDA VMM 保虚拟连续，物理按需映射 | v3 报告最高 1.23× 对比 paged kernel 路线 | 依赖 VMM 与平台特性；同容量目标，不同地址机制 |
| [FlashInfer](https://arxiv.org/abs/2501.01005) | 2025-01；MLSys 2025 | 可组合 KV layout、JIT、负载均衡、CUDA Graph | 特定集成报告 ITL 降低 29–69% | 集成 vLLM/SGLang/MLC；不是所有 2023 vLLM 负载的加速 |
| [vLLM APC 设计](https://docs.vllm.ai/en/latest/design/prefix_caching/) | 2026-10-29 核验 | 父 hash + token 块 + extra keys；缓存队列 | 自动复用可避免重复 prefill | 文档 full-block 模式与代码的可选 partial 功能要区分 |
| [Hybrid KV manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/) | 2026-10-29 核验 | Full attention、SWA、Mamba 等分组管理 | 统一 page byte 大小与各组分配需求 | 命中需跨组满足条件，不能只数一层 full-attention 长度 |
| [TorchTPU backend](https://vllm.ai/blog/2026-10-20-vllm-torchtpu) | 2026-10-20 | RaggedPagedAttention/Pallas，复用上游调度/KV 管理 | 展示分页抽象迁移到非 CUDA 平台 | 是官方特性范围，不是与 A100 严格同条件性能对照 |

这些倍数逐项来自对应论文/官方资料，**没有共同的 2026 benchmark**；模型、GPU、序列长度、SLO 与 baseline 版本不同。作者 2023 博客另报对 Hugging Face / TGI 的 24×/3.5×，属于另一组 benchmark，不混入 SOSP 对 Orca 的 2–4×。

### 当前 vLLM：固定源码点，避免把历史文档当现状

截至 2026-10-29，核验到最新发布为 **v0.32.0**（2026-10-18T06:50:48Z）。当前源码观察固定到 main commit **34091a07c7d78230e54d53d92b66b125b0f21a26**（2026-10-29T08:55:05Z），不是含糊的“最新 main”。

在这个点，V1 `KVCacheManager` 通过 coordinator 与 `BlockPool` 管理缓存；`get_computed_blocks` 查可用前缀，`allocate_slots` 申请容量，`free` 降引用。BlockPool 同时管理 free queue、引用计数与 hash 缓存。**refcount=0 的 block 可以保留可复用数据**，直到被淘汰/重用；它与 2023 简化的“free 即回池”示例在缓存语义上不同。

APC 文档说明 full blocks 的 parent hash 链，并列 extra hash keys，例如 LoRA、图像相关输入与 salt。当前源码还有可选 partial hash 功能，所以不能把“只缓存满块”写成现代 vLLM 永远成立的限制。上面的交互明确采用文档 full-block 模式。

Hybrid KV 设计处理 full-attention/SWA 等不同层的存储需求。不同 group 命中长度要取合法组合；单层历史可命中并不代表全模型可跳过同样长的 prefill。这与多模态拼接、滑窗退出和跨配置缓存的一致性直接相关。

官方 [Paged Attention kernel 文档](https://docs.vllm.ai/en/latest/design/paged_attention/) 开头明确说是 **historical implementation，不再描述今天代码**。其中历史 K layout 可写成 `[P,Hkv,d/x,B,x]`、V `[P,Hkv,d,B]`，Q 含 sequence/head/d 维；这些用于理解旧向量化布局。文档对 `PARTITION_SIZE` 的文字也有易误解之处：它对应 context partition，不能当 TP GPU 数量。

固定源码链接：[KVCacheManager](https://github.com/vllm-project/vllm/blob/34091a07c7d78230e54d53d92b66b125b0f21a26/vllm/v1/core/kv_cache_manager.py) · [BlockPool](https://github.com/vllm-project/vllm/blob/34091a07c7d78230e54d53d92b66b125b0f21a26/vllm/v1/core/block_pool.py) · [v0.32.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)

### 当前 SOTA 应如何表达

没有找到同一 checkpoint、KV 精度、TP、GPU、prompt/output、到达过程、TTFT/TPOT SLO 下的统一 2026 对照，因而不能宣称某引擎全局最快。更有证据的表述是：

- 容量轴：固定块管理、虚拟内存映射、KV 量化、GQA/MLA 各自减少不同来源的压力。
- 复用轴：APC/Radix 与跨调用程序结构决定哪些历史可跳过；命中率要按真实负载测。
- Kernel 轴：FlashInfer、FlashAttention 系列及硬件 backend 决定算子 IO 与并行效率。
- 调度轴：chunked prefill、P/D disaggregation 与恢复策略决定 TTFT、TPOT 和 goodput。
- 异构轴：hybrid KV、SWA、Mamba、多模态及 offload 使单一 full-attention 页表不足以描述全部成本。

工程选择应先识别业务瓶颈，再在固定条件下比指标。新版本号与新博客能证明功能或演进，不能单独证明 SOTA。

## 我的判断 — 贡献、隐含成本与未解问题

**Novelty 与正确性。** 最强贡献是把 KV 地址、生命周期和 batch 容量放进同一个服务抽象；它不是发明 softmax 或压缩 KV。原式 (4) 排版问题需要指出，数学等价可由正确全局归一化证明；公式审计不能替代对实际 kernel 的数值测试。

**实验质量。** 内存组成、batch 和系统性能形成了相互支持的证据链，且作者诚实展示慢 kernel 与 block size 代价。弱点是 Orca 重实现/先验长度、平均指标、Poisson 到达与缺少完整单因素拆分；结论应保持在这些条件内。

**工程复杂度。** 页表让分配更简单，却让 kernel 与 host/device 生命周期更复杂。COW copy、attention read、free/reuse 的回调和异步依赖是高优先级审查点；吞吐异常时，应先找每轮等待与对象释放逻辑，再观察算子形状。

**可扩展性与泛化。** TP 共用块号降低协调负担，但 scheduler 每轮控制成本、GQA head 复制、P/D 跨机 KV、hybrid groups 都会改变原型的成本模型。分页思想可迁移，2023 单一 OPT/MHA 路径不构成这些场景的完整评测。

**隐藏 trade-off。** 大 B 对短请求容量不利，小 B 对传输/元数据不利；更大 batch 可能提升吞吐而拉高 ITL；缓存保留能提高复用却占用可分配容量。最大 tokens/s 不一定是最佳业务体验。

如果继续研究，我会优先做以下四项：

1. **单因素实验。** 同一 scheduler/kernel 下仅切 allocator，然后分别打开共享、APC 和调度改进，避免把系统全部收益算给 paging。
2. **尾延迟与抢占。** 在 burst 到达、长短请求混合与不同 TTFT/TPOT SLO 下记录 p95/p99、recompute/swap 次数及队列时间。
3. **生命周期验证。** 检查 fork/copy/free 与 GPU completion 的所有分支，加入并发、异常、取消请求后的引用计数与 reuse 观测。
4. **现代模型成本。** 同时报告 GQA/MQA/MLA、低精度 KV、SWA/mixed groups 的 bytes/token、满块命中、实际搬运字节与各 rank 复制因子。

这些是本精读提出的方向，不是论文已经完成的实验。当前材料支持把 PagedAttention 视为重要服务系统基础，具体收益仍需用业务调用链与 per-op 测量兑现。

## References、关联阅读与复现范围

**原论文与官方材料。** [指定 v1 PDF](https://arxiv.org/pdf/2309.06180v1)、[arXiv metadata](https://arxiv.org/abs/2309.06180)、[ACM DOI](https://doi.org/10.1145/3600006.3613165)、[官方仓库](https://github.com/vllm-project/vllm)、[Berkeley project](https://sky.cs.berkeley.edu/project/vllm/)、[2023 作者博客](https://blog.vllm.ai/2023/06/20/vllm.html)。

**相关论文。** [Orca / OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)、[SGLang](https://arxiv.org/abs/2312.07104)、[DeepSpeed-FastGen](https://arxiv.org/abs/2401.08671)、[DistServe](https://arxiv.org/abs/2401.09670)、[Sarathi-Serve](https://arxiv.org/abs/2403.02310)、[vAttention](https://arxiv.org/abs/2405.04437)、[FlashInfer](https://arxiv.org/abs/2501.01005)。演进矩阵给出各自比较条件，不能把跨论文 speedup 相乘。

**当代设计与代码。** [APC](https://docs.vllm.ai/en/latest/design/prefix_caching/)、[Hybrid KV](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)、[历史 PagedAttention kernel](https://docs.vllm.ai/en/latest/design/paged_attention/)、[TorchTPU](https://vllm.ai/blog/2026-10-20-vllm-torchtpu)、[v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)。资料核验日期 2026-10-29；历史 kernel 文档不充当当前实现的保证。

<div class="related-readings"><p>Related Paper Readings</p><a href="../transformer/">Transformer → 自回归 attention 的计算基础</a><a href="../flashattention/">FlashAttention → attention IO 与精确归约</a><a href="../gqa/">GQA → 从 KV head 数减少每 token 字节</a><a href="../longformer/">Longformer → 稀疏注意力与分页的边界</a><a href="../eagle2/">EAGLE-2 → 分支验证与 KV 成本</a></div>

**可复现范围。** 本地保留原 PDF、作者 TeX/Bib、原图 PDF/PNG、完整 Table 1、图资产来源清单和本页 Markdown。所有 19 张图从作者独立图文件渲染，多 panel 仅按原序拼接，原图注以图外可读文字保留；未把 caption 截图当图片正文。两张 Mermaid 与三组交互为解释资产，算术有明确假设。

<div class="source-downloads"><a href="assets/evidence/2309.06180v1.pdf">Archived v1 PDF</a><a href="assets/evidence/reading-source.md">Reading Markdown</a><a href="assets/evidence/figure-manifest.json">Figure sources & hashes</a><a href="assets/evidence/metadata.yaml">Metadata</a><a href="assets/tables/table-01.html">Complete Table 1</a></div>

本次不声称复现任何 GPU benchmark。后续复现需要指定模型 revision、硬件/driver、engine commit、attention backend、KV dtype/TP、trace 与 SLO，并提供实际运行记录。

**Changelog · 2026-10-29。** 首次精读指定 v1；保留全部编号图表公式，审计式 (4) 与 Figure 19 的文字歧义，加入分页/COW、KV 容量和 full-block prefix 交互，扩展到固定 2026 源码点。
