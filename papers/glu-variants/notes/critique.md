# GLU Variants supporting note

[完整精读](../index.html)

## 实验审计与批判：哪些结论稳，哪些仍待验证？

| 维度 | 判断 | 下一项有价值的验证 |
|---|---|---|
| Novelty | GLU / Bilinear 非全新；贡献在变体家族与 Transformer FFN 的预算匹配验证 | 控宽度、控参数、控实际训练时间分别对照 |
| Correctness | 前向定义清晰；2/3 对主要矩阵预算成立 | 避免忽略 elementwise、bias、rounding 和融合差异 |
| Experiment quality | 同篇八变体对照有价值，短训练四次重复 | 长训练与微调多 seed、报告 CI 和 paired difference |
| Ablation | 包含线性、硬 gate、平滑 gate，但未分离交互结构与宽度作用 | 固定 H 与 matched H 两组；gate/up 共享或独立权重 |
| Generalization | 只覆盖该 encoder-decoder、C4 去噪和 dev 任务 | decoder-only、多规模、多数据、相同训练预算 |
| Engineering / hardware | 几行可实现；硬件 latency 并未对照 | 大 M、小 M、多 dtype；fusion 与 unfused 的 per-op profile |
| Memory / scalability | 权重匹配；激活驻留和异常值是隐藏成本 | 峰值内存、训练留存、MoE dispatch 和乘法分布 |
| Reproducibility | 原公式和相关代码公开，实验配置可部分追踪 | 八变体 checkpoint、seeds、precision、aggregate 明细 |
| Causal claims | dropout 解释和机制解释没有完整因果隔离 | 预训练 dropout × 微调 mixture 的二维控制实验 |

**Strengths。** 修改小、定义明确、预算逻辑透明；同时报告预训练与多组下游结果。Bilinear、GLU、ReGLU 的加入让读者看到“相乘”和“平滑激活”不是一回事。

**Weaknesses。** 一个模型设置，长训练和微调的重复不足；没有逐变体吞吐、精度和 memory；average 的复算细节缺失。报告 task-best dev checkpoint 也限制了同一部署 checkpoint 的解释。

**Hidden trade-offs。** 少一些 hidden channel 换来多一路乘法；乘积保留负值并产生更复杂的分布；训练质量收益可能与激活 IO、量化 outlier 或硬件映射冲突。强硬件融合可以改善执行效率，但不替代模型质量对照。

**Unanswered questions。** 哪些数据分布更受益？优势来自交互、优化还是正则化？GEGLU 与 SwiGLU 的差异是否跨规模稳定？在相同端侧延迟而非相同 GEMM FLOPs 下最佳方案是否变化？



## 我的见解与下一步：用什么实验把解释变成证据？

**Interpretation。** 这篇论文让我更重视“相同预算怎样分配”的问题：普通 FFN 把预算放在较宽单路，门控 FFN 把预算放在较窄的两个读取方向及其乘积。它提供的是一个有收益的结构选择，而不是低成本激活替换的万能规则。

**方向一：分离函数、结构与预算。** 同时训练两套比较：固定 H 测多加投影的收益，固定参数 / GEMM 测预算重分配的收益。保持 seed、token 顺序、dropout 和 checkpoint 选择一致，报告长训练和下游 paired CI。加上共享 `W=V`、独立 `W,V` 和 Bilinear，才更接近“交互是否关键”的检验。

**方向二：把 loss 与硬件成本放到同一曲线。** 先做 exact 实现对照：unpacked、packed、SiLU-product 融合、producer-consumer tile fusion；分别测 prefill / decode，再比较相同 TTFT / TPOT 约束下的质量。若基础计算图仍未映射稳定，不应先用稀疏近似遮盖问题。

**方向三：沿乘法追踪精度损耗。** 在真实标定语料上记录 gate、up、product、down 的分位数、饱和和误差传播。比较 per-tensor / per-channel、敏感层保高精度及实数等价 rescaling。只有逐 op 证据才能区分 checkpoint 能力与转换链路损失。

**方向四：把机制解释做成可证伪实验。** 梯度调制和负 gate 的写回作用是可检验的假说：统计负 gate 的频率与贡献，做受控 clamp / channel intervention，再观察 loss 与任务变化。2026 的 Weakening Neurons 提供了这一视角的研究素材；不能把“观察到某些抑制方向”直接等同于全部 SwiGLU 收益的原因。

**站内连接。** [Transformer 精读](../../transformer/) 解释 FFN 与 Attention 在残差流中的位置；[FlashAttention](../../flashattention/) 说明相同 FLOPs 为什么可能有不同 IO；[MQA](../../mqa/) 与 [GQA](../../gqa/) 则处理 KV 共享，与本文的 FFN 结构相互独立。
