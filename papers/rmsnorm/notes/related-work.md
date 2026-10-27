## Related Work & Technology Evolution：2016 到 2026

下面均为**外部一手资料**，不是原论文的实验结果。时间线表达机制演进和采用关系，不宣称每个节点严格替代前一个。

{{diagram:evolution}}

| 工作 / 年份 | 核心思想 | 优势 | 局限 | 与 RMSNorm 的关系 |
|---|---|---|---|---|
| Layer Normalization / 2016 | 单样本 feature 维均值与方差 | 不依赖 batch，适合序列 | 中心化与归约成本；placement 敏感 | 被拆分验证的直接前身 |
| Weight Normalization / 2016 | 分离权重长度和方向 | 重参数化，不用 activation batch stats | 不等价于单 token 尺度控制 | 原附录独立路线对照 |
| Group Normalization / 2018 | 分组 feature statistics | CV 小 batch 可用 | 轴与分组决定效果 | 说明“归一化轴”不可忽略 |
| Fixup / 2019 | 初始化与参数尺度安排 | 某些残差网络无需 Norm | 依赖架构与初始化规则 | 从初始化而非每步统计解决稳定性 |
| RMSNorm / 2019 | 用 RMS 保留尺度不变性 | 简单、质量／速度取舍可用 | 无平移不变性；p 估计风险 | 本文 |
| Pre-CRMSNorm / 2023 | 转换特定 Pre-LN 架构为等价 RMS／压缩表示 | 特定条件下等价，报告 1–10% 提效 | 转换前提严格，不是任意换层 | 解释为何 mean 方向在特定网络中可消去 |
| Llama / 2023 官方实现 | 在实际 decoder 用 RMSNorm | 大模型采用；FP32 statistics 明确 | 采用不等于独立因果消融 | 工业模型证据 |
| FlashNorm / 2024，v5 2026 | RMS 与后续 linear 的联合重写／融合 | 减少中间物化与 pass | 依赖投影、权重处理、融合与硬件 | 把层级简化继续推进到 IO |
| Transformers without Normalization / 2025 | Dynamic Tanh，α·x 后逐元素 tanh 与 affine | 不用激活统计归约 | 含非线性，替换需重新训练／验证 | 另一条去统计路线，不等价 RMS |
| PyTorch 2.11 / 2026 官方 benchmark | torch.compile 归约和 mixed-order 优化 | 常见 shape 接近专用实现 | 硬件、dtype、shape、方向敏感 | 当前应比较优化后的算子 |
| When Does Removing LayerNorm Help? / 2026 preprint | 检验 DyT 激活饱和与训练预算，另含 64M RMS 对照 | 提醒去统计路线依赖数据／模型规模 | 单一新预印本，非广泛共识 | RMS/LN 是对照；不能当成两者孰优的定律 |

### Pre-CRMSNorm：架构等价不等于单算子等价

[Jiang et al., 2023](https://arxiv.org/abs/2305.14858)，本文核验 v2。作者从 Pre-LN Transformer 的整体结构出发，分析可压缩的均值方向并构造等价转换；报告具体实验中 1–10% efficiency improvement。它的价值是把“经验上可替代”推进到满足条件的结构等价。不能拿这项结果证明普通含非零均值输入的 LN(a)=RMS(a)，更不能未经转换就加载任意 checkpoint。是否有 bias、affine、residual 与参数约束，必须按该文规则核对。

### Llama：实际代码怎么选？

[官方 Llama 实现](https://github.com/meta-llama/llama/blob/main/llama/model.py) 在 `_norm` 中先 `x.float()`，计算 `x * rsqrt(mean(x²)+eps)`，再转回输入类型并乘权重。统计在 FP32、epsilon 在 sqrt 内、Norm gain-only，这些都属于代码行为；原 2019 数学式没有描述这些数值选择。

Llama 的采用证明 RMS 能用于真实大规模 decoder，不能单独隔离它相对 LN 的收益，因为模型数据、FFN、位置编码等也同时不同。有关 RoPE、MHA／MQA／GQA 和 IO-aware attention 的站内文章，是与 RMS 同处一条推理链路的互补阅读。

### Dynamic Tanh：不用 reduction 的代价是什么？

[Zhu et al., CVPR 2025](https://arxiv.org/abs/2503.10622)，核验 v2。DyT 使用 `γ * tanh(αx) + β`，用可学习 α 调整输入尺度；避免 mean/RMS 的样本统计。tanh 是逐元素非线性，存在饱和，其 Jacobian 和 RMS 的全通道耦合不同。论文在自己的训练与视觉模型基准中评估替代，可视为新的路线；不是一个能保持任意 RMS checkpoint 函数的恒等变换。


## Current State / SOTA：截至 2026-10-27 的条件化判断

“哪种 normalization 是 SOTA”没有跨所有模型的唯一答案。这里把模型质量、kernel 速度与编译器能力分开，给出日期、基准和比较边界。

| 证据 | 截止与具体条件 | 可支持的结论 | 不能支持的结论 |
|---|---|---|---|
| PyTorch 官方 normalization 性能文 | 2026-04-08；2.11，H100/B200；与 2026-03-24 的 Quack trunk 对照 | 常见前向 shape 的 compile 已可接近专用 kernel | 所有 shape／backward／端到端都领先 |
| MixOrderReduction 文中实验 | B200 750W、CUDA 12.9、BF16，文中 late-2025 benchmark | 报告最高约 17.07× eager、1.45× Liger、1.34× Quack 的特定 shape 收益 | 这些倍数是整个 LLM 训练或 TPOT 提升 |
| Quack 公开实现 | 本次核验 HEAD 77a750c；H100、B200/B300、RTX 50，CUDA 12.9+ 要求 | 有可检查的高效 GPU 归约代码与硬件范围 | 能自动用于旧 GPU 或 HTP／NPU |
| FlashNorm v5 | 2026-04-24 更新；核验论文的 fused RMS+linear 推导 | 高效实现可以跨相邻算子消除物化 | 独立 RMS microbenchmark 的排序可直接预测所有模型 |
| 2026 compute-limited 预印本 | arXiv:2604.23434v1；全部实验处于 T/P&lt;1.84 的预算范围 | DyT 收益随数据／容量反转；64M RMS 与 LN 结果相差不足 0.7% | 把 1.84 当成优劣分界，或宣称该文证明 LN 全面优于 RMS |

PyTorch 官方文章还给出非 power-of-two N=384、较大 N 及 H100 的特定不足；forward 与 backward 不能互相替代。RMS 性能是一个 shape family 上的分布，应记录 [M,D]、dtype、GPU、功率、编译选项与 launch 方式。本文没有在这些 GPU 上重跑新论文数字，它们是**官方／论文报告值**，与本地代数和页面 QA 明确区分。

2026 年预印本的完整标题为 **When Does Removing LayerNorm Help? Activation Bounding as a Regime-Dependent Implicit Regularizer**（Lucky Verma）。其主要干预是 DyT 替换 LayerNorm；RMS 只在 64M 设置中作为第二基线，文中说明该对照为 2 seeds。它报告 RMS 与 LN 的结果在 0.7% 内，DyT 的效益则随数据预算改变。**T/P&lt;1.84 是全部实验的覆盖范围，不是归一化优劣阈值。** 这是对“去统计就能无条件获益”的提醒；单一预印本仍需独立复现。

对于实际选型，我会先看模型质量与 checkpoint 语义，然后在目标设备以真实 [M,D] 测相同优化等级的 forward/backward，最后接入完整 runner 看 TTFT、TPOT 与峰值内存。若 norm 只占总时间很小一部分，优先解决更大成本；若 FP32 统计跨设备回退，优先审计调用链与同步。


## References & Source Record

所有外部事实核验截至 **2026-10-27**。原图／原表从本地 1910.07467v1.pdf 无损 PNG 提取（4× PDF point resolution），同时保留 PDF、TeX、原始 figure PDFs 与资产清单。静态 MathML 为可选择公式；新增示意与交互均标 Agent-created，不代替原始证据。

| 类型 | 来源与用途 |
|---|---|
| Original Paper | [Root Mean Square Layer Normalization, 1910.07467v1](https://arxiv.org/abs/1910.07467)；全文、7 Figures、10 Table 编号、Eq.1–10、附录 |
| Official Code | [bzhangGo/rmsnorm](https://github.com/bzhangGo/rmsnorm)；[固定版本 PyTorch](https://github.com/bzhangGo/rmsnorm/blob/2e726f1a3f106bb719056422f4f9b6aca03d3ce6/rmsnorm_torch.py) |
| Related Paper | [Layer Normalization, 1607.06450](https://arxiv.org/abs/1607.06450)；[Weight Normalization, 1602.07868](https://arxiv.org/abs/1602.07868)；[Group Normalization, 1803.08494](https://arxiv.org/abs/1803.08494)；[Fixup, 1901.09321](https://arxiv.org/abs/1901.09321) |
| Follow-up | [Pre-CRMSNorm, 2305.14858v2](https://arxiv.org/abs/2305.14858)；[FlashNorm, 2407.09577v5](https://arxiv.org/html/2407.09577v5)；[Transformers without Normalization, 2503.10622v2](https://arxiv.org/abs/2503.10622) |
| Adoption | [Meta Llama model.py](https://github.com/meta-llama/llama/blob/main/llama/model.py) |
| Current Official Evidence | [PyTorch normalization performance, 2026-04-08](https://pytorch.org/blog/sota-normalization-performance-with-torch-compile/)；[PyTorch 2.11 RMSNorm docs](https://docs.pytorch.org/docs/2.11/generated/torch.nn.RMSNorm.html)；[Quack](https://github.com/Dao-AILab/quack) |
| New Preprint / 有限证据 | [arXiv:2604.23434v1](https://arxiv.org/abs/2604.23434)；compute-limited regime，单一来源，不作为通用选型结论 |

Initial publication / Last updated：2026-10-27。质量检查分为原文准确性复核与页面可读性复核；发布后另检查 GitHub Pages 的真实资源、公式、图表与交互。
