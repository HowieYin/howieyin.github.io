## 扩展调研：哪些工作修正了哪些假设

以下均为 **External Source**，与 2020 原文分开。核对原论文或官方仓库；版本年份写明初版 / 采用版本，避免把后续修订日期当成首次发表日期。

| 工作 | 年份 / 使用版本 | 核心机制 | 优势 | 限制 | 与 Kaplan 的关系 |
|---|---|---|---|---|---|
| [Hestness et al.](https://arxiv.org/abs/1712.00409) | 2017 | 多任务经验 learning/data scaling | 较早系统展示可预测幂律区间 | 配方与任务不同 | 历史先驱；不是 .73/.27 来源 |
| [McCandlish et al.](https://arxiv.org/abs/1812.06162) | 2018 | gradient noise scale 与 batch Pareto | 连接样本与 steps 效率 | 系统吞吐 / 通信不在噪声指标中 | 直接支撑 critical-batch 校正 |
| [Rosenfeld et al.](https://arxiv.org/abs/1909.12673) | 2019 | 联合数据 / 模型误差预测 | 构造可外推联合函数 | 形式与适用任务不同 | 同期相关路线 |
| [Chinchilla / Hoffmann et al.](https://arxiv.org/abs/2203.15556v1) | 2022 v1 | IsoFLOP、learning curve、parametric fit 三种估计 | 更充分扫描 training token；实际大模型对照 | 原始参数拟合和数据质量仍敏感 | 固定计算下转向参数 / token 近等比例增长 |
| [Muennighoff et al.](https://arxiv.org/abs/2305.16264v5) | 2023 初版；采用 2025 v5 | 区分唯一数据与重复 exposure，effective-data law | 描述数据受限多 epoch | 不能保证所有质量 / 重复范围 | 解除单 epoch \(D=E\) 假设 |
| [Sardana et al.](https://arxiv.org/abs/2401.00448v3) | ICML 2024；采用 2025 v3 | 训练 + 推理总计算目标 | 纳入 request/输出长度，优化较小长训模型 | 请求量、成本模型、真实吞吐要估计 | 原 B.4 的推理成本提示被系统化 |
| [Pearce & Song](https://arxiv.org/abs/2406.12907v3) | TMLR 2024；采用 v3 | 分析计数偏差、缩小尺度受控模拟 | 分离方法偏差解释 | 不等于真实 frontier 大模型全面复现 | 解释不同指数为何可能出现 |
| [Porian et al.](https://arxiv.org/abs/2406.19146v4) | NeurIPS 2024；采用 2025 v4 | head FLOPs、warmup、optimizer 受控消融 | 在 OWT2 / RefinedWeb 重建近 .5 前沿 | 有限模型与数据范围 | 直接检验 Kaplan–Chinchilla discrepancy |
| [Practical Scaling Laws](https://arxiv.org/abs/2605.09189v2) | 2026-05 初版；2026-09-21 v2 | 参数、processed token、unique data 与饱和统一 | 更宽 regime 的 bounded loss model | 公式灵活度 / 网格依赖，非处处胜出 | 修正单纯有限区间幂律形式 |
| [Skaling](https://arxiv.org/abs/2608.07222v1) | 2026-08-07 v1 | outer exponent 将 capacity/data 与联合形状解耦 | 在多个网格改善远距离预测 | 部分任务优势很小或弱于特定 baseline | 泛化 Kaplan/Chinchilla 联合式结构 |

### Chinchilla：修正的是 compute allocation，不是否认规模效应

2022 原文训练 **400 多个** 70M–16B 模型，tokens 约 5B–500B，分别用训练曲线最优包络、IsoFLOP 截面与参数联合拟合。前两种大致支持参数 / token 预算指数约 .5/.5；parametric fit 原始 \(\alpha\approx.34,\beta\approx.28\)，在 \(L=E_0+A/N^\alpha+B/T^\beta\)、\(C\approx6NT\) 下推得 \(p_N=\beta/(\alpha+\beta)\approx.452,\ p_T\approx.548\)。不是三个方法精确等于 .5。

70B Chinchilla 用约 1.4T token，与 280B Gopher 约 300B token 在相近训练计算下比较；原文报告 MMLU 约 67.5%。这是历史模型/任务对照，不能拿 Kaplan token loss 直接计算“提升多少 MMLU”，也不是 2026 当前 SOTA。按模型配置和数据质量补齐后比较才可因果归于 allocation。

工程含义：减少模型参数而增加训练 token 可以降低部署权重与 KV/每 token 计算负担，但训练时间、数据管理与总 token 增加。论文的约 20 tokens/parameter 是其大模型实验附近的配置，不是所有小模型、数据受限或 inference-heavy 场景的常数。

### 2024 受控复现：不是只换一种 schedule 就完了

Porian 等的原始研究与[官方代码](https://github.com/formll/resolving-scaling-law-discrepancies)检验三种混杂：**小模型输出 head FLOPs 被省略、fixed warmup 对短训练不成比例、优化器/LR 配方对规模与 batch 未对齐**。在 OWT2 和 RefinedWeb 上，修正后最优 \(N\) 指数分别约 **.518（95% CI .49–.54）** 与 **.497（.49–.50）**，并可在不使用 learning-rate decay 的设定中恢复 Chinchilla 类结果。

因此“差异完全因为 cosine schedule”过于简单。计数改变预算横轴，warmup 改变可用有效训练，optimizer 改变 loss 面；每项都可能改变最优点。论文给控制证据的区间和 CI，比只看图拟合斜率更能支持因果归因。

Pearce & Song 的 [Reconciling Kaplan and Chinchilla Scaling Laws](https://arxiv.org/abs/2406.12907v3) 从 counting / fitting 偏差和受控较小尺度实验解释趋势。它与 Porian 的多因素复现互补，不应混成同一个消融，也不应把其模拟结果声称为训练 trillion-parameter 模型的验证。

### 数据受限：重复 exposure 与唯一信息分别建模

Muennighoff 等 [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264v5) 表明在其测试范围内，约 **最多 4 epochs** 重复可接近无重复效果，更多重复出现收益递减；论文含最高约 9B model / 900B processed token 的实验。其结论带 regime 条件，不能写成“4 epoch 永远无代价”。

[官方仓库 datablations](https://github.com/huggingface/datablations)提供代码与实验资产。对本篇的修正是：唯一 \(D\) 限制与训练 token \(E\) 限制应独立，数据质量/可重复学习信号决定 effective data。重复语料、合成数据、重采样 mixture 都要记录来源，不能把 \(E\) 改名成 \(D\) 以绕过数据瓶颈。

### 推理需求：训练最优与生命周期最优不同

Sardana 等 [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448v3) 在 47 个模型中探索长训（最高约 10,000 token/parameter），将推理请求量与长度纳入 compute。高请求量如约 1B requests 的假设会推动更小、更长训的模型。相关代码位于作者 [anc 附件](https://arxiv.org/src/2401.00448v3/anc)。

对端侧/服务读者，最值得保留的是**目标函数迁移**：\(C_{\rm train}+C_{\rm serve}\)，再叠加权重/activation/KV 峰值、延迟与设备约束。请求量与输出长度变化会改变最优点。真实 decode 可能 memory-bound，无法只用 FLOPs 线性比例替代 TPOT 实测。

## 技术演进与当前进展：截至 2026-10-27

{{timeline-diagram}}

历史路径从“数据越多越好”的经验规律，发展到 Kaplan 的联合预测与算力包络，继而用 IsoFLOP 网格与受控复現修正指数，再把 unique data、processed tokens、推理负载和函数形式纳入。多条研究并行，不是每一代都完全替换上一代。

### 2026 Skaling：预测远端误差，不能只报告最有利列

Videau 等 [Skaling Laws](https://arxiv.org/abs/2608.07222v1) 采用

<div class="derived-formula"><code>L(N,D) = (A/N^α + B/D^β)^κ + E0</code><p>External Source · Skaling Eq. 形式；D 在该研究中表示训练数据 / token 轴，语义按其原设定解释。</p></div>

outer exponent \(\kappa\) 使单轴衰减与联合曲面形状可以更独立调整；固定某些参数可连接既有联合式。自建 SK-Grid 约 **134M–4.93B**、context 2048、vocabulary 128,256，并评测现有网格的远尺度外推。

| Skaling v1 Table 1 / Table 5 情形 | 指标（↓） | 对照 | Skaling | 解释 |
|---|---|---:|---:|---|
| SK-Grid full → far | MAPE | Chinchilla \(5.17\pm.28\)% | \(.70\pm.39\)% | absolute 下降 4.47 percentage points；相对约 86.46% |
| SK-Grid L-shaped → far | MAPE | Chinchilla \(14.63\pm.39\)% | \(1.15\pm.53\)% | absolute 下降 13.48 points；相对约 92.14%，对训练网格形状敏感 |
| Farseer far | MAPE | Farseer \(2.46\pm.19\)% | \(2.31\pm.18\)% | absolute .15 point，相对约 6.10%；差异较小 |
| Farseer-code，L-shaped ExtN | MAPE | Farseer .96% | 1.39% | 此列 Skaling 更差，不宣称全任务优胜 |

这些是 **loss prediction error**，不是模型 task performance；± 的统计口径保留原表，不擅自全部改成 95% CI。不同训练点切分改变外推难度。未在核查的原文链接中定位到独立官方训练仓库；不能据“未定位”断言闭源或不可复现。

### 2026 Practical Scaling Laws：饱和、训练与唯一数据同时入模

Bryant & Liu [Practical Scaling Laws](https://arxiv.org/abs/2605.09189v2) 采用 bounded / saturating form：

<div class="derived-formula"><code>L = E0 + (L0−E0) · h/(1+h); h = a/N^α + b/T^β + cN^γ/D^δ</code><p>External Source · T 是 processed training tokens，D 是 unique data；与原文 context-position T 区分。</p></div>

\(h\to\infty\) 逼近初始上界 \(L_0\)，\(h\to0\) 逼近熵底 \(E_0\)，三个项分别表达容量、优化暴露量与有限数据惩罚。它研究多个函数家族和五个已发表 LLM grids，使用 200 次 bootstrap、median 与 95% quantile 表达不确定性。

| v2 Table 2，高计算区域 | 指标（↓） | Chinchilla 形式 | Practical 形式 | 边界 |
|---|---|---:|---:|---|
| Chinchilla grid | log-loss RMSE | .023 | .008 | 下降 .015，相对约 65.22% |
| Porian grid | log-loss RMSE | .098 | .063 | 下降 .035，相对约 35.71% |
| TinyStories grid | log-loss RMSE | .167 | .183 | 新式更差；Muennighoff 对照 .079 也更低 |

这里 RMSE 在 log loss 空间，与 Skaling 的百分比 MAPE **不可横向排冠军**。模型尺度、tokenizer、网格和 extrapolation split 也不同。核查的论文正文没有定位到明确官方代码链接，保留此信息缺口。

### 当前应采用什么结论

不存在这类论文共享的统一“2026 SOTA”榜单：有人比较 language-model task score，有人比较 loss 预测误差，有人优化训练/推理成本。当前更可靠的实践是，在目标配方与硬件约束下做小/中模型 IsoFLOP 扫描，完整记 FLOPs，校准 warmup 与 optimizer，验证外推，区分唯一数据和 exposure，并根据服务负载选择适当长训程度。

因此本报告给出明确的 **截止日期、benchmark grid、metric、scale 与开源证据**，但不把一个预测公式称为所有任务上的最先进模型。开源仓库可帮助复现流程，不意味着其数据或所有原始训练 run 都公开。

