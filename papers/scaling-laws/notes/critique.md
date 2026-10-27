## Critique：优点、弱点与隐藏的系统成本

| 维度 | 评价 | 证据 / 后果 |
|---|---|---|
| Novelty | 联合统计框架与 compute frontier 是主要贡献 | 不把幂律这一概念本身当成首次提出 |
| Correctness | 观测区间拟合与多个控制扫描有说服力 | 熵底 conjecture、函数唯一性与普适指数未证明 |
| Experiment quality | 参数、数据、形状、context、batch 范围丰富 | 最小数据瞬态、optimizer 切换、固定 warmup 与不足的 CI |
| Engineering complexity | 统计后处理简单，训练 grid 成本高 | 需要持续、统一格式 checkpoint 日志和多 seed |
| Scalability | 宏观规律可用于预算 planning | 设备显存 / 通信 / 数据管线构成额外 feasible region |
| Generalization | OOD textual loss 与域内 loss 相关 | 不等价于任务能力、多模态或部署质量 |
| Compute cost | 区分 sample、steps、FLOPs 是优点 | non-embedding 近似在小网络会偏，未含重算/通信 |
| Memory cost | 原研究没有完整 peak-memory 评测 | 不能从同 \(N\) 推断同 activation/KV 或可部署性 |
| Hardware dependency | PF-day 是硬件中性的 FLOPs 单位 | 实际利用率、kernel、带宽和并行方案强相关 |
| Reproducibility | 原图、公式、配方可重建研究思路 | WebText2 / run logs / code 不完整公开，直接精确复现困难 |

**Strengths。**本论文改变“一个大模型跑完看分数”的研究单位，改为可复用的 loss 面和预算包络；用原始 Figure 2、17、18 把参数、样本、算力与串行更新区分清楚；主动讨论外推矛盾，而非只画无限向下的线。

**Weaknesses。**拟合依赖有限区间与特定训练配方，缺少完整 uncertainty propagation；多个常数/近似指数、factor-of-two 与交叉引用不一致，使用者需要重建口径；优化配置改变恰好与 model size 关联，影响 compute optimum 的可迁移性。

**Hidden trade-offs。**大模型短训可节省训练 FLOPs，却扩大权重、optimizer state、activation、KV、通信与推理开销。batch 增加可减少 updates，却可能降低样本效率，超过 critical batch 后尤其明显。模型形状 loss 相似可以让系统选择更自由，也意味着必须单独测 layer latency、kernel utilization 与峰值内存。

**Unanswered questions。**如何在配方变化后可靠估计指数区间？同一个 fitted entropy floor 是否可识别，还是与外推函数混淆？数据质量/去重与 repeated exposure 如何共同进入有效数据量？平均 loss 的改善对具体业务任务的收益是否稳健？动态改变模型尺寸的训练轨迹能否始终接近前沿？

**Possible improvements。**发布 raw/adjusted 全部曲线、完整 FLOPs 和 seed 统计；预注册 holdout scales；对 optimizer/warmup 做控制消融；比较多个拟合函数并报告外推而非只报训练拟合；在训练最优之外给推理负载和真实硬件约束下的 Pareto frontier。

