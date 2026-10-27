## 实验审计与可复现实现：从 Figure 到运行记录

| 审计项 | 原论文信息 | 对结论的影响 / 复现要求 |
|---|---|---|
| Dataset | WebText2；约 22.24B train token，另有 held-out / OOD corpora | 语料筛选和 tokenizer 改变后必须重新拟合 |
| Model | decoder-only Transformer；768–1.5B 非 embedding 参数 | 小模型区间需特别审计 head 与 attention FLOPs |
| Training tokens | 默认上限 131.072B processed；子集与早停 run 不同 | 每个 run 单独记录 \(D,E\) 与 epoch，不能统一假定一遍数据 |
| Batch / context | 默认 \(2^{19}\) token/update，1024 context | batch 校正依赖 loss；不能解释成每 run 都实际用 critical batch |
| Optimizer | Adam；>1B 用 Adafactor | optimizer change 是规模轴混杂，现代复现需要控制 |
| LR | 默认 warmup 3000、cosine-to-zero；另有 schedule scan | 短预算受 warmup token 占比影响，末端 decay 也改变 loss |
| Regularization | 有限数据实验 10% dropout、early stopping | \(D\) 要求并非对全部正则充分调优后结果 |
| Hardware / precision | 原文未完整披露设备数量、型号与数值精度 | 不编造 GPU-day、能耗、吞吐或训练内存实测 |
| Metric | next-token loss，nats/token | 没有 TP/FP/TN/FN；不是分类任务，不伪造 Accuracy/Precision/Recall/F1 |
| Noise / uncertainty | 第 4 节 .02；小模型 schedule scan .05；无完整 CI | 独立 seed、fit covariance 与误差传播需补充 |
| Baselines | LSTM / recurrent Transformer，有数据/参数对照 | FLOPs、hardware、optimizer tuning 不构成全维度严格公平比较 |
| Code / run logs | 原发布未定位到完整代码与 run grid | 不宣称已复现原训练；本报告验证的是公式和页面功能 |

**真正的 ablation。**模型 shape 控制、batch 扫描、context 分解、同数据 LSTM 对比以及共享权重计数比较分别回答不同机制。它们的目标并非证明一个新 module 贡献几分；“没有模块 ablation”在这篇经验规律论文中不是主要批评。关键缺项是优化配置与 FLOPs 口径的受控重复。

**建议复现数据结构与伪代码。**输入是每个训练 checkpoint 的测量记录；输出是预算网格的经验最佳模型，再独立拟合指数。所有日志保留总 / 非 embedding 参数、唯一与 processed token、实际 FLOPs 估计、schedule 状态和原始 loss。以下是 Agent-created 方法重建，不是原作者开源实现。

```python
# RunRecord: N_nonembed, N_total, D_unique, B_tokens, step,
# loss_nats, warmup_tokens, optimizer, context, flops_complete
for run in runs:
    for ckpt in run.checkpoints:
        E = run.B_tokens * ckpt.step
        C_body = 6 * run.N_nonembed * E
        Bcrit = B_star / ckpt.loss_nats ** (1 / alpha_B)
        Cmin = C_body / (1 + run.B_tokens / Bcrit)
        Smin = ckpt.step / (1 + Bcrit / run.B_tokens)
        save_raw_and_adjusted(run, ckpt, E, C_body, Cmin, Smin)

for budget in log_budget_grid:
    candidates = interpolate_loss_at_budget(runs, budget)
    # Report sensitivity to interpolation and exclusions as well.
    frontier[budget] = min(candidates, key=lambda point: point.loss)

# Split by model/budget regime before fitting; validate on unseen scales.
fit_power_laws(frontier, repeated_seeds=True, bootstrap=True)
```

记录 checkpoint 共 \(M\) 个，候选模型共 \(R\) 个、预算网格 \(K\) 个；排序索引后插值约 \(O(M\log M+KR\log M)\)，拟合成本远小于原训练。不要把后处理脚本的运行成本当成模型训练成本。实际 batch 与 model 的硬件可行域应先筛选，再进行最优分配。

**我会补的实验。**在相同数据 tokenizer 上，以完整 FLOPs 重计小模型、用 token-relative warmup、为不同 \(N,B\) 调 LR/Adam 参数，做同 horizon 的 IsoFLOP 网格与 independent-seed replicate；报告未经校正原曲线与校正后曲线，分别 bootstrap。之后保留大尺度 holdout 验证，增加多 epoch/数据质量/不同 context，最后加入服务负载、TTFT/TPOT、KV 与峰值内存。这样可辨别“规律迁移”与“优化未对齐”的贡献。

