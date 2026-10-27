## 3. Experiments：先统一指标，再看收益

作者以 MT-Bench 多类别对话做主要速度与质量评估，比较默认 autoregressive baseline、标准 speculative decoding、Medusa-1 和 Medusa-2。下面保留全部原始表格与原图。原表图片保留原版排版和 caption；可展开的全文转录保留所有行列及数值，移动端横向滚动。

**指标字典。** Acc. rate 是每轮平均提交 token 数，单位 token/轮，通常包括根；不是概率或分类准确率。Overhead 是 Medusa 一轮与普通一轮的时间比。Speedup 是 baseline 与 Medusa 生成速度之比。Quality 是 MT-Bench 的 GPT-4 judge 均分（越高越好），不是 Accuracy。该任务没有 TP/FP/TN/FN 定义，不能人为拼一张 confusion matrix。

{{EQ:speed}}

此式用于解释和粗略核验；聚合方式、输出长度和四舍五入可导致与表值略不同。单轮算得更慢并不矛盾：如果每轮时间是 1.22 倍、接受 3.47 token，则速度大约为 2.84 倍。speedup 不是接受率本身。

### 3.1 Vicuna-7B/13B：冻结与联合训练的差别

{{FIG:3}}

**Figure 3 深读。** 左图同时看 7B 和 13B 的 baseline、Medusa-1、Medusa-2。论文报告 Medusa-1 约 2.18×（7B）、2.33×（13B），Medusa-2 两者约 2.83×；联合训练相对冻结方案增加约 29.8% 和 21.5% 的速度收益。右图是 8 个 MT-Bench 类别，coding/math 等规律性更强的 continuation 更容易被未来 heads 命中。类别差异说明草稿可预测性影响效果，不能只看整体柱。

这里的 baseline 是论文默认实现，不能直接替换成今日优化服务引擎。更强 baseline 可能减少可摊薄时间，量化也可能转移瓶颈；加速比与 kernel、shape、采样策略和 hardware 共同决定。

{{TABLE:1}}

**Table 1 深读。** 四列分别对应 Vicuna-7B、Zephyr-7B、Vicuna-13B、Vicuna-33B。Medusa-2 的 speedup 为 2.83、2.66、2.83、2.35；相对标准 speculative baseline 的比值为约 1.925、未报告、1.814、1.469。以普通 baseline 为 1，7B 的 +183% throughput 与 64.7% 的等产出时间下降是不同写法，不要把两者混用。

3.47/1.22≈2.844，3.14/1.18≈2.661，3.51/1.23≈2.854，3.01/1.27≈2.370，与 speedup 小幅出入主要只能说明表值经过汇总/舍入，不能倒推出未公开的单样本计时。Quality 的差值是 +0.01、−0.07、−0.14、+0.05 分；13B 的 −0.14 不应被省略。没有多 seed、置信区间或 judge 稳定性分析，不能宣布“零质量损失”。

### 3.2 Self-Distillation：跨模型成立，收益也有差异

Zephyr-7B 和 Vicuna-33B 用自蒸馏训练，Table 1 显示有明显加速，并维持接近原模型的 MT-Bench 均分。33B 的接受量 3.01、overhead 1.27，高于 7B 的相对每轮成本，最终 speedup 仅 2.35。规模越大不意味着比例收益必然越大；接受质量、head 规模、attention 历史、kernel 利用率和训练约束都在作用。

自蒸馏解决“不提供原 SFT 回答”的问题，没有证明对所有领域、对齐策略和任务迁移都保持质量。公开 prompt 的覆盖、生成风格、multi-turn 分布与 teacher 精度仍是训练数据问题。

### 3.3.1 Tree Structure Ablation：64 个有价值节点优于更多无效节点

{{FIG:4}}

**Figure 4 深读。** 左图纵轴是 acc. rate，右图纵轴是 tokens/s，两者不能混为一张速度曲线。蓝点为随机 dense 树配置，红星为优化 sparse 配置。64 节点稀疏树可以获得不逊于、更好的接受长度，相比 256 节点 dense 树避免大量低价值验证。沿右图继续增加候选，速度可能下降：接受长度的收益趋于饱和，验证成本继续增长。

因此树预算应优化 `E[accepted tokens] / iteration latency`；单独最大化节点数、top-k 覆盖率或 acc. rate 都不够。图上的趋势线提供经验关系，没有证明固定 64 节点是任意模型/上下文/硬件的最优值。

### 3.3.2 Typical Threshold Ablation：把两个纵轴同时读完

{{FIG:5}}

**Figure 5 深读。** 温度固定 0.7，评估 writing 与 roleplay。横轴阈值 ε 从 0.01 到 0.25、步长 0.01；不同曲线使用质量均分和 acc. rate 的双纵轴。星形为 greedy，点为 random sampling，typical 曲线展示更宽允许集合可提升接受长度，但过宽会损害质量。ε=0 会对所有正概率候选开放，不具备质量保证。

实验符号中 α=√ε 与方法段 δ 的对应记号不够统一，读者应按阈值的实际表达式复现。单次 noisy judge 曲线、两个任务类别和固定温度不构成全场景最佳阈值。应进一步报告任务分布、长回答、不同温度和可靠性指标。

### 3.3.3 Heads Warmup：限制对 backbone 的早期冲击

{{TABLE:2}}

**Table 2 深读。** Vicuna-7B baseline 6.17，direct fine-tuning 5.925，下降 0.245 分；Medusa-1 6.23，Medusa-2 6.18。Medusa-1 相对 baseline 的 +0.06 不意味着冻结模型能力被训练增强，可能含采样、接受或 judge 波动。表支持所给训练 recipe 比直接联合微调更稳，但未通过相同预算、多 seed 与逐组件对照将所有改善归因给 warmup。

direct fine-tuning 的 speedup 是 N/A，而不是 0 或 1。表中质量接近是经验结论，不能取代概率保证。

{{TABLE:3}}

**Table 3 深读。** 无树的 Medusa-1 heads 约 1.5×；加 tree attention 约 1.9×；优化树约 2.2×；换 Medusa-2 训练约 2.8×。这是累计配置收益，不能将四个数相加。逐阶段比例是约 26.7%、15.8%、27.3%，但各阶段不是完整 factorial ablation，也没有单独拆出 typical 与采样模式的贡献。


## Appendix D. Standard Speculative Decoding：draft 大小与 γ 都要调

{{FIG:7}}

**Figure 7 深读。** 三个原始 panel 完整保留，分别为 Vicuna-7B/13B/33B。横轴 γ 是一次 draft 的 token 数，纵轴是 tokens/s；灰虚线为各自 AR baseline。较长 draft 虽可能减少 target 轮数，却要串行运行小模型更多次，接受也会在首个 mismatch 截断。

论文测试中较优配置包括 7B 对 Llama-68M 的 γ=4、13B 对 Llama-68M 的 γ=3、33B 对 TinyVicuna-1B 的 γ=3。draft 经过 compile，并非只选一个故意弱的 baseline。但这些是特定旧模型组合；相对于默认 target baseline 的优势不能证明 Medusa 永久优于所有独立 drafter。


## Appendix E. More Model Results：看模型间差异，也检查 caption

{{FIG:8}}

**Figure 8 深读。** 横向比较更多 Medusa-2 模型。不同 target 的 head 可预测性和训练约束带来不同 speedup。必须把柱形与 Table 1 的接受率、overhead 一起读，才能理解为何大模型并非必然获得更高加速比。

**原文一致性提醒。** Figure 8 caption 把 Vicuna-13B 也归到 self-distillation，但 §3.1 与 Appendix B.3 将 7B/13B 归为主数据两阶段训练；B.4 列 33B/Zephyr 为自蒸馏。本精读保留原 caption，按正文与配置分组分析，不将矛盾悄悄消掉。


## Appendix F. AlpacaEval：速度换算到 TPOT

{{TABLE:4}}

**Table 4 深读。** baseline→Medusa 的 tokens/s 为 37.07→106.76（7B）、29.01→91.54（13B）、17.87→40.43（33B）、34.21→99.50（Zephyr）。从原值重新算 speedup 是 2.880、3.155、2.262、2.909，表内分别舍入 2.88、3.16、2.26、2.91。

| 模型 | 吞吐绝对增加 token/s | 吞吐相对增加 | baseline TPOT → Medusa TPOT |
|---|---:|---:|---:|
| Vicuna-7B | +69.69 | +188.0% | 26.98 → 9.37 ms/token |
| Vicuna-13B | +62.53 | +215.5% | 34.47 → 10.92 ms/token |
| Vicuna-33B | +22.56 | +126.2% | 55.96 → 24.73 ms/token |
| Zephyr-7B | +65.29 | +190.9% | 29.23 → 10.05 ms/token |

这些 TPOT 是 `1000/tokens_per_second` 的平均速度换算，**不是本次实测逐 token 延迟分布**；Medusa 一轮会成块提交，用户可见 token 间隔不是均匀的。Table 4 没有 AlpacaEval win rate 或任务准确率，不能说该表证明质量保持。论文用 AlpacaEval 校准树，又在该数据上报告速度；校准与评估的独立拆分披露不足，应关注分布适配与潜在重复。


## Appendix G. Roofline：多算一些，是否能少花时间？

### G.1 不同阶段的算子复杂度

{{TABLE:5}}

**Table 5 深读。** 原表将 prefill、单 token decode 和 parallel decode 分开。q 从 1 增大，投影和 MLP 的 FLOPs 随 q 增大，而权重加载可在这些位置间复用；attention 对每个新 query 仍要读历史 KV，因此不同算子随 q、B、S 的变化不同。

**原表 shape 校正。** parallel decoding 中 QKᵀ 的 output 被写为 [b,n,s,q]，随后 PV 的输入也出现方向不匹配。为避免 q≠s 时照抄出错，保留原表并给出下面一致的工程记号：

| 算子 | 输入 → 输出 | FLOPs 主项 | 主要存储 / IO |
|---|---|---|---|
| QKV 投影（MHA 示意） | [B,q,d] × [d,d] → [B,q,d] | O(Bqd²) | 权重 O(d²)，激活 O(Bqd) |
| QKᵀ | [B,H,q,D_h] × [B,H,D_h,S+q] → [B,H,q,S+q] | O(BHq(S+q)D_h) | KV O(BH(S+q)D_h) |
| PV | [B,H,q,S+q] × [B,H,S+q,D_h] → [B,H,q,D_h] | O(BHq(S+q)D_h) | 若显式写 score，多 O(BHq(S+q)) |
| MLP up/gate/down | [B,q,d] → [B,q,i] → [B,q,d] | O(Bqdi) | 权重 O(di)，中间激活 O(Bqi) |

这里 H 为 MHA head 数；GQA 使用 H_q 与 H_kv 区分。树 mask 的零不能自动省掉 dense QK 的所有相应 FLOPs。Fused attention 可避免完整 score 写回，但需支持正确 mask。复杂度是主项，没有隐含包含 host 调度、sampling 与 KV copy。

### G.2 九张 Roofline 原图：按 GPU 和规模逐张读取

横轴 operational intensity（FLOP/byte），纵轴 FLOP/s，均为对数轴。斜线是 bandwidth roof，水平线是 compute roof。点接近哪条线，说明相应测量条件下受到哪种资源约束；离 roof 很远也可能有 kernel、launch、小矩阵利用率等原因，不能一概断言只缺带宽。`ar` 是 decode，`init` 是 prefill；qkv 与 up/gate/down 属于线性层，qk/pv 是 attention 两个矩阵乘。

{{FIG:9}}

**Figure 9（7B / A100）深读。** 对照相同 operator 的 ar/init 点：单 token 投影与 MLP 位于低算术强度区域，prefill 的多位置矩阵提高复用并更接近计算区。原 caption 的 1,935 GB/s 与 312 TFLOP/s 是图中引用的 roof 标签；不应把它们直接当作本地设备或所有 A100 PCIe 的实测规格。

{{FIG:10}}

**Figure 10（13B / A100）深读。** 同一硬件换更大模型，用点云的尺度和位置看“模型参数更多”与“算术强度更高”是否等价。decode 仍有低强度线性层，增加模型规模不能替代把 q 从 1 提高来复用权重；attention 随历史 KV 长度变动，又与投影不同。

{{FIG:11}}

**Figure 11（33B / A100）深读。** 更大模型的权重字节可使单请求 decode 明显受 memory 影响；prefill 位置较多时，线性层已有高强度。此图支持寻找 decode 余量，不是 Medusa 端到端速度测试。它也提醒：head 附加参数和完整词表投影会加入权重读写。

{{FIG:12}}

**Figure 12（7B / A40）深读。** 换 GPU 后 bandwidth/compute roof 的交点改变；同一 intensity 下能达到的绝对 FLOP/s 不同。不能把 A100 的 q=64 预算和 speedup 直接移到 A40。应比较相同 operator 点相对本图 roof 的距离。

{{FIG:13}}

**Figure 13（13B / A40）深读。** 横向对比 Figure 10 的同规模模型，再纵向对比 Figure 12 的同硬件较小模型，区分硬件效应与模型效应。多候选投影的价值来自更多有效行复用权重，而 qk/pv 的历史 KV 不会因此消失。

{{FIG:14}}

**Figure 14（33B / A40）深读。** 大模型在较低硬件 roof 下可更早遇到容量/算力/带宽边界。点云只能证明 benchmarked operator 条件；未加入训练 heads、接受器与 KV 重排，也没有生成质量维度，不能将 roofline 图当作全部 inference pipeline 的性能画像。

{{FIG:15}}

**Figure 15（7B / A6000）深读。** 同样观察 ar/init 分离与线性层、attention 的不同族群。A6000 的 roof 与 A40/A100 不同，说明“GPU 闲置算力可利用”需要针对设备核验。一般原理可以迁移，具体最佳树和收益不能直接迁移。

{{FIG:16}}

**Figure 16（13B / A6000）深读。** 对照 Figure 13 同规模/不同设备，避免只按显卡名字排序。低 intensity 点在斜 roof 下，高 intensity 点受水平 roof 限制；有效吞吐还要除去被拒候选的计算，FLOP/s 高并不保证 token/s 高。

{{FIG:17}}

**Figure 17（33B / A6000）深读。** 九图的共同结论是：单 token decode 与多 token prefill 有不同复用机会，不同算子也有不同瓶颈。共同形态支持 Medusa 的系统动机；它们没有证明每一种 batch、context、dtype 或新硬件都 memory-bound。

### G.3 增加 q：矩阵效率提高，工作量也提高

{{FIG:18}}

**Figure 18 深读。** Llama-33B、A100、B=16，改变 history sequence length 和候选数。q 增大让 attention 的 FLOP/s 与 intensity 上升，但增加了查询行数；S 越长，KV 工作越重。用 Table 6 的原始数值可以更精确拆解。

{{TABLE:6}}

**Table 6 深读。** 每格为 **TFLOP/s & FLOP/byte**，不是 latency。B 固定 16，S 有 128–8192 七行，q 有 1、16、32、48、64、80、96、112 八列。S=1024、q=1 为 1.24 & 0.99，q=64 为 54.8 & 40.96，即计算吞吐 44.19×、强度 41.37×。

若按固定 S 的同类 attention 矩阵工作量近似 q 倍，单次算子时间比例是 `64 / 44.19 ≈ 1.45`，即一次 64 行验证仍比一次 1 行慢约 45%；它只是远低于逐行做 64 次的时间。这解释了为何不能写“latency 降低 44×”。最终 token/s 还取决于接受几行及其他算子时间。

{{FIG:19}}

**Figure 19 深读。** 固定 S=1024，改变 batch 和 q。attention 的单 token operational intensity 不因 B 成比例提高：不同请求的 KV 不能共享成同一历史。B 和 q 都可能提升执行效率，但只有同一路径的候选能共享该请求的 KV 读入；仍受缓存、工作空间与 kernel 结构制约。

{{TABLE:7}}

**Table 7 深读。** q=1 时 B 从 1 到 64，attention TFLOP/s 从 0.37 到 1.48，而 intensity 保持 0.99；这是“更多 batch 提高利用率，却不自动增加单请求 KV 复用”的直接数值证据。q=64 时同样从 19.79 到 61.13 TFLOP/s；并不是 q 和 B 带来的收益能够无限相乘。

{{FIG:20}}

**Figure 20 深读。** linear up/gate/down 的点随有效行数 Bq 提高，权重矩阵的重复读取可被摊薄；进入 compute roof 后继续加行数主要增加时间。这是 Medusa 在小 batch 有利、在较大 batch 收益趋弱的系统依据。

{{TABLE:8}}

**Table 8 深读。** MLP 在 B=1、q=1 为 1.26 & 1.0；B=1、q=64 为 76.57 & 63.17；B=64、q=64 为 246.16 & 2221.31。线性层 intensity 随 Bq 大幅上升，但 TFLOP/s 后段接近平台，不能继续按低 batch 的倍率外推。所有 56 格完整保留，没有只选择最佳格。

{{DEMO:cost}}

该交互读取 Tables 6–8 全部 168 个原始数值对；只做格值查询与比例推导，没有插值，也没有伪造新硬件 benchmark。固定同一行时按矩阵主项推导 time ratio，并提示它不是全模型 latency。

### G.3 模拟端到端趋势：有解释力，不能当真实服务

{{FIG:21}}

**Figure 21 深读。** Llama-7B、A100、B=1、S=1024 的一层 block 简化模拟。横轴候选数，曲线分别是模拟 acc. rate、speedup 和归一化 latency；约 64 附近出现较好折中。acc. rate 增长但每轮成本也增长，最优点在两者比值，不在最大候选数。

模拟使用约 `0.477·log(q)` 的经验接受量近似，其在 q=1 给出 0，而真实普通 decode 至少生成根 token；这是拟合/定义范围问题，不能机械外推到边界。模型没有完整刻画 heads、词表输出、KV gather/copy、接受器、host 同步、通信和服务调度。

{{FIG:22}}

**Figure 22 深读。** 固定 S=1024，模拟不同 B。batch 更大时投影/MLP 原本就更能复用权重，额外候选的“空闲算力窗口”变小；图示 B>32 时可能失去收益。这个阈值属于给定简化模型，不能当成线上 continuous batching 的普适禁用规则。

{{FIG:23}}

**Figure 23 深读。** 固定 B=4，比较不同 S。更长上下文增加 attention 与 KV 成本，多候选要多次读大量历史，收益下降。真实服务还会受到 prefix sharing、KV precision、paging、长短请求混合和调度影响，因此应把它当趋势假设，设计进一步测量。


## Experiment Audit：支持了什么，还缺什么

| 审计项 | 已有证据 | 结论边界 / 建议补测 |
|---|---|---|
| 速度 | MT-Bench、AlpacaEval wall-time/tokens/s；多个旧模型 | 相对默认实现，未给今日强优化引擎统一对比 |
| 质量 | GPT-4 judge，主模型相对均分接近 | 没有统计区间；13B −0.14 分不能隐去；不等价于分布相同 |
| 训练 | M1/M2、两阶段与 self-distill recipe | warmup 文字/符号不一致；详细训练 token/成本未完全齐全 |
| 树 | dense/sparse 对比；校准 rank accuracy | 独立近似；校准域与线上分布变化；校准/评估拆分不足 |
| 采样 | T=0.7 的写作/角色扮演曲线 | 多温度、多任务、长回答、可靠性与分布偏差待补 |
| 算子 | 三种 GPU roofline、Tables 6–8 | 是 per-op 数据；不含完整 speculative pipeline |
| 服务 | 小 batch 动机与简化 batch 模拟 | continuous batching、p99、SLA、吞吐/延迟折中未充分验证 |
| 端侧 | 无 NPU/HTP 性能与功耗实测 | 不能从 GPU GEMM 利用率外推端侧动态树收益 |

**最值得补的公平对照。** 固定同一 target checkpoint、dtype、prompt/输出长度、temperature、GPU 和引擎版本；分别跑 AR、严格验证 M1、typical M1、严格验证 M2、typical M2。报告每轮候选 q、接受 ℓ 分布、per-op 时间、KV peak、TTFT、平均/尾部 TPOT、整体 tokens/s 和任务质量。这样才能把“模型改变”“规则改变”“runtime 改善”三个来源拆开。

质量至少分任务测：coding 的 pass@1、math 的 exact match、writing/roleplay 的偏好或 judge 分布；必要时统计失败类型。对确定分类任务才使用 TP/FP/TN/FN 及 Precision/Recall/F1，避免用不适配指标遮蔽真实质量变化。

