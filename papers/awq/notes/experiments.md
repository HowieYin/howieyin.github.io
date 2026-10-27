## 5. Experiments：把每组 claim 放回实验条件

### 5.1 设置与评价指标

| 项目 | 原文提供的设置 | 审计意义 |
|---|---|---|
| 模型 | OPT、LLaMA / Llama-2、Mistral/Mixtral、Vicuna、CodeLlama、OpenFlamingo、LLaVA、VILA | 跨架构和任务，但不是每个模型都有同样 baseline |
| 权重精度 | 主要 INT3/INT4，group=128；极低比特组合 INT2-g64 | 不能跨 group 和 bit 直接拼结论 |
| 校准 | 小规模通用文本；Figure 8 的 sequences 长度 2048 | 默认代码样本数与图中试验不同 |
| PPL | WikiText / WikiText-2，↓ | 衡量语言建模负对数似然；不是 classification accuracy |
| 指令评估 | 80 个问题，GPT-4；交换回答顺序合计 160 次评审 | 不能把 160 视作独立问题样本 |
| VLM | COCO 5k 的 CIDEr ↑，11 项 VILA benchmark | CIDEr、MME 与百分比准确率不可统一求平均 |
| 编程 / 数学 | MBPP pass@1、pass@10；GSM8K ↑ | 需要采样设置和置信区间理解小幅变化 |
| 速度 | batch=1，prompt=4，generate=200，median latency | 主要覆盖 warm 短 prompt 生成；不覆盖完整冷启动 / TTFT 分位数 |
| 硬件 | RTX 4090、laptop 4070 8 GB、Jetson Orin、Raspberry Pi 4；VILA 另有 A100 | 设备、内存容量和 kernel 都要匹配 |
| 训练项 | 本方法是 PTQ，没有新 optimizer / LR / training tokens | 不能编造一个 AWQ training recipe |

**Baseline fairness。** GPTQ 必须区分是否 reordering；系统速度必须同时看 HF FP16 与作者优化的 FP16。只报低优化 baseline 会混入融合收益；只报 OOM 模型则无法计算有限倍率。论文未提供多次独立量化、完整置信区间或跨所有任务统一资源预算。

### 5.2 精度：语言、指令、视觉、代码与极低比特

{{table4}}

**Table 4：不要跳过最强 GPTQ。** Llama-2-7B INT4 的 FP16=5.47，RTN=5.73，GPTQ=5.69，GPTQ with reordering=5.63，AWQ=5.60。AWQ 相对 RTN 改善 0.13 PPL，相对最强 GPTQ 只有 0.03，距 FP16 仍 0.13。INT3 的 LLaMA-7B 则从 GPTQ 8.81 → reordered GPTQ 6.53 → AWQ 6.35。AWQ 的作用明显，但不能把 reorder 带来的改善记到 AWQ。

**成本。** 完整 AWQ 搜索与直接 RTN 的校准成本不同；GPTQ 的 Hessian/补偿也不同。此表给出精度，不直接证明相同量化耗时下的最优 Pareto 点。

{{table5}}

**Table 5：架构可迁移，竞争排名证据不足。** Mistral-7B FP16 4.14 → INT4 4.30 / INT3 4.83；Mixtral FP16 5.94 → INT4 6.05 / INT3 6.52。这支持在 GQA/MoE 上获得可用结果，但该表没有 RTN、GPTQ 或旋转量化对手，不能独立支持 “superior to competing quantizers”。Mixtral 的所有 expert 常驻权重影响容量，active parameters 不能代替总存储。

{{fig5}}

**Figure 5：蓝色是胜出，灰色是持平，红色是失败。** 各行与同一模型 FP16 回答比较，7B 的 RTN `(6,3,71)`、GPTQ `(4,1,75)`、AWQ `(23,5,52)`；13B 分别 `(14,9,57)`、`(17,6,57)`、`(22,11,47)`，每行合计 80。7B 相对 RTN 多 17 个胜出，13B 多 8 个；AWQ 仍有大量输给 FP16 的案例。

**边界。** 文本报告双顺序 160 次评审，图中合并到 80；具体聚合规则没有充分展开。GPT-4 evaluator、prompt 和顺序偏差影响结果；它是 instruction generalization 的证据，不是精确人工偏好胜率。

{{table6}}

**Table 6：OpenFlamingo 的 in-context captioning。** COCO CIDEr ↑，保留 0/4/8/16/32-shot 所有列、INT3/INT4 所有 baseline。在 32-shot INT4 下，FP16 81.70，RTN 77.13，GPTQ 74.98，AWQ 80.53。相对 RTN 提升 3.40 CIDEr；距 FP16 的 gap 从 4.57 缩为 1.17，即恢复约 74.40% 的 RTN gap。

**不是所有 3-bit 配置都近乎无损。** INT3 32-shot 的 AWQ 是 74.47，距 FP16 7.23。这里仅量化语言部分，vision 保持高精度；caption 的“4× model size reduction”不能未经参数分解就套到整个 VLM。

{{table7}}

**Table 7：逐项看，不替作者放大 “lossless”。** VILA-7B 的 VizWiz 59.6 → 57.8，降 1.8 points；TextVQA 62.6 → 61.9，降 0.7；POPE 86.3 → 85.3，降 1.0。若应用是 OCR，TextVQA 的下降比某个通用指标的小幅增加更值得跟踪。MME 是 score，不是百分比 accuracy，不能和其他列直接平均。

**表格完整性问题。** caption 罗列 MMBench-CN，但表体没有对应列；本文保留原 caption 与实际 11 列，不补造一个 MMB-CN 数字。部分指标上升与下降并存，未给置信区间，支持“多数指标接近 FP16”，不支持所有 benchmark 的统计无损。

{{fig6}}

**Figure 6：LLaVA-13B 的 meme 理解。** 左图是食物摆出的世界地图，右侧红字指出 RTN 将它误读成“食物上有小地球图片”，AWQ 更接近图文语义。顺序是先看图，再读 prompt，最后比较原文着色回答。它显示一次量化可能扭曲视觉概念，却不构成整个 OCR / visual reasoning 集的错误率。

{{fig7}}

**Figure 7：OpenFlamingo 三个 caption 案例。** 依次辨别玩具飞机停在草地、人物和大象、两条狗，再比较 RTN/AWQ。错误对象、动作和数量用颜色标出。与 Table 6 的 COCO 定量结果互相补充，展示失真形式；案例数量少且可能经过选择，不能推断全部实体、计数和关系都可靠。

{{table8}}

**Table 8：编程和数学需要读到第二个指标。** MBPP pass@1 FP16 38.53 → AWQ 40.64，但 pass@10 是 49.77 → 49.25；GPTQ 的 pass@1 31.97，RTN 37.51。没有重复采样、seed/CI 时，“INT4 更聪明”不是可支持结论。GSM8K 的 70B 56.41 → 56.40，13B 26.16 → 25.25，7B 13.87 → 13.57，各规模退化不同。

**Interpretation。** PPL 的微小变化和 pass@k 的随机性都可能放大或掩盖真实应用错误。部署验收应固定 decoding、prompt 和题集，按任务逐项报告，而非挑一个上涨指标。

{{table9}}

**Table 9：这是 AWQ+GPTQ 的 INT2-g64 结果。** OPT-6.7B 的 GPTQ 16.65 → 组合 15.71，改进 0.94；仍明显差于 FP16 10.86，RTN 高达 7622 已崩溃。AWQ 调整坐标、GPTQ 在新坐标内补偿误差，存在互补性。不能把这张表写成 standalone AWQ 的 2-bit 无损能力，也不能拿 g64 对 g128 当成同条件优胜。

### 5.3 数据效率与分布迁移

{{fig8}}

**Figure 8 左图。** 横轴是 calibration sequences，纵轴 PPL ↓，固定 OPT-6.7B INT3-g128。AWQ 用 16 段达到好的结果，GPTQ 需要约 192 段；这个对照是 12×，原文“10×”是概括，不能将它变成所有模型固定 10×。

**右图容易读错方向。** 行是校准集 PubMed / Enron，列是评估集 PubMed / Enron。应固定评估列，比较校准分布改变造成的 PPL 增量；两个评估集本身的 PPL 基数不能当分布迁移惩罚。

| 方法 | 校准 PubMed → 评估 PubMed | 校准 Enron → 评估 PubMed | PubMed 列的迁移惩罚 | 校准 Enron → 评估 Enron | 校准 PubMed → 评估 Enron | Enron 列的迁移惩罚 |
|---|---:|---:|---:|---:|---:|---:|
| GPTQ | 32.48 | 34.81 | +2.33 | 45.52 | 50.41 | +4.89 |
| AWQ | 32.56 | 33.16 | +0.60 | 44.57 | 45.07 | +0.50 |

这是本文根据原图重排的方向审计表。它支持在一模型、两分布的试验中 AWQ 更稳健。不能从这里证明任意 calibration set 都不会过拟合，尤其不能推导“纯文本校准保证多图 OCR 没有退化”。

**机制解释。** 简化的平均幅度 scale search 减少了需要拟合的自由度，可能弱化特定校准样本的过度依赖；但它仍用校准输出 MSE，也仍可能遗漏罕见视觉 token、代码和数值场景。原文的强泛化叙述应与这份有限实验对应。

### 5.4 速度：量化与系统优化分别贡献了多少

{{fig9}}

**Figure 9：先看硬件栏，再看 baseline。** 4090 左栏、Orin 中栏、4070 laptop 右栏。蓝色 HF FP16、灰色作者优化 FP16、红色 AWQ；OOM 是容量失败，不是低速点。各栏模型顺序不同，应按标签逐项读。

RTX 4090、Llama-2-7B：HF 52 tok/s → 优化 FP16 62 tok/s → AWQ 194 tok/s。整体 3.731×；相对优化 FP16 是 3.129×。因此不是把所有 3.7× 记给量化：baseline 改善也有贡献。Orin 同模型 11 → 12 → 39 tok/s，整体 3.545×。4070 8 GB 的 Llama-2-13B 能跑到 33 tok/s，而 FP16 OOM，属于使部署可行。

**TTFT / TPOT。** `1000/194≈5.15 ms/token`、`1000/52≈19.23 ms/token` 可以作为 steady throughput 的平均间隔换算，不能叫实测 P50/P95 TPOT，更不是 TTFT。prompt=4 的测试没有展开 image encoder、长文 prefill、历史重用和 cold model load。

{{table10}}

**Table 10：VILA 的输出速度。** 7B 的 A100 81.6 → 155.3 tok/s，1.903×；4090 58.5 → 168.1，2.873×；Orin 11.5 → 35.6，3.096×。13B 在 A100 48.5 → 102.1，2.105×；Orin 6.1 → 17.5，2.869×；4090 FP16 OOM、AWQ 99 tok/s，无法定义有限倍率。表中的 tok/s 不独立测量视觉 tower 和首 token 时延。

{{fig10}}

**Figure 10：与其他 4-bit 系统比较。** 左图是 Jetson Orin 64 GB，单位 tok/s；标题说 latency comparison，但柱子越高代表越快。Llama-2-7B 的 AutoGPTQ 13.4、llama.cpp 22.5、exllama 15.9、TinyChat 39.1；TinyChat / llama.cpp=1.738×。13B 是 8.0 / 13.3 / 9.1 / 21.2；70B 是 1.3 / 3.0 / 1.4 / 3.5，TinyChat 对最快 baseline 仅 1.167×。不能把上界 3× 套到所有模型。

StarCoder-15.5B、StableCode-3B、Mistral-7B、Falcon-7B 中 AutoGPTQ / TinyChat 分别约 `6/17、10/32、14/37、3/22`。这是当时系统支持范围和 kernel 效率的比较，不描述 2026 年 llama.cpp 的支持能力。右图 Raspberry Pi 4 上 Llama-2-7B 两个可运行系统都约 0.7 tok/s；OPT 的 NS / Not Supported 是支持缺失，不能计算无限加速。

**系统实验未覆盖的关键条件。** 更长 prompt、更大 batch、context 增长、KV 量化、vision tower、持续功耗/温控和不同格式的 exporter。TinyChat 的“把容量收益变成速度”得到了特定协议支持；性能数字需要随设备和任务重新测量。

## 6. Conclusion：原文留下的结论

论文把 activation-aware scaling、统一低比特权重和硬件执行组合起来，证明不依赖大规模重训练也可以让多个 LLM/VLM 在端侧可部署。核心观察是权重误差不等价，保护重要输入通道能够有效改善量化；系统核心是让反量化留在计算 kernel 内，减少 DRAM 和 launch。

结论中“does not over-fit”“lossless”等语言比全部表格能直接支持的范围更强。较准确的阅读是：AWQ 在论文覆盖的多个模型和任务上接近高精度，并在一个分布迁移实验里更稳健；TinyChat 在 batch=1 的指定硬件上实现了明显速度收益。

致谢和参考文献是 v6 最后的内容，没有遗漏的 Appendix、补充训练算法或可借来填补消融的独立附录。

## 7. Experiment Audit：如果我是 Reviewer 还会要求什么

| 待审查问题 | 当前证据 | 还缺的实验 | 对部署的影响 |
|---|---|---|---|
| scale 与 clip 谁贡献更多 | Table 2/3 解释 scale | 同校准集、同 bit/group 的 scale-only、clip-only、both | 选择校准复杂度与敏感层策略 |
| 泛化是否覆盖视觉异常输入 | VLM benchmark 与两个文本分布 | OCR、多图、长图片 token、罕见语言校准/测试拆分 | TextVQA 退化不能被通用 PPL 替代 |
| “lossless”是否统计成立 | 单次表格，少数指标下降 | 多 seeds、配对置信区间与失败类别 | 小改进与小退化可能是采样波动 |
| 量化搜索资源是否更少 | 无 Hessian，较少序列 | 同机器 full-model walltime、CPU/GPU peak、IO | layerwise 搜索和 CPU 缓存仍有实际成本 |
| 加速主要来自何处 | 优化 FP16 和 AWQ 两个系统层次 | unpack/convert/scale/FMA/launch 分解 | 选择 runtime 比只选 quant_method 更重要 |
| cache 与上下文是否公平 | 短 prompt、固定生成量 | 相同 prefix cache、相同 KV 精度、多 T/B 网格 | 长上下文 decode 可能改由 KV 流量限制 |
| OOM 后是否仍可用于产品 | 能跑的大模型与 tok/s | 冷启动、TTFT/TPOT 分位数、功耗和热稳态 | 装得下是必要条件，体验需单独验收 |

**复现性审计。** 原 PDF、原图表、固定代码提交、设置和 checkpoint 名称提供了可追踪起点；但论文与当前实现、图表版本和 evaluator 并非自动完全一致。Figure 2 / Table 1 数字不一致、Table 1 / 3 的 OPT-1.3B RTN 119.00 / 119.47、Table 7 caption/列差异以及旧 Table 编号均已明确保留，不通过“清理排版”隐藏。
