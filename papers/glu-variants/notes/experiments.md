# GLU Variants supporting note

[完整精读](../index.html)

## 3.1 Model Architecture：真实实验是哪一种 T5？

**Paper §3.1。** 这是 encoder-decoder Transformer：encoder **12 层**，decoder **12 层**；`D=768`，Attention 12 heads，`dₖ=dᵥ=64`。普通 FFN `H₀=3072`；全部五种门控 FFN `H=2048`。encoder、decoder 的 FFN 都替换，Attention 架构不变。

**External Source 核对。** 原始 T5 的 `bi_bert_base.gin → bi_base.gin` 配置链明确设 `num_layers=12`，与论文一致。不能拿今天某个 Hugging Face `t5-base` 的 6+6 配置，把本论文“纠正”为 6+6。这里的名字和现代 checkpoint 名称不足以替代具体配置。

**Derived 参数账。** 单个普通 FFN 是 `2×768×3072 = 4,718,592` 权重；单个门控 FFN 是 `3×768×2048 = 4,718,592` 权重。每 token、每个 FFN 的主要 GEMM 为 **9,437,184 FLOPs**。24 个 FFN 合计 **113,246,208 权重**，不含 Attention、embedding、norm 等，不能称作模型总参数。

**Interpretation / Caveat。** 这里控制了 hidden-width 对主要预算的影响，但门控 FFN 的 hidden channel 数少了 1/3。实验比较的是“更窄的两路结构”与“更宽的单路结构”这一完整方案，不能从中单独分离 gate 函数、宽度和优化路径的因果贡献。



## 3.2 Pre-Training：低 log-perplexity 是最直接的证据

**What / How — Paper §3.2。** 使用 C4 的 span-filling 去噪目标：输入删除若干文本片段，并以 sentinel 标记；decoder 预测这些缺失片段及相应分隔标记。FFN 变体不改变目标文本构造。

### 训练与推理链路重建

<figure class="generated-figure"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 400" role="img" aria-label="去噪训练与逐 token 推理的链路重建"><style>text{font:14px sans-serif;fill:#d5e1f3}.small{font-size:12px;fill:#9fb1c9}.box{fill:#182335;stroke:#49617f}.gate{stroke:#90b8f4}.up{stroke:#d9b779}.line{fill:none;stroke:#8eafdf;stroke-width:1.5}</style><rect class="box " x="40" y="15" width="700" height="65" rx="6"/><text x="56" y="41">Corrupt C4 spans → encoder</text><text class="small" x="56" y="64">Input [B,512] → encoder states [B,512,D]</text><path class="line" d="M390 80v30m-5 -6l5 6 5 -6"/><rect class="box " x="40" y="110" width="700" height="65" rx="6"/><text x="56" y="136">Teacher forcing → causal decoder</text><text class="small" x="56" y="159">Target [B,114] + cross-attention → FFN gated projections</text><path class="line" d="M390 175v30m-5 -6l5 6 5 -6"/><rect class="box " x="40" y="205" width="700" height="65" rx="6"/><text x="56" y="231">Vocabulary logits → cross-entropy</text><text class="small" x="56" y="254">Predict missing spans; backpropagate through both branches</text><path class="line" d="M390 270v30m-5 -6l5 6 5 -6"/><rect class="box " x="40" y="300" width="700" height="65" rx="6"/><text x="56" y="326">Inference: encoder once, decoder per token</text><text class="small" x="56" y="349">FFN recomputes current token; attention KV reused</text></svg><figcaption>去噪训练与逐 token 推理的链路重建 · Agent-created / Simplified Diagram。[<a href="assets/generated/training-inference.mmd">Mermaid source</a>]</figcaption></figure>

**Derived loss。** 按有效 target token 的自回归交叉熵，可写为 `L=−(1/N_valid) Σ log pθ(tᵢ|t<ᵢ, corrupted input)`。原文没有给新的 loss 公式；此式是 T5 去噪训练目标的计算形式。FFN 输出经后续 block 与词表头影响 logits，gate 本身不是独立预测头。

训练时 decoder 用 teacher forcing，并通过两路乘法共同反传。推理时 encoder 对输入运行一次，decoder 逐 token 生成；FFN 对当前位置运行，Attention 可以复用 KV。**SwiGLU 不缩短 KV cache**，也不直接改变生成规则。

| 设置 | 原文值与审计 |
|---|---|
| 数据 / 评估 | C4 去噪预训练；heldout C4 shard 上的 log-perplexity，越低越好 |
| 长训练 / 短训练 | 每变体长训练 524,288 steps；另有 4 次 65,536-step 短训练 |
| Batch / 长度 | 128 examples；input 512 tokens，target 114 tokens |
| Optimizer / LR | Adafactor；inverse-square-root schedule；最后 10% steps 线性衰减 |
| Dropout | 预训练阶段不使用 dropout；与 Raffel et al. 的原始协议有区别 |
| Hardware | 32-core TPUv2；脚注约 0.15 秒 / step |
| Precision | 原文未报告，不能补写为 BF16 或 FP32 |
| 重复与方差 | 四次短训练用于 inter-run variability；长训练没有重复结果或置信区间 |

**Derived 训练量。** 若按固定 batch shape 算 token slots，长训练 input 为 `524,288×128×512 = 34,359,738,368`，target 为 `7,650,410,496`；短训练分别是 `4,294,967,296` 与 `956,301,312`。这些是序列槽位累计数，不是去重语料 token 数；也不能把 encoder 与 target 简单合成另一论文的“训练 tokens”口径。`0.15 s/step` 是原文脚注，未提供逐变体 latency 表。

### Table 1：GEGLU 最低，SwiGLU 紧随其后

<div class="table-scroll original-table" role="region" tabindex="0" aria-label="Table 1 完整原表重排"><table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r" scope="col">Training Steps</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column" scope="col">65,536</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column" scope="col">524,288</th></tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" scope="row">FFN_ReLU (baseline)</th>
<td class="ltx_td ltx_align_left ltx_border_t">1.997 (0.005)</td>
<td class="ltx_td ltx_align_left ltx_border_t">1.677</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_GELU</th>
<td class="ltx_td ltx_align_left">1.983 (0.005)</td>
<td class="ltx_td ltx_align_left">1.679</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_Swish</th>
<td class="ltx_td ltx_align_left">1.994 (0.003)</td>
<td class="ltx_td ltx_align_left">1.683</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" scope="row">FFN_GLU</th>
<td class="ltx_td ltx_align_left ltx_border_t">1.982 (0.006)</td>
<td class="ltx_td ltx_align_left ltx_border_t">1.663</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_Bilinear</th>
<td class="ltx_td ltx_align_left">1.960 (0.005)</td>
<td class="ltx_td ltx_align_left">1.648</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_GEGLU</th>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">1.942</span> (0.004)</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">1.633</span></td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_SwiGLU</th>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">1.944</span> (0.010)</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">1.636</span></td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_ReGLU</th>
<td class="ltx_td ltx_align_left">1.953 (0.003)</td>
<td class="ltx_td ltx_align_left">1.645</td></tr>
</tbody>
</table></div><details><summary>Table 1 原始截图与完整 caption · PDF p.2</summary><figure class="original-figure"><button type="button" data-zoom="assets/original/table-01-pretraining.png" aria-label="放大 Table 1"><img src="assets/original/table-01-pretraining.png" alt="Table 1：完整原始实验表、全部指标、作者加粗和英文caption" loading="lazy"></button><figcaption>Table 1: Heldout-set log-perplexity for Transformer models on the segment-filling task from [ Raffel et al., 2019 ] . All models are matched for parameters and computation. · Original Paper v1, p.2。点击放大。</figcaption></figure></details>

**如何阅读。** 两列分别是短训练和长训练。表中指标是 heldout log-perplexity，**↓**；作者同时把 GEGLU、SwiGLU 两行结果加粗，不能把所有加粗都理解为该列唯一最优。短训练括号数字对应运行间变动的报告，但原文没有明确说明它究竟是标准差、标准误或其他统计量。

**比较对象。** 两矩阵基线有 ReLU、GELU、Swish；三矩阵方案有 GLU、Bilinear、GEGLU、SwiGLU、ReGLU。长训练的最强单路基线是 ReLU=1.677，GELU=1.679、Swish=1.683 都略高。

**关键结果与重算。** GEGLU=1.633，比 ReLU 低 **0.044 log units**；SwiGLU=1.636，低 **0.041**；ReGLU=1.645，低 0.032；Bilinear=1.648，低 0.029；GLU=1.663，低 0.014。若 log 为自然对数，perplexity 比例为 `exp(ΔlogPPL)`，GEGLU 约降 **4.30%**，SwiGLU 约降 **4.02%**。原文未明示 log 底，因此比例换算带这一前提；不能把 `0.044/1.677` 当作 perplexity 降幅。

**公平性与 trade-off。** 同篇八种变体的主数据、步数与模型外围配置受控，支持“这组门控方案改善该去噪目标”。相同主要 GEMM 预算，不等于实测训练墙钟相等。短训练 GEGLU=1.942、SwiGLU=1.944 的差距很小；SwiGLU 括号为 0.010。缺少长训练多 seed，不能据此宣称 GEGLU 显著优于 SwiGLU。



## 3.3 Fine-Tuning：迁移收益并非每列一致

**Paper §3.3。** 每个预训练模型只做一次联合微调，将 SQuAD、GLUE、SuperGLUE 按 examples-proportional 混合。总共 131,072 steps，LR=`10⁻³`，每步 input 合计约 65,536 tokens；layer output、FFN hidden、Attention weights 的 dropout 为 0.1，embedding 冻结。

**评估口径。** 报告 dev 集结果，每个 task 选已记录 checkpoint 中的最好分数。因此表内多个任务不保证来自一个共同最佳 checkpoint，也不是 leaderboard test 结果。Raffel 的旧结果来自分别微调的协议，且预训练 dropout 不同。原文作者把旧结果较差归因于 dropout，这是作者判断；本实验没有完整控制这些变化来证明唯一因果。

### 先读懂指标，避免平均分掩盖任务差异

GLUE / SuperGLUE / SQuAD 的这些表均为 **↑**。`Acc=(TP+TN)/(TP+TN+FP+FN)`；分类 `Precision=TP/(TP+FP)`，`Recall=TP/(TP+FN)`，`F1=2PR/(P+R)`。CoLA 用 MCC，STSB 用 Pearson / Spearman 相关系数。SQuAD 的 EM 衡量规范化答案精确匹配，F1 衡量答案 token 重叠，与分类的混淆矩阵语义不同；MultiRC、ReCoRD 也有各自评估对象与聚合方式。

表中只有聚合指标，没有预测逐项结果，无法还原 TP / FP / TN / FN，也无法补出 Precision / Recall。本文保留全部列，不把 F1、Acc、EM 当作同一种分数相加。

### Table 2：GLUE 平均分由 ReGLU 领先

<div class="table-scroll original-table" role="region" tabindex="0" aria-label="Table 2 完整原表重排"><table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r" scope="col"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" scope="col">Score</th>
<th class="ltx_td ltx_align_center" scope="col">CoLA</th>
<th class="ltx_td ltx_align_center" scope="col">SST-2</th>
<th class="ltx_td ltx_align_center" scope="col">MRPC</th>
<th class="ltx_td ltx_align_center" scope="col">MRPC</th>
<th class="ltx_td ltx_align_center" scope="col">STSB</th>
<th class="ltx_td ltx_align_center" scope="col">STSB</th>
<th class="ltx_td ltx_align_center" scope="col">QQP</th>
<th class="ltx_td ltx_align_center" scope="col">QQP</th>
<th class="ltx_td ltx_align_center" scope="col">MNLIm</th>
<th class="ltx_td ltx_align_center" scope="col">MNLImm</th>
<th class="ltx_td ltx_align_center" scope="col">QNLI</th>
<th class="ltx_td ltx_align_center" scope="col">RTE</th></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r" scope="col"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" scope="col">Average</th>
<th class="ltx_td ltx_align_center" scope="col">MCC</th>
<th class="ltx_td ltx_align_center" scope="col">Acc</th>
<th class="ltx_td ltx_align_center" scope="col">F1</th>
<th class="ltx_td ltx_align_center" scope="col">Acc</th>
<th class="ltx_td ltx_align_center" scope="col">PCC</th>
<th class="ltx_td ltx_align_center" scope="col">SCC</th>
<th class="ltx_td ltx_align_center" scope="col">F1</th>
<th class="ltx_td ltx_align_center" scope="col">Acc</th>
<th class="ltx_td ltx_align_center" scope="col">Acc</th>
<th class="ltx_td ltx_align_center" scope="col">Acc</th>
<th class="ltx_td ltx_align_center" scope="col">Acc</th>
<th class="ltx_td ltx_align_center" scope="col">Acc</th></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" scope="row">FFN_ReLU</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">83.80</td>
<td class="ltx_td ltx_align_center ltx_border_t">51.32</td>
<td class="ltx_td ltx_align_center ltx_border_t">94.04</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">93.08</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">90.20</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">89.64</td>
<td class="ltx_td ltx_align_center ltx_border_t">89.42</td>
<td class="ltx_td ltx_align_center ltx_border_t">89.01</td>
<td class="ltx_td ltx_align_center ltx_border_t">91.75</td>
<td class="ltx_td ltx_align_center ltx_border_t">85.83</td>
<td class="ltx_td ltx_align_center ltx_border_t">86.42</td>
<td class="ltx_td ltx_align_center ltx_border_t">92.81</td>
<td class="ltx_td ltx_align_center ltx_border_t">80.14</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_GELU</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">83.86</td>
<td class="ltx_td ltx_align_center">53.48</td>
<td class="ltx_td ltx_align_center">94.04</td>
<td class="ltx_td ltx_align_center">92.81</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">90.20</span></td>
<td class="ltx_td ltx_align_center">89.69</td>
<td class="ltx_td ltx_align_center">89.49</td>
<td class="ltx_td ltx_align_center">88.63</td>
<td class="ltx_td ltx_align_center">91.62</td>
<td class="ltx_td ltx_align_center">85.89</td>
<td class="ltx_td ltx_align_center">86.13</td>
<td class="ltx_td ltx_align_center">92.39</td>
<td class="ltx_td ltx_align_center">80.51</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_Swish</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">83.60</td>
<td class="ltx_td ltx_align_center">49.79</td>
<td class="ltx_td ltx_align_center">93.69</td>
<td class="ltx_td ltx_align_center">92.31</td>
<td class="ltx_td ltx_align_center">89.46</td>
<td class="ltx_td ltx_align_center">89.20</td>
<td class="ltx_td ltx_align_center">88.98</td>
<td class="ltx_td ltx_align_center">88.84</td>
<td class="ltx_td ltx_align_center">91.67</td>
<td class="ltx_td ltx_align_center">85.22</td>
<td class="ltx_td ltx_align_center">85.02</td>
<td class="ltx_td ltx_align_center">92.33</td>
<td class="ltx_td ltx_align_center">81.23</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" scope="row">FFN_GLU</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">84.20</td>
<td class="ltx_td ltx_align_center ltx_border_t">49.16</td>
<td class="ltx_td ltx_align_center ltx_border_t">94.27</td>
<td class="ltx_td ltx_align_center ltx_border_t">92.39</td>
<td class="ltx_td ltx_align_center ltx_border_t">89.46</td>
<td class="ltx_td ltx_align_center ltx_border_t">89.46</td>
<td class="ltx_td ltx_align_center ltx_border_t">89.35</td>
<td class="ltx_td ltx_align_center ltx_border_t">88.79</td>
<td class="ltx_td ltx_align_center ltx_border_t">91.62</td>
<td class="ltx_td ltx_align_center ltx_border_t">86.36</td>
<td class="ltx_td ltx_align_center ltx_border_t">86.18</td>
<td class="ltx_td ltx_align_center ltx_border_t">92.92</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">84.12</span></td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_GEGLU</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">84.12</td>
<td class="ltx_td ltx_align_center">53.65</td>
<td class="ltx_td ltx_align_center">93.92</td>
<td class="ltx_td ltx_align_center">92.68</td>
<td class="ltx_td ltx_align_center">89.71</td>
<td class="ltx_td ltx_align_center">90.26</td>
<td class="ltx_td ltx_align_center">90.13</td>
<td class="ltx_td ltx_align_center">89.11</td>
<td class="ltx_td ltx_align_center">91.85</td>
<td class="ltx_td ltx_align_center">86.15</td>
<td class="ltx_td ltx_align_center">86.17</td>
<td class="ltx_td ltx_align_center">92.81</td>
<td class="ltx_td ltx_align_center">79.42</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_Bilinear</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">83.79</td>
<td class="ltx_td ltx_align_center">51.02</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">94.38</span></td>
<td class="ltx_td ltx_align_center">92.28</td>
<td class="ltx_td ltx_align_center">89.46</td>
<td class="ltx_td ltx_align_center">90.06</td>
<td class="ltx_td ltx_align_center">89.84</td>
<td class="ltx_td ltx_align_center">88.95</td>
<td class="ltx_td ltx_align_center">91.69</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">86.90</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">87.08</span></td>
<td class="ltx_td ltx_align_center">92.92</td>
<td class="ltx_td ltx_align_center">81.95</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_SwiGLU</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">84.36</td>
<td class="ltx_td ltx_align_center">51.59</td>
<td class="ltx_td ltx_align_center">93.92</td>
<td class="ltx_td ltx_align_center">92.23</td>
<td class="ltx_td ltx_align_center">88.97</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">90.32</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">90.13</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">89.14</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">91.87</span></td>
<td class="ltx_td ltx_align_center">86.45</td>
<td class="ltx_td ltx_align_center">86.47</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">92.93</span></td>
<td class="ltx_td ltx_align_center">83.39</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_ReGLU</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text ltx_font_bold">84.67</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">56.16</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">94.38</span></td>
<td class="ltx_td ltx_align_center">92.06</td>
<td class="ltx_td ltx_align_center">89.22</td>
<td class="ltx_td ltx_align_center">89.97</td>
<td class="ltx_td ltx_align_center">89.85</td>
<td class="ltx_td ltx_align_center">88.86</td>
<td class="ltx_td ltx_align_center">91.72</td>
<td class="ltx_td ltx_align_center">86.20</td>
<td class="ltx_td ltx_align_center">86.40</td>
<td class="ltx_td ltx_align_center">92.68</td>
<td class="ltx_td ltx_align_center">81.59</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_tt" scope="row"><cite class="ltx_cite ltx_citemacro_citep">[<a class="ltx_ref" href="https://arxiv.org/html/2002.05202v1#bib.bib5" title="">Raffel et al., 2019</a>]</cite></th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_tt">83.28</td>
<td class="ltx_td ltx_align_center ltx_border_tt">53.84</td>
<td class="ltx_td ltx_align_center ltx_border_tt">92.68</td>
<td class="ltx_td ltx_align_center ltx_border_tt">92.07</td>
<td class="ltx_td ltx_align_center ltx_border_tt">88.92</td>
<td class="ltx_td ltx_align_center ltx_border_tt">88.02</td>
<td class="ltx_td ltx_align_center ltx_border_tt">87.94</td>
<td class="ltx_td ltx_align_center ltx_border_tt">88.67</td>
<td class="ltx_td ltx_align_center ltx_border_tt">91.56</td>
<td class="ltx_td ltx_align_center ltx_border_tt">84.24</td>
<td class="ltx_td ltx_align_center ltx_border_tt">84.57</td>
<td class="ltx_td ltx_align_center ltx_border_tt">90.48</td>
<td class="ltx_td ltx_align_center ltx_border_tt">76.28</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">ibid. stddev.</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">0.235</td>
<td class="ltx_td ltx_align_center">1.111</td>
<td class="ltx_td ltx_align_center">0.569</td>
<td class="ltx_td ltx_align_center">0.729</td>
<td class="ltx_td ltx_align_center">1.019</td>
<td class="ltx_td ltx_align_center">0.374</td>
<td class="ltx_td ltx_align_center">0.418</td>
<td class="ltx_td ltx_align_center">0.108</td>
<td class="ltx_td ltx_align_center">0.070</td>
<td class="ltx_td ltx_align_center">0.291</td>
<td class="ltx_td ltx_align_center">0.231</td>
<td class="ltx_td ltx_align_center">0.361</td>
<td class="ltx_td ltx_align_center">1.393</td></tr>
</tbody>
</table></div><details><summary>Table 2 原始截图与完整 caption · PDF p.3</summary><figure class="original-figure"><button type="button" data-zoom="assets/original/table-02-glue.png" aria-label="放大 Table 2"><img src="assets/original/table-02-glue.png" alt="Table 2：完整原始实验表、全部指标、作者加粗和英文caption" loading="lazy"></button><figcaption>Table 2: GLUE Language-Understanding Benchmark [ Wang et al., 2018 ] (dev). · Original Paper v1, p.3。点击放大。</figcaption></figure></details>

**比较与提升。** 当前实验八行中 ReGLU 的原表 `Score Average=84.67` 最高，对 ReLU=83.80 为 **+0.87 个百分点**，相对所报分数约 +1.04%；SwiGLU=84.36 为 +0.56，GEGLU=84.12 为 +0.32。单路最强 GELU=83.86，对它 ReGLU 的优势是 +0.81，SwiGLU +0.50。

**逐项看证据。** ReGLU CoLA=56.16，对 ReLU=51.32 提升 +4.84；SwiGLU STSB PCC=90.32、QQP F1=89.14、QNLI=92.93 较强。Bilinear 的 MNLI matched / mismatched 为 86.90 / 87.08，GLU 的 RTE=84.12。相反，MRPC F1 的 ReLU=93.08 高于所有门控变体；GEGLU 的 RTE=79.42 低于 ReLU=80.14。gate 家族并不逐列统治。

**重要口径缺口。** 用表中展示的最佳任务值，按常见 GLUE 任务分组平均，ReLU 可算出约 84.4981，而原表写 83.80。原文没有提供足够的 average / checkpoint 聚合细节来消除差异。本文保留作者数值；“不同 checkpoint 的聚合可能导致”只能是待核假说，不能写成已确认原因。上面的平均分提升均基于 **原表所报 aggregate**。

**公平性。** 八种新实验使用联合微调，但各只有一轮；旧 Raffel 行及 `stddev.` 仅作历史参照。它们的标准差不能用作本次 ReGLU 或 SwiGLU 的显著性检验。

### Table 3：SuperGLUE 的 aggregate 由 SwiGLU 领先

<div class="table-scroll original-table" role="region" tabindex="0" aria-label="Table 3 完整原表重排"><table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_column ltx_th_row ltx_border_r" scope="col"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r" scope="col">Score</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">BoolQ</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">CB</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">CB</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">CoPA</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">MultiRC</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">MultiRC</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">ReCoRD</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">ReCoRD</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">RTE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">WiC</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">WSC</th></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_column ltx_th_row ltx_border_r" scope="col"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r" scope="col">Average</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">Acc</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">F1</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">Acc</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">Acc</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">F1</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">EM</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">F1</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">EM</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">Acc</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">Acc</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column" scope="col">Acc</th></tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" scope="row">FFN_ReLU</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">72.76</td>
<td class="ltx_td ltx_align_center ltx_border_t">80.15</td>
<td class="ltx_td ltx_align_center ltx_border_t">83.37</td>
<td class="ltx_td ltx_align_center ltx_border_t">89.29</td>
<td class="ltx_td ltx_align_center ltx_border_t">70.00</td>
<td class="ltx_td ltx_align_center ltx_border_t">76.93</td>
<td class="ltx_td ltx_align_center ltx_border_t">39.14</td>
<td class="ltx_td ltx_align_center ltx_border_t">73.73</td>
<td class="ltx_td ltx_align_center ltx_border_t">72.91</td>
<td class="ltx_td ltx_align_center ltx_border_t">83.39</td>
<td class="ltx_td ltx_align_center ltx_border_t">67.71</td>
<td class="ltx_td ltx_align_center ltx_border_t">77.88</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_GELU</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">72.98</td>
<td class="ltx_td ltx_align_center">80.64</td>
<td class="ltx_td ltx_align_center">86.24</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">91.07</span></td>
<td class="ltx_td ltx_align_center">74.00</td>
<td class="ltx_td ltx_align_center">75.93</td>
<td class="ltx_td ltx_align_center">38.61</td>
<td class="ltx_td ltx_align_center">72.96</td>
<td class="ltx_td ltx_align_center">72.03</td>
<td class="ltx_td ltx_align_center">81.59</td>
<td class="ltx_td ltx_align_center">68.34</td>
<td class="ltx_td ltx_align_center">75.96</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_Swish</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">72.40</td>
<td class="ltx_td ltx_align_center">80.43</td>
<td class="ltx_td ltx_align_center">77.75</td>
<td class="ltx_td ltx_align_center">83.93</td>
<td class="ltx_td ltx_align_center">67.00</td>
<td class="ltx_td ltx_align_center">76.34</td>
<td class="ltx_td ltx_align_center">39.14</td>
<td class="ltx_td ltx_align_center">73.34</td>
<td class="ltx_td ltx_align_center">72.36</td>
<td class="ltx_td ltx_align_center">81.95</td>
<td class="ltx_td ltx_align_center">68.18</td>
<td class="ltx_td ltx_align_center">81.73</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" scope="row">FFN_GLU</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">73.95</td>
<td class="ltx_td ltx_align_center ltx_border_t">80.95</td>
<td class="ltx_td ltx_align_center ltx_border_t">77.26</td>
<td class="ltx_td ltx_align_center ltx_border_t">83.93</td>
<td class="ltx_td ltx_align_center ltx_border_t">73.00</td>
<td class="ltx_td ltx_align_center ltx_border_t">76.07</td>
<td class="ltx_td ltx_align_center ltx_border_t">39.03</td>
<td class="ltx_td ltx_align_center ltx_border_t">74.22</td>
<td class="ltx_td ltx_align_center ltx_border_t">73.50</td>
<td class="ltx_td ltx_align_center ltx_border_t">84.12</td>
<td class="ltx_td ltx_align_center ltx_border_t">67.71</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">87.50</span></td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_GEGLU</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">73.96</td>
<td class="ltx_td ltx_align_center">81.19</td>
<td class="ltx_td ltx_align_center">82.09</td>
<td class="ltx_td ltx_align_center">87.50</td>
<td class="ltx_td ltx_align_center">72.00</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">77.43</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">41.03</span></td>
<td class="ltx_td ltx_align_center">75.28</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">74.60</span></td>
<td class="ltx_td ltx_align_center">83.39</td>
<td class="ltx_td ltx_align_center">67.08</td>
<td class="ltx_td ltx_align_center">83.65</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_Bilinear</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">73.81</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">81.53</span></td>
<td class="ltx_td ltx_align_center">82.49</td>
<td class="ltx_td ltx_align_center">89.29</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">76.00</span></td>
<td class="ltx_td ltx_align_center">76.04</td>
<td class="ltx_td ltx_align_center">40.92</td>
<td class="ltx_td ltx_align_center">74.97</td>
<td class="ltx_td ltx_align_center">74.10</td>
<td class="ltx_td ltx_align_center">82.67</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">69.28</span></td>
<td class="ltx_td ltx_align_center">78.85</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_SwiGLU</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text ltx_font_bold">74.56</span></td>
<td class="ltx_td ltx_align_center">81.19</td>
<td class="ltx_td ltx_align_center">82.39</td>
<td class="ltx_td ltx_align_center">89.29</td>
<td class="ltx_td ltx_align_center">73.00</td>
<td class="ltx_td ltx_align_center">75.56</td>
<td class="ltx_td ltx_align_center">38.72</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">75.35</span></td>
<td class="ltx_td ltx_align_center">74.55</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">85.20</span></td>
<td class="ltx_td ltx_align_center">67.24</td>
<td class="ltx_td ltx_align_center">86.54</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_ReGLU</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">73.66</td>
<td class="ltx_td ltx_align_center">80.89</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">86.37</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">91.07</span></td>
<td class="ltx_td ltx_align_center">67.00</td>
<td class="ltx_td ltx_align_center">75.32</td>
<td class="ltx_td ltx_align_center">40.50</td>
<td class="ltx_td ltx_align_center">75.07</td>
<td class="ltx_td ltx_align_center">74.18</td>
<td class="ltx_td ltx_align_center">84.48</td>
<td class="ltx_td ltx_align_center">67.40</td>
<td class="ltx_td ltx_align_center">79.81</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_tt" scope="row"><cite class="ltx_cite ltx_citemacro_citep">[<a class="ltx_ref" href="https://arxiv.org/html/2002.05202v1#bib.bib5" title="">Raffel et al., 2019</a>]</cite></th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_tt">71.36</td>
<td class="ltx_td ltx_align_center ltx_border_tt">76.62</td>
<td class="ltx_td ltx_align_center ltx_border_tt">91.22</td>
<td class="ltx_td ltx_align_center ltx_border_tt">91.96</td>
<td class="ltx_td ltx_align_center ltx_border_tt">66.20</td>
<td class="ltx_td ltx_align_center ltx_border_tt">66.13</td>
<td class="ltx_td ltx_align_center ltx_border_tt">25.78</td>
<td class="ltx_td ltx_align_center ltx_border_tt">69.05</td>
<td class="ltx_td ltx_align_center ltx_border_tt">68.16</td>
<td class="ltx_td ltx_align_center ltx_border_tt">75.34</td>
<td class="ltx_td ltx_align_center ltx_border_tt">68.04</td>
<td class="ltx_td ltx_align_center ltx_border_tt">78.56</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">ibid. stddev.</th>
<td class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">0.416</td>
<td class="ltx_td ltx_align_center">0.365</td>
<td class="ltx_td ltx_align_center">3.237</td>
<td class="ltx_td ltx_align_center">2.560</td>
<td class="ltx_td ltx_align_center">2.741</td>
<td class="ltx_td ltx_align_center">0.716</td>
<td class="ltx_td ltx_align_center">1.011</td>
<td class="ltx_td ltx_align_center">0.370</td>
<td class="ltx_td ltx_align_center">0.379</td>
<td class="ltx_td ltx_align_center">1.228</td>
<td class="ltx_td ltx_align_center">0.850</td>
<td class="ltx_td ltx_align_center">2.029</td></tr>
</tbody>
</table></div><details><summary>Table 3 原始截图与完整 caption · PDF p.4</summary><figure class="original-figure"><button type="button" data-zoom="assets/original/table-03-superglue.png" aria-label="放大 Table 3"><img src="assets/original/table-03-superglue.png" alt="Table 3：完整原始实验表、全部指标、作者加粗和英文caption" loading="lazy"></button><figcaption>Table 3: SuperGLUE Language-Understanding Benchmark [ Wang et al., 2019 ] (dev). · Original Paper v1, p.4。点击放大。</figcaption></figure></details>

**关键结果。** SwiGLU=74.56 对 ReLU=72.76 为 **+1.80 个百分点**，相对所报分数约 +2.47%；对最强单路 GELU=72.98 为 +1.58。GEGLU=73.96 比 ReLU +1.20。SwiGLU 的 RTE=85.20、ReCoRD F1=75.35 较强，GLU 的 WSC=87.50 最佳；Bilinear 的 BoolQ=81.53、CoPA=76.00、WiC=69.28 最佳。

**反例与 trade-off。** SwiGLU MultiRC F1=75.56、EM=38.72，分别低于 ReLU 的 76.93、39.14；WiC=67.24 也低于 67.71。GEGLU 的 MultiRC=77.43 / 41.03 最佳。这说明较好的去噪 loss 可以与部分下游退步并存。

**Caveat。** 对展示的任务列做常见分组平均，ReLU 约 74.6019，也无法重建原报 72.76。沿用上节的保留与标注策略，不自行覆盖 aggregate。旧 Raffel CB F1 / Acc 甚至高于本次多行，提醒读者不要把协议变化概括成所有任务都更好。没有本轮重复微调方差，+1.80 是观察值而非显著性结论。

### Table 4：SQuAD EM 与 F1 分别有不同赢家

<div class="table-scroll original-table" role="region" tabindex="0" aria-label="Table 4 完整原表重排"><table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r" scope="col"></th>
<th class="ltx_td ltx_align_left" scope="col">EM</th>
<th class="ltx_td ltx_align_left" scope="col">F1</th></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" scope="row">FFN_ReLU</th>
<td class="ltx_td ltx_align_left ltx_border_t">83.18</td>
<td class="ltx_td ltx_align_left ltx_border_t">90.87</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_GELU</th>
<td class="ltx_td ltx_align_left">83.09</td>
<td class="ltx_td ltx_align_left">90.79</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_Swish</th>
<td class="ltx_td ltx_align_left">83.25</td>
<td class="ltx_td ltx_align_left">90.76</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" scope="row">FFN_GLU</th>
<td class="ltx_td ltx_align_left ltx_border_t">82.88</td>
<td class="ltx_td ltx_align_left ltx_border_t">90.69</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_GEGLU</th>
<td class="ltx_td ltx_align_left">83.55</td>
<td class="ltx_td ltx_align_left">91.12</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_Bilinear</th>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">83.82</span></td>
<td class="ltx_td ltx_align_left">91.06</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_SwiGLU</th>
<td class="ltx_td ltx_align_left">83.42</td>
<td class="ltx_td ltx_align_left">91.03</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">FFN_ReGLU</th>
<td class="ltx_td ltx_align_left">83.53</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">91.18</span></td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_tt" scope="row"><cite class="ltx_cite ltx_citemacro_citep">[<a class="ltx_ref" href="https://arxiv.org/html/2002.05202v1#bib.bib5" title="">Raffel et al., 2019</a>]</cite></th>
<td class="ltx_td ltx_align_left ltx_border_tt">80.88</td>
<td class="ltx_td ltx_align_left ltx_border_tt">88.81</td></tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" scope="row">ibid. Standard Deviation</th>
<td class="ltx_td ltx_align_left">0.343</td>
<td class="ltx_td ltx_align_left">0.226</td></tr>
</tbody>
</table></div><details><summary>Table 4 原始截图与完整 caption · PDF p.4</summary><figure class="original-figure"><button type="button" data-zoom="assets/original/table-04-squad.png" aria-label="放大 Table 4"><img src="assets/original/table-04-squad.png" alt="Table 4：完整原始实验表、全部指标、作者加粗和英文caption" loading="lazy"></button><figcaption>Table 4: SQuAD [ Rajpurkar et al., 2016 ] v1.1 (dev). · Original Paper v1, p.4。点击放大。</figcaption></figure></details>

**如何阅读。** EM / F1 都越高越好，SQuAD v1.1 dev。八行同篇实验之后保留旧 Raffel 行和其 Standard Deviation；原始加粗保留，没有增添原文不存在的下划线或箭头。

**重算。** Bilinear EM=83.82 最高，对 ReLU=83.18 为 **+0.64 个百分点**，相对约 +0.77%；ReGLU F1=91.18 最高，对 ReLU=90.87 为 **+0.31 个百分点**，相对约 +0.34%。GEGLU=83.55 / 91.12，SwiGLU=83.42 / 91.03；GLU=82.88 / 90.69 两项都低于 ReLU。

**最强普通基线。** 若按两矩阵变体逐指标找，EM 最佳是 Swish=83.25，F1 最佳是 ReLU=90.87。因此 Bilinear 对最强普通 EM 是 +0.57；ReGLU 对最强普通 F1 是 +0.31。把两种 metric 的最佳硬归于一行会损失真实信息。

**Claim 边界。** 这张表支持某些门控变体的小幅增益；不支持“sigmoid gate 必然有利”，也没有跨 seed 的显著性证据。论文说“许多下游任务更好”，比“所有任务更好”更贴近完整表。



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
