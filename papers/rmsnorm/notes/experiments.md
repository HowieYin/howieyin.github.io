## 6. Experiments：先看质量，再看统一口径的时间

**原文 §6，PDF pp.5–9。** 未另说明时使用 TITAN X Pascal，计时是 3 次运行的均值与标准差。Transformer 改用 Tesla V100，CIFAR 改用 RTX 2080 Ti。论文横跨 TensorFlow、Theano、PyTorch，多样性有帮助，但不同框架 baseline 的细节也有差异，不能跨表把时间直接相除。

本文统一使用 **耗时降幅=(tLN−tnew)/tLN**，加速倍数=tLN/tnew。原表括号的百分比按前者保留。更低的 time 与 error 更好，更高的 BLEU 与 Recall 更好；未标注方向的原表不擅自改写，方向在解释中补充。

### 6.1 Machine Translation：RNN 与 Transformer 的收益不相同

**WMT14 En→De；原文 §6.1，PDF pp.6–7。** RNNSearch 在 recurrent connections 与 feed-forward layers 中加入归一化；Transformer 使用 base setting、训练 300k steps。配置详情见附录 A.1。

{{fig:2}}

**Figure 2 — TensorFlow RNNSearch 的开发集 BLEU。** 横轴为 training steps（每 30k 评估一次），纵轴为 newstest2013 SacreBLEU。先比较达到同一 BLEU 的步数，再比较平台值：LayerNorm、RMSNorm 与 pRMSNorm 比 Baseline 更快收敛，L2-Norm 较弱。图中没有时间横轴，因此不能仅凭曲线宣布实际更快；需要 Table 2。

{{table:2}}

**Table 2 — 相近质量，更低训练耗时。** 完整 RMS 相对 LN 的 Test14 为 −0.2 BLEU，Test17 为 +0.1；501 s 相对 665 s，降幅 24.66%，加速 1.327×。pRMS 的 Test14 持平、Test17 为 −0.5；493 s 降幅 25.86%，加速 1.349×。二者都比无归一化 Baseline 的 399 s 慢。作者称收敛步数约减少 50%，需要把每步成本与达标步数结合，才得到训练预算意义。

L2 不是最强 baseline：它只有 20.7／22.0 BLEU，且论文脚注承认前人只对最后一层使用 L2 并调尺度，本实验不是该方法的完整复现。不能将差结果归因于范数本身而忽略 placement 与调参。

{{fig:3}}

**Figure 3 — p 的消融。** 横轴是 partial ratio，纵轴是开发集 BLEU；TensorFlow Nematus 以 10% 步长改变 p。各点接近说明此 RNNSearch 配置对 p 不太敏感，但图没有多 seed 质量方差，也不是所有比率都充分测试。6.25% 是随后采用的设置，不能宣称图中每个点都直接证明 6.25% 最优。原 §6.1 的文字误指 Figure 7，PDF 实际对应本图；TeX 使用了重复 label。

{{table:3}}

**Table 3 — 框架依赖明显。** Theano 中 RMS 的 988→652 s 为 34.01% 降幅、1.515×，Test14 +0.2、Test17 −0.6 BLEU；pRMS 的 658 s 比完整 RMS 多 6 s，却获得 22.7／24.0 BLEU。PyTorch 中 RMS 的 857→763 s 为 10.97%、1.123×，Test14 −0.3、Test17 +0.2；Test17 最佳反而是无归一化 Baseline 的 24.7。这更支持“总体质量相近，时间可改善”，不支持“所有数据集指标全面胜出”。

{{table:4}}

**Table 4 — Transformer 的成本占比更低。** RMS 的 248→231 s 为 6.85%、1.074×；pRMS 的 225 s 为 9.27%、1.102×。RMS Test14 +0.2 BLEU、Test17 持平。Baseline 的 “−” 表示训练失败且 BLEU 为 0，210 s 只能用于说明无 Norm 的执行时间，不能作为同质量最快方案。RNN 归一化沿时间串行，Transformer 更易并行，这是作者对收益差异的解释；缺少独立 kernel trace，尚不能把全部差异精确归因给某一次 reduction。

{{table:5}}

**Table 5 — 均值不归零，也能更稳定。** M=mean，S=standard deviation；列 1–4 是 token 位置，ALL 是所有位置的平均。Baseline 的 S 从 7.35 跌到约 2.3–2.7，LN 与 RMS 基本维持约 1.5；RMS 的 M 从 −0.40 逐步到 −0.74，而 LN 也不是 0。这些测量是在指定 GRU hidden-to-hidden 表示上、包含模型上下文，不能误当成“裸 LN 核心输出仍不为零”的反证。它支持 RMS 在此模型中稳定统计，但没有完整分布尾部、跨层或跨 seed 的诊断。

{{fig:4}}

**Figure 4 — 将权重初始化中心移到 0.2。** 横轴 training steps，纵轴 dev BLEU；RMS 曲线比 LN 稳定，两者都低于正常初始化结果。先读纵轴平台，再看波动，不要把某个局部高点当最终质量。一个异常中心的 case 只能检验有限的 robustness，不能证明对任意 mean drift 都更稳；毕竟 RMS 在代数上仍不具备平移不变性。

### 6.2 CNN/Daily Mail Reading Comprehension：实体填空，不是摘要任务

**原文 §6.2，PDF pp.7–8。** 标题提 CNN/Daily Mail，实际实验训练在 CNN corpus；问题答案是 passage 中匿名化的实体。使用双向 attentive reader，top4 预处理，每篇最多 4 句。归一化只施于 LSTM，隐藏维度 240。

{{fig:5}}

**Figure 5 — 看验证错误率的下降。** 横轴 training steps，纵轴 validation error，越低越好。BatchNorm-LSTM 与 BatchNorm-Everywhere 均加快收敛，LN 与 RMS 曲线更低且相近，pRMS 稍弱。legend 区分归一化方法及 BatchNorm placement；不把后者当同一个算子。图提供收敛轨迹，没有正式完整 test 指标表，也没有 TP/FP/TN/FN。

{{table:6}}

**Table 6 — 每 100 steps 的时间。** RMS 392→333 s，降幅 15.05%、1.177×；pRMS 330 s，15.82%、1.188×。相对无 Norm 的 315 s 仍有开销。原脚注明确 BatchNorm 基于 cuDNN，不能直接把其 345／348 s 与其余非同等优化实现比较，进而宣布算法复杂度排名。

### 6.3 Image-Caption Retrieval：收益很大，也更受旧实现影响

**原文 §6.3，PDF pp.8–9。** COCO 图文双向检索。OE（order embedding）用 GRU 编码文本、预训练 VGG 的 10crop 特征表示图像，以 pairwise ranking loss 学联合空间。Caption retrieval 是 image→caption，image retrieval 是 caption→image。

{{fig:6}}

**Figure 6 — 三个 K，同一个训练过程。** 左到右为 Recall@1、@5、@10，横轴为 training steps，纵轴为 recall；每 300 steps 评估。先比较同一面板内方法，再跨面板看难度：K 越大，允许命中候选更多，Recall 通常越高。LN 与 RMS 较早到达更高曲线，RMS 在验证集略优。图不含 mean rank，也不能替代 Table 7 的测试集结果。

{{table:7}}

**Table 7 — 保留两方向的所有指标。** R@K 是正确候选进入 top-K 的查询比例，越高越好；Mean r 越低越好，不是 median rank。上半部来自 existing work，下半部才是本文自己复现的对照，前者存在实现与随机性差异，不应优先拿来算因果提升。

以 This Work 的 LN 为公平对照：RMS caption R@1 47.9→48.7，+0.8 个百分点，相对 +1.67%；image R@1 38.4→39.0，+0.6 个百分点，相对 +1.56%。Caption R@10 +0.3，但 image R@10 86.7→86.3 为 −0.4；caption Mean r 5.3 持平，image Mean r 7.5 持平。pRMS 的 caption R@1 反而是 46.8。没有一个方法在全部列严格占优；原文“better generalization”应读成总体观察。

每个测试 split 含 1000 图像、5000 captions，报告 5 个测试集合的平均；这不是 5 个训练 seed。Recall@K 的检索定义不提供完整二分类 confusion matrix，不能从表中虚构 TP/FP/TN/FN、Precision 或 F1。

{{table:8}}

**Table 8 — 摘要最高数字的来源。** LN 12.02→RMS 7.12 s，为 40.77% 耗时降低、1.688×；pRMS 4.34 s 为 63.89%、2.770×。单位是每 100 steps。Baseline 仅 2.11 s，所以 pRMS 仍约 2.06×于无 Norm 时间。原文 Table 8 没有内核融合、精度、带宽、launch 或 profiler 细节；旧 Theano RNN 上的巨大 overhead 不应直接迁移到今天单 kernel 的 RMS/LN 比较。

### 6.4 CIFAR-10 Classification：partial approximation 的代价显现

**原文 §6.4，PDF p.9。** Modified ConvPool-CNN-C；归一化沿图像 width/height，gain 和 bias 沿 channel。这里不是今天 LLM 最后一维 token RMSNorm 的同样轴。模型较小，p 调到 **12.5%**，不是前面通用的 6.25%。

{{table:9}}

**Table 9（原文编号）— 实际为训练曲线。** 横轴 epoch，纵轴 training error，越低越好。BN 收敛最好，各种 Norm 比 Baseline 快；训练曲线上的优势不保证测试 error 同样更低。原文 §6.4 叫它 “Figure 9”，但 PDF caption 及 TeX table 环境实际使它编号为 Table 9；本文按 PDF 原样保留，不自行重排后面的 Table 10。

{{table:10}}

**Table 10 — 最强质量对照是 BatchNorm。** BN test error 8.25%，WeightNorm 8.28%，均优于 RMS 8.83%；RMS 优于 LN 10.49%。RMS 相对 LN 降 1.66 个百分点，error 相对减少 15.82%；时间 39→31 s，降幅 20.51%、1.258×。RMS 相对无 Norm 的 8.96% 只改善 **0.13 个百分点**，相对减少 1.45%，不能写成原文正文的 “0.013%”。

pRMS 10.37% 比 RMS 多 1.54 个百分点的错误，只省 1 s/epoch（相对 RMS 3.23%）；原文所谓 “further 2.6%” 是二者相对 LN 降幅 23.1% 与 20.5% 的差，约 2.6 个百分点，不是 pRMS 相对 RMS 的时间降幅。相同表还显示 Baseline 学习率 0.0003、其余 0.003，比较质量时必须承认这一混杂因素。


## Appendix A：配置细节决定实验能否复现

### A.1 Machine Translation

**原附录 A.1，PDF pp.13–14。** WMT14 4.5M aligned sentence pairs；newstest2013 选择模型，newstest2014／2017 测试；32k BPE merge；最大似然训练、label smoothing 0.1。Case-sensitive detokenized SacreBLEU，原签名：

```text
BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt14+tok.13a+version.1.2.12
BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt17+tok.13a+version.1.2.12
```

RNNSearch：最大 source/target length=80，embedding=512，hidden=1024，batch=80，Adam lr=1e−4。一般权重为标准正态缩放 0.01，方阵随机正交。Transformer：D=512，FFN=2048，8 heads，batch=25000 source/target tokens，Adam β₁=0.9、β₂=0.98，inverse-square-root schedule，warmup=4000，Xavier 初始化，300k steps。

{{fig:7}}

**Figure 7 — Theano RNNSearch 的 WeightNorm 比较。** 横轴 steps，纵轴 dev BLEU。WeightNorm 收敛慢于 LN／RMS／pRMS；附录给出 test BLEU 21.7／23.5，质量未超过本文主要归一化方案。PyTorch 官方 WeightNorm API 的尝试导致 OOM，作者未给完整 memory trace；这是一条具体实现失败记录，不能等价于 WeightNorm 必然比 RMS 使用更多激活内存。

### A.2 CNN/Daily Mail Reading Comprehension

使用公开 Theano 模型，CNN 的 top4 setting；只在 LSTM 内使用 RMS 来匹配 LN placement，hidden=240，Adam batch=64、lr=8e−5，Xavier 初始化，方阵进一步正交。未披露总参数、全训练 tokens、完整 test accuracy 与质量 seed 方差。不能把任务标题扩大成 CNN+Daily Mail 上全部配置的验证。

### A.3 Image-Caption Retrieval

VGGNet 10crop 图像特征；word embedding=300，GRU hidden=1024；不同参数使用 Xavier、Gaussian、orthogonal 初始化。Adam batch=128、lr=1e−3。训练 ranking loss，不改变 loss 以适配 RMS。最终 R@K 为 5 个 split 的平均，每 split 1000 images／5000 captions。

### A.4 CIFAR-10 Classification

沿 width、height 归一化，沿 channel affine；Adam batch=100。Baseline lr=0.0003，其余 lr=0.003；pRMS=12.5%。原文未完整披露 RMS 的累加精度、epsilon 放置、warmup、CUDA/cuDNN 版本和所有 task 的 batch timing 边界。因此复现时必须使用代码补足，同时保留代码与论文不一致之处。


## Experiment Audit：哪些 Claim 被支持？

| Claim / 审计项 | 论文证据 | 判断与缺口 |
|---|---|---|
| 不中心化仍能训练有效 | RNN、Transformer、CNN QA、COCO、CIFAR | 支持跨多任务可用；不能证明所有模型都不需要中心化 |
| 质量与 LN 相近 | BLEU、validation error、R@K 多列接近 | 整体支持；缺 quality seed variance，非正式非劣效检验 |
| 运行更快 | 3 次 timing、3 框架、3 GPU 场景 | 支持所测环境；没有同等融合 kernel 与现代精度的严格对照 |
| pRMS 可更快 | Table 4、8、10 | 非一致，Table 3 Theano 更慢；CIFAR 有明显 error 代价 |
| 梯度有尺度性质 | Eq.8–10 | 理想局部代数成立；epsilon、Adam、decay、残差需另分析 |
| 初始化更鲁棒 | center=0.2 的 Figure 4 | 有限扰动实证；缺多尺度、多均值、多 seed 网格 |
| 显存、通信、KV 节约 | 无专门测量 | 不能声称改变 KV cache 或 attention 可见范围 |
| 比较是否公平 | 同框架主要对照，附录承认细节差异 | BN cuDNN；CIFAR lr不同；跨框架与 existing work 不可直接归因 |

时间的 standard deviation 来自三次计时，**不能当成 BLEU 或检索质量的置信区间**。三次测量也不足以完整描述延迟尾部。正文没有模型参数总量、完整训练 token 预算和每任务精度配置；将这些标成未披露比用现代模型习惯填空更可靠。

如果继续复现，我会做四项有辨别力的实验：相同 seed／数据顺序／优化器下 LN 与 RMS 多 seed 质量；同 dtype、epsilon、kernel 优化等级下的前向／反向 microbenchmark；按 hidden 分布与通道排列检验 pRMS 的 prefix bias；按 time-to-target-quality 而非只用固定 steps 评价收益。它们分别区分训练机制、实现成本、估计假设和真实预算。
