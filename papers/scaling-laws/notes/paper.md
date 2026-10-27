## 1. Introduction：从架构选择转向资源规律

### 1.1 Summary：宏观规律与研究问题

**What / Why。**原文问：在语言建模中，参数、数据、训练算力以及深度、宽度、attention heads 哪些最能解释性能？单个大模型的结果无法回答“下一笔预算该加在哪里”，因此作者进行跨多个数量级的控制扫描。

**How / Evidence。**固定 Transformer 与主要训练配方，改变一个或两个尺度变量，寻找验证损失的包络线。核心结果是合理模型形状范围内，规模变量比形状更能解释平均 loss。这里的“合理范围”排除少于两层和极端深宽比等异常。

{{fig1}}

**Figure 1 — 三条单轴幂律。**先读横轴：左是计算、中央是唯一数据 token、右是非 embedding 参数；纵轴都是 test loss。观察每条训练曲线与最佳包络线的区别，然后看直线拟合。双对数坐标下直线表示乘法增加资源带来固定比例的 loss 改善。中央和右侧成立的前提是另两种资源足够；左侧 \(C_{\min}\) 已做 batch 校正，不能把图中所有点描述为实际小 batch 重训。原 caption 的脚注明确提醒看 Figure 13 比较纯实测计算曲线。

**关键观察。**规模扩大时没有看到上端系统性弯折，但这仅描述观测区间；不证明损失可无限降至零。拟合尺度 \(N_c,D_c,C_c\) 是归一化常数，不是必须达到的门槛。

{{fig2}}

**Figure 2 — 样本效率不等于计算效率。**左图横轴 tokens processed，同目标 loss 的水平线与大模型曲线相交得更早：大模型可以更省 token。右图横轴 PF-days，每条曲线还乘上模型计算成本；大模型并非总是更好，最佳模型大小随目标 loss 改变。读右图黑色前沿时，应把它理解为“在候选训练 run 中取最佳”，而不是沿一条固定模型曲线继续训练。

**我的理解。**当目标足够困难时，小模型的容量地板使继续训练收效甚微；先扩大模型再短训能避免在容量瓶颈上浪费预算。这个解释不包含部署成本。

{{fig3}}

**Figure 3 — 将 \(10^9\) 倍计算拆成三个乘法因子。**底部到顶部是参数量、batch、serial steps 对总体计算的乘法贡献；高度使用对数尺度，不能读成资源百分比。以后文实测指数计算，参数增加 \(10^{6.57}\approx3.72\) 百万倍，batch 增加 \(10^{2.16}\approx145\) 倍，步数增加 \(10^{0.27}\approx1.86\) 倍，三者乘积 \(10^9\)。图中文字“约百倍 batch、少于十倍 steps”是数量级示意。它暗示对并行能力的强需求，但不是 wall-clock 加速实验。

### 1.2 Summary of Scaling Laws：原文摘要公式逐个拆解

{{eq1.1}}

**(1.1)** 当数据和优化不构成瓶颈，容量项 \(L_N=(N_c/N)^{\alpha_N}\)。\(N,N_c\) 均为参数数目，比值无量纲，输出是标量 loss。增加参数十倍使 loss 乘 \(10^{-0.076}=0.8395\)，即下降约 16.05%。这是有限区间幂律，不含不可约熵底的显式常数。

{{eq1.2}}

**(1.2)** 对足够大的模型，在固定数据子集上 early stop，数据项 \(L_D=(D_c/D)^{\alpha_D}\)。这里增大的是唯一语料量；多看十遍同一份语料不等于增大 \(D\) 十倍。十倍唯一数据按此拟合使 loss 乘 \(0.8035\)。

{{eq1.3}}

**(1.3)** 对预算扫描所得最优包络线，\(L_C=(C_c^{\min}/C_{\min})^{0.050}\)。十倍归一化计算使 loss 乘 \(0.8913\)，下降约 10.87%。把 loss 减半需要 \(2^{20}\approx1.05\) 百万倍计算，反映强烈的边际收益递减；外推这么远本身已超出可靠使用范围。

{{eq1.4}}

**(1.4)** loss 降低时临界 batch 增大，指数 \(1/\alpha_B\approx4.762\)。如 loss 下降 13%，batch 比例为 \(0.87^{-1/0.21}\approx1.94\)，接近翻倍。\(B_*\) 单位 token；它是特定 tokenizer 和训练分布下的拟合尺度。

{{eq1.5}}

**(1.5)** 容量与有限数据联合式，把 \((N_c/N)^{\alpha_N/\alpha_D}\) 与 \(D_c/D\) 相加，再整体取 \(\alpha_D\) 次幂。让 \(D\to\infty\) 恢复容量项，让 \(N\to\infty\) 恢复数据项。两项相近时两种瓶颈同时存在；不是直接把两个单轴 loss 相加。第 4 节解释为什么采用这个不对称形式。

{{eq1.6}}

**(1.6)** 学习曲线联合式是容量地板加尚未优化完的残差 \((S_c/S_{\min})^{\alpha_S}\)。两项都为标量，模型最终地板由 \(N\) 决定，残差按步数减少。早期 transient、warmup 和极小模型不应强行用此式。

{{eq1.7}}

{{eq1.8}}

**(1.7)–(1.8)** 给出推导型预算指数：\(p_N=\alpha_C/\alpha_N,\ p_B=\alpha_C/\alpha_B,\ p_S=\alpha_C/\alpha_S\)，三者之和为 1。摘要中的 \(D=BS\) 采用无数据重复情形；更一般应写 \(E=BS\)。用明确输入 \((.076,.76,.21)\) 重新计算得到 \(\alpha_C=0.051987\)、\(p_N=0.684039,\ p_B=0.247557,\ p_S=0.068404\)。它们不是第 6 节直接测到的 .73/.24/.03。把“推导大致支持实测”写成“精确等于”会丢失原文的拟合差异。

### 1.3 Notation：三个不同的最优问题

原文分别研究 \(D\to\infty\) 时的容量极限、足够大模型在有限 \(D\) 上的 early-stopped 数据极限、固定计算下不同 \(N,B,S\) 的最优前沿。它们约束不同，不能把 \(L(N,D)\) 的近收敛结果直接解释成固定计算最优配置。

## 2. Background and Methods：重建训练与成本链路

### 2.1 Parameter and Compute Scaling of Transformers

{{pipeline-diagram}}

**方法输入。**GPT 风格 decoder-only Transformer、WebText2 token 序列、模型形状、语料子集、优化配方与计算预算。**方法输出。**一组可比较的 loss 曲线、拟合参数和预算前沿，而非新 attention 算子。训练过程产生 logits / loss；统计过程把每个 run 的 \((N,D,E,B,S,C,L)\) 组成标量观测记录。

{{table1}}

**Table 1 — 参数与 forward FLOPs 账本。**逐行读 embedding、attention 投影、attention mask/score、FFN、de-embedding。参数列和计算列是两套账：无参数的 attention score 依然需要计算；tied embedding 不意味着输出词表投影免费。表内沿用原文 \(d_{\rm embd}\) 等记号，未静默修正为新的变量名。此表不是与 baseline 比较的结果表，没有“最强方法”或提升百分比；FLOPs 越低只有在 loss 与任务要求一致时才有意义。

{{eq2.1}}

**(2.1) 参数计数。**每层 Q/K/V/O 共 \(4d^2\) 个权重；FFN 为 \(d\to4d\to d\)，共 \(8d^2\)。合计 \(12d^2\)，乘层数 \(\ell\) 得 \(N\approx12\ell d^2\)。一般式允许 attention 内维与 FFN 内维变化。忽略 bias、LayerNorm 等低阶项；词表与位置 embedding 不在 \(N\) 中。

**Tensor shape。**输入 hidden \([b,t,d]\)；Q/K/V reshape 成 \([b,h,t,d_h]\)，\(h d_h=d_{\rm attn}\)。attention score 为 \([b,h,t,t]\)，经过 causal mask 和 softmax，再乘 V 得 \([b,h,t,d_h]\)；合并 heads 并输出投影回 \([b,t,d]\)。FFN intermediate 为 \([b,t,4d]\)，最终 logits 为 \([b,t,V]\)。本论文改变这些 shape 的大小，基础数据流并未改变。

{{eq2.2}}

**(2.2) 计算近似。**密集权重矩阵乘法每个 token 约 \(2N\) forward FLOPs，另加原文记账中的 attention context 项 \(2\ell t d\)。这是作者采用的近似口径，不能当作所有 dense kernel 的精确指令计数。QKᵀ 和 AV、因果三角利用情况、输出词表投影、非线性、softmax、重计算等都影响完整 FLOPs。

**何时 \(6NE\) 合理？**取 backward 约为 forward 的两倍，且权重 GEMM 占主导，得到训练 \(C\approx6NE\)。context 项与 \(2N\) 的比率按原式为 \(t/(12d)\)；在 \(t=1024,d=1024\) 时约 8.33%，在 \(d=128\) 时约 66.7%，因此极小模型端尤其要审计省略项。词表输出近似 \(2dV\) forward FLOPs/token，相对主体 \(2N\) 比值 \(dV/N\)。这正是后续受控复现强调小模型 head FLOPs 的原因。

**Toy example / Interpretation。**12 层、\(d=1024\)：非 embedding 参数约 151M；若处理 10B token，主体训练近似 \(9.06\times10^{18}\) FLOPs，约 0.105 PF-day。这只是示例账本，不是原论文某次 run 的实测值。权重 FP16 约 288 MiB，不含 embedding。常见 mixed precision Adam 若合计按 16 bytes/parameter 估算，主体约 2.25 GiB，还没算 activation、临时 workspace 和分布式 buffers；具体实现的字节数必须实测。

{{cost-diagram}}

**推理链路 / 工程补充。**训练时 teacher forcing 并行计算所有位置；生成时先 prefill prompt，再逐 token decode。普通 multi-head attention 的 KV cache 为 \(2\ell btd\) scalars，乘存储 bytes；无 GQA/MQA 的原始 dense 模型同等参数量可能因深宽比产生不同 KV 大小。prefill 会计算 attention 矩阵，decode 持续读取 KV 和权重；TTFT、TPOT、HBM 带宽与通信都不由 \(C=6NE\) 推出。原文没有量化、MoE routing、KV 压缩或新 kernel 实验，不能据其“形状影响较小”推断这些系统成本一样。

### 2.2 Training Procedures

默认 **Adam、250,000 updates、512 sequences × 1024 tokens/update**。故 \(B=524,288\) tokens；完整 run 的 \(E=131,072,000,000\) tokens，约 131.1B。学习率 3000-step 线性 warmup 后 cosine decay 到零；warmup 就处理约 1.573B token。大于 1B 参数因内存约束改用 Adafactor。附录 D 扫描学习率与 schedule；“收敛结果不敏感”不意味着“短预算扫描不受 warmup 影响”。

### 2.3 Datasets

WebText2 在 GPT-2 WebText 的 Reddit 外链上扩展 2018-01 至 2018-10，保持至少 3 karma 的筛选，Newspaper3k 提取文本。约 20.3M documents、96 GB text、16.2B words、22.9B tokenizer tokens，其中 660M token 留作 test，训练侧约 22.24B。

**为什么 \(D\) 与 \(E\) 要分开？**默认完整 run 处理 131.1B tokens，而训练语料约 22.24B；两者比例约 5.89。不能写成“使用 131B 不重复训练数据”。跨分布评测包括 Books Corpus、Common Crawl、英文 Wikipedia 与公开 Internet Books，训练仍只用 WebText2。

## 3. Empirical Results and Basic Power Laws：证据支持到哪里

扫描范围：768 至 1.5B 非 embedding 参数，22M 至约 23B token；大多数 run context 1024、batch \(2^{19}\) tokens，另有短 context 和 batch 扫描。极小网络不是今天通常说的“小型 LLM”，不能把宽度和 vocabulary cost 可忽略当成统一前提。

### 3.1 Approximate Transformer Shape and Hyperparameter Independence

{{fig5}}

**Figure 5 — 合理形状区间内的弱依赖。**左面板变 FFN ratio，同时调 hidden width 保持 \(N\)；中间变 depth/width aspect ratio；右面板变每 head dimension，补偿参数数量的细微偏差。纵轴是相对于参数量拟合基线的 loss 增量百分比，不是 Accuracy drop。先看低谷附近宽区间，再看两端恶化。例子中 \((6,4288)\) 和 \((48,1600)\) 接近，前者 loss 在后者约 3% 内；这不等于两者速度、通信、KV、训练稳定性相同。图中还提示某些形状需约 22% 更多计算补偿 1% loss，这类小效应在工程中并不自动可忽略。

**解释边界。**干预不是只改一个 shape tensor 后完全不调其他结构；作者通过宽度补偿保持参数数目。结果支持“同配方中的平均 loss 首先由规模解释”，不支持“架构不重要”，Figure 7 直接给出架构差异。

### 3.2 Performance with Non-Embedding Parameter Count N

{{fig6}}

**Figure 6 — 为什么排除 embedding？**左右面板纵轴完全相同，横轴分别含 / 不含 embedding。小网络中固定词表矩阵占比大，把它并入 \(N\) 会把不同深度曲线横向错开；排除后两层及以上的合理形状更接近统一趋势。零层 / 一层和极端深宽比仍偏离。正确结论是拟合的解释变量改善，不能从此推出 embedding 参数“无贡献”或 head FLOPs“可不计”。

{{eq3.1}}

**(3.1)** 是 (1.1) 的正文重述。此时模型在完整 WebText2 上近收敛，除最大模型可能轻微过拟合外，大部分视为数据充足。\(D=\infty\) 是实验近似标签，真实数据并非无限。

#### 3.2.1 Comparing to LSTMs and Universal Transformers

{{fig7}}

**Figure 7 — Transformer 的优势来自上下文利用。**左面板在相同非 embedding 参数附近比较平均 loss；右面板按 token 在 context 中的位置分解。LSTM 在靠前 token 可接近 Transformer，但约百 token 后收益趋缓，Transformer 持续从较长前缀获益。比较用相同训练数据与 context，却不是“相同 FLOPs / 相同 latency”的完整硬件对照。原 Figure 7 caption 基本为空，本报告保留这个事实，并用图轴与正文解释，未生成伪原 caption。

Universal Transformer 复用层权重，参数效率与计算效率可能相反；原文把完整图放在附录 D，见 Figure 17。

#### 3.2.2 Generalization Among Data Distributions

{{fig8}}

**Figure 8 — 域内 loss 是域外文本 loss 的强预测量。**左图各评测语料随 \(N\) 平滑改善，曲线存在分布偏移；右图把 Books/Wikipedia loss 对域内 test loss 作图，已收敛模型的点与大模型训练途中的虚线大致同轨。它支持 tested textual domains 中的相关性；未证明因果，也未覆盖多语言、代码、视觉或推理任务。偏移是“近似常数 / 缓慢变化”，不应当成严格固定常数。

### 3.3 Performance with Dataset Size and Compute

{{eq3.2}}

**(3.2)** 用 36 层、width 1280 的模型，在 WebText2 子集上训练到 test loss 不再降低，拟合数据幂律。最小子集过早过拟合等异常在第 4 节讨论。该式是 early-stopped test performance，不是所有子集都使用相同最终 training loss。

{{eq3.3}}

**(3.3)** 固定 batch 的实测计算包络线指数 \(\alpha_C=.057\)，区别于 batch 校正后 .050。固定某个 \(C\)，每个模型对应 \(S=C/(6NB)\)，比较这些时刻的 loss 后取最优 \(N\)。原文此处写 \(S=C/(6BS)\)，右侧含自身且缺 \(N\)，是笔误；报告原式资产保留，解释使用量纲正确的表达。

**如何复现包络线？**需要每个 run 的 checkpoint / loss trajectory，而非只取各 run 训练终点；否则大的 run 与小的 run 比较时预算不一致。对 log-space 预算网格插值，保留候选模型的原始 loss 与 FLOPs，然后寻找每个预算的最低点。

## 4. Charting the Infinite Data Limit and Overfitting

### 4.1 Proposed L(N,D) Equation：为何是不对称联合式

{{fig4}}

**Figure 4 — 两个联合函数预览。**左图固定颜色表示模型大小，增加唯一数据后曲线逼近各自容量地板；右图不同 \(N\) 的学习曲线经 batch-adjusted step 表示后可由地板加残差拟合。左边的 \(D\) 与右边的 \(S_{\min}\) 不是同一横轴对象；虚线拟合不是额外训练数据。摘要先放这张预览，当前两节分别展开其模型。

{{eq4.1}}

**(4.1) 三个建模要求。**第一，tokenizer 改变时，归一化常数应能吸收 loss 尺度变化；第二，两种无限极限要恢复对应单轴幂律；第三，希望大 \(D\) 下有整数次幂 \(1/D\) 展开。第三项理论支持最弱，作者明确说它是 ansatz，不是由 Transformer 机制严格推导。

令 \(x=(N_c/N)^{\alpha_N/\alpha_D}, y=D_c/D\)，则 \(L=(x+y)^{\alpha_D}\)。固定 \(N\) 而 \(D\) 很大，有 \(L=x^{\alpha_D}[1+\alpha_D y/x+O((y/x)^2)]\)。这就是有限样本惩罚首先按 \(1/D\) 增长的数学结构。选择对称幂律相加可得到不同外推；拟合充分好也不代表函数形式唯一。

### 4.2 Results：数据惩罚、拟合参数与 overfitting

{{table2}}

**Table 2 — 联合拟合的四个参数。**\(\alpha_N=.076,\alpha_D=.103,N_c=6.4\times10^{13},D_c=1.8\times10^{13}\)。这些是联合扫描拟合，不能用摘要单轴的 \(.095,8.8\times10^{13},5.4\times10^{13}\) 替换。没有 baseline 或性能提升列；应审计参数在什么条件拟合、是否存在置信区间，而非给出虚假的相对“提升”。

训练使用 **10% dropout** 和 test-loss early stopping，未随 \(N,D\) 充分调优正则。最小约 20M token 子集一个 epoch 仅约 40 updates，和 3000-step 默认 warmup 相差极大；这些 run 对联合式偏离更明显，处在优化瞬态与数据重复交织的范围。

{{fig9}}

**Figure 9 — 瓶颈与数据坍缩。**左边固定数据子集颜色，在 \(N\) 小时增加参数有效，在 \(N\) 大时逼近数据地板。不是“test loss 因增加参数突然暴涨”，而是 marginal benefit 饱和。右边把不同 \(N,D\) 的相对惩罚画成一个组合变量，点接近同一曲线，支持联合式。横轴含归一化 \(N^{\alpha_N/\alpha_D}/D\) 的实验坐标；不是有单位的参数/token 比直接通用。

{{eq4.2}}

{{eq4.3}}

**(4.2)–(4.3)** 定义 \(\delta L=L(N,D)/L(N,\infty)-1\)，单位是比例，区别于绝对 loss gap。令 \(z=(N/N_c)^{\alpha_N/\alpha_D}D_c/D\)，则 \(\delta L=(1+z)^{\alpha_D}-1\)。\(z\ll1\) 时 \(\delta L\approx\alpha_D z\)，揭示归一化的数据需求如何随 \(N\) 增长。

{{eq4.4}}

**(4.4)** 保持过拟合 gap 不超过约 0.02 的经验条件 \(D\gtrsim5000N^{0.74}\)。这里 \(0.74\approx.076/.103=.7379\)，来自 **Table 2 联合拟合**；摘要 \(.076/.095=.8\) 不能产生 .74。模型扩大 8 倍所需唯一数据约 \(8^{.74}=4.66\) 倍，作者概括为约 5 倍。\(N=10^9\) 时约需 22.85B token，和 WebText2 可训练量同量级。

**Caveat。**这是一种 near-convergence 正则与噪声阈值定义，既非固定计算最优的 token/parameter ratio，也非当代 LLM 防过拟合通用法则。0.02 是第 4 节估计的 run variation；不要与附录 Figure 22 的小模型 0.05 混为同一个误差条。

## 5. Scaling Laws with Model Size and Training Time

### 5.1 Adjustment for Training at Bcrit(L)

**问题。**固定 batch 的 run 在不同 loss 水平可能离临界 batch 很远，导致同样的计算统计掺杂不同噪声效率。作者借助 gradient noise scale 工作，对同目标 loss 的 steps–examples Pareto 曲线进行归一化。

{{fig10}}

**Figure 10 — batch 的控制量更接近 loss。**横轴 WebText2 train loss 从高向低读，纵轴 critical batch token；3M 和 85M 模型估计在相似 loss 处靠近。绿色梯度噪声测量与实测转折点同趋势，但散点明显，不能称为完全精确预测或“与参数量无任何关系”。\(B_{\rm crit}\) 是统计效率转折，不是系统吞吐峰值。

{{eq5.1}}

{{eq5.2}}

**(5.1)–(5.2) 重新推导。**令 \(s=S/S_{\min}\)、\(e=E/E_{\min}\)，Pareto 式为 \((s-1)(e-1)=1\)，而 \(E=BS\)、\(B_{\rm crit}=E_{\min}/S_{\min}\)，所以 \(e=(B/B_{\rm crit})s\)。取 \(r=B/B_{\rm crit}>0\)，解得 **\(s=1+1/r,\ e=1+r\)**。

因此小 batch 极限用最少 token，但需更多串行更新；大 batch 极限用最少 steps，却多消耗 token。当 \(r=1\)，实际训练同时需要 \(2S_{\min}\) 与 \(2E_{\min}\)，实际 \(C=2C_{\min}\)。不存在有限 batch 同时达到两个下限。

{{eq5.3}}

{{eq5.4}}

{{eq5.5}}

**(5.3)–(5.5) 如何校正一次观测？**从 run 的 loss 查 \(B_{\rm crit}(L)\)，用 \(S_{\min}=S/(1+B_{\rm crit}/B)\) 和 \(C_{\min}=C/(1+B/B_{\rm crit})\) 得到可比较坐标。若 \(B=4B_{\rm crit}\)，\(S_{\min}=0.8S\)，\(C_{\min}=0.2C\)；实际已花算力没变，改变的是用于拟合的归一化指标。该校正依赖经验 Pareto 模型，不能宣称作者逐点完成小 batch 重训。

{{batch-widget}}

**如何读计算器。**调整相对于临界值的 batch，图上标出 steps 与 processed-token 的互换。读数来自原式 (5.1)，不是吞吐 benchmark。默认 \(r=1\) 的两倍关系可以用来核查后续公式中常见的 factor-of-two。

### 5.2 Results for L(N,Smin) and Performance with Model Size and Compute

{{fig11}}

**Figure 11 — 固定计算存在最优参数，固定步数则主要看容量。**左边每个计算预算有一条曲线，随着 \(N\) 先降后升：小模型受容量限制，大模型受训练进度限制；底部虚线 / 包络取各预算的最优点。右边在固定校正 steps 下，不同 \(N\) 逐渐受训练残差控制。最初很小 \(S\) 的拟合较差，这恰是 warmup/transient 范围，而非简单归咎于随机噪声。

{{eq5.6}}

{{table3}}

**(5.6) / Table 3 — 地板与优化残差。**正文此处拟合 \(\alpha_N=.077,\alpha_S=.76,N_c=6.5\times10^{13},S_c=2100\)。与摘要 .076 不完全一致；输入这些值推导指数也会略变。若剩余优化项是容量项的两倍，增加 steps 的初始收益比继续扩参数更有潜力；若容量项占绝大多数，继续训练趋于浪费。

优化残差把 \(S_{\min}\) 十倍增加后乘 \(10^{-.76}=.1738\)，但总 loss 还含固定容量项，不会整体降低 82.62%。分项与总量要分别计算。所有变量为宏观标量，不是一个训练期间逐 tensor 执行的 loss 修正算子。

### 5.3 Lower Bound on Early Stopping Step

{{eq5.7}}

**(5.7) early stop 下界。**无限数据 learning curve 仍高于有限数据最优 test loss 时，不应声称已达到该数据地板。令 \((S_c/S_{\min})^{\alpha_S}\le L(N,D)-L(N,\infty)\)，解得该式。若允许 gap 越小，所需 steps 越大，且以 \(1/\alpha_S\) 次幂放大。它是下界，不给出真正 early stop 的精确位置，且需要同一 batch 口径。

**实现提示。**应同时保存 train loss、固定 held-out loss、平滑 / seed 统计和数据重复状态；单看 train–test gap 不能测出此式中的数据惩罚，附录 Figure 16 展示差别。

## 6. Optimal Allocation of the Compute Budget

### 6.1 Optimal Performance and Allocations：历史实测结果

{{fig12}}

**Figure 12 — 最优点附近存在宽容区。**左图横轴 \(N/N_{\rm eff}\)，纵轴为达同 loss 的 excess compute ratio；最低点在 1，约 .6 到 2.2 倍模型仍可在约 20% 额外计算内完成。右图是同目标 loss 的 steps ratio，大模型减少更新数。两图不是同时固定相同计算预算后比较 loss；这是固定目标质量的替代配置曲线。虚线远端提醒解析模型不能捕捉很大模型最早期训练。

{{fig13}}

**Figure 13 — 校正前后计算幂律。**蓝色 .050 使用 \(C_{\min}\)，橙色 .057 使用 fixed-batch \(C\)。一层到两层 transition 造成小规模凸起，被幂律拟合排除。图例 normalizer 为 \(2.3\times10^8\) 与 \(2.0\times10^7\) PF-days，而附录 Table 5 写 \(3.1\times10^8\) 与 \(1.6\times10^7\)；本报告把图与表原样保留，不强行统一常数。指数也不能互换。

{{eq6.1}}

{{eq6.2}}

{{fig14}}

**Figure 14 — .73/.03 是前沿扫描的经验拟合。**左边校正后最优模型随 \(C_{\min}^{.73}\) 增加，固定 batch 的未校正趋势更陡；右边校正步数近乎水平，而未校正步数增长明显。算力十倍，\(N\) 实测比例 \(10^{.73}=5.37\)，processed-token 余下约 \(10^{.27}=1.86\)。batch 趋势约 .24，steps 约 .03，作者指出 steps 指数可能与零一致，未给严谨的置信区间足以断言一定增长。

**Factor-of-two 审计。**本文统一定义 \(C_{\min}=6NE_{\min}=6NB_{\rm crit}S_{\min}\)，而在临界 batch 的实际 \(C=6NB_{\rm crit}(2S_{\min})=2C_{\min}\)。原第 6.1 节“\(C_{\min}\equiv6NB_{\rm crit}S\)”若将 \(S\) 当实际 steps 会混淆两者；附录 B 又把 adjusted/critical 标签省略。指数在常数归一化下不变，绝对 FLOPs、tokens 和尺度常数则会改变。

{{allocation-widget}}

**如何读预算比较器。**三种策略均从同一个基准出发，仅比较增加预算后的倍率：Kaplan 经验 .73/.27、按明确输入推导的理论指数、Chinchilla 等量近似 .5/.5。前两者显示解析 loss 比例，Chinchilla 模式不编造同一 loss 曲线或 batch/steps 建议。它们的 baseline 数据、优化和参数口径不同，倍率比较用于理解资源倾向，不能替代重新拟合或证明质量等价。

### 6.2 Predictions from L(N,Smin)：把“近似一致”拆开

{{eq6.3}}

{{eq6.4}}

{{eq6.5}}

**(6.3)–(6.5) 为什么产生倒数和？**固定预算时，增大 \(N\) 改善容量项，但降低训练步数；loss 下降又允许临界 batch 变大。在最优前沿，容量与优化残差的边际收益平衡，各尺度乘法组成 \(C\)。因而 \(p_N+p_B+p_S=1\)，代入 \(p_i=\alpha_C/\alpha_i\) 得倒数和式。附录 B 给出完整隐式求导。

原文 (6.4) 近似 .054、(6.5) 近似 .71；附录 B.7 近似 .052。固定输入 \(.076,.76,.21\) 的准确计算是 .051987 和 .684039。用 Table 3 的 \(.077,.76,.21\) 得约 .052453/.681207。它们从量级上支持主要增大 \(N\)，并不精确恢复 .73/.03。没有误差传播与 fit interval，不应将这几个近似数包装为严格一致。

### 6.3 Contradictions and a Conjecture：外推何时失效

{{fig15}}

**Figure 15 — 两条外推曲线互相矛盾。**同一计算预算横轴上，黄色最优计算损失按约 \(C^{-.05}\) 降，红色“把不复用数据的 token 用量带入数据幂律”降得更慢。越过交点后，一条式子声称能达到比另一条数据下限更好的 loss。应读为现有假设至少有一项失效；交点并非实测结果，也不是架构性能上限的证明。阴影 / 多线体现对指数敏感。

{{eq6.6}}

{{eq6.7}}

**(6.6)–(6.7)** 控制 near-convergence 数据惩罚需要 \(D\propto N^{.74}\propto C_{\min}^{.54}\)。但在无复用、临界 batch 下，\(D=E=2C_{\min}/(6N)\)，只按约 .26/.27 增长。因此长期外推会进入新的数据限制区间。式 (6.7) 约 \(4\times10^{10}\) tokens × \(C_{\min}^{.26}\)，而 Table 6 约 \(2\times10^{10}\) × \(C_{\min}^{.27}\)；绝对数 factor-of-two 与临界/最小计算口径有关，舍入指数也略不同。不能把这两组数据混在同一个绝对预算推荐中。

{{eq6.8}}

**(6.8) 仅为 conjecture。**\(C^*\sim10^4\) PF-days、\(N^*\sim10^{12}\)、\(D^*\sim10^{12}\)、\(L^*\sim1.7\) nats/token，位于观测范围之外多个数量级，原文说可随指数变化上下漂移约一个数量级。可靠结论是幂律必须在某处修正；“这里估计了自然语言熵”只是更强解释。改变 tokenizer、数据分布和上下文就会改变 token 熵，1.7 不能当通用能力上限。

**后见之明的正确用法。**今日数据/计算配方已经改变，不宜只用后来模型大于某个数字判定原始有限区间实验无效；应问其外推假设哪一条被新的控制实验修正。

## 7. Related Work：原论文的历史位置

原文连接三类研究：数据与参数规模的经验幂律（Banko & Brill、Hestness 等），过参数化与泛化 / NTK 等理论，梯度噪声与优化动态。最接近的 Hestness 2017 已研究数据与模型增长规律；本论文强化了统一 next-token loss、算力前沿与临界 batch 的联合研究。同期 Rosenfeld 等尝试构造 loss–size 联合预测，而不是所有幂律概念都首次来自 Kaplan。

原文也讨论 EfficientNet 的 compound scaling，但视觉任务中的 depth/width 法则不能直接迁移到语言模型。其 noisy quadratic / Hessian spectrum 解释属于可能机制，并未在本文给出从底层训练动力学推出这些指数的定理。

## 8. Discussion：宏观预测框架的贡献与边界

**作者主张。**规模律类似把复杂微观细节压缩为少数宏观变量的“热力学”描述；可以从短学习曲线估计长期 loss，用联合式预测有限数据惩罚、early stop 和预算配置。更大模型可能需要模型并行，宽网络与深网络对并行方式有不同要求。

**证据边界。**作者明确把语言建模 loss 是否对应具体任务能力列为开放问题。平滑平均 loss 可能与局部能力变化同时存在，但本文没有测试推理、对齐、安全、agent 或视觉能力。对图像、音频、视频 generative modeling 的迁移也是建议，并非已完成实验。

**读法。**保留“建立预测框架”这一贡献，同时把“big models may be more important than big data”理解为当时预算前沿与实验设定的结果。当前策略选择必须包含数据质量、优化配方、推理需求和硬件约束。

## Appendix A. Summary of Power Laws：原始六表的最后三表

{{table4}}

**Table 4 — 六种约束情形。**从上到下分别是容量极限、数据极限、fixed-batch 计算、校正最优计算、有限 \(N,D\) 联合式、有限 \(N,S\) learning curve。先看前三列 constraints，再读右侧 equation，避免拿同名 \(L\) 函数互换条件。这里没有 baseline superiority；“Optimal”需要说明在哪些变量和候选 run 上取最小。

{{table5}}

**Table 5 — 摘要拟合参数。**单轴 \(\alpha_D=.095\)，joint Table 2 为 .103；单轴 \(N_c=8.8\times10^{13}\)，学习曲线 Table 3 为 \(6.5\times10^{13}\)。表内 \(B_*=2.1\times10^8\)，正文常舍入到 \(2.0\times10^8\)。所有 scale 都有单位，且 tokenizer-dependent，不应截掉表头括号。高次幂与倒数会放大 rounding，因此报告给理论计算器明确固定输入。

{{table6}}

**Table 6 — 经验计算前沿的绝对尺度。**\(p_N=.73,N_e=1.3\times10^9\)；\(p_B=.24,B_e=2.0\times10^6\)；\(p_S=.03,S_e=5.4\times10^3\)；\(p_D=.27,D_e=2\times10^{10}\)。\(C_{\min}\) 在这个表中以 PF-days 数值代入，否则绝对量会错 \(8.64\times10^{19}\) 的单位因子。batch 行写 \(B\ll B_{\rm crit}\)，step 为下界；若改为实际在 critical batch 跑，不能仍把这些最小量当实际全部使用。

**公平性判断。**这些表是拟合摘要，不是同算力 benchmark 排名。没有硬件、精度、latency、confidence interval 或 task metric 列，不能为“当前最强模型”或“训练加速多少”提供证据。

## Appendix B. Empirical Model of Compute-Efficient Frontier

### B.1 Defining Equations：隐式方程与最优条件

{{eqB.1}}

{{eqB.2}}

{{eqB.3}}

**(B.1)–(B.3)** 重复 learning curve 和 critical batch，代入 \(S=C/(6NB(L))\)。令 \(A=(N_c/N)^{\alpha_N}\)，\(T=[6B_*S_cN/(L^{1/\alpha_B}C)]^{\alpha_S}\)，得到 \(L=A+T\)。右边包含 \(L\)，所以是**隐式方程**。求值不能把 unknown \(L\) 随便放为常数；应在指定有效区间做数值 root solving 或利用最优条件消元。

{{eqB.4}}

**(B.4) 固定 \(C\) 求导。**完整微分为 \(dL/dN=-\alpha_N A/N+\alpha_S T/N-(\alpha_S/\alpha_B)(T/L)dL/dN\)。最优点 \(dL/dN=0\)，故 \(\alpha_N A=\alpha_S T\)。原式括号用了 “5” 近似 \(1/.21=4.762\)，最终在驻点含导数项为零，不改变平衡式。原取消线 MathML 一并保留。

**直觉。**容量项对 \(N\) 的收益与预算约束引起的优化残差损失平衡。它是可微拟合面上的最优条件，不证明实际 optimizer 必然能到达，也不包含离散 shape/设备容量约束。

### B.2 Efficient Training：为何远未收敛

{{eqB.5}}

**(B.5)** \(T/A=\alpha_N/\alpha_S\approx.1\)，所以最优点总 loss 比同尺寸近收敛地板高约 10%。这是 relative loss gap；“只训练到 90% 能力”不是等价表述。它描述跨尺寸的最优 allocation，不能直接给现有业务模型制定通用 stop threshold。

{{eqB.6}}

{{eqB.7}}

{{eqB.8}}

**(B.6)–(B.8) loss 与计算归一化。**消元得到 \(\alpha_C\) 的倒数和，\(C_c\) 由 \(N_c,B_*,S_c\) 与幂次因子组成。若 \(C\) 以 FLOPs 记，则 \(6N_cB_*S_c\) 同样是 FLOPs；改为 PF-day 必须换算。原始 \(C_c\) 式常数不能同 Table 5 经验归一化直接混用。

{{eqB.9}}

{{eqB.10}}

**(B.9) 最优模型。**用 \(A=L/(1+\alpha_N/\alpha_S)\) 求 \(N\)，得到 \(C^{\alpha_C/\alpha_N}\)。比例系数包含幂次放大的 loss factor，不能忽略。

**(B.10) 额外公式核查。**从 \(S=C L^{1/\alpha_B}/(6NB_*)\)，令 \(q=1+\alpha_N/\alpha_S\)，代入 B.6 与 B.9，应得：

<div class="derived-formula"><code>S(C) = [Cc/(6NcB*)] · q^(-1/αN) · (C/Cc)^(αC/αS) · L(Cc)^(1/αB)</code><p>Agent-derived consistency check：按 B.6，L(Cc)=1，因此该末尾因子为 1。直接在未化简表达式中必须保留 L(C)^(1/αB)，再把它的 C 指数并入 αC/αS。</p></div>

这里**原 B.10 并不缺失最终的 loss 因子**：以 \(L(C)=(C_c/C)^{\alpha_C}\) 消元后，它已进入净指数。将未化简式的 \(L(C)^{1/\alpha_B}\) 再乘回 B.10 会重复计算，造成错误。这个核查强调先完成代数消元，再判断公式是否漏项。

### B.3 Comparison to Inefficient：相同目标 loss 的训练配方

{{eqB.11}}

{{eqB.12}}

{{eqB.13}}

{{eqB.14}}

**(B.11)–(B.14)** 定义 \(f\) 为距离近收敛地板的相对 gap，对同目标 \(L\) 比较 \(f\approx10\%\) 与 \(f'=2\%\)。原文近似得模型 2.7 倍、updates .13 倍、计算 .35 倍，故约 65% 少计算。它来自同目标 loss 下的解析模型，不是 wall-clock 65% 加速。用舍入 \(.076,.76,f=.1,f'=.02\) 得到的数会略偏离 2.7/.13/.35，不应把原近似当精确 benchmark。

同目标 \(L\) 时 \(B_{\rm crit}(L)\) 相同，因此这一局部比较可将计算比例写为 \(N\) 与 \(S\) 比例乘积。对不同目标 loss 或换 tokenizer，则还要把 batch factor 加回。

### B.4 Suboptimal Model Sizes：把推理成本带回预算讨论

{{eqB.15}}

**(B.15)** 固定目标 \(L\) 解所需计算，前提 \(L>(N_c/N)^{\alpha_N}\)。若小模型容量地板本就高于目标，训练再久也无法在这个模型里解出有限计算。括号差为非正数时，计算器明确标为解析不可达，不输出 NaN 或假数值。

{{eqB.16}}

{{eqB.17}}

**(B.16)–(B.17)** 令 \(r=N/N_{\rm eff}\)，\(g(r)=1+(\alpha_S/\alpha_N)(1-r^{-\alpha_N})\)，则 steps ratio \(g(r)^{-1/\alpha_S}\)，compute ratio \(r g(r)^{-1/\alpha_S}\)。\(r=1\) 时都等于 1；\(r\) 太小时容量不可达；\(r>1\) 时可少训但每步更贵。

{{suboptimal-widget}}

**原例重算。**输入 .076/.76，\(r=2.2\) 得约 **1.2035 倍计算、.5470 倍 steps**，与原文“约 20% 多计算、45% 少 steps”一致到近似精度。减少 steps 只有在每步耗时、并行性、通信和显存允许时才可能降低总训练时间。小于最优的模型则可牺牲少量训练算力以降低长期服务成本。

**原文交叉引用核查。**B.4 文本仍写 “A.1/A.6/A.9” 和 “Figure X/Y”；按上下文对应本附录 B.1/B.6/B.9 与 Figure 12 的两个面板。报告说明其残留占位，不制造额外图号。

## Appendix C. Caveats：作者明确承认的限制

原文提出以下未解决问题：缺少指数的理论来源；模型类别、任务、数据变化后是否通用；若无法描述幂律修正项，难以判断外推可信范围；优化未彻底调优；不同 regularization 或 tokenization 可能改变拟合；early-stage learning 不属于简单后期幂律区间。

**如何影响工程决策。**拟合误差小只说明在扫描网格中描述好，不意味着网格之外误差小。训练 run 的优化器从 Adam 切换为 Adafactor，也意味着规模轴并非完全同配方。本文没有硬件/精度/能耗细账，不能据其 FLOPs 标量比较 accelerator 实际耗时。

## Appendix D. Supplemental Figures：逐图读出补充证据

### D.1 Early Stopping and Test vs Train

{{fig16}}

**Figure 16 — 下界与真正 overfitting gap。**左图横轴是 (5.7) 下界表达式，纵轴实测 stop step；多数点在红线以上，支持下界而非精确预测。右图 300M 模型在不同数据子集上的 train/test curves，test loss 最初沿 full-data 轨道，随后偏离；黑色竖条的 train–test gap 明显大于“有限数据 test loss 相比无限数据 test loss 的损失”。不能用 train–test gap 直接代入 \(\delta L\)，也不能忽略重复 epoch 状态。

### D.2 Universal Transformers

{{fig17}}

**Figure 17 — 共享参数没有共享计算。**左图横轴把 reuse 也计入“有效参数使用量”，右图横轴 unique non-embedding parameters。与标准网络比较，反复使用同一权重的 recurrent Transformer 在固定 unique \(N\) 下可能略好，但按额外使用次数计 FLOPs 后略差。阅读顺序先确认各横轴 convention，再比较线的高低。它为“参数效率和计算效率分账”提供直接证据；没有推理 latency benchmark。

### D.3 Batch Size

{{fig18}}

**Figure 18 — 临界 batch 的实测来源。**左右分别 3M 与 85M 模型；横轴 step、纵轴 tokens processed，同一 loss 的不同 batch 终点形成一条 Pareto curve，颜色表示 loss 目标。沿曲线向左减少 steps，纵轴需要更多 token；这些拟合用于 Figure 10 的 \(B_{\rm crit}(L)\)。不是所有散点共享同一模型或同一目标。每个目标的经验渐近下限带拟合误差，后续 \(C_{\min}\) 的误差也继承它。

### D.4 Sample Efficiency vs Model Size

{{fig19}}

**Figure 19 — 同质量水平的大模型需要更少 steps/token。**左边最小串行 steps，右边在远低于 critical batch 时最少 examples；颜色分别是固定 loss。在容量刚刚够用处，小模型所需训练量急剧增大，参数足够时曲线逐渐平坦。原文“接近 100 倍样本效率”来自特定同目标、小模型到大模型范围，不是任意两种相差百倍参数的通用比例。

### D.5 Context Dependence

{{fig20}}

**Figure 20 — token 位置的 loss 不是平均 loss。**左图固定 context 位置 \(T\)，更多前文提供条件信息，loss 随 \(T\) 幂律下降；更大模型的下降更陡。右图固定 774M 模型，不同位置的 loss 随训练 steps 下降，早期位置 loss 地板更高。这里 \(T\) 是 context position，区别于外部 2026 Practical Scaling Laws 中的 processed training tokens \(T\)。

{{fig21}}

**Figure 21 — context 容量分配。**实线为 1024-context 下各位置，虚线为短 context 8 的实验；短模型在头几个 token 上可更好，因为训练目标更集中在这些位置。不能推断短 context 的全文平均建模一定优于长 context，也不能把首 token loss 当 prompt prefill TTFT。原文只测到这些 context 设置，不提供今日百万 token 长上下文结论。

### D.6 Learning Rate Schedules and Error Analysis

{{fig22}}

**Figure 22 — schedule 扫描的适用范围。**左图多条 schedule；右图在约 3M 参数模型中看 loss 与累计学习率的关系。只要学习率不太小、不太快衰减，多种配方最终 loss 接近；作者为这项扫描特意不 decay 到 zero，避免尾部固定 improvement 干扰。它与默认 run 的 cosine-to-zero 配方不同。这里 seed variation 约 **0.05 loss**，要验证更小变化需要多 seed；不能用这一个小模型扫描证明任何大模型、短训练 budget、warmup 都不重要。

{{eqD.1}}

**(D.1) 经验 learning-rate prescription。**原式写 “\(0.003239+-0.0001395\log(N)\)”；意图是负系数 log-linear fit，可写 \(0.003239-0.0001395\log N\)。log 默认数学自然对数，该单位拟合只对原扫描范围有意义；大范围外推可能给负 LR，不能直接做通用配置生成器。日志中同时记录实际 peak LR、warmup token 和 decay horizon，比仅存最终 LR 更有复现价值。

### D.7 Fit Details and Power Law Quality

{{fig23}}

**Figure 23 — 幂律比单一 log-linear 拟合更好。**同一组 loss–\(N\) 点上，蓝色 power law 更贴合，橙色 logarithmic alternative 在两端系统偏离。它提供定性函数形式比较，没有给独立外推集的统一误差表、bootstrap uncertainty 或所有候选函数的选择标准。因此不能从“优于这一条 log fit”推断幂律形式唯一。

### D.8 Generalization and Architecture

{{fig24}}

**Figure 24 — 约 1.5B 参数下的深度与域外 loss。**固定总参数，横轴 layers，比较多个语料 loss；整体差异较小，Books 与 Wikipedia 的局部起伏仍存在。12-layer 模型在 Internet Books 意外过拟合，用 early-stopped performance，作者说未在其他实验见过。它是异常个案说明，不能从均值曲线平滑把异常删掉。曲线不是严格水平，更没有误差条足以宣称所有深度等价。

