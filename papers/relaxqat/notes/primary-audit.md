# RelaxQAT 原始论文完整证据与准确性审校笔记

## 0. 来源、范围与审校约定

- 论文原题：**A Continuous Relaxation of Quantization-Aware Training**。
- 作者：Kartikeya Bhardwaj、Mohammed Almousa、Chaitanya Mullapudi、Ananth Balasubramanian、Nithin Ramesan、Tuc Nguyen、Harris Teague；Qualcomm AI Research。
- 版本：arXiv:2611.27056v1，首页标注 20 Nov 2026；PDF 元数据 DOI 为 `10.48550/arXiv.2611.27056`，许可元数据为 CC BY 4.0。
- 唯一研究对象：[本地原 PDF](file:///Users/bytedance/project/read_papers/paper/2611.27056v1.pdf)，17 页。
- SHA-256：`2b0e2e33b54c69a6d25cce14f848b5a94da3008572b0542d8d04ce2e368e9188`。
- 已完整阅读 `page_0001.txt` 至 `page_0017.txt`，并用 PDF 页面渲染核对公式上下标、主要表格、全部图像及图注。页引用均为 **PDF 从 1 开始的物理页码**，不使用文件名里的零基索引。
- 离线研究，没有检索作者代码、其他版本、外部基线论文或模型配置。因此，论文外的实现一致性、模型结构与可复现性均不宣称已确认。
- 全局 PyMuPDF 存在但因 `libmupdf.dylib` 缺失无法导入；改用已有 `pypdfium2` 成功渲染 17 页到 `/tmp/relaxqat-audit-renders/`。没有修改 `/tmp/paper-261127056-env`，没有安装依赖，没有修改原 PDF 或共享站点。
- 本文件仅为研究/审校笔记，不是最终网页或飞书报告。
- 标记：**[原述]** 为论文明确陈述；**[推导]** 为从所印公式/表格得出的代数或算术结果；**[未披露]** 为完整 PDF 中没有给出的信息；**[疑点]** 为内部不一致或不能由证据推出的强结论。全部经验结果属于单一论文来源，不代表独立复现实验。

### 0.1 逐页覆盖与原始章节树

| PDF 页 | 内容与核查对象 |
|---|---|
| 1 | 标题、作者、Abstract；1 Introduction 前半；6.3%/7.7% headline |
| 2 | 1 Introduction 后半及四项贡献；2 Related Work 前半 |
| 3 | 2 Related Work 后半；3 Proposed RelaxQAT；3.1 Bitwidth Schedule；3.2 RelaxSEQ 前半；Eq. (1)、Figure 1 |
| 4 | 3.2 后半；3.3 RelaxQAT 前半；Eq. (2)、(3)、Figure 2 |
| 5 | 3.3 后半；3.4 Systems Considerations；4 Theoretical Insights 前半；Algorithms 1、2，Eq. (4)，Table 1 |
| 6 | Algorithm 3；4 中 Proposition 1、2；Figure 3，Eq. (5) |
| 7 | Proposition 2 证明；5 Experiments；5.1 Experimental Setup；5.2 Language Modeling Results 前半；Eq. (6)、(7) |
| 8 | 5.2 后半；5.3 Vision-Language Modeling Results 前半；Tables 2、3，Figure 4 |
| 9 | 5.3 后半；5.4 Ablations 前半；Tables 4、5、6、7 |
| 10 | 5.4 后半；5.5 Does RelaxQAT work for Ternary Quantization?；6 Conclusion and Future Work；References 开始；Tables 8、9 |
| 11 | References，含 LSQ、BitNet、UPQ、WinQ、LLM-QAT 等引用 |
| 12 | References，含 ParetoQ、Triton、Qwen3 technical report 等引用 |
| 13 | References 结束；7 AI Usage Disclosure |
| 14 | Supplementary Information；A Existing SEQ vs. RelaxSEQ；B Proof of Proposition 1 前半；Figure 5、Table 10 |
| 15 | B 后半；C SEQ Simulation；D More Details on Experiments；E Ablation Results 的引导；Figure 6、Eq. (8) |
| 16 | Figure 7；F Ternary Quantization Results 的方法；Eq. (9) |
| 17 | E 的 Table 11，F 的 Tables 12、13 |

审校进度：逐页文本阅读、公式/理论核对、全部实验转录、图表目录、两轮审校均完成。82 行主实验/消融/附录分数已程序化逐行与原页文本比对，未发现转录差异；Table 1 / Table 10 的两组短表另经人工核对。

## 1. Introduction：问题、贡献与 headline 的正确边界（pp.1–2）

**[原述]** 目标是极低比特、尤其 2-bit 的 channelwise 权重量化。固定低比特 QAT 可能在 integer-code 不变的区域难以推进；UPQ 等离散阶段转移可能造成损失突变。RelaxQAT 提议预设浮点 bitwidth 日程，并修改 SEQ 的操作顺序，形成更平缓的极值变化。

**[原述]** 五个尺寸/型号：Qwen3-0.6B、Qwen3-4B、Qwen3.5-2B、Qwen3.5-4B、Gemma3-1B，覆盖 0.6B–4B。贡献段称“两种 VLM、三种 LLM”；实验部分又在语言域测试全部五个型号，其中 Qwen3.5 两个还测 VLM。这不是七个独立型号，不能重复计数。

**[原述]** 五个 QAT baseline：ParetoQ、UPQ、LLM-QAT、BitNet、WinQ；另有 PTQ AdaScale。每个模型训练 10B tokens；不是所有表都包含全部五个 QAT baseline，ternary 表缺少 LLM-QAT。

**[推导] headline 必须写 pp（百分点），且以 ParetoQ 为比较对象：**

| headline | 原始证据 | 正确差值 | 对当表最强其他 QAT baseline |
|---|---|---|---|
| vision “up to 6.3%” | Qwen3.5-2B，Table 7 p.9：63.52 vs ParetoQ 57.26 | **+6.26 pp**，一位小数约 +6.3 pp；相对百分比约 +10.93% | UPQ 59.77；**+3.75 pp** |
| language “up to 7.7%” | Qwen3-0.6B，Table 3 p.8：47.04 vs ParetoQ 39.34 | **+7.70 pp**；相对百分比约 +19.57% | BitNet 45.04；**+2.00 pp** |
| ternary “up to 4.2%” | Qwen3-4B，Table 12 p.17：65.97 vs ParetoQ 61.77 | **+4.20 pp** | UPQ 64.38；**+1.59 pp** |

不能写成“相对 SOTA 提升 7.7%/6.3%”，也不是多个模型整体平均提升；是特定模型的已定义 aggregate score 的最大改善。Table 6 p.9 上 Qwen3.5-4B VLM 的 RelaxQAT 平均分还低于 UPQ 0.16 pp，故“全部任务击败全部基线”不成立。

## 2. Related Work：与本文方法的区别（pp.2–3）

- **LSQ**：把 quantization step size 作为可学习参数。本文用其公式分析，但没有 LSQ-only 主实验行，也没有展示 RelaxLSQ 的真实 LLM/VLM 效果。
- **ParetoQ / SEQ**：本文强调 2-bit 对称 odd-integer grid 比 LSQ 的 signed asymmetric grid 更好；RelaxSEQ 是在 SEQ 上改操作顺序。
- **BitNet**：相关工作中描述 latent weight 与 quantized weight 的插值日程；RelaxQAT 调的是 bitwidth，而不是相同的插值系数。本文 2-bit BitNet baseline 是作者改造版本，见 5.1。
- **WinQ**：周期性将 latent / quantized weights 插值，以应对 saddle/flat 区域；不等同于 RelaxQAT 的 bit 日程。
- **UPQ**：原工作的概述是 blockwise PTQ 到 INT4，再用该 checkpoint 做 INT2 QAT。Figure 1 为策略示意，不能从该图认定本实验 UPQ 的实际 PTQ/QAT 资源预算。
- 文献优先性、“first”、channelwise 更高 hardware efficiency 等，是作者主张；本次离线审校没有独立验证这些比较或检索充分性。

## 3. Proposed RelaxQAT：完整方法重建（pp.3–6）

### 3.1 Bitwidth Schedule：公式 (1) 与时间索引

**Eq. (1)，p.3，原式：**

\[
b_t=b_s-(b_s-b_e)\sin\left(\frac{\pi}{2}\omega\right),\qquad \omega\in[0,1].
\]

- \(b_s\)：起始 bitwidth；\(b_e\)：目标结束 bitwidth。
- \(t\)：全局训练步；\(\omega\)：由 callback 状态/查表索引决定的日程变量，**不是训练 loss 学出的参数**。
- \(\omega=0\Rightarrow b_t=b_s\)；\(\omega=1\Rightarrow b_t=b_e\)。
- 初始 half cycle：\(b_s\to b_e\)。其后可做 \(n\) 个 full cycles：\(b_e\to b_s\to b_e\)，最后固定 \(b_e\) 微调。
- 默认 2-bit 实验 \(b_s=4,b_e=2,n=1\)；\(m_s\) 是一个 half schedule 中预计算样本的数量，不是全部训练步数。
- Figure 1 示例（不是每个实验的完整配置）：\(m_s=1200\)，初始约 0–1200 步由 4 到 2；3000 开始 relax，约 4200 回到 4，约 5400 再到 2，之后固定 2。

**[推导]**

\[
\frac{\partial b}{\partial\omega}
=-(b_s-b_e)\frac{\pi}{2}\cos\left(\frac{\pi}{2}\omega\right).
\]

在 \(b_e\) 端（\(\omega=1\)）斜率为 0；在 \(b_s\) 端（\(\omega=0\)）不是 0。反向查表再正向查表的 full cycle 在最高 bit 点形成尖峰，不能宣称整个周期全局可微/C1 smooth。作者的“gradient 为 0”指日程函数导数，不是 loss 对权重/bitwidth 的反传梯度；离散 callback 中也没有真的计算此导数。

Algorithm 1（p.5）明确用 `linspace(0,1,num=m_s)`，包含两端。因此零起始步解释下，初始 schedule 是 \(t=0,\ldots,m_s-1\)，实际首尾间隔 \(m_s-1\)，不是严格 \(m_s\)。文中的 1200/4200/5400 是粗粒度日程说明，与伪代码有 1–2 步的 endpoint 差异。

### 3.2 RelaxSEQ：张量、参数与全部量化公式

**[原述，p.3]**

\[
\theta\in\mathbb R^{n\times m},\qquad \alpha\in\mathbb R^n.
\]

\(n\) 是 output channels，\(m\) 是 input channels；训练参数为 latent FP weights \(\theta\) 与每 output channel 一个可学习 scale \(\alpha\)。每次 forward 重新算 QDQ weight，供 transformer 的 `nn.Linear` 使用（p.5）。

记号重用提醒：本小节 shape 中的 \(n\) 是 output-channel 数；日程/实验中的 \(n\) 是 full-cycle 数。二者没有数值关系。

**[推导，shape 的可执行解释，不是额外披露]**

| 量 | shape / 语义 |
|---|---|
| \(\theta,W_{\mathrm{QDQ}},r,q\) | \([n,m]\) |
| \(\alpha,\delta\) | 论文写 \([n]\)；沿输入维广播应视为 \([n,1]\) |
| \(b_t,k,n_{\mathrm{levels}}\) | 每 quantizer 的 scalar；Algorithm 1 给全部 SEQ modules 相同日程 |
| 常规 Linear 输入 \(X\) | \([B,L,m]\)，也可 flatten 为 \([BL,m]\) |
| 常规 Linear 输出 \(Y=XW_{\mathrm{QDQ}}^\top\) | \([B,L,n]\)；bias 若存在为 \([n]\)，其量化情况未披露 |
| 一个 weight matrix 的训练参数数 | \(nm+n\)，scale/weight 数量比 \(1/m\) |

**必须纠正的指数：** 论文所有相应公式及 Algorithm 3 使用

\[
k=2^{\,b-1},\qquad \delta=\frac{\alpha}{k}.
\]

绝非 \(2^b-1\)。Algorithm 3 变量 `n_levels` 就是这个 \(k\)，**不是 quantization code 的总个数**；2-bit 时该变量为 2，而总 code 数应为 4。

**Eq. (2)，p.4：原 ParetoQ SEQ，Clip-then-Round**

\[
w_p=\frac{\alpha}{k}
\left\{\operatorname{round}
\left[\operatorname{clip}\left(\frac{\theta}{\alpha},-1,1\right)k-0.5\right]+0.5\right\}.
\]

**Eq. (3)，p.4：RelaxSEQ，Round-then-Clip**

\[
w_r=\frac{\alpha}{k}
\left\{\operatorname{clip}
\left(\operatorname{round}\left[\frac{\theta}{\alpha}k-0.5\right],-k,k-1\right)+0.5\right\}.
\]

**Algorithm 3，p.6，与 Eq. (3) 对齐：**

1. `n_levels = 2**(b-1)`。
2. `shift = 0.5`。
3. `delta = alpha / n_levels`。
4. `q_bins = round(theta / delta - shift)`。
5. `q_bins = clip(q_bins, -n_levels, n_levels - 1)`。
6. `q = q_bins + shift`。
7. `w_r = q * delta`。

**[推导] 三种不同“code”切勿混写：**

- raw rounded/clipped \(r=\operatorname{clip}(\operatorname{round}(\theta/\delta-0.5),-k,k-1)\)。
- shifted \(q=r+0.5\)，内部非饱和 bin 为 half-integer，**不是 integer**。
- odd-grid code \(z_{\mathrm{odd}}=2q=2r+1\)，内部为 odd integers；\(w_r=(\alpha/(2k))z_{\mathrm{odd}}\)。
- \(b=2,k=2\)：\(r\in\{-2,-1,0,1\}\)，\(q\in\{-1.5,-0.5,0.5,1.5\}\)，\(z_{\mathrm{odd}}\in\{-3,-1,1,3\}\)，\(w_r\in(\alpha/4)\{-3,-1,1,3\}\)。
- 最终权重最大绝对幅值是 \(0.75\alpha\)，不是 \(\alpha\)。本文的 \(\alpha\) 不是最终 QDQ 极值。
- 浮点 \(b\) 时 \(r\) 的两端是 \(-k,k-1\) 的非整数；\(q\) 两端为 \(\pm(k-0.5)\)；odd-grid 两端为 \(\pm(2k-1)\)。内部 rounded bins 仍离散，不是 soft-round。
- 本文没有名为 `qmax` 的公式变量。若下游解释使用 `qmax`，必须注明是 raw bound \(k-1\)、shifted bound \(k-0.5\) 还是 odd-grid bound \(2k-1\)。其数值分别不同，不能用一个 \(2^b-1\) 代替整个 Eq. (3)。
- 非整数 \(b_t\) 是训练代理 quantizer 的连续控制参数，不是能用 \(b_t\) bit 固定位宽打包的硬件数据类型。有限 bin 的**数量**仍是整数、随阈值改变；连续的是日程和部分端点，不是整数计数意义上的“连续 bin 数”。

**Fig. 2 / 6 仿真，p.4、pp.14–15，原述：** \(\theta\sim\mathcal N(0,0.02)\)，channelwise \(\alpha=\max\{\theta\}\)，固定 \(\theta,\alpha\)，只把 bitwidth 从 4 降到 2。未说明 0.02 是 std 还是 variance、shape、sample count、seed、是否 max abs，以及 \(\alpha>0\) 的处理。这个初始化只属于仿真，不能写成所有真实模型的 scale 初始化。

Figure 6 横轴 `Quantized Integer Value` 及 ±15/±3 与 \(2q\) 对齐，而不是 raw \(r\)、half-integer \(q\) 或乘 scale 后的 \(w_r\)。Figure 2 的最大值 15→3 也应按此 odd-grid 表示解读；作者未显式给出作图转换公式。

### 3.3 round、STE、LSQ 与实现披露边界

**[原述，pp.3、5]** QAT backward 通过 STE 等修改；RelaxSEQ backward “similar to” 原 SEQ，差别是按 Eq. (3) 改 clipping ranges。**没有 backward 公式、kernel 源码或伪代码。**

**[未披露]**

- `round` ties（nearest-even、half-away-from-zero、特制 rounding）；Triton 与参考实现是否逐 bit 一致。
- STE 对 rounding 的具体 surrogate；clip 的梯度门控判断用 pre-round 还是 post-round 数值；等于 clipping endpoint 的梯度约定。
- scale 的初始化、正值约束/参数化、epsilon、是否允许 scale 跨零；clipping outliers 的具体处理。
- \(\alpha\) 的梯度计算、per-channel reduction、gradient normalization、独立 scale LR 或 LR multiplier。
- 是否使用 LSQ 常见的 scale-gradient normalization；例如不能擅自填入 `1/sqrt(N*qmax)` 并当作本论文实现。
- optimizer states 的精度、latent weight 精度、QDQ tensor 精度。
- 对 \(b\) 或 \(k\) 是否有自定义 autograd 参数；正文仅支持**外部查表设置 b，不学习 b**。

**[推导，仅作审校参考，非作者 backward]** 若自行选择 `round` identity STE、按 post-round 的 clip 开区间门控，记 \(M=1_{\{-k<r_{\rm rounded}<k-1\}}\)，固定 \(k\)：

\[
\frac{\partial w}{\partial\theta}\approx M,\qquad
\frac{\partial w}{\partial\delta}\approx q-\frac{\theta}{\delta}M,\qquad
\frac{\partial w}{\partial\alpha}\approx\frac1k
\left(q-\frac{\theta}{\delta}M\right).
\]

这个例子说明 backward 要补充的内容很多，**不证明作者真的采用此门控/梯度**；逐 channel 的 scale gradient 还需对输入维求和和可能的 normalization。不能根据前向公式倒推唯一 backward。

**[疑点：Eq. (2) 与常见 round tie 不完全自洽，p.4]** 取 \(b=2,k=2,\theta/\alpha=1\)，字面 Eq. (2) 的 `round(1.5)` 若为 nearest-even 得 2，odd-grid code 为 \(2(2+0.5)=5\)，超出声称的 \(\{-3,-1,1,3\}\)。负端 `round(-2.5)=-2`，结果为 -3，因此还会不对称。clip 产生精确 ±1 时不是可简单忽略的随机测度零事件。Eq. (3) 的 round 后 clip 则把该值钳到 1，修复正端超界。可能存在未印出的 epsilon/特殊 tie/clamp；本 PDF 无法确认。不能把论文对 Eq. (2) 极值的声称等同于任意 framework 的字面执行结果。

**Eq. (4)，p.5：本文用于理论分析的 LSQ**

\[
w_l=\frac{\alpha}{k}
\left\{\operatorname{clip}
\left(\operatorname{round}\left[\frac{\theta}{\alpha}k\right],-k,k-1\right)\right\}.
\]

\(b=2,k=2\) 时 \(w_l=(\alpha/2)\{-2,-1,0,1\}\)。与 RelaxSEQ 的差别不是“同一个 qmax 加不加零点”而已；shift、integer grid、有效正负幅值均不同。本文没有给出 LSQ 的 scale 更新和初始化，也没有声称 Eq. (4) 是部署整数 kernel。

### 3.4 Callback：Algorithms 1–2 的精确证据与内部错误（p.5）

Algorithm 1 条件：

- `css > m_s`。
- `cl > 2*m_s`。
- `b_s > b_e`。
- `(ces-css) % cl == 0`。
- `B = b_s - (b_s-b_e)*sin((pi/2)*linspace(0,1,num=m_s))`。
- **印刷的** `n = ((ces-css)/cl)+1`，注释 `number of cycles`。
- `C = linspace(css,ces,num=n)`。
- 各 SEQ module：`enable_relax_qat=true; i=0; B=B; cci=0; pci=-1`。

Algorithm 2 的逐步赋值：

- 若 `t < C[0]`：`i=min(t,m_s-1)`。
- 否则若 **`t < C[-1]`**：
- 若 `t == C[cci]`：`i=m_s-1; cci+=1; pci+=1`。
- 否则若 `t <= C[pci]+m_s-1`：`i=C[pci]+m_s-1-t`，反向查 B，由 \(b_e\) 到 \(b_s\)。
- 其余：`i=min(t-(C[pci]+m_s-1),m_s-1)`，正向查 B，由 \(b_s\) 到 \(b_e\)。
- 若 `t >= C[-1]`：`i=m_s-1`，永远固定 \(b_e\)。
- 最后 `b_t=B[i]`。

**[疑点，高置信]** 按上述字面代码，实际周期启动点只有 `C[0]` 至 `C[n-2]`，`C[-1]` 是 cutoff；因此实际 full cycles 是 `n-1`，而不是 Algorithm 1 注释的 `n`。如果按其 `n=1` 设 `ces=css`，C 长度 1，`C[0]==C[-1]`，周期分支完全不会执行，无法得到实验所称的一个 full cycle。若 C 长度 2，才执行一个 full cycle。此处很可能混淆“boundary count”和“cycle count”，不能判定真实代码也错。

**[未披露]** `n=0` 消融如何由这套 initializer 表达；真实各实验的 `css/ces/cl`；resume/callback state 恢复、step offset、global step 必须按 1 连续增长的约束。`t == C[cci]` 分支对浮点 C 的类型及训练恢复很敏感，论文未交代。

日程仅是查表，不在 forward 中计算 sin。仍有每步每 module 状态访问/赋值开销；正文所谓“removes any additional overhead”不能解读成逐算子数学意义的零开销。

### 3.5 Systems Considerations：Table 1（p.5）

| Method | GPU Hours（论文原值） |
|---|---|
| ParetoQ | 370 |
| RelaxQAT | 361（-2.4%） |

设置：Qwen3-0.6B，10B tokens，8× NVIDIA H100；作者为 SEQ 和 RelaxSEQ 都实现 Triton kernels。**[推导]** 下降 \(9/370=2.4324\%\)，表中的 -2.4% 是相对时间百分比，和 accuracy 表里的 pp 不同。若这是累计 GPU hours 且 8 GPU 全程占用，对应 wall time 约 46.25 h vs 45.125 h，**该换算不是另一个实测值**。

作者明确说略快但 within margin of error；没有 error bar、重复次数、variance、吞吐率/峰值显存、energy、kernel latency、各算子的前后向时间、Triton launch/fusion、GPU 软件版本、NPU 数据。最稳妥结论是“这个配置下端到端训练耗时相近”，不是“显著训练加速/推理加速”。

## 4. Theoretical Insights：全部公式与证明范围（pp.5–7，14–15）

### 4.1 Proposition 1 与 Eq. (8)

**[原述，p.6，证明 pp.14–15]** 增加 bit precision 会把 quantization hyperplane 的宽度减少 \(2^{b_2-b_1}\) 倍。

作者证明：用 LSQ，\(\theta/\delta\) 居于某内部 integer bin 中心；移动 ±0.5 到邻边，latent \(\theta\) 需移动 ±\(0.5\delta\)，因此 bin width 为 \(\delta=\alpha/2^{b-1}\)。

**Eq. (8)，p.15，原式：**

\[
\frac{\delta_1}{\delta_2}
=\frac{\alpha/2^{b_1-1}}{\alpha/2^{b_2-1}}
=2^{b_2-b_1}
\quad\Longrightarrow\quad
\delta_2=\frac{\delta_1}{2^{b_2-b_1}}.
\]

**准确边界：**

- 固定**相同正值 \(\alpha\)**，沿 \(\theta\) 方向测内部、未饱和的 bin 宽。若 \(\alpha\) 同时学习变化，比例为 \((\alpha_1/\alpha_2)2^{b_2-b_1}\)，不能只归因于 bitwidth。
- clipping 两端是向外无界的饱和区域，不具备有限宽 \(\delta\)；非整数 k 的 edge bins 也不是全部等宽。
- 在 \((\theta,\delta)\) 二维空间内，bin 是由 \(\theta=(z\pm0.5)\delta\) 等边界围出的区域；“width”并非严格定义的全局 hyperplane 几何厚度。沿 δ 方向 loss 可变，不是整个二维区域都 constant。
- \(\theta/\delta\) 正好 ±0.5 的 tie 在 nearest-even 下仍可映射 0。p.14 直接说 `-0.5 -> -1`, `+0.5 -> +1`，隐含特定 rounding convention，与常见 nearest-even 不同；应写边界/跨过边界，而非通用等号结论。
- 实际对 bitwidth 的离散/浮点变化会重新移动边界；缩窄 bin 只能提供更易 crossing 的几何直觉，**没有证明 optimizer 必定 escape saddle，必定 faster convergence 或全局最优**。
- Figure 3 还展示负 scale 的坐标范围，而正文未说明 scale 正值约束或 \(\delta=0\) 奇点如何处理。严谨宽度应为 \(|\delta|\)；原证明默认正 δ。

### 4.2 Toy loss，Figure 3（p.6）

**[原述]** 连续一维目标 \(f(x)=x^4-3x^3-2x\)，替换为 \(f(w_l)\)，变量从 x 变为 latent scalar \(\theta\) 和 scale scalar \(\delta\)。Figure 3(b) 用 Eq. (4) 的 2-bit LSQ，红线为 QAT trajectory。

图支撑“固定 scale 切片的 integer bins 会造成阶梯/flat 段”的直观展示。未给 toy 的初值、优化器、LR、step count、mesh resolution、随机种子、scale constraint，也未提供目标最小点的精确数值或收敛误差。不能当成真实 LLM 的 Hessian/saddle 诊断。

### 4.3 Proposition 2：公式 (5)–(7)

**Eq. (5)，pp.6–7：**

\[
w^{(t)}=\delta^{(t)}
\left\lfloor\frac{\theta^{(t)}}{\delta^{(t)}}\right\rceil
=\delta^{(t)}z^{(t)},\qquad
\delta^{(t)}=\frac{\alpha^{(t)}}{2^{b-1}}.
\]

这里 \(\lfloor\cdot\rceil\) **专指 Eq. (4) 的 Round-then-Clip**，不是单独 floor；\(b\) 在此证明中固定为 integer bitwidth，\(z\) 为整数 code。

**Eq. (6)，p.7，原式：**

\[
\begin{aligned}
w^{(t+1)}
&=\delta^{(t+1)}
\left\lfloor\frac{\theta^{(t+1)}}{\delta^{(t+1)}}\right\rceil\\
&=(\delta^{(t)}-\eta d_\delta^{(t)})
\left\lfloor
\frac{\theta^{(t)}-\eta d_\theta^{(t)}}
{\delta^{(t)}-\eta d_\delta^{(t)}}
\right\rceil\\
&=(\delta^{(t)}-\eta d_\delta^{(t)})z^{(t+1)}\\
&=\delta^{(t)}z^{(t+1)}-\eta z^{(t+1)}d_\delta^{(t)} .
\end{aligned}
\]

\(\eta\) 为 LR；\(d_\delta,d_\theta\) 是 optimizer 的实际 update directions（作者举 SGD/Adam），不能直接当作 loss 的裸梯度。

**Eq. (7)，p.7，原式，条件 \(z^{(t+1)}=z^{(t)}\)：**

\[
\begin{aligned}
w^{(t+1)}
&=\delta^{(t)}z^{(t+1)}-\eta z^{(t+1)}d_\delta^{(t)}\\
&=\delta^{(t)}z^{(t)}-\eta z^{(t)}d_\delta^{(t)}\\
&=w^{(t)}-\eta z^{(t)}d_\delta^{(t)}\\
&=w^{(t)}-\eta d_w^{(t)},\qquad
d_w^{(t)}=z^{(t)}d_\delta^{(t)}.
\end{aligned}
\]

**严谨重述：** 固定位宽 LSQ、单步整个 code tensor 不变时，QDQ weight 的显式步间变化只来自 broadcast scale 更新；每个输出 channel 的那一行只能沿该行当前 z 的方向缩放。

**不能扩大的边界：**

- 不意味着 latent weights 没有 STE 梯度、不更新或停止积累 optimizer state。它们可在多个 step 后 crossing；scale 梯度/surrogate 也可能间接依赖 latent \(\theta\)。
- 不意味着 loss 不变：scale 在变；若某 LSQ code 为 0，该元素的 QDQ 单步变化也为 0，非零元素仍可变化。
- 没有 convergence rate、escape probability、Hessian bound、generalization guarantee；“scale 比 weight 少，因此会更慢”是解释，不是由参数数量直接证明的定理。
- proof 用 fixed integer b；不能原样套到 \(b_t\) 改变的 RelaxQAT，因为 \(\delta=\alpha/k_t\) 还有外部 k 的变化，edge code 也可能非整数。
- 对整数 b 的 SEQ 可重新写成 half-integer q 或 odd-integer z 做类似恒等式，但原证明没有完成该重写；“without loss of generality”不等于已证明所有 quantizer。
- 参数更新本来是 \(\alpha\) 而非独立 \(\delta\)；固定 k 时可把 \(d_\delta=d_\alpha/k\) 重参数化。k 动态变化时还要包含该项，不可遗漏。

**[推导] RelaxSEQ 不保证所有 forward weight 或 loss 对 b 连续。** 取固定 \(\theta/\alpha=0.6\)，当 k 穿过 \(10/3\) 时，round argument \(0.6k-0.5\) 穿过 1.5，内部 r 从 1 到 2，此处 clip 不截断（upper bound \(k-1=7/3\)）。所以 \(w_r\) 有普通 rounding 跳变。论文减少的是**极值因 clip-then-round 造成的额外阶梯突变**，不是消除了所有 rounding 不连续，更不是 soft/differentiable quantization。

## 5. Experiments：设置、全部表格与准确性边界（pp.7–10）

### 5.1 Experimental Setup 与 Appendix D 补充（pp.7、15）

| 项目 | 论文披露 |
|---|---|
| 语言模型 | Qwen3-0.6B、Qwen3-4B、Qwen3.5-2B、Qwen3.5-4B、Gemma3-1B |
| 精确名字 | Qwen3-4B 明确为 `Qwen3-4B-Instruct-2507`；Gemma 在 p.10 为 `Gemma-3-1B-it`；其他模型未给完整 repo/revision |
| VLM | Qwen3.5-2B、Qwen3.5-4B |
| 量化对象 | 所有实验 **weight only**；channelwise，每 output channel 的 scale |
| 训练数据 | DCLM-Edu、tulu-3-sft-olmo-2-mixture、xlam-function-calling-60k 的 mixture |
| VLM 额外数据 | Llama-Nemotron-VLM-Dataset-v1 |
| token budget | all models 10B tokens |
| LR | all models/baselines 初始 0.0001，cosine decay 到 nearly zero |
| optimizer | AdamW，weight decay = 0（p.15） |
| total batch size | Qwen3-0.6B、Qwen3-4B、Gemma3-1B：256；Qwen3.5-2B、Qwen3.5-4B：320 |
| 硬件 | smaller models：单节点 8 H100；bigger models：两节点 A100；未逐模型列节点/GPU数映射 |
| 2-bit 日程 | \(b_s=4,b_e=2,n=1\)；\(m_s\) grid search `{400,800,1200}` |
| \(m_s\) 消融 | `{400,800,1200,1600,2000}`，固定 n=1 |
| n 消融 | `{0,1,2,3,4}`，固定 \(m_s=400\) |
| ternary 日程 | Appendix F：\(b_s=3,b_e=1.58\)；逐实验 \(m_s/css/ces/cl\) 未给 |
| CSR | 0-shot：ARC-Easy、ARC-Challenge、HellaSwag、WinoGrande、PIQA |
| CSR metric | ARC-Easy/Challenge、HellaSwag、PIQA 为 accuracy norm；WinoGrande 为 accuracy（p.15） |
| MMLU | 5-shot，loglikelihood multiple-choice；57 subjects（pp.7、15） |
| IFEval | generation；prompt-level-strict、inst-level-strict、prompt-level-loose、inst-level-loose 四种 accuracy 的平均（p.15） |
| VLM benchmark | MMMU、DocVQA、ScienceQA；未给 split、shot、DocVQA scoring 口径 |

**2-bit BitNet baseline 的明确改造：** 原引用的 Hugging Face BitNet 1.58-bit 版本没有 learnable scale；作者用 ParetoQ SEQ 生成 2-bit QDQ，再放入 BitNet algorithm（p.7）。因此应标注“本文适配的 2-bit BitNet”，不能暗示这是未修改的原发布配置。

**Aggregate 定义与公平解释：**

\[
\mathrm{CSR}=\frac{\mathrm{ArcE+ArcC+HS+WG+PIQA}}{5},
\qquad
\mathrm{GlobalAvg}=\frac{\mathrm{CSR+MMLU+IFEval}}{3}.
\]

Global Avg 不是 7 个原始 benchmark 的等权平均：五个 CSR 原始任务各权重 \(1/15\)，MMLU 与 IFEval 各 \(1/3\)。IFEval 内四项再等权，各权重 \(1/12\)。表格的 CSR 和 Global 列是派生 aggregate，不能当作独立任务再计数。VLM Avg 为三个任务等权平均。所有下列表格保留论文原值，不用舍入后重算值覆盖原值。

### 5.2 Language Modeling Results：Tables 2–5

以下表头顺序固定：ArcE、ArcC、HS、WG、PIQA、CSR、MMLU、IFEval、Global Avg。所有 score 越高越好；`ΔP` 是论文 Global 列括号中的相对 ParetoQ 差值，**按 pp 解读**。主 QAT 行包括 ParetoQ、UPQ、LLM-QAT、BitNet、WinQ、RelaxQAT；FP 和 AdaScale 单列其类型，不再复刻原表合并单元格。

#### Table 2，p.8：Qwen3-4B，2-bit

| Method | ArcE | ArcC | HS | WG | PIQA | CSR | MMLU | IFEval | Global Avg | ΔP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 81.22 | 58.44 | 69.08 | 67.48 | 75.73 | 70.39 | 72.63 | 86.55 | 76.52 | — |
| AdaScale (PTQ) | 25.54 | 26.36 | 26.94 | 50.19 | 50.76 | 35.95 | 26.89 | 20.27 | 27.70 | — |
| ParetoQ | 74.92 | 48.38 | 62.08 | 63.77 | 73.78 | 64.58 | 62.30 | 80.22 | 69.03 | +0.00 |
| UPQ | 75.97 | 49.66 | 61.79 | 62.67 | 73.23 | 64.66 | 61.21 | 79.18 | 68.35 | -0.68 |
| LLM-QAT | 76.94 | 48.29 | 61.29 | 63.93 | 73.45 | 64.78 | 59.66 | 79.73 | 68.05 | -0.98 |
| BitNet | 69.86 | 43.51 | 60.67 | 61.95 | 71.27 | 61.45 | 60.15 | 79.21 | 66.93 | -2.10 |
| WinQ | 75.42 | 48.55 | 62.23 | 65.51 | 73.99 | 65.14 | 62.80 | 80.26 | 69.40 | +0.37 |
| RelaxQAT | 76.47 | 50.26 | 63.05 | 64.48 | 74.43 | 65.73 | 62.84 | 81.54 | 70.03 | +1.00 |

相对 strongest other QAT WinQ：+0.63 pp；相对 FP：-6.49 pp。7/9 列为 QAT 中最佳，ArcE 和 WG 第二。Table 2 caption 没有像其他表一样明确写 “among QAT methods”，但 FP 的九列全高于 QAT；故原文“best overall”必须理解为 QAT 子集，而非连 FP 在内。

#### Table 3，p.8：Qwen3-0.6B，2-bit

| Method | ArcE | ArcC | HS | WG | PIQA | CSR | MMLU | IFEval | Global Avg | ΔP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 56.01 | 34.55 | 47.29 | 55.48 | 67.73 | 52.21 | 47.25 | 64.78 | 54.74 | — |
| AdaScale (PTQ) | 40.45 | 25.51 | 32.78 | 52.49 | 59.41 | 42.12 | 25.96 | 21 | 29.69 | — |
| ParetoQ | 50.79 | 27.64 | 40.19 | 52.40 | 64.03 | 47.01 | 37.18 | 33.85 | 39.34 | +0.00 |
| UPQ | 50.42 | 27.56 | 39.58 | 52.09 | 63.82 | 46.69 | 36.54 | 43.89 | 42.37 | +3.03 |
| LLM-QAT | 49.91 | 28.41 | 39.76 | 52.72 | 64.09 | 46.97 | 37.35 | 44.61 | 42.97 | +3.63 |
| BitNet | 52.44 | 28.49 | 40.34 | 55.64 | 64.20 | 48.22 | 37.62 | 49.29 | 45.04 | +5.70 |
| WinQ | 50.16 | 27.30 | 40.12 | 55.01 | 63.76 | 47.27 | 37.02 | 49.56 | 44.61 | +5.27 |
| RelaxQAT | 52.95 | 30.03 | 40.90 | 53.28 | 64.09 | 48.25 | 40.14 | 52.73 | 47.04 | +7.70 |

相对 strongest other QAT BitNet +2.00 pp；相对 FP -7.70 pp。IFEval 对 ParetoQ +18.88 pp，是 headline 改善的主要来源。按已报 aggregate 分解：CSR +1.24、MMLU +2.96、IFEval +18.88，三者平均约 +7.69 pp，与原 Global 差 +7.70 的 0.01 pp 差来自原值舍入。

Figure 4（p.8）仅为 Qwen3-0.6B train/loss vs global step；横轴到约 10000，纵轴 log scale。无精确可导出的逐步 loss 数据、error bars 或多 seed。后续 full cycle 暂时提高 bit 精度，所以 cycle 内更低的 training loss 不能全部解释为同精度模型更好；末段同为目标 bitwidth 后仍更低是比较更有意义的部分。

#### Table 4，p.9：Qwen3.5-4B，2-bit，语言域

| Method | ArcE | ArcC | HS | WG | PIQA | CSR | MMLU | IFEval | Global Avg | ΔP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 75.63 | 54.35 | 73.01 | 69.69 | 77.97 | 70.13 | 69.70 | 85.31 | 75.04 | — |
| AdaScale (PTQ) | 24.32 | 25 | 26.44 | 51.22 | 51.79 | 35.75 | 24.16 | 16.12 | 25.34 | — |
| ParetoQ | 73.94 | 49.57 | 67.48 | 67.48 | 75.40 | 66.77 | 63.95 | 74.32 | 68.34 | +0.00 |
| UPQ | 71.84 | 46.67 | 67.10 | 67.64 | 76.33 | 65.91 | 63.83 | 75.15 | 68.29 | -0.05 |
| LLM-QAT | 73.95 | 47.78 | 67.25 | 65.90 | 75.68 | 66.11 | 64.03 | 73.95 | 68.03 | -0.31 |
| BitNet | 72.90 | 48.21 | 66.65 | 67.32 | 75.24 | 66.06 | 63.93 | 76.29 | 68.76 | +0.42 |
| WinQ | 76.73 | 51.96 | 67.30 | 66.69 | 75.46 | 67.62 | 64.44 | 75.88 | 69.31 | +0.97 |
| RelaxQAT | 74.41 | 48.89 | 67.47 | 68.03 | 75.79 | 66.91 | 65.21 | 77.05 | 69.72 | +1.38 |

strongest other QAT WinQ +0.41 pp；FP gap -5.32 pp。8/9 列为 QAT best/second，ARC-Challenge 为第三；部分 CSR 子任务低于 ParetoQ，不能写逐任务全面提升。

#### Table 5，p.9：Qwen3.5-2B，2-bit，语言域

| Method | ArcE | ArcC | HS | WG | PIQA | CSR | MMLU | IFEval | Global Avg | ΔP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 66.03 | 41.46 | 62.06 | 62.58 | 72.52 | 60.93 | 57.19 | 72.84 | 63.65 | — |
| AdaScale (PTQ) | 24.32 | 25.00 | 26.44 | 51.22 | 51.79 | 35.75 | 24.16 | 16.12 | 25.34 | — |
| ParetoQ | 60.81 | 37.28 | 57.38 | 62.50 | 71.81 | 57.95 | 54.42 | 62.12 | 58.16 | +0.00 |
| UPQ | 60.77 | 36.60 | 56.92 | 62.66 | 71.27 | 57.64 | 54.63 | 62.59 | 58.28 | +0.12 |
| LLM-QAT | 58.54 | 36.77 | 56.62 | 62.11 | 71.16 | 57.04 | 53.40 | 62.15 | 57.53 | -0.63 |
| BitNet | 51.64 | 33.95 | 54.76 | 60.77 | 70.02 | 54.22 | 50.91 | 59.91 | 55.01 | -3.15 |
| WinQ | 61.36 | 36.94 | 56.70 | 62.03 | 71.32 | 57.67 | 53.43 | 59.07 | 56.72 | -1.44 |
| RelaxQAT | 62.58 | 37.45 | 56.72 | 60.85 | 71.59 | 57.83 | 54.34 | 63.39 | 58.52 | +0.36 |

strongest other QAT UPQ +0.24 pp；FP gap -5.13 pp。6/9 列 best/second。**[疑点]** AdaScale 行九个分数与 Table 4 的 Qwen3.5-4B 行完全相同（25 与 25.00 是显示差异），可能是拷贝错误，也可能碰巧重复，论文没有解释；不要擅改数据或认定已证实造假。

### 5.3 Vision-Language Modeling Results：Tables 6–7（p.9）

#### Table 6：Qwen3.5-4B，2-bit，VLM

| Method | MMMU | DocVQA | ScienceQA | Avg | ΔP (pp) |
|---|---:|---:|---:|---:|---:|
| FP | 41.22 | 94.51 | 85.52 | 73.75 | — |
| AdaScale (PTQ) | 26.00 | 0 | 0 | 8.66 | — |
| ParetoQ | 40.78 | 90.42 | 81.85 | 71.02 | +0.00 |
| UPQ | 42.44 | 90.80 | 82.75 | 72.00 | +0.98 |
| LLM-QAT | 39.22 | 90.76 | 79.67 | 69.88 | -1.14 |
| BitNet | 40.33 | 90.60 | 81.71 | 70.88 | -0.14 |
| WinQ | 34.00 | 7.76* | 65.94 | 35.90 | -35.12 |
| RelaxQAT | 40.33 | 91.25 | 83.94 | 71.84 | +0.82 |

`*` 原注：WinQ training instability caused loss of accuracy。RelaxQAT 平均第二，**比 UPQ 低 0.16 pp**；FP gap -1.91 pp。3/4 列 best/second，但 MMMU 的 40.33 与 BitNet 相同且低于 UPQ/ParetoQ，为并列第三。DocVQA 91.25 和 ScienceQA 83.94 在 QAT 中最高。AdaScale 的 Avg 8.66 与 26/3 四舍五入 8.67 有 0.01 pp 差，应保留原值并注明。

#### Table 7：Qwen3.5-2B，2-bit，VLM

| Method | MMMU | DocVQA | ScienceQA | Avg | ΔP (pp) |
|---|---:|---:|---:|---:|---:|
| FP | 39.77 | 92.05 | 82.89 | 71.57 | — |
| AdaScale (PTQ) | 25.55 | 0 | 0 | 8.52 | — |
| ParetoQ | 31.77 | 87.73 | 52.30 | 57.26 | +0.00 |
| UPQ | 30.55 | 88.09 | 60.68 | 59.77 | +2.51 |
| LLM-QAT | 30.66 | 87.34 | 56.66 | 58.22 | +0.96 |
| BitNet | 29.11 | 87.90 | 59.29 | 58.76 | +1.50 |
| WinQ | 29.11 | 72.16 | 57.01 | 52.76 | -4.50 |
| RelaxQAT | 33.11 | 87.71 | 69.75 | 63.52 | +6.26 |

strongest other QAT UPQ +3.75 pp；FP gap -8.05 pp。ScienceQA 相对 ParetoQ **+17.45 pp**，相对 UPQ **+9.07 pp**（原文“nearly 9%”也应按 pp）；MMMU 对 ParetoQ +1.34 pp，DocVQA **-0.02 pp**。RelaxQAT 的 DocVQA 在 QAT 中第四，并非每个单任务都高/次高。大 headline 主要源自 ScienceQA。

### 5.4 Ablations：Tables 8–9（p.10）

#### Table 8：Qwen3-0.6B，固定 n=1，改变 \(m_s\)

| \(m_s\)/Method | ArcE | ArcC | HS | WG | PIQA | CSR | MMLU | IFEval | Global Avg | ΔP (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 56.01 | 34.55 | 47.29 | 55.48 | 67.73 | 52.21 | 47.25 | 64.78 | 54.74 | — |
| ParetoQ | 50.79 | 27.64 | 40.19 | 52.40 | 64.03 | 47.01 | 37.18 | 33.85 | 39.34 | +0.00 |
| 400 | 53.11 | 29.01 | 40.82 | 53.91 | 63.82 | 48.13 | 40.01 | 51.28 | 46.47 | +7.13 |
| 800 | 52.95 | 30.03 | 40.90 | 53.28 | 64.09 | 48.25 | 40.14 | 52.73 | 47.04 | +7.70 |
| 1200 | 53.24 | 29.10 | 40.88 | 55.49 | 63.98 | 48.54 | 40.09 | 49.78 | 46.14 | +6.80 |
| 1600 | 50.88 | 28.92 | 41.19 | 54.14 | 63.71 | 47.77 | 40.57 | 49.26 | 45.87 | +6.53 |
| 2000 | 51.14 | 28.75 | 41.22 | 56.27 | 63.98 | 48.27 | 40.12 | 48.99 | 45.79 | +6.45 |

所有 m_s 对该 ParetoQ 单次结果 +6.45–7.70 pp，也超过 Table 3 BitNet 45.04；但 n=1/ms=800 为明显最佳 aggregate，不能写全部超参一样。800 行与 Table 3 RelaxQAT 全部九列一致，可合理识别为该主结果所用 m_s；其他模型最佳 m_s 不可由 PDF 得知。

#### Table 9：Qwen3-0.6B，固定 \(m_s=400\)，改变 n

| n/Method | ArcE | ArcC | HS | WG | PIQA | CSR | MMLU | IFEval | Global Avg | ΔP (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 56.01 | 34.55 | 47.29 | 55.48 | 67.73 | 52.21 | 47.25 | 64.78 | 54.74 | — |
| ParetoQ | 50.79 | 27.64 | 40.19 | 52.40 | 64.03 | 47.01 | 37.18 | 33.85 | 39.34 | +0.00 |
| 0 | 51.68 | 29.10 | 41.01 | 54.14 | 63.22 | 47.83 | 39.84 | 43.81 | 43.83 | +4.49 |
| 1 | 49.66 | 28.75 | 40.78 | 53.43 | 63.98 | 47.32 | 39.45 | 52.02 | 46.26 | +6.92 |
| 2 | 52.95 | 29.01 | 40.68 | 53.91 | 63.38 | 47.99 | 40.67 | 49.61 | 46.09 | +6.75 |
| 3 | 51.18 | 28.41 | 41.22 | 55.01 | 64.42 | 48.05 | 39.39 | 49.71 | 45.72 | +6.38 |
| 4 | 50.80 | 28.84 | 41.09 | 54.78 | 63.33 | 47.77 | 40.45 | 48.44 | 45.55 | +6.21 |

n=0 已比 ParetoQ +4.49 pp；n=1 最佳，+6.92 pp；非零 n 相对 n=0 +1.72–2.43 pp。说明本模型上 full cycle 有额外效果，不证明更多 cycles 单调更好、其他模型同样 robust，或机制已被因果隔离。

**[疑点，高置信]** 相同标签 n=1/ms=400，Table 8 是 46.47，Table 9 是 46.26，九列全部不同。可为不同 seed、不同 cycle timing、不同 run 或标注问题，但未披露，不能合并或选择其一当作唯一重复配置结果。

### 5.5 Ternary 结果的正文导读（p.10）

正文称 Qwen3-4B 比 ParetoQ +4.2%，Qwen3.5-2B +1.38%；均是 Tables 12–13 Global Avg 的 pp。具体量化式和全部结果按原附录顺序见第 F 节。不能用 Eq. (3) 的 shift=0.5 直接得到 ternary。

## 6. Conclusion and Future Work（p.10）

**[原述]** 连续控制训练中的 bitwidth、改善早期训练及 flat region 问题；再次给 6.3%/7.7% headline。正文明确：“although we have considered weight only quantization … activation or KV-cache … future work.”

**边界：** 激活量化、KV-cache 量化不是本论文已验证的 RelaxQAT 能力。没有 activation quantizer 的粒度、bitwidth、校准、量化误差实验；没有 KV-cache 的 bitwidth、per-head/per-token 粒度、容量、latency 或长上下文 accuracy 实验。

## References 与 7. AI Usage Disclosure（pp.10–13）

已阅读参考文献页；它们只给引用，不补足本文实验超参数或实现细节。Qwen3 technical report 被引用，未在本文列 architecture tensor sizes；没有 Qwen3.5 配置表。AIMET、BitNet、数据集等外链均未访问。

p.13 原述：ideation、literature search、experiment design、code implementation、writing 没有 AI；Claude Code 仅用于处理若干训练的 TensorBoard logs 与 matplotlib 绘图 Python code（Figure 4）。应保留这个明确例外，不能笼统写“完全没有使用 AI”。“code implementation”与绘图代码的区分应理解为方法实现 vs 辅助分析代码，不能仅据措辞认定不诚实。

## Appendix A. Existing SEQ vs. RelaxSEQ（pp.14–15）

同一 bit schedule 下比较 Clip-then-Round 与 Round-then-Clip，Figure 5 展示旧式的损失 spike、末端 loss 接近；Table 10 给 MMLU 小幅改善。该 early training 的 schedule、总 tokens、seed、是否和主表同 run 没有给出。

### Table 10，p.14：早期 Qwen3-4B-Instruct-2507 MMLU

| Method | MMLU |
|---|---:|
| ParetoQ SEQ + bit schedule | 62.47 |
| RelaxSEQ + bit schedule | 62.96（+0.5%） |

差值 **0.49 pp**，原表约 +0.5。不能与主 Table 2 MMLU 62.84 混写，后者是不同报告结果。这个对照支持操作顺序有作用，但只有一个 benchmark，未提供统计置信度或“same final loss implies generalization improvement”的机制证明。

**Figure 5 图读：** x=Global step 到 40000，y=Training loss 约 0–1.0；图例 orange/red 是 `Round-then-clip`，green 是 `Clip-then-round`。与 Figure 2 中两方法的红/绿色分配不同，不能跨图沿用颜色解释。细噪声线与粗线的关系（平滑/重复 run）未说明。约 12000 的 Clip-then-Round spike 视觉明显，未提供逐步数值；不将图读近似当成实验全精确值。

**Figure 6 全部 panel 文字数值，p.15：**

| Panel | Method | bits（图中） | n_levels（图中） | 两端标注 |
|---|---|---:|---:|---|
| a | Clip-then-Round | 4.00 | 8.0 | -15.0, +15.0 |
| b | Clip-then-Round | 3.71 | 6.5 | -13.0, +13.0 |
| c | Clip-then-Round | 3.21 | 4.6 | -9.0, +9.0 |
| d | Clip-then-Round | 2.00 | 2.0 | -3.0, +3.0 |
| e | Round-then-Clip | 4.00 | 8.0 | -15.0, +15.0 |
| f | Round-then-Clip | 3.71 | 6.5 | -12.1, +12.1 |
| g | Round-then-Clip | 3.21 | 4.6 | -8.3, +8.3 |
| h | Round-then-Clip | 2.00 | 2.0 | -3.0, +3.0 |

横轴 Quantized Integer Value；纵轴 histogram counts（到 100000），未给精确各柱 counts；不能从最高坐标刻度推出 sample total。p.14 正文写极值从 `{-15,15}` 到 `{-13,-13}`，后一个负号为明显文字 typo；Figure 6 图/图注写 `{-13,13}`。图中浮点极值 axis 仍称 integer 是术语不严谨，不应把小数改成整数。

## Appendix B. Proof of Proposition 1（pp.14–15）

完整证明和 Eq. (8) 已按正文逻辑放在第 4.1 节。这里没有额外的收敛或最优性 theorem；附录只证明固定 scale 下内部 bin width 的比例。

## Appendix C. SEQ Simulation（pp.15–16）

ParetoQ fixed 2-bit；RelaxQAT 使用只有 initial half cycle 的日程。原述：ParetoQ 约 2000 步 near optimum 后到 10000 步仍未达 optimum；RelaxQAT 约 2000 步达 optimum；日程在 1000 步结束。没有 objective 精确定义/归一化和 toy optimizer 设置；Figure 3 给的 quartic 不应未经核实直接赋予 Figure 7 的全部数值。

**Figure 7 全部 panel title，p.16：**

| Panel | Method | bits | Step（图中） |
|---|---|---:|---:|
| a | ParetoQ | 2.0 | 0 |
| b | ParetoQ | 2.0 | 800 |
| c | ParetoQ | 2.0 | 2000 |
| d | ParetoQ | 2.0 | 9900 |
| e | RelaxQAT | 4.00 | 0 |
| f | RelaxQAT | 2.10 | 800 |
| g | RelaxQAT | 2.00 | 2000 |
| h | RelaxQAT | 2.00 | 9900 |

原图注只称 `(a) ParetoQ`、`(b) RelaxQAT`，实际为上排 a–d、下排 e–h；面板索引误简化。最后实际展示 Step 9900，不是精确截图 10000。未知最优点 exact coordinates、error tolerance 与每步 loss，故“converges”是 toy 的视觉/作者解释，不能当成 quantitative convergence theorem。

## Appendix D. More Details on Experiments（p.15）

total batch、AdamW/zero decay、CSR metric、IFEval 四指标平均已经完整纳入 5.1。附录并未给 activation dtype、sequence length、warmup、training objective、loss mask、distillation、数据 mixture ratios、评测 command 等关键细节。

## Appendix E. Ablation Results：Table 11（p.17）

### Table 11：Gemma3-1B，2-bit

| Method | ArcE | ArcC | HS | WG | PIQA | CSR | MMLU | IFEval | Global Avg | ΔP (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 63.43 | 38.31 | 57.79 | 58.48 | 72.42 | 58.09 | 41.37 | 73.38 | 57.61 | — |
| AdaScale (PTQ) | 28.49 | 23.89 | 27.72 | 49.56 | 53.48 | 36.63 | 23.75 | 9.80 | 23.39 | — |
| ParetoQ | 52.48 | 30.12 | 46.13 | 58.41 | 67.90 | 51.01 | 25.28 | 53.12 | 43.14 | +0.00 |
| UPQ | 53.03 | 29.01 | 45.64 | 56.83 | 68.17 | 50.54 | 25.34 | 42.01 | 39.30 | -3.84 |
| LLM-QAT | 25.97 | 25.68 | 26.14 | 50.43 | 51.63 | 35.97 | 22.95 | —* | —* | — |
| BitNet | 52.82 | 28.92 | 45.24 | 56.99 | 67.63 | 50.32 | 26.23 | 46.61 | 41.05 | -2.09 |
| WinQ | 53.62 | 28.75 | 46.33 | 58.17 | 68.88 | 51.15 | 25.64 | 54.80 | 43.86 | +0.72 |
| RelaxQAT | 53.45 | 29.10 | 46.14 | 58.09 | 67.85 | 50.93 | 28.24 | 60.31 | 46.49 | +3.35 |

`*` 原注：LLM-QAT produced repeating tokens for IFEval, hence its accuracy cannot be reported。不得填成 0 或替它计算 Global；也需保留“未报告”可能改变基线比较完整性的限制。其他方法之间 strongest WinQ +2.63 pp；IFEval 相对 WinQ +5.51 pp（正文“nearly 6%”）；FP gap -11.12 pp。6/9 列 QAT best/second，CSR 本身 50.93 还低于 ParetoQ 51.01。

## Appendix F. Ternary Quantization Results（pp.16–17）

### Eq. (9)，p.16：ternary RelaxSEQ

\[
w_r=\frac{\alpha}{k}
\left\{\operatorname{clip}
\left(\operatorname{round}\left[\frac{\theta}{\alpha}k\right],
-k+0.5,k-0.5\right)\right\}.
\]

与 Eq. (3) 不同：round 前没有 -0.5，clip 后没有 +0.5；clip 范围对称 \([-k+0.5,k-0.5]\)。日程从 \(b_s=3\) 到 \(b_e=1.58\)。作者写
\(k=2^{b_e-1}=2^{1.58-1}\approx1.5\)，因此得到 \(\{-1,0,1\}\) grid。

**[疑点/推导]** 字面 \(b_e=1.58\)：

\[
k=1.494849248635,\qquad k-0.5=0.994849248635.
\]

round 后的 ±1 被 clip 为约 ±0.994849，而不是 ±1。严格整数终点需 \(k=1.5\)，即 \(b_e=\log_2 3=1.584962500721\)，或显式终点 snap。本论文没交代是否用精确 log2(3) 而把 1.58 当作展示缩写。**不能仅据印刷值断言实际导出非整数，也不能断言已严格得到 {-1,0,1} packed codes。** 即使采用 literal endpoint，仍有三个对称 levels，可重吸收 scale 表示为 ternary；这不是作者披露的导出过程。

起点 b=3 时 k=4，Eq. (9) clip 极值为 ±3.5，内部有整数 -3…3，共最多 9 个 levels（视饱和情况），不等于 8 个 strict 3-bit codes。其起始 b 也是 schedule 控制量，不是实际固定 3-bit 存储的证明。

### Table 12，p.17：Qwen3-4B-Instruct-2507，ternary

| Method | ArcE | ArcC | HS | WG | PIQA | CSR | MMLU | IFEval | Global Avg | ΔP (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 81.22 | 58.44 | 69.08 | 67.48 | 75.73 | 70.39 | 72.63 | 86.55 | 76.52 | — |
| ParetoQ | 75.08 | 47.44 | 58.28 | 61.56 | 71.87 | 62.85 | 54.27 | 68.18 | 61.77 | +0.00 |
| UPQ | 72.81 | 45.14 | 58.48 | 62.90 | 72.52 | 62.37 | 55.88 | 74.88 | 64.38 | +2.61 |
| BitNet | 70.96 | 44.88 | 57.16 | 60.93 | 71.49 | 61.08 | 54.46 | 74.87 | 63.47 | +1.70 |
| WinQ | 74.49 | 45.48 | 56.48 | 61.01 | 72.25 | 61.94 | 53.15 | 72.41 | 62.50 | +0.73 |
| RelaxQAT | 73.23 | 47.35 | 60.89 | 65.35 | 73.50 | 64.06 | 58.96 | 74.90 | 65.97 | +4.20 |

strongest other QAT UPQ +1.59 pp；FP gap -10.55 pp。ArcE / ArcC 都低于 ParetoQ，IFEval 只高于 UPQ 0.02 pp，不支持所有项显著胜出。AdaScale、LLM-QAT 没有在此表报告。

### Table 13，p.17：Qwen3.5-2B，ternary

| Method | ArcE | ArcC | HS | WG | PIQA | CSR | MMLU | IFEval | Global Avg | ΔP (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP | 66.03 | 41.46 | 62.06 | 62.58 | 72.52 | 60.93 | 57.19 | 72.84 | 63.65 | — |
| ParetoQ | 62.54 | 37.63 | 54.82 | 61.40 | 70.24 | 57.33 | 50.31 | 57.35 | 55.00 | +0.00 |
| UPQ | 28.66 | 22.35 | 24.65 | 51.38 | 50.38 | 35.49* | 24.08* | 18.34* | 25.97* | -29.03 |
| BitNet | 56.73 | 35.24 | 54.09 | 60.06 | 69.48 | 55.12 | 47.31 | 60.49 | 54.31 | -0.69 |
| WinQ | 61.99 | 36.86 | 54.92 | 61.56 | 71.00 | 57.27 | 50.28 | 56.60 | 54.72 | -0.28 |
| RelaxQAT | 59.93 | 36.69 | 55.33 | 61.25 | 70.29 | 56.70 | 51.64 | 60.80 | 56.38 | +1.38 |

`*` 原注：UPQ did not converge well for ternary training of Qwen3.5-2B。最强其他 baseline 实际是 ParetoQ 55.00，因此 +1.38 pp；FP gap -7.27 pp。CSR -0.63 pp、MMLU +1.33 pp、IFEval +3.45 pp，对 ParetoQ 的优势不是所有 CSR tasks 提升。AdaScale、LLM-QAT 均没有报告。

## 8. 全部 Figure / Table / Algorithm 目录与原始 caption

caption 按 PDF 转录，合并排版换行和断字；公式上下标、负号用等价可读记法。不保留正文中的字体强调。下述引用是原文，不意味着认同其 smoothness、causal 或 “best” 强措辞。

### 8.1 Figure 目录（共 7 幅）

| Figure | PDF 页 | 原始位置 | 内容 |
|---|---:|---|---|
| 1 | 3 | 3.1 | RelaxQAT / UPQ / ParetoQ 日程，a–c |
| 2 | 4 | 3.2 | Extreme range 随 bit schedule 变化 |
| 3 | 6 | 4 | quartic 1-d 目标与 2-d LSQ 量化表面，a–b |
| 4 | 8 | 5.2 | Qwen3-0.6B 的 ParetoQ / RelaxQAT train loss |
| 5 | 14 | Appendix A | 早期 SEQ / RelaxSEQ 的 train loss |
| 6 | 15 | Appendix A 延续 | 8 个 histogram panels，a–h |
| 7 | 16 | Appendix C 延续 | 8 个 toy surface/trajectory panels，a–h |

**Figure 1，p.3，caption：**

> (a) An example of our proposed RelaxQAT bitwidth schedule which gradually and smoothly reduces the bitwidth. The gradient of the schedule 0 whenever it is approaches 2-bits; this helps to ease into the 2-bit precision instead of an abrupt change. (b) Prior progressive quantization methods reduce the bitwidth suddenly during training like UPQ, or (c) keep it the same throughout.

**Figure 2，p.4，caption：**

> ParetoQ SEQ with a bitwidth schedule loses extreme integers as k = 2^(b_t-1) decreases with a shrinking bitwidth. RelaxSEQ shows a smooth change in extreme values which results in a smoothly changing loss.

图中额外标题：`Extreme Range Changes vs. Bitwidth Schedule`；x=`Schedule Step`，y=`Extreme Value (Max)`；red Clip-then-Round、green Round-then-Clip。注释原值：从 `{-13,13}` 到 `{-11,11}` 到 `{-9,9}` 等。没有离散点的完整数值文件，因此仅转录标注和范围，不捏造曲线采样值。

**Figure 3，p.6，caption：**

> (a) A simple 1-d function to minimize. (b) The 1-d function above results in a highly complex 2-d, 2-bit quantization loss surface (generated using LSQ) where the width of the hyperplanes (defined as the flat regions between integer changes) increases with larger scales. Red line shows the QAT trajectory.

图 3(a) 标题 `Single variable function: z=x^4-3x^3-2x`，axes x / z。3(b) 标题 `Loss Surface and Optimization Trajectory`，axes Latent Weight / Scale / z；标记 Initialization、Optimal point、Final point reached、Quantized trajectory。没有 exact coordinates。

**Figure 4，p.8，caption：**

> RelaxQAT converges significantly faster than ParetoQ.

图标题 `train/loss vs. step`；axes Global step / train/loss；blue ParetoQ、orange RelaxQAT。不能把训练步数轴当作 GPU-hour 轴。

**Figure 5，p.14，caption：**

> Existing SEQ (Clip-then-Round) with a bitwidth schedule shows an unstable training behavior with sudden loss spikes as predicted by Fig. 2. The new RelaxSEQ with Round-the-Clip has a smooth transition in loss with changing bitwidth. Although, the final loss after the end of schedule was similar for both Clip-then-Round vs. Round-then-Clip, we found that RelaxSEQ resulted in better benchmark scores (see Table 10).

`Round-the-Clip` 为原图注 typo。图中 axes Global step / Training loss；图例颜色与 Figure 2 对同方法的颜色不同，见 Appendix A。

**Figure 6，p.15，caption：**

> (a-d) Clip-then-Round operation in the existing SEQ (Eq. (2)) creates integers at the extremes (see vertical red dotted lines) since the round operation follows the clip. As the bitwidth reduces, the extreme integers are suddenly lost (e.g., from {-15,15} to {-13,13} and so on). This means that a training assumes that the previous range of integers (say {-15,15}) was available until k = 2^(b-1) reduces sufficiently; once k is sufficiently small, the round abruptly moves on to the new range of integers (i.e., {-13,13}). The sudden change in integer ranges can cause loss spikes (similar to UPQ (Lee et al., 2025)) and unstable training behavior. (e-h) RelaxSEQ employs the Round-then-Clip operation (Eq. (3)) which keeps the extremes in floating point as the k reduces. The range of the distribution changes slowly and the extremes stay floating point until the end of the schedule. This causes the loss to change smoothly as the bitwidth shrinks.

全部 panel 数值见 Appendix A 表。原注因果解释仅由 toy 图/early run 支持；极值“直到 end 才 integer”的说法并非排除途中 k 恰为整数的日程采样。

**Figure 7，p.16，caption：**

> (a) ParetoQ SEQ simulation for a 1-d optimization problem. QAT gets stuck near the minimum point around step 2000 and stays there until the end of training (10000 steps). (b) RelaxQAT (with just a half cycle bit schedule) converges within the first 2000 steps.

panel 编号与 caption 的 `(a)/(b)` 分组简写不匹配，见 Appendix C。额外标注：`Loss surface changes until step 1000 (end of bit schedule)`；`ParetoQ gets stuck near the optimal point and does not reach it even until the end (+)`；`RelaxQAT explores an alternative optimization path due to a changing loss surface. It converges and reaches the optimal point by step 2000`。

### 8.2 Table 目录（共 13 张）

| Table | PDF 页 | 内容 | 数值所在笔记节 |
|---|---:|---|---|
| 1 | 5 | Qwen3-0.6B GPU Hours | 3.5 |
| 2 | 8 | Qwen3-4B，2-bit，language | 5.2 |
| 3 | 8 | Qwen3-0.6B，2-bit，language | 5.2 |
| 4 | 9 | Qwen3.5-4B，2-bit，language | 5.2 |
| 5 | 9 | Qwen3.5-2B，2-bit，language | 5.2 |
| 6 | 9 | Qwen3.5-4B，2-bit，VLM | 5.3 |
| 7 | 9 | Qwen3.5-2B，2-bit，VLM | 5.3 |
| 8 | 10 | \(m_s\) 消融，Qwen3-0.6B | 5.4 |
| 9 | 10 | n 消融，Qwen3-0.6B | 5.4 |
| 10 | 14 | early SEQ/RelaxSEQ MMLU | Appendix A |
| 11 | 17 | Gemma3-1B，2-bit | Appendix E |
| 12 | 17 | Qwen3-4B，ternary | Appendix F |
| 13 | 17 | Qwen3.5-2B，ternary | Appendix F |

**Table 1，p.5，caption：**

> Training Speeds for Qwen3-0.6B model on 10B tokens.

**Table 2，p.8，caption：**

> Evaluation of Qwen3-4B on Commonsense Reasoning, MMLU, and IF-Eval benchmarks. WG and HS denote WinoGrande and HellaSwag, respectively. (↑): higher is better. Green and red denote improvement and degradation relative to ParetoQ, respectively. Bold is best and underline is second-best.

**Table 3，p.8，caption：**

> Evaluation of Qwen3-0.6B on Commonsense Reasoning, MMLU, and IF-Eval benchmarks. WG and HS denote WinoGrande and HellaSwag, respectively. (↑): higher is better. Green and red denote improvement and degradation relative to ParetoQ, respectively. Bold is best and underline is second-best among QAT methods.

**Table 4，p.9，caption：**

> Evaluation of Qwen3.5-4B on Commonsense Reasoning, MMLU, and IF-Eval benchmarks. WG and HS denote WinoGrande and HellaSwag, respectively. (↑): higher is better. Green and red denote improvement and degradation relative to ParetoQ, respectively. Bold is best and underline is second-best among QAT methods.

**Table 5，p.9，caption：**

> Evaluation of Qwen3.5-2B on Commonsense Reasoning, MMLU, and IF-Eval benchmarks. WG and HS denote WinoGrande and HellaSwag, respectively. (↑): higher is better. Green and red denote improvement and degradation relative to ParetoQ, respectively. Bold is best and underline is second-best among QAT methods.

**Table 6，p.9，caption：**

> Evaluation of Qwen3.5-4B on MMMU, DocVQA, and ScienceQA. (↑): higher is better. Green and red denote improvement and degradation relative to ParetoQ, respectively. Bold is best and underline is second-best among QAT methods. *WinQ showed some training instability which caused loss of accuracy.

**Table 7，p.9，caption：**

> Evaluation of Qwen3.5-2B on MMMU, DocVQA, and ScienceQA. (↑): higher is better. Green and red denote improvement and degradation relative to ParetoQ, respectively. Bold is best and underline is second-best among QAT methods.

**Table 8，p.10，caption：**

> Ablation study of Qwen3-0.6B with varying maximum schedule steps (m_s) and number of full cycles (n) fixed to 1 (after the initial half cycle of bit schedule). Evaluation is performed on Commonsense Reasoning, MMLU, and IF-Eval benchmarks. WG and HS denote WinoGrande and HellaSwag, respectively. (↑): higher is better. Bold is best and underline is second-best among QAT methods.

**Table 9，p.10，caption：**

> Ablation study of Qwen3-0.6B with varying number of full cycles, n (after the initial half cycle of bit schedule), with maximum schedule steps (m_s) fixed to 400. Evaluation is performed on Commonsense Reasoning, MMLU, and IF-Eval benchmarks. WG and HS denote WinoGrande and HellaSwag, respectively. (↑): higher is better. Bold is best and underline is second-best among QAT methods.

**Table 10，p.14，caption：**

> Clip-then-Round (Existing SEQ) vs. Round-then-Clip (RelaxSEQ): Early Benchmarking on MMLU.

**Table 11，p.17，caption：**

> Evaluation of Gemma3-1B on Commonsense Reasoning, MMLU, and IF-Eval benchmarks. WG and HS denote WinoGrande and HellaSwag, respectively. (↑): higher is better. Green and red denote improvement and degradation relative to ParetoQ, respectively. Bold is best and underline is second-best among QAT methods. *LLM-QAT produced repeating tokens for IFEval, hence, its accuracy cannot be reported.

**Table 12，p.17，caption：**

> Evaluation of Qwen3-4B Ternary model on Commonsense Reasoning, MMLU, and IF-Eval benchmarks. WG and HS denote WinoGrande and HellaSwag, respectively. (↑): higher is better. Green and red denote improvement and degradation relative to ParetoQ, respectively. Bold is best and underline is second-best among QAT methods.

**Table 13，p.17，caption：**

> Evaluation of Qwen3.5-2B Ternary model on Commonsense Reasoning, MMLU, and IF-Eval benchmarks. WG and HS denote WinoGrande and HellaSwag, respectively. (↑): higher is better. Green and red denote improvement and degradation relative to ParetoQ, respectively. Bold is best and underline is second-best among QAT methods. *UPQ did not converge well for the ternary training for Qwen3.5-2B, which explains its loss of accuracy.

### 8.3 Algorithm 目录（共 3 个）

| Algorithm | PDF 页 | 印刷标题 | 已重建位置 |
|---|---:|---|---|
| 1 | 5 | RelaxQATCallback: At initialization | 3.4 |
| 2 | 5 | RelaxQATCallback: At each training step | 3.4 |
| 3 | 6 | RelaxSEQ for 2-bit QAT | 3.2 |

## 9. 公平性、未披露项与可复现性审校

### 9.1 已有的公平比较控制

- 同样的 10B-token budget、起始 LR=1e-4、cosine schedule；Appendix D 给模型级相同 batch/AdamW/zero decay 的设定。
- 全部实验是 weight-only、本文强调 channelwise，避免直接把 groupwise 更精细但硬件成本不同的结果混入比较。
- 所有主 QAT 方法在同一组 benchmark 下报告；保留 FP 上限参照和 PTQ AdaScale 对照。
- 系统对比为双方都写 Triton kernel，至少不像“一方 Python，一方 fused GPU”那样明显不对称。

这些是作者报告的控制，**不能据此确认所有 baseline 已按其最佳官方 recipe 忠实重现**。

### 9.2 关键未披露项

| 类别 | 缺失信息 | 为什么影响审校 |
|---|---|---|
| 训练 objective | next-token CE、teacher KD 或组合 loss、teacher model、loss weights、temperature、masking | UPQ/LLM-QAT 的原 recipe 可能依赖 distillation，本文未给如何适配 |
| 数据 | mixture ratio、dataset revision、采样、顺序、dedup、tokenizer、SFT formatting、图文比例/vision tokens 计数 | 同 10B tokens 不保证内容与难度相同，也无法核对评测污染 |
| steps / 序列 | sequence length、packing、truncation、gradient accumulation、实际 update steps、warmup | 不能把 Figure 4 的约 10k 步严格换算成所有模型的 10B tokens 或 schedule fraction |
| callback | 每模型最佳 m_s、css/ces/cl、最后目标 bitwidth 训练长度、n=0 实现、恢复状态 | 无法复现 cycle，尤其 printed off-by-one |
| baseline tuning | 各 baseline 超参、search budget、checkpoint selection、WinQ window/interpolation、UPQ PTQ budget、BitNet schedule | 统一 LR 不等于公平最优 tuning；失败 baseline 是否可修复未知 |
| 搜索目标 | 用哪个 validation split 选 m_s、是否看 test benchmark、是否每模型调参 | 若用评测分数选最佳会有 selection bias；论文不足以判定是否发生 |
| 量化模块 | embedding/lm_head/bias/norm/vision encoder/projector 是否量化或保留 FP、shared/tied weights | 无法从“2-bit model”推出全部参数都 2 bit |
| scale / STE | 初始化、positivity、gradient factor、ties、门控、dtype | 影响最终 code count、边界行为、训练稳定性 |
| 软件 / 分布式 | PyTorch/Transformers/Triton 版本、parallelism/sharding、mixed precision、optimizer precision | GPU-hour 可移植性和数值一致性不明 |
| 评测 | lm-eval harness 版本/commit、task split、prompt/chat template、generation length、temperature、seed、MMMU subset、DocVQA metric | 分数不可完整复现；尤其生成退化/模板影响 IFEval 与 DocVQA |
| 重复性 | seeds、重复次数、std/CI、显著性检验 | +0.24、+0.41 等不能认定稳定显著 |
| 导出 / 设备 | packed checkpoint、bit layout、scale dtype、zero-point、INT kernel、hardware task accuracy | 论文主要是 QDQ 训练/评测，不是交付整数部署证明 |

### 9.3 特别需要保留的失败/比较限制

- Table 6 WinQ 的 DocVQA=7.76、Avg=35.90，作者归因训练不稳；无重跑或稳定 recipe 证据。
- Table 13 UPQ ternary 的 Avg=25.97，作者说未良好收敛；不能泛化为原 UPQ 方法必然失败。
- Table 11 LLM-QAT IFEval 重复 tokens 而 score 缺失；“无法报告”并不从数学上意味着不能给 0 分，是论文采用的报告选择；审校应保留缺失，不自行补分。
- AdaScale VLM 两个模型的 DocVQA/ScienceQA 均为 0，没有诊断 generation/template/校准失败原因；据此只能说“本文配置表现差”，不能证明所有 2-bit PTQ 都不可用。
- 若比较“每列最高/次高”，必须说明只比较 QAT，不含 FP；aggregate columns 非独立 evidence。
- Figure 4 相同训练步数并不等于相同 wall-clock；Triton runtime 数据只在 Table 1 给一组。

## 10. 权重/激活粒度、整数导出与 per-op / KV 成本

### 10.1 可以确认的粒度

- 权重：**output-channelwise / per-row** scale，\(\theta[n,m]\)、\(\alpha[n]\)，不是 groupwise、per-token 或 per-tensor scale。
- 日程：各 SEQ module 被赋相同 B / b_t；不是 learned per-layer mixed precision search。
- 2-bit grid：对称 odd code \(\{-3,-1,1,3\}\)，可代数表示为 \((\alpha/4)z\)。
- 激活：**未量化**是本文实验范围的结论；其 FP16/BF16/FP32 具体 dtype **未知**。不宜把 W2A16、W2BF16 写成论文已明示配置。
- KV-cache：没有 RelaxQAT 量化 experiment；bitwidth/granularity 未给，p.10 明确留 future work。

### 10.2 整数导出可知与不可知

**[推导]** 当 Eq. (3) 最终 b=2 时，有限四个 exact code levels 可映射为 2-bit index：
\(r\in\{-2,-1,0,1\}\)、\(u=r+2\in\{0,1,2,3\}\)，且
\(z_{\rm odd}=2u-3\)、\(w=(\alpha/4)(2u-3)\)。
这是数学上可编码的说明，不是作者公开的 packing/export实现。

odd codes ±3 本身不能直接装进普通 signed INT2 的范围 [-2,1]；需 index/codebook 映射或 affine form。最终 scales 仍是实数，dequantized weights 当然仍是实数；“整数导出”应指整数 codes + scales，而不是训练完所有 weight 数值都变为整数。

**[未披露]** 训练后是否真的冻结/导出 \(u,z,\alpha\)，量化异常模块是否单独处理，端点是否 snap，ternary storage 是 entropy/block packing 还是 2-bit index，每 channel scale storage dtype，NPU 支持的 signedness/zero-point、kernel accumulation/requantization、部署误差等。因此只能确认 Eq. (3) 最终数学四级 grid，不能确认论文已经实测整数模型与 QDQ accuracy 一致。

### 10.3 per-op 成本：仅给能从 shape 得出的公式

以下为**符号推导，非论文测量**。对某 `Linear` \([n,m]\)、输入 flatten \([BL,m]\)：

- 每 forward 要在 nm 个 latent weights 上做除 scale / scaling、shift、round、clip、add、multiply（Eq. (3) / Algorithm 3）；时间复杂度 O(nm)，实际 fusion/read-write/中间 tensor 数未知。Eq. (2) 操作数级别相近，换顺序不代表 FLOPs 显著减少。
- 后续 dense Linear 的 MAC 量级是 \(BLnm\)；若采用“一次乘加计 2 FLOPs”的 convention 则约 \(2BLnm\)。本文没有证明训练 matmul 本身用了 INT2 运算。
- 理想 2-bit code 存储为 \(nm/4\) bytes，加 n 个 per-row scales；若 scale 为 \(s_\alpha\) bytes，附加 \(ns_\alpha\)，每 weight 的平均附加 bits 为 \(8s_\alpha/m\)。alignment、metadata、未量化模块另计。
- 这个理想模型权重压缩比不能用于训练状态：训练仍保留 FP latent weights、scales、gradients 和 AdamW states，QDQ tensor dtype 未披露。
- 激活 \([B,L,n]\) memory 为 \(BLn\,s_a\) bytes，\(s_a\) 未知；量化权重不会直接把这部分也降为 2 bit。

论文没有逐层 n/m、层数、attention/MLP 配置，所以无法给每个真实 op 的绝对 bytes、MAC、KV memory 或参数覆盖率。不得把外部模型配置假装来自本文。

### 10.4 KV 成本：没有本文实测，不能与参数压缩混算

对**常规显式 K/V cache 的 attention layer**，若缓存长度 T、batch B、KV heads \(H_{\rm kv}\)、head dim \(d_h\)，两份 cache 的符号容量是：

\[
M_{\mathrm{KV,layer}}=2BT H_{\rm kv}d_h\,s_{\mathrm{KV}}\ \text{bytes}.
\]

这不是本文参数表或实验值；KV 量化后还应加 scales/zero-points 等 metadata，且不同粒度对应不同开销。对不缓存标准 K/V tensor 的 attention/其他状态结构，不能照搬此公式；论文也没有给 Qwen3.5 的实际 attention/cache 结构。本文未给 \(H_{\rm kv},d_h,T,s_{\rm KV}\) 和适用层数，故不能计算模型绝对 KV 成本，也不能声称 RelaxQAT 减少 KV memory/attention bandwidth 或改善长上下文。

## 11. 内部矛盾与审校优先级

| 优先级 | 问题 | 原始证据 | 审校判定 / 下游处理 |
|---|---|---|---|
| 高 | 6.3%/7.7% 易误写为 relative gain / strongest baseline gain | Abstract pp.1–2；T3 p.8；T7 p.9 | 明写 +6.26 pp vs ParetoQ，+7.70 pp vs ParetoQ；strongest 对比为 +3.75/+2.00 pp |
| 高 | full-cycle n 的计数与分支不同 | Algo1 L8–9 / Algo2 L3、5、18，p.5 | C 有 n boundaries 但只启动 n-1 cycles；n=1 不触发，要求源码澄清 |
| 高 | ternary literal 1.58 与 exact integer endpoint 不同 | Eq.9 p.16 | k≈1.494849；要求是否 log2(3)/snap；不得断言实际代码错 |
| 高 | Eq.2 的 saturation ties 与宣称 four-code grid 不同 | Eq.2 p.4；round convention 未给 | nearest-even 正端可生成 odd +5；要求 epsilon/tie/clamp说明 |
| 高 | “smooth loss / escape saddle / faster convergence”过强 | §3.2 p.4、Prop1–2 pp.6–7 | clip端点平滑不消除内部round不连续；证明仅 fixed-scale bin geometry 与 fixed-code步间恒等式 |
| 中 | 同配置 n=1/ms=400 两份分数 | T8/T9 p.10 | Global 46.47 vs46.26，九列都不同；可能 run/timing/seed，不做无证纠正 |
| 中 | AdaScale 两个 Qwen3.5 language 型号九列相同 | T4/T5 p.9 | 疑似复制，不能判定；保留原值并请求确认 |
| 中 | “outperforms baselines across the board”有反例 | §5.3 p.9，T6 | 4B VLM Avg 71.84 < UPQ72.00；多项个任务低于baseline |
| 中 | “all tasks highest or second”应只指选定 aggregate，若指每任务则不成立 | T7 DocVQA 87.71、T6 MMMU40.33 | T7 DocVQA第四；T6 MMMU并列第三 |
| 中 | runtime -2.4% 没有统计置信度 | T1 p.5 与 “within margin of error” | 表述“训练耗时相近”，不是显著性能提升 |
| 低 | “integers”混淆 raw/shifted/odd/QDQ | §3.2 p.4、AppA p.14、Fig6 p.15 | 明确 raw r、q=r+.5、zodd=2q；图浮点极值不称整数 |
| 低 | bin数量是否等于2^b、n_levels命名 | Algo3 p.6，Eq9 p.16 | n_levels是半范围k；非整数b不是真实物理bit count；ternary起点最多9级 |
| 低 | 附录极值写 {-13,-13} | p.14 vs Fig6 p.15 | typo，解释为图中的{-13,13}，不静默改原引文 |
| 低 | Fig7 caption只写(a)/(b)，实际a–h | p.16 | 标上排a–d与下排e–h |
| 低 | Table2 “best overall”漏QAT限定 | caption/正文p.8 | FP全列更高，最佳/次佳仅QAT |
| 低 | 末步10000叙述 vs Figure7截图9900 | pp.15–16 | 前者为作者总训练长度，后者为图显示；不伪造10000截图数值 |

以上问题不推翻所有经验结果；它们限定论文可复述的精度、理论强度与部署外推范围。本次没有复现实验或检查源码，无法判断多数疑点的真实实现原因。

## 12. 审校闭环与来源定位

### 12.1 格式/可读性复核

- 全文按原正文 1–7 与附录 A–F 重建，附录延页位置明确。
- 9 个编号公式全部记录；公式 (5) 的 round-and-clip 特殊记号、指数 \(b-1\)、shift 与 clip ranges 逐个核对。
- 7 幅图、13 张表、3 个 algorithm 已全部列目录、页码和标题/caption。
- Table 11 缺失分数与 Tables 6/13 的 failure footnotes 均保留。
- 原图 caption 独立引用；本文推导/疑点和作者原述分开，不把不存在的曲线原始数据写成精确实验数值。

### 12.2 内容准确性复核

- headline 算术、strongest baseline、pp vs relative percent 已独立重算。
- channelwise shape、scale 与 code 三种表示、LSQ vs RelaxSEQ vs ternary 的差别核对完成。
- 理论固定 scale/bitwidth/未饱和内部 bin 前提、round ties、scale梯度间接依赖、没有global guarantee均明确。
- 所有实验 score/超参依原表/正文转录；小于等于约0.01 pp的显示值平均差不覆盖原表数字，不据此认定重大错误。
- 82 行（Tables 2–9、11–13）已对原始逐页文本进行逐行数值比对，所有已报告 score 一致；CSR/Global/VLM 平均重新计算，没有超过0.01 pp量级的显示值平均差。图注计数7、表注计数13。
- `per-op`、KV容量、整数index映射均标记为符号推导，非论文实测/源码证据。
- 完整 PDF 未披露的实现/预算/评测事项已单列；没有外部网页或新版本内容混入本版本结论。
- 完成后复核原 PDF 的 SHA-256，与读取前一致。

### 12.3 关键源页

- [原 PDF，p.3](file:///Users/bytedance/project/read_papers/paper/2611.27056v1.pdf#page=3)：Eq1、日程、weight/scale shapes。
- [原 PDF，p.4](file:///Users/bytedance/project/read_papers/paper/2611.27056v1.pdf#page=4)：Eq2/3、k指数、极值仿真。
- [原 PDF，p.5](file:///Users/bytedance/project/read_papers/paper/2611.27056v1.pdf#page=5)：Algo1/2、backward披露、runtime、LSQ Eq4。
- [原 PDF，pp.6–7](file:///Users/bytedance/project/read_papers/paper/2611.27056v1.pdf#page=6)：Prop1/2、Eq5–7、setup。
- [原 PDF，p.8](file:///Users/bytedance/project/read_papers/paper/2611.27056v1.pdf#page=8)：T2/3、7.7 pp headline、Fig4。
- [原 PDF，p.9](file:///Users/bytedance/project/read_papers/paper/2611.27056v1.pdf#page=9)：T4–7、6.26 pp headline、UPQ反例。
- [原 PDF，p.10](file:///Users/bytedance/project/read_papers/paper/2611.27056v1.pdf#page=10)：T8/9、future activation/KV。
- [原 PDF，pp.14–15](file:///Users/bytedance/project/read_papers/paper/2611.27056v1.pdf#page=14)：A/B/C/D、Eq8、T10、Fig5/6。
- [原 PDF，p.16](file:///Users/bytedance/project/read_papers/paper/2611.27056v1.pdf#page=16)：Eq9、ternary终点、Fig7。
- [原 PDF，p.17](file:///Users/bytedance/project/read_papers/paper/2611.27056v1.pdf#page=17)：Gemma与ternary全数值。
