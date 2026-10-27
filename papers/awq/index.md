# AWQ：把重要通道的精度，变成端侧真正能用的速度

<p class="reading-meta">MLSys 2024 · BEST PAPER · arXiv 2306.00978v6 · 阅读 / 更新 2026-10-27<br>LLM · VLM · Quantization · Inference · Systems · Edge AI</p>

**AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration**

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han。机构包括 MIT、Shanghai Jiao Tong University、NVIDIA、Tsinghua University、MIT-IBM Watson AI Lab、UMass Amherst。

[版本锁定 PDF](assets/original/2306.00978v6.pdf) · [arXiv v6](https://arxiv.org/abs/2306.00978v6) · [官方代码](https://github.com/mit-han-lab/llm-awq) · [项目页](https://hanlab.mit.edu/projects/awq) · [作者 Slides](https://www.dropbox.com/scl/fi/dtnp6h6y1mnp7g036axu6/AWQ-slide.pdf?rlkey=ffgh50hxhx8dmsnjiu8kef0ou&dl=0) · [DOI](https://doi.org/10.48550/arXiv.2306.00978)

初稿发布于 2023-06-01，MLSys 2024 发表；本次精读的是 **2026-04-25 的 v6，15 页、10 张 Figure、10 张 Table、公式组 (1)–(5)，无 Appendix**。arXiv 页面标题省略了 PDF 标题里的 “On-Device”；这里按 PDF 标题记录。DOI 是 arXiv 标识，不把它当作另一个出版社 DOI。

> **来源约定。** “Paper” 指 v6 原文和原图表；“Code” 指固定提交 `d6e797a42b9ef7778de8ee2352116e0f48a78d61`；“External” 指注明日期的一手后续资料；“Interpretation” 是本文推导和审计。交互读数是可检查的小模型计算，没有在本地复现原论文的大模型精度或 GPU/ARM 吞吐。

## TL;DR：先建立全局模型

**一句话：** AWQ 用激活统计识别重要的输入通道，通过等价缩放重新分配权重量化误差，再由 TinyChat 把统一低比特权重的容量优势落实到端侧生成速度。

**一段话：** 单 token 解码通常反复读模型权重，4-bit 存储很有价值，但直接四舍五入会破坏重要通道。AWQ 先发现：按激活幅度挑选约 1% 通道、保留其权重为 FP16，可以挽救精度；最终算法随后放弃这个不规则混合格式，改为放大重要权重列、反向缩小对应激活，以较小校准集搜索缩放和裁剪。TinyChat 在 kernel 内解包和反量化，并优化 packing、Norm、QKV、KV 更新与调度，才得到论文里的实测加速。AWQ 是 weight-only PTQ，不是 INT4 全模型计算，也不量化 KV cache。

五个关键点：

1. **重要性来自激活，而非权重绝对值。** 同样的权重误差，乘上较大的输入激活会带来更大的输出误差。
2. **“保留 1% FP16”是观察实验。** 最终 AWQ 通过通道缩放保护精度，目标是硬件友好的统一 INT3/INT4 权重。
3. **等价变换精确，量化后不精确。** `WS · S⁻¹X = WX`；`Q(WS) · S⁻¹X` 的误差由缩放、group 步长、舍入和裁剪共同决定。
4. **存储 4-bit 不等于执行 INT4×FP16。** TinyChat 在计算循环中解包、转换和缩放，再执行浮点矩阵计算；这些 per-op 成本不能省略。
5. **精度和速度是两条证据链。** PPL、VQA、CIDEr、pass@k 说明任务退化；tok/s 说明特定硬件和协议下的性能。两者都不自动推广到长 prompt、多图 OCR、大 batch 或另一种导出格式。

## 1. Introduction：为什么这篇论文值得读

**What / Why。** 2023 年端侧 LLM 部署首先受模型容量约束。7B 参数只按 FP16 权重计算就约 14 GB，尚未计入 KV、激活和运行时。权重量化有两个目标：装得下，以及生成时读得少。前者可以靠文件格式实现，后者必须依赖真实 kernel。

**How。** 论文提出 AWQ 量化方法与 TinyChat 执行系统：前者针对精度，后者针对异构设备上的解包、访存和启动开销。这种方法与系统的组合是主要贡献；只拿一个 AWQ checkpoint 交给通用 PyTorch，并不能重现全部加速。

{{fig1}}

**Figure 1 怎样读。** 从左侧应用和模型能力，读到多种端侧平台，再看 TinyChat 小电脑实例。它解决的是“量化后是否可部署”的问题，而不是给出 controlled benchmark。8 GB、15 W 的 Orin Nano 展示与 Figure 10 的 64 GB Jetson Orin 测试属于不同设备配置，不能把后者的吞吐贴到前者。

**Interpretation。** 对端侧工程而言，这篇论文连接了三个层面：误差如何分配、权重如何排列、每次生成如何执行。量化校准成功只是第一步，export 和 runner 是同一条验收链路。

**Caveat。** 权重约 4× 压缩不是进程峰值内存约 4×。视觉编码器、未量化层、metadata、padding、KV、workspace 和 runtime 都可能成为剩余大头。

## 2. Related Work：先把三种问题分开

原文按模型量化、混合精度、系统部署等方向建立背景。其重要性是避免把不同精度目标的方法混作同一个 baseline。

| 路线 | 处理对象与核心动作 | 本文为什么需要它 | 比较边界 |
|---|---|---|---|
| RTN | 每个 group 直接舍入权重 | 显示低比特最简单基线的退化 | 没有 activation-aware scale 或二阶补偿 |
| GPTQ，2022 / ICLR 2023 | 用二阶近似与逐列误差补偿做 weight-only PTQ | AWQ 的主要精度对手，也可组合 | 应包含 reordering 版本；不能只选最弱 GPTQ |
| LLM.int8() 等混合精度 | 把异常部分交给高精度路径 | 解释“保留少数重要权重”为何有效 | 不规则索引、高精度旁路和融合成本影响部署 |
| SmoothQuant，2022 / ICML 2023 | 等价缩放平滑激活，支持 W8A8 | 与 AWQ 共享数学重参数化思路 | AWQ 保护 W4 权重误差；SmoothQuant 降低激活量化难度 |
| 系统 kernel / runtime | packing、反量化、融合、调度 | 容量改善能否变成速度取决于此 | 比较应同硬件、同 batch、同缓存策略、同优化水平 |

**Interpretation。** “Activation-aware” 不表示量化 activation。AWQ 使用 activation 来选 weight 的量化坐标；在 W4A16 配置里，activation 保持 16-bit 浮点。也不要把 AWQ 的均值幅度搜索，解释成 GPTQ Hessian 重建的简化名称。

## 3. AWQ：从 1% 观察到全整数权重格式

### 3.1 观察实验：重要的是输入列

设 `W ∈ R^(d_out×d_in)`，校准激活 `X ∈ R^(d_in×T)`，输出 `Y=WX ∈ R^(d_out×T)`。第 j 个输入通道对应 **W 的第 j 列**，而非一行，也不是任意散落的最大元素。PyTorch 使用 `X ∈ R^(B×T×d_in)` 与 `Y=XWᵀ`，讨论通道时必须先确认这个转置约定。

输入通道的平均绝对激活 `s_X,j = mean_t |X_j,t|` 是重要性代理。权重误差 `δW_ij` 对输出的贡献是 `δW_ij X_j,t`：大激活通道更值得保护。但最终误差还依赖相关性、抵消和后续非线性，均值不是每个样本的完备敏感度。

{{fig2}}

**Figure 2 解决的问题。** 从左到右读 RTN、按激活选择的少量 FP16 通道、最终 AWQ。图中的突出条带表示通道级保护；最后一栏用缩放取代混合精度旁路。因此“AWQ 把 1% 权重保留为 FP16”读错了方法阶段。

**来源异常。** 该原图保留 OPT-6.7B、INT3-g128 的 `43.2 → 13.0` 标签；同版本 Table 1/3 的相近名义设置却是 `23.54 → 11.39`。原文未提供足够 provenance 来统一这组数值。本文保留原图，量化比较以明确表格为准，不拼接出一个不存在的实验。

{{table1}}

**Table 1 怎样读。** 固定 INT3、group=128，WikiText PPL ↓。列分别保留按 activation、按 W、随机选出的 0.1%/1%/3% 通道。OPT-6.7B 从 RTN 23.54 降到 activation 1% 的 11.39，FP16 为 10.86；单按 W 选 1% 得到 22.37，几乎没有恢复。OPT-1.3B 的 RTN=119.00，按 activation 1% 后 16.91，仍高于 FP16 14.62。

**证据支持什么。** 低比特失真很不均匀，且 activation 统计能比权重幅度或随机选择更好地分配高精度预算。

**成本与边界。** 这里真实增加 FP16 旁路存储及调度成本，是“方法动机”，不是最终格式。选择更高比例不保证线性收益；PPL 改善也不能换算成准确率恢复百分比。

### 3.2 核心变换：扩大重要权重，抵消输入缩放

令 `s ∈ R^(d_in)` 为正数，`S=diag(s) ∈ R^(d_in×d_in)`：

{{equivalent}}

每一列权重乘 `s_j`，每一个输入通道除以相同 `s_j`。乘法前后一致，但 group 量化的格点随后发生变化。AWQ 争取让重要通道在原坐标里的量化间隔变小。

#### 公式 (1)：先读清简化量化模型

{{equation1}}

`w` 表示 group 中的权重，N 是 bit 数，Δ 是该 group 的步长，`Round` 取最近整数。`max(|w|)` 对 group 求最大值；在 4-bit 示例中分母为 8。

**数学直觉。** 舍入误差约在 ±Δ/2 内；乘以 x 后，输出误差幅度按 |x| 放大。若假设格点残差均匀分布，绝对残差均值约 0.25，带符号均值为 0、平方均值为 1/12。原文的 0.25 不能当作“每次误差都是正的 0.25”。

**工程含义。** 这是解释公式，并不是完整导出规范：没有写明整数饱和、signed range、zero-point 和 padding。实际官方 quantizer 使用 affine min/max，后文单独解析。用解释公式生成文件却交给另一种 affine kernel 会造成系统性精度损失。

#### 公式 (2)–(3)：缩放在哪个条件下减少误差

{{equation2}}

{{equation3}}

`Δ′` 是缩放后的 group 步长。重要通道在原坐标的步长是 `Δ′/s`；在舍入残差统计相近时，其误差比例约为 `Δ′/(Δs)`。这依赖 **Δ′ 没有随 s 大幅增大**。如果被放大的权重成为新的 group 最大值，它会拖动所有通道格点，其他通道可能更差。

**Shape 与复杂度。** `WS` 和 `S⁻¹X` 不需要真的构造 `d_in²` 对角矩阵：广播列缩放花 `O(d_out d_in)`，激活缩放花 `O(T d_in)`。其后仍是 `O(T d_out d_in)` 的 dense linear。

{{scale-widget}}

**交互怎样读。** 四个权重共享一个 group；可调重要通道的 s、x 和另一通道的最大权重。图展示量化重建权重的误差与激活相乘后的输出误差。固定 x=8、末项权重 −3.8 时，s=1 的输出为 −0.1425、参考为 0.92，绝对误差 1.0625；s=4 后输出 0.8075、误差 0.1125。舍入有跳变，曲线不必单调。组件遵循公式 (1) 的简化格点，无裁剪和整数饱和，不宣称复现官方 exporter。

{{table2}}

**Table 2 是关键 trade-off。** OPT-6.7B 中，将 1% salient channels 的 s 从 1 提到 2，PPL 从 23.54 到 11.92；s=4 却退回 12.36。与此同时，salient error proxy `Δ′/(Δs)` 仍从 0.519 降到 0.303。不是“保护得越强越好”，而是其他通道的步长被牵连。

**数值核对。** s=2 时 8.2% 的 group 步长改变，平均 Δ′/Δ=1.038；s=4 时改变比例升为 21.2%，平均比率=1.213。最优 PPL 与最小 salient proxy 不一致，正好解释下一步为什么要搜索整层输出误差。

#### 公式 (4)–(5)：用小校准集搜索，不能说完全不比较输出

{{equation4}}

{{equation5}}

`s_X ∈ R^(d_in)` 是平均绝对输入激活，α 是一个 scalar。将高维通道尺度约束在 `s=s_X^α` 曲线上，α=0 对应不做通道差异缩放，较大的 α 更积极地保护激活大的通道。`L(s)` 比较量化后输出与原 `WX`；它是局部代理，不是训练任务损失。

**Code。** 当前 `auto_scale.py` 实际试验 `α=0, 0.05, …, 0.95` 共 20 个点，未包含 1.0；对 s 做 `clamp(min=1e−4)`，再除以 `sqrt(max(s)min(s))`。每个候选会 fake quantize 并 forward，使用 FP32 mean squared output difference 评分。原文所说“不依赖 reconstruction / 只需要平均激活”应理解为没有回归拟合权重或反向传播训练；**不等于没有校准输入、没有 trial forward 或输出 MSE**。

**Clipping。** 当前 `auto_clip.py` 按 group 搜索 `1,0.95,…,0.55` 倍 absmax，按采样激活的 group 输出 MSE 选择阈值。默认最多采样 512 tokens；对 Q/K/Wqkv 名称跳过裁剪，原因是 QK 交互不能被孤立 linear 的局部 MSE 充分描述。不能对所有层机械套一个最优 clip 值。

{{table3}}

**Table 3：最终 AWQ 和 1% FP16 分开比较。** INT3-g128 中，OPT-6.7B 的 RTN 23.54 → AWQ 11.39，绝对减少 12.15、相对减少 51.61%；FP16 10.86，仍有 0.53 PPL gap。OPT-13B 的 1% FP16 是 10.43，AWQ 为 10.56，说明 comparable 并不等于每行更优。

**消融审计。** Table 2 拆出了一个固定 salient scaling 的效果，Table 3 比较了固定 s、mixed precision 与完整 AWQ。但原文未给出严格控制相同设置的 scale-only / clip-only / both 分解。v6 §3.2 末尾把消融指向 “Table 5”，而 Table 5 已是 Mistral/Mixtral 结果；本文不把这个编号错误当作缺失消融的证据。

### 方法重建：校准、折叠、导出三步各做了什么

{{pipeline-diagram}}

| 阶段 | Input → Output | 估计 / 更新的量 | 必须检查的边界 |
|---|---|---|---|
| 校准捕获 | 浮点模型、token sequences → 每层 linear 输入与原输出 | `s_X`、trial MSE、group clip | 数据应代表实际分布；hook 与缓存非零成本 |
| 搜索 | W、X → s、clip thresholds | α grid 与裁剪阈值 | 不更新训练 optimizer，没有新训练 loss |
| 等价折叠 | 相邻浮点算子 → 重参数化模型 | `WS` 与对应逆变换 | Norm、bias、残差与非线性必须同步处理 |
| Fake quant | 浮点 W → 落在量化格点的浮点 Ŵ | 重建权重 | 仍是 float tensor，容量并未按 4-bit 降低 |
| Real export | Ŵ / s / clip → packed integer、scale、zero metadata | 确定 signedness、layout、group | 不能跨格式直接复用 qweight |
| Runtime | 输入 / cache → 输出 / 更新 cache | 解包与 fused dequant、浮点 matmul | 选择 GEMV/GEMM、padding、累加精度、kernel |

**校准不是训练。** 当前 `pre_quant.run_awq` 用 hook 把 linear 输入缓存到 CPU，一次将一个 decoder block 搬到 GPU。默认 calibration 为 512 段、每段 512 tokens；Figure 8 的实验 sequence length 是 2048，不能用当前代码默认值反推所有论文实验。下一层的 `inps` 在本层施加 scale/clip **之前** forward 得到，属于按原模型轨迹逐层校准，并不保证完全匹配量化后上游分布。

**折叠必须尊重计算图。** Norm 的 gamma/bias 除以 s，Q/K/V 输入列乘 s，可以吸收额外激活除法。SwiGLU 下通常在 **up projection** 输出行和 bias 除以 s，再让 down projection 的输入列乘 s；不能把缩放随意穿过 SiLU/GELU，也不能漏掉共享残差路径。

**GQA 细节。** 当前实现仅在 v_proj 和 o_proj 权重 shape 匹配时尝试 V→O 折叠，否则跳过。GQA 的 KV heads 少于 query heads，V 的通道不能按 MHA 同尺寸假设映射；需要广播关系与 scale 一致。这里与 [GQA 精读](../gqa/) 和 [MQA 精读](../mqa/) 的 tensor / KV 账本直接衔接。

**计算与内存。** 一层 20 次 scale 试验大约要 `20×O(T d_in d_out)` 输出评估与 `20×O(d_in d_out)` 量化遍历；如果评分模块包含 attention，还会多出其成本。Clipping 另有约 10 轮 group 部分 dot product。它不构造 GPTQ 的 dense `d_in×d_in` Hessian，但缓存 `T×d_in` 激活、权重副本和 CPU↔GPU 搬运仍可能显著。参数规模大不代表校准峰值自动等于最终 INT4 模型大小。

## 4. TinyChat：为什么 W4 存储不自动产生 4× 加速

### 4.1 端侧解码的瓶颈：权重流量

{{fig3}}

**Figure 3 读图顺序。** 左图比较一个 200-token prompt 的 context 阶段与生成 20 tokens 的总耗时；中图放到 RTX 4090 roofline；右图分解 attention 和 FFN 的权重/激活访问。左图 10 ms 和 310 ms 对应不同 token 数，不能直接解释为“每个 decode token 慢 31 倍”。

**Roofline 重建。** 论文采用约 165 TFLOPS 与 1 TB/s，拐点是 165 FLOPs/byte。batch=1 GEMV 的理想权重算术强度近似 `2 FLOPs / 2 bytes = 1`；若用 INT4 读权重、忽略 metadata/KV/反量化，则约为 `2 / 0.5 = 4`。两个值仍远低于拐点，因此减少权重读取有意义。

**流量不等于峰值占用。** 右图 attention 的 134 MB weight vs 1.7 MB activation、FFN 的 271 MB vs 0.2 MB 是该分析设置的访问量；不能把它当成整个进程、长上下文 KV 或 VLM 图像预填充的内存图。

**Interpretation。** 更大 batch 会复用同一份权重，多 token prefill 则从 GEMV 转为更高算术强度的 GEMM。AWQ 的收益可能随之受计算效率、反量化和激活流量限制。长上下文还会让 KV 成为 decode 的另一条流量路径。

### 4.2 On-the-fly dequantization、SIMD packing 与融合

在论文测试的硬件路径上，没有可直接利用的 INT4×FP16 指令：packed qweight 被读入，在寄存器中解包、转换到 FP16、乘 scale / 处理 zero，再参与矩阵计算。融合避免将完整反量化权重写回 DRAM，也避免随后再次读取。

{{runtime-diagram}}

**per-op 账本。** 一次反量化至少涉及 shift/AND、整数到浮点转换、scale/zero 运算，之后才是 FMA/MMA。要同时看 load、位运算、convert、寄存器占用、spill、启动次数与访存。论文的“三条 SIMD 指令”是解包 32 个 nibble，不能理解成“三条指令完成整个 32-weight matmul”。

{{fig4}}

**Figure 4 逐模块解析。** 一个 128-bit NEON register 有 16 个 byte，容纳 32 个 INT4。离线将 `w_i` 放低 nibble、`w_(i+16)` 放同一 byte 的高 nibble。运行时低半用 `Pw & 0x0f`，高半用 `(Pw >> 4) & 0x0f`，一次得到两组连续权重，省去更复杂的 scalar 交错。

右图四种矩阵 shape 的 **Original weights → Packed weights** latency 为 `248→215`、`472→399`、`489→400`、`1172→954 μs`，对应约 `1.153×、1.183×、1.223×、1.229×`。packing 帮助消除解包成本，却不是额外降低量化 bit 数。这些是原图内的局部实测，不是任意 ARM 设备的速度保证。

{{packing-widget}}

**GPU layout 另有约定。** 原文 GPU 采用每 8 个权重按 `[0,2,4,6,1,3,5,7]` 的布局，ARM NEON 示意不能直接当作 CUDA exporter 的排列。组件只演示原图 ARM unsigned nibble 的打包/还原，zero-point 与 FP16 重建另行处理。

**融合还做了什么。** Norm 的多个算子收口为一条 kernel；QKV projection 合并，并结合位置编码；预分配 KV 并在 attention kernel 内更新。对 Falcon 等原 HF 路径，cache 行为与 launch overhead 会影响 baseline。论文指出某些 FP16 kernel 计算约 0.01 ms，与 launch overhead 同量级，所以“减少 kernel 数量”本身就能提高速度。

**Code 的运行分支。** 检查的 `WQLinear.forward` 按 flattened input token count `<8` 走 GEMV，否则走 GEMM。它是该提交的调度规则，不是算法定理，也不代表所有版本在 M=8 一定出现相同性能拐点。

### 存储规格：4.125、4.15625 与 4.25 bit/weight 为什么同时成立

官方 affine quantizer 采用 `Δ=(w_max−w_min)/(2^N−1)`、clamp 后的 `z=−round(w_min/Δ)`，随后 `q=clamp(round(w/Δ)+z)`，重建 `ŵ=Δ(q−z)`。Δ 下限、q/z 的整数范围和排列必须与 kernel 同步。

| INT4 group metadata 方案 | 每 group 附加 bits | group=128 的理想 bpw | 说明 |
|---|---:|---:|---|
| 对称：一个 FP16 scale | 16 | 4.125 | 不含零点 |
| 常见 affine：FP16 scale + packed INT4 zero | 20 | 4.15625 | zero 也按 nibble 存 |
| 本次官方 WQLinear：FP16 scale + FP16 scaled_zeros | 32 | 4.25 | zero 已折算为浮点项；不等同 packed zero |

当前官方 `qweight` 使用 INT16 容器、shape `[d_out/4,d_in]`，scale / scaled_zeros 是 `[padded_groups,d_out]` 的 FP16/BF16；容器 16-bit 不表示每个原权重用了 16-bit。`calculate_zeros_width` 会对 group 数做对齐，较小 group 还有额外规则，因此以上值未计 padding、embedding、head 或 FP16 层。

{{capacity-widget}}

**默认容量示例。** 假设 7.0B 个权重全量量化，group=128、官方双 FP16 metadata，理想权重 3.71875 GB，即约 3.463 GiB；FP16 权重是 14 GB，即 13.039 GiB。32 层、4096 context、8 个 KV heads、head_dim=128、FP16 KV、batch=1 时 cache 是 0.5 GiB；若改成 32 个 KV heads 则为 2 GiB。AWQ 降权重，GQA/MQA 降 KV，两者互补。

**KV 公式。** 逻辑 cache bytes=`2 L B T H_kv d_h b_cache`，2 对应 K 和 V，`b_cache` 是每 scalar 的 bytes。不包括 block padding、beam copy、paged allocator、临时工作区和视觉激活。NVFP4/INT8 cache 支持不能由这个容量计算器推断，组件固定 FP16 KV；没有把容量估算变成 latency 预测。

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

## 8. External Research：相关工作矩阵与技术演进

以下是后续一手资料，不能回写成 2023/2024 年 AWQ 原论文贡献。

| 工作 | 年份 | Core idea | 优点 | 限制 / 与 AWQ 的关系 |
|---|---:|---|---|---|
| [GPTQ](https://arxiv.org/abs/2210.17323v2) | 2022 / 2023 | 二阶代理和逐列误差补偿 | 低 bit weight-only 精度强 | 额外校准/Hessian 成本；可在 AWQ 坐标中继续做 |
| [SmoothQuant](https://arxiv.org/abs/2211.10438v7) | 2022 / 2023 | 缩放把激活异常迁到权重 | 对 W8A8 高效 kernel 友好 | 目标不同；缩放不等于相同量化精度路线 |
| [AWQ](https://arxiv.org/abs/2306.00978v6) | 2023 / 2024 | 激活感知 scale + group clip | 无反向训练，W4A16 易部署 | 不降 activation/KV bit，依赖 exporter 与 kernel |
| [QuaRot](https://arxiv.org/abs/2404.00456v2) | 2024 | 正交旋转分散 outliers | W4A4KV4 路线 | 与 weight-only 的容量/算子不同，不能只比 PPL |
| [QServe](https://arxiv.org/abs/2405.04532v3) | 2024 / 2025 | W4A8KV4 + progressive quant + serving | 优化大 batch 吞吐与反量化开销 | 服务场景、设备和精度不同于原 TinyChat |
| [SpinQuant](https://arxiv.org/abs/2405.16406v4) | 2024 / 2025 | 学习旋转矩阵 | 降 W4A4KV4 的精度损失 | 有旋转优化成本，不能继承 AWQ 的无训练叙述 |
| [Marlin](https://github.com/IST-DASLab/marlin) | 2024 | 高效 INT4×FP16 kernel、调度和复用 | 小到中 batch 仍有较高速度收益 | kernel 非量化搜索算法，硬件和格式有限定 |
| [TinyChat 2.0](https://hanlab.mit.edu/blog/tinychat20) | 2024-12 | prefill / FlashAttention / 历史复用 | 进一步改善 TTFT，VLM 异构精度 | 历史重算 baseline 与现代 prefix-cache 系统不可直接比 |
| [AAAC](https://arxiv.org/html/2605.08692v1) | 2026-05 | activation-weighted 的 affine codebooks | 可作为 AWQ 的后续编码调整 | preprint，额外编码/metadata；没有端到端速度证明 |
| [Schur Replay](https://arxiv.org/html/2609.36654v1) | 2026-09 | Schur 代理与 sequential replay | NVFP4 W4A4 的硬件可导出搜索 | 不同精度/模型；量化求解速度不是推理速度 |
| [Humming](https://vllm.ai/blog/2026-10-21-humming) | 2026-10 | 统一 quant 描述、repacking、JIT dispatch | 改善不同量化格式的 kernel 支持 | 当前官方 microbenchmark，不能统一宣称全场景第一 |

{{timeline-diagram}}

**Before。** RTN、混合精度、GPTQ 与 SmoothQuant 分别解决简单量化、异常通道、二阶误差和 activation 量化。

**This paper。** AWQ 把 activation 统计用于 weight-only 误差分配，并让规则化存储布局与端侧 kernel 协同。

**After。** 旋转路线开始把 W/A/KV 都压到低比特；Marlin 等系统改进扩大 W4A16 kernel 的有效 batch 范围；QServe 针对 serving；TinyChat 2.0 处理更完整的 prefill/历史链路。

**Current state。** 2026 年的竞争不只是“AWQ 还是 GPTQ”：量化误差模型、编码表示和 kernel dispatch 都在变化。它们是可组合但受格式约束的不同层，不是一条所有节点都全面取代前一个的线性排行榜。

## 9. Current State：截至 2026-10-27，SOTA 应怎样判断

本节资料检索截至 **2026-10-27**。结果为作者/项目方报告，本文未独立复现；不把不同 checkpoint、校准集、bit/group 和硬件混作严格排名。

### 精度路线：W4A16 仍在进化，但没有跨任务唯一赢家

**External · AAAC，2026-05。** 该预印本用 activation-weighted codebooks 改造 affine quantization，H100 上校准 4 段 C4×2048，覆盖至 32B 模型。其 Table 2、group=128 下，Llama-3.1-8B C4 PPL：FP16 6.24、RTN 6.82、AWQ 6.65、AAAC S=128 6.68、S=16 6.56。S=128 的零额外选择位方案并不胜过 AWQ；S=16 变好但引入约 0.0625 bpw 选择 metadata。Qwen2.5-7B 同样 AWQ 7.09 优于 S=128 7.13。

其 AWQ+AAAC 表报告平均恢复 70.5% 的 gap，但混有不同 group/model；个别 Qwen3.5-9B 组合 PPL 从 AAAC 8.56 变为 8.67。它支持编码与 scale 可组合，不支持每个模型的组合都更优，也没有表中可验证的推理吞吐。

**External · Schur Replay，2026-09。** 对 NVFP4 W4A4 的 Qwen3.5-397B-A17B、Llama3.3-70B，七项 benchmark 的 question-weighted recovery 分别报告 99.35% / 100.84%，对比 AWQ+GPTQ 的 97.69% / 98.85%。100.84% 是相对 baseline 的加权恢复，不是“准确率超过 100%”，也不证明统计优越。

其 15.17× / 23.14× 是对 ModelOpt / LLM Compressor 的 **每层量化求解**速度，full-model 时间部分是推算，不是推理加速。这种 W4A4、group=16、硬件 scale 格式与 AWQ W4A16-g128 无法直接作精度榜单。

### Kernel 路线：原 AWQ 速度图已不能代表今天

**External · Marlin。** 官方 README 在 A10、group=128 的 W4A16 条件下讨论接近理想上限的 linear kernel 收益：对称格式 4.125 bpw 对 FP16 的容量比约 3.879×；在 M≈16–32 仍保持较好收益。原仓库面向 Ampere/Ada，Hopper 优化和当前 vLLM 变体需单独核验。一个 linear microbenchmark 不等于整个生成 pipeline 的 3.87×。

**External · Humming，2026-10-21。** Ant Group/vLLM 的官方更新在 Hopper/Blackwell、M=1…8192 的 2 次幂网格、cold L2 / warm kernels、dense 与 MoE 上比较 kernel。其描述是 W4A16 在多数测试 case 匹配或超过 vLLM Marlin；没有足够一致的原始端到端数据可以给出“所有模型、所有 batch 的新第一”。项目开源，核心意义是把格式描述、离线 repacking 与硬件 dispatch 收口。

**External · QServe。** W4A8KV4 面向 serving，其 Llama3-8B 在 A100/L40S 相对 TensorRT-LLM 约 1.2×/1.4×，Qwen1.5-72B 约 2.4×/3.5×，属于不同 batch/精度/系统基线，不能作为原 AWQ Figure 9 的直接更新数值。

### VLM 和工具生态：TTFT 与格式维护必须另查

**TinyChat 2.0 官方博客，2024-12-12。** 报告相对旧版 1.5–1.7× TTFT 改善，涉及 FP16 accumulation、FlashAttention 和 chunk history reuse。其多轮对照包含历史重算，不能用于击败已开启 prefix cache 的现代系统。描述中的 NVILA 工作负载 vision tower 可占 prefill 的 90% 以上；该比例依赖 token 压缩和模型配置，不能泛化到所有 VLM。W8A8 vision + W4A16 language 是异构选择。

**维护状态。** [Transformers AWQ 文档](https://huggingface.co/docs/transformers/quantization/awq) 中的 bits/group/zero_point/version 必须与 checkpoint 匹配；[AutoAWQ](https://github.com/casper-hansen/AutoAWQ) 已明确 deprecated，最后测试版本是 Torch 2.6 / Transformers 4.51.3。当前 vLLM 文档推荐迁移到 [LLM Compressor AWQ 示例](https://github.com/vllm-project/llm-compressor/tree/main/examples/awq)。安装旧 AutoAWQ 不应作为 2026 年无条件推荐的默认步骤。

| 想判定的 “最好” | Benchmark / metric | 必须固定的比较条件 | 当前证据的可比性 |
|---|---|---|---|
| W4A16 精度 | 同 checkpoint 的 PPL 与真实任务退化 | calibration、group、zero、excluded layers | AAAC/AWQ 局部表可比，跨表不可统一 |
| W4A4 精度 | 七项 task 的 recovery / 原 accuracy | FP4 format、activation/KV、group、规模 | Schur Replay 与自己的 baselines 可比，非 AWQ 原配置 |
| 小 batch kernel | μs、带宽、M/N/K | 同 GPU、同 cold/warm、相同 metadata | Humming/Marlin microbench，非全链路性能榜 |
| Serving | requests/s、output tok/s、SLO | batch、并发、context、cache、KV bit | QServe 自己的部署对照，不可借给端侧 batch=1 |
| VLM 体验 | cold/warm TTFT、TPOT P50/P95 | image encoder、图数、分辨率、prefix reuse | TinyChat 2.0 部分链路，产品仍须端到端实测 |

## 10. Critique：新意、优势与隐藏成本

**Novelty。** 等价 scaling 不是 AWQ 首创；重要贡献是将 activation-aware saliency 用于低比特权重误差分配，避免不规则 mixed precision，并配合 TinyChat 落地。这个描述比“首次使用 activation 做量化”更准确。

**Correctness。** 重参数化在浮点实数代数中成立，误差降低依赖 group 步长变化和舍入；不能推广成逐通道无条件定理。α grid、裁剪和计算图折叠负责把直觉落实为可用模型。

**Experiment quality / Generalization。** 模型和任务覆盖广，包含 instruction、VLM、代码和数学，值得肯定。缺少严格 scale/clip 分解、置信区间、多种视觉校准对照以及统一耗时/峰值测量；Figure 8 证明的是有限分布更稳健。

**Engineering complexity。** 数学很小，导出/执行细节很多：输入列方向、Norm/bias、GQA、SwiGLU、group zero、container packing、layout、GEMV/GEMM dispatch 和 accumulator。错误通常沿转换链路产生，不能仅凭最终回答差就归咎于小模型能力。

**Scalability / Compute cost。** 不使用 Hessian 与训练 optimizer 降低搜索复杂度；但大模型仍需要逐层 trial forward 和 activation 缓存。MoE 的存储按所有 expert 计，不按 top-k active parameters 计。

**Memory cost。** 4-bit 权重省容量和带宽；scale/zero/padding、FP16 层、KV 和 vision 成为剩余成本。weight-only 不压缩 KV，也不压缩 activation 的 peak。

**Hardware dependency。** SIMD/GPU 布局需匹配实际指令、cache、寄存器与 warp/thread 调度。仅做“文件缩小”不能保证 TPOT 降低；大 batch 可能让混合反量化成本更突出。

**Reproducibility。** 公开代码、slides 和原图表是优势；固定代码版本及不同证据数值是必要条件。原论文的系统支持范围不能当作今天生态状态。

**Hidden trade-offs。** s 大会保护重要列但牵动非重要列；g 小通常降低 group 误差但增加 metadata；高压缩率可使大模型装入设备却留下很低 tok/s；clip 降局部 MSE 可能裁掉少量任务关键 outliers。

**Unanswered questions。** 动态多模态分布是否需要分场景 scale？按 layer/通道预算的保护能否同时约束输出误差与寄存器成本？怎样在校准阶段把 TTFT/TPOT 的真实 kernel代价纳入选择？这些是后续研究方向，不是 v6 已给出的答案。

## 11. My Insights：面向端侧 VLM/OCR 的执行建议

以下是 **Interpretation / proposed work**，与论文结果分开。

### 先证明转换链路正确，再决定敏感层

为每个 linear 比较四个节点：原 FP16、scale-fold 后 FP16、fake quant、真实 export→kernel 解包重建。第一步应近似等价；第二步引入预期量化误差；若第三步显著多损失，优先排查 zero_point、signedness、pack order、transpose、group 和 bias。

再捕获逐层输出的 max/mean absolute error、relative error、MSE 和 logits 差异；将第一次异常对齐到具体 op。不要在导出还有问题时讨论“模型本身太小”。Prefill/decode 应使用一致的首 token logits 路径，避免不同 kernel 或累加方式引入难以解释的漂移。

### 校准集覆盖真实图像输入

论文的文本校准在部分 VLM 上有效，是存在性证据。对于多图理解与 OCR，应包含真实 vision encoder 输出、不同分辨率、图数、文字密度、语言、数字、长文和难例；calibration 与 evaluation 隔离。

评估按业务输出拆解：OCR 可以用 exact match、CER/WER、数字/字段正确率；结构化抽取可报告 TP、FP、TN、FN、Accuracy、Precision、Recall、F1，并说明 TN 的样本空间；开放式 caption 则采用合适指标，不能伪造一个可直接计算的 confusion matrix。每项都对齐 FP16 和量化版本的相同输入、prompt、seed、decoding。

### 把 per-op 与整机体验同时记录

用原生常驻 runner 将模型加载与业务推理分开，避免测试工具的冷启动/RPC 成本覆盖 kernel 效果。记录 load、vision encode、prefill、first logits、decode，以及 unpack/convert/scale/matmul/cache 更新的耗时与流量；观察 launch、register pressure 和 spill。

完整指标包括 cold/warm TTFT、TPOT P50/P95、持续 tok/s、上下文增长、图像数量、功耗与温控。端侧异构部署若使用 HTP/dma-buf，可结合 `dmabuf_dump` 与 CPU `VmHWM` 检查物理峰值，不能只拿模型文件或逻辑 KV 当内存峰值。

### 选择方案时同时验收格式和 kernel

对本设备分别测试 AWQ/GPTQ/FP4 的可用 exporter 与 runner，固定 metadata、group 和未量化层。可尝试保留敏感层、调整 group/clip 或 activation-weighted codebook；每次只改一个因素。W4A16、W4A8KV4、W4A4KV4 的选择还要结合 prefill 与 decode 比例，不能只按“4-bit”命名。

## 12. Implementation Notes：可追踪的量化与推理流程

下面伪代码采用论文的列向量约定。它用于重建机制，不是可以跨模型直接执行的替代官方脚本。

```python
# W: [d_out, d_in]; X: [d_in, T]; Y_ref: [d_out, T]
Y_ref = W @ X
s_x = mean(abs(X), axis=1)
best = None
for k in range(20):
    alpha = k / 20
    s = maximum(s_x ** alpha, 1e-4)
    s /= sqrt(max(s) * min(s))
    W_scaled = W * s[None, :]
    W_fake = group_affine_fake_quant(W_scaled, bits=4, group=128)
    Y_trial = W_fake @ (X / s[:, None])
    score = mean((Y_trial.astype(fp32) - Y_ref.astype(fp32)) ** 2)
    best = choose_lower_mse(best, score, s)

# Fold only along supported graph patterns; include bias and all consumers.
fold_inverse_scale_into_previous_op(best.s)
W *= best.s[None, :]
W = activation_weighted_group_clip(W, cached_X)  # eligible layers only
qweight, scale, zero_metadata = real_export(W, target_kernel_format)
verify_export_unpack_roundtrip(qweight, scale, zero_metadata)
```

**输入/输出。** 搜索得到每输入通道 scale，clip 得到每 group 阈值，export 才得到压缩 tensor；runtime 接收 packed layout 与 metadata。实际模块评分、clip 输入与 fold 模式比此 linear 示例复杂，应以固定源码为准。

**固定源码入口。**

- [pre_quant.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/pre_quant.py)：逐层 hook、输入缓存与校准顺序。
- [auto_scale.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py)：α 搜索、输出 MSE、Norm/linear/activation 折叠和 shape guards。
- [auto_clip.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_clip.py)：group clip 搜索与 Q/K 例外。
- [quantizer.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/quantizer.py)：affine quantize、fake / real 的分界。
- [qmodule.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/qmodule.py)：qweight/scale/scaled_zeros、padding 与 forward dispatch。
- [TinyChat](https://github.com/mit-han-lab/llm-awq/tree/d6e797a42b9ef7778de8ee2352116e0f48a78d61/tinychat)：执行与硬件 kernel，不把其格式默认为所有 AWQ 工具通用。

**验收顺序。** 先检查 checkpoint quantization_config 与实际 tensor，再跑一层 export/unpack，随后 whole-model logits，再做真实任务精度，最后测常驻 warm/cold 性能。某种格式“加载成功”不是数学重建正确的证明。

## 13. References、关联精读与审阅记录

**Original paper。** [AWQ v6 abstract](https://arxiv.org/abs/2306.00978v6)、[PDF](https://arxiv.org/pdf/2306.00978v6)、[原始 HTML](https://arxiv.org/html/2306.00978v6)。全部 10 Figures 与 10 Tables 按原编号保留，表格可选择文字、横向滚动，并附完整原 PDF 页面证据。

**Official resources。** [AWQ project](https://hanlab.mit.edu/projects/awq)、[llm-awq](https://github.com/mit-han-lab/llm-awq)、[author slides](https://www.dropbox.com/scl/fi/dtnp6h6y1mnp7g036axu6/AWQ-slide.pdf?rlkey=ffgh50hxhx8dmsnjiu8kef0ou&dl=0)、[TinyChat 2.0](https://hanlab.mit.edu/blog/tinychat20)。

**Related papers。** [GPTQ](https://arxiv.org/abs/2210.17323v2)、[SmoothQuant](https://arxiv.org/abs/2211.10438v7)、[QuaRot](https://arxiv.org/abs/2404.00456v2)、[SpinQuant](https://arxiv.org/abs/2405.16406v4)、[QServe](https://arxiv.org/abs/2405.04532v3)、[Marlin paper](https://arxiv.org/abs/2408.11743)、[AAAC](https://arxiv.org/html/2605.08692v1)、[Schur Replay](https://arxiv.org/html/2609.36654v1)。

**Current implementation sources。** [Marlin code](https://github.com/IST-DASLab/marlin)、[QServe code](https://github.com/mit-han-lab/omniserve)、[Humming announcement](https://vllm.ai/blog/2026-10-21-humming)、[Humming code](https://github.com/vllm-project/humming)、[Transformers AWQ](https://huggingface.co/docs/transformers/quantization/awq)、[AutoAWQ deprecation](https://github.com/casper-hansen/AutoAWQ)、[LLM Compressor AWQ](https://github.com/vllm-project/llm-compressor/tree/main/examples/awq)。完整资料/证据映射见 [source register](sources/source-register.md)。

**Related Readings。** [MQA：多个读头，共用一份历史](../mqa/) · [GQA：让 Query 保留多样性，让 KV 共享历史](../gqa/) · [FlashAttention：算得更多，为什么反而更快？](../flashattention/)。AWQ 处理权重，MQA/GQA 处理 KV，FlashAttention 处理 attention IO；三者解决同一推理链路上的不同成本。

**Review / changelog · 2026-10-27。** 完成第一轮来源准确性审阅：章节、所有原图表、5 组公式、数值方向、bit/group、源码格式、后续资料日期与 claim 边界；完成第二轮可读性/渲染审阅：原图 caption/放大、表格完整性、公式原生 MathML、交互与桌面/移动端。详见 [review record](notes/review.md)。初次发布；后续研究数值均是引用的一手报告，未声称本地重现实验。
