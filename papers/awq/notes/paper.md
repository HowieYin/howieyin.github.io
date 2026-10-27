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
