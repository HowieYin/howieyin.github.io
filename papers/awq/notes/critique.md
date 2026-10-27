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
