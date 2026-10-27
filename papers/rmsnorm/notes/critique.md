## Implementation Notes：epsilon、精度与导出不能随意换

### 原论文、原作者代码与现代实现的差异

| 实现 | 分母／关键行为 | 迁移时的影响 |
|---|---|---|
| 2019 论文 | sqrt(mean(x²))；p 用 ceil(Dp) | 零向量未定义，代数讨论属理想形式 |
| 作者 PyTorch `rmsnorm_torch.py` | `sqrt(mean(x²)) + eps`，默认 eps=1e−8；p 用 `int(D*p)` | epsilon 在 sqrt 外；floor 和论文 ceil 不同；p=0 可产生空 slice |
| 作者 README TensorFlow 例子 | `sqrt(mean(x²)+eps)`，例中 eps=1e−5 | 与作者 PyTorch 默认也不严格相同 |
| Llama 官方代码 | FP32 累加，sqrt 内 eps，再 cast 并乘 gain | tiny input、BF16 rounding、scale 不变性近似程度不同 |
| PyTorch 2.11 RMSNorm docs | 默认 eps 使用 opmath 类型精度，FP16/BF16/FP32 对应 FP32 epsilon | 应显式写 eps，不凭记忆套旧版默认 |

作者代码本次核验 commit `2e726f1a3f106bb719056422f4f9b6aca03d3ce6`。PyTorch 文档本次检索的是 2.11；不同版本不能假设默认行为相同。pRMS 应显式验证 0&lt;p≤1、k≥1，避免空归约和 NaN。

下面是 **Agent-created reference**，遵循 sqrt 内 epsilon 与 FP32 累加；用于说明接口，不声称 bitwise 等同所有 kernel：

```python
def rmsnorm(x, weight, eps=1e-6):
    # x: [..., D], weight: [D], output: same shape/dtype as x
    xf = x.float()
    inv_rms = (xf.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    normalized = (xf * inv_rms).to(x.dtype)
    return normalized * weight
```

它的输出 cast 与 gain 的计算顺序必须与目标实现一致；一些融合算子在 FP32 中连 gain 一起做完再 cast，会产生不同 rounding。gain 若是 FP32 还可能触发 dtype promotion。模型导出、量化与 Native runner 需要核验实际张量 dtype，而不是只看 Python 的输入 dtype。

### 量化和端侧映射

平方、归约、rsqrt 对范围和精度较敏感。FP16 在大输入平方时可能溢出，BF16 的尾数较短；内部 FP32 累加通常更稳。INT8／INT4 权重模型不等于 Norm 的 activation statistics 也能无损整数化；特别是 pRMS 的小分母会放大误差。

gain 可在特定连接中折入后续线性权重，例如 W·diag(g)，但要同步更新量化 scale、bias 和校准；归一化的输入依赖分母不能作为常量全部折叠。若 graph 导出把 sqrt 内 epsilon 改到 sqrt 外，或者 Norm axis 从 D 变为 channel／spatial，函数会发生实质变化。

我会用真实、多样化 hidden activation 校准，记录每层 RMS、mean、outlier、near-zero 与饱和比例；对比框架 FP32、导出浮点、量化、设备执行四个阶段的 per-layer error。出现精度下降时先定位是哪一步的分母、cast 或 scale 改变，而不是把问题简单归因给小模型能力。

HTP/NPU 的内存验证用实际 `dmabuf_dump` 与 CPU `VmHWM` 分开统计；理论 4M bytes 统计量不代表设备整体物理峰值。常驻 Native runner 下测热态 TTFT/TPOT，避免工具冷启动和 RPC 干扰，并确认 prefill 与 decode 首 token 的 Norm 数值路径一致。


## My Takeaways：最值得迁移的是验证方式

**以下为本文独立判断。** 新颖性来自机制拆分而非复杂模块：把 LayerNorm 中心化与尺度约束分开，得到一个简单、容易实现、后来得到广泛采用的方法。正确性方面，核心代数扎实，但需要补齐正尺度、非零分母与 epsilon 条件；梯度分析支持局部性质，不能代替完整优化定理。

实验覆盖 RNN、self-attention、CNN、跨模态任务与多个框架，这比只测一种网络更有说服力。弱点在于缺少多 seed 质量误差、严格同优化等级 kernel 对照、训练预算的完整披露，以及不同轴／学习率等混杂因素。可复现性受公开代码帮助，但论文与代码的 epsilon、ceil/floor 差异仍需记录。

工程复杂度低，理论成本 Θ(MD)，扩展到大模型不增加 routing、通信或 KV；反过来说，它也不直接减少这些大项。gain-only 的参数节约很小，主要收益是 activation 统计与跨算子融合。硬件依赖表现在带宽、归约、布局与 launch，而不只是标量 FLOPs。

**隐藏取舍：** 保留均值方向使信息与噪声都可能留下；pRMS 的前缀估计对排列敏感；小分母可能放大异常通道；数值稳定 epsilon 会让“精确不变”变成近似。已经训练好的 LN 模型需要函数与质量验证，不能用接口相同代替语义相同。

值得继续回答的问题有三个：哪些架构真正利用均值方向，哪些可以满足 Pre-CRMS 的等价条件？在更完整的训练预算下 LN／RMS 的接近程度及 DyT 的收益反转是否复现，是否依赖数据与优化器？在目标 HTP/NPU 中最有价值的是省统计、融合 residual+Norm、融合 Norm+linear，还是避免一次 CPU fallback？

我的优先改进方向是将分布诊断与 per-op 调用链连起来：发现某层近零 RMS 或巨大 prefix bias 后，再检查量化分母、精度转换与设备回退；发现 kernel 加速后，再用 Amdahl 和真实 TTFT/TPOT 判断收益。论文 Figure 1 的“step 与 time 分开看”直到今天仍然适用。
