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
