# 扩展研究记录 · 2026-10-27

一手来源与链接集中在完整报告 References [3]–[17]。对照包含 2019 equalization；2022 LLM.int8、ZeroQuant、GPTQ；2023 AWQ/OmniQuant；2024–2025 QuaRot、SpinQuant、FlatQuant；2026 TwinQuant、Ascend reasoning case study；NVIDIA / AMD 官方工程材料。

当前状态采用同一论文、固定协议切片，而不建立跨论文混排：TwinQuant Table 1 的 W4A4 LLaMA3-8B / Qwen3-14B，FlatQuant 原表作独立核查。LLaMA3-8B 切片 TwinQuant 比 FlatQuant accuracy 高 0.8 pp；Qwen3-14B 则 FlatQuant 更好。这里的 SmoothQuant 是 W4A4 变体，不能反驳原论文 W8A8 的实测范围。

Ascend 2026 证据：1.5B W8A8KV8 均值下降 7.60 pp，不能概括任意规模无损；pseudo-quant 与真实 kernel 明确分开；QwQ-32B 原表 Average 算术疑点记录但未据此排名。

TensorRT-LLM 1.0 INT8 SmoothQuant 支持与 latest PyTorch FP8/FP4/KV 支持分开，避免拼接不同版本/backend 的保证。AMD 官方 MI300X 集成不等价于 Qualcomm HTP 已验证加速。

检索日期是快照日期，无持续更新或全领域 SOTA 声明。
