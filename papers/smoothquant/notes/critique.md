# 独立评价记录

贡献：识别固定异常通道与 INT8 GEMM 归约尺度矛盾，通过离线可融合的等价变换同时改善 activation / weight 量化，实现规则 W8A8；100B–530B 精度与真实 backend 证据互相支撑。

限制：absmax 与全局 α 不直接最小化任务损失；静态校准存在分布漂移；细粒度 O1 较稳但有 reduction / launch 成本；Norm/Softmax/缓存/通信/packing 不自动加速；新模型主要 PPL；decode timing 与部分内存定义不足；没有充分长推理、OCR/VLM、MoE routing 可靠性实验。

可检验方向：输出误差驱动逐层 α；多语域留出标定；敏感层零饱和与舍入误差联查；W8A8/KV8 交叉干预；固定后端 profiler 拆解 Q/DQ、absmax、GEMM、epilogue、cache 与通信；prefill/decode 统一首 token logits 路径。

端侧建议来自独立工程分析：真实 QNN/HTP 编码、融合和 granularity 要单独验证；fake-quant 的 `F.linear` 不能当整数性能证据；结合 `dmabuf_dump` / `VmHWM` 检查物理峰值，排除临时副本、加载和 RPC 冷启动。
