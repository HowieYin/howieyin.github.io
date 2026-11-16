# 本次实际运行的 Attention 教学实验

环境和原始结果见 results.json。脚本使用 Python 3 + NumPy，固定 seed=20261116，FP64，不下载训练模型。

```bash
python3 attention_properties.py
```

本次在 macOS ARM、Python 3.12.2、NumPy 2.5.5 实际执行并通过五组有意义的断言：在线 softmax 分块、因果 KV prefix、一处局部窗口信息丢失、低维状态碰撞、已经共享 K/V 的多头与 GQA 等价。原始 cache 字节计算另外记录在结果中。

online_softmax 的最大绝对误差约 6.66e−16，符合 FP64 累加次序改变后的舍入量级；不能外推 FP16/FP8 或 GPU 后端。局部 mask 和有限状态碰撞是明确给定矩阵上的构造反例，不是某个已训练模型的精度评测。KV 数字使用统一教学几何，不等于某个实际 checkpoint 的总内存，也未测量 RSS、VmHWM、dma-buf、GPU allocator 或能耗。

没有 GPU 吞吐、TTFT、TPOT、训练任务精度复现。公开正文将论文原报告与本次实测分列，不能混用。
