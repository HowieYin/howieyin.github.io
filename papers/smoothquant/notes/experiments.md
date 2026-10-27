# 实验审计记录

全部 11 张表保留完整单元格文本、rowspan / colspan、粗体 / 下划线和脚注；页面另保留原 PDF 排版截图。10 张图均附 caption、来源、逐图解析、缩放。

主要边界：

- 512 Pile 标定句子；OPT/BLOOM α=0.5、GLM=0.75；LLaMA/Table 7 的配置另列。
- Table 2 的 O1/O2/O3 权重均 per-tensor，activation 分别 token dynamic / tensor dynamic / tensor static。Table 7 不是 O3。
- Table 3 的七任务平均、Table 4 的四任务平均、GLM 四任务平均不能跨列直接比。
- Figure 9 的 tensor parallel 数变化；OPT-66B 部分 SQ 点稍慢，不能概括每配置更快。
- Table 8 的 latency 计时定义不充分，不能标为 TPOT；memory 的每卡/总量与 saving rounding 留疑点。
- Table 10 的 530B 结果主要是卡数 16→8；128 tokens 时 253 vs 232 ms，慢 9.1%。
- Table 11：13B/256，O3 112.1 vs FP16 152.6 ms，1.361×，延迟下降 26.54%；无 per-op profiler，不能唯一归因于 absmax。
- PPL/accuracy 变化非零；作者 “lossless” 不是逐元素零误差。

未下载模型权重、未重跑 A100/HTP 性能、未产生逐样本混淆矩阵。报告中所有 benchmark 数值来自明确的一手文献；计算器只输出逻辑容量或原 Figure 5 数学结果。
