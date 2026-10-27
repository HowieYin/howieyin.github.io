# 原论文阅读记录

指定来源：arXiv **2211.10438v7**，2024-03-29，13 页；ICML 2023。正文 1–7、References、Appendix A 均已阅读全文。报告正文按原章顺序组织，Abstract 的问题、方法与三项结果落实在 TL;DR。

结构与证据：

- §1 Introduction：资源缺口、activation outlier 与系统动机；Figures 1–2。
- §2 Preliminaries：对称整数编码与粒度；Eq. (1)、Figure 3。
- §3 Review of Quantization Difficulty：通道统计、GEMM 归约轴；Eq. (2)、Figure 4、Table 1。
- §4 SmoothQuant：离线等价变换、α、Norm 融合与 precision map；Eqs. (3)–(4)、Figures 5–6。
- §5 Experiments：setup、accuracy、speed/memory、530B、ablation；Figures 7–10、Tables 2–11。
- §6 Related Work：一般 equalization、LLM.int8、ZeroQuant、GPTQ 的历史定位。
- §7 Conclusion：规则 W8A8 的精度、显存、系统结果；没有独立 Limitations 章节。
- Appendix A：weight-only 与 W8A8、batch / KV / kernel 边界；未来 W4A4 并非已验证结果。

数学重建区分实数等价、量化近似、真实系统收益。Figure 5 原矩阵独立重算得到 `s=[1,4,1,3]` 与 `Y=[[-16,9,16],[11,0,-11]]`。源码审查固定官方 commit `c61476d728e42ae0d8a35e7e78494edcac3237b5`。

完整报告：[reading.md](../reading.md)。原始 PDF 与图表位置：[sources](../sources/visual-manifest.json)。
