# 审查记录 · 2026-10-27

交付遵循项目 `agent.md` 与 `PUBLISHING.md`，发布栏目使用现有网站 `/papers/` 路由。研究、页面设计与发布各项落实在实际页面和资产中。

第一轮：来源与准确性审查。核对 v7 metadata、全文结构、4 条原公式、10 图 / 11 表、Figure 5 数字、Table 2 与 Table 7 粒度、不同平均指标、GPU 数量与性能口径、fake / true INT8、缓存 dtype、当前文献版本与引用编号。11 张渲染表与来源 HTML 的全部单元格文本及 row/col spans 精确一致；原排版同时保留。

第二轮：读者与表达审查。核对章序、符号/shape/推导、论文事实/外部来源/解释的区分、逐图读法、指标方向、代价与公平性、toy/容量计算的边界、独立评价及实现定位顺序。修正过于绝对的速度/无损表述，保留 Table 8 timing 与 Ascend Average 疑点。

数学验证：202 组 α/bit 情况满足 Figure 5 浮点乘积不变；α=0 权重峰值为 1、α=1 激活峰值为 1、α=.5 得 `[1,4,1,3]`。独立权重/KV dtype、GQA head 数与逻辑容量核验通过。三组件均不预测未经测量的模型 accuracy/throughput。

页面 QA：真实 Chrome，375 / 390 / 430 / 1440 / 1920 px；Top / Method / Table / References 截图检查；10 原图、11 原表截图、2 Mermaid SVG 共 23 图像加载；6 MathML 公式；三 ECharts 组件；内部锚点；鼠标/键盘操作、图放大/Escape、目录折叠与跳转、表格横向滚动、原表展开、reduced-motion、无 JS 三组数值回退与静态结构图。沿用现有深色主题，不新增未验证的浅色模式。

复现范围：论文与源码审查、数学与网页 QA。未运行大模型训练/校准、未在 A100/HTP 上复测模型精度或速度。所有原文/后续 benchmark 均归属原作者。

发布地址：https://howieyin.github.io/papers/smoothquant/ 。上线后再次执行线上验收；临时脚本、浏览器 profile、截图及原始 QA log 不纳入仓库。
