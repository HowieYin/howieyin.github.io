# Stepped MoE v1 — 两轮审校

审校日期：2026-10-29。主来源为指定 15 页 PDF、arXiv v1 HTML 与作者 TeX，覆盖正文 §1–5 和附录 A.1–A.7。

## 第一轮：内容与数值准确性

- 逐项复核方法、张量、56 层 /12 Dense /44 MoE、D=1536、212 专家、256 expert hidden、12 Query /12 KV heads。
- 自动比较 9 张原表全部 538 个单元格，373 个十进制分数顺序与内容一致；逐表粗体数量一致。合并行列由 arXiv 的 rowspan/colspan 和 ltx_rowspan/ltx_colspan 元数据恢复。记录见 table-audit.json。
- 4 张原图使用作者原 PNG，未重绘或裁去轴、图例。Caption 和中文讲解均在图外；原图来源与 SHA256 已记录。
- 保留 3 个编号公式的原生 MathML 与 LaTeX。另解释原求和端点问题，未静默修改原式。
- MMLU 绝对与相对提升、SFT 百分点、KV 与专家参数条件估算、A.6 TPU core-hours 均重新计算。
- 明确 weighted CE 对 1B/3B 的相反影响；Table 3 与 Table 9 的训练/推理段长区别；Table 7 与主表 composite 差异。
- 将严格嵌套描述改为条件性推理：size embedding 可改变 logits；不同预算不能自动保证 mask 包含。
- 明确 Figure 3 为模拟器累计延迟、对数轴，不编造 speedup 或真实设备结果。
- 扩展来源截止 2026-10-29；Edge0/SlimWise 属于并行路线，不称作本文 follow-up 或共同 SOTA。
- 推理伪代码修正为 token 外层、layer 内层；明确 next-token shift 与 prompt prefill 调度仍待实现核对。

结果：通过。未声称本地训练、GPU 或 NPU 复现实验。

## 第二轮：排版与真实渲染

本地 Chromium 渲染宽度 375、390、430、1440、1920，测试高度 1000，覆盖桌面与手机布局。完整结果见 qa-local.json。

- 4 原图、2 Mermaid 解释图、9 完整原表、3 编号公式实际加载。
- 原图键盘/鼠标放大与 Escape 关闭正常；caption 文本独立显示。
- 宽表手机横向滚动，合并表头保留；页面级无水平溢出。
- 长公式容器滚动，MathML 非零尺寸；LaTeX 源码可展开并复制。
- 桌面 sticky TOC 与手机折叠目录跳转正确，锚点存在。
- 分段交互 S=64、1024 tokens 返回 16 mask 组、15 后续 refresh，与 Table 9 分数一致。
- 内存交互默认返回 336 KiB/token、8K=2.625 GiB KV；假设 4 KV heads 返回 0.875 GiB，并明确非论文配置。
- 深浅主题切换正常，图像不反色；浏览器 console/pageerror、HTTP 资源失败均为 0。
- 实际查看桌面首页、手机首页/公式/宽表/内存交互及两张 Mermaid 渲染图。

修复记录：保留原表的颜色类别并适配主题对比度；明确 image visibility，避免非视口图片的渲染检查受 content-visibility 影响。临时浏览器检查脚本先触发 eager loading 再等 decode，避免 lazy 图片阻塞检查。

结果：本地通过。线上验收另见 publishing-report.md 与 qa-online.json。
