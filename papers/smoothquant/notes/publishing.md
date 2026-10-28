# 发布验收记录 · 2026-10-27

已发布：[https://howieyin.github.io/papers/smoothquant/](https://howieyin.github.io/papers/smoothquant/)。

论文：SmoothQuant，指定 `2211.10438v7.pdf`。研究依 `agent.md` 全流程，发布依 `PUBLISHING.md`。完整章序 1–7 + Appendix A；10 原图、11 完整表、4 原公式与 2 解释性推导、2 Mermaid 结构图、3 ECharts 交互、17 一手来源、后续研究与当前状态、独立评价及实现审计。

提交：

- 初始内容：`b39fe7f51c53a00ac955b3e4ac5caa54da313cee`。
- Pages 资源与 SVG 修复：`5ca4c5c9f9f3934f701f50adf6e1da342d1c4625`。
- 成功部署：[Actions 40486868417](https://github.com/HowieYin/howieyin.github.io/actions/runs/40486868417)，build / deploy / report 均 success；部署 head `f500a8676b08827c5364e354b55986de27397d03`。Git ancestry 检查确认包含以上两项提交及同期其他文章。单独修复 run 被后续并行发布替代而取消，最终使用上述成功部署验收。

线上 QA：

- 375 / 390 / 430 / 1440 / 1920 px，页面 scrollWidth 均等于 viewport；桌面正文 880 px。
- 全部 23 图片资源（10 原图、11 原表截图、2 SVG）加载正常；SVG 实际尺寸与文本检查通过。
- 11 表所有原单元格文本、rowspan / colspan 完整；宽表可用键盘横向滚动；原表展开正常。
- 6 MathML 公式正常；LaTeX 可展开复制。
- α / bit 滑块、延迟配置、独立 KV bit 控件、图放大与 Escape、手机 TOC 跳转与收起正常。
- reduced-motion 和无 JS 三组数值回退 / 静态结构图正常。
- 控制台无页面错误，网络无失败资源。
- Paper Readings 索引标题、ICML 2023、v7、阅读日期与 tags 正确；实际点击进入；首页和 FlashAttention 关联入口存在。
- 指定线上 PDF HTTP 200，SHA-256：`d17856490ff664a99f8d08fe76eabd83219ed048cd1bc18ce3d0571e71056ff8`，与本地主源一致。
- arXiv v7、PMLR、DOI 和 NVIDIA 文档关键链接 HTTP 200。

部署适配：网站是静态 HTML，Pages legacy 流程实际使用 Jekyll；初始 `_shared` 被排除，已将脚本、字体与离线样式迁至公开 `assets/`，未修改公共部署流程。SVG 设置固有宽高，并使用 SVG 文本以避免外部图片模式下的 HTML 标签裁切。离线 `reading.html` 同样通过真实 Chrome 验证。

本次模型复现范围：原文 / 源码审查、202 组 α/bit 数学验证、容量计算及网页验证。未重新运行大模型训练、校准、A100/HTP 精度或性能 benchmark。

## 图注排版修正 · 2026-10-28

全部 10 张原图与 11 张原表重新裁切，仅包含图表主体；原说明段落、Table 7 / 10 夹入的相邻正文均移出图像。完整原图注和中文解读显示为图外文字，放大视图也显示独立图注。页面修订日期更新，研究检索快照不变。修复 Figure 5 等行内向量被引用链接误处理的问题。

本地验收：21 个裁切区域均不与 PDF Figure / Table 图注文本块相交；21 个图外 caption 完整；图像实际尺寸与 HTML 宽高一致；11 张表全部单元格文本与 rowspan / colspan 与来源一致。真实 Chrome 的五种屏宽无页面横向溢出，全部 23 图像、6 公式、3 交互组件加载，图放大 / Escape、原表展开与键盘滚动正常。检查 Figure 1 / 4 / 5 / 9、Table 7 与手机截图，确认说明段落在图像外；21 份浏览器获取的图像 SHA-256 与本地裁切文件逐一一致。
