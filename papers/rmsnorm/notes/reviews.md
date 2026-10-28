# RMSNorm review record

## Pass 1 — Formatting and readability

2026-10-27。原论文章节顺序为 §1–7、附录 A.1–4；前置 TL;DR 与后置审计／演进／工程／个人判断单独标明。原始 7 Figures、10 Table 编号共 17 份原貌 PNG；Table 9 原为训练曲线。9 个数据表全部转录，原貌与文字互补；两张 Mermaid 示意与两套交互明确 Agent-created。

真实 Chrome 检查 375、390、430、1440、1920px，审阅 top／method／interactive／table／current-update／zoom screenshots。12 组可选择 MathML（10 原式、2 补充推导）可渲染，宽表与公式在容器内滚动；原图可缩放，Escape 关闭并恢复焦点，目录移动端跳转与关闭正确。键盘 range 输入、前缀反例、零分母／epsilon、IO 默认与大 M 更新通过。无脚本静态读数／原图／公式仍在；reduced-motion 回退正确。独立页面全部本地资源及站内阅读链接 HTTP 200，无 Console exception。

修复：两张 Mermaid 内部 ID 名称冲突；reset 中数值 0.5 与 option 字符串 `.5` 不匹配造成空选择；原 Table 8 caption 下多余正文边缘。修复后全部五种宽度重新检查。

## Pass 2 — Content accuracy

依据 1910.07467v1.pdf、原 TeX 与附录逐项核验元数据、10 个编号公式、17 份原视觉证据与 9 个数据表。保留所有行列、空值、bold、计时标准差、原 caption 与脚注符号，转录只展开合并标题；原貌图保留合并结构。

确认并解释：

- Eq.6 正尺度条件与一般 |α| 形式；零向量及 epsilon 边界。
- RMS sphere 位于 affine 前；RMS²=σ²+μ²，variance 并非强制 1。
- Eq.9 原式完整保留；新增 outer-product 梯度 shape、Θ(D) backward，以独立中心差分核验。
- pRMS 固定前缀／ceil，原作者 PyTorch floor／sqrt 外 eps、README sqrt 内 eps 与 Llama FP32 行为分开。
- Figure 3／7 重复 label、CIFAR “Figure 9” 实为 Table 9；原编号未重排。
- 665→501s = 24.66% time reduction / 1.327×；988→652s = 34.01% / 1.515×；248→231s = 6.85% / 1.074×。
- 12.02→4.34s = 63.89% / 2.770×，属于 pRMS 与旧 Theano OE，不是全部 RMS 或现代 LLM 的保证。
- CIFAR 8.96→8.83 为 0.13 个百分点（原正文 0.013% 错）；10.49→8.83 为 1.66 个百分点；10.37−8.83=1.54 个百分点。
- Table 7 上半 existing work 与下半作者复现分开；Recall@K／mean rank、5 test splits 与 5 training seeds 区分。
- timing 的三次标准差不作为质量置信区间；BN cuDNN、CIFAR 学习率差异、未披露的精度／完整 token 预算明确。
- 后续研究更新与 2019 事实分开，2026 kernel 报告限 shape／dtype／GPU／方向；2604.23434v1 的主要干预是 DyT，RMS 在 64M 中为 2-seed 基线并与 LN 接近。T/P&lt;1.84 是全部实验的覆盖上界，不是 LN/RMS 优劣阈值；重新读取 HTML 正文及附录确认，已修正扩展稿的相关描述。

## Verification evidence

本地数学检查：正／负尺度、均值平移、epsilon、固定前缀失败、中心差分梯度、理想 byte accounting 和 Amdahl，全部通过。浏览器结果保存于 `qa-local.json`。这些不属于 H100/B200/NPU 性能复测；所引现代 benchmark 均为原来源报告值。

发布后的 Actions、production 资源／界面、索引与权限验证另保存于发布报告。

## Caption correction — 2026-10-28

按读者反馈修正全部 17 份展示图片：仅裁取原始图表主体，将 Figure/Table 图注及 Figure 1、Figure 6 的子图说明放在图片外，以可选择文字呈现。原图内的坐标轴、图例、数值和表格行列保持完整；完整 PDF 保留原始版面。裁剪采用原 4× PNG 的无损像素区域，没有重绘或重采样。

逐张核对 PDF 中 caption 的位置与裁剪边界，检查 17 图总览及 Figure 1、3、6 的高分辨率结果。Table 1 外部说明中的 TeX 命令名改为实际 “✓/✗” 符号。外部图注字号调至 14px，缩放链接和图片使用新版缓存参数。
