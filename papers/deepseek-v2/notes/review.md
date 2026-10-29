# DeepSeek-V2 v5 — 两轮审校与页面 QA

日期：2026-10-29。来源为指定 52 页 v5 PDF、作者 TeX/arXiv HTML、官方 HF config/modeling，以及逐条注明日期的外部来源。

## 第一轮：排版与可读性

- 已核对 §1–5、附录 A–G、扩展研究、个人观点的顺序与层级；长文有可折叠目录。
- 5 个原图、6 个图面板均来自作者向量 PDF；原 caption 在图片外，图内轴、图例和模块文字保留。
- 完整表格 37 张；附录 26 张 prompt 表采用逐表折叠，题面与标记全部保留，展开后可滚动。
- 47 个编号公式置于 15 个原公式组，另有 4 个分析推导；原文 MathML 与本篇推导分开注明，长公式局部横滚。
- 三个 Mermaid 流程图有 SVG、PNG 和源码；发现对齐图初版纵向过长，已调整分阶段布局并重渲染。
- 已查看真实浏览器截图：手机首屏、路由、原表浅色模式、桌面 GRPO 公式、架构图和长表。
- 已通过 375、390、430、1440、1920 五种宽度；无 document 横向溢出，深浅色切换、目录、图像 zoom/Escape、长表滚动和 prompt 开关均正常。

## 第二轮：来源准确性

- 全部 37 原表逐 cell 文本、rowspan、colspan 比对，1,813 cells 一致；原始强调保留。
- 全部 47 编号公式的 MathML alttext 逐组比对，公式编号无缺失或重复。
- 核验 PDF SHA256：f149f9c09aa8b46ce6a740a255a99b7cc2317ac5b065989339919b7e6c329c9b。
- 核验 Full V2/Lite 的层数、hidden、heads、KV/Q rank、RoPE、expert 数与 intermediate。
- 修正审计重点：67B 是 GQA8，93.3% 包含精度；同精度名义减少 82.24%；V2 MHA 假设与 Table9 不同分母。
- 核验 MLA 每 head 吸收、位置缩放 √192、normalized cKV、per-head latent aggregate，未把多个 head 聚合成一个。
- 官方 HF eager 可能缓存展开 KV，未把 checkpoint 的架构声明等同运行时压缩。
- 核验 42.5% GPUh/T 与 5.76×服务 throughput；未提供/捏造单请求 TTFT、TPOT、p95 或 GPU 本地复现。
- 核验 Table3 RL 的收益与退化、Table4微小分差、Table5 judge口径，Table8深度混杂、Table9 420B训练量。
- 补齐原 §4.4 的 SFT 数据量、alignment tax、online/offline三项；§5的知识更新、幻觉、其他语言限制及多模态计划。
- 附录 E 的标注分歧未被改写为去偏因果证明；F 的综合分与accuracy分开；G 的 WinoGrande prefix-selection方向保留。
- 外部研究按访问日与版本/固定SHA注明；2024论文结果与2026后续官方报告分开，不拼造统一SOTA榜。

## 数学与运行检查

`interactive/math-check.mjs` 在模块未实现时先失败，随后通过：cache 单位与128Ki边界、batch等价、expanded/absorbed输出一致、不同M下设备上限、M=8时global top-k等价。

五宽浏览器 QA 还验证了 BF16 与 6-bit 容量变化、切换 query 后的吸收等价、默认/无限制/集中路由、图像加载、内部锚点及控制台。离线 file:// 版本同样通过五宽检查，所有数学和交互资产本地提供。

详细记录：`numerical-check.json`、`qa-local.json`、`qa-offline.json`。上线后的记录另存 `qa-online.json` 与 `publishing-report.md`。

## 飞书镜像复查

镜像：https://bytedance.larkoffice.com/docx/YSovdytEvoLwa6x1RDecdFUmnnb 。已回查 9 个图片资源、50 张表（37 原表 + 13 分析表）、47 原编号公式与 18 个其他行内/分析公式。创建时飞书不支持 `mathds`、inline `split` 和 RGB color 写法，已只修正 (24)、(30)、(32)、(41)、(43) 五段，使用语义等价的 `mathbb`、`aligned` 与 `color{blue}`；修复后无降级 warning，原网页仍保留原 MathML。

通讯录邮箱核验与当前身份同为 yinhongyi@bytedance.com；新文档由该用户创建，`edit` 与 `manage_public` 权限测试均返回 true。没有另建文档或修改公开范围。
