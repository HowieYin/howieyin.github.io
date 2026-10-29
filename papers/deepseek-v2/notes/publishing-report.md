# DeepSeek-V2 v5 — 发布记录

2026-10-29，精读指定的 2405.04434v5。完整覆盖 §1–5 与附录 A–G；保留 5 编号图 / 6 原面板、37 原表、47 编号公式，补充 3 Mermaid 图和 3 交互。

已发布：https://howieyin.github.io/papers/deepseek-v2/

飞书镜像：https://bytedance.larkoffice.com/docx/YSovdytEvoLwa6x1RDecdFUmnnb

研究、两轮审校、原始单元格/公式比对及本地/离线五宽 QA 已通过。飞书资源、公式与账号编辑/管理权限已回查；记录见 `review.md` 与 `feishu-verification.json`。所有 GPU 性能引用作者报告，未本地复现。

内容提交：`79afbce`。与同期新增论文合并后的上线提交：`1a85115f2ca447c05857adb54ec1d8e87a82ffaa`，普通 push 至 `main`。

GitHub Pages 构建、状态回报和部署全部成功；部署完成于 2026-10-29 12:48:33 UTC（北京时间 20:48:33）。

构建记录：https://github.com/HowieYin/howieyin.github.io/actions/runs/40876047218

线上 QA 于 2026-10-29 12:49:20 UTC 完成：

- 375、390、430、1440、1920 五宽均通过，无页面横向溢出、缺失资源或 Console / JavaScript 错误。
- 核验 5 图 / 6 面板、37 原表、47 连续编号公式、93 MathML 元素、3 Mermaid 图与 3 交互。
- BF16 / 6-bit 缓存容量、两种 query 的吸收等价、受限 / 无限制 / 集中路由均通过。
- 深浅色、目录跳转、图片放大与 Escape、移动端长表滚动和 prompt 表展开均通过。
- 37 个站内页面及资源 URL 全部 HTTP 200；线上 HTML 与审校版本逐字节一致，SHA256 为 `f5862bbf318c54b8c75acd05ba623cc499f0ea132a838d872c2a367fb1b9f892`。
- 首页、Paper Readings 总览和 Model Architecture 专题在手机 / 桌面均出现本篇；GQA、MQA、Sparse MoE、DeepSeek-V3 四处反向链接正常。核验时目录包含 35 篇阅读，保留所有同期发布。

证据：`qa-online.json`、`qa-online-links.json`、`deployment.json` 和 `screenshots/`。上述记录对应内容上线版本；随后补交此发布记录和证据，正文与交互资产保持相同哈希。

离线交付：工作区 `deepseek-v2-offline.zip`，解压后打开 `deepseek-v2/offline.html`。正文、图表、字体、脚本、原 PDF 和审校记录均随包提供；外部参考链接需联网。
