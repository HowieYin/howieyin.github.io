# 两轮审校记录

## 第一轮：事实与方法准确性（2026-10-29）

已逐章对照 v1 PDF、作者 TeX、HTML 原表与固定提交源码，检查摘要、全部16表、3图、A–D附录、11组公式关系。

- 原表保留行列、合并表头、缺失值、星号和加粗；未选择性删除失败结果。
- 已修正评分关系的归属：completion / Answer: 比例用于 BoolQ / OpenBookQA 的共同评测说明，不能凭此推断 RACE 专有协议。
- 明确 Table3 的 BoolQ 正文矛盾；Table9 / 16 汇总差异各自保留；Table11 API 名称笔误解释在外部正文。
- GSM8k 单样本与多数投票分开，MATH / GSM8k 的 k 和 Minerva540 不同；HumanEval pass@1 与 pass@100 不混。
- 参数、KV、per-op FLOPs、训练时长、能源重算全部标为推导；训练精度不由参考 runtime 默认值推断。
- 用 Llama2 Table1 和 tokenizer 段核验第一代2k上下文与32k词表；第一代4模型均MHA、无MoE。
- 16样例全部保留；代码缩进直接从作者 TeX 还原；C3 返回类型为元组/标量/False；D3 第二轮是抽取 Python 函数定义，已修正分析。
- 原论文与2026官方资料分开；Qwen3.8等数字标 vendor-reported，禁止跨 harness / thinking budget 排名；未作本地 GPU / HTP 复现。

机器核验结果见 content-audit.json；原始表格证据与数值见 table-manifest.json。

## 第二轮：格式与可读性（2026-10-29）

已在真实 Chrome 对本地站点和独立离线页检查375/390/430/1440/1920五种宽度、深浅主题、目录锚点、公式、表格滚动、键盘缩放和交互边界，并复查顶部、方法、公式、表格、样例与底部截图。

- 无 document 横向溢出、缺图、坏锚点或页面脚本错误；MathML 真正渲染且保留可复制 LaTeX。
- 全部原图不截入 caption；原表完整并可横向滚动；16样例折叠保留全文，代码按原缩进。
- 发现最初 Mermaid 纵向图放大过度，改为横向机制图与独立滚动容器，保留关键 residual / cache 关系；不以图替代详细公式和源码说明。
- 核验4模型KV、weight/KV独立精度、batch倍率和假设GQA标签；成本组件检查正/零/负节省与盈亏平衡边界。
- 标题、来源、原图与原创图标记、原文显式关系与重构公式的标签均已区分。

最终生产环境验收记录在 publishing-report.md。本记录不把本地 QA 等同于已上线。
