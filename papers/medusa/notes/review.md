# Medusa v3 — 两轮审阅

审阅日期：2026-10-27。原文：2401.10774v3，27 页；正文、References 与 Appendix A–G 已完整阅读。

## 第一轮：排版与可读性

- 核心叙述依原章节排序，附录 A–G 单独标识，独立分析置于末尾。
- 保留 23 张 original figures、8 张 original tables；图例、多 panel、轴和原表 caption 均已目视核验。所有原表另有完整可滚动转录。
- 两幅 Mermaid 已实际渲染；修复其内嵌 SVG 节点 ID 重复，并同时更新样式/marker 引用。
- 原图 click-to-zoom、Escape 与返回焦点、mobile contents drawer 和 anchor jump、宽表滚动通过真实 Chrome 操作。
- 375、390、430、1440、1920 px 均无文档横向溢出；10 组可复制 MathML 公式正常，长公式局部横向滚动。
- 树选择/两步提交、温度与阈值、Tables 6–8 查询通过状态检查；键盘调整 range 有效。
- 无 JavaScript 时正文、全部原图/表、公式和三个静态机制回退仍存在。reduced-motion 有效。
- 查看 top、method、tree、experiment、cost、bottom 与两幅 Mermaid 的真实截图，层级、字距和图表清晰度可读。现有站点为 dark-only；没有凭空宣称通过不存在的 light-mode UI。
- 静态仓库按 README 原方法本地 serve；没有引入运行时 CDN、构建框架或新部署 workflow。

## 第二轮：原文准确性

- 对照 PDF/TeX 核验元数据、Eq.1–2 及所有方法中的未编号核心公式、23 个图号、8 个表号。
- 原 LM head 预测 t+1；Medusa head k 预测 t+k+1；示例 2×3 树为 8 个 Medusa 节点 + 1 个物理 LM 根。
- 核验 mask 为自身/祖先/旧前缀，position ID 沿 depth，KV 提交使用 accept_length+1。流程图修正为选中最长前缀后才保留终点 logits。
- W₂ 行/列方向与 Table 5 QK/PV shape 差异保留原式/原表，并另列一致的工程记号。
- λ₀ 按 Eq.2 乘 Medusa loss；训练 warmup 与 θ₀/α/δ 记号混乱明确说明，不自行补 schedule。
- 核验 self-distillation KL 方向；共享 teacher 权重只省第二份 backbone，不宣称无额外计算/显存。
- 用 Leviathan 原算法与证明纠正 §2.3.1 的 p=q 反例：严格 rejection 接受率为 1，独立样本匹配为 Σp²。
- 区分 M1 冻结、M2 目标变化、greedy、严格 rejection、typical 的不同保证；typical 未声明分布无损。
- Tables 1–4 数值、quality 差值与速度/TPOT 复算核验；保留 N/A 和 13B −0.14 分；GPT-4 judge score 未误标为 accuracy。
- Tables 6–8 全部 168 个数值对直接用于交互；54.8/1.24=44.19 只称 FLOP/s 比值，按工作量约64倍推导每轮算子时间约1.45倍。
- Figures 21–23 标为简化模拟，不当端到端服务实测；64 节点和 B>32 不当通用硬件阈值。
- Figure 8 自蒸馏 caption 与正文/配置分组冲突如实记录。
- 所有近期性能按来源模型、硬件、batch/concurrency、采样和任务条件引用，不宣布全局 SOTA，也不冒充本地 GPU/NPU 复现。

## 自动化证据

本地真实浏览器记录：`qa-local.json`；线上记录在正式部署后保存为 `qa-production.json`。脚本只测页面、状态机制和代数查询，不测 GPU 模型推理。
