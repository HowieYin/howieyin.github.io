# InstructGPT v1 — 双轮审校

审校日期：2026-10-29。目标来源：2203.02155v1.pdf，68页；全部50编号图、14编号表、2编号公式。

## 第一轮：内容准确性与数值

已按Abstract→§1–5→Appendix A–F复核正文、原图caption、配置和实验；14原表每一个单元格与提取来源逐格比较一致，2原式LaTeX与原文一致。

完成的关键修正与限定：

- Figure1/3参照是175B SFT，不误作对GPT的85%直接偏好率。
- C.3的5e−6/1.04e−5/2.45e−6属于初始化SFT；PPO actor LR按E.9扫描说明，不臆填最终逐variant值。
- 16epochs SFT展示基线与2epochs+10%ptx PPO初始化分开；实际RM public-task6B初始化与主文简图分开。
- 原式1归约歧义保留，示意实现采用有效pairs均值；KL sequence sum和sample可负、期望非负区分；old-policy ratio与reference ratio区分。
- 毒性basic/respectful/biased条件拆开，DROP/SQuAD/翻译的ptx恢复差值重算。
- 原表CNN/DM与TLDR重复、模板复制疑点、选择题lowest-logprob表述保留并解释。
- Figures46–50按PDF逐字阅读后，分别解释就业数据、serendipity造句、Ginsburg新闻QA、西葫芦购物清单与焦虑胸口感受，不将小图外观当成文本证据。
- 外部RewardBench2为2026v2快照；实时入口可达不等于全排名核验；GPRL分数单来源且未复现。

结论：内容审校通过；现存原文歧义已经就地记录，没有自行编造修正数据。

## 第二轮：格式、可读性与真实渲染

Chrome实际渲染375/390/430/1440/1920，检查dark/light、50原图、14原表、2原式与4推导、2解释图、3交互、TOC、zoom/Enter/Escape和LaTeX复制。修正行内MathML在Markdown阶段拆断段落的问题；修正原图文字转录的移动端换行，使西葫芦清单文字表在375宽下为295px，不撑破页面。

原图完整保留、caption独立图外、文字图有copyable转录；Mermaid有mmd+SVG+PNG；宽原表横向滚动；公式使用MathML与可复制LaTeX。全部页面内锚点与70本地资源/导航路径检查通过。

离线版将图片、字体、CSS、JS嵌入单HTML并采用eager，避免file协议的外部资源加载影响；线上版保留lazy。最终离线版五种宽度全部通过，原图/公式/交互完整。飞书保留52图与14原表，另含文字图转录表和解释表；两个原式因split环境不兼容，采用同数学内容的单行LaTeX并保留原LaTeX代码，局部修复同一个文档后回读确认。

本地完整QA及最终线上QA的JSON和屏幕证据汇总到publishing-report.md。这里的验证是内容与页面验证，不是原论文训练/性能复现。
