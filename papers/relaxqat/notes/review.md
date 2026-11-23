# Two-pass review — 2026-11-23

## Pass 1: content accuracy

独立逐页审校覆盖全部17页，完整原表82行分数另经程序比对；最终发布结构再次比对全部13表889个单元格、rowspan/colspan、232处加粗、101处下划线、83处颜色标记、图注及9公式的原MathML/LaTeX。7原图SHA-256一致。证据见 primary-audit.md、source-integrity.json。外部研究覆盖官方论文、仓库及部署文档，69条获取记录和来源验证另存 sources/external-manifest.json、external-validation.json。

已修正/明确：7.70与6.26为百分点且相对ParetoQ；最强基线差值分别2.00/3.75pp；4B视觉落后UPQ0.16pp；Algorithm2最后cycle不可达疑点；Tables8/9相同配置分数差；印刷Eq2 ties-to-even正端点疑点；literal1.58与log₂3端点区别；Figure5颜色与Figure2不同；HTML/PDF LLM-QAT引用年标差别。理论仅为条件几何/恒等式。没有宣称训练、GPU或HTP复现。

## Pass 2: formatting and real rendering

真实浏览器375/390/430/1440/1920全部通过，暗/亮主题、原图/生成图、公式非零尺寸、宽表键盘横滚、目录锚点、图像Enter/Escape焦点恢复、三组控件读数、零全页溢出、零资源/控制台错误。人工查看桌面首页/实验、手机网格与公式/附录截图；修复原HTML绿色#00E000及下划线样式遗漏后重跑。Mermaid原生标签渲染无裁字，使用SVG/PNG和可下载源码。

离线及最终线上验收结果见对应qa JSON和release.md。仅上述已完成的检查构成本次审校依据；网页交互是公式演示，不是性能复现实验。
